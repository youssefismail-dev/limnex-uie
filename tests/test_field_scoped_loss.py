"""
Tests for field-scoped loss.

The two benign corpora are sampled differently on purpose and each is WRONG
about the other's fields: the name corpus holds one URL per domain, so its path
depth is 1.29 with 27.7% bare against a population of 2.30 / 2.9%, while the
structure corpus covers only multi-page sites — a biased 4.6% of domains — so
its names are unrepresentative. Training every field on every row would rebuild
E2.1's bare-homepage artifact at ten times the scale.
"""
from __future__ import annotations

import torch

from uie.model import ModelConfig, URLTransformer
from uie.tokenizer import URLTokenizer
from uie.train import (CORPUS_SCOPE, NAME_FIELDS, STRUCTURE_FIELDS,
                       encode_corpus, field_mask)

CORPUS = [
    "https://blog.shopwave.io/posts/2024/review?page=2",
    "https://a.co.uk/",
    "http://brandx.com@evil.tk:8080/a/b?x=1#f",
]


def _tok():
    return URLTokenizer().fit(CORPUS, min_count=1)


def _trained(tok, ids, mask):
    """Which token TEXTS carry weight, for readable assertions."""
    return "".join(tok.itos[i] for i, w in zip(ids[1:], mask) if w)


def test_name_rows_train_naming_content_only():
    tok = _tok()
    ids = tok.encode(CORPUS[0], "benign")
    text = _trained(tok, ids, field_mask(ids, tok, "name"))
    assert "shopwave" in text and "io" in text and "blog" in text
    assert "posts" not in text and "review" not in text and "page=2" not in text


def test_structure_rows_train_path_content_only():
    tok = _tok()
    ids = tok.encode(CORPUS[0], "benign")
    text = _trained(tok, ids, field_mask(ids, tok, "structure"))
    assert "posts/2024/review" in text and "page=2" in text
    assert "shopwave" not in text and "blog" not in text


def test_markers_and_eos_are_trained_by_BOTH_corpora():
    """URL SHAPE — whether a path exists at all — is not a content question.

    Scoping markers to the structure corpus would teach "a path follows a
    label" only from multi-page sites, which have one essentially always, and
    generated URLs would come out with a bare rate near zero against a real
    2.9% benign / 47.4% phishing.
    """
    tok = _tok()
    ids = tok.encode(CORPUS[0], "benign")
    markers = {tok.marker_id(m) for m in ("SUF", "LBL", "SUB", "PTH", "QRY")}
    for corpus in ("name", "structure"):
        mask = field_mask(ids, tok, corpus)
        for target, w in zip(ids[1:], mask):
            if target in markers or target == tok.eos_id:
                assert w == 1.0, (corpus, tok.itos[target])


def test_the_two_scopes_partition_the_content_fields():
    """No content field may be trained twice or dropped entirely."""
    assert not set(NAME_FIELDS) & set(STRUCTURE_FIELDS)
    covered = set(NAME_FIELDS) | set(STRUCTURE_FIELDS)
    assert covered == {"SUF", "LBL", "SUB", "USR", "PTH", "QRY", "PRT", "FRG"}


def test_every_content_token_is_claimed_by_exactly_one_corpus():
    tok = _tok()
    for url in CORPUS:
        ids = tok.encode(url, "phishing")
        n = field_mask(ids, tok, "name")
        s = field_mask(ids, tok, "structure")
        markers = {tok.marker_id(m) for m in
                   ("SCH", "USR", "SUF", "LBL", "SUB", "PRT", "PTH", "QRY", "FRG")}
        control = {i for i, s_ in enumerate(tok.itos) if s_.startswith("<C:")}
        for target, wn, ws in zip(ids[1:], n, s):
            if (target in markers or target in control
                    or target in (tok.eos_id, tok.bos_id)):
                continue
            assert wn + ws == 1.0, (tok.itos[target], wn, ws)


def test_control_tokens_are_never_trained():
    """They are SUPPLIED at generation and never sampled, so predicting them
    is wasted capacity. Excluded deliberately rather than by omission."""
    tok = _tok()
    ids = tok.encode(CORPUS[0], "phishing")
    control = {i for i, s in enumerate(tok.itos) if s.startswith("<C:")}
    for corpus in ("name", "structure"):
        for target, w in zip(ids[1:], field_mask(ids, tok, corpus)):
            if target in control:
                assert w == 0.0, tok.itos[target]


def test_unknown_corpus_trains_shape_only():
    """A row whose corpus is unrecognised must not silently train everything."""
    tok = _tok()
    ids = tok.encode(CORPUS[0], "benign")
    mask = field_mask(ids, tok, "nonsense")
    assert _trained(tok, ids, mask).strip("<>") != ""      # markers survive
    assert "shopwave" not in _trained(tok, ids, mask)
    assert "posts" not in _trained(tok, ids, mask)


def test_encode_corpus_without_corpora_trains_everything():
    """Backwards compatibility: pre-E2.3 callers pass no corpus labels."""
    tok = _tok()
    enc, _ = encode_corpus(CORPUS, ["benign"] * 3, tok, 128)
    for ids, mask in enc["benign"]:
        assert all(w == 1.0 for w in mask)


def test_weighted_loss_matches_unweighted_when_all_ones():
    tok = _tok()
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1,
                      n_heads=2, d_ff=32, max_len=128)
    torch.manual_seed(0)
    model = URLTransformer(cfg).eval()
    ids = torch.tensor([tok.encode(CORPUS[0], "benign")])
    ones = torch.ones(ids.shape[0], ids.shape[1] - 1)
    assert torch.allclose(model.loss(ids, tok.pad_id),
                          model.loss(ids, tok.pad_id, ones), atol=1e-6)


def test_a_fully_masked_row_contributes_no_gradient():
    tok = _tok()
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1,
                      n_heads=2, d_ff=32, max_len=128)
    torch.manual_seed(0)
    model = URLTransformer(cfg)
    ids = torch.tensor([tok.encode(CORPUS[0], "benign")])
    zeros = torch.zeros(ids.shape[0], ids.shape[1] - 1)
    loss = model.loss(ids, tok.pad_id, zeros)
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters()
             if p.grad is not None]
    assert float(loss) == 0.0
    assert sum(grads) == 0.0


def test_loss_averages_over_its_own_weight_mass():
    """Rows contributing 8 scored tokens and 80 must not be pooled as equally
    informative — otherwise the effective learning rate on a field drifts with
    whatever mix of rows a batch happens to contain."""
    tok = _tok()
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1,
                      n_heads=2, d_ff=32, max_len=128)
    torch.manual_seed(0)
    model = URLTransformer(cfg).eval()
    ids = torch.tensor([tok.encode(CORPUS[0], "benign")] * 2)
    half = torch.zeros(2, ids.shape[1] - 1)
    half[0, :4] = 1.0
    half[1, :4] = 1.0
    # Identical rows, identical masks -> the mean must equal the single-row mean.
    single = model.loss(ids[:1], tok.pad_id, half[:1])
    assert torch.allclose(model.loss(ids, tok.pad_id, half), single, atol=1e-6)


# --- E5 preparation: naming the frame removes the need for scoping ----------

def test_naming_the_frame_removes_content_scoping():
    """The scoping exists ONLY because the model cannot tell which sampling
    frame a row came from. A tokenizer carrying `<F:*>` can, so every row may
    teach every field — which is the point, not a side effect: it moves path
    CONTENT from 1.65M benign / 46k phishing rows onto 2.55M / 239k, and puts
    path shape and path content under one teacher instead of two that disagree.
    """
    from uie.tokenizer import URLTokenizer
    from uie.train import field_mask

    url = "https://a.co.uk/login/verify"
    scoped = URLTokenizer(include_scheme=False, include_language=False)
    framed = URLTokenizer(include_scheme=False, include_language=False,
                          include_frame=True)
    for t in (scoped, framed):
        t.fit([url], min_count=1)

    ids_s = scoped.encode(url, "benign")
    ids_f = framed.encode(url, "benign", frame="domain")
    w_s = field_mask(ids_s, scoped, "name")
    w_f = field_mask(ids_f, framed, "name")

    i_s = ids_s.index(scoped.marker_id("PTH"))
    i_f = ids_f.index(framed.marker_id("PTH"))
    assert w_s[i_s] == 0.0        # scoped: path content masked on a name row
    assert w_f[i_f] == 1.0        # framed: path content taught on the same row

    # Name fields are unaffected either way.
    assert w_s[ids_s.index(scoped.marker_id("LBL"))] == 1.0
    assert w_f[ids_f.index(framed.marker_id("LBL"))] == 1.0


def test_frame_token_does_not_change_behaviour_when_disabled():
    """E4 must remain bit-identical until the frame is deliberately enabled."""
    from uie.tokenizer import URLTokenizer
    from uie.train import field_mask

    url = "https://a.co.uk/login/verify"
    tok = URLTokenizer(include_scheme=False, include_language=False)
    tok.fit([url], min_count=1)
    assert tok.include_frame is False
    ids = tok.encode(url, "benign")
    assert field_mask(ids, tok, "name") != field_mask(ids, tok, "structure")
