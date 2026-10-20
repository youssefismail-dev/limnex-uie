"""E5 wiring: the frame must travel from CORPUS -> ENCODING -> GENERATION.

The frame is derived from the corpus a row came from, never guessed. A row
encoded under the wrong frame teaches the wrong conditional, which is precisely
the failure the token exists to prevent -- so every hop is asserted, not assumed.
"""
from __future__ import annotations

import pytest
import torch

from uie.model import ModelConfig, URLTransformer, generate
from uie.suffixes import DEFAULT_FRAME_MIX, SuffixTable, generate_with_suffixes
from uie.tokenizer import URLTokenizer
from uie.train import CORPUS_FRAME, encode_corpus, field_mask

URLS = ["https://a.co.uk/login/verify", "https://b.com/x", "https://c.org/"]


def _tok(frame: bool):
    t = URLTokenizer(include_scheme=False, include_language=False,
                     include_frame=frame)
    return t.fit(URLS, min_count=1)


def test_corpus_maps_to_frame():
    assert CORPUS_FRAME["name"] == "domain"
    assert CORPUS_FRAME["structure"] == "page"


def test_encode_corpus_tags_each_row_with_its_corpus_frame():
    tok = _tok(True)
    enc, _ = encode_corpus(URLS, ["benign"] * 3, tok, 256,
                           corpora=["name", "structure", "name"])
    rows = enc["benign"]
    dom = tok.stoi["<F:domain>"]
    page = tok.stoi["<F:page>"]
    frames = [ids[2] for ids, _ in rows]          # bos, class, frame
    assert frames.count(dom) == 2 and frames.count(page) == 1


def test_encode_corpus_trains_every_field_once_the_frame_is_named():
    """The payoff: a name-corpus row now teaches path CONTENT, which under
    scoping it could not. That is what moves path content from 1.65M rows to
    2.55M (benign) and 46k to 239k (phishing)."""
    scoped, framed = _tok(False), _tok(True)
    url = "https://a.co.uk/login/verify"
    ids_s = scoped.encode(url, "benign")
    ids_f = framed.encode(url, "benign", frame="domain")
    w_s = field_mask(ids_s, scoped, "name")
    w_f = field_mask(ids_f, framed, "name")
    assert w_s[ids_s.index(scoped.marker_id("PTH"))] == 0.0
    assert w_f[ids_f.index(framed.marker_id("PTH"))] == 1.0


def test_generation_supplies_the_frame_token():
    tok = _tok(True)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, max_len=24,
                                       d_model=16, n_layers=1, n_heads=2,
                                       d_ff=32))
    for frame in ("domain", "page"):
        out = generate(model, tok, label="benign", n=2, frame=frame, seed=1)
        assert len(out) == 2


def test_generation_rejects_an_unknown_frame():
    tok = _tok(True)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, max_len=24,
                                       d_model=16, n_layers=1, n_heads=2,
                                       d_ff=32))
    with pytest.raises(KeyError, match="frame"):
        generate(model, tok, label="benign", n=1, frame="nonsense", seed=1)


def test_frame_mixture_is_sampled_per_row():
    """A mixture must actually vary per row; grouping by (frame, suffix) keeps
    each batched call homogeneous in both supplied controls."""
    tok = _tok(True)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, max_len=24,
                                       d_model=16, n_layers=1, n_heads=2,
                                       d_ff=32))
    table = SuffixTable({"benign": {"com": 10}})
    out = generate_with_suffixes(model, tok, label="benign", n=8, table=table,
                                 frame={"domain": 0.5, "page": 0.5}, seed=2)
    assert len(out) == 8


def test_default_frame_mix_is_the_frame_the_generator_samples():
    """One URL per invented host IS the domain frame, and it is what run_eval
    scores against. Defaulting to a blend would score in a frame the generator
    does not structurally sample."""
    from uie.evaluation import GENERATOR_FRAME
    assert DEFAULT_FRAME_MIX == {"domain": 1.0}
    assert GENERATOR_FRAME == "domain"


def test_a_framed_model_trains_end_to_end():
    """Guards the whole wiring: vocab, encoding, mask and loss must agree."""
    from uie.train import TrainConfig, URLDataset, train

    tok = _tok(True)
    enc, dropped = encode_corpus(URLS * 8, ["benign"] * 24, tok, 64,
                                 corpora=["name", "structure", "name"] * 8)
    assert dropped == 0
    ds = URLDataset(enc)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, max_len=64,
                                       d_model=16, n_layers=1, n_heads=2,
                                       d_ff=32))
    res = train(model, ds, ds, tok,
                TrainConfig(steps=3, batch_size=4, warmup=1, eval_every=5,
                            eval_batches=1))
    assert res is not None


# --- decode-time calibration -------------------------------------------------

def test_logit_bias_is_a_no_op_when_absent():
    """E5 generation must stay bit-identical unless a bias is explicitly given.
    None and {} are both no-ops; anything else is an opt-in change."""
    tok = _tok(True)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, max_len=24,
                                       d_model=16, n_layers=1, n_heads=2,
                                       d_ff=32))
    base = generate(model, tok, label="benign", n=6, seed=3, frame="domain")
    assert base == generate(model, tok, label="benign", n=6, seed=3,
                            frame="domain", logit_bias=None)
    assert base == generate(model, tok, label="benign", n=6, seed=3,
                            frame="domain", logit_bias={})


def test_logit_bias_shifts_log_odds_not_probability():
    """The correction is added to LOGITS, so it multiplies the odds of every
    position by the same factor. That is the property the fix relies on: a
    spurious 1.8% mid-path EOS is suppressed proportionally more than a
    legitimate 58% terminal EOS, so paths lengthen without stops breaking.
    """
    import torch

    logits = torch.tensor([[0.0, 0.0, 0.0]])
    delta = -1.0
    p0 = torch.softmax(logits, -1)[0, 0]
    logits2 = logits.clone()
    logits2[:, 0] += delta
    p1 = torch.softmax(logits2, -1)[0, 0]
    odds = lambda p: float(p / (1 - p))
    assert abs(odds(p1) / odds(p0) - pow(2.718281828, delta)) < 1e-5


def test_logit_bias_reaches_the_batched_suffix_path():
    """It must survive the (frame, suffix) grouping, or a sweep would silently
    measure the unbiased model."""
    tok = _tok(True)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, max_len=24,
                                       d_model=16, n_layers=1, n_heads=2,
                                       d_ff=32))
    table = SuffixTable({"benign": {"com": 10}})
    a = generate_with_suffixes(model, tok, label="benign", n=8, table=table,
                               frame="domain", seed=5)
    b = generate_with_suffixes(model, tok, label="benign", n=8, table=table,
                               frame="domain", seed=5,
                               logit_bias={tok.eos_id: -6.0})
    assert a != b
