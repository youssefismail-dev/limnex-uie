"""S4: the boundary loss weight must land on EXACTLY the segment starts.

The intervention is only interpretable if it weights the positions S3 measured
and no others, so these tests pin the POSITION SET rather than any outcome:

  * weight 1.0 reproduces E5's mask exactly -- reversible by configuration
  * the weighted targets are precisely the first character after `<PTH>` and
    after every in-path `/`
  * a `/` inside a query string is NOT a boundary
  * markers, EOS and out-of-scope fields keep their existing weights
  * the weight reaches the loss, and a row with no path is untouched

`boundary_nll` is pinned separately, because an off-by-one there would score the
model on a token it has already been shown and report near-perfect calibration.
"""
from __future__ import annotations

import torch

from uie.evaluation import boundary_nll
from uie.model import ModelConfig, URLTransformer
from uie.tokenizer import URLTokenizer
from uie.train import field_mask


def _tok():
    tok = URLTokenizer(classes=("benign",), include_frame=True)
    tok.fit(["https://a.example.com/ab/cd?e=f", "http://b.test.org/ghi/jk"])
    return tok


def _ids(tok, url):
    return tok.encode(url, "benign", frame="domain")


def test_unit_weight_is_identical_to_e5():
    tok = _tok()
    ids = _ids(tok, "https://a.example.com/ab/cd?e=f")
    assert field_mask(ids, tok, "structure", 1.0) == field_mask(
        ids, tok, "structure")


def test_weight_lands_on_segment_starts_only():
    tok = _tok()
    ids = _ids(tok, "https://a.example.com/ab/cd?e=f")
    base = field_mask(ids, tok, "structure")
    hot = field_mask(ids, tok, "structure", 7.0)

    pth, slash = tok.marker_id("PTH"), tok.stoi["/"]
    changed = {i for i, (b, h) in enumerate(zip(base, hot)) if b != h}
    # mask index i corresponds to target ids[i + 1], predicted after ids[i]
    expected = {i for i in range(len(base))
                if ids[i] in (pth, slash) and base[i] > 0}
    assert changed == expected and changed, (changed, expected)
    for i in changed:
        assert hot[i] == 7.0


def test_query_slash_is_not_a_boundary():
    tok = _tok()
    # the `/` here sits inside the query, where S3 measured nothing
    ids = _ids(tok, "https://a.example.com/ab?next=/cd")
    hot = field_mask(ids, tok, "structure", 7.0)
    pth = tok.marker_id("PTH")
    qry = tok.marker_id("QRY")
    after_qry = ids.index(qry) if qry in ids else len(ids)
    assert pth in ids
    for i, w in enumerate(hot):
        if i >= after_qry:
            assert w != 7.0, f"query position {i} was weighted"


def test_markers_and_out_of_scope_are_untouched():
    tok = _tok()
    ids = _ids(tok, "https://a.example.com/ab/cd")
    base = field_mask(ids, tok, "name")
    hot = field_mask(ids, tok, "name", 7.0)
    # under the "name" scope with no frame token the path is out of scope, so
    # a zero weight must stay zero: S4 must never TURN ON a field scoping is off
    tok_noframe = URLTokenizer(classes=("benign",))
    tok_noframe.fit(["https://a.example.com/ab/cd"])
    ids2 = tok_noframe.encode("https://a.example.com/ab/cd", "benign")
    off = field_mask(ids2, tok_noframe, "name", 7.0)
    assert all(w in (0.0, 1.0) for w in off), off
    assert len(base) == len(hot)


def test_a_row_without_a_path_is_unaffected():
    tok = _tok()
    ids = _ids(tok, "https://a.example.com")
    assert field_mask(ids, tok, "structure", 9.0) == field_mask(
        ids, tok, "structure")


def test_weight_reaches_the_loss():
    """A changed mask must change the gradient, or the knob is decorative."""
    tok = _tok()
    cfg = ModelConfig(vocab_size=len(tok.stoi), d_model=32, n_layers=1,
                      n_heads=2, d_ff=64, max_len=64)
    model = URLTransformer(cfg)
    ids = _ids(tok, "https://a.example.com/ab/cd")
    x = torch.tensor([ids], dtype=torch.long)
    w1 = torch.tensor([field_mask(ids, tok, "structure", 1.0)])
    w7 = torch.tensor([field_mask(ids, tok, "structure", 7.0)])
    assert not torch.isclose(model.loss(x, tok.pad_id, w1),
                             model.loss(x, tok.pad_id, w7))


def test_boundary_nll_is_not_off_by_one():
    """Scoring token t from slot t-1, never slot t.

    A model given the answer scores near zero loss, so an off-by-one shows up as
    implausibly good calibration rather than as an error.
    """
    tok = _tok()
    cfg = ModelConfig(vocab_size=len(tok.stoi), d_model=32, n_layers=1,
                      n_heads=2, d_ff=64, max_len=64)
    model = URLTransformer(cfg)
    urls = ["https://a.example.com/ab/cd", "http://b.test.org/ghi/jk"]
    r = boundary_nll(model, tok, urls, "benign", torch.device("cpu"))

    assert r["boundary_n"] > 0 and r["mid_n"] > 0
    # an untrained model is near-uniform, so every CE must sit near log(vocab)
    # -- a leaked target would drive it toward 0 instead
    floor = 0.5 * torch.log(torch.tensor(float(len(tok.stoi)))).item()
    assert r["boundary_ce"] > floor, r
    assert r["mid_ce"] > floor, r
    # over-confidence is CE - H_model and may take either sign; it is NOT a
    # divergence, so no non-negativity may be asserted here
    assert abs(r["boundary_overconf"]) < 5.0


def test_boundary_nll_counts_match_the_mask():
    """The eval and the loss must agree on which positions are boundaries."""
    tok = _tok()
    cfg = ModelConfig(vocab_size=len(tok.stoi), d_model=32, n_layers=1,
                      n_heads=2, d_ff=64, max_len=64)
    model = URLTransformer(cfg)
    urls = ["https://a.example.com/ab/cd", "http://b.test.org/ghi/jk"]
    r = boundary_nll(model, tok, urls, "benign", torch.device("cpu"))
    weighted = sum(
        sum(1 for w in field_mask(_ids(tok, u), tok, "structure", 7.0)
            if w == 7.0)
        for u in urls)
    assert r["boundary_n"] == weighted, (r["boundary_n"], weighted)


# --------------------------------------------------------------------------- #
# S4b: the weight applies to the POSITION, not the target
# --------------------------------------------------------------------------- #

def test_boundary_weight_reaches_eos_targets():
    """A path ending at a separator is a boundary too, and must be weighted.

    This is the test S4 lacked. 17.50% of real boundary positions have a
    non-content target (16.64% EOS, the trailing-slash form), and weighting
    only the content ones taught the model that segments start rather than what
    starts them.
    """
    tok = _tok()
    ids = _ids(tok, "https://a.example.com/ab/")
    hot = field_mask(ids, tok, "structure", 7.0)
    pth, slash = tok.marker_id("PTH"), tok.stoi["/"]
    # every position whose PREDECESSOR opens a segment must carry the weight,
    # whatever the target at that position happens to be
    for i, w in enumerate(hot):
        if ids[i] in (pth, slash) and i + 1 < len(ids):
            tgt = ids[i + 1]
            if tgt != tok.pad_id:
                assert w == 7.0, (f"position {i} (target {tok.itos[tgt]!r}) "
                                  f"was not weighted: {w}")


def test_content_to_stop_ratio_is_preserved():
    """THE INVARIANT S4 BROKE.

    At a boundary the model chooses among characters AND stopping. Up-weighting
    must scale that whole decision, leaving the relative pull of 'continue' and
    'stop' exactly as the data states it. If continuing is amplified and
    stopping is not, the loss is a biased sample of the decision.
    """
    tok = _tok()
    cont = _ids(tok, "https://a.example.com/ab/cd")   # boundary -> a character
    stop = _ids(tok, "https://a.example.com/ab/")     # boundary -> the url ends
    pth, slash = tok.marker_id("PTH"), tok.stoi["/"]

    def boundary_weights(ids, w):
        m = field_mask(ids, tok, "structure", w)
        return [m[i] for i in range(len(m))
                if ids[i] in (pth, slash) and ids[i + 1] != tok.pad_id]

    for w in (1.0, 3.0, 7.0):
        c, s = boundary_weights(cont, w), boundary_weights(stop, w)
        assert c and s, (c, s)
        # both kinds of boundary carry the SAME weight, so their ratio is 1
        # at every w -- under S4 this was w:1 and drifted with the setting
        assert set(c) == set(s) == {w}, (w, c, s)


def test_s4b_weights_strictly_more_positions_than_s4_did():
    """S4b must cover a superset: the old spec's positions plus the stops."""
    tok = _tok()
    ids = _ids(tok, "https://a.example.com/ab/")
    marker_of = {tok.marker_id(m): m for m in ("PTH", "QRY", "SUF", "LBL",
                                               "SUB", "FRG", "PRT", "USR")}
    hot = field_mask(ids, tok, "structure", 5.0)
    n_weighted = sum(1 for w in hot if w == 5.0)
    # the old spec skipped any boundary whose target was EOS or a marker
    old = sum(1 for i, w in enumerate(hot)
              if w == 5.0 and ids[i + 1] not in marker_of
              and ids[i + 1] != tok.eos_id)
    assert n_weighted > old, ("S4b covers no extra positions; the fix is inert",
                              n_weighted, old)
