"""
Tests for the per-field likelihood breakdown.

`field_nll` is the instrument that overturned E2.1's original plan: aggregate
perplexity said the model had learned URLs well, while the per-field split
showed the registrable label memorised (train 1.201 / val 2.346). A capacity
increase would have made that worse. So the two things tested here are that
the attribution is CORRECT — a field's loss is charged to that field — and
that a real memorisation gap actually shows up as one.
"""
from __future__ import annotations

import math

import pytest
import torch

from uie.evaluation import field_nll
from uie.model import ModelConfig, URLTransformer
from uie.tokenizer import URLTokenizer

DEVICE = torch.device("cpu")


def _corpus(n: int, tag: str):
    return [f"https://{tag}site{i:04d}.com/articles/{i}/read?ref={i}"
            for i in range(n)]


@pytest.fixture(scope="module")
def fixture():
    urls = _corpus(120, "alpha") + _corpus(120, "beta")
    tok = URLTokenizer().fit(urls, min_count=1)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2,
                      n_heads=2, d_ff=64, max_len=128)
    torch.manual_seed(0)
    return URLTransformer(cfg).eval(), tok, cfg


def test_reports_content_fields_and_never_the_markers(fixture):
    model, tok, _ = fixture
    out = field_nll(model, tok, _corpus(120, "alpha"), "benign", DEVICE,
                    min_tokens=50)
    assert {"LBL", "PTH", "SUF"} <= set(out)
    assert not any(key.startswith("<") for key in out)
    assert all(math.isfinite(v) and v > 0 for v in out.values())


def test_untrained_model_scores_every_field_near_uniform(fixture):
    """The strongest check that attribution is not mismatched: an untrained
    model must sit at ~ln(vocab) on EVERY field. A field reading far lower
    would mean loss from one field is being charged to another."""
    model, tok, cfg = fixture
    out = field_nll(model, tok, _corpus(120, "alpha"), "benign", DEVICE,
                    min_tokens=50)
    uniform = math.log(cfg.vocab_size)
    for name, value in out.items():
        assert abs(value - uniform) < 0.8, (name, value, uniform)


def test_rare_fields_are_suppressed(fixture):
    """A field seen a handful of times would otherwise report a number with no
    error bar beside the path's hundreds of thousands."""
    model, tok, _ = fixture
    urls = ["https://example.com:8443/a"] + _corpus(60, "alpha")
    out = field_nll(model, tok, urls, "benign", DEVICE, min_tokens=50)
    assert "PRT" not in out          # one port, four characters
    assert "PTH" in out


def test_memorisation_shows_up_as_a_positive_gap():
    """Train a tiny model to convergence on a fixed set, then compare its loss
    on those rows against unseen ones. This is the signature the E2 diagnosis
    turned on, so it is asserted rather than assumed."""
    seen = _corpus(96, "alpha")
    unseen = _corpus(96, "gamma")
    tok = URLTokenizer().fit(seen + unseen, min_count=1)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=48, n_layers=2,
                      n_heads=2, d_ff=96, max_len=128)
    torch.manual_seed(0)
    model = URLTransformer(cfg)

    ids = [tok.encode(u, "benign") for u in seen]
    width = max(len(e) for e in ids)
    batch = torch.tensor([e + [tok.pad_id] * (width - len(e)) for e in ids])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = model.loss(batch, tok.pad_id)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()

    on_seen = field_nll(model, tok, seen, "benign", DEVICE, min_tokens=50)
    on_unseen = field_nll(model, tok, unseen, "benign", DEVICE, min_tokens=50)
    assert on_unseen["LBL"] > on_seen["LBL"], (on_seen["LBL"], on_unseen["LBL"])


def test_a_field_the_corpus_never_uses_is_absent(fixture):
    model, tok, _ = fixture
    out = field_nll(model, tok, _corpus(120, "alpha"), "benign", DEVICE,
                    min_tokens=50)
    assert "FRG" not in out          # no fragments anywhere in the corpus
