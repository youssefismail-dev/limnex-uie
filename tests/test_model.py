"""
Tests for the transformer and its decoding.

These run against a TINY untrained model. That is deliberate: what is being
tested is the generation machinery — shapes, decoder contracts, batching
equivalence, and the field-injection rule — none of which depend on the weights
being any good. Sample quality is measured in E3, not asserted here.
"""
from __future__ import annotations

import pytest
import torch

from uie.factorization import parse
from uie.model import DECODERS, ModelConfig, URLTransformer, generate
from uie.tokenizer import URLTokenizer

CORPUS = [
    "https://blog.shopwave.io/posts/2024/review?page=2",
    "http://secure-verify.bankx.support/login",
    "https://a.co.uk/x",
    "http://brandx.com@evil.tk:8080/a",
    "https://news.tenjiku.jp/2024/03/article",
    "a.com/no/scheme",
]


@pytest.fixture(scope="module")
def fixture():
    tok = URLTokenizer().fit(CORPUS, min_count=1)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2,
                      n_heads=2, d_ff=64, max_len=128)
    torch.manual_seed(0)
    model = URLTransformer(cfg).eval()
    return model, tok, cfg


# --- shapes and likelihood -------------------------------------------------

def test_forward_shape(fixture):
    model, tok, cfg = fixture
    ids = torch.tensor([tok.encode(CORPUS[0], "benign")])
    assert model(ids).shape == (1, ids.shape[1], cfg.vocab_size)


def test_sequence_beyond_max_len_is_rejected(fixture):
    """Silently truncating would score a different string than the caller asked
    about, which corrupts perplexity rather than reporting a problem."""
    model, tok, cfg = fixture
    with pytest.raises(ValueError):
        model(torch.zeros((1, cfg.max_len + 1), dtype=torch.long))


def test_untrained_loss_is_near_uniform(fixture):
    """An untrained model must score ~ln(vocab). A much lower value would mean
    information is leaking from the targets into the inputs."""
    import math
    model, tok, cfg = fixture
    ids = torch.tensor([tok.encode(u, "benign")[:40] + [tok.pad_id] * (40 - len(tok.encode(u, "benign")[:40]))
                        for u in CORPUS])
    loss = float(model.loss(ids, tok.pad_id))
    assert abs(loss - math.log(cfg.vocab_size)) < 0.6


def test_sequence_log_prob_is_negative_and_finite(fixture):
    model, tok, _ = fixture
    ids = torch.tensor([tok.encode(CORPUS[0], "benign")])
    value = float(model.sequence_log_prob(ids, tok.pad_id)[0])
    assert value < 0 and value > -1e6


def test_padding_does_not_change_log_prob(fixture):
    """Padded positions are masked out, so a batch must score each row the same
    as it would alone — otherwise batch composition changes the metric."""
    model, tok, _ = fixture
    ids = tok.encode(CORPUS[2], "benign")
    alone = torch.tensor([ids])
    padded = torch.tensor([ids + [tok.pad_id] * 7])
    a = float(model.sequence_log_prob(alone, tok.pad_id)[0])
    b = float(model.sequence_log_prob(padded, tok.pad_id)[0])
    assert a == pytest.approx(b, abs=1e-4)


# --- decoders --------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(DECODERS))
def test_every_decoder_returns_a_valid_token(fixture, name):
    _, tok, cfg = fixture
    logits = torch.randn(cfg.vocab_size)
    token = DECODERS[name](logits, torch.Generator(), )
    assert 0 <= token < cfg.vocab_size


def test_greedy_is_deterministic(fixture):
    _, _, cfg = fixture
    logits = torch.randn(cfg.vocab_size)
    picks = {DECODERS["greedy"](logits, torch.Generator()) for _ in range(5)}
    assert len(picks) == 1
    assert picks.pop() == int(torch.argmax(logits))


def test_top_p_never_selects_outside_the_nucleus():
    """A peaked distribution must confine top-p to the peak."""
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])
    picks = {DECODERS["top_p"](logits, torch.Generator(), p=0.5) for _ in range(20)}
    assert picks == {0}


def test_typical_avoids_the_mode_on_a_flat_tail():
    """Typical sampling targets tokens near the distribution's expected
    information content, so on a spiked distribution it should not behave
    identically to greedy."""
    torch.manual_seed(0)
    logits = torch.tensor([6.0] + [1.0] * 40)
    g = torch.Generator().manual_seed(0)
    picks = [DECODERS["typical"](logits, g, mass=0.95) for _ in range(40)]
    assert len(set(picks)) > 1


def test_unknown_decoder_raises(fixture):
    model, tok, _ = fixture
    with pytest.raises(KeyError):
        generate(model, tok, label="benign", n=1, decoder="nope")


# --- generation ------------------------------------------------------------

def test_generated_urls_round_trip(fixture):
    model, tok, _ = fixture
    for url in generate(model, tok, label="benign", n=8, seed=1):
        assert parse(url) is not None            # never raises


def test_generation_is_deterministic_under_seed(fixture):
    model, tok, _ = fixture
    a = generate(model, tok, label="benign", n=6, seed=3)
    b = generate(model, tok, label="benign", n=6, seed=3)
    assert a == b


def test_unknown_class_raises(fixture):
    model, tok, _ = fixture
    with pytest.raises(KeyError):
        generate(model, tok, label="malware", n=1)


def test_unknown_language_falls_back(fixture):
    model, tok, _ = fixture
    assert len(generate(model, tok, label="benign", n=2, language="klingon", seed=1)) == 2


def test_batching_produces_the_same_count(fixture):
    model, tok, _ = fixture
    for bs in (1, 3, 16):
        assert len(generate(model, tok, label="benign", n=7, seed=1, batch_size=bs)) == 7


# --- supplied suffix: the regression that AUC 1.0000 exposed ---------------

def test_supplied_suffix_is_placed_in_the_output(fixture):
    """The injection guarantee is that the suffix is EMITTED, not that it parses
    back out.

    These tests run an untrained model, which wraps the suffix in malformed
    hosts (`ghkv/yonb.i.com` — the `/` ends the authority early), so
    `parse().public_suffix` is empty even though injection worked. Asserting
    structural recovery here would be testing the weights, not the mechanism;
    a trained model does recover it, which E3 measures.

    An EXISTENCE assertion, not a universal one: an untrained model frequently
    emits the end token before reaching the suffix field at all, and injection
    correctly does nothing when the model never asks for a suffix.
    """
    model, tok, _ = fixture
    for suffix in ("com", "co.uk", "jp"):
        urls = generate(model, tok, label="benign", n=24, seed=2, suffix=suffix)
        assert any(suffix in url for url in urls), urls[:4]


def test_supplied_suffix_does_not_suppress_the_scheme(fixture):
    """Regression: seeding the PREFIX with the suffix skipped the `[SCH]` field,
    because `[SCH]` precedes `[SUF]` in the sequence. Every sample came out
    scheme-less and a discriminator separated real from generated perfectly
    (measured AUC 1.0000) on that artefact alone. The suffix must be injected
    when the model REACHES that field, not prepended before it."""
    model, tok, _ = fixture
    free = generate(model, tok, label="benign", n=40, seed=5)
    supplied = generate(model, tok, label="benign", n=40, seed=5, suffix="com")
    free_rate = sum(parse(u).has_scheme for u in free) / len(free)
    supplied_rate = sum(parse(u).has_scheme for u in supplied) / len(supplied)
    # The scheme rate must not COLLAPSE relative to free generation.
    assert supplied_rate > free_rate - 0.25, (free_rate, supplied_rate)


def test_supplied_suffix_still_varies_the_rest(fixture):
    """Forcing one field must not freeze the others."""
    model, tok, _ = fixture
    urls = generate(model, tok, label="benign", n=12, seed=6, suffix="com")
    assert len(set(urls)) > 1
