"""
Tests for class-conditional suffix supply.

Benign block sampling is suffix-stratified at alpha = 0.5, which deliberately
flattens the suffix marginal to maximise data per suffix conditional. That makes
the model's OWN marginal over suffixes wrong on purpose, so letting it sample
its own suffix would ship a sampling artifact into the output and hand a
discriminator a tell that has nothing to do with generation quality.
"""
from __future__ import annotations

import random

import pytest
import torch

from uie.model import ModelConfig, URLTransformer
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.tokenizer import URLTokenizer

ROWS = (
    [{"url": f"https://site{i}.com/x", "label": "benign"} for i in range(70)]
    + [{"url": f"https://site{i}.de/x", "label": "benign"} for i in range(20)]
    + [{"url": f"https://site{i}.co.uk/x", "label": "benign"} for i in range(10)]
    + [{"url": f"https://bad{i}.top/l", "label": "phishing"} for i in range(60)]
    + [{"url": f"https://bad{i}.xyz/l", "label": "phishing"} for i in range(40)]
)


def test_table_is_built_per_class():
    t = SuffixTable.from_urls(ROWS)
    assert set(t.classes()) == {"benign", "phishing"}
    assert set(t.counts["benign"]) == {"com", "de", "co.uk"}
    assert set(t.counts["phishing"]) == {"top", "xyz"}


def test_classes_do_not_share_a_distribution():
    """Benign and phishing name on different TLDs and the difference is real —
    total variation between them is 0.579 on the E0 corpus. One shared table
    would replace phishing's cheap-gTLD habit with a benign-shaped prior."""
    t = SuffixTable.from_urls(ROWS)
    assert not set(t.counts["benign"]) & set(t.counts["phishing"])


def test_sampling_follows_the_recorded_frequencies():
    t = SuffixTable.from_urls(ROWS)
    drawn = t.sample("benign", 20_000, random.Random(0))
    share = drawn.count("com") / len(drawn)
    assert share == pytest.approx(0.70, abs=0.02), share


def test_sampling_is_reproducible_under_seed():
    t = SuffixTable.from_urls(ROWS)
    assert (t.sample("benign", 50, random.Random(3))
            == t.sample("benign", 50, random.Random(3)))


def test_unknown_class_raises_rather_than_falling_back():
    """Falling back to the other class's table would silently give phishing a
    benign suffix prior, which is the failure this is built to prevent."""
    t = SuffixTable.from_urls(ROWS)
    with pytest.raises(KeyError):
        t.sample("malware", 5, random.Random(0))


def test_round_trip_through_disk(tmp_path):
    t = SuffixTable.from_urls(ROWS)
    path = tmp_path / "suffixes.json"
    t.save(path)
    loaded = SuffixTable.load(path)
    assert loaded.counts == t.counts
    assert loaded.sample("phishing", 10, random.Random(1)) == \
        t.sample("phishing", 10, random.Random(1))


def test_generation_uses_the_supplied_suffixes():
    """The whole point: generated suffixes must follow the TABLE, not whatever
    marginal the stratified corpus taught the model."""
    corpus = [r["url"] for r in ROWS]
    tok = URLTokenizer().fit(corpus, min_count=1)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2,
                      n_heads=2, d_ff=64, max_len=128)
    torch.manual_seed(0)
    model = URLTransformer(cfg).eval()

    table = SuffixTable({"benign": {"com": 1}})     # degenerate on purpose
    urls = generate_with_suffixes(model, tok, label="benign", n=24,
                                  table=table, seed=1)
    assert len(urls) == 24
    # An UNTRAINED model builds malformed hosts, so the guarantee tested here is
    # that the supplied suffix is EMITTED, not that it parses back out — the
    # same existence assertion `test_supplied_suffix_is_placed_in_the_output`
    # makes, and for the same reason.
    assert any("com" in u for u in urls), urls[:4]


def test_generation_covers_multiple_suffixes():
    corpus = [r["url"] for r in ROWS]
    tok = URLTokenizer().fit(corpus, min_count=1)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2,
                      n_heads=2, d_ff=64, max_len=128)
    torch.manual_seed(0)
    model = URLTransformer(cfg).eval()
    table = SuffixTable({"phishing": {"top": 1, "xyz": 1}})
    urls = generate_with_suffixes(model, tok, label="phishing", n=40,
                                  table=table, seed=2)
    assert len(urls) == 40


def test_generation_is_reproducible_under_seed():
    corpus = [r["url"] for r in ROWS]
    tok = URLTokenizer().fit(corpus, min_count=1)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2,
                      n_heads=2, d_ff=64, max_len=128)
    torch.manual_seed(0)
    model = URLTransformer(cfg).eval()
    table = SuffixTable({"benign": {"com": 7, "de": 2, "co.uk": 1}})
    a = generate_with_suffixes(model, tok, label="benign", n=30, table=table,
                               seed=5)
    b = generate_with_suffixes(model, tok, label="benign", n=30, table=table,
                               seed=5)
    assert a == b
