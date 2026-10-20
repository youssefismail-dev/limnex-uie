"""D1 -- uie/typicality.py, the public typicality scorer (research/d1/D1_PLAN.md section 1)."""
from dataclasses import asdict

import pytest
import torch

from uie.model import ModelConfig, URLTransformer
from uie.tokenizer import URLTokenizer
from uie.typicality import TypicalityScorer, is_idn, normalize_url

URLS = [
    "https://www.example.com/",
    "http://secure-login.example.net/verify?id=7",
    "shop.example.co.uk/products/item-42",
    "example.org",
    "https://a.b.c.example.com/x/y/z?q=1#frag",
    "xn--80ak6aa92e.com/abc",
    "blog.example.com/2024/03/review",
]


def _tok():
    tok = URLTokenizer(classes=("benign", "phishing"), include_frame=True, include_fields=True,
                       root_slash=True, presence_plan=True, slash_plan=True)
    tok.fit(URLS + ["ABCDEFGHIJKLMNOPQRSTUVWXYZ.com/abcdefghijklmnopqrstuvwxyz0123456789-_~%=&?#"])
    return tok


def _scorer(max_len=64, batch_size=64, seed=0):
    torch.manual_seed(seed)
    tok = _tok()
    cfg = ModelConfig(vocab_size=len(tok.stoi), d_model=32, n_layers=2, n_heads=2, d_ff=64,
                      max_len=max_len)
    return TypicalityScorer(URLTransformer(cfg), tok, batch_size=batch_size)


def test_test_tokenizer_has_a_real_alphabet():
    tok = _tok()
    unk = tok.stoi["<unk>"]
    assert unk not in tok.encode("example.com/Login", "benign", frame="domain")


def test_normalize_lowercases_the_host_only():
    assert normalize_url("  HTTPS://WWW.Example.COM/Login?User=X#Top ") == "HTTPS://www.example.com/Login?User=X#Top"
    assert normalize_url("Example.COM/Path") == "example.com/Path"
    assert normalize_url("http://User@Host.COM:8080/A") == "http://user@host.com:8080/A"
    assert normalize_url("example.com") == "example.com"


def test_idn_flag():
    assert is_idn("xn--80ak6aa92e.com/abc")
    assert is_idn("https://bücher.example/")
    assert not is_idn("https://example.com/xn--not-a-host")


def test_log_prob_is_the_models_sequence_log_prob():
    s = _scorer()
    for u in URLS:
        ids = s.tokenizer.encode(normalize_url(u), "benign", frame="domain")
        want = s.model.sequence_log_prob(torch.tensor([ids]), s.tokenizer.pad_id).item()
        assert s.log_prob([u], "benign")[0] == pytest.approx(want, abs=1e-5)


def test_uses_the_domain_frame():
    s = _scorer()
    ids = s.encode("example.com/a", "benign").ids
    assert s.tokenizer.stoi["<F:domain>"] in ids
    assert s.tokenizer.stoi["<F:page>"] not in ids


def test_batch_size_and_order_do_not_change_scores():
    one = _scorer(batch_size=1).log_prob(URLS, "phishing")
    many = _scorer(batch_size=64).log_prob(URLS, "phishing")
    three = _scorer(batch_size=3).log_prob(URLS, "phishing")
    assert many == pytest.approx(one, abs=1e-4)
    assert three == pytest.approx(one, abs=1e-4)
    rev = _scorer(batch_size=64).log_prob(URLS[::-1], "phishing")
    assert rev[::-1] == pytest.approx(one, abs=1e-4)


def test_deterministic_and_finite():
    s = _scorer()
    a, b = s.log_prob(URLS, "benign"), s.log_prob(URLS, "benign")
    assert a == b
    assert all(torch.isfinite(torch.tensor(a)))
    assert all(x < 0 for x in a)


def test_llr_is_phishing_minus_benign():
    s = _scorer()
    p, b = s.log_prob(URLS, "phishing"), s.log_prob(URLS, "benign")
    assert s.llr(URLS) == pytest.approx([x - y for x, y in zip(p, b)], abs=1e-9)


def test_host_case_does_not_change_the_score():
    s = _scorer()
    assert s.llr(["HTTPS://WWW.EXAMPLE.COM/Login"])[0] == pytest.approx(s.llr(["https://www.example.com/Login"])[0], abs=1e-6)
    # the path is case-sensitive and keeps its case
    assert s.encode("example.com/Login", "benign").ids != s.encode("example.com/login", "benign").ids


def test_long_urls_are_truncated_and_flagged_the_same_for_both_classes():
    s = _scorer(max_len=48)
    long_url = "example.com/" + "a" * 200
    eb, ep = s.encode(long_url, "benign"), s.encode(long_url, "phishing")
    assert eb.truncated and ep.truncated
    assert len(eb.ids) == len(ep.ids) == 48
    row = s.score([long_url, "example.com"])
    assert row[0]["truncated"] and not row[1]["truncated"]
    assert row[0]["tokens"] == 48


def test_score_rows_hold_both_classes_and_flags():
    s = _scorer()
    rows = s.score(URLS)
    assert len(rows) == len(URLS)
    for u, r in zip(URLS, rows):
        assert r["llr"] == pytest.approx(r["log_p_phishing"] - r["log_p_benign"])
        assert r["idn"] == is_idn(u)
    assert set(rows[0]) == {"llr", "log_p_phishing", "log_p_benign", "tokens", "truncated", "idn"}


def test_load_from_a_checkpoint_file(tmp_path):
    """The consumer path: a checkpoint on disk, loaded through the public class only."""
    s = _scorer(seed=3)
    blob = {"model_config": asdict(s.model.cfg), "model": s.model.state_dict(),
            "tokenizer": s.tokenizer.state_dict()}
    path = tmp_path / "final.pt"
    torch.save(blob, path)
    loaded = TypicalityScorer.load(path)
    assert loaded.classes == ("benign", "phishing")
    assert loaded.llr(URLS) == pytest.approx(s.llr(URLS), abs=1e-6)
