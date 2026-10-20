"""The public interface (D2 step 4, research/d2/D2_PLAN.md): uie.load, Engine.generate (the
adopted decode, always gated), Engine.score, the command line, and the disclaimers' single text."""
import gzip
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import uie
from uie.api import ASSET_SHA256, ASSETS, Engine, _content_sha256, load, load_blocklist
from uie.bloom import BloomFilter
from uie.engine import generate_free
from uie.gate import Gate, normalize_for_dedup
from tests.test_typicality import _scorer

ROOT = Path(__file__).resolve().parents[1]
TINY_PS = {"benign": {"com": [1, 2, 1, 1, 3, 2], "org": [0, 1, 0, 2, 1, 1], "co.uk": [2, 0, 1, 0, 1, 0]},
           "phishing": {"com": [3, 1, 2, 2, 4, 3], "net": [1, 0, 1, 0, 2, 1]}}
TINY_SHIPPED = {"benign": {"com": 5, "org": 2, "co.uk": 1}, "phishing": {"com": 3, "net": 1}}


@pytest.fixture(scope="module")
def tiny():
    s = _scorer()
    s.model.eval()
    return s.model, s.tokenizer


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory, tiny):
    model, tok = tiny
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save({"model_config": asdict(model.cfg), "model": model.state_dict(), "tokenizer": tok.state_dict()}, path)
    return path


@pytest.fixture(scope="module")
def empty_bloom(tmp_path_factory):
    path = tmp_path_factory.mktemp("bloom") / "known.bloom.gz"
    BloomFilter(capacity=10, error_rate=0.01).save(path)
    return path


def test_packaged_tables_are_the_published_ones():
    for name, sha in ASSET_SHA256.items():
        assert hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() == sha


def test_packaged_decoder_equals_the_experiment_layer(tiny):
    for d in ("research/native", "research/e13", "research/e15", "research/e16", "research/e17"):
        if str(ROOT / d) not in sys.path:
            sys.path.insert(0, str(ROOT / d))
    import e12_common as E12
    import e13_common as E13
    import e17_common as E17
    model, tok = tiny
    for label, seed in (("benign", 11), ("phishing", 12)):
        research = E13.draw_and_generate(model, tok, label, 17, seed, E12.supplied_tables(TINY_PS, TINY_SHIPPED, label),
                                         lambda k: E17.decode_kwargs(True))
        assert generate_free(model, tok, label=label, n=17, seed=seed, presence_slash=TINY_PS,
                             shipped=TINY_SHIPPED) == research


def test_load_refuses_an_unpublished_checkpoint(tiny_ckpt):
    with pytest.raises(ValueError, match="not the published"):
        load(tiny_ckpt)
    assert isinstance(load(tiny_ckpt, verify=False), Engine)


def test_load_moves_the_model_to_the_requested_device(tiny_ckpt, monkeypatch):
    # a GPU run failed because the model stayed on the CPU (D2 step 6); no GPU here, so record the move
    moved = []
    real_to = torch.nn.Module.to
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: (moved.append(a[0] if a else k.get("device")),
                                                                     real_to(self, *a, **k))[1])
    load(tiny_ckpt, verify=False, device="cpu")
    assert torch.device("cpu") in [torch.device(d) for d in moved if d is not None]


def test_load_refuses_an_unpublished_known_url_filter(tiny_ckpt, tmp_path):
    fake = tmp_path / "known.bloom.gz"
    BloomFilter(capacity=10, error_rate=0.01).save(fake)      # an empty filter would let every copy through
    with pytest.raises(ValueError, match="not the published known-URL filter"):
        load(tiny_ckpt, known_urls=fake)                      # checked before the model, by content
    assert load(tiny_ckpt, verify=False, known_urls=fake).novelty_checked
    plain = tmp_path / "known.bloom"                          # decompressed on the way, as Kaggle does
    plain.write_bytes(gzip.decompress(fake.read_bytes()))
    assert _content_sha256(plain) == _content_sha256(fake)


def test_generate_is_gated_deterministic_and_says_what_it_did_not_check(tiny_ckpt, empty_bloom):
    bare = load(tiny_ckpt, verify=False)
    for label in ("benign", "phishing"):                  # never without the copy filter, either class
        with pytest.raises(ValueError, match="known-URL filter"):
            bare.generate(label, n=5)
    e = load(tiny_ckpt, verify=False, known_urls=empty_bloom)
    a = e.generate("benign", n=30, seed=3)
    assert a == e.generate("benign", n=30, seed=3) and e.last_seed == 3
    assert e.last_report.n_total == 30 and len(a) == e.last_report.n_passed
    assert all(Gate().check(u).passed for u in a)
    assert "novelty checked" in e.report_text() and "live threats NOT checked" in e.report_text()
    e.generate("phishing", n=5)
    assert isinstance(e.last_seed, int)
    with pytest.raises(ValueError):
        e.generate("spam", n=5)
    with pytest.raises(ValueError):
        e.generate("benign", n=0)


def test_novelty_and_threat_screens_remove_what_they_should(tiny, tiny_ckpt):
    model, tok = tiny
    from uie.factorization import parse
    kept = Engine(model, tok, known_urls=BloomFilter(capacity=10, error_rate=0.01)).generate("benign", n=60, seed=5)
    # the threat screen keys on the registrable domain, so its target must have one (the tiny
    # model also writes hosts that are a bare suffix, which no domain blocklist can name)
    with_domain = [u for u in kept if parse(u).registrable_domain]
    assert len(with_domain) >= 2
    copied, threat = with_domain[0], with_domain[1]
    domain = parse(threat).registrable_domain
    bloom = BloomFilter(capacity=100, error_rate=0.001)
    bloom.add(normalize_for_dedup(copied))
    e = Engine(model, tok, known_urls=bloom)
    again = e.generate("benign", n=60, seed=5, blocked_domains=[domain])
    assert copied not in again and threat not in again
    assert all(parse(u).registrable_domain != domain for u in again)
    assert e.last_report.by_reason["memorized"] >= 1 and e.last_report.by_reason["active_threat"] >= 1
    assert "novelty checked" in e.report_text() and "live threats checked" in e.report_text()


def test_load_blocklist_reduces_every_entry_to_its_registrable_domain(tmp_path):
    f = tmp_path / "feed.txt"
    f.write_text("# a snapshot\n\nevil.tk\nlogin.evil.tk\nhttps://a.b.example.co.uk/x?y=1\n", encoding="utf-8")
    assert load_blocklist(f) == ["evil.tk", "evil.tk", "example.co.uk"]


def test_score_is_the_typicality_scorer(tiny, tiny_ckpt):
    from uie.typicality import TypicalityScorer
    model, tok = tiny
    urls = ["example.com/login", "https://shop.example.co.uk/a?b=1"]
    rows = load(tiny_ckpt, verify=False).score(urls)
    direct = TypicalityScorer(model, tok).score(urls)
    assert [r["url"] for r in rows] == urls
    assert [r["llr"] for r in rows] == pytest.approx([d["llr"] for d in direct], abs=1e-6)


def test_command_line(tiny_ckpt, empty_bloom):
    run = lambda *a: subprocess.run([sys.executable, "-m", "uie", *a], cwd=ROOT, capture_output=True,  # noqa: E731
                                    text=True, encoding="utf-8")
    g = run("generate", "--weights", str(tiny_ckpt), "--no-verify", "--known-urls", str(empty_bloom),
            "--label", "benign", "-n", "8", "--seed", "2")
    assert g.returncode == 0, g.stderr
    assert "defensive research only" in g.stderr and "kept" in g.stderr and "--blocklist" in g.stderr
    assert len(g.stdout.splitlines()) <= 8
    s = run("score", "--weights", str(tiny_ckpt), "--no-verify", "example.com/a", "example.org")
    assert s.returncode == 0, s.stderr
    rows = [json.loads(l) for l in s.stdout.splitlines()]
    assert [r["url"] for r in rows] == ["example.com/a", "example.org"] and all("llr" in r for r in rows)
    refused = run("generate", "--weights", str(tiny_ckpt), "--label", "benign")            # unverified file
    assert refused.returncode == 2 and "not the published" in refused.stderr and "Traceback" not in refused.stderr
    for label in ("phishing", "benign"):
        no_filter = run("generate", "--weights", str(tiny_ckpt), "--no-verify", "--label", label)
        assert no_filter.returncode == 2 and "known-URL filter" in no_filter.stderr and not no_filter.stdout


def test_import_uie_is_light():
    out = subprocess.run([sys.executable, "-c", "import uie, sys; print('torch' in sys.modules); print(callable(uie.load))"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.split() == ["False", "True"], out.stderr


def _between_markers(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RESPONSIBLE-USE:BEGIN -->", "<!-- RESPONSIBLE-USE:END -->"
    assert text.count(start) == 1 and text.count(end) == 1, path
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def test_responsible_use_and_no_liability_are_one_text_in_readme_and_model_card():
    readme, card = _between_markers(ROOT / "README.md"), _between_markers(ROOT / "MODEL_CARD.md")
    assert readme == card
    for must in ("## Responsible Use", "## No Liability", "AS IS", "Apache License"):
        assert must in readme
