"""V2-R1 (research/v2/PREREG_R1.md): the registered rules and transforms, checked before any
reading runs. No data is loaded."""
import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def r1():
    sys.path.insert(0, str(ROOT / "research" / "v2"))
    spec = importlib.util.spec_from_file_location("v2_r1", ROOT / "research" / "v2" / "v2_r1.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v2_r1"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_n_f_rule(r1):
    assert r1.n_f([10_000, 9_950], [3_400, 3_350]) == 2_400       # a quarter of the real pool
    assert r1.n_f([90_000], [600, 590]) == 500                      # 0.9 x the fewest generated
    assert r1.n_f([10 ** 6], [10 ** 5]) == 6_000                    # the cap
    assert r1.n_f([1_000], [10_000]) == 200 and 200 < r1.N_MIN      # below 300: not measurable


def test_seeds_are_fixed_and_distinct(r1):
    seeds = set()
    for f in r1.FIELDS_R1:
        for lab in r1.LABELS:
            for g in range(r1.G):
                seeds.add(r1.seed_real(f, lab, g))
                for src in r1.SOURCES:
                    seeds.add(r1.seed_source(src, f, lab, g))
    for lab in r1.LABELS:
        for g in range(r1.G):
            seeds.add(r1.seed_shuffle(lab, g))
    assert len(seeds) == 4 * 2 * 5 * 3 + 2 * 5
    assert r1.seed_real("LBL", "benign", 0) == 20261030
    assert r1.seed_source("e15_new", "QRY", "phishing", 4) == 20261030 + 200_000 + 3_000 + 100 + 4


def test_d1_strings_isolate_the_field(r1):
    import strong_disc as SD
    assert r1.d1_string("LBL", "shopwave") == "shopwave"
    assert r1.d1_string("SUB", "login.secure") == "login.secure"
    assert r1.d1_string("PTH", "a/b.php") == "h.h/a/b.php"
    assert r1.d1_string("QRY", "id=7&next=home") == "h.h?id=7&next=home"
    # the constant host holds no word, so every word D1 sees in PTH and QRY is the field's
    assert SD.TOKEN.findall(r1.d1_string("QRY", "next=home")) == ["next", "home"]
    host, path, query, frag = SD.split_url(r1.d1_string("PTH", "login/verify"))
    assert host == "h.h" and path == "/login/verify" and query is None


def test_shuffle_words_keeps_words_and_separators(r1):
    import strong_disc as SD
    rng = random.Random(3)
    url = "secure.brandx-login.com/account/verify/update?next=home&lang=english"
    seen = set()
    for _ in range(50):
        out = r1.shuffle_words(url, rng)
        seen.add(out)
        assert len(out) == len(url)
        assert sorted(SD.TOKEN.findall(out)) == sorted(SD.TOKEN.findall(url))     # same words
        assert [c for c in out if not c.isalpha()] == [c for c in url if not c.isalpha()]
        h_out, h_in = SD.split_url(out)[0], SD.split_url(url)[0]
        assert sorted(SD.TOKEN.findall(h_out)) == sorted(SD.TOKEN.findall(h_in))  # host stays host
    assert len(seen) > 1                                                           # order changes
    assert r1.shuffle_words("abc.io/x1", random.Random(1)) == "abc.io/x1"        # under 2 words a part


def test_d4_tokens_and_features(r1):
    assert r1.d4_tokens("Shop.COM/item/12345?id=7") == ["shop", ".", "com", "/", "item", "/", "0",
                                                        "?", "id", "=", "0"]
    feats = r1.d4_features("a.b")
    assert "a" in feats and "a\x01." in feats and "a\x01.\x01b" in feats and len(feats) == 3 + 2 + 1


def test_edit1_is_exactly_levenshtein_one(r1):
    from qc1_common import levenshtein_le1
    for w in ("cat", "login", "aa"):
        v = r1.edit1(w)
        assert w not in v
        assert all(levenshtein_le1(w, x) for x in v)
        assert all(set(x) <= set(r1._AZ) for x in v)
    # every one-letter edit of "cat" over a-z is there
    assert {"at", "ct", "ca", "bat", "cut", "caat", "cats", "scat"} <= r1.edit1("cat")


def test_anatomy(r1):
    vocab = {"login": 50, "secure": 40, "shop": 30, "wave": 12, "rare": 3}

    def counts_of(words):
        return np.array([vocab.get(w, 0) for w in words])
    assert r1.anatomy("logim", counts_of) == "near"          # one substitution from a word seen 50 times
    assert r1.anatomy("shopwave", counts_of) == "join"       # two words, each seen at least 10 times
    assert r1.anatomy("rarex", counts_of) == "other"         # its neighbour is seen only 3 times
    assert r1.anatomy("zzqqx", counts_of) == "other"


def test_bands(r1):
    thr = (1000, 100, 10)
    assert [r1.band_of(c, thr) for c in (0, 1, 10, 99, 100, 999, 1000, 5000)] == [
        "unseen", "beyond", "8k-32k", "8k-32k", "1k-8k", "1k-8k", "top1k", "top1k"]


def test_shortfall_reading(r1):
    real = {"top1k": 0.5, "1k-8k": 0.2, "8k-32k": 0.1, "beyond": 0.12, "unseen": 0.08}
    common = {"top1k": 0.40, "1k-8k": 0.15, "8k-32k": 0.1, "beyond": 0.12, "unseen": 0.23}
    out = r1.shortfall(real, common)
    assert out["e"] == pytest.approx(0.15) and sum(out["d"].values()) == pytest.approx(out["e"])
    assert out["S_common"] == pytest.approx(1.0) and out["reading"] == "COMMON-WORD SHORTFALL"
    rare = {"top1k": 0.5, "1k-8k": 0.19, "8k-32k": 0.05, "beyond": 0.03, "unseen": 0.23}
    assert r1.shortfall(real, rare)["reading"] == "RARE-WORD SHORTFALL"
    assert r1.shortfall(real, dict(real))["reading"] == "NO SHORTFALL"
