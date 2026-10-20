"""V2-P1 step R0 (research/v2/PREREG_V2_P1.md section 15.3): the transforms, the N_R rule and
the seeds, checked before the reading runs. No data is loaded."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def r0():
    spec = importlib.util.spec_from_file_location("v2_r0", ROOT / "research" / "v2" / "v2_r0.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v2_r0"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("url, expected", [
    ("a.com", False), ("a.com/", False), ("a.com?", False), ("a.com#", False),
    ("a.com/x", True), ("a.com//", True), ("a.com?q=1", True), ("a.com/?q=1", True),
    ("a.com#top", True), ("sub.a.co.uk/p/q.html", True),
])
def test_rest_bearing(r0, url, expected):
    assert r0.rest_bearing(url) is expected


def test_mask_host_keeps_everything_after_the_host(r0):
    assert r0.mask_host("login.secure.bank.com/a/b.php?id=7&x=y#frag") == "h.h/a/b.php?id=7&x=y#frag"
    assert r0.mask_host("bank.com?q=1") == "h.h?q=1"
    assert r0.mask_host("bank.com#f") == "h.h#f"


def test_the_mask_holds_no_word(r0):
    import strong_disc as SD
    assert SD.TOKEN.findall(r0.MASK.lower()) == []
    # so every word the detector sees in a masked URL comes from its path, query or fragment
    masked = r0.mask_host("brandx-verify.com/account/login?next=home")
    assert SD.TOKEN.findall(masked.lower()) == ["account", "login", "next", "home"]


def test_host_only(r0):
    assert r0.host_only("login.secure.bank.com/a?b#c") == "login.secure.bank.com"
    assert r0.host_only("bank.com") == "bank.com"


def test_n_r_rule(r0):
    # a quarter of the real pool binds
    assert r0.n_r([10_000, 9_950], [3_400, 3_350]) == 2_400
    # 0.9 x the fewest e16_b5 rest-bearing URLs binds
    assert r0.n_r([90_000, 90_000], [6_000, 5_990]) == 5_300
    # never above 6,000, and always a multiple of 100
    assert r0.n_r([1_000_000], [100_000]) == 6_000
    assert r0.n_r([12_345], [9_999]) % 100 == 0


def test_seeds_are_fixed_and_distinct(r0):
    seeds = {r0.seed_real(lab, g) for lab in r0.LABELS for g in range(r0.G)}
    for src in r0.SOURCES:
        for kind in ("R", "H"):
            seeds |= {r0.seed_source(src, lab, g, kind) for lab in r0.LABELS for g in range(r0.G)}
    assert len(seeds) == 2 * 5 + 2 * 2 * 2 * 5
    assert r0.seed_real("benign", 0) == 20261022
    assert r0.seed_source("e16_b5", "phishing", 3) == 20261022 + 10_000 + 100 + 3
    assert r0.seed_source("e15_new", "benign", 0, "H") == 20261022 + 20_000 + 1_000


def test_folds_keep_e18s_indices(r0):
    # floor and e16_b5 split exactly as E18 split them (V-D1 depends on it)
    assert r0.TASK_INDEX["floor"] == 0 and r0.TASK_INDEX["e16_b5"] == 1
    assert r0.TASK_INDEX["e15_new"] not in (0, 1, 2, 3)
