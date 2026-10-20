"""V2-P1 amendment 1 (research/v2/PREREG_V2_P1_MEMORY.md section 12): the tiered copy rule,
checked on synthetic inputs before any P1 output exists. No data is loaded.

- paths and queries get +0.05 only while diversity holds; once-seen strings stay strict;
- W1 gets up to +0.10 only while G-PAIR passes and the blind-spot check holds;
- the blind-spot check: whatever passes under D1b must pass under D1b-x;
- G7's allowance acts upward only, and never on the once-seen rate;
- both readings, and the label when they differ.
"""
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "v2"))


@pytest.fixture(scope="module")
def C():
    import v2_p1_compare as mod
    return mod


def cls(pth_p1=0.20, once_p1=0.015, w1_p1=0.05, pair_p1=0.70):
    src = ("real", "base", "p1")
    return {
        "gate": {k: {"n": 8000, "memorized": 0.001, "duplicate_in_batch": 0.0, "no_public_suffix": 0.0005}
                 for k in src},
        "RD": {k: {"n": 8000, "copy": 0.01, "copy_once": 0.005, "duplicate": 0.0} for k in src},
        "PTH": {"real": {"n": 6000, "copy": 0.16, "copy_once": 0.023, "duplicate": 0.06},
                "base": {"n": 6000, "copy": 0.20, "copy_once": 0.016, "duplicate": 0.09},
                "p1": {"n": 6000, "copy": pth_p1, "copy_once": once_p1, "duplicate": 0.08}},
        "QRY": {k: {"n": 600, "copy": 0.23, "copy_once": 0.03, "duplicate": 0.05} for k in src},
        "blind_spot": {v: {"C": {"n": 25000, "b_only": 0.04, "a_only": 0.04},
                           "base": {"n": 25000, "b_only": 0.05, "a_only": 0.05},
                           "p1": {"n": 25000, "b_only": w1_p1, "a_only": w1_p1}} for v in ("R", "H")},
        "pair_novelty": {"real": 0.64, "base": 0.80, "p1": pair_p1},
        "distinct_host": {"n": 6000, "real": 650.0, "base": 730.0, "p1": 700.0},
        "distinct_rest": {"n": 10000, "real": 637.0, "base": 610.0, "p1": 625.0},
        "unseen_judge_host": {"real": [0.47, 12000], "base": [0.56, 12000], "p1": [0.50, 12000]},
        "unseen_judge_rest": {"real": [0.11, 29000], "base": [0.24, 21000], "p1": [0.15, 21000]},
    }


def guards(C, benign, blind=True):
    rows = C.metric_rows({"benign": benign, "phishing": cls()})
    ok = {"benign": blind, "phishing": True}
    return C.metric_guards(rows, False, ok), C.metric_guards(rows, True, ok)


def test_paths_get_five_points_while_diversity_holds(C):
    strict, tiered = guards(C, cls(pth_p1=0.235))            # strict limit is about 0.21
    assert not strict["G-COPY"]["benign"]["pass"]
    assert tiered["G-COPY"]["benign"]["pass"]
    _, tiered = guards(C, cls(pth_p1=0.27))                  # beyond +0.05
    assert not tiered["G-COPY"]["benign"]["pass"]


def test_once_seen_strings_stay_strict(C):
    _, tiered = guards(C, cls(pth_p1=0.235, once_p1=0.04))
    rows = tiered["G-COPY"]["benign"]["rows"]
    assert rows["PTH"]["pass"] and not rows["PTH once-seen"]["pass"]
    assert not tiered["G-COPY"]["benign"]["pass"]


def test_no_allowance_without_diversity(C):
    _, tiered = guards(C, cls(pth_p1=0.235, pair_p1=0.50))   # G-PAIR fails
    assert not tiered["G-PAIR"]["benign"]["pass"]
    assert not tiered["G-COPY"]["benign"]["diversity_condition"]
    assert not tiered["G-COPY"]["benign"]["pass"]


def test_site_words_get_ten_points_only_with_the_blind_spot_check(C):
    strict, tiered = guards(C, cls(w1_p1=0.11))
    assert not strict["W1"]["benign"]["pass"] and tiered["W1"]["benign"]["pass"]
    _, tiered = guards(C, cls(w1_p1=0.11), blind=False)
    assert not tiered["W1"]["benign"]["pass"]
    _, tiered = guards(C, cls(w1_p1=0.16))                   # beyond +0.10
    assert not tiered["W1"]["benign"]["pass"]


def test_the_blind_spot_check(C):
    k = {"benign": {"K1": True, "K2": True}, "phishing": {"K1": False, "K2": True}}
    kx = {"benign": {"K1": False, "K2": True}, "phishing": {"K1": False, "K2": True}}
    assert C.blind_spot_check(k, kx) == {"benign": False, "phishing": True}


def g7_rep(rate, singleton):
    real = {"n": 600, "rate": 0.25, "singleton": 0.03}
    return {"copy": {"free_benign": {"real_stream": real,
                                     "p1_mem": {"generated": {"n": 3000, "rate": rate, "singleton": singleton},
                                                "per_generation": 600}}},
            "copy_ref": {"free_benign": {"generated": {"n": 3000, "rate": 0.27, "singleton": 0.02}}},
            "copy_ref_real": {"free_benign": real}}


def test_g7_allowance_acts_upward_only_and_never_on_once_seen(C):
    rep = g7_rep(0.33, 0.03)
    assert not C.g7_tiered(rep, {"benign": 0.0, "phishing": 0.0})["pass"]
    assert C.g7_tiered(rep, {"benign": 0.05, "phishing": 0.05})["pass"]
    rep = g7_rep(0.33, 0.08)                                  # once-seen copying OVER
    assert not C.g7_tiered(rep, {"benign": 0.05, "phishing": 0.05})["pass"]


def test_both_readings_and_the_label(C):
    k = {lab: {"void": False, "K1": False, "K2": lab == "benign", "whole_url_cleared": False}
         for lab in ("benign", "phishing")}
    strict = {"G1": {"pass": True}, "G-COPY": {"pass": False}}
    tiered = deepcopy(strict)
    tiered["G-COPY"]["pass"] = True
    assert C.reading(k, strict)["reading"] == "PARTIAL WITH COST (G-COPY)"
    assert C.reading(k, tiered)["reading"] == "PARTIAL"
    k = {lab: dict(v, void=True) for lab, v in k.items()}
    assert C.reading(k, tiered)["reading"] == "UNREADABLE"
