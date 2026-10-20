"""E10 -- the registered criteria in research/native/e10_compare.py, on synthetic
readings: one where everything passes, then one targeted break per rule, and every
branch of the candidate rule, proven before any real data exists
(research/native/PREREG_E10.md section 2)."""
import copy
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "native"))
sys.path.insert(0, str(ROOT / "research" / "dc1"))


def _load():
    spec = importlib.util.spec_from_file_location("e10_compare_under_test",
                                                  ROOT / "research" / "native" / "e10_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load()
LABELS, STATES = ("benign", "phishing"), ("P1Q1", "P1Q0", "P0Q1", "P0Q0")
GATED = ("LBL", "SUF", "SUB", "PTH", "QRY")


def acc(x, v):
    return {"acceptance": {"x": x, "verdict": v}}


def report():
    free = {f: acc(0.02, "CLEARED") for f in GATED}
    free["QRY"] = acc(0.035, "UNDECIDED")
    ctrl = {f"{l}_{s}": {"label": l, "state": s, "fields": {f: acc(0.02, "CLEARED") for f in GATED}}
            for l in LABELS for s in STATES if (l, s) not in {("phishing", "P1Q1"), ("phishing", "P0Q1")}}
    rep = {"free": {}, "control": {}, "emission": {}, "novelty": {}, "honouring": {},
           "plan": {"benign": {"P": 0.74, "Q": 0.064}, "phishing": {"P": 0.40, "Q": 0.046}}}
    for who in (C.BASE, C.CAND):
        rep["free"][who] = {l: {"fields": copy.deepcopy(free)} for l in LABELS}
        rep["control"][who] = copy.deepcopy(ctrl)
        rep["novelty"][who] = {m: {"n": 80_000, "novel": 79_300, "q": 3_000, "copied": 600}
                               for m in ("free", "control")}
        rep["honouring"][who] = {f"{l}_{s}": {"rate": 0.95} for l in LABELS for s in STATES}
    rep["emission"][C.CAND] = {"benign": {"PTH": {"rate": 0.735}, "QRY": {"rate": 0.064}},
                               "phishing": {"PTH": {"rate": 0.395}, "QRY": {"rate": 0.046}}}
    rep["emission"][C.BASE] = {"benign": {"PTH": {"rate": 0.756}, "QRY": {"rate": 0.046}},
                               "phishing": {"PTH": {"rate": 0.382}, "QRY": {"rate": 0.029}}}
    return rep


def test_everything_passes_and_e10_plan_is_the_candidate():
    res = C.compare(report())
    assert all(res[k]["pass"] for k in ("K10a", "K10b", "K10c", "K10d"))
    assert res["candidate"].startswith("e10_plan is a candidate")


def test_k10a_fails_when_the_plan_query_rate_is_off_by_more_than_ten_percent():
    rep = report()
    rep["plan"]["benign"]["Q"] = 0.0653 * 0.85
    res = C.compare(rep)
    assert not res["K10a"]["pass"] and res["candidate"].startswith("K10a fails")


def test_k10a_fails_when_the_plan_path_rate_is_off_by_more_than_three_points():
    rep = report()
    rep["plan"]["phishing"]["P"] = 0.394 + 0.035
    assert not C.compare(rep)["K10a"]["pass"]


def test_k10a_holds_just_inside_its_limits():
    rep = report()
    rep["plan"]["benign"]["Q"] = 0.0653 * 1.099
    rep["plan"]["phishing"]["P"] = 0.394 - 0.029
    assert C.compare(rep)["K10a"]["pass"]


def test_emission_outside_its_band_fails_k10b_and_says_where_queries_are_lost():
    rep = report()
    rep["emission"][C.CAND]["benign"]["QRY"]["rate"] = 0.050
    res = C.compare(rep)
    assert not res["K10b"]["pass"] and "lost after the plan" in res["candidate"]


def test_a_regression_against_e10_base_fails_k10c():
    rep = report()
    rep["control"][C.CAND]["benign_P1Q1"]["fields"]["QRY"] = acc(0.09, "CONVICTED")
    res = C.compare(rep)
    assert not res["K10c"]["pass"] and "K10c" in res["candidate"]


def test_a_verdict_change_without_the_rise_is_not_a_regression():
    rep = report()
    rep["free"][C.CAND]["benign"]["fields"]["QRY"] = acc(0.045, "CONVICTED")   # rise 0.010
    assert C.compare(rep)["K10c"]["pass"]


def test_a_copying_rise_fails_k10d():
    rep = report()
    rep["novelty"][C.CAND]["free"]["copied"] = 900
    res = C.compare(rep)
    assert not res["K10d"]["pass"] and "K10d" in res["candidate"]


def test_reproducibility_against_e9_ln_is_reported_not_decided():
    rep = report()
    e9 = {"free": {"e9_ln_raw": copy.deepcopy(rep["free"][C.BASE])},
          "emission": {"e9_ln_raw": copy.deepcopy(rep["emission"][C.BASE])},
          "honouring": {"e9_ln_raw": copy.deepcopy(rep["honouring"][C.BASE])}}
    e9["free"]["e9_ln_raw"]["benign"]["fields"]["PTH"] = acc(0.04, "CLEARED")
    res = C.compare(rep, e9)
    r = res["reproducibility_e10_base_vs_e9_ln"]
    assert abs(r["max_abs_free_cell_difference"] - 0.02) < 1e-12
    assert r["within_retrain_allowance"] is False
    assert res["candidate"].startswith("e10_plan is a candidate")     # not a criterion
