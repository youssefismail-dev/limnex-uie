"""E8 (N1) -- the registered criteria in research/native/e8_compare.py, on
synthetic reports: one report where everything passes, then one targeted break
per rule, so every criterion and every branch of the candidate rule is proven
before any real data exists (PREREG_E8 section 6)."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "native"))
sys.path.insert(0, str(ROOT / "research" / "dc1"))


def _load():
    spec = importlib.util.spec_from_file_location("e8_compare_under_test",
                                                  ROOT / "research" / "native" / "e8_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load()
LABELS, STATES = ("benign", "phishing"), ("P1Q1", "P1Q0", "P0Q1", "P0Q0")
WHO = ("e7", "e8_base", "e8_rs", "e8_pf", "e8_pfrs")


def acc(x, v):
    return {"acceptance": {"x": x, "verdict": v}}


def good_report():
    """Every criterion passes: identical realism everywhere, plan arms honour and
    emit on target, slash arms write the slash, E7-conditioned arms do neither."""
    cells = {f: acc(0.02, "CLEARED") for f in C.GATED}
    rep = {"e7_free": {l: {"fields": copy.deepcopy(cells)} for l in LABELS},
           "free": {}, "control": {}, "honouring": {}, "emission": {}, "root_slash": {},
           "novelty": {}}
    for who in WHO:
        plan = C.K.ARMS.get(who, {}).get("presence_plan", False)
        slash = C.K.ARMS.get(who, {}).get("root_slash", False)
        if who != "e7":
            rep["free"][who] = {l: {"fields": copy.deepcopy(cells)} for l in LABELS}
        rep["control"][who] = {f"{l}_{s}": {"label": l, "state": s, "fields": copy.deepcopy(cells)}
                               for l in LABELS for s in STATES}
        rep["honouring"][who] = {f"{l}_{s}": {"rate": 0.995 if plan else 0.96}
                                 for l in LABELS for s in STATES}
        rep["emission"][who] = {
            "benign": {"PTH": {"rate": 0.73 if plan else 0.59}, "QRY": {"rate": 0.066 if plan else 0.036}},
            "phishing": {"PTH": {"rate": 0.395 if plan else 0.30}, "QRY": {"rate": 0.047 if plan else 0.017}}}
        rep["root_slash"][who] = {
            "benign": {"bare": {"rate": 0.98 if slash else 0.0}, "query_no_path": {"rate": 0.995 if slash else 0.0}},
            "phishing": {"bare": {"rate": 0.999 if slash else 0.0}, "query_no_path": {"rate": 1.0 if slash else 0.0}}}
        rep["novelty"][who] = {m: {"n": 100000, "novel": 99300, "q": 5000, "copied": 1100}
                               for m in ("free", "control")}
    return rep


def results(rep):
    res = {}
    for arm in C.K.ARMS:
        res[arm] = {"K1d": C.k1d(rep, arm), "K1e": C.k1e(rep, arm),
                    "K1a": C.k1a(rep, arm) if arm in C.PLAN_ARMS else {"pass": True},
                    "K1b": C.k1b(rep, arm) if arm in C.PLAN_ARMS else {"pass": True},
                    "K1c": C.k1c(rep, arm) if arm in C.SLASH_ARMS else {"pass": True}}
    return res


def test_the_good_report_passes_everything_and_names_the_candidate():
    rep = good_report()
    vb = C.v_base(rep)
    assert vb["pass"], vb
    res = results(rep)
    assert all(r[k]["pass"] for r in res.values() for k in r)
    assert C.candidate(res, vb).startswith("e8_pfrs is the candidate")


def test_k1d_needs_a_worse_verdict_and_a_rise_beyond_retraining_noise():
    base = {("benign", "PTH"): (0.040, "CLEARED")}
    assert C.regressions(base, {("benign", "PTH"): (0.0542, "UNDECIDED")})        # +0.0142
    assert not C.regressions(base, {("benign", "PTH"): (0.0540, "UNDECIDED")})    # +0.0140
    assert not C.regressions(base, {("benign", "PTH"): (0.0700, "CLEARED")})      # same verdict
    assert not C.regressions({("benign", "PTH"): (0.07, "CONVICTED")},
                             {("benign", "PTH"): (0.03, "CLEARED")})              # better


def test_emission_bands_are_the_registered_ones():
    rep = good_report()
    rep["emission"]["e8_pfrs"]["phishing"]["QRY"]["rate"] = 0.0420                # below 4.21%
    assert not C.k1b(rep, "e8_pfrs")["pass"]
    rep["emission"]["e8_pfrs"]["phishing"]["QRY"]["rate"] = 0.0515                # the edge
    assert C.k1b(rep, "e8_pfrs")["pass"]
    rep["emission"]["e8_pfrs"]["benign"]["PTH"]["rate"] = 0.7700                  # above 76.37%
    assert not C.k1b(rep, "e8_pfrs")["pass"]


def test_root_slash_tolerance_is_two_points():
    rep = good_report()
    rep["root_slash"]["e8_rs"]["benign"]["bare"]["rate"] = 0.960                   # 1.9 pts off
    assert C.k1c(rep, "e8_rs")["pass"]
    rep["root_slash"]["e8_rs"]["benign"]["bare"]["rate"] = 0.955                   # 2.4 pts off
    assert not C.k1c(rep, "e8_rs")["pass"]


def test_honouring_is_its_own_criterion_every_state_counts():
    rep = good_report()
    rep["honouring"]["e8_pfrs"]["phishing_P0Q1"]["rate"] = 0.975                  # an unmeasurable state
    assert not C.k1a(rep, "e8_pfrs")["pass"]


def test_only_k1a_failing_licenses_n1b_and_nothing_else():
    rep = good_report()
    for s in STATES:
        rep["honouring"]["e8_pfrs"][f"phishing_{s}"]["rate"] = 0.95
    out = C.candidate(results(rep), C.v_base(rep))
    assert out.startswith("only K1a fails: N1b")


def test_a_regression_in_the_candidate_stops_the_plan_and_lets_the_slash_stand_alone():
    rep = good_report()
    rep["free"]["e8_pfrs"]["benign"]["fields"]["QRY"] = acc(0.07, "CONVICTED")    # +0.05, worse
    out = C.candidate(results(rep), C.v_base(rep))
    assert "e8_pfrs fails K1d" in out
    assert "the plan is not taken forward" in out and "e8_rs stands alone" in out


def test_another_arms_failure_is_reported_with_the_outcome_never_dropped():
    """The registration names e8_pfrs the candidate when IT passes; a failure in
    the plan-only arm does not change that, but it must travel with the outcome."""
    rep = good_report()
    rep["free"]["e8_pf"]["benign"]["fields"]["QRY"] = acc(0.07, "CONVICTED")
    out = C.candidate(results(rep), C.v_base(rep))
    assert out.startswith("e8_pfrs is the candidate") and "reported: e8_pf fails K1d" in out


def test_novelty_worsening_counts_only_beyond_half_a_point_with_its_ci():
    rep = good_report()
    rep["novelty"]["e8_pfrs"]["free"]["copied"] = 1150                            # +1 pt of 5,000
    r = C.k1e(rep, "e8_pfrs")
    assert r["pass"]                                                              # CI crosses 0
    rep["novelty"]["e8_pfrs"]["free"]["copied"] = 1300                            # +4 pts
    assert not C.k1e(rep, "e8_pfrs")["pass"]


def test_v_base_fails_when_the_same_recipe_does_not_reproduce_e7():
    rep = good_report()
    rep["honouring"]["e8_base"]["benign_P1Q1"]["rate"] = 0.975                    # 1.5 pts from E7
    vb = C.v_base(rep)
    assert not vb["pass"]
    assert C.candidate(results(rep), vb).startswith("V-base FAILED")


def test_attribution_credits_only_effects_beyond_the_base_to_e7_gap():
    rep = good_report()
    rep["honouring"]["e7"]["phishing_P1Q1"]["rate"] = 0.955                       # noise 0.005
    att = C.attribution(rep)
    row = att["honour phishing P1Q1"]
    assert row["noise"] == pytest.approx(0.005)
    assert row["plan"] == pytest.approx(0.035) and row["plan_credited"] is True
    assert row["slash_on_e7"] == pytest.approx(0.0) and row["slash_on_e7_credited"] is False


def test_a_starved_cell_prints_its_verdict_and_never_crashes_the_scorer():
    """Phishing QRY is starved at the canonical size for E7-like arms (x is None);
    the scorer's progress line must print its verdict, not crash formatting it."""
    spec = importlib.util.spec_from_file_location("e8_score_under_test",
                                                  ROOT / "research" / "native" / "e8_score.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    assert S._cell("QRY", {"acceptance": {"x": None, "verdict": "INSUFFICIENT"}}) == "QRY INSUFFICIENT"
    assert S._cell("QRY", {"acceptance": {"x": float("nan"), "verdict": "UNDECIDED"}}) == "QRY UNDECIDED"
    assert S._cell("PTH", {"acceptance": {"x": 0.0335, "verdict": "CLEARED"}}) == "PTH +0.0335 CLEARED"
    assert S._cell("SUB", {}) == "SUB --"
