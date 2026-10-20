"""E9 -- the registered criteria in research/native/e9_compare.py, on synthetic
readings: one where everything passes, then one targeted break per rule, and every
branch of the candidate rule and of section 7's E10 rule, proven before any real
data exists (PREREG_E9 sections 6-7)."""
import copy
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "native"))
sys.path.insert(0, str(ROOT / "research" / "dc1"))


def _load():
    spec = importlib.util.spec_from_file_location("e9_compare_under_test",
                                                  ROOT / "research" / "native" / "e9_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load()
LABELS, STATES = ("benign", "phishing"), ("P1Q1", "P1Q0", "P0Q1", "P0Q0")
GATED = ("LBL", "SUF", "SUB", "PTH", "QRY")


def ratio(r, lo=None, hi=None):
    return {"ratio": r, "lo": r - 0.01 if lo is None else lo, "hi": r + 0.01 if hi is None else hi}


def mech(f1, f2, f3, tail=1.05, weighted=(1.0, 1.0, 0.97)):
    fam = lambda pl, wt: {"plain": ratio(pl), "weighted": ratio(wt)}  # noqa: E731
    one = {"F1": fam(f1, weighted[0]), "F2": fam(f2, weighted[1]), "F3": fam(f3, weighted[2]),
           "TAIL": {"plain": ratio(tail, tail - 0.05, tail + 0.05), "weighted": ratio(tail)}}
    return {"benign": copy.deepcopy(one), "phishing": copy.deepcopy(one)}


BASE_MECH = mech(1.16, 1.06, 0.85)
LN_MECH = mech(1.02, 1.01, 0.98, weighted=(0.90, 0.95, 1.05))


def acc(x, v):
    return {"acceptance": {"x": x, "verdict": v}}


def report():
    free = {f: acc(0.02, "CLEARED") for f in GATED}
    free["QRY"] = acc(0.09, "CONVICTED")
    ctrl = {f"{l}_{s}": {"label": l, "state": s, "fields": {f: acc(0.02, "CLEARED") for f in GATED}}
            for l in LABELS for s in STATES if (l, s) not in {("phishing", "P1Q1"), ("phishing", "P0Q1")}}
    rep = {"free": {}, "control": {}, "emission": {}, "novelty": {}, "qry": {}}
    for who in (C.BASE, C.CAND, C.ATTR):
        rep["free"][who] = {l: {"fields": copy.deepcopy(free)} for l in LABELS}
        rep["control"][who] = copy.deepcopy(ctrl)
        rep["novelty"][who] = {m: {"n": 80_000, "novel": 79_300, "q": 3_000, "copied": 800}
                               for m in ("free", "control")}
    rep["emission"][C.CAND] = {"benign": {"PTH": {"rate": 0.735}, "QRY": {"rate": 0.066}},
                               "phishing": {"PTH": {"rate": 0.395}, "QRY": {"rate": 0.047}}}
    qc = lambda v, t: {"q_val": acc(v, "CONVICTED"), "q_train": acc(t, "CONVICTED")}  # noqa: E731
    rep["qry"][C.ATTR] = {"benign": qc(0.095, 0.086)}
    rep["qry"][C.CAND] = {"benign": qc(0.090, 0.080)}
    return rep


def run(rep=None, base=None, ln=None):
    return C.compare(rep or report(), base or BASE_MECH, ln or LN_MECH)


def test_everything_passes_and_e9_ln_is_the_candidate():
    res = run()
    assert all(res[k]["pass"] for k in ("P0", "K9a", "K9b", "K9c", "K9d"))
    assert res["candidate"].startswith("e9_ln with the raw decode is a candidate")
    assert res["section7"].startswith("no data experiment")


def test_p0_voids_the_mechanism_test():
    base = copy.deepcopy(BASE_MECH)
    base["benign"]["F1"]["plain"] = ratio(0.98)
    res = run(base=base)
    assert not res["P0"]["pass"] and res["candidate"].startswith("P0 failed")


def test_k9a_needs_half_the_gap_closed():
    ln = copy.deepcopy(LN_MECH)
    ln["benign"]["F1"]["plain"] = ratio(1.10)                 # closure 1 - 0.10/0.16 = 0.375
    res = run(ln=ln)
    assert not res["K9a"]["pass"] and res["candidate"].startswith("K9a fails")


def test_k9a_needs_a_ci_excluding_the_baseline():
    ln = copy.deepcopy(LN_MECH)
    ln["benign"]["F2"]["plain"] = ratio(1.02, 1.00, 1.07)     # closure 0.67, CI holds 1.06
    res = run(ln=ln)
    assert not res["K9a"]["families"]["F2"]["pass"] and not res["K9a"]["pass"]


def test_k9a_needs_phishing_to_agree_in_direction():
    ln = copy.deepcopy(LN_MECH)
    ln["phishing"]["F3"]["plain"] = ratio(0.80)               # farther from 1 than 0.85
    assert not run(ln=ln)["K9a"]["pass"]


def test_emission_outside_its_band_is_reported_not_refuting():
    rep = report()
    rep["emission"][C.CAND]["benign"]["QRY"]["rate"] = 0.045
    res = run(rep=rep)
    assert res["K9a"]["pass"] and not res["K9b"]["pass"]
    assert res["candidate"].startswith("K9a passes") and "K9b" in res["candidate"]


def test_a_regression_against_todays_system_fails_k9c():
    rep = report()
    rep["free"][C.BASE]["benign"]["fields"]["PTH"] = acc(0.035, "UNDECIDED")
    rep["free"][C.CAND]["benign"]["fields"]["PTH"] = acc(0.070, "CONVICTED")
    res = run(rep=rep)
    assert not res["K9c"]["pass"]
    assert res["K9c"]["regressions"][0]["cell"] == ["benign", "PTH"]
    # a worse verdict inside the allowance is not a regression
    rep["free"][C.CAND]["benign"]["fields"]["PTH"] = acc(0.045, "CONVICTED")
    assert run(rep=rep)["K9c"]["pass"]


def test_attribution_regressions_are_reported_but_do_not_decide():
    rep = report()
    rep["free"][C.ATTR]["benign"]["fields"]["LBL"] = acc(0.01, "CLEARED")
    rep["free"][C.CAND]["benign"]["fields"]["LBL"] = acc(0.06, "UNDECIDED")
    rep["free"][C.BASE]["benign"]["fields"]["LBL"] = acc(0.06, "UNDECIDED")
    res = run(rep=rep)
    assert res["K9c"]["pass"] and res["K9c"]["attribution_regressions"]


def test_a_copying_rise_fails_k9d():
    rep = report()
    rep["novelty"][C.CAND]["control"]["copied"] = 1_000        # 26.7% -> 33.3%
    res = run(rep=rep)
    assert not res["K9d"]["pass"] and "control_copying" in res["K9d"]["failures"]


def test_section7_qry_cleared_closes_the_data_question():
    rep = report()
    rep["free"][C.CAND]["benign"]["fields"]["QRY"] = acc(0.03, "CLEARED")
    assert run(rep=rep)["section7"].startswith("QRY CLEARED")


def test_section7_sparsity_signature_licenses_e10_data():
    rep = report()
    rep["qry"][C.CAND]["benign"] = {"q_val": acc(0.092, "CONVICTED"),   # val closure 0.03
                                    "q_train": acc(0.030, "CONVICTED")}  # train closure 0.65
    assert run(rep=rep)["section7"].startswith("SPARSITY signature")


def test_section7_tail_signature_licenses_r1_only_when_f2_passes():
    ln = copy.deepcopy(LN_MECH)
    ln["benign"]["TAIL"]["plain"] = ratio(1.30, 1.20, 1.40)
    assert run(ln=ln)["section7"].startswith("TAIL signature")
    ln["benign"]["F2"]["plain"] = ratio(1.05)                 # F2 no longer closes
    assert not run(ln=ln)["section7"].startswith("TAIL signature")


def test_section7_otherwise_licenses_nothing():
    ln = copy.deepcopy(LN_MECH)
    ln["benign"]["TAIL"]["plain"] = ratio(1.08, 1.02, 1.14)   # CI includes 1.10
    assert run(ln=ln)["section7"].startswith("no data experiment")
