"""E12 -- the registered criteria in research/native/e12_compare.py, on synthetic
readings: one where everything passes and E3 is claimed, then one targeted break
per rule, proven before any real data exists (research/native/PREREG_E12.md
section 2). No data file is read."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("native", "dc1", "qc1", "cr1"):
    sys.path.insert(0, str(ROOT / "research" / p))

import e12_common as K  # noqa: E402
import e12_compare as C  # noqa: E402

GATED = ("LBL", "SUF", "SUB", "PTH", "QRY")
MEASURED = [(l, s) for l in K.LABELS for s in K.STATES if (l, s) not in K.UNMEASURABLE]
#: E11's targets, rounded, with free-mode slash populations of E11's size: phishing's
#: rates are 1.0, so the edge rule makes T = 2 / n (about 0.04% for bare hosts).
TARGETS = {label: {"path_rate_free": p, "query_rate_free": q,
                   "slash_P0Q0": {"free": b, "control": b, "free_n_per_generation": nb},
                   "slash_P0Q1": {"free": h, "control": h, "free_n_per_generation": nh}}
           for label, p, q, b, nb, h, nh in (("benign", 0.7337, 0.0662, 0.980, 1980, 0.9958, 75),
                                             ("phishing", 0.3951, 0.0468, 1.0, 4700, 1.0, 130))}


def acc(x, v):
    return {"acceptance": {"x": x, "verdict": v}}


def w(rate):
    return {"rate": rate}


def report():
    rep = {"free": {}, "control": {}, "emission": {}, "slash": {}, "novelty": {}, "copy": {}}
    for arm in K.ARMS:
        rep["free"][arm] = {l: {"fields": {f: acc(0.01, "CLEARED") for f in GATED}} for l in K.LABELS}
        rep["control"][arm] = {f"{l}_{s}": {"label": l, "state": s, "fields": {
            f: acc(0.01, "CLEARED") for f in GATED}} for l, s in MEASURED}
        rep["emission"][arm] = {l: {"PTH": w(TARGETS[l]["path_rate_free"]), "QRY": w(TARGETS[l]["query_rate_free"])}
                                for l in K.LABELS}
        rep["slash"][arm] = {}
        for l in K.LABELS:
            for st, key in (("P0Q0", "free_bare"), ("P0Q1", "free_host_query")):
                t = TARGETS[l][f"slash_{st}"]["free"]
                rep["slash"][arm][f"{key}_{l}"] = w(t)
                rep["slash"][arm][f"control_{l}_{st}"] = w(t)
        rep["novelty"][arm] = {m: {"n": 80_000, "novel": 79_600, "q": 4_000, "copied": 1_000}
                               for m in ("free", "control")}
    for cell, dd in (("free_benign", False), ("free_phishing", True)):
        node = {"real": {"rate": 0.30, "singleton": 0.05}}
        if dd:
            node["real_domain_disjoint"] = {"rate": 0.26, "singleton": 0.04}
        for arm in K.ARMS:
            node[arm] = {"generated": {"rate": 0.29, "singleton": 0.05}, "per_generation": 500}
        rep["copy"][cell] = node
    return rep


def train_dir(tmp_path, share=0.0):
    for arm in K.ARMS:
        (tmp_path / arm).mkdir(parents=True, exist_ok=True)
        bands = {f"after_1000 {b}": {"steps": 1000, "mean_norm": 0.5, "clipped_share": share, "nonfinite": 0}
                 for b in ("0-30", "30-60", "60-120", "120-inf")}
        (tmp_path / arm / "train_result.json").write_text(json.dumps({"grad_diagnostics": bands}))
    return tmp_path


def arm_result(rep, arm, train):
    r = {"K12a": C.k12a(rep, TARGETS, arm), "K12b": C.k12b(rep, TARGETS, arm),
         "K12c": C.k12c(rep, arm) if arm in K.REFERENCE else None,
         "K12d": C.k12d(rep, arm), "K12e": C.k12e(train, arm)}
    r["failing"] = [k for k in ("K12a", "K12b", "K12c", "K12d", "K12e") if r[k] is not None and not r[k]["pass"]]
    r["finish_line"] = C.finish_line(rep, arm, r["K12b"])
    return r


def test_everything_passes_and_the_finish_line_is_met(tmp_path):
    rep, train = report(), train_dir(tmp_path)
    for arm in K.ARMS:
        r = arm_result(rep, arm, train)
        assert r["failing"] == [] and r["finish_line"]["met"], (arm, r["failing"], r["finish_line"])


def test_a_free_cell_short_of_cleared_misses_the_finish_line_even_when_not_computable(tmp_path):
    for verdict in ("UNDECIDED", "CONVICTED", "NOT COMPUTABLE"):
        rep = report()
        rep["free"][K.CANDIDATE]["phishing"]["fields"]["QRY"] = acc(0.09 if verdict != "NOT COMPUTABLE" else None, verdict)
        fl = C.finish_line(rep, K.CANDIDATE, C.k12b(rep, TARGETS, K.CANDIDATE))
        assert not fl["met"] and [c["cell"] for c in fl["cells_not_cleared"]] == ["free phishing QRY"]


def test_a_control_cell_the_gate_cannot_compute_is_outside_the_finish_line():
    rep = report()
    rep["control"][K.CANDIDATE]["benign_P0Q0"]["fields"]["PTH"] = {"acceptance": {"verdict": "NOT COMPUTABLE"}}
    assert C.finish_line(rep, K.CANDIDATE, C.k12b(rep, TARGETS, K.CANDIDATE))["met"]
    rep["control"][K.CANDIDATE]["benign_P1Q1"]["fields"]["SUB"] = acc(0.03, "UNDECIDED")
    fl = C.finish_line(rep, K.CANDIDATE, C.k12b(rep, TARGETS, K.CANDIDATE))
    assert not fl["met"] and fl["cells_not_cleared"][0]["cell"] == "control benign_P1Q1 SUB"


def test_a_slash_row_outside_the_tolerance_fails_k12b_and_the_finish_line():
    rep = report()
    rep["slash"][K.CANDIDATE]["free_bare_phishing"] = w(0.9990)        # 0.1% misses; T is 2 / 4700
    b = C.k12b(rep, TARGETS, K.CANDIDATE)
    assert not b["pass"] and not b["rows"]["phishing free_bare"]["pass"]
    fl = C.finish_line(rep, K.CANDIDATE, b)
    assert fl["slash_rows_missed"] == ["phishing free_bare"] and not fl["met"]


def test_free_presence_outside_the_tolerance_fails_k12a():
    rep = report()
    rep["emission"][K.CANDIDATE]["benign"]["QRY"] = w(0.0662 - 0.0060)  # T is about 0.0056 at n = 8000
    a = C.k12a(rep, TARGETS, K.CANDIDATE)
    assert not a["pass"] and not a["rows"]["benign QRY"]["pass"] and a["rows"]["benign PTH"]["pass"]


def test_k12c_fails_only_when_the_verdict_worsens_and_x_rises_beyond_the_allowance():
    rep = report()
    rep["free"][K.CANDIDATE]["benign"]["fields"]["SUB"] = acc(0.01 + 0.0141, "UNDECIDED")
    assert C.k12c(rep, K.CANDIDATE)["pass"]                            # within the allowance
    rep["free"][K.CANDIDATE]["benign"]["fields"]["SUB"] = acc(0.01 + 0.0150, "UNDECIDED")
    c = C.k12c(rep, K.CANDIDATE)
    assert not c["pass"] and c["reference"] == "e12_base"


@pytest.mark.parametrize("change,fails", [
    (("host", "rate", 0.36), True),                  # copying OVER the natural rate + T
    (("host", "singleton", 0.08), True),             # strings seen once: OVER
    (("host", "rate", 0.10), False),                 # UNDER is reported, not a failure
])
def test_k12d_memorisation_fails_only_on_over_at_the_host_level(change, fails):
    rep = report()
    _, key, value = change
    rep["copy"]["free_benign"][K.CANDIDATE]["generated"][key] = value
    d = C.k12d(rep, K.CANDIDATE)
    assert d["pass"] is not fails
    assert (d["failures"] == ["memorisation free_benign"]) is fails


def test_k12d_reads_phishing_against_the_host_level_rate_and_reports_the_domain_disjoint_one():
    rep = report()
    rep["copy"]["free_phishing"][K.CANDIDATE]["generated"]["rate"] = 0.33   # OVER 0.26, within 0.30
    d = C.k12d(rep, K.CANDIDATE)
    m = d["memorisation"]["free_phishing"]
    assert m["domain_disjoint"]["rate"] == "OVER" and m["host"]["rate"] == "WITHIN" and d["pass"]


def test_k12d_novelty_worsening_against_e12_base_fails():
    rep = report()
    rep["novelty"][K.CANDIDATE]["free"]["novel"] = 78_000
    d = C.k12d(rep, K.CANDIDATE)
    assert not d["pass"] and "free_novelty" in d["failures"]
    assert C.k12d(rep, "e12_base")["novelty"] == {}                    # the reference itself


def test_k12e_fails_above_one_percent_clipped_or_without_diagnostics(tmp_path):
    assert C.k12e(train_dir(tmp_path / "a", 0.01), K.CANDIDATE)["pass"]
    assert not C.k12e(train_dir(tmp_path / "b", 0.011), K.CANDIDATE)["pass"]
    (tmp_path / "c" / K.CANDIDATE).mkdir(parents=True)
    (tmp_path / "c" / K.CANDIDATE / "train_result.json").write_text("{}")
    assert not C.k12e(tmp_path / "c", K.CANDIDATE)["pass"]


def test_differences_keep_free_and_control_cells_apart_and_skip_uncomputable_ones():
    rep = report()
    rep["free"]["e12_final"]["benign"]["fields"]["QRY"] = acc(0.004, "CLEARED")
    rep["free"]["e12_no_r1"]["phishing"]["fields"]["QRY"] = {"acceptance": {"verdict": "NOT COMPUTABLE"}}
    d = C.difference(rep, "e12_final", rep, "e12_no_r1")
    assert d["free_x"]["benign QRY"] == pytest.approx(-0.006)
    assert "phishing QRY" not in d["free_x"] and "benign P1Q1 QRY" in d["control_x"]


def results(rep, train):
    res = {"arms": {a: arm_result(rep, a, train) for a in K.ARMS}}
    res["leave_one_out"] = {"R1": C.difference(rep, "e12_final", rep, "e12_no_r1")}
    res["retrain_spread"] = C.difference(rep, "e12_base", rep, "e12_base")
    return res


E11_CMP = {"arms": {"e11_plan": {"K11b": {"rows": {
    "benign control P0Q1": {"pass": False}, "phishing free_bare": {"pass": False},
    "phishing control P0Q0": {"pass": False}, "benign free_bare": {"pass": True}}}}}}


def test_the_predictions_are_scored_from_the_readout(tmp_path):
    rep = report()
    for arm in K.ARMS:
        rep["free"][arm]["phishing"]["fields"]["QRY"] = acc(0.095, "CONVICTED")
    rep["free"]["e12_final"]["benign"]["fields"]["QRY"] = acc(0.030, "UNDECIDED")
    rep["free"]["e12_no_r1"]["benign"]["fields"]["QRY"] = acc(0.036, "UNDECIDED")
    for key, rate in (("control_benign_P0Q1", 0.9997), ("free_bare_phishing", 0.9953),
                      ("control_phishing_P0Q0", 0.9964)):
        rep["slash"]["e12_base"][key] = w(rate)
    res = results(rep, train_dir(tmp_path))
    held = {p["prediction"]: p["held"] for p in C.predictions(res, rep, E11_CMP)}
    assert held["K12b: every slash-plan arm passes"] is True
    assert held["K12b: e12_base fails the same three rows E11 did"] is True
    assert held["R1: benign free QRY lower with it, by 0.003-0.010"] is True
    assert held["phishing free QRY stays CONVICTED in every arm"] is True
    assert held["E3 finish line for e12_final: NOT met"] is True
    assert held["phishing free QRY is the expected failure"] is True
    assert held["the frame at every position (if licensed)"] is None
    assert all(v for k, v in held.items() if k.startswith("the run-to-run spread"))


def test_the_predictions_fail_when_the_readout_says_otherwise(tmp_path):
    rep = report()
    rep["free"]["e12_final"]["benign"]["fields"]["QRY"] = acc(0.030, "UNDECIDED")
    rep["free"]["e12_no_r1"]["benign"]["fields"]["QRY"] = acc(0.032, "UNDECIDED")   # only 0.002 lower
    res = results(rep, train_dir(tmp_path))
    held = {p["prediction"]: p["held"] for p in C.predictions(res, rep, E11_CMP)}
    assert held["R1: benign free QRY lower with it, by 0.003-0.010"] is False
    assert held["phishing free QRY stays CONVICTED in every arm"] is False     # CLEARED here
    assert held["K12b: e12_base fails the same three rows E11 did"] is False   # it fails none


def mix_file(path, dry=False, floors=(0.5, 0.5, 0.5)):
    draws = [{"floor": f, "n": 600, "plain": 0.06 + 0.001 * i, "mix": 0.07 + 0.001 * i}
             for i, f in enumerate(floors)]
    path.write_text(json.dumps({"n": 600, "n_plain": 600, "shares": {"host": 0.29},
                                "generated": {"found": 200, "novel": 1400}, "draws": draws,
                                "never_seen": {"insufficient": True, "n": 10},
                                "found": {"n": 300, "draws": [{"floor": 0.5, "x": 0.15}]}, "dry": dry}))


def test_qc1_2_applies_qc1s_rule_and_void_rule_and_refuses_a_dry_file_in_a_real_readout(tmp_path):
    (tmp_path / "qc1").mkdir()
    mix_file(tmp_path / "qc1" / f"mix_{K.CANDIDATE}_free_benign.json", floors=(0.5, 0.6, 0.5))
    out = C.qc1_2(tmp_path, dry=False)
    r = out["free_benign"]
    assert r["void"] == 1 and r["reading"] == "CONTENT-LED" and r["rho"] > 1
    assert r["x_never_seen"] is None and r["x_found"] == pytest.approx(0.15)
    assert out["free_phishing"] is None                                         # not measured
    mix_file(tmp_path / "qc1" / f"mix_{K.CANDIDATE}_free_phishing.json", dry=True)
    with pytest.raises(SystemExit):
        C.qc1_2(tmp_path, dry=False)
    assert C.qc1_2(tmp_path, dry=True)["free_phishing"]["reading"]
