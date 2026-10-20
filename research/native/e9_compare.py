"""E9 -- the registered criteria, the candidate rule and section 7's E10 rule
(PREREG_E9 sections 6-7). Reads data/e9/VERIFY.json, score_report.json and
mechanism_{e9_base,e9_ln}.json; writes data/e9/compare_report.json.

    python research/native/e9_compare.py

Every number is the registration's. The native candidate is `e9_ln` read with the
RAW decode; today's system, retrained on E9's start, is `e9_base` with the RESTING
decode. The objective's own effect at a fixed decode (`e9_base` raw) is reported for
attribution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dc1"))

import e9_common as K  # noqa: E402
from e8_compare import (P_BAND, Q_BAND, control_cells, free_cells, rate,  # noqa: E402
                        regressions)

BASE, CAND, ATTR = "e9_base_resting", "e9_ln_raw", "e9_base_raw"
CLOSES, LITTLE = 0.5, 0.25
TAIL_BAR = 1.10
FAMILIES = ("F1", "F2", "F3")
#: P0: E7's pattern -- plain above 1 (stopping too early) or below 1 (too few params).
ABOVE = {"F1": True, "F2": True, "F3": False}


def closure(r_arm: float, r_base: float) -> float:
    return 1.0 - abs(r_arm - 1.0) / abs(r_base - 1.0)


def p0(mech_base: dict) -> dict:
    """e9_base must show E7's pattern (benign), or the mechanism test is void."""
    fails = []
    for fam in FAMILIES:
        r = mech_base["benign"][fam]
        pl, wt = r["plain"]["ratio"], r["weighted"]["ratio"]
        if (pl > 1.0) != ABOVE[fam]:
            fails.append(f"{fam}: plain {pl:.3f} not on E7's side of 1")
        if abs(wt - 1.0) >= abs(pl - 1.0):
            fails.append(f"{fam}: weighted {wt:.3f} not closer to 1 than plain {pl:.3f}")
    return {"pass": not fails, "failures": fails}


def k9a(mech_base: dict, mech_ln: dict) -> dict:
    """Benign: closure >= 0.5 AND e9_ln's 95% CI excludes e9_base's value, in each
    family. Phishing: e9_ln closer to 1 than e9_base, in each family."""
    fams, fails = {}, []
    for fam in FAMILIES:
        b, a = mech_base["benign"][fam]["plain"], mech_ln["benign"][fam]["plain"]
        c = closure(a["ratio"], b["ratio"])
        excludes = not (a["lo"] <= b["ratio"] <= a["hi"])
        pb = mech_base["phishing"][fam]["plain"]["ratio"]
        pa = mech_ln["phishing"][fam]["plain"]["ratio"]
        agrees = abs(pa - 1.0) < abs(pb - 1.0)
        ok = c >= CLOSES and excludes and agrees
        fams[fam] = {"closure": c, "ci_excludes_base": excludes, "phishing_agrees": agrees,
                     "base": b["ratio"], "arm": a["ratio"], "arm_ci": [a["lo"], a["hi"]],
                     "phishing": [pb, pa], "pass": ok}
        if not ok:
            fails.append(f"{fam}: closure {c:.2f}, CI excludes base {excludes}, "
                         f"phishing agrees {agrees}")
    return {"pass": not fails, "failures": fails, "families": fams}


def k9b(rep: dict) -> dict:
    fails = []
    for label in K.LABELS:
        for f, band in (("QRY", Q_BAND[label]), ("PTH", P_BAND[label])):
            r = rate(rep, "emission", CAND, label, f)
            if r is None or not band[0] <= r <= band[1]:
                fails.append(f"{label} {f}: {r} outside {band}")
    return {"pass": not fails, "failures": fails}


def k9c(rep: dict) -> dict:
    """No gated cell's verdict worsens with x rising > 0.0141 against today's system;
    the same rule against e9_base / raw is reported (attribution), not decisive."""
    regs = (regressions(free_cells(rep, BASE), free_cells(rep, CAND))
            + regressions(control_cells(rep, BASE), control_cells(rep, CAND)))
    attr = (regressions(free_cells(rep, ATTR), free_cells(rep, CAND))
            + regressions(control_cells(rep, ATTR), control_cells(rep, CAND)))
    return {"pass": not regs, "regressions": regs, "attribution_regressions": attr}


def k9d(rep: dict) -> dict:
    from dc1_compare import prop_worsening
    out = {}
    for mode in ("free", "control"):
        b, a = rep["novelty"][BASE][mode], rep["novelty"][CAND][mode]
        out[f"{mode}_novelty"] = prop_worsening(b["novel"], b["n"], a["novel"], a["n"], False)
        out[f"{mode}_copying"] = prop_worsening(b["copied"], max(1, b["q"]), a["copied"],
                                                max(1, a["q"]), True)
    fails = [k for k, v in out.items() if v["worsened"]]
    return {"pass": not fails, "failures": fails, "readings": out}


def candidate(res: dict) -> str:
    if not res["P0"]["pass"]:
        return "P0 failed: e9_base does not show E7's pattern; the mechanism test is void (reported)"
    if not res["K9a"]["pass"]:
        return "K9a fails: the hypothesis is refuted; nothing is licensed"
    failing = [k for k in ("K9b", "K9c", "K9d") if not res[k]["pass"]]
    if failing:
        return (f"K9a passes (mechanism confirmed); {', '.join(failing)} fail: reported, "
                f"the owner decides")
    return ("e9_ln with the raw decode is a candidate (K9a-K9d pass); the owner decides "
            "adoption -> N3 with the frozen delta retired for it")


def section7(rep: dict, mech_ln: dict, k9a_res: dict) -> str:
    """Section 7: read on e9_ln / raw, benign deciding, in order."""
    qry = free_cells(rep, CAND).get(("benign", "QRY"))
    if qry and qry[1] == "CLEARED":
        return "QRY CLEARED: no data experiment; the data question is closed"
    q = rep.get("qry", {})
    base, cand = q.get(ATTR, {}).get("benign"), q.get(CAND, {}).get("benign")
    if base and cand:
        xb_t, xc_t = base["q_train"]["acceptance"]["x"], cand["q_train"]["acceptance"]["x"]
        xb_v, xc_v = base["q_val"]["acceptance"]["x"], cand["q_val"]["acceptance"]["x"]
        if 1 - xc_t / xb_t >= CLOSES and 1 - xc_v / xb_v < LITTLE:
            return ("SPARSITY signature: E10-data licensed -- R1 (a longer max_len from the "
                    "frozen corpus) first; external query data only if the owner unfreezes "
                    "the corpus")
    tail = mech_ln["benign"]["TAIL"]["plain"]
    if tail["lo"] > TAIL_BAR and k9a_res["families"]["F2"]["pass"]:
        return "TAIL signature: R1 licensed alone"
    return "no data experiment licensed; reported; the owner decides"


def compare(rep: dict, mech_base: dict, mech_ln: dict) -> dict:
    res = {"P0": p0(mech_base), "K9a": k9a(mech_base, mech_ln), "K9b": k9b(rep),
           "K9c": k9c(rep), "K9d": k9d(rep)}
    res["candidate"] = candidate(res)
    res["section7"] = section7(rep, mech_ln, res["K9a"])
    return res


def main() -> None:
    e9 = ROOT / "data" / "e9"
    verify = json.loads((e9 / "VERIFY.json").read_text(encoding="utf-8"))
    if verify.get("verdict") != "PASS":
        raise SystemExit(f"STOP: verification failed; nothing is read: {verify.get('failures')}")
    rep = json.loads((e9 / "score_report.json").read_text(encoding="utf-8"))
    mech = {a: json.loads((e9 / f"mechanism_{a}.json").read_text(encoding="utf-8"))
            for a in K.ARMS}
    res = compare(rep, mech["e9_base"], mech["e9_ln"])
    (e9 / "compare_report.json").write_text(json.dumps(res, indent=1, default=float),
                                            encoding="utf-8")
    for k in ("P0", "K9a", "K9b", "K9c", "K9d"):
        print(f"{k}: {'PASS' if res[k]['pass'] else 'FAIL'}  "
              f"{res[k].get('failures') or res[k].get('regressions') or ''}")
    print("CANDIDATE:", res["candidate"])
    print("SECTION 7:", res["section7"])


if __name__ == "__main__":
    main()
