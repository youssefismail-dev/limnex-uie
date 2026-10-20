"""E10 -- the registered criteria and the candidate rule (research/native/PREREG_E10.md
section 2). Reads data/e10/VERIFY.json and score_report.json; writes
data/e10/compare_report.json.

    python research/native/e10_compare.py              # everything, after VERIFY PASS
    python research/native/e10_compare.py --k10a-only  # the plan's calibration, after VERIFY_TRAIN

Every number is the registration's. The candidate is `e10_plan`; the comparison
arm is `e10_base` (the baseline recipe, same start, same session). `e10_base`
against `e9_ln` (same recipe, another run) is reported, never a criterion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dc1"))

import e10_common as K  # noqa: E402
from e8_compare import (P_BAND, Q_BAND, RETRAIN, control_cells, free_cells, rate,  # noqa: E402
                        regressions)

BASE, CAND = "e10_base", "e10_plan"
#: K10a's targets: FINDING_LENGTH_WEIGHTING.md 2.3, rows that fit 256 tokens.
Q_TARGET = {"benign": 0.0653, "phishing": 0.0451}
P_TARGET = {"benign": 0.733, "phishing": 0.394}
Q_REL, P_ABS = 0.10, 0.03


def k10a(rep: dict) -> dict:
    fails, out = [], {}
    for label in K.LABELS:
        pl = rep.get("plan", {}).get(label)
        if not pl:
            fails.append(f"{label}: no plan reading")
            continue
        qr = pl["Q"] / Q_TARGET[label]
        dp = pl["P"] - P_TARGET[label]
        out[label] = {"Q": pl["Q"], "Q_target": Q_TARGET[label], "Q_ratio": qr,
                      "P": pl["P"], "P_target": P_TARGET[label], "P_diff": dp}
        if abs(qr - 1) > Q_REL:
            fails.append(f"{label} plan Q {pl['Q']:.4f} is {qr:.3f} of {Q_TARGET[label]}")
        if abs(dp) > P_ABS:
            fails.append(f"{label} plan P {pl['P']:.4f} is {dp:+.4f} from {P_TARGET[label]}")
    return {"pass": not fails, "failures": fails, "readings": out}


def k10b(rep: dict) -> dict:
    fails = []
    for label in K.LABELS:
        for f, band in (("QRY", Q_BAND[label]), ("PTH", P_BAND[label])):
            r = rate(rep, "emission", CAND, label, f)
            if r is None or not band[0] <= r <= band[1]:
                fails.append(f"{label} {f}: {r} outside {band}")
    return {"pass": not fails, "failures": fails}


def k10c(rep: dict) -> dict:
    regs = (regressions(free_cells(rep, BASE), free_cells(rep, CAND))
            + regressions(control_cells(rep, BASE), control_cells(rep, CAND)))
    return {"pass": not regs, "regressions": regs}


def k10d(rep: dict) -> dict:
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
    """PREREG_E10 section 2, read literally."""
    if not res["K10a"]["pass"]:
        return ("K10a fails: the plan decision is itself miscalibrated; the explicit-decision "
                "hypothesis is refuted as stated. Next question: the hidden state's exposure "
                "(the owner's)")
    failing = [k for k in ("K10b", "K10c", "K10d") if not res[k]["pass"]]
    if not failing:
        return ("e10_plan is a candidate (K10a-K10d pass); the owner decides adoption "
                "-> N3")
    if "K10b" in failing:
        return (f"K10a passes; {', '.join(failing)} fail: the queries are lost after the plan "
                f"(K10b); reported, the owner decides")
    return f"K10a and K10b pass; {', '.join(failing)} fail: reported, the owner decides"


def reproducibility(rep: dict, e9: dict) -> dict:
    """Reported, not a criterion: e10_base against e9_ln (same recipe, another run)."""
    a, b = free_cells(rep, BASE), free_cells(e9, "e9_ln_raw")
    dx = {f"{k[0]} {k[1]}": a[k][0] - b[k][0] for k in sorted(set(a) & set(b))
          if a[k][0] is not None and b[k][0] is not None}
    em = {f"{label} {f}": (rate(rep, "emission", BASE, label, f) or 0)
          - (rate(e9, "emission", "e9_ln_raw", label, f) or 0)
          for label in K.LABELS for f in ("PTH", "QRY")}
    hon = {f"{label}_{st}": (rate(rep, "honouring", BASE, f"{label}_{st}") or 0)
           - (rate(e9, "honouring", "e9_ln_raw", f"{label}_{st}") or 0)
           for label in K.LABELS for st in K.STATES}
    return {"free_cell_x_difference": dx,
            "max_abs_free_cell_difference": max((abs(v) for v in dx.values()), default=None),
            "within_retrain_allowance": all(abs(v) <= RETRAIN for v in dx.values()),
            "emission_difference": em, "honouring_difference": hon}


def compare(rep: dict, e9: dict = None) -> dict:
    res = {"K10a": k10a(rep), "K10b": k10b(rep), "K10c": k10c(rep), "K10d": k10d(rep)}
    res["candidate"] = candidate(res)
    if e9 is not None:
        res["reproducibility_e10_base_vs_e9_ln"] = reproducibility(rep, e9)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e10", type=Path, default=ROOT / "data" / "e10")
    ap.add_argument("--k10a-only", action="store_true")
    args = ap.parse_args()
    e10 = args.e10
    if args.k10a_only:
        v = json.loads((e10 / "VERIFY_TRAIN.json").read_text(encoding="utf-8"))
        if v.get("verdict") != "PASS":
            raise SystemExit(f"STOP: the training side failed; nothing is read: {v.get('failures')}")
        plan = json.loads((e10 / "score_plan.json").read_text(encoding="utf-8"))
        r = k10a(plan)
        print(f"K10a: {'PASS' if r['pass'] else 'FAIL'}  {r['failures'] or ''}")
        for label, x in r["readings"].items():
            print(f"  {label}: plan Q {x['Q']:.4f} ({x['Q_ratio']:.3f} of {x['Q_target']}), "
                  f"plan P {x['P']:.4f} ({x['P_diff']:+.4f})")
        return
    verify = json.loads((e10 / "VERIFY.json").read_text(encoding="utf-8"))
    if verify.get("verdict") != "PASS":
        raise SystemExit(f"STOP: verification failed; nothing is read: {verify.get('failures')}")
    rep = json.loads((e10 / "score_report.json").read_text(encoding="utf-8"))
    e9_path = ROOT / "data" / "e9" / "score_report.json"
    e9 = json.loads(e9_path.read_text(encoding="utf-8")) if e9_path.is_file() else None
    res = compare(rep, e9)
    (e10 / "compare_report.json").write_text(json.dumps(res, indent=1, default=float),
                                             encoding="utf-8")
    for k in ("K10a", "K10b", "K10c", "K10d"):
        print(f"{k}: {'PASS' if res[k]['pass'] else 'FAIL'}  "
              f"{res[k].get('failures') or res[k].get('regressions') or ''}")
    print("CANDIDATE:", res["candidate"])


if __name__ == "__main__":
    main()
