"""E11 -- the registered criteria, per arm (research/native/PREREG_E11.md section 2).
Reads data/e11/VERIFY.json, score_report.json and the arms' training records, and
E10's score_report.json for `e11_rs`'s reference (`e10_base`); writes
data/e11/compare_report.json.

    python research/native/e11_compare.py

Every number is the registration's: the targets and n from data/e11/targets.json
(e11_targets.py), T = 2 sqrt(max(p(1-p), 1/n) / n), the retrain allowance 0.0141,
the guardrail rule, and at most 1% of steps clipped per band after step 1,000.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dc1"))

import e11_common as K  # noqa: E402
from e8_compare import RETRAIN, control_cells, free_cells, rate, regressions  # noqa: E402

CLIP_MAX = 0.01


def tolerance(p: float, n: float) -> float:
    """PREREG_E11 section 2: the detectability tolerance with the edge rule."""
    return 2 * math.sqrt(max(p * (1 - p), 1 / n) / n)


def check(q, p, n) -> dict:
    t = tolerance(p, n)
    return {"q": q, "p": p, "n": n, "T": t, "diff": None if q is None else q - p,
            "pass": q is not None and abs(q - p) <= t}


def k11a(rep, targets, arm) -> dict:
    out = {}
    for label in K.LABELS:
        for f, key in (("PTH", "path_rate_free"), ("QRY", "query_rate_free")):
            out[f"{label} {f}"] = check(rate(rep, "emission", arm, label, f),
                                        targets[label][key], 8000)
    return {"pass": all(v["pass"] for v in out.values()), "rows": out}


def k11b(rep, targets, arm) -> dict:
    out = {}
    for label in K.LABELS:
        for st, key in (("P0Q0", "free_bare"), ("P0Q1", "free_host_query")):
            t = targets[label][f"slash_{st}"]
            out[f"{label} {key}"] = check(rate(rep, "slash", arm, f"{key}_{label}"), t["free"],
                                          t["free_n_per_generation"])
            out[f"{label} control {st}"] = check(rate(rep, "slash", arm, f"control_{label}_{st}"),
                                                 t["control"], K.n_ctrl(label, st))
    return {"pass": all(v["pass"] for v in out.values()), "rows": out}


def cells(rep, e10, who):
    src = e10 if who == "e10_base" else rep
    return free_cells(src, who), control_cells(src, who)


def k11c(rep, e10, arm) -> dict:
    ref = K.REFERENCE[arm]
    (fb, cb), (fa, ca) = cells(rep, e10, ref), cells(rep, e10, arm)
    regs = regressions(fb, fa) + regressions(cb, ca)
    return {"pass": not regs, "reference": ref, "regressions": regs}


def k11d(rep, e10, arm) -> dict:
    from dc1_compare import prop_worsening
    ref = K.REFERENCE[arm]
    src = e10 if ref == "e10_base" else rep
    out = {}
    for mode in ("free", "control"):
        b, a = src["novelty"][ref][mode], rep["novelty"][arm][mode]
        out[f"{mode}_novelty"] = prop_worsening(b["novel"], b["n"], a["novel"], a["n"], False)
        out[f"{mode}_copying"] = prop_worsening(b["copied"], max(1, b["q"]), a["copied"],
                                                max(1, a["q"]), True)
    fails = [k for k, v in out.items() if v["worsened"]]
    return {"pass": not fails, "reference": ref, "failures": fails, "readings": out}


def k11e(train: Path, arm: str):
    """None where the arm has no repair; else the after-1,000 bands' clipping."""
    if not K.recipe(arm).get("grad_clip_late"):
        return None
    r = json.loads((train / arm / "train_result.json").read_text(encoding="utf-8"))
    bands = {k: v for k, v in (r.get("grad_diagnostics") or {}).items() if k.startswith("after_1000")}
    over = {k: v["clipped_share"] for k, v in bands.items() if v["clipped_share"] > CLIP_MAX}
    return {"pass": bool(bands) and not over, "bands": bands, "over": over}


def attribution(rep, e10, arm) -> dict:
    """Reported with each comparison: the change against the reference."""
    ref = K.REFERENCE[arm]
    src = e10 if ref == "e10_base" else rep
    em = {f"{label} {f}": (rate(rep, "emission", arm, label, f) or 0)
          - (rate(src, "emission", ref, label, f) or 0)
          for label in K.LABELS for f in ("PTH", "QRY")}
    hon = {f"{label}_{st}": (rate(rep, "honouring", arm, f"{label}_{st}") or 0)
           - (rate(src, "honouring", ref, f"{label}_{st}") or 0)
           for label in K.LABELS for st in K.STATES}
    (fb, cb), (fa, ca) = cells(rep, e10, ref), cells(rep, e10, arm)
    dx = {" ".join(map(str, k)): fa[k][0] - fb[k][0] for k in sorted(set(fa) & set(fb))}
    dx.update({" ".join(map(str, k)): ca[k][0] - cb[k][0] for k in sorted(set(ca) & set(cb))})
    return {"reference": ref, "emission": em, "honouring": hon, "cell_x": dx}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--e11", type=Path, default=ROOT / "data" / "e11")
    ap.add_argument("--train", type=Path, help="the training output (default: <e11>/train)")
    args = ap.parse_args()
    verify = json.loads((args.e11 / "VERIFY.json").read_text(encoding="utf-8"))
    if verify.get("verdict") != "PASS":
        raise SystemExit(f"STOP: verification failed; nothing is read: {verify.get('failures')}")
    rep = json.loads((args.e11 / "score_report.json").read_text(encoding="utf-8"))
    e10 = json.loads((ROOT / "data" / "e10" / "score_report.json").read_text(encoding="utf-8"))
    targets = json.loads((ROOT / "data" / "e11" / "targets.json").read_text(encoding="utf-8"))
    res = {"registration": "research/native/PREREG_E11.md", "arms": {}}
    for arm in K.ARMS:
        r = {"K11a": k11a(rep, targets, arm), "K11b": k11b(rep, targets, arm),
             "K11c": k11c(rep, e10, arm), "K11d": k11d(rep, e10, arm),
             "K11e": k11e(args.train or args.e11 / "train", arm),
             "attribution": attribution(rep, e10, arm)}
        crit = [k for k in ("K11a", "K11b", "K11c", "K11d", "K11e") if r[k] is not None]
        r["failing"] = [k for k in crit if not r[k]["pass"]]
        r["candidate"] = not r["failing"]
        if r["K11e"] is not None and not r["K11e"]["pass"]:
            r["note"] = "repair incomplete (K11e): this arm's repair-dependent readings are reported as such"
        res["arms"][arm] = r
        print(f"{arm:<10} " + "  ".join(f"{k} {'PASS' if r[k]['pass'] else 'FAIL'}" for k in crit)
              + f"  -> {'CANDIDATE' if r['candidate'] else 'not a candidate: ' + ', '.join(r['failing'])}")
    (args.e11 / "compare_report.json").write_text(json.dumps(res, indent=1, default=float),
                                                  encoding="utf-8")
    print("wrote", args.e11 / "compare_report.json")


if __name__ == "__main__":
    main()
