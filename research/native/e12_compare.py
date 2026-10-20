"""E12 -- the registered criteria, per arm, and the E3 finish line
(research/native/PREREG_E12.md section 2). Reads data/e12/VERIFY.json and
score_report.json, the arms' training records, and E11's step-20,000 reading
(data/e11_final/score_report.json) for the run-to-run spread; writes
data/e12/compare_report.json.

    python research/native/e12_compare.py

Every number is the registration's: E11's targets and tolerance with its edge rule
(data/e11/targets.json), the retrain allowance 0.0141, K11d's novelty rule, QC1-1's
memorisation reading (the host-level natural rate), and at most 1% of steps clipped
per band after step 1,000.

Reported beside the criteria (section 2): R1's leave-one-out, the slash plan's
effect (e12_no_r1 against e12_base; no arm ablates it, D6), the run-to-run spread,
QC1-2 on the candidate (data/e12/qc1, QC1's rule; one arm, so not replicated), and
section 3's predictions, scored.
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
sys.path.insert(0, str(HERE.parent / "qc1"))

import e12_common as K  # noqa: E402
from e8_compare import GATED, RANK, RETRAIN, control_cells, free_cells, rate, regressions  # noqa: E402
from e11_compare import check  # noqa: E402

CLIP_MAX = 0.01
#: Section 3's "no change beyond noise" and "stays under the retrain allowance",
#: read as |difference| <= the retrain allowance (fixed before any E12 data).
NOISE = RETRAIN


def k12a(rep, targets, arm) -> dict:
    """K11a's rule: free-mode PTH and QRY within T, both classes."""
    out = {}
    for label in K.LABELS:
        for f, key in (("PTH", "path_rate_free"), ("QRY", "query_rate_free")):
            out[f"{label} {f}"] = check(rate(rep, "emission", arm, label, f), targets[label][key], 8000)
    return {"pass": all(v["pass"] for v in out.values()), "rows": out}


def k12b(rep, targets, arm) -> dict:
    """K11b's rule: the 8 slash rows within T."""
    out = {}
    for label in K.LABELS:
        for st, key in (("P0Q0", "free_bare"), ("P0Q1", "free_host_query")):
            t = targets[label][f"slash_{st}"]
            out[f"{label} {key}"] = check(rate(rep, "slash", arm, f"{key}_{label}"), t["free"],
                                          t["free_n_per_generation"])
            out[f"{label} control {st}"] = check(rate(rep, "slash", arm, f"control_{label}_{st}"),
                                                 t["control"], K.n_ctrl(label, st))
    return {"pass": all(v["pass"] for v in out.values()), "rows": out}


def k12c(rep, arm) -> dict:
    """K11c's rule against e12_base: no gated cell's verdict worsens with x up by more
    than the retrain allowance."""
    ref = K.REFERENCE[arm]
    regs = (regressions(free_cells(rep, ref), free_cells(rep, arm))
            + regressions(control_cells(rep, ref), control_cells(rep, arm)))
    return {"pass": not regs, "reference": ref, "regressions": regs}


def k12d(rep, arm) -> dict:
    """Novelty: K11d's rule against e12_base, both modes. Memorisation: on QC1-1's
    cells, neither the copy rate nor the rate of copying strings seen once in
    training may be OVER the host-level natural rate (+ T); the domain-disjoint
    reading is reported for phishing."""
    import qc1_common as C
    from dc1_compare import prop_worsening
    fails, novelty, memo = [], {}, {}
    ref = K.REFERENCE.get(arm)
    if ref:
        for mode in ("free", "control"):
            b, a = rep["novelty"][ref][mode], rep["novelty"][arm][mode]
            novelty[mode] = prop_worsening(b["novel"], b["n"], a["novel"], a["n"], False)
            if novelty[mode]["worsened"]:
                fails.append(f"{mode}_novelty")
    for cell, node in rep["copy"].items():
        g, n = node[arm]["generated"], node[arm]["per_generation"]
        row = {}
        for ref_name, real in (("host", node["real"]),
                               ("domain_disjoint", node.get("real_domain_disjoint"))):
            if real is None:
                continue
            rr, t = C.rate_reading(g["rate"], real["rate"], n)
            sr, _ = C.rate_reading(g["singleton"], real["singleton"], n)
            row[ref_name] = {"generated": g["rate"], "natural": real["rate"], "T": t, "rate": rr,
                             "singleton_generated": g["singleton"],
                             "singleton_natural": real["singleton"], "singleton": sr}
        memo[cell] = row
        if row["host"]["rate"] == "OVER" or row["host"]["singleton"] == "OVER":
            fails.append(f"memorisation {cell}")
    return {"pass": not fails, "reference": ref, "failures": fails, "novelty": novelty,
            "memorisation": memo}


def k12e(train: Path, arm: str) -> dict:
    """K11e's rule: after step 1,000, at most 1% of steps clipped in every band."""
    r = json.loads((train / arm / "train_result.json").read_text(encoding="utf-8"))
    bands = {k: v for k, v in (r.get("grad_diagnostics") or {}).items() if k.startswith("after_1000")}
    over = {k: v["clipped_share"] for k, v in bands.items() if v["clipped_share"] > CLIP_MAX}
    return {"pass": bool(bands) and not over, "bands": bands, "over": over}


def finish_line(rep, arm, k12b_result) -> dict:
    """DECISION_2026-09-29 section 3: every gated cell CLEARED (free: all ten;
    control: every computable cell) and the slash within T."""
    missing = []
    for label in K.LABELS:
        for f in GATED:
            a = ((rep["free"][arm][label]["fields"].get(f) or {}).get("acceptance") or {})
            if a.get("verdict") != "CLEARED":
                missing.append({"cell": f"free {label} {f}", "x": a.get("x"), "verdict": a.get("verdict")})
    for key, part in rep["control"][arm].items():
        for f in GATED:
            a = ((part["fields"].get(f) or {}).get("acceptance") or {})
            if a.get("verdict") in RANK and a["verdict"] != "CLEARED":
                missing.append({"cell": f"control {key} {f}", "x": a.get("x"), "verdict": a["verdict"]})
    slash_missing = [k for k, v in k12b_result["rows"].items() if not v["pass"]]
    return {"met": not missing and not slash_missing, "cells_not_cleared": missing,
            "slash_rows_missed": slash_missing}


def difference(rep_a, who_a, rep_b, who_b) -> dict:
    """Reported: every cell's x and the emission, `a` against `b` (a - b)."""
    fa, fb = free_cells(rep_a, who_a), free_cells(rep_b, who_b)
    ca, cb = control_cells(rep_a, who_a), control_cells(rep_b, who_b)
    emission = {f"{label} {f}": (rate(rep_a, "emission", who_a, label, f) or 0)
                - (rate(rep_b, "emission", who_b, label, f) or 0)
                for label in K.LABELS for f in ("PTH", "QRY")}
    return {"free_x": {" ".join(k): fa[k][0] - fb[k][0] for k in sorted(set(fa) & set(fb))},
            "control_x": {" ".join(k): ca[k][0] - cb[k][0] for k in sorted(set(ca) & set(cb))},
            "emission": emission}


def qc1_2(e12: Path, dry: bool) -> dict:
    """Reported: QC1-2 on the candidate, with QC1's rule and its void rule
    (qc1_read). One arm, so it is not replicated and licenses nothing."""
    import statistics as st
    import qc1_common as C
    from qc1_read import valid_draws
    out = {}
    for cell in C.CELLS:
        name = C.cell_name(*cell)
        p = e12 / "qc1" / f"mix_{K.CANDIDATE}_{name}.json"
        if not p.is_file():
            out[name] = None
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("dry") and not dry:
            raise SystemExit(f"STOP: {p.name} is a dry run; it is never read")
        draws = valid_draws(d["draws"])
        if not draws:
            out[name] = {"n": d["n"], "void": len(d["draws"]), "reading": "VOID"}
            continue
        r = C.mix_reading([x["plain"] for x in draws], [x["mix"] for x in draws])
        r.update({"n": d["n"], "n_plain": d["n_plain"], "void": len(d["draws"]) - len(draws),
                  "shares": d["shares"], "generated": d["generated"],
                  "x_plain": st.mean(x["plain"] for x in draws),
                  "x_mix": st.mean(x["mix"] for x in draws)})
        if "mix_domain_disjoint" in draws[0]:
            r["x_mix_domain_disjoint"] = st.mean(x["mix_domain_disjoint"] for x in draws)
        for stratum in ("never_seen", "found"):
            sd = d.get(stratum, {})
            vd = [] if sd.get("insufficient") else valid_draws(sd["draws"])
            r[f"x_{stratum}"] = st.mean(x["x"] for x in vd) if vd else None
        out[name] = r
    return out


def predictions(res: dict, rep: dict, e11_cmp: dict) -> list:
    """Section 3's predictions, each scored from the readout; `held` is None where a
    prediction does not apply."""
    arms = res["arms"]
    slash = sorted(K.SLASH_ARMS)

    def failed(rows):
        return sorted(k for k, v in rows.items() if not v["pass"])

    def verdict(arm, label, f):
        return (((rep["free"][arm][label]["fields"].get(f) or {}).get("acceptance")) or {}).get("verdict")

    e11_rows = failed(e11_cmp["arms"]["e11_plan"]["K11b"]["rows"])
    lo = res["leave_one_out"]["R1"]["free_x"]
    b, ph = lo.get("benign QRY"), lo.get("phishing QRY")
    spread = res["retrain_spread"]["free_x"]
    worst = max(spread, key=lambda k: abs(spread[k])) if spread else None
    fl = arms[K.CANDIDATE]["finish_line"]
    missing = [c["cell"] for c in fl["cells_not_cleared"]]
    return [
        {"prediction": "K12b: every slash-plan arm passes",
         "held": all(arms[a]["K12b"]["pass"] for a in slash),
         "detail": {a: failed(arms[a]["K12b"]["rows"]) for a in slash}},
        {"prediction": "K12b: e12_base fails the same three rows E11 did",
         "held": failed(arms["e12_base"]["K12b"]["rows"]) == e11_rows,
         "detail": {"e12_base": failed(arms["e12_base"]["K12b"]["rows"]),
                    "e11_plan, step 20,000": e11_rows}},
        {"prediction": "K12a: every arm passes", "held": all(arms[a]["K12a"]["pass"] for a in K.ARMS),
         "detail": {a: failed(arms[a]["K12a"]["rows"]) for a in K.ARMS}},
        {"prediction": "R1: benign free QRY lower with it, by 0.003-0.010",
         "held": b is not None and -0.010 <= b <= -0.003, "detail": {"x difference": b}},
        {"prediction": f"R1: phishing free QRY, no change beyond noise (absolute difference at most {NOISE})",
         "held": ph is not None and abs(ph) <= NOISE, "detail": {"x difference": ph}},
        {"prediction": "the frame at every position (if licensed)", "held": None,
         "detail": "not applicable: QC1-9 read NO FRAME-LEAK, so D4 is not used"},
        {"prediction": "phishing free QRY stays CONVICTED in every arm",
         "held": all(verdict(a, "phishing", "QRY") == "CONVICTED" for a in K.ARMS),
         "detail": {a: verdict(a, "phishing", "QRY") for a in K.ARMS}},
        {"prediction": f"the run-to-run spread of free cells stays within the retrain allowance ({RETRAIN})",
         "held": worst is not None and abs(spread[worst]) <= RETRAIN,
         "detail": {"largest": worst, "x difference": spread.get(worst)}},
        {"prediction": "E3 finish line for e12_final: NOT met", "held": not fl["met"],
         "detail": {"cells not cleared": missing, "slash rows missed": fl["slash_rows_missed"]}},
        {"prediction": "phishing free QRY is the expected failure", "held": "free phishing QRY" in missing,
         "detail": {"free phishing QRY": verdict(K.CANDIDATE, "phishing", "QRY")}},
    ]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--e12", type=Path, default=ROOT / "data" / "e12")
    args = ap.parse_args()
    verify = json.loads((args.e12 / "VERIFY.json").read_text(encoding="utf-8"))
    if verify.get("verdict") != "PASS":
        raise SystemExit(f"STOP: verification failed; nothing is read: {verify.get('failures')}")
    dry = bool(verify.get("dry"))
    rep = json.loads((args.e12 / "score_report.json").read_text(encoding="utf-8"))
    targets = json.loads((ROOT / "data" / "e11" / "targets.json").read_text(encoding="utf-8"))
    e11 = json.loads((ROOT / "data" / "e11_final" / "score_report.json").read_text(encoding="utf-8"))
    e11_cmp = json.loads((ROOT / "data" / "e11_final" / "compare_report.json").read_text(encoding="utf-8"))
    res = {"registration": "research/native/PREREG_E12.md", "dry": dry, "arms": {}}
    for arm in K.ARMS:
        r = {"K12a": k12a(rep, targets, arm), "K12b": k12b(rep, targets, arm),
             "K12c": k12c(rep, arm) if arm in K.REFERENCE else None,
             "K12d": k12d(rep, arm), "K12e": k12e(args.e12 / "train", arm)}
        crit = [k for k in ("K12a", "K12b", "K12c", "K12d", "K12e") if r[k] is not None]
        r["failing"] = [k for k in crit if not r[k]["pass"]]
        r["candidate"] = not r["failing"]
        r["finish_line"] = finish_line(rep, arm, r["K12b"])
        res["arms"][arm] = r
        print(f"{arm:<10} " + "  ".join(f"{k} {'PASS' if r[k]['pass'] else 'FAIL'}" for k in crit)
              + f"  -> {'CANDIDATE' if r['candidate'] else 'not a candidate: ' + ', '.join(r['failing'])}"
              + f"; finish line {'MET' if r['finish_line']['met'] else 'not met'}")
    cand = res["arms"][K.CANDIDATE]
    res["E3"] = {"arm": K.CANDIDATE, "claimed": cand["candidate"] and cand["finish_line"]["met"]}
    res["leave_one_out"] = {lever: difference(rep, a, rep, b) for lever, (a, b) in K.LEAVE_ONE_OUT.items()}
    res["slash_plan_effect"] = difference(rep, "e12_no_r1", rep, "e12_base")
    res["retrain_spread"] = difference(rep, "e12_base", e11, "e11_plan")
    res["qc1_2"] = qc1_2(args.e12, dry)
    res["predictions"] = predictions(res, rep, e11_cmp)
    (args.e12 / "compare_report.json").write_text(json.dumps(res, indent=1, default=float),
                                                  encoding="utf-8")
    print(f"E3: {'CLAIMED' if res['E3']['claimed'] else 'NOT MET'} ({K.CANDIDATE})")
    print("wrote", args.e12 / "compare_report.json")


if __name__ == "__main__":
    main()
