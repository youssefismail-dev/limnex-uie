"""E14 -- the registered criteria (research/e14/PREREG_E14.md section 3). Reads the
three VERIFY files, V4, K14-2 (data/e14/k14_2.json), K14-3 (data/e14/k14_3.json),
E12's published record (e12_final) and E14's score_report.json; writes
data/e14/compare_report.json.

    python research/e14/e14_compare.py

Q    e14_q against e12_final (a retrained model): K14-1, K14-2, K14-3, G1-G8.
D1'  e12_d1p against e12_final (the same weights): E13 section 3.2's rules for D1,
     verbatim, with amendment 1 -- K14-4, R-a to R-f (e13_compare's functions).
The combination: e14_qd against e14_q, the same rules.
E3   on every arm (e12_compare.finish_line); MET counts only where every lever the
     arm carries is adopted (section 3.4).
Every rate compared between two arms uses amendment 1 (e13_compare.worse_rate).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e13", ROOT / "research" / "native",
          ROOT / "research" / "cr1", ROOT / "research" / "dc1", ROOT / "research" / "sc1",
          ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as K  # noqa: E402
import e13_compare as C13  # noqa: E402
import e14_common as E14  # noqa: E402
from dc1_compare import prop_worsening  # noqa: E402
from e8_compare import RETRAIN, control_cells, free_cells, regressions  # noqa: E402
from e12_compare import finish_line, k12a, k12b, k12d, k12e  # noqa: E402

E14D = ROOT / "data" / "e14"
# e13_compare's file readers (DC1's paired contrasts) find E14's arms here.
C13.DIRS.update({"e12_d1p": E14D / "decode" / "gen", "e14_q": E14D / "generate" / "gen",
                 "e14_qd": E14D / "generate" / "gen"})


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def merged() -> dict:
    """e12_final from E12's published record; E14's arms from E14's."""
    r12, r14 = load(ROOT / "data" / "e12" / "score_report.json"), load(E14D / "score_report.json")
    out = {}
    for block in ("free", "control", "emission", "slash", "honouring", "novelty", "s_honouring",
                  "plan_trace"):
        out[block] = {"e12_final": r12[block]["e12_final"], **(r14.get(block) or {})}
    out["copy"] = {}
    for cell, node in r12["copy"].items():
        out["copy"][cell] = {k: v for k, v in node.items() if k in ("real", "real_domain_disjoint",
                                                                     "e12_final")}
        out["copy"][cell].update({k: v for k, v in (r14.get("copy") or {}).get(cell, {}).items()
                                  if k in E14.ARMS})
    return out


def acceptance(rep, arm, mode, label, state, field):
    if mode == "free":
        row = rep["free"][arm][label]["fields"].get(field)
    else:
        row = rep["control"][arm][f"{label}_{state}"]["fields"].get(field)
    return (row or {}).get("acceptance") or {}


# --- Q (section 3.1) --------------------------------------------------------------------

def k14_1(rep, arm="e14_q", ref="e12_final") -> dict:
    """N2b's kill, inverted: some target query cell falls by more than 0.0141."""
    cells = {}
    for name, (mode, label, state) in E14.TARGET_CELLS.items():
        a, r = acceptance(rep, arm, mode, label, state, "QRY"), acceptance(rep, ref, mode, label, state, "QRY")
        cells[name] = {"x_ref": r["x"], "x_arm": a["x"], "fall": r["x"] - a["x"],
                       "v_ref": r["verdict"], "v_arm": a["verdict"],
                       "moves": (r["x"] - a["x"]) > RETRAIN}
    return {"cells": cells, "pass": any(c["moves"] for c in cells.values())}


def honouring_pq(rep, arm, ref) -> dict:
    """G4: no control state's P or Q honouring falls (amendment 1). S honouring is
    reported, not judged, for Q (section 3.1)."""
    rows = {f"PQ {key}": C13.worse_rate(r, rep["honouring"][arm][key], True)
            for key, r in rep["honouring"][ref].items() if rep["honouring"][arm].get(key)}
    s_rows = {}
    for key, sv in (rep["s_honouring"].get(ref) or {}).items():
        for s, r in sv.items():
            x = ((rep["s_honouring"].get(arm) or {}).get(key) or {}).get(s)
            if x and x.get("n"):
                s_rows[f"S {key} {s}"] = C13.worse_rate(r, x, True)
    return {"pass": not any(v["worse"] for v in rows.values()), "rows": rows, "s_reported": s_rows}


def copying_q(rep, arm, ref) -> dict:
    """G5: K12d's memorisation rule (never OVER the natural rate), and no cell's copy
    rate moves further from its natural rate (amendment 1). DC1's query-copying clause
    is NOT part of it (section 3.1)."""
    memo = k12d(rep, arm)
    dist = {}
    for cell, node in rep["copy"].items():
        if arm not in node or ref not in node:
            continue
        rn = node["real"]["rate"]
        r, a = node[ref]["generated"], node[arm]["generated"]
        dist[cell] = C13.worse_rate({"rate": r["rate"], "n": r["n"]},
                                    {"rate": a["rate"], "n": a["n"]}, r["rate"] <= rn)
    fails = list(memo["failures"]) + [f"distance {c}" for c, v in dist.items() if v["worse"]]
    return {"pass": not fails, "failures": fails, "memorisation": memo["memorisation"],
            "distance": dist}


def novelty_q(rep, arm, ref) -> dict:
    """G6: K12d's novelty rule against e12_final: whole-URL novelty, both modes."""
    nov = {}
    for mode in ("free", "control"):
        b, x = rep["novelty"][ref][mode], rep["novelty"][arm][mode]
        nov[mode] = prop_worsening(b["novel"], b["n"], x["novel"], x["n"], False)
    return {"pass": not any(v["worsened"] for v in nov.values()), "novelty": nov}


def criteria_q(rep, targets) -> dict:
    arm, ref = "e14_q", "e12_final"
    k2, k3 = load(E14D / "k14_2.json"), load(E14D / "k14_3.json")
    g1 = regressions(free_cells(rep, ref), free_cells(rep, arm)) + \
        regressions(control_cells(rep, ref), control_cells(rep, arm))
    full = {label: acceptance(rep, arm, "free", label, None, "FULL")["x"]
            - acceptance(rep, ref, "free", label, None, "FULL")["x"] for label in K.LABELS}
    a = k12a(rep, targets, arm)
    out = {"K14-1": k14_1(rep),
           "K14-2": {"pass": bool(k2["pass"]),
                     **{label: k2[label]["validation"] for label in K.LABELS}},
           "K14-3": {"pass": bool(k3["pass"]), **{label: k3[label] for label in K.LABELS}},
           "G1": {"pass": not g1, "regressions": g1},
           "G2": {"pass": a["pass"], "k12a": a,
                  "moves_reported": C13.presence_rule(rep, targets, arm, ref)["moves"]},
           "G3": C13.slash_kept(rep, targets, arm, ref),
           "G4": honouring_pq(rep, arm, ref),
           "G5": copying_q(rep, arm, ref),
           "G6": novelty_q(rep, arm, ref),
           "G7": {"delta_x": full, "pass": all(d <= RETRAIN for d in full.values())},
           "G8": k12e(E14D / "train", arm)}
    out["adopted"] = all(v["pass"] for v in out.values() if isinstance(v, dict) and "pass" in v)
    return out


# --- D1' and the combination (sections 3.2-3.3): E13's D1 rules verbatim ---------------

def criteria_decoder(rep, targets, arm, ref) -> dict:
    cells = C13.cell_contrasts(rep, arm, ref)
    out = {"K14-4": k12b(rep, targets, arm), "R-a": C13.r_a(cells),
           "R-b": C13.presence_rule(rep, targets, arm, ref),
           "R-c": C13.obedience_rule(rep, arm, ref),
           "R-d": C13.copying_rule(rep, arm, ref),
           "R-e": {"pass": True, "note": "the three VERIFY files and V4 passed (checked at start)"},
           "R-f": {"pass": True, "note": "single pass: D1' only rescales host-end logits"}}
    out["adopted"] = all(v["pass"] for v in out.values())
    return {**out, "cells": cells}


def e3_readings(rep, targets, adopted: dict) -> dict:
    """Section 3.4: E3 on every arm; MET counts only where every lever is adopted."""
    needs = {"e14_q": ("Q",), "e12_d1p": ("D1'",), "e14_qd": ("Q", "D1'", "combination")}
    out = {}
    for arm, levers in needs.items():
        f = finish_line(rep, arm, k12b(rep, targets, arm))
        counts = all(adopted[x] for x in levers)
        out[arm] = {**f, "levers": list(levers), "counts": counts,
                    "E3": "MET" if f["met"] and counts else
                          ("MET, NOT COUNTED (a lever is not adopted)" if f["met"] else "NOT MET")}
    return out


def main() -> None:
    for step in ("TRAIN", "DECODE", "GENERATE"):
        v = load(E14D / f"VERIFY_{step}.json")
        if v.get("verdict") != "PASS":
            raise SystemExit(f"STOP: VERIFY_{step} failed; nothing is read: {v.get('failures')}")
    if not load(E14D / "score" / "v4.json").get("pass"):
        raise SystemExit("STOP: V4 failed; nothing is read")
    rep = merged()
    targets = load(ROOT / "data" / "e11" / "targets.json")
    q = criteria_q(rep, targets)
    d1p = criteria_decoder(rep, targets, "e12_d1p", "e12_final")
    combo = criteria_decoder(rep, targets, "e14_qd", "e14_q")
    adopted = {"Q": q["adopted"], "D1'": d1p["adopted"],
               "combination": q["adopted"] and d1p["adopted"] and combo["adopted"]}
    res = {"registration": E14.REGISTRATION + " (section 3)", "Q": q, "D1'": d1p,
           "combination": combo, "adopted": adopted, "E3": e3_readings(rep, targets, adopted)}
    (E14D / "compare_report.json").write_text(json.dumps(res, indent=1, default=float),
                                              encoding="utf-8")
    for name, node in (("Q  e14_q", q), ("D1' e12_d1p", d1p), ("Q+D1' e14_qd", combo)):
        print(f"{name}: " + "  ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}" for k, v in node.items()
                                      if isinstance(v, dict) and "pass" in v)
              + f"  -> {'ADOPTED' if node['adopted'] else 'not adopted'}")
    for arm, e in res["E3"].items():
        print(f"E3 {arm}: {e['E3']}")
    print("wrote", E14D / "compare_report.json")


if __name__ == "__main__":
    main()
