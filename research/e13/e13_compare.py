"""E13 -- the registered criteria (research/e13/PREREG_E13.md section 3, amendment 1).
Reads the three VERIFY files, V4, E12's published record (e12_final, the owner's
decision 4) and E13's score_report.json; writes data/e13/compare_report.json.

    python research/e13/e13_compare.py

C   e13_ctx against e12_final (a retrained model): K13-1, K13-2 (DX8 on e13_ctx,
    data/e13/dx8_ctx.json), G1-G7.
D1  e12_d1 against e12_final, D2 e12_dec against e12_d1 (the same weights): DC1's
    paired cell rules with Holm per mode, K13-3 / K13-4, R-a to R-f.
Every rate compared between two arms uses amendment 1: at least T, and the
two-proportion 95% interval excludes 0 in the worsening direction.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "native", ROOT / "research" / "cr1",
          ROOT / "research" / "dc1", ROOT / "research" / "sc1", ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as K  # noqa: E402
import e13_common as E13  # noqa: E402
from dc1_compare import prop_worsening  # noqa: E402
from e8_compare import RANK, RETRAIN, control_cells, free_cells, regressions  # noqa: E402
from e12_compare import k12a, k12b, k12d  # noqa: E402
from sc1_compare import cap_of, contrast, holm, pairable  # noqa: E402

E13D = ROOT / "data" / "e13"
DIRS = {"e12_final": ROOT / "data" / "e12" / "gen", "e12_d1": E13D / "decode" / "gen",
        "e12_dec": E13D / "decode" / "gen", "e13_ctx": E13D / "ctx" / "gen"}
CELL_FIELDS = ("LBL", "SUF", "SUB", "PTH", "QRY", "FULL")
ALPHA = 0.05
K13_2_BASE = 0.0774              # e12_final's x_GV - x_RC, free phishing QRY (E13D_RESULTS)
D2_TARGETS = {"D2-Q": ("benign", "P0Q1", "QRY"), "D2-P": ("benign", "P1Q1", "PTH")}
BUNDLE = ROOT / "data" / "bundle2"


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def tolerance(p: float, n: float) -> float:
    return 2 * math.sqrt(max(p * (1 - p), 1 / n) / n)


def merged() -> dict:
    """e12_final from E12's published record; E13's arms from E13's."""
    r12, r13 = load(ROOT / "data" / "e12" / "score_report.json"), load(E13D / "score_report.json")
    out = {}
    for block in ("free", "control", "emission", "slash", "honouring", "novelty", "s_honouring",
                  "plan_trace"):
        out[block] = {"e12_final": r12[block]["e12_final"], **(r13.get(block) or {})}
    out["copy"] = {}
    for cell, node in r12["copy"].items():
        out["copy"][cell] = {k: v for k, v in node.items() if k in ("real", "real_domain_disjoint",
                                                                     "e12_final")}
        out["copy"][cell].update({k: v for k, v in (r13.get("copy") or {}).get(cell, {}).items()
                                  if k in E13.ARMS})
    return out


_files, _real = {}, {}


def files(arm, mode, label, state):
    key = (arm, mode, label, state)
    if key not in _files:
        names = ([f"free_{arm}_{label}_g{g}.json" for g in range(K.G)] if mode == "free" else
                 [f"ctrl_{arm}_{label}_{state}_g{g}.json" for g in range(K.G)])
        _files[key] = [load(DIRS[arm] / n) for n in names]
    return _files[key]


def real_count(label, state, field):
    """DC1's cap input: the real pool's non-empty strings of the field."""
    from cr1_score import pools
    from uie.evaluation import FIELDS
    key = (label, state, field)
    if key not in _real:
        real = pools(BUNDLE, label, K.STATES[state] if state else None)["name"]
        _real[key] = sum(1 for u in real if FIELDS[field](u))
    return _real[key]


def verdict(row):
    return ((row or {}).get("acceptance") or {}).get("verdict")


def cell_contrasts(rep, arm, ref) -> dict:
    """Every gated cell and FULL, per mode: the paired contrast where n pairs
    (DC1's rule), else unpaired."""
    from uie.evaluation import FIELDS
    out = {"free": {}, "control": {}}
    for label in K.LABELS:
        fa, fr = files(arm, "free", label, None), files(ref, "free", label, None)
        for f in CELL_FIELDS:
            ra = rep["free"][arm][label]["fields"].get(f)
            rr = rep["free"][ref][label]["fields"].get(f)
            fk = lambda xs: [sum(1 for u in x if FIELDS[f](u)) for x in xs]  # noqa: E731
            c = contrast(ra, rr, pairable(cap_of(real_count(label, None, f)), fk(fa), fk(fr)))
            out["free"][f"{label} {f}"] = c and {**c, "v_ref": verdict(rr), "v_arm": verdict(ra)}
    for key, part in rep["control"][arm].items():
        label, st = part["label"], part["state"]
        fa, fr = files(arm, "control", label, st), files(ref, "control", label, st)
        for f in CELL_FIELDS:
            ra, rr = part["fields"].get(f), rep["control"][ref][key]["fields"].get(f)
            if not (ra and rr):
                continue
            fk = lambda xs: [sum(1 for u in x if FIELDS[f](u)) for x in xs]  # noqa: E731
            c = contrast(ra, rr, pairable(cap_of(real_count(label, st, f)), fk(fa), fk(fr)))
            out["control"][f"{key} {f}"] = c and {**c, "v_ref": verdict(rr), "v_arm": verdict(ra)}
    return out


def r_a(cells: dict) -> dict:
    """DC1's REGRESSION rule per mode: Delta > 0 at Holm p < 0.05 over all the mode's
    cells, or the verdict worsens."""
    regs = []
    for mode, cs in cells.items():
        adj = holm({k: c["p"] for k, c in cs.items() if c})
        for k, c in cs.items():
            if not c:
                continue
            worse_v = (c["v_ref"] in RANK and c["v_arm"] in RANK and RANK[c["v_arm"]] > RANK[c["v_ref"]])
            if (c["delta"] > 0 and adj[k] < ALPHA) or worse_v:
                regs.append({"mode": mode, "cell": k, "delta": c["delta"], "p_holm": adj[k],
                             "v_ref": c["v_ref"], "v_arm": c["v_arm"]})
    return {"pass": not regs, "regressions": regs}


def worse_rate(ref: dict, arm: dict, worse_if_lower: bool) -> dict:
    """Amendment 1: a two-arm change counts only if it is at least T (at the
    reference rate and the arm's n) and its 95% interval excludes 0 in the
    worsening direction."""
    pr, pa = ref["rate"], arm["rate"]
    d = (pr - pa) if worse_if_lower else (pa - pr)
    se = math.sqrt(pr * (1 - pr) / ref["n"] + pa * (1 - pa) / arm["n"])
    t = tolerance(pr, arm["n"])
    worse = d >= t and d - 1.96 * se > 0
    return {"ref": pr, "arm": pa, "worsening": d, "T": t, "lo": d - 1.96 * se, "worse": worse}


def presence_rule(rep, targets, arm, ref) -> dict:
    """R-b / G2: K12a, and no free-mode PTH or QRY rate moves materially (amendment 1),
    in either direction."""
    a = k12a(rep, targets, arm)
    moves = {}
    for label in K.LABELS:
        for f in ("PTH", "QRY"):
            r, x = rep["emission"][ref][label][f], rep["emission"][arm][label][f]
            lo, hi = worse_rate(r, x, True), worse_rate(r, x, False)
            moves[f"{label} {f}"] = {"down": lo, "up": hi, "worse": lo["worse"] or hi["worse"]}
    return {"pass": a["pass"] and not any(m["worse"] for m in moves.values()), "k12a": a,
            "moves": moves}


def obedience_rule(rep, arm, ref) -> dict:
    """R-c / G4: no state's P/Q honouring, and no S honouring, falls materially."""
    rows = {}
    for key, r in rep["honouring"][ref].items():
        x = rep["honouring"][arm].get(key)
        if x:
            rows[f"PQ {key}"] = worse_rate(r, x, True)
    for key, sv in (rep["s_honouring"].get(ref) or {}).items():
        for s, r in sv.items():
            x = ((rep["s_honouring"].get(arm) or {}).get(key) or {}).get(s)
            if x and x.get("n"):
                rows[f"S {key} {s}"] = worse_rate(r, x, True)
    return {"pass": not any(v["worse"] for v in rows.values()), "rows": rows}


def copying_rule(rep, arm, ref) -> dict:
    """G5 / R-d: K12d's memorisation rule (never OVER), and no cell's copy rate moves
    materially further from its natural rate (amendment 1); and DC1's novelty rule."""
    memo = k12d(rep, arm)
    dist = {}
    for cell, node in rep["copy"].items():
        if arm not in node or ref not in node:
            continue
        rn = node["real"]["rate"]
        r, a = node[ref]["generated"], node[arm]["generated"]
        away_if_lower = r["rate"] <= rn               # under-copying: lower is further away
        dist[cell] = worse_rate({"rate": r["rate"], "n": r["n"]}, {"rate": a["rate"], "n": a["n"]},
                                away_if_lower)
    nov = {}
    for mode in ("free", "control"):
        b, x = rep["novelty"][ref][mode], rep["novelty"][arm][mode]
        nov[f"{mode} novelty"] = prop_worsening(b["novel"], b["n"], x["novel"], x["n"], False)
        nov[f"{mode} query copying"] = prop_worsening(b["copied"], b["q"], x["copied"], x["q"], True)
    ok = (not memo["failures"] and not any(v["worse"] for v in dist.values())
          and not any(v["worsened"] for v in nov.values()))
    return {"pass": ok, "memorisation": memo, "distance": dist, "novelty": nov}


def slash_kept(rep, targets, arm, ref) -> dict:
    """G3: no K12b row that the reference passed fails."""
    a, r = k12b(rep, targets, arm), k12b(rep, targets, ref)
    lost = [k for k, v in r["rows"].items() if v["pass"] and not a["rows"][k]["pass"]]
    return {"pass": not lost, "lost": lost, "k12b": a}


def criteria_c(rep, targets) -> dict:
    arm, ref = "e13_ctx", "e12_final"
    xa = rep["free"][arm]["phishing"]["fields"]["QRY"]["acceptance"]
    xr = rep["free"][ref]["phishing"]["fields"]["QRY"]["acceptance"]
    k1 = {"x_ref": xr["x"], "x_arm": xa["x"], "fall": xr["x"] - xa["x"], "verdict": xa["verdict"],
          "pass": (xr["x"] - xa["x"]) > RETRAIN and xa["verdict"] != "CONVICTED"}
    dx8 = load(E13D / "dx8_ctx.json")
    k2 = {"gap_arm": dx8["diff"], "gap_ref": K13_2_BASE, "pass": dx8["diff"] <= 0.5 * K13_2_BASE,
          "x_gv": dx8["x_gv"], "x_rc": dx8["x_rc"]}
    g1 = regressions(free_cells(rep, ref), free_cells(rep, arm)) + \
        regressions(control_cells(rep, ref), control_cells(rep, arm))
    full = (rep["free"][arm]["phishing"]["fields"]["FULL"]["acceptance"]["x"]
            - rep["free"][ref]["phishing"]["fields"]["FULL"]["acceptance"]["x"])
    out = {"K13-1": k1, "K13-2": k2, "G1": {"pass": not g1, "regressions": g1},
           "G2": presence_rule(rep, targets, arm, ref), "G3": slash_kept(rep, targets, arm, ref),
           "G4": obedience_rule(rep, arm, ref), "G5_G6": copying_rule(rep, arm, ref),
           "G7": {"delta_x": full, "pass": full <= RETRAIN}}
    out["adopted"] = all(v["pass"] for v in out.values() if isinstance(v, dict) and "pass" in v)
    return out


def criteria_d(rep, targets, settings) -> dict:
    out = {}
    d1_cells = cell_contrasts(rep, "e12_d1", "e12_final")
    d1 = {"K13-3": k12b(rep, targets, "e12_d1"), "R-a": r_a(d1_cells),
          "R-b": presence_rule(rep, targets, "e12_d1", "e12_final"),
          "R-c": obedience_rule(rep, "e12_d1", "e12_final"),
          "R-d": copying_rule(rep, "e12_d1", "e12_final"),
          "R-e": {"pass": True, "note": "the three VERIFY files and V4 passed (checked at start)"},
          "R-f": {"pass": True, "note": "single pass: no blend, no second prediction (by construction)"}}
    d1["adopted"] = all(v["pass"] for v in d1.values())
    out["D1"] = {**d1, "cells": d1_cells}
    applied = [s for s, r in settings["scopes"].items() if r.get("applied")]
    if not applied:
        out["D2"] = {"status": "NOT APPLICABLE: the fit applied no scope; e12_dec was not generated"}
        return out
    d2_cells = cell_contrasts(rep, "e12_dec", "e12_d1")
    targets_d2 = {}
    adj = holm({s: (d2_cells["control"].get(f"{D2_TARGETS[s][0]}_{D2_TARGETS[s][1]} {D2_TARGETS[s][2]}")
                    or {"p": 1.0})["p"] for s in applied})
    for s in applied:
        label, st, f = D2_TARGETS[s]
        c = d2_cells["control"].get(f"{label}_{st} {f}")
        targets_d2[s] = {"cell": f"control {label} {st} {f}", **(c or {}), "p_holm": adj[s],
                         "improved": bool(c) and c["delta"] < 0 and adj[s] < ALPHA,
                         "fixed": bool(c) and c["v_arm"] == "CLEARED"}
    k4 = {"targets": targets_d2,
          "pass": any(t["improved"] or t["fixed"] for t in targets_d2.values())}
    d2 = {"K13-4": k4, "R-a": r_a(d2_cells), "R-b": presence_rule(rep, targets, "e12_dec", "e12_d1"),
          "R-c": obedience_rule(rep, "e12_dec", "e12_d1"), "R-d": copying_rule(rep, "e12_dec", "e12_d1"),
          "R-e": d1["R-e"], "R-f": d1["R-f"]}
    d2["adopted"] = d1["adopted"] and all(v["pass"] for v in d2.values() if isinstance(v, dict))
    out["D2"] = {**d2, "applied": applied, "cells": d2_cells}
    return out


def main() -> None:
    for step in ("TRAIN", "DECODE", "GENERATE"):
        v = load(E13D / f"VERIFY_{step}.json")
        if v.get("verdict") != "PASS":
            raise SystemExit(f"STOP: VERIFY_{step} failed; nothing is read: {v.get('failures')}")
    v4 = load(E13D / "score" / "v4.json")
    if not v4.get("pass"):
        raise SystemExit("STOP: V4 failed; nothing is read")
    rep = merged()
    targets = load(ROOT / "data" / "e11" / "targets.json")
    settings = E13.load_settings(E13D / E13.SETTINGS_FILE)
    res = {"registration": E13.REGISTRATION + " (section 3, amendment 1)",
           "C": criteria_c(rep, targets), **criteria_d(rep, targets, settings)}
    (E13D / "compare_report.json").write_text(json.dumps(res, indent=1, default=float),
                                              encoding="utf-8")
    c = res["C"]
    print("C  e13_ctx: " + "  ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}" for k, v in c.items()
                                     if isinstance(v, dict) and "pass" in v)
          + f"  -> {'ADOPTED' if c['adopted'] else 'not adopted'}")
    for name in ("D1", "D2"):
        d = res[name]
        if "status" in d:
            print(f"{name}: {d['status']}")
            continue
        print(f"{name}: " + "  ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}" for k, v in d.items()
                                    if isinstance(v, dict) and "pass" in v)
              + f"  -> {'ADOPTED' if d['adopted'] else 'not adopted'}")
    print("wrote", E13D / "compare_report.json")


if __name__ == "__main__":
    main()
