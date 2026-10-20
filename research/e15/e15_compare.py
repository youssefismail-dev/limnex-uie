"""E15-B -- the registered criteria (research/e15/PREREG_E15.md section 9.8, with
section 13.2's instrument definitions). Reads the VERIFY files, V4, K15-2, K15-3, E14's
record (e12_d1p, the frozen baseline), E12's (e12_final, for G3) and E15's
score_report.json; writes data/e15/compare_report.json.

    python research/e15/e15_compare.py

e15_new against e12_d1p (a retrained model, so the retraining allowance 0.0141 applies):
K15-1, K15-2, K15-3 and G1-G10. Adopted iff all hold. E3 is read for the record.
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
import e15_common as E15  # noqa: E402
from dc1_compare import prop_worsening  # noqa: E402
from e8_compare import RETRAIN, control_cells, free_cells, regressions  # noqa: E402
from e12_compare import difference, finish_line, k12a, k12b, k12d, k12e  # noqa: E402

E15D = ROOT / "data" / "e15"
ARM, REF = E15.ARM, E15.BASELINE
#: K15-1's licensed cells: the admitted strata's query cells (section 9.4)
LICENSED = {"control benign_P0Q1 QRY": ("benign", "P0Q1"), "control benign_P1Q1 QRY": ("benign", "P1Q1")}
G3_MAX = 0.01


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def merged() -> dict:
    """e12_d1p from E14's record; e15_new from E15's."""
    r14, r15 = load(ROOT / "data" / "e14" / "score_report.json"), load(E15D / "score_report.json")
    out = {}
    for block in ("free", "control", "emission", "slash", "honouring", "novelty", "s_honouring",
                  "plan_trace"):
        out[block] = {REF: r14[block][REF], ARM: r15[block][ARM]}
    out["novelty_old_ref"] = r15["novelty_old_ref"]
    out["copy"], out["uptake"] = r15["copy"], r15["uptake"]
    out["copy_ref"] = {cell: node[REF] for cell, node in r14["copy"].items()}
    out["copy_ref_real"] = {cell: node["real"] for cell, node in r14["copy"].items()}
    return out


def acc(rep, who, mode, label, state, field):
    row = (rep["free"][who][label]["fields"].get(field) if mode == "free"
           else rep["control"][who][f"{label}_{state}"]["fields"].get(field))
    return (row or {}).get("acceptance") or {}


def k15_1(rep) -> dict:
    cells = {}
    for name, (label, state) in LICENSED.items():
        a, r = acc(rep, ARM, "control", label, state, "QRY"), acc(rep, REF, "control", label, state, "QRY")
        fall = r["x"] - a["x"]
        cells[name] = {"x_ref": r["x"], "x_arm": a["x"], "fall": fall, "v_ref": r["verdict"],
                       "v_arm": a["verdict"], "falls": fall > RETRAIN, "rises": -fall > RETRAIN}
    return {"cells": cells, "pass": any(c["falls"] for c in cells.values())
            and not any(c["rises"] for c in cells.values())}


def g2(rep) -> dict:
    """No PTH, SUB or LBL cell, free and control, both classes, rises by more than
    0.0141, whatever its verdict."""
    rows, fails = {}, []
    for cells in (lambda w: free_cells(rep, w), lambda w: control_cells(rep, w)):
        base, arm = cells(REF), cells(ARM)
        for key in sorted(set(base) & set(arm)):
            if key[-1] not in ("PTH", "SUB", "LBL"):
                continue
            d = arm[key][0] - base[key][0]
            name = " ".join(str(k) for k in key)
            rows[name] = {"x_ref": base[key][0], "x_arm": arm[key][0], "delta": d,
                          "v_ref": base[key][1], "v_arm": arm[key][1]}
            if d > RETRAIN:
                fails.append(name)
    return {"pass": not fails, "rises": fails, "rows": rows}


def g3() -> dict:
    """Every non-query field's validation NLL (the training record's per-field table,
    both classes) rises by no more than 0.01 nats per character against e12_final."""
    a = load(E15D / "train" / ARM / "train_result.json")["field_nll"]
    b = load(ROOT / "data" / "e12" / "train" / "e12_final" / "train_result.json")["field_nll"]
    rows = {f"{c} {f}": a[c]["val"][f] - b[c]["val"][f]
            for c in a for f in a[c]["val"] if f in b[c]["val"]}
    over = {k: v for k, v in rows.items() if not k.endswith("QRY") and v > G3_MAX}
    return {"pass": not over, "delta_val_nll": rows, "over": over}


def g6(rep) -> dict:
    """No control state's P, Q or S honouring falls (amendment 1)."""
    rows = {f"PQ {key}": C13.worse_rate(r, rep["honouring"][ARM][key], True)
            for key, r in rep["honouring"][REF].items() if rep["honouring"][ARM].get(key)}
    for key, sv in (rep["s_honouring"].get(REF) or {}).items():
        for s, r in sv.items():
            x = ((rep["s_honouring"].get(ARM) or {}).get(key) or {}).get(s)
            if x and x.get("n"):
                rows[f"S {key} {s}"] = C13.worse_rate(r, x, True)
    return {"pass": not any(v["worse"] for v in rows.values()), "rows": rows}


def g7(rep) -> dict:
    """K12d's memorisation rule (never OVER the natural rate), and no copy rate moving
    further from it (amendment 1). Section 13.2: e15_new is judged against its own
    natural rate (the real pool against old + stream counts); e12_d1p keeps its recorded
    one. 'Further' compares each arm's distance from its own natural rate: the arm's
    rate is translated by (natural_ref - natural_arm) and amendment 1's rule applied."""
    view = {cell: {"real": node["real_stream"],
                   "real_domain_disjoint": node.get("real_domain_disjoint_stream"),
                   ARM: node[ARM]} for cell, node in rep["copy"].items()}
    view = {c: {k: v for k, v in n.items() if v is not None} for c, n in view.items()}
    memo = k12d({"copy": view}, ARM)
    dist = {}
    for cell, node in rep["copy"].items():
        rn_arm, rn_ref = node["real_stream"]["rate"], rep["copy_ref_real"][cell]["rate"]
        r, a = rep["copy_ref"][cell]["generated"], node[ARM]["generated"]
        shifted = a["rate"] + (rn_ref - rn_arm)
        dist[cell] = {**C13.worse_rate({"rate": r["rate"], "n": r["n"]}, {"rate": shifted, "n": a["n"]},
                                       r["rate"] <= rn_ref),
                      "natural_ref": rn_ref, "natural_arm": rn_arm, "arm_raw": a["rate"]}
    fails = list(memo["failures"]) + [f"distance {c}" for c, v in dist.items() if v["worse"]]
    old_ref = {cell: {"generated": node[f"{ARM}_old_ref"]["generated"]["rate"],
                      "natural": node["real_old"]["rate"]} for cell, node in rep["copy"].items()}
    return {"pass": not fails, "failures": fails, "memorisation": memo["memorisation"],
            "distance": dist, "old_corpus_reading": old_ref}


def g8(rep) -> dict:
    """K12d's whole-URL novelty rule, both modes, against e12_d1p."""
    nov, old = {}, {}
    for mode in ("free", "control"):
        b, x = rep["novelty"][REF][mode], rep["novelty"][ARM][mode]
        nov[mode] = prop_worsening(b["novel"], b["n"], x["novel"], x["n"], False)
        o = rep["novelty_old_ref"][ARM][mode]
        old[mode] = prop_worsening(b["novel"], b["n"], o["novel"], o["n"], False)
    return {"pass": not any(v["worsened"] for v in nov.values()), "novelty": nov,
            "old_corpus_reading": old}


def criteria(rep, targets) -> dict:
    k2, k3 = load(E15D / "k15_2.json"), load(E15D / "k15_3.json")
    g1 = regressions(free_cells(rep, REF), free_cells(rep, ARM)) + \
        regressions(control_cells(rep, REF), control_cells(rep, ARM))
    full = {label: acc(rep, ARM, "free", label, None, "FULL")["x"]
            - acc(rep, REF, "free", label, None, "FULL")["x"] for label in K.LABELS}
    vt = load(E15D / "VERIFY_TRAIN.json")
    clip = k12e(E15D / "train", ARM)
    out = {"K15-1": k15_1(rep),
           "K15-2": {"pass": bool(k2["pass"]), "benign": k2["benign"]["validation"],
                     "by_stratum": k2["benign"].get("validation_by_stratum"),
                     "phishing_reported": k2["phishing"]["validation"]},
           "K15-3": {"pass": bool(k3["pass"]),
                     **{label: k3[label][ARM] for label in K.LABELS}},
           "G1": {"pass": not g1, "regressions": g1},
           "G2": g2(rep),
           "G3": g3(),
           "G4": {**k12a(rep, targets, ARM),
                  "moves_reported": C13.presence_rule(rep, targets, ARM, REF)["moves"]},
           "G5": k12b(rep, targets, ARM),
           "G6": g6(rep),
           "G7": g7(rep),
           "G8": g8(rep),
           "G9": {"delta_x": full, "pass": all(d <= RETRAIN for d in full.values())},
           "G10": {"pass": clip["pass"] and vt.get("verdict") == "PASS", "k12e": clip,
                   "verify_train": vt.get("verdict")}}
    out["adopted"] = all(v["pass"] for v in out.values() if isinstance(v, dict) and "pass" in v)
    return out


def main() -> None:
    for step in ("TRAIN", "GENERATE"):
        v = load(E15D / f"VERIFY_{step}.json")
        if v.get("verdict") != "PASS":
            raise SystemExit(f"STOP: VERIFY_{step} failed; nothing is read: {v.get('failures')}")
    if not load(E15D / "score" / "v4.json").get("pass"):
        raise SystemExit("STOP: V4 failed; nothing is read")
    rep = merged()
    targets = load(ROOT / "data" / "e11" / "targets.json")
    crit = criteria(rep, targets)
    f = finish_line(rep, ARM, k12b(rep, targets, ARM))
    res = {"registration": E15.REGISTRATION + " (sections 9.8, 13.2)", "arm": ARM, "baseline": REF,
           "criteria": crit, "adopted": crit["adopted"],
           "E3": {**f, "E3": "MET" if f["met"] and crit["adopted"] else
                  ("MET, NOT COUNTED (not adopted)" if f["met"] else "NOT MET")},
           "E3_baseline": finish_line(rep, REF, k12b(rep, targets, REF)),
           "difference": difference(rep, ARM, rep, REF), "uptake": rep["uptake"]}
    (E15D / "compare_report.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("e15_new vs e12_d1p: " + "  ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}"
                                            for k, v in crit.items() if isinstance(v, dict) and "pass" in v)
          + f"  -> {'ADOPTED' if crit['adopted'] else 'NOT ADOPTED'}")
    for name, c in crit["K15-1"]["cells"].items():
        print(f"  K15-1 {name}: {c['x_ref']:.4f} ({c['v_ref']}) -> {c['x_arm']:.4f} ({c['v_arm']}), "
              f"fall {c['fall']:+.4f}")
    print(f"E3: {res['E3']['E3']} ({len(f['cells_not_cleared'])} cells not cleared, "
          f"{len(f['slash_rows_missed'])} slash rows missed)")
    print("wrote", E15D / "compare_report.json")


if __name__ == "__main__":
    main()
