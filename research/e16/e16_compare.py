"""E16 -- the registered criteria and the end rule (research/e16/PREREG_E16.md sections 5-6).
Reads the VERIFY files, V4, K16-2 (data/e16/k16_2.json), K16-3 (data/e16/k16_3.json),
e15_new's record (data/e15/score_report.json and its training record) and E16's
score_report.json; writes data/e16/compare_report.json.

    python research/e16/e16_compare.py

e16_cap against e15_new (a retrained model: the retraining allowance 0.0141 applies):
K16-1, K16-2, K16-3, G1-G10; adopted iff all hold. Then the end rule's row, applied
mechanically. E3 is read for the record. Rates between two arms use amendment 1
(e13_compare.worse_rate). Both models trained on the old corpus plus the stream, so they
share one copying reference and one natural rate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e15", ROOT / "research" / "e13", ROOT / "research" / "native",
          ROOT / "research" / "cr1", ROOT / "research" / "dc1", ROOT / "research" / "sc1",
          ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as K  # noqa: E402
import e13_compare as C13  # noqa: E402
import e16_common as E16  # noqa: E402
from dc1_compare import prop_worsening  # noqa: E402
from e8_compare import RETRAIN, control_cells, free_cells, regressions  # noqa: E402
from e12_compare import difference, finish_line, k12a, k12b, k12d, k12e  # noqa: E402

E16D = ROOT / "data" / "e16"
ARM, REF = E16.ARM, E16.BASELINE
QUESTION = {"control benign_P1Q1 QRY": ("benign", "P1Q1", "QRY"),
            "control benign_P1Q1 PTH": ("benign", "P1Q1", "PTH")}
#: the end rule's partial response: 2 x each cell's between-run SD (E13-D DX4)
PARTIAL = {"control benign_P1Q1 QRY": 2 * 0.0041, "control benign_P1Q1 PTH": 2 * 0.0036}
G3_MAX = 0.01


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def merged() -> dict:
    r15, r16 = load(ROOT / "data" / "e15" / "score_report.json"), load(E16D / "score_report.json")
    out = {}
    for block in ("free", "control", "emission", "slash", "honouring", "novelty", "s_honouring",
                  "plan_trace"):
        out[block] = {REF: r15[block][REF], ARM: r16[block][ARM]}
    out["copy"] = {}
    for cell, node in r16["copy"].items():
        if node["real_stream"] != r15["copy"][cell]["real_stream"]:
            raise SystemExit(f"STOP: the natural rate of {cell} differs from E15's on the same counts")
        out["copy"][cell] = {"real": node["real_stream"], ARM: node[ARM],
                             REF: {"generated": r15["copy"][cell][REF]["generated"],
                                   "per_generation": r15["copy"][cell][REF]["per_generation"]}}
        if node.get("real_domain_disjoint_stream"):
            out["copy"][cell]["real_domain_disjoint"] = node["real_domain_disjoint_stream"]
    out["stream_reuse"] = r16.get("stream_reuse")
    return out


def acc(rep, who, mode, label, state, field):
    row = (rep["free"][who][label]["fields"].get(field) if mode == "free"
           else rep["control"][who][f"{label}_{state}"]["fields"].get(field))
    return (row or {}).get("acceptance") or {}


def k16_1(rep) -> dict:
    cells = {}
    for name, (label, state, field) in QUESTION.items():
        a, r = acc(rep, ARM, "control", label, state, field), acc(rep, REF, "control", label, state, field)
        fall = r["x"] - a["x"]
        cells[name] = {"x_ref": r["x"], "x_arm": a["x"], "fall": fall, "v_ref": r["verdict"],
                       "v_arm": a["verdict"], "falls": fall > RETRAIN, "rises": -fall > RETRAIN,
                       "partial": fall >= PARTIAL[name]}
    return {"cells": cells, "pass": any(c["falls"] for c in cells.values())
            and not any(c["rises"] for c in cells.values())}


def g2(rep) -> dict:
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
    a = load(E16D / "train" / ARM / "train_result.json")["field_nll"]
    b = load(ROOT / "data" / "e15" / "train" / REF / "train_result.json")["field_nll"]
    rows = {f"{c} {f}": a[c]["val"][f] - b[c]["val"][f] for c in a for f in a[c]["val"] if f in b[c]["val"]}
    over = {k: v for k, v in rows.items() if not k.endswith("QRY") and v > G3_MAX}
    return {"pass": not over, "delta_val_nll": rows, "over": over}


def g6(rep) -> dict:
    rows = {f"PQ {key}": C13.worse_rate(r, rep["honouring"][ARM][key], True)
            for key, r in rep["honouring"][REF].items() if rep["honouring"][ARM].get(key)}
    for key, sv in (rep["s_honouring"].get(REF) or {}).items():
        for s, r in sv.items():
            x = ((rep["s_honouring"].get(ARM) or {}).get(key) or {}).get(s)
            if x and x.get("n"):
                rows[f"S {key} {s}"] = C13.worse_rate(r, x, True)
    return {"pass": not any(v["worse"] for v in rows.values()), "rows": rows}


def g7(rep) -> dict:
    """K12d's memorisation rule, and no copy rate moving further from the shared natural
    rate (amendment 1)."""
    memo = k12d({"copy": {c: {k: v for k, v in n.items() if k != REF} for c, n in rep["copy"].items()}}, ARM)
    dist = {}
    for cell, node in rep["copy"].items():
        rn = node["real"]["rate"]
        r, a = node[REF]["generated"], node[ARM]["generated"]
        dist[cell] = C13.worse_rate({"rate": r["rate"], "n": r["n"]}, {"rate": a["rate"], "n": a["n"]},
                                    r["rate"] <= rn)
    fails = list(memo["failures"]) + [f"distance {c}" for c, v in dist.items() if v["worse"]]
    return {"pass": not fails, "failures": fails, "memorisation": memo["memorisation"], "distance": dist}


def g8(rep) -> dict:
    nov = {}
    for mode in ("free", "control"):
        b, x = rep["novelty"][REF][mode], rep["novelty"][ARM][mode]
        nov[mode] = prop_worsening(b["novel"], b["n"], x["novel"], x["n"], False)
    return {"pass": not any(v["worsened"] for v in nov.values()), "novelty": nov}


def criteria(rep, targets) -> dict:
    k2, k3 = load(E16D / "k16_2.json"), load(E16D / "k16_3.json")
    g1 = regressions(free_cells(rep, REF), free_cells(rep, ARM)) + \
        regressions(control_cells(rep, REF), control_cells(rep, ARM))
    full = {label: acc(rep, ARM, "free", label, None, "FULL")["x"]
            - acc(rep, REF, "free", label, None, "FULL")["x"] for label in K.LABELS}
    vt = load(E16D / "VERIFY_TRAIN.json")
    clip = k12e(E16D / "train", ARM)
    out = {"K16-1": k16_1(rep),
           "K16-2": {"pass": bool(k2["pass"]), "benign": k2["benign"]["validation"],
                     "P1Q1": k2["benign"]["validation_by_stratum"].get("benign_P1Q1"),
                     "by_stratum": k2["benign"]["validation_by_stratum"],
                     "phishing_reported": k2["phishing"]["validation"]},
           "K16-3": {"pass": bool(k3["pass"]), **{label: k3[label][ARM] for label in K.LABELS}},
           "G1": {"pass": not g1, "regressions": g1},
           "G2": g2(rep),
           "G3": g3(),
           "G4": {**k12a(rep, targets, ARM), "moves_reported": C13.presence_rule(rep, targets, ARM, REF)["moves"]},
           "G5": k12b(rep, targets, ARM),
           "G6": g6(rep),
           "G7": g7(rep),
           "G8": g8(rep),
           "G9": {"delta_x": full, "pass": all(d <= RETRAIN for d in full.values())},
           "G10": {"pass": clip["pass"] and vt.get("verdict") == "PASS", "k12e": clip,
                   "verify_train": vt.get("verdict")}}
    out["adopted"] = all(v["pass"] for v in out.values() if isinstance(v, dict) and "pass" in v)
    return out


def end_rule(crit) -> dict:
    k1, k2 = crit["K16-1"], crit["K16-2"]["pass"]
    partial = any(c["partial"] for c in k1["cells"].values())
    if k1["pass"]:
        row, reading = 1, "capacity is binding for P1Q1; the route to E3 continues"
    elif k2 and partial:
        row, reading = 2, "capacity-sensitive, but not enough at 1.77x; E17 is the owner's choice"
    elif k2:
        row, reading = 3, ("the wider model fits benign queries better and P1Q1 does not respond: "
                           "the final E3 report")
    else:
        row, reading = 4, "the probe did not add usable fit; it does not answer the question"
    return {"row": row, "reading": reading, "partial_thresholds": PARTIAL}


def main() -> None:
    for step in ("TRAIN", "GENERATE"):
        v = load(E16D / f"VERIFY_{step}.json")
        if v.get("verdict") != "PASS":
            raise SystemExit(f"STOP: VERIFY_{step} failed; nothing is read: {v.get('failures')}")
    if not load(E16D / "score" / "v4.json").get("pass"):
        raise SystemExit("STOP: V4 failed; nothing is read")
    rep = merged()
    targets = load(ROOT / "data" / "e11" / "targets.json")
    crit = criteria(rep, targets)
    rule = end_rule(crit)
    f = finish_line(rep, ARM, k12b(rep, targets, ARM))
    res = {"registration": E16.REGISTRATION + " (sections 5-6)", "arm": ARM, "baseline": REF,
           "criteria": crit, "adopted": crit["adopted"], "end_rule": rule,
           "E3": {**f, "E3": "MET" if f["met"] and crit["adopted"] else
                  ("MET, NOT COUNTED (not adopted)" if f["met"] else "NOT MET")},
           "E3_baseline": finish_line(rep, REF, k12b(rep, targets, REF)),
           "difference": difference(rep, ARM, rep, REF), "stream_reuse": rep["stream_reuse"]}
    (E16D / "compare_report.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("e16_cap vs e15_new: " + "  ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}"
                                            for k, v in crit.items() if isinstance(v, dict) and "pass" in v)
          + f"  -> {'ADOPTED' if crit['adopted'] else 'NOT ADOPTED'}")
    for name, c in crit["K16-1"]["cells"].items():
        print(f"  K16-1 {name}: {c['x_ref']:.4f} ({c['v_ref']}) -> {c['x_arm']:.4f} ({c['v_arm']}), "
              f"fall {c['fall']:+.4f}{' (partial)' if c['partial'] else ''}")
    print(f"END RULE: row {rule['row']} -- {rule['reading']}")
    print(f"E3: {res['E3']['E3']} ({len(f['cells_not_cleared'])} cells not cleared, "
          f"{len(f['slash_rows_missed'])} slash rows missed)")
    print("wrote", E16D / "compare_report.json")


if __name__ == "__main__":
    main()
