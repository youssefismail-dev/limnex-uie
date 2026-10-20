"""E17 -- the registered criteria (research/e17/PREREG_E17.md sections 2-4). Two parts,
read in order:

    python research/e17/e17_compare.py --part b5   # e16_b5 against e16_cap (the same weights)
                                                   #   -> data/e17/compare_b5.json
    python research/e17/e17_compare.py --part a    # e17_a against its reference
                                                   #   -> data/e17/compare_a.json

B5 (section 2), e16_b5 against e16_cap: K17-B1 (the 8 slash rows within T), K17-B2 (Q
honouring on every Q = 1 plan, from the files; no control state's P, Q or S honouring
falls), R-a' (DECISION_2026-10-05 section 2.3: e14_rerule.r_a_prime over
e13_compare.cell_contrasts), R-b (K12a; no free PTH or QRY rate moves materially, except
QRY toward its target), R-c and R-d (e13_compare's rules), R-e, R-f. Adopted iff all
hold. The result decides e17_a's decode (E17_generate_b5 or E17_generate_d1p).

A (section 3.3), e17_a against e16_b5 if B5 was adopted, else e16_cap -- a retrained
model, so the retraining allowance applies: K17-1, K17-2 (data/e17/k17_2.json), K17-3
(data/e17/k17_3.json), G1-G10 as PREREG_E16 section 5 (G5 strict). Copying and novelty
use each model's own training data (PREREG_E15 section 13.2): e17_a's natural rates are
on old + both streams' counts, the reference's on old + stream 1, so G7 compares each
arm's distance from its own natural rate (E15's g7). E3 for the record.
Every rate compared between two arms uses amendment 1 (e13_compare.worse_rate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e16", ROOT / "research" / "e15", ROOT / "research" / "e14",
          ROOT / "research" / "e13", ROOT / "research" / "native", ROOT / "research" / "cr1",
          ROOT / "research" / "dc1", ROOT / "research" / "sc1", ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as K  # noqa: E402
import e12_score as S12  # noqa: E402
import e13_compare as C13  # noqa: E402
import e17_common as E17  # noqa: E402
from dc1_compare import prop_worsening  # noqa: E402
from e8_compare import RETRAIN, control_cells, free_cells, regressions  # noqa: E402
from e12_compare import difference, finish_line, k12a, k12b, k12d, k12e  # noqa: E402
from e14_rerule import r_a_prime  # noqa: E402

E17D, E16D = ROOT / "data" / "e17", ROOT / "data" / "e16"
BASE, B5ARM, ARM = E17.BASELINE, E17.DECODE_ARM, E17.TRAIN_ARM
# e13_compare's file readers (DC1's paired contrasts) find E17's arms here.
C13.DIRS.update({BASE: E16D / "generate" / "gen", B5ARM: E17D / "decode" / "gen",
                 ARM: E17D / "generate" / "gen"})
BLOCKS = ("free", "control", "emission", "slash", "honouring", "novelty", "s_honouring", "plan_trace")
#: K17-1: the three cells the question names, and the four query cells none may raise
QUESTION = {"free phishing QRY": ("free", "phishing", None),
            "control benign_P0Q1 QRY": ("control", "benign", "P0Q1"),
            "control benign_P1Q1 QRY": ("control", "benign", "P1Q1")}
GUARDED = {**QUESTION, "free benign QRY": ("free", "benign", None)}
G3_MAX = 0.01
#: Amendment 2 (PREREG_E17 section 10): ADOPTED by the owner 2026-10-07, in force, before
#: any E17 output was read. True judges K17-B2's first clause as amended (no Q = 1 URL ends
#: or starts a fragment before its query begins; cut-offs at max_len counted apart); False
#: would judge it as first registered (Q honouring 100% by the instruments). Both readings
#: are always computed and written.
AMENDMENT_2 = True


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def require(*steps: str) -> None:
    for step in steps:
        v = load(E17D / f"VERIFY_{step}.json")
        if v.get("verdict") != "PASS":
            raise SystemExit(f"STOP: VERIFY_{step} failed; nothing is read: {v.get('failures')}")
    if not load(E17D / "score" / "v4.json").get("pass"):
        raise SystemExit("STOP: V4 failed; nothing is read")


def acc(rep, who, mode, label, state, field):
    row = (rep["free"][who][label]["fields"].get(field) if mode == "free"
           else rep["control"][who][f"{label}_{state}"]["fields"].get(field))
    return (row or {}).get("acceptance") or {}


# --- B5 (section 2) ----------------------------------------------------------------------

def merged_b5() -> dict:
    """e16_cap from E16's record; e16_b5 from E17's (the same natural rates: one training set)."""
    r16, r17 = load(E16D / "score_report.json"), load(E17D / f"score_report_{B5ARM}.json")
    out = {block: {BASE: r16[block][BASE], B5ARM: r17[block][B5ARM]} for block in BLOCKS}
    out["copy"] = {}
    for cell, node in r17["copy"].items():
        if node["real_stream"] != r16["copy"][cell]["real_stream"]:
            raise SystemExit(f"STOP: the natural rate of {cell} differs from E16's on the same counts")
        out["copy"][cell] = {"real": node["real_stream"], B5ARM: node[B5ARM],
                             BASE: {"generated": r16["copy"][cell][BASE]["generated"],
                                    "per_generation": r16["copy"][cell][BASE]["per_generation"]}}
        if node.get("real_domain_disjoint_stream"):
            out["copy"][cell]["real_domain_disjoint"] = node["real_domain_disjoint_stream"]
    out["stream_reuse"] = r17.get("stream_reuse")
    return out


def begun(u: str) -> bool:
    """The query has begun: a '?' before any '#'."""
    h, q = u.find("#"), u.find("?")
    return q >= 0 and (h < 0 or q < h)


def q1_rows(arm: str) -> list:
    """Every Q = 1 plan's URL in the arm's 50 files, with its plan reconstructed from the
    seeds (e12_score.state_positions, as S honouring is; never read from the output)."""
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    shipped = SuffixTable.load(ROOT / "data" / "bundle2" / "suffix_table.json")
    st = StateSuffixTables.load(ROOT / "data" / "state_suffix_tables.json")
    ps = load(ROOT / "data" / "e12" / "presence_slash_table.json")["table"]
    out = []
    for label in K.LABELS:
        ft = K.supplied_tables(ps, shipped.counts, label)
        for g, urls in enumerate(C13.files(arm, "free", label, None)):
            for u, k in zip(urls, S12.state_positions(ft, len(urls), K.seed_free(g))):
                if k[3] == "1":
                    out.append(("free", label, None, k, u))
        for state in K.STATES:
            ct = K.control_tables(ps, st, label, state)
            for g, urls in enumerate(C13.files(arm, "control", label, state)):
                for u, k in zip(urls, S12.state_positions(ct, len(urls), K.seed_ctrl(label, state, g))):
                    if k[3] == "1":
                        out.append(("control", label, state, k, u))
    return out


def q_honouring(arm: str) -> dict:
    """K17-B2's first clause, from the files: on every Q = 1 plan, both modes, the share of
    URLs with a query (the instruments' definition: FIELDS['QRY'] non-empty). Beside it,
    reported: the URLs whose query has begun ('?' before any '#'), the empty queries ('?'
    and nothing after), and the URLs without a begun query, split into those that ended
    and those cut off at max_len (their encoding with the end token exceeds max_len)."""
    import torch
    from uie.evaluation import FIELDS
    from uie.tokenizer import URLTokenizer
    blob = torch.load(E16D / "train" / BASE / "final.pt", map_location="cpu", weights_only=False)
    tok, max_len = URLTokenizer.from_state_dict(blob["tokenizer"]), blob["model_config"]["max_len"]
    groups: dict = {}
    for mode, label, state, k, u in q1_rows(arm):
        key = f"{mode} {label}" + (f" {state}" if state else "") + f" {k}"
        g = groups.setdefault(key, {"n": 0, "query": 0, "begun": 0, "empty": 0, "ended_without": 0,
                                    "cut_off_without": 0, "examples_without": []})
        g["n"] += 1
        has_q, has_b = bool(FIELDS["QRY"](u)), begun(u)
        g["query"] += has_q
        g["begun"] += has_b
        g["empty"] += has_b and not has_q
        if not has_b:
            cut = len(tok.encode(u, label)) > max_len
            g["cut_off_without" if cut else "ended_without"] += 1
            if len(g["examples_without"]) < 3:
                g["examples_without"].append(u)
    total = {f: sum(g[f] for g in groups.values()) for f in ("n", "query", "begun", "empty",
                                                             "ended_without", "cut_off_without")}
    as_registered = total["query"] == total["n"]
    as_amended = total["ended_without"] == 0
    return {"pass": as_amended if AMENDMENT_2 else as_registered,
            "judged_by": "amendment 2" if AMENDMENT_2 else "section 2 as registered",
            "pass_as_registered": as_registered, "pass_as_amended": as_amended,
            "total": total, "by_plan": groups}


def presence_b5(rep, targets, arm, ref) -> dict:
    """R-b (section 2): K12a within T, and no free PTH or QRY rate moves materially
    (amendment 1, either direction) -- except a QRY move toward its target, which is
    what the rule is for: such a move is allowed iff the arm's rate is nearer the target
    than the reference's."""
    a = k12a(rep, targets, arm)
    moves = {}
    for label in K.LABELS:
        for f, key in (("PTH", "path_rate_free"), ("QRY", "query_rate_free")):
            r, x = rep["emission"][ref][label][f], rep["emission"][arm][label][f]
            lo, hi = C13.worse_rate(r, x, True), C13.worse_rate(r, x, False)
            t = targets[label][key]
            toward = f == "QRY" and abs(x["rate"] - t) < abs(r["rate"] - t)
            material = lo["worse"] or hi["worse"]
            moves[f"{label} {f}"] = {"down": lo, "up": hi, "target": t, "toward_target": toward,
                                     "material": material, "worse": material and not toward}
    return {"pass": a["pass"] and not any(m["worse"] for m in moves.values()), "k12a": a, "moves": moves}


def part_b5(targets) -> dict:
    require("DECODE")
    rep = merged_b5()
    cells = C13.cell_contrasts(rep, B5ARM, BASE)
    obey = C13.obedience_rule(rep, B5ARM, BASE)
    qh = q_honouring(B5ARM)
    qh_ref = q_honouring(BASE)
    crit = {"K17-B1": k12b(rep, targets, B5ARM),
            "K17-B2": {"pass": qh["pass"] and obey["pass"], "q_honouring": qh,
                       "no_honouring_falls": obey["pass"]},
            "R-a'": r_a_prime(cells),
            "R-b": presence_b5(rep, targets, B5ARM, BASE),
            "R-c": obey,
            "R-d": C13.copying_rule(rep, B5ARM, BASE),
            "R-e": {"pass": True, "note": "VERIFY_DECODE and V4 passed (checked at start)"},
            "R-f": {"pass": True, "note": "single pass: B5 only rescales end-token logits"}}
    adopted = all(v["pass"] for v in crit.values())
    f = finish_line(rep, B5ARM, crit["K17-B1"])
    vd = load(E17D / "VERIFY_DECODE.json")
    return {"registration": E17.REGISTRATION + " (section 2)", "arm": B5ARM, "baseline": BASE,
            "criteria": crit, "adopted": adopted, "cells": cells,
            "report": {"slash_rows": {B5ARM: crit["K17-B1"]["rows"], BASE: k12b(rep, targets, BASE)["rows"]},
                       "q_honouring_e16_cap": qh_ref, "b5_trace": vd.get("b5_trace")},
            "E3": {**f, "E3": "MET" if f["met"] and adopted else
                   ("MET, NOT COUNTED (B5 not adopted)" if f["met"] else "NOT MET")},
            "difference": difference(rep, B5ARM, rep, BASE), "stream_reuse": rep["stream_reuse"]}


# --- A (section 3.3) ---------------------------------------------------------------------

def merged_a(ref: str) -> dict:
    """e17_a from E17's record; its reference from E17's (e16_b5) or E16's (e16_cap)."""
    ra = load(E17D / f"score_report_{ARM}.json")
    rr = load(E17D / f"score_report_{B5ARM}.json") if ref == B5ARM else load(E16D / "score_report.json")
    out = {block: {ref: rr[block][ref], ARM: ra[block][ARM]} for block in BLOCKS}
    out["copy"] = ra["copy"]
    out["copy_ref"] = {cell: node[ref] for cell, node in rr["copy"].items()}
    out["copy_ref_real"] = {cell: node["real_stream"] for cell, node in rr["copy"].items()}
    out["uptake"] = ra.get("uptake")
    return out


def k17_1(rep, ref) -> dict:
    cells = {}
    for name, (mode, label, state) in GUARDED.items():
        a, r = acc(rep, ARM, mode, label, state, "QRY"), acc(rep, ref, mode, label, state, "QRY")
        fall = r["x"] - a["x"]
        cells[name] = {"x_ref": r["x"], "x_arm": a["x"], "fall": fall, "v_ref": r["verdict"],
                       "v_arm": a["verdict"], "question": name in QUESTION,
                       "falls": fall > RETRAIN, "rises": -fall > RETRAIN}
    return {"cells": cells,
            "pass": any(c["falls"] for c in cells.values() if c["question"])
            and not any(c["rises"] for c in cells.values())}


def g2(rep, ref) -> dict:
    rows, fails = {}, []
    for cells in (lambda w: free_cells(rep, w), lambda w: control_cells(rep, w)):
        base, arm = cells(ref), cells(ARM)
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
    """Held-out NLL per field against e16_cap's training record (e16_b5 is e16_cap's weights)."""
    a = load(E17D / "train" / ARM / "train_result.json")["field_nll"]
    b = load(E16D / "train" / BASE / "train_result.json")["field_nll"]
    rows = {f"{c} {f}": a[c]["val"][f] - b[c]["val"][f] for c in a for f in a[c]["val"] if f in b[c]["val"]}
    over = {k: v for k, v in rows.items() if not k.endswith("QRY") and v > G3_MAX}
    return {"pass": not over, "delta_val_nll": rows, "over": over}


def g7(rep) -> dict:
    """K12d's memorisation rule (never OVER e17_a's own natural rate), and no copy rate
    moving further from its natural rate (amendment 1): the arm's rate is translated by
    (natural_ref - natural_arm) and amendment 1's rule applied (E15's g7)."""
    view = {cell: {"real": node["real_stream"], "real_domain_disjoint": node.get("real_domain_disjoint_stream"),
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
    return {"pass": not fails, "failures": fails, "memorisation": memo["memorisation"], "distance": dist}


def g8(rep, ref) -> dict:
    """K12d's whole-URL novelty rule, both modes, each model against its own training data."""
    nov = {}
    for mode in ("free", "control"):
        b, x = rep["novelty"][ref][mode], rep["novelty"][ARM][mode]
        nov[mode] = prop_worsening(b["novel"], b["n"], x["novel"], x["n"], False)
    return {"pass": not any(v["worsened"] for v in nov.values()), "novelty": nov}


def fit_vs_tell(rep, ref, e3_ref: dict, g3_rows: dict) -> dict:
    """Reported (section 4): for each cell the reference leaves short of E3, its change in
    x beside the held-out fit change: K17-2's query M3 (benign by stratum, phishing
    pooled) and, for every field, G3's per-field validation NLL change (E16 section 7)."""
    k2 = load(E17D / "k17_2.json")
    by = k2["benign"]["validation_by_stratum"]
    out = {}
    for c in e3_ref["cells_not_cleared"]:
        parts = c["cell"].split()
        mode, label = parts[0], (parts[1] if parts[0] == "free" else parts[1].split("_")[0])
        state = None if mode == "free" else parts[1].split("_")[1]
        field = parts[-1]
        a, r = acc(rep, ARM, mode, label, state, field), acc(rep, ref, mode, label, state, field)
        fit = (by.get(f"benign_{state}", {}).get("diff") if label == "benign" and state
               else k2[label]["validation"]["diff"])
        out[c["cell"]] = {"x_ref": r.get("x"), "x_arm": a.get("x"), "delta_x": (a.get("x") or 0) - (r.get("x") or 0),
                          "v_ref": r.get("verdict"), "v_arm": a.get("verdict"), "m3_diff": fit,
                          "field_nll_diff": g3_rows.get(f"{label} {field}")}
    return out


def part_a(targets) -> dict:
    require("TRAIN", "GENERATE")
    b5 = load(E17D / "compare_b5.json")
    ref = B5ARM if b5["adopted"] else BASE
    rep = merged_a(ref)
    k2, k3 = load(E17D / "k17_2.json"), load(E17D / "k17_3.json")
    g1 = regressions(free_cells(rep, ref), free_cells(rep, ARM)) + \
        regressions(control_cells(rep, ref), control_cells(rep, ARM))
    full = {label: acc(rep, ARM, "free", label, None, "FULL")["x"]
            - acc(rep, ref, "free", label, None, "FULL")["x"] for label in K.LABELS}
    vt = load(E17D / "VERIFY_TRAIN.json")
    clip = k12e(E17D / "train", ARM)
    crit = {"K17-1": k17_1(rep, ref),
            "K17-2": {"pass": bool(k2["pass"]), "benign": k2["benign"]["validation"],
                      "by_stratum": k2["benign"]["validation_by_stratum"],
                      "phishing_reported": k2["phishing"]["validation"]},
            "K17-3": {"pass": bool(k3["pass"]), **{label: k3[label][ARM] for label in K.LABELS}},
            "G1": {"pass": not g1, "regressions": g1},
            "G2": g2(rep, ref),
            "G3": g3(),
            "G4": {**k12a(rep, targets, ARM), "moves_reported": C13.presence_rule(rep, targets, ARM, ref)["moves"]},
            "G5": k12b(rep, targets, ARM),
            "G6": C13.obedience_rule(rep, ARM, ref),
            "G7": g7(rep),
            "G8": g8(rep, ref),
            "G9": {"delta_x": full, "pass": all(d <= RETRAIN for d in full.values())},
            "G10": {"pass": clip["pass"] and vt.get("verdict") == "PASS", "k12e": clip,
                    "verify_train": vt.get("verdict")}}
    adopted = all(v["pass"] for v in crit.values())
    f = finish_line(rep, ARM, crit["G5"])
    f_ref = finish_line(rep, ref, k12b(rep, targets, ref))
    return {"registration": E17.REGISTRATION + " (section 3.3)", "arm": ARM, "reference": ref,
            "decode": "raw + D1 + D1' + B5" if b5["adopted"] else "raw + D1 + D1'",
            "criteria": crit, "adopted": adopted,
            "E3": {**f, "E3": "MET" if f["met"] and adopted else
                   ("MET, NOT COUNTED (not adopted)" if f["met"] else "NOT MET")},
            "E3_reference": f_ref, "fit_vs_tell": fit_vs_tell(rep, ref, f_ref, crit["G3"]["delta_val_nll"]),
            "difference": difference(rep, ARM, rep, ref), "uptake": rep["uptake"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("b5", "a"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    targets = load(ROOT / "data" / "e11" / "targets.json")
    res = part_b5(targets) if args.part == "b5" else part_a(targets)
    dest = E17D / f"compare_{args.part}.json"
    dest.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    crit = res["criteria"]
    who = f"{res['arm']} vs {res.get('baseline') or res.get('reference')}"
    print(f"{who}: " + "  ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}" for k, v in crit.items())
          + f"  -> {'ADOPTED' if res['adopted'] else 'NOT ADOPTED'}")
    if args.part == "b5":
        qh = crit["K17-B2"]["q_honouring"]
        t = qh["total"]
        print(f"  Q = 1 plans: {t['n']:,} URLs; with a query {t['query']:,}; begun {t['begun']:,} "
              f"(empty {t['empty']}); without a begun query: ended {t['ended_without']}, "
              f"cut off at max_len {t['cut_off_without']}")
        print(f"  K17-B2 clause 1 judged by {qh['judged_by']}: as registered "
              f"{'PASS' if qh['pass_as_registered'] else 'FAIL'}, as amended "
              f"{'PASS' if qh['pass_as_amended'] else 'FAIL'}")
        for k, v in crit["K17-B1"]["rows"].items():
            print(f"  slash {k}: {v}")
        print(f"  NEXT: run {'E17_generate_b5' if res['adopted'] else 'E17_generate_d1p'}")
    else:
        for name, c in crit["K17-1"]["cells"].items():
            print(f"  K17-1 {name}: {c['x_ref']:.4f} ({c['v_ref']}) -> {c['x_arm']:.4f} ({c['v_arm']}), "
                  f"fall {c['fall']:+.4f}")
    print(f"E3: {res['E3']['E3']} ({len(res['E3']['cells_not_cleared'])} cells not cleared, "
          f"{len(res['E3']['slash_rows_missed'])} slash rows missed)")
    print("wrote", dest)


if __name__ == "__main__":
    main()
