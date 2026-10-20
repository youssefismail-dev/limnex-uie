"""
research/v2/v2_p1_compare.py

V2-P1 -- the registered criteria and the reading (research/v2/PREREG_V2_P1_MEMORY.md
section 5, with amendment 1's tiered copy rule, section 12). Laptop, after every earlier
step; nothing is fitted or generated here.

    python research/v2/v2_p1_compare.py      # -> data/v2/p1/compare.json; prints the readout

It reads only after every validity check has passed: VERIFY_TRAIN, K-used, VERIFY_GENERATE
and V4 (each from its own record), then V-D1b, the natural rates and V-floor (checked here,
first).

- K1 (section 5.1): D1b-R CLEARED, per class; VOID where e16_b5's D1b-R is not CONVICTED
  or the floor fails.
- K2 (section 5.2): D1b-R falls by at least 30% of e16_b5's excess, paired t <= -2.776.
- The guards (section 5.4):
  - gate: E17's own functions (e17_compare: g2, g7, g8; e12_compare: k12a, k12b, k12e;
    e8_compare: regressions; e13_compare: obedience_rule), on a report merging p1_mem's
    score with e16_b5's published one;
  - G3: from the training records;
  - G-D4: from the word fits;
  - G-COPY, W1, G-PAIR, C1, C2, G-DUP, G-VAL: from the metrics, with R2's tolerance
    arithmetic (v2_r2._tol).
- Amendment 1 (section 12): the same guards with the tiered copy rule. Paths and queries get
  +0.05 upward while diversity holds; once-seen strings stay strict, with new G-COPY rows
  for them; W1 gets up to +0.10 while G-PAIR passes and the blind-spot check (D1b-x) holds.
- Two readings, always both: the tiered one decides (section 12.4); the registered strict
  one is printed beside it. Then E3's finish line for the record.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e17", ROOT / "research" / "e16", ROOT / "research" / "e15",
          ROOT / "research" / "e14", ROOT / "research" / "e13", ROOT / "research" / "native",
          ROOT / "research" / "cr1", ROOT / "research" / "dc1", ROOT / "research" / "sc1",
          ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e17_compare as C17  # noqa: E402
from v2_arm import P1  # noqa: E402  (V2_ARM: p1 or p1b)
import v2_r1 as R1  # noqa: E402
import v2_r2 as R2  # noqa: E402
from e8_compare import RETRAIN, control_cells, free_cells, regressions  # noqa: E402
from e12_compare import difference, finish_line, k12a, k12b, k12e  # noqa: E402

SD = R2.SD
C13 = C17.C13
from v2_arm import DATA as P1D  # noqa: E402
E17D = ROOT / "data" / "e17"
GEN = P1D / "generate" / "gen"
ARM, REF = P1.ARM, P1.BASELINE
LABELS, G = SD.LABELS, SD.G
C17.ARM = ARM                                  # g2, g7 and g8 read it
C13.DIRS[ARM] = GEN
K2_SHARE = 0.30
VIEWS = {"R": "rest", "H": "host", "W": "whole"}
READINGS = (("D1b", "R"), ("D1b", "H"), ("D1b", "W"), ("D1", "R"), ("D1", "W"), ("D0", "W"), ("D4", "R"),
            ("D1x", "R"))
G3_MAX = 0.01
#: amendment 1 (section 12.3)
ALLOW_COPY = 0.05                              # paths and queries, upward, while diversity holds
ALLOW_W1 = 0.10                                # site words, while G-PAIR and the blind-spot check hold
DIVERSITY = ("C1", "C2", "G-PAIR", "G-DUP")


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def require() -> None:
    checks = {"VERIFY_TRAIN": load(P1D / "VERIFY_TRAIN.json").get("verdict") == "PASS",
              "K-used": bool(load(P1D / "k_used.json").get("pass")),
              "VERIFY_GENERATE": load(P1D / "VERIFY_GENERATE.json").get("verdict") == "PASS",
              "V4": bool(load(P1D / "score" / "v4.json").get("pass"))}
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise SystemExit(f"STOP: {bad} not passed; nothing is read")


def label_of(cell: str) -> str:
    return "phishing" if "phishing" in cell else "benign"


# --------------------------------------------------------------------------- #
# the word readings
# --------------------------------------------------------------------------- #

def v_d1b(fits: dict, meta: dict) -> dict:
    """V-D1b: e16_b5's D1b readings and the copier control reproduce R2 Part A's exactly
    (a class whose rest samples were cut is reported, not compared)."""
    r2 = load(R2.OUT / "A_D1b.json")
    off, skipped = [], []
    for view, part in VIEWS.items():
        for label in LABELS:
            if view == "R" and meta[label]["deviation"]:
                skipped.append(f"R {label}")
                continue
            for g in range(G):
                for task in ("floor", "base") + (("O4",) if view == "R" else ()):
                    a, b = fits[f"D1b|{view}|{label}|{g}|{task}"], r2[f"{part}|{label}|{g}|{task}"]
                    if a != b:
                        off.append((view, label, g, task, a, b))
    return {"pass": not off, "differences": off, "not_compared_cut": skipped}


def word_readings(fits: dict) -> dict:
    out = {}
    for det, view in READINGS:
        for label in LABELS:
            tasks = ("base", "p1") + (("O4",) if (det, view) == ("D1b", "R") else ())
            node = R1._verdicts(fits, f"{det}|{view}|{label}", tasks)
            node["p1 - base"] = SD.paired_t(node["p1"]["per_g"], node["base"]["per_g"])
            xb = node["base"]["mean"]
            node["closed"] = (xb - node["p1"]["mean"]) / xb if xb > 0 else None
            out[f"{det}|{view}|{label}"] = node
    return out


def k1_k2(words: dict, det: str = "D1b") -> dict:
    """K1 and K2 on the R view under `det` (D1b, or amendment 1's D1b-x). VOID follows
    D1b: it says whether the registered judge can see the tell in the class."""
    out = {}
    for label in LABELS:
        r, rb = words[f"{det}|R|{label}"], words[f"D1b|R|{label}"]
        void = rb["base"]["verdict"] != "CONVICTED" or rb["floor"]["flag"]
        d = r["p1 - base"]
        k2 = d["mean_diff"] <= -K2_SHARE * r["base"]["mean"] and d["t"] <= -SD.TCRIT
        out[label] = {"void": void, "K1": r["p1"]["verdict"] == "CLEARED", "K2": bool(k2),
                      "x_base": r["base"]["mean"], "x_p1": r["p1"]["mean"], "verdict_base": r["base"]["verdict"],
                      "verdict_p1": r["p1"]["verdict"], "closed": r["closed"], "paired": d,
                      "whole_url_cleared": words[f"D1b|W|{label}"]["p1"]["verdict"] == "CLEARED"}
    return out


def blind_spot_check(k: dict, kx: dict) -> dict:
    """Amendment 1: every K1 or K2 that passes under D1b also passes under D1b-x."""
    return {label: all(kx[label][c] for c in ("K1", "K2") if k[label][c]) for label in LABELS}


# --------------------------------------------------------------------------- #
# the guards
# --------------------------------------------------------------------------- #

def merged() -> dict:
    ra = load(P1D / f"score_report_{ARM}.json")
    rr = load(E17D / f"score_report_{REF}.json")
    rep = {block: {REF: rr[block][REF], ARM: ra[block][ARM]} for block in C17.BLOCKS}
    for cell, node in ra["copy"].items():
        if node["real_stream"] != rr["copy"][cell]["real_stream"]:
            raise SystemExit(f"STOP: the natural rate of {cell} differs from e16_b5's on the same counts")
    rep["copy"] = ra["copy"]
    rep["copy_ref"] = {cell: node[REF] for cell, node in rr["copy"].items()}
    rep["copy_ref_real"] = {cell: node["real_stream"] for cell, node in rr["copy"].items()}
    return rep


def g3() -> dict:
    a = load(P1D / "train" / ARM / "train_result.json")["field_nll"]
    b = load(ROOT / "data" / "e16" / "train" / "e16_cap" / "train_result.json")["field_nll"]
    rows = {f"{c} {f}": a[c]["val"][f] - b[c]["val"][f] for c in a for f in a[c]["val"] if f in b[c]["val"]}
    over = {k: v for k, v in rows.items() if not k.endswith("QRY") and v > G3_MAX}
    return {"pass": not over, "delta_val_nll": rows, "over": over}


def g7_tiered(rep: dict, allow: dict) -> dict:
    """G7 under amendment 1: C17.g7's two rules (K12d's memorisation rule on the copy rate
    and on the once-seen rate; amendment 1 of E13's distance rule), with `allow[label]`
    taken off p1_mem's copy rate in the upward direction only. The once-seen rule is
    unchanged."""
    import qc1_common as C
    memo, dist, fails = {}, {}, []
    for cell, node in rep["copy"].items():
        a = allow[label_of(cell)]
        g, n = node[ARM]["generated"], node[ARM]["per_generation"]
        real = node["real_stream"]
        if g["rate"] is None:
            continue
        rr, t = C.rate_reading(g["rate"] - a, real["rate"], n)
        sr, _ = C.rate_reading(g["singleton"], real["singleton"], n)
        memo[cell] = {"generated": g["rate"], "natural": real["rate"], "T": t, "allowance": a, "rate": rr,
                      "singleton_generated": g["singleton"], "singleton_natural": real["singleton"],
                      "singleton": sr}
        if rr == "OVER" or sr == "OVER":
            fails.append(f"memorisation {cell}")
        rn_arm, rn_ref = real["rate"], rep["copy_ref_real"][cell]["rate"]
        ref = rep["copy_ref"][cell]["generated"]
        shifted = g["rate"] + (rn_ref - rn_arm)
        worse_if_lower = ref["rate"] <= rn_ref
        if not worse_if_lower:                   # e16_b5 above natural: more copying is the risk
            shifted -= a
        dist[cell] = {**C13.worse_rate({"rate": ref["rate"], "n": ref["n"]}, {"rate": shifted, "n": g["n"]},
                                       worse_if_lower), "allowance": a if not worse_if_lower else 0.0}
        if dist[cell]["worse"]:
            fails.append(f"distance {cell}")
    return {"pass": not fails, "failures": fails, "memorisation": memo, "distance": dist}


def metric_rows(mtr: dict) -> dict:
    """Per class: every row the metric guards need, with the registered limit and, where
    amendment 1 changes it, the tiered limit beside it."""
    tol = R2._tol
    out = {}
    for label in LABELS:
        m = mtr[label]
        node = {"copy": {}, "once": {}, "field_dup": {}}
        for unit in ("URL", "RD", "PTH", "QRY"):
            if unit == "URL":
                v = {k: (m["gate"][k].get("memorized", 0.0), m["gate"][k]["n"]) for k in ("real", "base", "p1")}
            else:
                v = {k: (m[unit][k].get("copy", 0.0), m[unit][k]["n"]) for k in ("real", "base", "p1")}
            ref = max(v["real"][0], v["base"][0])
            node["copy"][unit] = {"real": v["real"][0], "base": v["base"][0], "p1": v["p1"][0],
                                  "limit": ref + tol(ref, v["p1"][1])}
        for unit in ("PTH", "QRY"):
            v = {k: (m[unit][k].get("copy_once", 0.0), m[unit][k]["n"]) for k in ("real", "base", "p1")}
            ref = max(v["real"][0], v["base"][0])
            node["once"][unit] = {"real": v["real"][0], "base": v["base"][0], "p1": v["p1"][0],
                                  "limit": ref + tol(ref, v["p1"][1])}
            node["once"][unit]["pass"] = node["once"][unit]["p1"] <= node["once"][unit]["limit"]
            d = {k: m[unit][k].get("duplicate", 0.0) for k in ("real", "base", "p1")}
            dref = max(d["real"], d["base"])
            node["field_dup"][unit] = {**d, "limit": dref + tol(dref, m[unit]["p1"]["n"]),
                                       "pass": d["p1"] <= dref + tol(dref, m[unit]["p1"]["n"])}
        node["W1"] = {}
        for view, bs in m["blind_spot"].items():
            ref = max(bs["C"]["b_only"], bs["base"]["b_only"])
            node["W1"][view] = {"real": bs["C"]["b_only"], "base": bs["base"]["b_only"], "p1": bs["p1"]["b_only"],
                                "limit": ref + tol(ref, bs["p1"]["n"]), "limit_tiered": ref + ALLOW_W1,
                                "a_only": {k: bs[k]["a_only"] for k in ("C", "base", "p1")}}
        pn = m["pair_novelty"]
        node["G-PAIR"] = {"real": pn["real"], "base": pn["base"], "p1": pn["p1"],
                          "floor": pn["real"] - tol(pn["real"], 20_000)}
        node["G-PAIR"]["pass"] = pn["p1"] >= node["G-PAIR"]["floor"]
        node["C1"] = {}
        for grp in ("host", "rest"):
            d = m[f"distinct_{grp}"]
            dp, db = abs(d["p1"] - d["real"]) / d["real"], abs(d["base"] - d["real"]) / d["real"]
            node["C1"][grp] = {"real": d["real"], "base": d["base"], "p1": d["p1"], "p1_off": dp, "base_off": db,
                               "pass": dp <= 0.05 or dp <= db + 0.02}
        node["C2"] = {}
        for grp in ("host", "rest"):
            (ur, nr), (ub, _), (up, npp) = (m[f"unseen_judge_{grp}"][k] for k in ("real", "base", "p1"))
            node["C2"][grp] = {"real": ur, "base": ub, "p1": up, "floor": ur - tol(ur, min(nr, npp))}
            node["C2"][grp]["pass"] = up >= node["C2"][grp]["floor"]
        dup = {k: m["gate"][k].get("duplicate_in_batch", 0.0) for k in ("real", "base", "p1")}
        dref = max(dup["real"], dup["base"])
        node["G-DUP"] = {**dup, "limit": dref + tol(dref, m["gate"]["p1"]["n"])}
        node["G-DUP"]["pass"] = dup["p1"] <= node["G-DUP"]["limit"]
        inv = {k: sum(v for r, v in m["gate"][k].items() if r not in ("n", "memorized", "duplicate_in_batch"))
               for k in ("real", "base", "p1")}
        node["G-VAL"] = {**inv, "limit": inv["base"] + tol(max(inv["base"], 1e-4), m["gate"]["p1"]["n"])}
        node["G-VAL"]["pass"] = inv["p1"] <= node["G-VAL"]["limit"]
        out[label] = node
    return out


def metric_guards(rows: dict, tiered: bool, blind_ok: dict) -> dict:
    """The seven metric guards, per class, under the registered rule or amendment 1's."""
    per = {}
    for label in LABELS:
        r = rows[label]
        diverse = (r["G-PAIR"]["pass"] and all(x["pass"] for x in r["C1"].values())
                   and all(x["pass"] for x in r["C2"].values()) and r["G-DUP"]["pass"])
        copy = {}
        for unit, x in r["copy"].items():
            limit = x["limit"]
            if tiered and unit in ("PTH", "QRY") and diverse and r["field_dup"][unit]["pass"]:
                limit += ALLOW_COPY
            copy[unit] = {**x, "limit_used": limit, "pass": x["p1"] <= limit}
        if tiered:
            for unit, x in r["once"].items():
                copy[f"{unit} once-seen"] = {**x, "limit_used": x["limit"]}
        w1 = {}
        for view, x in r["W1"].items():
            limit = x["limit_tiered"] if tiered and r["G-PAIR"]["pass"] and blind_ok[label] else x["limit"]
            w1[view] = {**x, "limit_used": limit, "pass": x["p1"] <= max(limit, x["limit"])}
        per[label] = {"G-COPY": {"pass": all(v["pass"] for v in copy.values()), "rows": copy,
                                 "diversity_condition": diverse},
                      "W1": {"pass": all(v["pass"] for v in w1.values()), "rows": w1},
                      "G-PAIR": r["G-PAIR"],
                      "C1": {"pass": all(x["pass"] for x in r["C1"].values()), "rows": r["C1"]},
                      "C2": {"pass": all(x["pass"] for x in r["C2"].values()), "rows": r["C2"]},
                      "G-DUP": r["G-DUP"], "G-VAL": r["G-VAL"]}
    return {name: {"pass": all(per[label][name]["pass"] for label in LABELS),
                   **{label: per[label][name] for label in LABELS}}
            for name in ("G-COPY", "W1", "G-PAIR", "C1", "C2", "G-DUP", "G-VAL")}


def gate_guards(rep: dict, targets: dict, words: dict) -> dict:
    g1 = regressions(free_cells(rep, REF), free_cells(rep, ARM)) + \
        regressions(control_cells(rep, REF), control_cells(rep, ARM))
    full = {label: C17.acc(rep, ARM, "free", label, None, "FULL")["x"]
            - C17.acc(rep, REF, "free", label, None, "FULL")["x"] for label in LABELS}
    d4 = {label: words[f"D4|R|{label}"]["p1"]["mean"] - words[f"D4|R|{label}"]["base"]["mean"] for label in LABELS}
    clip = k12e(P1D / "train", ARM)
    return {"G1": {"pass": not g1, "regressions": g1},
            "G2": C17.g2(rep, REF),
            "G3": g3(),
            "G4": k12a(rep, targets, ARM),
            "G5": k12b(rep, targets, ARM),
            "G6": C13.obedience_rule(rep, ARM, REF),
            "G7": C17.g7(rep),
            "G8": C17.g8(rep, REF),
            "G9": {"pass": all(d <= RETRAIN for d in full.values()), "delta_x": full},
            "G-D4": {"pass": all(d <= RETRAIN for d in d4.values()), "delta_x": d4},
            "G10": {"pass": clip["pass"], "k12e": clip}}


def all_guards(rep: dict, targets: dict, words: dict, mtr: dict, blind_ok: dict) -> tuple:
    """(strict, tiered): section 5.4 as registered, and with amendment 1's rule."""
    gate = gate_guards(rep, targets, words)
    rows = metric_rows(mtr)
    strict = {**gate, **metric_guards(rows, False, blind_ok)}
    tiered_metrics = metric_guards(rows, True, blind_ok)
    allow = {label: ALLOW_COPY if (tiered_metrics["G-COPY"][label]["diversity_condition"]
                                   and rows[label]["field_dup"]["QRY"]["pass"]) else 0.0
             for label in LABELS}
    tiered = {**gate, "G7": g7_tiered(rep, allow), **tiered_metrics}
    return strict, tiered


# --------------------------------------------------------------------------- #
# the reading
# --------------------------------------------------------------------------- #

def reading(k: dict, gd: dict) -> dict:
    live = [label for label in LABELS if not k[label]["void"]]
    failed = [name for name, v in gd.items() if not v["pass"]]
    if not live:
        return {"reading": "UNREADABLE", "classes_read": live, "failed_guards": failed}
    if all(k[label]["K1"] for label in live):
        base = "SUCCESS"
    elif any(k[label]["K1"] or k[label]["K2"] for label in live):
        base = "PARTIAL"
    else:
        base = "NO EFFECT"
    text = base + (f" WITH COST ({', '.join(failed)})" if failed else "")
    whole = base == "SUCCESS" and all(k[label]["whole_url_cleared"] for label in live)
    return {"reading": text, "base": base, "classes_read": live, "failed_guards": failed,
            "whole_url_too": whole}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    require()
    fits = load(P1D / "words" / "fits.json")
    meta = load(P1D / "words" / "samples_meta.json")
    mtr = load(P1D / "words" / "metrics.json")
    vd = v_d1b(fits, meta)
    if not vd["pass"]:
        raise SystemExit(f"STOP: V-D1b failed -- the word pipeline drifted from R2's: {vd['differences'][:5]}")
    words = word_readings(fits)
    floors = {key: v["floor"] for key, v in words.items()}
    k, kx = k1_k2(words, "D1b"), k1_k2(words, "D1x")
    blind_ok = blind_spot_check(k, kx)
    targets = load(ROOT / "data" / "e11" / "targets.json")
    rep = merged()
    strict, tiered = all_guards(rep, targets, words, mtr, blind_ok)
    rd_strict, rd_tiered = reading(k, strict), reading(k, tiered)
    label_text = rd_tiered["reading"] + ("" if rd_strict["reading"] == rd_tiered["reading"]
                                         else f" -- strict: {rd_strict['reading']}")
    f = finish_line(rep, ARM, strict["G5"])
    res = {"registration": P1.REGISTRATION, "arm": ARM, "baseline": REF,
           "validity": {"V-D1b": vd, "floors": floors},
           "K1_K2": k, "K1_K2_under_D1b_x": kx, "blind_spot_check": blind_ok, "words": words,
           "guards_strict": strict, "guards_tiered": tiered,
           "reading": {"binding": "tiered (amendment 1, section 12.4)", "label": label_text,
                       "tiered": rd_tiered, "strict": rd_strict},
           "E3_for_the_record": {**f, "E3": "MET (V2 does not reopen E3)" if f["met"] else "NOT MET"},
           "difference": difference(rep, ARM, rep, REF), "samples_meta": meta}
    (P1D / "compare.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("validity: VERIFY_TRAIN, K-used, VERIFY_GENERATE, V4, V-D1b PASS; floors flagged:",
          [key for key, v in floors.items() if v["flag"]] or "none")
    for det, view in READINGS:
        for label in LABELS:
            w = words[f"{det}|{view}|{label}"]
            c = f" closed {w['closed']:+.0%}" if w["closed"] is not None else ""
            o4 = f"; copier control {w['O4']['mean']:+.4f} {w['O4']['verdict']}" if "O4" in w else ""
            print(f"  {det}-{view} {label:<9} e16_b5 {w['base']['mean']:+.4f} {w['base']['verdict']:<10} "
                  f"{ARM} {w['p1']['mean']:+.4f} {w['p1']['verdict']:<10}{c}{o4}")
    for label in LABELS:
        x = k[label]
        print(f"{label}: K1 {'VOID' if x['void'] else ('PASS' if x['K1'] else 'FAIL')}; "
              f"K2 {'PASS' if x['K2'] else 'FAIL'} (paired diff {x['paired']['mean_diff']:+.4f}, "
              f"t {x['paired']['t']:+.2f}); blind-spot check {'holds' if blind_ok[label] else 'FAILS'}")
    print("guards, strict: " + "  ".join(f"{n} {'PASS' if v['pass'] else 'FAIL'}" for n, v in strict.items()))
    print("guards, tiered: " + "  ".join(f"{n} {'PASS' if v['pass'] else 'FAIL'}" for n, v in tiered.items()))
    print(f"READING: {label_text}" + ("  (whole URL too)" if rd_tiered.get("whole_url_too") else ""))
    print(f"E3 for the record: {res['E3_for_the_record']['E3']} ({len(f['cells_not_cleared'])} cells not cleared, "
          f"{len(f['slash_rows_missed'])} slash rows missed)")
    print("wrote", P1D / "compare.json")


if __name__ == "__main__":
    main()
