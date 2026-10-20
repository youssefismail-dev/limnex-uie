"""
research/v2/v2_p2_compare.py

V2-P2 -- the registered criteria and the reading (research/v2/PREREG_V2_P2.md sections 8-10).
Laptop, after every earlier step; nothing is fitted or generated here.

    python research/v2/v2_p2_compare.py      # -> data/v2/p2/compare.json; prints the readout

It reads only after every validity check has passed: V-gen-equiv, each arm's VERIFY_TRAIN,
VERIFY_GENERATE and V4 (each from its own record), then V-D1b and V-floor (checked here).

- **K1 (section 8.1):** `p2_a`'s D1b-R CLEARED; `p2_b`'s D1b-W CLEARED (host and rest recorded
  beside it). A class's view is VOID if `e16_b5` is not CONVICTED on it, or its floor fails.
- **K2 (section 8.2):** on one of the arm's own views (A: R; B: H or W), the mean cut is at
  least 30% of `e16_b5`'s excess and the paired t is <= -2.776. The readout names the view.
- **The blind-spot condition (section 8.3):** a K1 or K2 that passes under D1b must pass under
  D1b-x on the same view, or it is read as not passed ("blind spot").
- **Reported, never deciding:** B's host gain (B against A on D1b-H), A's specificity (A
  against `e16_b5` on D1b-H), the other views, D1, D0, D4, the bands, the copier control,
  closure shares, fit, the piece rule's removed mass, tokens per URL, the page-row name gap,
  examples, and E3's finish line per arm.
- **The guards (section 9), each arm against `e16_b5`:** the gate (E17's and E12's functions on
  a report merging the arm's score with `e16_b5`'s published one), G3 from the training
  records, G-D4 and G-D4-H from the word fits, the metric guards with the tiered copy rule
  (v2_p1_compare's rules, unchanged), and the host guards of section 9.6 (deciding for
  `p2_b`, reported for `p2_a`). Both readings are printed: the tiered one decides.
"""
from __future__ import annotations

import json
import os
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
import v2_p2_common as C  # noqa: E402
import v2_r1 as R1  # noqa: E402
import v2_r2 as R2  # noqa: E402
from e8_compare import RETRAIN, control_cells, free_cells, regressions  # noqa: E402
from e12_compare import difference, finish_line, k12a, k12b, k12e  # noqa: E402

SD = R2.SD
C13 = C17.C13
DATA = Path(os.environ["V2_DATA"]) if os.environ.get("V2_DATA") else ROOT / "data" / "v2"
P2D = DATA / "p2"
E17D = ROOT / "data" / "e17"
ARMS = ("p2_a", "p2_b")
REF = C.BASELINE
LABELS, G = SD.LABELS, SD.G
VIEWS = {"R": "rest", "H": "host", "W": "whole"}
READINGS = (("D1b", "R"), ("D1b", "H"), ("D1b", "W"), ("D1", "R"), ("D1", "W"), ("D0", "W"), ("D4", "R"),
            ("D4", "H"), ("D1x", "R"), ("D1x", "H"), ("D1x", "W"))
PASS_VIEW = {"p2_a": "R", "p2_b": "W"}                  # K1 (section 8.1)
K2_VIEWS = {"p2_a": ("R",), "p2_b": ("H", "W")}         # K2 (section 8.2)
K2_SHARE = 0.30
G3_MAX = 0.01
ALLOW_COPY = 0.05                              # paths and queries, upward, while diversity holds
ALLOW_W1 = 0.10                                # site words, while G-PAIR and the blind-spot check hold
ALLOW_LBL = 0.05                               # H-LBL, while the host diversity condition holds
STRICT_HOST = ("H-RD", "H-HOST", "H-LIVE", "H-LBL1", "H-SUB1")


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def arm_dir(arm: str) -> Path:
    return DATA / arm


def require() -> None:
    checks = {"V-gen-equiv": load(ROOT / "data" / "v2" / "p2" / "VERIFY_GEN_EQUIV.json").get("verdict") == "PASS"}
    for arm in ARMS:
        checks[f"{arm} VERIFY_TRAIN"] = load(arm_dir(arm) / "VERIFY_TRAIN.json").get("verdict") == "PASS"
        checks[f"{arm} VERIFY_GENERATE"] = load(arm_dir(arm) / "VERIFY_GENERATE.json").get("verdict") == "PASS"
        checks[f"{arm} V4"] = bool(load(arm_dir(arm) / "score" / "v4.json").get("pass"))
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise SystemExit(f"STOP: {bad} not passed; nothing is read")


def label_of(cell: str) -> str:
    return "phishing" if "phishing" in cell else "benign"


# --------------------------------------------------------------------------- #
# the word readings and K1, K2
# --------------------------------------------------------------------------- #

def v_d1b(fits: dict, meta: dict) -> dict:
    """V-D1b (section 7): e16_b5's D1b readings and the copier control reproduce R2 Part A's
    exactly, on every view (a class whose rest samples were cut is reported, not compared)."""
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
            tasks = ("base",) + ARMS + (("O4",) if (det, view) == ("D1b", "R") else ())
            node = R1._verdicts(fits, f"{det}|{view}|{label}", tasks)
            xb = node["base"]["mean"]
            for arm in ARMS:
                node[f"{arm} - base"] = SD.paired_t(node[arm]["per_g"], node["base"]["per_g"])
                node[f"{arm} closed"] = (xb - node[arm]["mean"]) / xb if xb > 0 else None
                # the closure share (section 10): the arm's cut over e16_b5's distance to the clear line
                line = node["base"]["clear_line"]
                node[f"{arm} closure"] = (xb - node[arm]["mean"]) / (xb - line) if xb > line else None
            node["p2_b - p2_a"] = SD.paired_t(node["p2_b"]["per_g"], node["p2_a"]["per_g"])
            out[f"{det}|{view}|{label}"] = node
    return out


def k_view(words: dict, det: str, view: str, label: str, arm: str) -> dict:
    """K1's and K2's parts on one view: CLEARED, and the cut (at least 30% of e16_b5's excess,
    paired t <= -2.776). VOID follows D1b on the view: can the registered judge see the tell?"""
    r, rb = words[f"{det}|{view}|{label}"], words[f"D1b|{view}|{label}"]
    d = r[f"{arm} - base"]
    # V-floor (section 12): a reading whose floor fails is VOID. For D1b that voids the view
    # (with e16_b5 not CONVICTED); a VOID D1b-x reading cannot confirm anything (section 8.3)
    own_floor_void = bool(r["floor"]["flag"])
    confirmable = not (det == "D1x" and own_floor_void)
    return {"void": rb["base"]["verdict"] != "CONVICTED" or rb["floor"]["flag"],
            "floor_void": own_floor_void,
            "cleared": confirmable and r[arm]["verdict"] == "CLEARED",
            "cut": confirmable and bool(d["mean_diff"] <= -K2_SHARE * r["base"]["mean"] and d["t"] <= -SD.TCRIT),
            "x_base": r["base"]["mean"], "x_arm": r[arm]["mean"], "verdict_base": r["base"]["verdict"],
            "verdict_arm": r[arm]["verdict"], "paired": d, "closure": r[f"{arm} closure"]}


def k1_k2(words: dict, arm: str) -> dict:
    """Per class: K1 on the arm's pass view, K2 on its own views, each also under D1b-x; a pass
    under D1b that fails under D1b-x is read as not passed (section 8.3)."""
    out = {}
    for label in LABELS:
        pv = PASS_VIEW[arm]
        k1, k1x = k_view(words, "D1b", pv, label, arm), k_view(words, "D1x", pv, label, arm)
        k2_views = {}
        for v in K2_VIEWS[arm]:
            a, ax = k_view(words, "D1b", v, label, arm), k_view(words, "D1x", v, label, arm)
            k2_views[v] = {"D1b": a, "D1b-x": ax, "void": a["void"], "holds": a["cut"] and not a["void"],
                           "holds_under_x": ax["cut"]}
        k1_pass = k1["cleared"] and not k1["void"]
        k2_pass_views = [v for v, x in k2_views.items() if x["holds"]]
        k2_x_views = [v for v in k2_pass_views if k2_views[v]["holds_under_x"]]
        blind = (k1_pass and not k1x["cleared"]) or (bool(k2_pass_views) and not k2_x_views)
        out[label] = {
            "void": k1["void"], "pass_view": pv, "K1_under_D1b": k1_pass, "K1_under_D1b_x": k1x["cleared"],
            "K1": k1_pass and k1x["cleared"],
            "K2_views_under_D1b": k2_pass_views, "K2_views": k2_x_views, "K2": bool(k2_x_views),
            "blind_spot": blind, "k1_detail": k1, "k1_detail_x": k1x, "k2_detail": k2_views,
            "recorded": {f"D1b-{v} CLEARED": words[f"D1b|{v}|{label}"][arm]["verdict"] == "CLEARED" for v in VIEWS}}
    return out


def host_gain_and_specificity(words: dict) -> dict:
    """Section 8.2, reported: B against A on D1b-H, recorded when the difference is <= -0.0141
    with t <= -2.776; A against e16_b5 on D1b-H, within 0.0141 or not significant."""
    out = {}
    for label in LABELS:
        w = words[f"D1b|H|{label}"]
        ba, ae = w["p2_b - p2_a"], w["p2_a - base"]
        out[label] = {"B's host gain": {**ba, "recorded": bool(ba["mean_diff"] <= -RETRAIN and ba["t"] <= -SD.TCRIT)},
                      "A's specificity": {**ae, "holds": bool(abs(ae["mean_diff"]) <= RETRAIN
                                                              or abs(ae["t"]) <= SD.TCRIT)}}
    return out


# --------------------------------------------------------------------------- #
# the guards
# --------------------------------------------------------------------------- #

def merged(arm: str) -> dict:
    ra = load(arm_dir(arm) / f"score_report_{arm}.json")
    rr = load(E17D / f"score_report_{REF}.json")
    rep = {block: {REF: rr[block][REF], arm: ra[block][arm]} for block in C17.BLOCKS}
    for cell, node in ra["copy"].items():
        if node["real_stream"] != rr["copy"][cell]["real_stream"]:
            raise SystemExit(f"STOP: the natural rate of {cell} differs from e16_b5's on the same counts")
    rep["copy"] = ra["copy"]
    rep["copy_ref"] = {cell: node[REF] for cell, node in rr["copy"].items()}
    rep["copy_ref_real"] = {cell: node["real_stream"] for cell, node in rr["copy"].items()}
    return rep


def g3(arm: str) -> dict:
    """G3 (section 9.2): no non-query field's validation NLL, in nats per character, up by more
    than 0.01 against e16_cap, from the training records."""
    a = load(arm_dir(arm) / "train" / arm / "train_result.json")["field_nll"]
    b = load(ROOT / "data" / "e16" / "train" / "e16_cap" / "train_result.json")["field_nll"]
    rows = {f"{c} {f}": a[c]["val"][f] - b[c]["val"][f] for c in a for f in a[c]["val"] if f in b[c]["val"]}
    over = {k: v for k, v in rows.items() if not k.endswith("QRY") and v > G3_MAX}
    return {"pass": not over, "delta_val_nll_per_char": rows, "over": over}


def g7_tiered(rep: dict, arm: str, allow: dict) -> dict:
    """G7 under the tiered rule (v2_p1_compare.g7_tiered, for the arm)."""
    import qc1_common as QC
    memo, dist, fails = {}, {}, []
    for cell, node in rep["copy"].items():
        a = allow[label_of(cell)]
        g, n = node[arm]["generated"], node[arm]["per_generation"]
        real = node["real_stream"]
        if g["rate"] is None:
            continue
        rr, t = QC.rate_reading(g["rate"] - a, real["rate"], n)
        sr, _ = QC.rate_reading(g["singleton"], real["singleton"], n)
        memo[cell] = {"generated": g["rate"], "natural": real["rate"], "T": t, "allowance": a, "rate": rr,
                      "singleton_generated": g["singleton"], "singleton_natural": real["singleton"],
                      "singleton": sr}
        if rr == "OVER" or sr == "OVER":
            fails.append(f"memorisation {cell}")
        rn_arm, rn_ref = real["rate"], rep["copy_ref_real"][cell]["rate"]
        ref = rep["copy_ref"][cell]["generated"]
        shifted = g["rate"] + (rn_ref - rn_arm)
        worse_if_lower = ref["rate"] <= rn_ref
        if not worse_if_lower:
            shifted -= a
        dist[cell] = {**C13.worse_rate({"rate": ref["rate"], "n": ref["n"]}, {"rate": shifted, "n": g["n"]},
                                       worse_if_lower), "allowance": a if not worse_if_lower else 0.0}
        if dist[cell]["worse"]:
            fails.append(f"distance {cell}")
    return {"pass": not fails, "failures": fails, "memorisation": memo, "distance": dist}


def metric_rows(mtr: dict, arm: str) -> dict:
    """v2_p1_compare.metric_rows for the arm (p1 -> the arm), plus section 9.6's host rows."""
    tol = R2._tol
    out = {}
    for label in LABELS:
        m = mtr[label]
        node = {"copy": {}, "once": {}, "field_dup": {}}
        for unit in ("URL", "RD", "PTH", "QRY"):
            if unit == "URL":
                v = {k: (m["gate"][k].get("memorized", 0.0), m["gate"][k]["n"]) for k in ("real", "base", arm)}
            else:
                v = {k: (m[unit][k].get("copy", 0.0), m[unit][k]["n"]) for k in ("real", "base", arm)}
            ref = max(v["real"][0], v["base"][0])
            node["copy"][unit] = {"real": v["real"][0], "base": v["base"][0], "arm": v[arm][0],
                                  "limit": ref + tol(ref, v[arm][1])}
        for unit in ("PTH", "QRY"):
            v = {k: (m[unit][k].get("copy_once", 0.0), m[unit][k]["n"]) for k in ("real", "base", arm)}
            ref = max(v["real"][0], v["base"][0])
            node["once"][unit] = {"real": v["real"][0], "base": v["base"][0], "arm": v[arm][0],
                                  "limit": ref + tol(ref, v[arm][1])}
            node["once"][unit]["pass"] = node["once"][unit]["arm"] <= node["once"][unit]["limit"]
            d = {k: m[unit][k].get("duplicate", 0.0) for k in ("real", "base", arm)}
            dref = max(d["real"], d["base"])
            node["field_dup"][unit] = {"real": d["real"], "base": d["base"], "arm": d[arm],
                                       "limit": dref + tol(dref, m[unit][arm]["n"]),
                                       "pass": d[arm] <= dref + tol(dref, m[unit][arm]["n"])}
        node["W1"] = {}
        for view, bs in m["blind_spot"].items():
            ref = max(bs["C"]["b_only"], bs["base"]["b_only"])
            node["W1"][view] = {"real": bs["C"]["b_only"], "base": bs["base"]["b_only"], "arm": bs[arm]["b_only"],
                                "limit": ref + tol(ref, bs[arm]["n"]), "limit_tiered": ref + ALLOW_W1,
                                "a_only": {k: bs[k]["a_only"] for k in ("C", "base", arm)}}
        pn = m["pair_novelty"]
        node["G-PAIR"] = {"real": pn["real"], "base": pn["base"], "arm": pn[arm],
                          "floor": pn["real"] - tol(pn["real"], 20_000)}
        node["G-PAIR"]["pass"] = pn[arm] >= node["G-PAIR"]["floor"]
        node["C1"] = {}
        for grp in ("host", "rest"):
            d = m[f"distinct_{grp}"]
            da, db = abs(d[arm] - d["real"]) / d["real"], abs(d["base"] - d["real"]) / d["real"]
            node["C1"][grp] = {"real": d["real"], "base": d["base"], "arm": d[arm], "arm_off": da, "base_off": db,
                               "pass": da <= 0.05 or da <= db + 0.02}
        node["C2"] = {}
        for grp in ("host", "rest"):
            (ur, nr), (ub, _), (ua, na) = (m[f"unseen_judge_{grp}"][k] for k in ("real", "base", arm))
            node["C2"][grp] = {"real": ur, "base": ub, "arm": ua, "floor": ur - tol(ur, min(nr, na))}
            node["C2"][grp]["pass"] = ua >= node["C2"][grp]["floor"]
        dup = {k: m["gate"][k].get("duplicate_in_batch", 0.0) for k in ("real", "base", arm)}
        dref = max(dup["real"], dup["base"])
        node["G-DUP"] = {"real": dup["real"], "base": dup["base"], "arm": dup[arm],
                         "limit": dref + tol(dref, m["gate"][arm]["n"])}
        node["G-DUP"]["pass"] = dup[arm] <= node["G-DUP"]["limit"]
        inv = {k: sum(v for r, v in m["gate"][k].items() if r not in ("n", "memorized", "duplicate_in_batch"))
               for k in ("real", "base", arm)}
        node["G-VAL"] = {"real": inv["real"], "base": inv["base"], "arm": inv[arm],
                         "limit": inv["base"] + tol(max(inv["base"], 1e-4), m["gate"][arm]["n"])}
        node["G-VAL"]["pass"] = inv[arm] <= node["G-VAL"]["limit"]
        # section 9.6: the host rows
        h = m["host"]
        n = h[arm]["n"]
        node["host"] = {}
        for code in STRICT_HOST + ("H-LBL", "H-DUP"):
            ref = max(h["real"][code], h["base"][code])
            node["host"][code] = {"real": h["real"][code], "base": h["base"][code], "arm": h[arm][code],
                                  "limit": ref + tol(ref, n)}
        ty = {k: h[k]["H-TYPO"] for k in ("real", "base", arm)}
        tt = tol(ty["real"], h[arm]["host_words"])
        node["host"]["H-TYPO"] = {**{k: ty[k] for k in ("real", "base")}, "arm": ty[arm], "T": tt,
                                  "pass": abs(ty[arm] - ty["real"]) <= tt
                                  or abs(ty[arm] - ty["real"]) <= abs(ty["base"] - ty["real"])}
        hv = {k: h[k]["H-VAL"] for k in ("real", "base", arm)}
        node["host"]["H-VAL"] = {"real": hv["real"], "base": hv["base"], "arm": hv[arm],
                                 "limit": hv["base"] + tol(max(hv["base"], 1e-4), n)}
        out[label] = node
    return out


def metric_guards(rows: dict, tiered: bool, blind_ok: dict) -> dict:
    """v2_p1_compare.metric_guards for the arm, plus the host guards (section 9.6)."""
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
            copy[unit] = {**x, "limit_used": limit, "pass": x["arm"] <= limit}
        if tiered:
            for unit, x in r["once"].items():
                copy[f"{unit} once-seen"] = {**x, "limit_used": x["limit"]}
        w1 = {}
        for view, x in r["W1"].items():
            limit = x["limit_tiered"] if tiered and r["G-PAIR"]["pass"] and blind_ok[label] else x["limit"]
            w1[view] = {**x, "limit_used": limit, "pass": x["arm"] <= max(limit, x["limit"])}
        hd = r["host"]["H-DUP"]
        hd_pass = hd["arm"] <= hd["limit"]
        host_diverse = r["C1"]["host"]["pass"] and r["C2"]["host"]["pass"] and hd_pass and r["G-DUP"]["pass"]
        hosts = {}
        for code in STRICT_HOST + ("H-DUP", "H-VAL"):
            x = r["host"][code]
            hosts[code] = {**x, "limit_used": x["limit"], "pass": x["arm"] <= x["limit"]}
        x = r["host"]["H-LBL"]
        limit = x["limit"] + (ALLOW_LBL if tiered and host_diverse else 0.0)
        hosts["H-LBL"] = {**x, "limit_used": limit, "pass": x["arm"] <= limit, "host_diversity": host_diverse}
        hosts["H-TYPO"] = r["host"]["H-TYPO"]
        per[label] = {"G-COPY": {"pass": all(v["pass"] for v in copy.values()), "rows": copy,
                                 "diversity_condition": diverse},
                      "W1": {"pass": all(v["pass"] for v in w1.values()), "rows": w1},
                      "G-PAIR": r["G-PAIR"],
                      "C1": {"pass": all(x["pass"] for x in r["C1"].values()), "rows": r["C1"]},
                      "C2": {"pass": all(x["pass"] for x in r["C2"].values()), "rows": r["C2"]},
                      "G-DUP": r["G-DUP"], "G-VAL": r["G-VAL"],
                      **{code: hosts[code] for code in hosts}}
    names = ("G-COPY", "W1", "G-PAIR", "C1", "C2", "G-DUP", "G-VAL") + STRICT_HOST + ("H-LBL", "H-TYPO", "H-DUP", "H-VAL")
    return {name: {"pass": all(per[label][name]["pass"] for label in LABELS),
                   **{label: per[label][name] for label in LABELS}}
            for name in names}


def gate_guards(rep: dict, targets: dict, words: dict, arm: str) -> dict:
    C17.ARM = arm                              # g2, g7 and g8 read it
    C13.DIRS[arm] = arm_dir(arm) / "generate" / "gen"
    g1 = regressions(free_cells(rep, REF), free_cells(rep, arm)) + \
        regressions(control_cells(rep, REF), control_cells(rep, arm))
    full = {label: C17.acc(rep, arm, "free", label, None, "FULL")["x"]
            - C17.acc(rep, REF, "free", label, None, "FULL")["x"] for label in LABELS}
    d4 = {v: {label: words[f"D4|{v}|{label}"][arm]["mean"] - words[f"D4|{v}|{label}"]["base"]["mean"]
              for label in LABELS} for v in ("R", "H")}
    # V-floor: a class whose D4 floor fails is a VOID reading, listed, neither pass nor fail
    d4_void = {v: [label for label in LABELS if words[f"D4|{v}|{label}"]["floor"]["flag"]] for v in ("R", "H")}
    clip = k12e(arm_dir(arm) / "train", arm)
    return {"G1": {"pass": not g1, "regressions": g1},
            "G2": C17.g2(rep, REF),
            "G3": g3(arm),
            "G4": k12a(rep, targets, arm),
            "G5": k12b(rep, targets, arm),
            "G6": C13.obedience_rule(rep, arm, REF),
            "G7": C17.g7(rep),
            "G8": C17.g8(rep, REF),
            "G9": {"pass": all(d <= RETRAIN for d in full.values()), "delta_x": full},
            "G-D4": {"pass": all(d <= RETRAIN for k, d in d4["R"].items() if k not in d4_void["R"]),
                     "delta_x": d4["R"], "void": d4_void["R"]},
            "G-D4-H": {"pass": all(d <= RETRAIN for k, d in d4["H"].items() if k not in d4_void["H"]),
                       "delta_x": d4["H"], "void": d4_void["H"]},
            "G10": {"pass": clip["pass"], "k12e": clip}}


def all_guards(rep: dict, targets: dict, words: dict, mtr: dict, blind_ok: dict, arm: str) -> tuple:
    """(strict, tiered) for the arm; the host guards and G-D4-H decide for p2_b only."""
    gate = gate_guards(rep, targets, words, arm)
    rows = metric_rows(mtr, arm)
    strict = {**gate, **metric_guards(rows, False, blind_ok)}
    tiered_metrics = metric_guards(rows, True, blind_ok)
    allow = {label: ALLOW_COPY if (tiered_metrics["G-COPY"][label]["diversity_condition"]
                                   and rows[label]["field_dup"]["QRY"]["pass"]) else 0.0
             for label in LABELS}
    C17.ARM = arm
    tiered = {**gate, "G7": g7_tiered(rep, arm, allow), **tiered_metrics}
    reported_only = () if arm == "p2_b" else ("G-D4-H",) + STRICT_HOST + ("H-LBL", "H-TYPO", "H-DUP", "H-VAL")
    return strict, tiered, reported_only


# --------------------------------------------------------------------------- #
# the reading
# --------------------------------------------------------------------------- #

def reading(k: dict, gd: dict, reported_only: tuple) -> dict:
    live = [label for label in LABELS if not k[label]["void"]]
    failed = [name for name, v in gd.items() if not v["pass"] and name not in reported_only]
    reported_fail = [name for name, v in gd.items() if not v["pass"] and name in reported_only]
    if not live:
        return {"reading": "UNREADABLE", "classes_read": live, "failed_guards": failed,
                "reported_guards_failing": reported_fail}
    if all(k[label]["K1"] for label in live):
        base = "SUCCESS"
    elif any(k[label]["K1"] or k[label]["K2"] for label in live):
        base = "PARTIAL"
    else:
        base = "NO EFFECT"
    text = base + (f" WITH COST ({', '.join(failed)})" if failed else "")
    return {"reading": text, "base": base, "classes_read": live, "failed_guards": failed,
            "reported_guards_failing": reported_fail}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    require()
    fits = load(P2D / "words" / "fits.json")
    meta = load(P2D / "words" / "samples_meta.json")
    mtr = load(P2D / "words" / "metrics.json")
    vd = v_d1b(fits, meta)
    if not vd["pass"]:
        raise SystemExit(f"STOP: V-D1b failed -- the word pipeline drifted from R2's: {vd['differences'][:5]}")
    words = word_readings(fits)
    floors = {key: v["floor"] for key, v in words.items()}
    targets = load(ROOT / "data" / "e11" / "targets.json")
    res = {"registration": C.REGISTRATION, "baseline": REF, "validity": {"V-D1b": vd, "floors": floors},
           "words": words, "host_gain_and_specificity": host_gain_and_specificity(words),
           "samples_meta": meta, "arms": {}}
    for arm in ARMS:
        k = k1_k2(words, arm)
        blind_ok = {label: not k[label]["blind_spot"] for label in LABELS}
        rep = merged(arm)
        strict, tiered, reported_only = all_guards(rep, targets, words, mtr, blind_ok, arm)
        rd_strict, rd_tiered = reading(k, strict, reported_only), reading(k, tiered, reported_only)
        text = rd_tiered["reading"] + ("" if rd_strict["reading"] == rd_tiered["reading"]
                                       else f" -- strict: {rd_strict['reading']}")
        f = finish_line(rep, arm, strict["G5"])
        vt, vg = load(arm_dir(arm) / "VERIFY_TRAIN.json"), load(arm_dir(arm) / "VERIFY_GENERATE.json")
        res["arms"][arm] = {
            "K1_K2": k, "guards_strict": strict, "guards_tiered": tiered, "reported_only": list(reported_only),
            "reading": {"binding": "tiered (section 9; PREREG_V2_P1_MEMORY.md section 12)", "label": text,
                        "tiered": rd_tiered, "strict": rd_strict},
            "reported": {"fit_per_character": vt.get("field_fit_per_character"),
                         "page_row_name_gap": vt.get("page_row_name_gap"),
                         "piece_rule_mass_removed": (vg.get("v_rules") or {}).get("piece_rule_mass_removed"),
                         "tokens_per_url": {label: mtr[label]["tokens_per_url"][arm] for label in LABELS},
                         "examples": {label: {"words": mtr[label]["examples"][arm],
                                              "hosts": mtr[label]["host_examples"][arm]} for label in LABELS}},
            "E3_for_the_record": {**f, "E3": "MET (V2 does not reopen E3)" if f["met"] else "NOT MET"},
            "difference": difference(rep, arm, rep, REF)}
    (P2D / "compare.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("validity: V-gen-equiv, VERIFY_TRAIN, VERIFY_GENERATE, V4 (both arms), V-D1b PASS; floors flagged:",
          [key for key, v in floors.items() if v["flag"]] or "none")
    for det, view in READINGS:
        for label in LABELS:
            w = words[f"{det}|{view}|{label}"]
            arms = "  ".join(f"{arm} {w[arm]['mean']:+.4f} {w[arm]['verdict']:<10}" for arm in ARMS)
            o4 = f"; copier control {w['O4']['mean']:+.4f} {w['O4']['verdict']}" if "O4" in w else ""
            print(f"  {det}-{view} {label:<9} e16_b5 {w['base']['mean']:+.4f} {w['base']['verdict']:<10} {arms}{o4}")
    for label, v in res["host_gain_and_specificity"].items():
        gain, spec = v["B's host gain"], v["A's specificity"]
        print(f"  {label}: B's host gain {gain['mean_diff']:+.4f} (recorded: {gain['recorded']}); "
              f"A's specificity holds: {spec['holds']}")
    for arm in ARMS:
        a = res["arms"][arm]
        for label in LABELS:
            x = a["K1_K2"][label]
            print(f"{arm} {label}: K1 ({x['pass_view']}) {'VOID' if x['void'] else ('PASS' if x['K1'] else 'FAIL')}; "
                  f"K2 {'PASS on ' + ','.join(x['K2_views']) if x['K2'] else 'FAIL'}"
                  f"{'; BLIND SPOT' if x['blind_spot'] else ''}")
        for name in ("guards_strict", "guards_tiered"):
            print(f"  {arm} {name[7:]}: " + "  ".join(
                f"{n}{'*' if n in a['reported_only'] else ''} {'PASS' if v['pass'] else 'FAIL'}"
                for n, v in a[name].items()))
        print(f"  {arm} READING: {a['reading']['label']}")
        e3 = a["E3_for_the_record"]
        print(f"  {arm} E3 for the record: {e3['E3']} ({len(e3['cells_not_cleared'])} cells not cleared, "
              f"{len(e3['slash_rows_missed'])} slash rows missed)")
    print("(* reported for this arm, not deciding)")
    print("wrote", P2D / "compare.json")


if __name__ == "__main__":
    main()
