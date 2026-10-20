"""PC1 -- the reading (research/native/PREREG_PC1.md sections 3-5): validity V1-V6,
the clip-weighted rates (M3), readings R1-R5 and the reported tables.

    python research/native/pc1_read.py > data/pc1/tables.md

Reads data/pc1/{rows.npz, rows_meta.json, plan.json, clip.json}, the shipped table
and e10_plan's training record; writes data/pc1/pc1_report.json. PC1_RESULTS.md
carries the printed tables verbatim as its appendix.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from pc1_measure import BUNDLE, FRAMES, LABELS, OUT, SHORT, TRAIN  # noqa: E402

#: V2 (e10_plan's train_result.json) and V3 (FINDING_LENGTH_WEIGHTING.md 2.3).
V2_FITS = {"benign": 2_545_305, "phishing": 238_199}
V2_DROPPED = 10_313
V3_TARGET = {"benign": {"P": 0.733, "Q": 0.0653}, "phishing": {"P": 0.394, "Q": 0.0451}}
V3_TOL = 0.0010
#: V5: the record's clipped share under 30 scored tokens per row, after step 1,000.
V5_RECORD = 0.327
V5_TOL = 0.15
V5_LONG_MAX = 0.02
#: R2's floor, as registered: 1.0 point, the benign path tolerance
#: (DECISION_2026-09-29 section 2: T = 0.99 points at 73.37%).
R2_FLOOR = 0.010
BANDS = (("common", 10_000, 10 ** 12), ("middle", 1_000, 10_000), ("rare", 0, 1_000))
LONG_BANDS = ((30, 60), (60, 120), (120, 10 ** 9))


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def logit(p: float) -> float:
    return math.log(p / (1 - p))


# --------------------------------------------------------------------------- #
# M3 -- clip factors by bin, then clip-weighted rates
# --------------------------------------------------------------------------- #

def clip_bins(batches: List[dict]) -> dict:
    """PREREG_PC1 M2: the short band in four bins of equal batch count by scored
    tokens per row; then [30, 60), [60, 120), [120, inf). omega(bin) = mean omega."""
    fin = [b for b in batches if b["omega"] is not None]
    short = sorted((b for b in fin if b["scored_per_row"] < SHORT), key=lambda b: b["scored_per_row"])
    k = len(short)
    groups = [short[round(g * k / 4):round((g + 1) * k / 4)] for g in range(4)]
    edges = [g[0]["scored_per_row"] for g in groups[1:]]           # lower edges of bins 2-4
    def norms(grp):                                  # descriptive: the norm range
        n = sorted(b["norm"] for b in grp)
        return {"norm_min": n[0], "norm_median": n[len(n) // 2], "norm_max": n[-1]} if n else {}

    bins = []
    lo = 0.0
    for g, grp in enumerate(groups):
        hi = edges[g] if g < 3 else SHORT
        bins.append({"lo": lo, "hi": hi, "batches": len(grp),
                     "omega": sum(b["omega"] for b in grp) / len(grp),
                     "clipped": sum(b["norm"] > 1 for b in grp), **norms(grp)})
        lo = hi
    for lo, hi in LONG_BANDS:
        grp = [b for b in fin if lo <= b["scored_per_row"] < hi]
        bins.append({"lo": lo, "hi": hi, "batches": len(grp),
                     "omega": sum(b["omega"] for b in grp) / len(grp) if grp else 1.0,
                     "clipped": sum(b["norm"] > 1 for b in grp), **norms(grp)})
    return {"bins": bins, "nonfinite": len(batches) - len(fin)}


def omega_of(m: np.ndarray, bins: List[dict]) -> np.ndarray:
    w = np.ones(len(m), dtype=np.float64)
    for b in bins:
        sel = (m >= b["lo"]) & (m < b["hi"])
        w[sel] = b["omega"]
    return w


def aggregate(rows, n_suf: int, omega: np.ndarray) -> Dict[str, np.ndarray]:
    """Per (class, frame, suffix) counts, plain and clip-weighted, over rows that fit."""
    fit = rows["fits"] == 1
    key = ((rows["cls"].astype(np.int64) * 2 + rows["frame"]) * n_suf + rows["suf"])[fit]
    size = 4 * n_suf
    P = rows["path"][fit].astype(np.float64)
    Q = rows["query"][fit].astype(np.float64)
    om = omega[fit]
    out = {}
    for name, wt in (("", np.ones_like(om)), ("w", om)):
        out[f"n{name}"] = np.bincount(key, wt, size)
        out[f"P{name}"] = np.bincount(key, wt * P, size)
        out[f"Q{name}"] = np.bincount(key, wt * Q, size)
        out[f"n0{name}"] = np.bincount(key, wt * (1 - P), size)
        out[f"Q0{name}"] = np.bincount(key, wt * (1 - P) * Q, size)
        out[f"Q1{name}"] = np.bincount(key, wt * P * Q, size)
    return out


class Cells:
    """Rates of one (class, frame) by suffix name."""

    def __init__(self, agg, names: List[str], c: int, f: int, n_suf: int) -> None:
        self.base = (c * 2 + f) * n_suf
        self.agg, self.index = agg, {s: i for i, s in enumerate(names)}
        sl = slice(self.base, self.base + n_suf)
        self.tot = {k: float(v[sl].sum()) for k, v in agg.items()}

    def get(self, s: str, k: str) -> float:
        i = self.index.get(s)
        return 0.0 if i is None else float(self.agg[k][self.base + i])

    def rate(self, s: str, num: str, den: str, pooled: bool = True) -> float:
        d = self.get(s, den)
        if d > 0:
            return self.get(s, num) / d
        return self.tot[num] / self.tot[den] if pooled else float("nan")

    def cond(self, s: str, w: str = "") -> Dict[str, float]:
        """r, r_Q, r_Q|0, r_Q|1 (empty conditional cells take the pooled rate)."""
        r = self.rate(s, f"P{w}", f"n{w}")
        n1 = self.get(s, f"n{w}") - self.get(s, f"n0{w}")
        q0 = self.rate(s, f"Q0{w}", f"n0{w}")
        q1 = (self.get(s, f"Q1{w}") / n1 if n1 > 0 else
              self.tot[f"Q1{w}"] / (self.tot[f"n{w}"] - self.tot[f"n0{w}"]))
        return {"r": r, "rQ": (1 - r) * q0 + r * q1, "rQ0": q0, "rQ1": q1}


def continuation(cells: Cells, s: str, names: List[str]) -> float:
    """Share of rows whose suffix starts with s and is longer, among rows whose
    suffix starts with s (plain counts)."""
    start = sum(cells.get(t, "n") for t in names if t.startswith(s))
    return (start - cells.get(s, "n")) / start if start else float("nan")


# --------------------------------------------------------------------------- #
# The readings
# --------------------------------------------------------------------------- #

def gap_terms(label: str, plan: dict, dom: Cells, shipped: Dict[str, int]) -> dict:
    """R1, R3's contributions and R5's split for one class, domain frame, shipped weights."""
    wsum = sum(shipped.values())
    t = {k: 0.0 for k in ("Delta", "C", "E", "DQ", "M", "K0", "K1", "CQ")}
    bands = {b: {"weight": 0.0, "Delta": 0.0, "suffixes": 0} for b, _, _ in BANDS}
    for s, cnt in shipped.items():
        w = cnt / wsum
        x = plan[s]
        d, dw = dom.cond(s), dom.cond(s, "w")
        t["Delta"] += w * (x["p"] - d["r"])
        t["C"] += w * (dw["r"] - d["r"])
        t["E"] += w * (x["p"] - dw["r"])
        t["DQ"] += w * (x["q"] - d["rQ"])
        t["M"] += w * (x["p"] - d["r"]) * (d["rQ1"] - d["rQ0"])
        t["K0"] += w * (1 - x["p"]) * (x["q_given_p0"] - d["rQ0"])
        t["K1"] += w * x["p"] * (x["q_given_p1"] - d["rQ1"])
        t["CQ"] += w * (dw["rQ"] - d["rQ"])
        n = dom.get(s, "n")
        band = next(b for b, lo, hi in BANDS if lo <= n < hi)
        bands[band]["weight"] += w
        bands[band]["Delta"] += w * (x["p"] - d["r"])
        bands[band]["suffixes"] += 1
    t["bands"] = bands
    t["target_P"] = sum(cnt / wsum * dom.cond(s)["r"] for s, cnt in shipped.items())
    t["target_Q"] = sum(cnt / wsum * dom.cond(s)["rQ"] for s, cnt in shipped.items())
    t["plan_P"] = sum(cnt / wsum * plan[s]["p"] for s, cnt in shipped.items())
    t["plan_Q"] = sum(cnt / wsum * plan[s]["q"] for s, cnt in shipped.items())
    return t


def frame_eps(plan_f: dict, cells: Cells, names: List[str]) -> dict:
    """Each frame on its own training rows: the model's mean P against the plain
    and the clip-weighted data rates, in log-odds."""
    num = den = dat = datw = 0.0
    for s in names:
        n = cells.get(s, "n")
        if n <= 0:
            continue
        num += n * plan_f[s]["p"]
        dat += cells.get(s, "P")
        datw += n * cells.cond(s, "w")["r"]
        den += n
    a_m, a_d, a_w = num / den, dat / den, datw / den
    return {"rows": den, "model": a_m, "data": a_d, "data_clip_weighted": a_w,
            "eps_plain": logit(a_m) - logit(a_d), "eps_residual": logit(a_m) - logit(a_w)}


def read_r1(t: dict) -> str:
    share = t["C"] / t["Delta"]
    return "CLIP-LED" if share >= 2 / 3 else "CLIP-PARTIAL" if share >= 1 / 3 else "CLIP-MINOR"


def read_r2(t: dict, eps: dict) -> str:
    if abs(t["E"]) < R2_FLOOR:
        return "NOT-NEEDED"
    ed, ep = eps["domain"]["eps_residual"], eps["page"]["eps_residual"]
    if ed > 0 and ep <= ed / 3:
        return "FRAME-PULL"
    if ed > 0 and ep >= 2 * ed / 3:
        return "OFFSET"
    return "OTHER"


def read_r3(t: dict) -> str:
    d = t["Delta"]
    if t["bands"]["common"]["Delta"] >= 0.5 * d:
        return "COMMON-LED"
    if t["bands"]["rare"]["Delta"] >= 0.5 * d:
        return "RARE-LED"
    return "SPREAD"


def read_r5(t: dict) -> str:
    d = t["DQ"]
    for name, k in (("Q-MIX", "M"), ("Q-NOPATH", "K0"), ("Q-PATH", "K1")):
        if t[k] / d >= 2 / 3:
            return name
    return "Q-SPLIT"


def main() -> None:
    from uie.suffixes import SuffixTable
    sys.stdout.reconfigure(encoding="utf-8")
    rows = dict(np.load(OUT / "rows.npz"))           # decompress each column once
    meta = load(OUT / "rows_meta.json")
    plan_all = load(OUT / "plan.json")["checkpoints"]
    clip = load(OUT / "clip.json")
    shipped = SuffixTable.load(BUNDLE / "suffix_table.json").counts
    names = meta["suffixes"]
    ns = len(names)
    binned = clip_bins(clip["batches"])
    omega = omega_of(rows["m"].astype(np.float64), binned["bins"])
    agg = aggregate(rows, ns, omega)
    cells = {(label, frame): Cells(agg, names, c, f, ns)
             for c, label in enumerate(LABELS) for f, frame in enumerate(FRAMES)}
    best = plan_all["best"]

    # ---- validity
    fails = []
    v1 = {label: {k: abs(x["here"] - x["score_plan"]) for k, x in v.items()}
          for label, v in best["V1"].items()}
    if max(d for v in v1.values() for d in v.values()) > 1e-9:
        fails.append("V1")
    if meta["fits"] != V2_FITS or meta["dropped"] != V2_DROPPED:
        fails.append("V2")
    terms = {label: gap_terms(label, best["plan"][label]["domain"], cells[(label, "domain")],
                              shipped[label]) for label in LABELS}
    v3 = {label: {"P": terms[label]["target_P"], "Q": terms[label]["target_Q"]} for label in LABELS}
    if any(abs(v3[l][k] - V3_TARGET[l][k]) > V3_TOL for l in LABELS for k in ("P", "Q")):
        fails.append("V3")
    if meta.get("V4_max_abs_diff", 1) != 0:
        fails.append("V4")
    fin = [b for b in clip["batches"] if b["omega"] is not None]
    short = [b for b in fin if b["scored_per_row"] < SHORT]
    v5 = {"short_clipped_share": sum(b["norm"] > 1 for b in short) / len(short),
          "short_batches": len(short)}
    for lo, hi in LONG_BANDS:
        grp = [b for b in fin if lo <= b["scored_per_row"] < hi]
        v5[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = (
            sum(b["norm"] > 1 for b in grp) / len(grp) if grp else None, len(grp))
    v5_pass = (abs(v5["short_clipped_share"] - V5_RECORD) <= V5_TOL
               and all((v5[k][0] or 0) <= V5_LONG_MAX for k in ("30-60", "60-120")))
    if max(best["V6"].values()) > 1e-9:
        fails.append("V6")
    verdict = "PASS" if not fails else "FAIL"

    # ---- readings (benign registered; phishing reported)
    eps = {label: {frame: frame_eps(best["plan"][label][frame], cells[(label, frame)], names)
                   for frame in FRAMES} for label in LABELS}
    readings = {}
    for label in LABELS:
        t = terms[label]
        readings[label] = {
            "R1": read_r1(t) if v5_pass else "UNREPRESENTATIVE",
            "R2": read_r2(t, eps[label]) if v5_pass else "UNREPRESENTATIVE",
            "R3": read_r3(t), "R5": read_r5(t)}
    legal = {}
    for label in LABELS:
        dom = cells[(label, "domain")]
        wsum = sum(shipped[label].values())
        pl = best["plan"][label]["domain"]
        dev = mean_m = mean_c = wdef = 0.0
        undefined = []
        for s, cnt in shipped[label].items():
            w = cnt / wsum
            c = continuation(dom, s, names)
            if math.isnan(c):
                # Amendment 2: no domain-frame row that fits carries s, so its
                # continuation is undefined; left out, its weight reported.
                undefined.append(s)
                continue
            wdef += w
            dev += w * abs(pl[s]["legal_mass_P"] - (1 - c))
            mean_m += w * pl[s]["legal_mass_P"]
            mean_c += w * (1 - c)
        dev, mean_m, mean_c = dev / wdef, mean_m / wdef, mean_c / wdef
        legal[label] = {"mean_abs_dev": dev, "model_legal_mass": mean_m,
                        "data_1_minus_cont": mean_c, "undefined": sorted(undefined),
                        "undefined_weight": 1 - wdef,
                        "reading": "LEGAL-EXPLAINED" if dev <= 0.02 else "LEGAL-DISCREPANT"}
        readings[label]["R4"] = legal[label]["reading"]
    final = {label: gap_terms(label, plan_all["final"]["plan"][label]["domain"],
                              cells[(label, "domain")], shipped[label]) for label in LABELS}
    short_rows = {}
    fit = rows["fits"] == 1
    for c, label in enumerate(LABELS):
        for f, frame in enumerate(FRAMES):
            sel = fit & (rows["cls"] == c) & (rows["frame"] == f)
            m, p = rows["m"][sel], rows["path"][sel]
            short_rows[f"{label} {frame}"] = {
                "no_path_rows": int((p == 0).sum()), "no_path_short": float((m[p == 0] < SHORT).mean()),
                "path_rows": int((p == 1).sum()), "path_short": float((m[p == 1] < SHORT).mean()),
                "userinfo_rows": int(rows["usr"][sel].sum())}

    report = {"registration": "research/native/PREREG_PC1.md", "verdict": verdict,
              "failures": fails, "V1": v1, "V2": {"fits": meta["fits"], "dropped": meta["dropped"]},
              "V3": v3, "V4": meta.get("V4_max_abs_diff"), "V5": {**v5, "pass": v5_pass},
              "V6": best["V6"], "V6_probe_mismatch": best["V6_probe_mismatch"],
              "K10a_training_prefix": best["K10a_training_prefix"],
              "clip_bins": binned, "terms": terms, "eps": eps,
              "legal": legal, "readings": readings, "final_terms": final,
              "short_rows": short_rows}
    (OUT / "pc1_report.json").write_text(json.dumps(report, indent=1, default=float),
                                         encoding="utf-8")
    render(report, best, cells, shipped, names)


# --------------------------------------------------------------------------- #
# Tables (the results document's appendix, verbatim)
# --------------------------------------------------------------------------- #

def pts(x: float) -> str:
    return f"{100 * x:+.2f}"


def render(rep: dict, best: dict, cells: Dict, shipped: Dict, names: List[str]) -> None:
    print(f"### Validity\n\nVerdict: **{rep['verdict']}**; failures: {rep['failures'] or 'none'}\n")
    for label, v in rep["V1"].items():
        print(f"- V1 {label}: max |K10a here − score_plan.json| = {max(v.values()):.1e}")
    print(f"- V2: rows that fit {rep['V2']['fits']}, dropped {rep['V2']['dropped']:,} "
          f"(training recorded {V2_FITS}, {V2_DROPPED:,})")
    for label, v in rep["V3"].items():
        print(f"- V3 {label}: shipped-weighted domain P {v['P']:.4f} (target {V3_TARGET[label]['P']}), "
              f"Q {v['Q']:.4f} (target {V3_TARGET[label]['Q']})")
    print(f"- V4: max |len − 3 − Σ field mask| over the sampled rows = {rep['V4']}")
    v5 = rep["V5"]
    print(f"- V5: clipped share under {SHORT} scored tokens per row {100 * v5['short_clipped_share']:.1f}% "
          f"of {v5['short_batches']} batches (record {100 * V5_RECORD:.1f}%); "
          + "; ".join(f"{k}: {'—' if v5[k][0] is None else f'{100 * v5[k][0]:.1f}%'} of {v5[k][1]}"
                      for k in ("30-60", "60-120", "120-inf"))
          + f" → **{'PASS' if v5['pass'] else 'FAIL'}**")
    print(f"- V6 (amendment 1): max |frame-generalised − plan_by_suffix| where the probe parses "
          f"to the suffix itself = " + ", ".join(f"{l} {x:.1e}" for l, x in rep["V6"].items()))
    for label, mm in rep["V6_probe_mismatch"].items():
        for s, x in mm.items():
            print(f"  - {label} `{s}` (probe parses as `{x['probe_parses_as']}`, shipped "
                  f"{100 * x['shipped_share']:.2f}%): plan P probe {x['probe']['p']:.4f}, "
                  f"training prefix {x['training_prefix']['p']:.4f}")
    for label, x in rep["K10a_training_prefix"].items():
        print(f"- K10a's quantities with the training prefix, {label}: P {x['P']:.6f}, "
              f"Q {x['Q']:.6f}, legal mass {x['legal_mass_P']:.6f} (reading unchanged)")
    print("\n### Clip factors on best.pt (M2)\n\n| scored tokens per row | batches | clipped | mean ω | gradient norm: min / median / max |\n|---|---|---|---|---|")
    for b in rep["clip_bins"]["bins"]:
        hi = "∞" if b["hi"] >= 10 ** 9 else f"{b['hi']:.1f}"
        rng = (f"{b['norm_min']:.2f} / {b['norm_median']:.2f} / {b['norm_max']:.2f}"
               if "norm_min" in b else "—")
        print(f"| {b['lo']:.1f}–{hi} | {b['batches']} | {b['clipped']} | {b['omega']:.4f} | {rng} |")
    print(f"\nNon-finite norms: {rep['clip_bins']['nonfinite']}\n")
    print("### Rows under 30 scored tokens (rows that fit)\n\n| class, frame | no-path rows | of them short | path rows | of them short | with userinfo |\n|---|---|---|---|---|---|")
    for k, v in rep["short_rows"].items():
        print(f"| {k} | {v['no_path_rows']:,} | {100 * v['no_path_short']:.1f}% | {v['path_rows']:,} | "
              f"{100 * v['path_short']:.1f}% | {v['userinfo_rows']:,} |")
    print("\n### R1 — clipping (domain frame, shipped weights; points)\n\n| class | plan P | data P | clip-weighted data P | gap Δ | clip part C | C / Δ | residual E | reading |\n|---|---|---|---|---|---|---|---|---|")
    for label in LABELS:
        t = rep["terms"][label]
        print(f"| {label} | {100 * t['plan_P']:.2f}% | {100 * t['target_P']:.2f}% | "
              f"{100 * (t['target_P'] + t['C']):.2f}% | {pts(t['Delta'])} | {pts(t['C'])} | "
              f"{t['C'] / t['Delta']:.2f} | {pts(t['E'])} | {rep['readings'][label]['R1']}"
              f"{'' if label == 'benign' else ' (reported)'} |")
    print("\n### R2 — each frame on its own training rows (log-odds)\n\n| class | frame | rows | model P | data P | clip-weighted | ε plain | ε residual |\n|---|---|---|---|---|---|---|---|")
    for label in LABELS:
        for frame in FRAMES:
            e = rep["eps"][label][frame]
            print(f"| {label} | {frame} | {e['rows']:,.0f} | {100 * e['model']:.2f}% | {100 * e['data']:.2f}% | "
                  f"{100 * e['data_clip_weighted']:.2f}% | {e['eps_plain']:+.3f} | {e['eps_residual']:+.3f} |")
    print("\nR2: " + "; ".join(f"{l} **{rep['readings'][l]['R2']}**" + ("" if l == "benign" else " (reported)")
                              for l in LABELS))
    print("\n### R3 — where the gap sits (domain-frame rows that fit per suffix; shipped weights)\n\n| class | band | suffixes | shipped weight | contribution to Δ | share of Δ |\n|---|---|---|---|---|---|")
    for label in LABELS:
        t = rep["terms"][label]
        for b, v in t["bands"].items():
            print(f"| {label} | {b} | {v['suffixes']} | {100 * v['weight']:.1f}% | {pts(v['Delta'])} | "
                  f"{v['Delta'] / t['Delta']:.2f} |")
    print("\nR3: " + "; ".join(f"{l} **{rep['readings'][l]['R3']}**" + ("" if l == "benign" else " (reported)")
                              for l in LABELS))
    print("\n### R4 — the plan slot's off-slot mass (domain frame, shipped weights)\n\n| class | model legal mass | data 1 − continuation | mean \\|difference\\| | left out (amendment 2) | reading |\n|---|---|---|---|---|---|")
    for label in LABELS:
        g = rep["legal"][label]
        print(f"| {label} | {g['model_legal_mass']:.4f} | {g['data_1_minus_cont']:.4f} | {g['mean_abs_dev']:.4f} | "
              f"{', '.join(g['undefined']) or 'none'} ({100 * g['undefined_weight']:.4f}% weight) | "
              f"**{g['reading']}** |")
    print("\n### R5 — the plan's query shortfall (domain frame, shipped weights; points)\n\n| class | plan Q | data Q | Δ_Q | path mix M | no-path K0 | after-path K1 | clip C_Q | reading |\n|---|---|---|---|---|---|---|---|---|")
    for label in LABELS:
        t = rep["terms"][label]
        print(f"| {label} | {100 * t['plan_Q']:.2f}% | {100 * t['target_Q']:.2f}% | {pts(t['DQ'])} | "
              f"{pts(t['M'])} ({t['M'] / t['DQ']:.2f}) | {pts(t['K0'])} ({t['K0'] / t['DQ']:.2f}) | "
              f"{pts(t['K1'])} ({t['K1'] / t['DQ']:.2f}) | {pts(t['CQ'])} | **{rep['readings'][label]['R5']}** |")
    print("\n### Reported: final.pt against best.pt (domain frame, shipped weights)\n\n| class | plan P best / final | plan Q best / final |\n|---|---|---|")
    for label in LABELS:
        b, f = rep["terms"][label], rep["final_terms"][label]
        print(f"| {label} | {100 * b['plan_P']:.2f}% / {100 * f['plan_P']:.2f}% | "
              f"{100 * b['plan_Q']:.3f}% / {100 * f['plan_Q']:.3f}% |")
    for label in LABELS:
        dom, page = cells[(label, "domain")], cells[(label, "page")]
        pd, pp = best["plan"][label]["domain"], best["plan"][label]["page"]
        wsum = sum(shipped[label].values())
        top = sorted(shipped[label], key=lambda s: -shipped[label][s])[:20]
        print(f"\n### Reported: top 20 shipped suffixes, {label}\n\n| suffix | shipped | domain rows | data P | clip-w. P | plan P | page rows | page data P | page plan P | legal mass | 1 − cont |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for s in top:
            d, dw = dom.cond(s), dom.cond(s, "w")
            n_p = page.get(s, "n")
            pdat = f"{100 * page.cond(s)['r']:.1f}%" if n_p else "—"
            pplan = f"{100 * pp[s]['p']:.1f}%" if s in pp else "—"
            c = continuation(dom, s, names)
            print(f"| {s} | {100 * shipped[label][s] / wsum:.2f}% | {dom.get(s, 'n'):,.0f} | {100 * d['r']:.1f}% | "
                  f"{100 * dw['r']:.1f}% | {100 * pd[s]['p']:.1f}% | {n_p:,.0f} | {pdat} | {pplan} | "
                  f"{pd[s]['legal_mass_P']:.3f} | {1 - c:.3f} |")
    print("\n### Readings\n\n| class | R1 | R2 | R3 | R4 | R5 |\n|---|---|---|---|---|---|")
    for label in LABELS:
        r = rep["readings"][label]
        tag = "" if label == "benign" else " (R1–R3 reported)"
        print(f"| {label}{tag} | {r['R1']} | {r['R2']} | {r['R3']} | {r['R4']} | {r['R5']} |")


if __name__ == "__main__":
    main()
