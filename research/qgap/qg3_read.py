"""QG3 -- the reading (research/qgap/PREREG_QG3.md sections 2-4): per (region, token)
frequency and calibration; the query's shortfall L split into R (the uncontrolled
reference at matched frequency) and S = L - R (the hidden-state residual); block
bootstrap (models paired on the same blocks); VR1-VR3; the registered reading.
Prints the tables for QG3_RESULTS.md; writes data/qgap/qg3_report.json.

    python research/qgap/qg3_read.py > data/qgap/qg3_tables.md
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

import numpy as np  # noqa: E402

import qg_common as Q  # noqa: E402

MODELS = ("e9_ln", "e9_base", "e7")
QG2D_S0 = {"benign": {"host": 0.541, "path": 0.806},      # QG2D_RESULTS.md section 3
           "phishing": {"host": 0.592, "path": 0.777}}
VR1_TOL, MIN_EVENTS = 0.005, 200
TARGETS = [("host", "<QRY>"), ("path", "<QRY>")]
SECONDARY = [("host", "<PTH>")]
UNC_STRUCT = {("host", "<SUB>"), ("host", "<PRT>"), ("query", "<FRG>"),
              ("query", "<eos>"), ("fragment", "<eos>")}
CONSTRAINED = {("host", "<eos>"), ("path", "<eos>"), ("host", "<FRG>"), ("path", "<FRG>")}
RI = {r: i for i, r in enumerate(Q.REGIONS)}


def load(model, label):
    base = Q.DATA / f"qg3_{model}_{label}"
    meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("limit"):
        raise SystemExit(f"{base.name} is a dry run; not read")
    z = np.load(base.with_suffix(".npz"))
    return z["P"], z["N"], z["POS"], meta


def status(region, t, chars):
    if (region, t) in TARGETS:
        return "target"
    if (region, t) in SECONDARY:
        return "secondary"
    if t in chars or (region, t) in UNC_STRUCT:
        return "uncontrolled"
    if (region, t) in CONSTRAINED:
        return "constrained"
    return "other"


def pairs_of(N, meta):
    """The fixed pair set: (region, token string, token id) with >= 200 events."""
    itos, chars = meta["itos"], set(meta["chars"])
    tot = N.sum(0)
    out = []
    for r, ri in RI.items():
        for t, n in enumerate(tot[ri]):
            if n >= MIN_EVENTS:
                out.append((r, itos[t], t, status(r, itos[t], chars)))
    return out


def measure(P, N, POS, pairs, idx=None):
    """f, c, n per pair from block totals (optionally resampled blocks)."""
    if idx is not None:
        P, N, POS = P[idx], N[idx], POS[idx]
    p, n, pos = P.sum(0), N.sum(0), POS.sum(0)
    rows = []
    for r, s, t, st in pairs:
        ev = n[RI[r], t]
        rows.append((r, s, st, ev / pos[RI[r]], p[RI[r], t] / ev if ev else np.nan, ev))
    return rows


def reference(f, rows, keep=lambda st, r, s: st == "uncontrolled", rule=True):
    """Event-weighted mean log c of the kept pairs within x3 (else x10) of f."""
    cand = [(ff, cc, nn) for r, s, st, ff, cc, nn in rows
            if keep(st, r, s) and nn > 0 and np.isfinite(cc) and cc > 0]
    for factor in (3, 10):
        sel = [(cc, nn) for ff, cc, nn in cand if f / factor <= ff <= f * factor]
        ok = (len(sel) >= 5 and sum(nn for _, nn in sel) >= 1000) if rule else len(sel) >= 1
        if ok:
            w = np.array([nn for _, nn in sel])
            return float((w * np.log([cc for cc, _ in sel])).sum() / w.sum()), factor, len(sel), float(w.sum())
    return float("nan"), None, 0, 0.0


def split(rows, which=TARGETS) -> dict:
    by = {(r, s): (ff, cc, nn) for r, s, st, ff, cc, nn in rows}
    t = [by[k] for k in which if k in by]
    if not t:
        return {}
    w = np.array([nn for _, _, nn in t])
    L = float((w * np.log([cc for _, cc, _ in t])).sum() / w.sum())
    refs = [reference(ff, rows) for ff, _, _ in t]
    if any(not np.isfinite(r[0]) for r in refs):
        return {"L": L, "R": float("nan"), "S": float("nan"), "resolved": False,
                "coverage": [r[1:] for r in refs]}
    R = float((w * np.array([r[0] for r in refs])).sum() / w.sum())
    return {"L": L, "R": R, "S": L - R, "share_R": R / L if L else float("nan"),
            "share_S": (L - R) / L if L else float("nan"), "resolved": True,
            "coverage": [r[1:] for r in refs]}


def ci(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    return [float("nan")] * 2 if not len(v) else [float(np.percentile(v, 2.5)),
                                                  float(np.percentile(v, 97.5))]


def reading(pt, c) -> str:
    """PREREG_QG3 section 3: rows in order, the first that holds."""
    if not pt.get("resolved"):
        return "UNRESOLVED (a target not covered)"
    below = lambda k: c[k][1] < 0
    if below("R") and below("S") and pt["share_R"] < 0.75 and pt["share_S"] < 0.75:
        return "SPLIT"
    if pt["share_R"] >= 0.75 and below("R"):
        return "RARITY-LED"
    if pt["share_S"] >= 0.75 and below("S"):
        return "STATE-LED"
    return "UNRESOLVED"


def curve(rows, kind):
    """Half-decade bins of f: event-weighted geometric-mean c of uncontrolled pairs."""
    keep = [(ff, cc, nn) for r, s, st, ff, cc, nn in rows if st == "uncontrolled"
            and ((kind == "structural") == ((r, s) in UNC_STRUCT)) and np.isfinite(cc)]
    out = []
    for lo in np.arange(-6.0, 0.0, 0.5):
        sel = [(cc, nn) for ff, cc, nn in keep if 10 ** lo <= ff < 10 ** (lo + 0.5)]
        if sel:
            w = np.array([nn for _, nn in sel])
            out.append({"bin": f"1e{lo:+.1f}", "pairs": len(sel), "events": float(w.sum()),
                        "c": float(np.exp((w * np.log([cc for cc, _ in sel])).sum() / w.sum()))})
    return out


def per_class(label) -> dict:
    data = {m: load(m, label) for m in MODELS}
    P0, N0, POS0, meta0 = data["e9_ln"]
    pairs = pairs_of(N0, meta0)
    out = {"pairs": len(pairs), "by_status": {}}
    for st in ("target", "secondary", "uncontrolled", "constrained", "other"):
        out["by_status"][st] = sum(1 for p in pairs if p[3] == st)
    rng = np.random.default_rng(Q.BOOT_SEED)
    nb = P0.shape[0]
    boots = {m: [] for m in MODELS}
    diffs = {"S": [], "R": []}
    for _ in range(Q.BOOT):
        idx = rng.integers(0, nb, nb)
        sp = {}
        for m in MODELS:
            P, N, POS, _ = data[m]
            sp[m] = split(measure(P, N, POS, pairs, idx))
            boots[m].append(sp[m])
        if sp["e9_ln"].get("resolved") and sp["e9_base"].get("resolved"):
            diffs["S"].append(sp["e9_ln"]["S"] - sp["e9_base"]["S"])
            diffs["R"].append(sp["e9_ln"]["R"] - sp["e9_base"]["R"])
    for m in MODELS:
        P, N, POS, meta = data[m]
        rows = measure(P, N, POS, pairs)
        pt = split(rows)
        cis = {k: ci([b.get(k) for b in boots[m]]) for k in ("L", "R", "S", "share_R", "share_S")}
        by = {(r, s): (ff, cc, nn) for r, s, st, ff, cc, nn in rows}
        detail = {}
        for key in TARGETS + SECONDARY:
            if key in by:
                ff, cc, nn = by[key]
                ref, factor, npairs, nev = reference(ff, rows)
                sref = reference(ff, rows, keep=lambda st, r, s: (r, s) in UNC_STRUCT, rule=False)
                detail[f"{key[0]} {key[1]}"] = {
                    "f": ff, "c": cc, "events": nn,
                    "reference_c": math.exp(ref) if np.isfinite(ref) else None,
                    "window": factor, "reference_pairs": npairs, "reference_events": nev,
                    "structural_reference_c": math.exp(sref[0]) if np.isfinite(sref[0]) else None,
                    "structural_window": sref[1], "structural_pairs": sref[2],
                    "structural_events": sref[3]}
        constrained = {f"{r} {s}": {"f": ff, "c": cc, "events": nn}
                       for r, s, st, ff, cc, nn in rows if st == "constrained"}
        out[m] = {"point": pt, "ci": cis, "reading": reading(pt, cis), "targets": detail,
                  "constrained": constrained,
                  "curve": {k: curve(rows, k) for k in ("characters", "structural")},
                  "vr2": meta["vr2"], "rows": meta["read"]}
    out["objective"] = {k: {"e9_ln_minus_e9_base": out["e9_ln"]["point"][k] - out["e9_base"]["point"][k],
                            "ci": ci(diffs[k])} for k in ("S", "R")}
    t = out["e9_ln"]["targets"]
    out["vr1"] = {r: {"qg3": t[f"{r} <QRY>"]["c"], "qg2d": QG2D_S0[label][r],
                      "pass": abs(t[f"{r} <QRY>"]["c"] - QG2D_S0[label][r]) <= VR1_TOL}
                  for r in ("host", "path")}
    return out


def r2(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


def r3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


def render(rep) -> None:
    print("### Validity\n")
    for label in Q.LABELS:
        x = rep[label]
        v1 = "; ".join(f"{r} {x['vr1'][r]['qg3']:.3f} against QG2-D {x['vr1'][r]['qg2d']:.3f}"
                       for r in ("host", "path"))
        v2 = max(x[m]["vr2"]["max_rel_prob_minus_positions"] for m in MODELS)
        e2 = max(x[m]["vr2"]["max_abs_events_minus_positions"] for m in MODELS)
        ok = all(v["pass"] for v in x["vr1"].values()) and e2 == 0 and v2 <= 1e-3
        cov = ", ".join(f"{k}: x{d['window']}" for k, d in x["e9_ln"]["targets"].items())
        print(f"- {label}: VR1 {v1}; VR2 events−positions {e2:.0f}, probability {v2:.1e}; "
              f"VR3 windows {cov}; pairs read {x['pairs']} {x['by_status']} → "
              f"**{'PASS' if ok else 'FAIL'}**")
    print("\n### The split of the query's shortfall (log scale; negative = under-predicted)\n")
    print("| class | model | L (the query) | R (rarity at matched frequency) | S (beyond rarity) | "
          "shares R / S | reading |\n|---|---|---|---|---|---|---|")
    for label in Q.LABELS:
        for m in MODELS:
            p, c = rep[label][m]["point"], rep[label][m]["ci"]
            f = lambda k: f"{r3(p.get(k))} [{r3(c[k][0])}, {r3(c[k][1])}]"
            print(f"| {label} | {m} | {f('L')} | {f('R')} | {f('S')} | "
                  f"{r2(p.get('share_R'))} / {r2(p.get('share_S'))} | **{rep[label][m]['reading']}** |")
    print("\n### The targets, and the uncontrolled outcomes at the same frequency\n")
    print("| class | model | pair | per-position frequency | c (model / data) | uncontrolled reference c "
          "(window, pairs, events) | structural-only reference c (pairs, events) |")
    print("|---|---|---|---|---|---|---|")
    for label in Q.LABELS:
        for m in MODELS:
            for k, d in rep[label][m]["targets"].items():
                print(f"| {label} | {m} | {k} | {100 * d['f']:.3f}% | {r3(d['c'])} | "
                      f"{r3(d['reference_c'])} (x{d['window']}, {d['reference_pairs']}, "
                      f"{d['reference_events']:,.0f}) | {r3(d['structural_reference_c'])} "
                      f"({d['structural_pairs']}, {d['structural_events']:,.0f}) |")
    print("\n### The uncontrolled reference curve (`e9_ln`): geometric-mean c by frequency bin\n")
    print("| class | kind | bin (per-position frequency from) | pairs | events | c |\n|---|---|---|---|---|---|")
    for label in Q.LABELS:
        for kind in ("characters", "structural"):
            for b in rep[label]["e9_ln"]["curve"][kind]:
                print(f"| {label} | {kind} | {b['bin']} | {b['pairs']} | {b['events']:,.0f} | {b['c']:.3f} |")
    print("\n### Constrained pairs (`e9_ln`; reported only)\n")
    print("| class | pair | frequency | c | events |\n|---|---|---|---|---|")
    for label in Q.LABELS:
        for k, d in rep[label]["e9_ln"]["constrained"].items():
            print(f"| {label} | {k} | {100 * d['f']:.3f}% | {r3(d['c'])} | {d['events']:,.0f} |")
    print("\n### The objective: `e9_ln` − `e9_base` (paired blocks)\n")
    print("| class | S difference | R difference |\n|---|---|---|")
    for label in Q.LABELS:
        o = rep[label]["objective"]
        print(f"| {label} | {r3(o['S']['e9_ln_minus_e9_base'])} [{r3(o['S']['ci'][0])}, {r3(o['S']['ci'][1])}] | "
              f"{r3(o['R']['e9_ln_minus_e9_base'])} [{r3(o['R']['ci'][0])}, {r3(o['R']['ci'][1])}] |")
    print(f"\n### Reading\n\n- benign, e9_ln (decides): **{rep['benign']['e9_ln']['reading']}**"
          f"\n- phishing, e9_ln: {rep['phishing']['e9_ln']['reading']}"
          f"\n- e9_base / E7 (benign): {rep['benign']['e9_base']['reading']} / {rep['benign']['e7']['reading']}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # a redirected Windows console is cp1252
    rep = {"registration": "research/qgap/PREREG_QG3.md"}
    for label in Q.LABELS:
        rep[label] = per_class(label)
    (Q.DATA / "qg3_report.json").write_text(json.dumps(rep, indent=1, default=float),
                                           encoding="utf-8")
    render(rep)


if __name__ == "__main__":
    main()
