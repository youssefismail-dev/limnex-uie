"""The baseline question after E12 (owner, 2026-10-01): should E12 replace E11 as the
official research baseline? Read-only; every number comes from the registered
reports. The same-session control (`e12_base`, E11's recipe retrained beside the
E12 arms) separates what the E12 levers did from run-to-run noise.

    python research/native/e12_baseline_compare.py > data/e12/baseline_compare.md

Reads data/e11_final/score_report.json (E11's `e11_plan` at step 20,000),
data/e12/{score_report, compare_report, VERIFY}.json and data/qc1/copy.json.
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

from e8_compare import GATED, RETRAIN  # noqa: E402

LABELS = ("benign", "phishing")
STATES = ("P1Q1", "P1Q0", "P0Q1", "P0Q0")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def acc(rep, who, mode, key, f):
    a = (((rep[mode][who].get(key) or {}).get("fields") or {}).get(f) or {}).get("acceptance") or {}
    x = a.get("x")
    return (None if x is None or x != x else x), a.get("verdict"), a.get("se")


def z2(k1, n1, k2, n2):
    """Two-proportion z for p1 - p2 (pooled)."""
    if not n1 or not n2:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return ((k1 / n1) - (k2 / n2)) / se if se else float("nan")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    e11 = load(ROOT / "data/e11_final/score_report.json")
    e12 = load(ROOT / "data/e12/score_report.json")
    cmp12 = load(ROOT / "data/e12/compare_report.json")
    v12 = load(ROOT / "data/e12/VERIFY.json")
    qc1 = load(ROOT / "data/qc1/copy.json")
    who = (("E11 e11_plan", e11, "e11_plan"), ("e12_base", e12, "e12_base"), ("e12_final", e12, "e12_final"))

    print("### A. Gated cells (x, verdict)\n")
    print("| cell | E11 `e11_plan` | `e12_base` | `e12_final` | final − base | final − E11 |\n|---|---|---|---|---|---|")
    tally = {w[0]: {"CLEARED": 0, "UNDECIDED": 0, "CONVICTED": 0} for w in who}
    cells = [("free", l, l, f) for l in LABELS for f in GATED]
    cells += [("control", f"{l}_{s}", f"{l} {s}", f) for l in LABELS for s in STATES for f in GATED]
    for mode, key, name, f in cells:
        vals = [acc(rep, w, mode, key, f) for _, rep, w in who]
        if all(v[0] is None for v in vals):
            continue
        for (lab, _, _), (x, v, _) in zip(who, vals):
            if v in tally[lab]:
                tally[lab][v] += 1
        fmt = lambda t: "—" if t[0] is None else f"{t[0]:+.4f} {t[1][:4]}"
        d1 = "—" if vals[2][0] is None or vals[1][0] is None else f"{vals[2][0] - vals[1][0]:+.4f}"
        d2 = "—" if vals[2][0] is None or vals[0][0] is None else f"{vals[2][0] - vals[0][0]:+.4f}"
        print(f"| {mode} {name} {f} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {d1} | {d2} |")
    print("\n| | CLEARED | UNDECIDED | CONVICTED |\n|---|---|---|---|")
    for lab, t in tally.items():
        print(f"| {lab} | {t['CLEARED']} | {t['UNDECIDED']} | {t['CONVICTED']} |")

    print(f"\nRetrain allowance {RETRAIN}. Same-session regressions (K12c): "
          f"{cmp12['arms']['e12_final']['K12c']['regressions'] or 'none'}.\n")

    print("### B. The slash rows (rate; generated − real, against ±T)\n")
    print("| row | E11 `e11_plan` | `e12_base` | `e12_final` |\n|---|---|---|---|")
    rows12 = {a: cmp12["arms"][a]["K12b"]["rows"] for a in ("e12_base", "e12_final")}
    e11_targets = load(ROOT / "data/e11/targets.json")
    from e11_compare import check
    import e12_common as K
    for name in rows12["e12_final"]:
        label, rest = name.split(" ", 1)
        if rest.startswith("control"):
            st = rest.split()[1]
            r11 = (e11["slash"]["e11_plan"].get(f"control_{label}_{st}") or {}).get("rate")
            t = e11_targets[label][f"slash_{st}"]
            c11 = check(r11, t["control"], K.n_ctrl(label, st))
        else:
            key = rest
            st = "P0Q0" if key == "free_bare" else "P0Q1"
            r11 = (e11["slash"]["e11_plan"].get(f"{key}_{label}") or {}).get("rate")
            t = e11_targets[label][f"slash_{st}"]
            c11 = check(r11, t["free"], t["free_n_per_generation"])
        fmt = lambda c: f"{100 * c['q']:.2f}% ({100 * c['diff']:+.2f} / ±{100 * c['T']:.2f}) {'ok' if c['pass'] else '**miss**'}"
        print(f"| {name} | {fmt(c11)} | {fmt(rows12['e12_base'][name])} | {fmt(rows12['e12_final'][name])} |")

    print("\n### C. Honouring the requested state (control, native)\n")
    print("| state | E11 `e11_plan` | `e12_base` | `e12_final` | z (final − base) |\n|---|---|---|---|---|")
    for l in LABELS:
        for s in STATES:
            h = [rep["honouring"][w][f"{l}_{s}"] for _, rep, w in who]
            z = z2(h[2]["k"], h[2]["n"], h[1]["k"], h[1]["n"])
            print(f"| {l} {s} | " + " | ".join(f"{100 * x['rate']:.2f}%" for x in h) + f" | {z:+.1f} |")

    print("\n### D. Copying and novelty\n")
    print("| arm | mode | novel | copied queries | z copied (vs `e12_base`) |\n|---|---|---|---|---|")
    base = e12["novelty"]["e12_base"]
    for lab, rep, w in who:
        for mode, c in rep["novelty"][w].items():
            z = "" if w == "e12_base" or rep is e11 else f"{z2(c['copied'], c['q'], base[mode]['copied'], base[mode]['q']):+.1f}"
            print(f"| {lab} | {mode} | {100 * c['novel'] / c['n']:.2f}% | {100 * c['copied'] / c['q']:.2f}% of {c['q']:,} | {z} |")
    print("\nQC1-1 cells, copy rate and the 20+ band (common training strings) against the natural rate:\n")
    print("| cell | natural | E11 `e11_plan` | `e12_base` | `e12_final` |\n|---|---|---|---|---|")
    for cell, node in e12["copy"].items():
        nat = node["real"]
        e11c = qc1[cell]["e11_plan"]["generated"]
        b, f_ = node["e12_base"]["generated"], node["e12_final"]["generated"]
        fmt = lambda g: f"{100 * g['rate']:.2f}% · 20+ {100 * g['bands']['20+']:.2f}%"
        print(f"| {cell} | {fmt(nat)} | {fmt(e11c)} | {fmt(b)} | {fmt(f_)} |")

    print("\n### E. S obedience (slash arms only)\n")
    print("| population | S = 0 | S = 1 |\n|---|---|---|")
    for pop, r in e12["s_honouring"]["e12_final"].items():
        cells_ = [f"{100 * r[s]['rate']:.3f}% of {r[s]['n']:,}" if s in r else "—" for s in ("S0", "S1")]
        print(f"| {pop} | " + " | ".join(cells_) + " |")

    print("\n### F. Validity and reproducibility\n")
    print(f"- E12 VERIFY: {v12['verdict']}; byte identity " + ", ".join(
        f"{k} {r['identical']}/{r['compared']}" for k, r in v12["identity"].items()) + ".")
    sp = cmp12["retrain_spread"]
    print(f"- Run-to-run spread (`e12_base` against E11's `e11_plan`, the same recipe): at most "
          f"{max(abs(x) for x in sp['free_x'].values()):.4f} on free cells and "
          f"{max(abs(x) for x in sp['control_x'].values()):.4f} on control cells.")
    print("- K12 for `e12_final`: " + ", ".join(
        f"{k} {'PASS' if cmp12['arms']['e12_final'][k]['pass'] else 'FAIL'}" for k in ("K12a", "K12b", "K12c", "K12d", "K12e")))


if __name__ == "__main__":
    main()
