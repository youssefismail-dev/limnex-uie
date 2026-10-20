"""Render E10's reports as markdown tables, so no number is transcribed by hand.
E10_RESULTS.md carries this output verbatim as its appendix. Reads
data/e10/{VERIFY, score_report, compare_report}.json and the arms' training
records.

    python research/native/e10_tables.py > data/e10/tables.md
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
sys.path.insert(0, str(HERE.parent / "dc1"))

import e10_common as K  # noqa: E402
from e8_compare import GATED, P_BAND, Q_BAND, rate  # noqa: E402

E10 = ROOT / "data" / "e10"
ARMS = tuple(K.ARMS)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def cell(row) -> str:
    a = (row or {}).get("acceptance", {})
    x = a.get("x")
    if x is None or x != x:
        return a.get("verdict", "—")
    return f"{x:+.4f} {a.get('verdict', '')[:4]}"


def pct(r) -> str:
    return "—" if r is None else f"{100 * r:.2f}%"


def main() -> None:
    import argparse
    global E10
    ap = argparse.ArgumentParser()
    ap.add_argument("--e10", type=Path, default=E10)
    E10 = ap.parse_args().e10
    sys.stdout.reconfigure(encoding="utf-8")   # a redirected Windows console is cp1252
    v, rep, res = load(E10 / "VERIFY.json"), load(E10 / "score_report.json"), load(E10 / "compare_report.json")
    print(f"### Validity\n\nVerdict: **{v['verdict']}**; failures: {v['failures'] or 'none'}\n")
    print(f"Shared start: {v['v_id']['starts']}; equals e9_ln's: {v['v_id']['start_equals_e9_ln']}\n")
    print("| arm / mode | URLs regenerated | identical |\n|---|---|---|")
    for k, r in v.get("identity", {}).items():
        print(f"| {k} | {r['compared']} | {r['identical']} |")
    for label, diffs in v.get("v4", {}).items():
        print(f"\nV4 {label}: max |x − published| = {max(diffs.values()):.2e}")
    print("\n### Training\n\n| arm | final val (benign / phishing / mean) | best step | minutes |\n|---|---|---|---|")
    for arm in ARMS:
        r = load(E10 / "train" / arm / "train_result.json")
        h = min(r["history"], key=lambda x: x["mean"])
        f = r["final"]
        print(f"| {arm} | {f['benign']:.4f} / {f['phishing']:.4f} / {f['mean']:.4f} | {h['step']} | "
              f"{r['seconds'] / 60:.1f} |")
    print("\n### K10a — the plan's own presence rates (shipped-weighted)\n")
    print("| class | plan P(path) | target | plan P(query) | target | ratio |\n|---|---|---|---|---|---|")
    for label, x in res["K10a"].get("readings", {}).items():
        print(f"| {label} | {pct(x['P'])} | {pct(x['P_target'])} | {pct(x['Q'])} | "
              f"{pct(x['Q_target'])} | {x['Q_ratio']:.3f} |")
    print(f"\nK10a: **{'PASS' if res['K10a']['pass'] else 'FAIL'}** {res['K10a']['failures'] or ''}\n")
    print("### K10b — free-mode emission\n\n| arm | benign PTH | benign QRY | phishing PTH | phishing QRY |\n|---|---|---|---|---|")
    for arm in ARMS:
        print(f"| {arm} | " + " | ".join(pct(rate(rep, "emission", arm, l, f))
                                         for l in K.LABELS for f in ("PTH", "QRY")) + " |")
    print(f"| bands | {100 * P_BAND['benign'][0]:.2f}–{100 * P_BAND['benign'][1]:.2f}% | "
          f"{100 * Q_BAND['benign'][0]:.2f}–{100 * Q_BAND['benign'][1]:.2f}% | "
          f"{100 * P_BAND['phishing'][0]:.2f}–{100 * P_BAND['phishing'][1]:.2f}% | "
          f"{100 * Q_BAND['phishing'][0]:.2f}–{100 * Q_BAND['phishing'][1]:.2f}% |")
    print(f"\nK10b: **{'PASS' if res['K10b']['pass'] else 'FAIL'}** {res['K10b']['failures'] or ''}\n")
    # Reported, not an E10 criterion: the owner adopted this standard after E10 was
    # registered (DECISION_2026-09-29 section 2). p = K9b's shipped-weighted target.
    tgt = load(ROOT / "data" / "native" / "emission_targets.json")
    print("### Reported: free-mode emission against the detectability tolerance "
          "(DECISION_2026-09-29 §2; not an E10 criterion)\n")
    print("| arm | class | field | generated | real | difference (points) | T (points) | within |"
          "\n|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for label in K.LABELS:
            for f in ("PTH", "QRY"):
                q = rate(rep, "emission", arm, label, f)
                p = tgt[label][f"{f}_achievable_under_shipped"]
                t = 2 * math.sqrt(p * (1 - p) / 8000)
                within = "—" if q is None else ("yes" if abs(q - p) <= t else "no")
                diff = "—" if q is None else f"{100 * (q - p):+.2f}"
                print(f"| {arm} | {label} | {f} | {pct(q)} | {pct(p)} | {diff} | ±{100 * t:.2f} | {within} |")
    print()
    print("### Gate v2, FULL FREE (x, verdict)\n\n| class | field | E7 (canonical) | "
          + " | ".join(ARMS) + " |\n|---|---|---|" + "---|" * len(ARMS))
    for label in K.LABELS:
        for f in GATED + ("FULL",):
            e7 = rep["e7_free"][label]["fields"].get(f)
            print(f"| {label} | {f} | {cell(e7)} | " + " | ".join(
                cell(rep["free"][a][label]["fields"].get(f)) for a in ARMS) + " |")
    print("\n### FULL CONTROL, measurable states (x, verdict)\n\n| state | field | "
          + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS))
    for key in sorted(rep["control"][ARMS[0]]):
        for f in GATED + ("FULL",):
            print(f"| {key} | {f} | " + " | ".join(
                cell(rep["control"][a][key]["fields"].get(f)) for a in ARMS) + " |")
    print(f"\nK10c: **{'PASS' if res['K10c']['pass'] else 'FAIL'}**; regressions: "
          f"{res['K10c']['regressions'] or 'none'}\n")
    print("### Honouring (control, native)\n\n| state | " + " | ".join(ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for label in K.LABELS:
        for st in K.STATES:
            print(f"| {label} {st} | " + " | ".join(
                pct(rate(rep, "honouring", a, f"{label}_{st}")) for a in ARMS) + " |")
    print("\n### Guardrails (K10d: e10_plan against e10_base)\n\n| arm | mode | novel | copied queries |\n|---|---|---|---|")
    for arm in ARMS:
        for mode, c in rep["novelty"].get(arm, {}).items():
            print(f"| {arm} | {mode} | {pct(c['novel'] / c['n'])} | "
                  f"{pct(c['copied'] / c['q']) if c['q'] else '—'} of {c['q']:,} |")
    d = res["K10d"]
    print(f"\nK10d: **{'PASS' if d['pass'] else 'FAIL'}** {d['failures'] or ''}")
    for k, x in d["readings"].items():
        print(f"- {k}: change {100 * x['worsening']:+.2f} pts [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]")
    pt = rep.get("plan_trace", {})
    if pt:
        print("\n### The plan's off-slot mass (mean per URL; generation records)\n")
        print("| arm | free | control |\n|---|---|---|")
        for arm in ARMS:
            print(f"| {arm} | {pt.get(arm, {}).get('free')} | {pt.get(arm, {}).get('control')} |")
    rp = res.get("reproducibility_e10_base_vs_e9_ln")
    if rp:
        print("\n### Reported: e10_base against e9_ln (same recipe, another run)\n")
        mx = rp["max_abs_free_cell_difference"]
        print(f"- largest free-cell x difference: {'—' if mx is None else f'{mx:.4f}'} "
              f"(within 0.0141: {rp['within_retrain_allowance']})")
        print("- emission difference: " + ", ".join(f"{k} {100 * x:+.2f} pts"
                                                  for k, x in rp["emission_difference"].items()))
        print("- honouring difference: " + ", ".join(f"{k} {100 * x:+.2f} pts"
                                                    for k, x in rp["honouring_difference"].items()))
    print(f"\n### Outcome\n\n- Candidate rule: **{res['candidate']}**")


if __name__ == "__main__":
    main()
