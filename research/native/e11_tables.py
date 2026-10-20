"""Render E11's reports as markdown tables, so no number is transcribed by hand.
E11_RESULTS.md carries this output verbatim as its appendix. Reads
data/e11/{VERIFY, score_report, compare_report}.json and the arms' training records.

    python research/native/e11_tables.py > data/e11/tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dc1"))

import e11_common as K  # noqa: E402
from e8_compare import GATED, rate  # noqa: E402

E11 = ROOT / "data" / "e11"
ARMS = tuple(K.ARMS)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def cell(row) -> str:
    a = (row or {}).get("acceptance", {})
    x = a.get("x")
    if x is None or x != x:
        return a.get("verdict", "—")
    return f"{x:+.4f} {a.get('verdict', '')[:4]}"


def pct(r, d=2) -> str:
    return "—" if r is None else f"{100 * r:.{d}f}%"


def row_check(x) -> str:
    if x["q"] is None:
        return "—"
    return (f"{pct(x['q'])} against {pct(x['p'], 3)} ({100 * x['diff']:+.2f}, ±{100 * x['T']:.2f}) "
            f"{'yes' if x['pass'] else '**no**'}")


def main() -> None:
    import argparse
    global E11
    ap = argparse.ArgumentParser()
    ap.add_argument("--e11", type=Path, default=E11)
    ap.add_argument("--train", type=Path, help="the training output (default: <e11>/train)")
    args = ap.parse_args()
    E11 = args.e11
    train = args.train or E11 / "train"
    sys.stdout.reconfigure(encoding="utf-8")   # a redirected Windows console is cp1252
    v, rep, res = load(E11 / "VERIFY.json"), load(E11 / "score_report.json"), load(E11 / "compare_report.json")
    print(f"### Validity\n\nVerdict: **{v['verdict']}**; failures: {v['failures'] or 'none'}\n")
    vid = v["v_id"]
    print(f"Shared start: {vid['starts']}; its part without <RS> equals E10's: {vid['base_equals_e10']}\n")
    print("| arm / mode | URLs regenerated | identical |\n|---|---|---|")
    for k, r in v.get("identity", {}).items():
        print(f"| {k} | {r['compared']} | {r['identical']} |")
    for label, diffs in v.get("v4", {}).items():
        print(f"\nV4 {label}: max |x − published| = {max(diffs.values()):.2e}")
    print("\n### Training\n\n| arm | final val (benign / phishing / mean) | best step | minutes | rows that fit | dropped |\n|---|---|---|---|---|---|")
    for arm in ARMS:
        r = load(train / arm / "train_result.json")
        h = min(r["history"], key=lambda x: x["mean"])
        f = r["final"]
        print(f"| {arm} | {f['benign']:.4f} / {f['phishing']:.4f} / {f['mean']:.4f} | {h['step']} | "
              f"{r['seconds'] / 60:.1f} | {r['counts']} | {r['dropped_over_max_len']:,} |")
    print("\n### Clipping after step 1,000 (K11e for the repair arms)\n\n| arm | band | steps | mean norm | clipped (own cap) | over 1.0 |\n|---|---|---|---|---|---|")
    for arm in ARMS:
        r = load(train / arm / "train_result.json")
        for k, b in (r.get("grad_diagnostics") or {}).items():
            if k.startswith("after_1000"):
                norm = "—" if b["mean_norm"] is None else f"{b['mean_norm']:.3f}"
                print(f"| {arm} | {k.split(' ', 1)[1]} | {b['steps']:,} | {norm} | "
                      f"{pct(b['clipped_share'], 1)} | {pct(b.get('over_early_cap_share'), 1)} |")
    print("\n### K11a — free-mode presence against the tolerance\n\n| arm | row | generated against real (difference, T) within |\n|---|---|---|")
    for arm in ARMS:
        for name, x in res["arms"][arm]["K11a"]["rows"].items():
            print(f"| {arm} | {name} | {row_check(x)} |")
    print("\n### K11b — the root slash against the tolerance\n\n| arm | row | generated against real (difference, T) within |\n|---|---|---|")
    for arm in ARMS:
        for name, x in res["arms"][arm]["K11b"]["rows"].items():
            print(f"| {arm} | {name} | {row_check(x)} |")
    print("\n### Gate v2, FULL FREE (x, verdict)\n\n| class | field | " + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS))
    for label in K.LABELS:
        for f in GATED + ("FULL",):
            print(f"| {label} | {f} | " + " | ".join(
                cell(rep["free"][a][label]["fields"].get(f)) for a in ARMS) + " |")
    print("\n### FULL CONTROL, measurable states (x, verdict)\n\n| state | field | " + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS))
    for key in sorted(rep["control"][ARMS[0]]):
        for f in GATED + ("FULL",):
            print(f"| {key} | {f} | " + " | ".join(
                cell(rep["control"][a][key]["fields"].get(f)) for a in ARMS) + " |")
    print("\n### K11c — no regression against each arm's reference\n\n| arm | reference | regressions |\n|---|---|---|")
    for arm in ARMS:
        c = res["arms"][arm]["K11c"]
        print(f"| {arm} | {c['reference']} | {c['regressions'] or 'none'} |")
    print("\n### Honouring (control, native)\n\n| state | " + " | ".join(ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for label in K.LABELS:
        for st in K.STATES:
            print(f"| {label} {st} | " + " | ".join(
                pct(rate(rep, "honouring", a, f"{label}_{st}")) for a in ARMS) + " |")
    print("\n### K11d — guardrails against each arm's reference\n\n| arm | mode | novel | copied queries |\n|---|---|---|---|")
    for arm in ARMS:
        for mode, c in rep["novelty"].get(arm, {}).items():
            print(f"| {arm} | {mode} | {pct(c['novel'] / c['n'])} | "
                  f"{pct(c['copied'] / c['q']) if c['q'] else '—'} of {c['q']:,} |")
    for arm in ARMS:
        d = res["arms"][arm]["K11d"]
        print(f"- {arm} against {d['reference']}: **{'PASS' if d['pass'] else 'FAIL'}** " + "; ".join(
            f"{k} {100 * x['worsening']:+.2f} pts [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]"
            for k, x in d["readings"].items()))
    pl = rep.get("plan", {})
    if pl:
        print("\n### Reported: e11_plan's own plan (checkpoint, shipped-weighted)\n\n| class | plan P | plan Q |\n|---|---|---|")
        for label, x in pl.items():
            print(f"| {label} | {pct(x['P'])} | {pct(x['Q'], 3)} |")
    print("\n### Attribution — each arm against its reference (reported)\n\n| arm | reference | emission change (points) |\n|---|---|---|")
    for arm in ARMS:
        a = res["arms"][arm]["attribution"]
        print(f"| {arm} | {a['reference']} | " + ", ".join(
            f"{k} {100 * x:+.2f}" for k, x in a["emission"].items()) + " |")
    print("\n### Outcome\n\n| arm | failing | candidate |\n|---|---|---|")
    for arm in ARMS:
        r = res["arms"][arm]
        print(f"| {arm} | {', '.join(r['failing']) or 'none'} | **{'yes' if r['candidate'] else 'no'}** |")


if __name__ == "__main__":
    main()
