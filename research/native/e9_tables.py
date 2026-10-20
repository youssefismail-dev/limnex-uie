"""Render E9's reports as markdown tables, so no number is transcribed by hand.
E9_RESULTS.md carries this output verbatim as its Appendix A, and
e9_check_results.py checks the text's numbers against it. Reads data/e9/{VERIFY, score_report,
compare_report, mechanism_*}.json and the two arms' training records; the
retrain-spread table also reads E8's already-reported baseline figures
(data/e8/score_report.json: e8_base only -- E8's arms stay unread).

    python research/native/e9_tables.py > data/e9/tables.md
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

import e9_common as K  # noqa: E402
from e8_compare import GATED, P_BAND, Q_BAND, free_cells, rate  # noqa: E402

E9 = ROOT / "data" / "e9"
E8 = ROOT / "data" / "e8"
CONDS_FREE = [f"{a}_{d}" for a, d in K.FREE_CONDITIONS]
CONDS_CTRL = [f"{a}_{d}" for a, d in K.CTRL_CONDITIONS]


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


def validity(v: dict) -> None:
    print("### Validity\n")
    print(f"Verdict: **{v['verdict']}**; failures: {v['failures'] or 'none'}\n")
    print("| condition | URLs regenerated | identical |\n|---|---|---|")
    for k, r in v.get("identity", {}).items():
        print(f"| {k} | {r['compared']} | {r['identical']} |")
    for label, diffs in v.get("v4", {}).items():
        print(f"\nV4 {label}: max |x − published| = {max(diffs.values()):.2e}")
    print()


def training() -> None:
    print("### Training\n")
    print("| arm | objective | final val (benign / phishing / mean) | best step | minutes | start |")
    print("|---|---|---|---|---|---|")
    recs = {}
    for arm in K.ARMS:
        r = load(E9 / "train" / arm / "train_result.json")
        recs[arm] = r
        h = min(r["history"], key=lambda x: x["mean"])
        f = r["final"]
        print(f"| {arm} | {r['loss_norm']} | {f['benign']:.4f} / {f['phishing']:.4f} / "
              f"{f['mean']:.4f} | {h['step']} | {r['seconds'] / 60:.1f} | "
              f"`{r['init_weights_sha256'][:12]}` |")
    print("\nGradient clipping after step 1,000, by scored tokens per row (share of steps clipped):\n")
    print("| band | " + " | ".join(K.ARMS) + " |\n|---|" + "---|" * len(K.ARMS))
    bands = sorted(k for k in recs["e9_base"]["grad_diagnostics"] if k.startswith("after_1000"))
    for b in bands:
        vals = []
        for arm in K.ARMS:
            d = recs[arm]["grad_diagnostics"].get(b, {})
            vals.append(f"{100 * d.get('clipped_share', 0):.1f}% (norm {d.get('mean_norm') or 0:.2f})")
        print(f"| {b.replace('after_1000 ', '')} | " + " | ".join(vals) + " |")
    print()


def mechanism(res: dict) -> None:
    print("### R1 — the mechanism (teacher-forced calibration; 1.0 = the data's own rate)\n")
    mech = {t: load(E9 / f"mechanism_{t}.json") for t in ("e7", "e9_base", "e9_ln")}
    for cls in ("benign", "phishing"):
        print(f"**{cls}**\n")
        print("| family | E7 plain | e9_base plain | e9_ln plain [95% CI] | e9_ln weighted | closure |")
        print("|---|---|---|---|---|---|")
        for fam in ("F1", "F2", "F3", "TAIL"):
            e7 = mech["e7"][cls][fam]["plain"]["ratio"]
            b = mech["e9_base"][cls][fam]["plain"]["ratio"]
            ln = mech["e9_ln"][cls][fam]["plain"]
            w = mech["e9_ln"][cls][fam]["weighted"]["ratio"]
            clo = 1 - abs(ln["ratio"] - 1) / abs(b - 1) if b != 1 else float("nan")
            print(f"| {fam} | {e7:.3f} | {b:.3f} | {ln['ratio']:.3f} [{ln['lo']:.3f}, "
                  f"{ln['hi']:.3f}] | {w:.3f} | {clo:.2f} |")
        print()
    print(f"P0: **{'PASS' if res['P0']['pass'] else 'FAIL'}**. K9a: "
          f"**{'PASS' if res['K9a']['pass'] else 'FAIL'}** {res['K9a']['failures'] or ''}\n")


def emission(rep: dict, res: dict) -> None:
    print("### R2 — free-mode emission (K9b reads e9_ln_raw)\n")
    print("| condition | benign PTH | benign QRY | phishing PTH | phishing QRY |\n|---|---|---|---|---|")
    for who in CONDS_FREE:
        print(f"| {who} | " + " | ".join(pct(rate(rep, "emission", who, l, f))
                                         for l in K.LABELS for f in ("PTH", "QRY")) + " |")
    print(f"| bands | {100 * P_BAND['benign'][0]:.2f}–{100 * P_BAND['benign'][1]:.2f}% | "
          f"{100 * Q_BAND['benign'][0]:.2f}–{100 * Q_BAND['benign'][1]:.2f}% | "
          f"{100 * P_BAND['phishing'][0]:.2f}–{100 * P_BAND['phishing'][1]:.2f}% | "
          f"{100 * Q_BAND['phishing'][0]:.2f}–{100 * Q_BAND['phishing'][1]:.2f}% |")
    print(f"\nK9b: **{'PASS' if res['K9b']['pass'] else 'FAIL'}** {res['K9b']['failures'] or ''}\n")


def gate(rep: dict, res: dict) -> None:
    print("### R3 — gate v2, FULL FREE (x, verdict)\n")
    print("| class | field | E7 (canonical) | " + " | ".join(CONDS_FREE) + " |")
    print("|---|---|---|" + "---|" * len(CONDS_FREE))
    for label in K.LABELS:
        for f in GATED + ("FULL",):
            e7 = rep["e7_free"][label]["fields"].get(f)
            vals = [cell(rep["free"][w][label]["fields"].get(f)) for w in CONDS_FREE]
            print(f"| {label} | {f} | {cell(e7)} | " + " | ".join(vals) + " |")
    print("\n### R3 — FULL CONTROL, measurable states (x, verdict)\n")
    print("| state | field | " + " | ".join(CONDS_CTRL) + " |\n|---|---|" + "---|" * len(CONDS_CTRL))
    keys = sorted(rep["control"][CONDS_CTRL[0]])
    for key in keys:
        for f in GATED + ("FULL",):
            vals = [cell(rep["control"][w][key]["fields"].get(f)) for w in CONDS_CTRL]
            print(f"| {key} | {f} | " + " | ".join(vals) + " |")
    k = res["K9c"]
    print(f"\nK9c (against today's system, e9_base_resting): **{'PASS' if k['pass'] else 'FAIL'}**; "
          f"regressions: {k['regressions'] or 'none'}")
    print(f"Attribution (against e9_base_raw, reported): {k['attribution_regressions'] or 'none'}\n")


def honouring(rep: dict) -> None:
    print("### R4 — honouring (control, native)\n")
    print("| state | " + " | ".join(CONDS_CTRL) + " |\n|---|" + "---|" * len(CONDS_CTRL))
    for label in K.LABELS:
        for st in K.STATES:
            print(f"| {label} {st} | " + " | ".join(
                pct(rate(rep, "honouring", w, f"{label}_{st}")) for w in CONDS_CTRL) + " |")
    print()


def guardrails(rep: dict, res: dict) -> None:
    print("### R6 — guardrails (K9d: e9_ln_raw against e9_base_resting)\n")
    print("| condition | mode | novel | copied queries |\n|---|---|---|---|")
    for who in dict.fromkeys(CONDS_FREE + CONDS_CTRL):
        for mode, c in rep["novelty"].get(who, {}).items():
            print(f"| {who} | {mode} | {pct(c['novel'] / c['n'])} | "
                  f"{pct(c['copied'] / c['q']) if c['q'] else '—'} of {c['q']:,} |")
    d = res["K9d"]
    print(f"\nK9d: **{'PASS' if d['pass'] else 'FAIL'}** {d['failures'] or ''}")
    for k, v in d["readings"].items():
        print(f"- {k}: change {100 * v['worsening']:+.2f} pts [{100 * v['lo']:+.2f}, "
              f"{100 * v['hi']:+.2f}]")
    print()


def qry(rep: dict) -> None:
    print("### R5 — the QRY cell against validation (Q-VAL) and training (Q-TRAIN) queries\n")
    print("| condition | class | Q-VAL | Q-TRAIN | n |\n|---|---|---|---|---|")
    for who, parts in rep.get("qry", {}).items():
        for label, r in parts.items():
            print(f"| {who} | {label} | {cell(r['q_val'])} | {cell(r['q_train'])} | "
                  f"{r['q_val'].get('n', '—')} |")
    print()


def spread(rep: dict) -> None:
    """Three independent trainings of E7's recipe: E7, e8_base, e9_base (resting)."""
    print("### Retrain spread — three trainings of E7's recipe (reported, not a criterion)\n")
    e8 = load(E8 / "score_report.json")
    runs = {"E7": (e8, "e7"), "e8_base": (e8, "e8_base"), "e9_base": (rep, "e9_base_resting")}
    print("| reading | E7 | e8_base | e9_base | max − min |\n|---|---|---|---|---|")
    for label in K.LABELS:
        for st in K.STATES:
            vals = [rate(r, "honouring", w, f"{label}_{st}") for r, w in runs.values()]
            if None in vals:
                continue
            print(f"| honour {label} {st} | " + " | ".join(pct(v) for v in vals)
                  + f" | {100 * (max(vals) - min(vals)):.2f} pts |")
        for f in ("PTH", "QRY"):
            vals = [rate(r, "emission", w, label, f) for r, w in runs.values()]
            if None in vals:
                continue
            print(f"| emission {label} {f} | " + " | ".join(pct(v) for v in vals)
                  + f" | {100 * (max(vals) - min(vals)):.2f} pts |")
    e7_cells = free_cells({"e7_free": load(E8 / "e7_free_fields.json")}, "e7")
    b8 = free_cells(e8, "e8_base")
    b9 = free_cells(rep, "e9_base_resting")
    for key in sorted(set(e7_cells) & set(b8) & set(b9)):
        xs = [e7_cells[key][0], b8[key][0], b9[key][0]]
        print(f"| free {key[0]} {key[1]} x | " + " | ".join(f"{x:+.4f}" for x in xs)
              + f" | {max(xs) - min(xs):.4f} |")
    print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # a redirected Windows console is cp1252
    v = load(E9 / "VERIFY.json")
    rep = load(E9 / "score_report.json")
    res = load(E9 / "compare_report.json")
    validity(v)
    training()
    mechanism(res)
    emission(rep, res)
    gate(rep, res)
    honouring(rep)
    guardrails(rep, res)
    qry(rep)
    spread(rep)
    print(f"### Outcome\n\n- Candidate rule: **{res['candidate']}**\n- Section 7: **{res['section7']}**")


if __name__ == "__main__":
    main()
