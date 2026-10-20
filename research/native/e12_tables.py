"""Render E12's reports as markdown tables, so no number is transcribed by hand.
E12_RESULTS.md carries this output verbatim as its appendix. Reads
data/e12/{VERIFY, score_report, compare_report}.json and the arms' training records.

    python research/native/e12_tables.py > data/e12/tables.md
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

import e12_common as K  # noqa: E402
from e8_compare import GATED, rate  # noqa: E402

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


def num(x, d=4) -> str:
    return "—" if x is None or x != x else f"{x:+.{d}f}"


def row_check(x) -> str:
    if x["q"] is None:
        return "—"
    return (f"{pct(x['q'])} against {pct(x['p'], 3)} ({100 * x['diff']:+.2f}, ±{100 * x['T']:.2f}) "
            f"{'yes' if x['pass'] else '**no**'}")


def detail(d) -> str:
    """A prediction's evidence as table-safe text: floats to 4 places, no pipes."""
    if isinstance(d, float):
        return f"{d:+.4f}"
    if isinstance(d, dict):
        return "; ".join(f"{k}: {detail(v)}" for k, v in d.items())
    if isinstance(d, list):
        return ", ".join(map(detail, d)) or "none"
    return str(d).replace("|", "/")


def difference_tables(d: dict) -> None:
    print("Emission (points): " + ", ".join(f"{k} {100 * x:+.2f}" for k, x in d["emission"].items()))
    for part, title in (("free_x", "free cell"), ("control_x", "control cell")):
        print(f"\n| {title} | x difference |\n|---|---|")
        for k, x in d[part].items():
            print(f"| {k} | {x:+.4f} |")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--e12", type=Path, default=ROOT / "data" / "e12")
    args = ap.parse_args()
    e12 = args.e12
    sys.stdout.reconfigure(encoding="utf-8")   # a redirected Windows console is cp1252
    v, rep, res = load(e12 / "VERIFY.json"), load(e12 / "score_report.json"), load(e12 / "compare_report.json")
    if res.get("dry"):
        print("**DRY RUN: a pipeline test on stand-in numbers. NOT A RESULT; never read.**\n")
    print(f"### Validity\n\nVerdict: **{v['verdict']}**; failures: {v['failures'] or 'none'}\n")
    print("| arm | E10's start | E11's start | new token rows | new positions |\n|---|---|---|---|---|")
    for arm, s in v["v_id"].items():
        print(f"| {arm} | {str(s['e10_start_sha256'])[:12]}… | {str(s['e11_start_sha256'])[:12]}… | "
              f"{str(s['new_token_rows_sha256'] or '—')[:12]} | {str(s['new_position_rows_sha256'] or '—')[:12]} |")
    print("\n| arm / mode | URLs regenerated | identical |\n|---|---|---|")
    for k, r in v.get("identity", {}).items():
        print(f"| {k} | {r['compared']} | {r['identical']} |")
    for label, diffs in v.get("v4", {}).items():
        print(f"\nV4 {label}: max |x − published| = {max(diffs.values()):.2e}")

    print("\n### Training (read at `final.pt`, step 20,000)\n\n| arm | final val (benign / phishing / mean) "
          "| lowest mean val at step | minutes | rows that fit | dropped |\n|---|---|---|---|---|---|")
    for arm in ARMS:
        r = load(e12 / "train" / arm / "train_result.json")
        h = min(r["history"], key=lambda x: x["mean"])
        f = r["final"]
        print(f"| {arm} | {f['benign']:.4f} / {f['phishing']:.4f} / {f['mean']:.4f} | {h['step']:,} | "
              f"{r['seconds'] / 60:.1f} | {r['counts']} | {r['dropped_over_max_len']:,} |")
    print("\n### K12e — clipping after step 1,000\n\n| arm | band | steps | mean norm | clipped "
          "| non-finite steps |\n|---|---|---|---|---|---|")
    for arm in ARMS:
        for k, b in res["arms"][arm]["K12e"]["bands"].items():
            norm = "—" if b.get("mean_norm") is None else f"{b['mean_norm']:.3f}"
            print(f"| {arm} | {k.split(' ', 1)[1]} | {b['steps']:,} | {norm} | {pct(b['clipped_share'], 1)} "
                  f"| {b.get('nonfinite', '—')} |")

    for code, title in (("K12a", "free-mode presence against the tolerance"),
                        ("K12b", "the root slash against the tolerance")):
        print(f"\n### {code} — {title}\n\n| arm | row | generated against real (difference, T) within |"
              "\n|---|---|---|")
        for arm in ARMS:
            for name, x in res["arms"][arm][code]["rows"].items():
                print(f"| {arm} | {name} | {row_check(x)} |")
    print("\n### S honouring (slash arms: generated URLs that obey their supplied S)\n\n"
          "| arm | population | S = 0 | S = 1 |\n|---|---|---|---|")
    for arm, pops in rep.get("s_honouring", {}).items():
        for pop, r in pops.items():
            print(f"| {arm} | {pop} | " + " | ".join(
                f"{pct(r[s]['rate'], 3)} of {r[s]['n']:,}" if s in r else "—" for s in ("S0", "S1")) + " |")

    print("\n### Gate v2, FULL FREE (x, verdict)\n\n| class | field | " + " | ".join(ARMS)
          + " |\n|---|---|" + "---|" * len(ARMS))
    for label in K.LABELS:
        for f in GATED + ("FULL",):
            print(f"| {label} | {f} | " + " | ".join(
                cell(rep["free"][a][label]["fields"].get(f)) for a in ARMS) + " |")
    print("\n### FULL CONTROL, measurable states (x, verdict)\n\n| state | field | " + " | ".join(ARMS)
          + " |\n|---|---|" + "---|" * len(ARMS))
    for key in sorted(rep["control"][ARMS[0]]):
        for f in GATED + ("FULL",):
            print(f"| {key} | {f} | " + " | ".join(
                cell(rep["control"][a][key]["fields"].get(f)) for a in ARMS) + " |")
    print("\n### Honouring (control, native)\n\n| state | " + " | ".join(ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for label in K.LABELS:
        for st in K.STATES:
            print(f"| {label} {st} | " + " | ".join(
                pct(rate(rep, "honouring", a, f"{label}_{st}")) for a in ARMS) + " |")

    print("\n### K12c — no regression against `e12_base`\n\n| arm | regressions |\n|---|---|")
    for arm in ARMS:
        c = res["arms"][arm]["K12c"]
        print(f"| {arm} | {'— (the reference)' if c is None else (c['regressions'] or 'none')} |")
    print("\n### K12d — guardrails\n\n| arm | mode | novel | copied queries |\n|---|---|---|---|")
    for arm in ARMS:
        for mode, c in rep["novelty"].get(arm, {}).items():
            print(f"| {arm} | {mode} | {pct(c['novel'] / c['n'])} | "
                  f"{pct(c['copied'] / c['q']) if c['q'] else '—'} of {c['q']:,} |")
    print("\nMemorisation (QC1-1's instrument; the host-level reference decides, the "
          "domain-disjoint one is reported):\n")
    print("| arm | cell | reference | copied, generated / natural (±T) | reading | seen once in training, "
          "generated / natural | reading |\n|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for cellname, row in res["arms"][arm]["K12d"]["memorisation"].items():
            for ref, m in row.items():
                print(f"| {arm} | {cellname} | {ref} | {pct(m['generated'])} / {pct(m['natural'])} "
                      f"(±{100 * m['T']:.2f}) | {m['rate']} | {pct(m['singleton_generated'])} / "
                      f"{pct(m['singleton_natural'])} | {m['singleton']} |")
    print("\nThe band profile (share of the cell's queries matching a training string seen that often):\n")
    print("| cell | who | 1 | 2–4 | 5–19 | 20+ |\n|---|---|---|---|---|---|")
    for cellname, node in rep.get("copy", {}).items():
        for who in ("real", "real_domain_disjoint") + ARMS:
            if who in node:
                bands = (node[who]["generated"] if who in ARMS else node[who])["bands"]
                print(f"| {cellname} | {who} | " + " | ".join(pct(bands[b]) for b in ("1", "2-4", "5-19", "20+")) + " |")
    print()
    for arm in ARMS:
        d = res["arms"][arm]["K12d"]
        nov = "; ".join(f"{k} novelty {100 * x['worsening']:+.2f} pts [{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]"
                        for k, x in d["novelty"].items())
        print(f"- {arm}: **{'PASS' if d['pass'] else 'FAIL'}**" + (f"; {nov}" if nov else "")
              + (f"; failures {d['failures']}" if d["failures"] else ""))

    print("\n### QC1-2 on `e12_final` (reported: one arm, not replicated, licenses nothing)\n\n"
          "| cell | n | x plain | x at the natural share | rho | t | reading | void seeds |\n"
          "|---|---|---|---|---|---|---|---|")
    for name, r in res.get("qc1_2", {}).items():
        if r is None:
            print(f"| {name} | — | — | — | — | — | not measured | — |")
        elif "x_plain" not in r:
            print(f"| {name} | {r['n']} | — | — | — | — | {r['reading']} | {r['void']} |")
        else:
            print(f"| {name} | {r['n']} | {num(r['x_plain'])} | {num(r['x_mix'])} | {r['rho']:.2f} | "
                  f"{r['t']:+.2f} | {r['reading']} | {r['void']} |")
    for name, r in res.get("qc1_2", {}).items():
        if r and "x_plain" in r:
            extra = (f"; at the domain-disjoint share {num(r['x_mix_domain_disjoint'])}"
                     if "x_mix_domain_disjoint" in r else "")
            print(f"- {name}: never-seen {num(r['x_never_seen']) if r['x_never_seen'] is not None else 'insufficient'}; "
                  f"found {num(r['x_found']) if r['x_found'] is not None else 'insufficient'}{extra}")

    print("\n### Leave-one-out: R1 (`e12_final` against `e12_no_r1`)\n")
    difference_tables(res["leave_one_out"]["R1"])
    print("\n### The slash plan's effect (`e12_no_r1` against `e12_base`; reported, no arm ablates it)\n")
    difference_tables(res["slash_plan_effect"])
    print("\n### Run-to-run spread (`e12_base` against E11's `e11_plan`, both at step 20,000)\n")
    sp = res["retrain_spread"]
    for part, what in (("free_x", "free"), ("control_x", "control")):
        xs = [abs(x) for x in sp[part].values()]
        if xs:
            print(f"Largest |x difference| over {len(xs)} {what} cells: {max(xs):.4f}\n")
    difference_tables(sp)

    print("\n### Section 3's predictions, scored\n\n| prediction | held | detail |\n|---|---|---|")
    for p in res.get("predictions", []):
        held = "n/a" if p["held"] is None else ("yes" if p["held"] else "**no**")
        print(f"| {p['prediction']} | {held} | {detail(p['detail'])} |")

    print("\n### Outcome\n\n| arm | failing | candidate | E3 finish line |\n|---|---|---|---|")
    for arm in ARMS:
        r = res["arms"][arm]
        fl = r["finish_line"]
        line = ("MET" if fl["met"] else f"not met: {len(fl['cells_not_cleared'])} cells, "
                f"{len(fl['slash_rows_missed'])} slash rows")
        print(f"| {arm} | {', '.join(r['failing']) or 'none'} | **{'yes' if r['candidate'] else 'no'}** | {line} |")
    fl = res["arms"][K.CANDIDATE]["finish_line"]
    print(f"\n**E3 for `{K.CANDIDATE}`: {'CLAIMED' if res['E3']['claimed'] else 'NOT MET'}.**")
    if fl["cells_not_cleared"] or fl["slash_rows_missed"]:
        print("\nWhat stands between it and E3:\n")
        for c in fl["cells_not_cleared"]:
            print(f"- {c['cell']}: {num(c['x'])} {c['verdict']}")
        for s in fl["slash_rows_missed"]:
            print(f"- slash row {s}")


if __name__ == "__main__":
    main()
