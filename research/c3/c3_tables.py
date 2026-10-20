"""Render C3-QRY's reports as the markdown tables of C3_RESULTS.md.

Reads only what the registered scripts wrote, so no number in the results is
transcribed by hand:

    data/c3/VERIFY.json           c3_verify.py       V1-V6, K1
    data/c3/F/report.json         c3_score.py        arm F's records
    data/sc1/T/cr1_report.json    (SC1, read only)   arm Q's records
    data/c3/c3a_report.json       c3_compare.py      C3-A
    data/c3/c3b_report.json       c3_ladder.py       C3-B
    data/c3/c3c_report.json       c3_describe.py     C3-C

    python research/c3/c3_tables.py --c3 data/c3 --sc1 data/sc1 > data/c3/tables.md

C3-C's query-shape table shows the statistics section 5 registers; the report's
hyphenated-value share is not registered and is not shown.
"""
import argparse
import json
from pathlib import Path

LABELS = ("benign", "phishing")
DESCRIPTIVE = ("LBL", "SUF", "SUB", "PTH", "FULL")
SHAPE = (("params_per_query", "params / query", "{:.3f}"),
         ("name_length", "name length", "{:.2f}"),
         ("value_length_p50", "value p50", "{:.1f}"),
         ("value_length_p90", "value p90", "{:.1f}"),
         ("percent_encoded_queries", "%-encoded", "{:.4f}"),
         ("numeric_values", "numeric values", "{:.3f}"),
         ("bare_flags", "bare flags", "{:.4f}"))


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def f4(x, sign=True):
    return "--" if x is None else (f"{x:+.4f}" if sign else f"{x:.4f}")


def pv(p):
    return "--" if p is None else (f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}")


def ci(r):
    return f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"


def acc(rep, tag, f):
    row = rep["states"].get(tag, {}).get("fields", {}).get(f) or {}
    a = row.get("acceptance", {})
    return a.get("x"), a.get("verdict")


def validity(v) -> list:
    out = ["## Validity (section 7) -- checked before any number was read", "",
           f"**VERIFY: {v['verdict']}**", "", "| check | result |", "|---|---|"]
    fails = v["failures"]
    for code, name in (("V1", "files"), ("V2", "queries"), ("V3", "identity"), ("V4", "table"),
                       ("V5", "arm Q"), ("V6", "S0'''")):
        mine = [f for f in fails if f.startswith(code)]
        detail = "; ".join(mine) if mine else "pass"
        if code == "V3":
            detail += " -- " + ", ".join(
                f"{l} {d['identical']}/{d['compared']}" if isinstance(d, dict) else f"{l} {d}"
                for l, d in v["identity"].items())
        out.append(f"| {code} {name} | {detail} |")
    other = [f for f in fails if not f[:2] in ("V1", "V2", "V3", "V4", "V5", "V6")]
    out.append(f"| K1 | {'; '.join(v['k1']) if v['k1'] else 'none'} |")
    if other:
        out.append(f"| other | {'; '.join(other)} |")
    return out + [""]


def c3a(a, q_rep, f_rep) -> list:
    out = ["## C3-A -- consistency (PRIMARY): QRY, Delta = x_Q - x_F", "",
           "| class | x_Q (gate verdict) | x_F (gate verdict) | Delta | 95% CI | p | Holm p "
           "| pairing | **reading** |", "|---|---|---|---|---|---|---|---|---|"]
    for label in LABELS:
        t = f"{label}_Q1"
        r = a["qry"][label]
        xq, vq = acc(q_rep, t, "QRY")
        xf, vf = acc(f_rep, t, "QRY")
        if "delta" not in r:
            out.append(f"| {label} | {f4(xq)} ({vq}) | {f4(xf)} ({vf}) | -- | -- | -- | -- | -- "
                       f"| **{r['reading']}** |")
            continue
        out.append(f"| {label} | {f4(xq)} ({vq}) | {f4(xf)} ({vf}) | {f4(r['delta'])} | {ci(r)} "
                   f"| {pv(r['p'])} | {pv(r['p_holm'])} | {'paired' if r['paired'] else 'UNPAIRED'} "
                   f"| **{r['reading']}** |")
    out += ["", "Descriptive, same scoring (Delta = x_Q - x_F; † unpaired, n differs):", "",
            "| class | " + " | ".join(DESCRIPTIVE) + " |", "|---|" + "---|" * len(DESCRIPTIVE)]
    for label in LABELS:
        c = a["classes"][label]
        if "k1" in c:
            out.append(f"| {label} | " + " | ".join("K1" for _ in DESCRIPTIVE) + " |")
            continue
        cells = []
        for f in DESCRIPTIVE:
            r = c["cells"].get(f)
            cells.append("n/c" if r is None else
                         f"{f4(r['delta'])}{'' if r['paired'] else '†'} {ci(r)}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")
    out += ["x per arm (gate acceptance, excess over the floor):", "",
            "| class | arm | " + " | ".join(DESCRIPTIVE) + " |", "|---|---|" + "---|" * len(DESCRIPTIVE)]
    for label in LABELS:
        for arm, rep in (("Q", q_rep), ("F", f_rep)):
            vals = [acc(rep, f"{label}_Q1", f) for f in DESCRIPTIVE]
            out.append(f"| {label} | {arm} | " + " | ".join(
                f"{f4(x)} {v or ''}".strip() for x, v in vals) + " |")
    return out + [""]


def c3b(b) -> list:
    out = [f"## C3-B -- where the tell lives (SECONDARY): Q3.1's ladder, D = {b['D']}, "
           f"t_crit {b['t_crit']}", ""]
    for label in LABELS:
        lq, lf = b["ladders"].get(f"{label}/Q", {}), b["ladders"].get(f"{label}/F", {})
        e5 = b["e5_published"].get(label, {})
        out += [f"### {label}", "",
                f"- ladder validity: Q {b['validity'][f'{label}/Q']}, F {b['validity'][f'{label}/F']}",
                f"- skeleton reading: Q **{b['shape'][f'{label}/Q']}**, F **{b['shape'][f'{label}/F']}**",
                f"- class reading (Q3.1's rule, the other mode as the second arm): "
                f"**{b['components'][label]['reading']}**"
                + (f"; identified {b['components'][label].get('identified')}"
                   if b['components'][label].get('identified') else "")
                + (f"; reduce in both modes {b['components'][label].get('reduce_in_both')}"
                   if b['components'][label].get('reduce_in_both') else ""), ""]
        if not lq and not lf:
            continue
        out += ["| rung | component | Q excess | Q R_T | Q t | Q verdict | F excess | F R_T "
                "| F t | F verdict | E5 R_T | E5 verdict |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name in list(dict.fromkeys(list(lq) + list(lf))):
            q, f, e = lq.get(name, {}), lf.get(name, {}), e5.get(name, {})
            comp = q.get("component") or f.get("component") or ""
            def cell(r):
                return (f"{f4(r.get('excess'))} | {r.get('R_T') if r.get('R_T') is not None else '--'} "
                        f"| {r.get('t') if r.get('t') is not None else '--'} | {r.get('verdict', '--')}")
            out.append(f"| {name} | {comp} | {cell(q)} | {cell(f)} "
                       f"| {e.get('R_T', '--')} | {e.get('verdict', '--')} |")
        out.append("")
    return out


def c3c(c) -> list:
    out = ["## C3-C -- emission and structure (DESCRIPTIVE)", "",
           "| class | F first-pass query rate | round-0 draws | E7, shipped table (HC1 C4) | real |",
           "|---|---|---|---|---|"]
    for label in LABELS:
        r = c["classes"][label]["query_rate"]
        out.append(f"| {label} | {f4(r['F_round0'], False)} | {r['round0_draws']:,} "
                   f"| {r['E7_free_shipped_table (HC1 C4)']:.4f} | {r['real']:.4f} |")
    out += ["", "Top-10 suffixes by draws, F first-pass query rate:", ""]
    for label in LABELS:
        top = c["classes"][label]["query_rate"]["top10_suffix_rate"]
        out.append(f"- {label}: " + ", ".join(f"`{s}` {v:.4f}" for s, v in top.items()))
    out += ["", "Path presence among query-bearing URLs (two-proportion 95% CIs):", "",
            "| class | real | Q | F | Q - F | Q - real | F - real |", "|---|---|---|---|---|---|---|"]
    for label in LABELS:
        p = c["classes"][label]["path_among_query_urls"]
        shares = [f"{p['share'][a]:.3f}" if a in p["share"] else "--" for a in ("real", "Q", "F")]
        diffs = [f"{f4(p[k]['delta'])} {ci(p[k])}" if k in p else "--"
                 for k in ("Q - F", "Q - real", "F - real")]
        out.append(f"| {label} | " + " | ".join(shares + diffs) + " |")
    out += ["", "Query shape (section 5's statistics, Q3.1's parser):", "",
            "| class | arm | queries | " + " | ".join(h for _, h, _ in SHAPE) + " |",
            "|---|---|---|" + "---|" * len(SHAPE)]
    for label in LABELS:
        for arm, s in c["classes"][label]["query_shape"].items():
            out.append(f"| {label} | {arm} | {s['queries']:,} | " + " | ".join(
                "--" if s[k] is None else fmt.format(s[k]) for k, _, fmt in SHAPE) + " |")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3", type=Path, required=True)
    ap.add_argument("--sc1", type=Path, required=True)
    args = ap.parse_args()
    lines = validity(load(args.c3 / "VERIFY.json"))
    lines += c3a(load(args.c3 / "c3a_report.json"), load(args.sc1 / "T" / "cr1_report.json"),
                 load(args.c3 / "F" / "report.json"))
    lines += c3b(load(args.c3 / "c3b_report.json"))
    lines += c3c(load(args.c3 / "c3c_report.json"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
