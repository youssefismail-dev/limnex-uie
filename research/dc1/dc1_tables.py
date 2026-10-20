"""Render DC1's reports as the markdown tables of DC1_RESULTS.md, so no number is
transcribed by hand. Reads VERIFY.json, settings.json, score_report.json and
compare_report.json from --dc1.

    python research/dc1/dc1_tables.py --dc1 data/dc1 > data/dc1/tables.md
"""
import argparse
import ast
import json
from pathlib import Path

FIELDS = ("LBL", "SUB", "PTH", "QRY")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def f4(x):
    return "--" if x is None else f"{x:+.4f}"


def pv(p):
    return "--" if p is None else (f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}")


def cell_name(key: str) -> str:
    label, state, field = ast.literal_eval(key)
    return f"{label} {state or 'free'} {field}"


def verify(v) -> list:
    out = [f"**VERIFY: {v['verdict']}**", ""]
    for mode, r in v["identity"].items():
        out.append(f"- V3 identity, {mode}: {r['identical']}/{r['compared']}")
    for label, d in v.get("v4", {}).items():
        out.append(f"- V4 scorer, {label}: max |x - published| {max(d.values()):.1e}")
    out.append(f"- failures: {'; '.join(v['failures']) if v['failures'] else 'none'}")
    return out + [""]


def settings_table(s) -> list:
    out = ["| class | mode | " + " | ".join(FIELDS) + " |", "|---|---|" + "---|" * len(FIELDS)]
    for label, modes in s["classes"].items():
        for mode, cell in modes.items():
            row = []
            for f in FIELDS:
                r = cell.get(f, {})
                if "T" not in r:
                    row.append("n/a")
                    continue
                txt = f"T {r['T']:.3f}, d {r['delta']:+.3f}" + (f", w {r['w']:.2f}" if r.get("w") is not None else "")
                row.append(txt if r["applied"] else f"off ({txt})")
            out.append(f"| {label} | {mode} | " + " | ".join(row) + " |")
    return out + [""]


def targets(mode) -> list:
    out = ["| target | baseline | DC1 | Delta (DC1 - base) | 95% CI | Holm p | reading |",
           "|---|---|---|---|---|---|---|"]
    for k, r in mode["targets"].items():
        if "delta" not in r:
            out.append(f"| {cell_name(k)} | -- | -- | -- | -- | -- | NOT COMPUTABLE |")
            continue
        reading = ", ".join(x for x, on in (("IMPROVED", r["improved"]), ("FIXED", r["fixed"])) if on) or "--"
        out.append(f"| {cell_name(k)} | {r['v_base']} | {r['v_dc1']} | {f4(r['delta'])} "
                   f"| [{f4(r['lo'])}, {f4(r['hi'])}] | {pv(r['p_holm'])} | {reading} |")
    return out + [""]


def all_cells(mode) -> list:
    out = ["| cell | baseline | DC1 | Delta | 95% CI | Holm p (all cells) | pairing |",
           "|---|---|---|---|---|---|---|"]
    for k, r in mode["cells"].items():
        if not r:
            out.append(f"| {cell_name(k)} | -- | -- | -- | -- | -- | not computable |")
            continue
        out.append(f"| {cell_name(k)} | {r['v_base']} | {r['v_dc1']} | {f4(r['delta'])} "
                   f"| [{f4(r['lo'])}, {f4(r['hi'])}] | {pv(r['p_holm_all'])} "
                   f"| {'paired' if r['paired'] else 'UNPAIRED'} |")
    return out + [""]


def novelty_cost(mode, name) -> list:
    n = mode["novelty"]
    out = ["| | baseline | DC1 | worsening | 95% CI | counts as harm |", "|---|---|---|---|---|---|"]
    for key, label in (("whole_url_novelty", "whole-URL novelty"), ("query_copying", "query strings copied from training")):
        r = n[key]
        out.append(f"| {label} | {r['base']:.4f} | {r['dc1']:.4f} | {r['worsening']:+.4f} "
                   f"| [{r['lo']:+.4f}, {r['hi']:+.4f}] | {'YES' if r['worsened'] else 'no'} |")
    c = mode["cost"]
    out += ["", f"Cost, next-token predictions per delivered URL: ratio DC1 / baseline "
                f"**{c['ratio']:.2f}** (limit 2.00)."]
    if "per_state" in c:
        out += ["", "| state | baseline | DC1 | ratio |", "|---|---|---|---|"]
        for st, v in c["per_state"].items():
            out.append(f"| {st} | {v['base']:.1f} | {v['dc1']:.1f} | {v['dc1'] / v['base']:.2f} |")
    else:
        out.append(f"(baseline {c['base']:.1f}, DC1 {c['dc1']:.1f})")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc1", type=Path, required=True)
    args = ap.parse_args()
    comp = load(args.dc1 / "compare_report.json")
    lines = ["## Validity", ""] + verify(load(args.dc1 / "VERIFY.json"))
    lines += ["## Settings (applied only where validation confirmed them)", ""]
    lines += settings_table(load(args.dc1 / "settings.json"))
    for name, mode in comp["modes"].items():
        lines += [f"## FULL {name.upper()} -- {'ADOPT' if mode['adopt'] else 'NOT ADOPTED'}",
                  "", f"criteria: {mode['criteria']}", "", "### Targets", ""]
        lines += targets(mode)
        lines += ["### Every cell", ""] + all_cells(mode)
        lines += ["### Novelty and cost", ""] + novelty_cost(mode, name)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
