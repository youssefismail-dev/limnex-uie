"""Render E8 (N1)'s reports as the markdown tables of E8_RESULTS.md, so no number
is transcribed by hand. Reads data/e8/{VERIFY,score_report,compare_report}.json.

    python research/native/e8_tables.py > data/e8/tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import e8_common as K  # noqa: E402

E8 = ROOT / "data" / "e8"
WHO = ("e7", "e8_base", "e8_rs", "e8_pf", "e8_pfrs")
GATED = ("LBL", "SUF", "SUB", "PTH", "QRY", "FULL")


def load(name):
    return json.loads((E8 / name).read_text(encoding="utf-8"))


def pct(r):
    return "--" if r is None else f"{100 * r:.2f}%"


def cell(row):
    a = (row or {}).get("acceptance", {})
    x = a.get("x")
    if x is None or x != x:
        return a.get("verdict", "--") or "--"
    return f"{x:+.4f} {a.get('verdict', '')}".strip()


def verify(v) -> list:
    out = [f"**VERIFY: {v['verdict']}**", ""]
    for key, r in v.get("identity", {}).items():
        out.append(f"- V2 identity, {key}: {r['identical']}/{r['compared']}")
    for label, d in v.get("v4", {}).items():
        out.append(f"- V4 scorer, {label}: max |x - published| {max(d.values()):.1e}")
    out.append(f"- failures: {'; '.join(v['failures']) if v['failures'] else 'none'}")
    return out + [""]


def honouring(rep) -> list:
    out = ["| state | " + " | ".join(WHO) + " |", "|---|" + "---|" * len(WHO)]
    for label in K.LABELS:
        for st in K.STATES:
            key = f"{label}_{st}"
            row = []
            for w in WHO:
                r = rep.get("honouring", {}).get(w, {}).get(key)
                row.append("--" if not r else f"{pct(r['rate'])} [{pct(r['lo'])}, {pct(r['hi'])}]")
            tag = " (honouring only)" if (label, st) in K.UNMEASURABLE else ""
            out.append(f"| {label} {st}{tag} | " + " | ".join(row) + " |")
    return out + [""]


def emission_and_slash(rep) -> list:
    whos = WHO + ("e7dc1",)
    out = ["| class | reading | " + " | ".join(whos) + " |", "|---|---|" + "---|" * len(whos)]
    for label in K.LABELS:
        for name, block, path in (("paths", "emission", ("PTH",)), ("queries", "emission", ("QRY",)),
                                  ("root slash, bare host", "root_slash", ("bare",)),
                                  ("root slash, query with no path", "root_slash", ("query_no_path",))):
            row = []
            for w in whos:
                node = rep.get(block, {}).get(w, {}).get(label, {})
                for p in path:
                    node = (node or {}).get(p)
                row.append(pct((node or {}).get("rate")))
            out.append(f"| {label} | {name} | " + " | ".join(row) + " |")
    return out + [""]


def realism_free(rep) -> list:
    whos = WHO + ("e7dc1",)
    out = ["| class | field | " + " | ".join(whos) + " |", "|---|---|" + "---|" * len(whos)]
    for label in K.LABELS:
        for f in GATED:
            row = []
            for w in whos:
                src = rep["e7_free"] if w == "e7" else rep.get("free", {}).get(w, {})
                row.append(cell(src.get(label, {}).get("fields", {}).get(f)))
            out.append(f"| {label} | {f} | " + " | ".join(row) + " |")
    return out + [""]


def realism_control(rep) -> list:
    out = ["| state | field | " + " | ".join(WHO) + " |", "|---|---|" + "---|" * len(WHO)]
    for label in K.LABELS:
        for st in K.STATES:
            if (label, st) in K.UNMEASURABLE:
                continue
            for f in GATED:
                row = [cell(rep.get("control", {}).get(w, {}).get(f"{label}_{st}", {})
                            .get("fields", {}).get(f)) for w in WHO]
                out.append(f"| {label} {st} | {f} | " + " | ".join(row) + " |")
    return out + [""]


def criteria(comp) -> list:
    keys = ("K1a", "K1b", "K1c", "K1d", "K1e")
    out = ["| arm | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for arm, r in comp["arms"].items():
        row = []
        for k in keys:
            v = r[k]
            if v.get("applies") is False:
                row.append("n/a")
            else:
                row.append("pass" if v["pass"] else "**FAIL**: " + "; ".join(
                    str(x) for x in (v.get("failures") or v.get("regressions") or [])))
        out.append(f"| {arm} | " + " | ".join(row) + " |")
    vb = comp["v_base"]
    out += ["", f"V-base (e8_base reproduces E7): **{'PASS' if vb['pass'] else 'FAIL'}**"
            + ("" if vb["pass"] else " -- " + "; ".join(vb["failures"])),
            "", f"**Outcome:** {comp['outcome']}", ""]
    return out


def attribution(comp) -> list:
    out = ["| metric | noise (e8_base - E7) | plan | credited | slash on E7 | credited "
           "| slash on plan | credited | interaction |", "|---|---|---|---|---|---|---|---|---|"]

    def f(x):
        return "--" if x is None else f"{x:+.4f}"

    def c(b):
        return "--" if b is None else ("yes" if b else "no")

    for name, r in comp["attribution"].items():
        out.append(f"| {name} | {f(r['noise'])} | {f(r['plan'])} | {c(r['plan_credited'])} "
                   f"| {f(r['slash_on_e7'])} | {c(r['slash_on_e7_credited'])} "
                   f"| {f(r['slash_on_plan'])} | {c(r['slash_on_plan_credited'])} "
                   f"| {f(r['interaction'])} |")
    return out + [""]


def novelty(comp) -> list:
    out = ["| arm | reading | e8_base | arm | worsening | 95% CI | counts |", "|---|---|---|---|---|---|---|"]
    for arm, r in comp["arms"].items():
        for key, v in r["K1e"].get("readings", {}).items():
            out.append(f"| {arm} | {key.replace('_', ' ')} | {v['base']:.4f} | {v['dc1']:.4f} "
                       f"| {v['worsening']:+.4f} | [{v['lo']:+.4f}, {v['hi']:+.4f}] "
                       f"| {'YES' if v['worsened'] else 'no'} |")
    return out + [""]


def main() -> None:
    rep, comp = load("score_report.json"), load("compare_report.json")
    lines = ["## Validity", ""] + verify(load("VERIFY.json"))
    lines += ["## Criteria", ""] + criteria(comp)
    lines += ["## Honouring (native, no resampling) -- separate from realism", ""] + honouring(rep)
    lines += ["## Emission and root slash (FULL FREE)", ""] + emission_and_slash(rep)
    lines += ["## Realism, FULL FREE (canonical gate)", ""] + realism_free(rep)
    lines += ["## Realism, FULL CONTROL (CR1 instrument, as delivered)", ""] + realism_control(rep)
    lines += ["## Novelty and copying", ""] + novelty(comp)
    lines += ["## Attribution", ""] + attribution(comp)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
