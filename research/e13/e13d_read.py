"""E13-D -- the registered readout (research/e13/PREREG_E13D.md sections 2-4).
Checks validity first and stops if it fails. Then it applies each reading's rule
and the licence table, writes data/e13d/readout.json, and prints the tables
E13D_RESULTS.md carries verbatim.

    python research/e13/e13d_read.py > data/e13d/tables.md

Refuses dry-run files unless --allow-dry (a pipeline test, stamped).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e13d_common as D  # noqa: E402

C = D.C
ALLOW_DRY = False
SRC = D.OUT


def load(name: str) -> dict:
    d = json.loads((SRC / name).read_text(encoding="utf-8"))
    if d.get("dry") and not ALLOW_DRY:
        raise SystemExit(f"STOP: {name} is a dry run; it is never read")
    return d


def valid(draws: list) -> list:
    """V4: a seed whose floor is outside Q3's bounds is void for every condition."""
    return [d for d in draws if C.FLOOR_LOW <= d["floor"] <= C.FLOOR_HIGH]


def gate_x(rep: dict, arm: str, cell) -> float:
    name, mode, label, state, field = cell
    node = rep["free"][arm][label] if mode == "free" else rep["control"][arm][f"{label}_{state}"]
    return node["fields"][field]["acceptance"]["x"]


def validity(e12rep: dict) -> tuple:
    from e13d_measure import sha256
    fails, v2 = [], {}
    v = load("validity.json")
    if v["v1"]["e12_verify"] != "PASS":
        fails.append("V1: E12's VERIFY is not PASS")
    for rel, h in v["v1"]["hashes"].items():
        if sha256(D.ROOT / rel) != h:
            fails.append(f"V1: {rel} changed since the measurement")
    for name, node in v["v3"].items():
        for split, r in node.items():
            if r["urls_one_per_host"] != r["distinct_hosts"] or not r["field_strings"]:
                fails.append(f"V3: {name} {split} pool is not one URL per host, or empty")
    for cell in D.CELLS:
        d = load(f"decomp_{cell[0]}.json")
        if not d["v2"]:
            fails.append(f"V2: {cell[0]} has no V2 draws")
            continue
        x = st.mean(d["v2"])
        g = gate_x(e12rep, D.PRIMARY, cell)
        v2[cell[0]] = {"harness": x, "gate": g, "diff": x - g}
        if abs(x - g) > 0.05:
            fails.append(f"V2: {cell[0]} harness {x:+.4f} against the gate's {g:+.4f}")
    return fails, v2


def v5(probe: dict) -> tuple:
    """V5 (amendment 1): in each generated population, the probe's mean violation
    probability lies within the 95% Wilson interval of the same URLs' observed
    violation rate -- the probe's own check."""
    out, ok = {}, True
    for name, p in probe["generated"].items():
        if not p["n"]:
            out[name] = {"n": 0, "within": None}
            continue
        k, n = sum(p["observed"]), p["n"]
        lo, hi = C.wilson(k, n)
        m = st.mean(p["viol"])
        within = lo <= m <= hi
        ok &= within
        out[name] = {"n": n, "probe_mean": m, "observed": k / n, "ci": [lo, hi], "within": within}
    return ok, out


def main() -> None:
    global ALLOW_DRY, SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dry", action="store_true")
    args = ap.parse_args()
    if args.allow_dry:
        ALLOW_DRY, SRC = True, D.OUT / "dry"
    sys.stdout.reconfigure(encoding="utf-8")
    e12rep = json.loads((D.E12 / "score_report.json").read_text(encoding="utf-8"))
    e11rep = json.loads((D.ROOT / "data" / "e11_final" / "score_report.json").read_text(encoding="utf-8"))
    fails, v2 = validity(e12rep)
    if args.allow_dry:
        print("**DRY RUN: a pipeline test at toy size. NOT A RESULT; never read.**\n")
    print(f"### Validity\n\nV1–V4: {'PASS' if not fails else 'FAIL'}; failures: {fails or 'none'}\n")
    print("| cell | harness x (gate n) | gate x | difference |\n|---|---|---|---|")
    for name, r in v2.items():
        print(f"| {name} | {r['harness']:+.4f} | {r['gate']:+.4f} | {r['diff']:+.4f} |")
    if fails and not args.allow_dry:
        json.dump({"validity": fails}, open(SRC / "readout.json", "w"), indent=1)
        raise SystemExit("STOP: validity failed; nothing is read")
    out = {"validity": {"fails": fails, "v2": v2}, "cells": {}, "quantity": {}, "dx4": {}, "dx5": {}}

    print("\n### DX1 and DX2 — where each cell's gap lives\n")
    print("| cell | n | x_GV | x_TV (share) | DX1 | x_GT final (share) | x_GT base (share) | DX2 | attribution | void seeds |"
          "\n|---|---|---|---|---|---|---|---|---|---|")
    for cell in D.CELLS:
        d = load(f"decomp_{cell[0]}.json")
        av, at = valid(d["against_val"]), valid(d["against_train"])
        voids = (len(d["against_val"]) - len(av)) + (len(d["against_train"]) - len(at))
        x_gv = [x[D.PRIMARY] for x in av]
        x_gv_b = [x[D.REPLICATION] for x in av]
        shift = D.shift_reading([x["train"] for x in av], x_gv)
        gp = D.gap_reading([x[D.PRIMARY] for x in at], x_gv)
        gb = D.gap_reading([x[D.REPLICATION] for x in at], x_gv_b)
        gap = D.replicated(gp["reading"], gb["reading"])
        attr = D.attribution(shift["reading"], gap)
        out["cells"][cell[0]] = {"shift": shift, "gap_primary": gp, "gap_replication": gb, "gap": gap,
                                 "attribution": attr, "n": d["n"], "void": voids}
        print(f"| {cell[0]} | {d['n']} | {shift['x_gv']:+.4f} | {shift['x_tv']:+.4f} ({shift['share']:.2f}) | "
              f"{shift['reading']} | {gp['x_gt']:+.4f} ({gp['share']:.2f}) | {gb['x_gt']:+.4f} ({gb['share']:.2f}) | "
              f"{gap} | **{attr}** | {voids} |")

    print("\n### DX3 — is there enough data? (query cells; free benign QRY is the CLEARED reference)\n")
    print("| cell | val | train | coverage 1/8 · 1/4 · 1/2 · 1 | g | unseen templates | singleton templates "
          "| learner x 1/8 · 1/4 · 1/2 · 1 | s (t) | DX3 |\n|---|---|---|---|---|---|---|---|---|---|")
    for cell in D.QUERY_CELLS:
        q = load(f"quantity_{cell[0]}.json")
        draws = valid(q["draws"])
        xs = {f: [x[str(f)] for x in draws] for f in D.FRACTIONS}
        g = q["coverage"]["1.0"] - q["coverage"]["0.5"]
        r = D.quantity_reading(g, xs[0.5], xs[1.0])
        out["quantity"][cell[0]] = {**r, "coverage": q["coverage"], "templates": q["templates"],
                                    "learner": {str(f): st.mean(v) for f, v in xs.items()}}
        cov = " · ".join(f"{100 * q['coverage'][str(f)]:.1f}%" for f in D.FRACTIONS)
        lx = " · ".join(f"{st.mean(xs[f]):+.3f}" for f in D.FRACTIONS)
        ref = " (reference)" if cell == D.REFERENCE else ""
        print(f"| {cell[0]}{ref} | {q['sizes']['val']:,} | {q['sizes']['train']:,} | {cov} | {100 * g:+.1f} pts | "
              f"{100 * q['templates']['val_unseen_share']:.1f}% | {100 * q['templates']['train_singleton_share']:.1f}% | "
              f"{lx} | {r['s']:+.4f} ({r['t']:+.2f}) | {r['reading'] if ref == '' else '—'} |")

    print("\n### DX4 — distance to a reliable clear (descriptive)\n")
    print("| cell | E11 | e12_base | e12_final | e12_no_r1 | mean | between-run SD | clear line | P(CLEARED) | mean x for P ≥ 0.8 |"
          "\n|---|---|---|---|---|---|---|---|---|---|")
    for cell in D.CELLS:
        xs = [gate_x(e11rep, "e11_plan", cell)] + [gate_x(e12rep, a, cell) for a in ("e12_base", "e12_final", "e12_no_r1")]
        _, mode, label, state, field = cell
        node = e12rep["free"][D.PRIMARY][label] if mode == "free" else e12rep["control"][D.PRIMARY][f"{label}_{state}"]
        se = node["fields"][field]["acceptance"]["se"]
        m, sd = st.mean(xs), st.stdev(xs)
        r = D.clear_probability(m, sd, se)
        out["dx4"][cell[0]] = {"xs": xs, "mean": m, "sd": sd, "se": se, **r}
        print(f"| {cell[0]} | " + " | ".join(f"{x:+.4f}" for x in xs) +
              f" | {m:+.4f} | {sd:.4f} | {r['line']:+.4f} | {r['p_clear']:.2f} | {r['x_for_0.8']:+.4f} |")

    probe = load("probe.json")
    ok5, v5r = v5(probe)
    print(f"\n### DX5 — slash obedience (teacher-forced, `e12_final`, its own URLs: amendment 1)\n\n"
          f"V5 (the probe against the same URLs' observed violations): {'PASS' if ok5 else '**FAIL**'}\n")
    print("| population | n | observed violations | probe mean | 95% interval | within |\n|---|---|---|---|---|---|")
    for name, r in v5r.items():
        if r.get("n"):
            print(f"| {name} | {r['n']:,} | {100 * r['observed']:.3f}% | {100 * r['probe_mean']:.3f}% | "
                  f"[{100 * r['ci'][0]:.3f}, {100 * r['ci'][1]:.3f}] | {'yes' if r['within'] else '**no**'} |")
    out["dx5"]["v5"] = {"pass": ok5, **v5r}
    print("\n| population | n | mean violation | top-decile share c | support: top decile / all | DX5 |\n|---|---|---|---|---|---|")
    for name, p in probe["generated"].items():
        if not p["n"]:
            continue
        r = D.obedience_reading(p["viol"], p["support"])
        reading = r["reading"] if ok5 else "VOID (V5)"
        out["dx5"][name] = {**r, "reading": reading}
        print(f"| {name} | {r['n']:,} | {100 * r['m']:.3f}% | {r['c']:.2f} | {r['top_support']:,.0f} / {r['support']:,.0f} | {reading} |")
    print("\nReported beside (as registered; licenses nothing): the same probe on real validation URLs\n")
    print("| population | n | mean violation (real) | mean violation (generated) |\n|---|---|---|---|")
    for name, p in probe["populations"].items():
        g = probe["generated"].get(f"gen_{name}")
        gm = f"{100 * st.mean(g['viol']):.3f}%" if g and g["n"] else "—"
        out["dx5"].setdefault("real", {})[name] = {"n": p["n"], "m": st.mean(p["viol"]) if p["n"] else None}
        print(f"| {name} | {p['n']:,} | {100 * st.mean(p['viol']):.3f}% | {gm} |" if p["n"] else f"| {name} | 0 | — | {gm} |")

    desc = load("describe.json")
    print("\n### DX6 — E12's side effects (descriptive)\n\nCopying by query length (free mode):\n")
    print("| class | length | e12_base | e12_final |\n|---|---|---|---|")
    for label, arms in desc["copy_by_length"].items():
        for b in ("1-10", "11-30", "31-80", "81+"):
            cells_ = []
            for arm in ("e12_base", "e12_final"):
                k, n = arms[arm].get(b, [0, 0])
                cells_.append(f"{100 * k / n:.1f}% of {n:,}" if n else "—")
            print(f"| {label} | {b} | " + " | ".join(cells_) + " |")
    print("\nPhishing control P1Q1 (path and query requested), what was written:\n")
    print("| arm | P1Q1 (honoured) | P1Q0 (query dropped) | P0Q1 (path dropped) | P0Q0 |\n|---|---|---|---|---|")
    for arm, c in desc["p1q1_phishing"].items():
        n = sum(c.values())
        print(f"| {arm} | " + " | ".join(f"{100 * c.get(k, 0) / n:.2f}%" for k in ("P1Q1", "P1Q0", "P0Q1", "P0Q0")) + " |")

    print("\n### Licences (section 3)\n\n| item | reading | licenses |\n|---|---|---|")
    lic = {}
    for cell in D.CELLS:
        c = out["cells"][cell[0]]
        qd = out["quantity"].get(cell[0], {}).get("reading")
        if c["shift"]["reading"] == "SHIFT-DOMINANT" or qd == "DATA-LIMITED":
            what = "a data intervention (the owner's six requirements first)"
        elif c["attribution"] == "LEARNING":
            what = "a learning-side lever (objective, weighting, conditioning)"
        else:
            what = "nothing"
        lic[cell[0]] = what
        print(f"| {cell[0]} | {c['attribution']}" + (f"; DX3 {qd}" if qd else "") + f" | {what} |")
    for name in D.GEN_PROBES:
        if name not in out["dx5"]:
            continue
        r = out["dx5"][name]["reading"]
        what = ("a data lever for those contexts" if r == "CONCENTRATED" else
                "a plan-decision objective lever" if r == "UNIFORM" else "nothing")
        lic[name] = what
        print(f"| slash {name} | {r} | {what} |")
    out["licences"] = lic
    json.dump(out, open(SRC / "readout.json", "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    main()
