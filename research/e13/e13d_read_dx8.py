"""E13-D amendment 3 -- DX8's registered readout (research/e13/PREREG_E13D.md section 9).
Validity first (V8, V9, and the cached sampler's V10), then the rule and its
licences. Writes data/e13d/readout_dx8.json and prints the tables.

    python research/e13/e13d_read_dx8.py > data/e13d/tables_dx8.md
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
from e13d_read import valid  # noqa: E402

C = D.C


def context_reading(x_gv, x_rc) -> dict:
    """DX8: CONTEXT-LED if mean x_RC <= 0.5 mean x_GV and the paired difference
    x_GV - x_RC is significant; FIELD-LED if mean x_RC >= 0.8 mean x_GV or the
    difference is not significant; SPLIT otherwise."""
    m, se, t = C.paired_t([a - b for a, b in zip(x_gv, x_rc)])
    gv, rc = st.mean(x_gv), st.mean(x_rc)
    sig = t == t and t > C.TCRIT
    reading = ("CONTEXT-LED" if rc <= 0.5 * gv and sig else
               "FIELD-LED" if rc >= 0.8 * gv or not sig else "SPLIT")
    return {"x_gv": gv, "x_rc": rc, "ratio": rc / gv if gv else float("nan"), "diff": m, "t": t,
            "reading": reading}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dry", action="store_true")
    args = ap.parse_args()
    src = D.OUT / "dry" if args.allow_dry else D.OUT
    sys.stdout.reconfigure(encoding="utf-8")
    data, fails = {}, []
    for cell in D.CELLS:
        d = json.loads((src / f"context_{cell[0]}.json").read_text(encoding="utf-8"))
        if d.get("dry") and not args.allow_dry:
            raise SystemExit(f"STOP: context_{cell[0]}.json is a dry run; it is never read")
        data[cell[0]] = d
        sampled = d["urls"] - d["skipped"]
        if sampled and d["sampled_valid"] / sampled < 0.98:
            fails.append(f"V8: {cell[0]} only {d['sampled_valid']} of {sampled} samples valid")
        if d["v10_max_abs_diff"] is None or d["v10_max_abs_diff"] > 1e-3:
            fails.append(f"V10: {cell[0]} cached sampler differs from the full pass by {d['v10_max_abs_diff']}")
        draws = d["draws"]
        v9 = max(abs(a["gen"] - b) for a, b in zip(draws, d["dx1_gen"]))
        if v9 > 0.02:
            fails.append(f"V9: {cell[0]} x_GV differs from DX1's by {v9:.4f}")
    if args.allow_dry:
        print("**DRY RUN: a pipeline test at toy size. NOT A RESULT; never read.**\n")
    print(f"### DX8 validity\n\nV8, V9, V10: {'PASS' if not fails else 'FAIL'}; failures: {fails or 'none'}\n")
    if fails and not args.allow_dry:
        raise SystemExit("STOP: validity failed; nothing is read")
    out = {"validity": fails, "cells": {}}
    print("| cell | n | valid samples | x_GV (generated) | x_RC (real context) | ratio | paired t | DX8 |"
          "\n|---|---|---|---|---|---|---|---|")
    for cell in D.CELLS:
        d = data[cell[0]]
        draws = valid(d["draws"])
        r = context_reading([x["gen"] for x in draws], [x["rc"] for x in draws])
        out["cells"][cell[0]] = r
        print(f"| {cell[0]} | {d['n']} | {d['sampled_valid']:,} of {d['urls'] - d['skipped']:,} | {r['x_gv']:+.4f} | "
              f"{r['x_rc']:+.4f} | {r['ratio']:.2f} | {r['t']:+.2f} | **{r['reading']}** |")
    print("\n### DX8 licences (amendment 3)\n\n| cell | DX8 | licenses |\n|---|---|---|")
    lic = {}
    for cell in D.CELLS:
        r = out["cells"][cell[0]]["reading"]
        what = ("a training-side lever on exposure to the model's own context (form and arms fixed in "
                "E13's registration)" if r == "CONTEXT-LED" else
                "no lever within the current rules (decoder-side only): an owner decision" if r == "FIELD-LED"
                else "nothing")
        lic[cell[0]] = what
        print(f"| {cell[0]} | {r} | {what} |")
    out["licences"] = lic
    json.dump(out, open(src / "readout_dx8.json", "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    main()
