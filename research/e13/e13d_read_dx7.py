"""E13-D amendment 2 -- DX7's registered readout (research/e13/PREREG_E13D.md section 8).
Validity first (V6, V7), then the rule and its licences. Writes
data/e13d/readout_dx7.json and prints the tables E13D_RESULTS.md's DX7 appendix
carries verbatim.

    python research/e13/e13d_read_dx7.py > data/e13d/tables_dx7.md
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

CELLS = D.CELLS + (D.REFERENCE,)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dry", action="store_true")
    args = ap.parse_args()
    src = D.OUT / "dry" if args.allow_dry else D.OUT
    sys.stdout.reconfigure(encoding="utf-8")
    fits, fails = {}, []
    for cell in CELLS:
        d = json.loads((src / f"fit_{cell[0]}.json").read_text(encoding="utf-8"))
        if d.get("dry") and not args.allow_dry:
            raise SystemExit(f"STOP: fit_{cell[0]}.json is a dry run; it is never read")
        fits[cell[0]] = d
        v = d["val"]
        share = v["mismatch"] / max(1, v["mismatch"] + len(v["nll_transformer"]))
        if share > 0.01:
            fails.append(f"V6: {cell[0]} span mismatches {100 * share:.2f}% of validation strings")
        if max(abs(s - 1) for s in d["v7"]) > 1e-9:
            fails.append(f"V7: {cell[0]} n-gram does not sum to 1")
    if args.allow_dry:
        print("**DRY RUN: a pipeline test at toy size. NOT A RESULT; never read.**\n")
    print(f"### DX7 validity\n\nV6, V7: {'PASS' if not fails else 'FAIL'}; failures: {fails or 'none'}\n")
    if fails and not args.allow_dry:
        raise SystemExit("STOP: validity failed; nothing is read")
    out = {"validity": fails, "cells": {}}
    print("| cell | strings (val) | mismatched / skipped | transformer NLL (val) | 6-gram NLL (val) | Δ [95% CI] "
          "| DX7 | transformer NLL (train sample) | its gap (val − train) |\n|---|---|---|---|---|---|---|---|---|")
    for cell in CELLS:
        d = fits[cell[0]]
        v, t = d["val"], d["train"]
        r = D.fit_reading([a - b for a, b in zip(v["nll_transformer"], v["nll_ngram"])])
        mt, mn = st.mean(v["nll_transformer"]), st.mean(v["nll_ngram"])
        tt = st.mean(t["nll_transformer"]) if t["nll_transformer"] else float("nan")
        ref = " (reference)" if cell == D.REFERENCE else ""
        out["cells"][cell[0]] = {**r, "nll_transformer_val": mt, "nll_ngram_val": mn, "nll_transformer_train": tt}
        print(f"| {cell[0]}{ref} | {r['n']:,} | {v['mismatch']} / {v['skipped']} | {mt:.3f} | {mn:.3f} | "
              f"{r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['reading'] if not ref else '—'} | "
              f"{tt:.3f} | {mt - tt:+.3f} |")
    print("\n### DX7 licences (amendment 2)\n\n| cell | DX7 | licenses |\n|---|---|---|")
    lic = {}
    for cell in D.CELLS:
        r = out["cells"][cell[0]]["reading"]
        what = ("raising the field's learning signal within the fixed steps (a content loss weight; "
                "value and arms fixed in E13's registration)" if r == "UNDER-FIT" else
                "no training lever (the gap is in sampling)" if r == "FITS" else "nothing")
        lic[cell[0]] = what
        print(f"| {cell[0]} | {r} | {what} |")
    out["licences"] = lic
    json.dump(out, open(src / "readout_dx7.json", "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    main()
