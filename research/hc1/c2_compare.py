"""HC1 -- C2 (carried over verbatim from E7) and C4: E7's unconditional mode
against E5, both under gate v2 in its canonical configuration.

C2, per gated cell (class x LBL, SUF, SUB, PTH, QRY):
    KILL              the verdict worsens AND x rises by more than 0.0141
    not attributable  the verdict worsens, x rises by 0.0141 or less
    no regression     otherwise
The effective verdict is the section-26.1 one; where it is NOT COMPUTABLE the
pooled rule's is used (TELL counts as CONVICTED, and it can never acquit).
FULL is auxiliary and never gates. C4: emission rates, gate pass and novelty.

    python research/hc1/c2_compare.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from uie.corpus.run_eval import pooled_verdict  # noqa: E402

E5 = "data/gate_v2_baseline_{}/eval_report.json"
E7 = "data/hc1_c2_e7_{}/eval_report.json"
NOISE = 0.0141                       # measured retraining noise (E7 section 4, C2)
RANK = {"CLEARED": 0, "UNDECIDED": 1, "CONVICTED": 2}
GATED = ("LBL", "SUF", "SUB", "PTH", "QRY")


def effective(block: dict, field: str):
    acc = block["fields"][field].get("acceptance") or {}
    if acc.get("verdict") in RANK:
        return acc["verdict"], acc["x"], "26.1"
    pooled = (block.get("pooled_fields") or {}).get(field)
    if pooled:
        v = pooled_verdict(pooled["excess_draws"])
        return ("CONVICTED" if v == "TELL" else "UNDECIDED"), round(pooled["excess"], 4), "pooled"
    return "NOT COMPUTABLE", None, "--"


def main() -> None:
    out = {"noise": NOISE, "cells": [], "c4": {}}
    print("| cell | E5 verdict | E5 x | E7 verdict | E7 x | dx | C2 |")
    print("|---|---|---:|---|---:|---:|---|")
    for cls in ("benign", "phishing"):
        e5 = json.loads((ROOT / E5.format(cls)).read_text(encoding="utf-8"))[cls]
        e7 = json.loads((ROOT / E7.format(cls)).read_text(encoding="utf-8"))[cls]
        for f in GATED:
            v5, x5, how5 = effective(e5, f)
            v7, x7, how7 = effective(e7, f)
            dx = None if x5 is None or x7 is None else round(x7 - x5, 4)
            worse = v5 in RANK and v7 in RANK and RANK[v7] > RANK[v5]
            status = ("KILL" if worse and dx is not None and dx > NOISE else
                      "not attributable" if worse else "no regression")
            out["cells"].append({"class": cls, "field": f, "e5": [v5, x5, how5],
                                 "e7": [v7, x7, how7], "dx": dx, "c2": status})
            num = lambda v: "--" if v is None else f"{v:+.4f}"  # noqa: E731
            print(f"| {cls} {f} | {v5} ({how5}) | {num(x5)} | {v7} ({how7}) | "
                  f"{num(x7)} | {num(dx)} | {status} |")
        full5 = e5["fields"]["FULL"]["acceptance"]["x"]
        full7 = e7["fields"]["FULL"]["acceptance"]["x"]
        out["c4"][cls] = {
            "full_auxiliary": {"e5": full5, "e7": full7},
            "emission": {f: {"e5": e5["emission_rates"][f]["generated"],
                             "e7": e7["emission_rates"][f]["generated"],
                             "real": e5["emission_rates"][f]["real"]}
                         for f in ("SUB", "PTH", "QRY")},
            "gate": {"e5": e5["gate"], "e7": e7["gate"]},
            "precision_recall": {"e5": e5["precision_recall"], "e7": e7["precision_recall"]}}
    kills = [c for c in out["cells"] if c["c2"] == "KILL"]
    out["verdict"] = "KILL" if kills else "NO ATTRIBUTABLE REGRESSION"
    print(f"\nC2: {out['verdict']}"
          + (f" -- {[(c['class'], c['field']) for c in kills]}" if kills else ""))
    print("\nC4 emission (generated; real in brackets):")
    for cls, c4 in out["c4"].items():
        print(f"  {cls:<9} " + "  ".join(
            f"{f} E5 {100 * v['e5']:.2f}% -> E7 {100 * v['e7']:.2f}% [{100 * v['real']:.2f}%]"
            for f, v in c4["emission"].items())
              + f"   FULL(aux) {c4['full_auxiliary']['e5']:+.4f} -> {c4['full_auxiliary']['e7']:+.4f}")
    (ROOT / "data/hc1/c2_compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
