"""E13 P -- the registered readout (research/e13/PREREG_E13.md section 2.6). Reads
data/e13p only; prints the tables and writes data/e13p/readout.json.

    python research/e13/e13p_read.py > data/e13p/tables.md

P-1: QC1-2's rule on each cell and arm (a draw is void, as in QC1 and E13-D, when
its real-against-real floor lies outside [0.42, 0.55]). P-2: QC1-4's rule. The
licence: subword tokens for a field if MIX-LED or BLENDING holds in both arms
(replicated) in any of that field's cells; otherwise ruled out by these
instruments (QC1's third route, the ladder's names, has no path or subdomain
form and is not run).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "research" / "qc1"))

import qc1_common as C  # noqa: E402
from e13p_measure import ARMS, CELLS, OUT  # noqa: E402

FLOOR_BAND = (0.42, 0.55)


def valid(draws):
    return [d for d in draws if FLOOR_BAND[0] <= d["floor"] <= FLOOR_BAND[1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dry", action="store_true")
    args = ap.parse_args()
    src = OUT / "dry" if args.allow_dry else OUT
    sys.stdout.reconfigure(encoding="utf-8")
    out = {"P-1": {}, "P-2": {}}
    if args.allow_dry:
        print("**DRY RUN: a pipeline test at toy size. NOT A RESULT; never read.**\n")
    print("### P-1 — the mix, or the content (QC1-2's rule)\n\n"
          "| cell | arm | n | x plain | x at the natural share | rho | t | reading | void | replicated |\n"
          "|---|---|---|---|---|---|---|---|---|---|")
    for cell in CELLS:
        rows = {}
        for arm in ARMS:
            d = json.loads((src / f"mix_{arm}_{cell}.json").read_text(encoding="utf-8"))
            if d.get("dry") and not args.allow_dry:
                raise SystemExit("STOP: a dry file is never read")
            draws = valid(d["draws"])
            r = C.mix_reading([x["plain"] for x in draws], [x["mix"] for x in draws])
            r.update({"n": d["n"], "void": len(d["draws"]) - len(draws),
                      "x_plain": st.mean(x["plain"] for x in draws),
                      "x_mix": st.mean(x["mix"] for x in draws), "share": d["share"]})
            rows[arm] = r
        rep = rows[ARMS[0]]["reading"] == rows[ARMS[1]]["reading"]
        out["P-1"][cell] = {"arms": rows, "replicated": rep}
        for arm, r in rows.items():
            print(f"| {cell} | {arm} | {r['n']} | {r['x_plain']:+.4f} | {r['x_mix']:+.4f} | "
                  f"{r['rho']:.2f} | {r['t']:+.2f} | {r['reading']} | {r['void']} | "
                  f"{'yes' if rep else '**no**'} |")
    nr = json.loads((src / "near.json").read_text(encoding="utf-8"))
    print("\n### P-2 — near misses (one edit from a training string seen 5+ times; QC1-4's rule)\n\n"
          "| cell | arm | generated never-seen, near | real never-seen, near | difference [95% CI] | "
          "reading | replicated |\n|---|---|---|---|---|---|---|")
    for cell in CELLS:
        node = nr[cell]
        rows = {arm: C.blending_reading(node[arm]["near"], node[arm]["n"], node["real"]["near"],
                                        node["real"]["n"]) for arm in ARMS}
        rep = all(r["reading"] == "BLENDING" for r in rows.values())
        out["P-2"][cell] = {"arms": rows, "replicated": rep}
        for arm, r in rows.items():
            print(f"| {cell} | {arm} | {100 * r['m_g']:.2f}% of {node[arm]['n']:,} | "
                  f"{100 * r['m_r']:.2f}% of {node['real']['n']:,} | {100 * r['diff']:+.2f} "
                  f"[{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] | {r['reading']} | "
                  f"{'yes' if rep else 'no'} |")
    lic = {}
    for field in ("PTH", "SUB"):
        cells = [c for c in CELLS if c.endswith(field)]
        mixled = [c for c in cells if out["P-1"][c]["replicated"]
                  and out["P-1"][c]["arms"][ARMS[0]]["reading"] == "MIX-LED"]
        blending = [c for c in cells if out["P-2"][c]["replicated"]]
        lic[field] = {"licensed": bool(mixled or blending),
                      "by": {"MIX-LED": mixled, "BLENDING": blending}}
    out["licences"] = lic
    print("\n### Licences (section 2.6)\n\n| field | subword tokens licensed for E14 | by |\n|---|---|---|")
    for field, r in lic.items():
        print(f"| {field} | **{'yes' if r['licensed'] else 'no: ruled out by these instruments'}** | "
              f"{r['by']} |")
    (src / "readout.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
