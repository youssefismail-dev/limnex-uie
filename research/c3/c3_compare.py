"""C3-QRY, C3-A -- is the query `<Q:1>` produces the query the model writes itself?

Arm Q = SC1 arm T's Q:1 records (data/sc1/T/cr1_report.json, read only); arm F =
c3_score.py's records. Per class and field, SC1's contrast (sc1_compare.contrast):
paired by generation when n is equal in every generation, else unpaired. The QRY
reading (PREREG_C3_QRY section 5):

    CONSISTENT    the 95% CI of Delta_QRY(Q - F) lies inside [-0.02, +0.02]
    DIFFERENT     |Delta| >= 0.02 and Holm-adjusted p < 0.05 over the two classes
    INCONCLUSIVE  otherwise

    python research/c3/c3_compare.py --c3 data/c3 --sc1 data/sc1 --bundle data/bundle2
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sc1"))
sys.path.insert(0, str(HERE.parent / "cr1"))

from c3_common import G, STATE, tag  # noqa: E402
from sc1_compare import cap_of, contrast, holm, pairable  # noqa: E402

#: section 5: QRY is the primary contrast; LBL, SUF, SUB, PTH and FULL are descriptive
FIELDS6 = ("LBL", "SUF", "SUB", "PTH", "QRY", "FULL")
MARGIN, ALPHA = 0.02, 0.05


def read_qry(rows: Dict[str, Optional[dict]], k1=()) -> Dict[str, dict]:
    """rows[label] = the QRY contrast (Q - F), or None. Holm over the registered
    family of two classes: a class that is not read -- K1 (PREREG section 8), or
    not computable -- stays in the family with p = 1, so the family never shrinks
    and the other class is never tested less strictly. Then each class's
    registered reading."""
    adj = holm({k: (r["p"] if r and k not in k1 else 1.0) for k, r in rows.items()})
    out = {}
    for label, r in rows.items():
        if label in k1:
            out[label] = {"reading": "K1 -- not read (PREREG section 8)"}
            continue
        if r is None:
            out[label] = {"reading": "NOT COMPUTABLE"}
            continue
        if r["lo"] > -MARGIN and r["hi"] < MARGIN:
            reading = "CONSISTENT"
        elif abs(r["delta"]) >= MARGIN and adj[label] < ALPHA:
            reading = "DIFFERENT"
        else:
            reading = "INCONCLUSIVE"
        out[label] = {**r, "p_holm": adj[label], "reading": reading}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3", type=Path, required=True)
    ap.add_argument("--sc1", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    args = ap.parse_args()

    from cr1_score import pools
    from uie.evaluation import FIELDS

    q_rep = json.loads((args.sc1 / "T" / "cr1_report.json").read_text(encoding="utf-8"))["states"]
    f_rep = json.loads((args.c3 / "F" / "report.json").read_text(encoding="utf-8"))["states"]
    out = {"registration": "research/c3/PREREG_C3_QRY.md", "contrast": "Q - F", "classes": {}}
    qry_rows = {}
    k1 = [label for label in ("benign", "phishing") if "k1" in f_rep[tag(label)]]
    for label in ("benign", "phishing"):
        t = tag(label)
        if label in k1:
            out["classes"][label] = {"k1": f_rep[t]["k1"]}
            qry_rows[label] = None
            continue
        real = pools(args.bundle, label, STATE)["name"]
        caps = {f: cap_of(sum(1 for u in real if FIELDS[f](u))) for f in FIELDS6}
        fakes = {}
        for arm, d in (("Q", args.sc1 / "T" / "gen"), ("F", args.c3 / "F" / "gen")):
            gens = [json.loads((d / f"{t}_g{g}.json").read_text(encoding="utf-8"))
                    for g in range(G)]
            fakes[arm] = {f: [sum(1 for u in x if FIELDS[f](u)) for x in gens] for f in FIELDS6}
        cells = {}
        for f in FIELDS6:
            can = pairable(caps[f], fakes["Q"][f], fakes["F"][f])
            cells[f] = contrast(q_rep[t]["fields"].get(f), f_rep[t]["fields"].get(f), can)
        if cells["QRY"] is not None and not cells["QRY"]["paired"]:
            cells["QRY"]["design_note"] = "UNPAIRED -- the registered sizes should have paired it"
        out["classes"][label] = {"cells": cells, "caps": caps}
        qry_rows[label] = cells["QRY"]
    out["qry"] = read_qry(qry_rows, k1)
    (args.c3 / "c3a_report.json").write_text(json.dumps(out, indent=1, default=float),
                                             encoding="utf-8")

    print("C3-A -- QRY, Delta = x_Q - x_F")
    for label, r in out["qry"].items():
        if "delta" in r:
            print(f"  {label:<9} delta {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
                  f"p {r['p']:.3g} holm {r['p_holm']:.3g} "
                  f"{'paired' if r['paired'] else 'UNPAIRED'}  ->  {r['reading']}")
        else:
            print(f"  {label:<9} {r['reading']}")
    print("descriptive, the other fields:")
    for label, c in out["classes"].items():
        if "k1" in c:
            print(f"  {label}  K1 -- not read: {c['k1']}")
            continue
        print("  " + label + "  " + "  ".join(
            f"{f} {r['delta']:+.4f}{'' if r['paired'] else '†'}" if r else f"{f} --"
            for f, r in c["cells"].items() if f != "QRY"))


if __name__ == "__main__":
    main()
