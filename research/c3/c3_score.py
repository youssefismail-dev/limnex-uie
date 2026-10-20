"""C3-QRY -- score arm F with CR1's scorer, unchanged (PREREG_C3_QRY section 5).

`cr1_score.pools` restricted to the Q:1 state and `cr1_score.score` with
n_main = 1,200, so every field is requested at min(1,200, 1,200) = 1,200 -- the
request arm Q's records (SC1 arm T) carry. Same SEED and split seeds, so the QRY
draws pair with arm Q's on identical real samples and floors. Records are
written in cr1_score.py's own format.

    python research/c3/c3_score.py --gens data/c3/F/gen --bundle data/bundle2 --out data/c3/F
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cr1"))

import cr1_score  # noqa: E402
from c3_common import G, N, STATE, tag  # noqa: E402
from uie.evaluation import FIELDS  # noqa: E402

N_MAIN = 1200


def k1_labels(gens: Path) -> dict:
    """K1 (PREREG section 8): a class with any FreeUnmet file has failed, and its
    C3-A and C3-B are not read. {label: the provenance's error lines}."""
    prov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    out = {}
    for r in prov["files"]:
        if r.get("error"):
            out.setdefault(r["label"], []).append(f"{r['file']}: {r['error']}")
    return out


def score_f(gens: Path, bundle: Path, label: str) -> dict:
    t0 = time.time()
    urls = [json.loads((gens / f"{tag(label)}_g{g}.json").read_text(encoding="utf-8"))
            for g in range(G)]
    assert all(len(x) == N[label] for x in urls), f"{label}: registered size"
    assert all(FIELDS["QRY"](u) for x in urls for u in x), f"{label}: a URL without a query"
    pop = cr1_score.pools(bundle, label, STATE)
    fields, pooled = cr1_score.score(pop, urls, N_MAIN)
    return {"label": label, "state": STATE, "real_name_pool": len(pop["name"]),
            "fields": fields, "pooled_fields": pooled, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"registration": "research/c3/PREREG_C3_QRY.md",
              "scorer": "research/cr1/cr1_score.py (pools, score), unchanged",
              "G": G, "D": cr1_score.D, "n_main": N_MAIN, "seed": cr1_score.SEED, "states": {}}
    k1 = k1_labels(args.gens)
    for label in ("benign", "phishing"):
        if label in k1:
            report["states"][tag(label)] = {"label": label, "state": STATE, "k1": k1[label]}
            (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float),
                                                  encoding="utf-8")
            print(f"  {tag(label):<12} K1 -- not scored: {k1[label]}", flush=True)
            continue
        rec = score_f(args.gens, args.bundle, label)
        report["states"][tag(label)] = rec
        (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float),
                                              encoding="utf-8")
        summary = "  ".join(
            f"{f} {r['acceptance'].get('x', float('nan')):+.4f} {r['acceptance']['verdict']}"
            if r.get("acceptance", {}).get("x") is not None
            else f"{f} {r.get('acceptance', {}).get('verdict', '--')}"
            for f, r in rec["fields"].items())
        print(f"  {tag(label):<12} {summary}  ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
