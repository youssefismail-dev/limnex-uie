"""E13 P -- QC1's subword questions, asked of paths and subdomains
(research/e13/PREREG_E13.md section 2.6). DIAGNOSTIC, laptop only: E12's files,
nothing trained or generated.

    python research/e13/e13p_measure.py --part near
    python research/e13/e13p_measure.py --part mix --cell control_benign_P1Q1_PTH --arm e12_final

Cells: control benign P1Q1 PTH, control benign P1Q1 SUB, control phishing P1Q0
PTH. Primary e12_final, replication e12_base. The populations are E13-D's (the
gate's real pool for the cell, one URL per host; E12's five generations pooled);
the training strings are QC1-1's set for the field (name_train + structure_train
of the class, with their counts).

  near (P-2, QC1-4's rule): never-seen strings one edit from a training string
       seen 5+ times, generated against real.
  mix  (P-1, QC1-2's rule): the generated strings as written ("plain") against the
       same strings re-drawn to the real share of exact training strings ("mix"),
       QC1's paired harness, D = 12 seeds.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e13d_common as D  # noqa: E402
import qc1_common as C  # noqa: E402

OUT = ROOT / "data" / "e13p"
CELLS = ("control_benign_P1Q1_PTH", "control_benign_P1Q1_SUB", "control_phishing_P1Q0_PTH")
ARMS = ("e12_final", "e12_base")
PRIMARY, REPLICATION = ARMS


def train_counts(label: str, field: str) -> collections.Counter:
    """QC1-1's training set, for the field: every name_train and structure_train
    row of the class, with each non-empty field string counted."""
    f = D.field_of(field)
    pre = "" if label == "benign" else f"{label}_"
    c = collections.Counter()
    for kind in ("name", "structure"):
        for u in C._jsonl(f"{pre}{kind}_train"):
            s = f(u)
            if s:
                c[s] += 1
    return c


def sample(xs, n, seed):
    return random.Random(seed).sample(list(xs), n)


def part_near(args) -> dict:
    out = {}
    for name in CELLS:
        cell = D.BY_NAME[name]
        counts = train_counts(cell[2], cell[4])
        idx = C.NearIndex(s for s, c in counts.items() if c >= C.FREQUENT)
        real = [s for s in D.pool(cell, "val") if s not in counts]
        node = {"real": {"n": len(real), "near": sum(idx.near(s) for s in real)}}
        for arm in ARMS:
            gen = [s for s in D.generated(cell, arm) if s not in counts]
            node[arm] = {"n": len(gen), "near": sum(idx.near(s) for s in gen)}
        out[name] = node
        print(f"  near {name}: real never-seen {len(real)}", flush=True)
        del counts, idx
        gc.collect()
    return out


def part_mix(args) -> dict:
    seeds = C.SEEDS[:2] if args.dry else C.SEEDS
    n_req = 200 if args.dry else C.N_REQ
    cell = D.BY_NAME[args.cell]
    counts = train_counts(cell[2], cell[4])
    real = D.pool(cell, "val")
    gen = D.generated(cell, args.arm)
    found = [s for s in gen if s in counts]
    novel = [s for s in gen if s not in counts]
    share = sum(s in counts for s in real) / len(real)
    n0 = min(n_req, len(real) // 3, len(gen))
    n = C.feasible_n(n0, len(found), len(novel), share)
    out = {"cell": args.cell, "arm": args.arm, "n_plain": n0, "n": n, "share": share,
           "generated": {"found": len(found), "novel": len(novel)}, "seeds": list(seeds),
           "draws": []}
    for s in seeds:
        fakes = {"plain": sample(gen, n, s + 101), "mix": C.stratified(found, novel, share, n, s + 202)}
        out["draws"].append(C.harness(real, fakes, s, n))
        print(f"  mix {args.arm} {args.cell} seed {s} done", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("near", "mix"))
    ap.add_argument("--cell", choices=CELLS)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--dry", action="store_true", help="toy size into data/e13p/dry, never read")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    if args.part == "mix" and not (args.cell and args.arm):
        ap.error("mix needs --cell and --arm")
    rep = part_near(args) if args.part == "near" else part_mix(args)
    rep.update({"registration": "research/e13/PREREG_E13.md (section 2.6)", "part": args.part,
                "dry": bool(args.dry), "seconds": round(time.time() - t0, 1)})
    tag = args.part + (f"_{args.arm}_{args.cell}" if args.part == "mix" else "")
    dest = (OUT / "dry" if args.dry else OUT) / f"{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(f"wrote {dest} after {rep['seconds']:.0f}s")


if __name__ == "__main__":
    main()
