"""Empirical noise floor: the 95th percentile of |sigma| under a KNOWN NULL.

The theoretical threshold failed. With G=5 the statistic has 4 df, so |t|>2 has
p~0.116, not 0.05 -- and across 4 fields that is a ~40% chance of a false
positive. The null self-test duly produced 2.9 sigma on PTH between two draws of
the SAME distribution.

Rather than pick a t-value and hope, this MEASURES the null distribution and
reads the threshold off it. No distributional assumption; it absorbs the df
issue, any pairing artifact and field-specific quirks in one number per field.

EFFICIENCY. The naive design (R replicates x 2 arms x G generations) is 80
generations at n=1800 ~ 9 hours. Instead:

  1. generate a POOL of M generations once           (the expensive step)
  2. compute excess per (generation, field, disc_seed) once, cached
  3. form R replicates by random disjoint G+G partitions of the pool

Steps 1-2 cost M generations and M*|fields|*D fits; step 3 is free, so R can be
large. Replicates share generations and are therefore correlated -- acceptable
for estimating a per-field percentile, and stated so it is not over-read.

n=1800 because SUB was UNMEASURABLE at n=600: ~42% field coverage gives ~250
values against a 3n requirement with a 250 floor.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.evaluation import discriminate_with_floor
from research.e5.harness import CACHE, FIELDS, POOL, generate_arm, rows

ap = argparse.ArgumentParser()
ap.add_argument("--label", default="benign")
ap.add_argument("--ckpt", default=str(ROOT / "data/best_e5.pt"))
ap.add_argument("-M", type=int, default=16, help="pool size (generations)")
ap.add_argument("-G", type=int, default=5, help="generations per arm")
ap.add_argument("-D", type=int, default=3)
ap.add_argument("-n", type=int, default=1800)
ap.add_argument("-R", type=int, default=200, help="null replicates (free)")
args = ap.parse_args()

pool_urls = rows(POOL[args.label])
SEEDS = [30000 + 17 * i for i in range(args.M)]
print(f"pool: M={args.M} generations, n={args.n}, label={args.label}", flush=True)
gens = generate_arm(args.ckpt, args.label, SEEDS, args.n, tag="pool")

# --- step 2: one excess per (generation, field, disc seed), cached ---------- #
store = CACHE / f"excess_{args.label}_n{args.n}_M{args.M}_D{args.D}.json"
if store.is_file():
    E = json.loads(store.read_text(encoding="utf-8"))
else:
    E = {}
    for f in FIELDS:
        for gi, urls in enumerate(gens):
            for ds in range(1, args.D + 1):
                r = discriminate_with_floor(pool_urls, urls, f, n=args.n,
                                            seed=ds)
                E.setdefault(f, {}).setdefault(str(gi), {})[str(ds)] = (
                    None if r.get("insufficient") else r["excess"])
        done = sum(1 for gi in E[f] for ds in E[f][gi]
                   if E[f][gi][ds] is not None)
        print(f"  {f}: {done}/{args.M * args.D} usable fits", flush=True)
    store.write_text(json.dumps(E), encoding="utf-8")

# --- step 3: replicate the null by partitioning the pool ------------------- #
rng = random.Random(20260812)
print(f"\nNULL DISTRIBUTION over R={args.R} disjoint {args.G}+{args.G} "
      f"partitions of the pool")
print(f"  {'field':<6}{'usable':>8}{'median|s|':>11}{'p90':>8}{'p95':>8}"
      f"{'p99':>8}{'max':>8}   THRESHOLD")
thresholds = {}
for f in FIELDS:
    per_gen = []
    for gi in range(args.M):
        vals = [v for v in E.get(f, {}).get(str(gi), {}).values()
                if v is not None]
        per_gen.append(statistics.mean(vals) if vals else None)
    usable = [i for i, v in enumerate(per_gen) if v is not None]
    if len(usable) < 2 * args.G:
        print(f"  {f:<6}{len(usable):>8}   insufficient usable generations")
        continue
    sigmas = []
    for _ in range(args.R):
        pick = rng.sample(usable, 2 * args.G)
        a = [per_gen[i] for i in pick[:args.G]]
        b = [per_gen[i] for i in pick[args.G:]]
        diffs = [y - x for x, y in zip(a, b)]
        se = (statistics.pstdev(diffs) / (len(diffs) - 1) ** 0.5) or 1e-9
        sigmas.append(abs(statistics.mean(diffs) / se))
    sigmas.sort()
    q = lambda p: sigmas[min(int(p * len(sigmas)), len(sigmas) - 1)]
    thresholds[f] = q(0.95)
    print(f"  {f:<6}{len(usable):>8}{statistics.median(sigmas):>11.2f}"
          f"{q(0.90):>8.2f}{q(0.95):>8.2f}{q(0.99):>8.2f}{max(sigmas):>8.2f}"
          f"   |sigma| > {q(0.95):.2f}")

out = CACHE / f"thresholds_{args.label}.json"
out.write_text(json.dumps(thresholds, indent=1), encoding="utf-8")
print(f"\nwrote {out}")
print("A theoretical 2.0 would have been used before; compare against p95 above.")
