"""Re-measure E6 and delta on the calibrated harness.

Both were rejected on a statistic (G=1, discriminator-only bars) whose null
distribution was never measured. The calibration since shows the NEW statistic
reaches |sigma| 8.84 between two draws of the SAME distribution, with per-field
p95 at 2.53-3.08 and family-wise (4 fields) thresholds near p99: 4.30-6.03.

E6 was rejected at 3.9 / 4.3 sigma, delta at 3.6 / 3.3. Every one of those sits
below the family-wise thresholds -- but they were computed with a different
statistic, so this re-runs both arms properly rather than rescoring old numbers.

Arms share the pool's generation seeds so the comparison is paired.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.e5.harness import (CACHE, FIELDS, POOL, compare, generate_arm,
                                 rows)

LABEL = "benign"
G, D, N = 5, 3, 1800
SEEDS = [30000 + 17 * i for i in range(G)]          # same schedule as the pool
E5 = str(ROOT / "data/best_e5.pt")
E6 = str(ROOT / "data/best_e6.pt")

thr = json.loads((CACHE / f"thresholds_{LABEL}.json").read_text(encoding="utf-8"))
FAMILY = {"LBL": 6.03, "SUF": 4.30, "SUB": 5.04, "PTH": 5.37}

pool = rows(POOL[LABEL])
base = generate_arm(E5, LABEL, SEEDS, N, tag="E5")

print("\n" + "=" * 78)
print("RE-JUDGE 1: E6 factorised head vs E5")
e6 = generate_arm(E6, LABEL, SEEDS, N, tag="E6")
compare(base, e6, pool, LABEL, ("E5", "E6"), tuple(range(1, D + 1)), N)

print("\n" + "=" * 78)
print("RE-JUDGE 2: E5 + path-scoped delta=0.22 vs E5")
dl = generate_arm(E5, LABEL, SEEDS, N, bias=0.22, tag="delta")
compare(base, dl, pool, LABEL, ("E5", "E5+d"), tuple(range(1, D + 1)), N)

print("\nthresholds  per-field p95: " +
      "  ".join(f"{k} {v:.2f}" for k, v in thr.items()))
print("            family-wise  : " +
      "  ".join(f"{k} {v:.2f}" for k, v in FAMILY.items()))
print("A verdict counts only if |sigma| exceeds the FAMILY-WISE threshold,")
print("because all four fields are scanned.")
