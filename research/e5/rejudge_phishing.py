"""Phishing delta on the calibrated harness. E6 is NOT re-run.

E6 is dead regardless of this scale: it failed its primary mechanistic
prediction (entropy curve 5.0x -> 4.5x, no flattening), which is teacher-forced
and untouched by discriminator variance. A clean phishing result would leave it
neutral-with-no-benefit at +385 parameters, so re-running it would only restate
the autopsy.

Phishing calibration is DEFERRED, not skipped. Benign p95 thresholds are
2.53-3.08 and family-wise 4.30-6.03. Phishing's pool is noisier (24,081 held-out
rows against 113,277) so its thresholds would be at least as permissive.
Decision rule, fixed BEFORE seeing the number:

    all |sigma| < 2.5   -> clears even the benign thresholds; adopt, no
                           calibration needed
    anything >= 2.5     -> borderline; run the phishing calibration before
                           judging

delta = 0.30, the empirical optimum from the powered sweep (+0.87 chars at
2.0 sigma; the derived 0.06 did nothing, which S1 explains -- phishing shows no
own-trajectory hazard inflation).
"""
from __future__ import annotations

import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.e5.harness import POOL, compare, generate_arm, rows

LABEL, G, D, N, DELTA = "phishing", 5, 3, 1800, 0.30
SEEDS = [30000 + 17 * i for i in range(G)]
E5 = str(ROOT / "data/best_e5.pt")

pool = rows(POOL[LABEL])
print(f"phishing delta={DELTA}  G={G} D={D} n={N}", flush=True)
base = generate_arm(E5, LABEL, SEEDS, N, tag="E5")
arm = generate_arm(E5, LABEL, SEEDS, N, bias=DELTA, tag="delta")
worst = compare(base, arm, pool, LABEL, ("E5", "E5+d"),
                tuple(range(1, D + 1)), N)
print(f"\n  worst |sigma| = {worst:.1f}")
print("  -> " + ("clears the benign thresholds; ADOPT without further "
                 "calibration" if worst < 2.5 else
                 "BORDERLINE; run the phishing calibration before judging"))
