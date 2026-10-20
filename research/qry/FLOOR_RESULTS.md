# FL — results: benign QRY's gate reading WAS a floor artifact

**Run:** 2026-09-23, `research/qry/floor_check.py`, registered in
`PREREG_FLOOR.md` (commit 3f544cc) before any number existed.
Nothing about the gate was changed.

## FL1 — H_UF SUPPORTED

The gate's three benign QRY floors, recomputed at its own split seeds
20260806-08: **F(360) = 0.5410**, against the gate's recorded 0.5374 (within
0.004) and the typical 0.4936 (NS, 20 draws at n=361). Seed 20260806 alone reads
0.52-0.57 across n = 330-390. The gate's floor is reproduced, and it was drawn
high.

## FL2 — the exact gate cell, re-scored with 30 split seeds

**Reproduction: 15/15** of R4's stored `excess_draws` matched to 4 decimals, so
this is the gate's own cell and not a lookalike. Per-generation n:
353, 369, 366, 359, 357.

| | x | SE | floor | AUC | verdict (section-26.1 rule) |
|---|---:|---:|---:|---:|---|
| **gate, D=3** | +0.0300 | 0.0096 | 0.5374 | 0.5675 | UNDECIDED |
| **D=30** | **+0.0619** | 0.0085 | **0.4973** | 0.5592 | **UNDECIDED** |

Generation means at D=30: 0.0759, 0.0624, 0.0394, 0.0515, 0.0802 — four of five
above the 0.05 bar. t vs the bar = +1.40 (t_crit 2.776).

> **Note, 2026-09-23 - SE convention.** SE here is sd / sqrt(G - 1), the published
> convention. Under section 26.1's sd / sqrt(G), which gate v2 encodes, the D=30
> SE is 0.0076 and t = +1.57 - **still UNDECIDED.**

**REGISTERED VERDICT: UNDECIDED. Benign QRY is NOT cleared under the adopted
decoder.**

Every registered prediction under H_UF held:

    floor near 0.494 (+-0.01)      0.4973      held
    x near ~0.07                   +0.0619     held
    not CLEARED                    UNDECIDED   held

Of the +0.0319 change in x, the floor accounts for **+0.0401** (0.5374 -> 0.4973)
and the AUC for -0.0083. The low reading was the floor.

## The structural finding: the gate's SE cannot see floor noise

This is larger than one cell, so it is stated carefully.

`run_eval` seeds each floor with `seed + dsi` — the split index only. All five
generations therefore share the **same three floors**. The section-26.1 standard
error is computed **across generation means**, so a floor offset common to all
generations never appears in it. It is invisible to the estimator.

Measured here: the gate's D=3 estimate was +0.0300 with SE 0.0096, so its 95%
interval was roughly [+0.003, +0.057]. The D=30 value, **+0.0619, lies outside
it.** The shared floor offset (0.040) was **four times the reported SE.**

Where it bites: NS measured single-floor sd at **~0.025-0.035 for n = 250-361**
against **~0.012-0.017 at n = 1200**. So reduced-n cells — both QRY cells, and
anything else the pool forces below 1,200 — carry the largest hidden offset,
and their verdicts are the most over-confident. At n=1200 the effect is smaller
but not zero.

## What is NOT decided here — for the user

Registered as out of scope: **any change to the gate.** The options, none
implemented:

1. **Seed floors per (generation, split)**, so the across-generation SE sees
   floor noise. This changes which floors every published cell used, so no
   published number would reproduce under the new seeding.
2. **Raise D** for reduced-n cells, averaging the shared offset down. Cheap: FL2's
   30 seeds took ~9 minutes for one cell.
3. Both.

Any of these changes the canonical gate, and every published verdict would need
re-reading under it. It warrants its own pre-registration.

## Caveats

- The cell is measured at n ~ 360, the gate's own n. NS found — descriptively,
  not at its registered standard — that tells read lower at small n. If that
  holds, +0.0619 understates the n=1200 value. Unmeasured.
- UNDECIDED is not CONVICTED: t = +1.40 against the bar. The honest statement is
  "not cleared", not "failed".
