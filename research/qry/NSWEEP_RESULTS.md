# NS — results: is the gate's excess biased at reduced n?

**Run:** 2026-09-23, 1,680 draws, `research/qry/n_sweep.py`, registered in
`PREREG_NSWEEP.md` (commit eee2194) before any number existed. Draws in
`data/qry/nsweep_draws.jsonl`, summary in `data/qry/nsweep_summary.json`.
Read-only; nothing adopted.

## 1. REGISTERED VERDICT: VOID

The validity condition required the null control (SUF) in **both** classes to
show |Delta| < 0.01 **and** a CI covering 0.

    benign SUF     Delta +0.0003  [-0.0158, +0.0163]   passes
    phishing SUF   Delta +0.0121  [-0.0049, +0.0292]   FAILS the point criterion

Phishing SUF's CI covers 0, so there is **no statistical evidence** that the null
drifts. But the rule as registered says VOID, and it stands.

### The rule was badly designed, and that is my error

The CI half-widths on the null controls came out at ~0.016 — **wider than the
0.01 tolerance the rule demanded.** With a true null and SE ~0.008, a single
control exceeds |Delta| = 0.01 by chance about 22% of the time, so two controls
**both** pass only ~61% of the time. The validity check would have voided a
perfectly healthy instrument roughly 4 times in 10.

The correct form is an **equivalence test**: the null control's whole CI must lie
inside a tolerance band (e.g. +-0.02), with R chosen from this run's variance so
the CI is narrow enough to fit. "The point estimate is small" is not evidence of
flatness; "the interval sits inside the band" is. This computation is post hoc
and explains the weakness of the rule — it does **not** overturn the verdict.

The registered H_LC branch would also have failed independently: its dose-response
ordering (null < weak < bar) broke because the weak-tell mean came out at -0.001,
below the null mean of +0.006.

## 2. What the data show — DESCRIPTIVE, not a registered conclusion

Delta = E[excess](n=1200) - E[excess](n=250), OLS on ln n, R = 20 per cell:

| cell | role | Delta excess [95% CI] | Delta AUC | Delta floor |
|---|---|---|---:|---:|
| benign SUF | null | +0.0003 [-0.016, +0.016] | +0.007 | +0.007 |
| phishing SUF | null | +0.0121 [-0.005, +0.029] | +0.003 | -0.009 |
| benign LBL | weak | +0.0192 [+0.000, +0.038] | +0.010 | -0.009 |
| benign SUB | weak | -0.0067 [-0.014, +0.000] | -0.000 | +0.007 |
| phishing LBL | weak | -0.0156 [-0.032, +0.001] | +0.003 | +0.018 |
| **benign PTH** | **bar** | **+0.0203 [+0.003, +0.038]** | +0.024 | +0.003 |
| **phishing SUB** | **bar** | **+0.0410 [+0.023, +0.059]** | +0.034 | -0.007 |
| **phishing PTH** | **bar** | **+0.0360 [+0.021, +0.051]** | +0.038 | +0.002 |
| benign FULL | strong | +0.0781 [+0.061, +0.095] | +0.082 | +0.004 |
| phishing FULL | strong | +0.0800 [+0.066, +0.094] | +0.090 | +0.010 |

Read plainly:

- **The gate's assumption ("weaker, not wrong") fails its own registered condition
  on all three bar-scale fields by a wide margin.** Every bar-scale and strong
  field shows excess rising from n=250 to n=1200, with CIs excluding 0.
- **The rise is in the AUC term, not the floor** — the learning-curve mechanism
  the pre-registration described. Floors are flat within noise.
- It **scales with the tell**: ~+0.02 to +0.04 at bar scale, ~+0.08 for FULL.
- Null controls are statistically flat; weak tells are indistinguishable from
  flat at this precision.

So the substance points toward H_LC for fields with a genuine tell, **but it did
not meet the registered standard**, and it must not be quoted as established.

**Practical implication, stated as a risk rather than a finding:** a cell near
the 0.05 bar measured at n~250 could plausibly read 0.02-0.04 lower than the same
cell at n=1200. Reduced-n "ok" verdicts near the bar should not be relied on to
acquit until this is confirmed.

## 3. QRY, secondary and descriptive

| arm | Delta over its own range | per-n excess |
|---|---|---|
| benign typical (150-476) | +0.043 [+0.024, +0.062] | 0.109 -> 0.149 |
| benign temperature (150-476) | +0.043 [+0.018, +0.068] | 0.091 -> 0.122 |
| phishing typical (150-356) | +0.056 [+0.029, +0.083] | 0.117 -> 0.163 |
| phishing temperature (150-356) | +0.058 [+0.032, +0.083] | 0.132 -> 0.176 |

QRY excess rises with n inside its available range, consistently across four
arms — the same direction as the main grid.

## 4. The benign QRY question — n does NOT explain the gate's low number

At **n=361, the gate's own n**, the sweep reads benign QRY (temperature) at
**+0.107**. The gate's canonical cell read **+0.0300**. The n-effect cannot
account for a gap that large at identical n.

**The floor does.** The gate cell's floor averaged **0.5374**. At n=361 the sweep's
20 floor draws average **0.4936** (sd 0.0258, range 0.448-0.533) — the gate's
mean sits **above every one of them.**

The script printed z = 6.57, treating the gate's 15 draws as independent. **They
are not:** `run_eval` seeds each floor with `seed + dsi`, the split index only, so
the 15 draws share **3** independent floors. The honest figure is
z ~ (0.5374 - 0.4936) / (0.0258 / sqrt 3) ~ **2.9** — unusual, not extreme.

If the gate cell's floor had been typical, its excess would have been roughly
0.5675 - 0.494 ~ **+0.074, above the bar.** That is an inference from one cell,
not a measurement, and it is the leading hypothesis for why benign QRY looked
cleared: **an unlucky floor draw, not a content improvement and not an n effect.**

**A structural weakness this exposes.** Floor noise at reduced n is large
(sd ~0.025-0.035 at n=250-361 against ~0.012 at n=1200), and because the floor is
seeded per split rather than per generation, the G=5 generations cannot average
it away. At reduced n the floor alone contributes a standard error of ~0.015 —
about a third of the 0.05 bar — **shared by every draw in the cell.**

## 5. Status

    NS  RUN 2026-09-23   REGISTERED VERDICT: VOID (null-control tolerance tighter
                         than the design could resolve -- my design error)
                         DESCRIPTIVE: bar-scale and strong tells rise with n via
                         the AUC term; the gate's assumption fails its own
                         condition on all three bar-scale fields
                         benign QRY: still UNRESOLVED; the floor, not n, is the
                         leading explanation for its low gate reading

## Operational note

The first two attempts died of memory: long-lived workers grew to ~3 GB each
(text-vectorizer allocations never returned to the OS), taking the laptop to
352 MB free. Fixed by recycling workers every 25 tasks plus a free-RAM watchdog.
Neither changes any value: 4 draws reproduced to the last digit across separate
runs on different workers.
