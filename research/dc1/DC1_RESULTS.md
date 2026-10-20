# DC1 — results: calibrated per-field decoding for E7

**Registered:** `PREREG_DC1.md` (8158269), before any DC1 code, fit or
generation; approved by the owner.
- **Sampler options:** 52e1ea2 (in `uie/model.py`, off by default and bit-identical).
- **Fit:** 6851edd (48.5 min on the laptop CPU; settings sha256 `68850b79…`).
- **Notebook and tools:** 53fc247, 6eabd6f, f3bfc0c — all committed before any DC1 number was read.

**Generation:** Kaggle Tesla T4, 2026-09-26: 56 files in 279 min, no errors.
**Validity:** V1–V4 PASS. Byte identity on the laptop CPU, DC1 settings
included: 2,831/2,831 free and 419/419 control. The free-mode scorer reproduced
E7's published free-mode record with zero difference on every field.
**Scoring:** laptop CPU, `cr1_score.py` unchanged; free 84 min and control 67 min in parallel.

**Frozen, and unchanged:**
- E7's weights (`best_e7.pt`);
- R4's decoder and delta (DC1 sits on top);
- gate v2 and its thresholds;
- the adopted state tables and held resampling;
- every published result.

## Verdicts

| mode | registered rule (section 5) | result |
|---|---|---|
| **FULL FREE** | at least 1 target IMPROVED, 0 REGRESSIONS, novelty held, cost at most 2× | **Criteria met.** 2 IMPROVED, 1 FIXED (within noise, section 1), 0 regressions, novelty held, cost **1.01×** |
| **FULL CONTROL** | the same | **Criteria met.** 1 IMPROVED, 0 regressions, novelty held, cost **1.94×** |
| K1–K4 | — | none fired |

**What the criteria do not say: DC1 reduced tells; it did not clear the failing
cells.** The eleven target cells, both arms read at the same n:

| targets (11) | CLEARED | UNDECIDED | CONVICTED |
|---|---|---|---|
| today's E7 | 0 | 5 | 6 |
| DC1 | 1 (within noise, section 1) | 5 | 5 |

**Adoption is the owner's decision.** The two modes are very different trades:
- free mode buys a real gain for almost no compute;
- control mode buys one improved cell for nearly twice the compute.

## 1. FULL FREE — today's E7 against DC1, same seeds, full n

The registered targets (Δ = DC1 − today):

| target | baseline | DC1 | Delta (DC1 - base) | 95% CI | Holm p | reading |
|---|---|---|---|---|---|---|
| benign free PTH | UNDECIDED | CLEARED | -0.0027 | [-0.0118, +0.0064] | 0.906 | FIXED |
| benign free QRY | CONVICTED | CONVICTED | -0.0239 | [-0.0311, -0.0167] | 0.00305 | IMPROVED |
| phishing free PTH | UNDECIDED | UNDECIDED | -0.0088 | [-0.0137, -0.0039] | 0.0228 | IMPROVED |
| phishing free QRY | CONVICTED | CONVICTED | +0.0036 | [-0.0288, +0.0360] | 0.906 | -- |

Each arm's own gate reading, at full n (QRY at 1,200 benign and 356 phishing URLs):

| class | arm | LBL | SUF | SUB | PTH | QRY |
|---|---|---|---|---|---|---|
| benign | today | +0.0021 CLEARED | +0.0193 CLEARED | +0.0080 CLEARED | +0.0363 UNDECIDED | +0.0957 CONVICTED |
| benign | DC1 | +0.0041 CLEARED | +0.0153 CLEARED | +0.0085 CLEARED | +0.0335 CLEARED | +0.0718 CONVICTED |
| phishing | today | +0.0152 CLEARED | +0.0083 CLEARED | +0.0285 CLEARED | +0.0534 UNDECIDED | +0.1141 CONVICTED |
| phishing | DC1 | +0.0171 CLEARED | +0.0076 CLEARED | +0.0314 CLEARED | +0.0446 UNDECIDED | +0.1177 CONVICTED |

- **Benign QRY: the query tell falls by a quarter.** It goes from +0.0957 to
  +0.0718 (−0.0239 [−0.0311, −0.0167]), the largest effect in free mode. It is
  still CONVICTED: the bar is 0.05.
- **Phishing PTH improves** from +0.0534 to +0.0446 (−0.0088). The point estimate
  is now under the bar; its interval still crosses it, so it stays UNDECIDED.
- **Benign PTH is FIXED by the registered definition, but that is not a
  demonstrated change.**
  - Today's cell was UNDECIDED by a hair: the upper end of its 95% interval was
    0.0507, against the 0.05 bar.
  - DC1's mean moved little (+0.0363 → +0.0335). Its five generations agreed more
    closely (SE 0.0052 → 0.0029), so the upper end fell to 0.0416.
  - The paired shift, −0.0027 [−0.0118, +0.0064], is **not** distinguishable from zero.
- **Phishing QRY shows no detectable change** (+0.0036, not significant). As registered, it
  carried no settings: its fit did not confirm on validation.
- **New at full n:** today's E7 is **CONVICTED on queries in both classes**
  (+0.0957 on 1,200 benign query URLs, +0.1141 on 356 phishing). The published
  readings, UNDECIDED at n = 289 and pooled UNDECIDED, had too few query URLs to
  decide; they were not passes.

Every cell, for regressions (Holm over all ten):

| cell | baseline | DC1 | Delta | 95% CI | Holm p (all cells) | pairing |
|---|---|---|---|---|---|---|
| benign free LBL | CLEARED | CLEARED | +0.0020 | [-0.0061, +0.0102] | 1 | paired |
| benign free SUF | CLEARED | CLEARED | -0.0040 | [-0.0075, -0.0005] | 0.276 | paired |
| benign free SUB | CLEARED | CLEARED | +0.0005 | [-0.0033, +0.0044] | 1 | paired |
| benign free PTH | UNDECIDED | CLEARED | -0.0027 | [-0.0118, +0.0064] | 1 | paired |
| benign free QRY | CONVICTED | CONVICTED | -0.0239 | [-0.0311, -0.0167] | 0.00762 | paired |
| phishing free LBL | CLEARED | CLEARED | +0.0019 | [-0.0037, +0.0076] | 1 | paired |
| phishing free SUF | CLEARED | CLEARED | -0.0007 | [-0.0061, +0.0047] | 1 | paired |
| phishing free SUB | CLEARED | CLEARED | +0.0030 | [-0.0045, +0.0104] | 1 | paired |
| phishing free PTH | UNDECIDED | UNDECIDED | -0.0088 | [-0.0137, -0.0039] | 0.0684 | paired |
| phishing free QRY | CONVICTED | CONVICTED | +0.0036 | [-0.0288, +0.0360] | 1 | paired |

| | baseline | DC1 | worsening | 95% CI | counts as harm |
|---|---|---|---|---|---|
| whole-URL novelty | 0.9926 | 0.9930 | -0.0004 | [-0.0009, -0.0000] | no |
| query strings copied from training | 0.2719 | 0.2254 | -0.0465 | [-0.0587, -0.0343] | no |

Cost, next-token predictions per delivered URL: ratio DC1 / baseline **1.01** (limit 2.00).
(baseline 32.7, DC1 32.9)

- **Fewer copied queries.** DC1's free queries repeat a training query verbatim
  less often.
  - The rate falls 27.2% → 22.5%, and the count falls too (2,514 → 2,295), although
    DC1 emits more queries: 10,182 against 9,246, of 325,000 URLs per arm.
  - This fits the benign query settings (temperature 1.062, a later end at
    δ −0.218), but the cause was not isolated.

## 2. FULL CONTROL — DC1 against SC1 arm T (the adopted path)

| target | baseline | DC1 | Delta (DC1 - base) | 95% CI | Holm p | reading |
|---|---|---|---|---|---|---|
| benign P1Q1 SUB | CONVICTED | UNDECIDED | -0.0259 | [-0.0345, -0.0173] | 0.00782 | IMPROVED |
| benign P1Q1 PTH | CONVICTED | CONVICTED | -0.0171 | [-0.0351, +0.0009] | 0.289 | -- |
| benign P1Q1 QRY | CONVICTED | CONVICTED | -0.0154 | [-0.0271, -0.0036] | 0.134 | -- |
| benign P0Q1 QRY | CONVICTED | CONVICTED | -0.0044 | [-0.0278, +0.0189] | 1 | -- |
| phishing P1Q0 LBL | UNDECIDED | UNDECIDED | -0.0009 | [-0.0048, +0.0030] | 1 | -- |
| phishing P1Q0 SUB | UNDECIDED | UNDECIDED | +0.0022 | [-0.0068, +0.0113] | 1 | -- |
| phishing P1Q0 PTH | UNDECIDED | UNDECIDED | -0.0059 | [-0.0208, +0.0091] | 1 | -- |

- **One significant improvement: benign P1 Q1 SUB**, CONVICTED → UNDECIDED
  (−0.0259 [−0.0345, −0.0173]).
- **The other benign P1 Q1 targets moved the right way without reaching
  significance:** PTH −0.0171, whose CI just crosses 0 (upper end +0.0009); QRY
  −0.0154, whose CI excludes 0 but not after Holm. Both stay CONVICTED.
- **Benign P0 Q1 QRY and the phishing P1 Q0 cells show no detectable change.**

Every cell (Holm over all computable cells):

| cell | baseline | DC1 | Delta | 95% CI | Holm p (all cells) | pairing |
|---|---|---|---|---|---|---|
| benign P1Q1 LBL | CLEARED | CLEARED | -0.0077 | [-0.0213, +0.0060] | 1 | paired |
| benign P1Q1 SUF | CLEARED | CLEARED | -0.0011 | [-0.0123, +0.0101] | 1 | paired |
| benign P1Q1 SUB | CONVICTED | UNDECIDED | -0.0259 | [-0.0345, -0.0173] | 0.0246 | paired |
| benign P1Q1 PTH | CONVICTED | CONVICTED | -0.0171 | [-0.0351, +0.0009] | 1 | paired |
| benign P1Q1 QRY | CONVICTED | CONVICTED | -0.0154 | [-0.0271, -0.0036] | 0.469 | paired |
| benign P1Q0 LBL | CLEARED | CLEARED | +0.0017 | [-0.0066, +0.0100] | 1 | paired |
| benign P1Q0 SUF | CLEARED | CLEARED | +0.0022 | [-0.0025, +0.0069] | 1 | paired |
| benign P1Q0 SUB | CLEARED | CLEARED | -0.0025 | [-0.0121, +0.0071] | 1 | UNPAIRED |
| benign P1Q0 PTH | CLEARED | UNDECIDED | -0.0029 | [-0.0110, +0.0053] | 1 | paired |
| benign P1Q0 QRY | -- | -- | -- | -- | -- | not computable |
| benign P0Q1 LBL | CLEARED | CLEARED | -0.0036 | [-0.0171, +0.0100] | 1 | paired |
| benign P0Q1 SUF | CLEARED | CLEARED | +0.0001 | [-0.0097, +0.0100] | 1 | paired |
| benign P0Q1 SUB | -- | -- | -- | -- | -- | not computable |
| benign P0Q1 PTH | -- | -- | -- | -- | -- | not computable |
| benign P0Q1 QRY | CONVICTED | CONVICTED | -0.0044 | [-0.0278, +0.0189] | 1 | paired |
| benign P0Q0 LBL | CLEARED | CLEARED | -0.0001 | [-0.0082, +0.0080] | 1 | paired |
| benign P0Q0 SUF | CLEARED | CLEARED | +0.0052 | [-0.0052, +0.0155] | 1 | paired |
| benign P0Q0 SUB | CLEARED | CLEARED | -0.0144 | [-0.0339, +0.0051] | 1 | UNPAIRED |
| benign P0Q0 PTH | -- | -- | -- | -- | -- | not computable |
| benign P0Q0 QRY | -- | -- | -- | -- | -- | not computable |
| phishing P1Q0 LBL | UNDECIDED | UNDECIDED | -0.0009 | [-0.0048, +0.0030] | 1 | paired |
| phishing P1Q0 SUF | CLEARED | CLEARED | -0.0079 | [-0.0148, -0.0011] | 0.652 | paired |
| phishing P1Q0 SUB | UNDECIDED | UNDECIDED | +0.0022 | [-0.0068, +0.0113] | 1 | paired |
| phishing P1Q0 PTH | UNDECIDED | UNDECIDED | -0.0059 | [-0.0208, +0.0091] | 1 | paired |
| phishing P1Q0 QRY | -- | -- | -- | -- | -- | not computable |
| phishing P0Q0 LBL | CLEARED | CLEARED | -0.0021 | [-0.0148, +0.0106] | 1 | paired |
| phishing P0Q0 SUF | CLEARED | CLEARED | +0.0042 | [-0.0050, +0.0134] | 1 | paired |
| phishing P0Q0 SUB | CLEARED | CLEARED | -0.0012 | [-0.0102, +0.0078] | 1 | paired |
| phishing P0Q0 PTH | -- | -- | -- | -- | -- | not computable |
| phishing P0Q0 QRY | -- | -- | -- | -- | -- | not computable |

- **No regression.** One non-target cell lost CLEARED without meeting the
  registered regression rule: **benign P1 Q0 PTH, CLEARED → UNDECIDED**.
  - Its mean fell (Δ −0.0029, not significant).
  - SC1 arm T had cleared it by a hair (upper end 0.0495, bar 0.05).
  - DC1's mean is lower (0.0383), but one generation read 0.0549, which widened
    the interval to 0.0527.
- **The two verdict changes at the bar are the same kind of event.** This one and
  free benign PTH's FIXED both flipped with the spread of five generations, not with
  a detectable change in the mean. Neither is evidence of a real change.

| | baseline | DC1 | worsening | 95% CI | counts as harm |
|---|---|---|---|---|---|
| whole-URL novelty | 0.9944 | 0.9948 | -0.0004 | [-0.0010, +0.0003] | no |
| query strings copied from training | 0.2927 | 0.2582 | -0.0345 | [-0.0416, -0.0274] | no |

Cost, next-token predictions per delivered URL: ratio DC1 / baseline **1.94** (limit 2.00).

| state | baseline | DC1 | ratio |
|---|---|---|---|
| benign_P1Q1 | 60.0 | 126.3 | 2.10 |
| benign_P1Q0 | 44.3 | 85.7 | 1.94 |
| benign_P0Q1 | 37.1 | 72.5 | 1.95 |
| benign_P0Q0 | 19.0 | 32.9 | 1.73 |
| phishing_P1Q0 | 41.5 | 78.1 | 1.88 |
| phishing_P0Q0 | 27.8 | 50.3 | 1.81 |

- **The blend is what doubles the cost.**
  - At every step of a blended field, the sampler makes a second prediction with the
    controls set to `<P:?> <Q:?>` (`uie/model.py`). In control mode nearly every
    field is blended, so the cost is 1.73–2.10× by state, and 2.10× for benign P1 Q1,
    the state that improved.
  - Temperature and stop bias add no prediction of their own; in free mode, where
    they are the only change, cost rose 1%.
- **Above 2×** (benign P1 Q1, g0), DC1 also needed more draws and more steps per
  draw: 3,161 draws for 3,000 delivered URLs against 3,112, and 119.8 predictions
  per draw against 57.9.

## 3. What DC1 did not do

- **No query cell cleared, in either mode.** Free mode's benign query tell fell by
  a quarter and is still CONVICTED.
  - The registration allowed for this: decoding corrects miscalibration only
    (section 6, "Not claimed").
  - DC1 does not show whether the remaining tell is structure E7 did not learn, or
    miscalibration that three numbers per field cannot reach. Unknown; more
    evidence required.
- **Free phishing QRY had no settings** and shows no detectable change.
- **Control mode's gains are mostly not significant**, apart from one subdomain cell.
- **The whole-URL reading barely changed.** In free mode, the point estimates of
  gate v2's auxiliary FULL reading (not a verdict cell; no contrast registered) are
  benign +0.2168 → +0.2189 and phishing +0.4018 → +0.4012.

## 4. For the owner's decision

| option | what it gives | what it does not give | cost |
|---|---|---|---|
| adopt DC1 for **FULL FREE** | benign QRY tell −25%; phishing PTH −16%; 4.7 points fewer copied queries; no regression | no cell robustly cleared: benign PTH's CLEARED is within noise, and both query cells stay CONVICTED | 1.01× compute |
| adopt DC1 for **FULL CONTROL** | benign P1 Q1 SUB out of CONVICTED; 3.5 points fewer copied queries; no regression | benign P1 Q1 PTH and QRY, and P0 Q1 QRY, stay CONVICTED | 1.94× compute (2.10× in benign P1 Q1) |

**Not decided here, and not run:**
- **A cheaper control variant**, temperature and stop bias without the blend, would
  drop the second prediction, so its cost would be close to today's. Whether it
  keeps any of the gain is unknown. It was not tested; testing it would be a new
  registration, because the registration forbids pruning settings after reading
  results.
- **Whether and how to adopt** is the owner's call.

## 5. Settings used (applied only where validation confirmed them)

| class | mode | LBL | SUB | PTH | QRY |
|---|---|---|---|---|---|
| benign | free | T 1.013, d -0.000 | off (T 1.026, d -0.077) | T 1.031, d -0.107 | T 1.062, d -0.218 |
| benign | P1Q1 | T 1.052, d -0.259, w 0.45 | T 1.080, d -0.016, w 1.08 | T 1.045, d -0.409, w 0.97 | T 1.066, d -0.165, w 0.87 |
| benign | P1Q0 | T 1.014, d -0.164, w 0.59 | T 1.045, d +0.173, w 1.42 | T 1.031, d -0.155, w 1.00 | n/a |
| benign | P0Q1 | T 1.030, d -0.336, w 0.30 | T 1.058, d +0.024, w 0.63 | n/a | T 1.056, d -0.136, w 0.46 |
| benign | P0Q0 | T 1.022, d -0.004, w 0.46 | T 1.041, d -0.194, w 1.00 | n/a | n/a |
| phishing | free | T 1.034, d +0.030 | T 1.037, d -0.030 | T 1.046, d +0.081 | off (T 1.021, d -0.444) |
| phishing | P1Q1 | T 1.092, d -0.171, w 0.88 | T 1.072, d -0.484, w 0.67 | T 1.040, d -0.227, w 0.60 | T 1.031, d -0.381, w 0.61 |
| phishing | P1Q0 | T 1.044, d +0.054, w 1.07 | T 1.027, d -0.008, w 1.06 | T 1.034, d +0.051, w 0.69 | n/a |
| phishing | P0Q1 | T 1.060, d -0.149, w 1.20 | T 1.076, d -0.466, w 0.56 | n/a | off (T 0.994, d -0.332, w 0.66) |
| phishing | P0Q0 | T 1.017, d +0.004, w 1.03 | T 1.027, d -0.007, w 0.88 | n/a | n/a |

## Validity

**VERIFY: PASS**

- V3 identity, free: 2831/2831
- V3 identity, control: 419/419
- V4 scorer, benign: max |x - published| 0.0e+00
- V4 scorer, phishing: max |x - published| 0.0e+00
- failures: none

## Reproduce

    python research/dc1/dc1_fit.py --checkpoint data/best_e7.pt --bundle data/bundle2 --out data/dc1
    python research/dc1/make_code_bundle.py     # then DC1_generate.ipynb on Kaggle
    python research/dc1/dc1_verify.py --dc1 data/dc1 --bundle data/bundle2 --checkpoint data/best_e7.pt
    python research/dc1/dc1_score.py --dc1 data/dc1 --bundle data/bundle2
    python research/dc1/dc1_compare.py --dc1 data/dc1 --sc1 data/sc1 --bundle data/bundle2
    python research/dc1/dc1_tables.py --dc1 data/dc1
