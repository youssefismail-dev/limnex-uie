# E9 — results: the length-neutral objective

- **Registration:** `PREREG_E9.md` (375e8a4), before any code or run.
- **Code:** 3b29288 (training and generation, run by the owner on Kaggle); laptop
  tools 440be86, ffc19e7 and 2c20958.
- **Read:** 2026-09-28, after VERIFY passed.
- **Numbers:** Appendix A is `e9_tables.py`'s output, rendered from the reports in
  `data/e9/` and pasted by script. Every number in the text appears there or has a
  declared source: a rounding, a formula, a cited document or the registered code.
  - `e9_check_results.py` checks this, checks table rows cell by cell in order, and
    checks that the appendix still equals the script's output.
  - It catches 13 of 14 planted errors (`--self-test`). The miss is its stated
    limit: a right number in the wrong place in the prose.

## In plain words

- **The cause of three biases is confirmed** (K9a).
  - Two models were trained from the same starting weights. Only the weighting in
    the training score differed.
  - The old rule gave every URL the same total say, however long it was. So each
    piece of a short URL counted up to about ten times as much as a piece of a long
    one. The model learned short URLs' habits: end early, stop query values early,
    add few parameters.
  - With every piece counted equally, the three habits match the real data. For
    benign URLs: 1.164 → **1.000**, 1.058 → **1.018**, 0.892 → **0.997**, where 1.0
    means "as often as real URLs". Phishing moves the same way.
- **Free-mode paths are fixed, with no patch.**
  - The share of URLs with a path rises from 58.7% to **75.6%** (benign) and from
    29.6% to **38.2%** (phishing), inside the target ranges.
  - Both free path cells now clear the gate. Phishing PTH was UNDECIDED in E7 and
    in today's system.
  - The frozen delta, the decode-time patch for early path endings, is not needed.
    On the new model it made both path cells worse.
- **Queries improve only part way.**
  - The share with a query rises from 3.8% to **4.6%** (benign) and from 2.0% to
    **2.9%** (phishing). The targets were 6.0–7.3% and 4.2–5.2%, so **K9b fails**.
  - The benign query cell improves from 0.063 to **0.035**, under the 0.05 bar, but
    stays UNDECIDED.
  - Why queries stay rare is **Unknown**. The weighting explains less of it than the
    earlier arithmetic suggested.
- **Nothing got worse beyond the retraining allowance** (K9c). **Copying fell**, from
  27–28% to 19–21% of generated queries (K9d).
- **Scoreboard,** today's system against `e9_ln` / raw: of 31 measurable gated cells,
  those not yet CLEARED fall from 9 to 6, and CONVICTED ones from 4 to 3.
- **What the registered rules say:** K9b failed, so `e9_ln` is not an automatic
  candidate; adoption is your decision. No query-data experiment is licensed.
- **A by-product:** three separate trainings of E7's recipe differ by up to 8 points
  in honouring (following the controls) in the hardest control states, but by at
  most 0.011 in any free gate cell.

## 1. Validity: PASS

| check | what was checked | result |
|---|---|---|
| V1 | all 160 registered files, each with its registered count, hash and seed; both checkpoints equal what training recorded and what generation used; E7's checkpoint and the suffix tables as registered; training and generation ran one code commit, 3b29288, which is in this history | pass |
| V2 | byte identity, regenerated on the laptop: 1,036 of 1,036 URLs in each of the 4 free conditions, 269 of 269 in each of the 3 control conditions (registered minimum 250) | pass |
| V3 | `uie/` and the two common modules unchanged since 3b29288 (VERIFY); flags off reproduce today's training byte for byte (`tests/test_e9.py`, 10 of 10 pass) | pass |
| V4 | the scorer reproduces E7's published free cells exactly: max \|x − published\| = 0.00e+00 in both classes | pass |
| V-id | both arms start from identical weights (hash `26926388bb57…`); tokenizer, model and every training setting equal E7's; `loss_norm` and C as registered; data counts equal E7's | pass |

## 2. Training

| arm | objective | final validation loss: benign / phishing / mean | best step | minutes |
|---|---|---|---|---|
| `e9_base` | per-batch mean (today's) | 1.6734 / 1.5301 / 1.6017 | 18,000 | 72.1 |
| `e9_ln` | length-neutral, C = 14,397.9 | 1.6671 / 1.5249 / 1.5960 | 18,000 | 72.6 |

- `e9_ln` ends with the lower validation loss, by 0.0057 on the mean. With one
  training per arm, how that compares with retrain noise is Unknown.
- **Clipping** acts on the same per-batch gradient in both arms, as D4 intended.
  After step 1,000, `e9_ln`'s batches of short rows (under 30 scored tokens per row)
  were clipped on 24.8% of steps, against 12.7%. In the longest band (120 or more)
  the figures are 2.2% against 3.1%.
- Clipping short batches more trims their weight a little. That pushes the opposite
  way from the removed bias: toward ending late, stopping values late and adding
  parameters. Whatever its size, no family in section 3 ends more than 2.4% past 1
  in that direction (the largest is phishing F2, 0.976).

## 3. R1, the mechanism: P0 and K9a pass

Each model's `best.pt` is teacher-forced on E7's training rows, with the controls
hidden.
- The **plain ratio** is how often the model takes a decision, divided by how often
  the data does. 1.0 is perfect.
- The 95% CIs come from a row bootstrap.
- **Closure** is the share of `e9_base`'s gap to 1 that `e9_ln` removes.

**Benign** (decides)

| family | E7 | `e9_base` | `e9_ln` [95% CI] | `e9_ln` weighted | closure |
|---|---|---|---|---|---|
| F1 · ending the URL inside a path (positions 0–50) | 1.184 | 1.164 | **1.000** [0.990, 1.010] | 0.852 | 1.00 |
| F2 · stopping a query value (positions 0–25) | 1.061 | 1.058 | **1.018** [1.009, 1.027] | 0.963 | 0.69 |
| F3 · another parameter at a value's end | 0.845 | 0.892 | **0.997** [0.987, 1.008] | 1.162 | 0.97 |
| TAIL · stopping a query value (positions 50–100); not a criterion | 1.259 | 1.228 | 1.079 [1.017, 1.146] | 0.938 | 0.65 |

**Phishing** (must agree in direction)

| family | E7 | `e9_base` | `e9_ln` [95% CI] | `e9_ln` weighted | closure |
|---|---|---|---|---|---|
| F1 | 1.113 | 1.081 | 0.995 [0.986, 1.003] | 0.913 | 0.94 |
| F2 | 1.098 | 1.105 | 0.976 [0.961, 0.992] | 0.878 | 0.77 |
| F3 | 0.724 | 0.755 | 0.939 [0.915, 0.964] | 1.193 | 0.75 |
| TAIL | 1.836 | 1.809 | 1.401 [1.314, 1.502] | 1.199 | 0.50 |

- **P0 passes.** `e9_base` shows E7's pattern: F1 and F2 above 1, F3 below, and its
  weighted ratios closer to 1 than its plain ones.
- **K9a passes.** Closure is 1.00, 0.69 and 0.97, each at least 0.5, and each CI
  excludes `e9_base`'s value. Phishing is closer to 1 in every family.
- **The moves are larger than the two old-objective trainings' differences.**
  - E7 and `e9_base` differ by 0.020, 0.003 and 0.047 in F1–F3 (benign).
  - `e9_ln` moves 0.164, 0.040 and 0.105 away from `e9_base`.
  - Two samples show this much and no more.
- **The weighted ratios mirror the plain ones** (benign 0.852, 0.963, 1.162), as a
  plainly calibrated model's should. The test's logic is in
  `FINDING_LENGTH_WEIGHTING.md` §3.
- **The long-value tail is only partly closed.** TAIL's CI excludes 1 in both
  classes, so long query values still stop too early (benign 1.079, phishing 1.401).

## 4. R2, free-mode emission: K9b fails, on queries

| condition | benign PTH | benign QRY | phishing PTH | phishing QRY |
|---|---|---|---|---|
| `e9_base` / resting (today's system) | 58.67% | 3.81% | 29.64% | 2.04% |
| `e9_base` / raw | 58.97% | 3.33% | 29.75% | 1.72% |
| **`e9_ln` / raw (the native candidate)** | **75.58%** | **4.58%** | **38.20%** | **2.94%** |
| `e9_ln` / resting | 75.32% | 5.01% | 38.29% | 3.55% |
| registered bands | 70.37–76.37% | 5.96–7.28% | 36.51–42.51% | 4.21–5.15% |

**K9b fails:** both query rates are below their bands. Both path rates are inside
theirs.

**How much of each gap the change closed.** This compares `e9_base` / raw with
`e9_ln` / raw, the same decode, against the data rates in
`FINDING_LENGTH_WEIGHTING.md` §2.3:

| | benign PTH | benign QRY | phishing PTH | phishing QRY |
|---|---|---|---|---|
| data rate | 73.3% | 6.53% | 39.4% | 4.51% |
| share of the gap closed | 116% | 39% | 88% | 44% |

- 116% means benign paths now slightly overshoot the data rate (75.58% against
  73.3%). They are still inside the band.
- **Queries closed less than the finding implied.**
  - The finding's arithmetic attributed "essentially all of E7's path shortfall and
    about 70% of its query shortfall" to the weighting.
  - Paths behaved as predicted. Queries closed 39% (benign) and 44% (phishing) of
    the gap; at the resting decode the figures are 44% and 61%.
  - Something besides the weighting holds query presence down. What it is is
    **Unknown**.
- **An observation.** The frozen delta raises query emission in both arms and both
  classes, by about 0.3 to 0.6 points.
  - Each rise is at least 2.8 standard errors, treating the two decodes as
    independent samples of 40,000 URLs.
  - So discouraging path endings sends some URLs on to a query.
  - What this implies for the shortfall is Unknown.

## 5. R3, the gate: K9c passes

**Free mode** (canonical gate v2, G = 5, D = 10):

| class | field | E7 | `e9_base` / resting | `e9_base` / raw | **`e9_ln` / raw** | `e9_ln` / resting |
|---|---|---|---|---|---|---|
| benign | LBL | +0.0017 CLEARED | +0.0057 CLEARED | +0.0055 CLEARED | +0.0073 CLEARED | +0.0068 CLEARED |
| benign | SUF | +0.0165 CLEARED | +0.0147 CLEARED | +0.0131 CLEARED | +0.0113 CLEARED | +0.0131 CLEARED |
| benign | SUB | +0.0048 CLEARED | +0.0022 CLEARED | +0.0056 CLEARED | +0.0031 CLEARED | −0.0004 CLEARED |
| benign | PTH | +0.0369 UNDECIDED | +0.0348 CLEARED | +0.0453 UNDECIDED | **+0.0078 CLEARED** | +0.0126 CLEARED |
| benign | QRY | +0.0541 UNDECIDED | +0.0630 UNDECIDED | +0.0504 UNDECIDED | **+0.0351 UNDECIDED** | +0.0317 UNDECIDED |
| phishing | LBL | +0.0238 CLEARED | +0.0197 CLEARED | +0.0203 CLEARED | +0.0122 CLEARED | +0.0063 CLEARED |
| phishing | SUF | +0.0054 CLEARED | +0.0037 CLEARED | +0.0064 CLEARED | +0.0080 CLEARED | +0.0071 CLEARED |
| phishing | SUB | +0.0334 CLEARED | +0.0353 CLEARED | +0.0327 CLEARED | +0.0189 CLEARED | +0.0176 CLEARED |
| phishing | PTH | +0.0604 UNDECIDED | +0.0533 UNDECIDED | +0.0527 UNDECIDED | **+0.0315 CLEARED** | +0.0453 UNDECIDED |
| phishing | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | +0.0923 UNDECIDED |

**Control mode** (CR1's instrument, native output). The table shows every cell that
is not CLEARED in at least one condition. Every other measurable control cell is
CLEARED in all three conditions (Appendix A).

| state | field | `e9_base` / resting | `e9_base` / raw | **`e9_ln` / raw** |
|---|---|---|---|---|
| benign P0Q1 | QRY | +0.0898 CONVICTED | +0.0884 CONVICTED | +0.0940 CONVICTED |
| benign P1Q0 | PTH | +0.0423 UNDECIDED | +0.0548 UNDECIDED | **+0.0119 CLEARED** |
| benign P1Q1 | SUB | +0.0728 CONVICTED | +0.0705 CONVICTED | **+0.0497 UNDECIDED** |
| benign P1Q1 | PTH | +0.0842 CONVICTED | +0.0828 CONVICTED | +0.0930 CONVICTED |
| benign P1Q1 | QRY | +0.0838 CONVICTED | +0.0861 CONVICTED | **+0.0636 CONVICTED** |
| phishing P1Q0 | LBL | +0.0365 UNDECIDED | +0.0401 UNDECIDED | **+0.0340 CLEARED** |
| phishing P1Q0 | SUB | +0.0404 CLEARED | +0.0496 UNDECIDED | +0.0473 UNDECIDED |
| phishing P1Q0 | PTH | +0.0526 UNDECIDED | +0.0501 UNDECIDED | **+0.0373 CLEARED** |

- **K9c passes.** No gated cell's verdict worsens with x rising more than 0.0141,
  against today's system or against `e9_base` / raw.
  - The only verdict that worsens is phishing P1Q0 SUB: CLEARED → UNDECIDED, x
    +0.0069.
  - The largest rise in x is phishing P0Q0 LBL, +0.0115, which stays CLEARED.
- **FULL** is the whole URL, reported but never gated. It falls from 0.2186 to
  0.1483 (benign) and from 0.4033 to 0.3850 (phishing).

**Cells not yet CLEARED:**

| | today's system (`e9_base` / resting) | `e9_ln` / raw |
|---|---|---|
| free (9 measurable) | 2: benign QRY, phishing PTH | 1: benign QRY |
| control (22 measurable) | 7 | 5 |
| **total** | **9**, of which 4 CONVICTED | **6**, of which 3 CONVICTED |

## 6. The frozen delta on the new model

| path cell, x | `e9_base`: raw → resting | `e9_ln`: raw → resting |
|---|---|---|
| benign PTH | +0.0453 → +0.0348 | +0.0078 → +0.0126 |
| phishing PTH | +0.0527 → +0.0533 | +0.0315 CLEARED → +0.0453 UNDECIDED |

- On `e9_base` the delta lowered benign's path x (by 0.0105) and left phishing's
  flat (+0.0006). On `e9_ln` it raises both: benign by 0.0048, and phishing by
  0.0138, from CLEARED to UNDECIDED.
- This is the direction the registration predicted ("the delta over-corrects a
  model that no longer needs it", §8).
  - No test of these differences was registered, so this is a direction, not a
    tested result.
  - It supports D5's choice to read the native candidate without the delta.
- **On the query cell the delta moves the other way:** benign 0.0351 → 0.0317 (both
  UNDECIDED). Phishing QRY becomes computable (0.0923, UNDECIDED) because the delta
  raises the number of queries.

## 7. R4, honouring (reported, not an E9 criterion)

| state | `e9_base` / resting | `e9_base` / raw | `e9_ln` / raw | range of the three old-recipe trainings (E7, `e8_base`, `e9_base`) |
|---|---|---|---|---|
| benign P1Q1 | 93.74% | 93.43% | 93.67% | 93.74–96.11% |
| benign P1Q0 | 98.11% | 98.13% | 98.12% | 98.11–98.71% |
| benign P0Q1 | 87.67% | 87.67% | 94.95% | 87.67–95.79% |
| benign P0Q0 | 98.33% | 98.35% | 96.89% | 97.85–98.33% |
| phishing P1Q1 | 81.56% | 79.86% | 86.76% | 81.56–89.52% |
| phishing P1Q0 | 95.91% | 96.32% | 96.50% | 95.91–97.35% |
| phishing P0Q1 | 82.68% | 82.68% | 89.76% | 82.68–90.76% |
| phishing P0Q0 | 98.67% | 98.75% | 97.95% | 98.15–98.67% |

- **The rises in the hard states are not attributed.**
  - In benign P0Q1, phishing P1Q1 and phishing P0Q1, `e9_ln` is 5.2 to 7.3 points
    above `e9_base`.
  - But it lands inside the range the three old-recipe trainings already span, and
    `e9_base` is the lowest of the three there.
- **P0Q0 is slightly lower.**
  - `e9_ln` sits below all three old-recipe trainings in benign P0Q0 (by 0.96
    points) and phishing P0Q0 (by 0.20). Benign P1Q1 is 0.07 below.
  - P0Q0 asks for neither a path nor a query. A model that writes both more readily
    would honour it slightly less [HYPOTHESIS].
  - Three trainings give a rough range, not a noise model.

## 8. R6, guardrails: K9d passes

| condition | mode | novel URLs | copied queries |
|---|---|---|---|
| `e9_base` / resting | free | 99.16% | 27.23% of 2,339 |
| `e9_base` / resting | control | 99.44% | 27.73% of 36,676 |
| **`e9_ln` / raw** | free | **99.53%** | **19.26% of 3,011** |
| **`e9_ln` / raw** | control | **99.67%** | **20.53% of 38,722** |

- Copying falls by 7.97 points in free mode [95% CI 5.68, 10.26] and by 7.20 in
  control [6.59, 7.81].
- Novelty rises by 0.37 and 0.24 points; both CIs exclude 0.
- Neither guardrail worsens.

## 9. R5 and §7: no data experiment is licensed

Q-VAL is the gate's query cell scored against held-out queries. Q-TRAIN is the same
cell scored against the model's own training queries.

| condition (benign) | Q-VAL | Q-TRAIN | generated queries per draw |
|---|---|---|---|
| `e9_base` / resting | +0.0630 | +0.0635 | 305 |
| `e9_base` / raw | +0.0504 | +0.0537 | 266 |
| `e9_ln` / raw | +0.0351 | +0.0410 | 366 |

§7's rules, read in order on `e9_ln` / raw:

| rule | reading | met? |
|---|---|---|
| QRY CLEARED | UNDECIDED at +0.0351 | no |
| sparsity signature, against `e9_base` / raw: Q-TRAIN closure ≥ 0.5 while Q-VAL closure < 0.25 | Q-TRAIN closure 0.24; Q-VAL closure 0.30 | no |
| tail signature: TAIL's 95% CI above 1.10 while F2 passes | CI [1.017, 1.146] | no |
| anything else | | **no data experiment licensed; reported; the owner decides** |

- **Not the sparsity pattern.** Q-TRAIN (+0.0410) is above Q-VAL (+0.0351). The
  model is no closer to its own training queries than to unseen ones.
- **Power.** The query cell can only use as many generated queries as a generation
  contains. For `e9_ln` / raw that is about 366 per draw, where the gate requests up
  to 1,200.
- **Phishing QRY stays NOT COMPUTABLE** for `e9_ln` / raw: only 1 of 5 generations
  reached the 250 queries the scorer needs.

## 10. Retrain spread, from three trainings of E7's recipe

E7, `e8_base` and `e9_base` are three independent trainings of one recipe (the
per-state figures are in Appendix A).

| reading | spread (max − min) |
|---|---|
| honouring, hard states (benign P0Q1, phishing P1Q1 and P0Q1) | 7.96–8.13 points |
| honouring, other states | 0.47–2.37 points |
| free path emission | 0.61–1.17 points |
| free query emission | 0.20–0.31 points |
| free gate cells, x | 0.0019–0.0108 |

- **For E8.**
  - E8's V-base failure (phishing P1Q1 honouring +5.24 points, benign path emission
    −1.17) lies inside the spread measured here.
  - This supports Amendment 1's reading: a retrain difference, not a pipeline
    change.
  - What to do with E8 remains your decision.
- **For any future registration.**
  - A difference in hard-state honouring needs a noise floor of about 8 points, or
    more seeds, before it can be attributed.
  - The 0.0141 gate-cell allowance is consistent with three trainings: every
    free-cell spread is below it (largest 0.0108).

## 11. The registered predictions (PREREG_E9 §8), scored

| prediction | result | scored |
|---|---|---|
| `e9_base`: F1 about 1.16, F2 about 1.06, F3 about 0.85 | 1.164, 1.058, 0.892 | right; F3 a little milder |
| `e9_base`: emission near E7's 59% / 3.6% (benign) | 58.67% / 3.81% | right |
| `e9_ln`: plain ratios near 1 in F1–F3 | 1.000, 1.018, 0.997 | right |
| `e9_ln` / raw: emission near 73% / 6.6% (benign) and 39.5% / 4.7% (phishing) | paths 75.58% / 38.20%; queries 4.58% / 2.94% | right for paths, **wrong for queries** |
| `e9_ln`: weighted ratios fall below 1 | F1 and F2 below 1; F3 above (1.162 benign, 1.193 phishing) | **wrong as written for F3.** F3's bias ran below 1, so its mirror lies above 1; the prediction only fitted the stopping decisions |
| `e9_ln` / resting: paths longer than the data; the delta over-corrects | no path-length reading was registered; both path cells worsen with the delta (section 6) | direction as predicted; length not measured |
| content | — | registered as Unknown |

## 12. What is and is not claimed

**Claimed** [CONFIRMED by the registered single-change test]:
- The old objective's length weighting caused the models' early ending inside paths
  (closure 1.00), most of their early stopping of short query values (0.69), and
  nearly all of their missing parameters (0.97). This is K9a.
- Removing it brings free-mode path presence into the target range and clears both
  free path cells, with no decode-time patch.
- Against today's system there is no regression beyond the allowance (K9c), and
  there is less copying (K9d).

**Not claimed:**
- that query presence is fixed: K9b fails, and the cause of the rest is Unknown;
- that the frozen delta harms the new model: a direction, untested;
- any honouring change: inside the retrain range, apart from a small P0Q0 drop;
- E3 passage: 6 gated cells are not CLEARED on `e9_ln` / raw.

## 13. Decisions for the owner

1. **Adopt `e9_ln` (raw decode), or not.** The registered rule leaves this to you
   because K9b failed.
   - For: K9a, K9c and K9d pass. 6 cells are not CLEARED instead of 9 (3 CONVICTED
     instead of 4). Both free path cells clear without a decode-time patch. Copying
     is down 7–8 points.
   - Against: query presence is below its band (4.58% and 2.94%). Phishing QRY stays
     NOT COMPUTABLE. P0Q0 honouring is slightly below the old recipe's range.
   - If adopted, the registration sends it to N3, the E3 reading, with the frozen
     delta retired for it (PREREG_E9 §6).
2. **E8.** The spread measured here explains its V-base failure. The options in
   `E8_RESULTS.md` §5 stand.
3. **What remains for E3** on `e9_ln` / raw:
   - query content: free benign QRY (UNDECIDED); control benign P0Q1 QRY and P1Q1
     QRY (CONVICTED);
   - the path-and-query state: control benign P1Q1 PTH (CONVICTED) and SUB
     (UNDECIDED);
   - phishing P1Q0 SUB (UNDECIDED, +0.0473).

   The query-presence shortfall has no measured cause. No registered reading
   measures the query-start decision itself.

## 14. Owner decision, recorded after the reading [DECISION]

Recorded 2026-09-28 in `DECISION_2026-09-28.md`; sections 1–13 above are the
registered reading and are unchanged.
1. **`e9_ln` with the raw decode is adopted as the baseline** (`uie/baseline.py`).
   K9b's failure stands in the record.
2. **E8 is permanently unread and obsolete.**
3. **The query-emission gap is the active research question**, investigated
   evidence-first and separately from the baseline.

## Appendix A — the rendered tables

`python research/native/e9_tables.py`, verbatim.

### Validity

Verdict: **PASS**; failures: none

| condition | URLs regenerated | identical |
|---|---|---|
| e9_base/resting/free | 1036 | 1036 |
| e9_base/raw/free | 1036 | 1036 |
| e9_ln/raw/free | 1036 | 1036 |
| e9_ln/resting/free | 1036 | 1036 |
| e9_base/resting/control | 269 | 269 |
| e9_base/raw/control | 269 | 269 |
| e9_ln/raw/control | 269 | 269 |

V4 benign: max |x − published| = 0.00e+00

V4 phishing: max |x − published| = 0.00e+00

### Training

| arm | objective | final val (benign / phishing / mean) | best step | minutes | start |
|---|---|---|---|---|---|
| e9_base | batch | 1.6734 / 1.5301 / 1.6017 | 18000 | 72.1 | `26926388bb57` |
| e9_ln | fixed | 1.6671 / 1.5249 / 1.5960 | 18000 | 72.6 | `26926388bb57` |

Gradient clipping after step 1,000, by scored tokens per row (share of steps clipped):

| band | e9_base | e9_ln |
|---|---|---|
| 0-30 | 12.7% (norm 0.78) | 24.8% (norm 0.93) |
| 120-inf | 3.1% (norm 0.47) | 2.2% (norm 0.41) |
| 30-60 | 0.0% (norm 0.54) | 0.0% (norm 0.58) |
| 60-120 | 0.0% (norm 0.46) | 0.0% (norm 0.45) |

### R1 — the mechanism (teacher-forced calibration; 1.0 = the data's own rate)

**benign**

| family | E7 plain | e9_base plain | e9_ln plain [95% CI] | e9_ln weighted | closure |
|---|---|---|---|---|---|
| F1 | 1.184 | 1.164 | 1.000 [0.990, 1.010] | 0.852 | 1.00 |
| F2 | 1.061 | 1.058 | 1.018 [1.009, 1.027] | 0.963 | 0.69 |
| F3 | 0.845 | 0.892 | 0.997 [0.987, 1.008] | 1.162 | 0.97 |
| TAIL | 1.259 | 1.228 | 1.079 [1.017, 1.146] | 0.938 | 0.65 |

**phishing**

| family | E7 plain | e9_base plain | e9_ln plain [95% CI] | e9_ln weighted | closure |
|---|---|---|---|---|---|
| F1 | 1.113 | 1.081 | 0.995 [0.986, 1.003] | 0.913 | 0.94 |
| F2 | 1.098 | 1.105 | 0.976 [0.961, 0.992] | 0.878 | 0.77 |
| F3 | 0.724 | 0.755 | 0.939 [0.915, 0.964] | 1.193 | 0.75 |
| TAIL | 1.836 | 1.809 | 1.401 [1.314, 1.502] | 1.199 | 0.50 |

P0: **PASS**. K9a: **PASS** 

### R2 — free-mode emission (K9b reads e9_ln_raw)

| condition | benign PTH | benign QRY | phishing PTH | phishing QRY |
|---|---|---|---|---|
| e9_base_resting | 58.67% | 3.81% | 29.64% | 2.04% |
| e9_base_raw | 58.97% | 3.33% | 29.75% | 1.72% |
| e9_ln_raw | 75.58% | 4.58% | 38.20% | 2.94% |
| e9_ln_resting | 75.32% | 5.01% | 38.29% | 3.55% |
| bands | 70.37–76.37% | 5.96–7.28% | 36.51–42.51% | 4.21–5.15% |

K9b: **FAIL** ['benign QRY: 0.045825 outside (0.0596, 0.0728)', 'phishing QRY: 0.02945 outside (0.0421, 0.0515)']

### R3 — gate v2, FULL FREE (x, verdict)

| class | field | E7 (canonical) | e9_base_resting | e9_base_raw | e9_ln_raw | e9_ln_resting |
|---|---|---|---|---|---|---|
| benign | LBL | +0.0017 CLEA | +0.0057 CLEA | +0.0055 CLEA | +0.0073 CLEA | +0.0068 CLEA |
| benign | SUF | +0.0165 CLEA | +0.0147 CLEA | +0.0131 CLEA | +0.0113 CLEA | +0.0131 CLEA |
| benign | SUB | +0.0048 CLEA | +0.0022 CLEA | +0.0056 CLEA | +0.0031 CLEA | -0.0004 CLEA |
| benign | PTH | +0.0369 UNDE | +0.0348 CLEA | +0.0453 UNDE | +0.0078 CLEA | +0.0126 CLEA |
| benign | QRY | +0.0541 UNDE | +0.0630 UNDE | +0.0504 UNDE | +0.0351 UNDE | +0.0317 UNDE |
| benign | FULL | +0.2191 AUXI | +0.2186 AUXI | +0.2251 AUXI | +0.1483 AUXI | +0.1508 AUXI |
| phishing | LBL | +0.0238 CLEA | +0.0197 CLEA | +0.0203 CLEA | +0.0122 CLEA | +0.0063 CLEA |
| phishing | SUF | +0.0054 CLEA | +0.0037 CLEA | +0.0064 CLEA | +0.0080 CLEA | +0.0071 CLEA |
| phishing | SUB | +0.0334 CLEA | +0.0353 CLEA | +0.0327 CLEA | +0.0189 CLEA | +0.0176 CLEA |
| phishing | PTH | +0.0604 UNDE | +0.0533 UNDE | +0.0527 UNDE | +0.0315 CLEA | +0.0453 UNDE |
| phishing | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | +0.0923 UNDE |
| phishing | FULL | +0.4014 AUXI | +0.4033 AUXI | +0.4032 AUXI | +0.3850 AUXI | +0.3833 AUXI |

### R3 — FULL CONTROL, measurable states (x, verdict)

| state | field | e9_base_resting | e9_base_raw | e9_ln_raw |
|---|---|---|---|---|
| benign_P0Q0 | LBL | +0.0014 CLEA | +0.0013 CLEA | +0.0115 CLEA |
| benign_P0Q0 | SUF | -0.0015 CLEA | -0.0005 CLEA | +0.0022 CLEA |
| benign_P0Q0 | SUB | +0.0100 CLEA | +0.0105 CLEA | +0.0168 CLEA |
| benign_P0Q0 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q0 | FULL | +0.4920 AUXI | +0.4918 AUXI | +0.4913 AUXI |
| benign_P0Q1 | LBL | +0.0172 CLEA | +0.0172 CLEA | +0.0135 CLEA |
| benign_P0Q1 | SUF | +0.0090 CLEA | +0.0090 CLEA | +0.0074 CLEA |
| benign_P0Q1 | SUB | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q1 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q1 | QRY | +0.0898 CONV | +0.0884 CONV | +0.0940 CONV |
| benign_P0Q1 | FULL | +0.4870 AUXI | +0.4870 AUXI | +0.4859 AUXI |
| benign_P1Q0 | LBL | +0.0023 CLEA | -0.0001 CLEA | +0.0123 CLEA |
| benign_P1Q0 | SUF | +0.0011 CLEA | +0.0031 CLEA | +0.0030 CLEA |
| benign_P1Q0 | SUB | +0.0076 CLEA | +0.0058 CLEA | +0.0032 CLEA |
| benign_P1Q0 | PTH | +0.0423 UNDE | +0.0548 UNDE | +0.0119 CLEA |
| benign_P1Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P1Q0 | FULL | +0.0359 AUXI | +0.0487 AUXI | +0.0128 AUXI |
| benign_P1Q1 | LBL | +0.0181 CLEA | +0.0145 CLEA | +0.0205 CLEA |
| benign_P1Q1 | SUF | -0.0006 CLEA | +0.0002 CLEA | +0.0020 CLEA |
| benign_P1Q1 | SUB | +0.0728 CONV | +0.0705 CONV | +0.0497 UNDE |
| benign_P1Q1 | PTH | +0.0842 CONV | +0.0828 CONV | +0.0930 CONV |
| benign_P1Q1 | QRY | +0.0838 CONV | +0.0861 CONV | +0.0636 CONV |
| benign_P1Q1 | FULL | +0.1088 AUXI | +0.1100 AUXI | +0.0851 AUXI |
| phishing_P0Q0 | LBL | +0.0221 CLEA | +0.0237 CLEA | +0.0336 CLEA |
| phishing_P0Q0 | SUF | -0.0046 CLEA | -0.0042 CLEA | -0.0016 CLEA |
| phishing_P0Q0 | SUB | +0.0279 CLEA | +0.0311 CLEA | +0.0326 CLEA |
| phishing_P0Q0 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P0Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P0Q0 | FULL | +0.4989 AUXI | +0.4988 AUXI | +0.4989 AUXI |
| phishing_P1Q0 | LBL | +0.0365 UNDE | +0.0401 UNDE | +0.0340 CLEA |
| phishing_P1Q0 | SUF | +0.0015 CLEA | +0.0002 CLEA | +0.0006 CLEA |
| phishing_P1Q0 | SUB | +0.0404 CLEA | +0.0496 UNDE | +0.0473 UNDE |
| phishing_P1Q0 | PTH | +0.0526 UNDE | +0.0501 UNDE | +0.0373 CLEA |
| phishing_P1Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P1Q0 | FULL | +0.0563 AUXI | +0.0620 AUXI | +0.0494 AUXI |

K9c (against today's system, e9_base_resting): **PASS**; regressions: none
Attribution (against e9_base_raw, reported): none

### R4 — honouring (control, native)

| state | e9_base_resting | e9_base_raw | e9_ln_raw |
|---|---|---|---|
| benign P1Q1 | 93.74% | 93.43% | 93.67% |
| benign P1Q0 | 98.11% | 98.13% | 98.12% |
| benign P0Q1 | 87.67% | 87.67% | 94.95% |
| benign P0Q0 | 98.33% | 98.35% | 96.89% |
| phishing P1Q1 | 81.56% | 79.86% | 86.76% |
| phishing P1Q0 | 95.91% | 96.32% | 96.50% |
| phishing P0Q1 | 82.68% | 82.68% | 89.76% |
| phishing P0Q0 | 98.67% | 98.75% | 97.95% |

### R6 — guardrails (K9d: e9_ln_raw against e9_base_resting)

| condition | mode | novel | copied queries |
|---|---|---|---|
| e9_base_resting | free | 99.16% | 27.23% of 2,339 |
| e9_base_resting | control | 99.44% | 27.73% of 36,676 |
| e9_base_raw | free | 99.15% | 25.32% of 2,018 |
| e9_base_raw | control | 99.43% | 27.66% of 36,459 |
| e9_ln_raw | free | 99.53% | 19.26% of 3,011 |
| e9_ln_raw | control | 99.67% | 20.53% of 38,722 |
| e9_ln_resting | free | 99.54% | 18.78% of 3,423 |

K9d: **PASS** 
- free_novelty: change -0.37 pts [-0.45, -0.29]
- free_copying: change -7.97 pts [-10.26, -5.68]
- control_novelty: change -0.24 pts [-0.29, -0.18]
- control_copying: change -7.20 pts [-7.81, -6.59]

### R5 — the QRY cell against validation (Q-VAL) and training (Q-TRAIN) queries

| condition | class | Q-VAL | Q-TRAIN | n |
|---|---|---|---|---|
| e9_base_raw | benign | +0.0504 UNDE | +0.0537 UNDE | 266 |
| e9_base_raw | phishing | NOT COMPUTABLE | NOT COMPUTABLE | — |
| e9_base_resting | benign | +0.0630 UNDE | +0.0635 UNDE | 305 |
| e9_base_resting | phishing | NOT COMPUTABLE | NOT COMPUTABLE | — |
| e9_ln_raw | benign | +0.0351 UNDE | +0.0410 UNDE | 366 |
| e9_ln_raw | phishing | NOT COMPUTABLE | NOT COMPUTABLE | 251 |

### Retrain spread — three trainings of E7's recipe (reported, not a criterion)

| reading | E7 | e8_base | e9_base | max − min |
|---|---|---|---|---|
| honour benign P1Q1 | 96.11% | 95.57% | 93.74% | 2.37 pts |
| honour benign P1Q0 | 98.65% | 98.71% | 98.11% | 0.61 pts |
| honour benign P0Q1 | 93.85% | 95.79% | 87.67% | 8.13 pts |
| honour benign P0Q0 | 97.85% | 98.19% | 98.33% | 0.47 pts |
| emission benign PTH | 59.08% | 57.91% | 58.67% | 1.17 pts |
| emission benign QRY | 3.62% | 3.72% | 3.81% | 0.20 pts |
| honour phishing P1Q1 | 84.28% | 89.52% | 81.56% | 7.96 pts |
| honour phishing P1Q0 | 96.89% | 97.35% | 95.91% | 1.44 pts |
| honour phishing P0Q1 | 87.10% | 90.76% | 82.68% | 8.08 pts |
| honour phishing P0Q0 | 98.15% | 98.49% | 98.67% | 0.52 pts |
| emission phishing PTH | 29.51% | 29.03% | 29.64% | 0.61 pts |
| emission phishing QRY | 1.74% | 2.05% | 2.04% | 0.31 pts |
| free benign LBL x | +0.0017 | +0.0041 | +0.0057 | 0.0040 |
| free benign PTH x | +0.0369 | +0.0330 | +0.0348 | 0.0039 |
| free benign QRY x | +0.0541 | +0.0522 | +0.0630 | 0.0108 |
| free benign SUB x | +0.0048 | +0.0051 | +0.0022 | 0.0029 |
| free benign SUF x | +0.0165 | +0.0135 | +0.0147 | 0.0030 |
| free phishing LBL x | +0.0238 | +0.0225 | +0.0197 | 0.0041 |
| free phishing PTH x | +0.0604 | +0.0539 | +0.0533 | 0.0071 |
| free phishing SUB x | +0.0334 | +0.0337 | +0.0353 | 0.0019 |
| free phishing SUF x | +0.0054 | +0.0076 | +0.0037 | 0.0039 |

### Outcome

- Candidate rule: **K9a passes (mechanism confirmed); K9b fail: reported, the owner decides**
- Section 7: **no data experiment licensed; reported; the owner decides**
