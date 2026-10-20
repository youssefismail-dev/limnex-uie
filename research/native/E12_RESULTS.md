# E12 — results: the final refinement pass

- **Registration:** `PREREG_E12.md` (39dc856), with its clarifications made before
  any data (6beaa4d, 2fa022c) and the training note made before generation (3bbef7a).
  The owner approved the Kaggle plan before any run (541f5bf).
- **Arms:** `e12_base` (E11's `e11_plan` recipe, retrained), `e12_final` (+ the
  slash plan + R1, the candidate), `e12_no_r1` (+ the slash plan only). One session,
  one start, read at `final.pt`, step 20,000.
- **Runs:** `E12_train` and `E12_generate`, run by the owner on Kaggle (T4, torch
  2.10): results (19) and results (20).
- **Read:** 2026-10-01, after VERIFY passed, with `e12_readout.py` (2fa022c).
- **Numbers:** Appendix A is `e12_tables.py`'s output, pasted by script. Every number
  in the text appears there or has a declared source: E11's numbers come from
  `E11_STEP20K_RESULTS.md`; standard errors come from `data/e12/score_report.json`.

## In plain words

- **E3 is not met.** `e12_final` has 6 gated cells that are not CLEARED and misses
  3 slash rows. E11's `e11_plan` had 7 and 3.
- **Free mode is one cell from done.** 9 of `e12_final`'s 10 free cells are
  CLEARED. The tenth, phishing free QRY, fails clearly: +0.1141 CONVICTED (E11:
  +0.0953). Every arm convicts it.
- **Two cells moved to CLEARED, one moved away:**

  | cell | E11 `e11_plan` | `e12_final` |
  |---|---|---|
  | benign free QRY | +0.0350 UNDECIDED | **+0.0325 CLEARED** |
  | phishing free PTH | +0.0431 UNDECIDED | **+0.0376 CLEARED** |
  | phishing control P1Q0 PTH | +0.0403 CLEARED | +0.0428 UNDECIDED |

  These flips are fragile. In each one x barely moved; the verdict changed mostly
  because the standard error changed (benign free QRY: 0.0104 in E11, 0.0055 here).
  All three changes are smaller than the run-to-run spread measured below.
- **The slash plan moved all three failing slash rows toward the real rates, and
  none into tolerance:**

  | row | E11 | `e12_final` | real (± T) | still missing |
  |---|---|---|---|---|
  | benign control P0Q1 (slash too often) | 99.97% | 99.81% | 99.551% (±0.24) | 0.02 points |
  | phishing free, bare hosts (slash too rarely) | 99.53% | 99.67% | 100% (±0.04) | 0.29 points |
  | phishing control P0Q0 (slash too rarely) | 99.64% | 99.76% | 100% (±0.07) | 0.17 points |

  - **Phishing:** told S = 1, the model writes the slash 99.693% (control P0Q0)
    and 99.790% (free mode) of the time, the populations behind the failing rows.
    The rows need at least 99.93–99.96%. This is the risk D2 stated before training.
  - **Benign `host?q`:** told S = 0, the model obeys 70.423% of the time (of 71
    URLs). That is much better than the training log's quick check suggested, but
    not enough. This is the warning recorded before generation (3bbef7a).
- **R1 (max_len 384) gave nothing measurable.** With it, benign free QRY was 0.0063
  *higher* (the prediction was 0.003–0.010 lower). Every cell difference stays
  within the retrain allowance.
- **No arm is a candidate.** Every arm fails K12b, the slash. Everything else
  passes in every arm:
  - K12a: presence within T;
  - K12c: no regression;
  - K12d: copying under the natural rate, no memorisation, novelty unchanged;
  - K12e: 0.0% of steps clipped after step 1,000.
- **The query tell is still in the content.** QC1-2 on `e12_final` reads
  CONTENT-LED in all four query cells, as QC1 found on E11.

## 1. Validity: PASS

| check | result |
|---|---|
| V1 | every `final.pt` as training recorded; generation read `final.pt`; one code commit (f4a1ee5), unchanged since; the 150 registered files with their n, seed and hash; the tables and both presence tables as registered |
| V2 | byte identity on the laptop: 712 of 712 (base) and 669 of 669 (slash arms) free URLs, 506 of 506 and 493 of 493 control URLs, located through the supplied plan's draws and shuffles |
| V3 | `uie/` and the common modules unchanged since f4a1ee5 |
| V4 | the scorer reproduces E7's published cells exactly: 0.00e+00 in both classes |
| V-id | every start contains E10's and E11's; the slash arms' new token rows are identical |

## 2. What stands between `e12_final` and E3, item by item

A cell is CLEARED when x is below 0.05 by more than t_crit (2.7764) standard
errors. UNDECIDED means the interval still crosses 0.05. "To clear" is x falling
to 0.05 − 2.7764 · SE, at this run's SE [computed from `score_report.json`].

| item | E11 | `e12_final` | SE | to clear, x must fall by |
|---|---|---|---|---|
| free phishing QRY | +0.0953 CONVICTED | +0.1141 CONVICTED | 0.0047 | 0.077 |
| control benign P0Q1 QRY | +0.0581 UNDECIDED | +0.0621 UNDECIDED | 0.0057 | 0.028 |
| control benign P1Q1 QRY | +0.0511 UNDECIDED | +0.0559 UNDECIDED | 0.0040 | 0.017 |
| control benign P1Q1 PTH | +0.0509 UNDECIDED | +0.0576 UNDECIDED | 0.0030 | 0.016 |
| control benign P1Q1 SUB | +0.0379 UNDECIDED | +0.0416 UNDECIDED | 0.0075 | 0.012 |
| control phishing P1Q0 PTH | +0.0403 CLEARED | +0.0428 UNDECIDED | 0.0032 | 0.002 |
| slash: benign control P0Q1 | 99.97% | 99.81% | — | 0.02 points |
| slash: phishing free, bare hosts | 99.53% | 99.67% | — | 0.29 points |
| slash: phishing control P0Q0 | 99.64% | 99.76% | — | 0.17 points |

- **One item is far:** phishing free QRY.
- **Four sit near the bar:** the benign control query and path cells, all within
  about 0.03 of their clearing point.
- **One is noise-level:** phishing control P1Q0 PTH, CLEARED in `e12_base`
  (+0.0377).
- **The slash rows:** benign is 0.02 points out. The phishing rows need
  near-perfect slash writing.

## 3. What each change did

**The slash plan** (`e12_no_r1` against `e12_base`; no arm ablates it):
- All three of E11's failing slash rows move toward the real rates. In
  `e12_no_r1`:
  - benign control P0Q1: +0.25 against `e12_base`'s +0.40;
  - phishing free bare: −0.26 against −0.48;
  - phishing control P0Q0: −0.18 against −0.44.
- Two other phishing rows fail in `e12_no_r1` that pass in the other arms:
  host-route queries −1.68 (±1.54), and control P0Q1 −0.22 (±0.20).
- Content cells move by at most 0.0128 (benign free QRY, lower), all within the
  allowance.

**R1, max_len 384** (`e12_final` against `e12_no_r1`):
- Rows dropped from training fell from 10,415 to 3,851.
- No cell moved beyond the allowance. Benign free QRY moved +0.0063, the opposite
  of the prediction.
- `e12_final` passes the two phishing slash rows `e12_no_r1` fails. With a single
  run each, that is not attributable.

## 4. Run-to-run spread

- `e12_base` retrains E11's `e11_plan` recipe from the same start, in a new session.
- Its cells differ from `e11_plan`'s by at most 0.0078 in control mode and 0.0148
  in free mode. The largest is phishing free QRY, just over the 0.0141 retrain
  allowance.
- So a verdict within about 0.015 of its clearing point can flip between runs of
  the same recipe. Section 2's near-bar items, and the three flips in the summary,
  should be read with that in mind.

## 5. Guardrails, copying and QC1-2

- **Copying** stays under the natural rate in every arm and cell:
  - `e12_final` free mode copies 16.86% of its queries from training;
  - real held-out URLs do 29.27% (benign) and 41.85% (phishing) at host level;
  - strings seen once in training are never OVER their natural rate, so there is
    no memorisation.
- **Novelty** is unchanged against `e12_base` in both modes.
- **The band profile** repeats QC1's finding. The arms produce the most common real
  query strings less often than real URLs do:
  - phishing free: strings seen 20+ times are 14.64% of `e12_final`'s queries,
    against 26.31% of real ones;
  - benign free: 7.98% against 13.10%.
- **QC1-2:** CONTENT-LED in every cell (rho 0.96–1.37). Forcing the natural mix
  of copied and new strings does not remove the tell.

## 6. Section 3's predictions, scored

Six held, three did not, one does not apply (Appendix A):
- **Held:** `e12_base` fails E11's three slash rows; K12a passes everywhere; R1
  leaves phishing free QRY within noise; phishing free QRY stays CONVICTED in every
  arm; the finish line is not met; phishing free QRY is among the failures.
- **Did not hold:**
  - every slash-plan arm passes K12b;
  - R1 lowers benign free QRY by 0.003–0.010;
  - the free-cell run-to-run spread stays within 0.0141 (it reached 0.0148).

## 7. The end rule, and what is decided next

- **The end rule** (`DECISION_2026-09-30.md` §8): E3 is not met, so this is the
  final E3 report of this programme.
- **Anything further is a new decision by the owner.** The owner has said E13 is
  open, to be decided after these results.
- **The facts that decision rests on:**
  - Free mode is one cell from done, but that cell is the far one. QC1 found no
    licensed lever for phishing query content, and E12 confirms the tell is still
    content-led.
  - The benign control cells are near-bar UNDECIDED. Their distances (0.012–0.028)
    are comparable to the run-to-run spread (up to 0.0148).
  - The slash plan works in the right direction, but it cannot reach the phishing
    rows' near-perfect requirement. The benign row is 0.02 points out.
  - R1 bought nothing measurable.
- **Not proposed here:** any new experiment. The owner's standing rules forbid
  inventing one because a metric failed. Any E13 would start from a registration
  and, for phishing queries, from new diagnosis.

## Appendix A — the rendered tables

`python research/native/e12_tables.py`, verbatim.

### Validity

Verdict: **PASS**; failures: none

| arm | E10's start | E11's start | new token rows | new positions |
|---|---|---|---|---|
| e12_base | 26926388bb57… | 5aa980f7c6ad… | — | — |
| e12_final | 26926388bb57… | 5aa980f7c6ad… | 3743d8971e06 | 4fe9099f54ae |
| e12_no_r1 | 26926388bb57… | 5aa980f7c6ad… | 3743d8971e06 | — |

| arm / mode | URLs regenerated | identical |
|---|---|---|
| e12_base/free | 712 | 712 |
| e12_base/control | 506 | 506 |
| e12_final/free | 669 | 669 |
| e12_final/control | 493 | 493 |
| e12_no_r1/free | 669 | 669 |
| e12_no_r1/control | 493 | 493 |

V4 benign: max |x − published| = 0.00e+00

V4 phishing: max |x − published| = 0.00e+00

### Training (read at `final.pt`, step 20,000)

| arm | final val (benign / phishing / mean) | lowest mean val at step | minutes | rows that fit | dropped |
|---|---|---|---|---|---|
| e12_base | 1.6018 / 1.4388 / 1.5203 | 19,000 | 62.9 | {'benign': 2545303, 'phishing': 238198} | 10,316 |
| e12_final | 1.5621 / 1.4006 / 1.4813 | 19,000 | 69.4 | {'benign': 2551259, 'phishing': 238707} | 3,851 |
| e12_no_r1 | 1.5705 / 1.4051 / 1.4878 | 15,000 | 63.6 | {'benign': 2545216, 'phishing': 238186} | 10,415 |

### K12e — clipping after step 1,000

| arm | band | steps | mean norm | clipped | non-finite steps |
|---|---|---|---|---|---|
| e12_base | 0-30 | 3,595 | 0.922 | 0.0% | 7 |
| e12_base | 30-60 | 9,428 | 0.562 | 0.0% | 0 |
| e12_base | 60-120 | 5,096 | 0.442 | 0.0% | 0 |
| e12_base | 120-inf | 881 | 0.400 | 0.0% | 0 |
| e12_final | 0-30 | 3,282 | 0.906 | 0.0% | 7 |
| e12_final | 30-60 | 9,473 | 0.557 | 0.0% | 0 |
| e12_final | 60-120 | 5,356 | 0.440 | 0.0% | 0 |
| e12_final | 120-inf | 889 | 0.404 | 0.0% | 0 |
| e12_no_r1 | 0-30 | 3,279 | 0.906 | 0.0% | 8 |
| e12_no_r1 | 30-60 | 9,495 | 0.554 | 0.0% | 0 |
| e12_no_r1 | 60-120 | 5,337 | 0.439 | 0.0% | 0 |
| e12_no_r1 | 120-inf | 889 | 0.399 | 0.0% | 0 |

### K12a — free-mode presence against the tolerance

| arm | row | generated against real (difference, T) within |
|---|---|---|
| e12_base | benign PTH | 73.46% against 73.369% (+0.09, ±0.99) yes |
| e12_base | benign QRY | 6.57% against 6.617% (-0.05, ±0.56) yes |
| e12_base | phishing PTH | 39.94% against 39.511% (+0.43, ±1.09) yes |
| e12_base | phishing QRY | 4.72% against 4.680% (+0.04, ±0.47) yes |
| e12_final | benign PTH | 73.51% against 73.369% (+0.14, ±0.99) yes |
| e12_final | benign QRY | 6.58% against 6.617% (-0.04, ±0.56) yes |
| e12_final | phishing PTH | 39.91% against 39.511% (+0.40, ±1.09) yes |
| e12_final | phishing QRY | 4.68% against 4.680% (-0.00, ±0.47) yes |
| e12_no_r1 | benign PTH | 73.45% against 73.369% (+0.08, ±0.99) yes |
| e12_no_r1 | benign QRY | 6.59% against 6.617% (-0.03, ±0.56) yes |
| e12_no_r1 | phishing PTH | 39.93% against 39.511% (+0.42, ±1.09) yes |
| e12_no_r1 | phishing QRY | 4.69% against 4.680% (+0.00, ±0.47) yes |

### K12b — the root slash against the tolerance

| arm | row | generated against real (difference, T) within |
|---|---|---|
| e12_base | benign free_bare | 98.11% against 98.004% (+0.10, ±0.63) yes |
| e12_base | benign control P0Q0 | 98.23% against 97.944% (+0.29, ±0.52) yes |
| e12_base | benign free_host_query | 99.53% against 99.576% (-0.05, ±1.50) yes |
| e12_base | benign control P0Q1 | 99.95% against 99.551% (+0.40, ±0.24) **no** |
| e12_base | phishing free_bare | 99.52% against 100.000% (-0.48, ±0.04) **no** |
| e12_base | phishing control P0Q0 | 99.56% against 100.000% (-0.44, ±0.07) **no** |
| e12_base | phishing free_host_query | 98.93% against 100.000% (-1.07, ±1.54) yes |
| e12_base | phishing control P0Q1 | 99.98% against 100.000% (-0.02, ±0.20) yes |
| e12_final | benign free_bare | 97.73% against 98.004% (-0.27, ±0.63) yes |
| e12_final | benign control P0Q0 | 97.70% against 97.944% (-0.24, ±0.52) yes |
| e12_final | benign free_host_query | 99.37% against 99.576% (-0.20, ±1.50) yes |
| e12_final | benign control P0Q1 | 99.81% against 99.551% (+0.26, ±0.24) **no** |
| e12_final | phishing free_bare | 99.67% against 100.000% (-0.33, ±0.04) **no** |
| e12_final | phishing control P0Q0 | 99.76% against 100.000% (-0.24, ±0.07) **no** |
| e12_final | phishing free_host_query | 98.79% against 100.000% (-1.21, ±1.54) yes |
| e12_final | phishing control P0Q1 | 99.96% against 100.000% (-0.04, ±0.20) yes |
| e12_no_r1 | benign free_bare | 97.80% against 98.004% (-0.21, ±0.63) yes |
| e12_no_r1 | benign control P0Q0 | 97.84% against 97.944% (-0.11, ±0.52) yes |
| e12_no_r1 | benign free_host_query | 98.91% against 99.576% (-0.67, ±1.50) yes |
| e12_no_r1 | benign control P0Q1 | 99.81% against 99.551% (+0.25, ±0.24) **no** |
| e12_no_r1 | phishing free_bare | 99.74% against 100.000% (-0.26, ±0.04) **no** |
| e12_no_r1 | phishing control P0Q0 | 99.82% against 100.000% (-0.18, ±0.07) **no** |
| e12_no_r1 | phishing free_host_query | 98.32% against 100.000% (-1.68, ±1.54) **no** |
| e12_no_r1 | phishing control P0Q1 | 99.78% against 100.000% (-0.22, ±0.20) **no** |

### S honouring (slash arms: generated URLs that obey their supplied S)

| arm | population | S = 0 | S = 1 |
|---|---|---|---|
| e12_final | control_benign_P1Q1 | — | 99.913% of 15,000 |
| e12_final | control_benign_P1Q0 | — | 99.920% of 15,000 |
| e12_final | control_benign_P0Q1 | 70.423% of 71 | 99.806% of 14,929 |
| e12_final | control_benign_P0Q0 | 97.468% of 316 | 99.748% of 14,684 |
| e12_final | free_benign | 94.872% of 195 | 99.887% of 39,805 |
| e12_final | control_phishing_P1Q1 | — | 99.740% of 5,000 |
| e12_final | control_phishing_P1Q0 | — | 99.880% of 15,000 |
| e12_final | control_phishing_P0Q1 | — | 99.620% of 5,000 |
| e12_final | control_phishing_P0Q0 | — | 99.693% of 15,000 |
| e12_final | free_phishing | — | 99.790% of 40,000 |
| e12_no_r1 | control_benign_P1Q1 | — | 99.927% of 15,000 |
| e12_no_r1 | control_benign_P1Q0 | — | 99.873% of 15,000 |
| e12_no_r1 | control_benign_P0Q1 | 61.972% of 71 | 99.799% of 14,929 |
| e12_no_r1 | control_benign_P0Q0 | 93.987% of 316 | 99.803% of 14,684 |
| e12_no_r1 | free_benign | 92.821% of 195 | 99.884% of 39,805 |
| e12_no_r1 | control_phishing_P1Q1 | — | 99.720% of 5,000 |
| e12_no_r1 | control_phishing_P1Q0 | — | 99.873% of 15,000 |
| e12_no_r1 | control_phishing_P0Q1 | — | 99.440% of 5,000 |
| e12_no_r1 | control_phishing_P0Q0 | — | 99.773% of 15,000 |
| e12_no_r1 | free_phishing | — | 99.823% of 40,000 |

### Gate v2, FULL FREE (x, verdict)

| class | field | e12_base | e12_final | e12_no_r1 |
|---|---|---|---|---|
| benign | LBL | +0.0104 CLEA | +0.0052 CLEA | +0.0077 CLEA |
| benign | SUF | +0.0173 CLEA | +0.0151 CLEA | +0.0174 CLEA |
| benign | SUB | +0.0015 CLEA | +0.0032 CLEA | +0.0030 CLEA |
| benign | PTH | +0.0050 CLEA | +0.0041 CLEA | +0.0097 CLEA |
| benign | QRY | +0.0390 UNDE | +0.0325 CLEA | +0.0262 UNDE |
| benign | FULL | +0.0168 AUXI | +0.0168 AUXI | +0.0182 AUXI |
| phishing | LBL | +0.0111 CLEA | +0.0125 CLEA | +0.0100 CLEA |
| phishing | SUF | +0.0054 CLEA | +0.0078 CLEA | +0.0120 CLEA |
| phishing | SUB | +0.0191 CLEA | +0.0191 CLEA | +0.0219 CLEA |
| phishing | PTH | +0.0377 UNDE | +0.0376 CLEA | +0.0372 CLEA |
| phishing | QRY | +0.1101 CONV | +0.1141 CONV | +0.1070 CONV |
| phishing | FULL | +0.0240 AUXI | +0.0219 AUXI | +0.0267 AUXI |

### FULL CONTROL, measurable states (x, verdict)

| state | field | e12_base | e12_final | e12_no_r1 |
|---|---|---|---|---|
| benign_P0Q0 | LBL | +0.0047 CLEA | +0.0100 CLEA | +0.0078 CLEA |
| benign_P0Q0 | SUF | +0.0007 CLEA | -0.0004 CLEA | +0.0027 CLEA |
| benign_P0Q0 | SUB | +0.0114 CLEA | +0.0108 CLEA | +0.0147 CLEA |
| benign_P0Q0 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q0 | FULL | +0.0080 AUXI | +0.0123 AUXI | +0.0097 AUXI |
| benign_P0Q1 | LBL | +0.0231 CLEA | +0.0152 CLEA | +0.0195 CLEA |
| benign_P0Q1 | SUF | +0.0103 CLEA | +0.0103 CLEA | +0.0092 CLEA |
| benign_P0Q1 | SUB | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q1 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q1 | QRY | +0.0566 CONV | +0.0621 UNDE | +0.0585 UNDE |
| benign_P0Q1 | FULL | +0.0772 AUXI | +0.0813 AUXI | +0.0844 AUXI |
| benign_P1Q0 | LBL | -0.0001 CLEA | +0.0030 CLEA | +0.0034 CLEA |
| benign_P1Q0 | SUF | -0.0013 CLEA | +0.0003 CLEA | +0.0010 CLEA |
| benign_P1Q0 | SUB | +0.0027 CLEA | +0.0027 CLEA | +0.0036 CLEA |
| benign_P1Q0 | PTH | +0.0086 CLEA | +0.0110 CLEA | +0.0086 CLEA |
| benign_P1Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P1Q0 | FULL | +0.0097 AUXI | +0.0103 AUXI | +0.0125 AUXI |
| benign_P1Q1 | LBL | +0.0256 CLEA | +0.0262 CLEA | +0.0241 CLEA |
| benign_P1Q1 | SUF | +0.0016 CLEA | +0.0041 CLEA | +0.0061 CLEA |
| benign_P1Q1 | SUB | +0.0408 UNDE | +0.0416 UNDE | +0.0498 UNDE |
| benign_P1Q1 | PTH | +0.0587 UNDE | +0.0576 UNDE | +0.0538 UNDE |
| benign_P1Q1 | QRY | +0.0509 UNDE | +0.0559 UNDE | +0.0594 UNDE |
| benign_P1Q1 | FULL | +0.0707 AUXI | +0.0699 AUXI | +0.0723 AUXI |
| phishing_P0Q0 | LBL | +0.0216 CLEA | +0.0145 CLEA | +0.0190 CLEA |
| phishing_P0Q0 | SUF | -0.0019 CLEA | -0.0033 CLEA | -0.0055 CLEA |
| phishing_P0Q0 | SUB | +0.0314 CLEA | +0.0265 CLEA | +0.0265 CLEA |
| phishing_P0Q0 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P0Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P0Q0 | FULL | +0.0326 AUXI | +0.0213 AUXI | +0.0290 AUXI |
| phishing_P1Q0 | LBL | +0.0199 CLEA | +0.0258 CLEA | +0.0236 CLEA |
| phishing_P1Q0 | SUF | +0.0022 CLEA | +0.0010 CLEA | -0.0016 CLEA |
| phishing_P1Q0 | SUB | +0.0251 CLEA | +0.0324 CLEA | +0.0300 CLEA |
| phishing_P1Q0 | PTH | +0.0377 CLEA | +0.0428 UNDE | +0.0430 UNDE |
| phishing_P1Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P1Q0 | FULL | +0.0331 AUXI | +0.0447 AUXI | +0.0470 AUXI |

### Honouring (control, native)

| state | e12_base | e12_final | e12_no_r1 |
|---|---|---|---|
| benign P1Q1 | 98.41% | 98.25% | 98.21% |
| benign P1Q0 | 99.63% | 99.68% | 99.58% |
| benign P0Q1 | 98.35% | 99.16% | 99.21% |
| benign P0Q0 | 99.47% | 99.58% | 99.57% |
| phishing P1Q1 | 97.06% | 96.02% | 97.26% |
| phishing P1Q0 | 99.43% | 99.46% | 99.43% |
| phishing P0Q1 | 99.20% | 99.22% | 98.88% |
| phishing P0Q0 | 99.63% | 99.65% | 99.65% |

### K12c — no regression against `e12_base`

| arm | regressions |
|---|---|
| e12_base | — (the reference) |
| e12_final | none |
| e12_no_r1 | none |

### K12d — guardrails

| arm | mode | novel | copied queries |
|---|---|---|---|
| e12_base | free | 99.55% | 20.83% of 4,517 |
| e12_base | control | 99.68% | 24.03% of 39,539 |
| e12_final | free | 99.47% | 16.86% of 4,502 |
| e12_final | control | 99.63% | 21.14% of 39,574 |
| e12_no_r1 | free | 99.56% | 17.94% of 4,509 |
| e12_no_r1 | control | 99.70% | 22.32% of 39,627 |

Memorisation (QC1-1's instrument; the host-level reference decides, the domain-disjoint one is reported):

| arm | cell | reference | copied, generated / natural (±T) | reading | seen once in training, generated / natural | reading |
|---|---|---|---|---|---|---|
| e12_base | free_benign | host | 20.82% / 29.27% (±3.97) | UNDER | 3.08% / 4.59% | WITHIN |
| e12_base | control_benign_P0Q1 | host | 30.41% / 42.90% (±1.82) | UNDER | 6.48% / 8.05% | UNDER |
| e12_base | control_benign_P1Q1 | host | 15.93% / 24.77% (±1.59) | UNDER | 1.93% / 3.45% | UNDER |
| e12_base | free_phishing | host | 20.85% / 41.85% (±5.07) | UNDER | 0.26% / 5.06% | UNDER |
| e12_base | free_phishing | domain_disjoint | 20.85% / 26.48% (±4.54) | UNDER | 0.26% / 4.03% | UNDER |
| e12_final | free_benign | host | 16.91% / 29.27% (±3.97) | UNDER | 3.08% / 4.59% | WITHIN |
| e12_final | control_benign_P0Q1 | host | 27.67% / 42.90% (±1.81) | UNDER | 6.29% / 8.05% | UNDER |
| e12_final | control_benign_P1Q1 | host | 16.06% / 24.77% (±1.59) | UNDER | 2.02% / 3.45% | UNDER |
| e12_final | free_phishing | host | 16.78% / 41.85% (±5.10) | UNDER | 0.59% / 5.06% | UNDER |
| e12_final | free_phishing | domain_disjoint | 16.78% / 26.48% (±4.56) | UNDER | 0.59% / 4.03% | UNDER |
| e12_no_r1 | free_benign | host | 19.28% / 29.27% (±3.96) | UNDER | 3.30% / 4.59% | WITHIN |
| e12_no_r1 | control_benign_P0Q1 | host | 29.22% / 42.90% (±1.81) | UNDER | 6.76% / 8.05% | UNDER |
| e12_no_r1 | control_benign_P1Q1 | host | 17.22% / 24.77% (±1.59) | UNDER | 1.93% / 3.45% | UNDER |
| e12_no_r1 | free_phishing | host | 16.06% / 41.85% (±5.10) | UNDER | 0.59% / 5.06% | UNDER |
| e12_no_r1 | free_phishing | domain_disjoint | 16.06% / 26.48% (±4.56) | UNDER | 0.59% / 4.03% | UNDER |

The band profile (share of the cell's queries matching a training string seen that often):

| cell | who | 1 | 2–4 | 5–19 | 20+ |
|---|---|---|---|---|---|
| free_benign | real | 4.59% | 4.63% | 6.95% | 13.10% |
| free_benign | e12_base | 3.08% | 3.58% | 3.62% | 10.54% |
| free_benign | e12_final | 3.08% | 3.12% | 2.74% | 7.98% |
| free_benign | e12_no_r1 | 3.30% | 3.57% | 3.68% | 8.73% |
| control_benign_P0Q1 | real | 8.05% | 7.71% | 10.71% | 16.43% |
| control_benign_P0Q1 | e12_base | 6.48% | 7.78% | 7.19% | 8.96% |
| control_benign_P0Q1 | e12_final | 6.29% | 7.29% | 5.86% | 8.22% |
| control_benign_P0Q1 | e12_no_r1 | 6.76% | 7.53% | 7.01% | 7.91% |
| control_benign_P1Q1 | real | 3.45% | 3.61% | 5.71% | 12.00% |
| control_benign_P1Q1 | e12_base | 1.93% | 1.69% | 2.54% | 9.77% |
| control_benign_P1Q1 | e12_final | 2.02% | 1.90% | 2.65% | 9.49% |
| control_benign_P1Q1 | e12_no_r1 | 1.93% | 2.20% | 2.60% | 10.49% |
| free_phishing | real | 5.06% | 5.34% | 5.15% | 26.31% |
| free_phishing | real_domain_disjoint | 4.03% | 5.30% | 5.72% | 11.44% |
| free_phishing | e12_base | 0.26% | 0.63% | 1.32% | 18.62% |
| free_phishing | e12_final | 0.59% | 0.80% | 0.75% | 14.64% |
| free_phishing | e12_no_r1 | 0.59% | 0.69% | 0.75% | 14.03% |

- e12_base: **PASS**
- e12_final: **PASS**; free novelty +0.07 pts [+0.00, +0.14]; control novelty +0.05 pts [-0.00, +0.10]
- e12_no_r1: **PASS**; free novelty -0.01 pts [-0.08, +0.05]; control novelty -0.02 pts [-0.07, +0.03]

### QC1-2 on `e12_final` (reported: one arm, not replicated, licenses nothing)

| cell | n | x plain | x at the natural share | rho | t | reading | void seeds |
|---|---|---|---|---|---|---|---|
| free_benign | 1522 | +0.0457 | +0.0628 | 1.37 | -3.74 | CONTENT-LED | 0 |
| free_phishing | 356 | +0.1237 | +0.1361 | 1.10 | -1.30 | CONTENT-LED | 0 |
| control_benign_P0Q1 | 600 | +0.0679 | +0.0649 | 0.96 | +0.38 | CONTENT-LED | 0 |
| control_benign_P1Q1 | 1800 | +0.0642 | +0.0819 | 1.28 | -4.38 | CONTENT-LED | 0 |
- free_benign: never-seen +0.0663; found +0.1190
- free_phishing: never-seen insufficient; found insufficient; at the domain-disjoint share +0.1056
- control_benign_P0Q1: never-seen +0.0462; found +0.1372
- control_benign_P1Q1: never-seen +0.0688; found +0.1600

### Leave-one-out: R1 (`e12_final` against `e12_no_r1`)

Emission (points): benign PTH +0.06, benign QRY -0.01, phishing PTH -0.02, phishing QRY -0.01

| free cell | x difference |
|---|---|
| benign LBL | -0.0025 |
| benign PTH | -0.0056 |
| benign QRY | +0.0063 |
| benign SUB | +0.0002 |
| benign SUF | -0.0023 |
| phishing LBL | +0.0025 |
| phishing PTH | +0.0004 |
| phishing QRY | +0.0071 |
| phishing SUB | -0.0028 |
| phishing SUF | -0.0042 |

| control cell | x difference |
|---|---|
| benign P0Q0 LBL | +0.0022 |
| benign P0Q0 SUB | -0.0039 |
| benign P0Q0 SUF | -0.0031 |
| benign P0Q1 LBL | -0.0043 |
| benign P0Q1 QRY | +0.0036 |
| benign P0Q1 SUF | +0.0011 |
| benign P1Q0 LBL | -0.0004 |
| benign P1Q0 PTH | +0.0024 |
| benign P1Q0 SUB | -0.0009 |
| benign P1Q0 SUF | -0.0007 |
| benign P1Q1 LBL | +0.0021 |
| benign P1Q1 PTH | +0.0038 |
| benign P1Q1 QRY | -0.0035 |
| benign P1Q1 SUB | -0.0082 |
| benign P1Q1 SUF | -0.0020 |
| phishing P0Q0 LBL | -0.0045 |
| phishing P0Q0 SUB | +0.0000 |
| phishing P0Q0 SUF | +0.0022 |
| phishing P1Q0 LBL | +0.0022 |
| phishing P1Q0 PTH | -0.0002 |
| phishing P1Q0 SUB | +0.0024 |
| phishing P1Q0 SUF | +0.0026 |

### The slash plan's effect (`e12_no_r1` against `e12_base`; reported, no arm ablates it)

Emission (points): benign PTH -0.01, benign QRY +0.02, phishing PTH -0.01, phishing QRY -0.04

| free cell | x difference |
|---|---|
| benign LBL | -0.0027 |
| benign PTH | +0.0047 |
| benign QRY | -0.0128 |
| benign SUB | +0.0015 |
| benign SUF | +0.0001 |
| phishing LBL | -0.0011 |
| phishing PTH | -0.0005 |
| phishing QRY | -0.0031 |
| phishing SUB | +0.0028 |
| phishing SUF | +0.0066 |

| control cell | x difference |
|---|---|
| benign P0Q0 LBL | +0.0031 |
| benign P0Q0 SUB | +0.0033 |
| benign P0Q0 SUF | +0.0020 |
| benign P0Q1 LBL | -0.0036 |
| benign P0Q1 QRY | +0.0019 |
| benign P0Q1 SUF | -0.0011 |
| benign P1Q0 LBL | +0.0035 |
| benign P1Q0 PTH | +0.0000 |
| benign P1Q0 SUB | +0.0009 |
| benign P1Q0 SUF | +0.0023 |
| benign P1Q1 LBL | -0.0015 |
| benign P1Q1 PTH | -0.0049 |
| benign P1Q1 QRY | +0.0085 |
| benign P1Q1 SUB | +0.0090 |
| benign P1Q1 SUF | +0.0045 |
| phishing P0Q0 LBL | -0.0026 |
| phishing P0Q0 SUB | -0.0049 |
| phishing P0Q0 SUF | -0.0036 |
| phishing P1Q0 LBL | +0.0037 |
| phishing P1Q0 PTH | +0.0053 |
| phishing P1Q0 SUB | +0.0049 |
| phishing P1Q0 SUF | -0.0038 |

### Run-to-run spread (`e12_base` against E11's `e11_plan`, both at step 20,000)

Largest |x difference| over 10 free cells: 0.0148

Largest |x difference| over 22 control cells: 0.0078

Emission (points): benign PTH +0.01, benign QRY +0.00, phishing PTH -0.04, phishing QRY +0.06

| free cell | x difference |
|---|---|
| benign LBL | +0.0023 |
| benign PTH | -0.0008 |
| benign QRY | +0.0040 |
| benign SUB | -0.0023 |
| benign SUF | +0.0012 |
| phishing LBL | +0.0004 |
| phishing PTH | -0.0054 |
| phishing QRY | +0.0148 |
| phishing SUB | +0.0000 |
| phishing SUF | -0.0053 |

| control cell | x difference |
|---|---|
| benign P0Q0 LBL | -0.0060 |
| benign P0Q0 SUB | -0.0039 |
| benign P0Q0 SUF | +0.0024 |
| benign P0Q1 LBL | +0.0004 |
| benign P0Q1 QRY | -0.0015 |
| benign P0Q1 SUF | +0.0018 |
| benign P1Q0 LBL | -0.0025 |
| benign P1Q0 PTH | -0.0039 |
| benign P1Q0 SUB | -0.0001 |
| benign P1Q0 SUF | -0.0023 |
| benign P1Q1 LBL | -0.0001 |
| benign P1Q1 PTH | +0.0078 |
| benign P1Q1 QRY | -0.0002 |
| benign P1Q1 SUB | +0.0029 |
| benign P1Q1 SUF | +0.0045 |
| phishing P0Q0 LBL | -0.0022 |
| phishing P0Q0 SUB | -0.0012 |
| phishing P0Q0 SUF | +0.0009 |
| phishing P1Q0 LBL | +0.0026 |
| phishing P1Q0 PTH | -0.0026 |
| phishing P1Q0 SUB | -0.0035 |
| phishing P1Q0 SUF | +0.0012 |

### Section 3's predictions, scored

| prediction | held | detail |
|---|---|---|
| K12b: every slash-plan arm passes | **no** | e12_final: benign control P0Q1, phishing control P0Q0, phishing free_bare; e12_no_r1: benign control P0Q1, phishing control P0Q0, phishing control P0Q1, phishing free_bare, phishing free_host_query |
| K12b: e12_base fails the same three rows E11 did | yes | e12_base: benign control P0Q1, phishing control P0Q0, phishing free_bare; e11_plan, step 20,000: benign control P0Q1, phishing control P0Q0, phishing free_bare |
| K12a: every arm passes | yes | e12_base: none; e12_final: none; e12_no_r1: none |
| R1: benign free QRY lower with it, by 0.003-0.010 | **no** | x difference: +0.0063 |
| R1: phishing free QRY, no change beyond noise (absolute difference at most 0.0141) | yes | x difference: +0.0071 |
| the frame at every position (if licensed) | n/a | not applicable: QC1-9 read NO FRAME-LEAK, so D4 is not used |
| phishing free QRY stays CONVICTED in every arm | yes | e12_base: CONVICTED; e12_final: CONVICTED; e12_no_r1: CONVICTED |
| the run-to-run spread of free cells stays within the retrain allowance (0.0141) | **no** | largest: phishing QRY; x difference: +0.0148 |
| E3 finish line for e12_final: NOT met | yes | cells not cleared: free phishing QRY, control benign_P1Q1 SUB, control benign_P1Q1 PTH, control benign_P1Q1 QRY, control benign_P0Q1 QRY, control phishing_P1Q0 PTH; slash rows missed: benign control P0Q1, phishing free_bare, phishing control P0Q0 |
| phishing free QRY is the expected failure | yes | free phishing QRY: CONVICTED |

### Outcome

| arm | failing | candidate | E3 finish line |
|---|---|---|---|
| e12_base | K12b | **no** | not met: 7 cells, 3 slash rows |
| e12_final | K12b | **no** | not met: 6 cells, 3 slash rows |
| e12_no_r1 | K12b | **no** | not met: 7 cells, 5 slash rows |

**E3 for `e12_final`: NOT MET.**

What stands between it and E3:

- free phishing QRY: +0.1141 CONVICTED
- control benign_P1Q1 SUB: +0.0416 UNDECIDED
- control benign_P1Q1 PTH: +0.0576 UNDECIDED
- control benign_P1Q1 QRY: +0.0559 UNDECIDED
- control benign_P0Q1 QRY: +0.0621 UNDECIDED
- control phishing_P1Q0 PTH: +0.0428 UNDECIDED
- slash row benign control P0Q1
- slash row phishing free_bare
- slash row phishing control P0Q0
