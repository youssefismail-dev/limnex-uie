# E13 — results

- **Registration:** `PREREG_E13.md` (ff2643a); amendment 1 (d784fb8), before any
  E13 URL existed.
- **Code:** a1ac866 (bundle `uie-e13-code` 0cd7c5d3…), unchanged since every run.
- **Runs:** E13_train, E13_decode and E13_generate on Kaggle (T4), 2026-10-02
  (results (21), (22), (23)). Verification, scoring and every reading ran on the
  laptop.
- **Read:** 2026-10-02, after validity passed.
- **Numbers:**
  - Appendix A is `e13_tables.py` (the registered readout);
  - Appendix B is `e13p_read.py` (P, read earlier the same day).

## In plain words

- **Nothing in E13 is adopted under its registered rules, and E3 is not met.** The
  registration said E3 could not be met here (§0); that holds.
- **D1, the slash rule, did what it was built to do. Its last row still misses,
  through a route it was registered not to touch.**
  - Every host planned without a path now obeys its slash:
    - phishing's 57 such misses fell to 0;
    - benign `host?q` told "no slash" went from obeying 70% of the time to 100%.
  - Seven of the eight slash rows now pass, up from five. No regression anywhere.
  - The eighth row, phishing free bare hosts, still misses: 19 of 23,385 hosts lack
    the slash. All 19 come from plans *with* a path, where the model dropped the
    path it was told to write and ended the URL at the host.
  - K13-3 needs all eight rows, so D1 is not adopted.
- **D2, field calibration, had no measurable effect**, as its tiny fitted settings
  suggested (Δ +0.0018 and −0.0018, not significant).
  - Both target cells sat on the UNDECIDED / CONVICTED boundary and tipped to
    CONVICTED.
  - The registered R-a counts any verdict that worsens, so it fails.
- **C, the context arm, moved free phishing QRY by 0.0137: from 0.1141 to 0.1004.**
  - That is just under the 0.0141 it needed, and the cell stays CONVICTED.
  - Its mechanism check passed on the margin, but for the wrong reason:
    - queries after the model's own context improved only a little (0.124 → 0.109);
    - queries after a *real* context got worse (0.046 → 0.071).
  - The model learned to lean less on the context. That is the decoupling risk the
    roadmap named.
- **N produced the most informative result in E13, and the registered rule cannot
  read it.**
  - **The registered reading is UNAVAILABLE.** The power rule voids one reference
    in each benign cell, so nothing is licensed. Free phishing reads EXPOSURE, which
    is reported only.
  - **Measured, four times the query practice (A2) cut the query tell sharply**,
    with copying rising toward the natural rate but never over it:
    - free phishing QRY: 0.114 → 0.055;
    - benign P0Q1 QRY: 0.058 → 0.028, which reads CLEARED;
    - benign P1Q1 QRY: barely moved.
  - That is the opposite of N2a on the old objective, where the same test changed
    nothing.
- **P: path and subdomain subwords are ruled out** (Appendix B).

## 1. Validity: PASS

| check | result |
|---|---|
| VERIFY_TRAIN | e13_ctx and N's arms as registered. e13_ctx started from e12_final's exact weights; ContextSwap matched 8,130 of 8,130 rows; its swap share was 0.5023, inside the 0.5 ± 0.0086 band |
| VERIFY_DECODE | 115 files; byte identity on the laptop CPU, D1 and D2 included |
| VERIFY_GENERATE | 50 files; byte identity 1,162 of 1,162 |
| V4 | e12_final's free cells re-scored from E12's files equal E12's record (difference 0.0) |

## 2. C — the context arm (not adopted)

- **K13-1 FAIL.** Free phishing QRY fell from 0.1141 to 0.1004. The fall, 0.0137,
  is under the 0.0141 the rule needs, and the cell is still CONVICTED.
- **K13-2 PASS, narrowly.** x_GV − x_RC is 0.0377, against a bar of 0.0387.
  - The gap shrank mostly because x_RC rose (0.0463 → 0.0713).
  - x_GV fell only a little (0.1237 → 0.1090).
  - The per-seed gaps range from −0.024 to 0.091.
- **G1–G4 and G7 PASS.** No CLEARED cell crossed the retrain rule, presence held,
  every passed slash row still passes, honouring held, and FULL held.
- **G5/G6 FAIL on DC1's query-copying clause.** Copying rose: 0.1686 → 0.1897 free,
  0.2114 → 0.2269 control.
  - Both are still far under the natural rate. The clause counts any rise as
    worsening, which conflicts with E12's adopted rule: judge copying against the
    natural rate, in both directions.
  - I registered it without noticing the conflict. It does not change the outcome,
    because K13-1 failed.

## 3. D1 — the slash rule (not adopted)

- **K13-3 FAIL: 7 of 8 rows pass.** Phishing free bare hosts read 99.919% against a
  real 100.000% (±0.042).
- **The misses, attributed from the seeds** (each URL's supplied state):

  | arm | bare phishing hosts without the slash | from P0 plans (D1's scope) | from P1 plans (path dropped) |
  |---|---|---|---|
  | e12_final | 76 | 57 | 19 |
  | e12_d1 | 19 | 0 | 19 |

- **S honouring in P0 states is exact at the token level** (Appendix A).
  - The rendered strings show 2 misses in about 30,000 control URLs.
  - Both are a `?` the model wrote inside the host. The parser then cuts the host
    there, and the slash it wrote lands in the "query".
- **R-a to R-f PASS.** No cell regressed (DC1's paired rule with Holm), and
  presence, honouring, copying and novelty all held.

## 4. D2 — field calibration (not adopted)

- **K13-4 FAIL.**
  - D2-P (benign P1Q1 PTH): Δ −0.0018, p 0.69.
  - D2-Q (benign P0Q1 QRY): Δ +0.0018, p 0.52.
- **R-a FAIL.** Both targets tipped UNDECIDED → CONVICTED at Δ ≈ 0.
  - DC1's own rule counted only CLEARED → CONVICTED; mine counts any worsening, so
    noise at the boundary fails it.
- **R-d FAIL.** Benign P0Q1's copy rate moved further from its natural rate
  (amendment 1's rule).

## 5. N — N2a on the current recipe (diagnostic)

- **Registered reading: UNAVAILABLE** (benign decides), so nothing is licensed.
  - Benign P0Q1: the reference Q-VAL (0.0583) is under its power floor (0.0633).
  - Benign P1Q1: the Q-TRAIN reference is void.
- **Free phishing QRY: EXPOSURE** (A2's validation closure is 0.52), reported. It
  does not extend, because benign is UNAVAILABLE.
- **A1 is far worse than e12_final everywhere.** Training on query rows alone, at
  e12_final's own exposure, loses the rest of the URL.

## 6. Predictions (section 4), scored

| prediction | held? |
|---|---|
| D1: all 8 slash rows within T | **no**: 7 of 8 (P1 path drops) |
| D1: S honouring 100% in P0 states; no R fails | yes |
| D2: D2-Q applies; D2-P uncertain | yes; both applied |
| D2: at most a quarter of the tell; neither clears | neither clears; there was no measurable effect at all |
| C: free phishing QRY to 0.05–0.08, UNDECIDED | **no**: 0.1004, CONVICTED |
| C: K13-2 holds | yes, narrowly, and mostly through x_RC rising |
| C: every guard holds | **no**: G6's query-copying clause |
| N: SPARSITY or CAPACITY-OR-REPRESENTATION | **no**: UNAVAILABLE; descriptively the opposite (exposure helped) |
| P: no MIX-LED, no BLENDING; subwords ruled out | yes |

## 7. What this leaves for the owner

None of these is licensed by E13's registered rules. Each needs the owner's
decision and a new registration.

1. **Extending D1 to plans with a path.** P = 1 implies S = 1, so forbidding a
   slash-less host end there too is slash obedience.
   - It would turn the 19 remaining misses into slashed URLs; the paths are still
     missing.
   - It is the one change that the evidence says closes the slash condition.
2. **Query exposure (N2b).** N's measurements point to it: free phishing reads
   EXPOSURE, and benign P0Q1's closure is 0.53 but void by the power rule.
   - The registered aggregation cannot license it.
   - `E12_ROADMAP.md` §2.4 left reopening N2b to the owner.
3. **The context arm and D2:** no case to continue.
4. **Capacity (R3): still locked.** N did not read CAPACITY-OR-REPRESENTATION, and
   N2b is now the open candidate.

## Appendix A — the registered readout

`python research/e13/e13_tables.py`, verbatim.

### Validity

VERIFY_TRAIN, VERIFY_DECODE, VERIFY_GENERATE: PASS, PASS, PASS; V4: max |x - published| 0.0

### The criteria

| lever | criterion | result |
|---|---|---|
| C (e13_ctx) | K13-1 | **FAIL** |
| C (e13_ctx) | K13-2 | PASS |
| C (e13_ctx) | G1 | PASS |
| C (e13_ctx) | G2 | PASS |
| C (e13_ctx) | G3 | PASS |
| C (e13_ctx) | G4 | PASS |
| C (e13_ctx) | G5_G6 | **FAIL** |
| C (e13_ctx) | G7 | PASS |
| C (e13_ctx) | **adopted** | **no** |
| D1 (e12_d1) | K13-3 | **FAIL** |
| D1 (e12_d1) | R-a | PASS |
| D1 (e12_d1) | R-b | PASS |
| D1 (e12_d1) | R-c | PASS |
| D1 (e12_d1) | R-d | PASS |
| D1 (e12_d1) | R-e | PASS |
| D1 (e12_d1) | R-f | PASS |
| D1 (e12_d1) | **adopted** | **no** |
| D2 (e12_dec) | K13-4 | **FAIL** |
| D2 (e12_dec) | R-a | **FAIL** |
| D2 (e12_dec) | R-b | PASS |
| D2 (e12_dec) | R-c | PASS |
| D2 (e12_dec) | R-d | **FAIL** |
| D2 (e12_dec) | R-e | PASS |
| D2 (e12_dec) | R-f | PASS |
| D2 (e12_dec) | **adopted** | **no** |

K13-1: free phishing QRY +0.1141 -> +0.1004 (fall 0.0137; needs > 0.0141 and not CONVICTED); verdict CONVICTED.
K13-2: x_GV +0.1090, x_RC +0.0713, gap 0.0377 (needs <= 0.0387; e12_final: x_GV +0.1237, x_RC +0.0463, gap 0.0774).
G5/G6 (DC1's novelty rule): free novelty 0.9947 -> 0.9956; free query copying 0.1686 -> 0.1897 WORSENED; control novelty 0.9963 -> 0.9966; control query copying 0.2114 -> 0.2269 WORSENED
K13-4: D2-P control benign P1Q1 PTH: delta -0.0018 (p 0.69), UNDECIDED -> CONVICTED; D2-Q control benign P0Q1 QRY: delta +0.0018 (p 0.52), UNDECIDED -> CONVICTED
D2 R-a: benign_P1Q1 PTH UNDECIDED -> CONVICTED (delta -0.0018, Holm p 1.00); benign_P0Q1 QRY UNDECIDED -> CONVICTED (delta +0.0018, Holm p 1.00)

### The slash rows (K12b; T as registered)

| row | real | e12_final | e12_d1 | e13_ctx |
|---|---|---|---|---|
| benign free_bare | 98.004% (±0.626) | 97.731% | 97.802% | 97.736% |
| benign control P0Q0 | 97.944% (±0.518) | 97.704% | 97.885% | 97.731% |
| benign free_host_query | 99.576% (±1.502) | 99.371% | 99.528% | 99.688% |
| benign control P0Q1 | 99.551% (±0.244) | 99.812% **miss** | 99.771% | 99.744% |
| phishing free_bare | 100.000% (±0.042) | 99.675% **miss** | 99.919% **miss** | 99.705% **miss** |
| phishing control P0Q0 | 100.000% (±0.067) | 99.759% **miss** | 100.000% | 99.793% **miss** |
| phishing free_host_query | 100.000% (±1.544) | 98.786% | 99.541% | 99.694% |
| phishing control P0Q1 | 100.000% (±0.200) | 99.960% | 100.000% | 99.899% |

### Gated cells that are not CLEARED in some arm, and FULL

| cell | e12_final | e12_d1 | e12_dec | e13_ctx |
|---|---|---|---|---|
| free benign FULL | +0.0168 AUXILIARY | +0.0164 AUXILIARY | +0.0158 AUXILIARY | +0.0183 AUXILIARY |
| free phishing QRY | +0.1141 CONVICTED | +0.1156 CONVICTED | +0.1156 CONVICTED | +0.1004 CONVICTED |
| free phishing FULL | +0.0219 AUXILIARY | +0.0215 AUXILIARY | +0.0215 AUXILIARY | +0.0234 AUXILIARY |
| control benign_P1Q1 LBL | +0.0262 CLEARED | +0.0262 CLEARED | +0.0252 CLEARED | +0.0371 UNDECIDED |
| control benign_P1Q1 SUB | +0.0416 UNDECIDED | +0.0416 UNDECIDED | +0.0399 UNDECIDED | +0.0450 UNDECIDED |
| control benign_P1Q1 PTH | +0.0576 UNDECIDED | +0.0576 UNDECIDED | +0.0558 CONVICTED | +0.0525 UNDECIDED |
| control benign_P1Q1 QRY | +0.0559 UNDECIDED | +0.0559 UNDECIDED | +0.0411 UNDECIDED | +0.0529 UNDECIDED |
| control benign_P1Q1 FULL | +0.0699 AUXILIARY | +0.0699 AUXILIARY | +0.0638 AUXILIARY | +0.0734 AUXILIARY |
| control benign_P1Q0 FULL | +0.0103 AUXILIARY | +0.0103 AUXILIARY | +0.0103 AUXILIARY | +0.0188 AUXILIARY |
| control benign_P0Q1 QRY | +0.0621 UNDECIDED | +0.0624 UNDECIDED | +0.0643 CONVICTED | +0.0617 UNDECIDED |
| control benign_P0Q1 FULL | +0.0813 AUXILIARY | +0.0812 AUXILIARY | +0.0835 AUXILIARY | +0.0868 AUXILIARY |
| control benign_P0Q0 FULL | +0.0123 AUXILIARY | +0.0129 AUXILIARY | +0.0129 AUXILIARY | +0.0138 AUXILIARY |
| control phishing_P1Q0 PTH | +0.0428 UNDECIDED | +0.0428 UNDECIDED | +0.0428 UNDECIDED | +0.0396 UNDECIDED |
| control phishing_P1Q0 FULL | +0.0447 AUXILIARY | +0.0447 AUXILIARY | +0.0447 AUXILIARY | +0.0453 AUXILIARY |
| control phishing_P0Q0 FULL | +0.0213 AUXILIARY | +0.0279 AUXILIARY | +0.0279 AUXILIARY | +0.0293 AUXILIARY |

### The slash misses, by supplied state (phishing free bare hosts; S from the seeds)

| arm | bare hosts | without the slash | by supplied state |
|---|---|---|---|
| e12_final | 23,376 | 76 | {'P0Q0S1': 56, 'P0Q1S1': 1, 'P1Q0S1': 17, 'P1Q1S1': 2} |
| e12_d1 | 23,385 | 19 | {'P1Q0S1': 17, 'P1Q1S1': 2} |

### S honouring (control P0 states, and free mode with every state)

| population | S | e12_final | e12_d1 | e13_ctx |
|---|---|---|---|---|
| control_benign_P1Q1 | S1 | 99.913% of 15,000 | 99.913% of 15,000 | 99.920% of 15,000 |
| control_benign_P1Q0 | S1 | 99.920% of 15,000 | 99.920% of 15,000 | 99.960% of 15,000 |
| control_benign_P0Q1 | S0 | 70.423% of 71 | 100.000% of 71 | 69.014% of 71 |
| control_benign_P0Q1 | S1 | 99.806% of 14,929 | 99.993% of 14,929 | 99.826% of 14,929 |
| control_benign_P0Q0 | S0 | 97.468% of 316 | 100.000% of 316 | 96.835% of 316 |
| control_benign_P0Q0 | S1 | 99.748% of 14,684 | 100.000% of 14,684 | 99.762% of 14,684 |
| free_benign | S0 | 94.872% of 195 | 100.000% of 195 | 94.872% of 195 |
| free_benign | S1 | 99.887% of 39,805 | 99.932% of 39,805 | 99.892% of 39,805 |
| control_phishing_P1Q1 | S1 | 99.740% of 5,000 | 99.740% of 5,000 | 99.860% of 5,000 |
| control_phishing_P1Q0 | S1 | 99.880% of 15,000 | 99.880% of 15,000 | 99.873% of 15,000 |
| control_phishing_P0Q1 | S1 | 99.620% of 5,000 | 100.000% of 5,000 | 99.700% of 5,000 |
| control_phishing_P0Q0 | S1 | 99.693% of 15,000 | 99.993% of 15,000 | 99.760% of 15,000 |
| free_phishing | S1 | 99.790% of 40,000 | 99.945% of 40,000 | 99.823% of 40,000 |

K13-2's per-seed gaps (x_GV - x_RC): -0.024 to +0.091; void 0; 1,040 of 1,068 real-context samples valid
VERIFY_TRAIN: ContextSwap matched 8,130 of 8,130; swapped 15,373 + 29 too long of 30,665 eligible draws: 0.5023 (band +- 0.0086)
VERIFY_DECODE byte identity: e12_d1/free 669/669; e12_d1/control 515/515; e12_dec/free 669/669; e12_dec/control 515/515; N reference 272/272
VERIFY_GENERATE byte identity: e13_ctx/free 669/669; e13_ctx/control 493/493

### N — N2a on the current recipe (registered reading, then the measurements)

Reading: **UNAVAILABLE** -> nothing; reported; free phishing QRY: EXPOSURE

| cell | model | Q-VAL x | Q-TRAIN x | copy rate (natural) |
|---|---|---|---|---|
| control_benign_P0Q1_QRY | e12_final | +0.0583 UNDECIDED | +0.0907 CONVICTED | 0.2803 (0.4290) |
| control_benign_P0Q1_QRY | A1 | +0.1400 CONVICTED | +0.1731 CONVICTED | 0.2006 (0.4290) |
| control_benign_P0Q1_QRY | A2 | +0.0276 CLEARED | +0.0447 UNDECIDED | 0.3636 (0.4290) |
| control_benign_P1Q1_QRY | e12_final | +0.0519 UNDECIDED | +0.0437 UNDECIDED | 0.1593 (0.2477) |
| control_benign_P1Q1_QRY | A1 | +0.1140 CONVICTED | +0.1103 CONVICTED | 0.1220 (0.2477) |
| control_benign_P1Q1_QRY | A2 | +0.0492 UNDECIDED | +0.0347 CLEARED | 0.2108 (0.2477) |
| free_phishing_QRY | e12_final | +0.1141 CONVICTED | +0.1667 CONVICTED | 0.1680 (0.4185) |
| free_phishing_QRY | A1 | +0.2097 CONVICTED | +0.2898 CONVICTED | 0.0715 (0.4185) |
| free_phishing_QRY | A2 | +0.0551 UNDECIDED | +0.0966 CONVICTED | 0.3167 (0.4185) |

| cell | reading | reference Q-VAL x / power floor | reference Q-TRAIN x / power floor | A2 closure (Q-VAL, Q-TRAIN) |
|---|---|---|---|---|
| control_benign_P0Q1_QRY | UNAVAILABLE | 0.0583 / 0.0633 (VOID) | 0.0907 / 0.0144 (usable) | +0.53, +0.51 |
| control_benign_P1Q1_QRY | UNAVAILABLE | 0.0519 / 0.0433 (usable) | 0.0437 / 0.0466 (VOID) | +0.05, +0.21 |
| free_phishing_QRY | EXPOSURE | 0.1141 / 0.0522 (usable) | 0.1667 / 0.0511 (usable) | +0.52, +0.42 |

## Appendix B — P (paths and subdomains)

`python research/e13/e13p_read.py`, verbatim.

### P-1 — the mix, or the content (QC1-2's rule)

| cell | arm | n | x plain | x at the natural share | rho | t | reading | void | replicated |
|---|---|---|---|---|---|---|---|---|---|
| control_benign_P1Q1_PTH | e12_final | 1800 | +0.0602 | +0.0749 | 1.24 | -5.58 | CONTENT-LED | 0 | yes |
| control_benign_P1Q1_PTH | e12_base | 1800 | +0.0596 | +0.0846 | 1.42 | -5.71 | CONTENT-LED | 0 | yes |
| control_benign_P1Q1_SUB | e12_final | 974 | +0.0363 | +0.0319 | 0.88 | +1.79 | CONTENT-LED | 0 | yes |
| control_benign_P1Q1_SUB | e12_base | 974 | +0.0359 | +0.0311 | 0.87 | +1.93 | CONTENT-LED | 0 | yes |
| control_phishing_P1Q0_PTH | e12_final | 1800 | +0.0431 | +0.0846 | 1.96 | -13.24 | CONTENT-LED | 0 | yes |
| control_phishing_P1Q0_PTH | e12_base | 1800 | +0.0363 | +0.0830 | 2.28 | -14.54 | CONTENT-LED | 0 | yes |

### P-2 — near misses (one edit from a training string seen 5+ times; QC1-4's rule)

| cell | arm | generated never-seen, near | real never-seen, near | difference [95% CI] | reading | replicated |
|---|---|---|---|---|---|---|
| control_benign_P1Q1_PTH | e12_final | 4.09% of 9,465 | 3.88% of 2,601 | +0.21 [-0.64, +1.05] | NO BLENDING | no |
| control_benign_P1Q1_PTH | e12_base | 3.94% of 9,307 | 3.88% of 2,601 | +0.06 [-0.78, +0.90] | NO BLENDING | no |
| control_benign_P1Q1_SUB | e12_final | 18.25% of 707 | 16.83% of 303 | +1.41 [-3.67, +6.50] | NO BLENDING | no |
| control_benign_P1Q1_SUB | e12_base | 17.92% of 692 | 16.83% of 303 | +1.09 [-4.00, +6.18] | NO BLENDING | no |
| control_phishing_P1Q0_PTH | e12_final | 5.57% of 7,285 | 4.33% of 3,048 | +1.24 [+0.35, +2.14] | NO BLENDING | no |
| control_phishing_P1Q0_PTH | e12_base | 5.72% of 7,309 | 4.33% of 3,048 | +1.39 [+0.49, +2.29] | NO BLENDING | no |

### Licences (section 2.6)

| field | subword tokens licensed for E14 | by |
|---|---|---|
| PTH | **no: ruled out by these instruments** | {'MIX-LED': [], 'BLENDING': []} |
| SUB | **no: ruled out by these instruments** | {'MIX-LED': [], 'BLENDING': []} |
