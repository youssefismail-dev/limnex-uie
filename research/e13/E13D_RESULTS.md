# E13-D — results: where the remaining E3 failures live

- **Registration:** `PREREG_E13D.md` (f8aaad4). Amendment 1 (d81e46b) moved DX5 and
  V5 onto the model's own URLs; it came after the dry run and before any real
  measurement. Amendments 2 (DX7, d5012c2) and 3 (DX8, c592af5) were each
  registered after the previous reading and before their own measurement.
- **Code:** d81e46b. The run was `e13d_run.py`, 2026-10-01, about 50 minutes on the
  laptop.
- **Material:** E12's registered files and the `e12_final` checkpoint. Nothing was
  trained or generated.
- **Read:** 2026-10-01, after validity passed.
- **Numbers:** four appendices, pasted by script (`e13d_paste.py`):
  - A, B and C are the registered readouts: `e13d_read.py`, `e13d_read_dx7.py` and
    `e13d_read_dx8.py`;
  - D is `e13d_posthoc.py`: descriptive, run after DX8's reading, not registered,
    and it licenses nothing (section 5).

## In plain words

- **Every remaining content cell has the same diagnosis: LEARNING.** This holds in
  all six cells, replicated on `e12_base`.
  - **The data is the right data.** Real training URLs are indistinguishable from
    the real validation URLs the gate tests against (x_TV between −0.0044 and
    +0.0054). There is no population shift.
  - **The model has not learned it.** Its output is as distinguishable from its own
    training data as from validation (x_GT is 0.76–1.18 of x_GV).
- **There is enough data.** A character 6-gram model, trained on the same 8,272
  phishing training queries, writes queries the gate's detector barely tells apart
  from real ones: +0.036, against the transformer's +0.124 at the same n. (Corrected
  after DX8: this line used to say the learner "proves it". See the last bullet
  below.)

  | cell | transformer x_GV | 6-gram learner, full data |
  |---|---|---|
  | free phishing QRY (CONVICTED) | +0.1237 | +0.036 |
  | control benign P0Q1 QRY | +0.0679 | +0.015 |
  | control benign P1Q1 QRY | +0.0642 | +0.067 |
  | free benign QRY (CLEARED, reference) | +0.0325 at the gate | +0.061 |

  - **The transformer loses to the 6-gram exactly where data per cell is
    smallest:** phishing (8,272 query rows) and benign host-route queries
    (14,906).
  - **It matches or beats it where data is large:** benign path-plus-query
    (42,862) and free benign (57,768).
  - **The 6-gram's own curve is flat at full data**, so more data would not help it
    (DX3: NOT DATA-LIMITED in every query cell).
  - **Correction, after DX8 (section 5, descriptive):**
    - The 6-gram can only write 6-grams it has seen. In the two small cells, 55.3%
      and 46.3% of its queries are exact training queries, above the natural
      40.3% and 35.7%.
    - Where it writes as many never-seen queries as the transformer (benign P1Q1,
      free benign), its tell is no lower.
    - So its low tell shows that the training strings themselves pass the detector,
      as DX1 shows directly. It does not show that a model writing new strings
      could get more out of the data. The sentence that stood here ("the
      transformer does not extract it in the small cells") is withdrawn.
    - NOT DATA-LIMITED has independent support in DX7: the transformer's
      validation fit is within 0.06 nats per character of its training fit.
- **No data intervention is licensed.** No cell reads SHIFT-DOMINANT or
  DATA-LIMITED. The owner's data authorization applies only with a defensible data
  cause, and the diagnosis found none.
- **What is licensed: a learning-side lever, for all six cells.** The registration
  requires the specific lever to be justified on its own before it is proposed.
  DX7 and DX8 (sections 3 and 4) then narrowed it to one cell.
- **Slash obedience: MIXED everywhere, licensing nothing.**
  - The probe passes its own check (V5) in all seven populations.
  - Misses are concentrated: in six of the seven populations, the top tenth of URLs
    carries 69–85% of the violation mass. In benign `host?q` without a slash,
    where misses are common (29%), it carries 34%.
  - They are not concentrated where training data is thin.
  - On the model's own hosts, misses are about twice as likely as on real hosts:
    phishing bare hosts 0.284% against 0.159%.
  - Benign `host?q` without a slash is thin as a whole: 67 training rows, about
    20 per suffix. The registered rule is relative (top decile against the
    population), so it cannot see a uniformly thin population. This is noted, not
    licensed.
- **Distance to a reliable clear (DX4):** the family's mean x would clear in under
  1% of new runs for five cells. Phishing control P1Q0 PTH clears about half the
  time (0.53): it is noise-level.
- **E12's side effects, explained (DX6):**
  - The copying drop is in short queries: free phishing queries of 1–10
    characters copy 41.5% against 51.1% (`e12_base`).
  - The phishing path-plus-query dip is mostly a dropped query: 3.18% against
    2.50%.

## 1. Validity: PASS

| check | result |
|---|---|
| V1 | E12's VERIFY PASS; every input file and the checkpoint unchanged since the run |
| V2 | the harness reproduces the gate within ±0.05 in all six cells; the largest difference is +0.0140 (free phishing QRY) |
| V3 | every pool and training twin is one URL per host, built by the gate's own functions |
| V4 | one void seed in 144 cell-seeds (benign P0Q1 QRY) |
| V5 (amended) | the probe's mean violation lies inside the observed violations' 95% interval in all seven generated populations |

## 2. What the evidence licenses next

- The six cells share one licence: a learning-side lever from the allowed set
  (objective, training-data weighting, conditioning).
- **Excluded by the project's record:**
  - more optimisation (R2: "more optimisation is not the repair");
  - the closed boundary-weighting branch (S4/S4b);
  - capacity, extra steps and decoder changes (the owner's rules).
- **Choosing among the remaining learning-side levers needs one more measurement,
  registered as amendment 2 before it is made:**
  - Does the transformer assign lower probability to real held-out queries and
    paths than a simple smoothed n-gram does (under-fitting)?
  - Or does it fit them, but sample badly?

  The two answers point to different levers, and guessing between them is what
  the owner ruled out.

## 3. Amendment 2 — DX7: the model fits its data (FITS in all six cells)

- **The transformer predicts real held-out field strings far better than an order-6
  n-gram does,** in every cell. The gap is 0.17–0.46 nats per character, and every
  95% interval is below 0.
- **It barely overfits:** validation is within +0.06 nats per character of a
  training sample.
- **So, by the registered test, the model is not under-fitting.**
  - The test is relative. It rules out a model worse than a simple smoothed one; it
    does not show that no better fit exists.
  - The model knows its strings. Its samples are what drift.
- **Licensed by DX7:** no lever on fitting real strings better. The gap is in
  sampling, which DX8 then split (section 4).
- **Prediction** (registered): UNDER-FIT in the two small cells, free phishing QRY
  and benign P0Q1 QRY. Wrong in both. FITS or UNRESOLVED in benign P1Q1 QRY and free
  benign QRY: right.
- (Appendix B: `e13d_read_dx7.py`.)

## 4. Amendment 3 — DX8: the context or the field's own sampling

The registered test: each field is sampled after a **real** context (the real URL
teacher-forced up to the field), and its tell is compared with fully generated
output, paired seed for seed (Appendix C: `e13d_read_dx8.py`; V8, V9, V10 PASS).

| cell | generated (x_GV) | real context (x_RC) | DX8 |
|---|---|---|---|
| free phishing QRY | +0.1237 | +0.0463 | **CONTEXT-LED** (ratio 0.37) |
| control benign P0Q1 QRY | +0.0679 | +0.0612 | FIELD-LED (ratio 0.90) |
| control benign P1Q1 QRY | +0.0642 | +0.0481 | SPLIT (0.75) |
| control benign P1Q1 PTH | +0.0602 | +0.0518 | FIELD-LED (0.86) |
| control benign P1Q1 SUB | +0.0363 | +0.0183 | SPLIT (0.50) |
| control phishing P1Q0 PTH | +0.0431 | +0.0304 | SPLIT (0.70) |

- **The far cell is mostly the context.** About 63% of the phishing query tell comes
  from the model's own host and path. Given a real one, its queries are 0.046.
  - It is consistent with a side observation in DX5, registered as licensing
    nothing. In the four slash-requested populations with a real counterpart, the
    probe's miss rate is 1.8–2.0 times higher on the model's own hosts (phishing
    bare hosts 0.284% against 0.159%).
  - Section 5 shows the same mechanism directly: the model's paths before a query
    are far more often never-seen than real ones.
  - Licensed: a training-side lever on exposure to the model's own context.
- **What remains, even with a perfect context, is the field's own sampling at
  temperature 1:**
  - 0.046 (phishing QRY), 0.061 (benign P0Q1 QRY), 0.048 and 0.052 (benign P1Q1
    QRY and PTH), 0.030 (phishing P1Q0 PTH), 0.018 (benign P1Q1 SUB);
  - four of the six are above their DX4 clear line: 0.037, 0.034, 0.039 and 0.042.
    The other two, phishing P1Q0 PTH and benign P1Q1 SUB, are below theirs (0.041
    and 0.029). These are the harness's scale, which matched the gate within 0.014
    (V2);
  - DX8's registered table matches this remainder only to a decoder-side lever,
    which the owner's rules exclude.
- **What this means for E3:**
  - Under the raw decode, the licensed lever can at best bring free phishing QRY
    to about its real-context level. That is 0.046 on the harness's scale. On the
    gate's scale it is about 0.043, a projection (the ratio 0.37 applied to the
    gate's 0.1141). Its clear line is 0.037, so the best case is UNDECIDED, not
    CLEARED.
  - The field-sampling remainder would still keep four of the six cells above
    their clear lines.
  - This is the evidence `E13_ROADMAP.md` puts to the owner.
- **Prediction** (registered): FIELD-LED for the query cells. Right for benign P0Q1
  QRY only: free phishing read CONTEXT-LED, benign P1Q1 QRY SPLIT. SPLIT or
  CONTEXT-LED for the PTH and SUB cells: right for SUB and phishing P1Q0 PTH, wrong
  for benign P1Q1 PTH (FIELD-LED). Three of six.

## 5. Post-hoc, descriptive: the copy deficit (diagnosis target 1)

Run after DX8's reading (`e13d_posthoc.py`, Appendix D). Not registered; it licenses
nothing. It answers, on E12, the first diagnosis target `DECISION_2026-10-01.md`
carried into E13: "copying moved further below the natural rate".

- **E12 writes too few exact real queries and too many never-seen ones** in every
  query cell, the CLEARED reference included:

  | cell | exact copies, real / E12 | never seen, real / E12 |
  |---|---|---|
  | free phishing QRY | 40.3% / 16.7% | 50.6% / 77.2% |
  | control benign P0Q1 QRY | 35.7% / 20.5% | 39.8% / 55.1% |
  | control benign P1Q1 QRY | 20.3% / 13.3% | 59.2% / 72.4% |
  | free benign QRY (CLEARED) | 25.2% / 14.0% | 53.7% / 68.0% |

- **The new strings are not near misses** of training strings: 2.1–3.3% of E12's
  queries are one edit from one, against 0.8–2.4% of real queries.
- **They are not another population's strings either.** Phishing queries that are
  benign training queries make up 1.6%, against 0.4% real.
- **In free phishing, the deficit starts in the path.**
  - Of the paths that come before a query, 24.0% are exact training paths, against
    57.8% real.
  - Even after an exact path, the query is exact 15.5% of the time, against 35.4%
    real.
  - This is DX8's CONTEXT-LED reading, seen in the strings.
- **This is QC1's picture on E11, replicated on E12** (QC1-1 UNDER, QC1-4 NO
  BLENDING, QC1-5 NO LEAK).
  - QC1-2 already showed what it means. Re-drawing the output to the natural copy
    share makes the tell bigger, because the strings the model copies are the
    wrong ones, at the wrong frequencies.
  - So the deficit is a symptom of which strings the model writes, not a lever.
    "Copy more" is not a repair.
- **The caveat on DX3** (see "In plain words"):
  - the 6-gram's queries are exact training queries 55.3% (phishing) and 46.3%
    (benign P0Q1) of the time;
  - where its never-seen share matches the transformer's (benign P1Q1: 73.1%
    against 72.4%; free benign: 68.0% against 68.0%), its tell is no lower (+0.067
    and +0.061).

## Appendix A — the rendered tables

`python research/e13/e13d_read.py`, verbatim.

### Validity

V1–V4: PASS; failures: none

| cell | harness x (gate n) | gate x | difference |
|---|---|---|---|
| free_phishing_QRY | +0.1281 | +0.1141 | +0.0140 |
| control_benign_P0Q1_QRY | +0.0618 | +0.0621 | -0.0003 |
| control_benign_P1Q1_QRY | +0.0450 | +0.0559 | -0.0109 |
| control_benign_P1Q1_PTH | +0.0547 | +0.0576 | -0.0029 |
| control_benign_P1Q1_SUB | +0.0394 | +0.0416 | -0.0022 |
| control_phishing_P1Q0_PTH | +0.0391 | +0.0428 | -0.0037 |

### DX1 and DX2 — where each cell's gap lives

| cell | n | x_GV | x_TV (share) | DX1 | x_GT final (share) | x_GT base (share) | DX2 | attribution | void seeds |
|---|---|---|---|---|---|---|---|---|---|
| free_phishing_QRY | 356 | +0.1237 | -0.0019 (-0.02) | NO SHIFT | +0.1161 (0.94) | +0.1208 (1.11) | LEARNING-GAP | **LEARNING** | 0 |
| control_benign_P0Q1_QRY | 600 | +0.0679 | -0.0033 (-0.05) | NO SHIFT | +0.0580 (0.85) | +0.0441 (0.76) | LEARNING-GAP | **LEARNING** | 1 |
| control_benign_P1Q1_QRY | 1800 | +0.0642 | -0.0029 (-0.04) | NO SHIFT | +0.0607 (0.95) | +0.0568 (0.93) | LEARNING-GAP | **LEARNING** | 0 |
| control_benign_P1Q1_PTH | 1800 | +0.0602 | +0.0054 (0.09) | NO SHIFT | +0.0614 (1.02) | +0.0541 (0.86) | LEARNING-GAP | **LEARNING** | 0 |
| control_benign_P1Q1_SUB | 974 | +0.0363 | -0.0044 (-0.12) | NO SHIFT | +0.0424 (1.17) | +0.0408 (1.18) | LEARNING-GAP | **LEARNING** | 0 |
| control_phishing_P1Q0_PTH | 1800 | +0.0431 | -0.0011 (-0.03) | NO SHIFT | +0.0380 (0.88) | +0.0336 (0.77) | LEARNING-GAP | **LEARNING** | 0 |

### DX3 — is there enough data? (query cells; free benign QRY is the CLEARED reference)

| cell | val | train | coverage 1/8 · 1/4 · 1/2 · 1 | g | unseen templates | singleton templates | learner x 1/8 · 1/4 · 1/2 · 1 | s (t) | DX3 |
|---|---|---|---|---|---|---|---|---|---|
| free_phishing_QRY | 1,068 | 8,272 | 56.9% · 63.0% · 68.1% · 72.2% | +4.1 pts | 27.8% | 28.4% | +0.097 · +0.061 · +0.035 · +0.036 | -0.0019 (-0.18) | NOT DATA-LIMITED |
| control_benign_P0Q1_QRY | 1,802 | 14,906 | 70.4% · 74.2% · 77.7% · 81.0% | +3.2 pts | 19.0% | 19.9% | +0.074 · +0.054 · +0.009 · +0.015 | -0.0065 (-0.60) | NOT DATA-LIMITED |
| control_benign_P1Q1_QRY | 5,451 | 42,862 | 53.0% · 58.3% · 63.0% · 67.3% | +4.3 pts | 32.7% | 32.0% | +0.116 · +0.084 · +0.073 · +0.067 | +0.0052 (+1.30) | NOT DATA-LIMITED |
| free_benign_QRY (reference) | 7,253 | 57,768 | 59.2% · 63.8% · 68.1% · 71.7% | +3.7 pts | 28.3% | 27.9% | +0.098 · +0.072 · +0.066 · +0.061 | +0.0046 (+1.23) | — |

### DX4 — distance to a reliable clear (descriptive)

| cell | E11 | e12_base | e12_final | e12_no_r1 | mean | between-run SD | clear line | P(CLEARED) | mean x for P ≥ 0.8 |
|---|---|---|---|---|---|---|---|---|---|
| free_phishing_QRY | +0.0953 | +0.1101 | +0.1141 | +0.1070 | +0.1066 | 0.0081 | +0.0370 | 0.00 | +0.0301 |
| control_benign_P0Q1_QRY | +0.0581 | +0.0566 | +0.0621 | +0.0585 | +0.0588 | 0.0023 | +0.0342 | 0.00 | +0.0322 |
| control_benign_P1Q1_QRY | +0.0511 | +0.0509 | +0.0559 | +0.0594 | +0.0543 | 0.0041 | +0.0389 | 0.00 | +0.0354 |
| control_benign_P1Q1_PTH | +0.0509 | +0.0587 | +0.0576 | +0.0538 | +0.0553 | 0.0036 | +0.0417 | 0.00 | +0.0387 |
| control_benign_P1Q1_SUB | +0.0379 | +0.0408 | +0.0416 | +0.0498 | +0.0425 | 0.0051 | +0.0292 | 0.00 | +0.0249 |
| control_phishing_P1Q0_PTH | +0.0403 | +0.0377 | +0.0428 | +0.0430 | +0.0410 | 0.0025 | +0.0411 | 0.53 | +0.0390 |

### DX5 — slash obedience (teacher-forced, `e12_final`, its own URLs: amendment 1)

V5 (the probe against the same URLs' observed violations): PASS

| population | n | observed violations | probe mean | 95% interval | within |
|---|---|---|---|---|---|
| gen_phishing_P0Q0_S1 | 15,000 | 0.307% | 0.284% | [0.230, 0.409] | yes |
| gen_phishing_P0Q1_S1 | 4,985 | 0.381% | 0.491% | [0.244, 0.595] | yes |
| gen_benign_P0Q1_S0 | 71 | 29.577% | 28.731% | [20.233, 41.019] | yes |
| gen_benign_P0Q1_S1 | 14,925 | 0.194% | 0.163% | [0.135, 0.279] | yes |
| gen_benign_P0Q0_S0 | 316 | 2.532% | 2.678% | [1.288, 4.915] | yes |
| gen_benign_P0Q0_S1 | 14,684 | 0.252% | 0.243% | [0.183, 0.347] | yes |
| gen_free_phishing_P0Q0_S1 | 23,398 | 0.261% | 0.232% | [0.203, 0.335] | yes |

| population | n | mean violation | top-decile share c | support: top decile / all | DX5 |
|---|---|---|---|---|---|
| gen_phishing_P0Q0_S1 | 15,000 | 0.284% | 0.79 | 12,973 / 13,085 | MIXED |
| gen_phishing_P0Q1_S1 | 4,985 | 0.491% | 0.84 | 308 / 308 | MIXED |
| gen_benign_P0Q1_S0 | 71 | 28.731% | 0.34 | 22 / 18 | MIXED |
| gen_benign_P0Q1_S1 | 14,925 | 0.163% | 0.78 | 262 / 214 | MIXED |
| gen_benign_P0Q0_S0 | 316 | 2.678% | 0.69 | 87 / 115 | MIXED |
| gen_benign_P0Q0_S1 | 14,684 | 0.243% | 0.85 | 3,965 / 3,381 | MIXED |
| gen_free_phishing_P0Q0_S1 | 23,398 | 0.232% | 0.74 | 12,973 / 13,085 | MIXED |

Reported beside (as registered; licenses nothing): the same probe on real validation URLs

| population | n | mean violation (real) | mean violation (generated) |
|---|---|---|---|
| phishing_P0Q0_S1 | 3,000 | 0.159% | 0.284% |
| phishing_P0Q1_S1 | 397 | 0.246% | 0.491% |
| benign_P0Q1_S0 | 3 | 30.045% | 28.731% |
| benign_P0Q1_S1 | 1,794 | 0.088% | 0.163% |
| benign_P0Q0_S0 | 561 | 3.163% | 2.678% |
| benign_P0Q0_S1 | 3,000 | 0.125% | 0.243% |

### DX6 — E12's side effects (descriptive)

Copying by query length (free mode):

| class | length | e12_base | e12_final |
|---|---|---|---|
| benign | 1-10 | 57.0% of 782 | 47.2% of 759 |
| benign | 11-30 | 8.8% of 1,091 | 7.3% of 1,101 |
| benign | 31-80 | 0.8% of 653 | 1.1% of 630 |
| benign | 81+ | 0.0% of 101 | 0.0% of 141 |
| phishing | 1-10 | 51.1% of 595 | 41.5% of 513 |
| phishing | 11-30 | 11.2% of 653 | 12.3% of 685 |
| phishing | 31-80 | 3.5% of 484 | 3.5% of 487 |
| phishing | 81+ | 0.0% of 158 | 0.0% of 186 |

Phishing control P1Q1 (path and query requested), what was written:

| arm | P1Q1 (honoured) | P1Q0 (query dropped) | P0Q1 (path dropped) | P0Q0 |
|---|---|---|---|---|
| e12_base | 97.06% | 2.50% | 0.22% | 0.22% |
| e12_final | 96.02% | 3.18% | 0.52% | 0.28% |
| e12_no_r1 | 97.26% | 2.08% | 0.44% | 0.22% |

### Licences (section 3)

| item | reading | licenses |
|---|---|---|
| free_phishing_QRY | LEARNING; DX3 NOT DATA-LIMITED | a learning-side lever (objective, weighting, conditioning) |
| control_benign_P0Q1_QRY | LEARNING; DX3 NOT DATA-LIMITED | a learning-side lever (objective, weighting, conditioning) |
| control_benign_P1Q1_QRY | LEARNING; DX3 NOT DATA-LIMITED | a learning-side lever (objective, weighting, conditioning) |
| control_benign_P1Q1_PTH | LEARNING | a learning-side lever (objective, weighting, conditioning) |
| control_benign_P1Q1_SUB | LEARNING | a learning-side lever (objective, weighting, conditioning) |
| control_phishing_P1Q0_PTH | LEARNING | a learning-side lever (objective, weighting, conditioning) |
| slash gen_phishing_P0Q0_S1 | MIXED | nothing |
| slash gen_phishing_P0Q1_S1 | MIXED | nothing |
| slash gen_benign_P0Q1_S0 | MIXED | nothing |
| slash gen_benign_P0Q1_S1 | MIXED | nothing |
| slash gen_benign_P0Q0_S0 | MIXED | nothing |
| slash gen_benign_P0Q0_S1 | MIXED | nothing |
| slash gen_free_phishing_P0Q0_S1 | MIXED | nothing |

## Appendix B — DX7 (amendment 2)

`python research/e13/e13d_read_dx7.py`, verbatim.

### DX7 validity

V6, V7: PASS; failures: none

| cell | strings (val) | mismatched / skipped | transformer NLL (val) | 6-gram NLL (val) | Δ [95% CI] | DX7 | transformer NLL (train sample) | its gap (val − train) |
|---|---|---|---|---|---|---|---|---|
| free_phishing_QRY | 1,048 | 0 / 20 | 1.998 | 2.362 | -0.364 [-0.420, -0.310] | FITS | 2.001 | -0.003 |
| control_benign_P0Q1_QRY | 1,797 | 0 / 5 | 1.481 | 1.745 | -0.265 [-0.294, -0.234] | FITS | 1.465 | +0.016 |
| control_benign_P1Q1_QRY | 5,407 | 0 / 44 | 1.592 | 2.047 | -0.456 [-0.472, -0.440] | FITS | 1.560 | +0.032 |
| control_benign_P1Q1_PTH | 5,407 | 0 / 44 | 1.224 | 1.521 | -0.297 [-0.312, -0.281] | FITS | 1.208 | +0.016 |
| control_benign_P1Q1_SUB | 2,891 | 0 / 33 | 0.597 | 0.766 | -0.169 [-0.191, -0.148] | FITS | 0.541 | +0.055 |
| control_phishing_P1Q0_PTH | 8,715 | 0 / 4 | 1.214 | 1.551 | -0.336 [-0.352, -0.321] | FITS | 1.154 | +0.060 |
| free_benign_QRY (reference) | 7,204 | 0 / 49 | 1.564 | 1.964 | -0.400 [-0.413, -0.387] | — | 1.516 | +0.048 |

### DX7 licences (amendment 2)

| cell | DX7 | licenses |
|---|---|---|
| free_phishing_QRY | FITS | no training lever (the gap is in sampling) |
| control_benign_P0Q1_QRY | FITS | no training lever (the gap is in sampling) |
| control_benign_P1Q1_QRY | FITS | no training lever (the gap is in sampling) |
| control_benign_P1Q1_PTH | FITS | no training lever (the gap is in sampling) |
| control_benign_P1Q1_SUB | FITS | no training lever (the gap is in sampling) |
| control_phishing_P1Q0_PTH | FITS | no training lever (the gap is in sampling) |

## Appendix C — DX8 (amendment 3)

`python research/e13/e13d_read_dx8.py`, verbatim.

### DX8 validity

V8, V9, V10: PASS; failures: none

| cell | n | valid samples | x_GV (generated) | x_RC (real context) | ratio | paired t | DX8 |
|---|---|---|---|---|---|---|---|
| free_phishing_QRY | 356 | 1,032 of 1,048 | +0.1237 | +0.0463 | 0.37 | +13.98 | **CONTEXT-LED** |
| control_benign_P0Q1_QRY | 600 | 1,796 of 1,797 | +0.0679 | +0.0612 | 0.90 | +1.62 | **FIELD-LED** |
| control_benign_P1Q1_QRY | 1800 | 5,378 of 5,407 | +0.0642 | +0.0481 | 0.75 | +5.07 | **SPLIT** |
| control_benign_P1Q1_PTH | 1800 | 5,396 of 5,407 | +0.0602 | +0.0518 | 0.86 | +2.29 | **FIELD-LED** |
| control_benign_P1Q1_SUB | 974 | 2,891 of 2,891 | +0.0363 | +0.0183 | 0.50 | +4.37 | **SPLIT** |
| control_phishing_P1Q0_PTH | 1800 | 8,689 of 8,715 | +0.0431 | +0.0304 | 0.70 | +2.47 | **SPLIT** |

### DX8 licences (amendment 3)

| cell | DX8 | licenses |
|---|---|---|
| free_phishing_QRY | CONTEXT-LED | a training-side lever on exposure to the model's own context (form and arms fixed in E13's registration) |
| control_benign_P0Q1_QRY | FIELD-LED | no lever within the current rules (decoder-side only): an owner decision |
| control_benign_P1Q1_QRY | SPLIT | nothing |
| control_benign_P1Q1_PTH | FIELD-LED | no lever within the current rules (decoder-side only): an owner decision |
| control_benign_P1Q1_SUB | SPLIT | nothing |
| control_phishing_P1Q0_PTH | SPLIT | nothing |

## Appendix D — post-hoc, descriptive (not registered; licenses nothing)

`python research/e13/e13d_posthoc.py`, verbatim.

**Post-hoc and descriptive: not a registered reading; licenses nothing.**

### A. Exact copies, digit variants, one-edit near misses, never seen (query cells)

| cell | source | n | exact copy | digit variant | one edit | never seen |
|---|---|---|---|---|---|---|
| free_phishing_QRY | real validation | 1,068 | 40.3% | 6.7% | 2.4% | 50.6% |
| free_phishing_QRY | transformer (e12_final) | 1,871 | 16.7% | 2.8% | 3.3% | 77.2% |
| free_phishing_QRY | 6-gram (DX3's learner) | 3,000 | 55.3% | 1.6% | 1.7% | 41.4% |
| control_benign_P0Q1_QRY | real validation | 1,802 | 35.7% | 23.6% | 0.8% | 39.8% |
| control_benign_P0Q1_QRY | transformer (e12_final) | 14,938 | 20.5% | 22.3% | 2.1% | 55.1% |
| control_benign_P0Q1_QRY | 6-gram (DX3's learner) | 3,000 | 46.3% | 6.6% | 1.5% | 45.6% |
| control_benign_P1Q1_QRY | real validation | 5,451 | 20.3% | 18.6% | 1.9% | 59.2% |
| control_benign_P1Q1_QRY | transformer (e12_final) | 14,779 | 13.3% | 12.0% | 2.4% | 72.4% |
| control_benign_P1Q1_QRY | 6-gram (DX3's learner) | 3,000 | 19.1% | 6.1% | 1.7% | 73.1% |
| free_benign_QRY | real validation | 7,253 | 25.2% | 19.4% | 1.7% | 53.7% |
| free_benign_QRY | transformer (e12_final) | 2,631 | 14.0% | 15.0% | 3.0% | 68.0% |
| free_benign_QRY | 6-gram (DX3's learner) | 3,000 | 23.5% | 6.4% | 2.1% | 68.0% |

### B. Copies of another population's training strings

| cell | source | own cell | other states | other class | none of these |
|---|---|---|---|---|---|
| free_phishing_QRY | real validation | 40.3% | 0.0% | 0.4% | 59.4% |
| free_phishing_QRY | transformer (e12_final) | 16.7% | 0.0% | 1.6% | 81.7% |
| control_benign_P0Q1_QRY | real validation | 35.7% | 1.7% | 0.0% | 62.6% |
| control_benign_P0Q1_QRY | transformer (e12_final) | 20.5% | 1.6% | 0.0% | 77.9% |
| control_benign_P1Q1_QRY | real validation | 20.3% | 0.8% | 0.0% | 78.8% |
| control_benign_P1Q1_QRY | transformer (e12_final) | 13.3% | 0.6% | 0.1% | 86.0% |
| free_benign_QRY | real validation | 25.2% | 0.0% | 0.0% | 74.8% |
| free_benign_QRY | transformer (e12_final) | 14.0% | 0.0% | 0.0% | 86.0% |

### C. Free phishing: the path before a query

| source | URLs | with a query | of which with a path | all paths: exact | paths before a query: exact | query: exact | (path, query) pair: exact | query exact, after an exact path | query exact, after a never-seen path |
|---|---|---|---|---|---|---|---|---|---|
| real validation (one URL per host) | 24,081 | 1,068 (4.44%) | 62.6% | 63.6% | 57.8% | 40.3% | 30.6% | 35.4% (n 387) | 28.0% (n 282) |
| e12_final, free mode | 40,000 | 1,871 (4.68%) | 64.8% | 46.6% | 24.0% | 16.7% | 9.2% | 15.5% (n 291) | 12.5% (n 921) |
