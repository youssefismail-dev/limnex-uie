# E16 — the capacity probe: can benign P1Q1 move at all?

- **Registered:** 2026-10-06, before any E16 code, training or generation. **Awaiting the
  owner's gate A.**
- **Rests on:** `research/e16/E16_DESIGN.md` (the evaluation), `research/native/
  DECISION_2026-10-06.md` (the new baseline, R3's unlock), `research/e15/E15B_RESULTS.md`.
- **Baseline:** `e15_new` (official since 2026-10-06), with its 50 files.
- **One arm:** `e16_cap`, a wider model; everything else is `e15_new`'s.

## 0. In plain words

- **The question:** benign P1Q1 QRY and PTH have not moved under any data or exposure
  lever. Does a model with 1.77× the parameters move them?
- **What changes:** only the model's width. 512 instead of 384 (feed-forward 2048 instead
  of 1536), 25,473,024 parameters instead of 14,386,176.
- **What does not change:**
  - the data, including E15's stream;
  - the batches, the steps and the objective;
  - the decode and the 50 generation jobs.
  So any difference is the width's.
- **How it is judged:** against `e15_new`, with every existing threshold.
  - **K16-1, the question:** P1Q1 QRY or PTH falls by more than 0.0141.
  - **K16-2, the check that the extra width was used:** held-out benign query fit falls,
    overall and on P1Q1 rows.
  - **Guards G1–G10 as in E15.** G5 is strict: all 8 slash rows within T.
- **An end rule, registered now:**
  - **P1Q1 moves:** the path to E3 continues.
  - **It moves partly:** E17 is the owner's choice of next step.
  - **Fit improves but P1Q1 does not move at all:** the next document is the final E3
    report.
- **Predicted:** most likely the third (about 50%). K16-1 passes about 20% of the time.

## 1. The owner's decisions, verbatim

**R3 and G5** (also `DECISION_2026-10-06.md` §5):

> I am officially unlocking R3 (Capacity). I agree with your analysis that we have
> exhausted all other avenues for P1Q1, and increasing the capacity (d_model 512,
> d_ff 2048) is the right next step.
> Second, regarding G5, leave the rule exactly as it is. We will maintain the strict E3
> criteria. If it fails on the same microscopic margin later, I will handle it then.

**The owner's question:**

> Since we are expanding the model's capacity, should we combine Option A and Option B1
> right now in E16? (i.e., feeding the larger model a doubled dose of data so it has
> enough exposure to utilize its new width).
> OR, is it better to run Option B1 alone as E16 to cleanly isolate the capacity
> variable, and if we still fall short of E3, we keep Option A in our back pocket to run
> as E17?

**The answer given, in short:** B1 alone.
- A's doubled stream adds about 1% of the training signal, in two query strata. It does
  not feed the wider model's width.
- Whether the width was used is measured directly (K16-2), and an unused width makes the
  probe inconclusive, not negative.
- P1Q1 already ignored one doubling, which `e16_cap` keeps.
- With G5 strict, A's extra slashed `host?query` practice adds a risk that could not be
  attributed.
- A also needs a new acquisition first.
- An end-rule row was added so that a partial response leads to E17, not to a final
  report.

**The owner's agreement:**

> yrs , agreed

**Not part of this registration:** the optional acquisition of A's rows on the laptop
during E16's Kaggle run. It needs its own explicit approval.

## 2. Why this experiment

`E16_DESIGN.md` §§2–4, in four lines:
- **Three exposure levers did not move P1Q1:**
  - N's 4× practice: 0.052 → 0.049;
  - E14's ×4 weight: 0.056 → 0.058;
  - E15-B's ×2 new rows: 0.052 → 0.050.
- **In E15, fit and tell came apart.** Held-out query fit improved by the same amount on
  P0Q1 rows (−0.0369) and P1Q1 rows (−0.0374); only P0Q1's tell moved.
- **Every non-capacity training family has now been tested** on P1Q1, or ruled out by a
  registered reading (`E16_DESIGN.md` §4.2). Capacity is the last one.
- **The probe is decisive either way.** P1Q1 decides whether E3 can be met.

## 3. The arm

| | `e15_new` (baseline) | `e16_cap` |
|---|---|---|
| width (d_model) | 384 | **512** |
| feed-forward (d_ff) | 1,536 | **2,048** (4 × d_model, as always) |
| layers, heads | 8, 8 | 8, 8 (head size 48 → 64) |
| parameters | 14,386,176 | **25,473,024** (1.771×) |
| start | E12's start at width 384 (`c7e71edf…`) | **E12's start procedure at width 512**: `new_model_e12` with the same seed (20260803) and the same new-row seeds. Its fingerprints are recorded in the dry run, before any training (V-start′) |
| data | corpus2 + `qry_extra.jsonl` (57,292 rows, `637accab…`), query targets only, p = 1.0, seeds 20261007 / 20261008 | the same |
| batches | the sampler at seed 20260803, batch 256, window 64; old-row digest `adbddc2a…` | the same: the same old rows and the same appended rows, in the same order |
| objective | fixed, C = 14397.885481960007 | the same |
| optimisation | 20,000 steps; lr 3e-4, warmup 500, weight decay 0.01; clip 1.0, then 5.0 after step 1,000; dropout 0.1; mixed precision | the same |
| **memory** | the whole batch at once (12.0 of 14.6 GB in the worst-case check) | **gradient accumulation: each batch split into 2 halves**, gradients summed before the clip and the step. The same batch and the same summed gradient (V-accum). If the worst-case check fails at 2, then 4, recorded; the arithmetic is unchanged |
| decode | raw + D1 + D1′ | the same |
| generation | E12's 50 jobs at E12's seeds (`data/e15/generate/gen`) | the same jobs and seeds; every cell paired with `e15_new`'s |

**What the accumulation is.** One training step today computes the loss on 256 rows (plus
the appended rows) and takes one optimiser step. `e16_cap` computes it on each half in
turn and adds the two gradients before clipping and stepping. Under the fixed objective
the loss is a sum divided by C, so the sum of the halves' gradients is the batch's
gradient. Only the order of floating-point additions and the dropout draws differ.

## 4. Instrument definitions

- **Baselines.**
  - Every cell, copy rate, novelty and honouring reading compares with **`e15_new`'s
    recorded numbers** (`data/e15/score_report.json`).
  - K16-2 and G3 compare with `e15_new`'s checkpoint and training record.
- **Training data,** for both arms: corpus2 plus the stream (`PREREG_E15.md` §13.2).
  - The copying (G7), novelty (G8) and band-profile (K16-3) references are the old
    corpus plus the stream, as `e15_new`'s readings were.
  - Both arms share one natural rate, so G7's distance rule needs no translation.
- **M3** (K16-2): `research/e14/e14_fit.py`'s `per_row_m3` and `paired`, unchanged.
  - The pooled per-character difference on every held-out query row, with a paired
    row-bootstrap 95% interval (1,000 draws, seed 20260806).
  - The self-check against N2a's own M3 runs first.
  - Per stratum, by the injector's stratum reader, as in K15-2.
- **The between-run SD** (the end rule's partial response) is E13-D DX4's, measured on four
  E12-family models: P1Q1 QRY 0.0041, P1Q1 PTH 0.0036.

## 5. Criteria: `e16_cap` against `e15_new`

A retrained model, so the retraining allowance (0.0141) applies. Every threshold below is
an existing one.

| code | criterion |
|---|---|
| **K16-1** the question | control benign P1Q1 QRY **or** control benign P1Q1 PTH falls by more than 0.0141, and neither rises by more than 0.0141 |
| **K16-2** the width was used | benign held-out query NLL (M3) falls, with its 95% interval below 0, **both** pooled over every benign validation query row **and** on the P1Q1 stratum's rows. Phishing M3 and the training record's per-field validation table are reported |
| **K16-3** band profile | M2: no overshoot of the null, both classes (`PREREG_E14.md` §3.1; old + stream reference) |
| G1 no regression | K12c: no gated cell's verdict worsens with x up by more than 0.0141 |
| G2 paths and subdomains | no PTH, SUB or LBL cell (free and control, both classes) rises by more than 0.0141, whatever its verdict |
| G3 non-query fit | no non-query field's validation NLL (the training record's per-field table, both classes) rises by more than 0.01 nats per character |
| G4 presence | K12a: free PTH and QRY within T, both classes |
| **G5 slash** | **all 8 K12b rows within T. No waiver carries over** (`DECISION_2026-10-06.md` §5) |
| G6 honouring | no control state's P, Q or S honouring falls (amendment 1's rule) |
| G7 copying | K12d: no cell OVER its natural rate (old + stream counts); no copy rate moving further from it (amendment 1's rule) |
| G8 novelty | K12d's whole-URL novelty rule, both modes (old + stream known set) |
| G9 FULL | free FULL up by no more than 0.0141, both classes |
| G10 training and validity | K12e (clipping after step 1,000 at most 1% in every band); V-start′, V-batch, V-mass′, V-stream, V-accum (§8) |

- **Adopted iff K16-1, K16-2, K16-3 and G1–G10 all hold.** There is no retry and no tuning.
- **"Adopted"** makes `e16_cap` a candidate baseline. Replacing `e15_new` is the owner's
  decision.
- **E3 is read for the record.** It is not expected: the far cell (free phishing QRY,
  0.0774) needs about 0.04–0.05 more.

## 6. The end rule

Applied mechanically after validity passes, in this order:

| row | condition | reading | what follows |
|---|---|---|---|
| 1 | K16-1 PASS | **capacity is binding for P1Q1** | the route to E3 continues. If `e16_cap` is adopted and made the baseline, E17 is option A (a second doubling of the stream) on `e16_cap`, for the small strata. If a guard blocks adoption, the reading stands and the owner decides |
| 2 | K16-1 FAIL; K16-2 PASS; **partial response**: P1Q1 QRY falls by at least 0.0082, or P1Q1 PTH by at least 0.0072 (2 × each cell's between-run SD) | **capacity-sensitive, but not enough at 1.77×** | no final report. E17 is the owner's choice: A on `e16_cap` (whether the wider model needs more data), or a further capacity step, which needs R3 approved again |
| 3 | K16-1 FAIL; K16-2 PASS; no partial response | **the wider model fits benign queries better and P1Q1 does not respond.** Every allowed training-side family has now been tested on P1Q1 | **the final E3 report**: E3 NOT MET, with the evidence for every item. The owner then decides whether to freeze the engine or to open a family the rules exclude, with its risk of tuning to the metric |
| 4 | K16-1 FAIL; K16-2 FAIL | the probe did not add usable fit, so it does not answer the question | reported. No automatic follow-up; the owner decides |

## 7. The report

- **Validity** (§8), then **the criteria** (§5), then **the end rule's row** (§6).
- **Fit against the tell.** For each remaining cell:
  - the change in held-out fit (M3 per stratum; the per-field table for paths) beside
    the change in its tell, `e16_cap` against `e15_new`;
  - beside it, E15's same table.
  This shows whether width changes the fit/tell split that E15 found.
- **The small strata, reported:** benign P0Q1 QRY and free phishing QRY. A single change,
  so they are attributable to width. They are not part of K16-1.
- **Copying, novelty, coverage, uptake:** as E15's report, `e16_cap` against `e15_new`.
- **Descriptive, licensing nothing:**
  - the long-tail query length (`e15_posthoc.py` A);
  - the free phishing path-before-query table (C).
- **Cost:** the measured training and generation time, and the peak memory.

## 8. Validity

| code | check |
|---|---|
| V1 | hashes and completeness: the bundle; the stream (`637accab…`); the tables; `e16_cap`'s `final.pt` as training recorded it; every generated file with its registered n and seed |
| V2 | byte identity on the laptop CPU, at least 250 URLs per mode, with `e16_cap`'s decode |
| V3 | options off are unchanged: the full test suite. `accum_steps = 1` and d_model 384 reproduce today's training bit for bit (a fingerprint test) |
| V4 | the scorer reproduces `e12_final`'s published cells exactly |
| **V-start′** | the width-512 start's fingerprints (E10-type base, E11-type start, and the start's own sha256) equal the laptop dry run's, recorded in Status before any training. The notebook stops if they differ |
| V-batch | the old rows' batch digest equals `adbddc2a…` over 20,000 steps |
| **V-mass′** | the appended rows and query positions equal `e15_new`'s **exactly**, per stratum: P0Q1 24,600 rows / 585,977 positions; P1Q1 70,187 / 2,231,497. The injector is deterministic given the same old rows and seeds |
| V-stream | the training record's stream hash and strata equal amendment 3's (as E15) |
| **V-accum** | a unit test on the laptop CPU, float64, dropout off. The summed gradient of two halves equals the whole batch's to 1e-10 (relative). One optimiser step from the same weights agrees to float32 tolerance |
| memory | the worst-case check (a full micro-batch at max_len 384, under autocast) passes on the T4 before training, as every run since E10 |

## 9. Predictions [HYPOTHESIS]

**Why I expect what follows:**
- A 1.77× model on the same data should fit better everywhere, modestly.
  - Scaling estimates put the reducible loss about 4% lower.
  - The query fields are where the reducible part is largest.
- **P1Q1 is the open question.** Its tell has not tracked fit (§2). Width buys mainly fit,
  so I expect a small move at most.
- The small strata learn faster in a larger model, so P0Q1 and the far cell should fall a
  little.
- **Phishing is the risk for fit.** Its rows are seen about 3.8 times each, and a wider
  model memorises more. Its validation fit may not improve (reported, not judged).

| item | `e15_new` | `e16_cap`, predicted (about 80% intervals) |
|---|---|---|
| benign M3, pooled and P1Q1 | — | −0.01 to −0.04 nats per character: **K16-2 PASS (about 90%)** |
| phishing M3 (reported) | — | −0.03 to +0.01 |
| control benign P1Q1 QRY | 0.0500 | 0.038–0.052 |
| control benign P1Q1 PTH | 0.0551 | 0.043–0.057 |
| control benign P0Q1 QRY | 0.0410 | 0.031–0.042 (CLEARED about 35%) |
| free phishing QRY | 0.0774 | 0.052–0.080 (CONVICTED most likely) |
| every other cell | — | within ±0.008. The marginal verdicts may flip either way: P1Q1 SUB (t −2.82), phishing P1Q0 PTH (t −3.06), free phishing PTH (t −2.47) |
| G5 | the "no slash" row: 20 of 71 plans kept their query; at least about 30 are needed | **FAIL about 55%.** The same data, so the same habit, unless the wider model learns the 67 rare training rows better |
| G7 copying | UNDER in every cell | rises toward the natural rate, stays UNDER |
| G10 clipping | 0% after step 1,000 | stays under 1% |

**The end rule, predicted:**
- row 3 (no response) about 50%;
- row 2 (partial) about 25%;
- row 1 (K16-1 PASS) about 20%;
- row 4 (no usable fit) about 5%.

**Adoption:** about 10% (K16-1 and every guard, G5 included). Most likely NOT ADOPTED.

## 10. Failure rules

- **No retry and no tuning.**
  - The learning rate, warmup and clip schedule stay `e15_new`'s even though the model is
    wider.
  - That is a stated risk: the same rate may suit a wider model less well.
- **If the worst-case memory check fails at 2 halves, use 4.** If it fails at 4, stop and
  report. Nothing else changes.
- **If training produces a non-finite loss or validation loss, stop and report.** No
  restart with other settings.
- **If a Kaggle session stops early,** the run is repeated as registered from its start.
  E16 does not resume from a mid-run checkpoint.

## 11. Execution steps and gates

| step | where | what | about | gate |
|---|---|---|---|---|
| 1 | laptop | this registration. Commit and push | done | **gate A: the owner approves it** |
| 2 | laptop | code (§11.1), unit tests (V-accum, off-by-default, the checkpoint layout), the full suite, the dry run (V-start′ fingerprints, a few CPU training steps at width 512 with 2 halves, the replay's V-batch digest), the bundle (no `.gz`), the notebooks `E16_train` and `E16_generate`, `E16_KAGGLE.md`, and the laptop's verify, fit, score and compare scripts (E15's, pointed at `e15_new`). Commit and push; the fingerprints and hashes in Status | 3–4 h | **gate B: the owner approves the measured numbers before any Kaggle run** |
| 3 | Kaggle | `E16_train` (T4) | 3.5–4 h | — |
| 4 | laptop | VERIFY_TRAIN (V1, V-start′, V-batch, V-mass′, V-stream, G10's clipping) and K16-2. Only then step 5 | about 1.5 h | — |
| 5 | Kaggle | `E16_generate` (T4): 50 files, 180,000 URLs | 3–3.5 h | — |
| 6 | laptop | VERIFY_GENERATE (V1, V2), V4, scoring, K16-3, the criteria, the end rule, `E16_RESULTS.md`. Commit and push | 3–4 h | **the owner's decision** |

### 11.1 Code, all off by default

- **`uie/train.py`:**
  - `TrainConfig.accum_steps: int = 1`. Each batch is split into that many consecutive
    parts, and the parts' gradients are summed.
  - At 1 the code path is today's, bit for bit.
  - Written to the checkpoint only when it is not 1, as E15's fields were.
- **`uie/kaggle_train.py`:** `run(..., accum_steps=1)`, passed through.
  - The width is already an argument (`d_model`). d_ff stays 4 × d_model.
- **`research/e16/e16_common.py`:**
  - the arm, the baseline, `recipe_cap(stream)` (= `e15_common.recipe_new(stream)`
    with `d_model = 512, accum_steps = 2`);
  - the start fingerprints (filled from the dry run);
  - E15's decode and jobs.
- **`tests/test_e16.py`:** V-accum; off-by-default; the width-512 start is deterministic;
  the checkpoint layout.

## 12. Unchanged and closed

- **Unchanged:**
  - gate v2, every threshold and tolerance, and E3 (`DECISION_2026-09-29.md` §3);
  - G5 as registered;
  - the data, the stream and its rules, the batches, the steps and the objective;
  - the decode (D1 + D1′) and the generation jobs;
  - `e15_new`'s files, which are not regenerated.
- **Closed and not reopened:**
  - the query weight (N2b);
  - the context arm;
  - D2;
  - the PTH attribution branch (free phishing PTH is read, not diagnosed);
  - boundary weighting (S4/S4b);
  - more steps (R2);
  - decoder-side sampling.
- **R3:** this one step only. A further capacity step needs the owner's approval again.

## 13. Amendment 1 (2026-10-06): gate A, the learning rate, the crawl script

**Registered before any E16 code is committed, and before any training.**

### 13.1 The owner's decision, verbatim

> I have reviewed the summary of PREREG_E16.md. The technical adjustments (gradient
> accumulation and narrowing K16-2 to benign rows) are solid engineering choices.
> However, I want to override one specific point regarding the Learning Rate. You
> mentioned keeping it unchanged to maintain strict isolation. My primary goal here is
> utility—I want the model to actually improve and succeed, not just run a perfectly
> isolated academic comparison.
> If the current learning rate is mathematically optimal for a width-512 model, keep it.
> But if a wider model realistically requires a tuned learning rate (or warmup schedule)
> to train stably and effectively, I authorize you to adjust it now. Tune it for success.
> With that addressed, I am formally giving my Gate A Approval.
> Regarding the parallel acquisition: I explicitly authorize you to prepare the script for
> the Option A crawl.
> Proceed immediately to Step 2 (Gate B). Provide the exact code, laptop dry-run commands,
> and Kaggle bundle instructions so I can launch the training now. We will sequence the
> background crawl after Kaggle is successfully running

**Gate A: APPROVED.**

### 13.2 The learning rate and warmup: kept, as the choice most likely to succeed

The owner authorized a change if a width-512 model needs one. On the evidence it does not:

| question | evidence | reading |
|---|---|---|
| Will it train stably? | In `e15_new`, the early cap (1.0, first 1,000 steps) clipped 26–78% of steps in every band, so it bounds every early update whatever the width. After step 1,000 no step was clipped at 5.0, in any band. At the same rate, a model 4/3 as wide takes updates about 4/3 as large in effect (Adam, standard parameterisation): like `e15_new` at 4e-4 | **stable, with a wide margin** |
| Is a different rate better? | 3e-4 has never been tuned here. Published guidance for a larger model points slightly **lower**, not higher: the μP width rule (×0.75 on hidden weights) and Kaplan et al.'s fit (about ×0.92 for 1.77× the parameters). This run's batch is small (about 14,400 scored tokens), which also argues against a higher rate | each correction is smaller than the uncertainty in the untuned base. Applying one would be a guess, not tuning |
| Would more learning per step help? | R2: twice the steps did not help an E12-family model; its binding constraint appears at about 1.1 epochs. A hotter rate is more learning per row in effect, with more memorisation risk where rows repeat (phishing, about 3.8 passes) | **no evidence it would** |

- **Decision:** lr 3e-4, warmup 500, cosine decay to zero, and the clip schedule, all
  unchanged. §3 and §10 stand.
- **The safety net, registered:**
  - If the wider model does not use its width (K16-2 FAIL, end-rule row 4), the
    learning rate is the first candidate the owner may approve for a rerun.
  - That would be a new decision with its own registration, not a retry inside E16.

### 13.3 The option-A crawl: the script is prepared, not run

- **Authorized:** preparing the script for a second doubling of the benign query stream
  (option A, E17's data).
- **Not authorized here:** running it. The owner sequences it after `E16_train` is
  running.
- **Its rows are held unused** until an E17 registration is approved. Its parameters are
  registered with the script (seeds, rounds, the per-crawl cap, and exclusions that add
  the first stream's domains), before it runs.

### 13.4 Order within step 2

- The laptop's reading scripts (verify, fit, score, compare for `e16_cap`) are written
  while `E16_train` runs, before any of its output exists on the laptop.
- Nothing is read before they are committed. This reorders §11's step 2; it changes no
  criterion.

## 14. Amendment 2 (2026-10-06): V-start′ moves to Kaggle

**Registered after the laptop dry run and before any training.**

**The fact.** The seeded initialisation depends on the torch build.
- Every recorded start (E10, E11, E12) was built on Kaggle's torch (2.10, then 2.11).
- The laptop now runs torch 2.13. There, E12's own start function gives E10's start as
  `32d43dfc…`, not the recorded `26926388…` (`data/e16/dryrun_start.json`).
- So the laptop cannot record the values Kaggle will build. §8's "equal the laptop dry
  run's" cannot be met as written.

**V-start′, redefined.**

| where | check |
|---|---|
| laptop (done) | on the laptop's torch, this rebuild **is** E12's procedure: its width-384 fingerprints equal E12's own `start_fingerprints()`. The width-512 start is deterministic (two builds identical) with 25,473,024 parameters |
| Kaggle, before training (E16_train cell 3) | the width-384 build reproduces `e12_final`'s **recorded** start exactly (init `c7e71edf…` and E12's four fingerprints). That proves Kaggle's torch runs the registered procedure; **otherwise the notebook stops**. Then the width-512 start is built twice, must be identical, and its fingerprints are written to PROVENANCE before training starts |
| laptop, VERIFY_TRAIN | the training record's init sha256 and E12-start fingerprints equal the ones the notebook recorded before training |

**Unchanged:** the start itself (E12's procedure at width 512, seed 20260803), and every
other check in §8.

## Status

    E16  registered 2026-10-06 (this document), after the owner unlocked R3 and agreed
         B1 alone with the refined end rule; before any E16 code, training or
         generation. Awaiting the owner's gate A.
    E16  Amendment 1, 2026-10-06 (section 13): gate A APPROVED by the owner. The
         learning rate and warmup kept on the evidence (section 13.2); the option-A
         crawl script authorized for preparation only; the reading scripts written
         while E16_train runs. Before any E16 code is committed and any training.
    E16  Amendment 2, 2026-10-06 (section 14): V-start' moves to Kaggle. The laptop's
         torch (2.13) initialises differently from Kaggle's (2.10/2.11), so the
         laptop cannot record Kaggle's start; E16_train checks the width-384 rebuild
         against e12_final's recorded start before training and records the
         width-512 start. Laptop dry run PASS: this rebuild is E12's own procedure
         on the same torch; the width-512 start is deterministic with 25,473,024
         parameters; real rows train at width 512 in 2 parts (CPU cost 1.62x).
         Before any training.
    E16  step 2 (bundle and notebooks), 2026-10-06: data/kaggle/uie-e16-code.zip from
         commit eeeb760, sha256 d7379975342d697a3687dc621030c08aeec7c943098fff358a8d8c681f4395bf
         (53.1 MB; 40 code files, 3 tables, e15_new.pt (99a04fef...), qry_extra.jsonl
         = 57,292 rows, 637accab...; nothing gzipped). Full suite PASS: 1,022 tests
         (12 new: V-accum equal to 1e-10 in float64 for 2, 3 and 4 parts, with and
         without weights; 2-part training equal to one pass within float32 tolerance,
         with the same batches and injection; off by default; the start procedure
         width-generic and deterministic). Dry run PASS (section 14). Both notebooks
         run end to end on the laptop CPU from the unzipped bundle (Kaggle's paths
         patched, tiny jobs; the start check bypassed for the simulation only, where
         the laptop's torch makes it fire). Run sheet: research/e16/E16_KAGGLE.md.
         Awaiting the owner's gate B.
    E16  READ 2026-10-07 (research/e16/E16_RESULTS.md; data/e16/compare_report.json).
         Validity PASS (VERIFY_TRAIN; VERIFY_GENERATE 957/957; V4 0.0). K16-1 FAIL
         (P1Q1 QRY 0.0500 -> 0.0400, now CLEARED, fall 0.0100 < 0.0141; PTH -0.0043);
         K16-2 PASS (benign -0.0512, P1Q1 rows -0.0512); K16-3 PASS; G1-G4, G6-G10
         PASS; G5 FAIL (benign control P0Q1 slash 0.99987 vs 0.99551). NOT ADOPTED.
         END RULE ROW 2: capacity-sensitive, not enough at 1.77x; E17 is the owner's
         choice (option A on e16_cap, or a further capacity step). E3 NOT MET: 3
         cells (free phishing QRY, benign P1Q1 PTH and SUB) + 1 slash row.
