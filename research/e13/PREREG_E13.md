# E13 — the levers E13-D licensed, and the owner's decoder override: pre-registration

- **Registered:** 2026-10-02, before any E13 code, fit, training or generation.
- **Decided by the owner** (2026-10-02, answering `E13_ROADMAP.md` §7; quoted in §1):
  - route A, with a decoder intervention added under an explicit amendment of the
    native-only rule, for the diagnosed targets only;
  - model-written context as training input, never as a target;
  - the comparison against `e12_final`'s existing results;
  - the lever's scope: the 8,272 phishing query rows;
  - N2a's two rules; R1 kept; the laptop check of paths and subdomains.
- **Decided by the executor:** the details below, from `E13D_RESULTS.md`, QC1, N2a,
  DC1 and E12.
- **Frozen, unchanged:**
  - gate v2, its thresholds and the detectability standard;
  - every instrument and every published result;
  - `e12_final` as the official baseline (`DECISION_2026-10-01.md`);
  - the library's defaults: every change is behind an option that is off by
    default, and off reproduces E12 bit for bit.

## 0. In plain words

- **E13 tests three levers and runs two diagnostics.**
  - **C, the context arm** (training). Phishing query rows are sometimes shown a
    host and path the baseline model wrote, and trained to write the real query
    after them. Target: free phishing QRY, which DX8 read CONTEXT-LED.
  - **D1, the slash rule** (decoder only). When the supplied plan says "slash" or
    "no slash" for a URL without a path, the host may only end in a way that obeys
    it. Target: the three missed slash rows.
  - **D2, field calibration** (decoder only). DC1's per-field temperature and stop
    bias, fitted on training data, in exactly the two cells DX8 read FIELD-LED:
    benign P0Q1 queries and benign P1Q1 paths.
  - **N, N2a re-run on the current recipe** (diagnostic, Kaggle). Settles whether
    more query practice could fix the query remainder (N2b), or whether capacity
    or representation is the limit (R3's recorded route).
  - **P, QC1's subword questions for paths and subdomains** (diagnostic, laptop).
- **Neither decoder rule changes a single weight.** Each is judged against
  `e12_final`'s own files under the owner's rule: no material regression anywhere,
  or it is rejected.
- **E3 is not reachable in E13, by construction.**
  - Two failing cells have no lever here: benign P1Q1 SUB and QRY, which DX8 read
    SPLIT. The owner's override covers only field-led and slash failures.
  - So no E3 candidate is generated. E14 assembles what E13 adopts, and reads E3.

## 1. The owner's decisions (2026-10-02), verbatim

> 1. Decoder Override: As the project owner, I am officially authorizing a
> decoder-side intervention, but only for the specifically diagnosed
> slash-obedience and field-led sampling failures. The previous "native-only"
> restriction for the decoder is hereby amended for these specific targets. The
> decoder intervention must be proposed and preregistered before implementation,
> must not alter the model's learned parameters, and must be evaluated against the
> existing E12 baseline using the same registered validity, copying, novelty, and
> detector-gap guardrails. It must not introduce any negative regression or
> degradation in other measured behaviors, fields, classes, validity, novelty,
> copying, structural obedience, or any other existing E3/E12 guardrail. Any
> intervention that improves the targeted failures but causes a material
> regression elsewhere must be rejected.
> 2. Route: Route A (Context arm + N2a + the new decoder fix).
> 3. Model-written context: Yes, allow host and path text written by the baseline
> model as training input (not as targets).
> 4. Comparison: Compare against e12_final's existing results.
> 5. Scope: Only the 8,272 phishing query rows.
> 6. N2a's rules: Yes, proceed with these rules.
> 7. R1: Keep R1.
> 8. Laptop check: Yes, run the paths and subdomains check.

**How this registration reads decision 1:**
- **"Specifically diagnosed" targets:**
  - slash obedience is the three missed K12b rows, with DX5's probe;
  - the field-led failures are the two cells DX8 read FIELD-LED: control benign
    P0Q1 QRY and control benign P1Q1 PTH.
  - The SPLIT cells are not targets. Their field part was not singled out by the
    registered reading.
- **"Must not alter the model's learned parameters":** D1 and D2 act on the
  logits at sampling time only.
- **"Material regression":** defined operationally in §3.2, before any data.

## 2. The levers and arms

### 2.1 C — training on the model's own context (`e13_ctx`)

**Recipe:** `e12_final`'s exactly (PREREG_E12 D1–D3, D5):
- the slash plan and R1 (`max_len` 384, kept, decision 7);
- E12's start;
- every setting, including the training seed 20260803 and the 20,000 steps;
- readings at `final.pt`.

**The one change: ContextSwap.**
- **Rows:** phishing name-corpus (domain-frame) training rows with a non-empty
  query, as the tokenizer encodes them. The registered count is 8,272 before the
  length limit; rows that fit 384 tokens are counted by the artifact's builder.
- **When:** each time such a row is drawn, with probability p = 0.5, from a
  generator seeded 20261009 and separate from training's own.
- **What:**
  - one of the row's model-written prefixes is chosen uniformly;
  - the row's tokens before its `<QRY>` are replaced by that prefix, which ends
    with the model's own `<QRY>`;
  - the row's real query and everything after it are kept.
- **What is trained:**
  - the weight of every target inside the model-written prefix is 0, including
    the prediction of its `<QRY>`;
  - targets from the first query character on keep the real row's weights;
  - the model-written tokens are input only (decision 3).
- **The length limit:** a swapped row longer than 384 tokens is used real for that
  draw, and counted.
- **Rows matched by their exact encoded ids.** A training row the artifact does not
  list is never swapped. The page-frame rows carry `<F:page>`, so they never match.

**The prefix artifact** (`data/e13/prefixes.json.gz`, frozen with its sha256 before
the notebook is built):
- **Made by:** `e12_final`'s `final.pt` on the laptop, with the registered raw
  decode (temperature 1.0, the full softmax, nothing else; no D1).
- **Per row, k = 4 samples** under the row's own suffix and plan (P, Q, S as its
  encoding states):
  - the model writes everything before the suffix;
  - the suffix, the plan and `<LBL>` are fed exactly as `generate()` feeds them;
  - the model then writes until it emits `<QRY>`.
- **Invalid samples:** a sample that ends (`<EOS>`, `<FRG>`) or reaches 384 tokens
  first is dropped, never re-drawn. A row keeps 0–4 prefixes, and a row with 0 is
  never swapped.
- **Batching:** rows that share a suffix and plan are sampled together, since the
  prefix depends on nothing else. Groups are sampled in sorted order, from one
  generator seeded 20261006.
- **Recorded:** the counts, the share of valid samples per row, and the prefixes'
  exact-copy rates against the training paths (descriptive).

### 2.2 D1 — the slash rule (decoder only)

**Applies only:**
- where the supplied plan has P = 0 (states P0Q0S0, P0Q0S1, P0Q1S0, P0Q1S1), in
  both modes and both classes;
- at every sampled step from the label marker until the host ends. The host ends
  with the first of `<PTH>`, `<RS>`, `<QRY>`, `<FRG>`, `<EOS>`; after it, D1 is off
  for the row.

**What it does at those steps:**
- **S = 1:** the slash-less ends (`<QRY>`, `<FRG>`, `<EOS>`) get probability 0.
- **S = 0:** the slashed ends (`<RS>`, `<PTH>`) get probability 0.
- **The removed mass goes to the remaining ends,** in proportion to their own
  probabilities, under the decoder's temperature:
  - the probability of ending the host at that step is unchanged;
  - every non-end token keeps its probability.
- **Untouched:** P1 states, where S is implied, and every step outside the host.

**Recorded:** per row, the end mass D1 moved (descriptive).

### 2.3 D2 — field calibration for the two field-led cells (decoder only)

**Mechanism:** DC1's single-pass calibration (`PREREG_DC1.md` §§2–3), with no blend:
- **T_f**, the temperature at every step inside field f;
- **δ_f**, a log-odds bias on every token that ends f (the end token, `<RS>` and
  every field marker);
- applied as `generate()` applies them: bias first, then temperature.

**Two scopes, keyed by class and supplied state** (the same per-state call in free
and control mode):
- **D2-Q:** benign, states P0Q1S0 and P0Q1S1, field QRY;
- **D2-P:** benign, state P1Q1S1, field PTH.

**The fit** (`e13_fit.py`, DC1's §3 objective, adapted to E12's tokenizer):
- **Data:** up to 10,000 real training URLs of the scope's class and state
  (`name_train`, domain frame, encodable within 384), selected with seed 20261007.
- **Objective:** teacher-forced cross-entropy at the steps the sampler samples
  inside f, as its field tracker follows the markers. The forced suffix, plan and
  `<LBL>` are never scored.
- **Search:** L-BFGS-B from (T, δ) = (1, 0), within T ∈ [0.5, 2] and δ ∈ [−3, 3].
  At most 200,000 positions, seeded subsample.
- **Validation confirms once, never tunes.**
  - Up to 5,000 validation URLs of the scope (`name_val`), seed 20261008.
  - The per-URL paired cross-entropy difference (fitted − default) must have a 95%
    CI below 0. Otherwise the scope stays at the defaults, meaning no change.
- **Fitted once,** on `e12_final`, before any D2 generation. The settings file is
  frozen with its sha256.
- **No setting is ever chosen by the gate,** and no setting is chosen from E13's
  generations.

**Not used, anywhere:**
- blend or any second forward pass;
- rejection or resampling;
- masking outside D1's host end;
- calibration of any other field, state or class.

### 2.4 The generated arms

| arm | weights | decode | role |
|---|---|---|---|
| `e13_ctx` | `e13_ctx` `final.pt` | raw | C |
| `e12_d1` | `e12_final` `final.pt` (E12's, unchanged) | raw + D1 | D1 alone |
| `e12_dec` | `e12_final` `final.pt` | raw + D1 + D2 | D2's effect against `e12_d1` |

- **Every arm runs E12's 50 jobs:** free mode 5 × 8,000 per class at
  `seed_free(g)`; control 8 states × 5 at `seed_ctrl`. S is supplied as PREREG_E12
  D2 says.
  - The seeds are `e12_final`'s, so every cell pairs with E12's own files.
  - Those files are the baseline, and they are not regenerated (decision 4).
- **There is no E3 candidate arm** (§0). E14 reads the adopted combination.

### 2.5 N — N2a re-measured on the current recipe (diagnostic)

**N2a's registered design** (`PREREG_N2A.md` with Amendments 1–4) is kept, with
exactly these changes (decision 6, and the cells E12 left failing):

1. **Recipe.**
   - `e12_final`'s tokenizer, model configuration and training settings, from
     E12's start. This includes the fixed loss constant, R1, the plan and the
     slash.
   - The query-bearing training rows of both frames and classes (N2a's rows).
2. **Exposure.**
   - EXPOSURE[c] is `e12_final`'s own draws per row of class c: 20,000 × 256 ×
     w_c / rows_c, from its recorded `class_alpha` and its encoded row counts.
   - A1 is 1×, A2 is 4×. Warm-up is 2.5% of steps, evaluation every 5% (N2a's).
   - The kept rows and the schedule are fixed from a dry run before any arm, and
     recorded in Status. A run on other counts stops.
3. **The cells read.**
   - The failing query cells: control benign P0Q1 QRY and control benign P1Q1 QRY
     (CR1's cells), and free phishing QRY. N2a read the free benign QRY cell,
     which `e12_final` already CLEARS: no deficit is left there to close.
   - Each model writes, per cell, 5 generations from new seed blocks s = 20, 21, 22
     (`20261001 + (100 s + g) × 10^9`), which no earlier experiment used:
     - for the control cells, 3,000 URLs of the state, with S supplied (PREREG_E12
       D2);
     - for free phishing, 1,500 URLs per generation, from E12's free procedure
       restricted to the query-bearing states.
   - **The reference is `e12_final`,** generated with the same procedure and seeds.
4. **Q-VAL and Q-TRAIN.**
   - Each cell's own scorer, at the gate's n, against the cell's validation pool
     (Q-VAL) and against the cell's training strings (Q-TRAIN).
   - Closure, "closes", "little" and the power rule are N2a's.
5. **The copying clause** (decision 6):
   - "copying rises" becomes "the copy rate is OVER the natural rate" (r_g > r_n +
     T, QC1-1's instrument, the cell's own natural rate), in SPARSITY's row and in
     the guard;
   - CAPACITY-OR-REPRESENTATION's "copying has not risen" becomes "the copy rate is
     not OVER the natural rate".
6. **The strong proof** (decision 6). CAPACITY-OR-REPRESENTATION in both benign
   cells, together with QC1's registered ruling-out of query subwords (QC1 §3),
   counts as R3's strong proof (`DECISION_2026-09-29.md` §4).
   - That decision named the plan's FREQUENCY-SPECIFIC or GENERAL, which N2a's
     Amendment 2 merged into this reading.
   - It unlocks a capacity proposal for the owner. It does not run one.
7. **Mechanism metrics M1–M3 are not re-measured.** They were descriptive in N2a.

**Aggregation.**
- Benign decides: a reading holds for the query content only if both benign cells
  read it. Otherwise the reading is MIXED.
- Free phishing QRY is reported, and agreement extends the reading to it (N2a's
  rule).

**A1 and A2 are diagnostic models.** Nothing they write is data, and they are never
candidates.

### 2.6 P — paths and subdomains: QC1's subword questions (laptop)

- **Cells:** control benign P1Q1 PTH, control benign P1Q1 SUB, control phishing
  P1Q0 PTH.
- **Material:** E12's files. `e12_final` is primary and `e12_base` the replication.
  Nothing is trained or generated.
- **P-1, the mix** (QC1-2's procedure and rule, on the field's strings):
  - the model's strings, re-drawn to the real share of exact training strings,
    against the model's strings as written;
  - MIX-LED if the re-drawn x is at most half the plain x; CONTENT-LED if it keeps
    at least 0.8; SPLIT otherwise. D = 12 seeds.
- **P-2, near misses** (QC1-4's procedure and rule):
  - never-seen strings one edit from a training string seen 5 or more times,
    generated against real;
  - BLENDING if generated exceeds real by QC1's registered margin.
- **Licence:** subword tokens for the field (DESIGN.md §4.2) are licensed for E14
  if MIX-LED or BLENDING, replicated in both arms, in any of that field's cells.
  Otherwise they are ruled out by these instruments.
- **Stated limit:** QC1's third route, the ladder's "names IDENTIFIED", is built on
  query parameter names and has no path or subdomain form. It is not run, and the
  ruling-out says so.

## 3. Criteria (fixed now)

### 3.1 C — `e13_ctx` against `e12_final` (a retrained model; the retrain allowance applies)

| code | criterion |
|---|---|
| **K13-1** target | free phishing QRY: x falls by more than 0.0141, and the verdict is not CONVICTED |
| **K13-2** mechanism | DX8 re-measured on `e13_ctx`: x_GV − x_RC at most half of `e12_final`'s 0.0774 |
| **G1** no regression | K12c's rule: no gated cell's verdict worsens with x up by more than 0.0141 (free and control, both classes) |
| **G2** presence | K12a: free PTH and QRY within T, both classes |
| **G3** slash | no K12b row that `e12_final` passed fails |
| **G4** honouring | no control state's P or Q honouring falls by more than its tolerance T (the owner's standard at `e12_final`'s rate and the arm's n) |
| **G5** copying | K12d's memorisation rule (never OVER), and no QC1-1 cell's distance to its natural rate grows by more than T |
| **G6** novelty | K11d's rule against `e12_final` |
| **G7** FULL | auxiliary FULL, free phishing: x up by no more than 0.0141 |

**Adopted iff** K13-1, K13-2 and G1–G7 all hold.

### 3.2 D — the decoder arms against the same weights' files (same checkpoint: DC1's paired rules, no retraining allowance)

D1 is `e12_d1` against `e12_final`; D2 is `e12_dec` against `e12_d1`.

| code | criterion |
|---|---|
| **K13-3** D1 target | all 8 K12b slash rows within T |
| **K13-4** D2 target | at least one of D2's applied cells IMPROVED (Δ < 0, Holm p < 0.05 over D2's applied cells) or FIXED (CLEARED). This is DC1's rule |
| **R-a** cells | REGRESSION if any gated cell (both classes, every computable cell, FULL included) has Δ > 0 at Holm p < 0.05 over all the cells of its mode, or its verdict worsens (DC1's rule) |
| **R-b** presence | K12a holds, and no free-mode PTH or QRY presence rate moves by more than T |
| **R-c** obedience | no state's P or Q honouring falls by more than T; S honouring does not fall in any state |
| **R-d** copying and novelty | G5's rule, and DC1's novelty rule: whole-URL novelty and the query copy share each worsen by less than 0.5 points (95% CI) |
| **R-e** validity | V1–V6 (§5) |
| **R-f** cost | forward passes per delivered URL: 1.0× (no second pass) |

- **D1 is adopted iff K13-3 holds and no R fails** for `e12_d1`.
- **D2 is adopted iff D1 is adopted, K13-4 holds and no R fails** for `e12_dec`
  against `e12_d1`.
  - If the fit applies neither scope, D2 is NOT APPLICABLE, and `e12_dec` is not
    generated. `e12_d1`'s jobs stand for it.
- **The owner's rule, stated as registered:** an intervention that meets its target
  but fails any R is rejected. Nothing is pruned after the fact; a reduced set is a
  new registration.

### 3.3 N and P — diagnostic readings (licenses, no adoption)

- N: §2.5's table and aggregation.
- P: §2.6's licence.
- Each licenses a lever for E14, or rules one out. Neither adopts anything.

### 3.4 Reported, not criteria

- Each lever's effect on every cell.
- Honouring per state, S honouring, and D1's moved mass.
- The prefix artifact's statistics.
- DX8's x_GV and x_RC on `e13_ctx`.
- Copying and novelty profiles.
- K12b's rows for every arm.
- The E3 state of each arm, for the record. No E3 claim is possible in E13 (§0).

## 4. Predictions, before any fit, training or generation [HYPOTHESIS]

- **D1:** K13-3 holds: all 8 slash rows within T. S honouring in P0 states becomes
  100%. No R fails, since D1 touches one decision on P0 URLs.
- **D2:** the fit applies D2-Q; D2-P is uncertain.
  - The applied cells improve by at most about a quarter of their tell, DC1's
    largest effect on E7.
  - Neither clears: benign P0Q1 QRY needs 0.034, and P1Q1 PTH 0.042.
- **C:** K13-1 holds, with free phishing QRY between 0.05 and 0.08, UNDECIDED.
  - K13-2 holds.
  - The phishing slash rows' misses shrink but stay outside T.
  - Every guard holds.
- **N:** benign reads SPARSITY or CAPACITY-OR-REPRESENTATION, not ALLOCATION or
  EXPOSURE. N2a's A2 already left the detector unchanged.
- **P:** no MIX-LED and no BLENDING, as QC1 found for queries. Path and subdomain
  subwords are ruled out.

## 5. Validity

| code | check | on failure |
|---|---|---|
| V1 | hashes and completeness: every checkpoint (`final.pt` as training recorded), the code bundle, the prefix artifact, the D2 settings, the tables (SC1's, the shipped table, the presence-slash table); every file with its registered n and seed | fixed under an amendment; nothing read |
| V2 | byte identity on the laptop CPU: at least 250 URLs per arm and mode, including D1 and D2 | same |
| V3 | options off are unchanged: tests, E12's fingerprints, and `uie/` unchanged since the code commit | same |
| V4 | the scorer reproduces `e12_final`'s published cells exactly from E12's files | same |
| V5 | ContextSwap, from the training log: the matched eligible rows equal the artifact's count; the swap share over eligible draws within ±3 SE of 0.5; zero weight on every model-written target (unit test) | same |
| V6 | D1 and D2, unit tests: (a) mass preservation; (b) only host-end tokens move, only in P0 states, only before the host ends; (c) defaults are bit-identical; (d) the fit's teacher-forced objective equals the sampler's own logits on fixtures | same |
| V-N | N2a's V and K, with `e12_final`'s tokenizer and settings in place of E7's; the kept rows and schedule as fixed in Status | same |

## 6. Order

1. **This registration.**
2. **Code and tests,** every option off by default:
   - D1 in `uie/model.py`; ContextSwap in `uie/train.py` and `uie/kaggle_train.py`;
   - `research/e13/`: the common module, prefix builder, D2 fit, N's training and
     scoring, P, the bundle and notebook builders.
3. **On the laptop:** the prefix artifact, the D2 fit, N's dry run (the counts and
   schedule recorded in Status), and P (read on its own).
4. **Smoke tests** on the CPU, then the bundle and notebooks:
   - `E13_decode`: `e12_d1`, `e12_dec`, and N's `e12_final` reference sets. Needs
     no training, so it can run first.
   - `E13_train`: `e13_ctx`, then N's A1 and A2 with their query sets.
   - `E13_generate`: `e13_ctx`'s 50 files, after the laptop has verified the
     training.
5. **The owner runs the notebooks.**
6. **The laptop:**
   - validity;
   - DX8 on `e13_ctx`;
   - the criteria and the readings;
   - `E13_RESULTS.md`.

## 7. Amendment 1 (2026-10-02): rate comparisons between two arms

**Registered before any E13 arm has generated a URL.** Nothing has been read.

**The flaw.** G4, G5's distance clause, R-b's "moves by more than T" and R-c compare
two generated rates: the arm against its reference.
- T = 2√(p(1−p)/n) is the owner's tolerance for ONE generated rate against a fixed
  real target, as K12a and K12b use it.
- The difference of two such rates has √2 times the spread, so "more than T" fires
  with no true change:
  - 7.9% per row, one-sided;
  - 15.7% per row, two-sided.
- Over the registered rows that is about a coin flip. With 8 honouring states, the
  chance that at least one fires is 0.48; with the 4 presence rows, 0.50.
- So a harmless lever would be rejected half the time, by noise. That is not the
  owner's "material regression".

**The rule, for every comparison of a rate between two arms** (G4, G5's distance
clause, R-b's second clause, R-c):
- a change counts against the arm only if it is **at least T** (the owner's
  standard, at the reference rate and the arm's n), the materiality bar;
- **and** its two-proportion 95% interval excludes 0 in the worsening direction.
  This is DC1's novelty rule, with T as the margin.

**Unchanged:**
- every one-sample rule: K12a, K12b, K12d's "never OVER", G3, K13-3;
- the cell rules: K13-1, G1, G7, R-a, K13-4;
- DC1's novelty rule.

## Status

    E13  registered 2026-10-02 (this document), before any E13 code, fit,
         training or generation
    E13  amendment 1 (section 7), before any E13 arm has generated a URL: rate
         comparisons between two arms need T and a 95% interval excluding 0; the
         one-sample rules are unchanged
    E13  code: the uie options, off by default (9d1bb3a): generate(slash_rule) and
         ContextSwap (TrainConfig.context_*, kaggle_train.run); research/e13: the
         common module, prefix builder, D2 fit, N's training and reader, P, the
         verifier, scorer, comparison, DX8 rerun, smoke tests, the bundle and
         notebook builders; tests/test_e13.py, test_e13_common.py, test_e13_fit.py.
         The full suite passes
    E13  laptop steps, before the notebooks were built:
         - the prefix artifact (section 2.1), data/e13/prefixes.json.gz, sha256
           849fa3dc7505d34d8a0d3dcd480c5f07f361ea442d4ad665f07a9d5655992841:
           - 8,130 rows (142 of the 8,272 over 384 tokens) in 371 (suffix, plan)
             groups;
           - 31,889 of 32,520 samples valid (622 ended, 2 without the supplied
             suffix, 7 at 384 tokens); every row has 1-4 prefixes (7,538 have 4),
             none is unswappable;
           - the cached sampler matches the full pass within 1.3e-5;
           - written paths exact in training: 28.4% (descriptive)
         - the D2 fit (section 2.3) on e12_final, data/e13/d2_settings.json, sha256
           0e7f4b7af92d4daf8ff4f7a8a0f52077d491e0ba59ca7893ecacfacd728371ca:
           - D2-Q: T 1.039, delta +0.079; val dCE/url -0.0200 [-0.0334, -0.0066];
             APPLIED;
           - D2-P: T 1.050, delta -0.065; val dCE/url -0.0564 [-0.0658, -0.0470];
             APPLIED;
           - seeds as run: training selection 20261007 + k, validation 20261008 + k,
             the position subsample 20261007 + 100 (k + 1), with k the scope's
             index (DC1's per-class offset; section 2.3 named the bases only)
         - N's dry run (section 2.5), fixed in e13n_train.py:
           - kept query rows: benign 225,533, phishing 13,673 (over 384 tokens:
             2,599 train, 402 validation);
           - exposure per row 1.6550 / 3.7604 (E7's: 1.6589 / 3.7684);
           - A1 1,659 steps (warm-up 41, evaluation every 83); A2 6,636 (166, 332)
         - smoke tests, all PASS:
           - training's tokenizer, rebuilt from the full bundle, is e12_final's
             exactly, and all 8,130 artifact rows match training rows;
           - train() with ContextSwap on the artifact: 399 of 821 eligible draws
             swapped;
           - both decoder arms generate in both modes with the fitted settings; D1
             obeys S in every tested state (40/40, three states) and stops at the
             host end (60/60 host/?query);
           - N's path from the data to its query sets
         - P (section 2.6) runs on the laptop while the owner runs Kaggle, and is
           read on its own
    E13  bundle uie-e13-code, sha256
         15c13092fb8be91c299d9c92a98a11d0d839d6c89801cbdbfa2cf1325cdd253a (commit
         887b928, 53.0 MB): 39 code files, the 5 registered tables and artifacts, and
         e12_final.pt (968ef622...). Checked unpacked, alone: every manifest hash, and
         the imports with no repository on the path. Notebooks E13_decode, E13_train,
         E13_generate; run sheet E13_KAGGLE.md
    E13  packaging fix, before anything ran on Kaggle: both first notebook runs
         stopped at the integrity check, because Kaggle decompresses .gz files
         inside an uploaded dataset (prefixes.json.gz arrived as prefixes.json).
         The bundle now carries the artifact's exact decompressed bytes,
         prefixes.json, sha256
         3de7e8cbd82a59688a1b8cc7e8677b97b124a9f481e5927ff177a7140fb94c73 (the
         registered prefixes.json.gz, decompressed). ContextSwap.load reads the
         artifact gzipped or plain (tested: the two load identically). Nothing
         else changes. The new bundle replaces 15c13092...
    E13  bundle uie-e13-code (replacement), sha256
         0cd7c5d3f5b32c2937c21b75492734cd2c6f8fa1f406381aa1b48965c8957249 (commit
         a1ac866, 53.1 MB): no .gz inside; unpacked alone, every manifest hash
         matches and its own code loads its own artifact (8,130 rows)
    E13  P READ 2026-10-02 (data/e13p/tables.md; e13p_run.py; 7 jobs, no void draw):
         - P-1: CONTENT-LED in all three cells, replicated in both arms. Re-drawn to
           the real share of exact training strings, the path tells grow (rho 1.24 /
           1.42 benign P1Q1 PTH, 1.96 / 2.28 phishing P1Q0 PTH); benign P1Q1 SUB
           0.88 / 0.87, not significant
         - P-2: NO BLENDING in every cell and arm (excess +0.06 to +1.41 points,
           under the 5-point margin)
         - licence: path and subdomain subwords are ruled out by these instruments
           (the ladder's names route is not run, as registered). Prediction held
    E13  trained 2026-10-02 (results (21)): VERIFY_TRAIN PASS
         - e13_ctx: final.pt as training recorded; code a1ac866, unchanged since; it
           started from e12_final's exact weights (c7e71edf...) with e12_final's
           settings and tokenizer; ContextSwap matched 8,130 of 8,130 rows, and
           swapped 15,373 + 29 too long of 30,665 eligible draws: 0.5023 (band
           0.5 +- 0.0086). 78.2 minutes; no step clipped after step 1,000. Final
           validation 1.5656 / 1.4082 (e12_final 1.5621 / 1.4006), reported only
         - N: A1 1,659 steps and A2 6,636 as registered, kept rows as registered,
           finite; their query sets as recorded
         - E13_decode returned (results (22)); its verification runs next
    E13  decode VERIFIED 2026-10-02: VERIFY_DECODE PASS -- the 115 files as recorded
         (e12_final's weights, the registered tables and D2 settings, code a1ac866);
         byte identity on the laptop CPU, D1 and D2 included: e12_d1 and e12_dec 1,184
         of 1,184 URLs each, both modes, N's reference 272 of 272. Scoring runs
         (e13_score_run.py)
    E13  generation VERIFIED 2026-10-02 (results (23)): VERIFY_GENERATE PASS -- e13_ctx's
         50 files from its verified final.pt; byte identity 1,162 of 1,162 URLs on
         the laptop CPU. V4 PASS: e12_final's free cells re-scored from E12's files
         equal E12's published record (max |difference| 0.0)
    E13  READ 2026-10-02 (E13_RESULTS.md): validity PASS. Nothing adopted; E3 not
         met (as section 0 said).
         - C: K13-1 FAIL (free phishing QRY 0.1141 -> 0.1004, fall 0.0137 < 0.0141,
           CONVICTED); K13-2 PASS narrowly (gap 0.0377), mostly through x_RC rising;
           G6 FAIL on DC1's query-copying clause
         - D1: K13-3 FAIL, 7 of 8 rows; phishing free bare hosts 99.919%, its 19
           misses all from P1 plans (paths dropped), outside D1's scope; every R
           passes
         - D2: no measurable effect (delta +-0.0018); R-a (verdict flips at the
           boundary) and R-d FAIL
         - N: UNAVAILABLE (power rule) -> nothing licensed; phishing EXPOSURE
           reported. Measured, A2 cut free phishing QRY to 0.055 and benign P0Q1 QRY
           to 0.028
         - predictions: 5 held, 4 did not
