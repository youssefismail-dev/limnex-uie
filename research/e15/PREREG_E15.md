# E15 — does more distinct query data help the transformer? A matched-exposure test, and the new-data experiment it licenses: pre-registration

> **AMENDED 2026-10-05: read §9 (Amendment 1) first.**
> - By the owner's decision, **E15-A is removed** and will not run.
> - **E15-B is the experiment,** re-specified in full in §9.
> - §0–§8 are kept below unedited, as first registered. Wherever they differ
>   from §9, §9 governs.
> - **AMENDED AGAIN, 2026-10-05: §10 (Amendment 2)** revokes matched exposure
>   and replaces §9.6's replacement rule with injection. Where §10 and §9
>   differ, §10 governs.
> - **Gate A approved; §11** records the owner's directives and the rules they
>   change.
> - **§12 (Amendment 3)** records the measured numbers for gate B.
> - **§13 (Gate B)** records the approval, the instrument definitions and the
>   formal predictions.

- **Registered:** 2026-10-05, before any E15 code, fit, training, generation or data
  acquisition. Nothing runs before the owner's gate 0 (§7).
- **Decided by the owner** (`research/native/DECISION_2026-10-05.md` §1):
  - the slash fix (D1 + D1′) is kept and is the decode from E15 onward;
  - the query weight is closed;
  - the next direction is query data from a genuinely new, separately auditable
    source, evaluated critically first; query data only;
  - comparisons are against frozen baselines;
  - no capacity, decoder, broad-corpus or open-ended changes.
- **Decided by the executor** (`E15_DESIGN.md`):
  - E15-A, a measured test of whether the transformer is data-limited on query
    content, comes first;
  - E15-B, the new-data experiment, runs only if E15-A licenses it.
- **Frozen, unchanged:**
  - gate v2, its thresholds, the detectability standard and E3;
  - every published result;
  - `e12_final` (weights) and `e12_d1p` (the adopted decode) as the baselines;
  - the library's defaults: every change is behind an option that is off by default,
    and off reproduces E12 bit for bit.

## 0. In plain words

- **E15-A (one Kaggle session).** The model is retrained exactly as `e12_final` was,
  twice.
  - Each time, it learns query content from only **half** of the query-bearing rows
    (a different half each time), at double weight. So the total query practice is
    unchanged.
  - Same start, same batches, same everything else.
  - If the query cells get clearly worse, the model is short of data, and doubling
    the data with a new source could help. The size of the damage bounds the
    possible gain.
  - If they do not, more data cannot help, and the data direction closes on the
    model's own evidence.
- **E15-B (only if E15-A licenses it).** New real URLs with queries, from **the same
  population** the gate tests against, are added as a separate, labelled stream.
  - **Same population** means more Common Crawl blocks from the same 16 crawls,
    built the same way, with new domains only.
  - **The model learns only their queries.** Their hosts, paths and plans are input,
    never targets.
  - **Query practice is held at today's level.** So the gain, if any, can only come
    from the new content, and the path and subdomain cells cannot be crowded the way
    E14 crowded them.
  - Decoded with the adopted slash fix, and judged against `e12_d1p`.
- **E3 is not reachable in E15.** Query data cannot reach the path and subdomain
  cells. A same-population phishing source of useful size does not exist
  (`E15_DESIGN.md` §1.3).

## 1. The owner's decisions

Verbatim in `DECISION_2026-10-05.md` §1. These bind E15:
- **"Keep the intervention narrow: QRY data only, while preserving the D1 slash
  fix."**
  - E15-A changes only query-content weights.
  - E15-B adds only query targets.
  - Every E15-B arm decodes with D1 + D1′.
- **"Do NOT simply duplicate, oversample, or reshape the existing QRY rows."**
  - E15-B's intervention adds new rows only.
  - E15-A reweights existing query content to *remove* half of it at fixed exposure.
    It is a diagnostic, never a candidate, and it needs the owner's explicit
    approval at gate 0 (`E15_DESIGN.md` §3).
- **"The new source must remain separately identifiable and auditable."** §4.3.
- **"Preserve the existing E12/E14 baselines and make all comparisons against the
  appropriate frozen baseline."**
  - E15-A: against E13's frozen `e12_final` reference sets (raw decode, N's
    instrument).
  - E15-B: against `e12_d1p` (E14's files).

## 2. E15-A — the transformer's query data curve at matched exposure

### 2.1 The arms: `e15_h1`, `e15_h2`

**The recipe is `e12_final`'s exactly** (`e12_common.recipe("e12_final")`):
- start c7e71edf…, seed 20260803;
- 20,000 steps, batch 256;
- the fixed objective with C = 14,397.885481960007 pinned;
- clip 1.0, then 5.0 from step 1,000;
- R1, the presence plan and the slash plan.

**The one change: which rows teach query content.**
- **Query-content positions,** as defined in `PREREG_E14.md` §2.1: every step taken
  while the query is open, from its first character through the step that predicts
  its end, but not the `<QRY>` step.
- **Rows in half K** get weight w_K,c on those positions.
- **Rows in half R** get weight 0 on those positions.
- **Every other position of every row** keeps `e12_final`'s weight, so plans,
  markers, hosts, paths and fragments train exactly as before.

**The halves:**
- A row's key is the corpus's own split key: its **registrable domain** (benign,
  `split_random` "domain") or its **host** (phishing, `split_phishing_host` "host").
  So all of one site's rows fall in the same half.
- A row is in K for `e15_h1` iff the first byte of sha256("E15-A:" + key) is even.
  For `e15_h2`, K is the complement.
- Both classes and both frames are split.

**The weight.**
- w_K,c = M_c / M_K,c for each class c.
  - M_c is the class's total count of query-content positions over its training
    rows, as encoded.
  - M_K,c is the same count over the class's half-K rows.
- So each class's expected query mass per draw equals `e12_final`'s exactly.
- The dry run fixes the four numbers (two arms × two classes, about 2.0) in Status
  before the bundle is built.

**Batches and validation.**
- **The batches are `e12_final`'s.** Masks do not enter the sampler, and V-batch
  proves it.
- **Validation rows are unweighted.**

### 2.2 Generation

- **N's per-cell query sets** (`e13_common.n_urls`, raw decode), for N's three cells:
  - control benign P0Q1 QRY;
  - control benign P1Q1 QRY;
  - free phishing QRY.
- **Five generations per cell,** seed blocks 20, 21 and 22 (`PREREG_E13.md` §2.5).
  That is 15 files per arm.
- **The reference** is E13's `n_e12_final_*` files (`data/e13/decode/gen`), frozen
  and not regenerated.

### 2.3 Measurement

- **The decision quantity: N's Q-VAL.** Each cell's own scorer
  (`n2a_qry.qry_cell`), at the gate's n, against the cell's real validation pool.
  This is how E13 measured `e12_final` (`data/e13/n_measure.json`: 0.1141, 0.0583,
  0.0519).
- **Reported, not decisive:**
  - Q-TRAIN;
  - held-out query NLL per class (`e14_fit.py` M3, per row, paired with
    `e12_final`), and its train–validation gap;
  - the copy rate against the natural rate;
  - every field's validation NLL, from the training records. This is the design
    check that only query content moved.

## 3. E15-A's reading, fixed now

**For each of the three cells c:**
- rise_c = ½ (x_h1 + x_h2) − x_ref, where x_ref is `e12_final`'s Q-VAL
  re-measured under V-ref.
- **DATA-LIMITED:** rise_c > 0.0141 (the retraining allowance), **and** x_h1 > x_ref
  **and** x_h2 > x_ref. Both halves must agree.
- **NOT DATA-LIMITED:** rise_c ≤ 0.0141.
- **UNRESOLVED:** anything else. It licenses nothing, and is reported.
- **VOID:** a void measurement (N2a's rule) in either arm or the reference.

**What each reading licenses:**

| reading | licenses |
|---|---|
| DATA-LIMITED in benign P0Q1 QRY or P1Q1 QRY | E15-B for that cell's stratum, through gates B0–B2 (§4). The gain from doubling is projected to be **at most** rise_c, since learning curves flatten |
| DATA-LIMITED in free phishing QRY | E15-B for phishing **only if** a same-population source passes B2 with at least one doubling: 8,272 domain-frame query rows from hosts new to the corpus. Expected infeasible: about 1.5 years of the feed (`E15_DESIGN.md` §1.3). Reported |
| no cell DATA-LIMITED | **The new-query-data direction closes**, on a registered transformer-level reading. DX3's stand-in is superseded. The data candidate counts as ruled out for R3's first case (`DECISION_2026-09-29.md` §4). R3 stays the owner's decision |

**Power** (from DX4's between-run SDs, which include recipe differences and so are
conservative):
- SD of the mean of two halves minus one reference = 1.22 × the between-run SD:
  0.0099 (phishing), 0.0028 (P0Q1), 0.0050 (P1Q1).
- **False DATA-LIMITED with no true effect,** before the both-halves condition:
  7.8%, below 0.1%, and 0.25%.
- **Chance of detecting a true rise of 0.020:** 72%, 98% and 88%.
- The phishing cell is the noisiest, and also the one no same-population source
  could act on.

## 4. E15-B — new same-population query data at matched exposure (conditional)

**Designed now, so the owner sees the whole data plan.** It runs only through gates
B0–B2.
- **Amendment 2:** its source-specific numbers are fixed after E15-A and before any
  data is acquired. They are the seed, block counts and target volume.
- **Amendment 3:** its stream's numbers are fixed after B2 and before any training.
  They are the hash, counts, weights and draws per batch.

### 4.1 B0 — the licence

E15-A reads DATA-LIMITED in at least one benign cell. Only those cells' strata
(class, frame, state) receive new rows.

### 4.2 B1 — data-source requirements

**Benign:**
- Common Crawl, **the same 16 crawls**:
  - live: CC-MAIN-2024-42, 2024-51, 2025-13, 2025-26, 2025-38, 2025-51, 2026-12,
    2026-30;
  - historical: CC-MAIN-2019-04, 2019-30, 2019-51, 2020-24, 2020-50, 2021-25,
    2021-49, 2022-49.
- **The same builder:** `uie/corpus/build.py`, `commoncrawl.py`, with the same
  filters.
- **A new registered seed,** and more index blocks per crawl. The number is fixed
  by amendment 2.
- **Rows are kept only if all of these hold:**
  - the registrable domain is absent from every split of the existing corpus;
  - the URL follows the name-corpus rule (one URL per registrable domain, domain
    frame);
  - the URL is in a licensed state (P0Q1 or P1Q1).
- **Stratified** to match the existing licensed stratum's crawl and era mix.
- **Volume:** one doubling of the licensed stratum's training rows (P1Q1 42,862;
  P0Q1 14,906), the quantity E15-A's halving mirrors.

**Phishing:** only as in §3's table. The source is the OpenPhish public feed after
commit 48f7a98 (2026-08-03), with the same parser and hosts new to the corpus.

**Recorded per row:** crawl or commit, block or fetch time, the builder's commit,
and the filter outcomes.

### 4.3 Leakage, contamination and auditability

- **Leakage, hard checks, which must be zero:**
  - registrable-domain (benign) or host (phishing) overlap with the existing
    train, validation and test splits (the split files' keys);
  - exact-URL overlap with validation and test (exact, plus the bundle's
    known-URL bloom).
  - The test split is never read.
- **Why domain disjointness is the guarantee.**
  - The existing build's block identities were not recorded (reused shards), so
    block-level disjointness cannot be shown.
  - Domain disjointness is the split's own key, and it is what keeps the gate's
    validation pools untouched.
- **Contamination:** no new benign row whose host is in the phishing corpus or the
  feed; the same benign filters as the existing build.
- **Auditability:**
  - the new rows live in a separate stream, `qry_extra_<class>_train.jsonl` (not
    gzipped), with its own manifest: sources, seed, blocks, filters, counts and
    sha256;
  - they carry their own corpus tag in training;
  - the training record reports the stream's rows, draws, query mass and loss
    separately;
  - the existing corpus files are unchanged and verified by hash.

### 4.4 B2 — population gates, before any training

All must pass, or the source is rejected and nothing trains.

| gate | test |
|---|---|
| **P1 no shift** | DX1's test (`PREREG_E13D.md`), with the new rows in place of the training twin: their real query strings against the cell's real validation pool, at DX1's harness and n, must read NO SHIFT under DX1's registered rule |
| **P2 new coverage** | the new rows raise the coverage of the cell's validation query templates by at least DX3's measured per-doubling gain less 1 point: P1Q1 ≥ 3.3 points, P0Q1 ≥ 2.2. The data must carry the held-out population's patterns, not merely more of the same or another population's |
| **P3 leakage** | §4.3's hard checks: all zero |
| **P4 mix** (reported) | suffix, era and crawl distributions against the existing stratum's |
| **P5 audit** (reported) | exact query-string overlap with the gate's validation queries, against the natural rate; the share of new templates |

### 4.5 The arm: `e15_new`

- **`e12_final`'s recipe and seed.** The existing corpus's batches are drawn
  exactly as in `e12_final` (V-batch).
- **The side stream.**
  - The new rows come from a separate stream with its own seeded generator: k_s
    extra rows per batch for each licensed stratum s.
  - k_s = (the expected old rows of s per batch) × (N_new,s / N_old,s).
  - So, per distinct query row, old and new are drawn equally often, as if the
    stratum's data had been larger.
- **Narrow targets.** On new rows, **only query-content positions are targets.**
  Plan, markers, host, path and fragment are input only, at weight 0. The new source
  cannot shift presence, hosts or paths.
- **Matched exposure, per stratum.** The query-content weight of every row in s, old
  and new, is N_old,s / (N_old,s + N_new,s). So the stratum's query mass per step
  equals `e12_final`'s.
  - Strata without new rows are unchanged.
  - Non-query mass is identical by construction.
- **Cost:** a few extra rows per batch, about 1–2% of compute, for their query
  targets only. Steps and parameters are unchanged.
- **Decode:** raw + D1 + D1′ (the adopted slash fix), on E12's 50 jobs at E12's
  seeds.

### 4.6 Criteria: `e15_new` against `e12_d1p` (a retrained model; the retrain allowance applies)

| code | criterion |
|---|---|
| **K15-1** target | in each licensed cell, x falls by more than 0.0141 |
| **K15-2** fit | held-out query NLL (M3) falls in the licensed class, with a paired 95% interval below 0, against `e12_final` |
| **K15-3** band profile | M2: no overshoot of the null |
| **G1** no regression | K12c: no gated cell's verdict worsens with x up by more than 0.0141 |
| **G2** paths, subdomains, labels | no PTH, SUB or LBL cell (free and control, both classes) rises by more than 0.0141, **whatever its verdict**. This is the owner's E14 concern made explicit |
| **G3** field fit (**new**) | no non-query field's validation NLL rises by more than 0.01 nats per character. That is 5× the benign noise reference (`e13_ctx`: 0.002); E14's breaches were 0.027–0.055 |
| **G4** presence | K12a; amendment 1's moves reported |
| **G5** slash | all 8 K12b rows within T under D1′, D1′'s prospective re-test on new weights |
| **G6** honouring | amendment 1's rule, P, Q and S |
| **G7** copying | K12d (never OVER the natural rate) and amendment 1's distance rule |
| **G8** novelty | K12d's rule |
| **G9** FULL | free FULL up by no more than 0.0141, both classes |
| **G10** training | K12e (clipping); V-batch on the existing rows; matched query mass (V-mass) |

- **Adopted iff** K15-1, K15-2, K15-3 and G1–G10 all hold.
- **Adopted means** the stream and the arm are kept. Making `e15_new` the official
  baseline is the owner's decision.
- **E3 is read for the record** (it cannot be met; §0).

## 5. Predictions, before any code, training or data [HYPOTHESIS]

- **E15-A:**
  - free phishing QRY: **NOT DATA-LIMITED** (moderate confidence: the cell is small
    and noisy);
  - benign P0Q1 QRY: **NOT DATA-LIMITED** (high confidence: exposure alone cleared
    it);
  - benign P1Q1 QRY: **NOT DATA-LIMITED**, but the closest, with rise_c between
    +0.005 and +0.014;
  - **so the most likely outcome is that the data direction closes;**
  - held-out query NLL rises slightly under halving, and its train–validation gap
    widens;
  - every other field's validation NLL moves by less than 0.01 nats per character.
- **E15-B, if it is licensed:**
  - P1 and P3 pass;
  - K15-1 holds only in the licensed cell, with a gain no larger than its E15-A
    rise;
  - G2 and G3 hold, because exposure is matched;
  - E3 is not met.

## 6. Validity (E15-A; E15-B's are fixed with amendment 3)

| code | check | on failure |
|---|---|---|
| V1 | hashes and completeness: the bundle; `e12_final.pt`; each arm's `final.pt` as training recorded; the tables; every file with its registered n and seed | fixed under an amendment; nothing read |
| V2 | byte identity on the laptop CPU, at least 250 URLs per arm across N's cells | same |
| V3 | options off are unchanged: the full suite; E12's fingerprints; with the split option off, masks equal `field_mask`'s bit for bit | same |
| V-start | the start is E12's (`c7e71edf…`, the E10 and E11 fingerprints) | same |
| V-batch | each arm records a digest of every batch's ids; `e15_h1`'s = `e15_h2`'s = the laptop's replay of `e12_final`'s sampler | same |
| V-mask | dry run and unit tests: K rows' query positions carry w_K,c, R rows' carry 0, every other position carries `e12_final`'s weight; each arm's per-class query mass within 0.1% of `e12_final`'s | same |
| V-ref | `e12_final`'s Q-VAL, re-measured from E13's files, equals `n_measure.json` exactly | same |

## 7. Execution steps and the owner's decision gates

| step | who | what | about |
|---|---|---|---|
| **gate 0** | owner | approve: (a) the verdict in `E15_DESIGN.md`, new query data not yet justified; (b) E15-A as the next step, including its half-content reweighting as a diagnostic only; (c) E15-B's design as the conditional continuation. Or direct otherwise, e.g. skip to the path and subdomain cells, or go straight to sourcing (not recommended, `E15_DESIGN.md` §4) | — |
| 1 | laptop | **Code and tests,** every option off by default. In `uie/`: the half-split query masks and the batch digest. In `research/e15/`: the common module, the dry run, the sampler replay, the N measurement and E15-A's reading, the bundle and notebook builders. Full suite; commit and push, verified on the remote | a session |
| 2 | laptop | **Pre-flight:** the dry run (the four w_K,c, K shares, V-mask) and the replay digest of `e12_final`'s sampler (V-batch's reference), both recorded in Status. CPU smoke tests; the bundle (no `.gz`); notebooks `E15A_h1` and `E15A_h2`; `E15_KAGGLE.md`. Commit and push | 1–2 h |
| **gate 1** | owner | review Step 2's numbers in Status; approve the Kaggle runs | — |
| 3 | Kaggle | **`E15A_h1` and `E15A_h2`, in parallel,** each training (about 80 min) plus its 15 query-set files. Inputs: `uie-corpus2`, `uie-e15-code`. GPU T4 | about 1.8 h each |
| 4 | laptop | **VERIFY** (V1, V2, V-start, V-batch, V-mask, V-ref), the measurements and the reading (§3); `E15A_RESULTS.md`. Commit and push | 2–3 h |
| **gate 2** | owner | read E15-A. **If DATA-LIMITED in a benign cell:** approve amendment 2 (E15-B's source parameters). **Otherwise:** the data direction is closed, and the owner decides what follows: the path and subdomain cells, or the R3 question | — |
| 5 | laptop | *(only after gate 2 licenses it)* amendment 2 → B1 the build → B2 the gates → amendment 3 (the stream's hash, counts, weights, k_s) | hours to a day |
| **gate 3** | owner | approve the source (B2's results) and amendment 3 | — |
| 6 | laptop | the side-stream code, tests, smoke tests, bundle, notebooks | a session |
| **gate 4** | owner | approve E15-B's Kaggle runs | — |
| 7 | Kaggle | `e15_new`: training, then 50 files with D1 + D1′ | about 1.5 h + 2–3 h |
| 8 | laptop | verify, score, criteria (§4.6), E3 for the record; `E15B_RESULTS.md` | about 5–6 h |
| **gate 5** | owner | adoption, and whether `e15_new` becomes the baseline | — |

## 8. What E15 does not do

| standing rule | E15 |
|---|---|
| no gate, threshold or tolerance changes | none. The decision margin is the existing retraining allowance; B2 reuses DX1's and DX3's rules. G3 is a new guard, declared as new (§4.6) |
| no capacity, extra steps or decoder changes merely to improve a result | none. The decoder is the adopted slash fix; E15-B's side stream adds about 1–2% of rows, for query targets only, at matched query mass |
| no duplication, oversampling or reshaping as the intervention | E15-B adds genuinely new rows. E15-A's reweighting is a diagnostic that removes data, approved at gate 0 |
| no optimising against the metric | one split, two complementary halves, every number fixed before training; no sweep, no retry |
| query data only | E15-A touches query-content weights only; E15-B's new rows teach query content only |
| the closed branches stay closed | the query weight (N2b), the context arm, D2 and the PTH attribution branch are not reopened |
| R2 rejected, R3 locked | unchanged. E15-A can supply one of the readings R3's first case asks for. It unlocks nothing by itself |
| the repository stays private | unchanged |

## 9. Amendment 1 (2026-10-05): E15-A removed by the owner; E15-B is the experiment

**Registered before any E15 code, data acquisition, training or generation.** It
supersedes §0–§8 wherever they differ. Those sections stay above, unedited, as the
record of what was first registered.

### 9.1 The owner's decision, verbatim

> I reviewed the E15 preregistration, and I want to change the execution path.
>
> Do not run E15-A. Close/remove E15-A as a prerequisite. I do not want the
> matched-exposure half-data diagnostic.
>
> Proceed directly to the E15-B new-data experiment, because the objective now is to
> directly test my original hypothesis: whether genuinely new and diverse QRY
> training data improves the remaining QRY failures without introducing the PTH/SUB
> trade-off caused by increasing QRY loss weight.
>
> Please update the preregistration openly before any code, data acquisition, or
> training. Do not silently modify the existing registration.
>
> Required E15-B design
>
> 1. Keep D1 + D1′
>    - Preserve the slash fix in the decoder from E15 onward.
>    - Do not reopen the E14 QRY-weight intervention.
>    - Do not use QRY loss weighting.
>    - Keep the model recipe, capacity, steps, optimizer, and decoder otherwise
>      unchanged.
> 2. New QRY data only
>    - Add a genuinely new, separately auditable data source containing real URLs
>      with queries.
>    - Do not duplicate, oversample, or reshape existing QRY rows.
>    - The new source must contribute genuinely new query strings/patterns and
>      measurable coverage/diversity.
>    - Keep the new stream separately identifiable with its own manifest, source
>      metadata, hash, counts, and corpus tag.
> 3. Population matching
>    - For benign data, use the same Common Crawl population/crawls and the same
>      corpus-building rules where possible.
>    - Require new registrable domains that are absent from the existing
>      train/validation/test split keys.
>    - Preserve the existing population characteristics and stratification.
>    - For phishing data, only use a source if it can satisfy the same-population
>      requirement and the registered leakage/contamination checks.
>    - If a same-population phishing source of sufficient quality/size is genuinely
>      unavailable, do NOT invent a substitute. Report that limitation and proceed
>      only with the licensed benign stratum if scientifically justified.
> 4. Query-only supervision
>    - The new rows may provide training targets only for QRY content.
>    - Host, path, subdomain, markers, plans, and other non-query fields must remain
>      inputs/context, not new supervised targets.
>    - The intervention must not intentionally increase training mass for PTH/SUB or
>      other fields.
> 5. Matched exposure
>    - Preserve the current overall QRY training exposure as much as possible.
>    - The purpose is to test the effect of new QRY content/coverage, not simply to
>      give the model more QRY gradient mass.
>    - Define the exact old/new sampling ratio and QRY loss weighting before
>      training and freeze them.
>    - No hyperparameter sweep, no retries, and no post-hoc tuning.
> 6. Pre-training gates
>    Before any training, verify and preregister:
>    - zero train/validation/test domain/host leakage;
>    - zero exact-URL validation/test contamination;
>    - source population compatibility;
>    - new-template coverage gain;
>    - source diversity;
>    - exact row counts;
>    - exact new/old sampling ratio;
>    - exact QRY target mass;
>    - hashes and reproducibility metadata.
> 7. Evaluation
>    Compare against the frozen "e12_d1p" baseline.
>    Keep all existing gate-v2 thresholds, tolerances, and E3 definitions unchanged.
>    The candidate must show:
>    - improvement in the licensed QRY cells;
>    - held-out QRY NLL improvement;
>    - no material PTH/SUB regression;
>    - no material non-query field NLL regression;
>    - slash obedience remains within the D1′ guard;
>    - presence/honoring/copying/novelty remain within their existing guards;
>    - FULL behavior remains within the existing allowance.
> 8. Important
>    Do not treat the fact that E15-A was skipped as evidence that the data
>    hypothesis is already proven.
>    E15-B itself is the direct test of the new-data hypothesis.
>    Also explicitly distinguish in the final report between:
>    - evidence that new QRY content helped;
>    - evidence that the model merely benefited from additional QRY exposure;
>    - evidence of improved coverage/diversity;
>    - and any PTH/SUB or slash trade-off.
>
> Execution boundary
> Do not start coding yet. First:
> 1. Amend the E15 preregistration to remove E15-A and make E15-B the primary/first
>    experiment.
> 2. Produce the exact revised E15-B design, source requirements, population gates,
>    leakage checks, sampling/exposure calculation, acceptance criteria, and
>    execution steps.
> 3. Commit and push the revised preregistration.
> 4. Stop there and wait for my review.
>
> I will approve the implementation/data-acquisition step separately after
> reviewing the revised registration.
>
> No Kaggle run, training, generation, or new-data acquisition before my approval.

### 9.2 What changes, stated openly

| original | amended |
|---|---|
| **E15-A** (§2, §3; its parts of §5, §6, §7) | **Removed, by the owner's decision. It will not run.** Its question stays unanswered by any registered reading: is the transformer data-limited at matched exposure? |
| E15-B conditional on E15-A (§4.1) | E15-B is **the experiment**. A stratum is admitted by the pre-training gates (§9.5) instead of by E15-A's licence |
| §4.5: matched exposure by down-weighting query content to N_old/(N_old+N_new) | **No QRY loss weighting** (decision 1). Matched exposure is achieved by sampling (§9.6): every supervised query position keeps weight 1.0 |
| §4.2, and `E15_DESIGN.md` §1.3: the benign corpus is "16 crawls, 200 index blocks each" | **Corrected, a factual error in the original text.** That describes E0 (`data/corpus/benign.manifest.json`), an earlier corpus. The training corpus is **corpus2**, the source of bundle2's name and structure corpora (§9.4) |
| `E15_DESIGN.md` §3: "the stronger next step is to measure" | the recommendation was not adopted. The evidence and verdict in `E15_DESIGN.md` stand as the executor's assessment. **Skipping E15-A is not evidence for the data hypothesis** (decision 8): E15-B is its direct test |
| E15-A as the data reading for R3's first case | lapses. E15-B's own reading is a registered, transformer-level test of new data, and §9.11 says what it can and cannot support |

**Unchanged:**
- gate v2, its thresholds, the tolerances and E3;
- `e12_d1p` and `e12_final` as frozen baselines;
- the library defaults (off reproduces E12 bit for bit);
- every closed branch.

### 9.3 What is tested

- **The hypothesis (the owner's).** Genuinely new, diverse, same-population training
  queries improve the remaining query failures, without the path and subdomain
  trade-off that the query weight caused.
- **The arm and the baseline.** The arm is `e15_new`. The baseline is `e12_d1p`:
  `e12_final` + D1 + D1′, E14's files at E12's seeds, frozen.
- **The target strata:**
  - benign **domain-frame** (name-corpus) rows in states **P0Q1** and **P1Q1**.
    These are the strata of the two remaining benign query failures, control benign
    P0Q1 QRY (`e12_d1p` 0.0624) and control benign P1Q1 QRY (0.0520).
  - Every gate cell is generated in the domain frame (`generate(frame="domain")`)
    and scored against the name pools, so these are the populations that matter.
- **Phishing,** for the free phishing QRY cell, enters only under §9.4.2's rule, and
  is expected not to.
- **The recipe is `e12_final`'s exactly:**
  - start c7e71edf…, seed 20260803;
  - 20,000 steps, batch 256, lr 3e-4, AdamW;
  - the fixed objective with C = 14,397.885481960007;
  - clip 1.0, then 5.0;
  - R1, the presence plan and the slash plan;
  - the model size and the tokenizer.
- **No QRY loss weighting:** every supervised position, query or not, keeps
  `e12_final`'s weight.
- **Decoder:** raw + D1 + D1′.

### 9.4 The data source and how it is built (frozen rules)

#### 9.4.1 Benign: corpus2's own population and procedure

**Corpus2, as built** (the manifests in `data/corpus2/`):
- **Crawls:** the 8 live crawls only: CC-MAIN-2024-42, 2024-51, 2025-13, 2025-26,
  2025-38, 2025-51, 2026-12, 2026-30.
- **Blocks:** suffix-stratified index-block sampling (`uie/corpus/stratified.py`,
  α = 0.5, seed 20260804), 4,251 blocks per crawl (34,014 in the manifest's total), with the
  extraction filters of `uie/corpus/extract.py`.
- **Name corpus** (`uie/corpus/merge.py`): one URL per registrable domain, chosen by
  reservoir (seed 20260805), 1,132,768 domains.
- **Split:** one global domain-disjoint map over both classes
  (`uie/corpus/build_split2.py`, seed 20260806; train / val / test 981,391 /
  122,673 / 122,675 domains).

**The new stream repeats that procedure on new blocks:**
- **Same crawls, same sampler and α,** the same extraction filters and the same
  name rule (reservoir, one URL per registrable domain).
- **Equal blocks per crawl,** which keeps the crawl and suffix stratification.
- **New seeds:** 20261005 for block draws, 20261006 for the reservoir.
- **Budget:** rounds of 500 blocks per crawl. Acquisition stops at the end of the
  first round in which both strata reach their targets, and never exceeds corpus2's
  own 4,251 blocks per crawl.
- **Exclusions:**
  - every registrable domain in `split_random` (any partition, either class);
  - every registrable domain or host in the phishing corpus
    (`data/corpus/phishing.jsonl`).
- **Kept:**
  - domain-frame rows (the name rule) whose state is P0Q1 or P1Q1, by the gate's
    `FIELDS`;
  - at most 384 tokens under `e12_final`'s tokenizer.
- **Targets:** one doubling, N_new,s = N_old,s, as encoded by the dry run (today:
  P0Q1 14,906, P1Q1 42,862 in `name_train`).
  - If a stratum exceeds its target, exactly the target is kept, in ascending order
    of sha256("E15-B:" + registrable domain).
  - If a stratum is short at the cap, all of it is kept, and §9.4.3 decides.
- **Where it lives:** `data/e15/qry_extra/benign_P0Q1.jsonl` and `benign_P1Q1.jsonl`,
  plain JSON lines, never gzipped, corpus tag `qry_extra`.
- **Its manifest records:**
  - per crawl, every drawn block (shard, offset) and every round;
  - the seeds, and the builder's commit;
  - each filter's and exclusion's outcome counts;
  - the row counts per stratum, and the sha256 of each file.

**Cost:** corpus2's extraction took 1.2–2.7 hours per crawl. The full cap is about
13 hours on the laptop, and one doubling is expected near 80% of it.

#### 9.4.2 Phishing: the owner's rule

- **The only admissible source** is the OpenPhish public feed's own later entries:
  commits after 48f7a98 (2026-08-03), parsed by the existing pipeline (one URL per
  host).
  - Hosts must be absent from `split_phishing_host`, and registrable domains absent
    from `split_random`.
  - Only domain-frame rows with a query are kept.
- **Admitted only if** N_new ≥ ½ × 8,272 = 4,136 rows **and** it passes every gate
  in §9.5.
- **Expected:** about 1,000 rows (≈ 15.7 query rows per feed-day since 2026-08-03),
  so **not admitted**. That would be reported as a limitation.
- **No substitute.** PhishTank, URLhaus, PhishStats and archived corpora are other
  populations, and none is used.

#### 9.4.3 Admitting a stratum

- **A stratum is admitted iff** N_new,s ≥ ½ N_old,s and it passes every gate in
  §9.5. Below half a doubling the coverage effect is too small to test.
- **If no stratum is admitted,** E15-B stops before training and reports.
- **The licensed cells** are the admitted strata's cells: control benign P0Q1 QRY
  and/or control benign P1Q1 QRY, plus free phishing QRY only if phishing is
  admitted.

### 9.5 Pre-training gates, all verified before any training

| gate | test | reference |
|---|---|---|
| **P1 population** | DX1's rule (`PREREG_E13D.md`), with the new rows as the twin: x_NV is the new real query strings against the cell's real validation pool, at DX1's harness, n and seeds. It must read NO SHIFT, i.e. not (x_NV ≥ 0.25 · x_GV with t > t_crit) | x_GV from E13-D: 0.0679 (P0Q1), 0.0642 (P1Q1), 0.1237 (phishing). The old training twins read −0.0033 / −0.0029 / −0.0019 |
| **P2 coverage gain** | Δcov = cov(old ∪ new) − cov(old), the share of the cell's validation templates covered (DX3's template and coverage functions). Δcov ≥ g_s · log2(1 + r_s) − 0.01, with r_s = N_new,s / N_old,s | g from DX3: 0.032 (P0Q1), 0.043 (P1Q1), 0.041 (phishing). At a full doubling: ≥ 2.2 and ≥ 3.3 points |
| **P3 leakage** | zero registrable-domain overlap with `split_random` (train, val **and test**); zero host overlap with `split_phishing_host`; zero exact-URL overlap with the validation files and the bundle's known-URL bloom (bloom hits resolved exactly). The test partition's rows are never read; its keys are in the map | — |
| **P4 mix** | suffix TV(new, old) and crawl TV(new, old) each at most the stratum's own train-against-validation TV. A same-population sample may differ from training by no more than the held-out rows already do | suffix 0.157 (P0Q1), 0.093 (P1Q1); crawl 0.029, 0.013 |
| **P5 diversity** | distinct templates in equal-size samples (n = min(N_new, N_old), 20 seeded draws, the median): new ≥ 0.9 × old. The top-10 templates' share of the new rows ≤ old's + 5 points. One row per registrable domain (verified) | the old stratum |
| **P6 genuinely new strings** | the share of new query strings absent from all old training queries ≥ the validation pool's never-seen share − 5 points. Reported: the share of new rows whose *template* is absent from old training | never-seen real (E13-D §5): 39.8% (P0Q1), 59.2% (P1Q1), 50.6% (phishing) |
| **P7 contamination** | zero new benign rows whose host or registrable domain is in the phishing corpus; corpus2's extraction filters applied unchanged | — |
| **P8 the numbers, frozen** | exact row counts per stratum; f_s (§9.6); the expected appended rows per batch; the expected QRY target mass per step, old against e12_final (V-mass in [0.95, 1.05] per stratum and [0.98, 1.02] for the class); the stream files' sha256; the builder's commit; every seed | — |

**Amendment 2** records every gate's result and every P8 number before any
training. The owner approves it (gate B, §9.12). A failed gate stops the stratum.
Nothing is tuned to make it pass.

### 9.6 Query-only supervision and matched exposure: the exact sampling rule

**Query-content positions** are as in `PREREG_E14.md` §2.1: every step taken while
the query is open, from its first character through the step that predicts its end
(`<FRG>` or the end token), but not the `<QRY>` step.

**New rows:**
- only their query-content positions are targets, at weight 1.0;
- every other position has weight 0: the plan tokens, every marker including
  `<QRY>`, the suffix, label, subdomains, the path, the fragment and the end token
  when it does not end the query.
- Their host and path are context only.

**Old rows:**
- drawn exactly as in `e12_final`: the same seed, sampler and batches (V-batch);
- every non-query position keeps `e12_final`'s weight on **every** draw.

**The replacement rule.**
- For each draw of an old row in an admitted stratum s, a side generator (seed
  20261007) flips a coin with probability f_s.
- **On heads:**
  - that draw's query-content positions are not supervised. This holds for that
    draw only; nothing about the row changes;
  - one new row of stratum s, drawn uniformly by a second side generator (seed
    20261008), is appended to the batch, its query content supervised at weight
    1.0.
- f_s = N_new,s / (N_old,s + N_new,s). It is the **frozen old/new sampling ratio**,
  (1 − f_s) : f_s = N_old,s : N_new,s, which is 1 : 1 at a full doubling.

**What this fixes, by construction:**
- **Non-query target mass per step is identical to `e12_final`'s,** in every class
  and every field. The intervention adds no mass to paths, subdomains or anything
  else, and removes none (decision 4).
- **The query rows supervised per step in s are unchanged in expectation,** so the
  overall query exposure is preserved (decision 5).
- **Every distinct query row in s, old or new, is supervised equally often:**
  n_s · T / (N_old,s + N_new,s) times. The data looks as if the stratum had been
  twice as large.
- **No query loss weight:** every supervised query position has weight 1.0
  (decision 1).

**Why this mechanism, and not another.** Only this rule satisfies decisions 1, 4
and 5 together:
- **adding the new rows on top** would add query gradient mass (decision 5);
- **swapping whole rows in the pool** would drop old rows' non-query targets on those
  draws, cutting path and subdomain mass in exactly the P1Q1 stratum whose path and
  subdomain cells are failing (decision 4);
- **down-weighting** is excluded (decision 1).

Old rows are never duplicated, oversampled or altered. A fraction f_s of their draws
do not supervise their query content, which is what holding query exposure fixed
requires. **The owner reviews this at gate A.**

**The numbers at a full doubling** (from `name_train`; the dry run fixes them
exactly as encoded):
- **Class probability per draw:** p_benign = 906,214 / (906,214 + 192,652) = 0.8247
  (class_alpha 1 over the label counts).
- **Old rows of s per batch:** n_s = 256 · p_benign · N_old,s / 2,551,259, giving
  n_P1Q1 = 3.547 and n_P0Q1 = 1.233.
- **f_s = 0.5.** The appended new rows per batch are 1.773 + 0.617 = 2.39, which is
  0.93% of the batch, for query targets only. Compute rises about 0.9%.
- **Query supervisions per row over 20,000 steps:** e12_final 1.655 per old row;
  E15-B 0.828 per row, old and new.
  - So most supervisions are first presentations already. The new rows replace
    second presentations of old rows.
  - This is why the expected effect is modest (§9.10).
- **Query target mass per step in s:** n_s · [(1 − f_s) · L_old,s + f_s · L_new,s],
  against e12_final's n_s · L_old,s.
  - L is the mean count of query-content positions per row (characters + the end
    step). Today's L_old: 25.7 (P0Q1) and 38.8 (P1Q1).
  - **V-mass (P8):** the ratio must lie in [0.95, 1.05] per stratum and
    [0.98, 1.02] for the benign class. Otherwise E15-B stops and reports. f is not
    adjusted.

### 9.7 Generation

- **`e15_new`'s 50 files:** E12's 50 jobs at E12's seeds, raw + D1 + D1′.
- **Every cell pairs with `e12_d1p`'s files,** the frozen baseline. Nothing of
  `e12_d1p` is regenerated.

### 9.8 Acceptance criteria: `e15_new` against `e12_d1p`

A retrained model, so the retraining allowance applies. Every threshold below is an
existing one except G3, which was declared new in §4.6.

| code | criterion |
|---|---|
| **K15-1** licensed query cells | at least one licensed cell's x falls by more than 0.0141, **and** no licensed cell's x rises by more than 0.0141 |
| **K15-2** held-out query fit | held-out query NLL (M3, per row, paired; `e14_fit.py`) falls in the licensed class against `e12_final`, with a 95% interval below 0. Reported per stratum too |
| **K15-3** band profile | M2: no overshoot of the null (`PREREG_E14.md` §3.1's definition) |
| **G1** no regression | K12c: no gated cell's verdict worsens with x up by more than 0.0141 |
| **G2** paths and subdomains | no PTH, SUB or LBL cell (free and control, both classes) rises by more than 0.0141, **whatever its verdict** |
| **G3** non-query fit | no non-query field's validation NLL (the training record's per-field table, both classes) rises by more than 0.01 nats per character against `e12_final`. That is 5× the benign noise reference, 0.002 |
| **G4** presence | K12a: free PTH and QRY within T, both classes (amendment 1's moves reported) |
| **G5** slash | all 8 K12b rows within T (the D1′ guard) |
| **G6** honouring | no control state's P, Q or S honouring falls (amendment 1's rule) |
| **G7** copying | K12d: no cell OVER its natural rate, and no copy rate moving further from it (amendment 1's rule) |
| **G8** novelty | K12d's whole-URL novelty rule, both modes |
| **G9** FULL | free FULL up by no more than 0.0141, both classes |
| **G10** training and validity | K12e (clipping); V-batch, V-mask, V-mass and V-stream (§9.10) |

- **Adopted iff** K15-1, K15-2, K15-3 and G1–G10 all hold. There is no retry and no
  tuning.
- **"Adopted"** keeps the stream and the arm. Whether `e15_new` becomes the
  baseline is the owner's decision.
- **E3 is read for the record.** It cannot be met in E15: the path and subdomain
  cells are untouched, and so is the phishing far cell unless phishing is admitted.

### 9.9 The report: four kinds of evidence, kept apart (decision 8)

| question | evidence used | how it is read |
|---|---|---|
| **Did new query content help?** | K15-1 under verified matched exposure (V-mass); K15-2 per stratum; **uptake**: the share of generated queries in the licensed cells that exactly match a *new-stream* query absent from old training, beside the share matching old training queries | new content helped only if K15-1 holds at matched exposure **and** the model demonstrably writes new-stream content. A gain with zero uptake is not attributed to the new content |
| **Was it merely more query exposure?** | the realized query target mass ratio (V-mass); E14 as the dose reference: 4× query mass moved P0Q1 by −0.026, P1Q1 by +0.002 and free phishing by −0.031 | upper bound on the exposure share of any change: \|ΔM/M\| × \|Δx_E14\| / 3. Within V-mass's ±5% that is at most 0.0005, against the 0.0141 margin |
| **Did coverage and diversity improve?** | in the generated output for the licensed cells, against `e12_d1p`: coverage of the validation templates; distinct templates per 1,000 queries; M2's rare-name share; exact-copy and never-seen rates (E13-D §5's table) | reported. "Improved" means a better template coverage of validation, not just more distinct strings |
| **Was there a path, subdomain or slash trade-off?** | G2, G3, G5; the per-field NLL table; slash misses by supplied state | any breach is reported as a trade-off even if the lever otherwise passes |

**E15-A was not run, and nothing in E15's report treats its absence as evidence.**

### 9.10 Validity

| code | check |
|---|---|
| V1 | hashes and completeness: the code bundle; `e12_final.pt`; `e15_new`'s `final.pt` as training recorded it; the stream files (amendment 2's sha256); the tables; every generated file with its registered n and seed |
| V2 | byte identity on the laptop CPU: at least 250 URLs per mode |
| V3 | options off are unchanged: the full suite; E12's fingerprints; without the stream, training is `e12_final`'s bit for bit |
| V4 | the scorer reproduces `e12_final`'s published cells exactly |
| V-start | the start is E12's (c7e71edf…; the E10 and E11 fingerprints) |
| V-batch | the old rows' batch digest, recorded per step on Kaggle, equals the laptop's replay of `e12_final`'s sampler |
| V-mask | unit tests and a dry run. New rows: only query-content positions are non-zero, at 1.0. Old rows: non-query weights equal `e12_final`'s on every draw. Replaced draws: their query positions are 0 |
| V-mass | the training record's appended-row and query-mass counts are within 3 SE of §9.6's expectations, and V-mass's bands hold |
| V-stream | the training record's stream hash, row counts and corpus tag equal amendment 2's |

### 9.11 Predictions [HYPOTHESIS]

- **Gates:**
  - P1 and P3–P7 pass for both benign strata, since they come from the same
    procedure;
  - P2 passes, since a same-population doubling should reproduce DX3's g;
  - phishing is not admitted (about 1,000 rows against 4,136).
- **K15-1 FAILS:** no licensed cell falls by more than 0.0141.
  - P1Q1 changes by between −0.012 and 0; P0Q1 by ±0.010.
  - **Why:** the record reads learning, not data (DX1, DX2). Each old row is already
    supervised about 1.65 times, so swapping half of those presentations for new
    rows changes coverage by about DX3's 4 points, which the stand-in learner did
    not convert into a lower tell.
- **K15-2 passes, small:** benign held-out query NLL −0.005 to −0.03 nats per
  character.
- **G2 and G3 pass,** because non-query mass is identical by construction. This is
  the design's central claim, and the one the owner asked for.
- **G5 passes (D1′),** and G1, G4 and G6–G10 pass.
- **Overall: not adopted, on K15-1.**
  - Then E15-B is the registered transformer-level test that genuinely new
    same-population query content, at matched exposure, does not materially
    improve the query cells.
  - That closes the data direction for these cells, and is a reading R3's first
    case can use. The decision stays the owner's.

### 9.12 Execution steps and the owner's gates (replaces §7)

| step | who | what | about |
|---|---|---|---|
| **gate A** | owner | approve Amendment 1: the design, the replacement rule (§9.6), the gates (§9.5), the criteria (§9.8). This approval starts implementation and data acquisition | — |
| 1 | laptop | **Code and tests,** every option off by default. In `uie/`: the `qry_extra` side stream, the replacement coins, the query-only masks, the old-row batch digest, the training record. In `research/e15/`: acquisition (extraction rounds, merge, exclusions, targets), gates P1–P8, the dry run, the sampler replay, the verifier, the scorer, the criteria and the four-way report, the bundle and notebook builders. Full suite; commit and push, verified on the remote | a session |
| 2 | laptop | **Acquisition and gates:** benign extraction rounds (at most about 13 h); merge, exclusions and targets; the phishing feed count; P1–P7; the dry run (P8). Then **amendment 2** with every number and hash. Commit and push | about 1 day |
| **gate B** | owner | review amendment 2 (admitted strata, counts, f_s, masses, gate results); approve the Kaggle runs | — |
| 3 | laptop | the bundle (the stream plain, never gzipped), notebooks `E15B_train` and `E15B_generate`, `E15_KAGGLE.md`, smoke tests. Commit and push | 1–2 h |
| 4 | Kaggle | **`E15B_train`**: `e15_new`, 20,000 steps. Inputs: `uie-corpus2`, `uie-e15-code`. GPU T4 | about 1.4 h |
| 5 | laptop | VERIFY_TRAIN (V1, V-start, V-batch, V-mask, V-mass, V-stream) and K15-2 (M3). Only after these pass does step 6 run | about 1.5 h |
| 6 | Kaggle | **`E15B_generate`**: 50 files, raw + D1 + D1′. GPU T4 | about 2 h |
| 7 | laptop | VERIFY_GENERATE, V4, scoring, the criteria, the four-way report (§9.9), E3 for the record; `E15B_RESULTS.md`. Commit and push | about 6 h |
| **gate C** | owner | adoption, and whether `e15_new` becomes the baseline | — |

**Nothing above runs before gate A.** No code, no data acquisition, no Kaggle run
and no training.

## 10. Amendment 2 (2026-10-05): matched exposure revoked; the new rows are injected alongside the old ones

**Registered before any E15 code, data acquisition, training or generation.** Amendment
1 was not approved ("not approved wait for instruction"). This amendment replaces
§9.6 and changes §9 only where listed in §10.2. §9's other sections stand.

### 10.1 The owner's decision, verbatim

> "I have reviewed the 'coin-flip swap' mechanism and I am explicitly rejecting it.
> I realize my previous instruction regarding 'Matched exposure' was overly
> restrictive and contradicted the core of my hypothesis. I am officially revoking
> the 'Matched exposure' constraint.
> My hypothesis is exactly this: Increasing QRY exposure by adding genuinely new,
> diverse data will improve learning without causing the destructive PTH/SUB
> capacity trade-off that we saw in E14 (which was caused by artificially
> overweighting repetitive, old data).
> Therefore, do not swap. Simply inject the new rows into the training stream
> alongside the old rows.
> For the new rows: They provide training targets for QRY only (all other fields
> are input-only).
> For the old rows: They train exactly as they do today, across all fields.
> This means the total QRY training mass will naturally increase. That is precisely
> what I want to test

### 10.2 What changes

| in §9 | amended |
|---|---|
| §9.1 decision 5 ("Matched exposure") | **revoked by the owner.** Decisions 1–4 and 6–8 stand |
| §9.6, the replacement ("coin-flip swap") rule | **rejected by the owner, and removed.** Replaced by §10.4 |
| §9.5 P8's V-mass band ([0.95, 1.05] / [0.98, 1.02]) | removed: query mass now rises by design. P8 instead records the exact added mass (§10.4) |
| §9.9's exposure row | replaced by §10.6: content and exposure are now confounded by design, and the report says so |
| §9.10 V-batch, V-mask, V-mass | restated in §10.7 |
| §9.11 predictions | replaced by §10.8 |
| §9.12, which calls the numbers amendment "amendment 2" | that amendment is renamed **amendment 3**. The steps are otherwise unchanged (§10.9) |

**Unchanged from §9:**
- the source and its build (§9.4: corpus2's 8 crawls and procedure, new domains only,
  one doubling of the two benign strata);
- the phishing rule and the admission rule;
- gates P1–P7;
- the criteria K15-1 to G10 (§9.8);
- the four-way report's other rows (§9.9);
- the frozen baseline `e12_d1p`.

### 10.3 What is tested, now

**The owner's hypothesis.** Adding genuinely new, diverse query data raises query
exposure and improves query learning, without the path and subdomain cost of E14.

**What a single arm can and cannot establish:**
- **It can establish** whether injecting one doubling of new same-population query
  rows into the two benign domain-frame strata:
  - improves the licensed query cells (K15-1, K15-2);
  - and does so without a material path, subdomain or other-field cost **at this
    dose** (G2, G3).
- **It cannot separate new content from added exposure.** Both rise together by
  design.
- **It cannot by itself show that new data avoids E14's trade-off mechanism.**
  - The added query mass is about 1/13 of E14's (§10.4).
  - An E14-proportional cost at this dose would be 0.002–0.004 nats per character:
    inside the noise reference (0.002) and under G3's 0.01.
  - So a clean G2/G3 is compatible both with the owner's hypothesis and with E14's
    mechanism at a smaller dose.
  - A material cost, by contrast, would be informative: it would contradict "new
    data avoids the trade-off".
- **[Reading, stated because the owner's text attributes E14's cost to "overweighting
  repetitive, old data"]** E14's evidence does not show that repetition caused the
  cost.
  - Every non-query field lost fit on its *training* rows as well as on validation,
    which is the signature of fields competing for a fixed budget.
  - Whether new content avoids that competition is the open question this design
    approaches. It does not test it at equal dose.
- **The only within-experiment way to test it at equal dose** is a control arm. That
  would be the same two strata's *old* query content given the same added mass, i.e.
  query weight 2 on those strata. It is **not registered.** It would use QRY loss
  weighting, which the owner excluded, so it is the owner's call (§10.9, gate A).

### 10.4 Injection: the exact rule

**Old rows** train exactly as in `e12_final`:
- the same seed, sampler and batches (V-batch);
- the same masks and weights on every draw;
- the same pinned loss constant C = 14,397.885481960007, so their gradient
  contribution per batch is unchanged in absolute terms.

**New rows** are injected alongside the old rows:
- **When.** Whenever the sampler draws an old row of an admitted stratum s, the next
  new row of s is appended to the same batch.
  - The new row is drawn from the stratum's new rows by a side generator (seed
    20261007), as a without-replacement shuffle that is re-shuffled at each pass.
  - So new rows are drawn **at exactly the rate old rows of their stratum are drawn**.
- **When a stratum is short** (N_new,s < N_old,s), a new row is appended with
  probability N_new,s / N_old,s per old draw (seed 20261008). So each new row is
  still drawn at the old rows' per-row rate.
- **What they teach.** New rows supervise **query content only,** at weight 1.0, as
  defined in `PREREG_E14.md` §2.1. Every other position is input only, at weight 0:
  plan, markers including `<QRY>`, suffix, label, subdomains, path, and the fragment.

**The frozen ratio.** Within each admitted stratum, new : old query supervisions
= N_new,s : N_old,s. That is **1 : 1 at a full doubling**, so the stratum's query
supervision doubles.

**No QRY loss weighting.** Every supervised position, old or new, keeps weight 1.0
(the owner's decision 1).

**The dose, at a full doubling** (from `name_train` and E14's training record; the dry
run fixes the exact values as encoded):

| quantity | value |
|---|---|
| old rows of the stratum per batch (= new rows appended) | P1Q1 3.547, P0Q1 1.233; **4.78 rows per batch** (+1.87% of 256) |
| query supervisions per row over 20,000 steps | old rows 1.655 (unchanged); new rows 1.655 (the same rate) |
| query positions per batch, `e12_final` | 750 (5.21% of the 14,398 scored per batch) |
| **added query positions per batch** | **169.3: query mass +22.6% overall, +100% in the two strata; total scored mass +1.18%** |
| E14's added mass, for scale | 2,249 per batch (+300% query mass everywhere; +15.6% scored mass). **This design is about 1/13 of it** |
| E14-proportional per-field cost at this dose | 0.0021–0.0041 nats per character (benign LBL 0.0029, PTH 0.0021; phishing PTH 0.0041, SUB 0.0031) |
| compute | about 2–3% more per step; the same 20,000 steps |

**What else changes, and what does not:**
- **The old rows' data, batches and loss weights are unchanged.** They are coupled to
  the new rows only through the shared parameters, AdamW's moment estimates and the
  gradient-norm clip.
  - The clip acts on the batch's total gradient; it bound on about 1–4% of
    `e12_final`'s steps.
  - These shared-parameter channels are where E14's trade-off arose, and are what
    G2 and G3 watch.
- **The class balance is unchanged.** The new stream is outside the sampler's class
  weights, which come from the label counts.

**P8 now records, before training:**
- the exact counts and the frozen ratio;
- the expected appended rows per batch;
- the expected added query mass per batch, per stratum and per class;
- the stream's sha256, the builder's commit and every seed.

### 10.5 Criteria

§9.8 stands unchanged: K15-1, K15-2, K15-3 and G1–G10, against the frozen
`e12_d1p`, with every existing gate-v2 threshold, tolerance and E3 definition.

**Added as a reported measure, not a criterion.** Each field's validation NLL on
the two target strata's own validation rows: benign domain-frame P0Q1 and P1Q1,
7,253 rows. That is where the added query gradient concentrates, so a local cost
would show there first.

### 10.6 The report's four kinds of evidence, under injection (replaces §9.9's exposure row)

| question | under injection |
|---|---|
| **Did new content help?** | it cannot be separated from added exposure within this arm. Reported: **uptake** (the share of generated queries in the licensed cells that exactly match a new-stream query absent from old training) and per-stratum M3. A gain with zero uptake is attributed to exposure, not to new content |
| **Was it merely more exposure?** | not separable. Reported: the realized added mass (+22.6% query, +1.18% scored, as verified), and E14 as an external, imperfect dose reference. E14 put +300% query mass everywhere and moved P0Q1 −0.026, P1Q1 +0.002 and free phishing −0.031. The doses differ by about 13× and in where they land, so no causal split is claimed |
| **Did coverage and diversity improve?** | as §9.9: the generated output's coverage of the validation templates, distinct templates, M2 and the copy/never-seen rates, against `e12_d1p` |
| **Was there a path, subdomain or slash trade-off?** | G2, G3, G5; the per-field NLL tables, including §10.5's stratum table, each shown beside its E14-proportional expectation at this dose (§10.4). A clean result is reported as "no cost detectable at about 1/13 of E14's dose", not as "no trade-off" |

### 10.7 Validity (restates §9.10's V-batch, V-mask, V-mass)

| code | check |
|---|---|
| V-batch | the old rows' batch digest, recorded on Kaggle, equals the laptop's replay of `e12_final`'s sampler |
| V-mask | unit tests and the dry run. New rows: only query-content positions are non-zero, at 1.0. Old rows: every mask equals `e12_final`'s, on every draw |
| V-mass | the training record's appended rows and added query positions are within 3 SE of §10.4's expectations, as fixed by amendment 3 |

V1–V4, V-start and V-stream stand as in §9.10.

### 10.8 Predictions, replacing §9.11 [HYPOTHESIS]

- **Gates:**
  - P1 and P3–P7 pass for both benign strata;
  - P2 passes;
  - phishing is not admitted (about 1,000 rows against 4,136).
- **K15-1 most likely FAILS, narrowly:**
  - **P0Q1** falls by 0.003–0.012. Exposure helps this cell (E14 cleared it), but the
    dose here is 2× and local to two strata, against E14's 4× everywhere.
  - **P1Q1** changes by −0.008 to +0.004. No exposure lever has moved it.
  - If K15-1 passes, it will be through P0Q1.
- **K15-2 passes:** held-out query NLL falls.
- **Uptake is non-zero.** The model writes new-stream queries at a rate comparable
  to its copy rate of old training queries.
- **G2 and G3 pass.** At about 1/13 of E14's dose, an E14-proportional cost is
  0.002–0.004 nats per character, below G3's 0.01. So this pass will not, by itself,
  show that new data avoids the trade-off (§10.3).
- **G5 passes (D1′),** and G1, G4 and G6–G10 pass.
- **Overall: not adopted (K15-1), most likely.**

### 10.9 Execution steps and gates

§9.12 stands, with the numbers amendment renamed **amendment 3**, and:
- **gate A** (owner) approves this Amendment 2. The owner also decides whether to
  add the dose-matched control arm (§10.3). It is not registered, and adding it
  would be its own amendment;
- step 1's code implements §10.4's injection, not §9.6's replacement;
- step 4's training time rises by about 2–3%.

**Nothing runs before gate A:** no code, no data acquisition, no Kaggle run and no
training.

## 11. Gate A (2026-10-05): approval, the owner's directives, and the rules they change

**Registered before any data acquisition.** It changes §9 and §10 only where listed
in §11.2.

### 11.1 The owner's approval, verbatim

> "Thank you for the precise, concrete breakdown. The pipeline logic and the
> bottleneck analysis are clear, and I now understand why the structural filtering
> demands high network I/O.
> I officially approve Gate A with the following execution directives:
> Benign Data Acquisition: Proceed with the full target volume (one doubling: 57,768
> rows). Execute the 8 crawls in parallel (the 2-4 hour option) to optimize network
> time.
> Phishing Data Acquisition: You are authorized to run the git fetch on the OpenPhish
> repo. However, do NOT drop the phishing injection if it falls short of the target.
> I want to use whatever genuinely new QRY-bearing phishing data is available (even
> if it's ~1,000 rows). Inject it using the fractional probability mechanism you
> already registered in Amendment 2 (p = N_new/N_old). Any new data is better than
> none.
> Control Arm: As previously stated, the Control Arm (w=2) is rejected and
> excluded.
> Proceed with the acquisition. Once complete, run the pre-training population
> gates/checks, and present the final measured numbers (Amendment 3) for my Gate B
> review"

### 11.2 What changes

1. **The phishing size minimum is waived** (the owner's directive). It replaces
   §9.4.2's ≥ 4,136 rows and §9.4.3's N_new,s ≥ ½ N_old,s, for phishing only.
   - The phishing strata are admitted at any size N_new,s ≥ 1, provided their gates
     pass: P1, P3, P4 (suffix only; the feed has no crawls), P5, P6 and P7.
   - P2 is reported for phishing. Its threshold, g · log2(1 + r) − 0.01, is
     negative below r ≈ 0.18, so it cannot bind there.
   - P1 runs at DX1's n (356 per seed for free phishing). If fewer new rows exist,
     it runs at n = N_new, reported as reduced power.
   - Injection follows §10.4's fractional rule: per old-row draw, a new row is
     appended with probability p_s = N_new,s / N_old,s.
   - **The benign minimum is unchanged:** N_new,s ≥ ½ N_old,s.
2. **Correction: the phishing leakage key is the host only.** This applies to
   `split_phishing_host`'s train, val and test keys.
   - The corpus's phishing split is keyed on the host. `merge_phishing.py` ignores
     `split_random` for phishing.
   - §9.4.2's extra condition, "registrable domains absent from `split_random`", is
     removed for phishing. About 73% of phishing sits on shared hosting (for
     example `x.vercel.app`), whose platform domain is in that split, so the
     condition would discard most phishing and skew the rest toward own-domain kits.
   - **Added:** no new phishing URL may equal any URL in the train or val files of
     either class.
3. **Parallel execution** (the owner's directive).
   - The 8 crawls run as parallel processes, one per crawl, round by round. Each
     round is 500 blocks per crawl, as in §9.4.1.
   - The procedure and seeds are those of a sequential run. Only network retries can
     differ, and the manifest records them: blocks drawn, HTTP errors and resync
     failures per crawl and round.
   - A crawl-round that delivers under 90% of its blocks is retried
     (`extract.py`'s own rule). Nothing short is ever recorded as done.
4. **The seeds, fully specified.** §9.4.1 named the base seeds but not their
   derivation.
   - **Block draws** for crawl i (`LIVE_CRAWLS` order, i = 0…7) in round r (1, 2, …):
     20261005 + 7919 · i + 1,000,003 · (r − 1). `extract.py`'s per-domain reservoir
     uses the same seed, as in corpus2.
   - **The benign name reservoir** (`merge.py`'s rule: k = 1, "c == 1 or
     randrange(c) == 0"): seed 20261006, over the shard files in round order, then
     crawl order.
   - **The phishing one-URL-per-host pick** (`merge_phishing.py`'s rule,
     `rng.choice` per host, hosts in first-seen order): seed 20261009.
5. **Contamination (P7) uses the updated feed.** Every host and registrable domain
   in the fetched OpenPhish history (old and new) is excluded from the benign
   stream, not only the corpus's phishing list.
6. **Cost, corrected.** §9.4.1's "about 13 hours" is the sequential figure (corpus2's
   own extraction: 13.6 h, one crawl after another). In parallel, 2–4 h are
   expected.

**Unchanged:**
- the benign targets (one doubling as encoded: today 14,906 and 42,862);
- the rounds, the cap (4,251 blocks per crawl) and the stop rule;
- every gate's definition except as above;
- the injection rule (§10.4), the criteria (§9.8) and the four-way report (§10.6);
- **Amendment 3 records the measured numbers before gate B.**

## 12. Amendment 3 (2026-10-06): the measured numbers, before any training (for gate B)

**Recorded after acquisition and the gates, before any E15 training or generation.**
- The tables below are `research/e15/e15_amendment3.py`'s output, pasted verbatim. No
  number in them was typed by hand.
- Gate B approves them.

### 12.1 In plain words

- **Benign: both strata are admitted.** They carry exactly one doubling as encoded:
  14,863 P0Q1 and 42,429 P1Q1, 57,292 rows in all.
- **Every benign gate passes:**
  - zero leakage and zero contamination;
  - the domain-ending and crawl mix inside the natural train-against-validation spread;
  - no population shift (DX1 reads NO SHIFT in both cells);
  - coverage gains above their thresholds (+2.77 and +3.80 points);
  - diversity on a par with the old data;
  - genuinely new strings at the validation pool's own rate.
- **Phishing: not admitted.** The size waiver (§11.2.1) applied, but the population
  gates failed.
  - **P1 SHIFT-DOMINANT.** A detector separates the 918 new phishing queries from the
    gate's held-out phishing at +0.281. The old training queries score −0.002, and the
    model's own output +0.124.
  - **P4 fails:** domain-ending distance 0.28–0.38, against a natural 0.13–0.14.
  - **P5 fails:** 0.56× the template diversity, and the top 10 templates make up 49%
    of the rows (26% before).
  - **The cause, described:** new kit templates absent from the corpus's window
    (`usrid=`, `rotated=`, `welcome=…&token248=…`), plus a Blogspot `m=1` wave on 163
    hosts.
- **The dose:**
  - 4.74 rows appended per batch (1.85%);
  - 141 query positions per batch: query mass +18.8%, scored mass +0.98%, 0.063 of
    E14's added mass;
  - the two strata's query supervision ×1.99.

### 12.2 Notes and deviations, stated

- **The targets are "one doubling as encoded":** 57,292 rows. The owner's "57,768"
  counted rows before the 384-token limit, which drops 43 P0Q1 and 433 P1Q1 rows.
- **Concurrency changed mid-run (f5508d9).** Rounds 1–3 ran 8 crawls in parallel.
  - Common Crawl throttled that: 3, 4 and then 7 crawls fell under 90% of their blocks
    in rounds 2, 3 and 4.
  - The run was stopped during a retry pause and resumed at round 4, at most 4 crawls
    at a time. No crawl-round was short after that.
  - Each crawl-round's draws are fixed by its seed, and nothing short was recorded as
    done.
  - The failed attempts are in `data/e15/qry_extra/logs/` (`*_a1.log`). The table's
    error counts are those of the successful attempts.
- **The per-file cap.** `extract.py` keeps at most 48 URLs per domain per *file*. Here a
  crawl's blocks span 8 round-files; in corpus2 they were one file per crawl. The name
  rule (one URL per domain, a reservoir over everything kept) is otherwise identical.
  "The same rules where possible."
- **The phishing step ran before its code was committed.** The file was unchanged
  afterwards. The manifest records the sha256, and the feed is frozen at d0642d7.
- **Where the files are.** The stream lives in `data/e15/qry_extra/`, not in git. Its
  hashes are above, and the bundle will check them.

### 12.3 The records

### Acquisition

- **OpenPhish archive:** https://github.com/openphish/public_feed.git, head 48f7a982b2a4 → d0642d73d6a3 (2026-10-05T12:00:06Z); 12,061 commits; 382,998 distinct URLs.
- **Phishing funnel:** 28,855 new URLs → 21,350 hosts → 1,134 seen before the cutoff → 20,216 new hosts (one URL each) → 19,286 without a query, 12 over 384 tokens, 0 exact-URL overlaps → **355 P0Q1 + 563 P1Q1**.
- **Benign:** 8 rounds of up to 500 blocks per crawl over the 8 crawls, in parallel (cap 4251). Seeds: blocks 20261005 + 7919*i + 1000003*(r-1); name reservoir 20261006.

| round | crawl | blocks | records | kept URLs | domains | minutes | HTTP errors | resync failures |
|---|---|---|---|---|---|---|---|---|
| 1 | CC-MAIN-2024-42 | 500 | 1,080,821 | 269,022 | 16,714 | 17.8 | 0 | 3 |
| 1 | CC-MAIN-2024-51 | 500 | 1,091,024 | 254,972 | 15,428 | 12.6 | 0 | 5 |
| 1 | CC-MAIN-2025-13 | 500 | 1,069,613 | 262,545 | 16,052 | 17.9 | 0 | 6 |
| 1 | CC-MAIN-2025-26 | 500 | 1,112,510 | 270,434 | 17,379 | 18.0 | 0 | 2 |
| 1 | CC-MAIN-2025-38 | 500 | 1,121,457 | 291,041 | 18,400 | 17.7 | 0 | 2 |
| 1 | CC-MAIN-2025-51 | 500 | 1,087,971 | 279,303 | 18,092 | 17.2 | 0 | 3 |
| 1 | CC-MAIN-2026-12 | 500 | 1,083,225 | 290,570 | 20,460 | 14.9 | 0 | 1 |
| 1 | CC-MAIN-2026-30 | 500 | 929,954 | 233,621 | 13,940 | 16.7 | 0 | 2 |
| 2 | CC-MAIN-2024-42 | 500 | 1,085,960 | 265,753 | 16,269 | 15.4 | 8 | 1 |
| 2 | CC-MAIN-2024-51 | 500 | 1,104,287 | 268,532 | 16,087 | 6.6 | 0 | 1 |
| 2 | CC-MAIN-2025-13 | 500 | 1,089,827 | 256,976 | 15,227 | 15.6 | 8 | 3 |
| 2 | CC-MAIN-2025-26 | 500 | 1,108,855 | 289,608 | 18,691 | 7.2 | 0 | 3 |
| 2 | CC-MAIN-2025-38 | 500 | 1,125,935 | 283,084 | 18,098 | 15.3 | 8 | 3 |
| 2 | CC-MAIN-2025-51 | 500 | 1,108,692 | 271,577 | 18,604 | 14.9 | 8 | 3 |
| 2 | CC-MAIN-2026-12 | 500 | 1,059,081 | 283,311 | 19,398 | 6.8 | 0 | 0 |
| 2 | CC-MAIN-2026-30 | 500 | 923,611 | 251,120 | 15,549 | 13.8 | 8 | 1 |
| 3 | CC-MAIN-2024-42 | 500 | 1,068,218 | 255,287 | 16,017 | 6.8 | 0 | 1 |
| 3 | CC-MAIN-2024-51 | 500 | 1,086,107 | 270,928 | 16,286 | 9.7 | 9 | 4 |
| 3 | CC-MAIN-2025-13 | 500 | 1,103,755 | 264,320 | 15,727 | 7.8 | 0 | 2 |
| 3 | CC-MAIN-2025-26 | 500 | 1,106,348 | 258,577 | 15,894 | 7.8 | 0 | 1 |
| 3 | CC-MAIN-2025-38 | 500 | 1,112,049 | 278,659 | 17,253 | 16.7 | 16 | 2 |
| 3 | CC-MAIN-2025-51 | 500 | 1,097,906 | 290,578 | 19,095 | 16.4 | 16 | 0 |
| 3 | CC-MAIN-2026-12 | 500 | 1,081,160 | 289,171 | 20,056 | 14.1 | 16 | 4 |
| 3 | CC-MAIN-2026-30 | 500 | 911,994 | 237,035 | 14,644 | 6.8 | 0 | 1 |
| 4 | CC-MAIN-2024-42 | 500 | 1,107,828 | 291,579 | 18,378 | 7.9 | 0 | 7 |
| 4 | CC-MAIN-2024-51 | 500 | 1,107,308 | 266,305 | 15,467 | 7.6 | 0 | 1 |
| 4 | CC-MAIN-2025-13 | 500 | 1,100,424 | 244,222 | 14,207 | 7.5 | 0 | 1 |
| 4 | CC-MAIN-2025-26 | 500 | 1,091,617 | 277,876 | 18,188 | 8.1 | 0 | 3 |
| 4 | CC-MAIN-2025-38 | 500 | 1,116,493 | 274,538 | 17,346 | 6.1 | 0 | 2 |
| 4 | CC-MAIN-2025-51 | 500 | 1,103,195 | 272,005 | 18,641 | 6.6 | 0 | 3 |
| 4 | CC-MAIN-2026-12 | 500 | 1,093,608 | 295,637 | 19,700 | 6.0 | 0 | 5 |
| 4 | CC-MAIN-2026-30 | 500 | 935,251 | 240,741 | 14,980 | 15.3 | 15 | 1 |
| 5 | CC-MAIN-2024-42 | 500 | 1,096,415 | 272,273 | 16,173 | 12.4 | 0 | 1 |
| 5 | CC-MAIN-2024-51 | 500 | 1,088,392 | 270,788 | 16,017 | 7.1 | 0 | 3 |
| 5 | CC-MAIN-2025-13 | 500 | 1,087,521 | 251,254 | 14,743 | 12.7 | 0 | 0 |
| 5 | CC-MAIN-2025-26 | 500 | 1,066,844 | 268,469 | 17,429 | 12.3 | 0 | 0 |
| 5 | CC-MAIN-2025-38 | 500 | 1,107,822 | 270,697 | 16,235 | 11.6 | 0 | 4 |
| 5 | CC-MAIN-2025-51 | 500 | 1,109,742 | 275,186 | 17,341 | 11.3 | 0 | 3 |
| 5 | CC-MAIN-2026-12 | 500 | 1,096,101 | 287,072 | 19,968 | 8.2 | 0 | 5 |
| 5 | CC-MAIN-2026-30 | 500 | 941,961 | 237,282 | 14,167 | 10.8 | 0 | 3 |
| 6 | CC-MAIN-2024-42 | 500 | 1,077,468 | 262,428 | 15,901 | 12.1 | 0 | 8 |
| 6 | CC-MAIN-2024-51 | 500 | 1,096,386 | 273,168 | 16,312 | 6.2 | 0 | 1 |
| 6 | CC-MAIN-2025-13 | 500 | 1,053,002 | 238,425 | 14,202 | 12.1 | 0 | 5 |
| 6 | CC-MAIN-2025-26 | 500 | 1,113,561 | 277,807 | 17,826 | 11.6 | 0 | 2 |
| 6 | CC-MAIN-2025-38 | 500 | 1,121,987 | 283,917 | 17,804 | 11.4 | 0 | 0 |
| 6 | CC-MAIN-2025-51 | 500 | 1,106,554 | 273,298 | 18,228 | 11.5 | 0 | 4 |
| 6 | CC-MAIN-2026-12 | 500 | 1,101,178 | 309,497 | 21,293 | 8.3 | 0 | 10 |
| 6 | CC-MAIN-2026-30 | 500 | 913,922 | 245,150 | 14,571 | 9.8 | 0 | 5 |
| 7 | CC-MAIN-2024-42 | 500 | 1,100,585 | 268,966 | 16,148 | 11.8 | 0 | 2 |
| 7 | CC-MAIN-2024-51 | 500 | 1,095,396 | 276,243 | 16,651 | 6.1 | 0 | 3 |
| 7 | CC-MAIN-2025-13 | 500 | 1,085,008 | 269,494 | 16,013 | 11.9 | 0 | 1 |
| 7 | CC-MAIN-2025-26 | 500 | 1,126,646 | 276,941 | 17,614 | 11.1 | 0 | 0 |
| 7 | CC-MAIN-2025-38 | 500 | 1,106,790 | 274,081 | 17,609 | 12.1 | 0 | 1 |
| 7 | CC-MAIN-2025-51 | 500 | 1,092,941 | 284,665 | 18,763 | 12.5 | 0 | 5 |
| 7 | CC-MAIN-2026-12 | 500 | 1,105,058 | 282,966 | 19,156 | 9.6 | 0 | 7 |
| 7 | CC-MAIN-2026-30 | 500 | 928,374 | 232,955 | 13,715 | 11.1 | 0 | 2 |
| 8 | CC-MAIN-2024-42 | 500 | 1,107,193 | 287,097 | 17,585 | 12.0 | 0 | 3 |
| 8 | CC-MAIN-2024-51 | 500 | 1,098,927 | 267,852 | 16,346 | 6.5 | 0 | 8 |
| 8 | CC-MAIN-2025-13 | 500 | 1,077,868 | 255,518 | 14,960 | 11.7 | 0 | 1 |
| 8 | CC-MAIN-2025-26 | 500 | 1,079,425 | 279,563 | 18,076 | 11.5 | 0 | 4 |
| 8 | CC-MAIN-2025-38 | 500 | 1,090,920 | 274,090 | 17,099 | 12.7 | 0 | 3 |
| 8 | CC-MAIN-2025-51 | 500 | 1,098,888 | 283,628 | 18,850 | 12.8 | 0 | 1 |
| 8 | CC-MAIN-2026-12 | 500 | 1,089,059 | 283,440 | 19,508 | 9.0 | 0 | 3 |
| 8 | CC-MAIN-2026-30 | 500 | 913,040 | 224,234 | 13,523 | 11.7 | 0 | 4 |
| all | | **32,000** | 68,781,058 | 17,247,371 | | | 112 | 179 |

- **Benign funnel:** 17,247,371 URLs read → 1,061,248 registrable domains (one URL each, reservoir) → 48,343 already in the corpus split → 13 in the phishing feed → 948,730 without a query → 442 over 384 tokens → **16,028 P0Q1 + 47,692 P1Q1** available.

| stream file | rows | available | target (N_old) | sha256 |
|---|---|---|---|---|
| benign_P0Q1.jsonl | 14,863 | 16,028 | 14,863 | `4ef1b466f1190e2705b07cfcd691dd48649bb9069e796be2a99987d78c30b746` |
| benign_P1Q1.jsonl | 42,429 | 47,692 | 42,429 | `ce66a7db77b4fabb079a0500889c099e0daabf7f18c484461b7e70c7e77ee8cd` |
| phishing_P0Q1.jsonl | 355 | — | — | `0a3c1fc047946fb4e2da9b1160e11a4fee687ad2a5de56108e8ee8f3fa68b774` |
| phishing_P1Q1.jsonl | 563 | — | — | `ca8c055826dcd4a43393a6201ca3273a378c19e8424ac145479618d56b3a4ef4` |

### The gates

| stratum | N_new / N_old | P3 leakage | P4 mix | P7 contamination | admitted |
|---|---|---|---|---|---|
| benign_P0Q1 | 14,863 / 14,863 | PASS (key overlap 0, exact URL 0) | suffix TV 0.104 vs 0.157; crawl TV 0.012 vs 0.029 PASS | PASS | **yes** |
| benign_P1Q1 | 42,429 / 42,429 | PASS (key overlap 0, exact URL 0) | suffix TV 0.087 vs 0.093; crawl TV 0.007 vs 0.013 PASS | PASS | **yes** |
| phishing_P0Q1 | 355 / 2,978 | PASS (key overlap 0, exact URL 0) | suffix TV 0.281 vs 0.130 **FAIL** | PASS | **no** |
| phishing_P1Q1 | 563 / 5,152 | PASS (key overlap 0, exact URL 0) | suffix TV 0.379 vs 0.139 **FAIL** | PASS | **no** |

| gate cell | P1 population (DX1) | P2 coverage gain | P5 diversity | P6 new strings |
|---|---|---|---|---|
| control_benign_P0Q1_QRY | NO SHIFT: x_NV -0.0012 (t -0.08, n 600); old twin -0.0033; x_GV +0.0679 PASS | +2.77 pts (needs ≥ +2.23; r 1.000) PASS | ratio 1.022 (≥ 0.9); top-10 0.412 vs 0.418 (+0.05) PASS | never seen 58.6% vs val 57.1% (−5) PASS; new templates 20.7% |
| control_benign_P1Q1_QRY | NO SHIFT: x_NV +0.0062 (t 1.51, n 1800); old twin -0.0029; x_GV +0.0642 PASS | +3.80 pts (needs ≥ +3.29; r 1.000) PASS | ratio 0.978 (≥ 0.9); top-10 0.167 vs 0.163 (+0.05) PASS | never seen 74.1% vs val 75.2% (−5) PASS; new templates 31.5% |
| free_phishing_QRY | SHIFT-DOMINANT: x_NV +0.2807 (t 18.72, n 306); old twin -0.0019; x_GV +0.1237 **FAIL** | +0.47 pts (needs ≥ -0.37; r 0.113) (reported) | ratio 0.555 (≥ 0.9); top-10 0.493 vs 0.258 (+0.05) **FAIL** | never seen 68.5% vs val 58.1% (−5) PASS; new templates 35.5% |

### P8: the frozen numbers (admitted strata only)

| stratum | admitted | N_new | N_old | p (inject) | old rows per batch | appended per batch | L_old | L_new | added query positions per batch | stratum query mass |
|---|---|---|---|---|---|---|---|---|---|---|
| benign_P0Q1 | yes | 14,863 | 14,863 | 1.0000 | 1.230 | 1.230 | 24.09 | 23.93 | 29.4 | ×1.993 |
| benign_P1Q1 | yes | 42,429 | 42,429 | 1.0000 | 3.511 | 3.511 | 32.27 | 31.81 | 111.7 | ×1.986 |
| phishing_P0Q1 | no | 355 | 2,978 | 0.0000 | 0.560 | 0.000 | 28.55 | 21.72 | 0.0 | — |
| phishing_P1Q1 | no | 563 | 5,152 | 0.0000 | 0.969 | 0.000 | 45.28 | 68.66 | 0.0 | — |

- **Appended rows per batch:** 4.741 (1.85% of 256).
- **Added query positions per batch:** 141.1, against `e12_final`'s 750: query mass **+18.8%**; scored mass +0.98% (C 14397.885); **0.063 of E14's added mass** (2249 per batch).
- **Seeds:** {"blocks": "20261005 + 7919*i + 1000003*(r-1)", "name": 20261006, "phishing_host_pick": 20261009, "inject_draw": 20261007, "inject_fraction": 20261008}.
- **V-batch reference** (`e15_replay.py`): the old rows' batch digest over 20,000 steps (seed 20260803, batch 256, window 64): `adbddc2ae4b85478395de46628cd7b5c1d1699511494ea6c84f836e19cb1d897`; counts {'benign': 2551259, 'phishing': 238707}, dropped 3,851.
- **Code:** the acquisition and gates at `f5508d98232d`.

### 12.4 For gate B

1. **The admitted strata and P8's frozen numbers above:** benign P0Q1 and P1Q1 at
   p = 1.0, with seeds 20261007 and 20261008.
2. **Phishing.** Under the registered rule it is not admitted. The owner asked
   (2026-10-06) whether the query-only mask makes its host shift harmless. That
   question, and the owner's proposed second arm with full supervision, are answered
   in the gate-B report. **Nothing in the design changes unless the owner decides.**

## 13. Gate B (2026-10-06): approval, instrument definitions, and the formal predictions

**Registered before the bundle, the notebooks and any training.**

### 13.1 The owner's decision, verbatim

> "I have reviewed Amendment 3 and your analysis of both proposals.
> Your reasoning is flawless on both counts. I concede the point regarding the
> Phishing QRY shift (learning future campaign vocabularies associated with generic
> hosting will corrupt the generation context) and the Presence bias in the full
> supervision arm (teaching the model that every URL requires a query is a critical
> artifact). You correctly identified the exact traps those proposals would create.
> Therefore, my final Gate B decisions are:
> I approve Amendment 3 and the measured numbers.
> The 918 phishing rows are explicitly REJECTED and excluded.
> The second arm (Full Supervision) is CANCELLED. We will execute the single, clean
> E15-B primary arm exactly as registered (Benign only, QRY-only supervision).
> However, before you build the bundle and the E15B_train Kaggle notebook, I want you
> to state your formal predictions for this run. Based on the 1.0 injection rate
> (+18.8% query mass, +0.98% overall mass) and the strict QRY-only mask, what is your
> exact expectation for the target QRY cells and the collateral PTH/SUB cells compared
> to e12_final and E14? Give me your hypothesis first, then proceed with building the
> bundle."

**So the experiment is one arm, `e15_new`:**
- **Injected data:** benign P0Q1 and P1Q1 only, at p = 1.0;
- **Supervision:** query-only;
- **Decode:** raw + D1 + D1′;
- **Judged against** the frozen `e12_d1p` (§9.8);
- **Phishing:** none.

### 13.2 Instrument definitions for `e15_new` (registered before any data from the arm)

The guards on copying, novelty and the band profile compare generated strings with
"the model's training data". For `e15_new` that data is **the old corpus plus the
57,292 injected rows**. The existing instruments know only the old corpus.
- **Without this definition:**
  - a query the model learned from the new stream and writes back would count as
    *not* copied;
  - a new-stream URL written back whole would count as *novel*;
  - name frequencies would miss the new rows.
- That would understate copying, overstate novelty and mis-band names: an artifact,
  not a property of the model.

**Registered for `e15_new` only** (`e12_d1p` and `e12_final` keep their recorded
numbers):
- **G7, copying (QC1-1's instrument, K12d):**
  - the reference counts are the old training queries plus the stream's query
    strings;
  - the generated rate, the singleton rate and the natural rate (the real held-out
    queries found in training) are all computed against that same reference;
  - the "never OVER" rule and amendment 1's distance rule compare each arm with its
    **own** natural rate.
  - The old-corpus-only reading is reported beside it.
- **G8, novelty:** the known URLs are the bundle's known-URL set plus the stream's
  URLs.
- **K15-3, band profile (M2):** frequency bands and the faithful-sampler null come from
  the model's own training queries (old plus stream).
- **Unchanged:** every gate-v2 cell, every threshold and tolerance, M3 (held-out rows
  only), and every other guard.

### 13.3 Formal predictions [HYPOTHESIS] (replace §10.8 and §9.11)

**Why I expect what follows:**
- **The dose is local and small.**
  - In the two target strata, query supervision doubles (×1.99) with genuinely new
    content.
  - Everywhere else nothing changes: benign page-frame queries (about 74% of benign
    query rows), all phishing, and every non-query field.
  - E14 put ×4 on every query position everywhere: about 16× this design's added mass.
- **The record reads "learning, not data"** (DX1, DX2, DX3).
- **Exposure moved P0Q1 strongly in E14, but never moved P1Q1** (E14, N's A2).
- **The trade-off E14 showed is a budget effect proportional to added gradient.** At
  0.98% added mass it should fall below detection.

**The cells.** The baseline column is what the criteria use. Ranges are about 80%
intervals.

| cell | `e12_final` | **`e12_d1p` (baseline)** | `e14_q` (E14) | **`e15_new`, predicted** | verdict, predicted |
|---|---|---|---|---|---|
| control benign P0Q1 QRY (target) | 0.0621 | **0.0624** | 0.0365 CLEARED | **0.054** (0.047–0.061) | UNDECIDED |
| control benign P1Q1 QRY (target) | 0.0559 | **0.0520** | 0.0579 | **0.049** (0.044–0.056) | UNDECIDED |
| free benign QRY (reference) | 0.0325 | **0.0307** | 0.0309 | 0.029 (0.025–0.034) | CLEARED, a flip possible |
| free phishing QRY (untouched) | 0.1141 | **0.1123** | 0.0830 | 0.112 (0.102–0.122) | CONVICTED |
| control benign P1Q1 SUB | 0.0416 | **0.0434** | 0.0536 | 0.044 (0.038–0.050) | UNDECIDED |
| control benign P1Q1 PTH | 0.0576 | **0.0575** | 0.0680 | 0.058 (0.052–0.064) | UNDECIDED or CONVICTED (noise) |
| control phishing P1Q0 PTH | 0.0428 | **0.0433** | 0.0481 | 0.044 (0.038–0.050) | UNDECIDED |
| free phishing PTH | 0.0376 | **0.0394** | 0.0457 | 0.040 (0.035–0.046) | CLEARED, a flip possible |

- **The target cells move the right way, but by less than the 0.0141 bar.**
  - P0Q1 falls by about 0.008. It is exposure-responsive, but this exposure is half of
    E14's and only local.
  - P1Q1 falls by about 0.003. New content gives it +3.8 points of coverage, but
    coverage has not translated into the detector anywhere so far.
- **K15-1 most likely FAILS.** The chance that P0Q1 falls by more than 0.0141 is about
  20%; for P1Q1 about 5%.
- **The path and subdomain cells move only by retraining noise.** At about 1/16 of E14's
  added mass, E14's costs (+0.010 to +0.012) scale to under 0.001. A shift of up to
  about ±0.008 is plain noise.
  - **G2 passes:** no path, subdomain or label cell rises by more than 0.0141.
  - A verdict may flip on a boundary cell, as before. For a retrained arm that is
    judged by the 0.0141 rule, not by the flip.

**Per-field validation fit, nats per character, against `e12_final`:**

| field | E14 (`e14_q`) | `e15_new`, predicted |
|---|---|---|
| benign QRY | −0.086 | **−0.02 to −0.05**: new content and ×2 exposure land exactly on the domain-frame validation rows |
| phishing QRY | −0.166 | −0.01 to +0.01 (untouched) |
| benign LBL / PTH / SUB / SUF | +0.038 / +0.028 / +0.001 / +0.002 | each within ±0.006 |
| phishing LBL / PTH / SUB | +0.027 / +0.055 / +0.041 | each within ±0.010 |

- **K15-2 passes:** held-out benign query NLL falls, with its interval below 0.
- **The train–validation query gap does not widen** (E14: +0.044). Per-row exposure is
  unchanged, and the extra exposure is on new rows.
- **G3 passes:** no non-query field worse by more than 0.01.

**Everything else:**
- **K15-3 passes:** the rare-name share stays far below the null.
- **G1, G4, G6, G9 and G10 pass.**
- **G5 passes:** 8 of 8 slash rows, by D1′.
- **G7 and G8 pass**, under §13.2's definitions.
- **V-batch:** the old-row digest equals `adbddc2a…`.
- **V-mass:**
  - about 94,820 rows appended over 20,000 steps (±915 at 3 SE);
  - about 2.82M query positions appended.
- **Training time:** about 2% longer than `e12_final`'s.

**The four-way report, predicted:**
- **Content:**
  - the model writes new-stream query strings that appear nowhere in the old corpus,
    about 3–8% of its target-cell queries (uptake well above zero);
  - the target cells' coverage of the validation templates rises by 1–3 points.
- **Exposure:** not separable from content. E14 suggests anything from about −0.002
  (scaled by global mass) to about −0.013 (scaled by local dose) for P0Q1. The truth is
  likely between.
- **Trade-off:** none detectable at this dose. That is reported as "no cost at about
  1/16 of E14's dose", not as "no trade-off".

**Overall: most likely NOT ADOPTED, on K15-1.** Held-out fit would improve while the
detector moves less than the bar. **If it is adopted, it will be through P0Q1.**

## Status

    E15  registered 2026-10-05 (this document and E15_DESIGN.md), before any
         E15 code, fit, training, generation or data acquisition. Awaiting the
         owner's gate 0 (section 7). Nothing has been executed.

    E15  Amendment 1, 2026-10-05 (section 9), by the owner's decision: E15-A
         removed; E15-B is the experiment, re-specified in full (corpus2's
         procedure, the replacement rule, gates P1-P8, criteria K15-1..G10,
         the four-way report). Also corrects the original text's description
         of the benign corpus (E0's 16 crawls -> corpus2's 8). Registered
         before any E15 code, data acquisition, training or generation.
         Awaiting the owner's gate A; nothing has been executed.
    E15  Amendment 1 not approved by the owner ("not approved wait for
         instruction").
    E15  Amendment 2, 2026-10-05 (section 10), by the owner's decision:
         matched exposure revoked; the replacement rule rejected and removed;
         new rows injected alongside the old ones at the old rows' per-row
         rate (query-only targets, weight 1.0; old rows unchanged). The
         numbers amendment is renamed amendment 3. Registered before any
         E15 code, data acquisition, training or generation. Awaiting the
         owner's gate A; nothing has been executed.
    E15  Gate A APPROVED by the owner 2026-10-05 (section 11, verbatim): full
         benign doubling with the 8 crawls in parallel; phishing fetched
         and injected at whatever volume exists (p = N_new/N_old); the
         control arm excluded. Section 11 also corrects the phishing
         leakage key (host only) and specifies the seeds. Registered before
         any data acquisition.
    E15  Amendment 3, 2026-10-06 (section 12): acquisition complete and gates
         run. Benign P0Q1 and P1Q1 ADMITTED (57,292 rows, every gate PASS);
         phishing NOT admitted (P1 SHIFT-DOMINANT, P4 and P5 FAIL). Dose: 4.74
         rows and 141 query positions per batch (query +18.8%, scored +0.98%).
         V-batch reference adbddc2a. Recorded before any training. Awaiting
         the owner's gate B.
    E15  Gate B APPROVED by the owner 2026-10-06 (section 13, verbatim): Amendment
         3's numbers; phishing rejected; the full-supervision arm cancelled.
         Section 13 registers, before the bundle and any training: the
         training-data reference (old corpus + the stream) for e15_new's
         copying, novelty and band-profile guards, and the formal predictions
         (most likely NOT ADOPTED on K15-1; no detectable PTH/SUB cost).
    E15  step 3 (bundle and notebooks), 2026-10-06: data/kaggle/uie-e15-code.zip from
         commit e48a078, sha256 637aca12635316a5f7fbefdcce81a4b5ac93e804b20e3a727d4192da41d8515e
         (53.1 MB; 39 code files, 3 tables, e12_final.pt, qry_extra.jsonl = 57,292
         rows, sha256 637accab5622f14aa432849fe9f4bc6ede7b2d18188bc3ef141d9d3c71e69e91;
         nothing gzipped). Smoke tests PASS: on the full encoded corpus the
         injector counts Amendment 3's strata exactly (p = 1.0), the old rows'
         batch digest over 400 batches equals the replay's, and it appends
         4.80 rows / 144 query positions per batch (P8: 4.74 / 141); a real
         train() with injected rows; the D1 + D1' decode. Both notebooks run
         end to end on the laptop CPU from the unzipped bundle (Kaggle's paths
         patched, tiny jobs; the start check bypassed for the simulation only).
         Run sheet: research/e15/E15_KAGGLE.md. Awaiting the owner's Kaggle run.
    E15  READ 2026-10-06 (research/e15/E15B_RESULTS.md; data/e15/compare_report.json).
         Validity PASS (VERIFY_TRAIN, VERIFY_GENERATE 957/957, V4 0.0). K15-1 PASS
         (benign P0Q1 QRY 0.0624 -> 0.0410), K15-2 PASS (-0.0373), K15-3 PASS,
         G1-G4 and G6-G10 PASS; G5 FAIL (benign control P0Q1 slash 0.99859 vs
         0.99551, T 0.00244). NOT ADOPTED under the registered rules; the baseline
         stays e12_d1p. Uptake at about the chance rate: the gain is not attributed
         to new content (section 10.6). E3 NOT MET. Any waiver is the owner's.
    E15  OWNER DECISION 2026-10-06 (research/native/DECISION_2026-10-06.md,
         verbatim there): G5 WAIVED for this adoption only; e15_new ADOPTED as the
         official baseline, replacing e12_d1p. G5's rule and T are unchanged for
         later experiments, and E3 still counts the benign control P0Q1 slash row
         as missed. E16 evaluated, not registered: research/e16/E16_DESIGN.md.
