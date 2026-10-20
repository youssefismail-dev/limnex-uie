# N2a — does this architecture learn the query tail when it gets the exposure? Pre-registration

**Registered:** 2026-09-26, before any N2a code, training, generation or reference
measurement.
- **Plan:** `NATIVE_REPAIR_PLAN.md` revision 2, §6 N2a, approved in direction and
  order by the owner.
- **Status: [DIAGNOSTIC].** N2a repairs nothing, ships nothing, and generates no
  data for any use outside this measurement.
- **Frozen:**
  - `data/best_e7.pt`, `data/bundle2`;
  - gate v2;
  - every instrument reused below: Q7's hazard, Q8/R2's band profile, the novelty
    code;
  - every published result.

## 1. Question

E7's queries are compressed:
- the model ends values too early (the termination hazard runs above the corpus's,
  on training rows too, Q7 on E5);
- rare parameter names collapse (R2's band profile);
- about 80% of the benign tell is unexplained (C3-B).

The candidate limits are U1 in the plan: allocation inside the training mix, the
quantity of exposure, capacity, character-level tokenization of long values, and
extreme-tail sparsity. N2a asks which of these the evidence supports. It
separates what it can, and says plainly what it cannot.

## 2. Arms

Each arm is **E7's recipe on query-bearing rows only**: E7's tokenizer (fitted on
all training rows, so the vocabulary is E7's 107), dropout 0.5, 14.34M parameters,
batch 256, lr 3e-4, seed 20260803, max_len 256.

**Exposure.** E7 drew each training row 1.839 times on average (5,120,000 draws
over 2,783,504 kept rows). Training holds 241,805 query-bearing rows [CONFIRMED,
`bundle2`].

| arm | query-row presentations | steps (batch 256) | warm-up (E7's 2.5%) |
|---|---|---|---|
| **A1** | E7's own, ≈ 1.839 × the kept query rows | ≈ 1,700 | ≈ 43 |
| **A2** | four times A1's | ≈ 6,800 | ≈ 170 |

The exact step counts are fixed from the kept-row count the run reports, before
any reading.

**Readings use `final.pt`.** A1 and A2 are budgets of exposure, not models chosen
for validation loss. `best.pt` is reported alongside.

**Stated confound [DECISION].** A1 gives the query rows E7's exposure in about
1,700 optimiser steps instead of 20,000, with a cosine schedule over its own
length (R2's caveat, again).
- If A1 closes a gap, that is strong evidence: less optimisation overall, the same
  query exposure.
- If A1 fails, it is ambiguous between allocation and under-optimisation. A2 then
  carries the reading.

**Implementation, flags default off:** `kaggle_train.run(query_rows_only=True,
warmup=…)`. Filtering happens after the tokenizer is fitted on every row, so the
vocabulary cannot change. Tests prove both flags off leave training byte-identical.

## 3. Instruments

All are teacher-forced or measured on files. Every model is read with its
presence controls **hidden** (`?`), E7's free mode.

| code | what | where | status |
|---|---|---|---|
| **M1 LENGTH** | termination hazard ratio model / corpus inside query values, per value-position bucket (Q7's buckets and definition), on TRAIN and on VALIDATION query rows; bootstrap 95% CI over rows, 1,000 resamples; a bucket needs ≥ 200 rows reaching it | laptop CPU | Q7's instrument, applied with hidden controls |
| **M2 FREQUENCY** | share of generated parameter names in corpus-frequency bands 1, 2–4 and 5–19, against the faithful-sampler null (R2's band profile) | laptop, from generated queries | R2's instrument |
| **M3 FIT** | teacher-forced QRY negative log-likelihood per character, by name-frequency band, TRAIN against VALIDATION | laptop CPU | Q2's machinery |
| **M4 COPYING** | share of generated query strings found verbatim among training queries | laptop | DC1's code |

**Row sets:**
- **validation:** every query-bearing validation row (benign `name_val` 7,253 and
  `structure_val`; phishing likewise);
- **train:** a fixed random sample of 20,000 query-bearing training rows per
  class, seed 20261001;
- the same rows for every model.

**Generated queries:**
- **Arms:** 20,000 per arm per class, free mode, resting decode, on the Kaggle GPU
  in the training notebook. Trained on query rows only, an arm's free mode always
  writes a query.
- **E7:** every query in its existing raw free-mode generations (DC1's base arm and
  HC1 C2's canonical files), about 8,600 benign and 2,700 phishing, counted
  exactly in N2a-0.
  - There is no top-up. Only about 3% of E7's free URLs carry a query, so 20,000
    phishing queries would take about a million URLs.
  - E7's `<Q:1>` mode is not a substitute: C3 showed its benign queries differ
    from free mode's.
  - N2a-0's power check decides whether this sample suffices.

## 4. N2a-0 — the E7 reference and the power check (read-only, before any arm)

On E7:
- M1–M4, as above;
- each metric's **minimum detectable change**: 2 × the bootstrap CI half-width of
  its estimate at the registered sizes.

**The gap and the power rule:**
- A metric's **gap** is |E7 − target|: the hazard ratio against 1.0, the band
  share against the null.
- A metric whose gap is **less than twice its minimum detectable change is void**
  for the decision, and is reported, not read.

This is recorded in this file's Status before any arm is trained.

## 5. Readings and the decision table (fixed now)

**Definitions:**
- **Sub-deficits:**
  - **LENGTH:** M1 in the pooled bucket 25–100;
  - **FREQUENCY:** M2's bands 1–19 pooled.
- **Closure** = 1 − gap(arm) / gap(E7).
- **"Closes"** means closure ≥ 0.5 **and** the arm's 95% CI excludes E7's value.
- **"Little"** means closure < 0.25.

| reading | condition | licenses |
|---|---|---|
| **ALLOCATION** | A1 closes both sub-deficits on validation | N2b: allocation inside the mixture |
| **EXPOSURE** | not ALLOCATION, and A2 closes both on validation | N2b: exposure (a query-content loss weight) |
| **SPARSITY** | A2 closes both on train (M1 train, M3 train) but little on validation, **and** M4 rises ≥ 0.5 points with its CI excluding 0 | no training fix inside the frozen corpus; the owner decides |
| **LENGTH-SPECIFIC** | on train, A2 closes FREQUENCY but LENGTH is little | N2c-T (a representation of long values); R1 as the known partial |
| **FREQUENCY-SPECIFIC** | on train, A2 closes LENGTH but FREQUENCY is little | N2c-C (capacity) — **R3's registered unlock condition** |
| **GENERAL** | on train, both little at A2 | N2c-C |
| **MIXED** | anything else | no content repair licensed; reported |

**Copy guard.** For an arm whose M4 exceeds E7's by ≥ 0.5 points with a CI
excluding 0, validation closures are void. They then count only as evidence for
SPARSITY.

**Per class.** The causal map (Q1–Q8) was built on benign, so the table is read on
benign. Phishing is read the same way and reported; it licenses only if it agrees.

**Not claimed:**
- N2a cannot prove tokenization or capacity; N2c exists for that, if licensed;
- N2a says nothing about any gate cell;
- A1 and A2 are diagnostic models and are never scored by the gate as candidates.

## 6. Kill and validity

| code | trigger | consequence |
|---|---|---|
| V | a checkpoint's hash, a tokenizer (not E7's), or a registered row set differs from its record | fixed under an amendment; nothing read |
| P | the power rule voids both sub-deficits | N2a cannot decide; reported, and the owner decides whether larger row sets justify a new registration |
| K | an arm's training diverges (non-finite loss) or its tokenizer differs from E7's | that arm is void |

## 7. Order

1. This registration.
2. N2a-0 on the laptop: E7's reference and the power check, recorded below.
3. Code and tests (flags off = byte-identical).
4. One Kaggle notebook: A1, A2 and their generated queries.
5. Laptop readings.
6. `N2A_RESULTS.md`.

Its reading decides N2b's registration.

## Status

    N2a  registered 2026-09-26
    N2a  run 2026-09-27 (6fdb067); verified; reading SPARSITY (benign), phishing UNAVAILABLE
         -> N2A_RESULTS.md. No training repair of query content licensed.

**Amendment 1, 2026-09-26 — definitions, before N2a-0 or any measurement.**
- **M1 and M3 rows are domain-frame only:** `name_train`/`name_val` (benign),
  `phishing_name_*` (phishing). That is Q7's population and the gate's, and free
  mode generates in the domain frame. This replaces section 3's inclusion of
  `structure_val`.
- **M2 is defined for E7's training regime.** R2 counted name frequencies on
  `structure_train` only, because in E5's masked regime only those rows taught
  query content (H6). E7 names the frame and trains every field on every row, so:
  - **bands** are frequencies over every query-bearing training row of the class
    (both frames);
  - **the null** is faithful draws from the class's domain-frame training queries
    (`name_train`), the distribution free mode is meant to reproduce.
- **Why.** R2's null targeted a different population. With it, the gap would be
  measured against the wrong reference. Nothing has been measured under either
  definition.

**N2a-0, benign (E7, controls hidden)** [CONFIRMED]: `n2a_reference.py` (15d2a13),
data in `data/native/n2a_reference_benign.json`.

| metric | train | validation | power rule |
|---|---|---|---|
| M1 LENGTH, pooled 25–100 | 1.088 [1.061, 1.118], 4,519 values | 1.107 [1.058, 1.155], 1,666 values | **void** on both |
| M1 tail bucket 50–100 (reported) | 1.259 [1.188, 1.335], 1,154 values | 1.276 [1.161, 1.392], 406 values | — |
| M2 FREQUENCY (rare-name share) | 0.0497 [0.0457, 0.0533] against null 0.2196, 12,382 names | — | measurable |
| M4 copying | 27.33% of 8,616 queries (DC1's base arm: 27.2%) | — | — |

**Amendment 2, 2026-09-26 — the primary readout. After N2a-0 (E7 only), before
any arm is trained, generated or measured.**

**Two findings force it:**
1. **LENGTH is void by the registered rule.** The length tail is still under-fit
   where values are long (50–100: about 1.27, CI excluding 1). But too few values
   reach it for this design to measure a closure. Validation cannot grow; the
   split is fixed. So N2a cannot license N2c-T.
2. **FREQUENCY is measurable, but for benign it is not the axis that carries the
   gate's tell.** C3-B found benign's QRY tell survives `skeleton`, which erases
   which names appear, keeping 95.5% of it in free mode. A decision on FREQUENCY
   alone would be clean and say little about E3's benign QRY cell.

**The change.** The decision moves onto **the gate's own QRY cell**, via
`n2a_qry.py`: `cr1_score`'s exact QRY call, G = 5, D = 10.
- **Q-VAL:** each arm's free-mode queries against real VALIDATION queries (the
  gate's pools). This measures generalisation. E7's reference is DC1's base-arm
  reading: benign +0.0957, phishing +0.1141, at n 1,200 / 356.
- **Q-TRAIN:** the same queries against the domain frame's TRAINING queries, at
  the same n. This measures whether the model "fits what it saw". E7's reference
  is measured below, before any arm.
- **Validity:** before anything else, the QRY-only scorer must reproduce DC1's
  recorded E7 Q-VAL cell exactly, in both classes.
- **Arms' generation:** 5 generations × 1,500 URLs per class per arm, free mode,
  resting decode. Seeds 20261001 + (100·s + g)·10⁹ with s = 18 (benign) and
  19 (phishing), blocks no earlier experiment used. A query-only model writes a
  query almost always, so QRY reaches its full n every generation.

**Readings:**
- **Closure** = 1 − x(arm) / x(E7) on each of Q-VAL and Q-TRAIN, where x is the
  excess over the floor and the ideal is 0.
- **"Closes"** means closure ≥ 0.5 **and** the arm's 95% CI excludes E7's x.
- **"Little"** means closure < 0.25.
- **Power:** the minimum detectable change is 2 × the 95% half-width of E7's
  estimate. A reading whose x(E7) is below twice that is void.

| reading | condition | licenses |
|---|---|---|
| **ALLOCATION** | A1 closes Q-VAL | N2b: allocation inside the mixture |
| **EXPOSURE** | not ALLOCATION, and A2 closes Q-VAL | N2b: exposure (a query-content loss weight) |
| **SPARSITY** | A2 closes Q-TRAIN but little on Q-VAL; **or** copying rises ≥ 0.5 points with its CI excluding 0 | no training fix inside the frozen corpus; the owner decides |
| **CAPACITY-OR-REPRESENTATION** | A2 is little on Q-TRAIN, and copying has not risen | N2c. N2a cannot tell capacity (R3's unlock) from representation |
| **MIXED** | anything else | no content repair licensed; reported |

**Unchanged:**
- the arms A1 and A2;
- the exposure definitions and the stated step-count confound;
- the 0.5 / 0.25 thresholds and the power rule;
- the copy guard;
- benign decides, and phishing licenses only if it agrees.

M1–M3 continue as descriptive mechanism evidence, not as decision inputs.

**Why this is an amendment and not a rescue:**
- N2a-0 is the registered pre-arm check, and no arm exists.
- No threshold moves.
- The decision moves onto the quantity E3 is judged on.

**N2a-0, phishing (E7, controls hidden)** [CONFIRMED]: recorded after Amendment 2,
which it does not change. Data in `data/native/n2a_reference_phishing.json`.

| metric | train | validation | power rule |
|---|---|---|---|
| M1 LENGTH, pooled 25–100 | 1.321 [1.265, 1.378], 3,086 values | 1.336 [1.183, 1.508], 370 values | train measurable; validation **void** |
| M1 tail bucket 50–100 (reported) | 1.836 [1.697, 1.976], 1,146 values | 1.761 [1.438, 2.172], 141 values | — |
| M2 FREQUENCY (rare-name share) | 0.0967 [0.0874, 0.1066] against null 0.3775, 3,351 names | — | measurable |
| M4 copying | 26.42% of 2,774 queries | — | — |

- **Phishing's length tail is far more under-fit than benign's.** At value
  positions 50–100, E7 stops about 1.8× as often as the corpus.
- This agrees with C3-B, where phishing's tell includes the long-value tail.
- It stays descriptive: under Amendment 2 the decision is the gate's QRY cell.

**Amendment 2's references, recorded before any arm** [CONFIRMED]: `n2a_qry.py
--check-e7`, data in `data/native/n2a_qry_e7_{benign,phishing}.json`.
- **Validity: exact in both classes.** On E7's raw free files, the QRY-only scorer
  reproduces DC1's recorded cells: +0.0957 and +0.1141.
- **The power rule** uses a half-width of 2.776 × SE (df 4) and a minimum
  detectable change of 2 × that half-width. A reading is usable only if x(E7) is
  at least twice the minimum detectable change.

| E7 reference | x | SE | usable if x ≥ | reading |
|---|---|---|---|---|
| benign Q-VAL | +0.0957 | 0.0038 | 0.042 | measurable |
| benign Q-TRAIN | +0.0862 | 0.0034 | 0.038 | measurable |
| phishing Q-VAL | +0.1141 | 0.0079 | 0.088 | measurable |
| phishing Q-TRAIN | +0.1170 | 0.0114 | 0.127 | **void** |

- **Benign, which decides, is fully measurable.** For phishing only the
  ALLOCATION and EXPOSURE readings can agree or disagree; its SPARSITY and
  CAPACITY readings are unavailable.
- **E7's own reading** [CONFIRMED]: its queries are about as detectable against its
  training queries as against held-out ones (benign +0.0862 against +0.0957). E7
  has not fit its training query distribution in the first place. This agrees
  with Q7's "under-fit on training data too". It is not a failure to generalise,
  which is what makes N2a's question well posed.

**Amendment 3, 2026-09-26 — exposure made exact and the implementation
simplified. Before any N2a code or arm.**

**1. Exposure is per class.** Section 2's "1.839 draws per row" is E7's average
across classes. E7's sampler is different:
- it weights CLASSES by name counts (906,214 : 192,652, `class_alpha` 1.0);
- it samples rows of both frames within each class.

So E7 gave benign rows **1.65888** draws each and phishing rows **3.76842**
(5,120,000 × 0.824681 / 2,545,305 and 5,120,000 × 0.175319 / 238,199) [CONFIRMED,
arithmetic on E7's recorded counts]. The corrected definitions:
- **A1** reproduces each class's exposure on its own query rows. The class weights
  are proportional to exposure × kept query rows, and steps = (1.65888 × kept
  benign query rows + 3.76842 × kept phishing query rows) / 256.
- **A2** is four times A1's steps.
- **Warm-up** is 2.5% of steps (E7: 500 / 20,000).

**2. The implementation leaves `uie/` untouched.**
- `research/native/n2a_train.py` loads **E7's own tokenizer and model
  configuration from `best_e7.pt`**, so the vocabulary is identical by
  construction.
- It filters to query-bearing rows, encodes validation with the controls hidden
  (as E7 was validated), and trains with `uie.train.train` unchanged: dropout 0.5
  and seed 20260803.
- No trainer flag is added, so the byte-identity burden section 2 carried for
  `uie/` disappears.

Nothing else changes.

**Amendment 4, 2026-09-26 — the exact schedule and the table's open wording.
After counting rows, before any arm is trained, generated or read.**

**1. The schedule, fixed from the counts** [CONFIRMED, `n2a_train.py --dry-run` on
the data bundle, `data/native/n2a_dryrun.log`]:
- query-bearing training rows: 241,805 (benign 227,823, phishing 13,982), the
  registered count reproduced exactly;
- kept under max_len 256: benign 223,342, phishing 13,270 (5,193 dropped);
- validation query rows: 30,570 benign, 1,903 phishing (822 dropped).

| arm | steps | warm-up | evaluation every |
|---|---|---|---|
| **A1** | 1,643 | 41 | 82 |
| **A2** | 6,572 | 164 | 329 |

- Section 2's "≈ 1,700 / ≈ 6,800" used the cross-class average exposure.
  Amendment 3's per-class exposure gives these.
- The notebook stops if Kaggle's counts differ (V).

**2. Two settings the registration left implicit:**
- **Evaluation every 5% of steps,** E7's share (1,000 of 20,000). It only decides
  which checkpoint becomes `best.pt`. Evaluation uses its own seeded generator and
  no dropout, so it draws no training randomness [CONFIRMED, code inspection of
  `uie.train.evaluate`]. Readings use `final.pt`.
- **Initialisation is seeded** with 20260803 before the model is built. E7's came
  from the process's default generator and cannot be recovered. Every other
  training setting is copied from E7's checkpoint, not restated.

**3. The decision table's open wording, settled:**
- **"The arm's 95% CI excludes E7's x":** the arm's CI is x ± 2.776 × SE, the
  half-width the power rule uses.
- **"Copying rises"** uses DC1's `prop_worsening`: the arm's M4 minus E7's is at
  least 0.5 points, and its normal 95% CI excludes 0. E7's M4 comes from N2a-0:
  27.33% of 8,616 benign queries, 26.42% of 2,774 phishing.
  - In the SPARSITY row it means a rise in **either** arm.
  - "Copying has not risen" in the CAPACITY row means in **neither**.
- **Order:** rows are read top to bottom; the first that holds is the reading.
  Every row that holds is reported.
- **Unavailable:**
  - SPARSITY and CAPACITY-OR-REPRESENTATION need a usable Q-TRAIN reference. For
    phishing it is void, so both rows are unavailable there, as recorded with the
    power table.
  - A phishing copying rise is still reported, and still voids that arm's
    validation closure.
  - When no available row holds and a reference is void, the reading is
    UNAVAILABLE, not MIXED.
- **Phishing:** benign's reading is the licence. Phishing's is reported as AGREES,
  DISAGREES or UNAVAILABLE. Only AGREES extends the reading to phishing.

**4. Code** (`n2a_train.py` trains and generates; `n2a_decide.py` verifies, then
applies the table):
- Tests pin the arithmetic, the sampler's delivered exposure, the row filter, E7's
  recipe end to end, and every row of the table: the guard, the power rule and the
  agreement, all on synthetic readings against E7's recorded references.

Nothing else changes: the arms, the exposure, the thresholds, the power rule, the
copy guard, and benign deciding.
