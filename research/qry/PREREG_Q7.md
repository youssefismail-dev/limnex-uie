# Q7 — pre-registration: why does the model underproduce a tail it saw?

**Registered:** 2026-09-22, before any measurement in this file was run.
**Frozen:** everything under `uie/`, `best_e5.pt`, the decoder, the sampling
policy, the gate, the 0.05 bar, training data and weights. **Read-only
diagnostics only. No repair is implemented in this phase.**
**Scope: BENIGN.** Phishing is measured only where a comparison is structurally
required, and is never generalised to.

## The gap this must explain

Q6 decomposed value-length p99 on `name_train`:

    corpus 108  --budget-->  87  --model-->  56  --decoder-->  40
                  -21 (31%)      -31 (46%)      -16 (24%)

The **46% middle term is the target**. The model saw p99 = 87 and values to 218
characters, and emits 56. Q5 excluded a tail-poor corpus; Q6 excluded truncation
and quantified deletion at ~31%.

Every hypothesis below is treated as falsifiable. None is assumed.

## Configuration facts read from the checkpoint (not hypotheses)

    steps 20,000 x batch 256   = 5,120,000 example-exposures
    d_model 384, 8 layers, 8 heads, d_ff 1536, max_len 256, vocab 101
    class_alpha 1.0            -> exposure-neutral across classes
    within-class sampling      -> shuffled cursor, WITHOUT replacement per epoch

Sampling without replacement matters: it means no row is seen zero times, which
rules out "the tail was never drawn" before it is proposed.

---

## H1 — LEARNING vs SAMPLING (the decisive split)

**Hypothesis.** The model did not learn the value-length tail: conditioned on a
real long-value prefix, it assigns too much probability to terminating the value.

**Method — the termination hazard**, reusing Q2's teacher-forced machinery
unchanged. Inside a query value, the next token either continues the value or
ends it (`&`, `<FRG>`, `<eos>`). Define

    h(i) = P(value ends at position i | value reached position i)

Empirical `h` comes from counting the corpus; model `h` comes from teacher
forcing on those same rows. **No sampling anywhere**, so decoder and prefix drift
are both excluded by construction.

**Prediction if H1 true.** Model `h(i)` exceeds empirical `h(i)` at large i —
the model is too eager to stop.
**Prediction if H1 false.** Model `h(i)` tracks empirical `h(i)` within noise at
large i, which would place the deficit downstream in sampling or prefix drift.

**Metric.** Hazard ratio `h_model / h_empirical` per position bucket, plus the
model-implied mean value length obtained by integrating the model hazard.
**Test.** Bootstrap 95% CI over rows, 1,000 resamples. A bucket counts as
divergent only if its CI excludes 1.0.
**Minimum sample.** >= 200 rows surviving to a bucket, else the bucket is
reported UNMEASURED rather than estimated.

**Confounders.** (a) Absolute sequence position is correlated with value
position — separated by H4. (b) Teacher forcing conditions on real prefixes, so
this measures the conditional only; that is deliberate and is what isolates
learning from drift.

---

## H2 — GENERALISATION vs MEMORISATION

**Hypothesis.** The model fits the tail it was trained on but does not
generalise it.

**Method.** Run H1's hazard on `name_train` and `name_val` **separately**.
**Support.** Model hazard tracks empirical on TRAIN and diverges on VAL.
**Refute.** Divergence of similar size on both -> a learning/capacity failure,
not a generalisation failure.
**Metric/test.** Same as H1; the comparison is the train-vs-val difference in
hazard ratio, bootstrap CI on the difference.
**Confounder.** Train and val differ in composition; Q5 already showed they are
distributionally consistent for query content, which is what licenses this
comparison.

---

## H3 — EFFECTIVE EXPOSURE / LEARNABILITY THRESHOLD

**Hypothesis.** Rare query structures receive too few gradient exposures to be
learnable, and the model reproduces only what it saw often enough.

**Method.** Compute the training frequency spectrum of parameter names, then ask
which names the model actually emits as a function of their training count.
**Support.** A threshold: emitted names concentrate among high-count names and
fall away below some count, with hapax (count 1) names essentially never
produced.
**Refute.** The model emits rare names at roughly the rate their training
frequency implies -> exposure is not the binding constraint.
**Metric.** Reproduction rate by training-count decile; hapax share of the real
name vocabulary; exposures-per-row derived from steps x batch / pool size.
**Test.** Descriptive with bootstrap CIs; no threshold is fitted, the curve is
reported as measured.
**Confounder.** Generated samples are small (758 parameters), so absence of a
rare name may be sampling. Controlled by comparing against a **frequency-matched
random draw of the same size from the training distribution** — the null of "a
faithful sampler of this size".

---

## H4 — POSITION / CONTEXT EFFECTS

**Hypothesis.** The deficit is driven by absolute position in the sequence
rather than by value length itself — late positions are modelled worse.

**Method.** Stratify H1's hazard by absolute token index at which the value
starts, holding within-value position fixed.
**Support.** Hazard ratio degrades with absolute position at matched within-value
position.
**Refute.** Hazard ratio depends on within-value position and not on absolute
position -> a length/field effect, not a context effect.
**Metric/test.** Hazard ratio by (absolute-position bucket x within-value
bucket); bootstrap CI.
**Confounder.** Long values tend to occur in long URLs, so the two are
correlated; the stratification is exactly what breaks that, and cells with
< 200 rows are reported UNMEASURED.

---

## H5 — QRY-SPECIFIC vs GENERAL LONG-FIELD FAILURE

**Hypothesis.** The tail deficit is not specific to queries but is a general
failure to sustain long fields.

**Method.** Run the identical hazard analysis on the **path** field.
**Support.** PTH shows a hazard divergence of similar shape and magnitude.
**Refute.** PTH tracks empirical while QRY diverges -> query-specific.
**Metric/test.** Same as H1, PTH vs QRY.
**Note.** This does **not** reopen PTH attribution, which stays closed. It uses
PTH only as a control field for a query question, and no PTH claim is made.

---

## H6 — OBJECTIVE / FIELD-SCOPED MASK VISIBILITY

**Hypothesis.** Query content receives systematically less effective loss weight
than other fields, so the objective under-trains it.

**Method.** **Read-only inspection of `field_mask` output** on real encoded rows:
count the fraction of total unmasked target positions that fall inside QRY,
versus QRY's share of characters in the corpus.
**Support.** QRY's share of trained positions is materially below its share of
characters.
**Refute.** The shares match -> the objective is not down-weighting queries.
**Metric.** Trained-position share per field, both corpora.
**Test.** Descriptive; the quantity is a deterministic count, not an estimate.
**Hard constraint.** This is diagnosis only. **S4/S4b stays closed and no
weighting is implemented or proposed as a consequence of this measurement
alone.**

---

## H7 — CAPACITY / REPRESENTATION

**Hypothesis.** The model cannot represent the required conditional at all.

**Status: NOT DIRECTLY TESTABLE READ-ONLY.** Capacity claims need a training
run, which is forbidden here. What *can* be done is to look for the signature
capacity failure leaves: if the model fits TRAIN hazard well and VAL poorly
(H2), capacity is not binding; if it fits neither, capacity becomes plausible
but remains unconfirmed. **It will be reported as PLAUSIBLE-UNMEASURED unless H2
produces the signature, and "bigger model" is not assumed to be the answer.**

---

## Decision rule for the causal map

Each mechanism is classified as:

- **CONFIRMED CONTRIBUTOR** — a read-only measurement with a stated CI supports
  it and the prediction that would have refuted it did not occur.
- **PLAUSIBLE, insufficient evidence** — consistent with observation but the
  discriminating measurement was not possible read-only.
- **REFUTED / NOT SUPPORTED** — the refuting prediction occurred.
- **UNMEASURED** — not tested, or sample below the stated minimum.

Contributions are quantified **only where a measurement supports a number**. A
cause that cannot be cleanly quantified is reported as such rather than given an
invented percentage.

## Status

    H1 H2 H4 H5   registered, one measurement (hazard ladder)
    H3            registered
    H6            registered
    H7            registered as not-read-only-testable
