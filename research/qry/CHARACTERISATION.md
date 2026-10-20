# QRY defect characterisation — findings

**Phase objective:** determine whether the two QRY defects — low emission rate
and distinguishable content — share a mechanism or are separable. **Not** to fix
either. No generator, checkpoint, decoder or gate code was modified; the 0.05 bar
is untouched.

**Experiments:** Q1 (decoder ablation), Q2/Q2b (teacher-forced probes), Q3
(content decomposition, see Q3_RESULTS.md), Q4 (selection test). All registered
experiments are now complete for the benign class; Q3 has not been run on
phishing.

---

## 1. Defect decomposition

The emission chain, measured end to end. Each row is a distinct measurement, not
a model fit.

| stage | benign | phishing | what it measures |
|---|---:|---:|---|
| corpus (training) | 6.361% | 4.294% | the target |
| corpus (validation) | 6.403% | 4.435% | agrees with training to 0.15 pts |
| **teacher-forced on real prefixes** | **5.560%** | **5.148%** | the model's DECISION |
| **model's own marginal** (ancestral) | **4.22%** | **1.95%** | decision + prefix drift |
| **production** (typical) | **2.30%** | **1.20%** | + decoder truncation |

Three separable terms, and they do **not** rank the same way in the two classes.

### Axis A — emission

| term | benign | phishing |
|---|---|---|
| conditional modelling | **no defect detected** (5.56% vs 6.40% target) | **no defect** — slightly OVER-assigns (5.148% vs 4.435%) |
| prefix drift | present, and **entirely structural** | present, and **mostly probabilistic** |
| decoder truncation | retains `<QRY>` at **3.11%** of eligible positions | retains at **4.62%** |

The decisive measurement for the drift term is p(QRY) at the **attachment point** —
the single unambiguous position per URL where its path ends and a query would
naturally attach:

| | benign real | benign generated | ratio | phishing real | phishing generated | ratio |
|---|---:|---:|---:|---:|---:|---:|
| p(QRY) mean | 0.17652 | 0.17812 | **1.01** | 0.13798 | 0.04431 | **0.32** |
| p(QRY) median | 0.01778 | 0.01720 | 0.97 | 0.03914 | 0.00966 | 0.25 |
| eligible positions/URL | 22.38 | 14.39 | 0.64 | 6.67 | 4.46 | 0.67 |
| pathless URLs | 27.25% | 42.42% | — | 62.58% | 68.67% | — |

**In benign the per-position query behaviour is indistinguishable from real.** The
query field is not independently defective there at all: the model has fewer
opportunities (0.64x the path steps, 1.56x the pathless URLs) and at each
opportunity behaves correctly. **The benign query deficit IS the benign path
deficit.**

**In phishing it is not.** Step count falls by a similar 0.67x, but the
per-position probability collapses to 0.32x, which structure does not explain.

### Axis B — content

**Q3 (full detail in Q3_RESULTS.md): the tell is SHAPE, not vocabulary.**
`skeleton` — every letter to `a`, every digit to `0`, punctuation and lengths
intact — retains the entire excess in both arms (R_T 1.11 t=+0.41; 1.06
t=+0.23). All lexical content can be destroyed with no loss, so parameter count,
separator layout, length profiles and character-class arrangement are
**sufficient** to separate generated queries from real ones.

No other component carries a detectable share: ordering, percent-encoding
identity, value length, digit identity, letter identity and value composition are
null in both arms. `count_only` was VOID degenerate and is not counted.

Descriptively the shape gap is unmistakable and the vocabulary gap is not:
value-length p90 **25 real against 15 generated** with an identical median of 4
(a missing long tail, not wrong short values), parameter count 1.79 -> 1.59,
name length 6.73 -> 5.31, percent-encoded queries 9.89% -> 3.15%, values 42.8% ->
55.6% purely numeric. The commonest parameter names coincide across sides.

Parameter-name identity plausibly contributes ~30%, but **this is not
established**: the transform that appeared to show it (`mask_names`, t=−10.80)
was confounded with intra-name punctuation, and the clean replacement
(`mask_name_chars`) gives the same point estimate at t=−1.91 / −1.49. Three
split seeds give an SE two degrees of freedom and cannot resolve it.

Q4, both arms at identical n=476 against the same real comparator with the floor
recomputed per split:

| arm | draws | mean excess | vs bar |
|---|---|---:|---:|
| T `typical` (production, truncated) | +0.1399, +0.1180, +0.1395 | **+0.1325** | 2.6x |
| A `temperature` (untruncated) | +0.1388, +0.1018, +0.0967 | **+0.1124** | 2.2x |

Paired differences −0.0011, −0.0162, −0.0428; mean **−0.0200**, SE 0.0122,
**t = −1.64 on 2 df** (|t| > 4.30 needed for p<0.05).

---

## 2. Is the content defect a selection artifact? NO.

Q1 established the survivorship condition H4 requires: production emits 2.38% of
queries out of a 4.22% model marginal, so the queries that reach output are the
subsample that survived truncation. H4 said that selection is what makes them
distinguishable.

**Removing truncation entirely reduces the content tell by 15.1%, and that
reduction is not statistically distinguishable from zero.** Arm A remains at
+0.1124 — **2.2x the bar**. Had selection been the cause, the excess should have
collapsed toward the floor; it did not move materially.

> **H4 (shared mechanism) is REFUTED. H5 (separable) is supported.**
> Emission and content are **separate defects with separate causes.** Fixing the
> emission rate — by any means, including replacing the decoder outright — would
> leave the content defect essentially intact.

**Power caveat, stated rather than buried:** 3 split seeds give 2 degrees of
freedom, so this test could not detect a *small* real reduction. What it does
exclude is selection as a *substantial* explanation, because a substantial
selection effect would have produced a large collapse and the observed arm A sits
at 85% of arm T.

---

## 3. The premature-termination hypothesis — evidence against

Section 30.8 offered premature termination, compounding rightward, as consistent
with the emission gradient. It has not survived.

**Against:**

1. Its one concrete prediction failed on both legs. It predicted delta would
   raise PTH emission and leave QRY flat. Measured: PTH 0.838 -> 0.842 (benign)
   and 0.789 -> **0.771** (phishing, wrong direction), QRY 0.388 -> 0.439 and
   0.293 -> 0.384.
2. The supporting account — truncation deletes the path exit so the model keeps
   writing path — is **benign-only**. Greedy drives benign paths to 91.70% and
   phishing paths to **12.90%**: opposite directions from the same intervention.
3. **Nothing terminates prematurely at the decision point.** Teacher forcing
   shows the query decision is healthy on real prefixes in both classes, and in
   phishing slightly over-assigns.
4. In benign, per-position behaviour is *identical* to real (ratio 1.01). There
   is no premature anything; there are simply fewer positions.

**For:** the emission ratios do fall monotonically with field depth
(benign 0.915 / 0.838 / 0.388; phishing 0.980 / 0.789 / 0.293), and step count
does fall (0.64x, 0.67x). But a monotone gradient is equally consistent with the
structural account, which is measured, so the gradient is not independent
evidence.

> **Verdict: not supported as a mechanism.** The observations it was invented to
> explain are better explained by (benign) too few path steps with correct
> per-step behaviour, and (phishing) a per-prefix probability collapse.

---

## 4. Mechanisms, ranked by evidential support

Mechanisms, **not** fixes. No intervention is selected.

| # | mechanism | evidence | strength |
|---|---|---|---|
| 1 | **Decoder truncation discards ~96% of query probability at the positions where it is legal** | direct per-position measurement (3.11% / 4.62% retention); monotone 4-decoder ladder; greedy reaches exactly 0/4000 in phishing | **strong** — measured, replicated in both classes, dose-responsive |
| 2 | **Benign: the query deficit is the path deficit.** Per-position behaviour is correct; opportunities are too few | attachment ratio 1.01 (mean) / 0.97 (median); step count 0.64x; pathless 27.25% -> 42.42% | **strong** — a tight null on the per-step term plus a measured structural gap |
| 3 | **Phishing: the model's own prefixes elicit ~1/3 the query probability of real prefixes** | attachment ratio 0.32 mean / 0.25 median, on the same 1,200 URLs per side | **strong** as a measurement, **unexplained** as a cause — *why* its prefixes differ is not established |
| 4 | **Content is a genuine modelling failure, independent of emission** | Q4: untruncated arm still +0.1124, 2.2x bar; selection explains 15.1%, t=−1.64 | **moderate–strong** — clear direction, underpowered at 2 df |
| 4a | **The content failure is SHAPE: counts, lengths, character-class layout — not vocabulary** | Q3: `skeleton` retains the full tell in both arms (t=+0.41, +0.23) with every character destroyed; descriptive p90 25 vs 15 at identical median; every other component null | **strong** — an isolating null replicated across two independently harvested arms, corroborated by an independent descriptive half |
| 5 | Premature termination / rightward compounding | failed prediction; benign-only support; contradicted by the healthy conditional | **rejected** |

---

## 5. What remains unknown

1. **Which shape property dominates — ANSWERED as DISTRIBUTED (Q3.1).** At
   D=12 three properties replicate in both arms and none reaches 50%:
   parameter-name identity (17-28%), parameter count (11-33%, one-way
   evidence), long-value tail (3-6%, dose-response non-monotone). Encoding
   presence and intra-field punctuation were rejected on replication failure
   (one sign-flipped). `skeleton` sufficiency replicates at R_T 1.03, which
   means **no single-property repair can remove the tell**. Mechanism measured:
   long-tail under-production, in parameter names (TTR 0.6023 real vs 0.3626
   generated at equal n) and value lengths (p90 25 vs 15). Whether those two
   share a cause is **not** established. See Q3_1_RESULTS.md.
2. **Why phishing prefixes elicit low query probability.** Mechanism 3 is
   measured but unexplained. Candidates not yet tested: the model generating
   hosts resembling the query-less subpopulation; suffix composition; path shape.
3. **Whether Q4 replicates in phishing.** Q4 ran on benign only, chosen because
   its decoder term was larger. Phishing's mechanism differs, so the selection
   test may not transfer.
4. **Whether mechanisms 1–3 account for the whole emission gap quantitatively.**
   The terms were measured separately; no additive model has been fitted or
   tested, and the naive product of the phishing ratios over-predicts the drop.
5. **Whether any of this affects the gate if fixed.** Untested and deliberately
   so.
6. Benign Q2b's generated marginal read 3.00% at n=1200 against 4.22–4.25% from
   two 4,000+ samples. The small sample was the outlier; **the 3.00% figure
   should not be quoted.** Its attachment ratio is unaffected, being computed on
   the same URLs on both sides.

---

## 6. Recommended next experiment

> **Diagnose WHY the model fails to learn a tail its corpus demonstrably contains.**

Q3.1 ranked the shape components and returned DISTRIBUTED: no property removes
half the tell, and `skeleton` proves the tell survives with all lexical content
destroyed, so a single-property repair cannot close it. The mechanism measured
underneath is **long-tail under-production** — of rare parameter names and of
long values alike.

**Q5 then settled where that comes from (Q5_RESULTS.md).** At a common
subsample the training corpus is **tail-rich** and matches validation on every
metric (TTR 0.5853 vs 0.5971, top-20 29.68% vs 28.62%, p90 27.0 vs 25.0), while
generated sits ~22 sd away. The structure corpus is richer still. **The model is
failing to learn a tail that is present in its training data**, so the
"faithful reproduction of a tail-poor corpus" explanation is excluded.

Q5 also split the deficit in two: name concentration is 54% model / 46% decoder,
but the value-length tail is **100% model / 0% decoder** — truncated and
untruncated sampling both stop at p90 = 15. They are two defects with different
mechanisms, so no single-site intervention addresses both.

**Q6 tested the length budget.** Training does not truncate over-length rows, it
**deletes** them, and deletion is strongly tail-selective: 41-45% of rows whose
longest value exceeds 100 characters are dropped against 0.16-0.25% at the
bottom bucket. But it is a **minority contributor** -- the loss still sees value
p99 = 87-88 while the model emits 56, so ~31% of the p99 gap.

**Q7/Q8 finished the root cause (Q7_RESULTS.md).** Using a teacher-forced
termination hazard -- no sampling, so decoder and prefix drift are excluded by
construction -- the model stops too eagerly on REAL prefixes: ~8% excess to
position 50, **1.315 [1.169, 1.494] at 50-100**, which compounds to ~54% fewer
values reaching 100 characters.

Critically it fails on **TRAIN as badly as on VAL** (1.346 vs 1.315), so this is
**UNDER-fitting, not a generalisation failure** -- which eliminates the whole
family of generalisation remedies. Q8 then found a sharp **learnability
threshold**: parameter names seen fewer than ~165 times in the corpus are
reproduced at a fifth to a tenth of the rate a faithful sampler of the same size
would give, outside the 95% null interval, while names seen 1000+ times are
over-produced 2.2x.

Two hypotheses were **refuted** along the way, one of them asserted
mid-investigation and withdrawn: the field mask does **not** under-weight
queries (trained/own = 1.0 in both corpora -- `CORPUS_SCOPE` is inert once the
frame token is present), and the deficit is **not** an absolute-position or
context effect (the direction reverses by bucket). The path field shows the same
eager-stopping at positions 4-50, so under-sustaining long fields is **general**;
only the deep-tail escalation is query-specific.

**Capacity and insufficient optimisation (~2 epochs) remain PLAUSIBLE and
UNMEASURED** -- separating them from the exposure threshold requires a training
run, which read-only diagnosis cannot do.

The completed plan is **REPAIR_PLAN.md**, which supersedes the earlier
`REPAIR_PLAN_DRAFT.md`. It covers all four confirmed contributors, orders them
(R1 length budget is a prerequisite; R2 optimisation follows; R3 capacity only
if R2 fails; R4 decoder is independent and must be tested separately, never
stacked), and names two interventions that must **not** be attempted -- any
loss-weighting change, whose premise H6 refuted, and corpus work, which Q5
excluded. **Nothing is implemented.**

**Still not recommended:** any fix. Q3 says the model writes queries of the wrong
shape — too few parameters, too short, missing the long tail, under-encoded — but
which of those to target is exactly what is unresolved, and the descriptive gap
(p90 25 vs 15) and the ablation (every component individually null) do not yet
agree on a single culprit.

**Not recommended yet, and why:** replacing the decoder is the obvious move and
would recover roughly half the benign emission gap and a quarter of the phishing
one — but Q4 shows it would leave the content defect essentially intact, and the
content defect is the larger of the two in excess terms. Choosing it now would be
optimising the measurable half before understanding the worse half.
