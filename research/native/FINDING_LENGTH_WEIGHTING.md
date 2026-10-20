# Finding — the training objective counts short URLs more than long ones

**Status: [DIAGNOSTIC] finding, post hoc.**
- **Found:** 2026-09-27, while checking that E8's plan arms' low free-mode emission
  (outside K1b's band) was not an artefact of the pipeline.
- **K1b itself is not read.** E8's V-base check failed, so by its registration no
  arm is read (`E8_RESULTS.md`). This finding concerns what the trained models
  learned, measured directly against the training data. It is not a criterion
  reading.
- **Nothing registered changes:** no criterion, no threshold, no published number.
- **The proposal at the end is for the owner.** It is not registered, and nothing
  has been run for it.

## In plain words

- **The mechanism.**
  - During training, each batch's loss is the average over every character in that
    batch.
  - Batches are built from URLs of about the same length. So a batch of short URLs
    and a batch of long URLs count the same in total.
  - Each URL therefore counts roughly in proportion to **1 / its length**. A 20-token
    URL has about ten times the say of a 200-token one.
- **What it does.** At every choice that changes how long the URL will be, the
  shorter option gets extra weight, and the model learns to take it too often. For
  example: add a path or not; end the URL here or keep going; stop this query value
  or continue; add another parameter or not.
- **What is measured on E7 itself** [CONFIRMED, section 3]. E7 is calibrated to the
  length-weighted data, not to the real data, in:
  - whether a path or query is written at all;
  - ending the URL inside a path — **exactly what the frozen path delta has been
    correcting at decode time**;
  - stopping short and mid-length query values;
  - most of its reluctance to add another query parameter.
  The same holds for E8's plan models: their plan is the length-weighted rate,
  suffix by suffix.
- **Where it reaches.** It has been there since E2 (3 August 2026), so it applies to
  every model in the line: E2–E7, E8's arms and N2a's arms.
- **What it does not explain:**
  - the extra stopping in long query values (positions 50–100), which remains
    beyond the weighting: residual 1.08 benign, 1.49 phishing;
  - E7's over-eagerness to start another path segment. The weighting would push the
    other way.
  - Both are **Unknown. More evidence required.**
- **Not claimed:** that removing the weighting clears any gate cell. Only
  retraining can show that (section 6).

## 1. The mechanism [CONFIRMED, code inspection]

- **The loss** (`uie/model.py`, `URLTransformer.loss`) returns
  `(per_token * mask).sum() / mask.sum()`. That is the mean over the batch's scored
  tokens.
  - Before E4 the unweighted path was `F.cross_entropy(..., ignore_index=pad)`, whose
    default reduction is also that mean.
- **The sampler** (`uie/train.py`, `batches`) draws a window of
  `batch_size × bucket_window` rows (256 × 64), sorts it by length, and cuts it into
  batches. So one batch holds rows of nearly one length L.
- **Every optimiser step** takes one batch's mean. So each scored token carries
  weight ≈ 1 / (256 × (L − 1)), and each row in total carries weight ≈ 1/256,
  **whatever its length**. A long row's tokens are each worth less.
- **At a decision,** the loss is minimised by the weighted rate:

      P*(option | context) = Σ w_r · 1[row r takes option] / Σ w_r ,  w_r = 1 / (L_r − 1)

  An option that makes rows shorter wins weight.
- **Against the code's own intent.** The loss's docstring says rows are averaged over
  their weight mass so that tokens count equally, and so that "the effective
  learning rate on a field [does not] drift with whatever mix of rows a batch
  happened to contain". That holds inside a batch. Length bucketing (E2) breaks it
  across batches: the per-token weight varies about tenfold between batches of
  short and batches of long rows.
- **History** (git): bucketing arrived in `079592f` (E2, 2026-08-03); the weighted
  mean in `2bdece5` (E4/E5, 2026-09-21), replacing an equivalent unweighted mean.

## 2. Presence decisions

**2.1 The plan models learn the weighted rate, suffix by suffix** [CONFIRMED]
(`e8_length_weighting.py --arm e8_pfrs`, `data/e8/length_weighting_e8_pfrs.json`).

In free mode a plan arm's plan depends only on the class, the frame and the
suffix, so for each suffix it is one number, read here exactly.

| class | suffix | P: data | P: 1/L-weighted | P: model | Q: data | Q: 1/L-weighted | Q: model |
|---|---|---|---|---|---|---|---|
| benign | com | 0.697 | 0.545 | **0.555** | 0.0552 | 0.0378 | **0.0373** |
| benign | de | 0.744 | 0.613 | **0.632** | 0.0568 | 0.0368 | **0.0364** |
| benign | org | 0.725 | 0.577 | **0.591** | 0.0625 | 0.0399 | **0.0398** |
| benign | net | 0.674 | 0.509 | **0.539** | 0.0737 | 0.0483 | **0.0472** |
| phishing | com | 0.319 | 0.233 | **0.227** | 0.0633 | 0.0349 | **0.0299** |
| phishing | dev | 0.302 | 0.202 | **0.195** | 0.0343 | 0.0167 | **0.0180** |
| phishing | net | 0.391 | 0.266 | **0.263** | 0.0848 | 0.0401 | **0.0297** |

Weighted by the shipped table:

| class | P: data | P: weighted | P: model | Q: data | Q: weighted | Q: model |
|---|---|---|---|---|---|---|
| benign | 0.733 | 0.585 | 0.610 | 0.0653 | 0.0449 | 0.0416 |
| phishing | 0.394 | 0.317 | 0.313 | 0.0451 | 0.0254 | 0.0204 |

**2.2 It is neither the labels nor the sampler** [CONFIRMED]
(`e8_plan_calibration.py`):
- **The labels are right.** The plan labels the arms trained on equal the gate's
  presence on every one of the 906,214 benign and 192,652 phishing `name_train`
  rows: 0 disagree.
- **The sampler is faithful.** The model's own plan gives an expected emission of
  benign P 0.610 and Q 0.0416. The generated files carry 0.615 and 0.0400.
- **Checked on real rows too.** Teacher-forced on real training rows:
  - domain-frame `nl` and `com`: model P 0.628 and 0.555, against 0.745 and 0.677
    in the data;
  - page-frame `com` and `nl`: 0.962 and 0.960, against 0.980 and 0.977.
- **One size of error.** In log-odds every gap is about −0.5 to −0.65, one
  systematic offset rather than per-suffix noise.

**2.3 E7's free-mode emission sits at the weighted rates** [CONFIRMED as
arithmetic] (`e8_length_weighting.py --e7`). The rates are computed with E7's own
tokenizer lengths.

| class | P: data | P: weighted | P: E7 free | Q: data | Q: weighted | Q: E7 free |
|---|---|---|---|---|---|---|
| benign | 0.733 | 0.574 | 0.591 | 0.0653 | 0.0445 | 0.0362 |
| phishing | 0.394 | 0.310 | 0.295 | 0.0451 | 0.0251 | 0.0174 |

The weighting accounts for essentially all of E7's path shortfall and about 70% of
its query shortfall.

## 3. Content decisions — the calibration test on E7 [CONFIRMED]

**The test** (`weighted_calibration.py`). A model trained on a loss that weights row
r by w_r = 1/(L_r − 1) becomes calibrated in the **weighted** sense: over a large
group of positions, Σ w·p ≈ Σ w·y. This is the first-order condition of its loss.
The two explanations predict different patterns:

| | plain ratio Σp / Σy | weighted ratio Σwp / Σwy |
|---|---|---|
| plainly calibrated model | ≈ 1 | **below 1** (stop events sit in shorter rows) |
| model fitted to the weighted loss | **above 1** | ≈ 1 |

**Setup:**
- E7, teacher-forced with its controls hidden (free mode), on its own
  domain-frame training rows: N2a-0's 20,000-row query sample (8,272 phishing), and
  20,000 path rows.
- **Validity:** the plain query ratios reproduce N2a-0's recorded M1 exactly, in
  every bucket and both classes.

**Stopping inside query values** (Q7's buckets and events):

| position in value | benign plain | benign weighted | phishing plain | phishing weighted |
|---|---|---|---|---|
| 0–4 | 1.075 | 0.995 | 1.110 | 1.068 |
| 4–10 | 1.054 | 1.008 | 1.113 | 0.956 |
| 10–25 | 1.066 | 1.008 | 1.083 | 0.942 |
| 25–50 | 1.044 | 0.940 | 1.133 | 0.971 |
| 50–100 | 1.259 | **1.083** | 1.836 | **1.487** |
| 100+ | 1.019 | 0.939 | 1.141 | 1.089 |

**Ending the URL inside a path** (the end token's mass at every path position):

| position in path | benign plain | benign weighted | phishing plain | phishing weighted |
|---|---|---|---|---|
| 0–4 | 1.560 | 0.970 | 1.107 | 1.022 |
| 4–10 | 1.290 | 1.004 | 1.145 | 1.016 |
| 10–25 | 1.157 | 0.996 | 1.100 | 1.018 |
| 25–50 | 1.120 | 1.015 | 1.094 | 1.003 |
| 50–100 | 1.091 | 1.023 | 1.185 | 1.081 |
| 100+ | 1.062 | 1.025 | 1.244 | 1.190 (105 events) |

**Structure decisions:**

| decision | class | plain | weighted |
|---|---|---|---|
| another query parameter ('&' at a value's end) | benign | 0.845 | 0.965 |
| another query parameter | phishing | 0.724 | 0.890 |
| another path segment ('/' at a segment's end) | benign | 1.094 | 1.176 |
| another path segment | phishing | 1.216 | 1.381 |

**What this establishes** [CONFIRMED]. E7 is fitted to the length-weighted
objective in:
- ending the URL inside paths: weighted 0.97–1.03 benign against plain up to 1.56;
- stopping short and mid query values: weighted 0.94–1.01 benign against plain
  1.04–1.08;
- most of its under-prediction of another query parameter.

**What remains unexplained** [CONFIRMED as measurements; cause Unknown]:
- **long query values** (50–100) still stop too often beyond the weighting;
- **another path segment** is over-predicted, which runs against the weighting's
  direction.

**Superseded.** An earlier corpus-only estimate (`hazard_length_weighting.py`)
grouped rows by position alone. It predicted about 1.2 everywhere, which did not
match. It ignored what the model conditions on. This test uses the model's own
predictions instead.

## 4. What it changes in the reading of earlier results

- **The frozen PTH end-of-string delta** (0.22 benign / 0.30 phishing, subtracted
  from the end token's logit inside paths at decode time) has been correcting a
  bias the training objective creates. At short path positions the plain ratio
  1.29–1.56 is log-odds 0.25–0.44. The resting decode keeps it until a native
  replacement exists.
- **E8 (N1).** The plan arms' free emission sits below K1b's band because they
  learned the length-weighted rates [CONFIRMED]. The plan samples exactly what the
  model believes.
  - No K1 criterion is read: V-base failed (`E8_RESULTS.md`).
  - Whatever E8's formal reading becomes, this finding explains its emission side
    and does not change it.
- **E7's free-mode under-emission** (NATIVE_REPAIR_PLAN §2, K1) has this cause for
  paths, and most of it for queries [CONFIRMED].
- **N2a.** Its arms trained under the same objective. "More exposure buys copying,
  not realism" is what one would also see if the objective set the limit: more
  exposure converges on the weighted optimum, whose queries are too short and too
  few-parametered.
  - N2a's registered reading (SPARSITY) stands. It now has a measured competing
    explanation [HYPOTHESIS: that the objective is what limits query realism].
  - N2a did not vary the objective and cannot separate the two.
- **Q7 and C3.** Q7's "under-fit on training data too" and C3-B's structural benign
  query tell are consistent with a model that is well fitted to the wrong
  (weighted) target [HYPOTHESIS].

## 5. Closed doors — why this is not one

- **S3 / S4 / S4b** (PTH loss weighting) are CLOSED. They **added** weights at chosen
  positions to move a metric.
- **The delta branch** is CLOSED. The frozen delta stays in the resting decode, and
  nothing here changes it.
- This finding is an **unintended** weight of about 1/length on every row. It comes
  from the interaction of the normalisation with bucketing, and it contradicts the
  normalisation's documented intent.
- Removing it re-weights nothing toward a target. It restores equal weight per
  token: the plain maximum-likelihood objective the model was meant to have.
- No other closed item (plan §8) touches the loss's normalisation.

## 6. Proposal for the owner — not registered

**One native change,** in the training objective, not at decode time: divide each
batch's summed loss by a **fixed constant** instead of by that batch's own token
count. For example, 256 × the mean scored tokens per row over the training set.
- Every scored token then weighs the same in every batch. The objective becomes the
  plain maximum-likelihood objective.
- Bucketing keeps its speed.
- It is a flag defaulting to off, so every earlier run stays reproducible.
- Seed the initialisation too (`E8_RESULTS.md` §3), so that the run is comparable.

**Sharp predictions it would test:**
1. **Free-mode emission** moves to the data rates (P ≈ 0.73 / 0.39, Q ≈ 0.065 /
   0.045).
2. **Plain calibration** of URL-ending inside paths, of stopping in short and mid
   query values, and of another parameter moves to ≈ 1 (section 3's plain column).
   So, **without the delta,** paths should stop no more often than the data.
3. **The open question,** answered by measurement: whether the gate's content
   cells move, especially QRY (the query tell) and PTH, against the
   retraining-noise rule.

**How it could be run.**
- Retrain E7's recipe with only this change (and, separately if wanted, E8's plan
  recipe), on the same data and seed. One Kaggle training session, then generation.
- It would be registered first, with its own criteria, if the owner chooses it.

**What it does not claim:** that it clears any gate cell, or that it passes E3.
