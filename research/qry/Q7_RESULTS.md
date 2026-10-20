# Q7 / Q8 — root cause: the model under-fits the tail, and rare structure is below a learnability threshold

**Run:** 2026-09-22. Artifacts `data/qry/q7_hazard_*.json`,
`data/qry/q8_exposure.json`, logs alongside. Pre-registered in `PREREG_Q7.md`,
committed as `60d4276` **before** any measurement.

**READ-ONLY throughout.** Teacher forcing and corpus counting only. No sampling
in Q7, no training, no weight, checkpoint, decoder, tokenizer, gate or corpus
modified. **Scope: BENIGN.** **No repair implemented.**

---

## The gap under investigation

Q6 left a 46% middle term: the loss saw value p99 = 87 and the model emits 56.

## The instrument

Inside a query value the next token either continues it or ends it (`&`,
`<FRG>`, `<eos>`). Define the **termination hazard**

    h(i) = P(value ends at position i | value reached position i)

Empirical `h` is counted from the corpus; model `h` is read from the
teacher-forced distribution on those same rows. **Because the prefix is held at
the real one and nothing is sampled, the decoder and prefix drift are excluded
by construction** — what remains is the model's conditional belief alone.

Bootstrap resamples **rows, not positions**: positions inside one value are not
independent, and resampling them would give intervals far too tight. Buckets
with fewer than 200 rows are reported **UNMEASURED**, never estimated.

---

## H1 — the model did not learn the tail. **CONFIRMED.**

Hazard ratio `h_model / h_empirical`, QRY, benign:

| within-value | rows (val) | val ratio | 95% CI | rows (train) | train ratio | 95% CI |
|---|---:|---:|---|---:|---:|---|
| 0–4 | 8362 | 1.128 | [1.050, 1.222] | 8215 | 1.038 | [0.968, 1.121] |
| 4–10 | 7947 | 1.067 | [1.041, 1.092] | 7739 | 1.061 | [1.036, 1.087] |
| 10–25 | 4566 | 1.085 | [1.057, 1.112] | 4334 | 1.086 | [1.060, 1.114] |
| 25–50 | 1159 | 1.106 | [1.049, 1.172] | 1080 | 1.068 | [1.011, 1.134] |
| **50–100** | 286 | **1.315** | **[1.169, 1.494]** | 269 | **1.346** | **[1.183, 1.517]** |
| 100+ | 61 | UNMEASURED | — | 58 | UNMEASURED | — |

Conditioned on **real** long-value prefixes, the model still assigns too much
probability to stopping: ~8% excess through position 50, **escalating to ~32% at
50–100**.

**Why a few percent matters.** The excess compounds. Using the observed survival
counts: survival to 50 is 286/8362 = 3.4%, and 50→100 is 61/286 = 21%. An 8%
hazard excess over the first stretch and 32% over the second gives

    survival to 50   x0.76      survival 50->100  x0.61      combined  x0.46

i.e. **~54% fewer values reach 100 characters**, measured without a single
sampled token. That is the right order to account for Q6's 87 → 56 term.

---

## H2 — this is NOT a generalisation failure. **REFUTED.**

Registered: support = tracks on TRAIN, diverges on VAL; refutation = similar
divergence on both.

**Similar divergence on both occurred.** Deep tail 1.346 on train against 1.315
on val; every bucket within overlapping CIs.

> **The model never fit the tail hazard even on rows it trained on. This is
> UNDER-fitting, not over-fitting.**

This is the most consequential result of the phase for the repair space: it
eliminates the whole family of generalisation remedies (more held-out coverage,
augmentation, regularisation) and puts the weight on **signal, optimisation or
capacity**.

---

## H4 — not a context/position effect. **NOT SUPPORTED.**

Hazard ratio by absolute token position, within-value position held fixed:

| within-value | 0–60 | 60–120 | 120–256 |
|---|---|---|---|
| 0–4 | 1.09 [1.00,1.20] | 1.18 [1.04,1.34] | 1.73 [1.10,4.10] |
| 4–10 | 1.06 [1.03,1.09] | 1.06 [1.02,1.12] | 1.20 [1.05,1.41] |
| 10–25 | 1.10 [1.06,1.14] | 1.07 [1.03,1.11] | 1.06 [0.97,1.16] |
| 25–50 | 1.21 [1.08,1.39] | 1.14 [1.07,1.22] | 0.89 [0.78,1.03] |
| 50–100 | — | 1.40 [1.20,1.67] | 1.19 [1.02,1.45] |

**The direction reverses**: short values worsen with absolute position, long
values improve. Registered refutation ("depends on within-value position, not
absolute") is what occurred. This is a **length effect, not a context effect**.
The 1.73 cell carries CI [1.10, 4.10] and is barely determined.

---

## H5 — partly general, partly query-specific. **SPLIT.**

The path field is used **purely as a control for a query question. PTH
attribution stays closed and no PTH claim is made here.**

| within-value | QRY/val | PTH/val |
|---|---:|---:|
| 0–4 | 1.128 eager | 1.013 tracks |
| 4–10 | 1.067 eager | 1.096 eager |
| 10–25 | 1.085 eager | 1.107 eager |
| 25–50 | 1.106 eager | **1.203 eager** |
| 50–100 | **1.315 eager** | 1.090 **tracks** [0.997, 1.193] |

- **Positions 4–50: GENERAL.** Both fields stop too eagerly, the path field if
  anything more so. Under-sustaining long fields is not query-specific.
- **Positions 50–100: QUERY-SPECIFIC.** The path field tracks empirical; the
  query field escalates to 1.32.

Both fields are structure-scoped and equally visible to the loss (H6), but paths
occupy 54–67% of field characters against queries' 4.4–5.7% — roughly an order
of magnitude more signal. **That is a hypothesis for the split, consistent with
H3, not a demonstrated link.**

---

## H6 — the mask does NOT under-weight queries. **REFUTED.**

Measured by running the **frozen `field_mask`** over real encoded rows:

| corpus | QRY char share | QRY trained share | trained / own chars |
|---|---:|---:|---:|
| name_train | 4.38% | 4.38% | **1.0** |
| structure_train | 5.66% | 5.66% | **1.0** |

Every field in both corpora returns `trained/own = 1.0`. The reason is in the
source:

```python
if getattr(tokenizer, "include_frame", False):
    scope = set(NAME_FIELDS) | set(STRUCTURE_FIELDS)   # E5: ALL fields
else:
    scope = set(CORPUS_SCOPE.get(corpus, ()))
```

E5's tokenizer sets `include_frame=True`, so **`CORPUS_SCOPE` is inert for this
checkpoint** — the frame token removed the need for scoping, exactly as the
docstring says it would.

> **A claim made mid-investigation is withdrawn.** On reading `CORPUS_SCOPE`
> alone it was asserted that the 906,214-row name corpus contributes zero
> gradient to query content. **That is false for this checkpoint.** The constant
> describes a pre-frame-token regime. The measurement corrected the reading,
> which is why it was run instead of trusting the constant.

Queries are not down-weighted. They are simply **rare**: 4.4–5.7% of field
characters against the path field's 54–67%. That is proportional representation,
not a mask defect. **S4/S4b stays closed; nothing here proposes a weighting
change.**

---

## H3 — rare structure sits below a learnability threshold. **CONFIRMED.**

Parameter-name frequency in the structure corpus (200,000 rows) against what the
model emits. The **null** is a faithful sampler drawing the same number of
parameters from the training frequency distribution — without it, a rare name's
absence at n=758 is unreadable.

| train count | distinct names | corpus token % | typical observed % | null % [95% CI] | temperature observed % |
|---|---:|---:|---:|---|---:|
| 1 (hapax) | 2,894 | 6.93 | **0.66** | 6.91 [5.14, 8.71] | 1.52 |
| 2–4 | 2,033 | 13.33 | **2.51** | 13.32 [10.95, 15.70] | 2.48 |
| 5–19 | 1,066 | 20.30 | **4.09** | 20.40 [17.41, 23.22] | 5.93 |
| 20–99 | 254 | 23.97 | 20.05 | 23.88 [20.98, 27.05] | 18.34 |
| 100–999 | 49 | 27.41 | **41.42** | 27.48 [24.27, 30.87] | 35.45 |
| 1000+ | 2 | 8.06 | **17.81** | 8.01 [6.07, 9.89] | 13.10 |

The null reproduces the corpus profile almost exactly, which is what validates
it. The observed profile is far outside it in **both** directions.

> **A sharp frequency threshold.** Names seen fewer than ~20 times in the sample
> (~165 in the full corpus) are reproduced at a fifth to a tenth of their due
> rate; names seen 100+ times are over-produced by 1.5–2.2x. Sampling at n=758
> is excluded by the null.

Scaling by the ~1.7 exposures per row implied by 5.12M exposures over a ~2.55M
benign pool, the threshold sits in the region of **a few hundred gradient
exposures**. That figure is an **order-of-magnitude estimate**, not a fitted
constant, and is labelled as such.

Two supporting details:

- Hapax names are **45.95% of the vocabulary but only 6.93% of tokens** — the
  rare tail is broad and thin.
- The model **does invent unseen names** (OOV 13.46% typical, 23.17%
  untruncated), so this is a *frequency-profile* failure, not an inability to
  generalise. Untruncated sampling reproduces rare names ~2x more often than
  production, consistent with Q5's 54/46 model/decoder split.

---

## H7 — capacity. **PLAUSIBLE, UNMEASURED.**

Registered in advance as not read-only testable, and it remains so. H2's
signature (fits neither train nor val) is *consistent* with capacity — but it is
equally consistent with insufficient exposure (H3, confirmed) and with
insufficient optimisation. **The measurement that would discriminate requires a
training run, which is forbidden here.** "Bigger model" is not assumed.

## H8 — insufficient optimisation. **PLAUSIBLE, UNMEASURED (added during the phase).**

Not pre-registered; surfaced by reading the checkpoint's own config. 20,000
steps x 256 = **5.12M exposures over a ~2.55M-row benign pool ≈ 2 epochs**. A
model that under-fits its training data after two epochs is exactly what
too-little optimisation looks like. **Flagged because the evidence raised it, and
marked UNMEASURED because separating it from capacity needs a training run.**

---

## Causal map

| mechanism | status | evidence | contribution |
|---|---|---|---|
| tail-poor corpus | **REFUTED** | Q5: corpus TTR 0.5853 vs generated 0.3626 at equal n; structure corpus richer still | — |
| long values truncated pre-loss | **REFUTED** | Q6: `encode_corpus` deletes, does not truncate | — |
| generalisation failure | **REFUTED** | Q7 H2: train 1.346 vs val 1.315 — fails on both | — |
| absolute position / context | **NOT SUPPORTED** | Q7 H4: direction reverses by bucket | — |
| field-mask under-weights QRY | **REFUTED** | Q8 H6: trained/own = 1.0 everywhere; frame token makes scope total | — |
| **max_len row deletion** | **CONFIRMED** | Q6: 41–45% of 100+ char rows deleted; 258x the base rate | **~31% of the p99 gap; ~18% at p90** |
| **model under-fits the tail hazard** | **CONFIRMED** | Q7 H1: ~8% excess to pos 50, ~32% at 50–100, on train *and* val | **~46% of the p99 gap; compounds to ~54% fewer values reaching 100** |
| **frequency / exposure threshold** | **CONFIRMED** | Q8 H3: bands <20 reproduced at 1/5–1/10 of null, outside 95% CI | **not cleanly separable from the hazard term — see below** |
| **decoder truncation** | **CONFIRMED** | Q1/Q4/Q5: monotone across 4 decoders; retention 3.11% of legal positions | **~24% of the p99 gap; ~46% of the name-TTR gap** |
| general long-field under-sustain | **CONFIRMED (not QRY-specific)** | Q7 H5: path field 1.096–1.203 at pos 4–50 | not quantified for QRY alone |
| capacity | **PLAUSIBLE, UNMEASURED** | H2 signature consistent; not discriminating | unquantifiable read-only |
| insufficient optimisation (~2 epochs) | **PLAUSIBLE, UNMEASURED** | checkpoint config; under-fits train | unquantifiable read-only |

**On double-counting.** The hazard term (H1) and the exposure threshold (H3) are
**not independent**: a plausible reading is that insufficient exposure *causes*
the under-fit hazard. They are reported as two measurements of what may be one
mechanism, and **their contributions must not be summed**. No percentage is
assigned to H3 for exactly this reason.

## What remains genuinely uncertain

1. Whether H3 causes H1, or both follow from a third cause (capacity /
   optimisation). **Not separable read-only.**
2. Whether capacity is binding. Requires a training run.
3. Whether ~2 epochs is the constraint. Requires a training run.
4. Whether any of this would move the gate — Q3.1's `skeleton` result still says
   shape alone reproduces the whole tell.
5. Everything about phishing. Benign only throughout.
