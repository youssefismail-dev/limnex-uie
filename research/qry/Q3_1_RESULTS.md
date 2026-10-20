# Q3.1 results — DISTRIBUTED, and a long-tail mechanism

**Run:** 2026-09-22. `data/qry/q3_1_benign.json`, log `data/qry/q3_1.log`.
Pre-registered in `PREREG_Q3_1.md`, committed as `53e75e5` **before** any
measurement. Continues Q3; **Q3_RESULTS.md is preserved unchanged.**

**Frozen:** `best_e5.pt`, the decoder, the sampling policy, the canonical gate,
the 0.05 bar, training data and weights. Scope **benign only**. **No fix
implemented.**

**Configuration:** D = 12 split seeds (Q3 used 3), df = 11, **t_crit = 2.201**.
The rule is unchanged — two-sided t at p<0.05; `scipy.stats.t.ppf(.975, 2) =
4.303` is exactly the 4.30 Q3 used, so only the data changed.

Harness checks: `null` byte-exact on both arms; `identity` +0.1405 (typical) and
+0.1032 (temperature), both inside Q3's own seed spread — Q3's three typical
draws were 0.1399 / 0.1718 / 0.0764, sd ~0.048, which is itself why D=3 could
rank nothing.

---

## 1. The ladder — 17 transforms, two stages

Stage 2 re-ran only what cleared threshold in stage 1. **A finding is promoted
only if it replicates.**

| transform | component | typical R_T | t | temperature R_T | t | replicates |
|---|---|---:|---:|---:|---:|:--|
| `null` | control | 1.00 | — | 1.00 | — | — |
| `clip_values_20` | tail > 20 | 0.98 | −2.15 | — | — | not promoted |
| **`clip_values_15`** | **tail > 15 (gen p90)** | **0.97** | **−2.43** | **0.94** | **−3.27** | **YES** |
| `clip_values_10` | tail > 10 | 0.99 | −0.67 | — | — | not promoted |
| **`first_param`** | **count → 1** | **0.89** | **−2.95** | **0.67** | **−6.71** | **YES** |
| `fixlen_names` | name length | 1.01 | +0.54 | — | — | not promoted |
| `mask_pct` | encoding presence | 1.03 | **+4.12** | 0.99 | −1.16 | **NO — sign flip** |
| `normalize_seps` | separator structure | 1.01 | +2.17 | — | — | not promoted |
| `mask_punct` | intra-field punctuation | 1.02 | **+3.23** | 1.00 | +0.43 | **NO** |
| `alnum_flat` | digit-vs-alpha | 0.90 | −1.06 | — | — | not promoted |
| **`mask_name_chars`** | **name identity only** | **0.83** | **−2.28** | **0.72** | **−3.27** | **YES** |
| `mask_values` | value composition | 1.03 | +0.45 | — | — | not promoted |
| `fixlen_values` | value length | 0.99 | −0.35 | — | — | not promoted |
| `sort_params` | ordering | 1.00 | −0.05 | — | — | not promoted |
| `skeleton` | structure + char class *(isolating)* | 1.03 | +0.41 | — | — | not promoted |
| `structure_only` | count + seps + lengths *(isolating)* | 0.74 | −3.17 | 0.90 | −1.08 | **NO** |

`count_only` was excluded in advance (VOID degenerate in Q3, both arms). The
pre-specified interaction `clip15_first` was **correctly not run**: the rule
required both components to be individually null and both were significant.

---

## 2. Decision: **DISTRIBUTED**

Against the rule fixed in advance:

**IDENTIFIED — NOT MET.** It required R_T <= 0.50. The largest replicated effect
is `first_param` at R_T 0.67 on one arm and 0.89 on the other. **No property
removes half the tell.**

**DISTRIBUTED — MET.** Three properties each reduce excess with |t| > 2.201 **in
both arms**, and none reaches 50% alone:

| property | typical | temperature | share of tell |
|---|---|---|---|
| parameter-name identity | t −2.28, R_T 0.83 | t −3.27, R_T 0.72 | 17–28% |
| parameter count | t −2.95, R_T 0.89 | t −6.71, R_T 0.67 | 11–33% |
| long-value tail | t −2.43, R_T 0.97 | t −3.27, R_T 0.94 | 3–6% |

### Caveats attached to each, none of which the decision rule erases

- **`first_param` is one-way evidence, by registration.** Count cannot be removed
  without also removing later parameters' content, so its *reduction* is
  ambiguous between the two. Its *survival* is clean, and 67–89% of the tell
  survives with every query at exactly one parameter.
- **`clip_values_15`'s dose-response is NON-MONOTONE** — caps 20/15/10 give
  t = −2.15 / −2.43 / −0.67. The tightest cap gives the *smallest* effect. This
  was registered as disconfirming. It replicates across arms yet fails its own
  series, so it is retained in the set with that contradiction on the record.
- **Magnitudes are not stable across arms** (`first_param` 11% vs 33%), so the
  decomposition itself is not reproducible in size, only in direction.

---

## 3. What was rejected

- **Percent-encoding presence** — t = **+4.12** on typical, **−1.16** on
  temperature. A sign flip between arms. Rejected, despite a 3x descriptive gap
  (9.89% vs 3.15%).
- **Intra-field punctuation** — +3.23 then +0.43. Rejected.
- **`structure_only`** — −3.17 then −1.08. Not replicated.
- **Never cleared stage 1:** name length, separator structure, digit-vs-alpha
  composition, value composition, value length, ordering, and the two outer clip
  caps.
- **Value length is doubly null**: `mask_values` (+0.45, kills identity keeps
  length) and `fixlen_values` (−0.35, kills both) agree, which is the registered
  pair confirming it from both sides.

**Both stage-1 "raises" hits died in replication.** That is the two-stage design
doing exactly what it was registered to do.

---

## 4. Sufficiency is unchanged, and it constrains any repair

`skeleton` — every letter to `a`, every digit to `0`, punctuation and lengths
intact — **retains the full tell at four times the resolution** (R_T 1.03,
t = +0.41). Q3's central finding replicates.

> **This is the most important constraint on any repair.** Shape alone
> reproduces the entire tell with every character destroyed. Therefore **fixing
> parameter names alone cannot remove it** — the shape carrier would remain. The
> carriers are redundant, not additive, and that is why no single property
> reaches 50%.

---

## 5. A mechanism — and a confounded metric of my own, withdrawn

Q3's descriptive half reported `name_ttr` as real 0.3086 vs typical 0.3562,
suggesting generated names were *more* diverse.

> **THAT COMPARISON IS INVALID AND IS WITHDRAWN.** Type-token ratio falls with
> sample size, and it was computed on 12,952 real parameters against 758
> generated. It compares sample sizes, not distributions.

Recomputed at equal n = 725, 30 subsamples:

| | TTR @ n=725 | top-1 share | top-5 | top-20 |
|---|---|---:|---:|---:|
| real | **0.6023 ± 0.0169** | 5.06% | 15.15% | 26.68% |
| typical | **0.3626 ± 0.0029** | 13.06% | 27.18% | 47.49% |
| temperature | **0.4648** | 7.86% | 28.28% | 42.62% |

**The direction reverses.** Real parameter names are far *more* diverse; the
model concentrates on a handful. Its top 20 names cover 47% of all parameters
against real's 27%.

This gives the `mask_name_chars` effect a measured mechanism: **the model
under-produces the long tail of rare parameter names.** Untruncated sampling is
less concentrated than production (7.86% vs 13.06% top-1) but still far from
real — the same **model-dominant, decoder-amplifies** split Q1 measured for
emission.

**The same signature appears twice, independently:** a missing long tail in
parameter *names* (this section) and in value *lengths* (Q3: p90 25 vs 15 at an
identical median of 4). Both are under-production of rare content.

**Not established:** that these two share a cause. They are two observations with
the same shape, measured separately. No experiment links them.

---

## 6. Is this strong enough to justify engineering work?

**Partly, and not for a targeted single-property fix.**

Justified: three replicated effects; a measured mechanism for the largest; a
long-tail signature appearing independently in two places; consistency with Q1's
model-dominant pattern.

Not justified: no property reaches 50%; magnitudes vary 3x between arms; and
`skeleton` proves shape alone suffices, so **any repair aimed at one field
property will leave the tell substantially intact.**

A repair plan follows under `REPAIR_PLAN_DRAFT.md`, targeting the *common*
mechanism rather than any single property, with the honest expectation that it
addresses a contributor and **will not by itself pass the gate.**

---

## 7. Status

    Q3    COMPLETE  content tell is SHAPE; preserved unchanged
    Q3.1  COMPLETE  DISTRIBUTED; 3 replicated properties, none dominant;
                    2 rejected by replication; name-TTR metric withdrawn;
                    long-tail under-production identified as a mechanism
    scope benign only; phishing content never measured
