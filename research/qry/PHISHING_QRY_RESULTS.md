# Phishing QRY — results

**Run:** 2026-09-23, frozen `data/best_e5.pt`, `data/bundle2`.
Registered in `PREREG_PHISHING_QRY.md` before any measurement.
**No intervention selected. Nothing adopted.**

---

## Headline: the prior I registered was WRONG, and that is the result

Section 4 of the pre-registration listed five recorded differences and predicted
phishing would behave differently from benign. **On CONTENT it does not.** The
phishing query tell is the same kind of defect as benign's, carried by the same
dominant component, at nearly the same magnitude.

Every one of those five differences — emission decomposition (76/24 vs 52/48),
attachment ratio (0.32 vs 1.01), pathless rate, corpus size — is about **axis A,
emission**. None of them predicted anything about **axis B, content**. Registering
the expectation in advance is what makes that separation visible rather than a
story told afterwards.

    PQ1  PH1 SUPPORTED   the tell is SHAPE, not vocabulary -- as in benign
    PQ2  criterion MET   selection term small -> PH5 separable, BUT sign reversed
    PQ3  PH3 REFUTED     no generalisation gap in termination
    PQ4  not run         lowest priority, deferred as registered

---

## PQ1 — PRIMARY. The tell is SHAPE. **PH1 supported.**

Registered: PH1 if `skeleton` retains the tell at **R_T >= 0.85**; PH2 if it
destroys it at **R_T <= 0.4**; between the two, neither is claimed.

`skeleton` replaces every character while preserving structure and character
class. It is an isolating null: if the tell survives it, no lexical content is
required to carry the tell.

| | typical | temperature | Q3.1 D=12 (typical) |
|---|---:|---:|---|
| **phishing** | **1.104** | **0.865** | **0.940**, t = -0.72, "no change" |
| benign | 1.114 | 1.061 | 1.029, "no change" |

**All three phishing measurements clear 0.85. PH2 is refuted by a wide margin.**
The D=12 run gives t = -0.72 against t_crit 2.201, i.e. destroying every
character produces no detectable change in the tell.

**Honest qualification:** the temperature arm at 0.865 is the weakest of the three
and sits close to the 0.85 boundary. The claim rests on three measurements
agreeing in direction, not on any one of them.

### The tell is DISTRIBUTED — no single property dominates

Replicated across both arms, D=12, t_crit 2.201. "Share" is `1 - R_T`, the
fraction of the tell a transform removes.

| transform | component | arm T | arm A | share T | share A |
|---|---|---:|---:|---:|---:|
| `alnum_flat` | digit-vs-alpha distinction | 0.79 | 0.68 | 21% | 32% |
| `mask_name_chars` | parameter-name identity | 0.84 | 0.73 | 16% | 27% |
| `mask_values` | value composition | 0.89 | 0.88 | 11% | 12% |
| `first_param` | parameter count | 0.94 | 0.87 | 6% | 13% |
| `clip_values_10/15/20` | long-value tail | 0.90-0.94 | 0.91-0.92 | 6-10% | 8-9% |
| `fixlen_values` | value LENGTH | 0.93 | 0.92 | 7% | 8% |
| `mask_pct` | percent-encoding presence | 1.02 | 1.02 | -2% | -2% |

Null in the registered set: `fixlen_names` (name LENGTH, R_T 1.06),
`normalize_seps` (separator structure, R_T 1.00). Also null: `mask_punct`,
`sort_params`.

**`alnum_flat` carries the largest share but is the least trustworthy reading.**
Its discriminator sits at distinct-fraction 0.52/0.54 against a degeneracy bound
of `DEGENERATE = 0.50` — closer to degenerate than any other non-void transform.
`structure_only` (0.456/0.462) and `count_only` (0.017/0.019) were both ruled
VOID degenerate.

**Compared with benign's Q3.1**, parameter-name identity carries 16-27% here
against 17-28% there — nearly identical. Parameter count is lower (6-13% vs
11-33%) and the long-value tail higher (6-10% vs 3-6%).

**Consequence, the same one benign reached:** no single-property repair can remove
this tell.

### Kill criteria — both cleared

The untransformed excess had to reproduce the canonical gate within +-0.05:

    temperature arm  +0.1616  vs gate +0.1431  |diff| 0.0185  CLEARED
    typical arm      +0.1392  vs gate +0.1488  |diff| 0.0096  CLEARED

No transform moved the real-vs-real floor outside `FLOOR_LOW 0.42 / FLOOR_HIGH
0.55`, so no transform was dropped.

---

## PQ2 — selection. Criterion MET, but the sign is REVERSED against benign.

Registered: `|A - T| < 0.04` supports PH5 (emission and content separable). A
phishing effect LARGER than benign's contradicts Q1's emission decomposition and
must be reported rather than absorbed.

| | typical | temperature | A - T | SE | t | |
|---|---:|---:|---:|---:|---:|---|
| **phishing** | +0.1392 | +0.1616 | **+0.0224** | 0.0049 | **+4.54** | SIGNIFICANT |
| benign | +0.1325 | +0.1124 | -0.0200 | 0.0149 | -1.34 | not significant |

`|A - T| = 0.0224 < 0.04`, so **the criterion is met and PH5 is supported**: the
selection term is small, and emission and content have different causes.

**But PH4 predicted the opposite sign.** Under PH4, truncation-surviving queries
are a biased, *more* distinguishable subsample, so removing truncation should
*reduce* the tell. Phishing does the reverse: **removing truncation makes queries
MORE distinguishable.** Truncation is not manufacturing the phishing content
defect — it is partially **masking** a worse one. Production understates this cell.

And the magnitude (0.0224) slightly exceeds benign's (0.0200) despite phishing's
decoder share being **24% against benign's 48%**. That is the case the
pre-registration said must be reported rather than absorbed: the decoder does
about the same amount of work with half the share, in the opposite direction. It
does not overturn Q1's decomposition, but it is not what the decomposition
predicted.

> **This result is fragile and must not be quoted as precise.** t = 4.54 barely
> clears t_crit = 4.303 at df = 2. It rests on three split seeds agreeing in sign
> (+0.0185, +0.0183, +0.0305), not on a large effect. The correlated-split-seed
> warning registered in section 7 applies: phishing's real pool is small, so this
> spread is a **LOWER BOUND** on the true uncertainty.

---

## PQ3 — generalisation. **PH3 REFUTED.**

PH3 was the lead this phase was built on: `E3_REPORT.md` records phishing QRY
val as HARDER than train (+0.162 NLL) while benign val is EASIER (-0.107), and
Q7 H2 refuted generalisation failure for benign only.

Registered prediction: if PH3 holds, the train/val hazard gap OPENS — train
tracks empirical (ratio -> 1.0) while val does not, i.e. **val minus train is
positive and large**.

Matched n (val 964 rows, train 956), class conditioning verified `phishing/name`:

| bucket | phishing val | phishing train | v-t | benign v-t | PH3? |
|---|---|---|---:|---:|---|
| 0-4 | 1.062 [0.99,1.14] | 1.219 [1.13,1.32] | **-0.157** | +0.091 | NO — train fits WORSE |
| 4-10 | 1.144 [1.03,1.28] | 1.161 [1.05,1.29] | -0.017 | +0.006 | no — flat |
| 10-25 | 1.019 [0.94,1.11] | 1.087 [1.01,1.17] | **-0.068** | -0.001 | NO — train fits WORSE |
| 25-50 | 1.112 [0.98,1.27] | 0.983 [0.88,1.09] | +0.129 | +0.038 | yes |
| 50-100 | UNMEASURED (131) | UNMEASURED (129) | — | -0.030 | — |
| 100-inf | UNMEASURED (47) | UNMEASURED (59) | — | — | — |

**Three of four measurable buckets go against PH3**, with train fitting *worse*
than val — the same under-fitting signature benign shows. At 0-4 the CIs are
nearly disjoint (overlap only 1.131-1.145), so that reversal is not noise. Only
25-50 goes PH3's way, with comfortably overlapping CIs.

**So the +0.162 NLL asymmetry is real but is NOT a termination effect.** Whatever
it is, it lives in which characters go into the query, not in when the query ends.
Termination in phishing looks like benign's: under-fitting, not memorisation.

**The registered limitation held.** 50-100 came back UNMEASURED on both sides
(131/129 against `MIN_ROWS` 200) — exactly the bucket where benign's deficit lived
(1.315), so PQ3 is silent on it. The pre-run estimate of ~62 rows was conservative
by ~2x but not enough to change the outcome.

**Power is ~5x lower than benign's** (964 rows vs 4,931). The three "tracks
empirical" readings partly reflect wide CIs rather than demonstrated tracking;
they are not evidence of health.

---

## PQ4 — not run

Lowest priority and explicitly deferrable in the registration, being axis A rather
than the convicted cell. `CHARACTERISATION.md` open question #2 — *why phishing
prefixes elicit low query probability* — **remains open and unexplained.**

---

## What this phase establishes

1. **Phishing QRY content is the same defect as benign QRY content**: shape, not
   vocabulary; distributed across components; dominated by parameter-name
   identity at nearly the same magnitude. The benign causal map transfers to
   content.
2. **The class differences are confined to emission.** Every difference the
   pre-registration cited is an axis-A quantity.
3. **Truncation masks part of the phishing content defect** rather than creating
   it. Production understates this cell.
4. **Phishing's termination behaviour is not a generalisation failure.** The NLL
   asymmetry is unexplained and is not about termination.

## What it does NOT establish

- Nothing about phishing **PTH**, the other convicted cell. Out of scope by
  registration.
- Nothing about whether any repair would move the gate. Untested, deliberately.
- **Why** phishing prefixes collapse (PQ4, open question #2).
- Anything in the 50-100 within-value bucket, which is unmeasurable at this pool
  size.

## The measurement-capacity problem stands

Registered in advance and unchanged by these results: at ~2.31% emission the cell
yields n_fake 180 at n_main 8000, below the canonical estimator's `MIN_N` of 250,
so it survives only on the pooled estimator — **which can convict or abstain but
never acquit.** Canonical measurability needs n_main ~11,100.

**A successful repair of this cell could not currently be VERIFIED as clearing
it.** That is logically prior to attempting one.

## Tooling changed during this phase, both verified before any number was read

    q7_hazard.py      class label was hard-coded "benign"; now derived from
                      --split, contradictions REFUSED. Verified bit-identical on
                      all three published benign artifacts (SHA-256).
    q3_1_resolve.py   cls was hard-coded "benign"; now --class. Verified
                      bit-identical on every statistic in the published benign
                      artifact (only the wall-clock `seconds` field differs).
