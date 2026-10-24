# U1 results

- **Registration:** `research/u1/PREREG_U1.md` (Amendments 1 and 2); Q1's implementation in
  `research/u1/Q1_PROTOCOL.md`.
- **Status:**
  - **Q1 READ 2026-10-21.**
  - **Q2 READ 2026-10-23** (with Q2 Amendment 1).
  - **Q3 READ 2026-10-24** (in LIMNEX: `phishguard-ml/research/u1_q3/RESULTS.md`, `RESULTS.json`).
- **Numbers:**
  - Q1: `data/u1/q1/Q1_REPORT.json` (git-ignored, with the generated files), produced by
    `research/u1/q1_score.py report`.
  - Q2: `research/u1/Q2_REPORT.json` (committed; run keys and numbers only, no URLs), produced by
    `research/u1/q2_detect.py test` on Kaggle from `F2_SELECTION.json`.
  - Q3: LIMNEX's `research/u1_q3/RESULTS.json` (numbers only), produced by `scripts/u1_q3_read.py read`.

---

## Q1: is UIE more realistic than simpler generators?

### In plain words
Five generators wrote 8,000 URLs × 5 generations per class from the same real training data, and
every URL passed the same known-URL filter. The same field-by-field test UIE was graded with in E3
then tried to tell each generator's URLs from real held-out URLs. **UIE was the most realistic
generator on every field where the test could tell them apart.** Out of 31 computable comparisons,
UIE was shown more realistic in 28. The other 3 were ties, and no baseline came out ahead of UIE on
any field.

The general-purpose LLM (Qwen2.5-7B-Instruct, shown 20 real examples per request) was the
**least** realistic of the learned generators. Its URLs look tidy and plausible to a person, but
they are stereotyped: few suffixes, few paths and queries in phishing-style URLs. Detectors
separate them from real URLs about as easily as the rule-based look-alikes.

### The generators
| id | generator | trained or prompted on | compute |
|---|---|---|---|
| `uie` | UIE `e16_b5`, free mode, fresh seeds from `30_000_000` | (released model) | 99 min generation, Kaggle GPU |
| `ngram` | B1, `NgramURLModel(order=5)` | UIE's training rows | 3 min, Kaggle CPU |
| `lstm` | B2, char LSTM 2×512, UIE's tokenizer | UIE's training rows | 2 GPU-hours training (47,182 steps, 4 epochs; val loss 1.656 → 1.533, still falling, never used to select) + 2 min generation, one T4 |
| `llm` | B3, `Qwen/Qwen2.5-7B-Instruct@a09a3545`, `transformers` backend, Appendix A prompt | 20 training rows per request | 6.8 h generation, 2×T4 |
| `twist` | B4, dnstwist-style look-alikes (phishing only) | benign training domains + top-500 phishing path templates | seconds, laptop |

Scoring ran on the laptop, about 30–70 min per generator.

### The readout (registered: §3)
x(UIE) − x(B) per field, Welch interval over 5 + 5 generation means. **UIE** = "UIE more realistic"
(one-sided 95% upper bound below 0); **tie** = not shown; **n/c** = not computable. Cells give the
difference, with the upper bound in brackets. FULL is auxiliary.

**Benign**
| UIE vs | LBL | SUF | SUB | PTH | QRY | FULL (aux.) |
|---|---|---|---|---|---|---|
| ngram | −0.216 [−0.207] **UIE** | +0.004 [+0.012] tie | −0.348 [−0.343] **UIE** | −0.266 [−0.250] **UIE** | −0.269 [−0.256] **UIE** | −0.310 [−0.302] |
| lstm | −0.024 [−0.018] **UIE** | −0.022 [−0.011] **UIE** | −0.004 [+0.000] tie | −0.086 [−0.073] **UIE** | −0.047 [−0.034] **UIE** | −0.136 [−0.128] |
| llm | −0.381 [−0.373] **UIE** | −0.288 [−0.281] **UIE** | −0.041 [−0.036] **UIE** | −0.351 [−0.339] **UIE** | n/c | −0.409 [−0.399] |

**Phishing**
| UIE vs | LBL | SUF | SUB | PTH | QRY | FULL (aux.) |
|---|---|---|---|---|---|---|
| ngram | −0.326 [−0.316] **UIE** | +0.004 [+0.013] tie | −0.254 [−0.243] **UIE** | −0.300 [−0.289] **UIE** | −0.223 [−0.202] **UIE** | −0.388 [−0.378] |
| lstm | −0.052 [−0.041] **UIE** | −0.010 [−0.002] **UIE** | −0.084 [−0.073] **UIE** | −0.146 [−0.136] **UIE** | −0.179 [−0.158] **UIE** | −0.107 [−0.097] |
| llm | −0.325 [−0.315] **UIE** | −0.171 [−0.159] **UIE** | −0.396 [−0.385] **UIE** | −0.273 [−0.261] **UIE** | n/c | −0.416 [−0.407] |
| twist | −0.349 [−0.339] **UIE** | −0.340 [−0.331] **UIE** | n/c | −0.212 [−0.202] **UIE** | n/c | −0.424 [−0.414] |

**Count (primary fields):** 28 UIE, 3 ties, 0 baseline ahead, 4 not computable.

- **The three ties:**
  - **SUF vs n-gram, in both classes:** the n-gram samples suffixes from the training table, as UIE
    does, so both clear the field.
  - **Benign SUB vs LSTM:** both clear it.
- **Not computable:**
  - **The LLM's QRY, in both classes:** it writes a query in only 1.9% (benign) and 1.1%
    (phishing) of URLs, against 9.6% and 6.3% in real data. That gives too few query URLs per
    generation for the per-generation test.
  - **Twist's SUB and QRY:** it writes no subdomains, and almost no queries.
  - The gate's pooled fallback (all 5 generations together, reduced n) reads them as TELL:
    - LLM QRY: +0.098 (benign), +0.184 (phishing);
    - twist QRY: +0.432.

  These pooled readings are not part of the registered comparison.

### Each generator's own cells (E3's verdicts: cl = CLEARED, U = UNDECIDED, C = CONVICTED)
| class | generator | LBL | SUF | SUB | PTH | QRY | FULL (aux.) |
|---|---|---|---|---|---|---|---|
| benign | **uie** | −0.001 cl | +0.016 cl | +0.003 cl | +0.009 cl | +0.046 U | +0.014 |
| benign | ngram | +0.214 C | +0.012 cl | +0.351 C | +0.274 C | +0.315 C | +0.324 |
| benign | lstm | +0.023 cl | +0.038 U | +0.007 cl | +0.095 C | +0.093 C | +0.150 |
| benign | llm | +0.379 C | +0.304 C | +0.044 cl | +0.360 C | n/c (pooled +0.098 TELL) | +0.423 |
| phishing | **uie** | +0.012 cl | +0.015 cl | +0.014 cl | +0.029 cl | +0.100 C | +0.019 |
| phishing | ngram | +0.338 C | +0.012 cl | +0.268 C | +0.328 C | +0.323 C | +0.407 |
| phishing | lstm | +0.064 C | +0.026 cl | +0.098 C | +0.175 C | +0.279 C | +0.126 |
| phishing | llm | +0.337 C | +0.186 C | +0.410 C | +0.302 C | n/c (pooled +0.184 TELL) | +0.435 |
| phishing | twist | +0.360 C | +0.355 C | n/c | +0.240 C | n/c (pooled +0.432 TELL) | +0.443 |

- **UIE with fresh seeds** clears 8 of 10 primary cells.
  - Phishing QRY is CONVICTED (+0.100), the weakness E3's final report names.
  - Benign QRY is UNDECIDED (+0.046).
- **The best baseline is the LSTM.** It clears 3 cells (benign LBL and SUB, phishing SUF), and is
  convicted on paths and queries in both classes.

### Also reported (§3)
| generator | class | validity | copy rate | domains / 8,000 | suffix entropy (bits) | share with SUB / PTH / QRY |
|---|---|---|---|---|---|---|
| **real** (val pools) | benign | | | 6,065 | 6.51 | 0.450 / 0.881 / 0.096 |
| uie | benign | 0.9988 | 0.0013 | 7,998 | 6.56 | 0.413 / 0.734 / 0.068 |
| ngram | benign | 0.9997 | 0.0001 | 7,394 | 6.57 | 0.314 / 0.967 / 0.108 |
| lstm | benign | 0.9910 | 0.0002 | 7,990 | 6.50 | 0.430 / 0.888 / 0.136 |
| llm | benign | 0.9893 | 0.0017 | 7,477 | 3.64 | 0.424 / 0.898 / 0.019 |
| **real** (val pools) | phishing | | | 3,343 | 4.41 | 0.759 / 0.494 / 0.063 |
| uie | phishing | 0.9938 | 0.0127 | 3,832 | 4.51 | 0.790 / 0.397 / 0.050 |
| ngram | phishing | 0.9938 | 0.0003 | 6,202 | 4.44 | 0.633 / 0.914 / 0.059 |
| lstm | phishing | 0.9824 | 0.0013 | 4,484 | 4.41 | 0.723 / 0.527 / 0.068 |
| llm | phishing | 0.9754 | 0.0221 | 5,315 | 4.29 | 0.591 / 0.163 / 0.011 |
| twist | phishing | 1.0000 | 0.0029 | 7,997 | 6.54 | 0.000 / 0.282 / 0.012 |

- **Copies:**
  - Every copy of a training URL is removed by the known-URL filter before scoring.
  - UIE's phishing copy rate (1.3%) is second only to the LLM's (2.2%).
- **Domain variety:**
  - Real phishing repeats domains (3,343 per 8,000, many URLs on the same platform hosts).
  - UIE comes closest (3,832); the n-gram (6,202) and the LLM (5,315) spread far wider than real
    data.
- **The LLM's benign suffixes are concentrated** (3.64 bits against 6.51 real): it falls back on a
  few common suffixes.

**B3 refusals (§6):**
- 3 of 4,480 responses (0.07%) gave no valid URL: 1 of 2,240 benign (0.04%) and 2 of 2,240
  phishing-style (0.09%).
- Two earlier B3 attempts ended before completing, and nothing from them was scored. The first
  failed on a vLLM install; the second on out-of-memory handling. See `Q1_PROTOCOL.md`'s note and commits
  57567f8, 9004de6 and 65d10e9.

**The word tell (E18's detector, informational):** excess over the real-vs-real floor, mean of 5
generations.

| | uie | lstm | llm | ngram | twist |
|---|---|---|---|---|---|
| benign | **+0.164** | +0.285 | +0.393 | +0.459 | — |
| phishing | **+0.167** | +0.316 | +0.404 | +0.469 | +0.481 |

### How far this result goes
- **UIE was developed against this instrument; the baselines were not.** E3 chose `e16_b5` with
  these field detectors and these validation pools. The fresh generation seeds rule out
  sampling luck, but not the home advantage. The word tell is the independent check:
  - E18's detector was built after `e16_b5` was adopted and never used to select it;
  - it gives the same order (UIE < LSTM < LLM < n-gram < twist);
  - so UIE's lead is not only an artefact of the instrument.
- **Q1 measures realism, not usefulness.** A less realistic generator can still help a detector,
  for example through variety. Whether UIE's data helps detectors, and helps more than B1–B3, is
  Q2's question (§4.6, S1 and S2). Q1 does not change Q2: every baseline B1–B3 enters it as
  registered.
- **One LLM, one prompt.** B3 is a single 7B instruct model, with the registered prompt and 20
  examples per request. A larger model, or a prompt engineered for realism, could do better.
  Q1 does not claim UIE beats LLMs in general.
- **The pooled readings** are not registered comparisons, and are quoted for completeness only.

---

## Q2: does UIE's data help detectors, and more than simpler generators?

### In plain words
Two detectors learned to flag phishing URLs from small, medium and large real training sets (1k,
10k and 100k URLs per class). Then the same detectors learned again with synthetic URLs added
(equally per class) from UIE or from one of the three learned baselines. All of them were scored
once on held-out phishing that appeared **after** UIE's training data ended.

- **UIE's URLs helped every time (S1: 6 of 6 tests).**
  - Phishing caught at 1% false alarms rose by **+2.7 to +24 points** on the primary test set.
  - The gain was largest when real data was scarce.
  - In each test, at most 1 of the 2,000 bootstrap resamples had a gain at or below zero.
- **Helping more than the best simpler generator is mostly NOT shown on the primary test set
  (S2: 1 of 6).**
  - UIE comes out ahead in 5 of the 6 settings, but only by 0.7 to 2.5 points, and only one lead
    survives the correction for multiple tests (the n-gram LR at 100k).
  - The LSTM, the best baseline in Q1, is usually the closest competitor.
  - On the registered window (T1, secondary), S2 holds in 4 of 6.
- **On other test sets, UIE's help is consistent, but its lead over the others is not:**
  - on T3, UIE is ahead in all 6 settings;
  - on PhreshPhish, the LLM's URLs help more than UIE's in 4 of 6.
- **The registered claims (§4.6):**
  - *"UIE's data improves detectors"*: for both detectors at every regime, on T1x.
  - *"UIE's data improves them more than simpler generators"*: only for the n-gram LR at 100k
    real URLs per class, on T1x.

### What ran (`Q2_PROTOCOL.md`, Amendment 1)
- **Detectors:**
  - **D1:** E18's character CNN, with the head's activation LeakyReLU(0.01) (Amendment 1); 5 seeds.
  - **D2:** a hashed character 1–5-gram logistic regression, fitted once (it is deterministic).
- **Arms, per regime R:**
  - A0 (R real URLs);
  - A_G(r) (R real + r·R from generator G, r ∈ {0.25, 0.5, 1.0});
  - A_S(G) (R synthetic only);
  - A_more (2R real).
- **Selection:** r\* was chosen on validation and frozen in F2 (69cbb95). UIE's r\* is 1.0 in every
  regime for both detectors.
- **The test read:**
  - Once, on Kaggle (one T4, about 10 minutes): 300 arm × test-set cells, 900 per-seed scores.
  - Bootstrap: 2,000 resamples by registrable domain.
  - Holm correction across the 6 tests of each criterion.
- **Test sets (phishing / benign):**
  - **T1x primary:** 24,588 / 53,482.
  - **T1 secondary:** 5,372 / 26,860.
  - **PhreshPhish:** 46,294 / 46,699.
  - **PhiUSIIL:** 97,018 / 127,910.
  - **T3:** 24,082 / 113,277.

### The registered readout (§4.6): recall at 1% FPR
Cells give Δ, with the unadjusted one-sided 95% lower bound in brackets and the Holm-adjusted p.
**holds** = Holm p < 0.05. Here 0.003 is the smallest a Holm-adjusted p can be (6 × 1/2,001).

**S1: A_UIE(r\*) − A0**
| detector | regime | T1x (primary) | T1 (secondary) |
|---|---|---|---|
| CNN | 1k | +0.242 [+0.089] p 0.003 **holds** | +0.411 [+0.196] p 0.003 **holds** |
| CNN | 10k | +0.027 [+0.015] p 0.003 **holds** | +0.028 [+0.017] p 0.003 **holds** |
| CNN | 100k | +0.035 [+0.022] p 0.003 **holds** | +0.026 [+0.014] p 0.003 **holds** |
| LR | 1k | +0.109 [+0.058] p 0.003 **holds** | +0.208 [+0.089] p 0.003 **holds** |
| LR | 10k | +0.124 [+0.064] p 0.003 **holds** | +0.124 [+0.077] p 0.003 **holds** |
| LR | 100k | +0.040 [+0.030] p 0.003 **holds** | +0.046 [+0.026] p 0.003 **holds** |

**S2: A_UIE(r\*) − the best of A_B(r\*_B), B ∈ {ngram, lstm, llm}** (the best baseline is re-taken
in every resample)
| detector | regime | best other on T1x | T1x (primary) | T1 (secondary) |
|---|---|---|---|---|
| CNN | 1k | lstm | +0.007 [−0.063] p 1.000 | +0.046 [−0.001] p 0.111 |
| CNN | 10k | lstm | −0.003 [−0.011] p 1.000 | +0.024 [+0.013] p 0.003 **holds** |
| CNN | 100k | llm | +0.015 [+0.003] p 0.150 | +0.026 [+0.014] p 0.003 **holds** |
| LR | 1k | lstm | +0.023 [−0.014] p 0.648 | +0.002 [−0.064] p 0.454 |
| LR | 10k | lstm | +0.014 [−0.019] p 1.000 | +0.044 [+0.029] p 0.003 **holds** |
| LR | 100k | lstm | +0.025 [+0.019] p 0.003 **holds** | +0.035 [+0.018] p 0.003 **holds** |

**Count:**
- S1: 6 of 6 hold on T1x, and 6 of 6 on T1.
- S2: 1 of 6 holds on T1x, and 4 of 6 on T1.
- CNN at 100k has an unadjusted lower bound above zero (+0.003) on T1x, but does not survive Holm.

### The arms on T1x: recall at 1% FPR, mean over seeds
| detector | regime | A0 (real R) | A_more (real 2R) | + UIE at r\* | + best other at r\* | UIE only (R) |
|---|---|---|---|---|---|---|
| CNN | 1k | 0.148 | 0.415 | 0.390 | 0.383 (lstm) | 0.342 |
| CNN | 10k | 0.606 | 0.614 | 0.633 | 0.637 (lstm) | 0.621 |
| CNN | 100k | 0.715 | 0.745 | 0.750 | 0.735 (llm) | 0.718 |
| LR | 1k | 0.209 | 0.245 | 0.318 | 0.295 (lstm) | 0.341 |
| LR | 10k | 0.393 | 0.468 | 0.517 | 0.503 (lstm) | 0.516 |
| LR | 100k | 0.634 | 0.676 | 0.674 | 0.649 (lstm) | 0.660 |

- **Against more real data (A_more, never a criterion, §4.6):** adding R UIE URLs beats adding R
  more real URLs in 4 of 6 settings on T1x, and in 6 of 6 on T1. On T1x it falls short at CNN 1k
  (0.390 against 0.415) and ties at LR 100k (0.674 against 0.676).
- **UIE alone (A_S(UIE)) against the same number of real URLs (A0):**
  - UIE alone is at least as good in every setting on T1x: +0.19 (CNN) and +0.13 (LR) at 1k,
    +0.12 at LR 10k, and level at CNN 100k (0.718 against 0.715).
  - Among the synthetic-only sets, UIE's is the best at 1k for both detectors and for the LR at
    every size. For the CNN at 10k and 100k, the LSTM's is level with it or slightly above (0.633
    against 0.621, 0.721 against 0.718).
- **Steadier training:** for the CNN on T1x, adding UIE roughly halves the spread over seeds (sd 0.130
  → 0.077 at 1k, 0.052 → 0.028 at 10k, 0.037 → 0.020 at 100k).
- **AUC moves the same way:** CNN 1k 0.897 → 0.946; LR 100k 0.975 → 0.978.

### Beside every claim (§4.6, not tested): the other test sets
Δ in recall at 1% FPR, means over seeds.

| detector | regime | PhreshPhish S1 / S2 (best other) | PhiUSIIL S1 / S2 (best other) | T3 S1 / S2 (best other) |
|---|---|---|---|---|
| CNN | 1k | +0.159 / −0.056 (llm) | +0.096 / −0.044 (ngram) | +0.261 / +0.051 (ngram) |
| CNN | 10k | +0.105 / +0.040 (llm) | +0.090 / +0.024 (ngram) | +0.051 / +0.042 (lstm) |
| CNN | 100k | +0.026 / +0.013 (llm) | +0.029 / +0.001 (ngram) | +0.044 / +0.035 (lstm) |
| LR | 1k | +0.071 / −0.043 (llm) | +0.079 / −0.037 (ngram) | +0.143 / +0.074 (llm) |
| LR | 10k | +0.092 / −0.010 (llm) | +0.067 / +0.004 (ngram) | +0.118 / +0.064 (lstm) |
| LR | 100k | +0.052 / −0.034 (llm) | +0.047 / +0.019 (llm) | +0.072 / +0.054 (lstm) |

- **S1 generalises:** UIE's data helps on every other test set, in all 18 settings.
- **S2 does not generalise to T2:**
  - On PhreshPhish, the LLM's data helps more than UIE's in 4 of 6 settings: the CNN at 1k, and
    the LR at every size.
  - On PhiUSIIL, the n-gram's helps more at 1k for both detectors. Elsewhere UIE leads by 0.1 to
    2.4 points.
  - On T3, UIE leads in all 6 settings.

### Overlap with the test sets (§1.3, reported, not filtered)
- **The share of each synthetic set whose host (phishing) or registrable domain (benign) occurs in a
  test set is at most 0.6%:** the LLM's benign domains in T2, which are mostly very common domains.
- **Phishing overlap is at most 0.08%** (the n-gram on PhreshPhish).
- The LLM's PhreshPhish lead is therefore not explained by copied test hosts.

### How far this result goes
- **"Synthetic beats real" is not a free lunch.** UIE learned from far more real URLs than the 1k–100k
  each arm adds, and its samples carry that knowledge. A_more is the reference for *how close*
  synthetic data comes to extra real data, not a fair contest. The fair contest is S2, where every
  generator learned from the same training rows.
- **S2's differences are small.**
  - At 1k, seed noise is large: a CNN's recall@1% ranges from 0.28 to 0.49 across the 5 seeds of
    one arm.
  - The registered precision on T1x is about 1 point, so leads of 1–2 points mostly stay unproven.
  - The registration expected S2's differences to be small.
- **Realism is not usefulness (Q1's caveat, now observed):**
  - The LLM was the least realistic learned generator in Q1, yet its URLs help detectors more than
    UIE's on PhreshPhish in 4 of 6 settings.
  - One possible reason, **untested**: its stereotyped, keyword-heavy phishing URLs may resemble
    PhreshPhish's phishing more than they resemble the feed-based T1x.
  - Q2 does not show which property of a synthetic set makes it useful.
- **Amendment 1:** D1's recipe was changed before F2, after its registered runs died (29 of 270).
  The amendment declares everything seen before it (D2's validation results, D1's broken
  validation results, the diagnosis and the fix check). The test sets were read once, after F2.
- **r\* = 1.0 is the top of the registered grid** for UIE everywhere. More synthetic data might help
  further; that is untested here.
- **Q3 is triggered:**
  - S1 holds for both detector families.
  - Per the LIMNEX registration (`phishguard-ml/research/u1_q3/PREREG_U1_Q3.md`), Q3 retrains
    LIMNEX's CNN and CatBoost members with UIE data and reads LIMNEX's frozen TEST once.

---

## Q3: does UIE's data help the real product, LIMNEX?

### In plain words
- **What ran:**
  - LIMNEX's CNN and CatBoost members were retrained with UIE's URLs added: 511,378 per class,
    about as many as LIMNEX's own ~1M real training URLs, by Q2's rule (r\* = 1.0).
  - Everything else stayed as in the served v3.7: the other two members, the fusion weights, every
    hyperparameter.
  - LIMNEX's frozen external TEST (41,808 URLs, PhiUSIIL + PhreshPhish) was then read once.
- **Q3 HOLDS.** Phishing caught at 1% false alarms rose from **27.2% to 54.4%** (+27.2 points,
  95% CI +15.1 to +40.9; one-sided lower bound +16.0, above 0 as registered).
  - ROC-AUC rose by 0.024, PR-AUC by 0.038, and recall at 5% FPR by 11 points.
- **The same two members got WORSE on LIMNEX's own validation fold:** CNN −2.4 points, CatBoost
  −6.0 points of recall@1%.
  - That fold comes from LIMNEX's older training sources, whose phishing URLs are long and full of
    paths and queries.
  - UIE's phishing, like the current feed it learned from, is mostly bare domains.
  - So the URLs teach a newer style. It costs a little on the old style and helps a lot on the
    external test.
- **Much of the gain is one phishing family:** look-alike sites on Blogger's country domains
  (blogspot.ro, .de, …). They are about a quarter of TEST's phishing; v3.7 caught 1.7% of them.
  - UIE produces this pattern only as phishing.
  - On TEST domains that never occur in UIE's rows, the gain is still +15.6 points, but its
    confidence interval just touches zero (exploratory, not registered).

### The registered readout (LIMNEX `PREREG_U1_Q3.md` §5)

| TEST | v3.7 | A = v3.7's weights with the Q3 CNN and CatBoost | Δ recall@1% | 95% CI | one-sided lower |
|---|---|---|---|---|---|
| recall@1% FPR | 0.2720 | **0.5442** | **+0.272** | [+0.151, +0.409] | **+0.160** |

- 2,000 resamples (seed 20261027). Phishing and benign are resampled separately by registrable
  domain (`uie.splits.group_key`), and every threshold is recomputed.
- One configuration, one test.

**Reported, deciding nothing:**
- **B** (F6, the fusion with UIE's scorer as a member, with the Q3 members): 0.4276 vs 0.4012,
  +0.026 [+0.008, +0.043]. That is smaller than A's gain, as registered: the scorer and the data
  partly carry the same signal.
- **Members alone on TEST:**
  - CNN 0.536 vs 0.219 (+0.317);
  - CatBoost 0.372 vs 0.242 (+0.131).
- **Members alone on the validation fold:**
  - CNN 0.754 vs 0.778 (−0.024 [−0.032, −0.016]);
  - CatBoost 0.495 vs 0.555 (−0.060 [−0.078, −0.047]).
  - Neither is broken (Q2's tie check).
- **The registered slices:**
  - TEST's UIE-disjoint rows: +0.273.
  - Rows whose host never occurs in the synthetic rows: +0.274.
  - Both cover 98–99.6% of TEST, so they say little about overlap.
- **IDN:**
  - A flags fewer of 1,455 real IDN websites: 46.4% vs 53.6% (McNemar p < 0.001).
  - B flags more: 48.2% vs 43.5% (p < 0.001).
  - Homograph recall is unchanged (0.988 / 0.994).

### How far this result goes
- **The registered prediction was wrong in size:** +0.00 to +0.03 was expected, and +0.27 came
  out. Q2 had shown gains shrink as real data grows, so a large effect at LIMNEX's ~1M rows was
  not expected.
  - The validation drop points to the likely reason: LIMNEX's corpus is older than the phishing
    on TEST.
  - So UIE's data acts less as "more data" and more as **current data**.
- **Concentration (exploratory):** Blogger ccTLD phishing carries much of the gain. Without those
  domains (the TEST rows whose domain never occurs in UIE's rows), the point gain stays large
  (+0.156) but is not established (lower bound −0.004).
- **One benchmark, one seed per member.** Seed variance at this scale is not measured.
- **A deviation before the read:** the rehearsal of the read printed the members-alone TEST numbers.
  The primary Δ was not computed early, and nothing registered could change. It is recorded in
  LIMNEX's `Q3_PROTOCOL.md` §6.
- **Not an adoption.** The served LIMNEX stays v3.7. Adopting the Q3 members is the owner's
  decision, through LIMNEX's own build and gate path. That path includes the in-distribution cost
  above and the IDN false-alarm checks.
