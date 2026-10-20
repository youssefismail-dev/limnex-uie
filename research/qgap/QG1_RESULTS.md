# QG1 — results: where the query-emission gap is lost

- **Registration:** `QG_PLAN.md` §5 (73774cf), before any code. **Code:** 144e4ea,
  committed before any reading.
- **Read:** 2026-09-28, after VQ1–VQ4 passed.
- **Numbers:** the numbers in the text come from Appendix A, which is
  `qg1_read.py`'s output pasted by script.
- **Before the reading,** a timing dry run of 1,500 rows per set (e9_ln only)
  exercised the code. Its numbers are not this reading and were not used.

## In plain words

- **The gap is in the model's decision itself.**
  - Shown the beginning of a real URL, the baseline gives "start a query now" about
    a quarter too little probability for benign (decision factor 0.735) and 29% too
    little for phishing (0.708).
  - That alone accounts for about 90% of the benign gap and 80% of the phishing gap.
- **The rest is not where the gap is.**
  - The model's own URL beginnings attract almost as much query probability as real
    ones (0.979 benign, 0.964 phishing).
  - The sampler draws faithfully (0.981 and 0.951; both intervals include 1).
- **It is not a generalisation failure.** The deficit is there on the rows the model
  trained on (0.735) and on held-out rows (0.704).
- **It is worst right after the site's name.** When a query follows the host with
  no path, the model gives about half the right probability (0.539 benign, 0.592
  phishing). After a path it gives 0.806 and 0.780.
- **E9's objective is not what fixes this decision.**
  - From the same start, the old objective's benign decision factor was 0.725; the
    new one is 0.735.
  - E9's emission gain came from the model's own starts (0.740 → 0.979): its paths
    are now written at the right rate.
  - For phishing the objective moved the decision part way (0.597 → 0.708).
- **So the remaining gap has a cause other than the length weighting, and that
  cause is Unknown.**
- **The registered reading is DECISION-LED** in both classes. It licenses QG2-D, a
  laptop diagnostic that asks which cause it is. It does not license any fix.

## 1. Validity

| check | result |
|---|---|
| VQ1 | every model verified by fingerprint or registered hash, and every generated file by E9's provenance, before measuring (`qg1_measure.py`) |
| VQ2, VQ3 | `tests/test_qgap.py`, 10 of 10 |
| VQ4 | post-stratified D is 6.36% (benign) and 4.37% (phishing), against FINDING §2.3's 6.53% and 4.51%: within the registered ±0.3 points; PASS |

The sample's D is used in the decomposition. With FINDING's D, G / D would be 0.70
instead of 0.706, and no conclusion changes.

## 2. The decomposition

Everything is from Appendix A, the "decomposition" and "by route" tables.

| | benign | phishing |
|---|---|---|
| G / D | 0.706 | 0.649 |
| **decision** T / D | **0.735** [0.709, 0.761] | **0.708** [0.674, 0.744] |
| own starts S / T | 0.979 [0.939, 1.018] | 0.964 [0.923, 1.004] |
| sampler G / S | 0.981 [0.926, 1.036] | 0.951 [0.882, 1.024] |
| share of the gap: decision / own starts / sampler | 0.89 / 0.06 / 0.05 | 0.80 / 0.09 / 0.12 |

**Reading (QG_PLAN §5.6), rows in their registered order:**
- **SAMPLER:** does not hold. G / S's CI includes 1 in both classes.
- **DECISION-LED:** holds. The decision factor's share is at least 0.5, and T / D's
  CI excludes 1.
- **→ DECISION-LED,** benign deciding. Phishing agrees.

**By route:**

| | host route (a query right after the host) | path route (a query after a path) |
|---|---|---|
| benign: share of real queries on this route | 1.68% of URLs, of D 6.36% | 4.69% |
| benign decision | **0.539** [0.500, 0.579] | 0.806 [0.775, 0.838] |
| benign part of the gap | 0.38 [0.30, 0.49] | 0.62 [0.51, 0.70] |
| phishing decision | **0.592** [0.547, 0.641] | 0.780 [0.738, 0.824] |
| phishing part of the gap | 0.50 [0.41, 0.61] | 0.50 [0.39, 0.59] |

- **The host route carries more than its share of the gap.** It has a quarter of
  benign real queries, 1.68 of 6.36 points, but 38% of the gap.
- **The model's own hosts attract slightly more query probability than real ones**
  (benign own-starts factor on the host route 1.081 [1.065, 1.098]). So the deficit
  on this route lies in the decision, not in the hosts the model writes.

## 3. What the contrasts say

**The objective.** `e9_base` is E7's objective from the same start, so the ratios
are paired.

| factor | benign: e9_base → e9_ln (ratio) | phishing: e9_base → e9_ln (ratio) |
|---|---|---|
| decision | 0.725 → 0.735 (1.014 [1.006, 1.022]) | 0.597 → 0.708 (1.186 [1.167, 1.206]) |
| own starts | 0.740 → 0.979 (1.323 [1.259, 1.390]) | 0.673 → 0.964 (1.431 [1.354, 1.514]) |
| sampler | 1.001 → 0.981 | 0.969 → 0.951 |

- **What E9 did.** Its objective restored the model's own starts: its paths are
  written at the data's rate (E9 K9a, K9b paths). It left the benign query-start
  decision almost where it was.
- **A second old-objective training agrees.** E7's decision factor is 0.703
  (benign) and 0.525 (phishing). The benign deficit is the same across three
  trainings and two objectives; phishing's varies more.

**Capacity and generalisation.** Training against validation rows: benign 0.735
against 0.704, phishing 0.708 against 0.706. The CIs overlap. The model
under-predicts the query start on rows it trained on, so this is not a failure to
generalise.

**The data side** (the whole pools):
- The band centres (6.62% / 4.68%) are all `name_train` rows; FINDING's D (6.53% /
  4.51%) is the rows within 256 tokens.
- Validation rows have the same rates.
- The page frame carries far more queries (plain 10.32% / 12.28%) than the domain
  frame (6.37% / 4.29%).
- So a frame blur that drew the model toward the page frame's overall rate would
  raise its domain-frame rate, not lower it. Whether a blur acts per route is not
  measured here.

**Decoder and sampling:**
- 0.01% of generated URLs carry an empty `?`.
- 4 and 5 of 20,000 sampled URLs could not be re-read.
- The re-encoded query flag equals the gate's on 100% of those that could.
- The per-generation spread (0.32 points benign) is close to its binomial size
  (0.23).
- Nothing here moves the gap.

## 4. The registered predictions (QG_PLAN §5.5), scored

| prediction | result | scored |
|---|---|---|
| G / S within [0.97, 1.03]; truncated plus empty queries under 0.1 points | 0.981 benign, 0.951 phishing; empty and unreadable about 0.03% | right for benign. **Wrong for phishing's point** (0.951), though its CI includes 1. The losses are right |
| T / D for e9_base below T / D for e9_ln | 0.725 < 0.735; 0.597 < 0.708 | right in direction. For benign the effect is 1.4%, far smaller than the gap |
| the length limit lowers D by 0.05–0.3 points | 0.09 (benign), 0.17 (phishing) | right |
| T / D for e9_ln within [0.80, 1.05]; no prediction of which factor leads | 0.735, 0.708 | **wrong: the decision deficit is larger than expected** |

## 5. What is and is not claimed

**Claimed** [CONFIRMED by the registered decomposition]:
- about 90% (benign) and 80% (phishing) of the query-emission gap is the model's
  query-start decision on real prefixes;
- the model's own prefixes and the sampler account for little of it;
- the deficit is present on training rows;
- it is largest on the host route.
- E9's objective did not change the benign decision factor materially.

**Not claimed:**
- **the cause of the decision deficit.** Data, conditioning, residual objective and
  capacity are not separated by QG1 (QG_PLAN §3). That is QG2-D.
- **anything about query content;**
- **anything about other trainings.** QG1 reads one training per objective. E7 and
  `e9_base` give the spread of two old-objective trainings.

## 6. What the reading licenses

**QG2-D** (QG_PLAN §6): locate the decision deficit, with laptop diagnostics only.
- It will be registered before it runs: hypotheses, predictions, a decision rule.
- Nothing changes the baseline.
- A training arm, whatever the finding, needs the owner's approval.

## Appendix A — the rendered tables

`python research/qgap/qg1_read.py`, verbatim.

### Validity

- VQ4 benign: post-stratified D 6.36% against FINDING's 6.53%: **PASS**
- VQ4 phishing: post-stratified D 4.37% against FINDING's 4.51%: **PASS**
- VQ1: every model and file was verified before measuring (qg1_measure.py); VQ2-VQ3: tests/test_qgap.py

### The decomposition (the baseline, `e9_ln` / raw)

| class | D | T | S | G | G / D |
|---|---|---|---|---|---|
| benign | 6.36% [6.12%, 6.62%] | 4.68% [4.55%, 4.79%] | 4.58% [4.43%, 4.73%] | 4.50% [4.21%, 4.76%] | 0.706 [0.654, 0.756] |
| phishing | 4.37% [4.17%, 4.57%] | 3.09% [3.00%, 3.17%] | 2.98% [2.88%, 3.07%] | 2.83% [2.63%, 3.05%] | 0.649 [0.593, 0.710] |

| class | decision T / D | own starts S / T | sampler G / S | shares (decision / own starts / sampler) | reading |
|---|---|---|---|---|---|
| benign | 0.735 [0.709, 0.761] | 0.979 [0.939, 1.018] | 0.981 [0.926, 1.036] | 0.89 [0.73, 1.08] / 0.06 [-0.06, 0.16] / 0.05 [-0.12, 0.19] | **DECISION-LED** |
| phishing | 0.708 [0.674, 0.744] | 0.964 [0.923, 1.004] | 0.951 [0.882, 1.024] | 0.80 [0.66, 0.98] / 0.09 [-0.01, 0.18] / 0.12 [-0.07, 0.26] | **DECISION-LED** |

### By route (`e9_ln`)

| class | route | D | T | S | G | decision | own starts | sampler | part of the gap |
|---|---|---|---|---|---|---|---|---|---|
| benign | host | 1.68% | 0.90% | 0.98% | 0.96% | 0.539 [0.500, 0.579] | 1.081 [1.065, 1.098] | 0.982 [0.850, 1.124] | 0.38 [0.30, 0.49] |
| benign | path | 4.69% | 3.77% | 3.60% | 3.54% | 0.806 [0.775, 0.838] | 0.955 [0.906, 1.003] | 0.981 [0.922, 1.046] | 0.62 [0.51, 0.70] |
| phishing | host | 1.66% | 0.98% | 0.97% | 0.89% | 0.592 [0.547, 0.641] | 0.987 [0.943, 1.036] | 0.920 [0.792, 1.049] | 0.50 [0.41, 0.61] |
| phishing | path | 2.70% | 2.11% | 2.01% | 1.94% | 0.780 [0.738, 0.824] | 0.953 [0.900, 1.009] | 0.965 [0.878, 1.057] | 0.50 [0.39, 0.59] |

### The objective: `e9_base` (same start, old objective) and the ratio `e9_ln` / `e9_base`

| class | factor | e9_base | e9_ln | e9_ln / e9_base |
|---|---|---|---|---|
| benign | decision | 0.725 [0.699, 0.752] | 0.735 [0.709, 0.761] | 1.014 [1.006, 1.022] |
| benign | own_starts | 0.740 [0.709, 0.776] | 0.979 [0.939, 1.018] | 1.323 [1.259, 1.390] |
| benign | sampler | 1.001 [0.931, 1.068] | 0.981 [0.926, 1.036] | 0.980 [0.901, 1.073] |
| phishing | decision | 0.597 [0.568, 0.626] | 0.708 [0.674, 0.744] | 1.186 [1.167, 1.206] |
| phishing | own_starts | 0.673 [0.641, 0.707] | 0.964 [0.923, 1.004] | 1.431 [1.354, 1.514] |
| phishing | sampler | 0.969 [0.870, 1.070] | 0.951 [0.882, 1.024] | 0.981 [0.857, 1.113] |

### Capacity: the decision on training and validation rows

| class | model | T / D, training rows | T / D, validation rows | D, validation |
|---|---|---|---|---|
| benign | e9_ln | 0.735 [0.709, 0.761] | 0.704 [0.656, 0.758] | 6.76% |
| benign | e9_base | 0.725 [0.699, 0.752] | 0.689 [0.644, 0.741] | 6.76% |
| benign | e7 | 0.703 [0.679, 0.729] | 0.678 [0.633, 0.731] | 6.76% |
| phishing | e9_ln | 0.708 [0.674, 0.744] | 0.706 [0.651, 0.771] | 4.48% |
| phishing | e9_base | 0.597 [0.568, 0.626] | 0.593 [0.547, 0.651] | 4.48% |
| phishing | e7 | 0.525 [0.499, 0.550] | 0.522 [0.481, 0.573] | 4.48% |

### Decoder and sampling (generated files)

| class | model | G, all 40,000 | per generation | SD (binomial SD) | empty `?` | unreadable in the subsample (their query rate) | re-encoded y = gate's |
|---|---|---|---|---|---|---|---|
| benign | e9_ln | 4.58% | 4.31%, 4.64%, 4.70%, 5.03%, 4.24% | 0.32% (0.23%) | 0.01% | 4 of 20,000 (50.00%) | 100.00% |
| benign | e9_base | 3.33% | 3.31%, 3.31%, 3.25%, 3.17%, 3.59% | 0.16% (0.20%) | 0.00% | 1 of 20,000 (0.00%) | 100.00% |
| phishing | e9_ln | 2.94% | 3.14%, 2.77%, 2.95%, 2.89%, 2.97% | 0.13% (0.19%) | 0.00% | 5 of 20,000 (40.00%) | 100.00% |
| phishing | e9_base | 1.72% | 1.85%, 1.71%, 1.69%, 1.60%, 1.74% | 0.09% (0.15%) | 0.01% | 2 of 20,000 (0.00%) | 100.00% |

### G against D by suffix group (`e9_ln`; groups with at least 100 query events on each side)

| class | group | real rows | D | generated | G | G / D |
|---|---|---|---|---|---|---|
| benign | com | 5,255 | 4.68% | 4,835 | 3.83% | 0.82 |
| benign | (other) | 17,801 | 7.45% | 18,972 | 5.09% | 0.68 |
| phishing | com | 10,328 | 6.58% | 11,047 | 3.93% | 0.60 |
| phishing | (other) | 6,144 | 5.66% | 6,874 | 4.86% | 0.86 |

### The data side (whole pools; `qg1_data.py`)

| class | pool | rows | plain, all | plain, fit | shipped, all | shipped, fit |
|---|---|---|---|---|---|---|
| benign | name_train | 906,214 | 6.37% | 6.29% | 6.62% | 6.53% |
| benign | name_val | 113,277 | 6.40% | 6.32% | 6.61% | 6.53% |
| benign | structure_train | 1,648,460 | 10.32% | 10.14% | 10.27% | 10.12% |
| phishing | phishing_name_train | 192,652 | 4.29% | 4.13% | 4.68% | 4.51% |
| phishing | phishing_name_val | 24,081 | 4.44% | 4.28% | 4.63% | 4.48% |
| phishing | phishing_structure_train | 46,491 | 12.28% | 11.59% | 11.63% | 10.97% |

### Reading

- benign (decides): **DECISION-LED**
- phishing: DECISION-LED
