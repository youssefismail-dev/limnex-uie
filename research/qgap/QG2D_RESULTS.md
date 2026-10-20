# QG2-D — results: where the query-start probability goes

- **Registration:** `PREREG_QG2D.md` (53049d3), before any code. Approved by the
  owner, laptop only.
- **Code:** bfd502d, committed before any reading. After it, one reader fix made
  intervals return "no estimate" for a category with no events. It came before
  any number was printed.
- **Read:** 2026-09-28, after the validity checks.
- **Numbers** come from three places:
  - Appendix A, `qg2_read.py`'s output pasted by script;
  - `data/qgap/qg2_audit.json`, for §2;
  - `data/qgap/qg2_report.json`, unrounded, for the model-to-data ratios in §4.
- **A 300-row dry run** exercised the code. It is not this reading.
- **The baseline** (`e9_ln` / raw) is only read, never changed.

## In plain words

- **The training code is not the cause** (the audit).
  - Every training row carries its right frame.
  - Each presence control is hidden exactly half the time, whatever the row holds.
  - The prefix the model trains on is token for token what generation gives it.
- **Mixing the two kinds of site is not the cause either.**
  - URLs from multi-page sites are under-predicted in the same way (0.479
    benign, 0.502 phishing, on the host route).
  - Pooling would have pushed them the other way.
- **Telling the model whether a path follows barely helps.** It closes only 20%
  (benign) and 13% (phishing) of the host-route deficit.
- **Telling the model that a query exists fixes it.**
  - The model then puts almost exactly one query's worth of probability into the
    URL (0.98 benign, 1.01 phishing).
  - It splits it between "after the host" and "after the path" nearly as the data
    does (0.238 against 0.264; 0.354 against 0.381).
- **So the model can write these queries; it under-chooses them when left to
  decide.** The registered reading is **QUERY-STATE**: the deficit lives in the
  model's decision under the hidden query control.
- **An observation, not a registered criterion.** Across every decision point
  measured, the rarer an outcome, the more the model under-predicts it:
  - common outcomes (38% and up) come out at 1.00–1.04 of the data's rate;
  - outcomes at 6–7% at 0.82–0.85;
  - outcomes at 1–2% at 0.47–0.64.
  - The query start (1–2% at the end of a host) is one of several rare outcomes on
    that curve. So the cause may be a general under-prediction of rare structural
    choices, not something special about queries. **This is a hypothesis for the
    next registration.**

## 1. Validity: PASS

| check | benign | phishing |
|---|---|---|
| VQ1 | the model fingerprint verified | the same |
| VP1: S0 against QG1 (±0.05) | host 0.541 against 0.539; path 0.806 against 0.806 | 0.592 against 0.592; 0.777 against 0.780 |
| VP2: max \|sum of categories − 1\| (≤ 1e-4) | 6.1e-07 | 5.9e-07 |
| VP3: mis-encoded controls | 0 | 0 |
| rows per slice | 40,000 | 40,000 |

## 2. Probe 3 — the audit: no asymmetry (A1–A5 pass)

| item | finding |
|---|---|
| A1 | 99,484 sampled training rows: 0 under the wrong frame |
| A2 | each control is hidden at 0.50 in every group (present or not, either frame, either class, the other control hidden or not). Every 95% CI includes 0.5. Both are hidden on 25.04% [24.77, 25.31]. The hidden token is free mode's `?` |
| A3 | generation's prefix is identical to training's hidden-control encoding, in both classes |
| A4 | `<QRY>` has weight 1 as a target in all 8,042 sampled query rows, both corpora |
| A5 | the host-route URLs (`host/?q`, `host?q`) get P = 0, Q = 1, matching their markers |
| A6 | free mode's exact slice (domain frame, both controls hidden) is 7.3% of benign training exposures and 3.5% of phishing ones |

## 3. Probe 1 — the host-route query decision by slice

| slice | benign | phishing |
|---|---|---|
| S0 free mode | 0.541 [0.506, 0.583] | 0.592 [0.550, 0.640] |
| S1 multi-page sites | 0.479 [0.448, 0.513] | 0.502 [0.457, 0.552] |
| S2 frame swapped to page | 0.664 [0.623, 0.716] | 0.806 [0.746, 0.878] |
| S3 path state told | 0.633 [0.593, 0.683] | 0.645 [0.600, 0.696] |
| S4 query state told | 1.168 [1.097, 1.250] | 1.298 [1.220, 1.379] |

- **S3's closure** of S0's deficit is 0.20 (benign) and 0.13 (phishing). The
  paired gains are 0.093 [0.079, 0.106] and 0.052 [0.040, 0.065]: real, but small.
- **S4, told a query exists:**
  - per row, the model spends 0.980 [0.952, 1.010] (benign) and 1.012 [0.974,
    1.048] (phishing) of a query's probability;
  - its host-route share is 0.238 [0.229, 0.247] against the data's 0.264, and
    0.354 [0.336, 0.376] against 0.381.
  - Told there is no query, it still spends 0.012 per row on one.
  - The host-route ratio above 1 in S4 reflects the told query's probability
    spread over the host positions before the model can know which route it takes.
    The per-row total and the split are the readings.
- **S2:** the page token raises the query decision on the same domain URLs, on the
  host route and on the path route (1.171 benign). So the model does condition on
  the frame.

**The path route (description):**

| slice | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| benign | 0.806 | 0.911 | 1.171 | 0.793 | 1.154 |
| phishing | 0.777 | 0.842 | 1.008 | 0.722 | 1.291 |

## 4. Probe 2 — what the model predicts instead

**At the end of a real host, in free mode (S0).** The model's split over path,
query, end and fragment, renormalised, against the data's:

| | path | query | end of URL | fragment |
|---|---|---|---|---|
| benign: model / data | 75.02% / 71.90% | 0.98% / 1.66% | 24.00% / 26.44% | 0.00% / 0.00% |
| phishing: model / data | 38.89% / 38.75% | 1.02% / 1.60% | 60.08% / 59.60% | 0.01% / 0.05% |

- **Benign:** the missing query mass, together with some of the end-of-URL mass,
  goes to "a path follows": 3.12 points [2.71, 3.54] too much path.
- **Phishing:** no one outcome takes it (path +0.14 points [−0.19, 0.46]).

**Per-category calibration over the whole host region (S0):**

| | QRY | PTH | EOS | SUB | content |
|---|---|---|---|---|---|
| benign | 0.541 | 0.973 | 0.832 | 0.935 | 1.008 |
| phishing | 0.592 | 0.938 | 0.930 | 0.996 | 1.004 |

The model gives slightly too much probability to "the host continues" (content
1.008 / 1.004). At a real host end, 45% (benign) and 54% (phishing) of its
probability still says the host goes on.

**At the end of a real path (S0):** query 5.51% against 6.45% (benign) and 5.50%
against 6.73% (phishing). The rest goes to the end of the URL.

**The observation** [DIAGNOSTIC; not a registered criterion].
- **What it is:** the model-to-data ratio of each outcome at the end of a host or
  a path, across the slices above, with the unrounded values from
  `qg2_report.json`.
- **The pattern:** it falls as the outcome gets rarer.

| how often the outcome occurs at that point | model / data | cases |
|---|---|---|
| 38–97% | 1.00–1.04 | path and end-of-URL at host and path ends, both classes and frames (7 cases) |
| 26% | 0.91 | benign end-of-URL at a domain host end |
| 5.6–6.7% | 0.77–0.85 | query at a path end (both classes); phishing end-of-URL at a multi-page host end |
| 1.1–1.8% | 0.47–0.64 | query at a host end (both classes and frames); benign end-of-URL at a multi-page host end |

- **Fragments,** the only outcome without a presence control, follow the trend too
  (0.17–0.86). They occur at 0.3% or less and have too few events to weigh.
- **Why it matters.** The same curve holds for an outcome with no query control
  involved: the URL ending at a multi-page host, 1.09% → 0.55. That weighs
  against a cause specific to the query control.
- **Status:** a hypothesis to be tested, not a finding.

## 5. Reading

In registered order (PREREG_QG2D §7):

| reading | benign | phishing |
|---|---|---|
| AUDIT-FOUND | no: no audit finding | no |
| FRAME-POOLING | no: the page frame is also below 1 (0.479 [0.448, 0.513]) | no |
| PATH-COMPETITION | no: S3's closure is 0.20 < 0.5 | no: 0.13 |
| QUERY-STATE | **yes:** S3's closure < 0.25 and S1's CI lies entirely below 1 | **yes** (agrees) |

**The registered reading is QUERY-STATE.** It licenses a registered question on the
hidden-query state's learning, with capacity as the alternative. R3 stays LOCKED,
and the owner decides.

## 6. The registered predictions (PREREG_QG2D §6), scored

| prediction | result | scored |
|---|---|---|
| the audit finds no asymmetry | A1–A5 pass | right |
| at the host end, `<PTH>` over-predicted and `<QRY>` under-predicted | benign +3.12 points path, query 0.98% against 1.66%; phishing path +0.14 (CI includes 0), query under | right for benign; **for phishing only the query half holds** |
| S3 closes at least half of the host-route deficit | 0.20 and 0.13 | **wrong** |
| pooling across frames: no confident prediction; weighed against by the similar deficits | the page frame is under-predicted too | the note held |

## 7. What is and is not claimed

**Claimed** [CONFIRMED by the registered probes]:
- The query deficit is not an artefact of the training code's frame or control
  handling.
- It is not frame pooling.
- It is barely reduced by knowing the path state.
- It disappears when the query state is told. It lives in the model's decision
  under the hidden query control.

**Not claimed:**
- **why** that decision is low. The rarity pattern is a hypothesis.
- **a fix.**
- **anything for another training.** One model, `e9_ln`, was read.

## 8. What the reading licenses (owner approval needed, as for QG2-D)

**QG3,** laptop only, registered first. It separates two explanations:
- **(a) rarity-general:** every rare structural outcome is under-predicted on one
  curve of frequency, whatever its control;
- **(b) hidden-control-specific:** the under-prediction is confined to decisions
  whose own presence control is hidden.

**Possible probes:**
- the calibration-against-frequency curve over all structural outcomes with enough
  events, split by the outcome's control status (hidden, told, or none);
- the same curve on `e9_base` and E7, to see whether it depends on the objective.

**Which lever families each explanation would point to** (all later, all the
owner's decision):
- **(a):** training dynamics or capacity. R2 was rejected and R3 is LOCKED; both
  are the owner's to reopen.
- **(b):** conditioning, meaning how the hidden state is trained.

## Appendix A — the rendered tables

`python research/qgap/qg2_read.py`, verbatim.

### Validity

- benign: VP1 host 0.541 against QG1 0.539; path 0.806 against QG1 0.806; VP2 max |sum − 1| 6.1e-07; VP3 mis-encoded controls 0; rows S0 40,000, S1 40,000, S2 40,000, S3 40,000, S4 40,000 → **PASS**
- phishing: VP1 host 0.592 against QG1 0.592; path 0.777 against QG1 0.780; VP2 max |sum − 1| 5.9e-07; VP3 mis-encoded controls 0; rows S0 40,000, S1 40,000, S2 40,000, S3 40,000, S4 40,000 → **PASS**
- Audit (A1–A5): findings none

### Probe 1 — the host-route query decision by slice (Σ p(<QRY>) / queries)

| class | S0 free mode | S1 multi-page sites | S2 frame swapped | S3 path told | S4 query told |
|---|---|---|---|---|---|
| benign | 0.541 [0.506, 0.583] | 0.479 [0.448, 0.513] | 0.664 [0.623, 0.716] | 0.633 [0.593, 0.683] | 1.168 [1.097, 1.250] |
| phishing | 0.592 [0.550, 0.640] | 0.502 [0.457, 0.552] | 0.806 [0.746, 0.878] | 0.645 [0.600, 0.696] | 1.298 [1.220, 1.379] |

| class | S3 closure of S0's host-route deficit | r_S3 − r_S0 (paired) |
|---|---|---|
| benign | 0.20 | 0.093 [0.079, 0.106] |
| phishing | 0.13 | 0.052 [0.040, 0.065] |

**The path route, by slice (description only):**

| class | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| benign | 0.806 [0.777, 0.840] | 0.911 [0.887, 0.935] | 1.171 [1.130, 1.222] | 0.793 [0.764, 0.827] | 1.154 [1.124, 1.185] |
| phishing | 0.777 [0.736, 0.824] | 0.842 [0.822, 0.863] | 1.008 [0.952, 1.068] | 0.722 [0.684, 0.766] | 1.291 [1.246, 1.339] |

### Probe 2 — at the end of a real host (S0): the model's split against the data's

| class | slice | model PTH / data PTH | model QRY / data QRY | model EOS / data EOS | model FRG / data FRG | model: host continues |
|---|---|---|---|---|---|---|
| benign | S0 | 75.02% / 71.90% | 0.98% / 1.66% | 24.00% / 26.44% | 0.00% / 0.00% | 45.17% |
| benign | S1 | 98.57% / 97.16% | 0.83% / 1.76% | 0.60% / 1.09% | 0.00% / 0.00% | 10.44% |
| benign | S3 | 71.64% / 71.90% | 1.14% / 1.66% | 27.22% / 26.44% | 0.00% / 0.00% | 44.29% |
| phishing | S0 | 38.89% / 38.75% | 1.02% / 1.60% | 60.08% / 59.60% | 0.01% / 0.05% | 54.25% |
| phishing | S1 | 95.00% / 93.19% | 0.69% / 1.23% | 4.30% / 5.58% | 0.00% / 0.01% | 30.41% |
| phishing | S3 | 38.73% / 38.75% | 1.10% / 1.60% | 60.16% / 59.60% | 0.01% / 0.05% | 53.65% |

| class | S0: model PTH share − data PTH share | page − domain: data's host-end query share |
|---|---|---|
| benign | 3.12% [2.71%, 3.54%] | 0.09% [-0.09%, 0.27%] |
| phishing | 0.14% [-0.19%, 0.46%] | -0.38% [-0.55%, -0.20%] |

**Per-category calibration in the host region (S0): Σ p / events**

| class | QRY | PTH | EOS | FRG | PRT | SUB | content |
|---|---|---|---|---|---|---|---|
| benign | 0.541 [0.506, 0.583] | 0.973 [0.964, 0.981] | 0.832 [0.817, 0.845] | — | 0.959 [0.746, 1.306] | 0.935 [0.922, 0.948] | 1.008 [1.007, 1.008] |
| phishing | 0.592 [0.550, 0.640] | 0.938 [0.927, 0.950] | 0.930 [0.920, 0.939] | 0.161 [0.110, 0.276] | 0.874 [0.712, 1.110] | 0.996 [0.991, 1.001] | 1.004 [1.003, 1.004] |

**At the end of a real path (S0): the model's split against the data's**

| class | model QRY / data QRY | model EOS / data EOS | model FRG / data FRG | model: path continues |
|---|---|---|---|---|
| benign | 5.51% / 6.45% | 94.48% / 93.54% | 0.01% / 0.01% | 42.67% |
| phishing | 5.50% / 6.73% | 94.29% / 92.99% | 0.21% / 0.28% | 34.50% |

### S4 — told a query exists: where the model puts it

| class | model host-route share | data host-route share | hazard per told row | hazard per row told none |
|---|---|---|---|---|
| benign | 0.238 [0.229, 0.247] | 0.264 | 0.980 [0.952, 1.010] | 0.012 [0.012, 0.012] |
| phishing | 0.354 [0.336, 0.376] | 0.381 | 1.012 [0.974, 1.048] | 0.012 [0.012, 0.013] |

### Reading

- benign (decides): **QUERY-STATE**
- phishing: QUERY-STATE
