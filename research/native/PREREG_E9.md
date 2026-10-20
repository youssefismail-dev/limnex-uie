# E9 — the length-neutral objective. Decision record and pre-registration

**Registered:** 2026-09-27, before any E9 code, training or generation.
- **Approved by the owner:** the experiment (the fixed-constant loss normalisation)
  and the fixed-weight theory as a hypothesis to test. The QRY-data question (§0,
  D7) was left to the evidence.
- **Basis:** `FINDING_LENGTH_WEIGHTING.md` (cc5fb32, 5984a7b).
- **Frozen, unchanged:** the data bundle, gate v2 and its thresholds, every
  instrument, every published result, the frozen delta (the resting decode), and
  E8's unread state (its option is still the owner's).

## 0. Decisions made before implementation, with their reasons

**D1 — One change, plus a seeded start.**
- E9 changes only the training objective's weighting.
- Both arms seed their initialisation, so they start from **identical weights**.
- **Why:** E8 showed that two trainings of one recipe from uncontrolled starts
  differ by up to 5.2 points in honouring (`E8_RESULTS.md`). Only a shared start
  makes the contrast attributable to the objective. Every other setting is E7's.

**D2 — Two arms, not one against E7.**

| arm | objective | start | recipe |
|---|---|---|---|
| `e9_base` | today's: each batch's loss is its mean over its scored tokens | seeded, 20260803 | E7's |
| `e9_ln` | length-neutral: every scored token weighs the same in every batch | the same seed, so identical initial weights | E7's |

- **Why not compare `e9_ln` with E7 directly:** E7 started from an unrecorded
  state; `e9_base` is E7's recipe from `e9_ln`'s start.
- **A bonus:** `e9_base` is a third independent training of E7's recipe (after E7
  and `e8_base`), so retrain spread becomes measurable from three samples. It is
  reported, not used as a gate.

**D3 — The constant.**
- C = 256 × 56.242 = **14,398** scored tokens per batch. 56.242 is the expected
  number of scored tokens per drawn row under E7's sampler (class probabilities by
  name count, rows uniform within a class, over-length rows dropped) [CONFIRMED,
  `e9_evidence_signal_share.py`].
- With this C the average loss scale equals today's, so the learning rate keeps
  its meaning.

**D4 — Clipping: scale after clipping, not inside the loss.** The recipe clips
the gradient norm at 1.0. Measured with E7's field masks, on real 256-row batches
per length band [CONFIRMED, `e9_evidence_clipping.py`, `data/native/e9_clipping.json`]:

| | tokens per batch | today's loss: norm | fixed divisor inside the loss: norm |
|---|---|---|---|
| **late training** (E7's weights), rows 8–29 | 4,710 | 0.78 | 0.26 |
| rows 30–59 | 9,984 | 0.60 | 0.41 |
| rows 60–119 | 19,039 | 0.49 | 0.64 |
| rows 120–256 | 39,923 | 0.37 | **1.03** |
| **early training** (a fresh start), rows 8–29 | 4,710 | 7.5 | 2.5 |
| rows 120–256 | 39,923 | 6.9 | **19.1** |

- **With the divisor inside the loss, clipping would undo the change.**
  - Early in training every batch exceeds 1.0 and is cut to norm 1, whatever its
    length. That is exactly today's equal weight per batch.
  - Late in training the longest batches (1.03) are still trimmed.
- **So E9 keeps today's loss and today's clipping unchanged,** then multiplies the
  clipped gradient by s_b = T_b / C before the optimiser step, T_b being the batch's
  scored tokens.
  - For any batch and any weights, `e9_ln`'s update direction is `e9_base`'s times
    s_b. Clipping acts identically in both arms, and the only difference is the
    per-batch weight, in every phase.
  - Unclipped, this is exactly the gradient of Σ losses / C, the approved
    fixed-constant objective. Clipped, it is the same weighting applied to the
    same clipped direction.
- **s_b spans about 0.33–2.8.** AdamW's per-parameter normalisation absorbs the
  scale on average; the warm-up (500 steps) is unchanged.
- **Measured in the run:** each arm records its per-step norms and the clipped
  fraction by length band.

**D5 — Decodes.**
- The frozen delta (resting decode) was fitted to today's objective's early path
  ending. The finding shows that bias is the weighting itself.
- A de-biased model read with the delta would be over-corrected, so the delta
  cannot be applied blindly to `e9_ln`.
- The **native** system is `e9_ln` read with the **raw** decode: temperature 1.0,
  no delta, no other adjustment.
- The conditions generated:

| condition | free (canonical) | control (native) | role |
|---|---|---|---|
| `e9_base` / resting | yes | yes | **today's system**, retrained on E9's start: the adoption baseline |
| `e9_base` / raw | yes | yes | the objective contrast at a fixed decode |
| `e9_ln` / raw | yes | yes | **the native candidate** |
| `e9_ln` / resting | yes | — | reported: does the delta now over-correct? |

  This does not reopen the delta branch: the delta itself is untouched, and the
  resting decode stays the decode of every earlier result.

**D6 — Validity by identity, not by behavioural tolerances.** E8's V-base assumed a
retrain would stay within ±1 point and failed on spread nobody had measured. E9's
validity checks identity directly:
- the arms' tokenizer, model and training settings (apart from the one flag), and
  data counts, against E7's and each other's;
- **the arms' initial weights: identical hashes;**
- files, code and scorer as in E8.

Behavioural differences between `e9_base` and E7 or `e8_base` are **reported as
retrain spread**, never used as a gate.

**D7 — The owner's QRY-data suggestion: not in E9. E10 is conditional on E9's
reading, by the rule in §7.** The evidence:
1. **E9 already raises the query field's share of the training signal** from 3.19%
   to 4.76% (×1.49), and the path field's from 42.4% to 51.8%, without adding a row
   [CONFIRMED, `e9_evidence_signal_share.py`]. Extra query data in the same run
   would stack a second change on the first, and attribution would be lost. R1's
   own record says the same: "must not be bundled … or attribution is lost".
2. **More of the same query rows did not help.** N2a's A2 saw each query row four
   times as often and landed on E7's own values: closure +0.01 on held-out queries,
   +0.03 on training queries. Copying rose by 6.85 points [CONFIRMED, N2a].
3. **E7 does not fit the queries it already has.** It is about as detectable
   against its training queries as against held-out ones (+0.0862 against
   +0.0957) [CONFIRMED]. A model that does not reproduce its own training
   distribution is not limited by how much data it has.
4. **The objective explains most of what E7 gets wrong in queries.** Short and mid
   value stops are calibrated to the weighted data, and so is most of the
   parameter shortfall [CONFIRMED]. What remains is the long-value tail
   (positions 50–100): residual 1.08 benign, 1.49 phishing. That is where data is
   thinnest (1,154 benign values in the 20,000-row sample reach it) and where the
   256-token budget deletes 41–45% of rows with long values (Q6, R1).
   - If data limits anything after E9, it is there.
   - The in-corpus lever for it already exists and is sized: R1, a longer
     `max_len`.
5. **The corpus is frozen.** New external query data would change the frozen
   bundle, and that is the owner's decision.

**Decision:** E9 stays a single-change test. After E9, §7's rule decides between
three outcomes:
- the data question is licensed (R1 first, external data only on the owner's
  decision);
- it is not licensed;
- it is unnecessary.

**D8 — Checkpoint.**
- Both arms read `best.pt`, as E7 and E8 did.
- Both are selected by the same, unchanged validation metric (`uie.train.evaluate`).
- `final.pt` is reported alongside.

## 1. Question and hypotheses

**Question.** Does removing the objective's roughly 1/length weighting of each row
remove the biases it measurably causes? Does it change the gate's content cells?

- **H9a, mechanism** [HYPOTHESIS, the owner's approved test]. Trained without the
  weighting, the model is plainly calibrated where E7 is calibrated to the weighted
  data:
  - ending the URL inside paths;
  - stopping short and mid query values;
  - adding another parameter;
  - and free-mode presence, which reaches the data rates.
- **H9b, content** [open; Unknown]. The gate's content cells, especially QRY and
  PTH, improve beyond retrain noise, and no cell regresses.

## 2. The change, exactly

**Flags,** all defaulting off, so every earlier run stays byte-identical (tests):
- `TrainConfig.loss_norm`: `"batch"`, today's (default), or `"fixed"`, E9;
  `loss_constant`, C.
- `seed_init` in `kaggle_train.run`: seeds torch with the training seed before the
  model is built.
- **Diagnostics recorded for every run:**
  - per-step gradient norm and scored-token count, summarised by length band;
  - the fraction of clipped steps;
  - a hash of the initial weights.

**`"fixed"`:** after today's clipping, every gradient is multiplied by
s_b = T_b / C before the optimiser step (D4). T_b is the batch's weighted
scored-token mass, the same quantity the loss divides by. Nothing else in the step
changes.

## 3. Generation (one Kaggle session after training)

- Every call as E8's (`e8_common`): the shipped suffix table in free mode; SC1's
  state tables in control mode; the plan and root slash off (E7-type tokenizer).
- **Free:** 5 × 8,000 per class per condition, at the canonical seeds
  (`seed_free(g)`), so every condition pairs with E7's canonical files.
- **Control:** 8 joint states × 5 generations; 3,000 per measurable state, 1,000 in
  phishing P1Q1 and P0Q1 (honouring only). E8's seeds (`seed_ctrl`), so it pairs
  with E8's E7 control files.
- **Raw decode:** `decoder="temperature"`, temperature 1.0, no field bias.
- **Resting decode:** the same plus the frozen PTH delta (0.22 / 0.30).
- **Size:** about 620,000 URLs, about 4.2 h on a T4 (E8's rate).

## 4. Readings (laptop)

| code | reading | instrument |
|---|---|---|
| **R1** mechanism | plain and weighted calibration ratios, teacher-forced on each arm's `best.pt`, E7's rows | `weighted_calibration.py`, with row-bootstrap 95% CIs added |
| **R2** emission | free-mode PTH and QRY presence per class and condition | E8's definition |
| **R3** realism | gate v2 free canonical (G = 5, D = 10); CR1's instrument per measurable control state | `cr1_score` unchanged |
| **R4** honouring | control mode, per state and condition | E8's definition |
| **R5** QRY diagnostics | Q-VAL and Q-TRAIN (the gate's QRY cell against validation and training queries); M1 by bucket | `n2a_qry.py`, `n2a_reference.py` |
| **R6** guardrails | whole-URL novelty and query copying, free and control | E8's (DC1's `prop_worsening`) |

**The mechanism's decision families (benign decides; phishing must agree in
direction):**
- **F1:** ending the URL inside a path, positions 0–50, pooled;
- **F2:** stopping inside a query value, positions 0–25, pooled;
- **F3:** another query parameter at a value's end.

## 5. Validity

| code | check | on failure |
|---|---|---|
| V1 | hashes and completeness: checkpoints, code bundle, tables, every file with its registered n and seed | fixed under an amendment; nothing read |
| V2 | byte identity on the laptop for at least 250 URLs per arm, condition and mode | same |
| V3 | flags off reproduce today's training byte for byte (tests); `uie/` unchanged since the code commit | same |
| V4 | the scorer reproduces HC1 C2's E7 cells exactly | same |
| V-id | the arms' initial-weight hashes are identical; their tokenizer, model and training settings are identical to each other and to E7's, except `seed_init` (both) and `loss_norm` (e9_ln); data counts equal E7's | same |

## 6. Criteria (fixed now)

**Precondition P0.** `e9_base` shows E7's pattern in F1–F3:
- plain ratio above 1 in F1 and F2, below 1 in F3;
- weighted ratio closer to 1 than the plain one.
- If not, the mechanism test is void and reported.

| code | criterion |
|---|---|
| **K9a** mechanism | In each of F1–F3 (benign): `e9_ln`'s plain ratio closes ≥ 0.5 of `e9_base`'s gap to 1. Closure = 1 − \|R(e9_ln) − 1\| / \|R(e9_base) − 1\|, and `e9_ln`'s 95% CI must exclude `e9_base`'s value. Phishing: `e9_ln` closer to 1 than `e9_base` in each family. |
| **K9b** emission | `e9_ln` / raw, free: QRY within [5.96, 7.28]% / [4.21, 5.15]% and PTH within [70.37, 76.37]% / [36.51, 42.51]% (E8's achievable-target bands) |
| **K9c** no regression | `e9_ln` / raw against `e9_base` / resting, today's system: no gated cell (free 10; control, every computable cell) has its verdict worsen **and** its x rise by more than 0.0141. The same rule is also reported against `e9_base` / raw, for attribution |
| **K9d** guardrails | against `e9_base` / resting: neither novelty nor query copying worsens by ≥ 0.5 points with a 95% CI excluding 0, in either mode |

**Candidate.** If K9a–K9d all pass, `e9_ln` with the raw decode is a candidate, and
the owner decides adoption.
- Adopted, it goes to **N3**, the E3 reading, with the frozen delta retired for it.
- K9a failing means the hypothesis is refuted; nothing is licensed.
- K9a passing with K9b, K9c or K9d failing is reported, and the owner decides.

**Not claimed:**
- E3 passage;
- any cell's CLEARED verdict as proof of realism outside the gate;
- anything about the presence plan or the root slash (E8 remains unread).

## 7. What E9 licenses next, including D7's QRY-data rule

Read on `e9_ln` / raw, benign deciding, with E9's own gate reading of the QRY cell.
- **QRY is CLEARED:** no data experiment; the data question is closed.
- **Sparsity signature,** read against `e9_base` / raw: Q-TRAIN closure ≥ 0.5 while
  Q-VAL closure < 0.25. The model now fits the queries it saw but does not
  generalise. Licensed: **E10-data**. First R1 (a longer `max_len`, from the frozen
  corpus, sized by `r1_sizing.py`); external query data only if the owner unfreezes
  the corpus.
- **Tail signature:** M1 at positions 50–100, plain ratio, with a 95% CI excluding
  1.10, while F2 passes. The residual is the long tail the length budget cuts.
  Licensed: **R1** alone.
- **Anything else:** no data experiment is licensed; reported, and the owner
  decides.

## 8. Predictions, stated before any run [HYPOTHESIS]

- **`e9_base`** reproduces E7's pattern:
  - F1 plain about 1.16 pooled, F2 about 1.06, F3 about 0.85;
  - emission near E7's 59% / 3.6% (benign).
- **`e9_ln`:**
  - plain ratios near 1 in F1–F3;
  - raw free emission near 73% / 6.6% (benign) and 39.5% / 4.7% (phishing);
  - its weighted ratios fall below 1.
- **`e9_ln` / resting:** paths longer than the data. The delta over-corrects a
  model that no longer needs it.
- **Content:** Unknown.

## 9. Cost and order

1. This registration.
2. Code and tests: flags off must be byte-identical.
3. Kaggle session 1: train both arms, about 2.5 h.
4. Kaggle session 2: generation, about 4.2 h.
5. The laptop: V1–V4 and V-id; R1–R6; K9a–K9d; §7.
6. `E9_RESULTS.md`. The owner decides adoption.

## Status

    E9  registered 2026-09-27 (375e8a4); code 3b29288; bundle uie-e9-code d360d80b
        laptop tools built and dry-run validated (see the commit after 3b29288);
        E7's mechanism reference in data/e9/mechanism_e7.json
        READ 2026-09-28 (E9_RESULTS.md): VERIFY PASS; P0, K9a, K9c, K9d pass;
        K9b FAILS (query emission below its band; paths inside). Candidate rule:
        "K9a passes (mechanism confirmed); K9b fail: reported, the owner decides".
        Section 7: no data experiment licensed. Adoption is the owner's decision.
        OWNER DECISION 2026-09-28 (DECISION_2026-09-28.md): e9_ln with the raw
        decode is ADOPTED as the baseline (uie/baseline.py). This registration and
        its results are unchanged; K9b's failure stands in the record.
