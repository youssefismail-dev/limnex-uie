# N2a — results

**Status: [DIAGNOSTIC].** Registration `PREREG_N2A.md` (Amendments 1–4); code
`6fdb067`; Kaggle T4, 2026-09-27; laptop readings the same day. A1 and A2 are
diagnostic models. Nothing they wrote is data, and they are never candidates.

## In plain words

- **The question.** Why are E7's queries still detectable? Either the model did not
  get enough query practice, which training could fix, or the data cannot teach
  it more, which training cannot fix.
- **The test.** Two models were trained only on URLs that carry a query.
  - **A1** saw each query row as often as E7 did.
  - **A2** saw each four times as often.
- **What happened** (benign decides):
  - A1 is worse than E7.
  - A2 is exactly as detectable as E7, both against held-out queries and against
    the training queries themselves.
  - What A2 gained was copying: 34.2% of its queries are training queries, word
    for word, against E7's 27.3%.
- **The registered reading is SPARSITY.** More exposure buys memorisation, not
  realism. Its registered consequence: *no training fix inside the frozen corpus;
  the owner decides.*
- **What it rests on.** This reading comes from the copying clause alone, which
  Amendment 2 wrote as an **or** where the plan had an **and** (section 4). Under
  the plan's wording it would be MIXED. Both wordings license the same thing: no
  training repair of query content (N2b, N2c), and R3 stays locked.
- **Phishing is UNAVAILABLE,** as recorded before any arm. Its Q-TRAIN reference is
  too noisy to read the fit rows, and neither arm closed its validation gap.

## 1. Validity — all passed (`n2a_decide.py verify`)

| check | result |
|---|---|
| V: checkpoints and every generated file against their records (sha256) | match |
| V: tokenizer | E7's, byte for byte, in both arms |
| V: training settings | E7's recorded settings, except steps, warm-up and evaluation spacing, which are exactly Amendment 4's |
| V: kept query rows | 223,342 / 13,270, the registered counts (the run stops otherwise) |
| V: generation | 10 files per arm, 1,500 URLs each, Amendment 2's seeds |
| K: non-finite loss | none |

| arm | steps | trained | final validation loss, benign / phishing | best `best.pt` step | generated URLs with a query, per 1,500 |
|---|---|---|---|---|---|
| A1 | 1,643 | 16.4 min | 1.958 / 2.316 | 1,640 | 1,448–1,491 |
| A2 | 6,572 | 31.6 min | 1.885 / 1.945 | 5,593 | 1,488–1,495 |

- Every QRY cell reached its full n (1,200 benign, 356 phishing) in all 50 draws,
  with all 5 generations usable.
- Readings use `final.pt` (section 2). `best.pt` is reported only.

## 2. The primary readout — the gate's own QRY cell (Amendment 2)

x is the excess over the floor, and 0 is ideal. Closure = 1 − x(arm) / x(E7).

| class | model | Q-VAL x (SE) | closure | Q-TRAIN x (SE) | closure |
|---|---|---|---|---|---|
| benign | E7 (reference) | +0.0957 (0.0038) | — | +0.0862 (0.0034) | — |
| benign | A1 | +0.1077 (0.0034) | −0.13 | +0.0989 (0.0019) | −0.15 |
| benign | A2 | +0.0945 (0.0046) | +0.01 | +0.0839 (0.0062) | +0.03 |
| phishing | E7 (reference) | +0.1141 (0.0079) | — | +0.1170 (0.0114), **void** | — |
| phishing | A1 | +0.1856 (0.0062) | −0.63 | +0.1871 (0.0060) | (void) |
| phishing | A2 | +0.1151 (0.0043) | −0.01 | +0.1167 (0.0062) | (void) |

- Every cell is CONVICTED; none is anywhere near the 0.05 bar.
- "Closes" needs closure ≥ 0.5. The best closure recorded is +0.03.

## 3. Copying (M4) and the copy guard

Queries found word for word among the class's training queries. The change is
against E7, using DC1's `prop_worsening` with a 95% CI.

| class | model | copied | change vs E7 | guarded |
|---|---|---|---|---|
| benign | E7 | 27.33% of 8,616 | — | — |
| benign | A1 | 23.22% of 7,429 | −4.11 pts [−5.46, −2.77] | no |
| benign | A2 | 34.19% of 7,471 | **+6.85 pts [+5.42, +8.28]** | **yes** |
| phishing | E7 | 26.42% of 2,774 | — | — |
| phishing | A1 | 19.10% of 7,293 | −7.32 pts [−9.20, −5.45] | no |
| phishing | A2 | 36.14% of 7,454 | **+9.72 pts [+7.75, +11.69]** | **yes** |

The cleanest contrast is inside the experiment: A1 → A2 is the same data and
recipe with four times the exposure. Copying rises by 11.0 points (benign) and
17.0 points (phishing) [CONFIRMED].

## 4. The decision — Amendment 2's table, read in order (Amendment 4)

**Benign:**

| row | holds? | why |
|---|---|---|
| ALLOCATION | no | A1's Q-VAL closure is −0.13 |
| EXPOSURE | no | A2 is guarded, so its validation closure is void; its value, +0.01, would not close anyway |
| **SPARSITY** | **yes** | copying rises in A2 (+6.85 pts, CI excludes 0). The fit clause does **not** hold: A2 is "little" on Q-TRAIN (+0.03) |
| CAPACITY-OR-REPRESENTATION | no | A2 is little on Q-TRAIN, but copying rose |

**Reading: SPARSITY. Licence: no training fix inside the frozen corpus; the owner
decides.**

**Phishing: UNAVAILABLE.**
- Neither available row holds: A1's closure is −0.63; A2's is −0.01 and guarded.
- The fit rows are unavailable, because the Q-TRAIN reference is void by the power
  rule, recorded before any arm.
- Nothing extends to phishing.

**What the reading rests on — a disclosure.**
- The plan (§6) and this registration's original section 5 defined SPARSITY with
  **and**: "A2 closes on train but little on validation, **and** copying rises".
  Amendment 2 moved the decision onto the gate's QRY cell and wrote the row with
  **or**. Amendment 2 was committed before any arm existed, so its wording is the
  registered rule, and it decides.
- Under the original **and**, SPARSITY would not hold, because A2 does not close
  Q-TRAIN. CAPACITY-OR-REPRESENTATION would not hold either, because copying rose.
  The reading would be **MIXED**: "no content repair licensed".
- **The practical consequence is the same under both wordings.** No training repair
  of query content is licensed: N2b and N2c are not licensed, and R3 stays locked.
  Only the label and the licence's wording differ.

## 5. Mechanism evidence (descriptive; not decision inputs)

**M3 FIT** — teacher-forced QRY negative log-likelihood per character, pooled over
bands (controls hidden):

| class | model | train | validation | gap |
|---|---|---|---|---|
| benign | E7 | 1.643 | 1.674 | +0.030 |
| benign | A1 | 1.966 | 1.999 | +0.033 |
| benign | A2 | **1.549** | 1.656 | **+0.107** |
| phishing | E7 | 2.639 | 2.632 | −0.007 |
| phishing | A1 | 2.912 | 2.888 | −0.024 |
| phishing | A2 | **2.267** | 2.371 | **+0.104** |

**M1 LENGTH** — the value-termination hazard, model against corpus, pooled over
positions 25–100; 1.0 is ideal:

| class | model | train | validation |
|---|---|---|---|
| benign | E7 | 1.088 [1.061, 1.118] | 1.107 [1.058, 1.155] |
| benign | A1 | 1.115 [1.082, 1.149] | 1.122 [1.067, 1.177] |
| benign | A2 | 1.140 [1.113, 1.168] | 1.158 [1.105, 1.214] |
| phishing | E7 | 1.321 [1.265, 1.378] | 1.336 [1.183, 1.508] |
| phishing | A1 | 1.333 [1.281, 1.385] | 1.301 [1.169, 1.455] |
| phishing | A2 | 1.125 [1.088, 1.165] | 1.219 [1.094, 1.373] |

**M2 FREQUENCY** — the share of generated parameter names from rare bands (1–19),
against the faithful-sampler null:

| class | E7 | A1 | A2 | null |
|---|---|---|---|---|
| benign | 0.0497 | 0.0384 | 0.0409 | ≈ 0.219 |
| phishing | 0.0967 | 0.0753 | 0.0772 | ≈ 0.377 |

**What these show:**
- **A2 fits the training query text far better than E7,** but not held-out text
  [CONFIRMED, M3]:
  - benign training NLL falls 0.095 nats per character, validation only 0.018;
  - the train/validation gap roughly triples in both classes.
- **A2's validation loss had flattened.** From about step 4,600 it stayed within
  ±0.005 of 1.915, while its training fit kept improving [CONFIRMED, the run's
  history].
- **Likelihood and realism come apart** [CONFIRMED as measurements]:
  - phishing validation NLL improves by 0.26 nats per character, yet phishing Q-VAL
    does not move (+0.1141 → +0.1151);
  - benign training NLL improves, yet Q-TRAIN does not move.
  - Fitting the text better did not make sampled queries harder to detect, even
    against the very queries the model was fitted on.
- **Neither sub-deficit improves with exposure:**
  - rare names become rarer, not commoner: 0.0497 → 0.0409 benign;
  - benign's length tail does not improve: 1.088 → 1.140 on train;
  - phishing's length tail does improve in A2: 1.321 → 1.125.
- **[HYPOTHESIS, untested]** The part of the tell that exposure cannot reach sits in
  the structure of sampled queries, which per-character likelihood does not
  measure. This agrees with C3-B, where benign's QRY tell survived `skeleton`. It
  is not established here.

## 6. What N2a does and does not say

**Says** [CONFIRMED]:
- Up to four times E7's per-row query exposure does not make this model family's
  queries more realistic. A2 lands on E7's own values on both readouts.
- The extra exposure goes into verbatim copying.
- ALLOCATION and EXPOSURE are refuted as explanations of E7's query tell. Even the
  arm with the exposure advantage closes nothing.

**Stated confound, as registered:** A1 gives E7's exposure in 1,643 steps instead
of 20,000 and is worse than E7 everywhere. So A1 alone was ambiguous, and A2
carried the reading, as section 2 specified.

**Does not say:**
- that capacity or representation is, or is not, the limit (N2a cannot prove
  either, as stated in advance);
- anything about any other gate cell;
- anything about E3;
- A1 and A2 are never scored as candidates.

## 7. Consequences under the registered plan

- **N2b** (allocation or exposure repair): **not licensed.**
- **N2c** (tokenization or capacity): **not licensed.**
- **R3** (capacity): **stays locked.** Its unlock needed a FREQUENCY-SPECIFIC or
  GENERAL reading, and neither arises under either wording.
- **The owner decides** (the licence's own words) whether anything beyond the
  frozen corpus is justified. No new experiment is proposed here.
- In the approved order, **N3**, the E3 reading of the adopted candidate, follows
  E8 (N1). E8's own results are reported separately in `E8_RESULTS.md`.

## Provenance

- **Registration:** `PREREG_N2A.md`, Amendments 1–4. Amendment 4 fixed the schedule
  and the table's wording before any arm.
- **Code:** `6fdb067` (`n2a_train.py`, `n2a_decide.py`); Kaggle bundle
  `uie-n2a-code.zip`, sha256 `8ec6d9c0…`.
- **Run:** Kaggle T4, torch 2.10.0+cu128. A1 16.4 min training + 15.0 min
  generation; A2 31.6 + 15.2.
- **Data (local):**
  - `data/n2a/`: checkpoints, `N2A_RECORD.json`, generated files,
    `verify.json`, `decision.json`;
  - per arm: `n2a_qry_<label>.json` and `n2a_reference_<label>.json`;
  - E7's references in `data/native/`.

## Note added later the same day — a competing explanation (reading unchanged)

**`FINDING_LENGTH_WEIGHTING.md`**, found after this result, shows that every model in
the line, A1 and A2 included, trains on an objective that weights each URL by about
1/length. E7 is measurably calibrated to that weighted target [CONFIRMED], in:
- stopping short and mid query values;
- most of its reluctance to add another parameter;
- ending the URL inside paths.

**What that means for N2a:**
- Exposure cannot move a model past its objective's optimum. So "more exposure buys
  copying, not realism" is also what a biased objective would produce.
- **The registered reading, SPARSITY, stands.** Its licence stands too: no training
  repair of query content inside the present training setup.
- The claim that the corpus is what limits query realism now has a measured
  competing explanation: the objective [HYPOTHESIS].
- N2a did not vary the objective, so it cannot separate the two.
- Whether an unweighted objective changes the QRY cell is **Unknown**. The
  finding's section 6 describes the test, for the owner to decide.
