# E10 — the explicit presence decision on the new objective: pre-registration

- **Registered:** 2026-09-28, before any E10 code, training or generation.
- **Chosen by:** the owner's delegation after QG3 ("choose whatever you like and the
  order that you see right").
- **Basis:** the query-gap investigation (`research/qgap/`: QG1, QG2-D, QG3).
- **Frozen, unchanged:**
  - the adopted baseline (`uie/baseline.py`, `e9_ln` / raw) and every library
    default;
  - the data bundle, gate v2 and its thresholds, and every instrument;
  - every published result;
  - E8's state: permanently unread.
- **Naming.** PREREG_E9 §7 named a possible "E10-data". It was not licensed, and
  this experiment is not that.

## 0. In plain words

- **The investigation's finding.** The model under-produces queries because, left
  to decide on its own with the query control hidden, it chooses one too rarely.
- **Two facts support making the choice explicit:**
  - told whether a query exists, it decides correctly;
  - it predicts even rare outcomes correctly when no hidden control governs them.
- **E10 makes the choice explicit.** Right after the suffix, the model writes a
  one-token plan, "path: yes/no, query: yes/no", and then writes the URL following
  its own plan. In control mode the plan is supplied instead.
- **The test.** Two models are trained from the same starting weights. One is
  today's baseline recipe; the other is the same recipe plus the plan. They are
  compared.

## 1. Decisions, with their reasons

**D1 — Why the explicit decision** [CONFIRMED evidence, `research/qgap/`]:
- **QG1:** about 90% (benign) and 80% (phishing) of the query-emission gap is the
  model's query-start decision on real URL beginnings.
- **QG2-D:** the deficit lives in the decision under the hidden query control.
  - It is not an artefact of the training code (audit A1–A5) and not frame
    pooling.
  - Told a query exists, the model spends 0.98 / 1.01 of a query per URL and
    routes it nearly as the data does.
- **QG3:** 97–98% of the deficit is the hidden state, not rarity. Uncontrolled
  outcomes that are equally rare per position come out at 0.98–1.00.
- **A plan turns the hidden decision into two parts:**
  - one explicit, single-position choice, which QG3 predicts will be calibrated;
  - a told-state continuation, which QG2-D S4 found calibrated.

**D2 — Two arms from one seeded start, trained in one session.**

| arm | recipe |
|---|---|
| `e10_base` | the baseline recipe, exactly `uie.baseline.BASELINE.recipe`: E7's settings, `loss_norm="fixed"`, C = 14,397.885, `field_dropout` 0.5, `seed_init`, seed 20260803 |
| `e10_plan` | the same, plus `presence_plan=True` |

- **Why not compare with `e9_ln` directly.** A seeded start does not reproduce
  across environments. On the laptop's torch, the same seed and shapes give the
  start `32d43dfc…`; E9 recorded `26926388…` on Kaggle. Only arms trained in one
  session are guaranteed an identical start.
- **The plan's vocabulary is `e9_ln`'s:** 107 tokens in the same order
  [CONFIRMED, laptop]. So the arms' shapes, and hence their starts, match.
- **A bonus.** `e10_base` against `e9_ln` measures same-recipe, run-to-run spread,
  which the three-training spread did not isolate. It is reported, not a criterion.

**D3 — The plan is the one change.**
- `presence_plan=True`: the plan tokens follow the suffix and are always trained.
- The plan replaces the hidden prefix controls, so `field_dropout` is 0 for this
  arm. The trainer refuses anything else (`uie/kaggle_train.py`). This is part of
  the change, not a second one.
- Everything else is the baseline recipe, including the **same C** (14,397.885).
  The plan's two extra tokens change the mean scored mass by about 2 tokens per
  row, and AdamW absorbs the scale. Keeping C fixed removes one difference.
- **E8's plan code is reused as code, with its tests. None of E8's results is read
  or used.**

**D4 — Decode.** Both arms use raw: temperature 1.0 and nothing else, the
baseline's decode. In free mode the plan arm fills its own plan; in control mode
the plan is supplied. The plan arm records `plan_trace`: the mass outside a free
plan slot's two legal tokens.

**D5 — Generation** (E9's calls, with raw decode):

| | per arm | total |
|---|---|---|
| free | 5 × 8,000 per class at `seed_free(g)` | 160,000 URLs |
| control | 8 states × 5 generations at `seed_ctrl`, `n_ctrl` (3,000 per measurable state, 1,000 honouring-only) | 200,000 URLs |
| **all** | | **100 files, 360,000 URLs** |

**D6 — Validity by identity,** as in E9 (§4).

**D7 — Readings reuse E9's instruments.**
- emission, gate v2 free cells, CR1 control cells, honouring, novelty and copying:
  as in `e9_score.py`;
- the plan's calibration: `e8_plan_calibration.plan_by_suffix`, weighted by the
  shipped table, the method of FINDING §2.1.

## 2. Criteria (fixed now)

| code | criterion |
|---|---|
| **K10a** mechanism | the plan's query probability, shipped-weighted, is within ±10% (relative) of the data rate on rows that fit (6.53% / 4.51%, FINDING §2.3); its path probability is within ±3 points of 73.3% / 39.4% |
| **K10b** emission | `e10_plan`, free: QRY within [5.96, 7.28]% / [4.21, 5.15]% and PTH within [70.37, 76.37]% / [36.51, 42.51]% (the K9b bands) |
| **K10c** no regression | against `e10_base`: no gated cell (free 10; control, every computable cell) has its verdict worsen **and** its x rise by more than 0.0141 |
| **K10d** guardrails | against `e10_base`: neither novelty nor query copying worsens by ≥ 0.5 points with a 95% CI excluding 0, in either mode |

**Outcomes:**
- **K10a–K10d all pass:** `e10_plan` is a candidate. Adopting it as the baseline is
  the owner's decision; an adopted candidate goes to N3.
- **K10a fails:** the plan decision is itself miscalibrated, so the
  explicit-decision hypothesis is refuted as stated. The next question is the
  hidden state's exposure (the owner's).
- **K10a passes and K10b fails:** the queries are lost after the plan. Reported;
  the owner decides.
- **K10c or K10d fails:** reported; the owner decides.

**Reported, not criteria:**
- `e10_base` against `e9_ln`;
- honouring in both arms;
- `plan_trace`;
- QG1's decomposition for `e10_base`.

## 3. Predictions, before any run [HYPOTHESIS]

- **K10a passes.** Reasons:
  - an explicit, uncontrolled decision is calibrated even when rare (QG3);
  - the token-weighted objective weights every row's plan equally.
- **K10b passes.** Emission is about the plan's rate times honouring when told,
  about 0.98 (QG2-D S4). That is about 6.4% (benign), inside the band.
- **K10c and K10d: Unknown.** More queries also means more query content to judge.
- **`e10_base` reproduces `e9_ln`** within the retrain allowance on free cells
  (≤ 0.0141).

## 4. Validity

| code | check | on failure |
|---|---|---|
| V1 | hashes and completeness: checkpoints, the code bundle, the tables, every file with its registered n and seed | fixed under an amendment; nothing read |
| V2 | byte identity on the laptop for at least 250 URLs per arm and mode | same |
| V3 | flags off are unchanged (tests); `uie/` is unchanged since the code commit | same |
| V4 | the scorer reproduces HC1 C2's E7 cells exactly | same |
| V-id | the arms' initial-weight hashes are identical; their tokenizer, model and training settings are the baseline recipe's, apart from `presence_plan` and its `field_dropout` 0; data counts equal E9's | same |

## 5. Not claimed

- E3 passage;
- anything about E8's arms, which stay unread;
- that the plan fixes query content: E10 targets presence.

## 6. Cost and order

1. This registration.
2. Code and tests; the bundle; the notebooks.
3. `E10_train`, which the owner runs: 2 arms, about 2.5 h.
4. `E10_generate`, which the owner runs: 360,000 URLs, about 2.5 h.
5. The laptop: V1–V4, V-id, K10a–K10d, and the reported readings.
6. `E10_RESULTS.md`. The owner decides adoption.

## Status

    E10  registered 2026-09-28 (this document), before any E10 code
    E10  code 1c00650 (tests 8; local smoke of both arms: identical seeded start,
         settings as registered); bundle uie-e10-code sha256 4e47a4c3...;
         notebooks E10_train / E10_generate handed to the owner
    E10  trained 2026-09-29 (results (14)): VERIFY_TRAIN PASS -- identical start,
         equal to e9_ln's (26926388...); settings as registered; E9's data counts
    E10  K10a READ 2026-09-29, before any generation data: FAIL.
         benign   plan Q 0.0605 (0.927 of 0.0653, within); plan P 0.7710 (+0.038 > 0.03)
         phishing plan Q 0.0403 (0.893 of 0.0451, outside); plan P 0.3983 (+0.004)
         Candidate rule: the plan decision is itself miscalibrated; the
         explicit-decision hypothesis is refuted as stated. K10b-K10d await
         generation; running it is the owner's call.
    E10  note 2026-09-29: K10a's probe `x.{s}/` reads another suffix's plan for
         benign bg, se, sapporo.jp and phishing bg, se (public-suffix rules make
         `x.se` etc. suffixes themselves). With the training prefix K10a's
         quantities move by under 0.0001; the reading is unchanged
         (PREREG_PC1.md amendment 1). E10_generate started by the owner.
    E10  READ 2026-09-29 (E10_RESULTS.md): VERIFY PASS. K10a FAIL, K10b FAIL,
         K10c PASS, K10d FAIL (control-mode query copying +2.34 points).
         Candidate rule: the explicit-decision hypothesis is refuted as stated;
         next question the hidden state's exposure (E11's e11_joint).
         Reported: a supplied plan honours 94.99-99.72% and turns every
         convicted control cell UNDECIDED; e10_base reproduces e9_ln within
         0.0069 on free cells.
