# E8 (N1) — results

**Registration:** `PREREG_E8.md`. Code `338ed0a` (training and generation); Kaggle
T4, 2026-09-26/27; laptop verification and scoring 2026-09-27.

## In plain words

- **The files are sound.**
  - Every file, checkpoint and table matches its record.
  - The laptop regenerates the Kaggle URLs byte for byte.
  - The scorer reproduces E7's published gate cells exactly.
- **E8 could not be read.**
  - Before any arm is judged, the registration requires the retrained baseline,
    `e8_base` (E7's recipe, E7's seed), to reproduce E7 within fixed tolerances
    (V-base). It does not. In the hardest control states its honouring differs
    from E7's by up to 5.2 points, against a ±1-point tolerance, and its free path
    emission by 1.2 points, against ±0.5.
  - Its free gate cells do reproduce E7's, all within the measured 0.0141.
  - The registration's consequence is: **no arm is read.** No K1 criterion was
    evaluated, and the candidate rule was not applied. No arm is adopted, and
    nothing (N1b included) is licensed.
- **The investigation** (Amendment 1 of the registration) finds **no pipeline
  change**:
  - tokenizer, model and training settings, data, training code and environment are
    all identical to E7's;
  - `e8_base` is simply a second, independent training of the same recipe. Its
    starting weights are not controlled by the seed, so it is not a copy of E7;
  - V-base's tolerances were never measured. This pair is the first measurement of
    how much two retrains differ in honouring and emission, and they differ by more
    than the tolerances.
- **Your call.** What to do with E8 now needs a decision only you can make,
  because it concerns a registered threshold. The options are in section 5.
- **Separately,** checking the plan arms' low emission uncovered a finding about
  the training objective itself. It is recorded in `FINDING_LENGTH_WEIGHTING.md`.
  It is not a reading of any E8 criterion.

## 1. Validity V1–V4 — all passed (`e8_verify.py`, `data/e8/VERIFY.json`)

| check | result |
|---|---|
| **V1** hashes and completeness | 250 of 250 registered files present, with registered n and seeds; checkpoints match both the training and generation records; E7 and tables match; no code under `uie/` or `e8_common.py` changed since the code the GPU ran |
| **V2** byte identity on this CPU | 20 files, one per checkpoint and mode: every regenerated URL identical (for example 203/203, 833/833, 135/135) |
| **V3** flags off reproduce E7 | a code property, proven by the suite at `36e1b26`; no `uie/` file changed since |
| **V4** scorer reproduces HC1 C2 | E7's canonical free cells reproduced exactly: max \|x − published\| = 0 in both classes |

## 2. V-base — FAILED

V-base requires `e8_base` to reproduce E7 within noise (section 5, fixed before
any run):
- honouring within ±1 point per joint state;
- emission within ±0.5 points;
- no free gated cell's verdict changing with |Δx| > 0.0141.

**Honouring** (control mode, native; n = 15,000 per state, 5,000 in phishing P1Q1
and P0Q1):

| state | E7 | e8_base | difference | within ±1? |
|---|---|---|---|---|
| benign P1Q1 | 96.11% | 95.57% | −0.53 | yes |
| benign P1Q0 | 98.65% | 98.71% | +0.07 | yes |
| benign P0Q1 | 93.85% | 95.79% | **+1.95** | **no** |
| benign P0Q0 | 97.85% | 98.19% | +0.34 | yes |
| phishing P1Q1 | 84.28% | 89.52% | **+5.24** | **no** |
| phishing P1Q0 | 96.89% | 97.35% | +0.46 | yes |
| phishing P0Q1 | 87.10% | 90.76% | **+3.66** | **no** |
| phishing P0Q0 | 98.15% | 98.49% | +0.33 | yes |

**Emission** (free mode; 40,000 URLs per class):

| | E7 | e8_base | difference | within ±0.5? |
|---|---|---|---|---|
| benign paths | 59.08% | 57.91% | **−1.17** | **no** |
| benign queries | 3.62% | 3.72% | +0.11 | yes |
| phishing paths | 29.51% | 29.03% | −0.48 | yes |
| phishing queries | 1.74% | 2.05% | +0.31 | yes |

**Free gated cells:** within the allowance; see section 2.1.

**These are real differences between two models, not sampling noise.**
- Phishing P1Q1's +5.24 points is about 7.7 standard errors of the difference (n
  5,000 per model).
- Benign path emission's −1.17 is about 3.4 (n 40,000).

**Registered consequence:** *"If it fails, the pipeline changed. No arm is read,
and it is investigated under an amendment."*
- No arm's K1a–K1e has been evaluated or reported.
- The candidate rule has not been applied.

### 2.1 Free gated cells (E7 against e8_base) — within the allowance

Canonical gate v2, G = 5, D = 10. x is the excess over the floor.

| class | field | E7 x, verdict | e8_base x, verdict | Δx | V-base |
|---|---|---|---|---|---|
| benign | LBL | +0.0017 CLEARED | +0.0041 CLEARED | +0.0024 | ok |
| benign | SUF | +0.0165 CLEARED | +0.0135 CLEARED | −0.0030 | ok |
| benign | SUB | +0.0048 CLEARED | +0.0051 CLEARED | +0.0003 | ok |
| benign | PTH | +0.0369 UNDECIDED | +0.0330 CLEARED | −0.0039 | ok (verdict changes, \|Δx\| ≤ 0.0141) |
| benign | QRY | +0.0541 UNDECIDED | +0.0522 UNDECIDED | −0.0019 | ok |
| phishing | LBL | +0.0238 CLEARED | +0.0225 CLEARED | −0.0013 | ok |
| phishing | SUF | +0.0054 CLEARED | +0.0076 CLEARED | +0.0022 | ok |
| phishing | SUB | +0.0334 CLEARED | +0.0337 CLEARED | +0.0003 | ok |
| phishing | PTH | +0.0604 UNDECIDED | +0.0539 UNDECIDED | −0.0065 | ok |
| phishing | QRY | NOT COMPUTABLE | NOT COMPUTABLE | — | ok (both) |

**What this shows:**
- The two retrains agree closely on content: every |Δx| ≤ 0.0065, well inside the
  measured 0.0141.
- Where they differ is honouring in the hardest control states, and free path
  emission.
- Benign PTH changes verdict across the two retrains on a Δx of 0.0039. A verdict
  this close to the bar can flip on retraining alone.

## 3. The investigation (PREREG_E8 Amendment 1)

The registration reads a V-base failure as "the pipeline changed". Every part of
the pipeline that can be checked without training was checked [CONFIRMED]:

| part | E7 | e8_base |
|---|---|---|
| tokenizer state | — | identical to E7's (full state dict) |
| model configuration | — | identical |
| training settings (steps, batch, lr, warm-up, seed, dropout, class_alpha, …) | — | identical |
| data | 2,545,305 / 238,199 kept rows; 10,313 dropped | identical |
| training code | commit `7405606` | `338ed0a`: the network, the loss and the batching unchanged; `train.py`'s E8 additions are empty for E7's tokenizer; the other changes are generation-only |
| environment | torch 2.10.0+cu128, Kaggle | the same |
| weights | — | all 101 tensors differ |

**Why the two differ.**
- `train()` seeds the batch stream and control dropout, but the model is built
  **before** it seeds (`kaggle_train._run`). So the initial weights come from the
  process's unseeded random state. GPU arithmetic is not deterministic either.
- The validation histories already differ at step 1,000 (E7 2.18807, e8_base
  2.18216). e8_base is lower until step 8,000 and ends 0.0094 higher (1.6087
  against 1.5993).
- So `e8_base` is an **independent retrain** of E7's recipe, not a replicate.

**What this means:**
- The V-base failure is **retrain-to-retrain spread, not a pipeline change**
  [CONFIRMED by the identity checks above].
- The ±1-point and ±0.5-point tolerances were **not calibrated on measured
  retraining spread** (plan §6 N1, `PREREG_E8` §5 give no measurement). Only the
  0.0141 gate-cell allowance was measured.
- This pair is the first measurement of that spread for honouring and emission.
  One pair is a weak estimate of it.

## 4. What is and is not claimed

**Claimed** [CONFIRMED]:
- the generation and scoring pipeline is faithful (V1–V4);
- two trainings of E7's recipe differ by up to 5.2 points in honouring and 1.2
  points in path emission;
- no pipeline change was found.

**Not claimed:**
- any K1 outcome for any arm;
- adoption;
- N1b's licence;
- any statement about the presence plan or the root-slash token's effects.

**Not a reading of E8, recorded separately:**
- the plan arms' free emission, which is set by the training objective's length
  weighting (`FINDING_LENGTH_WEIGHTING.md`), verified against the training data
  directly.

## 5. Options for the owner (none taken)

1. **Leave E8 unread,** as registered. Nothing more is spent. E8 then answers
   neither question (plan, slash).
2. **Amend V-base to what it was meant to detect.** Its purpose was to catch a
   pipeline change, and the identity checks in section 3 now rule that out
   directly. Then read E8 exactly as registered.
   - The K1 criteria compare each arm with `e8_base`, the same pipeline and the
     same kind of retrain.
   - The registered attribution rule already refuses any effect smaller than
     |e8_base − E7| on the same metric. So the spread measured here would act as
     the noise floor for each metric.
   - **Caveat:** this amendment would come after the arms' numbers exist. It would
     be recorded as post hoc, and it changes a registered threshold's role, which
     is yours to decide.
3. **Measure the spread first.**
   - Retrain `e8_base` with one or two more seeds, and this time seed the
     initialisation.
   - Generate only what V-base needs: free emission, control honouring and free
     cells.
   - Set V-base from that measurement in a registered amendment, then read E8.
   - Cleanest, and it would give the attribution rule a real noise yardstick.
   - Cost: about 1.2 h training plus about 2 h generation per seed on Kaggle.

**Whichever is chosen, for any future training:** seed the model's initialisation
before building it, so that "E7's seed" really fixes the starting weights.

## 6. Owner decision, 2026-09-28 [DECISION]

Option 1, made permanent (`DECISION_2026-09-28.md`):
- E8 is **permanently unread and obsolete**. E9 addressed directly the
  objective-induced path and query emission bias that E8 was measuring.
- No K1 criterion is ever evaluated, and no further budget goes to E8.
- The artifacts (`data/e8/`) and this record are kept for provenance only.

Sections 1–5 above are the record as written before the decision. They are
unchanged.

## Provenance

- **Kaggle:**
  - training: `data/e8/train/` (PROVENANCE, 4 arms);
  - generation: `data/e8/gen/`, 250 files, 363 min; code `338ed0a`.
- **Laptop:**
  - `data/e8/VERIFY.json` and `verify.log`;
  - `data/e8/score_*.json` (computed as registered; arm cells not read).
- **The investigation's comparisons:** checkpoints `data/best_e7.pt` and
  `data/e8/train/e8_base/best.pt`; histories in `data/e7_kaggle/train_result.json`
  and `data/e8/train/e8_base/train_result.json`.
