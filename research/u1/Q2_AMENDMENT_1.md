# U1 Q2, Amendment 1: D1's dying ReLU (2026-10-23)

**Status:** written and committed **before F2** (no r\* chosen) and before any test set left the
laptop. It changes one line of D1's recipe and the D1 runs that follow from it. Nothing else in
`Q2_PROTOCOL.md` changes.

## 1. What happened
`Q2_train_cnn` ran the registered D1 (E18's CNN) on every arm: 3 regimes × 18 arms × 5 seeds = 270
runs, all scored on the validation set only.

- **29 of 270 runs were broken**: their 1%-FPR operating point did not exist.
  - **15 gave one constant score to every URL** (AUC exactly 0.5): 13 at R100k, 2 at R10k.
  - **14 gave a block of URLs the same top score.** That block held ≥ 1% of the benign validation
    URLs, so recall@1% was 0 by construction, although AUC was 0.89–0.98.
- **Every arm was hit**, every generator, A0's neighbours and A_more alike. Seed 40,000,004 was hit
  most often (13 times).
- **The breaks decided the selection:** with them included or left out, r\* changed in 5 of 12
  (generator, regime) cells, UIE's at R10k among them. For example, R100k A_more's mean validation recall@1% was 0.526 with
  its 2 dead seeds and 0.877 without them. A dead model is dead on the test sets too.

D2 (the n-gram logistic regression) was not affected: all 54 runs converged and none is broken.

## 2. What was looked at before this amendment
Declared, because it was seen before the rule below was written:
- **D2's validation results** (`VAL_lr.json`, 54 runs).
- **D1's registered validation results** (`q2_repair/VAL_cnn_registered_relu_pasted.json`). This is a
  copy pasted from Kaggle's file preview, so its bytes may differ from Kaggle's file; the numbers are
  the same.
- **The diagnosis** (`q2_repair/DIAG_summary.txt`) and **the fix check**
  (`q2_repair/FIXCHECK_summary.txt`, `FIXCHECK_cnn.json`).

No test set was read. The fix was chosen on the mechanism, and the fix check's pass rule is
stability only. Its recall numbers are reported, but they were not a criterion.

## 3. Diagnosis (`q2_diagnose.py`, commits 811e93e and 6b41137; Kaggle CPU)
Every saved D1 model was opened on the validation set, layer by layer:
- **The pooled features never die:** 0 of 320 dead in all 270 runs.
- **The head's 64-unit hidden layer (Linear → ReLU) dies.**
  - Broken runs: median **64 of 64** units never fire.
  - Even healthy runs: median **51 of 64** never fire.
- **In every broken run, the tied score equals the output bias exactly.** The tied URLs are exactly
  those for which no hidden unit fires.
- **Timing:** CPU replays of the registered recipe, verified bit-identical to `cnn_fit` by test, show
  units dying in the first 10–25 steps. One run was fully dead by step 23 and its loss stayed at 0.69
  for the remaining 900 steps.
  - The replays reproduce 10 of 14 GPU breaks, so this is the recipe, not GPU noise.

**Mechanism:** the head's inputs are pooled maxima after a ReLU, all non-negative and large (median
pre-ReLU maximum about 4). Adam moves each of a hidden unit's 320 weights by about the learning rate
per step, so one step can move the unit's input by roughly 2e-3 × 320 × 4 ≈ 2.6 for every URL at
once. A unit pushed below zero for every URL gets no gradient through ReLU and never returns. When
all 64 die, the output is the bias: a constant.

## 4. The change
**D1's head activation is `LeakyReLU(0.01)` instead of `ReLU`.** Nothing else changes: architecture
otherwise, character map, truncation, the pooled ReLU, Adam 2e-3, 6 epochs, batch 256, length
buckets, the seeds, data, arms, regimes, measures, selection rule and test.

- **How:** the activation is swapped after E18's model is built verbatim (`q2_detect.D1_HEAD =
  "leaky_relu_0.01"`). The swap adds no weights and draws no random numbers, so every run starts from
  exactly the weights the registered run started from (tested).
- **Why this fix:** it acts exactly where the failure arises. A leaky unit always passes gradient
  (1%), so it can recover, and the head can no longer output one constant for every URL.
- **Rejected alternatives:**
  - *A lower learning rate:* it changes training everywhere and makes dying rarer, not impossible.
  - *Changing the pooled ReLU:* that layer is not where the failure is.
  - *Re-rolling the broken seeds:* it would select survivors after seeing results, and the healthy
    runs would still be training on about 13 units.
  - *Dropping D1:* it loses the CNN half of Q2, and Q3's CNN path depends on D1.

## 5. Fix check (`Q2_fixcheck_cnn`, commit 0d7bff2; Kaggle CPU) — PASS
With the change, on CPU:
- **The runs:** every broken R1k and R10k run (14), the first three broken R100k runs in training
  order, and the two A0 controls of the diagnosis (19 runs in all).
- **Pass rule:** no run broken by the tie rule (section 6).
- **Result:** **0 of 19 broken.** The largest tie at the 1% threshold is 1 benign URL in 20,000, and
  at least 39,750 of 40,000 validation scores are distinct.
- **Performance (reported, not a criterion):** each repaired run lands in its arm's healthy range.
  The controls barely move: R1k A0 0.142 → 0.147; R10k A0 0.704 → 0.700.

## 6. Consequences, fixed now
1. **All 270 D1 runs are retrained** with the amended recipe (a new `Q2_train_cnn` run; same seeds
   and everything else). The registered D1 runs are superseded. They are reported as the diagnosis
   only and are never used for selection or test.
2. **D2 is unchanged.** Its 54 runs stand.
3. **The broken-run rule** stays a report, not a filter. After the new D1 training, every run is
   checked: *broken = the tie block at the 1%-FPR threshold holds ≥ 1% of the benign validation URLs*
   (a model that cannot be operated at the measured point).
   - Every broken run is **reported and kept as it is**: no re-rolls, no replacement seeds, no
     exclusion.
4. **Selection (F2) and the test read** follow `Q2_PROTOCOL.md` sections 5–6 unchanged, on the new
   D1 records and D2's existing records.
   - Every D1 record and F2 carry `"d1_head": "leaky_relu_0.01"`.
   - `select` refuses D1 records without it, and `test` refuses a selection without it.
5. **Q3** (`PREREG_U1_Q3.md`, LIMNEX repo) is unaffected. Its CNN is LIMNEX's own v3.5 recipe; only
   r\* flows from F2.
6. **Reporting:** U1's results say that D1 is "E18's CNN with a LeakyReLU head (Amendment 1)", and
   give the 29/270 count, the diagnosis and this amendment.

## 7. Evidence in the repo
| file | sha256 |
|---|---|
| `q2_repair/DIAG_summary.txt` | `4df65bd84909db34b083ad413b541f1b90515b49bf9b3bbf080c30139f95681e` |
| `q2_repair/FIXCHECK_summary.txt` | `365debb964923a18fc9119d697f17f0a6d097513a6cc48e2273eec32f5b3e0f2` |
| `q2_repair/FIXCHECK_cnn.json` | `8340c15b6f6fc189e3af2695739b7df2ac903fd629ed7bc45f5960d1931f431f` |
| `q2_repair/VAL_cnn_registered_relu_pasted.json` | `97da5feecd172dc5fb370a6b654359abf2675e17a3257b614d1092cf232504d1` |

These files hold run keys and numbers only: no URL.
