# E11 at step 20,000 — results (amendment 3)

- **Registration:** `PREREG_E11.md` §3.3, amendment 3 (0a1acc5), approved by the
  owner before any run.
  - Everything is the first reading's: the 200 jobs, their seeds and calls, the
    code bundle (commit 6d32e2d), the tables, the instruments and the criteria.
  - Only the checkpoint differs: each arm's `final.pt`, step 20,000.
- **Tools:** b154a06 (`--checkpoint final`, `--train`), plus `e11_side_by_side.py`
  and `e11_final_paste.py` for this document.
- **Generation:** `E11_generate_final`, run by the owner on Kaggle (T4, torch 2.10).
- **Read:** 2026-09-30, after VERIFY passed.
- **The first reading stays in the record as registered** (`E11_RESULTS.md`, read
  at `best.pt`). This document reports the reading at step 20,000 and sets the two
  side by side.
- **Numbers:** two appendices, pasted by script. Every number in the text appears
  there or has a declared source.
  - Appendix A is `e11_tables.py` at step 20,000;
  - Appendix B is `e11_side_by_side.py`: both readings, and amendment 3's
    predictions scored.

## In plain words

- **The question.** E11's first reading compared arms saved at different steps
  (14,000 and 19,000). That confounded every "this change did that". Read at one
  common step, do the results hold?
- **Presence holds.** `e11_plan` passes K11a again, each rate within its tolerance:
  benign path +0.09 points and query −0.05; phishing +0.47 and −0.01.
- **Content was better than the first reading showed.** At step 20,000 `e11_plan`
  has one CONVICTED cell left, against four at step 14,000:

  | `e11_plan` cell | step 14,000 (`best.pt`) | step 20,000 (`final.pt`) |
  |---|---|---|
  | benign control P0Q1 QRY | 0.0847 CONVICTED | **0.0581 UNDECIDED** |
  | benign control P1Q1 QRY | 0.0745 CONVICTED | **0.0511 UNDECIDED** |
  | benign control P1Q1 PTH | 0.0578 CONVICTED | **0.0509 UNDECIDED** |
  | phishing free QRY | 0.1382 CONVICTED | **0.0953 CONVICTED** |
  | phishing control P1Q0 PTH | 0.0592 UNDECIDED | **0.0403 CLEARED** |

  - Every measurable phishing control cell is now CLEARED for the plan.
  - Phishing query content, which barely moved all project, fell by a third.
- **Each change, measured at the same step:**
  - **the repair** moves path presence toward the real rate in both classes
    (benign −1.40 points, phishing +0.70) and leaves content unchanged;
  - **joint dropout** gains nothing: presence moves by 0.3 points at most, and
    the query rates slip;
  - **the plan** makes presence exact, and its control query cells are slightly
    better than `e11_ref`'s;
  - **the slash** leaves content unharmed. This one comparison still crosses
    steps (`e10_base` was read at 18,000).
- **Still no candidate.** `e11_plan` fails the same two criteria as before:
  - K11b, the slash: 3 of 8 rows fail, against 5 before. Phishing bare hosts miss
    the slash 0.36–0.47% of the time, where real ones never do. Benign host-route
    queries in control mode get it too often (99.97% against 99.551%).
  - K11d: control-mode copying is 0.96 points above `e11_ref`'s.
- **What E3 still needs from `e11_plan`:** 7 gated cells (1 CONVICTED, 6
  UNDECIDED) and the 3 slash rows. Four of the seven are query cells, and two more
  are in the path-and-query state.
- **The step matters.** It changed four of `e11_plan`'s verdicts, all toward E3,
  and removed every regression K11c found at step 14,000. The proposal (the owner
  decides): read `final.pt` in every later registration.

## 1. Validity: PASS

| check | result |
|---|---|
| V1 | the 200 registered files with their n, seed and hash; each arm's `final.pt` as E11_train recorded it; generation recorded `final.pt`; one code commit (6d32e2d); the tables and the presence table as registered |
| V2 | byte identity on the laptop from `final.pt`: 1,036 of 1,036 free and 269 of 269 control URLs per hidden arm. The supplied-plan free mode: 712 of 712, located through both shuffles |
| V3 | `uie/` and the common modules unchanged since 6d32e2d |
| V4 | the scorer reproduces E7's published cells exactly: 0.00e+00 in both classes |
| V-id | one shared start (`5aa980f7…`), its part without `<RS>` equal to E10's |

## 2. The criteria, both readings

| arm | K11a | K11b | K11c | K11d | K11e | candidate |
|---|---|---|---|---|---|---|
| `e11_rs` | FAIL → FAIL | FAIL → FAIL | FAIL → **PASS** | FAIL → FAIL | — | no → no |
| `e11_ref` | FAIL → FAIL | FAIL → FAIL | PASS → PASS | PASS → PASS | PASS | no → no |
| `e11_joint` | FAIL → FAIL | FAIL → FAIL | FAIL → **PASS** | FAIL → **PASS** | PASS | no → no |
| `e11_plan` | PASS → PASS | FAIL → FAIL | PASS → PASS | FAIL → FAIL | PASS | no → no |

Each cell reads `best.pt` → `final.pt`. K11e is a training record, the same for
both.

## 3. K11a — presence

| arm | benign PTH | benign QRY | phishing PTH | phishing QRY | K11a |
|---|---|---|---|---|---|
| `e11_rs` | 76.28% (+2.91) | 5.41% (−1.21) | 38.42% (−1.09) | 3.60% (−1.08) | FAIL |
| `e11_ref` | 74.89% (+1.52) | 5.23% (−1.39) | 39.12% (−0.39) | 3.37% (−1.31) | FAIL |
| `e11_joint` | 75.17% (+1.81) | 5.04% (−1.57) | 39.41% (−0.10) | 3.11% (−1.57) | FAIL |
| `e11_plan` | **73.45% (+0.09)** | **6.57% (−0.05)** | **39.98% (+0.47)** | **4.67% (−0.01)** | **PASS** |

- The tolerances are ±0.99, ±0.56, ±1.09 and ±0.47.
- At the common step, every hidden arm's phishing path rate is inside its
  tolerance (`e11_rs` exactly at its edge). None of the hidden arms reaches the
  query tolerance in either class.
- **The plan's own plan** (reported): benign P 74.41%, Q 6.009%. It is not what
  free mode uses; free mode's plan is supplied (D6).

## 4. K11b — the root slash

`e11_plan`'s rows; the other arms are in Appendix A.

| row | step 14,000 | step 20,000 | real | T | within |
|---|---|---|---|---|---|
| benign bare hosts, free | 98.27% | 97.96% | 98.004% | ±0.63 | yes → yes |
| benign bare hosts, control | 98.57% | 98.21% | 97.944% | ±0.52 | no → **yes** |
| benign host-route queries, free | 100.00% | 98.91% | 99.576% | ±1.50 | yes → yes |
| benign host-route queries, control | 99.95% | 99.97% | 99.551% | ±0.24 | no → **no** |
| phishing bare hosts, free | 99.30% | 99.53% | 100.000% | ±0.04 | no → **no** |
| phishing bare hosts, control | 99.65% | 99.64% | 100.000% | ±0.07 | no → **no** |
| phishing host-route queries, free | 98.04% | 98.93% | 100.000% | ±1.54 | no → **yes** |
| phishing host-route queries, control | 99.98% | 100.00% | 100.000% | ±0.20 | yes → yes |

- **Every arm still misses the phishing bare-host rows.** The residuals are the
  same two as before: phishing takes a little of benign's "no slash", and benign
  control mode is too regular on host-route queries.
- The host-route rows are small populations, with tolerances of ±1.50 and ±1.54.
  They moved by up to 1.40 points between the checkpoints; the bare-host rows by
  at most 0.36 (§8).

## 5. K11c and the content cells

- **K11c passes in every arm.** The five regressions found at step 14,000 are
  gone. Each was a 14,000-step arm read against a later reference (Appendix B).
- **`e11_plan`'s gated cells that are not CLEARED at step 20,000:**

  | mode | cell | x | verdict |
  |---|---|---|---|
  | free | benign QRY | 0.0350 | UNDECIDED |
  | free | phishing PTH | 0.0431 | UNDECIDED |
  | free | phishing QRY | 0.0953 | CONVICTED |
  | control | benign P0Q1 QRY | 0.0581 | UNDECIDED |
  | control | benign P1Q1 SUB | 0.0379 | UNDECIDED |
  | control | benign P1Q1 PTH | 0.0509 | UNDECIDED |
  | control | benign P1Q1 QRY | 0.0511 | UNDECIDED |

  Every other gated cell of `e11_plan`, free and control, is CLEARED.
- **FULL** (auxiliary) is at its lowest: 0.0094 benign and 0.0249 phishing in free
  mode.
- **A caution about benign free QRY across arms.** `e11_rs` and `e11_ref` read it
  CLEARED (0.0266 and 0.0268), and `e11_plan` UNDECIDED (0.0350). But the cell
  scores the generated URLs that carry a query. The hidden arms write about a
  fifth fewer (5.41% and 5.23% against 6.57%). A detector trained on fewer
  queries is weaker, so those two CLEARED verdicts are partly lower power, not
  better content. The same holds for phishing QRY, which is not computable for
  `e11_ref` and `e11_joint` at their rates.

## 6. K11d — guardrails

| arm | reference | control-mode copying (points) | reading |
|---|---|---|---|
| `e11_rs` | `e10_base` | +0.92 [+0.35, +1.50] | **FAIL** |
| `e11_ref` | `e11_rs` | +0.57 [−0.01, +1.16] | PASS |
| `e11_joint` | `e11_ref` | +0.26 [−0.33, +0.85] | PASS |
| `e11_plan` | `e11_ref` | +0.96 [+0.38, +1.55] | **FAIL** |

- Novelty moves by at most 0.05 points in any arm.
- **Context, not a criterion.** Every arm copies well below the rate real URLs
  do. Of the gate's real held-out queries, 29.3% (benign) and 41.9% (phishing)
  appear word for word in the training queries [MEASURED, `data/e12/sizing.json`,
  `E12_ROADMAP.md` §2.3]. `e11_plan` copies 19.47% (free) and 22.98% (control),
  both classes pooled. These rises are toward the real rate. K11d is read as
  registered.

## 7. Each change at the same step

The attributions amendment 3 exists for. From Appendix B, the rows at step 20,000:

| change | benign PTH | benign QRY | phishing PTH | phishing QRY | benign free QRY x | benign P0Q1 QRY x | benign P1Q1 QRY x |
|---|---|---|---|---|---|---|---|
| the slash (`e11_rs` against `e10_base`) | +0.40 | +0.34 | +0.55 | +0.71 | −0.0020 | −0.0113 | −0.0077 |
| the repair (`e11_ref` against `e11_rs`) | −1.40 | −0.18 | +0.70 | −0.23 | +0.0002 | −0.0041 | +0.0012 |
| joint dropout (`e11_joint` against `e11_ref`) | +0.29 | −0.19 | +0.29 | −0.26 | +0.0123 | +0.0017 | +0.0000 |
| the plan (`e11_plan` against `e11_ref`) | −1.43 | +1.34 | +0.86 | +1.30 | +0.0082 | −0.0038 | −0.0073 |

- **The slash** leaves content unharmed. Its cells move the right way, within the
  retrain allowance. `e10_base` was read at step 18,000, so this row still
  crosses steps.
- **The repair does what PC1 said, and nothing else.**
  - benign path −1.40 points, toward the real rate from above;
  - phishing path +0.70, toward it from below;
  - every content cell within 0.0041.
  - At the saved checkpoints the same comparison read −2.72 and +2.90: the step
    had inflated both.
- **Joint dropout gains nothing.**
  - Path presence moves by 0.29 points in each class.
  - The query rates slip (−0.19 and −0.26 points).
  - At the saved checkpoints it read +2.92 on benign paths: that was the step.
  - This answers E10's registered follow-up question: doubling the exposure of
    the states generation uses does not raise the hidden state's query rate.
- **The plan** makes presence exact, and its control query cells are slightly
  better than `e11_ref`'s. Its free QRY cell is 0.0082 higher, read at more
  queries, so at higher power (§5).

## 8. Amendment 3's predictions, scored

From Appendix B:

| prediction | scored |
|---|---|
| `e11_plan` passes K11a again | right |
| FULL moves by at most 0.02 in every arm | **wrong**: six readings moved by more, the largest −0.0274, all downward |
| the slash rows move by at most 0.3 points | **wrong**: the small host-route rows moved by up to 1.40, and three bare-host rows by 0.32–0.36 |
| K11b still fails on the phishing bare-host rows | right, in every arm |
| benign free PTH about 76% / 74.5% / 74.5–75% | 76.28% / 74.89% / 75.17% |
| joint dropout moves benign free PTH by under 1 point | right (+0.29) |
| the three 14,000-step arms improve on benign free QRY, to within 0.01 of `e11_ref`'s | partly: all three improve; `e11_joint` stays outside 0.01 |
| `e11_plan`'s control query cells improve | right: both reach UNDECIDED (no prediction was made on that) |
| K11c's regressions shrink | right: all five vanish |
| K11d's copying rises shrink | partly: three shrink; `e11_plan`'s grows |

## 9. Candidates, and the E3 finish line

- **No arm is a candidate.** `e11_plan` fails K11b and K11d. The hidden arms also
  fail K11a.
- **The E3 finish line** (`DECISION_2026-09-29.md` §3), for `e11_plan` at step
  20,000:
  - gated cells: the 7 in §5 are not CLEARED;
  - the slash: 3 of 8 rows miss their tolerance.

## 10. What this changes (the owner decides)

1. **Read `final.pt` from now on.**
   - The step changed four of `e11_plan`'s verdicts, all toward E3.
   - It removed all five of K11c's regressions.
   - It shrank every attribution in §7, and reversed one (the slash on phishing
     paths, −1.83 at the saved checkpoints, +0.55 at step 20,000).
   - With the validation curve flat from step 14,000, `best.pt`'s step is noise.
     The learning rate ends at zero, so step 20,000 is where training settles.
2. **`e11_plan`'s recipe is the natural base for E12:** the slash, the repair, the
   plan, and the supplied plan in free mode. It is exact on presence, the best arm
   on content, and clears every phishing control cell.
3. **Joint dropout can be dropped.** At the same step it changes nothing that
   matters.
4. **Query content is what is left.** `E12_ROADMAP.md` is the draft.

## 11. What is and is not claimed

**Claimed** [CONFIRMED by the registered readout at step 20,000]:
- `e11_plan` meets the presence tolerance at step 20,000, as it did at 14,000;
- at the same step, the repair moves path presence toward the real rate in both
  classes and leaves content unchanged, and joint dropout gains nothing;
- at step 20,000 `e11_plan`'s benign control query cells and its P1Q1 path cell
  read UNDECIDED, and its phishing free QRY cell falls to 0.0953, still CONVICTED;
- no arm is a candidate.

**Not claimed:**
- that step 20,000 is better than step 14,000 in general: one training per arm;
- anything for another training;
- that the hidden arms' CLEARED benign free QRY reflects better content (§5).

## Appendix A — the rendered tables at step 20,000

`python research/native/e11_tables.py --e11 data/e11_final --train data/e11/train`, verbatim.

### Validity

Verdict: **PASS**; failures: none

Shared start: {'e11_rs': '5aa980f7c6ada9dacaef3d9a03646c23973a57e9901d622388935c69a5050aa0', 'e11_ref': '5aa980f7c6ada9dacaef3d9a03646c23973a57e9901d622388935c69a5050aa0', 'e11_joint': '5aa980f7c6ada9dacaef3d9a03646c23973a57e9901d622388935c69a5050aa0', 'e11_plan': '5aa980f7c6ada9dacaef3d9a03646c23973a57e9901d622388935c69a5050aa0'}; its part without <RS> equals E10's: True

| arm / mode | URLs regenerated | identical |
|---|---|---|
| e11_rs/free | 1036 | 1036 |
| e11_rs/control | 269 | 269 |
| e11_ref/free | 1036 | 1036 |
| e11_ref/control | 269 | 269 |
| e11_joint/free | 1036 | 1036 |
| e11_joint/control | 269 | 269 |
| e11_plan/free | 712 | 712 |
| e11_plan/control | 269 | 269 |

V4 benign: max |x − published| = 0.00e+00

V4 phishing: max |x − published| = 0.00e+00

### Training

| arm | final val (benign / phishing / mean) | best step | minutes | rows that fit | dropped |
|---|---|---|---|---|---|
| e11_rs | 1.6650 / 1.5082 / 1.5866 | 14000 | 60.4 | {'benign': 2545303, 'phishing': 238198} | 10,316 |
| e11_ref | 1.6652 / 1.5063 / 1.5858 | 19000 | 60.5 | {'benign': 2545303, 'phishing': 238198} | 10,316 |
| e11_joint | 1.6652 / 1.5040 / 1.5846 | 14000 | 60.5 | {'benign': 2545303, 'phishing': 238198} | 10,316 |
| e11_plan | 1.5995 / 1.4376 / 1.5186 | 14000 | 60.2 | {'benign': 2545303, 'phishing': 238198} | 10,316 |

### Clipping after step 1,000 (K11e for the repair arms)

| arm | band | steps | mean norm | clipped (own cap) | over 1.0 |
|---|---|---|---|---|---|
| e11_rs | 0-30 | 4,353 | 0.894 | 20.4% | — |
| e11_rs | 30-60 | 9,005 | 0.578 | 0.0% | — |
| e11_rs | 60-120 | 4,861 | 0.449 | 0.0% | — |
| e11_rs | 120-inf | 781 | 0.403 | 1.4% | — |
| e11_ref | 0-30 | 4,353 | 0.871 | 0.0% | 18.1% |
| e11_ref | 30-60 | 9,005 | 0.577 | 0.0% | 0.0% |
| e11_ref | 60-120 | 4,861 | 0.448 | 0.0% | 0.0% |
| e11_ref | 120-inf | 781 | 0.402 | 0.0% | 1.7% |
| e11_joint | 0-30 | 4,353 | 0.875 | 0.0% | 18.5% |
| e11_joint | 30-60 | 9,005 | 0.579 | 0.0% | 0.0% |
| e11_joint | 60-120 | 4,861 | 0.450 | 0.0% | 0.0% |
| e11_joint | 120-inf | 781 | 0.404 | 0.0% | 1.5% |
| e11_plan | 0-30 | 3,595 | 0.921 | 0.0% | 27.5% |
| e11_plan | 30-60 | 9,428 | 0.562 | 0.0% | 0.0% |
| e11_plan | 60-120 | 5,096 | 0.442 | 0.0% | 0.0% |
| e11_plan | 120-inf | 881 | 0.402 | 0.0% | 1.0% |

### K11a — free-mode presence against the tolerance

| arm | row | generated against real (difference, T) within |
|---|---|---|
| e11_rs | benign PTH | 76.28% against 73.369% (+2.91, ±0.99) **no** |
| e11_rs | benign QRY | 5.41% against 6.617% (-1.21, ±0.56) **no** |
| e11_rs | phishing PTH | 38.42% against 39.511% (-1.09, ±1.09) yes |
| e11_rs | phishing QRY | 3.60% against 4.680% (-1.08, ±0.47) **no** |
| e11_ref | benign PTH | 74.89% against 73.369% (+1.52, ±0.99) **no** |
| e11_ref | benign QRY | 5.23% against 6.617% (-1.39, ±0.56) **no** |
| e11_ref | phishing PTH | 39.12% against 39.511% (-0.39, ±1.09) yes |
| e11_ref | phishing QRY | 3.37% against 4.680% (-1.31, ±0.47) **no** |
| e11_joint | benign PTH | 75.17% against 73.369% (+1.81, ±0.99) **no** |
| e11_joint | benign QRY | 5.04% against 6.617% (-1.57, ±0.56) **no** |
| e11_joint | phishing PTH | 39.41% against 39.511% (-0.10, ±1.09) yes |
| e11_joint | phishing QRY | 3.11% against 4.680% (-1.57, ±0.47) **no** |
| e11_plan | benign PTH | 73.45% against 73.369% (+0.09, ±0.99) yes |
| e11_plan | benign QRY | 6.57% against 6.617% (-0.05, ±0.56) yes |
| e11_plan | phishing PTH | 39.98% against 39.511% (+0.47, ±1.09) yes |
| e11_plan | phishing QRY | 4.67% against 4.680% (-0.01, ±0.47) yes |

### K11b — the root slash against the tolerance

| arm | row | generated against real (difference, T) within |
|---|---|---|
| e11_rs | benign free_bare | 98.13% against 98.004% (+0.12, ±0.63) yes |
| e11_rs | benign control P0Q0 | 98.08% against 97.944% (+0.13, ±0.52) yes |
| e11_rs | benign free_host_query | 98.96% against 99.576% (-0.61, ±1.50) yes |
| e11_rs | benign control P0Q1 | 99.86% against 99.551% (+0.31, ±0.24) **no** |
| e11_rs | phishing free_bare | 99.59% against 100.000% (-0.41, ±0.04) **no** |
| e11_rs | phishing control P0Q0 | 99.58% against 100.000% (-0.42, ±0.07) **no** |
| e11_rs | phishing free_host_query | 99.40% against 100.000% (-0.60, ±1.54) yes |
| e11_rs | phishing control P0Q1 | 99.94% against 100.000% (-0.06, ±0.20) yes |
| e11_ref | benign free_bare | 98.36% against 98.004% (+0.36, ±0.63) yes |
| e11_ref | benign control P0Q0 | 98.42% against 97.944% (+0.48, ±0.52) yes |
| e11_ref | benign free_host_query | 98.20% against 99.576% (-1.37, ±1.50) yes |
| e11_ref | benign control P0Q1 | 99.89% against 99.551% (+0.34, ±0.24) **no** |
| e11_ref | phishing free_bare | 99.51% against 100.000% (-0.49, ±0.04) **no** |
| e11_ref | phishing control P0Q0 | 99.66% against 100.000% (-0.34, ±0.07) **no** |
| e11_ref | phishing free_host_query | 98.77% against 100.000% (-1.23, ±1.54) yes |
| e11_ref | phishing control P0Q1 | 99.98% against 100.000% (-0.02, ±0.20) yes |
| e11_joint | benign free_bare | 98.20% against 98.004% (+0.19, ±0.63) yes |
| e11_joint | benign control P0Q0 | 98.42% against 97.944% (+0.48, ±0.52) yes |
| e11_joint | benign free_host_query | 96.36% against 99.576% (-3.22, ±1.50) **no** |
| e11_joint | benign control P0Q1 | 99.86% against 99.551% (+0.31, ±0.24) **no** |
| e11_joint | phishing free_bare | 99.56% against 100.000% (-0.44, ±0.04) **no** |
| e11_joint | phishing control P0Q0 | 99.74% against 100.000% (-0.26, ±0.07) **no** |
| e11_joint | phishing free_host_query | 96.70% against 100.000% (-3.30, ±1.54) **no** |
| e11_joint | phishing control P0Q1 | 99.94% against 100.000% (-0.06, ±0.20) yes |
| e11_plan | benign free_bare | 97.96% against 98.004% (-0.05, ±0.63) yes |
| e11_plan | benign control P0Q0 | 98.21% against 97.944% (+0.27, ±0.52) yes |
| e11_plan | benign free_host_query | 98.91% against 99.576% (-0.67, ±1.50) yes |
| e11_plan | benign control P0Q1 | 99.97% against 99.551% (+0.42, ±0.24) **no** |
| e11_plan | phishing free_bare | 99.53% against 100.000% (-0.47, ±0.04) **no** |
| e11_plan | phishing control P0Q0 | 99.64% against 100.000% (-0.36, ±0.07) **no** |
| e11_plan | phishing free_host_query | 98.93% against 100.000% (-1.07, ±1.54) yes |
| e11_plan | phishing control P0Q1 | 100.00% against 100.000% (+0.00, ±0.20) yes |

### Gate v2, FULL FREE (x, verdict)

| class | field | e11_rs | e11_ref | e11_joint | e11_plan |
|---|---|---|---|---|---|
| benign | LBL | +0.0099 CLEA | +0.0059 CLEA | +0.0111 CLEA | +0.0081 CLEA |
| benign | SUF | +0.0148 CLEA | +0.0129 CLEA | +0.0163 CLEA | +0.0161 CLEA |
| benign | SUB | +0.0050 CLEA | +0.0047 CLEA | +0.0032 CLEA | +0.0038 CLEA |
| benign | PTH | +0.0069 CLEA | +0.0031 CLEA | +0.0034 CLEA | +0.0058 CLEA |
| benign | QRY | +0.0266 CLEA | +0.0268 CLEA | +0.0391 UNDE | +0.0350 UNDE |
| benign | FULL | +0.0270 AUXI | +0.0185 AUXI | +0.0221 AUXI | +0.0094 AUXI |
| phishing | LBL | +0.0133 CLEA | +0.0100 CLEA | +0.0108 CLEA | +0.0107 CLEA |
| phishing | SUF | +0.0095 CLEA | +0.0093 CLEA | +0.0055 CLEA | +0.0107 CLEA |
| phishing | SUB | +0.0186 CLEA | +0.0213 CLEA | +0.0192 CLEA | +0.0191 CLEA |
| phishing | PTH | +0.0356 CLEA | +0.0350 CLEA | +0.0388 CLEA | +0.0431 UNDE |
| phishing | QRY | +0.0868 CONV | NOT COMPUTABLE | NOT COMPUTABLE | +0.0953 CONV |
| phishing | FULL | +0.0223 AUXI | +0.0188 AUXI | +0.0227 AUXI | +0.0249 AUXI |

### FULL CONTROL, measurable states (x, verdict)

| state | field | e11_rs | e11_ref | e11_joint | e11_plan |
|---|---|---|---|---|---|
| benign_P0Q0 | LBL | +0.0133 CLEA | +0.0094 CLEA | +0.0074 CLEA | +0.0107 CLEA |
| benign_P0Q0 | SUF | +0.0016 CLEA | -0.0003 CLEA | +0.0013 CLEA | -0.0017 CLEA |
| benign_P0Q0 | SUB | +0.0079 CLEA | +0.0142 CLEA | +0.0152 CLEA | +0.0153 CLEA |
| benign_P0Q0 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q0 | FULL | +0.0145 AUXI | +0.0100 AUXI | +0.0082 AUXI | +0.0107 AUXI |
| benign_P0Q1 | LBL | +0.0271 CLEA | +0.0115 CLEA | +0.0197 CLEA | +0.0227 CLEA |
| benign_P0Q1 | SUF | +0.0092 CLEA | +0.0094 CLEA | +0.0110 CLEA | +0.0085 CLEA |
| benign_P0Q1 | SUB | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q1 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q1 | QRY | +0.0660 CONV | +0.0619 CONV | +0.0636 CONV | +0.0581 UNDE |
| benign_P0Q1 | FULL | +0.0804 AUXI | +0.0749 AUXI | +0.0709 AUXI | +0.0762 AUXI |
| benign_P1Q0 | LBL | -0.0013 CLEA | +0.0042 CLEA | +0.0018 CLEA | +0.0024 CLEA |
| benign_P1Q0 | SUF | +0.0008 CLEA | -0.0009 CLEA | +0.0031 CLEA | +0.0010 CLEA |
| benign_P1Q0 | SUB | +0.0057 CLEA | +0.0040 CLEA | +0.0079 CLEA | +0.0028 CLEA |
| benign_P1Q0 | PTH | +0.0110 CLEA | +0.0115 CLEA | +0.0091 CLEA | +0.0125 CLEA |
| benign_P1Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P1Q0 | FULL | +0.0131 AUXI | +0.0126 AUXI | +0.0158 AUXI | +0.0135 AUXI |
| benign_P1Q1 | LBL | +0.0253 CLEA | +0.0270 CLEA | +0.0261 CLEA | +0.0257 CLEA |
| benign_P1Q1 | SUF | +0.0008 CLEA | +0.0026 CLEA | -0.0017 CLEA | -0.0029 CLEA |
| benign_P1Q1 | SUB | +0.0669 CONV | +0.0644 CONV | +0.0708 CONV | +0.0379 UNDE |
| benign_P1Q1 | PTH | +0.0806 CONV | +0.0836 CONV | +0.0773 CONV | +0.0509 UNDE |
| benign_P1Q1 | QRY | +0.0572 UNDE | +0.0584 CONV | +0.0584 CONV | +0.0511 UNDE |
| benign_P1Q1 | FULL | +0.0786 AUXI | +0.0853 AUXI | +0.0825 AUXI | +0.0688 AUXI |
| phishing_P0Q0 | LBL | +0.0335 CLEA | +0.0303 CLEA | +0.0288 CLEA | +0.0238 CLEA |
| phishing_P0Q0 | SUF | -0.0019 CLEA | -0.0025 CLEA | -0.0034 CLEA | -0.0028 CLEA |
| phishing_P0Q0 | SUB | +0.0326 CLEA | +0.0339 UNDE | +0.0349 CLEA | +0.0326 CLEA |
| phishing_P0Q0 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P0Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P0Q0 | FULL | +0.0439 AUXI | +0.0363 AUXI | +0.0322 AUXI | +0.0322 AUXI |
| phishing_P1Q0 | LBL | +0.0346 CLEA | +0.0337 CLEA | +0.0386 CLEA | +0.0173 CLEA |
| phishing_P1Q0 | SUF | +0.0031 CLEA | -0.0005 CLEA | -0.0008 CLEA | +0.0010 CLEA |
| phishing_P1Q0 | SUB | +0.0454 UNDE | +0.0459 UNDE | +0.0428 UNDE | +0.0286 CLEA |
| phishing_P1Q0 | PTH | +0.0362 CLEA | +0.0377 CLEA | +0.0400 CLEA | +0.0403 CLEA |
| phishing_P1Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P1Q0 | FULL | +0.0460 AUXI | +0.0521 AUXI | +0.0500 AUXI | +0.0387 AUXI |

### K11c — no regression against each arm's reference

| arm | reference | regressions |
|---|---|---|
| e11_rs | e10_base | none |
| e11_ref | e11_rs | none |
| e11_joint | e11_ref | none |
| e11_plan | e11_ref | none |

### Honouring (control, native)

| state | e11_rs | e11_ref | e11_joint | e11_plan |
|---|---|---|---|---|
| benign P1Q1 | 93.71% | 94.67% | 93.55% | 98.34% |
| benign P1Q0 | 98.50% | 98.37% | 98.63% | 99.71% |
| benign P0Q1 | 98.16% | 97.81% | 97.79% | 98.59% |
| benign P0Q0 | 99.15% | 99.11% | 99.11% | 99.56% |
| phishing P1Q1 | 82.00% | 84.04% | 81.98% | 96.82% |
| phishing P1Q0 | 97.82% | 97.85% | 98.05% | 99.38% |
| phishing P0Q1 | 97.72% | 97.70% | 97.10% | 98.68% |
| phishing P0Q0 | 98.93% | 98.90% | 98.81% | 99.55% |

### K11d — guardrails against each arm's reference

| arm | mode | novel | copied queries |
|---|---|---|---|
| e11_rs | free | 99.55% | 21.28% of 3,604 |
| e11_rs | control | 99.64% | 21.45% of 38,506 |
| e11_ref | free | 99.52% | 20.43% of 3,441 |
| e11_ref | control | 99.69% | 22.02% of 38,645 |
| e11_joint | free | 99.53% | 21.35% of 3,260 |
| e11_joint | control | 99.66% | 22.28% of 38,282 |
| e11_plan | free | 99.55% | 19.47% of 4,495 |
| e11_plan | control | 99.64% | 22.98% of 39,535 |
- e11_rs against e10_base: **FAIL** free_novelty +0.01 pts [-0.06, +0.07]; free_copying +0.24 pts [-1.71, +2.19]; control_novelty +0.05 pts [-0.00, +0.10]; control_copying +0.92 pts [+0.35, +1.50]
- e11_ref against e11_rs: **PASS** free_novelty +0.03 pts [-0.04, +0.10]; free_copying -0.85 pts [-2.75, +1.05]; control_novelty -0.05 pts [-0.10, +0.01]; control_copying +0.57 pts [-0.01, +1.16]
- e11_joint against e11_ref: **PASS** free_novelty -0.01 pts [-0.08, +0.06]; free_copying +0.92 pts [-1.03, +2.87]; control_novelty +0.03 pts [-0.02, +0.08]; control_copying +0.26 pts [-0.33, +0.85]
- e11_plan against e11_ref: **FAIL** free_novelty -0.03 pts [-0.10, +0.04]; free_copying -0.96 pts [-2.74, +0.81]; control_novelty +0.05 pts [-0.00, +0.10]; control_copying +0.96 pts [+0.38, +1.55]

### Reported: e11_plan's own plan (checkpoint, shipped-weighted)

| class | plan P | plan Q |
|---|---|---|
| benign | 74.41% | 6.009% |
| phishing | 39.02% | 3.890% |

### Attribution — each arm against its reference (reported)

| arm | reference | emission change (points) |
|---|---|---|
| e11_rs | e10_base | benign PTH +0.40, benign QRY +0.34, phishing PTH +0.55, phishing QRY +0.71 |
| e11_ref | e11_rs | benign PTH -1.40, benign QRY -0.18, phishing PTH +0.70, phishing QRY -0.23 |
| e11_joint | e11_ref | benign PTH +0.29, benign QRY -0.19, phishing PTH +0.29, phishing QRY -0.26 |
| e11_plan | e11_ref | benign PTH -1.43, benign QRY +1.34, phishing PTH +0.86, phishing QRY +1.30 |

### Outcome

| arm | failing | candidate |
|---|---|---|
| e11_rs | K11a, K11b, K11d | **no** |
| e11_ref | K11a, K11b | **no** |
| e11_joint | K11a, K11b | **no** |
| e11_plan | K11b, K11d | **no** |

## Appendix B — side by side with best.pt, and amendment 3's predictions

`python research/native/e11_side_by_side.py`, verbatim.

### The criteria at both checkpoints

| arm | best.pt step | K11a | K11b | K11c | K11d | K11e | candidate |
|---|---|---|---|---|---|---|---|
| e11_rs | 14,000 | FAIL → **FAIL** | FAIL → **FAIL** | FAIL → PASS | FAIL → **FAIL** | — | no → **no** |
| e11_ref | 19,000 | FAIL → **FAIL** | FAIL → **FAIL** | PASS → PASS | PASS → PASS | PASS → PASS | no → **no** |
| e11_joint | 14,000 | FAIL → **FAIL** | FAIL → **FAIL** | FAIL → PASS | FAIL → PASS | PASS → PASS | no → **no** |
| e11_plan | 14,000 | PASS → PASS | FAIL → **FAIL** | PASS → PASS | FAIL → **FAIL** | PASS → PASS | no → **no** |

Each cell reads best.pt → final.pt (step 20,000). K11e is a training record, the same for both.

### K11a — free-mode presence, best.pt → final.pt

| arm | row | best.pt | final.pt | change (points) | real | T | within |
|---|---|---|---|---|---|---|---|
| e11_rs | benign PTH | 77.09% | 76.28% | -0.81 | 73.369% | ±0.99 | no → **no** |
| e11_rs | benign QRY | 5.55% | 5.41% | -0.15 | 6.617% | ±0.56 | no → **no** |
| e11_rs | phishing PTH | 36.04% | 38.42% | +2.38 | 39.511% | ±1.09 | no → yes |
| e11_rs | phishing QRY | 3.28% | 3.60% | +0.32 | 4.680% | ±0.47 | no → **no** |
| e11_ref | benign PTH | 74.36% | 74.89% | +0.52 | 73.369% | ±0.99 | no → **no** |
| e11_ref | benign QRY | 4.99% | 5.23% | +0.24 | 6.617% | ±0.56 | no → **no** |
| e11_ref | phishing PTH | 38.94% | 39.12% | +0.18 | 39.511% | ±1.09 | yes → yes |
| e11_ref | phishing QRY | 3.33% | 3.37% | +0.04 | 4.680% | ±0.47 | no → **no** |
| e11_joint | benign PTH | 77.28% | 75.17% | -2.10 | 73.369% | ±0.99 | no → **no** |
| e11_joint | benign QRY | 5.43% | 5.04% | -0.39 | 6.617% | ±0.56 | no → **no** |
| e11_joint | phishing PTH | 38.78% | 39.41% | +0.64 | 39.511% | ±1.09 | yes → yes |
| e11_joint | phishing QRY | 3.08% | 3.11% | +0.03 | 4.680% | ±0.47 | no → **no** |
| e11_plan | benign PTH | 73.38% | 73.45% | +0.08 | 73.369% | ±0.99 | yes → yes |
| e11_plan | benign QRY | 6.57% | 6.57% | -0.00 | 6.617% | ±0.56 | yes → yes |
| e11_plan | phishing PTH | 39.90% | 39.98% | +0.08 | 39.511% | ±1.09 | yes → yes |
| e11_plan | phishing QRY | 4.71% | 4.67% | -0.04 | 4.680% | ±0.47 | yes → yes |

### K11b — the root slash, best.pt → final.pt

| arm | row | best.pt | final.pt | change (points) | real | T | within |
|---|---|---|---|---|---|---|---|
| e11_rs | benign free_bare | 98.30% | 98.13% | -0.17 | 98.004% | ±0.63 | yes → yes |
| e11_rs | benign control P0Q0 | 98.32% | 98.08% | -0.24 | 97.944% | ±0.52 | yes → yes |
| e11_rs | benign free_host_query | 98.31% | 98.96% | +0.65 | 99.576% | ±1.50 | yes → yes |
| e11_rs | benign control P0Q1 | 99.91% | 99.86% | -0.05 | 99.551% | ±0.24 | no → **no** |
| e11_rs | phishing free_bare | 99.48% | 99.59% | +0.11 | 100.000% | ±0.04 | no → **no** |
| e11_rs | phishing control P0Q0 | 99.59% | 99.58% | -0.01 | 100.000% | ±0.07 | no → **no** |
| e11_rs | phishing free_host_query | 98.95% | 99.40% | +0.45 | 100.000% | ±1.54 | yes → yes |
| e11_rs | phishing control P0Q1 | 99.98% | 99.94% | -0.04 | 100.000% | ±0.20 | yes → yes |
| e11_ref | benign free_bare | 98.04% | 98.36% | +0.33 | 98.004% | ±0.63 | yes → yes |
| e11_ref | benign control P0Q0 | 98.38% | 98.42% | +0.04 | 97.944% | ±0.52 | yes → yes |
| e11_ref | benign free_host_query | 98.38% | 98.20% | -0.18 | 99.576% | ±1.50 | yes → yes |
| e11_ref | benign control P0Q1 | 99.91% | 99.89% | -0.02 | 99.551% | ±0.24 | no → **no** |
| e11_ref | phishing free_bare | 99.59% | 99.51% | -0.08 | 100.000% | ±0.04 | no → **no** |
| e11_ref | phishing control P0Q0 | 99.73% | 99.66% | -0.07 | 100.000% | ±0.07 | no → **no** |
| e11_ref | phishing free_host_query | 98.41% | 98.77% | +0.35 | 100.000% | ±1.54 | no → yes |
| e11_ref | phishing control P0Q1 | 99.96% | 99.98% | +0.02 | 100.000% | ±0.20 | yes → yes |
| e11_joint | benign free_bare | 98.29% | 98.20% | -0.10 | 98.004% | ±0.63 | yes → yes |
| e11_joint | benign control P0Q0 | 98.45% | 98.42% | -0.03 | 97.944% | ±0.52 | yes → yes |
| e11_joint | benign free_host_query | 97.07% | 96.36% | -0.71 | 99.576% | ±1.50 | no → **no** |
| e11_joint | benign control P0Q1 | 99.86% | 99.86% | -0.00 | 99.551% | ±0.24 | no → **no** |
| e11_joint | phishing free_bare | 99.46% | 99.56% | +0.10 | 100.000% | ±0.04 | no → **no** |
| e11_joint | phishing control P0Q0 | 99.66% | 99.74% | +0.08 | 100.000% | ±0.07 | no → **no** |
| e11_joint | phishing free_host_query | 98.10% | 96.70% | -1.40 | 100.000% | ±1.54 | no → **no** |
| e11_joint | phishing control P0Q1 | 99.98% | 99.94% | -0.04 | 100.000% | ±0.20 | yes → yes |
| e11_plan | benign free_bare | 98.27% | 97.96% | -0.32 | 98.004% | ±0.63 | yes → yes |
| e11_plan | benign control P0Q0 | 98.57% | 98.21% | -0.36 | 97.944% | ±0.52 | no → yes |
| e11_plan | benign free_host_query | 100.00% | 98.91% | -1.09 | 99.576% | ±1.50 | yes → yes |
| e11_plan | benign control P0Q1 | 99.95% | 99.97% | +0.02 | 99.551% | ±0.24 | no → **no** |
| e11_plan | phishing free_bare | 99.30% | 99.53% | +0.23 | 100.000% | ±0.04 | no → **no** |
| e11_plan | phishing control P0Q0 | 99.65% | 99.64% | -0.01 | 100.000% | ±0.07 | no → **no** |
| e11_plan | phishing free_host_query | 98.04% | 98.93% | +0.89 | 100.000% | ±1.54 | no → yes |
| e11_plan | phishing control P0Q1 | 99.98% | 100.00% | +0.02 | 100.000% | ±0.20 | yes → yes |

### Gate v2, FULL FREE, best.pt → final.pt (x, verdict)

| class | field | e11_rs | e11_ref | e11_joint | e11_plan |
|---|---|---|---|---|---|
| benign | LBL | +0.0080 CLEA → +0.0099 CLEA | +0.0061 CLEA → +0.0059 CLEA | +0.0110 CLEA → +0.0111 CLEA | +0.0086 CLEA → +0.0081 CLEA |
| benign | SUF | +0.0171 CLEA → +0.0148 CLEA | +0.0148 CLEA → +0.0129 CLEA | +0.0124 CLEA → +0.0163 CLEA | +0.0184 CLEA → +0.0161 CLEA |
| benign | SUB | +0.0076 CLEA → +0.0050 CLEA | +0.0063 CLEA → +0.0047 CLEA | +0.0035 CLEA → +0.0032 CLEA | +0.0030 CLEA → +0.0038 CLEA |
| benign | PTH | +0.0163 CLEA → +0.0069 CLEA | +0.0046 CLEA → +0.0031 CLEA | +0.0142 CLEA → +0.0034 CLEA | +0.0129 CLEA → +0.0058 CLEA |
| benign | QRY | +0.0489 UNDE → +0.0266 CLEA | +0.0299 UNDE → +0.0268 CLEA | +0.0512 UNDE → +0.0391 UNDE | +0.0387 UNDE → +0.0350 UNDE |
| benign | FULL | +0.0295 AUXI → +0.0270 AUXI | +0.0212 AUXI → +0.0185 AUXI | +0.0299 AUXI → +0.0221 AUXI | +0.0243 AUXI → +0.0094 AUXI |
| phishing | LBL | +0.0333 CLEA → +0.0133 CLEA | +0.0101 CLEA → +0.0100 CLEA | +0.0212 CLEA → +0.0108 CLEA | +0.0215 CLEA → +0.0107 CLEA |
| phishing | SUF | +0.0095 CLEA → +0.0095 CLEA | +0.0068 CLEA → +0.0093 CLEA | +0.0082 CLEA → +0.0055 CLEA | +0.0079 CLEA → +0.0107 CLEA |
| phishing | SUB | +0.0378 CLEA → +0.0186 CLEA | +0.0225 CLEA → +0.0213 CLEA | +0.0365 CLEA → +0.0192 CLEA | +0.0306 CLEA → +0.0191 CLEA |
| phishing | PTH | +0.0556 UNDE → +0.0356 CLEA | +0.0354 UNDE → +0.0350 CLEA | +0.0582 UNDE → +0.0388 CLEA | +0.0533 UNDE → +0.0431 UNDE |
| phishing | QRY | NOT COMPUTABLE → +0.0868 CONV | NOT COMPUTABLE → NOT COMPUTABLE | NOT COMPUTABLE → NOT COMPUTABLE | +0.1382 CONV → +0.0953 CONV |
| phishing | FULL | +0.0497 AUXI → +0.0223 AUXI | +0.0206 AUXI → +0.0188 AUXI | +0.0394 AUXI → +0.0227 AUXI | +0.0393 AUXI → +0.0249 AUXI |

### FULL CONTROL, measurable states, best.pt → final.pt (x, verdict)

| state | field | e11_rs | e11_ref | e11_joint | e11_plan |
|---|---|---|---|---|---|
| benign_P0Q0 | LBL | +0.0124 CLEA → +0.0133 CLEA | +0.0081 CLEA → +0.0094 CLEA | +0.0123 CLEA → +0.0074 CLEA | +0.0095 CLEA → +0.0107 CLEA |
| benign_P0Q0 | SUF | +0.0010 CLEA → +0.0016 CLEA | -0.0023 CLEA → -0.0003 CLEA | -0.0017 CLEA → +0.0013 CLEA | -0.0005 CLEA → -0.0017 CLEA |
| benign_P0Q0 | SUB | +0.0119 CLEA → +0.0079 CLEA | +0.0139 CLEA → +0.0142 CLEA | +0.0126 CLEA → +0.0152 CLEA | +0.0168 CLEA → +0.0153 CLEA |
| benign_P0Q0 | FULL | +0.0124 AUXI → +0.0145 AUXI | +0.0107 AUXI → +0.0100 AUXI | +0.0137 AUXI → +0.0082 AUXI | +0.0104 AUXI → +0.0107 AUXI |
| benign_P0Q1 | LBL | +0.0223 CLEA → +0.0271 CLEA | +0.0196 CLEA → +0.0115 CLEA | +0.0221 CLEA → +0.0197 CLEA | +0.0179 CLEA → +0.0227 CLEA |
| benign_P0Q1 | SUF | +0.0085 CLEA → +0.0092 CLEA | +0.0095 CLEA → +0.0094 CLEA | +0.0138 CLEA → +0.0110 CLEA | +0.0111 CLEA → +0.0085 CLEA |
| benign_P0Q1 | QRY | +0.0883 CONV → +0.0660 CONV | +0.0615 CONV → +0.0619 CONV | +0.0817 CONV → +0.0636 CONV | +0.0847 CONV → +0.0581 UNDE |
| benign_P0Q1 | FULL | +0.0979 AUXI → +0.0804 AUXI | +0.0791 AUXI → +0.0749 AUXI | +0.0875 AUXI → +0.0709 AUXI | +0.1013 AUXI → +0.0762 AUXI |
| benign_P1Q0 | LBL | +0.0059 CLEA → -0.0013 CLEA | +0.0031 CLEA → +0.0042 CLEA | +0.0057 CLEA → +0.0018 CLEA | +0.0110 CLEA → +0.0024 CLEA |
| benign_P1Q0 | SUF | -0.0000 CLEA → +0.0008 CLEA | +0.0003 CLEA → -0.0009 CLEA | +0.0010 CLEA → +0.0031 CLEA | +0.0008 CLEA → +0.0010 CLEA |
| benign_P1Q0 | SUB | +0.0009 CLEA → +0.0057 CLEA | +0.0031 CLEA → +0.0040 CLEA | +0.0015 CLEA → +0.0079 CLEA | +0.0041 CLEA → +0.0028 CLEA |
| benign_P1Q0 | PTH | +0.0202 CLEA → +0.0110 CLEA | +0.0108 CLEA → +0.0115 CLEA | +0.0216 CLEA → +0.0091 CLEA | +0.0204 CLEA → +0.0125 CLEA |
| benign_P1Q0 | FULL | +0.0203 AUXI → +0.0131 AUXI | +0.0145 AUXI → +0.0126 AUXI | +0.0225 AUXI → +0.0158 AUXI | +0.0215 AUXI → +0.0135 AUXI |
| benign_P1Q1 | LBL | +0.0207 CLEA → +0.0253 CLEA | +0.0277 CLEA → +0.0270 CLEA | +0.0225 CLEA → +0.0261 CLEA | +0.0190 CLEA → +0.0257 CLEA |
| benign_P1Q1 | SUF | +0.0013 CLEA → +0.0008 CLEA | +0.0031 CLEA → +0.0026 CLEA | -0.0009 CLEA → -0.0017 CLEA | +0.0009 CLEA → -0.0029 CLEA |
| benign_P1Q1 | SUB | +0.0768 CONV → +0.0669 CONV | +0.0663 CONV → +0.0644 CONV | +0.0766 CONV → +0.0708 CONV | +0.0513 UNDE → +0.0379 UNDE |
| benign_P1Q1 | PTH | +0.0863 CONV → +0.0806 CONV | +0.0849 CONV → +0.0836 CONV | +0.0783 CONV → +0.0773 CONV | +0.0578 CONV → +0.0509 UNDE |
| benign_P1Q1 | QRY | +0.0708 CONV → +0.0572 UNDE | +0.0566 CONV → +0.0584 CONV | +0.0814 CONV → +0.0584 CONV | +0.0745 CONV → +0.0511 UNDE |
| benign_P1Q1 | FULL | +0.0984 AUXI → +0.0786 AUXI | +0.0830 AUXI → +0.0853 AUXI | +0.0968 AUXI → +0.0825 AUXI | +0.0916 AUXI → +0.0688 AUXI |
| phishing_P0Q0 | LBL | +0.0478 UNDE → +0.0335 CLEA | +0.0284 CLEA → +0.0303 CLEA | +0.0393 UNDE → +0.0288 CLEA | +0.0414 CLEA → +0.0238 CLEA |
| phishing_P0Q0 | SUF | -0.0071 CLEA → -0.0019 CLEA | -0.0022 CLEA → -0.0025 CLEA | -0.0009 CLEA → -0.0034 CLEA | -0.0024 CLEA → -0.0028 CLEA |
| phishing_P0Q0 | SUB | +0.0439 CLEA → +0.0326 CLEA | +0.0334 CLEA → +0.0339 UNDE | +0.0458 UNDE → +0.0349 CLEA | +0.0371 CLEA → +0.0326 CLEA |
| phishing_P0Q0 | FULL | +0.0495 AUXI → +0.0439 AUXI | +0.0371 AUXI → +0.0363 AUXI | +0.0491 AUXI → +0.0322 AUXI | +0.0496 AUXI → +0.0322 AUXI |
| phishing_P1Q0 | LBL | +0.0419 CLEA → +0.0346 CLEA | +0.0342 CLEA → +0.0337 CLEA | +0.0420 UNDE → +0.0386 CLEA | +0.0335 CLEA → +0.0173 CLEA |
| phishing_P1Q0 | SUF | -0.0006 CLEA → +0.0031 CLEA | -0.0010 CLEA → -0.0005 CLEA | -0.0006 CLEA → -0.0008 CLEA | -0.0005 CLEA → +0.0010 CLEA |
| phishing_P1Q0 | SUB | +0.0626 CONV → +0.0454 UNDE | +0.0390 CLEA → +0.0459 UNDE | +0.0600 UNDE → +0.0428 UNDE | +0.0389 CLEA → +0.0286 CLEA |
| phishing_P1Q0 | PTH | +0.0533 UNDE → +0.0362 CLEA | +0.0433 UNDE → +0.0377 CLEA | +0.0605 CONV → +0.0400 CLEA | +0.0592 UNDE → +0.0403 CLEA |
| phishing_P1Q0 | FULL | +0.0733 AUXI → +0.0460 AUXI | +0.0508 AUXI → +0.0521 AUXI | +0.0714 AUXI → +0.0500 AUXI | +0.0638 AUXI → +0.0387 AUXI |

### Honouring (control, native), best.pt → final.pt

| state | e11_rs | e11_ref | e11_joint | e11_plan |
|---|---|---|---|---|
| benign P1Q1 | 92.50% → 93.71% | 94.84% → 94.67% | 94.23% → 93.55% | 98.25% → 98.34% |
| benign P1Q0 | 98.27% → 98.50% | 98.44% → 98.37% | 98.30% → 98.63% | 99.74% → 99.71% |
| benign P0Q1 | 98.10% → 98.16% | 97.91% → 97.81% | 97.97% → 97.79% | 98.76% → 98.59% |
| benign P0Q0 | 99.20% → 99.15% | 99.06% → 99.11% | 98.83% → 99.11% | 99.44% → 99.56% |
| phishing P1Q1 | 82.36% → 82.00% | 83.96% → 84.04% | 82.24% → 81.98% | 96.40% → 96.82% |
| phishing P1Q0 | 97.43% → 97.82% | 98.04% → 97.85% | 97.68% → 98.05% | 99.45% → 99.38% |
| phishing P0Q1 | 97.28% → 97.72% | 97.54% → 97.70% | 96.16% → 97.10% | 98.80% → 98.68% |
| phishing P0Q0 | 98.71% → 98.93% | 98.90% → 98.90% | 98.79% → 98.81% | 99.59% → 99.55% |

### Copying and novelty, best.pt → final.pt

| arm | mode | novel | copied queries |
|---|---|---|---|
| e11_rs | free | 99.62% → 99.55% | 19.76% → 21.28% |
| e11_rs | control | 99.72% → 99.64% | 22.49% → 21.45% |
| e11_ref | free | 99.51% → 99.52% | 19.03% → 20.43% |
| e11_ref | control | 99.66% → 99.69% | 21.72% → 22.02% |
| e11_joint | free | 99.60% → 99.53% | 21.59% → 21.35% |
| e11_joint | control | 99.73% → 99.66% | 23.04% → 22.28% |
| e11_plan | free | 99.65% → 99.55% | 17.69% → 19.47% |
| e11_plan | control | 99.78% → 99.64% | 22.51% → 22.98% |

K11d readings against each arm's reference (points, 95% CI), best.pt → final.pt:

- e11_rs against e10_base: free_novelty -0.06 → +0.01 [-0.06, +0.07]; free_copying -1.29 → +0.24 [-1.71, +2.19]; control_novelty -0.03 → +0.05 [-0.00, +0.10]; control_copying +1.97 → +0.92 [+0.35, +1.50]
- e11_ref against e11_rs: free_novelty +0.11 → +0.03 [-0.04, +0.10]; free_copying -0.72 → -0.85 [-2.75, +1.05]; control_novelty +0.06 → -0.05 [-0.10, +0.01]; control_copying -0.77 → +0.57 [-0.01, +1.16]
- e11_joint against e11_ref: free_novelty -0.09 → -0.01 [-0.08, +0.06]; free_copying +2.56 → +0.92 [-1.03, +2.87]; control_novelty -0.07 → +0.03 [-0.02, +0.08]; control_copying +1.32 → +0.26 [-0.33, +0.85]
- e11_plan against e11_ref: free_novelty -0.14 → -0.03 [-0.10, +0.04]; free_copying -1.35 → -0.96 [-2.74, +0.81]; control_novelty -0.12 → +0.05 [-0.00, +0.10]; control_copying +0.79 → +0.96 [+0.38, +1.55]

### e11_plan's own plan (checkpoint, shipped-weighted), best.pt → final.pt

| class | plan P | plan Q |
|---|---|---|
| benign | 74.43% → 74.41% | 6.379% → 6.009% |
| phishing | 39.15% → 39.02% | 4.252% → 3.890% |

### Each change at the saved checkpoints and at step 20,000

| change | arm (step) against reference (step) | benign PTH | benign QRY | phishing PTH | phishing QRY | benign free QRY x | benign P0Q1 QRY x | benign P1Q1 QRY x |
|---|---|---|---|---|---|---|---|---|
| the slash | e11_rs (14,000) against e10_base (18,000) | +1.21 | +0.49 | -1.83 | +0.39 | +0.0203 | +0.0110 | +0.0059 |
| the slash | e11_rs (20,000) against e10_base (18,000) | +0.40 | +0.34 | +0.55 | +0.71 | -0.0020 | -0.0113 | -0.0077 |
| the repair | e11_ref (19,000) against e11_rs (14,000) | -2.72 | -0.56 | +2.90 | +0.05 | -0.0190 | -0.0268 | -0.0142 |
| the repair | e11_ref (20,000) against e11_rs (20,000) | -1.40 | -0.18 | +0.70 | -0.23 | +0.0002 | -0.0041 | +0.0012 |
| joint dropout | e11_joint (14,000) against e11_ref (19,000) | +2.92 | +0.44 | -0.17 | -0.25 | +0.0213 | +0.0202 | +0.0248 |
| joint dropout | e11_joint (20,000) against e11_ref (20,000) | +0.29 | -0.19 | +0.29 | -0.26 | +0.0123 | +0.0017 | +0.0000 |
| the plan | e11_plan (14,000) against e11_ref (19,000) | -0.99 | +1.58 | +0.96 | +1.38 | +0.0088 | +0.0232 | +0.0179 |
| the plan | e11_plan (20,000) against e11_ref (20,000) | -1.43 | +1.34 | +0.86 | +1.30 | +0.0082 | -0.0038 | -0.0073 |

Emission in points; cells as the change in x. `e10_base` is E10's registered reading (best.pt, step 18,000), the one comparison that still crosses steps.

### Amendment 3's predictions, scored

| prediction (`PREREG_E11` §3.3) | result | scored |
|---|---|---|
| `e11_plan` passes K11a again | passes | right |
| FULL moves by at most 0.02 in every arm | largest move -0.0274 (e11_rs free phishing); over 0.02: e11_rs free phishing -0.0274, e11_rs phishing_P1Q0 -0.0273, e11_joint phishing_P1Q0 -0.0214, e11_plan benign_P1Q1 -0.0228, e11_plan benign_P0Q1 -0.0251, e11_plan phishing_P1Q0 -0.0251 | **wrong** |
| the slash rows move by at most 0.3 points | largest move -1.40 (e11_joint phishing free_host_query); over 0.3: e11_rs benign free_host_query +0.65, e11_rs phishing free_host_query +0.45, e11_ref benign free_bare +0.33, e11_ref phishing free_host_query +0.35, e11_joint benign free_host_query -0.71, e11_joint phishing free_host_query -1.40, e11_plan benign free_bare -0.32, e11_plan benign control P0Q0 -0.36, e11_plan benign free_host_query -1.09, e11_plan phishing free_host_query +0.89 | **wrong** |
| K11b still fails on the phishing bare-host rows | e11_rs: 2 of 2 fail; e11_ref: 2 of 2 fail; e11_joint: 2 of 2 fail; e11_plan: 2 of 2 fail | right |
| benign free PTH about 76% (`e11_rs`), 74.5% (`e11_ref`), 74.5–75% (`e11_joint`) | e11_rs 76.28%, e11_ref 74.89%, e11_joint 75.17% | reported (see the difference from each stated value) |
| joint dropout moves benign free PTH by under 1 point | +0.29 points | right |
| the three 14,000-step arms improve on benign free QRY, to within 0.01 of `e11_ref`'s (`e11_ref` at step 20,000: +0.0268) | e11_rs +0.0489 → +0.0266 (improves, within 0.01); e11_joint +0.0512 → +0.0391 (improves, not within 0.01); e11_plan +0.0387 → +0.0350 (improves, within 0.01) | partly |
| `e11_plan`'s control query cells improve (whether they reach UNDECIDED: no prediction) | P1Q1 QRY +0.0745 CONV → +0.0511 UNDE; P0Q1 QRY +0.0847 CONV → +0.0581 UNDE | right |
| K11c: the regressions found at step 14,000 shrink | e11_rs phishing PTH +0.0181 → -0.0019; e11_rs phishing P1Q0 PTH +0.0151 → -0.0020; e11_rs phishing P1Q0 SUB +0.0185 → +0.0013; e11_joint phishing P1Q0 PTH +0.0172 → +0.0023; e11_joint phishing P1Q0 SUB +0.0210 → -0.0031 | right |
| K11d: the copying rises found at step 14,000 shrink | e11_rs control_copying +1.97 → +0.92; e11_joint free_copying +2.56 → +0.92; e11_joint control_copying +1.32 → +0.26; e11_plan control_copying +0.79 → +0.96 | partly |

Checkpoint files: best.pt as E11_RESULTS.md; final.pt e11_rs c606d93e…, e11_ref c86e60c5…, e11_joint c71081b4…, e11_plan 1031739d….
