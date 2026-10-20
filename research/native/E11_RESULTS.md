# E11 — results: presence and the root slash

> **Read again at step 20,000** (amendment 3, 2026-09-30): `E11_STEP20K_RESULTS.md`.
> This reading, at `best.pt`, stays as registered. Each change's effect is read
> there, where all four arms share the step.

- **Registration:** `PREREG_E11.md` (9e89dcd), before any code or run.
  - amendment 1 (2c1be2a): predictions from E10, before training;
  - amendment 2 (18bbd33; instrument 4b38f29): post hoc, reported only; the
    checkpoint-step check, registered before it was run.
- **Code:** 6d32e2d (training and generation, run by the owner on Kaggle).
  - laptop tools: f9389e0, 90b2fda, f8acf68 and 69fb4ab.
- **Read:** 2026-09-30, after VERIFY passed.
- **Numbers:** two appendices, pasted by script. Every number in the text appears
  there or has a declared source.
  - Appendix A is `e11_tables.py`'s output;
  - Appendix B is `e11_step_check.py`'s.

## In plain words

- **Presence is solved when the plan is supplied.** In free mode, `e11_plan` writes
  paths and queries at the real rates in both classes. It is the first arm to meet
  the tolerance (K11a), and it lands where E10 predicted (amendment 1).

  | class | field | generated | real | predicted |
  |---|---|---|---|---|
  | benign | path | 73.38% | 73.37% | 73.32% |
  | benign | query | 6.57% | 6.62% | 6.61% |
  | phishing | path | 39.90% | 39.51% | 39.54% |
  | phishing | query | 4.71% | 4.68% | 4.63% |

- **The root slash takes away most of the whole-URL detector's edge.** FULL for
  `e11_plan`:

  | reading | before (`e10_base`) | `e11_plan` |
  |---|---|---|
  | benign, free | 0.146 | 0.024 |
  | phishing, free | 0.385 | 0.039 |
  | benign bare hosts, control | 0.491 | 0.010 |
  | phishing bare hosts, control | 0.499 | 0.050 |

  Benign bare hosts get the slash 98.27% of the time; the real rate is 98.00%.
- **The strict slash criterion still fails in every arm** (K11b), by fractions of a
  percent:
  - phishing bare hosts miss the slash 0.3–0.7% of the time, where real ones never
    do;
  - benign host-route queries in control mode get it slightly too often: 99.95%
    against 99.55%.
- **The clipping repair works** (K11e). No batch was clipped after step 1,000. The
  plan's own benign path choice moved from 77.10% to 74.43%.
- **The plan honours requests** 96.40–99.74% of the time, and in 7 of 8 states at
  or above 0.98.
- **Content is still the blocker.** With presence right, phishing free-mode queries
  are fully measurable, and they read CONVICTED (+0.138). The benign query cells
  in control mode are CONVICTED in every arm.
- **Copying of training queries rises in three arms** (K11d).
- **No arm is a candidate.** `e11_plan` fails only K11b and K11d.
- **A method problem was found and measured** (amendment 2).
  - `best.pt` is picked on a validation curve that is flat from step 14,000, so the
    step it lands on is effectively random: 14,000 for three arms, 19,000 for
    `e11_ref`.
  - That step moves the benign path decision by up to 2.4 points and the query
    cells by about 0.02: as much as the changes being compared.
  - So "each change's effect" is **confounded** here.
  - Nor can this run tell whether E10's content gain for the plan survives.
    `e11_plan` was read at its 14,000-step checkpoint, and E10's plan arm at
    19,000.

## 1. Validity: PASS

| check | result |
|---|---|
| V1 | the 200 registered files present with their n, seed and hash; checkpoints and the presence table as registered; one code commit (6d32e2d) |
| V2 | byte identity on the laptop: 1,036 of 1,036 free and 269 of 269 control URLs per hidden arm. The supplied-plan free mode: 712 of 712, located through both shuffles |
| V3 | `uie/` and the common modules unchanged since 6d32e2d |
| V4 | the scorer reproduces E7's published cells exactly: 0.00e+00 in both classes |
| V-id | one shared start (`5aa980f7…`), its part without `<RS>` equal to E10's (`26926388…`); settings as registered |

## 2. Training and K11e (the repair): PASS

| arm | final validation mean | best step | rows dropped |
|---|---|---|---|
| `e11_rs` | 1.5866 | 14,000 | 10,316 |
| `e11_ref` | 1.5858 | 19,000 | 10,316 |
| `e11_joint` | 1.5846 | 14,000 | 10,316 |
| `e11_plan` | 1.5186 | 14,000 | 10,316 |

- E10's run dropped 10,313 rows. The three extra rows gained `<RS>` and now exceed
  256 tokens.
- **K11e passes for every repair arm.** After step 1,000, no step was clipped in any
  length band. Under the old cap of 1.0, the shortest band would have been clipped:

  | arm | share of the shortest band over 1.0 |
  |---|---|
  | `e11_ref` | 18.1% |
  | `e11_joint` | 18.5% |
  | `e11_plan` | 27.5% |

  `e11_rs`, which keeps the old cap, clipped 20.4%.
- **The plan's own plan** (reported; checkpoint, shipped-weighted):
  - benign P 74.43%, Q 6.379%;
  - phishing P 39.15%, Q 4.252%;
  - E10's plan arm was 77.10% / 6.05% and 39.83% / 4.03%.

## 3. K11a — presence against the tolerance

| arm | benign PTH | benign QRY | phishing PTH | phishing QRY | K11a |
|---|---|---|---|---|---|
| `e11_rs` | 77.09% (+3.72) | 5.55% (−1.06) | 36.04% (−3.47) | 3.28% (−1.40) | FAIL |
| `e11_ref` | 74.36% (+0.99, just over ±0.99) | 4.99% (−1.63) | 38.94% (−0.57) | 3.33% (−1.35) | FAIL |
| `e11_joint` | 77.28% (+3.91) | 5.43% (−1.18) | 38.78% (−0.73) | 3.08% (−1.60) | FAIL |
| `e11_plan` | **73.38% (+0.01)** | **6.57% (−0.05)** | **39.90% (+0.39)** | **4.71% (+0.03)** | **PASS** |

- The real rates are 73.369%, 6.617%, 39.511% and 4.680%.
- The tolerances are ±0.99, ±0.56, ±1.09 and ±0.47.

## 4. K11b — the root slash: FAIL in every arm

`e11_plan`'s rows, with the other arms in Appendix A:

| row | generated | real | T | within |
|---|---|---|---|---|
| benign bare hosts, free | 98.27% | 98.00% | ±0.63 | yes |
| benign bare hosts, control | 98.57% | 97.94% | ±0.52 | **no** (+0.63) |
| benign host-route queries, free | 100.00% (647 of 647) | 99.58% | ±1.50 | yes |
| benign host-route queries, control | 99.95% | 99.55% | ±0.24 | **no** (+0.40) |
| phishing bare hosts, free | 99.30% (164 of 23,378 miss it) | 100% | ±0.04 | **no** |
| phishing bare hosts, control | 99.65% | 100% | ±0.07 | **no** |
| phishing host-route queries, free | 98.04% | 100% | ±1.54 | **no** |
| phishing host-route queries, control | 99.98% | 100% | ±0.20 | yes |

- **Every arm misses the phishing bare-host rows.** It writes the slash on 99.3–99.7%
  of phishing bare hosts, against 100% in 114,639 real ones.
- **Every arm overshoots the benign control host-route row:** 99.86–99.95% against
  99.55%.
- **What that means.** The model generalises slightly across classes and states:
  phishing inherits a little of benign's 2% "no slash", and benign control mode is
  too regular.
- **What it does not mean.** It is not a failure to write the slash. That was 0%
  before E11.

## 5. K11c — the gate, each arm against its reference

| arm | reference | regressions (verdict worse **and** x up by more than 0.0141) |
|---|---|---|
| `e11_rs` | `e10_base` | phishing free PTH: 0.0375 CLEARED → 0.0556 UNDECIDED; phishing P1Q0 PTH: 0.0382 CLEARED → 0.0533 UNDECIDED; phishing P1Q0 SUB: 0.0441 UNDECIDED → 0.0626 CONVICTED. **FAIL** |
| `e11_ref` | `e11_rs` | none. **PASS** |
| `e11_joint` | `e11_ref` | phishing P1Q0 PTH: 0.0433 UNDECIDED → 0.0605 CONVICTED; phishing P1Q0 SUB: 0.0390 CLEARED → 0.0600 UNDECIDED. **FAIL** |
| `e11_plan` | `e11_ref` | none. **PASS** |

All five regressions are in arms read at step 14,000, against references read
later (see §7).

## 6. K11d — guardrails

| arm | reference | control-mode copying (points) | reading |
|---|---|---|---|
| `e11_rs` | `e10_base` | +1.97 [+1.39, +2.55] | **FAIL** |
| `e11_ref` | `e11_rs` | −0.77 [−1.36, −0.19] | PASS |
| `e11_joint` | `e11_ref` | +1.32 [+0.73, +1.91], and free mode +2.56 [+0.64, +4.48] | **FAIL** |
| `e11_plan` | `e11_ref` | +0.79 [+0.21, +1.37] | **FAIL** |

- Novelty moves by at most 0.14 points in any arm.
- `e11_plan`'s free-mode copying is lower than its reference's: 17.69% against
  19.03%.

## 7. The checkpoint step confounds the per-change readings (amendment 2)

**The check** (Appendix B). The model's share of "a path follows" at the end of
3,000 real hosts per class, the same rows for every checkpoint. The real share on
these rows is 72.07% (benign) and 38.37% (phishing).

| arm | best step | benign share at `best.pt` | at `final.pt` (step 20,000) | change |
|---|---|---|---|---|
| `e11_rs` | 14,000 | 76.53% | 75.75% | −0.78 |
| `e11_ref` | 19,000 | 73.63% | 74.24% | +0.61 |
| `e11_joint` | 14,000 | 77.05% | 74.63% | **−2.42** |

- **The registered rule triggers.** A 14,000-step arm moved by more than 1 point
  while `e11_ref` did not. The per-change effects in Appendix A's attribution table
  are reported as **confounded by the checkpoint step**.
- **At the same step** (both at step 20,000), the changes look different:

  | change | benign path share, at saved checkpoints | at step 20,000 |
  |---|---|---|
  | the repair (`e11_rs` → `e11_ref`) | −2.9 points | **−1.5 points** |
  | joint dropout (`e11_ref` → `e11_joint`) | +3.4 points | **+0.4 points** |

- **The plan's own plan between its two checkpoints:**
  - path almost unmoved: 74.43% → 74.41% (benign), 39.15% → 39.02% (phishing);
  - query down: 6.379% → 6.009% and 4.252% → 3.890%.
- **The same pattern in the gate cells,** not measured step by step: every arm read
  at step 14,000 has worse query cells than `e11_ref` at 19,000. For example, benign
  free QRY is 0.0489 (`e11_rs`) and 0.0512 (`e11_joint`), against 0.0299
  (`e11_ref`).
- **Why this happens** [HYPOTHESIS]. At step 14,000 the learning rate is still about
  a fifth of its peak, and the rare decisions and query values are still settling.
  The validation loss averages all tokens and hides that.

## 8. Reported: FULL and honouring

**FULL (auxiliary)** falls sharply wherever the slash matters. `e10_base`'s column is
from `E10_RESULTS.md`, Appendix A. It rises a little in
path-only control states, where the 14,000-step arms are weaker.

| reading | `e10_base` | `e11_rs` | `e11_ref` | `e11_joint` | `e11_plan` |
|---|---|---|---|---|---|
| free, benign | 0.1464 | 0.0295 | 0.0212 | 0.0299 | 0.0243 |
| free, phishing | 0.3845 | 0.0497 | 0.0206 | 0.0394 | 0.0393 |
| control P0Q0, benign | 0.4910 | 0.0124 | 0.0107 | 0.0137 | 0.0104 |
| control P0Q0, phishing | 0.4990 | 0.0495 | 0.0371 | 0.0491 | 0.0496 |

**Honouring:**
- **The slash fixed the hardest state.** A query right after the host rose from
  94.47% to 98.10% (benign) and from 88.70% to 97.28% (phishing), `e10_base` to
  `e11_rs`. "`host/?q`" now has its own structure.
- **The plan arm honours 96.40–99.74%.** Phishing path-and-query is its only state
  under 0.98.

## 9. The registered predictions, scored

| prediction (`PREREG_E11` §3 and amendment 1) | result | scored |
|---|---|---|
| `e11_rs`: benign slash rows pass | 3 of 4 (control host-route overshoots) | partly |
| `e11_rs`: phishing slash rows uncertain | the bare-host rows fail | — |
| `e11_rs`: FULL falls under +0.08 benign and +0.10 phishing (free) | 0.0295 / 0.0497 | right |
| `e11_rs`: paths within ±1 point of `e10_base` | +1.21 / −1.83 | wrong |
| `e11_ref`: K11e passes | passes | right |
| `e11_ref`: benign free PTH falls 1.5–3 points, into the tolerance | falls 2.72, lands 0.005 over | partly |
| `e11_ref`: phishing PTH falls | rises 2.90 | **wrong** (confounded, §7) |
| `e11_joint`: free QRY rises; honouring rises | benign up, phishing down; honouring not up | wrong |
| `e11_plan`: K11a passes at 73.32 / 6.61 / 39.54 / 4.63 | 73.38 / 6.57 / 39.90 / 4.71: passes | **right** |
| `e11_plan`: honouring stays high | 96.40–99.74% | right |
| `e11_plan`: may fail K11d on control copying | fails, +0.79 | right |
| `e11_plan`'s own benign P inside the tolerance (about 73.98%) | 74.43%, 0.07 over | nearly |

## 10. Candidates, and the E3 finish line

- **No arm passes all its criteria.**
  - `e11_plan` fails K11b (the slash's strict rows) and K11d (control copying).
  - The others also fail K11a.
- **The E3 finish line** (`DECISION_2026-09-29.md` §3) is not reached either:
  - query cells are CONVICTED (benign P1Q1 and P0Q1 in control; phishing free);
  - the slash misses its tolerance in 5 of 8 rows.

## 11. What I propose (the owner decides)

1. **Read E11 again at step 20,000.**
   - Generate the same 200 files, seeds and calls from each arm's `final.pt`, which
     training already wrote, and score them with the same criteria.
   - This removes the confound in §7 for every comparison inside E11.
   - It also answers the most important open question: whether the plan's content
     gain from E10 holds with the slash and the repair.
   - Kaggle about 6.5 h; laptop about 3 h. It would be registered as an amendment
     before it runs. This reading stays as registered.
2. **Use `final.pt` for generation in every later registration.** The learning rate
   ends at zero, and the validation curve is too flat to choose a step. This is a
   change of procedure, so it is the owner's decision.
3. **The slash's residual errors** (phishing bare hosts without the slash) and
   **copying** are small next to content. Revisit them after the step-20,000
   reading, not before.

## 12. What is and is not claimed

**Claimed** [CONFIRMED by the registered readout]:
- the supplied plan puts free-mode presence at the real rates, within the
  tolerance;
- the slash token cuts FULL by 79–98% across free mode and the no-path control
  states (`e11_plan` against `e10_base`);
- the repair stops clipping from binding after step 1,000;
- the slash raises control honouring of host-route queries;
- no arm is a candidate.

**Not claimed:**
- the size of any single change's effect on the gate or on emission: confounded by
  the checkpoint step (§7);
- why the checkpoint step matters: a hypothesis;
- anything for another training: one training per arm.

## Appendix A — the rendered tables

`python research/native/e11_tables.py`, verbatim.

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
| e11_rs | benign PTH | 77.09% against 73.369% (+3.72, ±0.99) **no** |
| e11_rs | benign QRY | 5.55% against 6.617% (-1.06, ±0.56) **no** |
| e11_rs | phishing PTH | 36.04% against 39.511% (-3.47, ±1.09) **no** |
| e11_rs | phishing QRY | 3.28% against 4.680% (-1.40, ±0.47) **no** |
| e11_ref | benign PTH | 74.36% against 73.369% (+0.99, ±0.99) **no** |
| e11_ref | benign QRY | 4.99% against 6.617% (-1.63, ±0.56) **no** |
| e11_ref | phishing PTH | 38.94% against 39.511% (-0.57, ±1.09) yes |
| e11_ref | phishing QRY | 3.33% against 4.680% (-1.35, ±0.47) **no** |
| e11_joint | benign PTH | 77.28% against 73.369% (+3.91, ±0.99) **no** |
| e11_joint | benign QRY | 5.43% against 6.617% (-1.18, ±0.56) **no** |
| e11_joint | phishing PTH | 38.78% against 39.511% (-0.73, ±1.09) yes |
| e11_joint | phishing QRY | 3.08% against 4.680% (-1.60, ±0.47) **no** |
| e11_plan | benign PTH | 73.38% against 73.369% (+0.01, ±0.99) yes |
| e11_plan | benign QRY | 6.57% against 6.617% (-0.05, ±0.56) yes |
| e11_plan | phishing PTH | 39.90% against 39.511% (+0.39, ±1.09) yes |
| e11_plan | phishing QRY | 4.71% against 4.680% (+0.03, ±0.47) yes |

### K11b — the root slash against the tolerance

| arm | row | generated against real (difference, T) within |
|---|---|---|
| e11_rs | benign free_bare | 98.30% against 98.004% (+0.29, ±0.63) yes |
| e11_rs | benign control P0Q0 | 98.32% against 97.944% (+0.38, ±0.52) yes |
| e11_rs | benign free_host_query | 98.31% against 99.576% (-1.27, ±1.50) yes |
| e11_rs | benign control P0Q1 | 99.91% against 99.551% (+0.36, ±0.24) **no** |
| e11_rs | phishing free_bare | 99.48% against 100.000% (-0.52, ±0.04) **no** |
| e11_rs | phishing control P0Q0 | 99.59% against 100.000% (-0.41, ±0.07) **no** |
| e11_rs | phishing free_host_query | 98.95% against 100.000% (-1.05, ±1.54) yes |
| e11_rs | phishing control P0Q1 | 99.98% against 100.000% (-0.02, ±0.20) yes |
| e11_ref | benign free_bare | 98.04% against 98.004% (+0.03, ±0.63) yes |
| e11_ref | benign control P0Q0 | 98.38% against 97.944% (+0.43, ±0.52) yes |
| e11_ref | benign free_host_query | 98.38% against 99.576% (-1.20, ±1.50) yes |
| e11_ref | benign control P0Q1 | 99.91% against 99.551% (+0.36, ±0.24) **no** |
| e11_ref | phishing free_bare | 99.59% against 100.000% (-0.41, ±0.04) **no** |
| e11_ref | phishing control P0Q0 | 99.73% against 100.000% (-0.27, ±0.07) **no** |
| e11_ref | phishing free_host_query | 98.41% against 100.000% (-1.59, ±1.54) **no** |
| e11_ref | phishing control P0Q1 | 99.96% against 100.000% (-0.04, ±0.20) yes |
| e11_joint | benign free_bare | 98.29% against 98.004% (+0.29, ±0.63) yes |
| e11_joint | benign control P0Q0 | 98.45% against 97.944% (+0.50, ±0.52) yes |
| e11_joint | benign free_host_query | 97.07% against 99.576% (-2.51, ±1.50) **no** |
| e11_joint | benign control P0Q1 | 99.86% against 99.551% (+0.31, ±0.24) **no** |
| e11_joint | phishing free_bare | 99.46% against 100.000% (-0.54, ±0.04) **no** |
| e11_joint | phishing control P0Q0 | 99.66% against 100.000% (-0.34, ±0.07) **no** |
| e11_joint | phishing free_host_query | 98.10% against 100.000% (-1.90, ±1.54) **no** |
| e11_joint | phishing control P0Q1 | 99.98% against 100.000% (-0.02, ±0.20) yes |
| e11_plan | benign free_bare | 98.27% against 98.004% (+0.27, ±0.63) yes |
| e11_plan | benign control P0Q0 | 98.57% against 97.944% (+0.63, ±0.52) **no** |
| e11_plan | benign free_host_query | 100.00% against 99.576% (+0.42, ±1.50) yes |
| e11_plan | benign control P0Q1 | 99.95% against 99.551% (+0.40, ±0.24) **no** |
| e11_plan | phishing free_bare | 99.30% against 100.000% (-0.70, ±0.04) **no** |
| e11_plan | phishing control P0Q0 | 99.65% against 100.000% (-0.35, ±0.07) **no** |
| e11_plan | phishing free_host_query | 98.04% against 100.000% (-1.96, ±1.54) **no** |
| e11_plan | phishing control P0Q1 | 99.98% against 100.000% (-0.02, ±0.20) yes |

### Gate v2, FULL FREE (x, verdict)

| class | field | e11_rs | e11_ref | e11_joint | e11_plan |
|---|---|---|---|---|---|
| benign | LBL | +0.0080 CLEA | +0.0061 CLEA | +0.0110 CLEA | +0.0086 CLEA |
| benign | SUF | +0.0171 CLEA | +0.0148 CLEA | +0.0124 CLEA | +0.0184 CLEA |
| benign | SUB | +0.0076 CLEA | +0.0063 CLEA | +0.0035 CLEA | +0.0030 CLEA |
| benign | PTH | +0.0163 CLEA | +0.0046 CLEA | +0.0142 CLEA | +0.0129 CLEA |
| benign | QRY | +0.0489 UNDE | +0.0299 UNDE | +0.0512 UNDE | +0.0387 UNDE |
| benign | FULL | +0.0295 AUXI | +0.0212 AUXI | +0.0299 AUXI | +0.0243 AUXI |
| phishing | LBL | +0.0333 CLEA | +0.0101 CLEA | +0.0212 CLEA | +0.0215 CLEA |
| phishing | SUF | +0.0095 CLEA | +0.0068 CLEA | +0.0082 CLEA | +0.0079 CLEA |
| phishing | SUB | +0.0378 CLEA | +0.0225 CLEA | +0.0365 CLEA | +0.0306 CLEA |
| phishing | PTH | +0.0556 UNDE | +0.0354 UNDE | +0.0582 UNDE | +0.0533 UNDE |
| phishing | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | +0.1382 CONV |
| phishing | FULL | +0.0497 AUXI | +0.0206 AUXI | +0.0394 AUXI | +0.0393 AUXI |

### FULL CONTROL, measurable states (x, verdict)

| state | field | e11_rs | e11_ref | e11_joint | e11_plan |
|---|---|---|---|---|---|
| benign_P0Q0 | LBL | +0.0124 CLEA | +0.0081 CLEA | +0.0123 CLEA | +0.0095 CLEA |
| benign_P0Q0 | SUF | +0.0010 CLEA | -0.0023 CLEA | -0.0017 CLEA | -0.0005 CLEA |
| benign_P0Q0 | SUB | +0.0119 CLEA | +0.0139 CLEA | +0.0126 CLEA | +0.0168 CLEA |
| benign_P0Q0 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q0 | FULL | +0.0124 AUXI | +0.0107 AUXI | +0.0137 AUXI | +0.0104 AUXI |
| benign_P0Q1 | LBL | +0.0223 CLEA | +0.0196 CLEA | +0.0221 CLEA | +0.0179 CLEA |
| benign_P0Q1 | SUF | +0.0085 CLEA | +0.0095 CLEA | +0.0138 CLEA | +0.0111 CLEA |
| benign_P0Q1 | SUB | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q1 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q1 | QRY | +0.0883 CONV | +0.0615 CONV | +0.0817 CONV | +0.0847 CONV |
| benign_P0Q1 | FULL | +0.0979 AUXI | +0.0791 AUXI | +0.0875 AUXI | +0.1013 AUXI |
| benign_P1Q0 | LBL | +0.0059 CLEA | +0.0031 CLEA | +0.0057 CLEA | +0.0110 CLEA |
| benign_P1Q0 | SUF | -0.0000 CLEA | +0.0003 CLEA | +0.0010 CLEA | +0.0008 CLEA |
| benign_P1Q0 | SUB | +0.0009 CLEA | +0.0031 CLEA | +0.0015 CLEA | +0.0041 CLEA |
| benign_P1Q0 | PTH | +0.0202 CLEA | +0.0108 CLEA | +0.0216 CLEA | +0.0204 CLEA |
| benign_P1Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P1Q0 | FULL | +0.0203 AUXI | +0.0145 AUXI | +0.0225 AUXI | +0.0215 AUXI |
| benign_P1Q1 | LBL | +0.0207 CLEA | +0.0277 CLEA | +0.0225 CLEA | +0.0190 CLEA |
| benign_P1Q1 | SUF | +0.0013 CLEA | +0.0031 CLEA | -0.0009 CLEA | +0.0009 CLEA |
| benign_P1Q1 | SUB | +0.0768 CONV | +0.0663 CONV | +0.0766 CONV | +0.0513 UNDE |
| benign_P1Q1 | PTH | +0.0863 CONV | +0.0849 CONV | +0.0783 CONV | +0.0578 CONV |
| benign_P1Q1 | QRY | +0.0708 CONV | +0.0566 CONV | +0.0814 CONV | +0.0745 CONV |
| benign_P1Q1 | FULL | +0.0984 AUXI | +0.0830 AUXI | +0.0968 AUXI | +0.0916 AUXI |
| phishing_P0Q0 | LBL | +0.0478 UNDE | +0.0284 CLEA | +0.0393 UNDE | +0.0414 CLEA |
| phishing_P0Q0 | SUF | -0.0071 CLEA | -0.0022 CLEA | -0.0009 CLEA | -0.0024 CLEA |
| phishing_P0Q0 | SUB | +0.0439 CLEA | +0.0334 CLEA | +0.0458 UNDE | +0.0371 CLEA |
| phishing_P0Q0 | PTH | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P0Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P0Q0 | FULL | +0.0495 AUXI | +0.0371 AUXI | +0.0491 AUXI | +0.0496 AUXI |
| phishing_P1Q0 | LBL | +0.0419 CLEA | +0.0342 CLEA | +0.0420 UNDE | +0.0335 CLEA |
| phishing_P1Q0 | SUF | -0.0006 CLEA | -0.0010 CLEA | -0.0006 CLEA | -0.0005 CLEA |
| phishing_P1Q0 | SUB | +0.0626 CONV | +0.0390 CLEA | +0.0600 UNDE | +0.0389 CLEA |
| phishing_P1Q0 | PTH | +0.0533 UNDE | +0.0433 UNDE | +0.0605 CONV | +0.0592 UNDE |
| phishing_P1Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P1Q0 | FULL | +0.0733 AUXI | +0.0508 AUXI | +0.0714 AUXI | +0.0638 AUXI |

### K11c — no regression against each arm's reference

| arm | reference | regressions |
|---|---|---|
| e11_rs | e10_base | [{'cell': ['phishing', 'PTH'], 'base': [0.0375, 'CLEARED'], 'arm': [0.0556, 'UNDECIDED']}, {'cell': ['phishing', 'P1Q0', 'PTH'], 'base': [0.0382, 'CLEARED'], 'arm': [0.0533, 'UNDECIDED']}, {'cell': ['phishing', 'P1Q0', 'SUB'], 'base': [0.0441, 'UNDECIDED'], 'arm': [0.0626, 'CONVICTED']}] |
| e11_ref | e11_rs | none |
| e11_joint | e11_ref | [{'cell': ['phishing', 'P1Q0', 'PTH'], 'base': [0.0433, 'UNDECIDED'], 'arm': [0.0605, 'CONVICTED']}, {'cell': ['phishing', 'P1Q0', 'SUB'], 'base': [0.039, 'CLEARED'], 'arm': [0.06, 'UNDECIDED']}] |
| e11_plan | e11_ref | none |

### Honouring (control, native)

| state | e11_rs | e11_ref | e11_joint | e11_plan |
|---|---|---|---|---|
| benign P1Q1 | 92.50% | 94.84% | 94.23% | 98.25% |
| benign P1Q0 | 98.27% | 98.44% | 98.30% | 99.74% |
| benign P0Q1 | 98.10% | 97.91% | 97.97% | 98.76% |
| benign P0Q0 | 99.20% | 99.06% | 98.83% | 99.44% |
| phishing P1Q1 | 82.36% | 83.96% | 82.24% | 96.40% |
| phishing P1Q0 | 97.43% | 98.04% | 97.68% | 99.45% |
| phishing P0Q1 | 97.28% | 97.54% | 96.16% | 98.80% |
| phishing P0Q0 | 98.71% | 98.90% | 98.79% | 99.59% |

### K11d — guardrails against each arm's reference

| arm | mode | novel | copied queries |
|---|---|---|---|
| e11_rs | free | 99.62% | 19.76% of 3,533 |
| e11_rs | control | 99.72% | 22.49% of 38,360 |
| e11_ref | free | 99.51% | 19.03% of 3,326 |
| e11_ref | control | 99.66% | 21.72% of 38,642 |
| e11_joint | free | 99.60% | 21.59% of 3,404 |
| e11_joint | control | 99.73% | 23.04% of 38,496 |
| e11_plan | free | 99.65% | 17.69% of 4,512 |
| e11_plan | control | 99.78% | 22.51% of 39,491 |
- e11_rs against e10_base: **FAIL** free_novelty -0.06 pts [-0.12, +0.00]; free_copying -1.29 pts [-3.22, +0.64]; control_novelty -0.03 pts [-0.08, +0.02]; control_copying +1.97 pts [+1.39, +2.55]
- e11_ref against e11_rs: **PASS** free_novelty +0.11 pts [+0.04, +0.17]; free_copying -0.72 pts [-2.60, +1.15]; control_novelty +0.06 pts [+0.01, +0.11]; control_copying -0.77 pts [-1.36, -0.19]
- e11_joint against e11_ref: **FAIL** free_novelty -0.09 pts [-0.16, -0.03]; free_copying +2.56 pts [+0.64, +4.48]; control_novelty -0.07 pts [-0.11, -0.02]; control_copying +1.32 pts [+0.73, +1.91]
- e11_plan against e11_ref: **FAIL** free_novelty -0.14 pts [-0.20, -0.07]; free_copying -1.35 pts [-3.08, +0.39]; control_novelty -0.12 pts [-0.16, -0.07]; control_copying +0.79 pts [+0.21, +1.37]

### Reported: e11_plan's own plan (checkpoint, shipped-weighted)

| class | plan P | plan Q |
|---|---|---|
| benign | 74.43% | 6.379% |
| phishing | 39.15% | 4.252% |

### Attribution — each arm against its reference (reported)

| arm | reference | emission change (points) |
|---|---|---|
| e11_rs | e10_base | benign PTH +1.21, benign QRY +0.49, phishing PTH -1.83, phishing QRY +0.39 |
| e11_ref | e11_rs | benign PTH -2.72, benign QRY -0.56, phishing PTH +2.90, phishing QRY +0.05 |
| e11_joint | e11_ref | benign PTH +2.92, benign QRY +0.44, phishing PTH -0.17, phishing QRY -0.25 |
| e11_plan | e11_ref | benign PTH -0.99, benign QRY +1.58, phishing PTH +0.96, phishing QRY +1.38 |

### Outcome

| arm | failing | candidate |
|---|---|---|
| e11_rs | K11a, K11b, K11c, K11d | **no** |
| e11_ref | K11a, K11b | **no** |
| e11_joint | K11a, K11b, K11c, K11d | **no** |
| e11_plan | K11b, K11d | **no** |

## Appendix B — the checkpoint-step check (amendment 2)

`python research/native/e11_step_check.py`: the model's split at the end of 3,000 real hosts per class, teacher-forced in free mode's conditioning, renormalised over the five ways a host can end.

| arm | checkpoint | class | model PTH | model RS | model QRY | model EOS | data PTH |
|---|---|---|---|---|---|---|---|
| e11_rs | best | benign | 76.53% | 23.05% | 0.02% | 0.40% | 72.07% |
| e11_rs | best | phishing | 39.16% | 60.69% | 0.01% | 0.14% | 38.37% |
| e11_rs | final | benign | 75.75% | 23.80% | 0.02% | 0.44% | 72.07% |
| e11_rs | final | phishing | 38.64% | 61.22% | 0.01% | 0.13% | 38.37% |
| e11_ref | best | benign | 73.63% | 25.91% | 0.02% | 0.44% | 72.07% |
| e11_ref | best | phishing | 39.15% | 60.65% | 0.01% | 0.19% | 38.37% |
| e11_ref | final | benign | 74.24% | 25.30% | 0.02% | 0.44% | 72.07% |
| e11_ref | final | phishing | 39.28% | 60.52% | 0.01% | 0.19% | 38.37% |
| e11_joint | best | benign | 77.05% | 22.57% | 0.02% | 0.36% | 72.07% |
| e11_joint | best | phishing | 40.75% | 59.05% | 0.01% | 0.18% | 38.37% |
| e11_joint | final | benign | 74.63% | 24.89% | 0.03% | 0.45% | 72.07% |
| e11_joint | final | phishing | 39.48% | 60.32% | 0.02% | 0.18% | 38.37% |

**The plan's own plan at both checkpoints** (amendment 2, reading 1; shipped-weighted, PC1's training prefix; `data/e11/step_check_plan.json`)

| arm | checkpoint | benign P | benign Q | phishing P | phishing Q |
|---|---|---|---|---|---|
| e11_plan | best | 74.43% | 6.379% | 39.15% | 4.252% |
| e11_plan | final | 74.41% | 6.009% | 39.02% | 3.890% |
| e10_plan | best | 77.10% | 6.048% | 39.83% | 4.027% |
| e10_plan | final | 77.20% | 6.126% | 39.98% | 4.066% |
