# E11 — presence and the root slash: pre-registration

- **Registered:** 2026-09-29, before any E11 code, training or generation.
- **Decided by:** the owner (`DECISION_2026-09-29.md` §§1–5 and §7). The four arms,
  the clipping repair, the supplied plan and the shared start are the owner's
  decisions.
- **Basis:**
  - PC1 (`PC1_RESULTS.md`): gradient clipping discounts short rows, CLIP-LED;
  - E10 (`PREREG_E10.md`): the explicit presence plan;
  - the query-gap investigation (`research/qgap/`);
  - `E11_ROADMAP.md`.
- **Frozen, unchanged:**
  - the adopted baseline and every library default;
  - the data bundle, gate v2 and its thresholds, and every instrument;
  - every published result;
  - E8, which stays unread.

## 0. In plain words

- **Four models from one start, trained in one session.** Each adds one change to
  the one before it, so every change is measured on its own:

  | arm | adds | measured against |
  |---|---|---|
  | `e11_rs` | the root-slash token: the model can finally write `host/` | `e10_base` (E10's baseline arm, the same start) |
  | `e11_ref` | the clipping repair: short training rows stop being discounted | `e11_rs` |
  | `e11_joint` | the two hidden controls are hidden together or shown together, never one alone | `e11_ref` |
  | `e11_plan` | the presence plan; in free mode it is drawn from the data, as the suffix already is | `e11_ref` |

- **What must be true for an arm to be a candidate:**
  - its path and query rates, and its root-slash rate, meet the detectability
    tolerance;
  - nothing the gate measures gets worse beyond the retraining allowance;
  - clipping no longer binds by length.

## 1. Decisions, with their reasons

**D1 — The root slash, every arm.**
- `root_slash=True`: `<RS>` is written where `<PTH>` would be when the path is the
  bare `/`. It covers `host/` and `host/?q`.
- `<PTH>` still means a non-empty path, so every presence definition, control and
  gate reading is unchanged.
- The code and its tests are E8's; no E8 result is read.
- **Why** (the owner's decision, `DECISION_2026-09-29.md` §3): E3 is not claimed
  while the slash is missing. The tokenizer cannot write it today, so it appears
  0% of the time, against 98.0% of real benign bare hosts and 100% of phishing
  ones.

**D2 — The shared start.**
- **How it is built.** `<RS>` adds one token (108 instead of 107), which would
  change every seeded initial weight. So each arm's start is built from the
  107-token seeded start, E10's own:
  - every existing row and weight is copied;
  - `<RS>`'s row is drawn from N(0, 0.02), the model's own initialisation, from a
    separate generator seeded 20260929.
- **What each run records:**
  - the 107-token start's hash, which must equal E10's `26926388…`;
  - the 108-token start's hash, identical across the four arms.
- **Why:** the owner's decision (§7.4). `e11_rs` against `e10_base` then differs by
  the slash alone, down to the start.

**D3 — The clipping repair** (`e11_ref`, `e11_joint`, `e11_plan`).
- **The setting.** The gradient-clip cap stays at 1.0 for the first 1,000 steps,
  then rises to 5.0.
  - The warm-up is 500 steps. Steps 0–999 are the early window the training
    records report separately, where every length band is clipped often.
- **Why 5.0** [CONFIRMED facts, `PC1_RESULTS.md` and E10's training records]:
  - At the end of training the shortest batches' norms reach 2.16, and every other
    band stays under 0.9.
  - After step 1,000 the shortest band's mean norm was 0.99.
  - A cap of 5.0 is well above both, so it should stop the length-dependent
    discount. It still stops a genuine spike.
- **What it is not:** a gate threshold. It is a training setting.
- The owner's decision (§7.1).

**D4 — Joint dropout** (`e11_joint`).
- **The setting.** With probability 0.5 both presence controls are hidden;
  otherwise both are shown. Never one alone.
- **Why:**
  - Generation uses only those two states: free mode hides both, and control mode
    shows both.
  - Independent dropout spends half of all rows on partial states that are never
    used.
  - Joint dropout doubles the exposure of each state in use (25% → 50% of rows).
  - This is the question E10 registered after K10a's failure: the hidden state's
    exposure.

**D5 — The plan** (`e11_plan`).
- `presence_plan=True` with `field_dropout` 0, as in E10, plus `<RS>` and the
  repair.
- Control mode supplies the plan from the request, as E10 did.

**D6 — The supplied plan in free mode** (`e11_plan`; owner-approved, §7.2).
- **The table.** From every `name_train` row, by the gate's own field definitions,
  per class and suffix: the counts of the four presence states.
  - Written by `research/native/e11_targets.py` to `data/e11/presence_table.json`.
  - It ships in the code bundle with its hash.
- **One free generation of n URLs:**
  1. draw the state counts n_k from a multinomial with probabilities
     π_k = Σ_s shipped(s) · n(s, k) / n(s);
  2. for each state, generate n_k URLs with control mode's native call and the
     state supplied, drawing the suffix from w_k(s) ∝ shipped(s) · n(s, k) / n(s);
  3. shuffle the URLs.
- **Why this is the right distribution.** It is exactly the distribution of drawing
  each URL's suffix from the shipped table and its state from the data's rate for
  that suffix. The model writes every character.
- **The state shares** [CONFIRMED, `data/e11/targets.json`]:

  | class | P0Q0 | P0Q1 | P1Q0 | P1Q1 | path | query |
  |---|---|---|---|---|---|---|
  | benign | 24.967% | 1.664% | 68.415% | 4.953% | 73.369% | 6.617% |
  | phishing | 58.870% | 1.619% | 36.450% | 3.061% | 39.511% | 4.680% |

  The path and query columns are K9b's targets exactly.
- **Not generated:** `e11_plan` filling its own plan in free mode. Its plan
  calibration is read from the checkpoint instead (K10a's method with PC1's
  training prefix) and reported. That is the repair's effect on the plan.

**D7 — Decode.** Raw, the baseline's: temperature 1.0 and nothing else.

**D8 — Generation.**

| mode | what | size |
|---|---|---|
| free | every arm, 5 × 8,000 per class at `seed_free(g)`; the hidden arms run their controls `?`; `e11_plan` uses D6 | 320,000 URLs |
| control | every arm, 8 states × 5 generations at `seed_ctrl`, `n_ctrl`, native, SC1's state tables | 400,000 URLs |
| **all** | | **200 files, 720,000 URLs** |

**D9 — Instruments.**
- **E10's, unchanged:** gate v2 free cells, CR1 control cells, emission,
  honouring, novelty and copying.
- **New:** the root-slash rate. It counts URLs whose path is the bare `/` (a
  non-empty path with an empty body), among:
  - free: bare hosts, and host-route queries;
  - control: the P0Q0 and P0Q1 URLs that honour their state.

## 2. Criteria (fixed now), per arm and class

**The tolerance** (`DECISION_2026-09-29.md` §2): |q − p| ≤ T, with
T = 2 · sqrt(max(p(1 − p), 1/n) / n).
- n is the population's expected count in one generation:
  - free: 8,000 × the real share;
  - control: the state's per-generation size.
- **The floor 1/n is this registration's edge rule.**
  - Where fewer than one real minority event is expected per generation, the
    formula would give T ≈ 0. Phishing's slash is 100% in the data.
  - A single generation cannot resolve a rate more finely than one event, so the
    variance is floored there.
  - It applies to the rows marked *.

**The targets** [CONFIRMED, `data/e11/targets.json`]:

| outcome | benign: p, n, T | phishing: p, n, T |
|---|---|---|
| path, free | 73.369%, 8,000, ±0.99 | 39.511%, 8,000, ±1.09 |
| query, free | 6.617%, 8,000, ±0.56 | 4.680%, 8,000, ±0.47 |
| slash, free bare hosts | 98.004%, 1,997, ±0.63 | 100.000%, 4,710, ±0.04 * |
| slash, free host-route queries | 99.576%, 133, ±1.50 * | 100.000%, 130, ±1.54 * |
| slash, control P0Q0 (honoured URLs) | 97.944%, 3,000, ±0.52 | 100.000%, 3,000, ±0.07 * |
| slash, control P0Q1 (honoured URLs) | 99.551%, 3,000, ±0.24 | 100.000%, 1,000, ±0.20 * |

**The criteria:**

| code | criterion |
|---|---|
| **K11a** presence | free-mode PTH and QRY within T, both classes |
| **K11b** root slash | every slash row above within T, both classes |
| **K11c** no regression | against the arm's reference (§0), no gated cell has its verdict worsen **and** its x rise by more than 0.0141. Free: the 10 cells; control: every computable cell |
| **K11d** guardrails | against the reference, neither novelty nor query copying worsens by ≥ 0.5 points with a 95% CI excluding 0, in either mode |
| **K11e** the repair | arms with the repair only: in the training record after step 1,000, at most 1% of steps clipped in every length band |

**Outcomes:**
- **An arm that passes all its criteria is a candidate** (K11e only where it
  applies). Adopting one is the owner's decision.
  - The expected candidate is `e11_plan`: the supplied plan in free mode, the
    requested plan in control mode.
- **Every comparison with its reference is reported as that change's effect,**
  whatever the criteria say:
  - emission, slash, FULL, the gated cells and honouring;
  - the slash (`e11_rs`), the repair (`e11_ref`), exposure (`e11_joint`) and the
    plan (`e11_plan`).
- **If K11e fails for an arm,** its readings are reported as "repair incomplete".
- **If K11b fails in every arm,** the slash token does not produce the real rate.
  The cause is the next question.

**Reported, not criteria:**
- FULL (auxiliary), free and control, every arm;
- honouring per state, against the objective's 0.98 (K1a);
- `e11_plan`'s own plan calibration (D6);
- each arm's clipping record per band, with the share of steps whose norm exceeds
  1.0 (the old cap);
- the shared-start check (D2);
- the E3 finish line (`DECISION_2026-09-29.md` §3) for any candidate: every gated
  cell and the slash.

## 3. Predictions, before any run [HYPOTHESIS]

**`e11_rs` against `e10_base`:**
- The benign slash rows pass K11b.
- The phishing slash rows are uncertain: T allows about two misses in 4,710 bare
  hosts, and the benign class (98%) may leak.
- FULL falls sharply. In free mode, benign from about +0.15 to under +0.08, and
  phishing from about +0.39 to under +0.10. The E7 slash reference gave +0.072 /
  +0.052.
- Path and query rates within ±1 point of `e10_base`'s.
- K11c passes.

**`e11_ref` against `e11_rs`:**
- K11e passes.
- Benign free PTH falls by 1.5–3 points, into the tolerance.
- Phishing PTH falls by up to about 1.5 points, below the tolerance. PC1's
  phishing residual was −1.02 points on top of clipping's +1.45.
- Queries fall by 0.1–0.3 points, still outside.

**`e11_joint` against `e11_ref`:**
- Free QRY rises; the size is Unknown.
- Honouring rises.
- K11a queries are probably still outside.

**`e11_plan`:**
- Free-mode presence is about π_k × honouring, so K11a turns on honouring. See §3.1.
- Its own plan's benign P lands inside the tolerance with the repair. PC1
  projected about 73.98%.

### 3.1 Predictions informed by E10's readout

- **When.** E10's generation arrived as this document was written, and its readout
  (`PREREG_E10.md`) was still running.
- **What.** These predictions are added as an amendment when the readout is done,
  before any E11 training. Two things will be used:
  - the plan arm's per-state field rates in control mode, which turn D6's supplied
    plan into a predicted free-mode emission;
  - E10's content comparison (K10c).
- **The design above does not depend on them.**

**Amendment 1** (2026-09-29, after E10's VERIFY passed and before any E11 training).
The predictions from E10's registered readout. Sources: `data/e10/score_misc.json`
and `data/e11/predict_from_e10.json`, from `e11_predict.py`.

- **`e11_plan` passes K11a** in both classes, if it honours as `e10_plan` did.
  - Weighting `e10_plan`'s per-field rates when told each state by D6's state
    shares predicts:

    | class | path | query |
    |---|---|---|
    | benign | 73.32% (real 73.37, −0.04; T ±0.99) | 6.61% (real 6.62, −0.01; T ±0.56) |
    | phishing | 39.54% (real 39.51, +0.03; T ±1.09) | 4.63% (real 4.68, −0.05; T ±0.47) |

  - Told a state, `e10_plan` writes the requested path and query almost exactly.
    Its largest leak is 0.53% of benign "no path, no query" URLs getting a path.
  - `e10_base`, told the same states, would miss queries by about +1 point
    (7.69% / 5.62%). It leaks queries into "no query" states (1.2–2.4%).
- **Honouring stays high.** `e10_plan` honoured every control state at 95.0–99.7%,
  against `e10_base`'s 85.4–98.1%. The hardest state is still a query without a
  path (95.0% / 95.1%).
- **`e11_plan` may fail K11d on control-mode query copying.** `e10_plan` copied
  22.87% of its control-mode queries from training, against `e10_base`'s 20.53%
  (+2.34 points). Free mode moved the other way (20.30% against 21.04%).
- **The run-to-run spread is not small for queries.**
  - `e10_base` against `e9_ln` has the same recipe and the same start, differing
    only by GPU nondeterminism.
  - Benign query emission 5.07% against 4.58% (+0.49); paths within 0.32 points.
  - So a hidden arm's query reading near its tolerance (±0.56) is within one
    training's noise. The seed-to-seed spread S of the edge flag is at least this.
- **No change to the design, the criteria or the arms.**

### 3.2 Amendment 2 — the checkpoint-step check (2026-09-30, post hoc, reported only)

- **Why.** Added after VERIFY passed and after K11a, K11b and K11d had been read. It
  changes no criterion.
  - `best.pt` is chosen by validation loss. From step 14,000 the loss is flat
    within 0.001, so the chosen step is effectively random: 14,000 for `e11_rs`,
    `e11_joint` and `e11_plan`, 19,000 for `e11_ref`.
  - That step difference was flagged in the status before the readout.
  - The readings then show that the two hidden arms at 14,000 agree with each other
    and differ from `e11_ref` at 19,000: benign free PTH 77.09% and 77.28% against
    74.36%.
- **The check.** Two readings, laptop only:
  1. `e11_plan`'s own plan (K10a's quantity, PC1's training prefix) at `best.pt` and
     at `final.pt`, step 20,000;
  2. for the hidden arms, the model's path share at the end of 5,000 real
     domain-frame hosts per class, teacher-forced in free mode, at `best.pt` and
     `final.pt`.
- **How it is read.**
  - If a 14,000-step arm's benign path share moves by more than 1 point between its
    two checkpoints while `e11_ref`'s does not, the checkpoint step confounds the
    attributions (§2's "each change's effect"). They are reported as confounded.
  - The criteria are read as registered either way.

### 3.3 Amendment 3 — the reading at step 20,000 (2026-09-30, owner-approved, before any run)

**Why.**
- Amendment 2's rule triggered: the checkpoint step confounds every per-change
  reading (`E11_RESULTS.md` §7).
- The owner approved reading E11 again at one common checkpoint, the final step,
  to remove the checkpoint-selection confound. The question is whether the
  presence and slash results, especially `e11_plan`'s, hold there.

**What stays the same:**
- the 200 registered jobs: files, n and seeds;
- `e11_common`'s calls and the raw decode;
- the code bundle `uie-e11-code` (commit 6d32e2d), unchanged;
- the tables, including the presence table (`7a743452…`);
- the instruments (`e11_score.py`, the CR1 scorer);
- the criteria K11a–K11e, the tolerance with its edge rule, and the references.

**What changes: the checkpoint.** Each arm is read from its `final.pt`, step
20,000, with the hashes E11_train recorded:

| arm | `final.pt` hash |
|---|---|
| `e11_rs` | `c606d93e…` |
| `e11_ref` | `c86e60c5…` |
| `e11_joint` | `c71081b4…` |
| `e11_plan` | `1031739d…` |

- The files go to `data/e11_final/`.
- The same laptop tools read them with `--checkpoint final`.

**Validity:** V1–V4 and V-id as in §4, with `final.pt` in place of `best.pt`.

**One comparison still crosses steps.** `e11_rs`'s reference is E10's registered
`e10_base` reading, from `best.pt` at step 18,000. It is noted, not changed.

**How it is read:**
- the criteria at step 20,000, per arm, with §2's outcome rules;
- each change's effect is read from this reading, where all four arms share the
  step;
- the `best.pt` reading stays in the record as registered, and both are reported
  side by side.

**Predictions** [HYPOTHESIS], before any run:
- **`e11_plan` passes K11a again.** Its presence is supplied, and its honouring
  barely depends on the step.
- **FULL** moves by at most 0.02 from the `best.pt` reading in every arm.
- **The slash rows** move by at most 0.3 points, and K11b still fails on the
  phishing bare-host rows.
- **Benign free PTH at step 20,000:** `e11_rs` about 76%, `e11_ref` about 74.5%,
  `e11_joint` about 74.5–75%. So joint dropout moves it by under 1 point
  (amendment 2's host-end shares: 75.75, 74.24, 74.63).
- **The three arms read at step 14,000 improve on queries.** Their benign free QRY
  cell comes within 0.01 of `e11_ref`'s.
- **`e11_plan`'s control query cells** (P1Q1 QRY 0.0745, P0Q1 QRY 0.0847) improve.
  Whether they reach E10's UNDECIDED (0.054 / 0.064): no confident prediction.
- **K11c and K11d:** the regressions and copying rises found at step 14,000 shrink;
  no confident prediction on pass or fail.

## 4. Validity

| code | check | on failure |
|---|---|---|
| V1 | hashes and completeness: checkpoints, the code bundle, the tables (SC1's, the shipped table, the presence table), every file with its registered n and seed | fixed under an amendment; nothing read |
| V2 | byte identity on the laptop for at least 250 URLs per arm and mode, including the supplied-plan free mode | same |
| V3 | flags off are unchanged (tests); `uie/` unchanged since the code commit | same |
| V4 | the scorer reproduces HC1 C2's E7 cells exactly | same |
| V-id | all four arms' 108-token starts are identical; their 107-token start equals E10's; settings are the registered recipes; the data counts are recorded (the slash adds a token to root-slash rows, so the count that fits may differ slightly from E10's) | same |

## 5. Not claimed

- E3 passage: content cells stay E12's question;
- anything about E8's arms;
- that joint dropout or the plan fixes query content: E11 targets presence and
  structure.

## 6. Cost and order

1. This registration.
2. Code and tests, then the bundle and notebooks.
3. `E11_train`, run by the owner: 4 arms, about 4.6-5 h (E10's arms took 69 minutes each).
4. `E11_generate`, run by the owner: 720,000 URLs, about 7 h (E10's 360,000 took 3.4 h).
5. The laptop: validity, K11a–K11e and the reported readings.
6. `E11_RESULTS.md`. The owner decides adoption.

## Status

    E11  registered 2026-09-29 (this document), before any E11 code (9e89dcd)
    E11  code 6d32e2d (tests; full suite 866 pass; local smoke of the four arms);
         bundle uie-e11-code sha256 01b78a57...; notebooks E11_train /
         E11_generate handed to the owner (Downloads)
    E11  laptop readout tools f9389e0, 90b2fda, f8acf68, 69fb4ab
    E11  amendment 1 (section 3.1): E10-informed predictions, before any training
    E11  trained 2026-09-29 (results (16)): VERIFY_TRAIN PASS -- one shared start
         (5aa980f7...), its part without <RS> equal to E10's (26926388...);
         settings as registered; 10,316 rows dropped (E10: 10,313).
         K11e PASS for e11_ref, e11_joint, e11_plan: 0.00% of steps clipped in
         every band after step 1,000 (the old cap would have clipped 18.2 /
         18.5 / 27.5% of the shortest band); e11_rs, at 1.0, clipped 20.4%.
         Reported: e11_plan's own plan (checkpoint, shipped-weighted) benign
         P 74.43% Q 6.38%, phishing P 39.15% Q 4.25% (e10_plan: 77.10 /
         6.05, 39.83 / 4.03). best.pt at step 14,000 (e11_rs, e11_joint,
         e11_plan) and 19,000 (e11_ref); validation flat within 0.001 from
         14,000 on, so the step is noise -- but e11_ref against e11_rs also
         differs by checkpoint step, noted for the readout.
    E11  READ 2026-09-30 (E11_RESULTS.md): VERIFY PASS. No arm is a candidate.
         e11_plan: K11a PASS (73.38 / 6.57, 39.90 / 4.71), K11b FAIL (phishing
         bare hosts miss the slash 0.7%; benign control host-route slash too
         regular), K11c PASS, K11d FAIL (control copying +0.79), K11e PASS.
         The slash cuts FULL 79-98%. Amendment 2's rule triggered: the
         checkpoint step (14,000 vs 19,000) confounds every per-change
         attribution. Proposed, owner to decide: read E11 again from final.pt.
    E11  amendment 3 (section 3.3, 0a1acc5): the owner approved the step-20,000
         re-read. E11_generate_final.ipynb (same bundle uie-e11-code, commit
         6d32e2d, unchanged) reads each arm's final.pt; laptop tools take
         --checkpoint final / --train; the final path's training-side
         verification passes (final.pt hashes as E11_train recorded).
    E11  READ AT STEP 20,000 2026-09-30 (amendment 3; E11_STEP20K_RESULTS.md):
         VERIFY PASS. No arm is a candidate: e11_plan fails K11b (3 of 8 slash
         rows, 5 before) and K11d (control copying +0.96). K11c passes in every
         arm. At the common step e11_plan's benign control query cells and P1Q1
         PTH go CONVICTED -> UNDECIDED, and phishing free QRY 0.1382 -> 0.0953;
         the repair moves paths toward the real rates and leaves content
         unchanged; joint dropout gains nothing. Proposed (owner decides):
         final.pt in every later registration; E12_ROADMAP.md drafts E12.
