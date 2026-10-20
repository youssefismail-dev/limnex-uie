# E14 — query exposure (N2b) and the slash rule on plans with a path: pre-registration

- **Registered:** 2026-10-02, before any E14 code, training or generation. Execution
  waits for the owner's review of this document and its execution steps (§6).
- **Decided by the owner** (2026-10-02, answering `E13_RESULTS.md` §7; quoted in §1):
  - N2b, query exposure, is E14's primary lever;
  - D1 is extended to plans with a path, to close the slash condition;
  - D2 and the context arm are discarded.
- **Decided by the executor:** the details below, from `NATIVE_REPAIR_PLAN.md` (N2b),
  N2a, `E13_RESULTS.md`, `PREREG_E13.md` and E12.
- **Frozen, unchanged:**
  - gate v2, its thresholds and the detectability standard;
  - every instrument and every published result;
  - `e12_final` as the official baseline (`DECISION_2026-10-01.md`);
  - the library's defaults: every change is behind an option that is off by
    default, and off reproduces E12 and E13 bit for bit.

## 0. In plain words

- **E14 tests two levers, one change each.**
  - **Q, query exposure** (training). The model is retrained exactly as `e12_final`
    was: same start, same data, same batches, same number of steps. One thing
    changes: every character inside a query counts four times in the loss.
    Everything else counts exactly what it counted before.
    - Target: the three query cells that are not CLEARED. These are free phishing
      QRY, control benign P0Q1 QRY and control benign P1Q1 QRY.
    - Why: in E13's N, four times the query practice (A2) halved the free phishing
      query tell, and pushed benign P0Q1's to a CLEARED reading. But A2 trained on
      query rows alone. Q asks whether the same practice works inside the full
      training mix, without costing the other fields.
  - **D1′, the slash rule on plans with a path** (decoder only). A plan with a
    path also promises the slash. So when the host ends, it may not end without
    one.
    - Target: the last slash row. Its 19 misses all came from plans with a path,
      where the model dropped the path and ended the URL at the host.
- **D2 and the context arm are not run** (the owner's decision 3).
- **E3 is still out of reach in E14, by construction.** Three failing cells have no
  lever here:
  - control benign P1Q1 SUB (0.0416);
  - control benign P1Q1 PTH (0.0576);
  - control phishing P1Q0 PTH (0.0428).

  Q targets queries only, and D1′ touches only how the host ends. E14 reads E3 on
  its candidate for the record. A MET reading would need cells it does not target
  to clear on their own.

## 1. The owner's decisions (2026-10-02), verbatim

> Let's execute exactly what I outlined earlier: 1. Adopt N2b (Query Exposure) as
> the primary lever for E14. 2. Extend D1 to cover P=1 plans where the path is
> dropped by the model, to fully close the slash condition. 3. Discard D2 and the
> Context Arm. Please register this plan and give me the E14 execution steps. Do not
> begin execution yet; I want to review the registered plan and execution steps
> first.

**How this registration reads them:**
- **"N2b (Query Exposure)":**
  - This is `NATIVE_REPAIR_PLAN.md`'s N2b, the lever its decision table attaches to
    the EXPOSURE reading: "N2b: exposure (query-content loss weight)".
  - It carries N2b's registered guards and kill (§3.1), with the one amendment the
    owner already made: copying is judged against the natural rate (E13 decision
    6, E12 decision 4).
  - E13's registered aggregation could not license N2b, because benign read
    UNAVAILABLE (`E13_RESULTS.md` §5). This decision is the licence. It is the
    owner's, as `E12_ROADMAP.md` §2.4 left it.
- **"Extend D1 to cover P=1 plans":**
  - The extension is D1's own S = 1 rule, applied where S = 1 is implied by P = 1
    (§2.2).
  - It stays inside the decoder override of 2026-10-02 (`PREREG_E13.md` §1): it
    acts only on slash obedience, the diagnosed target. It changes no learned
    parameter. It is judged by D1's own registered rules.
- **"Discard D2 and the Context Arm":** neither is run.
  - Their code stays in the library, off by default. Removing it would change
    nothing measured, and the off-state is already proven bit-identical.

## 2. The levers and arms

### 2.1 Q — query exposure inside the mixture (`e14_q`)

**The recipe is `e12_final`'s, unchanged in every respect but one.** It keeps:
- the tokenizer, the model configuration, the corpus and the seed;
- E12's start (initial weights sha256 `c7e71edf…`, the same start E13's arms used);
- 20,000 steps at batch 256, and the learning-rate schedule;
- the fixed-loss objective (E9) and the clip schedule (1.0, then 5.0 from step
  1,000);
- R1 (`max_len` 384), the presence plan and the slash plan;
- `final.pt` as the weights read.

**The one change: a query-content loss weight, w = 4.**
- **Which positions.** Every prediction step taken while the query field is open:
  - from the first query character through the step that predicts the query's end
    (`<FRG>` or `<EOS>`);
  - not the step that predicts `<QRY>` itself. That step is the presence decision,
    which belongs to the field before it.
- **The whole step is weighted, never particular targets** (S4b's lesson,
  `uie/train.py` `field_mask`).
  - Weighting a step leaves the data's own ratio of "another character" to "the
    query ends here" untouched.
  - So the weight cannot teach the model that queries run longer or shorter. It
    only raises how much each query step matters.
- **Every row, both classes, both frames.** This is N2a's row set: the
  query-bearing rows of both classes, the rows E13's N trained A1 and A2 on.
- **The loss constant stays at `e12_final`'s, C = 14,397.885481960007.**
  - Under the fixed objective, each scored step's gradient weighs (its weight) / C.
  - With C pinned, every non-query step weighs exactly what it did in `e12_final`.
    Each query step weighs 4×.
  - Recomputing C from the weighted mass would instead shrink every other field's
    weight. That would make the trade-off a design choice rather than a
    measurement.
- **The validation set is unweighted.** The training log's validation losses stay
  comparable with `e12_final`'s.
- **The batches are `e12_final`'s.** The weight touches the loss, never the
  sampler, so the same seed draws the same rows in the same order.

**Why w = 4, fixed now, with no sweep.**
- A2's 4× exposure is the only query practice with measured evidence on the current
  recipe. A1's 1× tells nothing about the mixture: it trained on query rows alone.
- One value is registered. If Q fails, no other weight is tried in E14. Trying
  weights until a cell clears would be tuning the generator against the metric.

**Why a loss weight, and not oversampling query rows or more steps.**
- More steps are excluded by the standing constraint.
- Oversampling query rows would also re-expose their labels, hosts and paths. That
  is the class-exposure route to memorisation E2.2 measured. It would also change
  every batch, so the arm would differ from `e12_final` in more than one thing.
- The weight changes one thing: how much the query content counts.

**What the weight is not, stated because predictions rest on it.**
- Under AdamW, a 4× gradient weight on a subset is not the same as 4× the
  repetitions. The optimiser normalises each parameter's step by its own gradient
  history.
- Query steps share every parameter with the other fields. So the extra weight can
  still crowd the other fields at the optimiser level, even with C pinned.
- G1 (§3.1) catches that if it reaches a cell. The per-field validation NLL
  reports it in any case (§3.5). `NATIVE_REPAIR_PLAN.md` calls a trade-off
  "itself evidence of capacity in the mixture".

### 2.2 D1′ — the slash rule on plans with a path (decoder only)

**D1 as registered in E13 is unchanged for P = 0 plans.**

**The extension, for P = 1 plans (states P1Q0S1 and P1Q1S1), in both modes and both
classes:**
- **Where:** at every sampled step from the label marker until the host ends. The
  host ends with the first of `<PTH>`, `<RS>`, `<QRY>`, `<FRG>`, `<EOS>`. After that,
  the rule is off for the row. These are D1's own bounds.
- **What:** the slash-less ends (`<QRY>`, `<FRG>`, `<EOS>`) get probability 0. Their
  mass goes to `<PTH>` and `<RS>` in proportion to those two ends' own
  probabilities, under the decoder's temperature. This is D1's S = 1 rule exactly.
- **What it guarantees, by construction:**
  - the probability of ending the host at each step is unchanged, so host lengths
    are unchanged;
  - every non-end token keeps its probability;
  - `<PTH>`'s probability can only rise, so P honouring in P1 states can only rise.
- **What happens to the 19 path drops:** each now ends the host with `<PTH>` (the
  path is written) or `<RS>` (`host/`, the path still missing). The split follows
  the model's own odds of those two ends.

**Why `<RS>` stays allowed, rather than forcing `<PTH>`:**
- Forcing `<PTH>` would be a path-presence rule. The override covers slash
  obedience. A P1 plan's slash is honoured by either slashed end.
- Allowing both is D1's own rule with nothing new added. It is the change
  `E13_RESULTS.md` §7.1 described.

**Its scope, measured in E13's files** (`e12_d1`, descriptive; P1-plan URLs whose
host ended without a slash):

| population | e12_d1 |
|---|---|
| free phishing | 22 of 15,956 (19 bare, 3 `host?q`) |
| free benign | 26 of 29,410 |
| control phishing P1Q0 / P1Q1 | 18 of 15,000 / 13 of 5,000 |
| control benign P1Q0 / P1Q1 | 12 of 15,000 / 13 of 15,000 |

- These URLs are the same in `e12_final`'s files. D1 left every URL of a P1 plan
  byte-identical: free benign 29,410 of 29,410, free phishing 15,956 of 15,956,
  and every control P1 file.
- That record is the basis of V7 (§5).
- **Outcomes against strings:**
  - In distribution, D1′ changes how the host ends for about 0.1% of URLs: the
    table's rows.
  - The text of other P1-plan URLs can differ too. Changing a step's
    probabilities can re-draw that step's sample, and every later step diverges
    after it.
  - In P0 plans, D1 moved a lot of end mass. It fixed 57 phishing misses and
    changed the text of 13,094 of 24,044 free phishing P0 URLs.
  - In P1 plans, the slash-less ends carry little mass (the table's rates), so
    fewer re-draws are expected. The count is recorded, not predicted.
  - A re-drawn URL is a fresh draw from the same distribution. DC1's paired rules
    (R-a) are built for exactly this.

**Recorded:** per row, the end mass D1′ moved, by plan (descriptive).

### 2.3 Discarded (decision 3)

- **C (the context arm)** missed K13-1 by 0.0004 (0.0137 against 0.0141). Its
  mechanism check passed only because real-context queries got worse.
- **D2 (field calibration)** had no measurable effect.
- Neither is generated, trained or read in E14.

### 2.4 The arms

| arm | weights | decode | role | judged against |
|---|---|---|---|---|
| `e14_q` | `e14_q` `final.pt` | raw | Q alone | `e12_final` (retrained: the retrain allowance applies) |
| `e12_d1p` | `e12_final` `final.pt` (E12's, unchanged) | raw + D1 + D1′ | D1′ alone | `e12_final` (same weights: DC1's paired rules) |
| `e14_qd` | `e14_q` `final.pt` | raw + D1 + D1′ | the combination | `e14_q` (same weights: DC1's paired rules) |

- **Every arm runs E12's 50 jobs:**
  - free mode: 5 × 8,000 per class at `seed_free(g)`;
  - control: 8 states × 5 at `seed_ctrl`.
  - S is supplied as PREREG_E12 D2 says.
- **The seeds are `e12_final`'s,** so every cell pairs with E12's own files.
  - `e12_final`'s files are not regenerated. They are the baseline.
  - `e12_d1`'s files (E13) are reused for V7, never regenerated.

## 3. Criteria (fixed now)

### 3.1 Q — `e14_q` against `e12_final`

| code | criterion |
|---|---|
| **K14-1** target (N2b's kill, inverted) | at least one of the three target cells (free phishing QRY, control benign P0Q1 QRY, control benign P1Q1 QRY) has x fall by more than 0.0141 |
| **K14-2** fit (N2b guard 1) | held-out, unweighted query NLL per character falls in each class (definition below) |
| **K14-3** band profile (N2b guard 2) | the rare-band share of generated parameter names does not overshoot the null, in either class (definition below) |
| **G1** no regression (N2b guard 4) | K12c's rule: no gated cell's verdict worsens with x up by more than 0.0141 (free and control, both classes) |
| **G2** presence | K12a: free PTH and QRY within T, both classes |
| **G3** slash | no K12b row that `e12_final` passed fails |
| **G4** honouring | no control state's P or Q honouring falls (amendment 1's rule: by at least T, with the two-proportion 95% interval excluding 0) |
| **G5** copying (N2b guard 3, as amended) | K12d's memorisation rule: no QC1-1 cell's copy rate or singleton-copy rate is OVER its natural rate. And no cell's copy rate moves further from its natural rate (amendment 1's rule) |
| **G6** novelty | K12d's novelty rule against `e12_final`: whole-URL novelty, both modes (DC1's `prop_worsening`) |
| **G7** FULL | auxiliary FULL, free mode, both classes: x up by no more than 0.0141 |
| **G8** training health | K12e: after step 1,000, at most 1% of steps clipped in every band (bands on the w = 1 mass, as V5) |

**Q is adopted iff K14-1, K14-2, K14-3 and G1–G8 all hold.**
- **Otherwise Q is killed, and N2b is closed as registered.** No second weight and
  no second form are tried.
- **"Adopted"** means the lever is kept, and `e14_q` is the candidate baseline.
  Making it the official baseline is the owner's decision, as it was for
  `e12_final`.

**K14-2, defined.**
- **Instrument:** N2a's M3 (`research/native/n2a_reference.py` `teacher_forced`).
  It is teacher-forced negative log-likelihood per query character. It uses the
  plan the tokenizer scores, the domain frame, and every validation query row of
  the class.
- **The comparison:** computed per row for both models on the same rows.
- **"Falls":** the pooled difference (`e14_q` − `e12_final`) is below 0, and its
  paired row-bootstrap 95% interval excludes 0 (1,000 draws, seed 20260806).
- **Why the interval:**
  - N2b's guard exists to catch the S3/S4 failure: detector gains without a gain
    in held-out fit.
  - A fall inside the noise is not a gain in held-out fit. So it does not meet the
    guard.

**K14-3, defined.**
- **Instrument:** N2a's M2 (`band_profile`), on `e14_q`'s free-mode files. That is
  every query-bearing URL in all 5 generations of the class.
- **The null:** the faithful-sampler null over the class's training queries.
- **"Overshoots":** the observed rare-band share's 95% interval lies wholly above
  the null's share.

**G5 and G6: what changed from E13, and why** (decided now, before any data):
- E13's C carried DC1's query-copying clause: "the query copy share must not
  worsen by 0.5 points".
- That clause counts any rise as worsening. E12's adopted rule judges copying
  against the natural rate, in both directions (`E12_ROADMAP.md` §2.4,
  decision 4). E13 recorded the conflict at read time.
- Q is expected to raise copying toward the natural rate: A2 did, and stayed
  under it. That clause would reject Q for moving toward the real distribution.
- So the retrained arm is judged by K12d, E12's own rule, together with amendment
  1's distance rule. Neither is weakened. Q fails if copying goes OVER the natural
  rate, or moves away from it.

**N2b's fourth guard, "no other field regresses",** is read on the gate's cells
(G1). That is E12's registered regression rule for a retrained model. Per-field
NLL is reported (§3.5), not judged. `NATIVE_REPAIR_PLAN.md` gives no threshold for
it, and inventing one now would be a new gate.

### 3.2 D1′ — `e12_d1p` against `e12_final`

**These are E13 §3.2's rules for D1, verbatim, with amendment 1.** D1′ is judged by
exactly the rules D1 passed.

| code | criterion |
|---|---|
| **K14-4** target | all 8 K12b slash rows within T |
| **R-a** cells | REGRESSION if any gated cell (both classes, every computable cell, FULL included) has Δ > 0 at Holm p < 0.05 over all the cells of its mode, or its verdict worsens (DC1's rule) |
| **R-b** presence | K12a holds, and no free-mode PTH or QRY presence rate moves (amendment 1's rule) |
| **R-c** obedience | no state's P or Q honouring falls (amendment 1's rule); S honouring does not fall in any state |
| **R-d** copying and novelty | G5's rule, and DC1's novelty rule: whole-URL novelty and the query copy share each worsen by less than 0.5 points (95% CI) |
| **R-e** validity | V1–V7 (§5) |
| **R-f** cost | forward passes per delivered URL: 1.0× (no second pass) |

**D1′ is adopted iff K14-4 holds and no R fails.**
- R-d keeps DC1's query-copying clause here, though G5 drops it for Q:
  - D1′ is judged by the rules D1 passed, unchanged;
  - and a decoder rule on host ends has no business moving query copying at all.
- R-a keeps "or its verdict worsens", the clause that failed D2 on boundary noise:
  - it is the owner's no-regression standard as registered for the decoder;
  - D1 passed it with the same kind of re-draw (§2.2).
  - **The known risk:** a cell sitting on a verdict boundary can tip on re-drawn
    strings alone. If that happens, D1′ is rejected as registered.
    - The cells at risk are the UNDECIDED ones in P1 states: control benign P1Q1
      SUB, PTH and QRY, and control phishing P1Q0 PTH.
    - P1Q1 PTH already tipped once to CONVICTED at Δ ≈ 0, in E13's `e12_dec`.
    - Control P0-state cells cannot tip from D1′. In P0 plans D1′ is exactly D1,
      so their strings are `e12_d1`'s (V7), and those passed R-a in E13.

### 3.3 The combination — `e14_qd` against `e14_q`

- **K14-4 on `e14_qd`:** all 8 slash rows within T.
- **R-a to R-f against `e14_q`,** as in §3.2. Same weights, so DC1's paired rules
  apply.
- **This checks that D1′ behaves on Q's weights as it does on `e12_final`'s.** It
  holds only if Q and D1′ are both adopted.

### 3.4 The E3 reading

- **E3 (`DECISION_2026-09-29.md` §3; `e12_compare.finish_line`):** every gated cell
  is CLEARED, free and control, and all 8 slash rows are within T.
- **It is read on every arm,** for the record.
- **A MET reading counts only on an arm whose levers are all adopted:**
  - `e14_qd` needs Q, D1′ and §3.3;
  - `e14_q` needs Q;
  - `e12_d1p` needs D1′.
- **E3's bar is not changed.** No cell is excused, and no tolerance is moved.

### 3.5 Reported, not criteria

- Each arm's effect on every cell.
- M3 on training rows: the train/validation gap (A2's memorisation signature).
- M1 (the value-termination hazard) and M4 (copying), N2a's instruments.
- Per-field validation NLL for every field, `e14_q` against `e12_final` (§3.1).
- The query share of the scored mass, unweighted and weighted (the dry run).
- Honouring per state, S honouring, and D1′'s moved mass.
- Copying and novelty profiles, and K12b's rows for every arm.
- The clip record, per band.

## 4. Predictions, before any code, training or generation [HYPOTHESIS]

- **Q:**
  - K14-1 holds through free phishing QRY, which falls by more than 0.0141 to
    roughly 0.07–0.09. It is not CLEARED.
  - Control benign P0Q1 QRY falls, but stays UNDECIDED.
  - Control benign P1Q1 QRY moves by less than 0.0141. N's A2 barely moved it.
  - K14-2 holds in both classes.
  - K14-3 holds: the rare-band share stays far under the null, as in N2a.
  - Copying rises toward the natural rate, and is not OVER it.
  - The guard most at risk is G1. A non-query cell may cross the retrain rule
    through the optimiser-level crowding §2.1 names.
- **D1′:**
  - K14-4 holds: 8 of 8 rows on `e12_d1p`.
  - No R fails.
  - V7 holds: every P0-plan URL is byte-identical to `e12_d1`'s.
- **The combination:** K14-4 holds on `e14_qd`, and no R fails.
- **E3:** NOT MET on every arm. At least these cells stay uncleared:
  - control benign P1Q1 SUB;
  - control benign P1Q1 PTH;
  - control phishing P1Q0 PTH;
  - free phishing QRY;
  - control benign P0Q1 QRY;
  - control benign P1Q1 QRY.

## 5. Validity

| code | check | on failure |
|---|---|---|
| V1 | hashes and completeness: the code bundle; `e12_final.pt`; `e14_q`'s `final.pt` as training recorded it; the tables (the shipped suffix table, the state and presence-slash tables); every generated file with its registered n and seed | fixed under an amendment; nothing read |
| V2 | byte identity on the laptop CPU: at least 250 URLs per arm and mode, D1′ included | same |
| V3 | options off are unchanged: the full test suite; E12's fingerprints; with `query_weight` 1.0 the masks equal `field_mask`'s bit for bit; with the P1 extension off, decoding equals E13's D1 bit for bit; `uie/` unchanged since the code commit | same |
| V4 | the scorer reproduces `e12_final`'s published cells exactly from E12's files | same |
| V5 | Q's training, from its record. The start weights are E12's (`c7e71edf…`). The row counts equal `e12_final`'s: 2,551,259 benign and 238,707 phishing, with 3,851 dropped over `max_len`. C equals 14,397.885481960007. **The same batches:** the per-band step counts on the w = 1 mass (`e12_final`'s mask) equal `e12_final`'s (3,454 / 9,973 / 5,637 / 936). Unit tests show the weight sits on exactly §2.1's positions, and zero-weight positions stay zero | same |
| V6 | D1′, unit tests: (a) the host-end probability is preserved; (b) in P1 plans, only `<QRY>`, `<FRG>` and `<EOS>` go to 0, and only `<PTH>` and `<RS>` gain, and only before the host ends; (c) P0 plans behave exactly as D1; (d) defaults are bit-identical | same |
| V7 | D1′'s scope, on the generated files: every URL of a P0 plan in `e12_d1p` is byte-identical to the URL at the same position in `e12_d1`'s E13 files (same weights, seeds and GPU type; D1′ equals D1 there). The basis is §2.2's record | the decoder touched URLs outside its scope: stop, nothing read |

## 6. Execution steps

Nothing below starts until the owner approves this registration. "Laptop" steps are
mine. "Kaggle" steps are the owner's.

| step | who | what | about |
|---|---|---|---|
| 1 | laptop | **Code and tests,** every option off by default. In `uie/train.py`: `query_weight` (in `field_mask` and `TrainConfig`, written to a checkpoint only when set), the unweighted mass beside T_b in the clip record, and validation always unweighted. In `uie/kaggle_train.py`: pass `query_weight` and the pinned `loss_constant` through. In `uie/model.py`: D1′ as an option of the slash rule. In `research/e14/`: the common module, the training wrapper, K14-2 and K14-3's measurement, the verifier (V5 and V7 included), the scorer wrapper (E13's), the comparison, the tables, and the bundle and notebook builders. Tests. The full suite passes. Commit and push, verified on the remote | a session |
| 2 | laptop | **Pre-flight.** Q's dry run on the corpus: the row counts, the dropped rows, and the query share of the scored mass, recorded in Status. CPU smoke tests: a short weighted training run, and D1′ decoding a few URLs of each plan. The bundle `data/kaggle/uie-e14-code.zip`, from a clean committed tree with no `.gz` inside. The three notebooks and `E14_KAGGLE.md`. Hashes in Status. Commit and push | 1–2 h |
| 3 | Kaggle | **Upload** `uie-e14-code.zip` as a new private dataset, `uie-e14-code` | minutes |
| 4 | Kaggle | **`E14_decode`:** `e12_d1p`'s 50 files (`e12_final` + D1′). Needs no training, so it can run first. Inputs: `uie-e14-code`. GPU T4 | ≈ 1.5 h |
| 5 | Kaggle | **`E14_train`:** `e14_q` (20,000 steps). Inputs: `uie-corpus2`, `uie-e14-code`. GPU T4. Runs in either order with step 4, or at the same time | ≈ 1.5 h |
| 6 | laptop | **VERIFY_TRAIN (V1, V5) and K14-2,** from the downloaded `E14_train` output. K14-2 needs only the two checkpoints. `E14_generate` runs only after this passes | ≈ 1 h |
| 7 | laptop | **VERIFY_DECODE (V1, V2, V7),** from the downloaded `E14_decode` output | < 1 h |
| 8 | Kaggle | **`E14_generate`:** `e14_q` raw and `e14_qd`, 50 files each. Inputs: `uie-e14-code` and `E14_train`'s output. GPU T4 | ≈ 3.2 h |
| 9 | laptop | **VERIFY_GENERATE (V1, V2), V4, scoring** (E13's memory-guarded pool, three arms), K14-3, the criteria, the E3 readings, the tables and `E14_RESULTS.md`. Commit and push, verified on the remote | ≈ 5 h, mostly scoring |
| 10 | owner | **Read the results.** Decide whether `e14_q` becomes the official baseline, if Q is adopted | — |

- **About 6 GPU hours on Kaggle in all.** Every notebook writes `PROVENANCE.json`
  after each file. If a session stops early, a rerun keeps what finished.
- Each download goes to `Downloads` as Kaggle names it (`results (N).zip`). I
  extract and verify it before anything is read.
- The run sheet, `E14_KAGGLE.md`, is written in step 2. It lists exactly what to
  upload and attach, and what each notebook must print.

## 7. What this registration does not do

| standing constraint | E14 |
|---|---|
| no threshold, gate or tolerance changes | none. G5 and G6 apply E12's own copying rule (§3.1, stated before data) |
| no extra capacity or training steps | same model, 20,000 steps, same batches |
| decoder changes only under the override, proposed and registered first | D1′ is D1's S = 1 rule, for the diagnosed slash misses, registered here before any code |
| no optimising the generator against the metric | one weight, fixed now from A2; no sweep, and no retry if Q fails |
| no reopening the PTH attribution branch | E14 does not touch PTH attribution |
| R2 rejected, R3 locked | unchanged. Q's failure would not unlock R3. That needs `DECISION_2026-09-29.md` §4's strong proof, which E14 does not measure |
| the repository stays private | unchanged |

## Status

    E14  registered 2026-10-02 (this document), before any E14 code, training or
         generation. Awaiting the owner's review of the registration and of the
         execution steps (section 6); nothing has been executed.
         - Section 2.2's scope counts and the out-of-scope identity record were
           read from E13's existing files (e12_final, e12_d1), which E13 had
           already read; no E14 data exists.
    E14  APPROVED by the owner 2026-10-02, unchanged, verbatim:
           "I have reviewed the E14 registration and I approve the choices you
           made, including: Applying w=4 to both classes. Sticking to w=4 without
           retry if it fails. Keeping the non-query weights and the loss constant
           intact. The D1' design and the copying rule adjustment. I also
           understand and accept that E14 cannot fully clear E3 because of the
           remaining PTH and SUB cells. We will secure the QRY and slash fixes
           first, then address those remaining cells in a subsequent step. You
           may proceed with Execution Steps 1 and 2."
         Steps 1-2 (code, tests, pre-flight, bundle, notebooks) begin; nothing
         runs on Kaggle before the owner does.
    E14  code (step 1), every option off by default:
         - uie/model.py: generate(slash_paths), slash_ends(paths) -- D1';
         - uie/train.py: field_mask / encode_corpus(query_weight), TrainConfig.
           query_weight (in a checkpoint only when set); train() reads the clip
           bands on the w = 1 mass, records the query share, and refuses an
           unpinned C under the fixed objective;
         - uie/kaggle_train.py: run(query_weight), on the training rows only;
         - research/e14: the common module, K14-2/K14-3 (e14_fit), the verifier
           (V1, V2, V5, V7), the scorer and its runner, the criteria
           (e14_compare), the tables, the smoke tests, the bundle and notebook
           builders;
         - tests/test_e14.py (17) and tests/test_e14_common.py (6). The full
           suite passes.
    E14  pre-flight (step 2), on the laptop:
         - Q's dry run (data/e14/dryrun.json): every training row encoded as
           E14_train will encode it. Counts 2,551,259 benign / 238,707 phishing,
           3,851 dropped: e12_final's exactly. Query-bearing rows 225,533 /
           13,673 (N's kept rows). In every row the weighted positions are
           exactly section 2.1's query steps. Query steps are 5.13% (benign) and
           5.76% (phishing) of the scored positions: at w = 4 they carry about
           17.8% / 19.6% of the scored mass, and a batch's mass rises by about
           15% / 17% with C pinned (descriptive)
         - smoke tests, all PASS: train_q (the same batches at w = 1 and w = 4
           on real rows); decode (all three arms generate; D1' slashes every P1
           host, 180 of 180; P0 plans byte-identical to D1); fit (the per-row M3
           equals N2a's M3 on real rows, both classes); compare (on E13's
           already-read files, the D1' rules reproduce E13's D1 reading exactly,
           and K14-1 recovers e13_ctx's fall of 0.0137)
         - the bundle: data/kaggle/uie-e14-code.zip, from commit 2f4f01c, sha256
           dac834260eaccd1a847db5284e10c830522b70a413d19e329dbc34da3b6e2498
           (51.2 MB; 39 code files, 3 tables, e12_final.pt; nothing gzipped)
         - the three notebooks, run end to end on the laptop CPU from the
           unzipped bundle (Kaggle's paths patched, tiny jobs): E14_decode and
           E14_generate write and record their files; E14_train trains, and its
           readback cell holds the weight, the extra setting and the pinned C.
           The start check fired as designed (this laptop's torch 2.13 rebuilds
           another start than Kaggle's 2.10) and was bypassed for the
           simulation only. Through run(), the weight reached the rows (query
           share 0.10 on the tiny data) with the same batches as w = 1
         - the verifier on stand-ins (outside data/e14): decode on E13's e12_d1
           files renamed -- no failure; V7 counted 84,634 P0-plan URLs identical.
           Train on e12_final in e14_q's place -- only the two missing-weight
           checks fail; every other V5 check passes against e12_final's record
         - run sheet: research/e14/E14_KAGGLE.md. Awaiting the owner's Kaggle
           runs (steps 3-5)
    E14  E14_train returned (results (24)); E14_decode returned (results (25)):
         - E14_train: 80.0 minutes on a T4, torch 2.11 (E12/E13 ran 2.10); the
           start check PASSED on 2.11: both fingerprints equal the recorded ones.
           VERIFY_TRAIN PASS (data/e14/VERIFY_TRAIN.json): e12_final's settings,
           tokenizer, start, row counts, dropped rows, C and per-band step counts
           (same batches) all equal; query weight 4.0; query share of scored
           positions 0.0521. e14_q final.pt sha256 d11726b1...; final validation
           loss benign 1.5273, phishing 1.4173 (e12_final: 1.5621, 1.4006;
           descriptive, whole-URL)
         - E14_decode: VERIFY_DECODE PASS (data/e14/VERIFY_DECODE.json): byte
           identity 669/669 free and 413/413 control; V7: 84,634 of 84,634 P0-plan
           URLs identical to e12_d1's (torch 2.11 against E13's 2.10); D1' re-drew
           18,280 of 95,366 P1-plan URLs (descriptive)
         - K14-2 PASS (data/e14/k14_2.json; N2a's M3 per row; the per-row sums
           equal N2a's on a 480-row self-check). Query NLL per character,
           e12_final -> e14_q, difference [95% CI]:
             benign validation  1.5998 -> 1.5269  -0.0729 [-0.0786, -0.0676]  7,204 rows
             phishing validation 2.5377 -> 2.3528  -0.1848 [-0.2194, -0.1543]  1,048 rows
             benign train       1.5588 -> 1.4422  -0.1166 [-0.1201, -0.1133]  19,827 rows
             phishing train     2.5110 -> 2.2479  -0.2631 [-0.2774, -0.2512]  8,130 rows
           Descriptive: the validation-minus-train gap widens, benign +0.041 ->
           +0.085, phishing +0.026 -> +0.105 (A2's was +0.107 / +0.104)
         - E14_generate (step 8) is licensed to run
    E14  E14_generate returned (results (26)): VERIFY_GENERATE PASS
         (data/e14/VERIFY_GENERATE.json): 100 files, byte identity 1,082 of
         1,082 over e14_q's and e14_qd's four target files each. Scoring started
         2026-10-03 21:17 (e14_score_run.py: V4, N2a's M1-M4 / K14-3, and the
         free / control / misc / copy parts for e12_d1p, e14_q, e14_qd)
    E14  READ 2026-10-05 (E14_RESULTS.md; data/e14/compare_report.json; scoring
         finished 2026-10-04 01:55, 14 jobs, all exit 0; V4 0.0). Nothing adopted;
         E3 NOT MET on every arm, as registered:
         - Q: K14-1, K14-2, K14-3, G1, G2, G4-G8 PASS; G3 FAIL (phishing
           free_host_query 98.165% against 100% +-1.544; e12_final passed it).
           Free phishing QRY 0.1141 -> 0.0830; benign P0Q1 QRY CLEARED (0.0365)
         - D1': K14-4 PASS (8 of 8 slash rows); R-a FAIL: benign P1Q1 PTH
           UNDECIDED -> CONVICTED at delta -0.0001, Holm p 1.00 (the risk named in
           section 3.2)
         - combination: K14-4 PASS; R-a FAIL: free benign QRY CLEARED ->
           UNDECIDED at delta +0.0008, Holm p 1.00
         - predictions: 10 held, 5 did not
    E14  owner decisions 2026-10-05 (research/native/DECISION_2026-10-05.md):
         - Q closed as REJECTED: e14_q and e14_qd not adopted; N2b closed; no
           retry, no further weight tuning
         - R-a registered anew as R-a' (cb389b5) before use; applied post-hoc at
           the owner's direction: D1' passes every criterion under it and is
           ADOPTED; the slash fix (D1 + D1') is the decode from E15 onward
