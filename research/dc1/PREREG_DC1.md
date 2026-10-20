# DC1 — calibrated per-field decoding for E7: pre-registration

**Registered:** 2026-09-26, before any DC1 code, fit or generation.
**Approved by the owner (2026-09-26):** generation-level repair of E7's failing
cells in the two shipped modes, with no retraining.

**Frozen:**
- `data/best_e7.pt` (sha256 `916d0385…`);
- R4's decoder (temperature 1.0, delta 0.22 / 0.30), kept as is — DC1 settings sit on top of it;
- gate v2 and its thresholds;
- `research/cr1/cr1_score.py`;
- `data/bundle2`;
- the adopted state tables (`0c6f2760…`) and held resampling (`uie/controlled.py`);
- every published result.

**Scope:** FULL FREE (`<P:?> <Q:?>`) and FULL CONTROL (`<P:0/1> <Q:0/1>`, both explicit) only. Single-field states are out of scope.

## 1. Targets — E7's failing cells

| mode | cell | today |
|---|---|---|
| free (HC1 C2) | benign PTH | UNDECIDED, +0.0369 |
| free | benign QRY | UNDECIDED, +0.0541 (n = 289) |
| free | phishing PTH | UNDECIDED, +0.0604 |
| free | phishing QRY | pooled UNDECIDED, +0.0924 |
| control (SC1 arm T) | benign P1 Q1: SUB, PTH, QRY | CONVICTED |
| control | benign P0 Q1: QRY | CONVICTED |
| control | phishing P1 Q0: LBL, SUB, PTH | UNDECIDED |

## 2. The change — generation only

For each class, each mode (free, and each joint state) and each field f in LBL, SUB, PTH, QRY:

- **T_f** — the temperature at every step inside f.
- **δ_f** — a log-odds bias on every token that ends f (the end token and the field markers).
  - A uniform shift keeps which field comes next unchanged; it changes only how long f runs.
  - It adds to the frozen path delta.
- **w_f** (control mode only) — blends content characters: z = z_free + w_f · (z_control − z_free).
  - Field markers and the end token always come from z_control, so the control keeps sole authority over structure.

Defaults (T = 1, δ = 0, w = 1) reproduce today's generator bit-for-bit (tested against the existing fixtures).

The sampler's field tracking is corrected so injected suffix/label markers update it. Otherwise LBL settings would never apply. Existing output is unchanged; the same fixtures prove it.

In control mode, honouring stays the adopted held resampling.

## 3. Fitting — never on the data the gate scores

- **Objective.** Each (class, mode, field) setting minimises teacher-forced cross-entropy on real URLs from the **training** split (`name_train`, `phishing_name_train`, domain frame), fitted by L-BFGS-B.
- **Samples.** Free mode: 20,000 URLs per class, controls `?`. Each joint state: up to 10,000 URLs, with that state's controls.
- **Bounds.** T in [0.5, 2], δ in [−3, 3], w in [0, 2].
- **Validation confirms once; it never tunes.** The **validation** split (the gate's real data) is used once, per (class, mode, field):
  - on up to 5,000 URLs, the per-URL paired cross-entropy difference (fitted − default) must have a 95% CI below 0;
  - otherwise that field stays at the defaults.
- **No setting is ever chosen by the gate.**

## 4. Evaluation — gate v2 code, unchanged

- **Free.** Two arms, today's E7 and DC1, with the same seeds, both classes, G = 5.
  - n_main is 40,000 benign / 25,000 phishing per generation, so QRY reaches min(1,200, pool / 3) instead of 289 or pooled.
  - Contrasts are paired.
- **Control.** The DC1 arm on the six measurable joint states, 5 × 3,000 per state, scored by `cr1_score`, paired against SC1 arm T's files (read-only).
- **Validity (K0).**
  - checkpoint, settings and code-bundle hashes;
  - every controlled URL is in its state;
  - CPU byte identity on at least 250 URLs per mode;
  - the free-mode scorer reproduces HC1 C2's published E7 record from its stored generations.
- Generation runs on Kaggle; verification and scoring run on the laptop.

## 5. Criteria, per mode

Throughout, Δ = x(DC1) − x(baseline).

- **FIXED:** a target cell's verdict becomes CLEARED.
- **IMPROVED:** a target cell's Δ < 0, with Holm p < 0.05 over that mode's targets.
- **REGRESSION:** any cell in the mode with Δ > 0 and Holm p < 0.05 over all its cells, or a CLEARED cell becoming CONVICTED.
  - The checkpoint is the same, so no retraining-noise allowance applies (R4's rule).
- **Adopt DC1 for a mode iff all of:**
  - at least 1 target cell is IMPROVED;
  - 0 REGRESSIONS;
  - novelty is held: whole-URL novelty and the share of query strings copied from training each worsen by less than 0.5 points (95% CI);
  - the forward passes per delivered URL are at most 2× the baseline (resampling and blending included).

## 6. Kill

| code | trigger | consequence |
|---|---|---|
| K0 | a validity check fails | fixed under an amendment; never read |
| K1 | no field in a mode passes the validation confirmation | that mode stops **before any generation** |
| K2 | any REGRESSION | not adopted; no post-hoc pruning of settings — a reduced set is a new registration |
| K3 | the cost limit is exceeded | not adopted |
| K4 | novelty or copying worsens | not adopted |

**Not claimed.** Decoding corrects miscalibration only. A cell whose tell is structure E7 never learned may stay failing, and it will be reported as such.

## 7. Order

1. This registration.
2. The sampler extension, with tests (off by default, bit-identical).
3. The fit and its validation confirmation (laptop CPU).
4. The Kaggle notebook and code bundle; the owner runs it.
5. Verification, scoring and the comparison.
6. `DC1_RESULTS.md`.

The owner decides adoption.

## Status

    DC1  registered 2026-09-26, owner-approved
    DC1  scored 2026-09-26: criteria met in both modes; adoption pending the owner

**Implementation note, 2026-09-26 — before any fit or generation.** Section 2's
"the sampler's field tracking is corrected" proved not to be bit-identical in
every sequence.
- The 96-case E7 fixture caught it: when `<PTH>` is sampled before `<SUF>`, an
  ungrammatical order a random test model produces, the legacy tracker stays in
  PTH through the injected label. The frozen path delta then keeps acting there,
  and the corrected tracker stopped that.
- Changed accordingly, with nothing about what DC1 fits or tests:
  - the legacy tracker behind `field_bias` (and so the frozen delta) is
    untouched;
  - DC1 has its **own** tracker, which follows supplied markers;
  - δ_f is applied through DC1's own `field_stop_bias`, not through `field_bias`.
- The fit replicates both trackers. A test proves that the fit's teacher-forced
  log-probabilities equal the sampler's, token for token, with every option on:
  matched settings agree to 2e-6, and mismatched ones are caught.

**Clarification, 2026-09-26 — before any DC1 generation.** Section 5's novelty
clause, "each worsens by less than 0.5 points (95% CI)", is read as SC1's K3:
- a worsening counts only if it is **at least 0.5 points and its 95% CI excludes 0**;
- the stricter reading (the upper bound must be below 0.5 points) would fail on noise
  alone: at zero change the upper bound is 0.86 points for FULL FREE's ~9,400
  query URLs per arm.

**Fit, 2026-09-26 (48.5 min, laptop CPU, peak 1.6 GB):** `data/dc1/settings.json`
sha256 `68850b79f83851cbc49dd14fd91cb20e8de404eacb208a04b3eada151efb5f35`,
fitted by `dc1_fit.py` at 6851edd.
- **K1 passes in both modes.** Settings confirmed on validation:
  - FULL FREE: benign LBL, PTH, QRY; phishing LBL, SUB, PTH. Benign SUB and phishing QRY are off (their CIs cross 0).
  - FULL CONTROL: every field in every state, except phishing P0 Q1 QRY.
- Ten of the eleven target cells carry settings. Free phishing QRY does not, so it can move only through the fields before it.

**Results, 2026-09-26:** `DC1_RESULTS.md`. Registration, thresholds and settings unchanged.
- Generation: Kaggle Tesla T4, 56 files in 279 min. VERIFY PASS (V1–V4).
- FULL FREE, criteria met: 2 IMPROVED (benign QRY, phishing PTH), 1 FIXED within
  noise (benign PTH), 0 regressions, novelty held, cost 1.01×.
- FULL CONTROL, criteria met: 1 IMPROVED (benign P1 Q1 SUB), 0 regressions,
  novelty held, cost 1.94×.
- No query target cleared in either mode. Of the eleven targets: today 0 CLEARED,
  5 UNDECIDED, 6 CONVICTED; DC1 1 (within noise), 5, 5.
