# PC1 — why the plan over-predicts benign paths: pre-registration

- **Registered:** 2026-09-29, before any PC1 code, measurement or reading.
- **Approved by:** the owner, 2026-09-29: "Go ahead and run your registered laptop
  diagnostics on why the plan overshoots benign paths"
  (`DECISION_2026-09-29.md` §1).
- **Laptop only.**
- **What it reads:**
  - `e10_plan`'s `best.pt`, with `final.pt` as a report;
  - the training bundle (`data/bundle2`);
  - `e10_plan`'s training record.
- **What it does not read:**
  - E10_generate's files, whose readout is `PREREG_E10.md`'s;
  - `e10_base`;
  - E8.
- **Frozen, unchanged:**
  - the baseline;
  - E10's registration and its K10a reading;
  - gate v2 and its thresholds;
  - every published result.
- **Diagnostic.** It licenses questions and design inputs for E11, never a fix.

## 0. In plain words

- **The failure.** In free mode the plan says "a path follows" for 77.1% of benign
  URLs (K10a). The data says 73.3%.
- **Why it can be read exactly.** The plan is chosen right after the suffix, so it
  depends only on the class, the frame and the suffix. PC1 reads it suffix by
  suffix and compares it with the very rows the model was trained on.
- **Three candidate reasons, each testable:**
  1. **Clipping.** Short training batches, where most bare hosts sit, had their
     gradients clipped in a third of steps. That makes no-path rows count a little
     less, so the model learns a rate tilted toward paths.
  2. **The frame.** The multi-page frame (97% paths, 65% of benign training rows)
     pulls up the one-URL-per-site frame, which free mode uses.
  3. **Rare suffixes.** Estimates for suffixes with little data drift.
- **Two checks:**
  - what the 9% of plan-slot probability outside the plan tokens is;
  - where the plan's query shortfall sits.

## 1. Known before this registration

| fact | status and source |
|---|---|
| K10a: benign plan P 0.7710 against 0.733, Q 0.0605 against 0.0653; phishing P 0.3983 against 0.394, Q 0.0403 against 0.0451; legal mass 0.9087 / 0.9528 | [CONFIRMED] `data/e10/score_plan.json` |
| top suffixes, plan against FINDING §2.1's data rates: `com` 0.7301 / 0.697, `de` 0.7757 / 0.744, `org` 0.7692 / 0.725, `net` 0.7054 / 0.674 | [CONFIRMED], published before PC1 |
| e10_plan's clipping after step 1,000, by the batch's scored tokens per row: 0–30: 32.7% of steps (mean norm 0.99); 30–60: 0.0%; 60–120: 0.0%; 120+: 1.1%. e10_base: 24.4%, 0.0%, 0.0%, 1.7% | [CONFIRMED] `train_result.json`, `grad_diagnostics` |
| clipping acts on the batch-mean gradient before E9's T_b / C scaling (`uie/train.py` 583–586), so a clipped batch's tokens each weigh less than 1/C | [CONFIRMED] code |
| E9 saw the same asymmetry (`e9_ln` 24.8% in 0–30). It bounded the effect on content decisions (≤ 2.4% past 1) but did not size it for presence | [CONFIRMED] `E9_RESULTS.md` §2 |
| in free mode the plan depends only on class, frame and suffix, and free mode's frame is `domain` | [CONFIRMED] `e8_plan_calibration.py`, `uie/free.py` |
| the shipped table weights suffixes by every training row, both frames; every shipped suffix occurs in `name_train` | [CONFIRMED] `emission_targets.py`, `data/native/emission_targets.json` |
| `e9_ln` (hidden controls) also over-predicts benign paths at real host ends: +3.12 points in the domain frame, +1.41 in the page frame. There the path competes with the query and the end of the URL. Not read here | [CONFIRMED] QG2-D |

The clipping record was seen while preparing this registration. H-CLIP (§2) is
registered because of it, and the prediction in §6 says so.

## 2. Questions and hypotheses

**Q1 (primary).** What makes the benign plan's shipped-weighted P(path) 3.8 points
too high?

- **H-CLIP — length weighting through clipping.**
  - Under E9's objective every scored token should weigh 1/C. A clipped batch's
    tokens weigh ω_b / C instead, with ω_b = min(1, 1 / norm_b).
  - Short batches are clipped and long ones are not. No-path rows are short. So
    the plan converges to a clip-weighted rate that under-counts no-path rows.
  - AdamW's stationary point is where the weighted sum of gradients is zero.
    FINDING §2.1 showed that the plan reaches such a weighted optimum under E8's
    1/L weighting.
- **H-FRAME — frame pull.** The domain frame's plan is pulled toward the page
  frame's, which is the majority of benign exposures and carries 97% paths.
- **H-RARE — suffix smoothing.** The error sits in suffixes with few domain-frame
  rows.

**Q2.** Is the 9% of plan-slot mass outside the legal pair explained?
- **H-LEGAL:** it is the suffix's own continuation, for example `com` → `com.br`,
  which the renormalisation over the legal pair handles correctly.

**Q3 (secondary).** Where does the plan's query shortfall sit? (Phishing 0.893 of
target, benign 0.927.)

## 3. Measurements

**M0 — the rows the model trained on.**
- **Which rows:** every row of the four training streams (`uie.kaggle_train.STREAMS`).
- **How they are encoded:** with `e10_plan`'s tokenizer under the row's corpus
  frame, exactly as `encode_corpus` encodes them. Rows over 256 tokens are dropped,
  as training dropped them.
- **What is recorded per row:**
  - class, frame and suffix;
  - the plan labels (path, query);
  - scored tokens m = len(ids) − 3. The control targets `<C:*>` and `<F:*>` weigh 0,
    and the length check is V4.

**M1 — the plan's distribution.**
- **Where:** each class and frame, over every suffix present in that class's
  training rows of that frame, plus every shipped suffix in the domain frame.
- **What:** the plan distribution as `e8_plan_calibration.plan_by_suffix` computes
  it, generalised to the frame:
  - P̂ = P(`<P:1>`) renormalised over the legal pair;
  - the legal mass;
  - q̂|p0 and q̂|p1;
  - q̂ = (1 − P̂) q̂|p0 + P̂ q̂|p1.
- **Checkpoints:** `best.pt`; `final.pt` as a report.

**M2 — clip factors on `best.pt`.**
- **The batches** are built as `uie.train.batches` builds them from the M0 rows:
  - class drawn by label counts;
  - rows uniform within the class;
  - windows of 64 × 256 rows, sorted by length and chunked into batches of 256;
  - 12 windows, seed 20260929.
- **Per batch:**
  - the gradient norm of the batch's training loss (`model.loss` with the field
    masks), in training mode with dropout on and a fixed torch seed, fp32 on CPU;
  - its scored tokens per row, T_b / rows;
  - ω_b = min(1, 1 / norm).
- **Which batches:**
  - every batch under 30 scored tokens per row;
  - from each longer band, every 8th batch in window order.
- **Bins:**
  - the short band is split into four bins of equal batch count by scored tokens
    per row;
  - the longer bands are [30, 60), [60, 120) and [120, ∞).
- **ω(bin)** is the mean ω_b over the batch sample in the bin.

**M3 — clip-weighted data rates.**

    r_w(c, f, s) = Σ_i ω(bin of m_i) · y_i / Σ_i ω(bin of m_i)

The sums run over the M0 rows of (class, frame, suffix). The same weighting gives
the query rates and the continuation rates.

## 4. Validity

| code | check | on failure |
|---|---|---|
| V1 | K10a reproduced: the domain-frame plan per shipped suffix, shipped-weighted, equals `score_plan.json`'s P, Q and legal mass for both classes within 1e-9 | nothing read |
| V2 | M0 reproduces training's encoding: rows that fit 2,545,305 benign and 238,199 phishing, 10,313 dropped (`train_result.json`) | nothing read |
| V3 | M0 reproduces K10a's targets: shipped-weighted domain-frame P and Q on rows that fit within ±0.0010 of 0.733 / 0.394 and 0.0653 / 0.0451. FINDING §2.3 used E7's lengths; the plan adds 2 tokens per row | nothing read |
| V4 | m = len − 3 equals the field mask's sum on 1,000 random rows of each stream (a test) | nothing read |
| V5 | the clip measurement resembles training: the clipped share under 30 scored tokens per row is within ±15 points of the record's 32.7%, and at most 2% in [30, 60) and [60, 120) | R1 is UNREPRESENTATIVE, not read |
| V6 | the frame-generalised plan function equals `plan_by_suffix` in the domain frame (a test) | nothing read |

## 5. Readings, fixed now

Every reading uses exact numbers. The model side is forward passes. The data side
is counts of the rows the model trained on. No sampling error enters, so no
confidence intervals are needed. The shipped weights w(s) are
`data/bundle2/suffix_table.json`.

**R1 — clipping** (benign, domain frame, shipped weights):

    Δ = Σ w (P̂ − r)      the gap K10a measured, on rows that fit
    C = Σ w (r_w − r)    the part clip weighting predicts

| reading | rule |
|---|---|
| CLIP-LED | C / Δ ≥ 2/3 |
| CLIP-PARTIAL | 1/3 ≤ C / Δ < 2/3 |
| CLIP-MINOR | C / Δ < 1/3 |

**R2 — the frame, on what clipping leaves.**
- The residual is E = Σ w (P̂ − r_w). If |E| < 1.0 point, R2 is **NOT-NEEDED**: a
  residual below the benign path tolerance needs no explanation
  (`DECISION_2026-09-29.md` §2).
- Otherwise, on each frame's own training rows (benign):
  - ε_f = logit(Σ n P̂ / N) − logit(Σ n r_w / N);
  - n is the frame's rows per suffix and N their total.

| reading | rule |
|---|---|
| FRAME-PULL | ε_domain > 0 and ε_page ≤ ε_domain / 3 |
| OFFSET | ε_page ≥ (2/3) ε_domain > 0 |
| OTHER | anything else |

**R3 — where** (benign, the plain gap Δ).
- Suffixes are banded by their benign domain-frame rows that fit: common
  ≥ 10,000; middle 1,000–9,999; rare < 1,000.
- **COMMON-LED** if the common band carries ≥ 50% of Δ; otherwise **RARE-LED** if
  the rare band does; otherwise **SPREAD**.

**R4 — the off-slot mass** (each class, domain frame, shipped weights).
- The data's continuation rate cont(s) is the share of rows whose suffix starts
  with s and is longer, among rows whose suffix starts with s.
- **LEGAL-EXPLAINED** if the weighted mean of |legal mass(s) − (1 − cont(s))| is
  at most 0.02; otherwise **LEGAL-DISCREPANT**.

**R5 — the query** (each class, domain frame, shipped weights). The shortfall
splits exactly:

    Δ_Q = Σ w (q̂ − r_Q) = M + K0 + K1
    M  = Σ w (P̂ − r) (r_Q|1 − r_Q|0)      the path mix
    K0 = Σ w (1 − P̂) (q̂|p0 − r_Q|0)       queries without a path
    K1 = Σ w P̂ (q̂|p1 − r_Q|1)             queries after a path

- A term carrying ≥ 2/3 of Δ_Q names the reading: **Q-MIX**, **Q-NOPATH** or
  **Q-PATH**. Otherwise **Q-SPLIT**.
- C_Q = Σ w (r_w,Q − r_Q) is reported.
- An empty conditional cell (a suffix with no path rows, or no no-path rows) takes
  the class-and-frame pooled conditional rate.

**Reported, not readings:**
- phishing's R1–R3 quantities;
- `final.pt` against `best.pt`;
- the top 20 shipped suffixes, both frames;
- the share of each class's no-path and path rows under 30 scored tokens.

## 6. Predictions, before any run [HYPOTHESIS]

- **V1–V6 pass.**
- **R1: CLIP-PARTIAL.**
  - My arithmetic: suppose about 90% of benign bare-host rows and 25% of path rows
    sit under 30 scored tokens, and a short batch's mean weight is about 0.87.
    Then the tilt is about +0.1 in log-odds.
  - The published top suffixes are off by +0.15 to +0.23.
  - This prediction is informed by the clipping record.
- **R2:** no confident prediction.
- **R3: COMMON-LED,** informed by the published top-suffix records.
- **R4: LEGAL-EXPLAINED.**
- **R5: Q-PATH or Q-SPLIT, not Q-MIX.**
  - C_Q > 0: clip weighting pushes queries up, since query rows are long, so it
    cannot explain a query shortfall.
- **Phishing:** its clip shift is smaller than benign's, because phishing bare
  hosts are longer.
- **`final.pt` against `best.pt`:** the plan P differs by under 0.5 points.

## 7. What each reading licenses

Each is an input to E11's registration (`E11_ROADMAP.md`). None is a fix, and
all are the owner's decisions.

| reading | licenses |
|---|---|
| CLIP-LED / CLIP-PARTIAL | An objective-level repair of the clipping interaction becomes a candidate for every E11 arm, hidden or plan, since all share the objective. The candidate: a clip that no longer binds by length after warm-up, checked by the new run's clipped share per band. Whether `e9_ln`'s own +2.2-point benign path emission shares the cause is not claimed; E11 can test it |
| CLIP-MINOR | clipping is not the explanation; no objective change is licensed by PC1 |
| FRAME-PULL | the frame's share of exposures enters E11's design, or a plan supplied from the domain-frame data (the owner's ruling on "native") |
| OFFSET | an unexplained class-wide offset; no E11 lever licensed; reported |
| RARE-LED | per-suffix smoothing: a supplied plan, or acceptance where it is inside the tolerance |
| LEGAL-DISCREPANT | the renormalised plan is not the model's conditional; free-mode plan sampling is reviewed before any adoption of the plan |
| R5 | names which half of the plan's query decision falls short; E11's query criteria use it |

## 8. Not claimed

- anything about `e10_base`, `e9_ln` or E10_generate's files;
- anything about content cells;
- a fix, and any statement for another training: one checkpoint is read.

## 9. Cost and order

1. This registration.
2. Code and tests (`research/native/pc1_measure.py`, `pc1_read.py`), committed
   before any measurement.
3. The measurement, on the laptop:
   - data about 5 minutes;
   - model about 1 minute;
   - clip factors about 20 minutes.
4. The reading, then `PC1_RESULTS.md`, with its appendix pasted by script.

## Amendment 1 — V6 (2026-09-29, after the model pass, before any reading)

- **What happened.** V6 failed. The largest difference between the
  frame-generalised plan and `plan_by_suffix` was 0.023 (benign) and 0.234
  (phishing). V1 passed: K10a reproduces exactly.
- **The cause** [CONFIRMED].
  - `plan_by_suffix` builds its prefix by encoding the probe URL `x.{s}/`.
  - For three suffixes, the public-suffix list treats the probe's own host as a
    suffix:
    - `x.bg` and `x.se`: single-letter second levels;
    - `x.sapporo.jp`: the rule `*.sapporo.jp`.
  - The probe then encodes the suffix `x.bg`, not `bg`.
  - Affected: benign `bg`, `se` and `sapporo.jp`, 1.71% of the shipped weight;
    phishing `bg` and `se`, 0.05%.
  - On every other shipped suffix the two functions agree exactly (difference 0.0).
- **Which prefix is right.** PC1's `plan_prefix` builds the prefix a training row
  with suffix s carries. `tests/test_pc1.py` checks it against encoded rows. For
  the three suffixes, it is the probe that reads another suffix's plan.
- **K10a's quantities with the training prefix** [CONFIRMED]. Every difference is
  under 0.0001, so K10a's reading (FAIL: benign P +0.038, phishing Q 0.893) is
  unchanged. K10a stands as registered; this corrects nothing in it.

  | class | quantity | registered | with the training prefix |
  |---|---|---|---|
  | benign | P | 0.771033 | 0.771018 |
  | benign | Q | 0.060509 | 0.060477 |
  | phishing | P | 0.398309 | 0.398347 |
  | phishing | Q | 0.040277 | 0.040274 |

- **V6, restated.** The frame-generalised plan must equal `plan_by_suffix` within
  1e-9 on every shipped suffix whose probe parses to the suffix itself. The
  mismatched suffixes are listed with both values.
- **PC1 reads `plan_prefix`'s values throughout,** as M1 registered ("generalised
  to the frame", the prefix built directly).
- **Also touched, not read.** FINDING_LENGTH_WEIGHTING §2.1 used the same probe on
  E8's arm. E8 stays unread.
- `e8_plan_calibration.py` is left unchanged, so the published instruments
  reproduce.

## Amendment 2 — R4's undefined suffixes (2026-09-29, AFTER the reading was printed)

- **What happened.** R4 printed `nan` and so read LEGAL-DISCREPANT. The rule did not
  say what to do with a suffix whose continuation rate is undefined.
- **The cause** [CONFIRMED].
  - Six shipped suffixes each have exactly one domain-frame training row, and that
    row is longer than 256 tokens. So no row that fits carries the suffix.
  - They are benign `edu.ec`, `gob.bo`, `gov.kh`, `puglia.it`, `vt.us` and phishing
    `aws`.
  - Their shipped weight is benign 0.0002%, phishing 0.0004%.
- **The fix.** These suffixes are left out of R4's weighted mean, and their weight is
  reported. No threshold or rule changes.
- **Honesty note.** This amendment was made after the reading was printed and the
  per-suffix values had been seen.
  - Any value for the six suffixes moves R4's mean by at most their weight
    (≤ 0.000004). The reading cannot depend on the choice.
- **Also recorded.** R1, R3 and R5 give these suffixes the pooled domain-frame
  rate, because their own rate is undefined. The effect is bounded by the same
  weight.

## Status

    PC1  registered 2026-09-29 (this document), before any PC1 code
    PC1  instrument 948a3ad (tests 8); data pass: V2 exact, V4 0
    PC1  amendment 1 (V6, the probe's public-suffix artefact), before any reading
    PC1  read 2026-09-29; amendment 2 (R4's undefined suffixes), after the reading
    PC1  results (PC1_RESULTS.md): validity PASS. benign CLIP-LED (0.83 of the
         +3.77-point gap; residual +0.65), R2 NOT-NEEDED, COMMON-LED,
         LEGAL-EXPLAINED, Q-PATH; phishing Q-PATH, LEGAL-EXPLAINED (R1-R3
         reported). Licenses an objective-level clipping repair as an E11
         candidate (E11_ROADMAP.md); the owner decides.
