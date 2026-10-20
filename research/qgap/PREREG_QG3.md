# QG3 — rarity or the hidden state? Registration

- **Registered:** 2026-09-28, before any QG3 code or measurement.
- **Licensed by:** QG2-D's reading, QUERY-STATE (`QG2D_RESULTS.md`, c04749b).
- **Approved by the owner:** "to isolate the rarity effect versus the hidden-state
  effect". Laptop only.
- **Rules:** `QG_PLAN.md`'s. These are diagnostics; the baseline is never changed;
  no reading licenses a fix.

## 0. In plain words

- QG2-D showed that the model under-chooses the query when the query control is
  hidden, and that rare outcomes in general looked under-chosen.
- QG3 asks which it is:
  1. the model under-predicts **anything rare**, or
  2. it under-predicts **outcomes whose own presence control is hidden.**
- The test compares like with like. The query start is compared with outcomes that
  no control governs and that are **equally rare per position**.
- **If those are predicted at the right rate,** the query's shortfall comes from the
  hidden control. **If they are equally short,** rarity explains it.

## 1. Known at registration [CONFIRMED]

- **The output layer is tied to the input embeddings and has no bias**
  (`uie/model.py:145-148`). So no parameter forces a token's total predicted
  probability to match its count. A general rare-token effect is therefore
  possible.
- **QG2-D's host-region calibrations** (S0, benign / phishing):

  | token | benign | phishing |
  |---|---|---|
  | `<SUB>` | 0.935 | 0.996 |
  | `<PRT>` | 0.959 | 0.874 |
  | `<FRG>` | — | 0.161 (few events) |
  | `<EOS>` | 0.832 | 0.930 |
  | `<PTH>` | 0.973 | 0.938 |
  | content | 1.008 | 1.004 |

- **QG2-D's snapshot pattern:** at field ends, the rarer an outcome, the more
  under-predicted. It was not a registered criterion.
- **E9's uncontrolled path and query families** are calibrated (F1 1.000, F2
  1.018, F3 0.997, benign) at common frequencies.

## 2. The measurement: calibration by token, region and control status

**Rows.** QG1's 40,000 domain-frame training rows per class, in free mode's
encoding (frame domain, both controls `?`). They are paired across models.

**Regions.** Each position is assigned by where it sits. It predicts the next
token.
- **host:** from `<LBL>` to the path, or to the first stop;
- **path:** from `<PTH>` to the first stop;
- **query:** from `<QRY>` to `<FRG>` or `<EOS>`;
- **fragment:** from `<FRG>` to `<EOS>`.

**For every (region, token) pair,** over the region's positions:
- **frequency** f = the number of times the token is next ÷ the number of
  positions;
- **calibration** c = Σ p(token) ÷ the number of times it is next.

**Status of a pair:**
- **HIDDEN-CONTROLLED:** the token is the subject of a hidden presence control.
  `<QRY>` in host and path are the **primary targets** (the query gap). `<PTH>` in
  host is a secondary target.
- **UNCONTROLLED:** no control is about the token, and it implies nothing about a
  controlled field. These form the reference:
  - every character, in every region;
  - `<SUB>` and `<PRT>` in host;
  - `<FRG>` and `<EOS>` in query;
  - `<EOS>` in fragment.
- **CONSTRAINED:** reported only.
  - `<EOS>` in host and path: it implies no query, and in host no path.
  - `<FRG>` in host and path: it implies no query.

**A pair is read only if it has at least 200 events**, fixed on the point estimate.

## 3. The test (primary: benign; phishing reported with its agreement)

**The reference at a frequency.** For a target pair with frequency f:
- r(f) is the event-weighted mean of log c over the UNCONTROLLED pairs whose
  frequency lies within a factor of 3 of f.
- It needs at least 5 pairs and at least 1,000 events. Otherwise the window widens
  to a factor of 10, and failing that the target is **not resolved**.

**The split, over the two `<QRY>` targets** (event-weighted):
- **L** = mean log c, the query's shortfall;
- **R** = mean r(f), what rarity alone predicts at the same frequencies;
- **S** = L − R, what remains beyond rarity: the hidden-state effect.
- The shares R / L and S / L sum to 1.

**CIs.** Rows are cut into 400 blocks of 100. The bootstrap resamples blocks, 1,000
draws with seed 20260806. Every quantity, the neighbourhoods included, is recomputed
in each draw.

**Reading,** rows in order, the first that holds:

| reading | condition | points to (the owner decides any next step) |
|---|---|---|
| SPLIT | R's and S's CIs both lie entirely below 0, and neither share is ≥ 0.75 | both rarity and the hidden state |
| RARITY-LED | R / L ≥ 0.75 and R's CI lies entirely below 0 | a general under-prediction of rare outputs: training dynamics, capacity, or the output's form (R2 was rejected and R3 is LOCKED: the owner's) |
| STATE-LED | S / L ≥ 0.75 and S's CI lies entirely below 0 | the hidden-query state specifically: how the hidden state is trained, or making the query decision explicit (a new registration) |
| UNRESOLVED | anything else | reported; the owner decides |

**Secondary, reported but not deciding:**
- `<PTH>` in host;
- a structural-only reference, from uncontrolled structural pairs alone, with its
  power stated;
- the reference curve, the event-weighted mean of c in half-decade frequency bins,
  for characters and structural tokens separately;
- the CONSTRAINED pairs.

**The objective** (as approved): the same L, R, S and curve for `e9_base` and E7,
on the same rows.

## 4. Validity (checked before any number is read)

| code | check |
|---|---|
| VQ1 | every model's fingerprint or registered hash (as in QG1) |
| VR1 | e9_ln's `<QRY>` calibrations in host and path reproduce QG2-D's S0 (0.541 / 0.806 benign; 0.592 / 0.777 phishing) within ±0.005: same rows, same positions |
| VR2 | in every region, the events summed over all tokens equal the region's positions, and the probabilities summed over all tokens equal them within 1e-3 |
| VR3 | the neighbourhood coverage of every target, reported. A target without coverage at a factor of 10 is not resolved |

## 5. Predictions, before any measurement [HYPOTHESIS]

- **STATE-LED, moderate confidence.** At the query's frequencies the uncontrolled
  reference sits near 1 (r within [0.9, 1.1] in ratio terms), for two reasons:
  - content is calibrated in aggregate (QG2-D);
  - E9's uncontrolled families are calibrated.
- **Against it:** a tied output with no bias could under-predict every rare token.
  That would give RARITY-LED.
- **`e9_base` and E7 give the same reading as `e9_ln`.**

## 6. Not claimed

- a fix;
- a mechanism inside the hidden state (that would be the next question);
- anything for another training beyond the three read.

**One limit, stated now.** Characters and structural choices are different kinds of
decision. Uncontrolled structural outcomes are too few at the query's frequency to
form a reference alone, so the structural-only comparison is reported with its
power, and the reading rests on the combined reference.

## Status

    QG3  registered 2026-09-28 (this document), before any QG3 code or measurement
    QG3  code 04fe96f; READ 2026-09-28 (QG3_RESULTS.md): VR1-VR3 PASS; STATE-LED
         in both classes and all three models (S share 0.97-0.99). Any next step
         is the owner's decision and a new registration.
