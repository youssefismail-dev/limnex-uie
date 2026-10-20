# QG3 — results: rarity or the hidden state?

- **Registration:** `PREREG_QG3.md` (8f54c55), before any code. Approved by the
  owner, laptop only.
- **Code:** 04fe96f, committed before any reading.
- **Read:** 2026-09-28, after VR1–VR3 passed.
- **Numbers:** from Appendix A, `qg3_read.py`'s output pasted by script. A ratio
  written in the text is exp of a log value in the tables.
- **A 600-row dry run,** read with loosened thresholds, exercised the code. Its
  output is not this reading.
- **The baseline** (`e9_ln` / raw) is only read, never changed.

## In plain words

- **Rarity is not the reason.**
  - Take outcomes that no presence control governs and that are just as rare per
    position as the query start: 15–99 of them per comparison, mostly characters.
    The model predicts them at 98–100% of their real rate.
  - The query start comes out at about 73% (benign) and 70% (phishing).
- **The hidden control is the reason.** Of the query's shortfall, 97% (benign) and
  98% (phishing) is beyond anything rarity predicts.
  - The registered reading is **STATE-LED**, in both classes.
  - It is the same for the old objective (`e9_base`) and for E7.
- **QG2-D's "rarity pattern" was a confusion of two things.** The rare outcomes in
  it were almost all ones the hidden presence controls govern:
  - the query start;
  - the URL ending right after the host (no path, no query);
  - a fragment right after the host.
  Once rarity and control are separated, rarity carries almost none of the effect.
- **So the question narrows to one:** when the model is not told whether a query
  exists, why does it choose one too rarely? It chooses correctly when told
  (QG2-D S4).

## 1. Validity: PASS

| check | benign | phishing |
|---|---|---|
| VQ1 | the three models verified | the same |
| VR1: the `<QRY>` calibrations reproduce QG2-D's S0 (±0.005) | host 0.541 = 0.541; path 0.806 = 0.806 | 0.592 = 0.592; 0.777 = 0.777 |
| VR2 | events − positions = 0; probabilities within 6.1e-08 | 0; within 1.3e-07 |
| VR3: every target's reference window | ×3, the narrow window, for all three targets | the same |
| pairs read (≥ 200 events) | 146: 2 targets, 1 secondary, 141 uncontrolled, 2 constrained | 184 |

## 2. The split

| | benign | phishing |
|---|---|---|
| L, the query (log) | −0.321 [−0.361, −0.285] | −0.356 [−0.398, −0.313] |
| R, rarity at the same frequency | −0.011 [−0.017, −0.005] | −0.007 [−0.016, 0.003] |
| S, beyond rarity | −0.310 [−0.352, −0.274] | −0.349 [−0.390, −0.305] |
| shares R / S | 0.03 / 0.97 | 0.02 / 0.98 |
| **reading** | **STATE-LED** | **STATE-LED** (agrees) |

**Target by target, `e9_ln`.** The reference is the uncontrolled outcomes within a
factor of 3 of the same per-position frequency.

| class | target | frequency per position | the model's c | reference c (pairs, events) |
|---|---|---|---|---|
| benign | query after the host | 0.118% | 0.541 | 0.981 (19, 22,244) |
| benign | query after the path | 0.218% | 0.806 | 0.992 (41, 75,182) |
| phishing | query after the host | 0.070% | 0.592 | 0.983 (15, 6,777) |
| phishing | query after the path | 0.392% | 0.777 | 1.000 (99, 165,165) |

**The uncontrolled curve.** Characters from 10⁻⁴ to 10⁻¹·⁵ per position come out at
0.95–1.02 of their rate (benign); phishing is the same, apart from one bin holding a
single character. The model's calibration of uncontrolled outcomes does not fall
with rarity.

**The registered limit.**
- No uncontrolled **structural** outcome occurs at the query's frequency, so the
  structural-only reference is empty there. The phishing path target has one pair,
  1.055 on 1,643 events. The reading rests on the combined reference, as
  registered.
- At higher frequencies, the uncontrolled structural pairs come out at 0.935–1.055.
- **Context, not criteria:** E9's uncontrolled length decisions were calibrated
  (F1–F3), and so is `<EOS>` after a path (0.997 / 1.003).

## 3. The secondary and constrained pairs, and the objective

| | benign | phishing |
|---|---|---|
| host `<PTH>` (the P control's subject) | 0.973, reference 1.002 | 0.938, reference 1.003 |
| host `<EOS>` (no path and no query: constrained) | 0.832 | 0.930 |
| path `<EOS>` (no query: constrained, common) | 0.997 | 1.003 |

- **The path decision under the hidden P control** is slightly low too. So are the
  URL endings that the hidden controls govern. The rare ones are the most affected.
- **The three models all read STATE-LED:** `e9_ln`, `e9_base` and E7, in both
  classes. Their S shares are 0.97–0.99.
- **The objective's part** (paired, `e9_ln` − `e9_base`):
  - S changes by +0.013 [0.003, 0.022] (benign): almost nothing;
  - and by +0.157 [0.142, 0.178] (phishing): E9's objective reduced phishing's
    hidden-state deficit, and left most of it.

## 4. The registered predictions (PREREG_QG3 §5), scored

| prediction | result | scored |
|---|---|---|
| STATE-LED, with the uncontrolled reference within [0.9, 1.1] at the query's frequencies | STATE-LED; reference 0.981–1.000 | **right** |
| (against: a tied output with no bias could under-predict every rare token) | rare characters are calibrated | the counter-case did not occur |
| `e9_base` and E7 give the same reading | STATE-LED in both classes | **right** |

## 5. What is and is not claimed

**Claimed** [CONFIRMED by the registered test]:
- In all three models the query start's under-prediction is not a rarity effect.
- About 97–98% of it is specific to deciding while the query control is hidden.
- The under-prediction of rare outcomes seen in QG2-D was the controlled outcomes,
  not rarity.

**Not claimed:**
- **the mechanism inside the hidden state.**
- **a fix.**
- **that characters and structural choices are the same kind of decision.** This is
  the registered limit; the structural outcomes that exist agree with it.

## 6. Where the investigation stands

- **QG1:** the gap is the query-start decision (about 90% / 80%).
- **QG2-D:** the decision under the hidden query control. The training code is
  symmetric, and the model decides correctly when told.
- **QG3:** it is the hidden state, not rarity (97% / 98%).

**What STATE-LED points to** (PREREG_QG3 §3; any step is the owner's decision and a
new registration):
- **how the hidden state is trained.** Free mode's exact slice is 7.3% of benign
  and 3.5% of phishing training exposures (QG2-D A6).
- **making the query decision explicit.** Told, the decision is calibrated
  (QG2-D S4).

**Laptop questions that remain, before any training:**
- whether the hidden state's shortfall tracks how the model's told states behave:
  it leaks 1.2% when told "no query";
- whether it changes along the training history. Only `best.pt` and `final.pt`
  exist.

## Appendix A — the rendered tables

`python research/qgap/qg3_read.py`, verbatim.

### Validity

- benign: VR1 host 0.541 against QG2-D 0.541; path 0.806 against QG2-D 0.806; VR2 events−positions 0, probability 6.1e-08; VR3 windows host <QRY>: x3, path <QRY>: x3, host <PTH>: x3; pairs read 146 {'target': 2, 'secondary': 1, 'uncontrolled': 141, 'constrained': 2, 'other': 0} → **PASS**
- phishing: VR1 host 0.592 against QG2-D 0.592; path 0.777 against QG2-D 0.777; VR2 events−positions 0, probability 1.3e-07; VR3 windows host <QRY>: x3, path <QRY>: x3, host <PTH>: x3; pairs read 184 {'target': 2, 'secondary': 1, 'uncontrolled': 179, 'constrained': 2, 'other': 0} → **PASS**

### The split of the query's shortfall (log scale; negative = under-predicted)

| class | model | L (the query) | R (rarity at matched frequency) | S (beyond rarity) | shares R / S | reading |
|---|---|---|---|---|---|---|
| benign | e9_ln | -0.321 [-0.361, -0.285] | -0.011 [-0.017, -0.005] | -0.310 [-0.352, -0.274] | 0.03 / 0.97 | **STATE-LED** |
| benign | e9_base | -0.324 [-0.362, -0.290] | -0.002 [-0.009, 0.004] | -0.322 [-0.361, -0.286] | 0.01 / 0.99 | **STATE-LED** |
| benign | e7 | -0.364 [-0.402, -0.328] | -0.012 [-0.019, -0.006] | -0.352 [-0.391, -0.314] | 0.03 / 0.97 | **STATE-LED** |
| phishing | e9_ln | -0.356 [-0.398, -0.313] | -0.007 [-0.016, 0.003] | -0.349 [-0.390, -0.305] | 0.02 / 0.98 | **STATE-LED** |
| phishing | e9_base | -0.523 [-0.568, -0.477] | -0.018 [-0.023, -0.003] | -0.506 [-0.553, -0.465] | 0.03 / 0.97 | **STATE-LED** |
| phishing | e7 | -0.654 [-0.700, -0.609] | -0.012 [-0.017, 0.003] | -0.642 [-0.689, -0.603] | 0.02 / 0.98 | **STATE-LED** |

### The targets, and the uncontrolled outcomes at the same frequency

| class | model | pair | per-position frequency | c (model / data) | uncontrolled reference c (window, pairs, events) | structural-only reference c (pairs, events) |
|---|---|---|---|---|---|---|
| benign | e9_ln | host <QRY> | 0.118% | 0.541 | 0.981 (x3, 19, 22,244) | — (0, 0) |
| benign | e9_ln | path <QRY> | 0.218% | 0.806 | 0.992 (x3, 41, 75,182) | — (0, 0) |
| benign | e9_ln | host <PTH> | 5.113% | 0.973 | 1.002 (x3, 63, 1,184,299) | 0.948 (2, 19,309) |
| benign | e9_base | host <QRY> | 0.118% | 0.645 | 0.993 (x3, 19, 22,244) | — (0, 0) |
| benign | e9_base | path <QRY> | 0.218% | 0.753 | 1.000 (x3, 41, 75,182) | — (0, 0) |
| benign | e9_base | host <PTH> | 5.113% | 0.826 | 0.996 (x3, 63, 1,184,299) | 0.932 (2, 19,309) |
| benign | e7 | host <QRY> | 0.118% | 0.537 | 0.978 (x3, 19, 22,244) | — (0, 0) |
| benign | e7 | path <QRY> | 0.218% | 0.762 | 0.991 (x3, 41, 75,182) | — (0, 0) |
| benign | e7 | host <PTH> | 5.113% | 0.852 | 0.996 (x3, 63, 1,184,299) | 0.936 (2, 19,309) |
| phishing | e9_ln | host <QRY> | 0.070% | 0.592 | 0.983 (x3, 15, 6,777) | — (0, 0) |
| phishing | e9_ln | path <QRY> | 0.392% | 0.777 | 1.000 (x3, 99, 165,165) | 1.055 (1, 1,643) |
| phishing | e9_ln | host <PTH> | 1.696% | 0.938 | 1.003 (x3, 130, 985,689) | 0.998 (2, 39,680) |
| phishing | e9_base | host <QRY> | 0.070% | 0.538 | 0.953 (x3, 15, 6,777) | — (0, 0) |
| phishing | e9_base | path <QRY> | 0.392% | 0.629 | 1.002 (x3, 99, 165,165) | 1.332 (1, 1,643) |
| phishing | e9_base | host <PTH> | 1.696% | 0.878 | 1.000 (x3, 130, 985,689) | 1.004 (2, 39,680) |
| phishing | e7 | host <QRY> | 0.070% | 0.470 | 0.975 (x3, 15, 6,777) | — (0, 0) |
| phishing | e7 | path <QRY> | 0.392% | 0.554 | 0.996 (x3, 99, 165,165) | 1.339 (1, 1,643) |
| phishing | e7 | host <PTH> | 1.696% | 0.889 | 0.999 (x3, 130, 985,689) | 1.001 (2, 39,680) |

### The uncontrolled reference curve (`e9_ln`): geometric-mean c by frequency bin

| class | kind | bin (per-position frequency from) | pairs | events | c |
|---|---|---|---|---|---|
| benign | characters | 1e-4.0 | 5 | 1,189 | 1.019 |
| benign | characters | 1e-3.5 | 4 | 1,508 | 0.954 |
| benign | characters | 1e-3.0 | 14 | 15,306 | 0.979 |
| benign | characters | 1e-2.5 | 34 | 99,624 | 1.001 |
| benign | characters | 1e-2.0 | 53 | 412,005 | 1.009 |
| benign | characters | 1e-1.5 | 29 | 865,209 | 1.001 |
| benign | structural | 1e-2.0 | 1 | 16,789 | 0.935 |
| benign | structural | 1e-1.5 | 1 | 2,520 | 1.044 |
| phishing | characters | 1e-4.0 | 1 | 219 | 1.168 |
| phishing | characters | 1e-3.5 | 1 | 234 | 0.973 |
| phishing | characters | 1e-3.0 | 21 | 11,549 | 0.979 |
| phishing | characters | 1e-2.5 | 68 | 114,269 | 0.999 |
| phishing | characters | 1e-2.0 | 62 | 377,148 | 1.003 |
| phishing | characters | 1e-1.5 | 24 | 639,074 | 1.003 |
| phishing | structural | 1e-2.0 | 1 | 1,643 | 1.055 |
| phishing | structural | 1e-1.5 | 1 | 38,037 | 0.996 |

### Constrained pairs (`e9_ln`; reported only)

| class | pair | frequency | c | events |
|---|---|---|---|---|
| benign | host <eos> | 1.880% | 0.832 | 10,575 |
| benign | path <eos> | 3.164% | 0.997 | 26,902 |
| phishing | host <eos> | 2.609% | 0.930 | 23,839 |
| phishing | path <eos> | 5.424% | 1.003 | 14,414 |

### The objective: `e9_ln` − `e9_base` (paired blocks)

| class | S difference | R difference |
|---|---|---|
| benign | 0.013 [0.003, 0.022] | -0.009 [-0.011, -0.006] |
| phishing | 0.157 [0.142, 0.178] | 0.011 [0.001, 0.013] |

### Reading

- benign, e9_ln (decides): **STATE-LED**
- phishing, e9_ln: STATE-LED
- e9_base / E7 (benign): STATE-LED / STATE-LED
