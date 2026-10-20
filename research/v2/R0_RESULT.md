# V2-P1, R0 — the frozen engines' word tell in the path, query and fragment

- **Registered:** `PREREG_V2_P1.md` §14.10 step 1 and §15.3, committed before it ran (f6d9074).
- **Code:** `research/v2/v2_r0.py`.
- **Record:** `data/v2/r0/` (`r0_result.json`, `D1W.json`, `D1R.json`, `D1H.json`, `meta.json`,
  `samples/`).
- **Read once, 2026-10-09,** after the run finished (14 min on the laptop). Nothing is tuned
  against it.

## In plain words

- **The detector sees the tell in both classes.** `e16_b5`'s words in the path, query and
  fragment are CONVICTED in benign (+0.192) and in phishing (+0.100). Both classes can be
  read, so the early stop does not fire and P1 goes ahead.
- **The bar is far.** To pass, `v2_hyb` must bring the reading below about:
  - benign: 0.036. The tell must shrink by about four-fifths (81%);
  - phishing: 0.041. The tell must shrink by about three-fifths (59%).
- **For scale:** E16's capacity step (width 384 → 512) moved the benign reading by 0.023,
  about a tenth of it, and the phishing reading not measurably (0.004).
- **Where the words go wrong.** In benign paths, queries and fragments, 77% of the engine's
  words are real words, against 92% for real URLs. It writes 0.79 never-seen words per such
  URL, against 0.28: nearly three times as many.
- **Phishing's word tell is mostly in the host:** +0.147 there, against +0.100 in the path,
  query and fragment. V2 leaves the host as it is, so in phishing it works on the smaller
  part.

## 1. Validity

| check | result |
|---|---|
| V-D1 | E18's whole-URL D1, rerun on E18's cached samples: 20 of 20 tasks, largest difference from E18's record 0 → **PASS** |
| V-floor | floors: D1-R 0.4958 (benign), 0.4948 (phishing); D1-H 0.5037, 0.5032. All within 0.03 of 0.5 |
| N_R | benign **5,300**: set by 0.9 × the fewest rest-bearing URLs in an `e16_b5` file (5,966). Phishing **2,400**: set by a quarter of the real pool's rest-bearing URLs (9,793) |
| the samples | real URLs with a path body, query or fragment, per generation: benign 85,323–85,340 of 115,009; phishing 9,793–9,811 of 24,081. In `e16_b5`'s files: benign 5,966–6,072 of 8,000; phishing 3,298–3,385 of 8,000 |

## 2. The readings

Excess over the real-vs-real floor; the mean of 5 generations; t against the 0.05 bar
(±2.776). The clear line is 0.05 − 2.776 × SE: where a source with this SE would have to sit
to be CLEARED.

**D1-R: the path, query and fragment (the host masked)**

| class | source | excess | SE | t | verdict | clear line |
|---|---|---|---|---|---|---|
| benign | `e16_b5` | **+0.1924** | 0.0050 | +28.42 | **CONVICTED** | 0.0361 |
| benign | `e15_new` | +0.2152 | 0.0058 | +28.30 | CONVICTED | 0.0338 |
| phishing | `e16_b5` | **+0.0997** | 0.0032 | +15.73 | **CONVICTED** | 0.0412 |
| phishing | `e15_new` | +0.1035 | 0.0055 | +9.66 | CONVICTED | 0.0346 |

**D1-H: the host alone**

| class | source | excess | SE | t | verdict |
|---|---|---|---|---|---|
| benign | `e16_b5` | +0.0682 | 0.0029 | +6.36 | CONVICTED |
| benign | `e15_new` | +0.0786 | 0.0038 | +7.47 | CONVICTED |
| phishing | `e16_b5` | +0.1469 | 0.0063 | +15.49 | CONVICTED |
| phishing | `e15_new` | +0.1691 | 0.0066 | +18.09 | CONVICTED |

**`e15_new` − `e16_b5`, paired by generation** (the capacity yardstick, `DESIGN_V2.md` §3)

| reading | benign | phishing |
|---|---|---|
| D1-R | +0.0227 (t 8.28) | +0.0039 (t 0.77) |
| D1-H | +0.0105 (t 2.33) | +0.0221 (t 7.66) |

`e15_new` differs from `e16_b5` in width (384 against 512) and in decode (no B5), and it is a
separate training, so this yardstick mixes all three.

## 3. Words in the path, query and fragment (descriptive)

Per URL that has a path, query or fragment; the mean of 5 generations; vocabulary from the
class's real training URLs (E18's).

| class | source | known words | never-seen words per URL |
|---|---|---|---|
| benign | real | **91.9%** | **0.28** |
| benign | `e16_b5` | 77.3% | 0.79 |
| benign | `e15_new` | 75.2% | 0.85 |
| phishing | real | **83.1%** | **0.41** |
| phishing | `e16_b5` | 78.4% | 0.50 |
| phishing | `e15_new` | 78.1% | 0.51 |

- These shares count every word, over the URLs that have a path, query or fragment, across
  all 5 generations.
- E18's 94% → 84% was computed another way, on generation 0. The two sets of figures are not
  comparable number for number, but they point the same way.

## 4. What it means for P1

- **P1 goes ahead.** The detector is sensitive in both classes, so neither class is VOID and
  the early stop (§15.3) does not fire. Building starts on the owner's word.
- **The distance to the bar,** at `e16_b5`'s SE (the real clear line is set by `v2_hyb`'s own
  SE):
  - benign: from +0.192 to below about 0.036, a cut of about 81%;
  - phishing: from +0.100 to below about 0.041, a cut of about 59%.
- **The capacity yardstick cuts both ways.**
  - A pass needs an effect many times larger than E16's capacity step had on this tell
    (benign: 0.023 against the 0.156 needed).
  - If V2 moves the tell far more than that, its extra embedding rows (+16% parameters,
    against E16's +77%) are an unlikely explanation.
- **The odds, revised after R0.** This is my judgement, not a registered prediction; §14.8
  stands as registered.
  - Benign CLEARED: about 10% (registered: ~25%). The benign tell is larger than I assumed.
  - Phishing CLEARED: about 30%.
  - SUCCESS: about 5–10% (registered: ~10–15%).
- **Recorded, not acted on:** in phishing, the host carries more of the word tell than the
  path, query and fragment do (+0.147 against +0.100). V2 does not touch it by design
  (`DESIGN_V2.md` §0), so any later work on host words is the owner's call.
