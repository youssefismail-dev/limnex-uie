# Q1 results — the emission deficit has TWO causes, and the model is the larger

**Run:** 2026-09-22, `data/qry/q1_decoder_emission.json`, log `data/qry/q1.log`.
**Frozen:** `best_e5.pt`, delta=0, n=4000 per cell, seed 4242 (null control 14241),
suffix draws paired across decoders. Exit 0.

## Both kill criteria cleared, so the result is interpretable

The `typical` control had to reproduce the canonical rates within +-0.6 points or
the experiment was void. It did, in both classes, on all three fields:

| | QRY canonical | QRY here | PTH canonical | PTH here | SUB canonical | SUB here |
|---|---:|---:|---:|---:|---:|---:|
| benign | 2.49% | 2.30% | 60.69% | 61.62% | 38.10% | 38.50% |
| phishing | 1.30% | 1.20% | 30.78% | 30.50% | 77.31% | 77.70% |

The second criterion — that `greedy` must not emit a non-trivial query rate, or
the four-way decision-structure model is wrong — also held: 0.78% benign and
**0.00%** phishing.

## The ladder

Wilson 95% intervals; `distinct` is distinct URLs out of 4000.

**Benign** (corpus: train 6.361%, val 6.403%)

| decoder | truncation | QRY | 95% CI | PTH | SUB | distinct |
|---|---|---:|---|---:|---:|---:|
| `temperature` t=1.0 | none | **4.25%** | [3.67, 4.92] | 58.30% | 38.58% | 4000 |
| `typical` mass 0.95 | moderate | 2.30% | [1.88, 2.81] | 61.62% | 38.50% | 4000 |
| `top_p` p=0.90 | tighter | 1.82% | [1.45, 2.29] | 62.17% | 39.20% | 4000 |
| `greedy` | total | 0.78% | [0.55, 1.10] | 91.70% | 54.80% | 360 |
| `typical` (null, seed 14241) | moderate | 2.45% | [2.01, 2.98] | 61.05% | 38.67% | 4000 |

**Phishing** (corpus: train 4.294%, val 4.435%)

| decoder | truncation | QRY | 95% CI | PTH | SUB | distinct |
|---|---|---:|---|---:|---:|---:|
| `temperature` t=1.0 | none | **1.95%** | [1.57, 2.43] | 31.18% | 74.80% | 3999 |
| `typical` mass 0.95 | moderate | 1.20% | [0.91, 1.59] | 30.50% | 77.70% | 3998 |
| `top_p` p=0.90 | tighter | 1.12% | [0.84, 1.50] | 29.98% | 78.60% | 3989 |
| `greedy` | total | **0.00%** | [0.00, 0.10] | 12.90% | 89.45% | 215 |
| `typical` (null, seed 14241) | moderate | 1.25% | [0.95, 1.64] | 30.15% | 77.05% | 3994 |

**Null spread** — same decoder, different seed: 0.15 points benign (2.30 vs 2.45),
0.05 points phishing (1.20 vs 1.25). Every effect discussed below is 5x to 14x
that spread.

## The pre-registered verdict

`decode_temperature` at t=1.0 is untruncated ancestral sampling, so its rate **is**
the model's implied marginal.

| | H1 decoder | H2 model | H3 both |
|---|---|---|---|
| registered benign | temp in [5.4, 7.4] | temp < 3.5 | rises, excludes corpus |
| observed benign | 4.25% — **refuted** | 4.25% — **refuted** | **CONFIRMED** |
| registered phishing | temp in [3.4, 5.4] | temp < 2.0 | rises, excludes corpus |
| observed phishing | 1.95% — **refuted** | 1.95% — met on the point estimate | also consistent |

**Decomposition of the gap** (against the validation rate):

| | corpus | model marginal | production | model share | decoder share |
|---|---:|---:|---:|---:|---:|
| benign | 6.403% | 4.25% | 2.30% | 2.15 pts (**52%**) | 1.95 pts (**48%**) |
| phishing | 4.435% | 1.95% | 1.20% | 2.42 pts (**76%**) | 0.75 pts (**24%**) |

**Conclusion.** Both mechanisms are real in both classes. The model's own
under-assignment is the larger and more consistent component — co-equal in benign,
dominant in phishing. **The decoder amplifies a defect it did not create.**

Phishing's point estimate technically satisfies the registered H2 threshold
(1.95 < 2.0), but the interval reaches 2.43 and the rise from `typical` (1.20 ->
1.95) has barely-overlapping intervals against a null spread of 0.05, so the
decoder component there is small but real rather than absent. The honest reading
is H3 in both classes with different proportions, not H3 in one and H2 in the
other.

**This rules out the cheap fix.** Changing the decoder would recover roughly half
the benign deficit and a quarter of the phishing one, and would leave the larger
share — the model's own under-assignment — untouched in both.

## A mechanism claim that was too broad, and the narrower one that survived

Query suppression is monotone in truncation aggressiveness in both classes, with
no exceptions across four decoders spanning a 5.4x range. That part is solid.

The account offered mid-run — that truncation deletes the rare exit from the path
so the model keeps writing path instead — **is benign-only and was over-general.**
Path presence moves in opposite directions under the same intervention:

| | benign PTH | phishing PTH |
|---|---:|---:|
| `temperature` | 58.30% | 31.18% |
| `typical` | 61.62% | 30.50% |
| `top_p` | 62.17% | 29.98% |
| `greedy` | **91.70%** | **12.90%** |

Benign greedy drives paths to 92%; phishing greedy drives them to 13%. The path
response is class-conditional and the story does not transfer.

Two effects hold in **both** classes and are what the ladder actually supports:

1. **QRY falls monotonically with truncation** — reaching exactly 0/4000 under
   greedy in phishing, i.e. the query marker is never the argmax at any decision
   point in that class.
2. **SUB rises monotonically with truncation** — 38.5 -> 54.8 benign,
   77.7 -> 89.5 phishing.

Truncation reliably suppresses the query field and inflates the subdomain field.
Where the displaced mass goes otherwise depends on the class, plausibly because
61% of benign URLs carry a path against 30% of phishing ones, so the fallback
option available at a query decision differs. **That last clause is a hypothesis,
not a measurement.**

## What Q1 does NOT establish

- **Why** the model under-assigns. Q1 measures that it does, not the cause.
- Anything about axis B. No content measurement was taken here, and the emission
  rate is not evidence about query content.
- That fixing either mechanism would help the gate. No intervention is selected.

## Status

    Q1  COMPLETE — H3, model-dominant. Both kill criteria cleared.
    Q2  registered, not yet run
    Q3  registered, not yet run
    Q4  UNBLOCKED by Q1 — and now the phase's central question, because Q1
        established that production queries ARE a truncated subsample
        (2.30% emitted out of a 4.25% model marginal in benign), which is
        exactly the survivorship condition H4 requires.
