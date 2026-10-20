# Q5 — the corpus is tail-rich; the model is failing to learn the tail

**Run:** 2026-09-22. `data/qry/q5_corpus_tail.json`,
`research/qry/q5_corpus_tail.py`. **READ-ONLY**: corpus files and
already-harvested generated queries only. No generation, no model load, no
training, no gate. Nothing modified. Scope **benign**.

## Why this ran before any repair

Q3.1 measured the model's OUTPUT against HELD-OUT data and found long-tail
under-production. That could not distinguish two worlds:

    corpus tail-RICH, model tail-poor -> genuine learning failure; a generator
                                         change is meaningful
    corpus itself tail-POOR           -> model faithfully reproduces training;
                                         NO generator change is warranted

Q1 set the precedent: train and validation query *rates* were verified to agree
(6.361% vs 6.403%) before the emission deficit was called a model defect.

## Method — and the confound it exists to avoid

Populations differ by three orders of magnitude: 906,214 training rows against
476 generated queries. **Every metric requested — TTR, top-k concentration, p90,
parameter count — moves with sample size.** Comparing them raw would measure
corpus size and report it as corpus shape, which is exactly the error that
produced Q3's withdrawn TTR row.

So all five populations are subsampled to a **common 725 parameters / 476
queries**, 30 draws, reported mean ± sd. Corpora are **host-deduped**, matching
`discriminate_with_floor`, so a multi-page site contributing forty near-identical
queries cannot deflate TTR for reasons unrelated to the tail.

Raw pool sizes after dedup: TRAIN name 57,768 queries / 102,558 parameters;
TRAIN structure 6,555 / 13,554; VAL name 7,253 / 12,952; GEN typical 476 / 758;
GEN temperature 507 / 725.

## Results at common n

| metric | TRAIN name | TRAIN structure | VAL name | GEN typical | GEN temperature |
|---|---:|---:|---:|---:|---:|
| name TTR | **0.5853** ±0.01 | **0.6449** ±0.01 | **0.5971** ±0.01 | 0.3626 ±0.00 | 0.4648 ±0.00 |
| top-1 name % | 5.35 ±0.78 | 4.61 ±0.66 | 5.20 ±0.83 | 13.03 ±0.32 | 7.86 ±0.00 |
| top-5 name % | 15.78 ±1.52 | 12.80 ±0.99 | 15.27 ±1.33 | 27.19 ±0.37 | 28.28 ±0.00 |
| top-20 name % | 29.68 ±1.60 | 25.03 ±1.31 | 28.62 ±1.42 | 47.54 ±0.39 | 42.62 ±0.00 |
| value len median | 4.1 ±0.2 | 4.3 ±0.5 | 4.1 ±0.2 | 4.0 ±0.0 | 3.0 ±0.0 |
| value len p90 | **27.0** ±3.2 | **28.0** ±3.2 | **25.0** ±2.7 | 15.0 ±0.0 | 15.0 ±0.0 |
| value len p99 | **110.7** ±27.9 | **116.5** ±24.2 | **100.9** ±19.9 | 40.0 ±0.0 | 56.0 ±0.0 |
| param count mean | 1.767 ±0.09 | 2.057 ±0.10 | 1.759 ±0.07 | 1.592 ±0.00 | 1.426 ±0.01 |
| param count >=3 % | 16.39 ±1.43 | 21.29 ±1.75 | 17.05 ±1.37 | 13.03 ±0.00 | 8.40 ±0.32 |

## Finding 1 — train and validation are indistinguishable

TTR 0.5853 vs 0.5971, top-20 29.68% vs 28.62%, p90 27.0 vs 25.0, param count
1.767 vs 1.759 — every pair within about one sd. The domain-disjoint split is
**distributionally faithful for query content**, so validation is a sound
comparator and the tail deficit is not a split artifact.

## Finding 2 — the corpus is tail-RICH and the model is not

Generated sits far outside both real populations: TTR 0.3626 against training's
0.5853 is roughly **22 sd** below; top-20 concentration 47.54% against 29.68% is
roughly **11 sd** above.

> **VERDICT: the model is failing to learn the tail.** The "faithful
> reproduction of a tail-poor corpus" world is **excluded**. A generator-side
> intervention is warranted in principle.

**The structure corpus is richer still** (TTR 0.6449, top-20 25.03%, param count
2.057, p99 116.5). The model trains on an *even longer* tail than the population
it is scored against and still under-produces it, which removes the last
alternative explanation — it is not that the scored population is unusually
tail-heavy.

## Finding 3 — the two deficits have DIFFERENT mechanisms

Decomposing each against the untruncated model sample:

| | training | model (untruncated) | production | model share | decoder share |
|---|---:|---:|---:|---|---|
| name TTR | 0.5853 | 0.4648 | 0.3626 | **54%** | **46%** |
| value len p90 | 27.0 | 15.0 | 15.0 | **100%** | **0%** |

**Name concentration splits almost exactly like Q1's emission finding** (52% /
48%) — the model concentrates, the decoder concentrates it further.

**The value-length tail is entirely the model.** Truncated and untruncated
sampling both stop at p90 = 15; the decoder contributes nothing. At p99 the
decoder even helps slightly (56 untruncated vs 40 truncated), still against
~101-117 real.

> These are **two distinct defects**, not one. A decoder change would partially
> address name diversity and do **nothing whatsoever** for value length.

> **SCOPE CORRECTION (Q6).** The "100% model / 0% decoder" figure is derived from
> **p90** and holds only there. At p99 the decoder does contribute (model emits
> 56 untruncated, 40 in production, ~24% of the gap). The split is
> percentile-dependent and should have been scoped when written. The broader
> claim — that name concentration and value length have different mechanisms —
> stands.

## Finding 4 — the deficit is purely in the tail

Value-length **medians match** across every population (4.1 / 4.3 / 4.1 / 4.0 /
3.0) while p90 and p99 diverge sharply. The model writes the common case
correctly. This was visible in Q3 at p90; p99 shows it is far more extreme than
p90 suggested — a 2-3x shortfall at the extreme.

## What this does NOT establish

1. **That fixing the tail would pass the gate.** Q3.1's `skeleton` result stands:
   shape alone reproduces the entire tell with all lexical content destroyed, so
   a name-diversity or value-length fix leaves the shape carrier intact.
2. **A cause for the learning failure.** Q5 establishes the corpus contains the
   tail and the model does not reproduce it. **Q6 tested the length budget
   (Q6_RESULTS.md): it deletes the tail selectively — 41-45% of rows with a
   100+ char value — but accounts for only ~31% of the p99 gap and ~18% at p90.
   Truncation is EXCLUDED; rows are deleted, not truncated.** The majority
   remains a genuine failure to learn a tail the loss did see. Sparsity
   (~630 surviving 100+ char examples), capacity and tokenisation are untested.
3. **Anything about phishing.** Benign only.
4. **That name-tail and value-tail failures share a cause.** Finding 3 is now
   evidence they may NOT: their model/decoder splits differ sharply (54/46 vs
   100/0).

## Consequence for the repair plan

`REPAIR_PLAN_DRAFT.md` Site C was the registered precondition and is now
**satisfied**: the corpus is not the problem. The plan's other constraints are
unchanged, and Finding 3 adds one — **any single-site intervention is now known
to be insufficient**, because the two deficits do not share a mechanism.

Still not recommended for implementation without a cause for the learning
failure (item 2 above).
