# V2-R3 — the model chooses from a word memory, in context: registration

- **Registered 2026-10-09, after R2's Part A and before any R3 generation.** Laptop only: no
  training, no GPU. `e16_b5`'s weights are read, never changed.
- **The test is R2's Part B** (`PREREG_R2.md` §4–§5), unchanged: the same mechanism, judge,
  readings, success criteria and validity checks. Its code is `research/v2/v2_r2.py` (cccf492),
  and V-wrap has already passed.
- **Why it has its own registration.** R2's rule said Part B runs only if the realistic memory
  oracle (O2e) closed at least half the tell. It closed 39% in benign paths, so by R2's rule
  Part B did not run, and that rule stands. R3 is a new decision, taken for the reason below,
  and does not override it.

## 0. In plain words

- **R2's Part A filled each invented word with a real rare word picked by a dice roll,**
  ignoring the surrounding URL.
- **The diagnosis afterwards** (`research/v2/v2_r2_diag.py`, post-hoc) showed two things:
  - The dice favoured one-off words. About half of the replacements are words the judge has
    never seen either, so the oracle under-states what a memory can do.
  - The remaining tell is about *which* real words appear: the engine is short of rare but
    recurring real words in every class and field.
- **Neither point says what happens when the engine itself chooses** the real word that
  fits its context. R3 measures exactly that, on the frozen model.

## 1. What carries over from R2, unchanged

| | where |
|---|---|
| the honest judge D1b (Vocab_B) and the honesty rule | `PREREG_R2.md` §2 |
| the mechanism: the lexicon wrapper, the free-word rate e_f = min(1, r_f / u_f) from training data, the −10,000 penalty | §4 |
| the generation: free mode, E12's seeds, 5 × 1,600 URLs per class | §4 |
| the readings: D1b (whole URL, path part, host), D1, D0 (whole URL, paths), D4; copying, near misses, word-pair novelty, diversity, structure, the wrapper's record | §4 |
| the success criteria: MEETS, PARTIAL, COSTLY, NO EFFECT | §5 |
| the validity checks: V-split, V-honest, V-wrap (PASS), V-floor, V-trie | §7 |

The free-word rates fixed by Part A (`data/v2/r2/oracle.json`):
- benign: LBL 0.881, SUB 0.920, PTH 0.460, QRY 0.732, FRG 0.326;
- phishing: 1.0 everywhere except FRG 0.559.

So in phishing the wrapper almost never acts. By the data's own measure, phishing does not
over-invent, and phishing serves as the check that the mechanism does no harm where it is
not needed.

## 2. Predictions [HYPOTHESIS]

| reading | prediction |
|---|---|
| benign D1b on paths, queries and fragments | falls by 30–60% from base (point guess ~45%, against O2e's 39%) |
| benign whole-URL D1b | falls by 20–45%; MEETS unlikely (~10%) |
| phishing | within noise of base on every reading (~80%) |
| path copies | rise but stay within T of the natural rate (~70%) |
| word-pair novelty | stays within T of real paths (~75%) |
| benign reading | PARTIAL ~50%, COSTLY ~20%, MEETS ~10%, NO EFFECT ~20% |

## 3. Cost

About 2 hours of CPU generation in the background (the two classes in parallel), then about 1
hour of scoring. No GPU.

## Status

- **2026-10-09:** registered, after R2's Part A, before any R3 generation.
- **2026-10-10:** generated (16,000 URLs, 0 memory violations), VERIFY PASS, read once: `R3_RESULTS.md`. Benign COSTLY (path tell −53%, whole URL −35%, domain copies 1.4% → 3.9%); phishing NO EFFECT.
