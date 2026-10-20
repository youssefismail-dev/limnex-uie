# V2-R2 — can a vocabulary memory beat the word tell without copying? Registration

- **Registered 2026-10-09, before any R2 code.** Laptop only: no training, no GPU. `e16_b5`'s
  weights are read, never changed. `uie/` is untouched, and E3 stays closed.
- **Why:** R1 found the word tell is a *rare-word* shortfall. The engine reproduces common
  words but invents where real URLs use rare real words, and word pieces hold almost none of
  those words (`R1_RESULTS.md`).
- **The owner's direction (2026-10-09, verbatim in §1):** the lead engineer chooses the next
  step. The engine must stay creative and must not become a copier.

## 0. In plain words

- **The question:** if the engine had a memory of real words — and used its own judgement
  to choose among them, while still inventing new words as often as real URLs do — would the
  word giveaway go away, without the engine starting to copy?
- **An honest judge first.**
  - Any memory of real words comes from the training data, and so does the word detector's
    vocabulary. A memory could therefore pass the detector by copying.
  - So the training data is split in two by domain. **Memories may draw only from half A;
    the judge counts a word as real only if it appears in half B.**
  - A fix passes only by using words the way the web does, not by reusing training strings.
- **Part A, ceilings on existing files (no generation).** Take `e16_b5`'s output and:
  - fix its spelling slips;
  - replace its invented words with real rare words from half A;
  - replace only the *excess* invented words, keeping invention at the rate the data shows;
  - and, as a warning control, paste in real training paths: a pure copier.
- **Part B, the real mechanism (only if Part A says it can work).** `e16_b5`'s frozen weights
  generate as before, but each word must come from half A's vocabulary, except for a share of
  words left free for invention. That share is measured from training data, never from the
  judge. The model still chooses every word.

## 1. The owner's message, verbatim (2026-10-09)

> I have read your analysis and the findings regarding the long-tail vocabulary shortfall and
> the limitations of BPE. I understand that simply spelling common words correctly won't
> solve the issue if the engine is forced to invent pseudo-words because it lacks memory for
> the rare ones.
> I don't want to make a premature architectural choice, nor do I want to restrict you to
> only the two ceiling tests you proposed. I am putting the decision entirely in your hands
> as the lead engineer. You decide the best next step.
> If running those ceiling tests helps you measure what works, do it. If you have a
> completely different architectural idea that solves the root cause, explore it.
> My only strict condition for the final solution:
> Whatever path or lever we choose, the engine must remain genuinely creative and
> intelligent. It must learn the patterns of real URLs and generate highly realistic, novel
> outputs. I absolutely do not want a solution that forces the engine to become a 'dumb
> copier' that just memorizes and regurgitates training data to pass the word test. Novelty
> and structural intelligence are non-negotiable.
> Take the wheel. Run whatever local, offline tests you need to gather evidence, and report
> back with hard numbers and your technical recommendation on how we actually beat this
> vocabulary tell while keeping the engine creative. Keep e16_b5 frozen and use no Kaggle GPU
> for now

## 2. The honest judge

- **The split.** Every training row of a class goes to half A or half B by its registrable
  domain (blake2b of the lower-cased domain; even → A, odd → B). A site's pages stay
  together.
- **Vocabularies.** Vocab_A and Vocab_B count D1's words (`[a-z]{3,}`, lower-cased) over each
  half's training URLs, per class.
- **The judge, D1b:** E18's D1 detector, its features and protocol unchanged, with **Vocab_B**
  as its vocabulary. It reads like a gate cell: excess over its own real-against-real floor,
  5 generations, t against ±2.776.
- **The rule:** nothing tested in R2 may see Vocab_B. Its words come from the model or from
  Vocab_A.
- **The data-derived invention rate, per field f** (training data only):
  - r_f = the share of half B's field-f words that are not in Vocab_A, i.e. how often new
    data brings a word the memory does not hold;
  - u_f = the share of `e16_b5`'s own field-f words not in Vocab_A, i.e. how often the
    engine's words fall outside the memory;
  - the engine keeps a fake with probability min(1, r_f / u_f).

## 3. Part A — ceilings, on existing files

**The samples** (no new generation): R0's path, query and fragment samples (host masked; N
5,300 benign, 2,400 phishing), R0's host samples and E18's whole-URL samples (N 6,000). On
`e16_b5`'s side, a **fake** is a word not in Vocab_A.

| source | what is done to `e16_b5`'s words | what it measures |
|---|---|---|
| **base** | nothing | the tell under the honest judge |
| **O1 slips** | a fake one edit from a Vocab_A word seen ≥ 10 times becomes that word (the most frequent one) | the most that fixing spelling can give (what word pieces aim at) |
| **O2 memory** | every fake becomes a real rare word: drawn from Vocab_A's words beyond its 8,000 commonest, in proportion to their counts | the most a rare-word memory can give |
| **O2e memory + invention** | each fake is kept with probability min(1, r_f / u_f); the rest are replaced as in O2 | a memory that still invents at the data's rate: the realistic ceiling |
| **O4 copier** (paths only) | each path, query and fragment is replaced by a random one from half A's training URLs | a warning control: what pure copying does to the judge |

- The replacement keeps the original word's case pattern (lower, upper or title).
- **Seeds:** R2_SEED = 20261040. Oracle draws use R2_SEED + 10,000 × oracle + 100 × class +
  g, with oracles numbered O1 1, O2 2, O2e 3, O4 4.
- **Readings:**
  - D1b on the path, query and fragment ("D1b-R"), on the host ("D1b-H") and on the whole
    URL. In each case: base, O1, O2, O2e, and O4 (on the path part only).
  - **The share of the base excess closed:** (x_base − x_oracle) / x_base.
  - **D4** (word order, E18 protocol, `research/v2/v2_r1.py`) on the path part for base, O2e
    and O4, to check that substitution does not create an order tell.
  - **Copying:** for O4 and O2e, the share of paths found exactly in training.
- **Decision rule for Part B (fixed now):** Part B runs if, in either class, O2e closes at
  least half of the base excess on D1b-R or on D1b-H.

## 4. Part B — the mechanism, on `e16_b5`'s frozen weights

- **The engine:** `e16_cap`'s `final.pt`, with `e16_b5`'s decode (raw, T 1.0, + D1 + D1′ +
  B5), plus a **lexicon wrapper** around the model. The weights are not changed.
- **The wrapper's rule, inside LBL, SUB, PTH, QRY and FRG:**
  - **Where a word starts,** a deterministic draw decides whether that word is free. The draw
    is a hash of the row's tokens so far. The word is free with probability e_f = min(1, r_f /
    u_f).
  - **A free word** is the model's own, unconstrained.
  - **Any other word** must stay inside Vocab_A. Each next letter must keep the word a prefix
    of some Vocab_A word, both letter cases allowed. The word may end only once it is a
    complete Vocab_A word, or while it is under 3 letters.
  - Disallowed tokens get a logit of −10,000, not −∞, so D1's and B5's redistributions stay
    finite.
  - Among the allowed words, the model's own probabilities decide.
- **Generation:** free mode, `e17_common.free_urls` at E12's seeds, 5 generations × 1,600 URLs
  per class.
- **Readings,** against the real pools, with `e16_b5`'s published free files sub-sampled to
  the same N as the base:
  - **realism:** D1b on the whole URL, D1b-R, D1b-H; D1 (E18's vocabulary) for reference only
    (the memory's words are all in it); D0 (the gate's own character model) on the whole URL
    and on paths; D4;
  - **novelty and copying:**
    - whole-URL novelty (the sample gate's MEMORIZED);
    - exact copies per field (LBL, SUB, PTH, QRY) and of the registrable domain, against the
      natural rate (R1-6's method);
    - near misses;
    - **word-pair novelty:** the share of the generated paths' neighbouring word pairs that
      never occur in training, against real held-out paths, which catches patchwork copying;
  - **diversity:** distinct words per 1,000 at an equal n; duplicates within a file;
  - **structure and control:** path and query presence; path length in characters and
    segments; the sample gate's validity failures;
  - **what the memory did:** the share of words left free, and the mass the wrapper removed
    per constrained word.

## 5. What counts as success (fixed now)

The mechanism **MEETS** the bar in a class if:
1. **Realism:** D1b on the whole URL is CLEARED, or falls by at least 70% from base; and D0
   on paths does not rise by more than 0.0141 (the character gate's allowance).
2. **Not a copier:**
   - exact copies of the path and of the registrable domain do not exceed the natural rate,
     or base's rate where base is already above it, by more than T;
   - whole-URL novelty does not fall below base by more than T;
   - word-pair novelty stays within T of real held-out paths, or above them.
3. **Still creative:**
   - distinct words per 1,000 stay within ±5% of real;
   - never-seen words (by the judge's vocabulary) do not fall below the real rate by more
     than T. The engine must still invent.
4. **Structure intact:** path and query presence within T of base, and validity failures not
   above base + T.

T is the programme's detectability tolerance, 2√(p(1−p)/n). The readings are **MEETS**,
**PARTIAL** (realism improves by at least 30% and guards 2–4 hold), **COSTLY** (realism
improves but a guard fails), or **NO EFFECT**.

## 6. Predictions [HYPOTHESIS]

| reading | prediction |
|---|---|
| base under the honest judge | still CONVICTED, close to D1's values (benign D1b-R about +0.18) |
| O1 slips | closes 15–35% of the benign D1b-R excess |
| O2 memory | closes 70–95%; may overshoot (too few new words) |
| O2e memory + invention | closes 55–85% of the benign D1b-R excess; 40–75% of the phishing D1b-H excess |
| O4 copier | close to the floor (it passes the judge) while copying 100% of its paths |
| Part B mechanism | closes 40–70% of the benign D1b-R excess; path copies rise but stay near the natural rate; word-pair novelty within T of real; D0 unchanged |
| overall | Part B runs (~85%); PARTIAL ~45%, MEETS ~25%, COSTLY ~20%, NO EFFECT ~10% |

## 7. Validity, before anything is read

| code | check |
|---|---|
| V-split | every training row lands in exactly one half; the halves' row counts are recorded |
| V-honest | the code that builds oracles and the wrapper never opens Vocab_B (checked by a test) |
| V-wrap | with every e_f set to 1, the wrapper's output is identical to `e16_b5`'s own on a test slice |
| V-floor | every D1b, D0 and D4 floor within 0.03 of 0.5 |
| V-trie | every constrained word that is generated is a Vocab_A word (counted from the files) |
| V-inputs | every input file hashed |

## 8. Steps, each checked before the next

1. This registration committed.
2. The code (`research/v2/v2_r2.py`) and its tests committed.
3. **vocab:** the split, the two vocabularies, r_f. V-split.
4. **oracle:** Part A's samples. **runA:** the D1b and D4 fits. The Part A reading, and the
   decision for Part B.
5. If Part B runs: **gen** (V-wrap first, then generation in the background, the two classes
   in parallel), **runB**, **metrics**. V-trie.
6. **verify,** then **read** once: `R2_RESULTS.md`, with the technical recommendation.

## 9. Cost

- **Part A:** about 1–1.5 hours on the laptop.
- **Part B:** about 2–3 hours of CPU generation in the background (measured at 1.6 benign and
  2.3 phishing URLs per second), plus about 1 hour of scoring. No GPU.

## Status

- **2026-10-09:** registered, before any R2 code.
