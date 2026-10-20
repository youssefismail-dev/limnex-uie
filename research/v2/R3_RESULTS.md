# V2-R3 — the frozen model chooses from a word memory, in context: results and recommendation

- **Registration:** `research/v2/PREREG_R3.md` (a61164a), which runs R2's Part B unchanged.
  **Code:** `research/v2/v2_r2.py` (cccf492).
- **Record:**
  - `data/v2/r2/gen/`: 10 files, 16,000 URLs, with their traces;
  - `B_fits.json`, `B_metrics.json`, `B_n.json`, `verify.json`, `r2_result.json`,
    `r3_read.log`.
- **Read once, 2026-10-10,** after VERIFY passed.
- **Cost:** about 2.5 h of CPU generation and 15 min of scoring on the laptop. No GPU; no
  weight changed.

## In plain words

- **When the engine itself picks real words in context, the benign path tell halves.** It
  closes 53%, against 39% for the blind memory: the engine's judgement is the ingredient that
  matters.
- **It is still not enough.** The whole benign URL closes only 35% and stays CONVICTED.
- **And in names it starts copying.** URLs that reuse a real training domain rise from 1.4% to
  3.9%, and exact training URLs from 0.14% to 0.96%. That is the copier you ruled out, caught
  by the guards.
- **Phishing is unchanged,** as expected: by the data's own measure it does not over-invent,
  so the memory almost never acted (57 constrained steps in 8,000 URLs). Its tell is *which*
  real words it uses, which a decode-time memory cannot fix.
- **Verdicts:** benign COSTLY, phishing NO EFFECT. A decode-time memory is not the solution.

## 1. Validity: all PASS

- V-split, V-honest, V-wrap (48 of 48 identical per class), V-floor, and **V-trie: 0 memory
  violations in 16,000 URLs.**
- Free-word shares as registered: benign LBL 0.86, SUB 0.91, PTH 0.46, QRY 0.73;
  phishing 1.0.

## 2. Realism (excess over the floor; mean of 5 generations)

| judge, part, class | `e16_b5` | with memory | closed |
|---|---|---|---|
| **D1b, whole URL, benign** | +0.1330 CONVICTED | **+0.0867 CONVICTED** | 35% |
| **D1b, path/query/fragment, benign** | +0.1340 CONVICTED | **+0.0629 UNDECIDED** | **53%** |
| D1b, host, benign | +0.0405 UNDECIDED | +0.0240 UNDECIDED | 41% |
| D1b, whole URL, phishing | +0.0913 CONVICTED | +0.0757 UNDECIDED | within noise |
| D1b, path/query/fragment, phishing | +0.0339 UNDECIDED | +0.0594 UNDECIDED | within noise |
| D1b, host, phishing | +0.0652 UNDECIDED | +0.0634 UNDECIDED | 3% |
| D0 (the gate's character model), whole URL, benign / phishing | +0.0210 / +0.0095 | +0.0223 / +0.0054 | unchanged, CLEARED |
| D0, paths, benign / phishing | +0.0062 / +0.0109 | +0.0082 / +0.0089 | unchanged |
| D4 (word order), benign / phishing | +0.0244 / +0.0145 | +0.0208 / +0.0120 | unchanged, CLEARED |

- **N is smaller than in R2** (1,000–1,400 benign, 500–1,400 phishing), so excesses read lower
  and verdicts are weaker than at R2's N.
- In phishing the mechanism barely acted, so its differences are noise.
- **The character gate and word order are untouched,** so the memory did not damage what
  already worked.

## 3. Copying, creativity, structure (8,000 URLs per source)

| measure, benign | real | `e16_b5` | with memory |
|---|---|---|---|
| **URL reuses a training registrable domain** | 0.01% | 1.39% | **3.87%** ✗ |
| **exact training URL (memorised)** | 0.03% | 0.14% | **0.96%** ✗ |
| label is a training label | 8.75% | 5.76% | 11.89% |
| path is a training path | 16.3% | 20.0% | 20.2% |
| query is a training query | 23.2% | 23.6% | 24.9% |
| word pairs in paths never seen in training | 64.5% | 79.7% | 79.1% |
| words new to the judge, paths | 10.7% | 23.9% | 18.4% |
| words new to the judge, hosts | 47.0% | 56.3% | 53.6% |
| distinct words per 1,000, hosts | 649 | 730 | 735 |
| path present / query present | 87.5% / 9.3% | 74.0% / 6.6% | 74.2% / 7.0% |

- **The memory raised copying only through names.** Paths, queries and word pairs are
  unchanged, and the engine still invents more than real data does.
- **The names effect is the copier failure mode.** Forcing label words toward real words makes
  the engine regenerate real domains.
- **Phishing:** every copying measure is within noise of base. The guard flags in phishing
  (query presence, validity, path word diversity) are noise or already present in base, since
  the wrapper hardly acted.

## 4. Reading (`PREREG_R2.md` §5)

| class | realism met | failed guards | reading |
|---|---|---|---|
| benign | no (whole URL 35%, not ≥ 70%) | registrable-domain copies, whole-URL novelty; host word diversity (already failing in base) | **COSTLY** |
| phishing | no | none attributable to the mechanism | **NO EFFECT** |

## 5. What R1, R2 and R3 establish together

| | evidence |
|---|---|
| the tell is the **long tail of real vocabulary**, not URL structure and not word order | R1 (bands), R2 diagnosis (vocabulary features 0.657 against structural 0.526), D4 CLEARED |
| word pieces cannot reach the tail | R1: 32k pieces hold 42% of real benign path words, and under 1% of rare ones |
| a blind memory helps paths a little and harms names | R2: 39% on paths; names +0.058 → +0.202 |
| **the engine's own judgement in context is the key ingredient** | R3: 53% on paths, against 39% blind |
| a bolt-on memory cannot calibrate *which* words the model prefers | phishing untouched (its tell is too many common words) |
| a bolt-on memory leaks copying where real data coins | R3: domain copies 1.4% → 3.9% |
| the word test alone rewards copying | R2: a pure copier CLEARS benign paths |

## 6. Recommendation

**Beat the tell inside the model, not at the decoder.** The engine has to *learn* the long
tail of real words — which rare words occur, how often, and where — and must decide
explicitly when to coin a new one.

**The design: a lexicon-aware generator.**

1. **Structure stays as it is.** Fields, the plan, the slash rules, B5, the suffix supply,
   digits and punctuation keep `e16_b5`'s character-level machinery, which already clears the
   gate.
2. **At each word, the model makes one explicit choice: reuse or coin.**
   - **Reuse:** pick a real word from a large lexicon (every training word seen at least
     twice, roughly 400k–900k). Each word's representation is **built from its spelling** by a
     small shared character encoder, so the lexicon can be huge without a parameter per word.
     Rare words still get meaningful representations, and the model learns their frequencies
     by maximum likelihood.
   - **Coin:** spell a new word letter by letter, exactly as today.
   - The model learns *when* to coin from the data itself. R2 measured how strongly that rate
     differs by field: names about 67% coined, benign paths about 11%. So names stay
     inventive and paths reuse.
3. **Why this answers each finding:**
   - it reaches the tail (pieces cannot);
   - it keeps the engine's judgement in context (R3's working ingredient);
   - it calibrates frequencies, so phishing's over-use of common words is learned away;
   - coinage stays a learned, explicit act, so names are not forced into dictionary words
     (R3's copying leak).
4. **Guards, every one now shown to be needed:**
   - the honest split judge (D1b);
   - registrable-domain and whole-URL copying (they caught R3's leak);
   - word-pair novelty;
   - the copier control;
   - the character gate and D4.

**What I do not recommend:**
- **Adopting the decode-time memory:** COSTLY, and the bar is not met.
- **P1 as registered** (BPE arms A and B): R1 and R2 show pieces cannot reach the tail.
- **Training against the word detector, or filtering:** rejected in `REVIEW_V2.md`; R2's
  copier result shows why.

**What this needs.**
- **This is a model change, so it needs GPU training.**
- **Next, laptop only, if you agree:**
  - design and register it as the new P1, replacing the BPE arms;
  - build and test the lexicon and the word encoder;
  - measure its size and cost.
- **Nothing reaches Kaggle before your approval.**
- **The alternative you reserved, a pretrained model's word knowledge, remains the fallback**
  if this design falls short.

## 7. Predictions against outcome (`PREREG_R3.md` §2)

| prediction | outcome |
|---|---|
| benign path part falls 30–60% (point ~45%) | ✓ 53% |
| benign whole URL falls 20–45%; MEETS unlikely | ✓ 35%; not met |
| phishing within noise (~80%) | ✓ |
| path copies within T of the natural rate (~70%) | ✓ unchanged (20.2% against base 20.0%) |
| word-pair novelty within T of real (~75%) | ✓ (it stays above real) |
| benign: PARTIAL ~50%, COSTLY ~20% | COSTLY, through names, which I did not foresee |
