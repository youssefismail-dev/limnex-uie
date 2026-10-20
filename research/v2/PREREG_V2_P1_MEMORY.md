# V2-P1 — a learned memory layer for rare words: registration

- **Registered:** 2026-10-10, before any P1 code, training or generation exists.
- **The owner's GO:** §1, verbatim. It covers this phase only: training, generation, the
  registered evaluation and the report. Nothing beyond it runs without a new decision.
- **Replaces:** the hybrid-tokenizer pilot in `PREREG_V2_P1.md`. That pilot was never built
  or run, and its registration is kept as it was.
- **Rests on:** `R1_RESULTS.md`, `R2_RESULTS.md`, `R3_RESULTS.md`, and the language check
  reported to the owner on 2026-10-10 (§2).
- **Frozen:** `e16_b5` (its weights, decode and 50 files), every accepted E3 result, and
  every prior criterion. E3 stays closed.
- **One arm:** `p1_mem`. It is `e16_cap` plus one memory layer. Everything else is
  `e16_cap`'s: data, batches, steps, objective, start and decode.

## 0. In plain words

- **The question:** suppose the engine has a large memory inside its own network, learned
  in the same training run from the same data. Does it then write real rare words where it
  now invents them? And does it do so without copying whole URLs, paths, queries or
  domains, and without losing novelty, diversity or structure?
- **What changes: one layer.** It sits in the middle of the network. It is a large table of
  learned vectors (262,144 slots) that the network searches by content.
  - At every letter, the network asks "what do I remember about contexts like this one?"
    It gets back a blend of the 128 slots that match best.
  - **No slot is a word.** The slots are learned numbers. There is no word list, no lookup
    of training strings and no template.
  - The engine still writes every URL letter by letter, and it can still invent.
- **What does not change:** the training data, the batches and their order, the 20,000
  steps, the learning rate and the objective. Every existing weight starts where
  `e16_cap`'s started. The decode, the 50 generation jobs and every instrument stay the same.
  So the difference from `e16_b5` is the memory, plus ordinary retraining noise.
- **The pass mark is your gold standard, with the honest judge.** The word judge D1b, on the
  path, query and fragment, must be CLEARED (proven below 0.05) in every class where it
  convicts `e16_b5`.
- **Passing the judge alone is not success.** Every guard must also hold:
  - copying of whole URLs, paths, queries and registrable domains;
  - the judge's one known blind spot (new: W1, §4.2);
  - novelty, invention and diversity;
  - structure: the 32 gate cells, the 8 slash rows, presence, plan honouring and word order;
  - fit, and every validity check.
- **Stops, fixed now (§6):**
  - before training, if the start or the memory check fails;
  - during training, on a non-finite loss;
  - after training, if the memory is not used (no generation is spent on it);
  - on any failed validity check;
  - after the reading: no retry and no tuning.
- **Cost:** about 7–7.5 GPU-hours on a T4, about 6–8 laptop hours, about 2–3 days.
- **My odds, honestly:** a full pass about 5%; a clear reduction in the word tell about 35%.

## 1. The owner's decision, verbatim (2026-10-10)

> I appreciate the objective data on English/generic domains. The vocabulary tell appears to
> be a structural generative issue, not merely a language-distribution mismatch.
>
> Your proposed Memory Layer approach is the architectural direction I want to pursue. It
> should preserve letter-by-letter generation, avoid explicit lookup dictionaries and
> template matching, and allow the network to learn useful representations of rare words
> internally.
>
> We will skip the tiny laptop experiment. I have sufficient GPU credits, and I want to
> evaluate the approach within the full architecture rather than rely on a scaled-down
> model that may not reproduce its learning dynamics.
>
> You have my GO to proceed directly to the GPU phase for the full Memory Layer
> implementation, subject to these mandatory conditions:
>
> 1. No external lookup dictionaries: Do not introduce discrete word lists, lexicon-based
>    output selection, template matching, or external pretrained language models. The memory
>    must be learned through training within our own architecture.
> 2. Strict anti-copying guards: Retain all existing copying guards, the copier control, and
>    the honest D1 judge. The expanded memory must not become a mechanism for memorizing and
>    regurgitating complete URLs, paths, queries, or registrable domains.
> 3. Frozen baseline: Preserve "e16_b5" and all accepted E3 results. Do not modify the
>    baseline or retroactively change prior evaluation criteria.
> 4. New P1 preregistration: Design and formally log a new P1 preregistration specifically
>    for the Memory Layer architecture before examining the new model's evaluation results.
>    Define the hypothesis, controls, metrics, sample sizes, pass/fail criteria, copying
>    thresholds, and stop conditions in advance.
> 5. Evidence-based evaluation: Evaluate both the reduction in the word-vocabulary tell and
>    the preservation of novelty, diversity, structural realism, and anti-copying
>    performance. Passing D1 alone is not sufficient to declare success.
> 6. Execution and reporting: Once the design and preregistration are formally recorded,
>    proceed directly to GPU training without the tiny laptop experiment. Report the actual
>    training configuration, resource usage, results, guard checks, and deviations from the
>    preregistration.
>
> Do not substitute the lexicon-aware approach or a pretrained model. The Memory Layer is
> the direction to implement and test.
>
> My approval is for this defined research phase, not for unregistered architecture changes
> or uncontrolled follow-up experiments. Stop after the preregistered evaluation and report
> the results before proceeding further.

**How each condition is met:**

| condition | where |
|---|---|
| 1. no list, no lookup, no template, no pretrained model; a memory learned in our own architecture | §3.1: the memory holds learned vectors only, trained by the engine's own loss in the same run; generation stays letter by letter |
| 2. every copying guard, the copier control, the honest judge; no regurgitation of complete URLs, paths, queries or domains | §4.1 (D1b unchanged), §4.3 (the copier control, rerun), §5.4 (G7, G8, G-COPY on each complete unit, W1) |
| 3. `e16_b5` and E3 frozen; no criterion changed | §11; `e16_b5`'s numbers are recomputed only as reproduction checks |
| 4. this registration, before any P1 result | this document, committed and pushed before any P1 code |
| 5. reduction and preservation, both required | §5: K1 and K2 for the tell; §5.4's guards for preservation; §5.5's reading needs both |
| 6. straight to the GPU; full reporting | §10; §9 (cost); the results report covers the configuration, resources, results, guards and deviations |

## 2. Why this design

**What the evidence says the tell is:**
- **A shortage of real words that are rare but recur.** It appears in every class, field and
  language: the engine produces 55–75% of the real rate (R2's diagnosis).
- **Benign fills the gap with invented words.** On English and generic sites it writes 2.6–
  3.7 times as many unknown words as real URLs do. Country sites show the same habit (the
  language check, 2026-10-10).
- **Phishing fills it with common words instead.**
- **Structure carries almost none of it.** On benign paths, structural features alone score
  0.53 against 0.66 for vocabulary features, and word order (D4) is CLEARED in benign.

**What has been ruled out:**
- **Word pieces** cannot hold the tail: 32k merges hold under 1% of the rare words (R1).
- **A memory bolted on at decode time** halves the benign path tell, but it copies through
  names and cannot calibrate phishing (R3).
- **Width alone** helps, but slowly: 1.77 times the parameters closed about a tenth of the
  word tell (E16, through R0's yardstick).
- **A word list in the model and a pretrained model** are excluded by the owner (§1).

**Why a product-key memory layer** (Lample et al., NeurIPS 2019):
- It is the established way to give a transformer far more room for rare facts at almost no
  extra compute. Each letter reads only 128 of 262,144 slots, chosen by an exact top-k
  search over two small key tables.
- It is trained end to end by the engine's own loss, so the engine learns what to store and
  when to use it, at every letter, in context. R3 showed that this in-context judgement is
  the ingredient that matters (53% against 39% for a blind memory).
- It adds storage, not a new way of writing. The output is still a distribution over the next
  character.

## 3. The arm

| | `e16_cap` (the weights of `e16_b5`) | `p1_mem` |
|---|---|---|
| network | 8 blocks, d_model 512, 8 heads, d_ff 2,048, max_len 384, tied embeddings | **the same, plus one memory layer after block 4** (§3.1) |
| parameters | 25,473,024 | **161,795,072**: the same 25,473,024 plus 136,322,048 in the memory |
| start | E12's procedure at width 512, seed 20260803 (`caca0107…`) | **every one of those weights at exactly that start**; the memory drawn from its own generator (§3.2) |
| data, batches, steps, objective, optimiser, precision, accumulation | §3.3 | **the same** |
| decode | raw, T 1.0, + D1 + D1′ + B5 (`e16_b5`'s) | **the same** |
| generation | E12's 50 jobs at E12's seeds | **the same** jobs and seeds, so every file pairs with `e16_b5`'s |

### 3.1 The memory layer, exactly

**Where it sits.** Between block 4 and block 5:
`x ← x + Dropout_0.1( M( LayerNorm(x) ) )`.
- It is a new residual branch. It uses the network's own dropout and a LayerNorm of its own.
- Every existing block is unchanged.

**What M computes, for each position that is not padding:**
1. **The query:** q = BatchNorm(W_q·u + b_q), with u = LayerNorm(x). q has 4 heads × 512
   dimensions.
2. **The candidate slots,** per head. The 512 dimensions split into two halves of 256. Each
   half is scored against its own table of 512 sub-keys, and the top 32 are kept in each.
3. **The chosen slots.** The 32 × 32 sums of a sub-key score from each half are the candidate
   slots (slot = 512 × i₁ + i₂). Their top 32 are kept. This search is exact: it returns the
   32 best of all 262,144 slots.
4. **The weights:** a softmax over those 32 scores.
5. **The output:** the weighted sum of the 32 chosen value rows (512-d), summed over the 4
   heads, so each position reads 128 slots.

**Its size and settings:**

| setting | value |
|---|---|
| slots | 512² = 262,144 |
| heads, top-k | 4, 32 |
| key size | 512 (two halves of 256) |
| value size | 512 (= d_model) |
| query normalisation | BatchNorm (Lample et al.'s choice, which keeps the slots in use) |
| precision | float32: the memory runs outside autocast |

**Its parameters (136,322,048):**

| part | parameters |
|---|---|
| values | 134,217,728 |
| sub-keys | 1,048,576 |
| query projection | 1,050,624 |
| BatchNorm | 4,096 |
| LayerNorm | 1,024 |

**Padding and evaluation:**
- Padding positions are skipped: their output is 0.
- In training, the BatchNorm's batch statistics come from real tokens only.
- In evaluation and generation it uses its running statistics, so a URL's output never
  depends on the other URLs in its batch.

**What it is not:**
- No slot holds a string.
- Nothing in it was built from a word list, a lexicon or another model.
- It changes no decode rule. Generation calls `uie.model.generate` unchanged.

**The code lives in `research/v2/`. Nothing under `uie/` changes:**
- `v2_mem.py` holds the layer, the model (a subclass of `uie.model.URLTransformer`), its
  start, and its checkpoint loader.
- `v2_p1_train.py` runs `uie.kaggle_train.run` unchanged. It swaps only the function that
  builds the starting model, and restores it afterwards.

### 3.2 The start

**`e16_cap`'s part:**
- Every weight `e16_cap` has starts at its recorded start. That start is built by the same
  procedure (`new_model_e12`, seed 20260803) and checked on Kaggle before training, exactly
  as `E17_train` did:
  - the width-384 rebuild must equal `e12_final`'s recorded start;
  - the width-512 rebuild must equal `e16_cap`'s (`caca010736da…`).
- **If this torch cannot rebuild them,** the notebook STOPS before training (§6).

**The memory's part,** drawn from its own generator, seed **20261050**, in this order:

| part | drawn from |
|---|---|
| sub-keys | uniform(±1/√512) |
| values | normal(0, 1/√512) |
| query weights | normal(0, 0.02), the network's own initialisation; bias 0 |
| BatchNorm, LayerNorm | their defaults |

**Recorded on Kaggle before training:**
- the `e16_cap` part's fingerprint, which must equal `caca010736da…`;
- the memory's fingerprint, built twice, which must be identical;
- the whole start's fingerprint.

### 3.3 Training: `e16_cap`'s recipe, unchanged

| | value |
|---|---|
| data | corpus2 plus stream 1 (`qry_extra.jsonl`, 57,292 rows, `637accab…`), as `e16_cap` |
| batches | the sampler at seed 20260803, batch 256, window 64; old-row digest `adbddc2a…` |
| steps | 20,000 |
| optimiser | AdamW, lr 3e-4, warmup 500, cosine to 0, weight decay 0.01, betas (0.9, 0.95). **The memory's parameters join the same optimiser with the same settings:** no learning rate of their own, no extra steps |
| clipping | 1.0, then 5.0 after step 1,000 (global norm, memory included) |
| other | dropout 0.1; mixed precision; the fixed objective, C = 14,397.885481960007 |
| accumulation | 2 parts, or 4 if the worst-case check fails at 2 |
| checkpoint | `final.pt` is the registered one |

**The worst-case check, before training:**
- One full micro-batch at max_len 384, forward, backward and an optimiser step, under
  training's autocast. The step allocates the optimiser's state, which the memory makes
  large.
- It runs at 2 parts, then at 4. If it fails at 4, the notebook STOPS.

### 3.4 Generation

- E12's 50 jobs at E12's seeds:
  - free mode: 5 files × 8,000 URLs per class;
  - control mode: 40 files.
  - 180,000 URLs in all.
- `e16_b5`'s decode exactly: raw, T 1.0, + D1 + D1′ + B5.
- One notebook, `V2P1_generate`. It runs only after the laptop has verified training (§10).

## 4. Instruments

### 4.1 The honest judge, D1b (the primary instrument), unchanged

- E18's D1 detector with Vocab_B, as `PREREG_R2.md` §2 defines it. Its features, protocol,
  folds and verdict rule are unchanged.
- **Three views:**

| view | strings | N per side |
|---|---|---|
| **D1b-R** (primary) | the URL with its host replaced by `h.h`, over URLs with a path body, query or fragment | N_R: benign 5,300, phishing 2,400 |
| D1b-H | the host alone | 6,000 |
| D1b-W | the whole URL | 6,000 |

- **The samples:**
  - **Real:** R2 Part A's own, exactly: R0's rest and host samples, and E18's whole-URL
    samples.
  - **`e16_b5`:** R2 Part A's `base` samples, exactly.
  - **`p1_mem`:** drawn from its free files by R0's rule, as source index 2. The seed is
    20261022 + 10,000 × 3 + 100 × class + g. D1b-H adds 1,000 and D1b-W adds 2,000.
  - **If a `p1_mem` file holds fewer rest-bearing URLs than N_R,** the D1b-R samples of
    every source in that class are cut to the largest multiple of 100 that fits, each
    keeping its first n. The cut is recorded as a deviation.
- **Folds:** E18's rule, seed_of(class, g) × 10 + the task's index, with `p1_mem` at index
  12.
- **The reading:** x = AUC(real C against the source) − AUC(real A against real B), as the
  mean over g = 0…4, read like a gate cell: t = (x − 0.05)/SE against ±2.776.
- **V-D1b:** in the same run, `e16_b5`'s readings must reproduce R2 Part A's base values
  exactly (benign R +0.1720, H +0.0581, W +0.1500; phishing +0.0857, +0.1019, +0.1209).
  Otherwise the pipeline has drifted, and nothing is read.

### 4.2 The judge's known blind spot, and W1

- **The blind spot** is R2's limitation L3. The judge knows the words of half B of the
  training data. `e16_b5` trained on both halves, so a word it memorised from a half-B site
  looks known to the judge, and it gets credit that real URLs from unseen sites do not get.
- **Why it matters more now.** A memory layer is built to remember. Left unchecked, it could
  pass the judge partly by reusing words from the sites the judge happens to know. That is
  exactly the copying the honest judge exists to block.
- **Why `p1_mem` still trains on both halves:**
  - Keeping the memory to half A would honour R2's rule literally. But it would change the
    data the engine learns from: half the sites, and phishing rows seen about 8 times
    instead of 3.8 (half A holds 109,351 of the 238,707 phishing rows).
  - That raises the copying risk for its own reasons, and it would break the match with
    `e16_b5`.
  - The blind spot is measured directly instead.
- **W1, the measurement:**
  - A word is **half-B-only** if it is in Vocab_B and not in Vocab_A, per class.
  - In the D1b-R and D1b-H samples, W1 is the share of URLs with at least one half-B-only
    word, pooled over the 5 generations, for real C, `e16_b5` and `p1_mem`.
  - **Real URLs come from sites in neither half,** so their W1 is the natural,
    coincidental rate.
  - A memory that reuses half-B site words pushes `p1_mem`'s W1 above it.
- **Reported beside it:** the half-A-only share. It should rise in step with W1 if the memory
  memorises site words from both halves.

### 4.3 Other word readings

| reading | role |
|---|---|
| D1 (E18's full training vocabulary) on R and W | reported: continuity with R0 and the original V2 bar |
| D0 (the gate's character model) on W | reported |
| D4 (word order, E18 protocol) on R | guard G-D4 |
| **the copier control O4** (R2 Part A: each path, query and fragment replaced by a random half-A training one), on D1b-R | **rerun in the same run, and reported.** It must reproduce R2's values (benign +0.0098, phishing +0.1524). It is the standing proof that a word judge alone rewards copying, which is why §5.4's copying guards are part of success |
| word bands (R2's diagnosis): never seen, 1–9, 10–99, 100–999, 1,000+ by the judge's counts | reported: what the memory changed |

### 4.4 The gate and structure: E12's instruments, unchanged

- `research/e17/e17_score.py`'s parts (free, control, misc, copy), pointed at `p1_mem`'s
  files. They give:
  - the 32 gate cells;
  - FULL;
  - the 8 slash rows;
  - presence and plan honouring;
  - whole-URL novelty;
  - query copying (K12d).
- **Each model's training data is the old corpus plus stream 1.** So `p1_mem` and `e16_b5`
  share one natural rate.
- **The reference** is `e16_b5`'s published report (`data/e17/score_report_e16_b5.json`).
- **V4 runs first:** the scorer must reproduce `e12_final`'s published cells exactly.

### 4.5 Copying, novelty and creativity

These are R3's instrument (`v2_r2.part_metrics`'s measures), on the first 1,600 URLs of each
free file: 8,000 per class, the same rows R3 measured for `e16_b5`.

| measure | defined as |
|---|---|
| URL copied | the sample gate's MEMORIZED share: exact training URLs |
| registrable domain, label, path, query copied | the share whose value occurs in training; near misses beside each |
| word-pair novelty | the share of neighbouring word pairs in paths never seen in training |
| words new to the judge | the share of host and path words not in Vocab_B: the engine must still invent |
| distinct words per 1,000 | hosts and paths, at an equal n |
| duplicates within the files | the sample gate's `duplicate_in_batch` |
| validity failures | the sample gate's other reasons |
| structure | path and query presence; path length in characters and in segments |

The natural rate of each is the same measure on 8,000 real held-out URLs per class (R3's).

### 4.6 Fit, and whether the memory was used (VERIFY_TRAIN, laptop CPU)

**The rows:** per class, 2,000 validation rows from the name frame and 2,000 from the page
frame, seed 20261050.

**What is measured:**
- **Per-row NLL** per scored character, `p1_mem` against `e16_cap` on the same rows.
  - Benign: over every content character.
  - Reported by field (LBL, SUB, PTH, QRY, FRG), with a paired row-bootstrap 95% interval
    (1,000 draws, seed 20261051).
  - Phishing: the same, reported.
- **Slot usage:** the share of the 262,144 slots chosen at least once by those rows. Its
  spread (KL from uniform) is reported beside it.
- **The ablation (reported only):** NLL with the memory's output set to 0. It shows how much
  the engine leans on the memory.
- **From the training record:** the per-field train/validation gap (memorisation) and the
  per-field validation table, against `e16_cap`'s.

## 5. Criteria (fixed now)

### 5.1 K1, the pass mark, per class

- **K1:** `p1_mem`'s D1b-R is **CLEARED**: (x − 0.05)/SE < −2.776 over the 5 generations.
- **A class is VOID** if, in this run, `e16_b5`'s D1b-R is not CONVICTED, or V-floor fails
  for D1b-R. A judge that cannot tell the old engine from real URLs there would "clear" any
  engine.
- **"Whole URL too"** is recorded when D1b-W is also CLEARED in every class that is not VOID.

### 5.2 K2, the memory's effect, per class

- Let Δ(g) = x(`p1_mem`, g) − x(`e16_b5`, g) on D1b-R, paired by g.
- **K2 holds if both:**
  - mean Δ ≤ −0.30 × mean x(`e16_b5`): the memory removes at least 30% of the tell;
  - the paired t ≤ −2.776.

### 5.3 K-used, the memory was used (read at VERIFY_TRAIN, before generation)

**K-used holds if both:**
- (a) benign held-out NLL per content character falls against `e16_cap`, with its 95%
  interval below 0 (§4.6);
- (b) at least 10% of the slots are used by §4.6's rows.

**If K-used fails, the reading is MEMORY UNUSED.** Generation does not run, and the owner
decides.

### 5.4 Guards: `p1_mem` against `e16_b5`

- `p1_mem` is a retrained engine, so the retraining allowance (0.0141) applies, as in E16 and
  E17.
- T is the programme's tolerance, 2√(p(1−p)/n), with p the threshold's own rate.

**Structure and the gate** (existing thresholds):

| code | guard |
|---|---|
| G1 | no gated cell, free or control, has its verdict worsen with x up by more than 0.0141 (K12c) |
| G2 | no PTH, SUB or LBL cell, free or control, rises by more than 0.0141 |
| G4 | free PTH and QRY presence within T, both classes (K12a) |
| G5 | all 8 slash rows within T (K12b) |
| G6 | no control state's P, Q or S honouring falls (E16's G6) |
| G9 | free FULL up by no more than 0.0141, both classes |
| G-D4 | word order: D4 on R up by no more than 0.0141, both classes |

**Fit** (existing threshold):

| code | guard |
|---|---|
| G3 | no non-query field's validation NLL rises by more than 0.01 nats per character against `e16_cap`. Read from the training records' per-field table |

**Copying of complete units** (the owner's condition 2):

| code | guard |
|---|---|
| G7 | queries: K12d. No cell OVER its natural rate, and none further from it than `e16_b5`'s |
| G8 | whole-URL novelty: K12d's rule, both modes |
| **G-COPY** | for whole URLs, registrable domains, paths and queries (§4.5), each copy rate ≤ max(natural, `e16_b5`) + T, per class. This is R2 §5's rule |

**The judge's blind spot** (new):

| code | guard |
|---|---|
| **W1** | half-B-only URLs (§4.2) ≤ max(real, `e16_b5`) + T, in D1b-R and in D1b-H, per class |

**Creativity and diversity:**

| code | guard |
|---|---|
| G-PAIR | word-pair novelty in paths ≥ real − T, per class (R2 §5) |
| C1 | distinct words per 1,000, hosts and paths: within ±5% of real, or no further from real than `e16_b5` plus 2 points |
| C2 | words new to the judge, hosts and paths, ≥ real − T: the engine must still invent (R2 §5) |
| G-DUP | duplicates within the free files ≤ max(real, `e16_b5`) + T |

**Validity:**

| code | guard |
|---|---|
| G-VAL | the sample gate's validity failures ≤ `e16_b5` + T, per class |
| G10 | K12e (clipping after step 1,000 at most 1% in every band) and every check in §7 |

### 5.5 The reading

| K1 over the classes that are not VOID | K2 | guards | reading |
|---|---|---|---|
| CLEARED in every one | — | all pass | **SUCCESS** ("whole URL too" if §5.1's record holds) |
| CLEARED in some, or K2 in at least one class | — | all pass | **PARTIAL** |
| CLEARED in none | in no class | all pass | **NO EFFECT** |
| any of the above | — | one or more fail | that reading, **WITH COST (G…)**: copying, blind-spot, creativity or structure costs named |
| both classes VOID | — | — | **UNREADABLE** |
| K-used fails | — | — | **MEMORY UNUSED** (stopped before generation) |
| a validity check fails | — | — | **INVALID** |

- **SUCCESS is V2's bar, not E3.** E3 stays closed.
- E3's finish line (32 cells and 8 slash rows) is read on `p1_mem` for the record.
- **Reported with every reading, never deciding it:**
  - D1b-H and D1b-W;
  - D1 on R and W;
  - D0;
  - the copier control;
  - the word bands;
  - the half-A-only share;
  - fit and the memory's use;
  - examples of the words each engine writes.

## 6. Stop conditions (fixed now)

| when | condition | what happens |
|---|---|---|
| Kaggle, before training | this torch does not rebuild `e12_final`'s or `e16_cap`'s recorded start | STOP. Nothing trained. The owner chooses: a pinned torch, or the session's own start as an amendment |
| Kaggle, before training | the worst-case check fails at 4 parts | STOP. Nothing trained |
| Kaggle, training | a non-finite training or validation loss that the scaler does not skip | STOP and report. No restart with other settings |
| Kaggle, training | the session ends early | rerun as registered from the start; no resume from a mid-run checkpoint |
| laptop, VERIFY_TRAIN | any validity check fails (§7) | INVALID. No generation |
| laptop, VERIFY_TRAIN | K-used fails (§5.3) | MEMORY UNUSED. No generation |
| laptop, VERIFY_GENERATE | any validity check fails | INVALID. No scoring |
| laptop, scoring | V4 or V-D1b fails | STOP. Nothing is read until the pipeline is fixed; the fix is recorded |
| after the reading | always | the owner decides. No retry, no tuning, no follow-up run without a new registration |

## 7. Validity

| code | check | on failure |
|---|---|---|
| V-iso | `uie/` is byte-identical to its state at this registration's commit | INVALID |
| V-equiv | with every value row at 0, `p1_mem` computes exactly `e16_cap`'s logits from the same weights (a unit test); a `p1_mem` checkpoint loads back to identical weights and outputs | INVALID |
| V1 | hashes and completeness: the bundle and its manifest; `final.pt` as the training record has it; every generated file with its registered n and seed | INVALID |
| V-start | the `e16_cap` part of the start equals `caca010736da…`; the memory part matches the fingerprint recorded before training; the training record's start equals it | INVALID |
| V-recipe | every training setting equals `e16_cap`'s record, except the memory's own settings and, if the fallback ran, 4 parts | INVALID |
| V-batch | the old rows' batch digest over 20,000 steps equals `adbddc2a…` | INVALID |
| V-mass | the appended stream rows equal `e16_cap`'s record exactly (benign P0Q1 24,600 rows and 585,977 positions; P1Q1 70,187 and 2,231,497) | INVALID |
| V-stream | the stream's hash equals `637accab…` | INVALID |
| K12e | clipping after step 1,000 at most 1% in every band | INVALID |
| V-gen | see below | INVALID |
| V-rules | B5: no URL under a Q = 1 plan ends, or starts a fragment, before its query (from the files) | INVALID |
| V4 | the scorer reproduces `e12_final`'s published cells exactly | STOP (§6) |
| V-D1b | `e16_b5`'s D1b readings reproduce R2 Part A's exactly; the copier control reproduces R2's | STOP (§6) |
| V-floor | every word-judge floor within 0.03 of 0.5 | that reading is VOID |

**V-gen in detail:** a leading sample of `p1_mem`'s files is regenerated on the laptop CPU.
- **The bar: at least 1,000 URLs compared, at least 98% byte-identical.**
- **Why the bar is not 100%:**
  - `e16_b5`'s dense network regenerated 957 of 957 (E17).
  - The memory makes discrete choices at every position: the top 32 of 262,144 slots. A
    floating-point near-tie at the edge of a choice can fall differently on the GPU and the
    CPU.
- **Why 98% is still decisive:** a wrong checkpoint, seed, table or decode diverges on
  nearly every URL.
- **Reported:** the mismatch rate, and the position of each first divergence.

**When the checks run:**
- Nothing is read before VERIFY_TRAIN and VERIFY_GENERATE pass.
- The reading is produced by script, pasted, and read once.

## 8. Predictions, before any measurement [HYPOTHESIS]

| reading | prediction |
|---|---|
| K-used | PASS (about 85%). The memory is used; benign held-out NLL falls 0.01–0.05 nats per character, most in paths and labels |
| memorisation | the per-field train/validation gap widens, most in phishing, whose rows are seen about 3.8 times |
| D1b-R, benign | falls 10–45% from `e16_b5`'s (point about 25%); CLEARED about 5% |
| D1b-R, phishing | falls 0–25% (point about 10%); CLEARED about 5%. Its tell is word frequency, which a memory addresses less directly |
| K2 | benign about 40%; phishing about 15% |
| guards | G-COPY or W1 fails about 35%: a memory raises reuse of domains and site words. G1/G2 about 25%; G5 about 15%; C1/C2 about 15% |
| **overall** | SUCCESS about 5%; PARTIAL about 30% (half of it WITH COST); NO EFFECT about 45%; MEMORY UNUSED about 10%; INVALID or UNREADABLE about 10% |

**Stated plainly:** the bar is the right target, and a first-shot pass is unlikely. The run
is designed to show clearly whether a learned memory moves the word tell, and at what cost.

## 9. Cost

| where | what | about |
|---|---|---|
| Kaggle GPU (T4) | `V2P1_train`: the start and memory checks, then training. `e16_cap`'s loop took 2.73 h; the memory adds gathers and a larger optimiser step. Then the training record's evaluations | 3.5–4.5 h |
| Kaggle GPU (T4) | `V2P1_generate`: 50 files, 180,000 URLs (`e16_b5`'s took 2.53 h) | 3–3.5 h |
| **GPU total** | | **about 7–8 GPU-hours** |
| laptop | code, tests, a CPU dry run and notebook simulation, the bundle (before any run) | 6–8 h, mostly mine |
| laptop | VERIFY_TRAIN and K-used | about 1.5 h |
| laptop | VERIFY_GENERATE | about 1 h |
| laptop | scoring: the gate (about 1.5 h, 3 processes), the word judges (about 1.5–2 h), the metrics (about 0.5 h) | about 3–4 h, unattended |
| downloads | the training output: `final.pt` (about 650 MB) and its record. `best.pt` is not needed. The generation output: about 25 MB | |
| **calendar** | | **about 2–3 days** |

## 10. Execution steps

1. **This registration,** committed and pushed before any P1 code.
2. **Code and preparation,** committed and pushed before any Kaggle run:
   - the layer and model, the runner, the common module and the tests (V-iso, V-equiv);
   - the full suite;
   - a CPU dry run: the start, a few training steps in 2 parts on real rows, the
     checkpoint's round trip, and determinism of generation;
   - both notebooks simulated end to end on the CPU from the unzipped bundle;
   - the bundle `uie-v2p1-code.zip`, with its sha256 recorded in Status;
   - the run sheet, `V2P1_KAGGLE.md`.
3. **Kaggle, the owner:** `V2P1_train`.
   - While it runs, the laptop's reading scripts (verify, fit, score, words, compare) are
     written and committed. This is E16's order (`PREREG_E16.md` §13.4).
   - Nothing is read before they are committed.
4. **Laptop:** VERIFY_TRAIN and K-used. STOP if either fails.
5. **Kaggle, the owner, only after step 4 passes:** `V2P1_generate`.
6. **Laptop:**
   - VERIFY_GENERATE;
   - scoring;
   - the criteria;
   - `V2P1_RESULTS.md`, read once, covering the configuration, resources, results, guards
     and deviations.
   - Commit and push, then **STOP: the owner decides.**

## 11. Unchanged and closed

**Unchanged:**
- `e16_b5`: its weights, decode, 50 files and published numbers;
- every accepted E3 result and E3's criteria;
- gate v2 and every threshold;
- the honest judge (D1b, Vocab_A/B);
- `uie/`.

**Not used, though each might raise the odds:**
- a separate learning rate or a sparse optimiser for the memory;
- more steps;
- several memory layers or a larger memory;
- any decode change.

Each would be a different experiment, with its own registration.

**Excluded by the owner:** a word list or lexicon in any form, template matching, and a
pretrained model.

## 12. Amendment 1 (2026-10-12): the tiered copy and novelty rule, logged blind

- **Logged while blind.** `V2P1_train` is still running on Kaggle. It is logged before
  VERIFY_TRAIN, before any generated URL exists and before any reading. The only P1 output
  seen is the training log up to step 0: the start and memory checks, the data counts and
  the first loss line.
- **Whose:** the proposal is the owner's (§12.1, verbatim). The exact numbers and the
  measurable conditions are mine, at the owner's request (§12.3).
- **The owner's message ends at "3." with nothing after it.** Nothing is assumed for that
  point.

### 12.1 The owner's proposal, verbatim

> Before any training results, verification checks, or generated URLs are produced or
> examined, I want to propose and formally log a Pre-Evaluation Amendment regarding how we
> judge copying vs. language learning in `V2-P1`.
>
> Because we are still completely blind to the outcome of `V2P1_train`, recording this now
> preserves full preregistration integrity.
>
> Look at the problem practically: a Memory Layer is designed to internalize vocabulary and
> common web structures. Treating all levels of reuse with the exact same zero-tolerance
> threshold conflates memorizing whole training records with learning natural language and
> standard web conventions.
>
> I am proposing the following **Tiered Copy & Novelty Framework** for evaluating `p1_mem`:
>
> 1. **Whole-URL Copying (`URL copied` / `G8`):**
>    - **Handling:** Very strict requirement (Zero tolerance beyond the existing strict
>      threshold). The engine must never regurgitate complete training URLs.
>
> 2. **Registrable Domain Copying (`Registrable domain copied`):**
>    - **Handling:** Strict requirement, especially regarding training-set domains. The
>      engine must not pass by memorizing and reusing training domains.
>
> 3. **Path and Query Copying (`Path / Query copied` in `G-COPY`, `G7`):**
>    - **Handling:** Limited additional breathing room, * strictly conditional on
>      maintaining overall diversity*. Real web traffic naturally reuses common structural
>      paths and query patterns. A modest increase here should not automatically disqualify
>      the model if diversity holds.
>
> 4. **Word Reuse (Vocabulary overlap / `W1` / site-word reuse):**
>    - **Handling:** Larger allowance. Reusing learned words—including rare words from the
>      training distribution—is a natural and necessary part of learning a language,
>      provided those words are recombined into new contexts rather than copied as whole
>      strings.
>
> 5. **URL Diversity and Novel Combinations (`G-PAIR` word-pair novelty, `C1` distinct
>    words, `C2` new words, `G-DUP`):**
>    - **Handling:** Core mandatory requirement for success. The model must prove it is
>      genuinely generative by combining learned words and sub-structures into novel
>      combinations without batch duplication.
>
> **My question and directive before any results exist:**
> 1. Give me your technical assessment of this 5-part tiered proposal.
> 2. Log this formally as a pre-evaluation amendment (or secondary decision tier alongside
>    the strict baseline guards) *now*, while we are still blind to the run's results, and
>    define the exact numerical tolerances for items 3 and 4 before `VERIFY_TRAIN` runs.
> 3.

### 12.2 In plain words

- **The idea.** A memory layer will reuse more of what it learned. Reusing real words and
  common web patterns is learning; reproducing whole training records is copying. The rule
  should tell the two apart instead of treating all reuse alike.
- **What stays strict:**
  - whole URLs and registrable domains (items 1 and 2), unchanged;
  - **copies of unique training records.** A path or query string that occurs only once in
    training is one site's record. These copies are held to the registered rule even where
    repeated patterns get room.
- **What gets room, and only under conditions:**
  - common paths and queries (item 3): 5 more percentage points, only while diversity
    holds;
  - words the judge knows from particular training sites (W1, item 4): up to 10 more
    points, only if the words are recombined into new pairs **and** the result does not
    depend on them.
- **Diversity (item 5) stays mandatory, and it gates both allowances.**
- **Both readings are always computed.** The tiered reading decides; the registered strict
  reading is printed beside it, so any difference is visible.

### 12.3 The rules, exactly

T is §5.4's: 2√(p(1−p)/n), with p the threshold's own rate.

| item | registered (§5.4) | amended |
|---|---|---|
| 1. whole URLs | G8; G-COPY's URL row ≤ max(natural, `e16_b5`) + T | **unchanged** |
| 2. registrable domains | G-COPY's RD row ≤ max(natural, `e16_b5`) + T | **unchanged** |
| 3. paths and queries: the total copy rate | G-COPY ≤ max(natural, `e16_b5`) + T. G7 (K12d): not OVER natural + T, and no further from natural than `e16_b5`'s | **+0.05 upward, only if the diversity condition holds in the class.** G-COPY ≤ max(natural, `e16_b5`) + T + 0.05. G7's copy rate reads OVER only above natural + T + 0.05. Where `e16_b5` is above its natural rate, a move toward more copying than `e16_b5`'s counts only beyond 0.05. Moves toward less copying are judged as registered |
| 3. paths and queries: unique records (strings seen once in training) | G7: K12d's singleton rule (not OVER natural + T). G-COPY: not measured separately | **Strict.** G7's singleton rule is unchanged. **New G-COPY rows:** the share copying a once-seen path, and the share copying a once-seen query, each ≤ max(natural, `e16_b5`) + T |
| 4. word reuse (W1) | half-B-only URLs ≤ max(real, `e16_b5`) + T, views R and H | **≤ max(real, `e16_b5`) + 0.10, only if G-PAIR passes in the class and the blind-spot check holds there**; otherwise as registered |
| 5. diversity: G-PAIR, C1, C2, G-DUP | mandatory | **unchanged and mandatory**; the condition for items 3 and 4 |

**The diversity condition (item 3), per class and field:**
- C1, C2, G-PAIR and G-DUP pass in the class;
- the field's own duplicate share in the generated sample (R3's `duplicate`: 1 − distinct /
  n) is ≤ max(real, `e16_b5`) + T.

**The blind-spot check (item 4), per class:**
- **The instrument, D1b-x:** a new judge, D1b with its vocabulary restricted to words found
  in both halves (half B's counts). A site word from half B looks exactly as unseen to it as
  one from half A.
- **How it is run:** on the R view, with the same samples and folds as D1b (task indices:
  floor 0, base 1, `p1_mem` 12).
- **The check holds** if every K1 or K2 that passes in the class under D1b also passes under
  D1b-x.

**Why these numbers:**
- **0.05 for common paths and queries.** The copier control bounds what copying can do to
  the pass mark.
  - Copying 100% of rests moved benign D1b-R by 0.162 (from +0.172 to +0.010).
  - For a fixed scorer the excess is linear in the share of copied URLs. So 5 more points
    of copying can move it by about 0.008, far below the about 0.14 that benign needs to
    clear.
  - In phishing, copying made the judge's reading worse (+0.152 against +0.086), so it
    cannot help phishing pass at all.
  - The allowance therefore cannot buy the pass mark. As a share of today's rates it is
    modest: for benign paths, the limit moves from about 21% to about 26%.
- **0.10 for site words.** W1 counts no general vocabulary at all: a word found in both
  halves is never half-B-only. It counts only words the judge knows from particular
  training sites.
  - The blind-spot check makes sure the result does not depend on them.
  - The cap stops the engine leaning on site vocabularies wholesale.
  - The value is a policy choice, far above the noise: with 12,000–30,000 URLs per view, T
    stays under one point at any rate.
- **Unique records stay strict.** Reproducing a string that one training site alone
  contains is the regurgitation §1's condition 2 forbids, whatever the field.

### 12.4 The binding reading

- **The tiered reading decides:** §5.5 with §12.3's guards.
- **The registered strict reading (§5.4 as registered) is always computed and printed
  beside it.** When the two differ, the label shows both, for example "PARTIAL — strict:
  PARTIAL WITH COST (G-COPY)".
- **Everything else is unchanged:** K1, K2, K-used, every other guard, the stop conditions
  and validity.

### 12.5 Code

- `v2_p1_words.py` gains D1b-x on the R view.
- `v2_p1_compare.py` gains the tiered guards and both readings.
- Both are committed before any P1 output is read. The scoring has not run.

## Status

- **2026-10-10:** registered, before any P1 code, training or generation. The owner's GO
  (§1) covers training, generation and the registered evaluation.
- **2026-10-12: code and preparation (§10 step 2).**
  - **Code:** `cda67de` (the layer, its training path, the notebooks and 16 tests) and
    `605f776` (first drafts of the reading scripts). `uie/` is unchanged.
  - **Full suite:** 1,102 passed.
  - **The CPU dry run at full size** (`data/v2/p1/dryrun.json`): PASS.
    - The start is deterministic. Its `e16_cap` part equals this torch's `e16_cap` start,
      and the count is 161,795,072.
    - V-equiv: with every value row at 0, the start computes exactly the logits of
      `e16_cap`'s start on real rows.
    - The checkpoint (617 MB) loads back to identical weights and logits.
    - Generation with `e16_b5`'s decode is deterministic.
  - **The bundle:** `data/kaggle/uie-v2p1-code.zip`, built from commit `605f776`, sha256
    `c87fd80e8ab9e092bc20c099c3070e70a29e107a38e6c9ca00e48066d1662f82`. It is 2.2 MB and
    holds 44 code files, the three tables, stream 1 (57,292 rows) and `e16_cap`'s record.
  - **Deviation, §10 step 2's notebook simulation:**
    - `V2P1_train` ran from the unzipped bundle through cells 1–3: the manifest's hashes,
      the start built twice and identical, and the memory check's code path.
    - **Its training cell then froze the laptop.** Training the full-size engine on the
      CPU needs about 3 GB on its own. The simulation's small data copy also made the
      stream injector add about 780 new rows per old row (the real run's rate is 1.0), so
      one batch asked for 2.65 GB more. The owner shut the laptop down.
    - **On the owner's decision, nothing trains on the laptop.** The simulation is retired:
      `v2_p1_simulate.py` refuses to run.
    - **What covers the training path instead:** the end-to-end toy test through
      `uie.kaggle_train.run` with the memory builder, the full-size dry run, and the
      notebook's own checks, which stop before any training.
    - **`V2P1_generate` was not simulated.** Its setup now asserts that every weight is on
      the GPU. Before it runs, VERIFY_TRAIN loads the trained checkpoint on the laptop CPU
      and generates a few URLs with the notebook's own calls (inference only).
  - **Next:** the owner runs `V2P1_train` (`V2P1_KAGGLE.md`).
- **2026-10-12: Amendment 1 (§12), the tiered copy and novelty rule, logged blind** while
  `V2P1_train` runs, before VERIFY_TRAIN and before any generated URL or reading exists.
- **2026-10-13: STOPPED at K-used: MEMORY UNUSED** (`V2P1_RESULTS.md`).
  - VERIFY_TRAIN PASS.
  - K-used (a) FAIL: benign +2.947 nats per character against `e16_cap`. (b) PASS: 84.4% of
    the slots are used.
  - Generation did not run.
  - The cause (post hoc): the query BatchNorm depends on the length-grouped batch
    composition, so evaluation mode breaks.
