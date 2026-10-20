# V2-P1 — the hybrid-tokenizer pilot: registration

- **Registered 2026-10-09, before any V2 code, data, tokenizer or training exists.** The
  design is `research/v2/DESIGN_V2.md`. This document fixes the pilot's arms, rules,
  criteria and predictions.
- **The owner's gates:**
  - **A** — approve this registration, which starts the code and the laptop steps;
  - **B** — approve Amendment 1 (the frozen numbers), which starts Kaggle;
  - **C** — the reading, and what follows it.
- **Outside E3.** E3 stays CLOSED, NOT MET. `e16_b5` stays `BASELINE` and is read-only.
  Nothing here changes `uie/`, the gate, or any published number.

## 0. In plain words

- **The question.** If the engine writes paths, queries and fragments in learned pieces
  instead of letters, does it invent fewer fake words there, without breaking anything that
  works today?
- **How.** Train two engines that are identical in every way except that one choice:
  - `v2_char`, a letter-by-letter twin of `e16_b5`'s engine, trained again;
  - `v2_hyb`, the same with pieces in those three fields.
  - Same data, same batches in the same order, same starting weights, same width, steps and
    decode rules.
- **Why a twin, and not just `e16_b5`.** Two trainings of the same recipe never come out
  identical; the programme measured differences of up to 0.0148 on a cell. Comparing the
  hybrid with a twin trained alongside it isolates the tokenizer.
  - `e16_b5` stays the fixed reference.
  - The twin-against-`e16_b5` difference measures how large plain retraining noise is.
- **What is read.** Free mode only: 5 × 8,000 URLs per class per engine.
  - The main reading: the word-aware detector on the path, query and fragment.
  - The same detector on the host, which should not move.
  - The gate's free cells, the slash rows, presence, plan following, copying, novelty and
    fit, as guards.
- **Cost:** about 8 Kaggle GPU-hours over three notebooks, and about 12–16 hours of laptop
  machine time, mostly unattended (§10).
- **What decides.** In every class where the twin's tell exists, the hybrid must cut it by
  at least a quarter, by more than retraining alone moves it, and pass every guard. Nothing
  is tuned and nothing is retried.

## 1. The question and the hypothesis

> **H1.** Writing the path, query and fragment in learned pieces (BPE, 8,000 merges)
> instead of characters reduces the engine's word-level tell in those fields, at a matched
> training budget, without a structural, obedience, copying or char-level regression.

- **Scope:** free mode, both classes, `e16_cap`'s recipe at full scale.
- **Not claimed, even if H1 holds:**
  - the host's word gap (unchanged by design);
  - control mode (P2);
  - the free phishing QRY cell (63% context-led);
  - any E3 result.
- **The mechanism** is in `DESIGN_V2.md` §2.3. The pilot tests its consequence, not its
  cause.

## 2. Arms and references

| | role | tokenizer | weights | generated in P1 |
|---|---|---|---|---|
| **`v2_hyb`** | the arm | hybrid: characters for the host, pieces for PTH, QRY and FRG | trained in P1 | yes, free mode |
| **`v2_char`** | the matched twin | characters (`e16_cap`'s tokenizer) | trained in P1 | yes, free mode |
| `e16_b5` | **frozen reference** | characters | `e16_cap`'s `final.pt` (`98b10b46…`) | no: its existing free files (`data/e17/decode/gen`) |
| `e15_new` | capacity reference, word readings only | characters | width 384; decoded without B5 | no: its existing free files (`data/e15/generate/gen`) |

**Identical between `v2_hyb` and `v2_char`:**
- the rows: training and validation rows whose character encoding fits 384 tokens, and
  E15's stream (`637accab…`) at rate 1;
- the batches: the same rows in the same order (sampler keyed on character length; V-batch);
- the start: `e16_cap`'s start (`caca0107…`); the hybrid adds only its 8,000 piece rows;
- the architecture: d_model 512, 8 layers, 8 heads, d_ff 2048, `max_len` 384, tied
  embeddings;
- every training setting in `uie/baseline.py` `E16_B5.recipe`:
  - 20,000 steps, batch 256, accumulation 2;
  - lr 3e-4, warmup 500, cosine decay;
  - clipping 1.0, then 5.0 after step 1,000; AdamW (0.9, 0.95), weight decay 0.01;
  - class α 1, the frame token, the presence plan, the root slash, the slash plan, the
    fixed objective;
- the checkpoint read: `final.pt` at step 20,000;
- generation: E12's free jobs and seeds, the presence-slash and suffix tables, and the
  decode raw (T 1.0) + D1 + D1′ + B5.

**Different, and only these:**
- the tokenizer (`DESIGN_V2.md` §4);
- what follows from it:
  - 4,096,000 extra embedding parameters (non-embedding parameters identical);
  - C, re-derived by E9's rule;
  - the piece rule at decode (it does nothing for `v2_char`);
  - fewer tokens per URL, so the hybrid spends less compute per URL. If that biases
    anything, it biases against the hybrid.

## 3. The tokenizer, as registered

`DESIGN_V2.md` §4, with these values fixed now:

| item | value |
|---|---|
| merges | **8,000**; never changed after this registration |
| learning data | the PTH, QRY and FRG bodies of every training row `v2_char` keeps, plus the QRY bodies of the stream's 57,292 rows |
| weighting | each body by its row's draw probability under the training sampler (class α = 1); stream rows at the old rows' per-row rate, as the injector draws them |
| fallback | if learning on all rows exceeds 4 hours in the dry run, a uniform sample of 1,000,000 training rows (seed 20261021) replaces it. Recorded in Amendment 1, before training |
| ties | the pair whose two pieces come first in code-point order |
| piece rows at the start | N(0, 0.02), the model's own initialisation, drawn in vocabulary order from a generator seeded 20261020 |
| ids | every `e16_cap` token keeps its id; the pieces follow |
| the piece rule | `DESIGN_V2.md` §4.4. The allowed set per field is computed once from the training encodings and hashed |

## 4. Training

### 4.1 The recipe

- `E16_B5.recipe` exactly (§2), for both arms.
- Both arms train through V2's trainer, with the two matching rules (`DESIGN_V2.md` §6).

### 4.2 The start

- **V-start′, E16's procedure, on Kaggle:**
  - rebuild `e12_final`'s width-384 start; STOP unless it is `c7e71edf…`;
  - build the width-512 start; STOP unless it is `caca0107…`.
- `v2_char` trains from that start.
- `v2_hyb` trains from the same weights with the 8,000 piece rows added (§3). Check:
  removing the piece rows gives back `caca0107…` exactly.
- **Plan B**, only if V-start′ cannot be reproduced on Kaggle's current image: one notebook
  trains both arms in sequence, from one start built in that session and saved. The twin's
  comparison with `e16_b5` then also includes a different start, and the report says so.

### 4.3 The objective

- `v2_char`: C = 14,397.885 (pinned; `e16_cap`'s).
- `v2_hyb`: C = 256 × the expected scored-token mass per drawn row under the hybrid
  encoding (`uie.train.expected_scored_tokens`, old rows only). Computed on the laptop and
  recorded in Amendment 1.

### 4.4 Kaggle

- Two notebooks, `V2P1_train_char` and `V2P1_train_hyb`, one arm each. They may run in
  parallel.
- Before training, each checks:
  - torch and the GPU;
  - V-start′;
  - a worst-case memory preflight at `max_len` 384 (fallback: accumulation 4, as in E16);
  - the bundle and corpus hashes.

## 5. Generation

- **Free mode only.** E12's 10 free jobs per arm: 5 generations × 8,000 URLs per class, at
  E12's seeds.
  - The six-state plan is supplied from the presence-slash table, and suffixes come from the
    shipped table, as in `e17_common.free_urls`.
- **Decode:** `e17_common.decode_kwargs(True)` (raw, T 1.0, + D1 + D1′ + B5) for both arms,
  plus the piece rule for `v2_hyb`.
- **Recorded:**
  - per URL: its supplied plan and its token ids (for the piece-rule check and the
    non-canonical rate);
  - per file: the mass D1 and B5 moved, and the mass the piece rule removed.
- **One notebook,** `V2P1_generate`, for both arms, after the laptop has verified training.
- **Not generated in P1: control mode.** P1 therefore does not read the 4 control slash
  rows, the 22 control cells, or plan following under control. Those wait for P2.

## 6. Instruments

### 6.1 The word detectors (E18's code; the detectors are unchanged)

All four use `research/e18/strong_disc.py`'s detectors exactly as they ran in E18:
- 5-fold out-of-fold ROC AUC, with E18's fold seeds;
- the class's vocabulary files (`data/e18/strong_disc/vocab_{class}.npz`);
- real samples from the gate's whole-URL validation pool (one URL per host, scheme
  stripped).

Each reading is x = AUC(real C against the source) − AUC(real A against real B). It is the
mean over g = 0…4 and is read like a gate cell: t = (x − 0.05)/SE against ±2.776.

| name | the strings | the population | N per side |
|---|---|---|---|
| **D1-R** (primary) | the URL with its host replaced by the constant `h.h`, so only the path, query and fragment carry information | URLs with a non-empty path body, query or fragment | N_R(class) |
| **D1-H** (negative control) | the host alone | every URL | 6,000 (E18's) |
| **D1** (comparability) | the whole URL, exactly as in E18 | every URL | 6,000 |
| **D0** (comparability) | the gate's own model on the whole URL, exactly as in E18 | every URL | 6,000 |

- **N_R(class)** is the largest multiple of 100 not above the smallest of:
  - 6,000;
  - a quarter of the real pool's URLs that carry a path body, query or fragment;
  - 0.9 × the fewest such URLs in any of `e16_b5`'s free files of that class.
  - It is computed at step R0 and recorded in Amendment 1, before any V2 file exists.
- **Pairing.** For a given class and g, the real samples are the same for every source, so
  differences between sources are paired by g. The sampling seeds are fixed in
  `v2_common.py` and recorded in Amendment 1.
- **Why D1-R masks the host and restricts the population.**
  - Close to 60% of phishing URLs and about a quarter of benign ones carry no path, query or
    fragment. Left in, they become identical strings on both sides and dilute the reading
    heavily: for phishing, most comparisons would be between identical strings.
  - The gate avoids the same dilution by scoring a field only where it exists
    (`discriminate_with_floor`).
- **D1-R's floor** is computed on the same transformed strings, and V-floor applies to it.

### 6.2 The gate (character-level)

- The gate's free cells (LBL, SUF, SUB, PTH, QRY per class) and FULL.
- Scored by `e17_score`'s free part, with the gate's real pools, floors and seeds (gate v2,
  unchanged).

### 6.3 Structure and obedience

| reading | definition | source |
|---|---|---|
| presence | free PTH and QRY rates against real, within tolerance T | K12a; `DECISION_2026-09-29.md` §2 |
| slash rows | the 4 free rows (benign and phishing; bare hosts and host-route queries), within T | K12b |
| plan following | per supplied state, the share of URLs whose P, Q and S, read from the URL string, equal the plan | E16's G6, here on free mode's supplied plans |
| rule integrity | D1: every host on a plan D1 covers ends as S says. B5: no URL under a Q = 1 plan ends or starts a fragment before its query. The piece rule: no piece in a host field, and none outside its allowed fields | by construction; counted from the records |

### 6.4 Copying and novelty

- **Queries (G7):** K12d's copy rule on the free QRY cells. No cell is OVER its natural rate,
  and no copy rate moves further from it (old + stream counts, `PREREG_E15.md` §13.2).
- **Paths (G7-P, new):** the same rule on the free PTH cells. The natural rate comes from the
  gate's real pool against the training paths.
- **Novelty (G8):** K12d's whole-URL novelty rule, with the old + stream known set.

### 6.5 Fit

- Validation NLL per field, in **nats per character** under the canonical split, for both
  arms on the same validation rows.
- V2's evaluator must reproduce `v2_char`'s training record where the two overlap (a
  self-check).

### 6.6 Reported, never decisive

| code | what |
|---|---|
| R0 | `e16_b5` and `e15_new` on D1-R and D1-H, read before any V2 training |
| X-1 | tokens per URL, by field and class; training and generation time per arm |
| X-2 | E18's word statistics (unseen words per URL, longest unseen word, known-word share in the host and in path + query), real against each source |
| X-3 | the share of path segments found whole in the training paths (E5's measure, `E3_REPORT.md` §13.4) |
| X-4 | the mass the piece rule removed, per field; the non-canonical split rate |
| X-5 | E3's far cell (free phishing QRY) and the free PTH cells, hybrid against twin |
| X-6 | the twin against `e16_b5` on every reading: the retraining spread |
| X-7 | the share of generated URLs longer than 384 characters |

## 7. Validity: checked before anything is read

| code | check | on failure |
|---|---|---|
| V-T1 | losslessness: every training, validation and stream URL round-trips (`DESIGN_V2.md` §4.5) | stop; fixed before training |
| V-T2 | the pieces were learned from training rows only (the files read, hashed) | stop |
| V-T3 | the merge table and the per-field allowed sets are hashed in Amendment 1, and equal on Kaggle | stop |
| V-equiv | V2's trainer in character mode equals `uie.train.train` bit for bit on the CPU (first steps, at toy scale); its 20,000-step batch digest equals `adbddc2a…` | stop |
| V-gen-equiv | V2's free generator, with `e16_cap`'s weights and `e16_b5`'s decode, reproduces the start of `e16_b5`'s free files byte for byte on the CPU | stop |
| V-start′ | §4.2 | STOP on Kaggle; Plan B only by §4.2's rule |
| V-batch | in both arms, the digest of the old rows' batches over their character encodings equals `adbddc2a…` | the arm is INVALID |
| V-mass | the stream rows appended equal `e16_cap`'s record in both arms: 24,600 (benign P0Q1) and 70,187 (benign P1Q1) | INVALID |
| V-stream, V-accum | as in E16 | INVALID |
| K12e | clipping after step 1,000 at most 1% in every band, both arms | INVALID |
| V-rules | every rule-integrity count of §6.3 is zero | INVALID: a rule did not run |
| V-gen | a sample of every file, regenerated on the laptop CPU, matches byte for byte | INVALID |
| V-score | the scorer reproduces `e16_b5`'s published free cells exactly from its files | stop |
| V-D1 | the word pipeline reproduces E18's `e16_b5` readings exactly: D1 +0.1675 / +0.1637, D0 +0.0269 / +0.0437 | stop |
| V-floor | every word-detector floor within 0.03 of 0.5 (E18's rule) | that reading is VOID |

- Nothing is read before VERIFY_TRAIN and VERIFY_GENERATE pass.
- The readout is produced by script and pasted. The results are read once.

## 8. Criteria (fixed now)

### 8.1 K1, the primary, per class

Write x_R for the D1-R excess, and Δ_R(g) = x_R(`v2_hyb`, g) − x_R(`v2_char`, g).

K1 passes in a class if and only if **all three** hold:
1. **Size:** mean Δ_R ≤ −max(0.0141, 0.25 × mean x_R(`v2_char`)). The fall must be at
   least the retraining allowance, and at least a quarter of the twin's tell.
2. **Certainty:** the paired t over the 5 generations is ≤ −2.776.
3. **Beyond retraining:** |mean Δ_R| > |mean x_R(`v2_char`) − mean x_R(`e16_b5`)|. The
   tokenizer must move the tell more than retraining alone did.

- **STRONG** is recorded when K1 passes and `v2_hyb`'s D1-R is CLEARED.
- **A class is VOID** (no tell to remove, or no working instrument) if:
  - `e16_b5`'s or `v2_char`'s D1-R is not CONVICTED in it; or
  - V-floor fails for D1-R in it.
- **Why a quarter.** A pilot that removes less than a quarter of the tell would not, on its
  own, justify a new programme. Fixing the bar now stops it from moving later.

### 8.2 K2, specificity (the negative control)

- On D1-H (the host, which both engines write identically), the reading is **SPECIFIC** if
  mean Δ_H is within ±0.0141, or its paired t is within ±2.776. Otherwise it is
  **NOT SPECIFIC**.
- It qualifies K1 and is not a failure. A hybrid that also changed the host would mean the
  gain may not come from the pieces themselves.

### 8.3 Guards: `v2_hyb` against `v2_char`

Existing thresholds, except where marked new.

| code | guard |
|---|---|
| G1′ | no free gated cell's verdict worsens with x up by more than 0.0141 (K12c) |
| G2′ | no free PTH, SUB or LBL cell rises by more than 0.0141 |
| G3′ | no host field's (SUF, LBL, SUB) validation NLL rises by more than 0.01 nats per character. PTH, QRY and FRG are reported |
| G4 | free PTH and QRY presence within T, both classes (K12a), on `v2_hyb` |
| G5′ | the 4 free slash rows within T (K12b), on `v2_hyb` |
| G6′ | no supplied state's P, Q or S following falls. A fall counts only if it is at least T and its two-proportion 95% CI excludes 0 (E13 Amendment 1's rule) |
| G7 | queries: K12d's copy rule (§6.4) |
| G7-P | paths: the same rule (**new**) |
| G8 | K12d's whole-URL novelty rule |
| G9 | free FULL up by no more than 0.0141, both classes |
| G10 | K12e, and every validity check in §7 |

- G4 and G5′ are absolute. If `v2_char` fails the same row, the failure is put down to
  retraining. It is recorded and does not count against the tokenizer.

### 8.4 The reading

| K1, over the non-void classes | guards | reading | what follows |
|---|---|---|---|
| passes in every one | all pass | **SUPPORTED** | the owner decides whether to open P2 |
| passes in some | all pass | **PARTIAL** | the owner decides; a P2 would diagnose the failing class first |
| passes in none | all pass | **NOT SUPPORTED** | pieces at 8,000 merges do not reduce the word tell, and the tokenizer route closes as registered; the owner decides |
| any | one or more fail | the reading above, **WITH COST (G…)** | no P2 before the cost is understood; the owner decides |
| both classes VOID | — | **UNREADABLE** | no tell to remove at this scale; the owner decides |
| — | a validity check fails | **INVALID** | nothing is read; an amendment decides what happens |

- K2's qualifier is attached to every reading in which K1 passes somewhere.
- **No retry, no second merge count, no tuning.** A different merge count, a second hybrid
  arm or control mode would each need a new registration.
- **The early stop.** If `e16_b5`'s D1-R is CONVICTED in neither class at R0, P1 stops there
  and comes back to the owner before any code is built.

## 9. Predictions, before any measurement [HYPOTHESIS]

| reading | prediction |
|---|---|
| R0: `e16_b5`'s D1-R | CONVICTED in benign (~90%); in phishing (~60%) |
| `v2_char` against `e16_b5` | every free cell within 0.0141 (~75%); D1-R within 0.02 (~70%) |
| tokens per URL, hybrid ÷ twin | 0.60–0.75 |
| known words in path + query, benign | twin ~84%; hybrid 88–92%; real 94% |
| unseen words per benign URL | twin ~1.45; hybrid 1.05–1.25; real 0.93 |
| K1, benign | passes (~60%) |
| K1, phishing | passes (~30%); VOID (~35%) |
| K2 | SPECIFIC (~75%) |
| guards | G4, G5′ and G6′ pass (~90%); G7 and G7-P (~80%; the risk is overshoot on common queries); G1′ and G2′ (~70%); G3′ (~85%); G9 (~80%) |
| free phishing QRY (X-5) | hybrid − twin between −0.025 and +0.005 (~70%); CLEARED (~10%) |
| **overall** | SUPPORTED ~25%; PARTIAL ~25%; any reading WITH COST ~30%; NOT SUPPORTED ~15%; INVALID or UNREADABLE ~5% |

## 10. Cost

### 10.1 Kaggle (T4; measured where a measurement exists)

| notebook | about | basis |
|---|---|---|
| `V2P1_train_char` | about 3 h | `e16_cap`, the same recipe: its training loop took 2.73 h, memory 8.1 of 14.6 GB (`data/e16/train/e16_cap/train_result.json`); E17's notebook spent about 0.15 h around its own loop |
| `V2P1_train_hyb` | 2.5–3.5 h | estimate: about 30% fewer tokens per row, an output layer that adds about 16% compute per token, and encoding the corpus into pieces on Kaggle. Measured in the dry run and recorded in Amendment 1 |
| `V2P1_generate` | about 2 h | `e16_b5`'s 50 files (180,000 URLs) took 2.53 h. P1 generates 80,000 free URLs per arm, the hybrid in fewer steps |
| **total** | **about 8 GPU-hours** (7.5–8.5) | well within one week's Kaggle GPU allowance; plus waits for a machine (up to about 50 min seen) |

- **For the owner:**
  - upload one dataset (the code bundle; the corpus dataset is already on Kaggle);
  - run three notebooks (the two training ones can run at the same time);
  - download three outputs.
- **Wall-clock on Kaggle:** about 3.5 h of training if run in parallel, then about 2 h of
  generation, after the laptop's check.

### 10.2 Laptop (CPU, mostly unattended; the laptop must stay on)

| step | about |
|---|---|
| R0 (the reference word readings, N_R) | 0.5 h |
| learning the pieces | 1–4 h (pure Python, once) |
| losslessness on about 2.8M URLs | 0.5 h |
| equivalence tests, notebook simulations, timing | 1.5 h |
| VERIFY_TRAIN | 1.5 h |
| fit (nats per character, two arms) | 1.5–2 h |
| VERIFY_GENERATE (byte identity) | 0.7 h |
| free gate scoring, two arms | 2–3 h |
| word detectors, copying, novelty, diagnostics | 2 h |
| **total** | **about 12–16 h** |

### 10.3 Calendar

About 3–4 days from gate A, set mostly by Kaggle availability and the owner's two remaining
gates.

## 11. Execution steps and gates

0. **Gate A:** the owner approves this registration. Nothing runs before it.
1. **R0, on the laptop:** `e16_b5` and `e15_new` on D1-R and D1-H; N_R. If `e16_b5`'s D1-R
   is CONVICTED in neither class, stop and report.
2. **Code, committed before any Kaggle run:**
   - `research/v2/` and its tests;
   - V-equiv and V-gen-equiv;
   - learning the pieces; V-T1 to V-T3; C for the hybrid;
   - notebook simulations with the packages Kaggle lacks blocked, in Kaggle's real mount
     layout;
   - per-step timing and the memory estimate;
   - the bundle (no nested `.gz`) and the notebooks.
3. **Amendment 1 (gate B):** the frozen numbers, added below and never edited above:
   - the merge-table and allowed-set hashes;
   - C, N_R and the sampling seeds;
   - the bundle hash, the timings and the memory.
   - The owner approves the Kaggle plan.
4. **Kaggle:** `V2P1_train_char` and `V2P1_train_hyb`.
5. **Laptop:** VERIFY_TRAIN. Then the owner runs `V2P1_generate`.
6. **Laptop:** VERIFY_GENERATE, scoring and the readings, then `V2P1_RESULTS.md`. Read once,
   after every validity check passes.
7. **Gate C:** the owner decides what follows.

## 12. Alternatives considered (not registered)

| option | Kaggle | what it would cost the answer |
|---|---|---|
| **as registered:** the full recipe, free mode | about 8 h | control mode waits for P2 |
| add control mode now | about +3 h (100,000 more URLs per arm); laptop about +4 h | nothing. It adds the 4 control slash rows, the 22 control cells and plan following under control |
| a smaller model (width 384), or half the steps | about 4–5 h | the twin is no longer a retraining of `e16_b5`'s engine, so there is no retraining yardstick and no direct comparison with `e16_b5`'s numbers. Tokenizer effects are often larger in small models, so a positive result would still need a full-scale run |
| no twin: the hybrid against `e16_b5` | about 4–5 h | the tokenizer's effect cannot be separated from retraining noise, or from V2's own code path |

## 13. Unchanged and closed

- **E3:** CLOSED, NOT MET. No V2 number is an E3 number, and the E3 documents are not
  edited.
- **`e16_b5`:** `BASELINE`. Its weights, files and records are read-only inputs, pinned by
  hash.
- **`uie/`:** untouched. The gate, its thresholds, T and the retraining allowance are
  unchanged.
- **Closed branches stay closed:**
  - decode calibration knobs (D2);
  - decoder sampling changes;
  - more steps (R2);
  - width or depth (R3);
  - new or synthetic training data.

## 14. Amendment 1 (2026-10-09, before gate A): the owner's practical redesign

- **Written at the owner's request, before any V2 code, data or training exists.** It
  replaces the sections named in §14.3. The text above is kept exactly as it was registered.
- **Gate A now approves the registration as amended.**
- The frozen-numbers amendment that §11 called "Amendment 1" becomes **Amendment 2**
  (gate B).

### 14.1 The owner's message, verbatim

> I've been thinking about the pilot design, and I'd like to discuss a few practical
> adjustments. I want to shift our focus slightly to align more directly with our ultimate
> goals. Here is what I have in mind:
> Shifting the Comparison Paradigm (Dropping the Twin): Let's drop the v2_char twin engine.
> Honestly, perfectly isolating the BPE variable isn't as critical to me right now as simply
> building the strongest V2 possible. I want you to have the freedom to optimize V2's design
> choices to maximize its chances of passing the success criteria. We will still compare
> v2_hyb against the frozen e16_b5, but we'll treat it as a practical comparison between two
> completely distinct architectures (Hybrid/BPE vs. Pure Char) rather than a strict A/B
> control experiment. This saves us GPU budget and time while remaining highly valuable
> research.
> Moving Scoring to Kaggle: 12-16 hours of laptop scoring just isn't practical for our
> workflow. Let's run the full 8,000-sample evaluation, but shift the heavy scoring pipeline
> to a Kaggle notebook (e.g., V2_score.ipynb) to leverage their infrastructure.
> Absolute Success Criteria: Let's drop the relative '25% reduction' rule and aim straight
> for the gold standard. The success condition for v2_hyb should be the absolute E3 bar:
> Excess AUC on the D1 vocabulary detector for the path, query, and fragment must be
> strictly < 0.05.
> Adding Control Mode: We should definitely include Control Mode in both generation and
> evaluation. I'm completely fine with the extra ~3 Kaggle hours this will require.
> Keeping the Core Specs: I'm totally on board with keeping the 8,000 BPE merges and the
> full-size training (d_model=512). I also fully understand and accept the host limitation
> (character-level hosts will still trigger the detector).
> What do you think of this practical, goal-oriented approach? If you agree, please amend
> PREREG_V2_P1.md with these constraints, commit the update, and let me know the revised
> timeline and cost before we pull the trigger

### 14.2 In plain words

- **One new engine, no twin.** `v2_hyb` is compared directly with the frozen `e16_b5`, as
  two different designs (pieces against letters), not as a controlled experiment.
  - It saves about 3 GPU-hours.
  - The cost: where the two differ, part of the difference can be ordinary retraining
    noise, and it can no longer be separated out.
- **The pass mark is absolute.** The word detector on the path, query and fragment must
  clear the gate's bar — proven below 0.05 — in both classes. A relative improvement does
  not count as a pass.
- **Control mode is in.** All 32 cells and all 8 slash rows are read.
- **Scoring moves to Kaggle,** in a CPU session that uses no GPU allowance. `e16_b5` is
  scored again in the same session, so both engines are measured by the same software.
- **V2 is designed to pass.** Its choices are made with that aim, but all of them before any
  V2 result exists. Nothing is changed after a result has been seen.

### 14.3 What this replaces

| registered | amended |
|---|---|
| §0 and §1: the twin comparison; H1, relative | §14.2; H1′ (§14.4) |
| §2: two arms, `v2_hyb` and `v2_char` | `v2_hyb` only. `e16_b5` is the frozen reference, scored again in the Kaggle run; `e15_new` is a word-reading reference |
| §4.1: the two matching rules | §14.5, choices 1 and 2 |
| §4.2: Plan B | not needed with one arm. If V-start′ fails on Kaggle, `v2_hyb` starts from a fresh seeded start built in the session, and records it |
| §4.4: two training notebooks | one: `V2P1_train` |
| §5: free mode only | free and control mode: E12's 50 jobs |
| §6: the instruments | as registered, extended to control mode; §14.4 says which reading decides |
| §7: validity | §14.7 |
| §8: K1, K2 and the guards against the twin | §14.4 and §14.6 |
| §9, §10, §11 | §14.8, §14.9, §14.10 |

### 14.4 The pass mark (replaces §1's H1 and §8.1–8.2, §8.4)

> **H1′.** `v2_hyb`'s word tell in the path, query and fragment (D1-R, free mode) is
> CLEARED in every class where the instrument can see the tell, without a regression in
> structure, obedience, copying or the character-level gate.

- **CLEARED is the gate's verdict:** (x − 0.05)/SE < −2.776 over the 5 generations, that is,
  x proven below 0.05 with 95% confidence.
  - "Strictly below 0.05" is read as this, the E3 standard.
  - A point estimate under 0.05 that is not proven reads UNDECIDED, as E3's SUB cell (0.0431)
    did.
- **The instrument must be able to see the tell.** In the same Kaggle run, D1-R must CONVICT
  `e16_b5` in the class. Otherwise the class is VOID: a detector that cannot tell the old
  engine from real URLs there would "clear" any engine.
- **The reading:**

| D1-R on `v2_hyb`, over the classes that are not VOID | guards (§14.6) | reading |
|---|---|---|
| CLEARED in every one | all pass | **SUCCESS** |
| CLEARED in some | all pass | **PARTIAL** |
| CLEARED in none | all pass | **NOT MET** |
| any | one or more fail | the reading above, **WITH COST (G…)** |
| both classes VOID | — | **UNREADABLE** |
| — | a validity check fails | **INVALID** |

- **Reported with every reading, never deciding it:**
  - `v2_hyb` against `e16_b5` on D1-R: the difference, paired by generation, with its t. It
    compares two designs and includes retraining noise. `e15_new` against `e16_b5` is shown
    beside it, as a yardstick of how much two letter-by-letter engines differ;
  - D1-H (the host), the whole-URL D1 and D0, and D1-R in each measurable control state;
  - the E3 finish line read on `v2_hyb` (all 32 cells and the 8 slash rows), for the
    record. E3 stays closed.
- **SUCCESS is V2's pilot bar, not E3.** E3 also needs every gated cell CLEARED.
- **No retry and no tuning,** as in §8.4.

### 14.5 Choices made to give V2 its best chance, fixed now

- **What the owner's "freedom to optimize" covers:** every choice made before any V2 output
  exists.
  - It does not cover changing V2 after a reading. Decode knobs fitted to a reading stay
    closed (D2).
  - **Why:** if V2 were adjusted after seeing what the word detector says, passing the
    detector would show that V2 had been fitted to it, not that V2 is realistic.
  - Any later variant (another merge count, for example) needs its own registration. A
    winner chosen among variants must be confirmed on fresh real samples that played no part
    in choosing it.
- **The choices:**
  1. **Long rows kept.** `v2_hyb` trains on every training row whose own encoding fits 384
     tokens, including the rows the letter-by-letter engine dropped for length.
     - It is the same corpus, not new data. `e16_cap` dropped 3,851 training rows (about
       0.14%) and 543 validation rows; Amendment 2 records `v2_hyb`'s count.
     - Why: real URLs include such rows, a letter-by-letter engine can never write them,
       and QC1-6 measured that benign's missing long queries carry part of the query tell.
       The effect is expected to be small.
  2. **Its own batches.** Batches are sorted by `v2_hyb`'s own token length, which wastes
     less compute on padding. The batch and injection checks stay exact, against the
     laptop's replay of `v2_hyb`'s own sampler.
  3. **Everything else as registered:**
     - 8,000 merges, the data weighting and the piece rule;
     - `e16_cap`'s start plus the piece rows;
     - `E16_B5.recipe`, with C re-derived;
     - `final.pt`;
     - `e16_b5`'s decode plus the piece rule.
- **Not used, although they might raise the odds:** more steps (R2), more width or depth
  (R3), a different decode temperature or top-p, new or synthetic data. All stay closed.

### 14.6 Guards (replaces §8.3): `v2_hyb` against `e16_b5`, from the same Kaggle run

The retraining allowance (0.0141) applies, as when E16 and E17 compared a retrained engine
with the baseline before it. The thresholds are existing ones, except where marked new.

| code | guard |
|---|---|
| G1 | no gated cell, free or control, has its verdict worsen with x up by more than 0.0141 (K12c) |
| G2 | no PTH, SUB or LBL cell, free or control, rises by more than 0.0141 |
| G3′ | no host field's (SUF, LBL, SUB) validation NLL rises by more than 0.01 nats per character against `e16_cap`, by the same evaluator, on the rows both engines can encode |
| G4 | free PTH and QRY presence within T, both classes (K12a) |
| G5 | **all 8 slash rows within T (K12b)**, E3's second condition |
| G6 | no control state's P, Q or S honouring falls (E16's G6, with its amendment-1 rule); free-mode plan following is reported |
| G7 | queries: no cell OVER its natural rate, and none moving further from it than `e16_b5`'s (K12d) |
| G7-P | the same for paths (**new**) |
| G8 | K12d's whole-URL novelty rule, both modes |
| G9 | free FULL up by no more than 0.0141, both classes |
| G10 | K12e, and every validity check in §14.7 |

- **Why the guards stay under an absolute bar:**
  - The cheapest way to pass a real-words detector is to copy real URLs. G7, G7-P and G8
    close that route.
  - The rules' carry-over has to be shown, not assumed: G4–G6 and V-rules.

### 14.7 Validity (replaces §7 where it differs)

| code | check | on failure |
|---|---|---|
| V-T1 to V-T3 | as registered | stop |
| V-equiv | V2's trainer, run in character mode, equals `uie.train.train` bit for bit on the CPU for its first steps at toy scale (the shared code path) | stop |
| V-gen-equiv | as registered | stop |
| V-start′ | as registered. A failure on Kaggle falls back to a fresh seeded start, which is recorded (§14.3) | — |
| V-batch, V-mass | exact, against the laptop's replay of `v2_hyb`'s own sampler and injector; the reference values go in Amendment 2 | INVALID |
| V-stream, V-accum, K12e | as registered | INVALID |
| V-rules | as registered, now in both modes | INVALID |
| V-gen | about 1,000 URLs across `v2_hyb`'s files, regenerated on the CPU of the scoring session (not the GPU machine that made them), match byte for byte | INVALID |
| **V-env** (new) | the scoring session records its package versions. `e16_b5` is scored again there, free and control, and every comparison uses numbers from that one run. The largest difference from `e16_b5`'s published cells is reported. The laptop's versions are Python 3.14.3, scikit-learn 1.9.0, numpy 2.4.4 and scipy 1.18.0 | published and Kaggle numbers are never mixed |
| V-D1′ | in the same run, `e16_b5`'s whole-URL D1 and D0 are reported against E18's published values (+0.1675 / +0.1637; +0.0269 / +0.0437) | reported |
| V-floor | as registered | that reading is VOID |

- **The checks run where the files are.**
  - VERIFY_TRAIN runs as the first cells of `V2P1_generate`, and VERIFY_GENERATE as the first
    cells of `V2P1_score`. Each STOPs on failure.
  - No laptop round trip sits between the notebooks. The laptop re-reads every record
    afterwards.

### 14.8 Predictions (replaces §9) [HYPOTHESIS]

| reading | prediction |
|---|---|
| R0: `e16_b5`'s D1-R | CONVICTED in benign (~90%); in phishing (~60%) |
| `v2_hyb`'s D1-R against `e16_b5`'s (reported) | benign falls by 30–60% |
| `v2_hyb`'s D1-R CLEARED | benign ~25%; phishing ~35%, where not VOID |
| guards | G4–G6 pass (~90%); G7 and G7-P (~80%); G1 and G2 (~65%, now over 32 cells); G3′ (~85%); G9 (~80%) |
| **overall** | SUCCESS ~10–15%; PARTIAL ~20%; NOT MET ~40%; any reading WITH COST ~20%; INVALID or UNREADABLE ~5% |

- **Stated plainly:** the absolute bar is the right target, and a first-shot pass is
  unlikely. R0 (30 laptop minutes, before any code) will show how far `e16_b5` is from the
  bar, and it is reported before anything is built.

### 14.9 Cost (replaces §10)

| where | what | about |
|---|---|---|
| Kaggle GPU (T4) | `V2P1_train`: `v2_hyb`, plus `e16_cap`'s fit by the same evaluator | 2.5–3.5 h |
| Kaggle GPU | `V2P1_generate`: VERIFY_TRAIN, then free and control mode, 50 files | 2–2.5 h |
| **GPU total** | | **about 5–6 GPU-hours** (was about 8) |
| Kaggle CPU (no GPU allowance used) | `V2P1_score`: VERIFY_GENERATE; for `v2_hyb` and `e16_b5`, the gate's 32 cells, the slash rows, presence, following, copying and novelty; the word detectors for `v2_hyb`, `e16_b5` and `e15_new` | 3–5 h |
| laptop | R0 (0.5 h); learning the pieces (1–4 h); encoding and the replay references (about 1 h); equivalence tests and notebook simulations (about 1.5 h); reading the results (about 1 h) | **about 5–8 h**, nearly all of it unattended preparation before the first Kaggle run |

- **The basis:**
  - `e16_cap`'s training loop took 2.73 h, and `e16_b5`'s 50 files took 2.53 h to generate.
  - One engine's full scoring took 96 min on the laptop with three processes (free 46 min,
    control 95 min). A Kaggle CPU session has 4 cores, slower than the laptop's, so the
    estimate allows 1.5–2× per job, for two engines.
- **For the owner:**
  - upload one dataset: the code bundle, which carries `e16_b5`'s 50 files (8 MB) and
    `e15_new`'s free files (3.4 MB). The corpus is already on Kaggle;
  - run three notebooks in order (train, generate, score), attaching each one's output to
    the next, and E16_train's output to the first;
  - download the score output, which is small.

### 14.10 Steps and gates (replaces §11)

0. **Gate A:** the owner approves the registration as amended.
1. **R0, on the laptop (30 min):** `e16_b5` and `e15_new` on D1-R and D1-H; N_R. Reported to
   the owner before anything is built.
2. **Code and preparation,** committed before any Kaggle run: as §11 step 2, without the
   twin.
3. **Amendment 2 (gate B):** the frozen numbers, as §11 step 3, plus the long-row count and
   the replay references.
4. **Kaggle:** `V2P1_train`, then `V2P1_generate`, then `V2P1_score` (CPU).
5. **Laptop:** re-read the records, then write `V2P1_RESULTS.md`, read once, after every
   validity check passes.
6. **Gate C:** the owner decides.

- **Timeline:** about 2–3 days from gate A.
  - About a day of code and preparation, much of it long unattended compute.
  - Then the three Kaggle runs back to back: about 8–11 h of machine time, plus waits of up
    to about 50 min for a GPU.

## 15. Gate A (2026-10-09): approved, with the step-by-step workflow

### 15.1 The owner's message, verbatim

> You have a GO for Gate A. Please start the 30-minute reading on e16_b5 and report back.
> Regarding the workflow: I actually prefer our old step-by-step verification method. I do
> NOT want to chain the Kaggle notebooks back-to-back blindly. I want to download the
> training results, run the local laptop VERIFY_TRAIN to ensure everything is perfect, and
> only then proceed to the Kaggle generation. The same goes for generation before scoring.
> Please keep the local verification scripts and the workflow exactly as we did in E16 and
> E17. I like to check things myself before moving to the next step.", also V1 WAS CALLED
> WITH E like E9 E16 AND SUCH THESE ONES WHAT WILL CALL THEM the first one will be what 1

### 15.2 The workflow: E16's and E17's, step by step

§14.7's last bullet ("The checks run where the files are") is withdrawn. Each Kaggle run waits
for the laptop's check of the one before it:

1. `V2P1_train` on Kaggle. The owner downloads its output, and the laptop runs VERIFY_TRAIN
   (`--step train`, as `e16_verify.py` and `e17_verify.py` did). VERIFY_TRAIN also recomputes
   the training notebook's fit numbers on a sample of validation rows, on the laptop CPU.
   **The owner runs `V2P1_generate` only after it passes.**
2. `V2P1_generate` on Kaggle. Download; the laptop runs VERIFY_GENERATE (`--step generate`:
   byte identity on the laptop CPU, as in E17, and rule integrity). **The owner runs
   `V2P1_score` only after it passes.**
3. `V2P1_score` on Kaggle (CPU). Download; the laptop checks the score output's provenance
   (its inputs are the verified files, the code commit, the package versions), then reads
   the result.

- V-gen moves back to the laptop CPU, where E17 ran it (§14.7 had put it in the scoring
  session).
- **Cost:** GPU and CPU-session hours are unchanged. The laptop gains VERIFY_TRAIN (about
  1.5 h) and VERIFY_GENERATE (about 0.7 h): **about 7–10 h** in all. The calendar becomes
  **about 3 days** from gate A, because each step waits for a download and a check.

### 15.3 R0, fixed before it runs

`research/v2/v2_r0.py`, committed with this record:
- **D1-R real samples:** per class and generation g, the gate's whole-URL validation pool,
  one URL per host by E18's own selection (`seed_of(class, g)`). URLs with a path body,
  query or fragment are kept and shuffled with seed 20261022 + 100 × class + g. A, B and C
  are consecutive blocks of N_R.
- **Source samples:** N_R such URLs from each free file, seed 20261022 + 10,000 × (1 +
  source) + 100 × class + g, with `e16_b5` source 0 and `e15_new` source 1. D1-H's
  `e15_new` sample adds 1,000 to its seed.
- **D1-H:** E18's cached samples (A, B, C and its `e16_b5` sample), hosts only; for
  `e15_new`, 6,000 URLs drawn from its files.
- **Folds:** E18's rule, `seed_of(class, g)` × 10 + the task index (floor 0, `e16_b5` 1,
  `e15_new` 4).
- **V-D1:** E18's whole-URL D1, rerun on E18's cached samples, must reproduce E18's record
  exactly. If it does not, the pipeline has drifted and R0 is not read.
- **The reading,** as §14.4:
  - per class, `e16_b5`'s D1-R verdict: CONVICTED means the instrument sees the tell and
    the class can be read;
  - its clear line, `e15_new` against `e16_b5`, D1-H, and descriptive word statistics.
- **Early stop:** if `e16_b5`'s D1-R is CONVICTED in neither class, P1 stops and returns to
  the owner.

### 15.4 The naming: decided

- The owner, the same day, verbatim: "oh I see you called V2 Experiments as P thats cool".
- **V2's experiments are numbered P1, P2, P3, …**, written "V2-P1" in full. This pilot is P1.
- V1's E-numbers (E1 to E18) stay as they are and are not continued, so no V2 name can be
  mistaken for a V1 one (for example "E3", which is also V1's finish line).
- Code, notebooks and results follow the same names: `research/v2/v2_*.py`, `V2P1_*.ipynb`,
  `V2P1_RESULTS.md`.

## Status

- **2026-10-09:** registered; documents only.
- **2026-10-09, the same day:** Amendment 1 (§14), the owner's redesign, added before gate A.
- **2026-10-09: gate A APPROVED** by the owner, with the step-by-step workflow (§15).
- **2026-10-09: R0 read** (`R0_RESULT.md`).
  - V-D1 PASS.
  - `e16_b5`'s D1-R is CONVICTED in both classes (benign +0.1924, phishing +0.0997), so
    both classes are readable and the early stop does not fire.
  - N_R: benign 5,300, phishing 2,400.
  - **Next, on the owner's word:** the code and preparation (§14.10 step 2).
- **2026-10-09: implementation PAUSED by the owner** before any code, for a scientific review
  of the plan: `REVIEW_V2.md` (not a registration). Nothing is built or run while the owner
  decides on it.
- **2026-10-10: superseded, never built or run.** After R1–R3, the owner chose a learned
  memory layer for P1. Its registration is `PREREG_V2_P1_MEMORY.md`, with the owner's
  decision verbatim. This document is kept unchanged as the record of the tokenizer plan.
