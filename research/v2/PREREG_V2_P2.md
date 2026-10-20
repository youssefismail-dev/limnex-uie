# V2-P2 — word pieces in two arms: the path, query and fragment (A), and the host too (B)

- **Registered 2026-10-13: gate A approved by the owner** (verbatim in Status), before any
  V2-P2 code. The text above Status is the registered text; later changes go in
  amendments.
- **The owner's gates:**
  - **A:** approve this registration. That starts the laptop work: code, piece tables,
    checks and bundles.
  - **B:** approve Amendment 1 (the frozen numbers: the tables, their checks, C, the
    bundles). That starts Kaggle.
  - **C:** the reading, and what follows.
- **Outside E3.** E3 stays CLOSED. `e16_b5` stays `BASELINE` and read-only. Nothing here
  changes a published number.
- **Builds on:**
  - `DESIGN_V2.md` (the hybrid tokenizer);
  - `REVIEW_V2.md` §3 and §5.3 (the two arms, the host guards);
  - R1, R2 and R3 (the diagnosis, the honest judge, the ceilings);
  - `PREREG_V2_P1_MEMORY.md` (the judge, the guards, the tiered copy rule);
  - `WORDS_INVESTIGATION.md` (why pieces, and §8, why the host is in and how piece tables
    memorise).

## 0. In plain words

- **The question.** Does the engine write fewer made-up words when the path, query and
  fragment (arm A) — and the host too (arm B) — are written in learned word pieces instead
  of single letters? And does it do so without copying more, without losing creativity, and
  without breaking what works?
- **Two new engines, trained the same way,** compared with the frozen `e16_b5`:
  - **A:** pieces in the path, query and fragment; the host in letters, as today;
  - **B:** the same pieces, plus pieces in the host (names and subdomains). The suffix stays
    in letters.
  - Same data, same batches, same start, same recipe and decode. So A against B shows
    exactly what host pieces add.
- **The piece tables learn vocabulary from independent sites, not from pages.** Revised
  after the owner's review; §4.2 records why the first draft's rules failed.
  - R1's host table held whole real phishing hosts as single pieces (for example
    `turkiyecumhuriyetiziraatbankasi`), because it counted every page.
  - **A fragment's weight is now the number of independent sites that use it.** A site on a
    shared platform is its own site, and a campaign of lookalike hosts on one platform counts
    once.
  - **No length cap and no frequency threshold.** Rare and long words keep their natural
    place; only the table's size limits it, as in standard word pieces.
  - **In training, strings are sometimes split differently** (BPE-dropout), so the engine
    learns how pieces are built instead of memorising them whole.
  - **Domain names are taught by the name corpus only,** one URL per site, as the engine was
    originally designed. Page rows no longer rehearse each site's name up to 40 times.
  - An audit verifies the tables before anything trains.
- **The pass marks are the gold standard,** proven below 0.05 as in E3:
  - **A:** the honest word judge on the path, query and fragment (D1b-R) is CLEARED in
    each class;
  - **B:** the same judge on the whole URL (D1b-W) is CLEARED in each class.
  - A smaller but clear cut, at least 30% and statistically sure, reads PARTIAL.
- **The guards decide too.** Every copying guard stays, plus host guards for B:
  - copies of real domains and real full hosts, against the natural rate;
  - live phishing hosts;
  - names that are one site's own, kept strict;
  - lookalike (typosquat) names, at the real rate;
  - cliché and repeated names;
  - validity.

  A pass with a failed guard reads **WITH COST**.
- **Cost:** about 10–12 GPU-hours for four Kaggle runs, plus laptop scoring, mostly
  unattended. About 5–7 days from gate A.
- **My honest odds:**
  - a clear cut with every guard passed (PARTIAL or better): about 50% for A and 40% for B;
  - the gold standard: about 10% for A and 5–8% for B.

## 1. The owner's decision, verbatim (2026-10-13)

> I approve this revised direction. Proceed with drafting the new preregistration for both
> Arm A (paths/queries/fragments) and Arm B (including hosts).
> Ensure the draft explicitly incorporates the new piece-table generation rules to prevent
> memorization (per-site counting, length cap, no-whole-string check) and the rigorous host
> guards we discussed.
> Send me the draft for review. Do not write any code or start any Kaggle runs until I
> officially approve the preregistration

### 1.1 The owner's review of the first draft, verbatim (2026-10-13)

> I am not approving or rejecting any of the decisions in Section 17 right now.
> Looking closely at your proposed piece-table rules (like the 10-site minimum and the
> 16-character cap), it is clear this design has multiple major logical flaws. These rules
> seem like lazy patches that actively work against our primary objective of capturing and
> preserving rare long-tail vocabulary.
> I want you to step back and rigorously review your entire piece-table design against our
> actual goals. Identify the flaws and contradictions in these constraints yourself,
> explain why they fail our objective, and propose a structurally sound solution that
> prevents domain memorization without destroying naturally rare or long words.
> Review your own design, find the flaws, and fix them before we discuss any approvals

The review and the fix are §4.2 to §4.5.

**The owner's standing conditions** (from the GO of 2026-10-10, carried over):
- no word list or lexicon chooses the output, no template matching, no pretrained model;
- every copying guard, the copier control and the honest judge;
- `e16_b5` and every accepted E3 result frozen;
- preregistration before any result;
- a pass needs reduction AND preservation, never the judge alone;
- a full report of configuration, resources, results, guards and deviations, then STOP.

## 2. Why this design

| evidence | what it says | source |
|---|---|---|
| the tell is the long tail of real words | benign paths write words seen 1–99 times at less than half the real rate (12.7% against 28.8%), and fill the gap with inventions | R1; `WORDS_INVESTIGATION.md` §1 |
| letters cannot hold that tail | `e16_cap` scores words seen 1–9 times almost like unseen ones (2.19 against 2.45 nats per letter). The training vocabulary holds 1.49M word types seen 1–9 times | `WORDS_INVESTIGATION.md` §1 |
| pieces model that band almost as well as a word list | words seen 1–99 times: 1.70 bits per character for pieces, 1.64 for a word list, 2.08 for letters. Pieces beat the word list on all 7 multilingual languages tested | Mielke & Eisner, 2019 |
| phishing's largest tell is the host | D1-H +0.147 against D1-R +0.100. R1's tables hold 51–59% of phishing host words whole, against 34–38% of benign host words and 32–42% of benign path words. Only phishing paths are held better (54–66%) | R0; R1-4 |
| forcing whole real words into names backfires | R2's blind oracle doubled the phishing host tell; R3 raised domain copies from 1.4% to 3.9% | R2, R3 |
| a store that rewards repetition memorises sites | P1b's memory recalled training site names; R1's draw-weighted piece tables hold whole sites as single pieces (host 20.7% of pieces; 184 of 189 heavily drawn strings) | `V2P1B_RESULTS.md`; `WORDS_INVESTIGATION.md` §8 |
| draft-and-revise and decode-time memories are not the route | they cannot add knowledge; R3 leaked copying | `WORDS_INVESTIGATION.md` §2 |

## 3. The arms

| arm | path, query, fragment | name (LBL) and subdomains (SUB) | suffix, userinfo, port |
|---|---|---|---|
| **`p2_a`** | pieces (the rest table) | letters | letters |
| **`p2_b`** | pieces (the same rest table) | pieces (the host table) | letters |
| `e16_b5` (frozen reference) | letters | letters | letters |

**Identical in `p2_a` and `p2_b`:**
- **The rows:** `e16_cap`'s. A training or validation row is kept if and only if its
  *character* encoding fits 384 tokens. Stream 1 (`637accab…`) is added at rate 1.
- **The batches:** sorted by character length, so both arms draw the same rows in the same
  batches, and `e16_cap`'s batch digest `adbddc2a…` holds for both. This reverses V2-P1's
  Amendment 1 choices (long rows, own batches): exact matching is worth more than 0.14% of
  rows.
- **The start:** `e16_cap`'s start (`caca0107…`) plus the piece rows.
  - The rest pieces' rows come from one seeded generator, identical in both arms.
  - `p2_b` adds its host pieces' rows from a second seeded generator.
  - So `p2_a`'s weights are a subset of `p2_b`'s.
- **The recipe:** `E16_B5.recipe`: 20,000 steps, batch 256 in 2 parts (4 if the memory
  check fails at 2), the same rate, schedule, clipping, optimiser, class α, frame, presence
  plan, root slash, slash plan and the fixed objective. `final.pt` is read, never `best.pt`.
- **Two training changes, in both arms:**
  - names are taught by the name corpus only (§4.5);
  - BPE-dropout, p = 0.1, on piece fields (§4.4).
- **Generation:** E12's 50 jobs (free and control) at E12's seeds, with `e16_b5`'s decode
  (raw T 1.0 + D1 + D1′ + B5) plus the piece rule.

**Different, and only these:**
- **`p2_a` against `e16_b5`:** the rest table, BPE-dropout and the name scope, and what
  follows from them (embedding rows, C, the piece rule).
  - Their separate shares are not isolated.
  - A third arm, letters with the name scope, would isolate the scope for about 5 more
    GPU-hours. It is offered in §17 and is not part of this draft.
- **`p2_b` against `p2_a`:** the host table and what follows from it. This comparison stays
  exact.

## 4. The piece tables (revised after the owner's review; §4.2 records the first draft's flaws)

### 4.1 What a piece table is for

- **Its one job is vocabulary:** the fragments a model needs to write the words of *new*
  sites, rare and long ones included.
  - Every reading uses held-out sites only: benign is split by domain, phishing by host.
  - So the right measure of a fragment is **how many independent sites use it**, not how
    often it appears.
- **Memorisation is not the table's job.** It arises in the model, from training on
  repeated strings.
  - `e16_cap` memorises page-row site names in plain letters, with no pieces at all. P1b's
    diagnosis: 0.60 nats per letter on sites it trained on, 3.23 on new ones.
  - It is prevented where it arises (§4.5) and measured by the guards (§9).

### 4.2 The first draft's rules, reviewed: why they failed the objective

| first draft's rule | the flaw | what it did to the objective |
|---|---|---|
| **P-3: a fragment needs 10 sites** | An arbitrary threshold, applied to exactly the band we must capture. Words seen 1–99 times are, by their nature, used by few sites | It cut the long tail out of the vocabulary to block a hazard that counting by site already removes (§4.3) |
| **P-2: no piece longer than 16 characters** | Arbitrary, and wrong in both directions. It blocks long real words: compounds, multi-word slugs, many of the engine's 159 languages. It cannot see a short memorised name at all: one site's 8-letter name passes a 16-character cap | R2 found the engine's inventions are *longer* than real new words. Long words are where letter-by-letter writing fails most, and the cap withdrew help exactly there |
| **P-4: no piece may equal a whole string fewer than 10 sites use** | Real names are often real words (`northvale`, `shopwave`). A word that is one site's entire name but common inside other strings is vocabulary, not a record | It banned real words for being somebody's name |
| **"a shared platform is one site"** | `snargudaryan.github.io` and `blcatfreedom.github.io` are different sites run by different people. The Public Suffix List's private section exists to say exactly that | Phishing sites crowd onto platforms, so this starved phishing hosts of vocabulary: the very tell arm B exists to fix |
| **one threshold for both classes** | Phishing has far fewer sites than benign, so the same absolute bar is far harsher on it | Again against arm B's purpose |
| **all four together** | They act on the table only. Training still rehearses each multi-page site's name up to 40 times, once per page row | They left untouched the mechanism that memorises names, so they could not deliver even their own purpose |

**What was right, and is kept:** counting by site instead of by page. R1's tables counted
pages, and that is what made whole sites into pieces.

### 4.3 The corrected table design

1. **The independent source.**
   - A row's source is its registrable domain under the full Public Suffix List, private
     section included (tldextract's bundled snapshot, hashed in Amendment 1). Each
     platform-hosted site (`name.github.io`) is its own source.
   - The fields themselves are unchanged; only the counting uses this unit.
   - **Within one shared platform, near-duplicate sites are one source,** by Lee et al.'s
     (2022) deduplication rule:
     - candidate pairs come from MinHash over the character 3-grams of each site's own
       label (the one left of the platform suffix);
     - a pair is confirmed when its edit similarity is above 0.8;
     - the clusters are the connected components.
   - **What this does:**
     - a campaign that spins up hundreds of lookalike subdomains on one platform
       (`case10009zwpol0lm…`) counts once;
     - unrelated sites that share a word (`paypal-secure-login`, `paypal-account-verify`;
       edit similarity about 0.4) stay separate.
   - **Why only within a platform:**
     - Across separately registered domains, identical names are normal and independent:
       `shop.de` and `shop.fr` are two sites. Merging them would make every common word
       count as one source and keep it out of the table.
     - Within a platform each subdomain name is unique, so near-duplicates there are spun
       up, not chosen independently.
2. **The count: source frequency.**
   - A pair of adjacent pieces counts once for each source in which it occurs, however
     many pages, rows or strings of that source contain it.
   - Merges take the highest count first. Ties go to the pair whose two pieces come first
     in code-point order (`DESIGN_V2.md` §4.3).
3. **Both classes on an equal footing.**
   - Counts are taken per class, as a share of that class's sources, then combined with
     each class's training draw share.
   - A fragment used by 5% of phishing sources then weighs what a fragment used by 5% of
     benign sources weighs, scaled by how often training draws each class.
   - Phishing's smaller vocabulary is no longer drowned out by benign's.
4. **No length cap, no frequency threshold, no banned strings.**
   - The merge budget is the only limit, as in standard word pieces: R1's registered sizes
     (DR2), 32,000 merges for the rest table and 16,000 for the host table.
   - A piece forms only from adjacencies that many independent sources share. A long word
     that many sites use can therefore become a piece.
   - One site's own string, a platform campaign's stem or a site's junk cannot: each counts
     as one source. If many independent sources share such a string, it is vocabulary by
     this definition. Copying it into an output is then still judged by the guards (§9).
5. **One natural stop:** learning ends early if the best remaining pair occurs in only one
   source. A fragment no two sources share is not vocabulary. That is what "shared" means,
   not a tuned value.

**Unchanged from the first draft:**
- the learning data (the rest table: path, query and fragment bodies of the kept training
  rows plus stream 1's query bodies; the host table: registrable and subdomain labels, one
  label per sequence);
- the alphabet (every single character stays a piece);
- no pre-splitting inside a field.

### 4.4 Training with subword regularisation (BPE-dropout)

- **What:** every time a row is drawn in training, its piece fields are split with
  BPE-dropout at p = 0.1 (Provilkov et al., 2020). Each merge is skipped with probability
  0.1, so the same string is seen in several segmentations. Generation uses no dropout.
- **Why:**
  - No string is ever learned only as one atomic unit. Every piece stays spelled out by its
    parts in the model's experience. A memorised whole-string shortcut is weaker, and rare
    pieces are learned together with the pieces they are made of.
  - In translation, BPE-dropout kept rare tokens integrated with the rest of the
    vocabulary instead of isolated, and made models more robust to misspelled input.
    Kudo (2018) found the same idea, with a unigram model, helps most on out-of-domain
    data. New sites are our out-of-domain data.
- **p = 0.1** is the paper's standard value, not tuned here.
- **Reproducible:** the splits come from a seeded generator keyed on the step and the row.
  Batches stay keyed on character length, so V-batch holds.
- **Fit** is still read in nats per character under the canonical split. That is
  conservative for the piece engines.

### 4.5 Where memorisation is prevented

| layer | what prevents it | evidence |
|---|---|---|
| **the training signal** | **Names are taught by the name corpus only.** In page rows, the letters (or pieces) inside the suffix, name, subdomain and userinfo fields are context, not targets. Markers, the plan tokens and the end token stay trained on every row, as `field_mask` always trains them. This restores the field scope the engine was designed with: in `uie/train.py`'s words, names "want every domain weighted equally", and the page corpus "covers only multi-page sites ... so its NAMES are not representative". Turning on the frame token switched the scope off, so each multi-page site's name was trained up to 40 times. Generation uses only the domain frame, so nothing generated loses a teacher. **Both arms** | P1b's diagnosis: page-row names score 0.60 nats per letter on trained sites against 3.23 on new ones (`e16_cap`). Deduplicated training data emits memorised text about ten times less (Lee et al., 2022). Memorisation grows with how often an example is duplicated (Carlini et al., 2023) |
| **the table** | source frequency, with near-duplicate sources merged (§4.3) | the hazard check: counting pages made whole sites into pieces |
| **the model's use of pieces** | BPE-dropout (§4.4) | Provilkov et al., 2020 |
| **the output** | the copy guards decide (§9): whole URLs, registrable domains, full hosts, live hosts, names and subdomains of a single source | — |

### 4.6 Encoding, ids, the piece rule

- **Encoding is canonical:** merges are applied in the order they were made, so each string
  has exactly one canonical encoding (the one generation's likelihood and fit use).
- **Ids:** every `e16_cap` token keeps its id. The rest pieces follow, then (in `p2_b` only)
  the host pieces.
- **The piece rule at decode** (`DESIGN_V2.md` §4.4):
  - a rest piece may be sampled only in PTH, QRY and FRG;
  - a host piece only in LBL and SUB;
  - single characters stay legal wherever they are legal today.

  The rule forbids no string, only token paths. The mass it removes is recorded per field.
- **Lossless by requirement:** for every training, validation and stream URL, each arm's
  tokenizer, under any dropout split, gives back exactly what `e16_cap`'s gives back. Zero
  failures.

### 4.7 The audit before training (V-T4)

Run on the laptop on the final tables. It is reported to the owner in Amendment 1, before
any Kaggle run.

| required | bar |
|---|---|
| every merge occurs in at least two independent sources | true by construction; verified |
| losslessness of both tables, canonical and under dropout | zero failures (V-T1) |

**Reported, against R1's draw-weighted tables** (`v2_piece_hazard.py`'s measures, R1-4's
coverage):
- the smallest source count among the merges made (the floor the budget implies), and
  whether the budget bound;
- pieces that are a whole training string of a single source (expected near 0);
- heavily drawn strings that became one piece (R1: 184 of 189 host, 208 of 210 rest);
- the near-duplicate clusters: how many, the largest, examples;
- coverage of real held-out words by frequency band (never seen, 1–9, 10–99, 100+), per
  field and class. Expected: not below R1's tables in the 1–99 band;
- long words: real held-out words over 16 letters, and how many pieces each needs;
- phishing host words: coverage against R1's host table.

**If a reported measure looks wrong** (for example, coverage of the 1–99 band below R1's
tables), it goes to the owner at gate B with the evidence. Nothing is patched with a new
threshold.

## 5. Training

- **One notebook per arm,** `V2P2_train_a` and `V2P2_train_b`. They may run at the same time.
- **Before training, each checks:**
  - the bundle's manifest;
  - torch and the GPU;
  - V-start′: the width-384 rebuild against `e12_final`'s record, and the width-512 start
    against `caca0107…`;
  - that removing the piece rows gives back `caca0107…` exactly;
  - the worst-case memory check at `max_len` 384 (2 parts, else 4).
- **C** is re-derived per arm by E9's rule: batch size × the expected scored-token mass per
  drawn row, old rows only. It is taken under the arm's own encoding, the name scope and
  BPE-dropout, from a seeded sample of draws, and recorded in Amendment 1.
- **The training record:**
  - the per-field validation NLL in nats per *character* (canonical split), so fit
    compares fairly with `e16_cap`;
  - the per-field train/validation gap;
  - the clipping bands.
- **Read at VERIFY_TRAIN, reported:** the page-row name gap, P1b's measure (domain names on
  page rows of sites the engine trained on, against new sites). It shows whether the name
  scope removed the memorisation it targets.

## 6. Generation

- **One notebook per arm,** `V2P2_generate_a` and `V2P2_generate_b`, each run only after the
  laptop's VERIFY_TRAIN for that arm has passed.
- **E12's 50 jobs per arm,** free and control, at E12's seeds, with the presence-slash and
  suffix tables. Decode: `e17_common.decode_kwargs(True)` plus the piece rule.
- **How the rules carry over in `p2_b`, whose host is now in pieces:**
  - **D1 and D1′** act at every step where a host-ending token (`<PTH>`, `<RS>`, `<QRY>`,
    `<FRG>`, `<EOS>`) can be sampled. With host pieces, those steps fall at piece
    boundaries. The guarantee, S honoured exactly on every covered plan, is unchanged and
    checked from the files.
  - **B5** forbids ending, or starting a fragment, before the query, at every step from
    `<LBL>` to `<QRY>`. Host and path steps are now piece steps, constrained exactly as
    before.
- **Recorded per file:** the mass D1, B5 and the piece rule moved, and the plan traces.

## 7. Instruments

**The honest judge, D1b** (`PREREG_V2_P1_MEMORY.md` §4.1, unchanged), in three views:

| view | strings | N per side |
|---|---|---|
| **D1b-R** | the URL with its host replaced by `h.h`, over URLs with a path body, query or fragment | benign 5,300, phishing 2,400 |
| **D1b-H** | the host alone | 6,000 |
| **D1b-W** | the whole URL | 6,000 |

- **The samples:** real and `e16_b5` exactly as R2 Part A's.
- **Each arm** is drawn from its free files by R0's rule, as source index 3 (`p2_a`) and 4
  (`p2_b`), with seed 20261022 + 10,000 × (source + 1) + 100 × class + g. D1b-H adds 1,000
  and D1b-W adds 2,000.
- **Folds:** E18's rule, at task indices 13 (`p2_a`) and 14 (`p2_b`).
- **The reading:** x = AUC(real C against the source) − AUC(real A against real B), the mean
  over g = 0…4, read like a gate cell: t = (x − 0.05)/SE against ±2.776.
- **If an arm's file holds fewer rest-bearing URLs than N_R,** the memory phase's rule applies
  (`PREREG_V2_P1_MEMORY.md` §4.1): every source's D1b-R samples in that class are cut to the
  largest multiple of 100 that fits, and the cut is recorded as a deviation.

**Also read:**

| reading | role |
|---|---|
| **D1b-x** (the judge with its vocabulary restricted to words found in both halves) on R, **and new here on H and W** | the blind-spot check (§9) |
| W1 (half-B-only URLs), views R and H | guard (§9) |
| the copier control O4 on D1b-R | rerun in the same run; must reproduce R2 (benign +0.0098, phishing +0.1524) |
| D1 on R and W; D0 on W; the word bands | reported |
| D4 (word order) on R, **and new here on H for `p2_b`** | guards G-D4 and G-D4-H |
| the gate (`e17_score`'s parts): 32 cells, FULL, 8 slash rows, presence, plan honouring, whole-URL novelty, query copying | guards |
| R3's measures (first 1,600 URLs of each free file: 8,000 per class): URL, registrable-domain, label, path and query copies; word-pair novelty; words new to the judge; distinct words per 1,000; duplicates; validity | guards; natural rates from 8,000 real held-out URLs per class |
| **new host measures** (§9.6) | guards for `p2_b`, reported for `p2_a` |

**V-D1b:** in the same run, `e16_b5`'s readings reproduce R2 Part A exactly (benign R
+0.1720, H +0.0581, W +0.1500; phishing R +0.0857, H +0.1019, W +0.1209). Otherwise nothing is
read.

## 8. Criteria (fixed now)

### 8.1 The pass mark, per arm and class

- **K1-A:** `p2_a`'s D1b-R is **CLEARED**: (x − 0.05)/SE < −2.776.
- **K1-B:** `p2_b`'s D1b-W, the whole URL, is **CLEARED**.
  - Recorded beside it: "host CLEARED" (D1b-H) and "rest CLEARED" (D1b-R).
- **VOID:** a class's view is VOID if, in this run, `e16_b5` is not CONVICTED on it, or
  V-floor fails. A judge that cannot see the old engine's tell would clear any engine.

### 8.2 The effect, per arm and class

- **K2,** read per view, paired by generation against `e16_b5`. On a view, both must hold:
  - the mean cut is at least 30% of `e16_b5`'s excess;
  - the paired t is ≤ −2.776.
- **The arm's K2 holds in a class** if it holds on one of the arm's own views (A: D1b-R;
  B: D1b-H or D1b-W). The readout names the view.
- **B's host gain** (reported, never deciding): B against A on D1b-H, paired by g. It is
  recorded when the difference is ≤ −0.0141 with t ≤ −2.776. This is exactly what host
  pieces add.
- **A's specificity** (reported): A against `e16_b5` on D1b-H is within 0.0141 or not
  significant. Host words written in letters should not move.

### 8.3 The blind-spot condition

Any K1 or K2 that passes under D1b must also pass under D1b-x on the same view, or it is
read as not passed and recorded "blind spot". This is the memory phase's check, extended to
the H and W views.

## 9. Guards: each arm against `e16_b5`, from the same scoring run

- T is the programme's tolerance, 2√(p(1−p)/n), with p the threshold's own rate.
- Both arms are retrained engines, so the retraining allowance 0.0141 applies where marked.
- **The tiered copy rule** (`PREREG_V2_P1_MEMORY.md` §12, the owner's framework) carries
  over unchanged, and is extended to hosts in §9.6. The tiered reading decides; the strict
  reading is always printed beside it.

### 9.1 Structure and the gate (existing thresholds)

| code | guard |
|---|---|
| G1 | no gated cell, free or control, has its verdict worsen with x up by more than 0.0141 |
| G2 | no PTH, SUB or LBL cell, free or control, rises by more than 0.0141 |
| G4 | free PTH and QRY presence within T, both classes |
| G5 | all 8 slash rows within T |
| G6 | no control state's P, Q or S honouring falls |
| G9 | free FULL up by no more than 0.0141, both classes |
| G-D4 | word order: D4 on R up by no more than 0.0141 |
| **G-D4-H** (`p2_b`) | word order in hosts: D4 on H up by no more than 0.0141 |

### 9.2 Fit

| code | guard |
|---|---|
| G3 | no non-query field's validation NLL, in nats per character, rises by more than 0.01 against `e16_cap`, from the training records |

### 9.3 Copying of complete units (tiered, as the owner's framework)

| code | guard |
|---|---|
| G7 | queries: K12d, with the tiered +0.05 under the diversity condition (`PREREG_V2_P1_MEMORY.md` §12.3); the once-seen-query row strict |
| G8 | whole-URL novelty: K12d's rule, both modes. **Strict** |
| G-COPY | whole URLs and registrable domains ≤ max(natural, `e16_b5`) + T (**strict**). Paths and queries ≤ max(natural, `e16_b5`) + T + 0.05, only under the diversity condition. Once-seen paths and once-seen queries ≤ max(natural, `e16_b5`) + T (**strict**) |

### 9.4 The judge's blind spot

| code | guard |
|---|---|
| W1 | half-B-only URLs ≤ max(real, `e16_b5`) + 0.10 in views R and H, only if G-PAIR and the blind-spot check (§8.3) hold in the class; otherwise ≤ max(real, `e16_b5`) + T |

### 9.5 Creativity and diversity (mandatory)

| code | guard |
|---|---|
| G-PAIR | word-pair novelty in paths ≥ real − T |
| C1 | distinct words per 1,000, hosts and paths: within ±5% of real, or no further from real than `e16_b5` plus 2 points |
| C2 | words new to the judge, hosts and paths, ≥ real − T: the engine must still invent |
| G-DUP | duplicates within the free files ≤ max(real, `e16_b5`) + T |

### 9.6 The host guards (new; deciding for `p2_b`, reported for `p2_a`)

From the same 8,000 URLs per class as §7's R3 measures. The natural rate is the same
measure on 8,000 real held-out URLs. Real held-out sites are never training sites: benign is
split by domain and phishing by host.

| code | guard | tier |
|---|---|---|
| **H-RD** | the share whose registrable domain is a training registrable domain ≤ max(natural, `e16_b5`) + T. This is G-COPY's domain row, restated for hosts | strict (owner's item 2) |
| **H-HOST** | the share whose **full host** (every label, and the suffix) equals a training URL's host ≤ max(natural, `e16_b5`) + T. It catches a real site regenerated on a shared platform, which the registrable domain cannot see | strict |
| **H-LIVE** | the share whose full host equals a host in the offline threat snapshot (300 URLs, hash recorded) ≤ max(real, `e16_b5`) + T. Matching full hosts, not registrable domains, fixes R1's L7: the old match counted shared platforms | strict |
| **H-LBL1** | the share whose registrable label is a label found in only one training source (§4.3's unit) ≤ max(natural, `e16_b5`) + T | strict: one site's own name |
| **H-SUB1** | the share containing a subdomain label found in only one training source ≤ max(natural, `e16_b5`) + T | strict: one site's own record |
| **H-LBL** | the share whose registrable label is any training label ≤ max(natural, `e16_b5`) + T + 0.05, only under the host diversity condition; otherwise + T | tiered (owner's item 3: common names may be reused) |
| **H-TYPO** | the share of host words that are lookalikes: one edit from a word seen 10+ times in training (R1-3's rule). Within T of real, or no further from real than `e16_b5` | two-sided: pieces must not crowd out typosquats, nor add them |
| **H-DUP** | the share of URLs whose full host repeats within its file ≤ max(real, `e16_b5`) + T | cliché names |
| **H-VAL** | the sample gate's host validity failures (labels over 63 characters, hosts too long, no public suffix, unparseable) ≤ `e16_b5` + T | validity |

- **The host diversity condition** (for H-LBL's room): C1's host row, C2's host row, H-DUP
  and G-DUP all pass in the class.
- **Why H-HOST and H-LIVE are new.**
  - Phishing reuses platform domains at a high natural rate (registrable domains 63.7% in
    R1-6), so a domain match cannot tell a new site on a platform from a regenerated one.
  - The full host can: real held-out hosts never match training hosts, by the split.

### 9.7 Validity of the outputs

| code | guard |
|---|---|
| G-VAL | the sample gate's validity failures, all reasons, ≤ `e16_b5` + T |
| G10 | K12e and every validity check in §12 |

## 10. The reading, per arm

| K1 over the views and classes that are not VOID | K2 | guards | reading |
|---|---|---|---|
| CLEARED in every class | — | all pass | **SUCCESS** |
| CLEARED in some class, or K2 in at least one class | — | all pass | **PARTIAL** |
| CLEARED in none | in no class | all pass | **NO EFFECT** |
| any of the above | — | one or more fail | that reading, **WITH COST (G…/H…)**, naming each failed guard |
| every class VOID | — | — | **UNREADABLE** |
| a validity check fails | — | — | **INVALID** |

- **Both readings are printed:** the tiered one decides, the strict one sits beside it. For
  example, "PARTIAL — strict: PARTIAL WITH COST (G-COPY)".
- **Reported with every reading, never deciding it:**
  - the other views (D1b-H for A; D1b-R for B);
  - B's host gain and A's specificity;
  - D1, D0, the word bands, the copier control;
  - the closure share per view: (`e16_b5`'s excess − the arm's) ÷ (`e16_b5`'s excess − the
    clear line);
  - fit, the piece rule's removed mass, tokens per URL;
  - the page-row name gap (§5) against `e16_cap`'s;
  - examples of each engine's words and hosts;
  - E3's finish line read on each arm, for the record.
- **SUCCESS is V2's bar, not E3.** Adoption of either arm is the owner's separate decision.

## 11. Stop conditions (fixed now)

| when | condition | what happens |
|---|---|---|
| laptop, before any bundle | V-T1 to V-T4 fail (losslessness, the tables, the audit's required rows) | STOP. Fixed before Kaggle, and recorded in Amendment 1 |
| Kaggle, before training | V-start′ fails, or the piece rows do not peel off to `caca0107…` | STOP. Nothing trained. The owner chooses |
| Kaggle, before training | the worst-case memory check fails at 4 parts | STOP |
| Kaggle, training | a non-finite loss the scaler does not skip | STOP and report. No restart with other settings |
| Kaggle, training | the session ends early | rerun as registered from the start; no resume |
| laptop, VERIFY_TRAIN | any validity check fails | that arm is INVALID; its generation does not run |
| laptop, VERIFY_GENERATE | any validity check fails | that arm is INVALID; no scoring |
| laptop, scoring | V4, V-D1b or the copier control fails to reproduce | STOP. Nothing is read until the pipeline is fixed |
| after the reading | always | the owner decides. No retry, no tuning, no follow-up without a new registration |

## 12. Validity

| code | check | on failure |
|---|---|---|
| V-iso | `uie/` byte-identical to `422cb23` (its current state). All P2 code lives in `research/v2/` | INVALID |
| V-T1 | losslessness, both arms, every training, validation and stream URL, canonical and under dropout splits | STOP |
| V-T2 | the tables were learned from training rows and stream 1 only (the files read, hashed) | STOP |
| V-T3 | the tables, the source clusters and the per-field allowed sets are hashed in Amendment 1, and equal on Kaggle | STOP |
| **V-T4** | the audit's required rows (§4.7) | STOP |
| **V-scope** | the training masks follow §4.5 exactly: in page rows, name-field content weight 0; markers, plan and end tokens weight 1; every other weight as `field_mask` gives it (a unit test on real rows) | STOP |
| **V-drop** | BPE-dropout is reproducible (same seed, same splits) and lossless (a unit test) | STOP |
| V-equiv | V2's trainer in character mode equals `uie.train.train` bit for bit on the CPU, at toy scale only | STOP |
| V-gen-equiv | V2's generator with `e16_cap`'s weights and `e16_b5`'s decode reproduces the start of `e16_b5`'s files byte for byte on the CPU | STOP |
| V-start′ | §5 | STOP |
| V-recipe | every setting equals `e16_cap`'s record, except the pieces' own, the name scope, BPE-dropout and, if the fallback ran, 4 parts | INVALID |
| V-batch | both arms' old-row batch digest over 20,000 steps equals `adbddc2a…` | INVALID |
| V-mass, V-stream | the appended stream rows and the stream hash equal `e16_cap`'s record | INVALID |
| K12e | clipping after step 1,000 at most 1% in every band | INVALID |
| V-rules | zero violations, from the files: D1 and D1′ (S honoured on every covered plan), B5, and the piece rule (no piece outside its fields) | INVALID |
| V-gen | at least 1,000 URLs per arm regenerated on the laptop CPU, at least 99% byte-identical, with every mismatch listed | INVALID |
| V4 | the scorer reproduces `e12_final`'s published cells exactly | STOP |
| V-D1b | as §7 | STOP |
| V-floor | every word-judge floor within 0.03 of 0.5 | that reading is VOID |

- **V-gen's bar is 99%, not 100%.** `e16_b5`'s dense engine regenerated 957 of 957. A
  softmax over tens of thousands of pieces has more near-ties that the GPU and the CPU can
  break differently. A wrong checkpoint, table, seed or decode diverges on nearly every URL.
- **Laptop rule:** nothing trains on the laptop. V-equiv runs at toy scale, and every other
  check is inference only, at low priority with the low-RAM alarm.

## 13. Predictions, before any measurement [HYPOTHESIS]

| reading | prediction |
|---|---|
| the audit (§4.7) | the required rows hold. Heavily drawn strings that are one piece fall from 184 of 189 (host) and 208 of 210 (rest) to under 10 each. Coverage of real held-out words seen 1–99 times is at least R1's tables' (about 70%) |
| the tables' sizes | both reach their budgets (rest about 85%, host about 70%) |
| the page-row name gap | falls by at least half in both arms, against `e16_cap`'s (about 75%) |
| tokens per field, against letters | rest fields 0.30–0.50 of their letters; hosts in `p2_b` 0.35–0.55 |
| `p2_a`, benign D1b-R | falls 25–60% (point about 40%); CLEARED about 10–15% |
| `p2_a`, phishing D1b-R | falls 10–40%; CLEARED about 20–25% |
| `p2_b`, phishing D1b-H | falls 15–50% (point about 30%); CLEARED about 15% |
| `p2_b`, D1b-W | benign CLEARED about 5–10%; phishing about 10% |
| B's host gain over A | recorded in phishing (about 50%), in benign (about 35%) |
| guards | host copy guards (H-RD, H-HOST, H-LBL1) fail in `p2_b` about 25%; G-COPY on paths about 20%; G1/G2 about 25%; C1/C2 about 15%; H-TYPO about 20% |
| **overall, `p2_a`** | SUCCESS about 10%; PARTIAL about 40%; NO EFFECT about 25%; any reading WITH COST about 20%; INVALID or UNREADABLE about 5% |
| **overall, `p2_b`** | SUCCESS about 5–8%; PARTIAL about 32%; NO EFFECT about 25%; any reading WITH COST about 30%; INVALID or UNREADABLE about 5% |

**Stated plainly:** the bar is far. R2's oracles and R3 did not reach it, and the
copier did only by copying. A clear PARTIAL would still show, for the first time, a
representation that moves the word tell inside the model, and by how much in each field.

## 14. Cost

| where | what | about |
|---|---|---|
| laptop, after gate A | the source clustering and the source-frequency learner, both tables, V-T1 to V-T4, V-scope, V-drop, the tokenizer, trainer and generator code, tests, the full suite, toy-scale checks, bundles, notebooks, run sheet | about 2–2.5 days, mostly mine; clustering and table learning about 2–4 h unattended (R1's standard learner took 15 min for the rest table) |
| Kaggle GPU (T4) | `V2P2_train_a`: `e16_cap`'s loop took 2.73 h. A 32,000-piece output layer and on-the-fly dropout splits add time; fewer tokens per URL saves some | 3–4 h |
| Kaggle GPU | `V2P2_train_b`: as A, with host pieces too | 3.5–4.5 h |
| Kaggle GPU | `V2P2_generate_a` and `_b`: 50 files, 180,000 URLs each (`e16_b5`'s took 2.53 h) | 2–2.5 h each |
| **GPU total** | | **about 11–13 GPU-hours** |
| laptop | VERIFY_TRAIN ×2 (about 1 h), VERIFY_GENERATE ×2 (about 2 h), scoring ×2 (about 7–8 h unattended), reading (about 1 h) | about 11–12 h, mostly unattended |
| downloads | per arm: `final.pt` (about 170 MB for A and 200 MB for B; `e16_cap`'s is 102 MB) and its record; the generated files (about 25 MB) | |
| **calendar** | | **about 5–7 days from gate A** |

## 15. Steps and gates

0. **Gate A:** the owner approves this registration (as revised in review). It is committed
   and pushed. Nothing is built before it.
1. **Laptop, code and preparation,** committed and pushed before any Kaggle run:
   - the source clustering and the source-frequency learner (§4.3), and both tables;
   - V-T1 to V-T4, with §4.7's audit;
   - the tokenizer;
   - the trainer: character-length batches, the name scope (§4.5) and BPE-dropout (§4.4),
     with V-scope and V-drop;
   - the generator with the piece rule;
   - V-equiv and V-gen-equiv;
   - the reading scripts' skeletons;
   - the full suite;
   - bundles, notebooks and the run sheet `V2P2_KAGGLE.md`.
2. **Amendment 1 (gate B):** the frozen numbers, added below and never edited above:
   - the table and cluster hashes, the tables' sizes, and §4.7's audit;
   - C per arm;
   - the bundle hashes;
   - the timings and memory.

   The owner approves the Kaggle runs.
3. **Kaggle, the owner:** `V2P2_train_a` and `V2P2_train_b`.
4. **Laptop:** VERIFY_TRAIN per arm. A failed arm stops there.
5. **Kaggle, the owner, per arm that passed:** `V2P2_generate_a`, `V2P2_generate_b`.
6. **Laptop:** VERIFY_GENERATE, scoring, the criteria, then `V2P2_RESULTS.md`, read once:
   configuration, resources, results, guards and deviations. Commit and push, then
   **STOP: the owner decides (gate C).**

The workflow is gate A's from V2-P1 (`PREREG_V2_P1.md` §15): each Kaggle run waits for the
laptop's check of the run before it, and nothing is chained.

## 16. Unchanged and closed

**Unchanged:**
- `e16_b5`: its weights, decode, 50 files and published numbers;
- every accepted E3 result and E3's criteria;
- gate v2 and every threshold;
- the honest judge (D1b, Vocab_A/B), the copier control and the tiered copy rule;
- `uie/`.

**Not used, though each might raise the odds:**
- more steps;
- more width or depth;
- a different merge count after any result;
- any decode change;
- a second training of an arm before the reading.

Each would be a different experiment, with its own registration.

**Excluded by the owner:** a word list or lexicon in any form, template matching, a
pretrained model.

## 17. For the owner's review: the choices made here

The first draft's items 1 and 2 (K = 10, L = 16, a platform as one site) are withdrawn.
§4.2 says why.

1. **The independent source** (§4.3): the Public Suffix List with its private section, so a
   site on a shared platform is its own source. Near-duplicate sources are merged by Lee et
   al.'s rule (edit similarity above 0.8).
2. **Both classes on an equal footing** (§4.3): per-class source shares, combined by the
   classes' training draw shares.
3. **The name scope, in both arms** (§4.5): this restores the engine's original design. A
   third arm (letters with the name scope) would isolate its share, for about 5 more
   GPU-hours. It is not in this draft.
4. **BPE-dropout at p = 0.1** (§4.4), the published standard value.
5. **Pass marks:** A on D1b-R; **B on the whole URL (D1b-W)**, the strongest statement of
   word realism, with the host reading recorded beside it (§8.1).
6. **The tiered copy rule carries over,** extended to hosts (§9.6):
   - domains, full hosts, live hosts and one-source names are strict;
   - common names get the +0.05 room under the diversity condition.
7. **Scoring on the laptop,** as in the memory phase (about 7–8 h unattended for both
   arms), rather than in a Kaggle CPU session.
8. **The merge budgets** are R1's registered sizes (32,000 rest, 16,000 host), the only
   limit on the tables.

## 18. Amendment 1 (2026-10-13, before any piece table exists): how the tables are computed

- **Logged after gate A and before any table, training or output exists.** The only P2 data
  seen are the corpus sizes below, counted from the training files.
- **The numbering:** the frozen-numbers amendment that §15 calls "Amendment 1" becomes
  **Amendment 2** (gate B).

### 18.1 The registered statistic, on a stratified source sample (the rest table only)

**The measured sizes** (training rows and stream 1, sources by §4.3's unit):

| table | class | sources | characters | in multi-page sources |
|---|---|---|---|---|
| rest | benign | 670,876 | 89.1M | 72.0M, in 81,217 sources |
| rest | phishing | 68,464 | 2.9M | 2.0M, in 11,268 sources |
| host | benign | 906,504 | 11.6M | 5.7M, in 382,188 sources |
| host | phishing | 151,818 | 3.5M | 3.1M, in 112,234 sources |

**The problem:**
- §4.3 counts a pair once per source. For a source with many strings, that needs a running
  count of how many of its strings contain each pair.
- For the benign rest table's multi-page sources (72M characters) this is tens of millions
  of counters: more than the laptop's memory can hold in the pure-Python learner.

**The change:**
- In the rest table, the benign multi-page sources are learned on a uniform sample of
  **15%** (seed 20261070).
- Each sampled source counts 1 ÷ 0.15, so every pair's source frequency stays an unbiased
  estimate of the full count.
- **Everything else is complete:** every single-page source, every phishing source and the
  whole host table.
- **Why this is the right place to sample:**
  - under §4.3 a multi-page site is one source, however many pages it has;
  - these sites are 12% of benign rest sources, so they carry about 12% of every count;
  - sampling them adds a little variance to that share and no bias.
- **The rate is a memory limit, not a modelling choice.** The learner's peak memory is
  measured and reported in Amendment 2.
- **The stop rule** (§4.3, point 5) counts sources in the learning data as sampled.

### 18.2 Specified now, consistent with the registration

| item | specification |
|---|---|
| near-duplicate clustering (§4.3) | Applies to sources whose suffix is in the private section (tldextract's `is_private`). The label compared is the source's own label. Candidates come from MinHash LSH over character 3-grams: 120 hash functions in 40 bands of 3, which catches 3-gram Jaccard 0.5 with probability above 0.99. Confirmed when the edit similarity, 1 − distance ÷ the longer length, is above 0.8. Clusters are connected components |
| class shares (§4.3) | Each source counts w_c ÷ N_c: N_c is the class's number of sources in the table's full data; w_c is the class's training draw share at class α = 1 (benign 906,214 ÷ 1,098,866; phishing 192,652 ÷ 1,098,866). Scores are computed from integer source counts, so they are exact and the same on every machine |
| a pair's presence | A pair counts once per source when any string of that source contains it, however many times |
| BPE-dropout (§4.4) | Provilkov et al.'s procedure. At each step every applicable merge occurrence is dropped with probability 0.1, and the lowest-ranked kept occurrence (leftmost on a tie) is merged. Segmentation stops when no kept occurrence remains. At p = 0 it gives the canonical encoding exactly (a unit test). It applies to every training row of a batch, the stream rows included. Each row's generator is seeded from (20261071, the batch's index, the row's position) |
| the piece rule (§4.6) | By table: a rest piece only in PTH, QRY and FRG; a host piece only in LBL and SUB. Applied by a wrapper around the model at generation, which reads the current field from the row's last field marker |
| V-mass (§12) | The appended stream rows are counted as registered. Their query positions are counted in characters, so the reference values (585,977 and 2,231,497) still apply |
| C (§5) | batch size × the expected scored mass per drawn row under the arm's encoding, the name scope and the dropout. It is computed from every training row with 4 dropout draws per row (seed 20261072) and pinned in Amendment 2 |
| sources not on the private list | Shared hosts that the Public Suffix List does not list (for example `000webhostapp.com`, `weebly.com`) are one source each, by §4.3's registered unit. The audit reports the largest sources so this is visible |

## 19. Amendment 2 (2026-10-13, before any piece table exists): the stop rule's reading, and three seeds

- **Logged before any table, training or output exists.** A first build ran under the literal
  reading below. It was stopped while learning the rest table, before it wrote any table,
  and its files were deleted. Seen from it: the kept-row counts and the cluster count (3,162
  clusters over 17,733 sites). The stop rule does not affect either.
- **The numbering:** the frozen-numbers amendment (gate B) becomes **Amendment 3**. Where §18
  says "Amendment 2", it means that amendment.

### 19.1 The stop rule (§4.3, point 5)

**The registered text:** "learning ends early if the best remaining pair occurs in only one
source. A fragment no two sources share is not vocabulary."

**The two sentences disagree under the registered weights.**
- They agree only when every source weighs the same. Then the best pair is in one source
  only when every pair is.
- Under §4.3 point 3 (class shares) and §18.1 (the 1 ÷ 0.15 sample weight), sources do not
  weigh the same:
  - a sampled benign multi-page rest source weighs 0.8247 ÷ 670,876 ÷ 0.15 ≈ 8.2 × 10⁻⁶;
  - a phishing rest source weighs 0.1753 ÷ 68,464 ≈ 2.6 × 10⁻⁶;
  - so a pair inside one sampled benign site outranks a pair that three phishing sites
    share (7.7 × 10⁻⁶).
- Read literally, the first sentence ends learning at that point. Every shared pair ranked
  below it is never learned. Where learning ends is then set by the weights, not by the
  data running out of shared pairs.
- **A unit test found it:** one heavily weighted benign source and the word `login` in 5
  phishing sources. The literal reading learns nothing (0 merges), and `login` never
  becomes a piece.

**The reading adopted is the second sentence, which states the purpose:**
- a pair found in fewer than two sources is never merged, however high its weighted score.
  It is passed over; it is not a stop;
- learning stops at the budget, or when no pair that two sources share is left.

**What this changes, and what it does not:**
- Under both readings, no pair inside a single source is ever merged.
- The only difference: the shared pairs ranked below a heavy single-source pair can still
  be learned.
- With equal weights the two readings give the same table.
- The criteria, the guards, the budgets and the statistic are unchanged.

**Checked:**
- The learner equals a brute-force reference under this reading in 8 random cases. In every
  one of them the learner passes over single-source pairs.
- Amendment 3 reports, per table, where the literal reading would have stopped: the merge
  count at the first pair passed over.

### 19.2 Seeds, fixed before use

| seed | use |
|---|---|
| 20261073 | the MinHash hash functions (§18.2's clustering) |
| 20261074 | the start's rest piece rows, N(0, 0.02) (§3) |
| 20261075 | the start's host piece rows (`p2_b`) |

### 19.3 Two records the build adds (V-T2, §4.3)

- **The inputs, hashed:** the four training files, stream 1, `e16_cap`'s record (its
  tokenizer filters rows by length) and tldextract's Public Suffix List snapshot, with
  tldextract's version (5.3.1 here).
- **The laptop's start:**
  - The laptop's torch does not rebuild `caca0107…`, as E16's amendment 2 found. Only
    Kaggle's does.
  - So the laptop test checks the peel-off against this torch's own `e16_cap` start: every
    weight, and the embedding's first rows, equal.
  - The notebook checks `caca0107…` on Kaggle, as §5 says.

## 20. Amendment 3 (2026-10-13, gate B): the frozen numbers, and what the implementation needed

- **Logged after the tables, the checks and the dry runs, before any Kaggle run.** No P2 model
  exists, and nothing here was chosen after seeing an arm's output.
- **Nothing in §8–§11 changes.** §20.4 records what the implementation needed to run the
  registered design as written, each item with its evidence.

### 20.1 The tables (V-T2, V-T3, V-T4)

| | rest | host |
|---|---|---|
| merges | 32,000 (the budget) | 16,000 (the budget) |
| sha256 | `a48312b8f097e259…` | `a4ed9ca41e3e8dde…` |
| units per class (benign / phishing) | 727,503 / 66,777 | 963,035 / 137,130 |
| multi-page units learned (§18.1) | benign 12,052 (the 15% sample); phishing 11,258 (all) | all |
| sources sharing a merge: smallest / 10% / median | 14 / 35 / 79 | 29 / 44 / 91 |
| where the literal stop rule would have stopped (§19.1) | never | never |
| learning time, peak memory | 578 s, 2.73 GB | 160 s, 2.29 GB |

- **Amendment 2's reading changed nothing here.** The literal reading would not have stopped
  either table before its budget, so both readings give exactly these tables.
- **The budget binds.** Every piece is shared by at least 14 sources (rest) or 29 (host),
  far above the two-source floor.
- **The near-duplicate clusters:** 3,162 clusters over 17,733 platform sites (map
  `338502e534ce…`). The largest holds 732 lookalike KuCoin login sites on webflow.io.
- **V-T2, the inputs:** the four training files, stream 1 (`637accab…`), `e16_cap`'s record
  (`3b13b0e9…`) and tldextract 5.3.1's suffix-list snapshot (`b69315c0…`), each hashed in
  `data/v2/p2/tables/build.json`.
- **V-T3:** the per-field allowed sets hash to `e9488f82a672…`.
- **V-T4's required rows:** every merge in at least two sources (rest and host), and
  losslessness on R1's held-out samples. Both PASS.

### 20.2 §4.7's reported rows, with the evidence (to the owner; nothing patched)

1. **Pieces that are a whole training string of a single source: rest 945 of 32,000 (3.0%),
   host 2,024 of 16,000 (12.7%). §4.7 expected "near 0"; that expectation was wrong.**
   - Each is a fragment many sources share that is also, by coincidence, one site's entire
     string. Examples: `.html`, `/search`, `-contact`, `aargau`, `abeauty`, `2004`.
   - The merge that made each one was shared by at least 16 sources (rest) or 29 (host); the
     median is 73 (rest) and 77 (host).
   - §4.2 calls such strings vocabulary: "a word that is one site's entire name but common
     inside other strings is vocabulary, not a record". The guards judge the outputs (H-LBL1,
     H-SUB1, G-COPY).
2. **Heavily drawn strings that are one of the pieces** (§4.7's definition, the one behind its
   R1 counts): **rest 201 of 210 (R1: 208), host 139 of 189 (R1: 184). §13 predicted "under
   10 each"; that prediction was badly wrong.**
   - Most heavily drawn strings are words many sites share: `privacy-policy`, `impressum/`,
     `index.php`, `contact`, `www`, `shop`, `support`. Source frequency correctly keeps them
     as vocabulary. My prediction assumed they were mostly single-site strings.
   - Some rest ones are kit strings that many registered domains share, for example the
     obfuscated query `3a36efff&cf6=5a&1eG=K9dHF3da4b=D400e`. By §4.3's unit they are
     vocabulary. G7 and G-COPY judge any copying of them.
   - The audit's first version counted strings that encode as a single token, which also
     counts single characters (rest 203 vs 210, host 144 vs 189). It now reports both counts,
     each labelled.
3. **Coverage of real held-out words in §4.7's count bands, against R1's tables.** The 1–99
   band is not below R1's in 3 of 7 rows and is below in 4:

   | field, class | 1–99 band: P2 vs R1 | 100+ band: P2 vs R1 |
   |---|---|---|
   | LBL benign | 0.063 vs 0.054 | 0.680 vs 0.651 |
   | SUB benign | 0.069 vs 0.071 | 0.988 vs 0.989 |
   | PTH benign | 0.015 vs 0.015 (below in the 4th decimal) | 0.622 vs 0.621 |
   | QRY benign | 0.032 vs 0.036 | 0.735 vs 0.770 |
   | LBL phishing | **0.178 vs 0.202** | **0.822 vs 0.989** |
   | SUB phishing | 0.234 vs 0.201 | 0.907 vs 0.964 |
   | PTH phishing | 0.191 vs 0.189 | 0.913 vs 0.920 |

   - **The clear shortfall is phishing labels.** R1 counted draws, so a label repeated across
     one campaign's or one platform's rows became a piece. Source frequency counts that
     campaign or platform once (`weebly`, for example, is a single source).
   - This is the trade §4.3 chose: the memorisation hazard, against coverage of names that
     repeat inside one source.
   - §13's "at least R1's tables' (about 70%)" quoted a wrong level. R1's tables hold 1.5–20%
     of the 1–99 band whole (and 62–99% of the 100+ band). This audit's measure is R1-4's: its
     R1 column reproduces R1-4's published benign path figures (85.0 / 31.3 / 4.5 / 0.9% by
     rank band, against 84.6 / 29.7 / 4.2 / 0.9% here).
4. **The largest sources.** Shared hosts that the suffix list's private section does not list
   are one source each, as §18.2 recorded: `weebly.com` (10,146 host strings),
   `weeblysite.com` (7,176) and `godaddysites.com` (6,049). One `.com.au` domain (13,351) is random
   subdomains under one registrable domain.

### 20.3 C and V-T1 (§5, §18.2)

| | `p2_a` | `p2_b` |
|---|---|---|
| **C** (batch 256 × the expected scored mass per drawn row: the arm's pieces, the name scope, 4 dropout draws per row) | **6208.307292719264** | **5067.853260707934** |
| beside it: canonical split, no dropout | 6149.647 | 4969.939 |
| mass per drawn row, benign / phishing | 22.705 / 31.526 | 19.691 / 20.290 |
| tokens per row, benign: characters → canonical | 61.6 → 35.4 | 61.6 → 26.8 |
| tokens per row, phishing: characters → canonical | 46.6 → 38.3 | 46.6 → 24.3 |
| V-T1: every training, validation and stream URL, canonical and under dropout; then every row over 384 characters (3,851 training, 543 validation) | PASS, zero failures | PASS, zero failures |
| rows kept and dropped | `e16_cap`'s exactly (2,551,259 / 238,707; 3,851) | the same |

**The script is checked against `uie`'s own arithmetic.**
- On a 12,000-row subset, its C equals `uie.train.expected_scored_tokens` on the same rows
  encoded by the run's own `encode_corpus`, bit for bit.
- This holds in characters (13944.155428948041) and in canonical pieces with the name scope
  (A 6226.512506385456, B 4980.839899345481).
- The full pass gives the same character value for both arms (14333.846590747171), as it
  must.

**A finding about the old constant (reported, no change):**
- `e16_cap`'s C, 14397.885481960007, is not E9's rule applied to `e16_cap`'s data. It is
  256 × 56.24174016390628, the mass PREREG_E9 D3 computed for E7's sampler and tokenizer.
- It has been carried unchanged since E9. `e12_base` (max_len 256, vocab 108) records the
  same number.
- E9's rule on `e16_cap`'s own data gives 14333.846590747171, 0.44% less.
- §5 and §18.2 register the rule, re-derived on each arm's own encoding, and that is what is
  pinned here.
- A first version of the script checked its character value against the old constant and
  failed. That check expected the wrong thing. It was replaced by the check against `uie`'s
  own function, and the arm's result was re-read with it; nothing was recomputed.
- With Adam, a constant scale on the gradient makes almost no difference either way.

### 20.4 What the implementation needed (no criterion changes)

1. **The vocabulary head's memory.**
   - With 32,110 (A) or 48,110 (B) tokens, the logits are the largest tensor in training.
     §5's worst case is 288 rows of 384 positions, every target scored.
   - Even in 4 parts (72 rows), the logits alone need about 13–19 GB for B, at 10–14 bytes
     each under autocast. For A they need about 9–12 GB, beside about 4 GB of activations.
   - So the registered check (2 parts, else 4) would STOP both arms on a T4 (14.6 GB).
   - **The fix computes the same loss:** the head is applied to chunks of positions, and each
     chunk's logits are recomputed in the backward pass.
     - It is the same per-target cross-entropy over the same mass. The loss and every
       gradient equal `URLTransformer.loss`'s within float rounding (a unit test).
     - The batch, the parts, the objective and the registered check are unchanged.
   - Validation, the fit pass and the after-training structure NLL use the same chunked head,
     without gradients.
   - Generation computes the last position's logits only, the one `generate` reads.
   - S4's segment-start diagnostic (`boundary_nll`) is skipped for a piece model. It reads `/`
     as a token of its own, and pieces span it. It is not a P2 reading, and it would not fit a
     T4 after training.
   - Character mode keeps `uie`'s own code paths, so V-equiv stays bit-identical.
2. **V-rules read the generation's own token-level record (§12).**
   - The URL strings cannot show D1's guarantee exactly. A delimiter written as a character
     inside a host field parses as a host end there. `e16_b5`'s own files show 5 such URLs in
     180,000, and its published S honouring is 99.99%, not 100%.
   - So each generation call records, from the rows' tokens as they were sampled, each row
     against its supplied plan:
     - D1 and D1′: the host ended on a token the plan allows;
     - B5: a Q = 1 row's `<QRY>` came before any `<EOS>` or `<FRG>`. This mirrors the
       decoder's own state, which closes at the first `<QRY>`;
     - the piece rule's violations;
     - rows cut at max_len before the decision, counted apart.
   - Zero violations is required. The string reading is reported beside it.
   - Tests: hand-made rows, and 360 rows from random models, where the decoder enforces the
     rules and the record finds none broken.
3. **Cut-offs at max_len are recorded by the generation.** With pieces, a URL's character
   length no longer tells whether it reached 384 tokens.
4. **D1's moved mass is recorded per file,** as §6 asks; E17's calls did not record it.
5. **The generation calls are E17's, with the record added.**
   - A unit test shows their URLs equal to E17's own calls'.
   - **V-gen-equiv PASS:** 1,355 of 1,355 URLs of `e16_b5`'s files regenerated byte for byte
     on this CPU, by `e16_cap`'s weights through P2's tokenizer, piece rule and calls.
6. **The host guards' training data:**
   - H-HOST, H-LBL1, H-SUB1 and H-LBL use the rows the tables learned from (every kept
     training row of both classes, and stream 1) and §4.3's unit;
   - H-RD uses the class's training rows, as G-COPY's domain row does;
   - H-TYPO counts words over the class's whole training data (R1-3's counts).
7. **The page-row name gap (§5)** uses P1b's rows exactly:
   - the trained-site rows of its reading C: 2,000 `structure_train` rows per class, seed
     20261054 + the class's index;
   - the new-site rows of its reading A: K-used's draw (seed 20261050, 2,000 rows from each
     validation stream in order), its page rows.
   - Shown equal to P1b's own rows. It reads the name fields per character, the arm's and
     `e16_cap`'s on the same rows.
8. **The start:** training's own start (the patched procedure, at the run's config and seed)
   equals the start the notebook checks before training, at full size (a unit test).
9. **V-T1 covers every URL:** the main pass skips rows over 384 characters (neither the run
   nor the main pass encodes them). A second pass checks those rows the same way.
10. **V-floor in the reading (§12):**
    - a D1b-x reading whose floor fails cannot confirm a pass (§8.3), so it can only remove
      one;
    - a class whose D4 floor fails makes that class's G-D4 or G-D4-H reading VOID: listed,
      neither pass nor fail.
11. **V-scope's test** asserts the registered wording: on a page row every weight equals
    `field_mask`'s (the frame on) except name-field content, which is 0.

### 20.5 The dry runs

They decide nothing; they show that the code does what this registration says, before any
GPU time is spent (`research/v2/v2_p2_dryrun.py`).

1. **V-gen-equiv PASS** (§20.4, item 5): 1,355 of 1,355 URLs identical.
2. **The reading pipeline, on stand-ins.** `e16_b5`'s own 50 files stand in for both arms (as
   sources 3 and 4, with their own sample seeds and folds); everything runs through the real
   scripts.
   - **V-D1b PASS, exactly:** `e16_b5`'s D1b readings reproduce R2 Part A on every view (R
     +0.1720 / +0.0857, H +0.0581 / +0.1019, W +0.1500 / +0.1209), and the copier control
     reproduces it (+0.0098 / +0.1524). Every floor is within 0.03 of 0.5.
   - **The stand-ins read like `e16_b5`,** as they must: for both stand-ins, on all 11 readings
     in both classes (44 paired t's), the paired t against `e16_b5` stays inside ±2.776
     (largest: D4-R, phishing, −2.48).
   - **V4 PASS:** the scorer reproduces `e12_final`'s published cells exactly (largest
     difference 0).
   - **V-rules, string half:** `e16_b5`'s files show the 5 in-field delimiter cases of §20.4
     item 2, and no B5 case.
   - **The gate scorer:** through `v2_p1_score.py` (`V2_ARM=p2_a`: V4, free, control, misc,
     copy, merge), the stand-in's report reproduces `e16_b5`'s published report exactly:
     - every block (free, control, emission, slash, honouring, novelty, S honouring, plan
       trace);
     - the 4 query-copy cells.
   - **The metrics and the host guards:** natural rates follow the splits. Benign held-out
     hosts almost never repeat a training domain or host (0.0001). Phishing reuses
     registrable domains 62.9% of the time (R1-6: 63.7%), but full hosts only 0.06%.
     `e16_b5`'s rates match R3's (benign domain copies 1.39%); the stand-ins equal
     `e16_b5`'s.
   - **The reading:** both arms read NO EFFECT, every guard PASS (strict and tiered, the host
     guards included). B's host gain is not recorded, A's specificity holds, and E3 for the
     record is NOT MET (3 cells, 0 slash rows: `e16_b5`'s own).
     - Arm A's score report is the scorer's own. Arm B's is `e16_b5`'s published report under
       its name (part `standin_scores`): the stand-ins are the same files, and arm A's run
       shows the scorer reproduces that report exactly.
3. **The notebook pre-flight.** The four notebooks' own cells ran on this CPU from the
   extracted bundle. Only their Kaggle-only lines were replaced (folders, the GPU query, this
   torch's start values, one-row memory check, two jobs of 2 URLs).
   - Training notebooks, before training: the manifest, the tables, the start built twice
     and peeled off to `e16_cap`'s start, the parameters (A 41,857,024, vocab 32,110; B
     50,049,024, vocab 48,110), and the memory check's code path. Training and its readback
     were not run: nothing trains on the laptop.
   - Generation notebooks, on an untrained full-size start: every cell. Zero violations of
     the piece rule, D1 and B5. The per-file records hold the removed mass per field, the
     token-level rules, D1's moved mass and the cut-offs.
   - The removed mass behaves as it must. In A it is close to all of an untrained model's
     mass outside PTH, QRY and FRG (32,000 of 32,110 tokens are rest pieces), and none
     inside. In B it is about two thirds in LBL and SUB and one third in QRY.

### 20.6 The bundle, the notebooks, the cost

- **The bundle:** `uie-v2p2-code.zip`, built by `research/v2/make_v2p2_bundle.py` from a clean,
  committed tree.
  - sha256 `8d18d9530a5992bb2893ac4e8e75a61c7b9505d59cc46e259f8f12544f79bbb8`, from commit
    `22867b4`, 2.4 MB, 47 code files.
  - The notebook pre-flight (§20.5, item 3) ran on this exact zip and passed for all four
    notebooks.
  - It holds `uie/` (= `422cb23`), the common modules E8–E17, V2's P2 modules, the two piece
    tables, stream 1, `e16_cap`'s record and the three generation tables.
  - Its manifest pins every file's hash, the piece tables' and rules' hashes, and both C's.
- **The notebooks:** `V2P2_train_a`, `V2P2_train_b`, `V2P2_generate_a`, `V2P2_generate_b`
  (`make_v2p2_notebooks.py`). The run sheet is `V2P2_KAGGLE.md`.
- **The cost, re-estimated:**

  | run | about |
  |---|---|
  | training, each arm (§14 registered 3–4 h and 3.5–4.5 h) | 3.5–4.5 h: the chunked head recomputes each chunk's logits in the backward pass, and the dropout splits run on the CPU |
  | generation, each arm (§14: 2–2.5 h) | 1.5–2.5 h: shorter token sequences, and the head computed at the last position only |
  | **GPU total** | **about 11–14 T4-hours** |

- **Laptop timings and memory:** the tables 1,012 s (peak 2.73 GB); C and V-T1 about 2.7 h
  per arm (peak 1.0 GB); the audit 3 min; the dry runs about 2 h. All ran at low priority
  with the low-RAM alarm; nothing trained.

## Status

- **2026-10-13:** drafted for the owner's review, after the owner approved the direction
  (§1). Documents only; no code; no run.
- **2026-10-13: revised after the owner's review (§1.1).**
  - The first draft's piece-table rules are withdrawn. §4.2 records each one's flaw.
  - §4 is rewritten:
    - independent sources (the private suffix list, near-duplicates merged);
    - source frequency, with no caps or thresholds;
    - BPE-dropout;
    - the name scope restored in training.
  - Every section that depended on the old rules now follows the new design.
  - Still a draft: no code; no run.
- **2026-10-13: REGISTERED. Gate A approved by the owner,** verbatim:

  > Proceed with implementing V2-P2 based on the revised "PREREG_V2_P2.md".
  > Complete the code changes, piece-table construction, training updates, anti-memorization
  > guards, evaluation pipeline, unit tests, local audits, and toy-scale checks. Review the
  > implementation critically and fix any issues you find.
  > Preserve the frozen "e16_b5" baseline and all previously accepted results. Do not weaken
  > or retroactively change preregistered criteria to make the experiment pass.
  > You may run local tests and prepare all required Kaggle scripts, configuration, bundles,
  > and execution instructions. Do not attempt to launch Kaggle yourself.
  > When everything is ready, commit and push the completed implementation, verify that the
  > remote matches your final commit and the working tree is clean, then give me a concise
  > report [...]
  > Stop after preparation and wait for my approval before any experiment run that requires
  > me to launch Kaggle.

  - §17's choices stand as written.
  - Gate B (Amendment 1, the frozen numbers) comes before any Kaggle run.
- **2026-10-13: Amendment 1 (§18),** logged after gate A and before any table, training or
  output exists. The benign rest table learns on a stratified source sample (a memory
  limit; the estimate stays unbiased), and the implementation details are specified.
- **2026-10-13: Amendment 2 (§19),** logged before any table exists.
  - The stop rule follows its stated purpose: a single-source pair is passed over, not a
    stop.
  - The build and start seeds are fixed.
  - The build records its inputs' hashes.
  - The frozen numbers become Amendment 3 (gate B).
- **2026-10-14: Amendment 3 (§20), the frozen numbers, logged before any Kaggle run.**
  - The tables, C per arm, V-T1 to V-T4, the audit's reported rows, what the implementation
    needed, the dry runs and the bundle.
  - No criterion changed.
  - **Gate B is the owner's:** the Kaggle runs wait for the owner's approval.
- **2026-10-14: GATE B APPROVED by the owner,** verbatim: "yup I am starting the runs now".
  `V2P2_train_a` and `V2P2_train_b` start on Kaggle from the bundle `8d18d953…` (commit
  `22867b4`).
- **2026-10-14: READ ONCE (23:23).** Every validity check passed in both arms. **`p2_a`: NO
  EFFECT WITH COST (G1, G2, G3, G6, G7, G9, G-D4, C1). `p2_b`: NO EFFECT WITH COST (G1, G2,
  G3, G6, G7, G9, G-D4, G-D4-H, C1, G-VAL, H-TYPO).** The word tells rose in every view.
  Nothing is adopted; `e16_b5` stays the baseline. The owner decides what follows. See
  `V2P2_RESULTS.md`.
- **2026-10-15: CLOSED by the owner** under the current rules, with the rest of V2's word-level
  line; `e16_b5` stays the baseline (`research/native/DECISION_2026-10-15.md`).
