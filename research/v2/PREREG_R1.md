# V2-R1 — the lexical diagnosis: registration

- **Registered 2026-10-09, before any R1 code.** The design is `REVIEW_V2.md` §5.2, approved
  by the owner (§1).
- **Laptop only, on existing files.** No model is trained, no GPU is used, and nothing is
  generated.
- **Unchanged:** `e16_b5` stays frozen and `BASELINE`; `uie/` is untouched; E3 stays closed;
  P1 is not begun.
- **Output:** `R1_RESULTS.md` (evidence, findings, limits, implications for P1). Then R1
  **stops for the owner's review.**

## 0. In plain words

- **Four questions, on files we already have:**
  1. Which parts of the URL carry the word problem?
  2. Do the engine's fake words stand in for *common* real words or *rare* ones?
  3. What do the fakes look like: near-misspellings, two real words joined, or new
     inventions?
  4. At most, how many real words could a set of pieces hold whole?
- **Two more things P1 needs:**
  - **Is word *order* already a giveaway?** A new detector reads words in sequence. It is
    first tested on cases where we know the answer.
  - **The reference numbers** for copying, diversity and validity that P1 will be judged
    against.
- **Nothing is trained.** The only "learning" is building tables of pieces in order to count
  how many real words they would hold whole: counting, not modelling.
- **R1 decides P1's fields and piece counts** by rules fixed here, or concludes that pieces
  cannot plausibly reach the bar.

## 1. The owner's decisions, verbatim (2026-10-09)

> Proceed with R1 only, using existing local files and no training.
>
> My decisions are:
>
> - R1 first: laptop-only diagnosis.
> - P1: compare both Arm A and Arm B.
> - Evaluation: approve the six-part scorecard, with thresholds preregistered before the
>   experiment.
> - V2 finish line: review its relationship to E3's conditions before finalizing it. E3
>   itself remains permanently closed.
> - Pretrained model: retain as a future option only.
>
> Keep "e16_b5" frozen. Do not change the baseline, write or run training code, launch any
> GPU job, or begin P1 yet.
>
> For R1, preserve the existing E16/E17 step-by-step workflow and report the evidence,
> findings, limitations, and implications for P1. After R1, stop and wait for my review.
> Then present the revised V2 finish line and final P1 preregistration for approval before
> any training or GPU execution.

## 2. Material (read-only; every file hashed in the record)

| what | where |
|---|---|
| `e16_b5`'s free files, 5 per class | `data/e17/decode/gen/free_e16_b5_{class}_g{g}.json` |
| `e15_new`'s free files (the capacity yardstick) | `data/e15/generate/gen/free_e15_new_{class}_g{g}.json` |
| `e9_ln`'s free files (D4's positive control) | `data/e9/gen/free_e9_ln_raw_{class}_g{g}.json` |
| real held-out URLs | the gate's validation pools (`data/bundle2/*_val`), one URL per host by E18's own selection |
| real training URLs | `data/bundle2/*_train` (name and structure, both classes) |
| the class vocabularies | E18's (`data/e18/strong_disc/vocab_{class}.npz`): the words of the class's training URLs, with counts |
| the known-URL filter | `data/bundle2/known_urls.bloom.gz` |
| the threat snapshot | `data/feeds/openphish/feed.txt` (300 URLs) |
| cached samples | E18's (`data/e18/strong_disc/samples`) and R0's (`data/v2/r0/samples`) |

A **word** is D1's: a run of three or more letters, lower-cased (`[a-z]{3,}`). It is
**known** if the class's vocabulary holds it.

## 3. The questions and their measures

### R1-1 Where is the word tell? (D1, one field at a time)

- **The detector:** E18's D1, unchanged.
- **The fields:** taken by the gate's own extractors (`uie/evaluation.py` `FIELDS`): LBL, SUB,
  PTH and QRY.
- **The strings:** LBL and SUB are read alone, as a host. PTH is read as `h.h/<body>` and QRY
  as `h.h?<query>`, so only the field carries information.
- **The population:** URLs whose field is non-empty, as the gate scores a field.
- **N_f:** the largest multiple of 100 not above any of:
  - 6,000;
  - a quarter of the real pool's URLs that carry the field (the fewest over g);
  - 0.9 × the fewest such URLs in an `e16_b5` file.
  - **Below 300, the field is not measurable** in that class.
- **Seeds:**
  - R1_SEED = 20261030. Fields are numbered LBL 0, SUB 1, PTH 2, QRY 3; sources `e16_b5` 0
    and `e15_new` 1.
  - Real shuffle: R1_SEED + 1,000 × field + 100 × class + g.
  - Source draws: R1_SEED + 100,000 × (1 + source) + 1,000 × field + 100 × class + g.
  - Folds: E18's rule, `seed_of(class, g)` × 10 + the task index (floor 0, `e16_b5` 1,
    `e15_new` 4).
- **The reading:** as a gate cell (excess, SE, t, verdict, clear line), plus `e15_new` −
  `e16_b5`, paired by g.

### R1-2 Common or rare words? (the frequency profile)

- **On R1-1's samples** (real C, `e16_b5`, `e15_new`), per field and class.
- **Every word is placed in a band** by its rank in the class vocabulary: top 1k, 1k–8k,
  8k–32k, beyond 32k, never seen. The bands' word shares are the mean over g.
- **The shortfall:**
  - e = the never-seen share in generated text − the never-seen share in real text;
  - for each known band b, d_b = real share − generated share, so that Σ d_b = e;
  - S_common = (d_top1k + d_1k–8k) / e.
- **The reading:**
  - **COMMON-WORD SHORTFALL** if S_common ≥ 0.5;
  - **RARE-WORD SHORTFALL** if S_common < 0.5;
  - **NO SHORTFALL** if e < 0.01.

### R1-3 What are the fakes like? (anatomy)

- **Every never-seen word,** generated and real, is classed in this order:
  1. **near miss:** one edit (inserting, deleting or substituting one letter) from a word
     seen at least 10 times, the edit rule of QC1's near misses;
  2. **join:** it splits into two words, each of at least 3 letters and seen at least 10
     times;
  3. **other.**
- **Reported:** per field and class, the shares and the counts per URL. The *excess* per
  category (generated − real, per URL) shows which kind of fake the engine overproduces.
- **The typosquat reference:** the near-miss share of LBL words, real against generated.

### R1-4 How many real words could pieces hold? (coverage)

- **Two piece tables, learned on the laptop** by `research/v2/v2_bpe.py`. This is greedy BPE:
  deterministic, ties broken by code-point order, no splitting inside a field body.
  - **rest:** path, query and fragment bodies;
  - **host:** registrable labels and subdomain labels, each label a sequence of its own;
  - **from one sample:** 600,000 training rows drawn as training draws them. The class is
    picked by label count (906,214 benign against 192,652 phishing, α = 1); the row is
    uniform within the class. Seed R1_SEED + 7.
  - **Up to 32,000 merges.** The 2k, 4k, 8k and 16k tables are prefixes of the same run.
  - If learning a table takes more than 3 hours, its sample is halved, and the change is
    recorded.
- **Coverage:** on R1-1's real C samples, the share of words whose whole span sits inside one
  piece, with the field encoded canonically. Given per field group, class and table size;
  also by R1-2's bands; with characters per piece.
- **These are diagnostic tables.** P1 may reuse them only if its registration names them by
  hash.

### R1-5 Is word order already a tell? (D4)

- **D4:**
  - the URL is lower-cased and split into letter runs, digit runs (each becomes `0`) and
    single other characters;
  - TF-IDF over 1–3-grams of those tokens (sublinear tf, at most 50,000 features), then
    logistic regression (C = 1);
  - E18's protocol: N = 6,000, 5-fold out-of-fold AUC, E18's cached real samples and folds.
- **Sources:**
  - `e16_b5`, `e9_ln` and the chimeras: E18's cached samples;
  - `e15_new`: R0's D1-H draw, as whole URLs;
  - **shuffled real URLs:** the D block of E18's real pool, rebuilt by E18's own selection
    and seed. In each URL, the words of the host, and separately the words of the path,
    query and fragment, are permuted among their own positions (seed R1_SEED + 50,000 +
    100 × class + g). The share of URLs this changes is reported.
- **Validity:**
  - its floors must sit within 0.03 of 0.5;
  - it must CONVICT `e9_ln` in both classes (E18's V-P1), or D4 is VOID.
- **Sensitivity to order:**
  - if D4 convicts the shuffled real URLs in a class, it can see word order there;
  - if not, D4 is recorded as a word-sequence detector whose order sensitivity is not shown.
    That is a limit, not a failure.
  - D1 is also run on the shuffled URLs, as a check on the control: it should see little.

### R1-6 The reference values for P1's scorecard

On `e16_b5`'s 5 free files per class (and `e15_new`'s, for the yardstick), against the real
held-out pool:

- **The sample gate** (`uie/gate.py`), per file:
  - validity failures, by reason;
  - MEMORIZED, against the known-URL filter;
  - DUPLICATE_IN_BATCH;
  - ACTIVE_THREAT, against the 300-URL snapshot. Its size is a stated limit.
- **Exact copies,** per field (LBL, SUB, PTH, QRY) and for the registrable domain: the share
  of non-empty values found among the class's training values. The *natural rate* is the
  same share for real held-out values, one per host.
- **Copies of values seen exactly once in training,** generated against real.
- **Near misses,** per field: never-seen values one edit from a training value seen at least
  5 times. This is QC1-4's definition, using `research/qc1/qc1_common.py`'s `NearIndex`.
- **Diversity:**
  - duplicate values within a file, per field;
  - distinct words per 1,000 words at an equal n (20,000 words), per field group, with
    real-against-real at the same n as the baseline.

## 4. Validity: checked before anything is read

| code | check | on failure |
|---|---|---|
| V-D1′ | E18's D1 on its cached benign g0 floor and `e16_b5` tasks reproduces E18's record exactly | stop |
| V-floor | every per-field D1 floor and every D4 floor within 0.03 of 0.5 | that reading is VOID |
| V-N | N_f set by §3's rule; measurability flagged | — |
| V-BPE | decoding returns every encoded string exactly; learning twice gives the same table (a test) | stop |
| V-D4 | D4 convicts `e9_ln` in both classes | D4 is VOID |
| V-inputs | every input file's sha256 recorded | stop if a pinned hash differs |

## 5. Decision rules for P1 (fixed now)

- **DR1, the fields:**
  - arm A's table always covers PTH, QRY and FRG (R0's D1-R convicted both classes);
  - arm B adds LBL and SUB, each if its R1-1 reading CONVICTS `e16_b5` in either class, or
    is not measurable (R0's D1-H convicted the host as a whole);
  - if neither LBL nor SUB qualifies, arm B is dropped.
- **DR2, the piece count per table:** the smallest of 8k, 16k and 32k whose coverage of real
  words is within 5 points of the 32k coverage, in both classes. Its parameter cost (merges
  × 512) is stated.
- **DR3, go or no-go per table:** NO-GO if, in both classes, R1-2 reads RARE-WORD SHORTFALL
  and the 32k coverage of real words is below 50%. Then pieces cannot plausibly make the
  missing words whole.
  - If both tables are NO-GO, P1 does not run as designed, and the plan returns to the owner
    with the alternatives (`REVIEW_V2.md` §2.3).
- **DR4, D4:** if valid, D4 joins P1's guards for word order, with its threshold set in P1's
  registration. If not valid, P1 has no word-order judge, and that limit is stated.
- **These rules decide P1's design only.** Nothing in R1 passes or fails an engine.

## 6. Predictions, before any measurement [HYPOTHESIS]

| reading | prediction |
|---|---|
| R1-1 LBL | CONVICTED in both classes (~85%) |
| R1-1 SUB | CONVICTED in phishing (~60%); benign possibly not measurable, or weak |
| R1-1 PTH | CONVICTED in both classes (~90%) |
| R1-1 QRY | benign CONVICTED (~60%, low power); phishing not measurable (~80%) |
| R1-2, benign rest | COMMON-WORD SHORTFALL ~50% |
| R1-2, phishing host | RARE-WORD SHORTFALL ~55% |
| R1-3 | the generated excess is mostly "other" (fluent invention), ~60%; near misses are 15–25% of the fakes |
| R1-4, rest table | 8k coverage of real words 60–75% (benign); 32k 75–85% |
| R1-4, host table | 8k 40–55%; 32k 55–70% |
| R1-5 D4 | valid (~90%); convicts the shuffled URLs (~60%); convicts `e16_b5` (~90%) |
| R1-6 | copies below the natural rate in every field (~80%); MEMORIZED under 1%; validity failures under 1% |
| decision | both tables GO (~70%); arm B kept (~85%); 8k chosen for rest (~55%), more for host (~50%) |

## 7. Steps (the E16/E17 order: each step checked before the next)

1. This registration is committed.
2. The code (`research/v2/v2_bpe.py`, `research/v2/v2_r1.py`) and its tests are committed
   before any run.
3. **data:** build the samples. V-inputs.
4. **bpe:** learn the two tables. V-BPE.
5. **run:** V-D1′ first, then per-field D1, D4 and its controls. V-floor, V-D4.
6. **words, cover, base:** R1-2, R1-3, R1-4 and R1-6.
7. **verify:** every check in §4 → PASS, or the reading stops there.
8. **read:** `R1_RESULTS.md`, read once.
9. **Stop.** The owner reviews.

## 8. Cost

About 2–4 hours of laptop computing, mostly unattended; learning the pieces is the longest
part. No Kaggle.

## Status

- **2026-10-09:** registered, before any R1 code.
- **2026-10-09:** the code committed (cbcc213; full suite 1,077 passed), run step by step,
  VERIFY PASS, read once: `R1_RESULTS.md`.
  - DR1: arm B kept (LBL, SUB). DR2: path table 32k, host table 16k. DR3: GO for both
    tables, each through phishing only. DR4: D4 valid.
  - The main finding: the shortfall is in rare real words, which pieces do not hold.
  - **Stopped for the owner's review.**
