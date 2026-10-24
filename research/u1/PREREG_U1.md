# U1: does Limnex UIE help phishing detectors? Registration

- **Why, 2026-10-20:** the release-readiness assessment named two open gaps:
  (1) whether training a detector on UIE's generated URLs improves it on real phishing, and
  (2) how UIE compares with other generators.
- **Status: registered before any generator is trained, any data is drawn or any detector runs.**
  Every rule below is fixed now. A rule that must change before its freeze point (section 7) is
  changed in this file, in a dated amendment, before the data it touches is read.
- **The released model is not changed by U1.** `e16_b5` stays the published engine.

## In plain words

Three questions, answered in this order:

| | question | where |
|---|---|---|
| **Q1** | Are UIE's URLs more realistic than those of simpler generators, on the same field-by-field test UIE already passes? | this repository |
| **Q2** | Does adding generated URLs to a detector's training data help it catch **real, future** phishing, and does UIE help more than simpler generators? | this repository |
| **Q3** | Does the single best Q2 setting improve the real LIMNEX detector? | the LIMNEX repository, registered there |

Q1 is cheap and needs no new real data. Q2 is the main experiment. Q3 is one run, chosen by Q2,
read once.

---

## 1. Data

### 1.1 Real training pools
- From `e16_b5`'s own training partitions in the bundle: benign = `name_train` +
  `structure_train`; phishing = `phishing_name_train` + `phishing_structure_train`.
- **Three regimes, per class:** R1k = 1,000, R10k = 10,000 and R100k = 100,000 URLs. They are
  nested (R1k ⊂ R10k ⊂ R100k), drawn once with seed `20261020`, and their sha256 is recorded at F1.
- **Validation (model selection only):** 20,000 per class from `name_val` + `structure_val` and
  `phishing_name_val` + `phishing_structure_val`, seed `20261021`.

### 1.2 Test sets (never used for training or selection)
- **T1, "future", the primary test set.**
  - **Phishing:** OpenPhish public-feed records **first seen after 2026-10-05T12:00:06Z** (the last
    feed commit any UIE training data read, `d0642d73…`), up to a pin commit fixed and recorded at
    F1. One URL per host, by `merge_phishing`'s host rule.
  - **Benign:** one Common Crawl crawl **newer than `CC-MAIN-2026-30`**, extracted with
    `uie.corpus.extract` at seed `20261022` with a reduced block budget fixed at F1 (about 1,000
    blocks; the owner's internet is limited). One URL per registrable domain;
    n = max(20,000, 5 × the phishing count).
  - **Exclusions:** any host or registrable domain present anywhere in UIE's training data (the
    corpus2 train and val partitions, both phishing corpora, the E15 and E17 query streams).
- **T2, external.** PhreshPhish (test split) and PhiUSIIL, from their public sources, with
  versions and sha256 recorded at F1, and with registrable domains in UIE's training data removed.
- **T3, in-distribution.** UIE's own held-out test partitions. A sanity check only, never a
  success criterion.

### 1.3 Integrity
- **No test URL, host or domain is used in training, generation or selection.**
- Overlap between synthetic data and test domains is **reported, not filtered**. Filtering would
  use the test set.
- Everything generated or collected for U1 stays under git-ignored `data/u1/`.

---

## 2. Generators

All are trained on, or prompted from, **the same real data UIE was trained on**: the bundle's
training partitions plus the E15 stream. All pass through **the same known-URL filter** before use.

| | generator | detail |
|---|---|---|
| **G0** | **UIE `e16_b5`** | the public API, free mode, filter required; seeds from `30_000_000`, disjoint from every gate seed |
| **B1** | **character n-gram** | `uie.ngram.CharNgram(order=5)`, the project's own E1 baseline |
| **B2** | **character LSTM**, class-conditional | 2 layers of 512 units, UIE's tokenizer, trained for a fixed budget of 2 GPU-hours on a Kaggle T4; the DeepPhish-style approach |
| **B3** | **an open-weight instruct LLM**, about 7–8B parameters | name and revision fixed at F1; run on Kaggle, no commercial API; few-shot prompt (appendix A) with 20 real training URLs per call; temperature 1.0; refusals counted and reported |
| **B4** | **rule-based look-alikes** (phishing class only) | dnstwist-style permutations of real benign training domains (homoglyph, insertion, omission, transposition, hyphenation, suffix swap), plus paths drawn from the 500 most frequent phishing training path templates |

**B4 has no benign class.** It enters Q1 for the phishing cells only, and is **excluded from
Q2**: adding synthetic URLs to one class only gives a detector a "synthetic = phishing" shortcut
(see 4.3).

**Volumes:**
- **Q1:** 8,000 URLs × 5 generations per class per generator, as in E3's free mode.
- **Q2:** up to 100,000 per class per generator, generated once and reused by every arm.

---

## 3. Q1: the gate comparison

- **The instrument:** E12's free-mode instrument, unchanged (`research/native/e12_score.py`
  `part_free`), against the same real pools as E3. It gives each generator's x (excess over the
  real-vs-real floor) and its verdict per field: LBL, SUF, SUB, PTH, QRY and the auxiliary FULL.
- **Also reported per generator:**
  - **validity**, the share passing the gate's validity checks;
  - **copy rate**, the share removed by the known-URL filter;
  - **diversity:** distinct registrable domains per 8,000, and suffix entropy;
  - **the word tell** (E18's word-lookup detector, informational).
- **The readout is descriptive, with intervals; there is no pass/fail.** For each field and each
  baseline B: the difference x(UIE) − x(B), with a Welch interval over the 5 + 5 generation means.
  **"UIE more realistic than B on field f"** is claimed only where the one-sided 95% upper bound of
  that difference is below 0.
- **Control mode is UIE-only** (the baselines cannot take plans), so it is not compared.

---

## 4. Q2: the detector-training experiment

### 4.1 Detectors (fixed now; no tuning)
- **D1, a character CNN**, with E18's architecture and schedule (`research/e18/strong_disc.py`):
  embedding 32, convolutions of 64 channels with kernels 2–6, max-pool, 6 epochs, Adam at 2e-3,
  batch 256.
- **D2, a character n-gram logistic regression:** scikit-learn
  `HashingVectorizer(analyzer='char', ngram_range=(1, 5), n_features=2**20)` +
  `LogisticRegression(C=1.0, max_iter=1000)`.

### 4.2 Arms (for each detector × regime R ∈ {R1k, R10k, R100k})
| arm | training data |
|---|---|
| **A0** | real only (R per class) |
| **A_G(r)** | real + synthetic from generator G ∈ {UIE, B1, B2, B3}, **r × R per class**, r ∈ {0.25, 0.5, 1.0} |
| **A_S(G)** | synthetic only from G, R per class |
| **A_more** | real + R **additional real** URLs per class: the upper reference ("what if you simply had more real data?") |

Each arm is trained with **5 seeds**. The data samples are fixed; the seeds change initialisation
and order only.

### 4.3 The shortcut guard
E18 found that a word-aware detector separates UIE's URLs from real ones, because UIE invents
words. **Synthetic URLs are therefore always added in equal numbers to both classes**, so that
"looks generated" carries no class information.

### 4.4 Selection, then one test read
- **r\*** for each (detector, regime, generator) = the r with the highest mean **validation**
  recall at 1% FPR over the 5 seeds. Recorded at **F2**, before any test score is computed.
- The test sets are then scored **once**, for A0, A_more, A_S(G) and A_G(r\*).

### 4.5 Measures
- **Primary: recall at 1% false-positive rate on T1.** The threshold is set where the model's FPR
  on T1's benign URLs is 1%. This is LIMNEX's own primary measure.
- **Secondary:** recall at 5% FPR, ROC AUC, on T1 and T2; T3 as the sanity check.

### 4.6 Success criteria (6 tests: 2 detectors × 3 regimes; Holm-adjusted at family α = 0.05)
- **S1, UIE helps.** Δ1 = mean over seeds of recall@1%(A_UIE(r\*)) − recall@1%(A0). **Holds if
  the one-sided 95% lower bound of Δ1 is above 0**, from a bootstrap that resamples T1's
  registrable domains (2,000 resamples), Holm-adjusted.
- **S2, UIE helps more than simpler generators.** Δ2 = recall@1%(A_UIE(r\*)) − the best of
  recall@1%(A_B(r\*_B)) over B1–B3. **Holds if its one-sided lower bound is above 0**, by the same
  method, Holm-adjusted.
- **Allowed claims:**
  - "UIE's data improves detectors" only where S1 holds;
  - "UIE's data improves them more than simpler generators" only where S1 **and** S2 hold.
  
  T2 results are reported beside every claim (does it generalise?), but are not gated.
- **A_more is never a criterion.** It shows how close synthetic data comes to real data, and
  synthetic data is not expected to beat it.

---

## 5. Q3: the LIMNEX case study (registered in the LIMNEX repository before it runs)

- **One configuration only:** the Q2 setting (detector family, r\*) with the best validation
  result, applied to the LIMNEX members of the matching family. That means the CNN member for a D1
  win, and the CatBoost member for a D2 win (both if both won). They are retrained on LIMNEX's own
  training corpus plus UIE synthetic URLs at r\*, equal per class, with their existing
  hyperparameters, on Kaggle.
- **Fusion weights stay v3.7's** (benchmark_v2's DEV split is spent; nothing is refitted).
- **LIMNEX's frozen TEST (41,808) is read once.**
  - Primary: recall at 1% FPR against v3.7.
  - Reported both **without and with** the UIE scorer member, because augmentation and the scorer
    are different mechanisms that may add up or overlap.
  - Same one-sided bootstrap rule as S1.
- **The served LIMNEX model changes only by the owner's separate decision.**

---

## 6. Reported whatever the outcome
Every number of Q1 and Q2, the null results included, goes in `research/u1/U1_RESULTS.md`:
- the T1 and T2 composition and overlaps;
- B3's refusal rate;
- the compute used.

Nothing is dropped because it disappoints.

## 7. Freeze points, in order
1. **F0:** this registration, committed.
2. **F1, the data freeze** (before any test score): the T1 feed pin and crawl, the T2 versions, the
   pools' sha256, B3's model name and revision. Recorded in `research/u1/F1_FREEZE.json`.
3. **Q1** runs and is read (it touches no test set).
4. **F2:** Q2 trained; validation selection (r\*) recorded in `research/u1/F2_SELECTION.json`.
5. **F3:** the Q2 test sets scored once; U1_RESULTS written.
6. **Q3**, registered and run in LIMNEX.

## 8. Safety and compute
- Generated phishing-style URLs and the T1 phishing set are **never published, resolved,
  registered or visited**. They stay in `data/u1/`. Results report counts and rates only.
- **All training runs on Kaggle** (generators, detectors, LIMNEX members). The laptop prepares data
  and computes metrics only.
- **Rough cost:** Q1 about a week; Q2 about two to three weeks of Kaggle sessions (generation about
  3 h per generator; about 70 detector runs); Q3 one Kaggle run.

---

## Amendment 1 (2026-10-20): F1's open details, before any U1 data is drawn

Written before the feed is fetched, the crawl is read, any pool is drawn or any T2 row is filtered.
Nothing below was chosen after seeing U1 data.

### A1.1 The exclusion key, per class (corrects §1.2)
§1.2 excludes test rows whose "host or registrable domain" is in UIE's training data. Read literally,
that removes every phishing URL on a shared platform, because the platform's registrable domain
(`webflow.io`, `vercel.app`, `pages.dev`, `github.io`, …) is in the training data.
- **Measured on UIE's own held-out phishing** (the 24,082 hosts of the host split's *test* partition,
  never trained on): the literal reading removes **15,367 of 24,082 (63.8%)**; by host alone, 0.
  The extra removals are platforms: webflow.io 1,852, vercel.app 1,767, pages.dev 1,633, github.io
  1,624, weebly.com 610, …
- A test set without platform phishing (73% of the feed, `merge_phishing.py`) would be easier and
  unrepresentative.
- **The rule therefore follows UIE's own splits:**
  - **phishing** is keyed by **host** (`group_key_host`, as in `split_phishing_host`);
  - **benign** is keyed by **registrable domain** (`group_key`, as in `split_random`).
- **Exclusion rules:**
  - A **phishing** test row is dropped if its host is in any training host set (phishing or benign).
  - A **benign** test row is dropped if its registrable domain is in the benign or the phishing
    training domains, or its host is in any training host set.
- **"UIE's training data"** means:
  - `bundle2`'s `name_*`, `structure_*`, `phishing_name_*` and `phishing_structure_*` train and val
    files;
  - every top-level `.jsonl` of the E15 (`data/e15/qry_extra/`) and E17 (`data/e17/qry_extra2/`)
    query streams. That is a superset of the trained streams, so it can only exclude more.
- **Also reported, not used:** how many T1 and T2 phishing rows the literal reading would have
  removed.
- **Label hygiene for T1 benign** (E15's P7 rule): a benign row whose host or registrable domain
  appears anywhere in the OpenPhish archive at the T1 pin is dropped.

### A1.2 Seeds and draws
| what | rule |
|---|---|
| **pools** (§1.1) | per class: the train files' rows, de-duplicated by URL and sorted, then shuffled once with `random.Random(20261020)`. R1k, R10k and R100k are the first 1k / 10k / 100k. **A_more** for regime R is ranks R … 2R−1 of the same order, so it is disjoint from R. |
| **validation** (§1.1) | per class: the val files' rows, de-duplicated and sorted, then `random.Random(20261021).sample(…, 20000)` |
| **T1 phishing** | `uie.corpus.feeds` walk of the full archive up to the pin (origin HEAD at fetch time). Keep URLs first seen after 2026-10-05T12:00:06Z. Pick one URL per host with `random.Random(20261023).choice`, hosts in first-seen order. |
| **T1 benign budget** | `blocks = ceil(1.25 × n_target / 28.6)`, where `n_target = max(20000, 5 × n_T1_phishing)` and 28.6 is the lowest domains-per-block of the 8 training crawls (CC-MAIN-2026-30). Capped at 2,000 blocks (about 0.8 GB, for the owner's connection). |
| **T1 benign draw** | `extract_crawl` at seed 20261022 on the newest crawl after CC-MAIN-2026-30 listed in `collinfo.json` at fetch time. One URL per registrable domain (`merge.py`'s reservoir, k = 1, `random.Random(20261024)`), then the exclusions. Then the first `n_target` by `sha256("U1-T1B:" + domain)` order. If fewer than 20,000 remain, one top-up of the missing blocks at seed 20261022 + 1,000,003. Otherwise all available rows are kept and the shortfall against `n_target` is reported. |
| **T3** | benign: the `name` corpus rows in `split_random`'s test partition. Phishing: the archive's test-partition hosts, one URL each with `random.Random(20261025).choice`. |
| **T2** | PhreshPhish `test-*` (the local copy `phreshphish_urls.csv`, 104,514 rows: `phish` = phishing). PhiUSIIL (the local `PhiUSIIL_Phishing_URL_Dataset.csv`, 235,795 rows: **label 0 = phishing, 1 = legitimate**). De-duplicated by URL, then the A1.1 exclusions. |

### A1.3 B3
**`Qwen/Qwen2.5-7B-Instruct`** (Apache-2.0, about 7.6B parameters, no access gate). The Hugging Face
commit is pinned at F1.

---

## Amendment 2 (2026-10-20): T1 starts where UIE's training data ends

**Made after F1 (`259e6d4`) and before any detector, generator or score exists.** Decided by the
owner on 2026-10-20.

### A2.1 The correction
- §1.2 starts T1 phishing after 2026-10-05T12:00:06Z, described as "the last feed commit any UIE
  training data read". **That description was wrong about training.**
- **UIE's phishing training data ends at 2026-08-03T00:00:08Z.** That is the latest `first_seen` in
  `phishing_name_train` and `phishing_structure_train`, from the corpus pinned at `48f7a982b2a4`.
- The feed between 2026-08-03 and 2026-10-05 was *read* by E15's acquisition, but **none of it was
  trained on**:
  - **E15's 918 phishing query rows** from that window (`phishing_P0Q1` 355, `phishing_P1Q1` 563)
    were **not admitted** (`PREREG_E15.md`, "Phishing: not admitted"). They are in U1's exclusion
    list (A1.1), so they stay out of T1 regardless.
  - **E15 and E17 used the window's phishing hosts and domains only to remove benign rows** from
    their query streams (rule P7): 13 domains (E15) and 7 (E17), out of about 1.06 million candidate
    domains each. Nothing about those phishing URLs entered UIE's training.
- The 2026-10-05 line was therefore a cautious "last contact" line, not the training cutoff. It
  discarded **19,398 never-trained phishing hosts** first seen between 2026-08-03 and 2026-10-05.

### A2.2 The change
- **T1 phishing (primary)** becomes every phishing URL first seen **after 2026-08-03T00:00:08Z**, up to
  the **same F1 pin `290d4544`** (2026-10-20T12:00:06Z). The end does not move.
  - The rules are unchanged: one URL per host (`random.Random(20261023)`) and the A1.1 exclusions.
  - **24,588 hosts**, against 5,372 for the registered window.
- **T1 benign** follows the registered rules unchanged. `n_target = max(20000, 5 × 24,588)` gives
  `block_budget` = the A1.2 cap of **2,000 blocks**.
  - 1,174 blocks were drawn at F1 (seed 20261022, round `r00`, kept as they are).
  - The other **826** are drawn as round `r01` at seed 20261022 + 1,000,003.
  - The selection (A1.2) then runs over both rounds. Because of the cap, T1 benign holds all
    available rows, and the shortfall against `n_target` is reported.
- **The registered window is still reported.** F1's original files (`t1_phishing`, 5,372; `t1_benign`,
  26,860) stay byte-for-byte as frozen in `F1_FREEZE.json`. U1_RESULTS reports S1 and S2 on them too,
  beside the primary results.
- The new files are frozen in `research/u1/F1_AMENDMENT2_FREEZE.json`, which also records the sha256
  of `F1_FREEZE.json`.

### A2.3 Why this is not tuning to the result
- **What existed when the owner decided:** counts only.
  - T1's funnel (5,372 hosts; 217 dropped on training hosts) and T1 benign's funnel.
  - The gap's size (19,398 hosts over 64 days, about 303 a day, against 358 a day in T1).
  - T2's and T3's sizes.
- **What did not exist:** any detector, any generated URL from any U1 generator, any score on any U1
  test set, and any reading of individual test URLs.
- **The reason is precision, judged before any result.** With 5,372 phishing hosts the smallest
  recall difference S1/S2 can reliably show is roughly 2 points. With 24,588 it is roughly 1 point,
  which matters for R100k and for S2, where small differences are expected.
- **Every hold-out property stays:**
  - every T1 URL is first seen after UIE's last training URL;
  - no T1 host is a training host;
  - T1 still ends at the F1 pin.

---

## Appendix A: B3 prompt template (fixed)

> You are helping build a **training set for a phishing detector**. Below are 20 real `{label}` URLs
> from a research corpus. Write `{k}` new, different URLs of the same kind, one per line, with no
> numbering, no explanation and no real company login pages. Do not copy any example.
>
> `{20 examples, one per line}`

`{label}` is "benign" or "phishing-style"; `{k}` = 20. Outputs pass the same validity and
known-URL filter as every generator. A response with no valid URL counts as a refusal.
