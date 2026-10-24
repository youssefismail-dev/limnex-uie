# U1 Q2: the detector-training protocol

- **What this is:** the implementation choices for Q2 (`PREREG_U1.md` §4, with §1 and Amendments
  1–2) that the registration left open.
- **Fixed 2026-10-21, after Q1 was read and before any Q2 URL is generated, any detector is
  trained or any test set is scored.**
  - Q1 read only realism scores on validation pools; it touched no test set.
  - Any change after a Q2 step runs is recorded here as a dated amendment.
- **Code:**
  - Q1's `q1_common.py` and `q1_generate.py` generate.
  - `q2_data.py` composes and freezes the training data.
  - `q2_detect.py` trains, selects and tests.
  - `make_q2_kaggle.py` builds the notebooks.

## 1. The synthetic training sets
| | rule |
|---|---|
| **generators** | `uie`, `ngram`, `lstm`, `llm` (B4 `twist` is excluded, §2) |
| **one stream per generator and class** | files g = 0, 1, 2, … made by Q1's code at Q1's seeds (`base + 100_000·c + 1_000·g + k`). **g0–g4 are Q1's files, unchanged**; Q2 generates g ≥ 5 the same way, with the same fresh gate per file. |
| **the set** | the **first 100,000 distinct URLs** of the stream: files in order of g, URLs in file order, a URL already taken skipped. Files are generated in order until 100,000 is reached (planned: g5–g12, and g5–g13 for `llm`, whose phishing stream repeats more: 1.0% across Q1's files, against at most 0.24% for the others). |
| **B2** | the LSTM trained in Q1 (`lstm_b2.pt`, sha256 `b0bc64ec16675fdae78fabbda1f5c49fd722d50882e76cc9601bdf38463efd8f`), loaded, not retrained |
| **B3** | as in Q1: same model, revision, prompt, sampling and `transformers` backend; refusals counted |
| **use** | for volume v, a set contributes its **first v URLs**. Every arm, regime and r reads a prefix of the same list, so smaller sets are nested in larger ones. |

## 2. Inputs every detector sees
- **Real training:** `data/u1/pools/{class}_order.jsonl.gz` (F1).
  - Regime R is ranks 0 … R−1.
  - A_more adds ranks R … 2R−1 (A1.2).
- **Validation:** `data/u1/pools/{class}_val.jsonl.gz`, 20,000 per class (F1).
- **The scheme is dropped from every URL**, in training, validation and test alike
  (`uie.factorization.strip_scheme`).
  - **Why:** real URLs carry `http(s)://`, and synthetic ones never do (they are written as UIE
    writes them). Real phishing also uses `http://` far more often than real benign (37% against
    13% in the pools).
  - **Without stripping:** a detector would learn "no scheme = synthetic" and "http = phishing",
    neither of which is about the URL.
  - This is E3's rule (its FULL detector strips the scheme from real URLs) and LIMNEX v3.5's.
- **Labels:** benign 0, phishing 1.

## 3. Arms (§4.2), per detector and regime R ∈ {1k, 10k, 100k}, per class
| arm | training rows per class |
|---|---|
| A0 | real[0 : R] |
| A_G(r), r ∈ {0.25, 0.5, 1.0} | real[0 : R] + synth_G[0 : r·R] |
| A_S(G) | synth_G[0 : R] |
| A_more | real[0 : 2R] |

That is 1 + 12 + 4 + 1 = **18 arms per detector and regime**.

## 4. Detectors (§4.1)
### D1, the character CNN
> **Amended 2026-10-23 (`Q2_AMENDMENT_1.md`):** the head's hidden activation is LeakyReLU(0.01).
> The registered ReLU let 29 of 270 runs die. Everything else below stands.

- **Architecture:** E18's `CNN` (`research/e18/strong_disc.py`).
  - Its fixed character map: printable ASCII to 2–96, anything else 1, 0 = pad.
  - Truncation at 160 characters.
  - Embedding 32, five convolutions of 64 channels (kernels 2–6), masked max-pool.
  - Head: dropout 0.3, then 320 → 64 → 1.
- **Training:** 6 epochs, Adam at 2e-3, BCE with logits.
  - Length-bucketed batches (sorted by length, chunked, chunk order shuffled each epoch by the
    run's seed), as E18.
- **Batch 256, as the registration states.** E18's own code trains at 128 and scores at 256; the
  registration's figure is the binding one, so 256 is used.
- **Seeds:** `40_000_000 + k`, k = 0 … 4. The same five seeds in every arm, so arms are paired by
  seed. The seed sets PyTorch's initialisation and the batch order.
- **Runs on a Kaggle GPU.** GPU kernels are not bit-deterministic, and that is recorded, not
  corrected.
- **Known property, reported and not corrected:** a fixed epoch count means more rows give more
  steps. At R1k, A0 trains for only 48 steps (2,000 rows / 256 × 6), so part of any gain from added
  rows may be the extra steps.
  - A_more shows the same effect with real rows.
  - **S2 compares generators at equal row counts, so it is not affected.**

### D2, the character n-gram logistic regression
- **Model:** `HashingVectorizer(analyzer='char', ngram_range=(1, 5), n_features=2**20)` +
  `LogisticRegression(C=1.0, max_iter=1000)`, with scikit-learn's defaults otherwise.
- **D2 is deterministic.** The default solver (lbfgs) has no random state, and row order does not
  change its solution. Its five seeds would be five identical models, so it is **fitted once per
  arm**, and its seed mean equals that single model.
- **Runs on a Kaggle CPU.**

### Both detectors
**The score** is the model's phishing logit (D1) or decision function (D2); higher means phishing.

## 5. Selection (§4.4) and freeze F2
- **The measure:** recall at 1% FPR on the validation set.
  - The threshold t is the k-th smallest benign validation score, k = ⌈0.99·n⌉.
  - A URL is flagged when its score is > t.
- **The choice:** r\* for each (detector, regime, generator) is the r with the highest mean over
  seeds. **On a tie, the smaller r.**
- **Recorded in `research/u1/F2_SELECTION.json` and committed before any test set is uploaded or
  scored.** It holds:
  - every validation number;
  - r\*;
  - the sha256 of every trained model.
- The test stage refuses models whose sha256 differs.

## 6. The test read (§4.4–4.6, freeze F3)
- **Scored once:** A0, A_more, A_S(G) and A_G(r\*), for every detector and regime, on:
  - **T1x** (primary, Amendment 2);
  - **T1** (the registered window, secondary);
  - **T2 PhreshPhish** and **T2 PhiUSIIL**;
  - **T3** (sanity).
- **Measures:** recall at 1% and 5% FPR (threshold on that set's own benign, as in §5) and ROC AUC.
- **Δ1 (S1)** = mean over seeds of recall@1%(A_UIE(r\*)) − mean over seeds of recall@1%(A0).
- **Δ2 (S2)** = mean recall@1%(A_UIE(r\*)) − the **largest** mean recall@1%(A_B(r\*_B)) over
  B ∈ {ngram, lstm, llm}. That largest is taken again inside each bootstrap resample, which can
  only make S2 harder to pass.
- **Bootstrap:** 2,000 resamples, seed `20261027`.
  - Phishing and benign are resampled separately, each **by registrable domain** (`uie`'s
    `group_key`), so URLs on one platform domain move together.
  - In each resample, every model's threshold and recall are recomputed.
- **The one-sided p-value:** p = (1 + #{Δ\* ≤ 0}) / (2,000 + 1).
- **The decision:**
  - **S1 and S2 are each a family of 6 tests** (2 detectors × 3 regimes), Holm-adjusted at
    α = 0.05.
  - A test "holds" when its Holm-adjusted p is below 0.05. That is the Holm form of "the one-sided
    lower bound is above 0".
  - The unadjusted 95% lower bound (the 5th percentile of Δ\*) is reported beside it.
- **The registered window (T1):** the same computation, reported as secondary.
- **T2 and T3:** reported beside every claim, without a test (§4.6).
- **Overlap (§1.3):** the share of each synthetic set whose host (phishing) or registrable domain
  (benign) occurs in T1x or T2, reported and not filtered.

## 7. Where each step runs
| step | where | notebook or command |
|---|---|---|
| generate g ≥ 5 | Kaggle | `Q2_gen_llm_benign`, `Q2_gen_llm_phishing` (GPU T4 x2 each), `Q2_gen_uie_ngram_lstm` (GPU T4 x1) |
| compose the sets, freeze the training data | laptop | `q2_data.py compose`, then `freeze` |
| train D1, score validation | Kaggle GPU | `Q2_train_cnn` |
| train D2, score validation | Kaggle CPU | `Q2_train_lr` |
| select r\*, freeze F2 | laptop | `q2_detect.py select` |
| score the test sets, bootstrap, Holm | Kaggle CPU | `Q2_test`, after F2 is committed |
| write U1_RESULTS Q2 | laptop | — |

- **Safety:**
  - Every Q2 notebook and dataset stays PRIVATE.
  - Generated phishing-style URLs and the T1 phishing set are never published, resolved or
    visited (§8). The test sets are uploaded only after F2.
