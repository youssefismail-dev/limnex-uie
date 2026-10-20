# The stronger-discriminator test — registration

- **Requested 2026-10-08**, after the big-picture assessment: "If you can
  run this test (like using a character CNN or a more advanced classifier with the files
  we already have), please go ahead and do it."
- **Registered before any result exists.** The code is written after this document is
  committed, and nothing is read until the full run finishes.
- **Informational only.** E3 is closed (`research/E3_FINAL_REPORT.md`). This test reopens
  nothing, changes no threshold or gate, and adopts nothing. It uses only files already on
  disk: no training of the generator, no generation, no new data.

## 1. The question

Every gate cell is judged by a SURFACE detector: character 2–5-grams + logistic regression
(`uie/evaluation.py::real_vs_fake_auc`). Passing it proves there is no surface tell. It
does not prove a stronger detector could not find one. So:

> **Can a stronger whole-URL detector tell `e16_b5`'s free-mode URLs from real held-out
> URLs, beyond the real-vs-real floor?**

## 2. The detectors

| | detector | what it can see that D0 cannot |
|---|---|---|
| **D0** | the gate's own model: char 2–5-gram TF-IDF + logistic regression | — (the reference, run on the same data as the others) |
| **D1** | gradient boosting on global features: field lengths and counts, character-class shares, entropy, depth — plus **vocabulary features**: the share of the URL's word tokens that occur in the real *training* corpus, and their frequencies | whole-URL statistics, and whether the words are *real words* — a direct probe of the "too inventive" finding |
| **D2** | character CNN (URLNet / eXpose style: embedding, parallel convolutions of width 2–6, max-pool, dense) | non-linear combinations of local patterns |
| **D3** | small character transformer (2 layers, positional embeddings, mean-pooled) | interactions across the whole URL (host ↔ path ↔ query) |

D1's vocabulary is built from the real training pools (`*_train`), which a real-world
detector would also have. It never sees the held-out URLs it is tested on.

## 3. The sources tested against real

| source | role |
|---|---|
| **`e16_b5`** free files (5 per class) | **the reading** |
| **`e9_ln` raw** free files (5 per class) | **positive control P1, a known surface tell.** An E7-type tokenizer cannot write the root slash, so its whole URLs carry a tell every working detector must catch |
| **chimeras** (5 constructions per class) | **positive control P2, a coherence tell.** Real held-out URLs whose path and query are replaced by those of *other* real URLs of the same class, with each URL's own presence pattern kept. Every field is genuinely real, so per-field or bag-of-n-gram detectors should see little; a detector that reads the URL as a whole should see the mismatch |

## 4. Protocol

Per class (benign, phishing) and generation g = 0…4:

- **Real data:** the gate's own whole-URL population (the `mixed` validation pool, one URL
  per host, scheme stripped — `FIELDS["FULL"]`): 115,009 benign, 24,081 phishing. Four
  disjoint samples of **N = 6,000**, seeded by (class, g):
  - A and B: the floor;
  - C: the real side of every main comparison;
  - D: hosts and donors for the chimeras.
- **Generated data:** N URLs drawn from generation g's file (8,000 available).
- **Each task:** 5-fold stratified cross-validation, out-of-fold ROC AUC over all 2N URLs.
  - floor = AUC(A vs B);
  - **excess = AUC(C vs source) − floor**, the gate's own quantity.
- Deep models: fixed seeds, fixed epochs, no tuning on any result.

## 5. Decision rules (fixed now)

Each (source, class, detector) is read exactly like a gate cell: the mean excess over the 5
generations, SE, t = (mean − 0.05)/SE against ±2.776 → CLEARED / UNDECIDED / CONVICTED.

**Validity — is each detector working?**
- **V-P1:** a detector that does not CONVICT `e9_ln` in both classes is not working, and
  its `e16_b5` reading is void.
- **V-P2:** a detector's chimera excess measures its sensitivity to whole-URL coherence.
  If it is not above D0's (paired t > 2.776 over the 5 generations), its `e16_b5` reading
  says nothing about coherence. That is reported as a limit, not as a failure.
- **V-floor:** a detector whose mean floor AUC departs from 0.5 by more than 0.03 is
  flagged: something in its pipeline is broken.

**The reading on `e16_b5`:**

| outcome | rule |
|---|---|
| **HOLDS** | every valid detector CLEARS `e16_b5` in both classes |
| **DEEPER TELL** | any valid detector CONVICTS `e16_b5` in a class, or exceeds D0 by more than 0.05 with paired t > 2.776 |
| **MIXED** | anything else |

A DEEPER TELL is not a failure of the engine. It would mean the surface gate was missing
something, and the follow-up is to characterize the tell, not to tune against it.

## 6. Predictions (made before the run)

- **P1:** every detector convicts `e9_ln` strongly. Near-certain.
- **P2:** D0 sees little in the chimeras; D3, and probably D1, see more. Moderate
  confidence (~60%): real URLs' cross-field dependence may be weaker than it seems.
- **`e16_b5`:**
  - D0 at N = 6,000 lands near or slightly above the gate's whole-URL +0.015, because
    more data means a slightly stronger detector.
  - **D1 is the most likely to find a tell**, through the vocabulary features: the engine
    invents words where real URLs use real ones.
  - D2 roughly matches D0: with a few thousand examples per side, a CNN is rarely much
    stronger than TF-IDF + LR.
  - D3 is data-limited.
  - A tell is more likely in phishing than in benign.
  - Overall: HOLDS ~35%, MIXED ~35%, DEEPER TELL ~30%.

## 7. What a positive or negative result licenses

- **HOLDS:** the realism claim survives the stronger detectors tried here, at this sample
  size. It still does not cover a pretrained language-model detector or a human panel.
- **DEEPER TELL:** record which detector, which class and how large, then characterize the
  tell read-only. Nothing is tuned against it, and E3 stays closed.
- **Either way, the next stronger step** would be fine-tuning a small pretrained character
  or byte model (for example ByT5-small or CANINE) as the discriminator, on a Kaggle GPU.
  That needs the owner's approval, because it brings in an external pretrained model.

## 8. Records

Code: `research/e18/strong_disc.py`. Output: `data/e18/strong_disc/`. Result:
`research/e18/STRONG_DISC_RESULT.md`.
