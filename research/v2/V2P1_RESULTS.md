# V2-P1 — results: stopped at K-used (MEMORY UNUSED, by the registered rule)

- **Registration:** `research/v2/PREREG_V2_P1_MEMORY.md` (904a79a; amendment 1, 81060e8).
- **Records:** `data/v2/p1/`: `train/`, `VERIFY_TRAIN.json`, `k_used.json`, `diag_bn.json`,
  `verify_train_job.log`.
- **Read once, 2026-10-13.** VERIFY_TRAIN passed first.
- **Generation did not run.** That is §6's stop: K-used failed.

## In plain words

- **The run itself is valid.** It used the registered start and `e16_cap`'s recipe and data
  exactly. Nothing was clipped after step 1,000, and VERIFY_TRAIN passed.
- **But the trained engine is broken in evaluation mode,** the mode generation uses. On
  validation it scores 4.32 nats per character in benign against `e16_cap`'s 1.61. Paths
  and phishing queries are the worst hit.
- **K-used fails, so by the registration the reading is MEMORY UNUSED and P1 stops before
  generation.** That label is misleading here. The memory *is* used: 84% of its slots are
  chosen. It is used wrongly, and that is my design error (below).
- **The cause, confirmed after the fact: the memory's query BatchNorm.**
  - Training batches are grouped by length, and BatchNorm normalises with each batch's own
    statistics. The engine learned to depend on that.
  - Scored the way it trained (batch statistics, length-grouped batches), it fits as well
    as `e16_cap`, a little better on benign paths.
  - Scored any other way it breaks: with running statistics, random batches, or
    re-estimated statistics.
  - Generation cannot reproduce training's batches, so this engine cannot be used as it is.
- **What it says about the memory idea:** the idea has not yet had a fair test. Even in its
  own conditions, the fit gain was small: benign paths −0.027 nats per character (for scale,
  E16's width step gave −0.032).

## 1. Validity: VERIFY_TRAIN PASS

| check | result |
|---|---|
| V1 | `final.pt` equals the checkpoint training recorded (sha256) |
| V-iso | `uie/` unchanged since 904a79a |
| V-recipe | `e16_cap`'s settings exactly, 2 parts; memory as registered; 161,795,072 parameters |
| V-start | `e16_cap`'s part `caca0107…` on Kaggle; the memory's start as recorded before training |
| V-batch, V-stream, V-mass | `adbddc2a…`; `637accab…`; the injection equals `e16_cap`'s exactly |
| K12e | 0% clipped after step 1,000 in every band |
| tests | V-equiv and the training-path tests pass |
| generation check | 9 URLs generated on the laptop CPU, all parse |

**Resources:**
- Kaggle T4: the training loop took 3.90 h (`e16_cap`'s: 2.73 h); the notebook ran 242 min
  in all.
- Peak GPU memory was 11.2 of 14.6 GB, in 2 parts.
- The laptop's checks took about 25 min.

## 2. K-used (§5.3): FAIL

The rows: 3,991 benign and 3,992 phishing validation rows (§4.6). The figures are nats per
content character.

| | `e16_cap` | `p1_mem` | difference [95% interval] |
|---|---|---|---|
| benign | 1.697 | 4.645 | **+2.947 [+2.846, +3.045]** |
| phishing | 1.701 | 4.163 | +2.463 [+2.357, +2.567] |

- **(a) fit:** FAIL, since the interval is far above 0.
- **(b) usage:** PASS, with 221,230 of 262,144 slots (84.4%) and a KL from uniform of 2.30.
- **The ablation, reported only:** with the memory's output set to 0, benign NLL falls by
  2.90, to about 1.74. In evaluation mode the memory branch actively harms the engine.

## 3. The training record, reported

Evaluation-mode validation loss, benign (`e16_cap` in brackets):

| step | 1,000 | 6,000 | 7,000 | 13,000 | 15,000 | 18,000 | 20,000 |
|---|---|---|---|---|---|---|---|
| `p1_mem` | 2.030 | 1.805 | 2.307 | 3.202 | 3.380 | 3.462 | **4.318** |
| `e16_cap` | 2.016 | 1.559 | 1.551 | 1.592 | 1.603 | 1.607 | 1.607 |

- The loss tracked `e16_cap`'s until about step 6,000, then swung and climbed.
- Training rows score as badly as validation rows, so this is not memorisation.

## 4. The cause (post hoc: `v2_p1_diag_bn.py`, committed 5a99092 before it ran; decides nothing)

The same 1,024 validation rows per class, in nats per content character:

| scoring | benign all | benign PTH | benign QRY | phishing all | phishing PTH | phishing QRY |
|---|---|---|---|---|---|---|
| `e16_cap` | 1.687 | 1.482 | 1.470 | 1.677 | 1.769 | 2.028 |
| A. running statistics (as registered) | 4.529 | 5.775 | 3.899 | 3.969 | 6.131 | 8.092 |
| B. batch statistics, random batches | 2.242 | 2.314 | 1.926 | 2.401 | 3.044 | 4.543 |
| **C. batch statistics, length-grouped batches (as trained)** | **1.677** | **1.455** | **1.477** | **1.673** | **1.784** | **2.024** |
| D. running statistics re-estimated (global) | 2.248 | 2.322 | 1.943 | 2.421 | 3.021 | 4.859 |

- **Only C works,** so the engine depends on how its batch was composed. That cannot be
  reproduced in generation, and re-estimating the statistics (D) does not fix it.
- **The error is mine.** I took BatchNorm from the memory-layer paper without accounting for
  our length-grouped batches. Neither the unit tests nor the CPU dry run could catch it:
  both run single steps, with no long training to drift the running statistics.
- **In its own conditions (C), the memory buys little fit:** benign all −0.010, paths
  −0.027, phishing all −0.004.

## 5. Deviations from the registration

- §10 step 2's notebook simulation was retired after it froze the laptop (Status,
  2026-10-12). This finding is unrelated to that.
- Nothing else.

## 6. What follows (the owner decides; nothing runs before that)

- **The fix:** normalise the memory's query per token (LayerNorm) instead of per batch
  (BatchNorm). Training and generation then compute the same thing. Everything else would
  stay as registered.
- **It needs a new registration and a new training run** (about 4 h on a T4).
- **An honest expectation:** in its own conditions this memory improved fit only slightly
  (paths −0.027 nats per character). A large cut in the word tell is not the likely outcome.
- **The other route** is to close V2's word work with the evidence gathered so far.
