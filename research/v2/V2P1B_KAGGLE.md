# V2-P1b on Kaggle — exact steps

This is the companion to `PREREG_V2_P1B.md` (§7). There are two notebooks, and you run
both, one at a time. Verification, scoring and every reading run on the laptop.

## Upload: one new dataset

`uie-v2p1b-code.zip` is built by `research/v2/make_v2p1b_bundle.py` from a clean, committed
tree. The commit and the zip's sha256 are in `PREREG_V2_P1B.md`'s Status. It holds:
- `uie/`, with T2's optional optimiser hook, off by default;
- the common modules E8 to E17;
- V2's P1b modules: `v2_mem.py`, `v2_p1_common.py`, `v2_p1b_common.py` and
  `v2_p1b_train.py`;
- `qry_extra.jsonl`: stream 1, the 57,292 rows `e16_cap` trained with;
- `e16_cap_record.json`: `e16_cap`'s tokenizer and settings;
- the three tables;
- `V2P1B_CODE_MANIFEST.json`, which both notebooks check first.

Upload it like this:
1. Kaggle → *Datasets* → *New Dataset*.
2. Upload `uie-v2p1b-code.zip` as is.
3. Title it **`uie-v2p1b-code`**, set it to **Private**, then *Create*.

## Order

| # | notebook | inputs to attach | about | download |
|---|---|---|---|---|
| 1 | `V2P1B_train.ipynb` | `uie-corpus2`, `uie-v2p1b-code` | 4–4.5 h | `p1b_mem/final.pt` (about 620 MB), `p1b_mem/train_result.json`, `PROVENANCE.json`. **Not** `best.pt` |
| 2 | `V2P1B_generate.ipynb` — **only after I say the laptop's checks passed** | `uie-v2p1b-code`, the **V2P1B_train** notebook's output | 3–3.5 h | `v2p1b_gen/` (about 25 MB) |

- **After 1, wait for the laptop's checks** before running 2. They take about 30 min:
  VERIFY_TRAIN, K-used and the audit on the trained model.
- If K-used fails, notebook 2 is not run. That is a registered stop.

For each notebook:
1. *Code* → *New Notebook* → *File* → *Import Notebook* → the `.ipynb`.
2. *Add Input*, as in the table.
3. *Settings* → GPU **T4** (check it is really on), Internet **On**.
4. *Save Version* → *Save & Run All*.

## What to check

**V2P1B_train**

| cell | must show |
|---|---|
| 2 | `code: commit …; … stream 1 (57,292 rows) and e16_cap's record verified`; `torch: … \| gpu: Tesla T4`; `data bundle: all 8 streams present; e16_cap's record loaded` |
| 3 | in this order: **`T4 audit PASSED: largest gap … logits (bar 0.05); no stored statistic changed`**; `width 384 rebuilds e12_final's recorded start; width 512 rebuilds e16_cap's (caca010736da...)`; `p1b_mem's start: e16_cap's part caca010736da..., memory …, 161,792,000 parameters`; `start check PASSED; memory check PASSED in 2 parts (… of 14.6 GB)` |
| 4 | in this order: **`T4 audit (in run, the exact model about to train): {… 'pass': True …}`**; `model: 161.79M parameters`; `E15: query injection ON -> …`; then the training log: a loss line every 200 steps and an `eval` line every 1,000 |
| 5 | `MISMATCHES: none -- p1b_mem is as registered` |

**V2P1B_generate**

| cell | must show |
|---|---|
| 1 | `checkpoint verified` |
| 3 | `50 files, 180,000 URLs; decode {… 'slash_rule': True, 'slash_paths': True, 'query_rule': True}`; `loaded p1b_mem: width 512, memory {…}` |
| 4 | `all 50 files written` |

**If a cell stops with "STOP", nothing was trained or written.** Report it to me, and do not
work around it. The registered stops before training are:
- the start check, which fails if Kaggle's torch has changed;
- **the batch-coupling audit (T4)**;
- the memory check, which fails if even 4 parts do not fit.

There is no tripwire: once training starts, it runs its full 20,000 steps.

## Back on the laptop

1. Put `V2P1B_train`'s output in `data/v2/p1b/train`, keeping the `p1b_mem` folder. Then I
   run, at low priority with the low-RAM alarm:
   - `V2_ARM=p1b python research/v2/v2_p1_verify.py --step train`;
   - `V2_ARM=p1b python research/v2/v2_p1_fit.py`.
2. Put `V2P1B_generate`'s output in `data/v2/p1b/generate/gen`. Then VERIFY_GENERATE,
   scoring and the reading, with `V2_ARM=p1b`, as P1 registered.
