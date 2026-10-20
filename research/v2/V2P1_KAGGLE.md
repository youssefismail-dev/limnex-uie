# V2-P1 on Kaggle — exact steps

This is the companion to `PREREG_V2_P1_MEMORY.md` (§10). There are two notebooks, and you
run both, one at a time. Verification, scoring and every reading run on the laptop.

## Upload: one new dataset

`uie-v2p1-code.zip` is built by `research/v2/make_v2p1_bundle.py` from a clean, committed
tree. The commit and the zip's sha256 are in `PREREG_V2_P1_MEMORY.md`'s Status. It holds:
- `uie/`, unchanged;
- the common modules E8 to E17;
- V2's three P1 modules: `v2_mem.py` (the memory layer), `v2_p1_common.py` and
  `v2_p1_train.py`;
- `qry_extra.jsonl`: stream 1, the 57,292 rows `e16_cap` trained with;
- `e16_cap_record.json`: `e16_cap`'s tokenizer and settings, so no earlier notebook output
  needs attaching;
- the three tables;
- `V2P1_CODE_MANIFEST.json`, which both notebooks check first.

Upload it like this:
1. Kaggle → *Datasets* → *New Dataset*.
2. Upload `uie-v2p1-code.zip` as is.
3. Title it **`uie-v2p1-code`**, set it to **Private**, then *Create*.

## Order

| # | notebook | inputs to attach | about | download |
|---|---|---|---|---|
| 1 | `V2P1_train.ipynb` | `uie-corpus2`, `uie-v2p1-code` | 4–4.5 h | `p1_mem/final.pt` (about 620 MB), `p1_mem/train_result.json`, `PROVENANCE.json`. **Not** `best.pt` |
| 2 | `V2P1_generate.ipynb` — **only after I say the laptop's check passed** | `uie-v2p1-code`, the **V2P1_train** notebook's output | 3–3.5 h | `v2p1_gen/` (about 25 MB) |

- **After 1, wait for the laptop's check** before running 2. It takes about 1.5 h:
  VERIFY_TRAIN, plus the check that the memory is actually used.
- If the memory is not used, notebook 2 is not run. That is a registered stop.

For each notebook:
1. *Code* → *New Notebook* → *File* → *Import Notebook* → the `.ipynb`.
2. *Add Input*, as in the table.
3. *Settings* → GPU **T4** (check it is really on), Internet **On**.
4. *Save Version* → *Save & Run All*.

## What to check

**V2P1_train**

| cell | must show |
|---|---|
| 2 | `code: commit …; … stream 1 (57,292 rows) and e16_cap's record verified`; `torch: … \| gpu: Tesla T4`; `data bundle: all 8 streams present; e16_cap's record loaded` |
| 3 | `width 384 rebuilds e12_final's recorded start; width 512 rebuilds e16_cap's (caca010736da...)`, then `p1_mem's start: e16_cap's part caca010736da..., memory …, 161,795,072 parameters`, then `start check PASSED; memory check PASSED in 2 parts (… of 14.6 GB)` |
| 4 | `E15: query injection ON -> …`, `model: 161.80M parameters`, then the training log: a loss line every 200 steps and an `eval` line every 1,000 |
| 5 | `MISMATCHES: none -- p1_mem is as registered` |

**V2P1_generate**

| cell | must show |
|---|---|
| 1 | `checkpoint verified` |
| 3 | `50 files, 180,000 URLs; decode {… 'slash_rule': True, 'slash_paths': True, 'query_rule': True}`; `loaded p1_mem: width 512, memory {…}` |
| 4 | `all 50 files written` |

**If a cell stops with "STOP", nothing was trained or written.** Report it to me, and do not
work around it. The registered stops are:
- the start check, which fails if Kaggle's torch has changed since E17;
- the memory check, which fails if even 4 parts do not fit.

## Back on the laptop

1. Put `V2P1_train`'s output in `data/v2/p1/train` (`p1_mem/final.pt`,
   `p1_mem/train_result.json`, `PROVENANCE.json`). Then run VERIFY_TRAIN and K-used,
   before step 2 runs.
2. Put `V2P1_generate`'s output in `data/v2/p1/generate/gen`. Then run VERIFY_GENERATE,
   scoring, the criteria, and `V2P1_RESULTS.md`.
