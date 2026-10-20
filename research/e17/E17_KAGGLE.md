# E17 on Kaggle — exact steps

Companion to `PREREG_E17.md` (§7; amendment 1). Four notebooks; you run three of them.
Verification, scoring and every reading run on the laptop.

## Upload: ONE new dataset

`uie-e17-code.zip`, built by `research/e17/make_e17_bundle.py` from a clean, committed
tree (the commit and the zip's sha256 are in `PREREG_E17.md`'s Status). It holds:
- `uie/`, with E17's two options (B5 and the injector's rate above 1), off by default;
- the common modules E8 to E17;
- `qry_extra_s12.jsonl`: streams 1 and 2 pooled (114,584 rows);
- the three tables;
- `E17_CODE_MANIFEST.json`, which every notebook checks first.

No checkpoint is inside. The notebooks attach the outputs of earlier notebooks.

1. Kaggle → *Datasets* → *New Dataset* → upload `uie-e17-code.zip` as is → title
   **`uie-e17-code`** → **Private** → *Create*.

## Order

| # | notebook | inputs to attach | about | download |
|---|---|---|---|---|
| 1 | `E17_decode.ipynb` | `uie-e17-code`, the **E16_train** notebook's output | 2.5 h | *Output* (`e17_decode_gen/`) |
| 2 | `E17_train.ipynb` | `uie-corpus2`, `uie-e17-code`, the **E16_train** notebook's output | 3.5 h | *Output* (`e17_a/`, `PROVENANCE.json`) |
| 3 | `E17_generate_b5.ipynb` **or** `E17_generate_d1p.ipynb`: **I will say which** | `uie-e17-code`, the **E17_train** notebook's output | 2.5 h | *Output* (`e17_gen/`) |

- **1 and 2 can run one after the other right away.** While 2 trains, the laptop reads
  1: B5's verdict decides which version of 3 to run.
- **After 2, wait for the laptop's check** (about 1.5 h) before running 3.

For each notebook:
1. *Code* → *New Notebook* → *File* → *Import Notebook* → the `.ipynb`.
2. *Add Input* as in the table.
3. *Settings* → GPU **T4** (check it is really on), Internet **On**.
4. *Save Version* → *Save & Run All*.

## What to check

**E17_decode**

| cell | must show |
|---|---|
| 2 | `code: commit …; … the pooled stream (114,584 rows) verified`; `checkpoint verified` |
| 3 | `50 files, 180,000 URLs; decode {… 'slash_rule': True, 'slash_paths': True, 'query_rule': True}`; `loaded e16_cap: width 512` |
| 4 | `all 50 files written` |

**E17_train**

| cell | must show |
|---|---|
| 2 | `… E17 options present; e16_cap verified` |
| 3 | `width 384 rebuilds e12_final's recorded start; width 512 rebuilds e16_cap's (caca010736da...)`; `start check PASSED; memory check PASSED in 2 parts`. The recipe shows `'d_model': 512` and `'qry_extra': '…/qry_extra_s12.jsonl'` |
| 4 | `E15: query injection ON -> benign_P0Q1: 29,726 new / 14,863 old, p = 1.0000, rate 2.0000 per old draw; benign_P1Q1: 84,858 new / 42,429 old, …, rate 2.0000 per old draw`, then the training log |
| 5 | `MISMATCHES: none -- e17_a is as registered` |

**E17_generate (either)**

| cell | must show |
|---|---|
| 1 | `checkpoint verified` |
| 3 | `50 files, 180,000 URLs; decode {…}`, with `'query_rule': True` only in the `_b5` version; `loaded e17_a: width 512` |
| 4 | `all 50 files written` |

If a cell stops with "STOP", nothing was trained or written. Report it to me; do not
work around it.

## Back on the laptop

1. `E17_decode`'s output → `data/e17/decode/gen`: verify, score, B5's criteria (R-a′ …).
2. `E17_train`'s output → `data/e17/train`: VERIFY_TRAIN and K17-2, before step 3 runs.
3. `E17_generate`'s output → `data/e17/generate/gen`: verify, score, K17-1 … G10.
