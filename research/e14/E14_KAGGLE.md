# E14 on Kaggle — exact steps

Companion to `PREREG_E14.md` (§6, steps 3–8). Three notebooks. Verification, scoring
and every reading run on the laptop.

## Upload: ONE new dataset

`uie-e14-code.zip`, built by `research/e14/make_e14_bundle.py` from a clean,
committed tree (commit 2f4f01c; sha256 `dac83426…`, 51.2 MB). It holds:
- `uie/`, with E14's two options (the query weight, D1′), both off by default;
- `research/native/` (E8, E11, E12's common modules), `research/e13/e13_common.py`
  and `research/e14/e14_common.py`;
- `e12_final.pt`, E12's baseline weights, checked against E12's records;
- the suffix, state and presence-slash tables;
- `E14_CODE_MANIFEST.json`, which every notebook checks before doing anything.

Nothing inside is gzipped.

1. Kaggle → *Datasets* → *New Dataset* → upload `uie-e14-code.zip` as is (not
   unzipped) → title **`uie-e14-code`** → **Private** → *Create*.

## Order

**E14_decode** and **E14_train** do not depend on each other. Run them in either
order, or both at once. **E14_generate** runs only after the laptop has verified
E14_train's output; I will say when.

| notebook | inputs to attach | GPU | about | download |
|---|---|---|---|---|
| `E14_decode.ipynb` | `uie-e14-code` | **T4** (required) | 1.5 h | the *Output* tab (`e14_decode/`) |
| `E14_train.ipynb` | `uie-corpus2`, `uie-e14-code` | T4 | 1.5 h | the *Output* tab (`e14_q/`, `PROVENANCE.json`) |
| `E14_generate.ipynb` | `uie-e14-code`, the E14_train notebook's output | T4 | 3.2 h | the *Output* tab (`e14_gen/`) |

For each notebook:
1. *Code* → *New Notebook* → *File* → *Import Notebook* → the `.ipynb`.
2. *Add Input* as in the table.
3. *Settings* → GPU **T4**, Internet **On**.
4. *Save Version* → *Save & Run All*.

It keeps running with the laptop off.

**E14_decode must run on a T4.** It compares its URLs with E13's T4 run (V7), and
stops in cell 3 on any other GPU. "GPU T4 x2" is fine.

## What to check

**E14_decode**

| cell | must show |
|---|---|
| 2 | `code: commit 2f4f01c85c34; 39 files, 3 tables and e12_final.pt verified`, then `torch: … \| gpu: Tesla T4` |
| 3 | `50 files, 180,000 URLs` |
| 4 | one progress line per file; no error; `all 50 files written` |

**E14_train**

| cell | must show |
|---|---|
| 2 | `… 39 files, 3 tables and e12_final.pt verified`; `all 8 streams present; E14 option present`; the recipe ends `'query_weight': 4.0` |
| 3 | `start check PASSED; memory preflight PASSED` |
| 4 | `E14: query weight 4.0 on every step inside the query …`, then the training log |
| 5 | `MISMATCHES: none -- e14_q is as registered` |

**E14_generate**

| cell | must show |
|---|---|
| 1 | `checkpoint verified` |
| 3 | `100 files, 360,000 URLs` |
| 4 | `all 100 files written` |

`PROVENANCE.json` is rewritten after every file. If a session stops early, a
rerun keeps what finished. Report it to me first, and do not delete anything.

## Back on the laptop

Put each downloaded zip in `Downloads` as Kaggle names it. The laptop then:
1. extracts E14_train's output to `data/e14/train` and runs
   `e14_verify.py --step train` and `e14_fit.py --part m3` (K14-2), before
   E14_generate is run;
2. extracts E14_decode's output to `data/e14/decode/gen` and runs
   `e14_verify.py --step decode` (V1, V2, V7);
3. extracts E14_generate's output to `data/e14/generate/gen` and runs
   `e14_verify.py --step generate`;
4. runs the readings: `e14_score_run.py --arms e12_d1p e14_q e14_qd --v4 --n2a`,
   `e14_score.py --merge`, then `e14_compare.py` and `e14_tables.py`.
