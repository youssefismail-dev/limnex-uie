# E13 on Kaggle — exact steps

Companion to `PREREG_E13.md`. Three notebooks. Verification, scoring and every
reading run on the laptop.

## Upload: ONE new dataset

`data/kaggle/uie-e13-code.zip`, built by `research/e13/make_e13_bundle.py` from a
clean, committed tree (commit 887b928; sha256 `15c13092…`, 53.0 MB). It holds:
- `uie/`, with E13's options (the slash rule, ContextSwap; both off by default);
- `research/native/` (E8, E11, E12's common modules) and `research/e13/`
  (`e13_common.py`, `e13n_train.py`);
- `e12_final.pt`, E12's baseline weights, checked against E12's records;
- `prefixes.json` (the context arm's model-written prefixes, uncompressed:
  Kaggle decompresses `.gz` files inside an uploaded dataset) and
  `d2_settings.json` (D2's fitted settings), each frozen on the laptop;
- the suffix, state and presence-slash tables;
- `E13_CODE_MANIFEST.json`, which every notebook checks before doing anything.

1. Kaggle → *Datasets* → *New Dataset* → upload `uie-e13-code.zip` as is (not
   unzipped) → title **`uie-e13-code`** → **Private** → *Create*.

## Order

**E13_decode** and **E13_train** do not depend on each other: run them in
either order, or both at once. **E13_generate** runs only after the laptop has
verified E13_train's output.

| notebook | inputs to attach | about | download |
|---|---|---|---|
| `E13_decode.ipynb` | `uie-e13-code` | 3.5 h | the *Output* tab (`e13_decode/`) |
| `E13_train.ipynb` | `uie-corpus2`, `uie-e13-code` | 2.5 h | the *Output* tab (`e13_ctx/`, `n_A1/`, `n_A2/`, `PROVENANCE.json`) |
| `E13_generate.ipynb` | `uie-e13-code`, the E13_train notebook's output | 1.6 h | the *Output* tab (`e13_gen/`) |

For each: *Code* → *New Notebook* → *File* → *Import Notebook* → the `.ipynb` →
*Add Input* as in the table → *Settings* → GPU **T4**, Internet **On** → *Save
Version* → *Save & Run All*.

## What to check

**E13_decode**

| cell | must show |
|---|---|
| 2 | `code: commit …; … files, 5 tables and e12_final.pt verified`, then `torch: … \| gpu: Tesla T4` |
| 3 | `D2 scopes applied: ['D2-P', 'D2-Q']`; `115 files` |
| 4 | one progress line per file; no error; `all 115 files written` |

**E13_train**

| cell | must show |
|---|---|
| 2 | `… files, 5 tables and e12_final.pt verified`; `all 8 streams present; E13 options present` |
| 3 | `start check PASSED; memory preflight PASSED` |
| 4 | `E13 C: ContextSwap ON -> 8,130 of 8,130 listed rows matched; p = 0.5` |
| 5 | `MISMATCHES: none -- e13_ctx is as registered` |
| 6 | `E13 N A1 ... steps 1659` and `E13 N A2 ... steps 6636`; `all arms trained` |

**E13_generate**

| cell | must show |
|---|---|
| 1 | `checkpoint verified` |
| 4 | `all 50 files written` |

`PROVENANCE.json` is rewritten after every file and every arm. If a session stops
early, a rerun keeps what finished; report it first, and do not delete anything.

## Back on the laptop

Put each downloaded zip in `Downloads`. The laptop then:
1. extracts E13_train's output to `data/e13/train` and runs
   `e13_verify.py --step train`, before E13_generate is run;
2. extracts E13_decode's output to `data/e13/decode/gen`, and E13_generate's to
   `data/e13/ctx/gen`, and runs `e13_verify.py --step decode` and
   `--step generate`;
3. runs the readings: `e13_score.py`, `e13_dx8.py`, `e13n_read.py`, then
   `e13_compare.py`.
