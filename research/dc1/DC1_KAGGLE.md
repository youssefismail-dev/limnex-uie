# DC1 on Kaggle — exact steps

Companion to `PREREG_DC1.md`. Generation only; verification, scoring and the
comparison run on the laptop.

## Upload: ONE new dataset

`data/kaggle/uie-dc1-code.zip`, built by `research/dc1/make_code_bundle.py` from a
clean, committed tree, holds:
- `uie/`, with DC1's sampler options (off by default);
- `dc1_common.py`, the registered sizes and seeds;
- `settings.json`, the fit's output (only settings that passed validation are applied);
- the adopted state tables and the shipped suffix table;
- `DC1_CODE_MANIFEST.json`, checked by the notebook before anything is generated.

1. Kaggle → *Datasets* → *New Dataset* → upload `uie-dc1-code.zip` (as is, not
   unzipped) → title **`uie-dc1-code`** → **Private** → *Create*.
2. Kaggle → *Code* → *New Notebook* → *File* → *Import Notebook* →
   `DC1_generate.ipynb`.
3. *Add Input* → `uie-dc1-code`; *Add Input* → *Your Work* → *Notebooks* → the E7
   training notebook (`e-7-experience`).
4. *Settings* → GPU T4 x2, Internet **On**. *Save Version* → *Save & Run All*.

## What to check

| cell | must show |
|---|---|
| 1 | `sha256 verified: best_e7` |
| 2 | `tldextract: 5.3.1`, `code integrity: 36/36 files match`, `settings sha256 … verified` |
| 3 | `FULL FREE ON, FULL CONTROL ON; 56 files to make` |
| 4 | one progress line per file; no `ERROR` |
| 5 | `all complete: True` and `DONE` |

About 5–6 h. Download the *Output* tab (`dc1_gen/`) into `Downloads`.

`PROVENANCE.json` is rewritten after every file, so if the session stops early
the finished files keep their records; do not delete or rerun anything — report it
first.
