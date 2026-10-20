# CR1 on Kaggle — exact steps

Companion to `PREREG_CR1.md`. Generation only; scoring runs on the laptop.

## Upload: ONE new dataset — the code

`data/kaggle/uie-cr1-code.zip` (built by `research/cr1/make_code_bundle.py` from
a clean, committed tree): the current `uie/` package, including
`uie/presence.py`, and `CR1_CODE_MANIFEST.json` — the commit and a SHA-256 per
file, checked by the notebook before it generates anything. The data bundle and
the checkpoint are **not** re-uploaded.

1. Kaggle → *Datasets* → *New Dataset* → upload `uie-cr1-code.zip` → title
   **`uie-cr1-code`** → **Private** → *Create*.
2. Kaggle → *Code* → *New Notebook* → *File* → *Import Notebook* →
   `research/cr1/CR1_generate.ipynb`.
3. *Add Input* → `uie-corpus`; *Add Input* → `uie-cr1-code`; *Add Input* →
   *Your Work* → *Notebooks* → the E7 training notebook (`e-7-experience`).
4. *Settings* → GPU T4 x2, Internet **On**. *Save Version* → *Save & Run All*.

## What to check

| cell | must show |
|---|---|
| 1 | `sha256 verified: best_e7` |
| 2 | `tldextract: 5.3.1`, `code integrity: 31/31`, `uie.presence present` |
| 3 | `model parameters on cuda:0; mode …; 70 files to make` |
| 4 | 70 progress lines, no `ERROR` |
| 5 | `all complete: True` and `DONE` |

About 1.5 h. Download the *Output* tab (`cr1_gen/`, ~13 MB) into `Downloads`.
