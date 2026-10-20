# C3-QRY on Kaggle — exact steps

Companion to `PREREG_C3_QRY.md`. Arm F's generation only; arm Q is SC1's existing
files. Verification, scoring and every reading run on the laptop.

## Upload: ONE new dataset — the code and the tables

`data/kaggle/uie-c3-code.zip`, built by `research/c3/make_code_bundle.py` from a
clean, committed tree, holds:
- the `uie/` package, including the adopted `uie/controlled.py`;
- `c3_common.py`, the registered sizes and seeds;
- `free_matched.py`, arm F's generator;
- `state_suffix_tables.json`, the adopted tables, sha256 `0c6f2760…13cc9`.

It also holds `C3_CODE_MANIFEST.json`, which the notebook checks before it
generates anything. No data bundle is needed, and the checkpoint is not
re-uploaded.

1. Kaggle → *Datasets* → *New Dataset* → upload `uie-c3-code.zip` (as is, not
   unzipped) → title **`uie-c3-code`** → **Private** → *Create*.
2. Kaggle → *Code* → *New Notebook* → *File* → *Import Notebook* →
   `research/c3/C3_generate.ipynb`.
3. *Add Input* → `uie-c3-code`; *Add Input* → *Your Work* → *Notebooks* → the E7
   training notebook (`e-7-experience`).
4. *Settings* → GPU T4 x2, Internet **On**. *Save Version* → *Save & Run All*.

## What to check

| cell | must show |
|---|---|
| 1 | `sha256 verified: best_e7` |
| 2 | `tldextract: 5.3.1`, `code integrity: 35/35 files match`, `tables sha256 0c6f2760c033c532 verified` |
| 3 | `model parameters on cuda:0; arm F (matched free); 10 files to make (benign 5 x 1200, phishing 5 x 400 query-bearing URLs)` |
| 4 | 10 progress lines, benign first, each with its query rate; no `ERROR`. A line saying `kept from an earlier run` appears only if cell 4 was run twice in one session — that is correct: the file is kept only with its own record |
| 5 | `all complete: True` and `DONE` |

About 4–5 h: a simulation of this exact run (its 10 seeds, its suffix groups, E7's own
per-suffix query rates from its unconditional C2 generations, SC1's measured 34 draws a
second) gives 4.1 h median, 4.8 h at the 99th percentile and 5.6 h at worst in 400
replicates, inside Kaggle's 12 h. Download the *Output* tab (`c3_gen/F/gen/`) into
`Downloads`.

`PROVENANCE.json` is rewritten after every file, so if the session stops early the
finished files keep their records; do not delete or rerun anything — report it first.

An `ERROR FreeUnmet` line is **K1**: a registered outcome for that class, not a
fault. Do not rerun with other settings; report it as it stands.
