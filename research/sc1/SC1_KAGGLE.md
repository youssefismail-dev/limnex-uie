# SC1 on Kaggle — exact steps

Companion to `PREREG_SC1.md`. Generation only; verification and scoring run on
the laptop.

## Upload: ONE new dataset — the code and the tables

`data/kaggle/uie-sc1-code.zip`, built by `research/sc1/make_code_bundle.py` from a
clean, committed tree, holds three things:
- the `uie/` package;
- `sc1_common.py`, the registered arms, states and seeds;
- `tables.json`, the suffix tables, sha256 `62aa7d9b…d453`.

It also holds `SC1_CODE_MANIFEST.json` — the commit and a SHA-256 per file — which
the notebook checks before it generates anything.

The data bundle is **not** needed: the tables travel in the code dataset. The
checkpoint is not re-uploaded.

1. Kaggle → *Datasets* → *New Dataset* → upload `uie-sc1-code.zip` → title
   **`uie-sc1-code`** → **Private** → *Create*.
2. Kaggle → *Code* → *New Notebook* → *File* → *Import Notebook* →
   `research/sc1/SC1_generate.ipynb`.
3. *Add Input* → `uie-sc1-code`; *Add Input* → *Your Work* → *Notebooks* → the E7
   training notebook (`e-7-experience`). (Attaching `uie-corpus` as well is
   harmless: its older `uie/` is detected and never imported.)
4. *Settings* → GPU T4 x2, Internet **On**. *Save Version* → *Save & Run All*.

## What to check

| cell | must show |
|---|---|
| 1 | `sha256 verified: best_e7` |
| 2 | `tldextract: 5.3.1`, `code integrity: 33/33 files match`, `tables sha256 62aa7d9bafd5bb79 verified` |
| 3 | `model parameters on cuda:0; arms ['C', 'T']; mode held; 140 files to make` |
| 4 | 140 progress lines (`C/…` first, then `T/…`), no `ERROR` |
| 5 | `all complete: True` and `DONE` |

About 4 h. Download the *Output* tab (`sc1_gen/`, holding `C/gen/` and `T/gen/`)
into `Downloads`.

## If the session stops early

Nothing already on disk is lost within a session, because each file is written
when it completes. A **new** session starts empty, though. Set
`RUN_ARMS = ["T"]` (or `["C"]`) in cell 3 to regenerate only the arm that is
missing. Its seeds are fixed per file, so the result is identical.

## Replication (section 7.4) — only if a K2 flag fires

Set `REPLICATE = ["<flagged tag>", …]` in cell 3. That arm is `T_rep`, with the
replication seeds. Download `sc1_gen/T_rep/`. Nothing else changes.
