# E15-B on Kaggle — exact steps

Companion to `PREREG_E15.md` (§9.12 steps 4–6; §§12–13). Two notebooks. Verification,
scoring and every reading run on the laptop.

## Upload: ONE new dataset

`uie-e15-code.zip`, built by `research/e15/make_e15_bundle.py` from a clean, committed
tree (the commit and the zip's sha256 are in `PREREG_E15.md`'s Status). It holds:
- `uie/`, with E15's option (the query injector), off by default;
- `research/native/` (E8, E11, E12's common modules), `research/e13/e13_common.py` and
  `research/e15/e15_common.py`;
- `e12_final.pt`, E12's baseline weights, checked against E12's records;
- `qry_extra.jsonl`: **the new rows**, 57,292 benign URLs with queries (Amendment 3's
  files, checked by hash);
- the suffix, state and presence-slash tables;
- `E15_CODE_MANIFEST.json`, which every notebook checks before doing anything.

Nothing inside is gzipped.

1. Kaggle → *Datasets* → *New Dataset* → upload `uie-e15-code.zip` as is (not
   unzipped) → title **`uie-e15-code`** → **Private** → *Create*.

## Order

**E15B_train first.** Then the laptop verifies it, which takes about 1.5 h; I will say
when. **Only then E15B_generate.**

| notebook | inputs to attach | GPU | about | download |
|---|---|---|---|---|
| `E15B_train.ipynb` | `uie-corpus2`, `uie-e15-code` | T4 | 1.5 h | the *Output* tab (`e15_new/`, `PROVENANCE.json`) |
| `E15B_generate.ipynb` | `uie-e15-code`, the E15B_train notebook's output | T4 | 2 h | the *Output* tab (`e15_gen/`) |

For each notebook:
1. *Code* → *New Notebook* → *File* → *Import Notebook* → the `.ipynb`.
2. *Add Input* as in the table.
3. *Settings* → GPU **T4**, Internet **On**.
4. *Save Version* → *Save & Run All*.

It keeps running with the laptop off.

## What to check

**E15B_train**

| cell | must show |
|---|---|
| 2 | `code: commit …; … files, 3 tables, e12_final.pt and the stream (57,292 rows) verified`; `all 8 streams present; E15 option present`; the recipe ends `'qry_extra': '…/qry_extra.jsonl'` |
| 3 | `start check PASSED; memory preflight PASSED` |
| 4 | `E15: query injection ON -> benign_P0Q1: 14,863 new / 14,863 old, p = 1.0000; benign_P1Q1: 42,429 new / 42,429 old, p = 1.0000`, then the training log |
| 5 | `MISMATCHES: none -- e15_new is as registered`, and `injected: … rows (about 4.74 per batch …)` |

**E15B_generate**

| cell | must show |
|---|---|
| 1 | `checkpoint verified` |
| 3 | `50 files, 180,000 URLs; decode {… 'slash_rule': True, 'slash_paths': True}` |
| 4 | `all 50 files written` |

`PROVENANCE.json` is rewritten after every file. If a session stops early, a rerun
keeps what finished. Report it to me first, and do not delete anything.

## Back on the laptop

Put each downloaded zip in `Downloads` as Kaggle names it. The laptop then:
1. extracts E15B_train's output to `data/e15/train` and verifies it (V1, V-start,
   V-batch, V-mask, V-mass, V-stream) and K15-2, before E15B_generate is run;
2. extracts E15B_generate's output to `data/e15/generate/gen` and verifies it;
3. scores against `e12_d1p`, applies the criteria (§9.8, with §13.2's definitions),
   writes the four-way report and reads E3.
