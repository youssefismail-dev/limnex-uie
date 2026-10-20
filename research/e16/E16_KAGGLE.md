# E16 on Kaggle — exact steps

Companion to `PREREG_E16.md` (§11; amendments 1–2). Two notebooks. Verification, scoring
and every reading run on the laptop.

## Upload: ONE new dataset

`uie-e16-code.zip`, built by `research/e16/make_e16_bundle.py` from a clean, committed
tree (the commit and the zip's sha256 are in `PREREG_E16.md`'s Status). It holds:
- `uie/`, with E16's option (gradient accumulation), off by default;
- `research/native/` (E8, E11, E12's common modules), `research/e13/e13_common.py`,
  `research/e15/e15_common.py` and `research/e16/e16_common.py`;
- `e15_new.pt`, the baseline's checkpoint (its tokenizer rebuilds the start; its record
  is what the run is read back against);
- `qry_extra.jsonl`: E15's stream, 57,292 rows, unchanged;
- the suffix, state and presence-slash tables;
- `E16_CODE_MANIFEST.json`, which every notebook checks before doing anything.

Nothing inside is gzipped.

1. Kaggle → *Datasets* → *New Dataset* → upload `uie-e16-code.zip` as is (not unzipped)
   → title **`uie-e16-code`** → **Private** → *Create*.

## Order

**E16_train first.** Then the laptop verifies it (about 1.5 h; I will say when). **Only
then E16_generate.**

| notebook | inputs to attach | GPU | about | download |
|---|---|---|---|---|
| `E16_train.ipynb` | `uie-corpus2`, `uie-e16-code` | T4 | 3.5 h | the *Output* tab (`e16_cap/`, `PROVENANCE.json`) |
| `E16_generate.ipynb` | `uie-e16-code`, the E16_train notebook's output | T4 | 3–3.5 h | the *Output* tab (`e16_gen/`) |

For each notebook:
1. *Code* → *New Notebook* → *File* → *Import Notebook* → the `.ipynb`.
2. *Add Input* as in the table.
3. *Settings* → GPU **T4**, Internet **On**.
4. *Save Version* → *Save & Run All*.

It keeps running with the laptop off.

## What to check

**E16_train**

| cell | must show |
|---|---|
| 2 | `code: commit …; 40 files, 3 tables, e15_new.pt and the stream (57,292 rows) verified`; `all 8 streams present; E16 option present` |
| 3 | `width 384 rebuilds e12_final's recorded start exactly (c7e71edf983f...)`; `width 512 start: … (25,473,024 parameters), built twice, identical`; `start check PASSED; memory check PASSED in 2 parts (… of 14.6 GB)`. The recipe line shows `'d_model': 512`, `'lr': 0.0003` and `'accum_steps': 2`. (`in 4 parts` is the registered fallback and is fine) |
| 4 | `model: 25.47M parameters`; `E15: query injection ON -> benign_P0Q1: 14,863 new / 14,863 old, p = 1.0000; benign_P1Q1: 42,429 new / 42,429 old, p = 1.0000`; `E16: gradient accumulation ON -> each batch in 2 parts …`; then the training log |
| 5 | `MISMATCHES: none -- e16_cap is as registered`; every clipped share after step 1,000 at most 0.01 |

**If cell 3 stops** with "This torch does not rebuild e12_final's recorded start", Kaggle
has changed its torch. Nothing was trained. Report it to me; do not work around it.

**E16_generate**

| cell | must show |
|---|---|
| 1 | `checkpoint verified` |
| 3 | `50 files, 180,000 URLs; decode {… 'slash_rule': True, 'slash_paths': True}`; `loaded e16_cap: width 512` |
| 4 | `all 50 files written` |

`PROVENANCE.json` is rewritten after every file. If a session stops early, a rerun keeps
what finished. Training is never resumed half-way (§10): a stopped training session is
rerun from the start. Report it to me first, and do not delete anything.

## Back on the laptop

Put each downloaded zip in `Downloads` as Kaggle names it. The laptop then:
1. extracts E16_train's output to `data/e16/train`, verifies it (V1, V-start′, V-batch,
   V-mass′, V-stream, clipping) and reads K16-2, before E16_generate is run;
2. extracts E16_generate's output to `data/e16/generate/gen` and verifies it (V1, V2);
3. scores against `e15_new`, applies the criteria and the end rule (§§5–6).
