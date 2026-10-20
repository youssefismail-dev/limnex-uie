# V2-P2 on Kaggle — exact steps

This is the companion to `PREREG_V2_P2.md` (§5, §6, §15). There are four notebooks: one
training and one generation notebook per arm. Verification, scoring and every reading run on
the laptop.

## Upload: one new dataset

`uie-v2p2-code.zip` is built by `research/v2/make_v2p2_bundle.py` from a clean, committed
tree. The commit and the zip's sha256 are in `PREREG_V2_P2.md` (Amendment 3, §20). It holds:
- `uie/`, unchanged since `422cb23` (V-iso);
- the common modules E8 to E17;
- V2's P2 modules: `v2_bpe.py`, `v2_mem.py`, `v2_p1_common.py`, `v2_p2_common.py`,
  `v2_p2_tokenizer.py` and `v2_p2_train.py`;
- the two piece tables: `p2_rest.json` (32,000 pieces) and `p2_host.json` (16,000 pieces);
- `qry_extra.jsonl`: stream 1, the 57,292 rows `e16_cap` trained with;
- `e16_cap_record.json`: `e16_cap`'s tokenizer and settings;
- the three generation tables;
- `V2P2_CODE_MANIFEST.json`, which every notebook checks first.

Upload it like this:
1. Kaggle → *Datasets* → *New Dataset*.
2. Upload `uie-v2p2-code.zip` as is.
3. Title it **`uie-v2p2-code`**, set it to **Private**, then *Create*.

## Order

| # | notebook | inputs to attach | about | download |
|---|---|---|---|---|
| 1 | `V2P2_train_a.ipynb` | `uie-corpus2`, `uie-v2p2-code` | 3.5–4.5 h | `p2_a/final.pt` (about 165 MB), `p2_a/train_result.json`, `PROVENANCE.json`. **Not** `best.pt` |
| 2 | `V2P2_train_b.ipynb` | `uie-corpus2`, `uie-v2p2-code` | 3.5–4.5 h | `p2_b/final.pt` (about 200 MB), `p2_b/train_result.json`, `PROVENANCE.json`. **Not** `best.pt` |
| 3 | `V2P2_generate_a.ipynb` — **only after I say arm A's laptop check passed** | `uie-v2p2-code`, the **V2P2_train_a** notebook's output | 1.5–2.5 h | `v2p2_gen_a/` (about 25 MB) |
| 4 | `V2P2_generate_b.ipynb` — **only after I say arm B's laptop check passed** | `uie-v2p2-code`, the **V2P2_train_b** notebook's output | 1.5–2.5 h | `v2p2_gen_b/` (about 25 MB) |

- **1 and 2 may run at the same time** (§5), each in its own session.
- **After each training, wait for the laptop's check** (VERIFY_TRAIN, about 30–45 min) before
  that arm's generation. A failed check stops that arm: its generation is not run.

For each notebook:
1. *Code* → *New Notebook* → *File* → *Import Notebook* → the `.ipynb`.
2. *Add Input*, as in the table.
3. *Settings* → GPU **T4** (check it is really on), Internet **On**.
4. *Save Version* → *Save & Run All*.

## What to check

**V2P2_train_a / V2P2_train_b**

| cell | must show |
|---|---|
| 2 | `code: commit …; … files, 3 tables, the piece tables […], stream 1 (57,292 rows) and e16_cap's record verified`; `torch: … \| gpu: Tesla T4`; `data bundle: all 8 streams present; e16_cap's record loaded` |
| 3 | in this order: `width 384 rebuilds e12_final's recorded start; width 512 rebuilds e16_cap's (caca010736da...)`; `p2_a's start: e16_cap's part caca010736da..., the piece rows peeled off give caca010736da...; pieces {'rest': 32000, 'host': 0}; …; 41,857,024 parameters; vocab 32,110` (arm B: `pieces {'rest': 32000, 'host': 16000}`, `50,049,024 parameters; vocab 48,110`); `start check PASSED; memory check PASSED in 2 parts (… of 14.6 GB)` (4 parts is the registered fallback) |
| 4 | `vocab: 32,110` (B: `48,110`), `model: 41.86M parameters` (B: `50.05M`), `E15: query injection ON -> …`, then the training log: a loss line every 200 steps and an `eval` line every 1,000 |
| 5 | `MISMATCHES: none -- p2_a is as registered` (B: `p2_b`) |

**V2P2_generate_a / V2P2_generate_b**

| cell | must show |
|---|---|
| 1 | `checkpoint verified` |
| 3 | `50 files, 180,000 URLs; decode {… 'slash_rule': True, 'slash_paths': True, 'query_rule': True} + the piece rule`; `loaded p2_a: width 512, vocab 32,110 …; the piece rule: {'PTH': 'rest', 'QRY': 'rest', 'FRG': 'rest'}` (B: also `'LBL': 'host', 'SUB': 'host'`) |
| 4 | each file's line ends `violations: pieces 0, D1 0, B5 0`; at the end `all 50 files written; violations (pieces, D1, B5): none` |

**If a cell stops with "STOP", nothing was trained or written.** Report it to me, and do not
work around it. The registered stops before training are:
- the start check, which fails if Kaggle's torch has changed;
- the memory check, which fails if even 4 parts do not fit.

Once training starts, it runs its full 20,000 steps. If a session ends early, the arm is
rerun from the start as registered (no resume).

## Back on the laptop

1. Put `V2P2_train_a`'s output in `data/v2/p2_a/train` (keeping the `p2_a` folder), and
   `V2P2_train_b`'s in `data/v2/p2_b/train`. Then I run, at low priority with the low-RAM alarm:
   - `V2_ARM=p2_a python research/v2/v2_p2_verify.py --step train` (and `p2_b`).
2. Put `V2P2_generate_a`'s output in `data/v2/p2_a/generate/gen` and `V2P2_generate_b`'s in
   `data/v2/p2_b/generate/gen`. Then, per arm, VERIFY_GENERATE
   (`V2_ARM=p2_a python research/v2/v2_p2_verify.py --step generate`), V4 and the gate
   (`V2_ARM=p2_a python research/v2/v2_p1_score.py --v4`, `--part free`, `control`, `misc`,
   `copy`, `--merge`); then the words and host guards (`v2_p2_words.py --part samples`, `run`,
   `metrics`) and the reading (`v2_p2_compare.py`), read once.
