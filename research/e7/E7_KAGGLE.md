# E7 on Kaggle — exact steps

Companion to `PREREG_E7.md`. E7 trains a **new** checkpoint; `best_e5.pt` is never
touched.

## What you upload: ONE small dataset — the code

The data does not change. Presence is derived from each URL at encode time, so
the existing **`uie-corpus2`** dataset is used as it is. Only the code moves:

- **File:** `data/kaggle/uie-e7-code.zip` in the E7 working copy (a separate checkout of this
  repository)
  (built by `research/e7/make_code_bundle.py` from a clean, committed tree)
- It holds the `uie/` package plus `E7_CODE_MANIFEST.json`: the git commit and a
  SHA-256 for every file. The notebook checks every one before training.

## Steps

1. **Upload the code.** Kaggle -> *Datasets* -> *New Dataset* -> upload
   `uie-e7-code.zip` -> title **`uie-e7-code`** -> keep it **Private** -> *Create*.
2. **Open the notebook.** Kaggle -> *Code* -> *New Notebook* -> *File* ->
   *Import Notebook* -> `research/e7/E7_train.ipynb`.
3. **Attach both inputs.** Right panel -> *Add Input* -> your existing
   **`uie-corpus2`**; then *Add Input* -> **`uie-e7-code`**.
4. **Settings.** *Accelerator* -> **GPU T4 x2** (or P100). *Internet* -> **On —
   required**: `tldextract` is not in the Kaggle image, and cell 2 installs it
   (pinned, 5.3.1) before anything imports `uie`. Training itself makes no
   network calls.
5. **Run All.** About 1.1 h.

## What to check while it runs

| cell | must show |
|---|---|
| 2 | `tldextract: 5.3.1`, then `uie imported from:` a path containing `uie-e7-code`, then `code integrity: 30/30 files match` |
| 4 | `E7: presence controls ON` and `vocab: 107` in the log |
| 4 | loss falling from ~4.6 (ln 107) to under 3.0 within the first 1,000 steps |
| 5 | every line `OK`, and `MISMATCHES: none` |

If cell 2 fails with **SHADOWED**, the notebook is importing the data bundle's
older `uie`. Stop: that run would train a plain E5 model while looking healthy.

Kaggle now mounts inputs as `/kaggle/input/datasets/<user>/<slug>/`; the
notebook discovers both by their manifests, so the slug does not matter. Cell 3
stops the run if the data dataset is not the two-corpus bundle.

## C2 generations on the GPU — `E7_C2_generate.ipynb`

Generation only; nothing is scored on Kaggle. No new upload.

1. **Import** `research/e7/E7_C2_generate.ipynb` as a new notebook.
2. **Attach three inputs:** `uie-corpus`, `uie-e7-code`, and *Your Work →
   Notebooks →* the E7 training notebook (its output holds `best.pt`).
3. **Settings:** GPU T4 x2, Internet **On**. **Save & Run All** — about an hour.
4. **Check:** cell 1 `sha256 verified: best_e7`; cell 2 `tldextract: 5.3.1` and
   `30/30`; cell 3 `suffix table: sha256 verified` and **`model parameters on
   cuda:0`**; cell 4 ten progress lines; cell 5 `DONE`.
5. **Download** the Output (`gen_cache_e7_gpu/`, ~5 MB) into `Downloads`.

Locally, `c2_gpu_verify.py` runs before the gate reads a single file.

## Attempt log

- **Attempt 1, 2026-09-23 — failed in cell 2, before any training.**
  `ModuleNotFoundError: No module named 'tldextract'`: the notebook imported
  `uie.tokenizer` (which needs `tldextract`) before installing it. The local
  checks missed it because the laptop has `tldextract`. **Fix:** install first,
  pinned. **Verified** in a simulated Kaggle layout with `tldextract` absent: the
  attempt-1 notebook reproduces the identical traceback; the fixed one runs every
  cell end to end on a miniature model. Neither dataset changed — re-import the
  notebook only.
- **C2 generation, attempt 1, 2026-09-23 — failed at the first forward pass.**
  `Expected all tensors to be on the same device`: `load_checkpoint` builds the
  model on the CPU, and the gate moves it on the next line
  (`model.to(device).eval()`, run_eval.py:277), which the notebook had not
  copied. Invisible on the laptop, where everything is on the CPU, and a CPU-only
  simulation cannot exercise it. **Fix:** the gate's own line, plus an assertion
  on the parameters' actual device before any generation.

## Afterwards

Download the *Output* tab (the zip is fine) into `Downloads` and tell me. I will
extract it as **`best_e7.pt` / `final_e7.pt`** — never as `best.pt`, and never over
`best_e5.pt` — and run the registered evaluation: C1 honouring, C2 gate v2 on the
unconditional mode, then C3 consistency.
