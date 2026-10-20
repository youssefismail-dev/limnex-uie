# R2 on Kaggle — exact deployment steps

Companion to `PREREG_R2.md`. **R2 is a DIAGNOSTIC** (does more optimisation move
the Q8 learnability threshold?), not a repair. It produces a new checkpoint under
a new name; `best_e5.pt` is never overwritten.

---

## 1. What you need to upload: **NOTHING**

Verified, not assumed. The `uie` package **ships inside the dataset** —
`kaggle_train.py` says so: *"It travels inside the upload bundle, so the notebook
needs no network access and no pip install from an unpublished repository."*

Comparing every `.py` in the uploaded dataset against the local repo by SHA-256:

    shipped .py files          30
    IDENTICAL                  29
    STALE                       1   uie/corpus/run_eval.py
    MISSING locally             0
    in repo but not shipped     0

The one stale file is the evaluation gate, edited today for R4. **It is not on
the training import path.** `kaggle_train`'s full import closure is:

    uie/kaggle_train.py -> evaluation, factorization, model, ngram,
                           tokenizer, train

`run_eval` does not appear in it; the only two mentions elsewhere in `uie/` are
in comments. So the uploaded dataset is **correct as-is for R2**.

> This check exists because a stale shipped package has bitten this project
> before — an older bundle carried an out-of-date `uie` and would have raised a
> TypeError mid-session. Re-verify with the hash comparison before any future
> Kaggle run; do not assume.

### The dataset to select

The one built from `data/bundle2/` (uploaded as `uie-corpus2.zip`, 172 MB). It
must contain all **eight** streams, which `kaggle_train` loads by name:

    name_train  structure_train  phishing_name_train  phishing_structure_train
    name_val    structure_val    phishing_name_val    phishing_structure_val

Plus `uie/`, `suffix_table.json`, `known_urls.bloom.gz`, `bundle.manifest.json`.

If your Kaggle dataset only has `train.jsonl` / `val.jsonl`, that is the **old
E2 bundle** and is the wrong one — it predates the two-corpus architecture.

---

## 2. STALE DOCUMENTATION WARNING

**`KAGGLE.md` is written for E2 and its command is WRONG for E5.** It says:

```bash
!python -m uie.kaggle_train --data ... --steps 20000 --batch-size 256
```

Run verbatim, that trains a **4.8M-parameter model with no frame token**,
because `kaggle_train`'s defaults are `--d-model 256 --n-layers 6` and `--frame`
off. E5 is 14.33M at `d_model 384 / 8 layers / frame on`.

**A run without `--frame` is not comparable to `best_e5.pt` at all** — the frame
token changes the vocabulary *and* the loss (it disables field-scoped content
masking), which is exactly why Q8 H6 found `trained/own = 1.0`.

Use the command in section 4, not the one in `KAGGLE.md`.

---

## 3. Notebook setup

1. Kaggle → **Code → New Notebook**
2. **Settings → Accelerator → GPU T4 x2** (or P100; the model uses one GPU)
3. **Settings → Internet → On** — needed only for `tldextract` in the first
   cell. Training itself makes no network calls.
4. **Add Input** → your `uie-corpus2` dataset.
5. Confirm the mount path in the file browser. It will be
   `/kaggle/input/<your-dataset-name>` — substitute it everywhere below.

**Cell 1**

```python
!pip install -q tldextract
```

**Cell 2 — verify the environment and the dataset BEFORE burning GPU time**

```python
import sys, pathlib
DATA = "/kaggle/input/uie-corpus2"          # <-- your dataset name
sys.path.insert(0, DATA)

import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))

need = ["name_train", "structure_train", "phishing_name_train",
        "phishing_structure_train", "name_val", "structure_val",
        "phishing_name_val", "phishing_structure_val"]
present = {p.name for p in pathlib.Path(DATA).iterdir()}
for s in need:
    ok = f"{s}.jsonl" in present or f"{s}.jsonl.gz" in present
    print(("OK   " if ok else "MISS "), s)
print("uie package:", (pathlib.Path(DATA) / "uie" / "kaggle_train.py").is_file())
```

Every line must read `OK` and the package must be `True`. Kaggle **decompresses
gzip members** when processing an upload, so streams may arrive as `.jsonl`
rather than `.jsonl.gz` — `resolve()` accepts either, which is why the check
tests both.

---

## 4. The training command

```bash
!python -m uie.kaggle_train \
    --data /kaggle/input/uie-corpus2 \
    --out  /kaggle/working \
    --steps 40000 \
    --batch-size 256 \
    --lr 3e-4 \
    --d-model 384 \
    --n-layers 8 \
    --n-heads 8 \
    --class-alpha 1.0 \
    --boundary-weight 1.0 \
    --eval-every 1000 \
    --frame
```

Every flag and why it is stated explicitly:

| flag | value | why |
|---|---|---|
| `--steps` | **40000** | **THE ONLY VARIABLE UNDER TEST.** 20,000 -> 40,000 doubles exposures, 5.12M -> 10.24M (~2 -> ~4 epochs) |
| `--d-model` | 384 | default is 256 — would silently train a different model |
| `--n-layers` | 8 | default is 6 — same trap |
| `--frame` | on | default is OFF; without it the vocabulary and loss differ from E5 |
| `--boundary-weight` | 1.0 | S4 stays OFF; stated so the artifact records it |
| `--class-alpha` | 1.0 | exposure-neutral; the memorisation guard stays as-is |
| `--batch-size`, `--lr` | 256, 3e-4 | unchanged from the baseline |

`max_len` is **not** a CLI flag and stays at the `ModelConfig` default of 256 —
which is what we want, since R1 was demoted on measured cost/benefit and
bundling it here would destroy attribution.

---

## 5. What to expect

`KAGGLE.md` quotes ~40 min for the 4.8M model at 20,000 steps. E5 is ~3x the
compute per step (`(384/256)^2 x (8/6) ≈ 3`), so:

    20,000 steps  ~2 h        40,000 steps  ~4 h

Comfortably inside a 12 h session and inside the 30 GPU-h weekly quota. **No
checkpoint-resume should be needed** — but confirm your own baseline wall-clock
before launching, since this estimate is derived, not measured.

Sanity checks while it runs:

- Initial loss near `ln(vocab) ≈ 4.6`. If it has not fallen below 3 within the
  first thousand steps, **stop rather than burn quota**.
- Validation loss is reported **per class**. Both should improve. If phishing
  stalls while benign improves, class sampling is not working.

---

## 6. Outputs, and the one thing that must not go wrong

The notebook's *Output* tab will contain `final.pt`, `best.pt` and
`train_result.json`.

> **RENAME ON DOWNLOAD.** The script writes `best.pt` / `final.pt`, the same
> names the E2-era run used. Download them as **`best_r2_40k.pt`** and
> **`final_r2_40k.pt`**. **Do not place a file named `best.pt` over anything in
> `data/`, and never overwrite `best_e5.pt`** — it is the frozen baseline every
> result in `E3_REPORT.md` section 30 is measured against.

Also keep `train_result.json`: it records the config actually used, which is how
we verify after the fact that `--frame` and `--d-model 384` really took effect.

---

## 7. What happens when it lands (locally, nothing adopted)

In the order registered in `PREREG_R2.md`:

1. **`q8_exposure.py`** — band profile vs the faithful-sampler null. **Primary.**
   Does band 5–19 rise from 4.09% toward its null of 20.40%?
2. **`q7_hazard.py`** on `name_val` **and** `name_train` — hazard at 50–100
   (currently 1.315 / 1.346), and the **train/val gap that is the memorisation
   kill criterion**.
3. **`run_eval.py`** at `n_main 8000, G=5 x D=3` — safety, read against the
   ±0.0141 retraining-noise confound. **benign SUB clears by only 0.011 and is
   therefore NOT assessable from a single run.**

Adoption is a separate decision that neither this document nor `PREREG_R2.md`
pre-authorises.
