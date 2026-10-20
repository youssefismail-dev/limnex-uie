# Training E2 on Kaggle

The model is small (4.8M parameters) but ~25 hours on a laptop CPU and **under an
hour on a Kaggle T4**. This is the whole procedure.

---

## What to upload

**One thing: the folder `data/kaggle_bundle/`.**

Build it first:

```bash
python -m uie.corpus.pack_kaggle
```

That produces:

```
data/kaggle_bundle/
    train.jsonl.gz          55 MB    1,503,176 URLs  (1.2M benign + 303k phishing)
    val.jsonl.gz           2.6 MB       80,339 URLs  (60k benign + 20k phishing)
    uie/                             the package itself
    bundle.manifest.json             counts, seed, and the split hash
```

**~58 MB total.** Upload the whole folder — nothing else is needed.

Two things worth knowing about what is in it:

- Rows come from the **frozen** `train` and `val` partitions. They are not
  re-split here; re-splitting would produce a different, unverifiable division
  and could leak val domains into training. `bundle.manifest.json` records the
  split hash so the bundle can be traced to the artifact it came from.
- All 303k phishing URLs are included; benign is capped at 1.2M. Phishing is the
  scarce class, and a 4.8M-parameter model over a ~140-token vocabulary saturates
  well before 5M benign URLs. This is an upload-time decision, not a statistical
  one.

---

## Steps

**1. Create the dataset**

Kaggle → *Datasets* → *New Dataset* → drag in `data/kaggle_bundle/`.
Name it `uie-corpus`. Private is fine.

It will mount at `/kaggle/input/uie-corpus`.

**2. Create the notebook**

Kaggle → *Code* → *New Notebook*.

- **Settings → Accelerator → GPU T4 x2** (or P100 — either is fine; the model
  uses one GPU).
- **Settings → Internet → On** *(only needed for the `pip install` in step 3; if
  `tldextract` is already in the image, turn it off — the training itself makes
  no network calls).*
- **Add Input** → your `uie-corpus` dataset.

**3. Run these cells**

```python
!pip install -q tldextract
```

```python
import sys
sys.path.insert(0, "/kaggle/input/uie-corpus")

import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

```python
!python -m uie.kaggle_train \
    --data /kaggle/input/uie-corpus \
    --out  /kaggle/working \
    --steps 20000 \
    --batch-size 256
```

**4. Download the outputs**

From the notebook's *Output* tab:

| file | what it is |
|---|---|
| `final.pt` | weights + architecture + **tokenizer**, in one file |
| `best.pt` | the lowest-validation-loss checkpoint |
| `train_result.json` | loss history, config, and 20 samples per class per decoder |

Put both `.pt` files in `data/` locally. E3 reads them from there.

---

## What to expect

| | |
|---|---|
| parameters | ~4.8M |
| vocabulary | ~140 tokens |
| steps | 20,000 at batch 256 ≈ 3.4 epochs |
| GPU time | **~40 minutes** on a T4 |
| dropped over `max_len` | ~1.5% of training rows |

Initial loss should be near `ln(vocab) ≈ 4.9` — that is the model correctly
knowing nothing. If it does not fall below 3 within the first thousand steps,
something is wrong; stop rather than burn the quota.

Validation loss is reported **per class**, never pooled. Benign and phishing
should both improve. If phishing stalls while benign improves, the class
balancing in the sampler is not working.

---

## Adjustments

**Out of memory** — halve `--batch-size` to 128. The model is small; this is
almost always a sequence-length spike rather than a real capacity problem.

**More capacity** — `--d-model 384 --n-layers 8` gives roughly 16M parameters and
still fits comfortably. Worth trying only *after* the 4.8M model is measured, so
the comparison means something.

**Kaggle quota** — 30 GPU-hours per week. A full run is well under one, so there
is room to iterate; there is no reason to rush a bad configuration through.

---

## The bar this has to clear

E1's published baseline (`BENCHMARK.md` §6), on held-out data:

| | benign | phishing |
|---|---|---|
| perplexity / char | 22.050 | 18.834 |
| discrimination | balanced accuracy 0.885 | |

E2 must beat **both** perplexity numbers *and* sample quality, with novelty
reported alongside — a model that wins by memorising has not won.

And the qualitative bar, which matters more than the numbers: the n-gram produces

```
http://awr.oudlisyrroinaacogiriwwigdhmfowloenr.com/1o%sawvn?D07r%4fw5
```

Structurally perfect, and obviously not a real URL. E2 has to put **meaning
inside the skeleton**.
