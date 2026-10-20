"""Is the boundary conditional UNDERFIT or OVERFIT? The two demand opposite fixes.

This is the pre-check that has to pass before any training-side S4 is proposed,
because the obvious S4 -- up-weight boundary positions in the loss -- helps in
one regime and actively harms in the other:

  train CE ~ val CE, both high   UNDERFIT. The model has not learned the
                                 conditional. Boundary positions are ~6% of
                                 in-path steps, so they contribute ~6% of the
                                 gradient; re-weighting gives them more and the
                                 fit should improve.

  train CE << val CE             OVERFIT. The model has MEMORISED segment
                                 openings. Up-weighting makes it memorise
                                 harder: sharper on training answers, no better
                                 on held-out ones, and over-confidence RISES.
                                 The fix would have to be regularisation or more
                                 data, never more weight.

Over-confidence alone cannot distinguish these -- both regimes produce it. Only
the train/val gap can, which is why this is measured before anything is designed.
"""
from __future__ import annotations

import gzip
import json
import sys

import torch

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.tokenizer import _MARKERS
from uie.train import load_checkpoint

B = str(ROOT / "data/bundle2")
model, tok, cfg = load_checkpoint(str(ROOT / "data/best_e5.pt"))
model.eval()
PTH = tok.marker_id("PTH")
ENDS = set([tok.marker_id(m) for m in _MARKERS] + [tok.eos_id])
SLASH = tok.stoi["/"]

SPLITS = {
    ("benign", "train"): "name_train", ("benign", "val"): "name_val",
    ("phishing", "train"): "phishing_name_train",
    ("phishing", "val"): "phishing_name_val",
}


def measure(label, stem, limit=3000):
    seqs = []
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            ids = tok.encode(json.loads(line)["url"], label, frame="domain")
            if PTH not in ids or len(ids) > cfg.max_len:
                continue
            a = ids.index(PTH) + 1
            b = a
            while b < len(ids) and ids[b] not in ENDS:
                b += 1
            if b > a:
                seqs.append((ids, a, b))
            if len(seqs) >= limit:
                break

    acc = {"bnd": [0.0, 0.0, 0], "mid": [0.0, 0.0, 0]}   # ce_sum, H_sum, n
    with torch.no_grad():
        for s in range(0, len(seqs), 24):
            ch = seqs[s:s + 24]
            w = max(len(i) for i, _, _ in ch)
            x = torch.full((len(ch), w), tok.pad_id, dtype=torch.long)
            for r, (ids, _, _) in enumerate(ch):
                x[r, :len(ids)] = torch.tensor(ids)
            lp = torch.log_softmax(model(x).float(), dim=-1)
            H = -(lp.exp() * lp).sum(-1)
            for r, (ids, a, b) in enumerate(ch):
                since = 0
                for pos in range(a, b):
                    k = "bnd" if since == 0 else "mid"
                    acc[k][0] += float(-lp[r, pos - 1, ids[pos]])
                    acc[k][1] += float(H[r, pos - 1])
                    acc[k][2] += 1
                    since = 0 if ids[pos] == SLASH else since + 1
    return {k: (v[0] / v[2], v[1] / v[2], v[0] / v[2] - v[1] / v[2], v[2])
            for k, v in acc.items() if v[2]}


for label in ("benign", "phishing"):
    print(f"\n{'=' * 70}\n=== {label} ===")
    print(f"  {'split':<7}{'kind':<6}{'n':>8}{'CE':>9}{'H_model':>10}"
          f"{'overconf':>10}")
    got = {}
    for split in ("train", "val"):
        got[split] = measure(label, SPLITS[(label, split)])
        for kind in ("bnd", "mid"):
            ce, h, g, n = got[split][kind]
            print(f"  {split:<7}{kind:<6}{n:>8,}{ce:>9.3f}{h:>10.3f}{g:>+10.3f}")

    gap_b = got["val"]["bnd"][0] - got["train"]["bnd"][0]
    gap_m = got["val"]["mid"][0] - got["train"]["mid"][0]
    print(f"\n  generalisation gap (val CE - train CE):"
          f"  boundary {gap_b:+.3f}   mid-segment {gap_m:+.3f}")
    # The comparison that matters is the boundary gap against the mid-segment
    # gap, not against zero: some gap is expected everywhere. A boundary gap
    # much larger than mid-segment's means memorisation localised to the very
    # positions an S4 would up-weight.
    if gap_b > gap_m + 0.15:
        print("  -> OVERFIT AT THE BOUNDARY. Up-weighting would make it worse.")
    elif abs(gap_b - gap_m) <= 0.15:
        print("  -> boundary generalises like everywhere else: UNDERFIT, not "
              "memorised. Re-weighting is a coherent S4.")
    else:
        print("  -> boundary generalises BETTER than mid-segment; unexpected, "
              "investigate before designing anything.")
