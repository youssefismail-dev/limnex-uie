"""E9 design evidence -- would gradient clipping undo a length-neutral loss?
READ-ONLY: forward and backward passes on CPU, no optimiser step, no training.

    python research/native/e9_evidence_clipping.py

With a fixed denominator C the loss of a batch is exactly the current loss times
T_b / C (T_b = the batch's scored tokens), so its gradient is the current gradient
times T_b / C. One backward pass per batch gives both norms. Clipping at the
recipe's 1.0 rescales exactly the batches whose norm exceeds it; if those are
mostly the LONG batches, clipping partly restores the 1/L weighting.

Batches are built as training builds them: real training rows (both frames, both
classes by E7's class probabilities), grouped into near-uniform length bands, 256
rows each, with E7's field masks. Two models: E7's trained weights (late
training) and a fresh seeded initialisation (early training).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from uie.kaggle_train import STREAMS, load_streams  # noqa: E402
from uie.model import URLTransformer  # noqa: E402
from uie.train import CORPUS_FRAME, _pad, field_mask, load_checkpoint  # noqa: E402

BANDS = [(8, 30), (30, 60), (60, 120), (120, 257)]
PER_BAND = 3
BATCH = 256
MICRO = 32


def main() -> None:
    torch.set_num_threads(6)
    share = json.loads((ROOT / "data" / "native" / "e9_signal_share.json").read_text(encoding="utf-8"))
    mbar = share["expected_scored_tokens_per_draw"]
    C = BATCH * mbar
    e7, tok, cfg = load_checkpoint(ROOT / "data" / "best_e7.pt", device=torch.device("cpu"))
    torch.manual_seed(20260803)
    fresh = URLTransformer(cfg)
    urls, labels, _, corpora, label_counts = load_streams(ROOT / "data" / "bundle2", STREAMS)
    rng = random.Random(20261001)
    idx = rng.sample(range(len(urls)), 200_000)
    by_band = {b: [] for b in BANDS}
    for i in idx:
        ids = tok.encode(urls[i], labels[i], frame=CORPUS_FRAME[corpora[i]])
        for lo, hi in BANDS:
            if lo <= len(ids) < hi and len(by_band[(lo, hi)]) < BATCH * PER_BAND:
                by_band[(lo, hi)].append((ids, field_mask(ids, tok, corpora[i])))
    out = {"C": C, "mbar": mbar, "clip": 1.0, "bands": {}}
    print(f"C = 256 x {mbar:.2f} = {C:.0f} scored tokens")
    for name, model in (("E7 weights (late training)", e7), ("fresh init (early training)", fresh)):
        model.train()
        res = {}
        for (lo, hi), rows in by_band.items():
            norms_now, norms_new, tokens = [], [], []
            for k in range(PER_BAND):
                chunk = rows[k * BATCH:(k + 1) * BATCH]
                if len(chunk) < BATCH:
                    break
                # The full-batch gradient, accumulated over micro-batches (memory):
                # loss = sum_batch / T_b, so each micro-batch contributes
                # loss_mb * T_mb / T_b -- exactly the full batch's gradient.
                parts = [_pad(chunk[j:j + MICRO], tok.pad_id)
                         for j in range(0, len(chunk), MICRO)]
                t = sum(float((w * (ids[:, 1:] != tok.pad_id)).sum()) for ids, w in parts)
                model.zero_grad(set_to_none=True)
                for ids, w in parts:
                    t_mb = float((w * (ids[:, 1:] != tok.pad_id)).sum())
                    (model.loss(ids, tok.pad_id, w) * (t_mb / t)).backward()
                g = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                   if p.grad is not None)).item()
                norms_now.append(g)
                norms_new.append(g * t / C)
                tokens.append(t)
            res[f"{lo}-{hi - 1}"] = {"tokens_per_batch": tokens, "norm_now": norms_now,
                                     "norm_length_neutral": norms_new,
                                     "scale": [t / C for t in tokens]}
            print(f"  {name:<28} band {lo:>3}-{hi - 1:<3} tokens/batch "
                  f"{sum(tokens) / len(tokens):>8.0f}  norm now "
                  f"{sum(norms_now) / len(norms_now):.3f}  length-neutral "
                  f"{sum(norms_new) / len(norms_new):.3f}  (x{sum(tokens) / len(tokens) / C:.2f})",
                  flush=True)
        out["bands"][name] = res
    dest = ROOT / "data" / "native" / "e9_clipping.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
