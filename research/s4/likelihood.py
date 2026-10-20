"""Likelihood suite: E5 vs S4 w=1 / w=2 / w=4, one protocol for all four.

Replicates `uie.kaggle_train._run`'s reporting EXACTLY -- same streams, same
order, same first-3,000-per-class sample, same `boundary_nll` and `field_nll` --
so S4 w=1's numbers here must reproduce the ones its own Kaggle run printed. If
they do not, the local harness and the training job disagree and nothing
downstream is interpretable.

The four checkpoints were separately verified to share an architecture and a
byte-identical tokenizer, which is what makes cross-model NLL comparison valid
at all.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.evaluation import boundary_nll, field_nll
from uie.kaggle_train import STREAMS, VAL_STREAMS
from uie.train import load_checkpoint

B = ROOT / "data/bundle2"
MODELS = {"E5": "data/best_e5.pt", "w1": "data/s4_w1_best.pt",
          "w2": "data/s4_w2_best.pt", "w4": "data/s4_w4_best.pt"}
N = 3_000


def load(streams):
    """Same concatenation order as the trainer; label decides the bucket."""
    out = {"benign": [], "phishing": []}
    for stem, label, _corpus in streams:
        path = B / f"{stem}.jsonl.gz"
        if not path.is_file():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if len(out[label]) >= N * 4:      # headroom before slicing
                    break
                out[label].append(json.loads(line)["url"])
    return {k: v[:N] for k, v in out.items()}


@torch.no_grad()
def overall_ce(model, tok, urls, label, device, batch=64):
    """Unweighted next-token CE over every non-pad target.

    Reported alongside the field breakdown because the two can move in OPPOSITE
    directions: this one includes markers and EOS -- URL shape -- which the
    field breakdown deliberately excludes.
    """
    tot, n = 0.0, 0
    for i in range(0, len(urls), batch):
        enc = [tok.encode(u, label)[:model.cfg.max_len] for u in urls[i:i + batch]]
        w = max(len(e) for e in enc)
        ids = torch.tensor([e + [tok.pad_id] * (w - len(e)) for e in enc],
                           dtype=torch.long, device=device)
        lp = torch.log_softmax(model(ids).float(), dim=-1)
        nll = -lp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        keep = (ids[:, 1:] != tok.pad_id)
        tot += float((nll * keep).sum())
        n += int(keep.sum())
    return tot / n


if __name__ == "__main__":
    dev = torch.device("cpu")
    data = {"train": load(STREAMS), "val": load(VAL_STREAMS)}
    print({k: {c: len(v) for c, v in d.items()} for k, d in data.items()})

    report = {}
    for name, ckpt in MODELS.items():
        model, tok, _ = load_checkpoint(ckpt)
        model.eval()
        report[name] = {}
        for label in ("benign", "phishing"):
            report[name][label] = {}
            for split in ("train", "val"):
                urls = data[split][label]
                report[name][label][split] = {
                    "boundary": boundary_nll(model, tok, urls, label, dev),
                    "field": field_nll(model, tok, urls, label, dev),
                    "overall_ce": overall_ce(model, tok, urls, label, dev),
                }
                print(f"  {name} {label} {split} done", flush=True)
        del model

    out = ROOT / "data/s4_likelihood.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", out)
