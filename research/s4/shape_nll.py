"""Trained-position CE and marker/EOS NLL -- the two metrics `likelihood.py` got
wrong or omitted.

WHY THE FIRST OVERALL CE WAS INVALID. It averaged every non-pad target,
including the control tokens (`<bos>` -> `<C:phishing>` -> `<F:domain>`) that
`field_mask` assigns weight 0. Those positions are SUPPLIED at generation and
never trained, so each model's prediction there is arbitrary. Measured on
phishing val, E5 and S4 w=1 differed by +0.310 nats over all positions and by
+0.001 over trained positions alone: the whole apparent gap was one untrained
position, not model quality.

WHAT IS REPORTED INSTEAD:
  trained_ce   CE over positions with field_mask weight > 0 -- the objective
               actually optimised, and the only overall number comparable
               across checkpoints
  marker_nll   markers plus EOS, i.e. URL SHAPE: which fields exist and where
               the URL stops. Trained on every row from both corpora, and
               excluded from `field_nll` by design, so nothing else in the
               suite measures it.

Shape is worth isolating because S4 could plausibly move it in either
direction: predicting where a segment STARTS is adjacent to predicting where
one ends.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.kaggle_train import STREAMS, VAL_STREAMS
from uie.tokenizer import _MARKERS
from uie.train import field_mask, load_checkpoint

B = ROOT / "data/bundle2"
MODELS = {"E5": "data/best_e5.pt", "w1": "data/s4_w1_best.pt",
          "w2": "data/s4_w2_best.pt", "w4": "data/s4_w4_best.pt"}
N = 3_000


def load(streams):
    out = {"benign": [], "phishing": []}
    for stem, label, _c in streams:
        p = B / f"{stem}.jsonl.gz"
        if not p.is_file():
            continue
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                if len(out[label]) >= N * 4:
                    break
                out[label].append(json.loads(line)["url"])
    return {k: v[:N] for k, v in out.items()}


@torch.no_grad()
def measure(model, tok, urls, label, batch=48):
    marker_ids = {tok.marker_id(m) for m in _MARKERS}
    shape_ids = marker_ids | {tok.eos_id}
    acc = {"trained": [0.0, 0], "marker": [0.0, 0], "eos": [0.0, 0]}
    for i in range(0, len(urls), batch):
        enc = [tok.encode(u, label)[:model.cfg.max_len] for u in urls[i:i + batch]]
        w = max(len(e) for e in enc)
        ids = torch.tensor([e + [tok.pad_id] * (w - len(e)) for e in enc],
                           dtype=torch.long)
        lp = torch.log_softmax(model(ids).float(), dim=-1)
        nll = -lp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        for r, e in enumerate(enc):
            # With a frame token the mask scopes every field on every row, so
            # the corpus name no longer changes the result; passed for clarity.
            mask = field_mask(e, tok, "structure")
            for t in range(len(e) - 1):
                if mask[t] == 0:
                    continue
                v = float(nll[r, t])
                acc["trained"][0] += v
                acc["trained"][1] += 1
                tgt = e[t + 1]
                if tgt in shape_ids:
                    k = "eos" if tgt == tok.eos_id else "marker"
                    acc[k][0] += v
                    acc[k][1] += 1
    return {k: (s / n if n else 0.0) for k, (s, n) in acc.items()} | \
           {f"{k}_n": n for k, (_s, n) in acc.items()}


if __name__ == "__main__":
    data = {"train": load(STREAMS), "val": load(VAL_STREAMS)}
    out = {}
    for name, ckpt in MODELS.items():
        model, tok, _ = load_checkpoint(ckpt)
        model.eval()
        out[name] = {}
        for label in ("benign", "phishing"):
            out[name][label] = {}
            for split in ("train", "val"):
                out[name][label][split] = measure(model, tok,
                                                  data[split][label], label)
                print(f"  {name} {label} {split} done", flush=True)
        del model
    p = ROOT / "data/s4_shape.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", p)
