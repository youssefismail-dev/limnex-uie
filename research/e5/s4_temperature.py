"""Can a boundary-only TEMPERATURE remove the over-confidence spike, no retrain?

Measured defect (S3): the over-confidence gap CE - H_model is +0.329 at segment
starts against +0.032 mid-segment (10.4x, benign). The model is OVER-CONFIDENT
at exactly one decision -- which word to start.

Over-confidence means the distribution is too sharp, and a temperature T > 1
flattens it. The objective here is CROSS-ENTROPY, not the S3 gap:

    KL(p_real || p_T) = CE(T) - H(p_real),   H(p_real) independent of T

so argmin_T CE = argmin_T KL exactly, and CE(1) - CE(T*) is the KL REDUCTION
temperature can buy. Note what is NOT available: H(p_real) is not estimable from
samples, so the ABSOLUTE KL is unmeasurable here -- only differences in T are.

Decision rule, fixed before the numbers:
  T* at the boundary exceeds T* mid-segment
      -> miscalibration is boundary-SPECIFIC (a targeted decode-time fix is
         well-posed rather than a global sharpness knob)
  the CE gain at T* is large
      -> temperature recovers the defect
  the CE gain at T* is small
      -> the defect is the SHAPE of the distribution, which a one-parameter
         rescaling cannot reach, and a training-side S4 is justified

Whether a small CE gain still shifts GENERATION is a separate question that this
script cannot answer; it is settled by the calibrated harness, not here.
"""
from __future__ import annotations

import collections
import gzip
import json
import statistics
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
TEMPS = (0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.6)


def rows(stem, limit):
    out = []
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line)["url"])
            if len(out) >= limit:
                break
    return out


for label, stem in (("benign", "name_val"), ("phishing", "phishing_name_val")):
    seqs = []
    for u in rows(stem, 3000):
        ids = tok.encode(u, label, frame="domain")
        if PTH not in ids or len(ids) > cfg.max_len:
            continue
        a = ids.index(PTH) + 1
        b = a
        while b < len(ids) and ids[b] not in ENDS:
            b += 1
        if b > a:
            seqs.append((ids, a, b))

    # Collect raw logits as STACKED TENSORS. Keeping 62,000 individual tensors
    # and looping over them in Python for each temperature is ~500k log_softmax
    # calls; stacking makes every temperature one vectorised pass.
    bnd_l, bnd_t, mid_l, mid_t = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(seqs), 24):
            ch = seqs[s:s + 24]
            w = max(len(ids) for ids, _, _ in ch)
            x = torch.full((len(ch), w), tok.pad_id, dtype=torch.long)
            for r, (ids, _, _) in enumerate(ch):
                x[r, :len(ids)] = torch.tensor(ids)
            lg = model(x).float()
            for r, (ids, a, b) in enumerate(ch):
                since = 0
                for pos in range(a, b):
                    if since == 0:
                        bnd_l.append(lg[r, pos - 1]); bnd_t.append(ids[pos])
                    else:
                        mid_l.append(lg[r, pos - 1]); mid_t.append(ids[pos])
                    since = 0 if ids[pos] == SLASH else since + 1
    bnd = (torch.stack(bnd_l), torch.tensor(bnd_t))
    mid = (torch.stack(mid_l), torch.tensor(mid_t))
    del bnd_l, mid_l

    @torch.no_grad()
    def stats_at(items, T):
        """CROSS-ENTROPY is the objective; over-confidence is diagnostic.

        KL(p_real||p_T) = CE - H(p_real), and H(p_real) does not depend on T, so
        MINIMISING CE minimises KL exactly. This is standard temperature scaling.

        An earlier version used CE - H(p_MODEL) as if it were KL. It is not:
        that equals KL + (H_real - H_model), so inflating the model's entropy
        with T drove it NEGATIVE -- impossible for a divergence, which is what
        exposed the error. CE - H_model remains reported as `overconf`, a valid
        over-confidence measure (confidence exceeding accuracy), but it is not
        the objective.
        """
        lg, tgt = items
        lp = torch.log_softmax(lg / T, dim=-1)
        ce = float(-lp.gather(1, tgt.unsqueeze(1)).mean())
        H = float(-(lp.exp() * lp).sum(-1).mean())
        return ce, ce - H

    print(f"\n{'=' * 74}\n=== {label} ===   boundary n={len(bnd[1]):,}  "
          f"mid n={len(mid[1]):,}")
    print(f"  {'T':>6}{'bnd CE':>10}{'bnd overconf':>14}"
          f"{'mid CE':>10}{'mid overconf':>14}")
    # BASELINE IS T=1.0, NOT TEMPS[0]. An earlier version captured `base` on the
    # first sweep entry (0.9), inflating every reported gain by the 0.9->1.0 drop.
    bestb, bestm, base = (1e9, None), (1e9, None), None
    for T in TEMPS:
        cb, ob = stats_at(bnd, T)
        cm, om = stats_at(mid, T)
        if T == 1.0:
            base = (cb, ob, cm, om)
        if cb < bestb[0]:
            bestb = (cb, T)
        if cm < bestm[0]:
            bestm = (cm, T)
        print(f"  {T:>6.2f}{cb:>10.4f}{ob:>+14.4f}{cm:>10.4f}{om:>+14.4f}")
    print(f"\n  boundary    CE {base[0]:.4f} at T=1 -> {bestb[0]:.4f} at "
          f"T*={bestb[1]}   gain {base[0]-bestb[0]:.4f} nats")
    print(f"  mid-segment CE {base[2]:.4f} at T=1 -> {bestm[0]:.4f} at "
          f"T*={bestm[1]}   gain {base[2]-bestm[0]:.4f} nats")
    print("  -> " + ("MISCALIBRATION IS BOUNDARY-SPECIFIC"
                     if bestb[1] > bestm[1] else
                     "miscalibration is NOT boundary-specific"))
