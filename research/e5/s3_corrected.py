"""S3, corrected: is the model OVER-CONFIDENT specifically at segment boundaries?

The original S3 compared the model's CONDITIONAL entropy at segment starts
(2.836 benign / 2.182 phishing) against the MARGINAL entropy of first characters
pooled over all contexts (3.292 / 3.505), and read the ratios 0.86 / 0.62 as
over-confidence. That is invalid: H(X) >= H(X|C) always, so a PERFECT model also
scores below 1.0. The confound pointed in the direction of the conclusion.

This measures a quantity that is zero for a calibrated model:

    gap = E_real[-log p_model(x)] - E[H(p_model)]

The first term is cross-entropy, the second the model's own entropy; their
difference is the KL divergence and is >= 0 by construction. Levels do not
matter -- only whether the gap at BOUNDARY positions materially exceeds the gap
at MID-SEGMENT positions, which is what "over-confidence localised to the
decision point" means.

  boundary     the prefix ends in "/" (or the path just opened) -- the model is
               choosing which word to start, the decision that fixes length
  mid-segment  the model is continuing a word already begun

CONFIRMED IF: boundary gap >> mid-segment gap.
REFUTED IF:   the two are comparable -- then the model is no more miscalibrated
              at the boundary than anywhere else, and S3 is not the mechanism.
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

    # kind -> (nll list, H list); plus a within-segment position breakdown
    agg = collections.defaultdict(lambda: ([], []))
    bypos = collections.defaultdict(lambda: ([], []))
    with torch.no_grad():
        for s in range(0, len(seqs), 24):
            ch = seqs[s:s + 24]
            w = max(len(ids) for ids, _, _ in ch)
            x = torch.full((len(ch), w), tok.pad_id, dtype=torch.long)
            for r, (ids, _, _) in enumerate(ch):
                x[r, :len(ids)] = torch.tensor(ids)
            lp = torch.log_softmax(model(x).float(), dim=-1)
            p = lp.exp()
            for r, (ids, a, b) in enumerate(ch):
                since = 0
                for pos in range(a, b):
                    q, l = p[r, pos - 1], lp[r, pos - 1]
                    H = float(-(q * l).sum())
                    nll = -float(l[ids[pos]])
                    kind = "boundary" if since == 0 else "mid-segment"
                    agg[kind][0].append(nll)
                    agg[kind][1].append(H)
                    bypos[min(since, 8)][0].append(nll)
                    bypos[min(since, 8)][1].append(H)
                    since = 0 if ids[pos] == SLASH else since + 1

    print(f"\n{'='*74}\n=== {label} ===")
    print(f"  {'position kind':<16}{'n':>9}{'NLL':>9}{'H':>9}"
          f"{'gap = KL':>11}")
    res = {}
    for kind in ("boundary", "mid-segment"):
        nll, H = agg[kind]
        if len(nll) < 100:
            continue
        mn, mh = statistics.mean(nll), statistics.mean(H)
        res[kind] = mn - mh
        print(f"  {kind:<16}{len(nll):>9,}{mn:>9.3f}{mh:>9.3f}{mn-mh:>+11.3f}")
    if len(res) == 2:
        ratio = res["boundary"] / res["mid-segment"] if res["mid-segment"] else 0
        print(f"\n  boundary KL / mid-segment KL = {ratio:.2f}")
        print(f"  -> S3 {'CONFIRMED' if ratio > 1.5 else 'REFUTED'}"
              f"  (over-confidence {'is' if ratio > 1.5 else 'is NOT'} "
              f"localised to the boundary)")

    print(f"\n  by characters since the last separator")
    print(f"  {'since':<8}{'n':>9}{'NLL':>9}{'H':>9}{'gap':>10}")
    for k in sorted(bypos):
        nll, H = bypos[k]
        if len(nll) < 100:
            continue
        mn, mh = statistics.mean(nll), statistics.mean(H)
        print(f"  {('0 (start)' if k == 0 else str(k)):<8}{len(nll):>9,}"
              f"{mn:>9.3f}{mh:>9.3f}{mn-mh:>+10.3f}")
