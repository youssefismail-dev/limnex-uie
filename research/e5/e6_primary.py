"""E6 PRIMARY TEST: did factorising the head stop the entropy leak?

Measured on E5, at positions where the path DEFINITIVELY CONTINUES, EOS mass
tracked the model's uncertainty about the next character:

    entropy 0.0-0.5 nats -> 0.111%   (benign)
    entropy 2.5-3.0 nats -> 3.022%   (27x)

That coupling is what the factorised head removes by construction: P(structural)
is now sigmoid(gate(h)) and cannot be moved by the content softmax.

PREDICTION, pre-registered: the entropy->EOS slope FLATTENS. If it does not, the
entropy-leakage diagnosis is wrong regardless of what happens to path length.

Both checkpoints are measured on IDENTICAL positions from identical sequences,
so the comparison is paired -- the only difference is the head.
"""
from __future__ import annotations

import collections
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
CKPTS = [("E5", str(ROOT / "data/best_e5.pt")),
         ("E6", str(ROOT / "data/best_e6.pt"))]


def rows(stem, limit):
    out = []
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line)["url"])
            if len(out) >= limit:
                break
    return out


for label, stem in (("benign", "name_val"), ("phishing", "phishing_name_val")):
    print(f"\n{'='*72}\n=== {label} ===")
    curves = {}
    for tag, path in CKPTS:
        model, tok, cfg = load_checkpoint(path)
        model.eval()
        PTH = tok.marker_id("PTH")
        ENDS = [tok.marker_id(m) for m in _MARKERS] + [tok.eos_id]

        seqs = []
        for u in rows(stem, 2500):
            ids = tok.encode(u, label, frame="domain")
            if PTH not in ids or len(ids) > cfg.max_len:
                continue
            i = ids.index(PTH)
            j = i + 1
            while j < len(ids) and ids[j] not in ENDS:
                j += 1
            if j - i - 1 >= 8:
                seqs.append((ids, i + 1, j))

        bins = collections.defaultdict(list)
        with torch.no_grad():
            for s in range(0, len(seqs), 32):
                ch = seqs[s:s + 32]
                w = max(len(x[0]) for x in ch)
                x = torch.full((len(ch), w), tok.pad_id, dtype=torch.long)
                for r, (ids, _, _) in enumerate(ch):
                    x[r, :len(ids)] = torch.tensor(ids)
                p = torch.softmax(model(x).float(), dim=-1)
                for r, (ids, a, b) in enumerate(ch):
                    for pos in range(a, min(a + 16, b)):
                        q = p[r, pos - 1]
                        ent = float(-(q * (q + 1e-12).log()).sum())
                        bins[min(int(ent / 0.5), 7)].append(float(q[tok.eos_id]))
        curves[tag] = bins

    print(f"  {'entropy (nats)':<16}{'n':>8}{'E5 EOS':>10}{'E6 EOS':>10}"
          f"{'ratio':>9}")
    e5b, e6b = curves["E5"], curves["E6"]
    for k in sorted(set(e5b) | set(e6b)):
        a, b_ = e5b.get(k, []), e6b.get(k, [])
        if len(a) < 50 or len(b_) < 50:
            continue
        ma, mb = sum(a) / len(a), sum(b_) / len(b_)
        print(f"  {k*0.5:>5.1f}-{k*0.5+0.5:<10.1f}{len(a):>8,}{ma:>10.3%}"
              f"{mb:>10.3%}{(mb/ma if ma else 0):>9.2f}")

    # The slope is the whole claim: leak should stop tracking entropy.
    def slope(bins):
        lo = [v for k in (0, 1) for v in bins.get(k, [])]
        hi = [v for k in (5, 6) for v in bins.get(k, [])]
        if not lo or not hi:
            return None
        return (sum(hi) / len(hi)) / (sum(lo) / len(lo))

    sa, sb = slope(e5b), slope(e6b)
    if sa and sb:
        print(f"\n  high/low entropy EOS ratio   E5 {sa:>6.1f}x   E6 {sb:>6.1f}x")
        print(f"  -> {'FLATTENED' if sb < sa * 0.5 else 'NOT flattened'}")
