"""The path-end hazard runs ~1pp high at every position. WHERE does that mass sit?

The field grammar is strict. Inside a path body, the only legal continuations are
a content character, or one of {QRY, FRG, EOS}. Every other marker is
STRUCTURALLY IMPOSSIBLE: SUF/LBL/SUB/SCH/USR/PRT all precede PTH in the emission
order, and a second PTH cannot follow the first.

So any probability the model places on those markers is pure error. It is also
counted as "the path ends here" by the sampler, because `_generate_chunk` stops
the field on any marker. If that illegal mass is an appreciable share of the
hazard excess, then:

  * the cause is an unlearned grammar constraint, not a length preference, and
  * it is fixable at DECODE time by masking illegal tokens -- no retrain.

If instead the excess sits on LEGAL enders (QRY/FRG/EOS), grammar masking cannot
help and the cause is a genuine length preference.

Reports the split, and what fraction of total end-mass is illegal.
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
CKPT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/best_e5.pt")
N, BATCH = 1200, 32

model, tok, cfg = load_checkpoint(CKPT)
model.eval()
FRAMED = getattr(tok, "include_frame", False)

PTH = tok.marker_id("PTH")
MARK = {m: tok.marker_id(m) for m in _MARKERS}
ALL_END = list(MARK.values()) + [tok.eos_id]
LEGAL = [MARK["QRY"], MARK["FRG"], tok.eos_id]
ILLEGAL = [i for i in ALL_END if i not in LEGAL]
NAMES = {v: k for k, v in MARK.items()}
NAMES[tok.eos_id] = "EOS"

CONFIGS = [("benign", "name_val", "domain"),
           ("benign", "structure_val", "page"),
           ("phishing", "phishing_name_val", "domain"),
           ("phishing", "phishing_structure_val", "page")]


def rows(stem, limit):
    out = []
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line)["url"])
            if len(out) >= limit:
                break
    return out


@torch.no_grad()
def decomp(urls, label, frame):
    seqs = []
    for u in urls:
        ids = (tok.encode(u, label, frame=frame) if FRAMED
               else tok.encode(u, label))
        if PTH not in ids or len(ids) > cfg.max_len:
            continue
        i = ids.index(PTH)
        j = i + 1
        while j < len(ids) and ids[j] not in ALL_END:
            j += 1
        if j > i + 1:
            seqs.append((ids, i + 1, j))
    if not seqs:
        return None

    mass = collections.defaultdict(float)
    n = 0
    real_end = 0
    for s in range(0, len(seqs), BATCH):
        chunk = seqs[s:s + BATCH]
        width = max(len(ids) for ids, _, _ in chunk)
        x = torch.full((len(chunk), width), tok.pad_id, dtype=torch.long)
        for r, (ids, _, _) in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids)
        p = torch.softmax(model(x).float(), dim=-1)
        for r, (ids, a, b) in enumerate(chunk):
            for pos in range(a, b):          # CONTENT positions only: the path
                q = p[r, pos - 1]            # does NOT end here in the data
                for t in ALL_END:
                    mass[t] += float(q[t])
                n += 1
            real_end += 1
    return mass, n, len(seqs)


print(f"checkpoint: {CKPT.split(chr(92))[-1]}   framed={FRAMED}")
print(f"legal after a path body : {[NAMES[i] for i in LEGAL]}")
print(f"structurally impossible : {[NAMES[i] for i in ILLEGAL]}")

for label, stem, frame in CONFIGS:
    r = decomp(rows(stem, N), label, frame)
    if r is None:
        continue
    mass, n, nseq = r
    tot = sum(mass.values()) / n
    leg = sum(mass[t] for t in LEGAL) / n
    ill = sum(mass[t] for t in ILLEGAL) / n
    print(f"\n=== {label}/{frame} ===  {n:,} non-terminal path positions, "
          f"{nseq:,} paths")
    print(f"  mean end-mass per position   {tot:.3%}")
    print(f"    legal   (QRY/FRG/EOS)      {leg:.3%}   ({leg/tot:.1%} of it)")
    print(f"    ILLEGAL (impossible here)  {ill:.3%}   ({ill/tot:.1%} of it)")
    print(f"  breakdown:")
    for t in sorted(ALL_END, key=lambda t: -mass[t]):
        v = mass[t] / n
        if v < 1e-5:
            continue
        tag = "legal" if t in LEGAL else "ILLEGAL"
        print(f"    {NAMES[t]:<5}{v:>9.3%}   {tag}")
