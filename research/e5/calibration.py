"""Is the model MISCALIBRATED, or is generation drifting away from calibration?

Both open problems -- uppercase at ~0.56x and paths ~7 characters short -- are
statements about SAMPLED output. Neither says where the defect lives. Under
teacher forcing on REAL data the model's conditional distribution can be read
directly, with no autoregression and therefore no accumulation:

  PROBE U (uppercase)
    At every path-content position, sum the model's predicted probability mass
    over uppercase characters and compare to the empirical uppercase frequency
    at those same positions.

      model mass ~ empirical  -> conditionals are CALIBRATED; the deficit is
                                 created during free generation
      model mass << empirical -> the model genuinely under-predicts uppercase

  PROBE L (path length)
    At each position k inside a path, compare the model's hazard -- P(next token
    ends the path) -- against the empirical hazard, i.e. among real paths that
    reached length k, the fraction that stop at k.

      hazard matches           -> length is not a conditional defect
      hazard systematically high -> the model is too eager to stop, and the
                                 excess is measurable per position

These are diagnostics, not gates. They localise the defect; they do not score
the model.
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.tokenizer import _MARKERS
from uie.train import load_checkpoint

B = str(ROOT / "data/bundle2")
CKPT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/best_e5.pt")
N = 1000
BATCH = 32

model, tok, cfg = load_checkpoint(CKPT)
model.eval()
FRAMED = getattr(tok, "include_frame", False)

PTH = tok.marker_id("PTH")
MARKER_IDS = [tok.marker_id(m) for m in _MARKERS]
END_IDS = MARKER_IDS + [tok.eos_id]
UPPER_IDS = [i for ch, i in tok.stoi.items()
             if len(ch) == 1 and ch.isupper()]

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
def probe(urls, label, frame):
    """Returns (upper_model, upper_real, hazard_by_k)."""
    seqs = []
    for u in urls:
        ids = (tok.encode(u, label, frame=frame) if FRAMED
               else tok.encode(u, label))
        if PTH not in ids or len(ids) > cfg.max_len:
            continue
        i = ids.index(PTH)
        j = i + 1
        while j < len(ids) and ids[j] not in END_IDS:
            j += 1
        if j > i + 1:                       # non-empty path body
            seqs.append((ids, i + 1, j))    # content is [i+1, j), ends at j
    if not seqs:
        return None

    up_model = up_real = n_pos = 0.0
    haz_model = collections.defaultdict(float)
    haz_real = collections.defaultdict(float)
    haz_n = collections.Counter()

    for s in range(0, len(seqs), BATCH):
        chunk = seqs[s:s + BATCH]
        width = max(len(ids) for ids, _, _ in chunk)
        x = torch.full((len(chunk), width), tok.pad_id, dtype=torch.long)
        for r, (ids, _, _) in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids)
        logits = model(x).float()
        probs = torch.softmax(logits, dim=-1)

        for r, (ids, a, b) in enumerate(chunk):
            for k, pos in enumerate(range(a, b + 1)):
                # position pos is predicted by the logits at pos-1
                p = probs[r, pos - 1]
                ends_here = (pos == b)
                haz_model[k] += float(p[END_IDS].sum())
                haz_real[k] += 1.0 if ends_here else 0.0
                haz_n[k] += 1
                if not ends_here:           # a real content character
                    up_model += float(p[UPPER_IDS].sum())
                    up_real += 1.0 if tok.itos[ids[pos]].isupper() else 0.0
                    n_pos += 1
    return (up_model / n_pos, up_real / n_pos, n_pos,
            {k: (haz_model[k] / haz_n[k], haz_real[k] / haz_n[k], haz_n[k])
             for k in sorted(haz_n)})


print(f"checkpoint: {Path(CKPT).name}   framed={FRAMED}")
print(f"uppercase token ids in vocab: {len(UPPER_IDS)}")
if not UPPER_IDS:
    print("  !! NO UPPERCASE IN VOCAB -- generation cannot produce any")

for label, stem, frame in CONFIGS:
    r = probe(rows(stem, N), label, frame)
    if r is None:
        print(f"\n=== {label}/{frame} ===  no usable rows")
        continue
    um, ur, npos, haz = r
    print(f"\n=== {label}/{frame} ({stem}) ===  {npos:,} path-content positions")
    print(f"  PROBE U  empirical uppercase {ur:.2%}   "
          f"model predicted mass {um:.2%}   ratio {um/ur if ur else 0:.2f}")
    print(f"  PROBE L  hazard = P(path ends here)")
    print(f"    {'k':>4}{'n':>8}{'empirical':>12}{'model':>10}{'excess':>10}")
    for k in sorted(haz):
        m, e, n = haz[k]
        if n < 40 or k > 40:
            continue
        if k % 2 and k > 8:
            continue
        print(f"    {k:>4}{n:>8,}{e:>12.2%}{m:>10.2%}{m-e:>+10.2%}")
