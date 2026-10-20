"""S1 and S3: localise the selection failure at the segment boundary.

PARADOX TO RESOLVE. P("/") is calibrated at every within-segment position on
real prefixes (0.95-1.05, overall 1.02) and NLL falls with position inside a
segment (2.203 -> 1.458), yet free sampling produces segments 27% short. If the
hazard is right everywhere, sampling must be right -- unless the model's own
prefixes arrive at those positions carrying different content.

S1  OWN-TRAJECTORY HAZARD INFLATION. The model picks head-like words; a head word
    is complete at ~6 characters, so it then correctly closes. Termination is not
    the bug -- starting short words is.
      CONFIRMED IF: P("/") at matched within-segment positions is HIGHER on the
      model's own prefixes than on real ones.
      REFUTED IF:  the two curves coincide -- then the hazard is innocent AND the
      content is innocent, and the deficit must be elsewhere.

S3  BOUNDARY OVER-CONFIDENCE. Every calibration test so far measured P("/"), the
    TERMINATION decision. None measured the diversity of what FOLLOWS a "/". If
    predictive entropy at segment starts is below the empirical diversity, the
    model is over-confident at exactly the moment that fixes segment length, and
    a calibrated sampler will correctly pick head words most of the time.
      CONFIRMED IF: model entropy at segment starts << empirical entropy of the
      real first-character distribution at those same positions.
"""
from __future__ import annotations

import collections
import gzip
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.tokenizer import _MARKERS
from uie.train import load_checkpoint

B = str(ROOT / "data/bundle2")
TMP = ROOT / "data/scratch"
gen = json.loads((TMP / "e5_domain_gen.json").read_text(encoding="utf-8"))
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


def spans(urls, label):
    out = []
    for u in urls:
        ids = tok.encode(u, label, frame="domain")
        if PTH not in ids or len(ids) > cfg.max_len:
            continue
        a = ids.index(PTH) + 1
        b = a
        while b < len(ids) and ids[b] not in ENDS:
            b += 1
        if b > a:
            out.append((ids, a, b))
    return out


@torch.no_grad()
def probe(items):
    """P("/") by chars-since-separator, plus entropy at segment STARTS."""
    haz = collections.defaultdict(list)
    start_ent, start_top1, start_first = [], [], collections.Counter()
    for s in range(0, len(items), 24):
        ch = items[s:s + 24]
        w = max(len(ids) for ids, _, _ in ch)
        x = torch.full((len(ch), w), tok.pad_id, dtype=torch.long)
        for r, (ids, _, _) in enumerate(ch):
            x[r, :len(ids)] = torch.tensor(ids)
        p = torch.softmax(model(x).float(), dim=-1)
        for r, (ids, a, b) in enumerate(ch):
            since = 0
            for pos in range(a, b):
                haz[min(since // 4, 5)].append(float(p[r, pos - 1, SLASH]))
                if since == 0:                      # first char of a segment
                    q = p[r, pos - 1]
                    start_ent.append(float(-(q * (q + 1e-12).log()).sum()))
                    start_top1.append(float(q.max()))
                    start_first[ids[pos]] += 1
                since = 0 if ids[pos] == SLASH else since + 1
    return haz, start_ent, start_top1, start_first


LB = ["0-3", "4-7", "8-11", "12-15", "16-19", "20+"]

for label, stem in (("benign", "name_val"), ("phishing", "phishing_name_val")):
    real = spans(rows(stem, 2500), label)
    own = spans(gen[label], label)
    rh, re_, rt, rf = probe(real)
    oh, oe, ot, of = probe(own)

    print(f"\n{'='*76}\n=== {label} ===   real {len(real):,} paths, "
          f"model {len(own):,}")
    print("  S1  P(\"/\") by chars since the last separator")
    print(f"  {'position':<12}{'n real':>9}{'real':>9}{'n own':>9}{'own':>9}"
          f"{'own/real':>10}")
    for i, lb in enumerate(LB):
        a, b_ = rh.get(i, []), oh.get(i, [])
        if len(a) < 100 or len(b_) < 40:
            continue
        ma, mb = sum(a) / len(a), sum(b_) / len(b_)
        print(f"  {lb:<12}{len(a):>9,}{ma:>9.2%}{len(b_):>9,}{mb:>9.2%}"
              f"{(mb/ma if ma else 0):>10.2f}")

    def ent(counter):
        n = sum(counter.values())
        return -sum((v / n) * math.log(v / n) for v in counter.values() if v)

    print("\n  S3  at SEGMENT STARTS: what follows a \"/\"")
    print(f"  {'':<22}{'real prefixes':>16}{'own prefixes':>16}")
    print(f"  {'model entropy':<22}{sum(re_)/len(re_):>16.3f}"
          f"{sum(oe)/len(oe):>16.3f}")
    print(f"  {'model top-1 mass':<22}{sum(rt)/len(rt):>16.2%}"
          f"{sum(ot)/len(ot):>16.2%}")
    print(f"  {'EMPIRICAL entropy':<22}{ent(rf):>16.3f}{ent(of):>16.3f}")
    print(f"  {'distinct first chars':<22}{len(rf):>16}{len(of):>16}")
    print(f"  -> model/empirical entropy ratio, real prefixes: "
          f"{(sum(re_)/len(re_))/ent(rf):.2f}")
