"""D1: is the separator "/" over-emitted, and is that CAUSE or SYMPTOM?

Arithmetic: the model emits internal "/" at 3.79% of path characters against a
real 2.75% (x1.38), which is what makes segments 27% short while depth stays
exactly right.

But that ratio is mechanically implied by short segments -- shorter segments mean
more separators per character regardless of cause. So the ratio alone cannot
distinguish:

  D1-CALIBRATION  the model genuinely over-predicts P("/") given a real prefix.
                  TEST: teacher-forced P("/") at real path positions vs the
                  empirical rate, binned by characters since the last separator
                  (P("/") must rise with segment length in both).
                  CONFIRMED IF: model curve sits above empirical.

  D1-SYMPTOM      P("/") is calibrated on real prefixes, and over-emission
                  appears only on the model's OWN prefixes -- i.e. it is
                  downstream of inventing short segments, not the cause.
                  CONFIRMED IF: model matches empirical on real prefixes.

D2 already showed the model reproduces KNOWN words at the right length (ratio
1.00) and truncates only INVENTED ones (0.80), which predicts D1-SYMPTOM.
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
        i = ids.index(PTH)
        j = i + 1
        while j < len(ids) and ids[j] not in ENDS:
            j += 1
        if j > i + 1:
            seqs.append((ids, i + 1, j))

    emp = collections.defaultdict(lambda: [0, 0])   # bin -> [reached, was "/"]
    mod = collections.defaultdict(list)
    with torch.no_grad():
        for s in range(0, len(seqs), 32):
            ch = seqs[s:s + 32]
            w = max(len(x[0]) for x in ch)
            x = torch.full((len(ch), w), tok.pad_id, dtype=torch.long)
            for r, (ids, _, _) in enumerate(ch):
                x[r, :len(ids)] = torch.tensor(ids)
            p = torch.softmax(model(x).float(), dim=-1)
            for r, (ids, a, b) in enumerate(ch):
                since = 0
                for pos in range(a, b):
                    # predicting ids[pos] from everything before it
                    bkt = min(since // 4, 6)
                    emp[bkt][0] += 1
                    if ids[pos] == SLASH:
                        emp[bkt][1] += 1
                    mod[bkt].append(float(p[r, pos - 1, SLASH]))
                    since = 0 if ids[pos] == SLASH else since + 1

    print(f"\n{'='*70}\n=== {label} ===   P(\"/\") on REAL prefixes")
    print(f"  {'chars since /':<16}{'n':>9}{'empirical':>12}{'model':>10}"
          f"{'ratio':>9}")
    tot_e = tot_m = tot_n = 0
    for bkt in sorted(emp):
        n, hits = emp[bkt]
        if n < 200:
            continue
        e = hits / n
        m_ = sum(mod[bkt]) / len(mod[bkt])
        tot_e += hits; tot_m += sum(mod[bkt]); tot_n += n
        lbl = f"{bkt*4}-{bkt*4+3}" if bkt < 6 else "24+"
        print(f"  {lbl:<16}{n:>9,}{e:>12.2%}{m_:>10.2%}"
              f"{(m_/e if e else 0):>9.2f}")
    print(f"  {'OVERALL':<16}{tot_n:>9,}{tot_e/tot_n:>12.2%}"
          f"{tot_m/tot_n:>10.2%}{(tot_m/tot_e if tot_e else 0):>9.2f}")
