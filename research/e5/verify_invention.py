"""Verify -- or falsify -- "accurate when recalling, deficient when inventing".

The theory is CONFOUNDED as stated. Long segments are rare segments, so every
observation supporting "deficient when inventing" is equally consistent with
"deficient at long horizons". These need different cures.

V4  DECONFOUND. Bin real segments jointly by LENGTH and training FREQUENCY, and
    compare the model's share of each cell.
      rarity drives it   -> deficit at fixed length, across frequency
      horizon drives it  -> deficit at fixed frequency, across length
      both               -> theory needs restating as two effects

V2  COMPOUNDING. Predictive entropy at step k of FREE generation vs entropy at
    the same position on REAL prefixes.
      gap GROWS with k     -> drift into predictable regions; collapse confirmed
      gap CONSTANT         -> one-step conservatism, not compounding; "trajectory
                              collapse" is the wrong description
"""
from __future__ import annotations

import collections
import gzip
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.factorization import parse
from uie.model import DECODERS
from uie.tokenizer import _MARKERS
from uie.train import load_checkpoint

B = str(ROOT / "data/bundle2")
TMP = ROOT / "data/scratch"
gen = json.loads((TMP / "e5_domain_gen.json").read_text(encoding="utf-8"))
model, tok, cfg = load_checkpoint(str(ROOT / "data/best_e5.pt"))
model.eval()
PTH = tok.marker_id("PTH")
ENDS = set([tok.marker_id(m) for m in _MARKERS] + [tok.eos_id])


def segs(u):
    return [s for s in (parse(u).path or "").split("/") if s]


def rows(stem, limit=200000):
    out = []
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line)["url"])
            if len(out) >= limit:
                break
    return out


# ========================= V4: rarity vs length ============================ #
print("=" * 88)
print("V4  DECONFOUND: is the deficit driven by RARITY or by LENGTH?")
print("    cell value = model share / real share.  1.00 = faithful.")

for label, vstem, stem in (("benign", "name_train", "name_val"),
                           ("phishing", "phishing_name_train",
                            "phishing_name_val")):
    freq = collections.Counter(s.lower() for u in rows(vstem) for s in segs(u))
    real = [s for u in rows(stem) for s in segs(u)]
    fake = [s for u in gen[label] for s in segs(u)]

    def cell(s):
        f = freq.get(s.lower(), 0)
        fb = ("rare" if f <= 2 else "mid" if f <= 100 else "common")
        L = len(s)
        lb = ("short" if L <= 6 else "medium" if L <= 15 else "long")
        return lb, fb

    rc = collections.Counter(cell(s) for s in real)
    gc = collections.Counter(cell(s) for s in fake)
    rn, gn = sum(rc.values()), sum(gc.values())

    print(f"\n  --- {label} ---   real {rn:,} segments, model {gn:,}")
    print(f"  {'':<10}" + "".join(f"{f:>22}" for f in ("common", "mid", "rare")))
    for lb in ("short", "medium", "long"):
        cells = []
        for fb in ("common", "mid", "rare"):
            r, g = rc.get((lb, fb), 0) / rn, gc.get((lb, fb), 0) / gn
            cells.append(f"{r:>7.1%}/{g:<7.1%}{(g/r if r else 0):>6.2f}")
        print(f"  {lb:<10}" + "".join(cells))
    print("            (real share / model share, then ratio)")

# ========================= V2: compounding ================================= #
print("\n" + "=" * 88)
print("V2  COMPOUNDING: predictive entropy at step k, free vs real prefixes")

strategy = DECODERS["typical"]
for label, stem in (("benign", "name_val"), ("phishing", "phishing_name_val")):
    seqs = []
    for u in rows(stem, 1200):
        ids = tok.encode(u, label, frame="domain")
        if PTH not in ids or len(ids) > cfg.max_len:
            continue
        i = ids.index(PTH)
        j = i + 1
        while j < len(ids) and ids[j] not in ENDS:
            j += 1
        if j - i - 1 >= 24:
            seqs.append((ids, i + 1, j))
    if len(seqs) < 40:
        print(f"\n  {label}: only {len(seqs)} long paths"); continue
    seqs = seqs[:240]

    STARTS = 4                      # branch after 4 real path characters
    real_H, free_H = collections.defaultdict(list), collections.defaultdict(list)
    gen_ = torch.Generator(device="cpu"); gen_.manual_seed(11)
    with torch.no_grad():
        for s in range(0, len(seqs), 24):
            ch = seqs[s:s + 24]
            base = [list(ids[:a + STARTS]) for ids, a, _ in ch]
            free = [list(b) for b in base]
            for k in range(20):
                # real branch: entropy at the TRUE prefix, same absolute position
                w = max(len(ids) for ids, _, _ in ch)
                x = torch.full((len(ch), w), tok.pad_id, dtype=torch.long)
                for r, (ids, _, _) in enumerate(ch):
                    x[r, :len(ids)] = torch.tensor(ids)
                pr = torch.softmax(model(x).float(), -1)
                for r, (ids, a, b) in enumerate(ch):
                    pos = a + STARTS + k
                    if pos < b:
                        q = pr[r, pos - 1]
                        real_H[k].append(float(-(q * (q + 1e-12).log()).sum()))
                # free branch: entropy at the model's OWN prefix. Rows are
                # grouped by length because each row's real prefix ends at a
                # different absolute position, so the free branches are ragged.
                by_len = collections.defaultdict(list)
                for r, f in enumerate(free):
                    by_len[len(f)].append(r)
                for L, idxs in by_len.items():
                    y = torch.tensor([free[r] for r in idxs], dtype=torch.long)
                    pf = torch.softmax(model(y).float(), -1)
                    for slot, r in enumerate(idxs):
                        q = pf[slot, -1]
                        free_H[k].append(float(-(q * (q + 1e-12).log()).sum()))
                        free[r].append(strategy(pf[slot, -1].log(), gen_))

    print(f"\n  --- {label} ---")
    print(f"  {'step k':>8}{'real H':>10}{'free H':>10}{'gap':>9}")
    for k in (0, 2, 4, 8, 12, 16, 19):
        if len(real_H.get(k, [])) < 20 or len(free_H.get(k, [])) < 20:
            continue
        a, b_ = statistics.mean(real_H[k]), statistics.mean(free_H[k])
        print(f"  {k:>8}{a:>10.3f}{b_:>10.3f}{b_-a:>+9.3f}")
