"""Two open root causes. Hypotheses stated, then measured.

TASK 1 -- why does the model over-predict EOS mid-path on REAL prefixes?

  H1a POSITIONAL LENGTH PRIOR. The model may key EOS on ABSOLUTE token position
      rather than path context, having learned "sequences end around token ~46"
      (the benign mean). If so, EOS mass at the SAME path-relative position k
      should rise with absolute position -- i.e. a long host inflates mid-path
      EOS even though nothing about the path changed.
      TEST: hold k fixed, split by host length, compare EOS mass.
      REFUTED IF: EOS mass at fixed k is flat across host length.

  H1b ENTROPY LEAKAGE. Residual uncertainty spreads over the whole vocabulary,
      and EOS collects a share. Predicts EOS mass correlates with per-position
      predictive entropy.
      TEST: bin positions by entropy, compare mean EOS mass.
      REFUTED IF: no monotone relation.

  These are not exclusive. Both are measured on the same positions so their
  contributions can be compared directly.

TASK 2 -- why do benign paths never enter uppercase (~0%) when phishing does (20%)?

  H2a DATA RARITY. Benign uppercase is simply much rarer per path.
  H2b CONTEXT DEPENDENCE. Uppercase lives in contexts the model rarely
      generates (file extensions, ID tokens, CamelCase segments). If the model
      never produces the CONTEXT, it never enters the mode -- entry failure is
      then a symptom, not the cause.
      TEST: characterise where uppercase occurs in real benign paths, then check
      whether the model generates those contexts at all.
"""
from __future__ import annotations

import collections
import gzip
import json
import math
import re
import sys

import torch

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.factorization import parse
from uie.tokenizer import _MARKERS
from uie.train import load_checkpoint

B = str(ROOT / "data/bundle2")
TMP = str(ROOT / "data/scratch")

model, tok, cfg = load_checkpoint(str(ROOT / "data/best_e5.pt"))
model.eval()
PTH = tok.marker_id("PTH")
ALL_END = [tok.marker_id(m) for m in _MARKERS] + [tok.eos_id]


def rows(stem, limit):
    out = []
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line)["url"])
            if len(out) >= limit:
                break
    return out


def pb(u):
    return "/".join(s for s in (parse(u).path or "").split("/") if s)


# =========================================================================== #
print("=" * 78)
print("TASK 1 -- H1a positional length prior / H1b entropy leakage")

for label, stem, frame in (("benign", "name_val", "domain"),
                           ("phishing", "phishing_name_val", "domain")):
    seqs = []
    for u in rows(stem, 2500):
        ids = tok.encode(u, label, frame=frame)
        if PTH not in ids or len(ids) > cfg.max_len:
            continue
        i = ids.index(PTH)
        j = i + 1
        while j < len(ids) and ids[j] not in ALL_END:
            j += 1
        if j - i - 1 >= 16:                 # need depth in k to compare
            seqs.append((ids, i + 1, j))
    if len(seqs) < 60:
        print(f"\n{label}: only {len(seqs)} usable paths")
        continue

    # (k, host_len_bucket) -> eos mass ; entropy bin -> eos mass
    by_k_host = collections.defaultdict(list)
    ent_bins = collections.defaultdict(list)
    with torch.no_grad():
        for s in range(0, len(seqs), 32):
            ch = seqs[s:s + 32]
            w = max(len(x[0]) for x in ch)
            x = torch.full((len(ch), w), tok.pad_id, dtype=torch.long)
            for r, (ids, _, _) in enumerate(ch):
                x[r, :len(ids)] = torch.tensor(ids)
            p = torch.softmax(model(x).float(), dim=-1)
            for r, (ids, a, b) in enumerate(ch):
                host = a - 1                       # tokens before path content
                hb = "short" if host < 16 else ("mid" if host < 24 else "long")
                for k, pos in enumerate(range(a, min(a + 16, b))):
                    q = p[r, pos - 1]
                    eos = float(q[tok.eos_id])
                    by_k_host[(k, hb)].append(eos)
                    ent = float(-(q * (q + 1e-12).log()).sum())
                    ent_bins[min(int(ent / 0.5), 7)].append(eos)

    print(f"\n--- {label} ---")
    print("  H1a  EOS mass at fixed path position k, by HOST length")
    print(f"    {'k':>3}" + "".join(f"{h:>12}" for h in ("short", "mid", "long"))
          + f"{'long/short':>12}")
    for k in (2, 4, 6, 8, 10, 12, 14):
        vals = {}
        for hb in ("short", "mid", "long"):
            v = by_k_host.get((k, hb), [])
            vals[hb] = sum(v) / len(v) if len(v) >= 25 else None
        if vals["short"] and vals["long"]:
            ratio = vals["long"] / vals["short"]
            cells = "".join(f"{vals[h]:>12.3%}" if vals[h] else f"{'-':>12}"
                            for h in ("short", "mid", "long"))
            print(f"    {k:>3}{cells}{ratio:>12.2f}")

    print("  H1b  EOS mass by predictive ENTROPY of the same position")
    print(f"    {'entropy':>12}{'n':>8}{'EOS mass':>12}")
    for bkt in sorted(ent_bins):
        v = ent_bins[bkt]
        if len(v) < 50:
            continue
        print(f"    {bkt*0.5:>6.1f}-{bkt*0.5+0.5:<5.1f}{len(v):>8,}"
              f"{sum(v)/len(v):>12.3%}")

# =========================================================================== #
print("\n" + "=" * 78)
print("TASK 2 -- where does uppercase live in real paths, and does the model")
print("          ever generate that context?")

gen = json.loads(open(f"{TMP}/e5_domain_gen.json", encoding="utf-8").read())

PATTERNS = {
    "ALLCAPS token (>=3)": re.compile(r"(?:^|[^A-Za-z])[A-Z]{3,}(?:[^A-Za-z]|$)"),
    "CamelCase": re.compile(r"[a-z][A-Z]"),
    "capitalised word": re.compile(r"(?:^|[/_\-])[A-Z][a-z]{2,}"),
    "mixed alnum ID": re.compile(r"[A-Za-z]*[0-9][A-Za-z0-9]*[A-Z]"),
    "uppercase extension": re.compile(r"\.[A-Z]{2,4}$"),
    "percent-encoding": re.compile(r"%[0-9A-F]{2}"),
}

for label, stem in (("benign", "name_train"), ("phishing", "phishing_name_train")):
    real = [pb(u) for u in rows(stem, 60000)]
    real = [b for b in real if b]
    fake = [pb(u) for u in gen[label]]
    fake = [b for b in fake if b]

    def frac(bodies, rx):
        return sum(1 for b in bodies if rx.search(b)) / len(bodies)

    ru = sum(1 for b in real if any(c.isupper() for c in b)) / len(real)
    fu = sum(1 for b in fake if any(c.isupper() for c in b)) / len(fake)
    print(f"\n--- {label} ---   real paths {len(real):,}  model paths {len(fake):,}")
    print(f"  any uppercase          real {ru:>7.2%}   model {fu:>7.2%}"
          f"   ratio {fu/ru if ru else 0:>5.2f}")
    print(f"  {'context':<24}{'real':>9}{'model':>9}{'ratio':>8}")
    for name, rx in PATTERNS.items():
        a, b_ = frac(real, rx), frac(fake, rx)
        print(f"  {name:<24}{a:>9.2%}{b_:>9.2%}{(b_/a if a else 0):>8.2f}")

    # Of the real paths that DO contain uppercase, which pattern explains them?
    up = [b for b in real if any(c.isupper() for c in b)]
    if up:
        print(f"  of the {len(up):,} real paths WITH uppercase:")
        for name, rx in PATTERNS.items():
            print(f"    {name:<24}{frac(up, rx):>9.1%}")
        print(f"    examples: {', '.join(up[:6])}")
