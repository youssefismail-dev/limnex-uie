"""Does the model lose the thread INSIDE a long segment, or avoid starting one?

Surviving regularity: fidelity degrades with segment LENGTH, in every V4 column
of both classes. Rarity is benign-only; compounding entropy collapse was not
supported. Two mechanisms remain and need opposite cures:

  HORIZON    the model cannot sustain a long token -- per-character NLL rises
             with position INSIDE the segment.
  SELECTION  it sustains fine but rarely commits to a long segment -- NLL is
             flat or falling with position, and the deficit is upstream.

CONFOUND, and the design that removes it: position 25 only exists in segments of
26+ characters, so comparing position 5 across all segments against position 25
is comparing different content. Both positions are therefore measured WITHIN THE
SAME segments, restricted to those long enough to have both.

Baseline expectation: NLL should FALL with position -- more context makes the
rest of a word more predictable ("nectar" -> "ine"). Flat or rising is the
anomaly.

Second curve, the decisive one: the same measurement on the model's OWN
generated segments. Teacher-forced hazards say the model is nearly calibrated on
real prefixes (P("/") 1.02, path-exit 1.10-1.15x, together predicting only a
0.9-character segment deficit against an observed 4.3), so the deficit lives on
its own trajectories. If the own-curve rises where the real-curve falls, the
model finds its own inventions progressively harder, which raises termination.
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
MINLEN = 26          # long enough to hold both position 5 and position 25


def rows(stem, limit=200000):
    out = []
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line)["url"])
            if len(out) >= limit:
                break
    return out


def segments_with_positions(urls, label, freq=None):
    """(ids, [(abs_pos, within_seg_pos, freq_bucket)]) for long segments only."""
    out = []
    for u in urls:
        ids = tok.encode(u, label, frame="domain")
        if PTH not in ids or len(ids) > cfg.max_len:
            continue
        a = ids.index(PTH) + 1
        b = a
        while b < len(ids) and ids[b] not in ENDS:
            b += 1
        marks, start = [], a
        for pos in range(a, b + 1):
            if pos == b or ids[pos] == SLASH:
                if pos - start >= MINLEN:
                    text = tok.decode_fields(ids[start:pos]) if False else None
                    seg = "".join(tok.itos[t] for t in ids[start:pos])
                    fb = "-"
                    if freq is not None:
                        f = freq.get(seg.lower(), 0)
                        fb = "rare" if f <= 2 else "mid" if f <= 100 else "common"
                    for k in range(pos - start):
                        marks.append((start + k, k, fb))
                start = pos + 1
        if marks:
            out.append((ids, marks))
    return out


@torch.no_grad()
def nll_curve(items):
    """Mean per-character NLL by within-segment position, and by freq bucket."""
    by_pos = collections.defaultdict(list)
    by_pos_fb = collections.defaultdict(list)
    for s in range(0, len(items), 16):
        ch = items[s:s + 16]
        w = max(len(ids) for ids, _ in ch)
        x = torch.full((len(ch), w), tok.pad_id, dtype=torch.long)
        for r, (ids, _) in enumerate(ch):
            x[r, :len(ids)] = torch.tensor(ids)
        lp = torch.log_softmax(model(x).float(), dim=-1)
        for r, (ids, marks) in enumerate(ch):
            for abs_pos, k, fb in marks:
                v = -float(lp[r, abs_pos - 1, ids[abs_pos]])
                by_pos[min(k // 5, 5)].append(v)
                by_pos_fb[(min(k // 5, 5), fb)].append(v)
    return by_pos, by_pos_fb


LB = ["0-4", "5-9", "10-14", "15-19", "20-24", "25+"]

for label, vstem, stem in (("benign", "name_train", "name_val"),
                           ("phishing", "phishing_name_train",
                            "phishing_name_val")):
    freq = collections.Counter(
        s.lower() for u in rows(vstem)
        for s in (x for x in (parse(u).path or "").split("/") if x))

    real_items = segments_with_positions(rows(stem, 6000), label, freq)
    gen_items = segments_with_positions(gen[label], label, freq)
    if not real_items:
        print(f"\n{label}: no long segments"); continue

    rp, rpf = nll_curve(real_items)
    gp, _ = nll_curve(gen_items) if gen_items else ({}, {})

    print(f"\n{'='*82}\n=== {label} ===   real segments >={MINLEN} chars: "
          f"{len(real_items):,} urls   model: {len(gen_items):,}")
    print(f"  {'position':<10}{'n':>8}{'REAL nll':>11}{'n':>8}{'OWN nll':>11}"
          f"{'own-real':>11}")
    for i, lb in enumerate(LB):
        r = rp.get(i, [])
        g = gp.get(i, [])
        if len(r) < 30:
            continue
        mr = statistics.mean(r)
        if len(g) >= 30:
            mg = statistics.mean(g)
            print(f"  {lb:<10}{len(r):>8,}{mr:>11.3f}{len(g):>8,}{mg:>11.3f}"
                  f"{mg-mr:>+11.3f}")
        else:
            print(f"  {lb:<10}{len(r):>8,}{mr:>11.3f}{len(g):>8,}"
                  f"{'--':>11}{'--':>11}")

    print(f"\n  REAL segments, split by training frequency (controls rarity)")
    print(f"  {'position':<10}" + "".join(f"{f:>14}" for f in
                                          ("common", "mid", "rare")))
    for i, lb in enumerate(LB):
        cells = []
        for fb in ("common", "mid", "rare"):
            v = rpf.get((i, fb), [])
            cells.append(f"{statistics.mean(v):>9.3f}({len(v):>3})"
                         if len(v) >= 30 else f"{'--':>14}")
        if any("--" not in c for c in cells):
            print(f"  {lb:<10}" + "".join(cells))
