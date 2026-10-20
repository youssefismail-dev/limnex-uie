"""Is the mid-path EOS mass a LEAK, or correct behaviour badly calibrated?

The earlier probe only looked at positions where the path definitively
continues, so the empirical hazard there is 0 by construction and ANY model mass
looked like excess. That cannot distinguish the two stories. This measures the
EMPIRICAL hazard alongside the model's, over ALL positions including terminal
ones, split by context:

    boundary      the prefix so far ends in "/" -- a complete segment just
                  closed, so a real path plausibly ends here
    non-boundary  mid-segment -- a real path almost never ends here

HYPOTHESIS (mine, unproven): the model's excess is legitimate boundary-aware
behaviour, merely mis-scaled.
    PREDICTED IF TRUE:  model/empirical ratio is similar at boundary and
                        non-boundary positions -- a uniform mis-scaling.
    PREDICTED IF FALSE: the excess concentrates at NON-boundary positions, where
                        real paths essentially never end. That would be a true
                        leak and would refute the hypothesis.

"End" means leaving the path field by ANY route -- EOS or a following marker --
not EOS alone, which is what the previous probe measured.
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
    print(f"\n{'='*74}\n=== {label} ===")
    results = {}
    empirical = None

    for tag, path in CKPTS:
        model, tok, cfg = load_checkpoint(path)
        model.eval()
        PTH = tok.marker_id("PTH")
        END_IDS = [tok.marker_id(m) for m in _MARKERS] + [tok.eos_id]
        END_SET = set(END_IDS)
        SLASH = tok.stoi.get("/")

        seqs = []
        for u in rows(stem, 3000):
            ids = tok.encode(u, label, frame="domain")
            if PTH not in ids or len(ids) > cfg.max_len:
                continue
            i = ids.index(PTH)
            j = i + 1
            while j < len(ids) and ids[j] not in END_SET:
                j += 1
            if j > i + 1:
                seqs.append((ids, i + 1, j))

        # empirical: reached / ended, by context -- model-independent, so it is
        # computed once and reused for both checkpoints.
        if empirical is None:
            emp = {"boundary": [0, 0], "non-boundary": [0, 0]}
            for ids, a, b in seqs:
                for i in range(0, b - a + 1):
                    kind = ("boundary" if i > 0 and ids[a + i - 1] == SLASH
                            else "non-boundary")
                    if i == 0:
                        continue          # path just opened; not a stop point
                    emp[kind][0] += 1
                    if i == b - a:
                        emp[kind][1] += 1
            empirical = {k: (v[1] / v[0] if v[0] else 0.0, v[0])
                         for k, v in emp.items()}

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
                    for i in range(1, b - a + 1):
                        kind = ("boundary" if ids[a + i - 1] == SLASH
                                else "non-boundary")
                        mod[kind].append(float(p[r, a + i - 1, END_IDS].sum()))
        results[tag] = {k: sum(v) / len(v) for k, v in mod.items() if v}

    print(f"  {'context':<16}{'n':>10}{'empirical':>12}{'E5 model':>11}"
          f"{'E5 ratio':>10}{'E6 model':>11}{'E6 ratio':>10}")
    for kind in ("boundary", "non-boundary"):
        e, n = empirical[kind]
        m5, m6 = results["E5"].get(kind), results["E6"].get(kind)
        r5 = m5 / e if e else float("inf")
        r6 = m6 / e if e else float("inf")
        print(f"  {kind:<16}{n:>10,}{e:>12.3%}{m5:>11.3%}{r5:>10.2f}"
              f"{m6:>11.3%}{r6:>10.2f}")

    eb = empirical["boundary"][0]
    en = empirical["non-boundary"][0]
    print(f"\n  empirical boundary/non-boundary hazard ratio: "
          f"{eb/en if en else float('inf'):.1f}x")
    for tag in ("E5", "E6"):
        rb = results[tag]["boundary"] / eb if eb else 0
        rn = results[tag]["non-boundary"] / en if en else 0
        verdict = ("UNIFORM mis-scaling" if 0.5 < rb / rn < 2.0
                   else "excess concentrated at NON-boundary -> real leak")
        print(f"  {tag}: over-prediction {rb:.2f}x boundary vs {rn:.2f}x "
              f"non-boundary  ->  {verdict}")
