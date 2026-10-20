"""E9 R1 -- the mechanism reading (PREREG_E9 sections 4, 6, 7): plain and
length-weighted calibration ratios for the decision families, with row-bootstrap
95% CIs, from a checkpoint's teacher-forced predictions (controls hidden: E7's
free mode). No generation, no training.

    python research/native/e9_mechanism.py --tag e7          # E7, the reference
    python research/native/e9_mechanism.py --tag e9_base --checkpoint data/e9/train/e9_base/best.pt
    python research/native/e9_mechanism.py --tag e9_ln   --checkpoint data/e9/train/e9_ln/best.pt

Families (weighted_calibration.py's events and rows, pooled):
  F1    ending the URL inside a path, positions 0-50 from the path's start
        (20,000 domain-frame path rows)
  F2    stopping inside a query value, positions 0-25 within the value
        (N2a-0's query sample: 20,000 benign / 8,272 phishing)
  F3    another parameter: '&' against the query ending, at a value's end
        (the same query rows)
  TAIL  stopping inside a query value, positions 50-100 (section 7's tail rule)

Ratio = sum of model mass / sum of events, over rows; weighted uses w = 1/(L-1)
per row. CIs resample ROWS (Q7's `boot_ci`, 1,000 draws, seed 20260806).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from n2a_reference import SAMPLE_SEED, STREAMS, TRAIN_ROWS, rows  # noqa: E402
from research.qry.q7_hazard import BATCH, boot_ci, field_spans  # noqa: E402
from uie.evaluation import FIELDS  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402

FAMILIES = ("F1", "F2", "F3", "TAIL")
SPANS = {"F1": (0, 50), "F2": (0, 25), "TAIL": (50, 100)}


def _forward(model, enc, pad_id):
    width = max(len(e) for e in enc)
    ids = torch.tensor([e + [pad_id] * (width - len(e)) for e in enc])
    with torch.no_grad():
        return F.softmax(model(ids).float(), dim=-1)


def per_row(model, tok, cfg, cls: str):
    """Per-row (events, mass) and (w*events, w*mass) for every family."""
    out = {f: {"plain": [], "weighted": []} for f in FAMILIES}
    amp = tok.stoi.get("&")
    slash = tok.stoi.get("/")
    eos = tok.eos_id
    stoppers = [t for t in (eos, tok.marker_id("FRG"), tok.marker_id("QRY"), amp) if t is not None]
    ends = [t for t in (eos, tok.marker_id("FRG")) if t is not None]

    def add(fam, y, p, w):
        if y or p:
            out[fam]["plain"].append((y, p))
            out[fam]["weighted"].append((w * y, w * p))

    # F2, F3, TAIL on N2a-0's query sample
    qrows = [u for u in rows(STREAMS[cls][0]) if FIELDS["QRY"](u)]
    qrows = random.Random(SAMPLE_SEED).sample(qrows, min(TRAIN_ROWS, len(qrows)))
    for start in range(0, len(qrows), BATCH):
        enc = [tok.encode(u, cls, frame="domain", hide_fields=("PTH", "QRY"))
               for u in qrows[start:start + BATCH]]
        enc = [e for e in enc if len(e) <= cfg.max_len]
        if not enc:
            continue
        probs = _forward(model, enc, tok.pad_id)
        for b, seq in enumerate(enc):
            w = 1.0 / (len(seq) - 1)
            acc = {f: [0.0, 0.0] for f in ("F2", "F3", "TAIL")}
            for s, e in field_spans(seq, tok, "QRY"):
                pos = s
                while pos < e:
                    end = pos
                    while end < e and seq[end] != amp:
                        end += 1
                    for i in range(pos, end):
                        within = i - pos
                        y = 1.0 if i + 1 >= end else 0.0
                        p = float(probs[b, i, stoppers].sum())
                        for fam in ("F2", "TAIL"):
                            lo, hi = SPANS[fam]
                            if lo <= within < hi:
                                acc[fam][0] += y
                                acc[fam][1] += p
                    if end > pos:
                        i = end - 1
                        nxt = seq[i + 1] if i + 1 < len(seq) else eos
                        if nxt == amp or nxt in ends:
                            pa = float(probs[b, i, amp])
                            pe = float(probs[b, i, ends].sum())
                            acc["F3"][0] += 1.0 if nxt == amp else 0.0
                            acc["F3"][1] += pa / (pa + pe) if pa + pe > 0 else 0.0
                    pos = end + 1
            for fam, (y, p) in acc.items():
                add(fam, y, p, w)

    # F1 on 20,000 domain-frame path rows
    prow = [u for u in rows(STREAMS[cls][0]) if FIELDS["PTH"](u)]
    prow = random.Random(SAMPLE_SEED).sample(prow, min(TRAIN_ROWS, len(prow)))
    for start in range(0, len(prow), BATCH):
        enc = [tok.encode(u, cls, frame="domain", hide_fields=("PTH", "QRY"))
               for u in prow[start:start + BATCH]]
        enc = [e for e in enc if len(e) <= cfg.max_len]
        if not enc:
            continue
        probs = _forward(model, enc, tok.pad_id)
        for b, seq in enumerate(enc):
            w = 1.0 / (len(seq) - 1)
            y_sum = p_sum = 0.0
            for s, e in field_spans(seq, tok, "PTH"):
                for i in range(s, e):
                    if not SPANS["F1"][0] <= i - s < SPANS["F1"][1]:
                        continue
                    nxt = seq[i + 1] if i + 1 < len(seq) else eos
                    y_sum += 1.0 if nxt == eos else 0.0
                    p_sum += float(probs[b, i, eos])
            add("F1", y_sum, p_sum, w)
    del slash
    return out, len(qrows), len(prow)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "best_e7.pt")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    model, tok, cfg = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    model.eval()
    report = {"registration": "research/native/PREREG_E9.md (R1)", "tag": args.tag,
              "checkpoint": str(args.checkpoint)}
    for cls in ("benign", "phishing"):
        fams, nq, np_ = per_row(model, tok, cfg, cls)
        rng = random.Random(20260806)
        res = {"query_rows": nq, "path_rows": np_}
        for fam in FAMILIES:
            res[fam] = {}
            for kind in ("plain", "weighted"):
                pairs = fams[fam][kind]
                point, lo, hi = boot_ci(pairs, rng)
                res[fam][kind] = {"ratio": point, "lo": lo, "hi": hi, "rows": len(pairs),
                                  "events": sum(p[0] for p in pairs)}
            print(f"  {args.tag:<8} {cls:<9} {fam:<5} plain {res[fam]['plain']['ratio']:.3f} "
                  f"[{res[fam]['plain']['lo']:.3f}, {res[fam]['plain']['hi']:.3f}]   weighted "
                  f"{res[fam]['weighted']['ratio']:.3f} [{res[fam]['weighted']['lo']:.3f}, "
                  f"{res[fam]['weighted']['hi']:.3f}]   rows {res[fam]['plain']['rows']:,}",
                  flush=True)
        report[cls] = res
    dest = args.out or ROOT / "data" / "e9" / f"mechanism_{args.tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
