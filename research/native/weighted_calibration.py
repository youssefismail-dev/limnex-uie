"""Is E7 calibrated to the plain data, or to the length-weighted data?
[DIAGNOSTIC], post hoc (FINDING_LENGTH_WEIGHTING.md section 3); teacher forcing
on E7's own training rows, no training.

    python research/native/weighted_calibration.py --label benign

A model trained on a loss that weights row r by w_r = 1/(L_r - 1) is, at a
stationary point, calibrated in the WEIGHTED sense: for a token and a large
group of positions, sum w * p ~= sum w * y (the output-bias first-order
condition). A plainly calibrated model has sum p ~= sum y instead. So for each
decision, over the same rows and positions:

    plain ratio    = sum p / sum y            (N2a-0's M1 is exactly this)
    weighted ratio = sum w p / sum w y

Weighted ~= 1 with plain != 1 means the length weighting explains the
miscalibration; weighted != 1 means something else is at work.

Decisions (Q7's conventions; controls hidden, E7's free mode):
- STOP inside query values, by Q7's within-value buckets (stoppers: end, <FRG>,
  <QRY>, '&'). The plain ratio must reproduce N2a-0's M1 train ratios exactly: a
  validity check;
- ANOTHER PARAMETER at the end of a query value: '&' against the query ending.

`--field PTH` (path-bearing domain-frame training rows, same sample rules):
- END OF URL inside the path: the end token's mass at every path position, by
  position from the path's start -- the decision the frozen PTH end delta damps;
- ANOTHER SEGMENT at a segment's end: '/' against the path ending.
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
from research.qry.q7_hazard import BATCH, BUCKETS, bucket_of, field_spans, label  # noqa: E402
from uie.evaluation import FIELDS  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402


def path_mode(cls: str, checkpoint: Path) -> None:
    model, tok, cfg = load_checkpoint(checkpoint, device=torch.device("cpu"))
    model.eval()
    urls = [u for u in rows(STREAMS[cls][0]) if FIELDS["PTH"](u)]
    urls = random.Random(SAMPLE_SEED).sample(urls, min(TRAIN_ROWS, len(urls)))
    slash = tok.stoi.get("/")
    eos = tok.eos_id
    after_path = {tok.marker_id("QRY"), tok.marker_id("FRG"), eos}
    end_eos = {bk: [0.0, 0.0, 0.0, 0.0] for bk in BUCKETS}
    seg = [0.0, 0.0, 0.0, 0.0]
    for start in range(0, len(urls), BATCH):
        enc = [tok.encode(u, cls, frame="domain", hide_fields=("PTH", "QRY"))
               for u in urls[start:start + BATCH]]
        enc = [e for e in enc if len(e) <= cfg.max_len]
        if not enc:
            continue
        width = max(len(e) for e in enc)
        ids = torch.tensor([e + [tok.pad_id] * (width - len(e)) for e in enc])
        with torch.no_grad():
            probs = F.softmax(model(ids).float(), dim=-1)
        for b, seq in enumerate(enc):
            w = 1.0 / (len(seq) - 1)
            for s, e in field_spans(seq, tok, "PTH"):
                for i in range(s, e):
                    nxt = seq[i + 1] if i + 1 < len(seq) else eos
                    y = 1.0 if nxt == eos else 0.0
                    p = float(probs[b, i, eos])
                    acc = end_eos[bucket_of(i - s)]
                    acc[0] += p
                    acc[1] += y
                    acc[2] += w * p
                    acc[3] += w * y
                    # a segment's last character: another '/' segment, or the path ends
                    if seq[i] != slash and (nxt == slash or nxt in after_path):
                        if nxt == slash and (i + 2 >= len(seq) or seq[i + 2] in after_path):
                            continue                  # a trailing '/', not a new segment
                        ps = float(probs[b, i, slash])
                        pe = float(probs[b, i, list(after_path)].sum())
                        p = ps / (ps + pe) if ps + pe > 0 else 0.0
                        y = 1.0 if nxt == slash else 0.0
                        seg[0] += p
                        seg[1] += y
                        seg[2] += w * p
                        seg[3] += w * y
    out = {"status": "DIAGNOSTIC, post hoc", "label": cls, "field": "PTH",
           "checkpoint": str(checkpoint), "rows": len(urls), "end_of_url": {}}
    print(f"== {cls} PTH: {len(urls):,} path rows, controls hidden")
    print(f"   END OF URL inside the path  {'plain':>7} {'weighted':>9} {'end events':>11}")
    for bk in BUCKETS:
        sp, sy, swp, swy = end_eos[bk]
        if not sy:
            continue
        out["end_of_url"][label(bk)] = {"plain": sp / sy, "weighted": swp / swy, "events": sy}
        print(f"   {label(bk):<28}{sp / sy:>7.3f} {swp / swy:>9.3f} {sy:>11,.0f}")
    sp, sy, swp, swy = seg
    out["another_segment"] = {"plain": sp / sy if sy else None,
                              "weighted": swp / swy if swy else None, "events": sy}
    print(f"   ANOTHER SEGMENT ('/' at a segment's end): plain {sp / sy:.3f}  weighted "
          f"{swp / swy:.3f}  ({sy:,.0f} events)")
    dest = ROOT / "data" / "e8" / f"weighted_calibration_pth_{cls}.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=("benign", "phishing"), required=True)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "best_e7.pt")
    ap.add_argument("--field", choices=("QRY", "PTH"), default="QRY")
    args = ap.parse_args()
    cls = args.label
    if args.field == "PTH":
        path_mode(cls, args.checkpoint)
        return
    model, tok, cfg = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    model.eval()
    train_q = [u for u in rows(STREAMS[cls][0]) if FIELDS["QRY"](u)]
    train_q = random.Random(SAMPLE_SEED).sample(train_q, min(TRAIN_ROWS, len(train_q)))
    amp = tok.stoi.get("&")
    stoppers = [t for t in (tok.eos_id, tok.marker_id("FRG"), tok.marker_id("QRY"), amp)
                if t is not None]
    ends = [t for t in (tok.eos_id, tok.marker_id("FRG")) if t is not None]
    # per bucket: [sum p, sum y, sum w p, sum w y]
    stop = {bk: [0.0, 0.0, 0.0, 0.0] for bk in BUCKETS}
    more = [0.0, 0.0, 0.0, 0.0]            # '&' at a value's end
    for start in range(0, len(train_q), BATCH):
        enc = [tok.encode(u, cls, frame="domain", hide_fields=("PTH", "QRY"))
               for u in train_q[start:start + BATCH]]
        enc = [e for e in enc if len(e) <= cfg.max_len]
        if not enc:
            continue
        width = max(len(e) for e in enc)
        ids = torch.tensor([e + [tok.pad_id] * (width - len(e)) for e in enc])
        with torch.no_grad():
            probs = F.softmax(model(ids).float(), dim=-1)
        for b, seq in enumerate(enc):
            w = 1.0 / (len(seq) - 1)
            for s, e in field_spans(seq, tok, "QRY"):
                pos = s
                while pos < e:
                    end = pos
                    while end < e and seq[end] != amp:
                        end += 1
                    for i in range(pos, end):
                        y = 1.0 if i + 1 >= end else 0.0
                        p = float(probs[b, i, stoppers].sum())
                        acc = stop[bucket_of(i - pos)]
                        acc[0] += p
                        acc[1] += y
                        acc[2] += w * p
                        acc[3] += w * y
                    if end > pos:                       # the value's last token
                        i = end - 1
                        nxt = seq[i + 1] if i + 1 < len(seq) else tok.eos_id
                        if nxt == amp or nxt in ends:
                            pa = float(probs[b, i, amp])
                            pe = float(probs[b, i, ends].sum())
                            p = pa / (pa + pe) if pa + pe > 0 else 0.0
                            y = 1.0 if nxt == amp else 0.0
                            more[0] += p
                            more[1] += y
                            more[2] += w * p
                            more[3] += w * y
                    pos = end + 1
    measured = json.loads((ROOT / "data" / "native" / f"n2a_reference_{cls}.json")
                          .read_text(encoding="utf-8"))["train"]["M1_buckets"]
    out = {"status": "DIAGNOSTIC, post hoc", "label": cls, "checkpoint": str(args.checkpoint),
           "rows": len(train_q), "stop": {}, "another_parameter": {}}
    print(f"== {cls}: {len(train_q):,} query rows (N2a-0's training sample), controls hidden")
    print(f"   STOP in query values   {'plain':>7} {'weighted':>9} {'N2a-0 M1':>9}")
    for bk in BUCKETS:
        sp, sy, swp, swy = stop[bk]
        if not sy:
            continue
        m = measured.get(label(bk)) or measured.get(f"{bk[0]}-{bk[1]}")
        mr = m.get("ratio") if isinstance(m, dict) else None
        out["stop"][label(bk)] = {"plain": sp / sy, "weighted": swp / swy, "n2a0_m1": mr,
                                  "stop_events": sy}
        print(f"   {label(bk):<22}{sp / sy:>7.3f} {swp / swy:>9.3f} "
              f"{(mr if mr is not None else float('nan')):>9.3f}")
    sp, sy, swp, swy = more
    out["another_parameter"] = {"plain": sp / sy if sy else None,
                                "weighted": swp / swy if swy else None, "events": sy}
    print(f"   ANOTHER PARAMETER ('&' at a value's end): plain {sp / sy:.3f}  weighted "
          f"{swp / swy:.3f}  ({sy:,.0f} events)")
    dest = ROOT / "data" / "e8" / f"weighted_calibration_{cls}.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
