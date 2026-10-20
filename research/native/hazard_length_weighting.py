"""Does the length weighting of the training objective predict E7's query-value
termination hazard? [DIAGNOSTIC], post hoc; corpus only -- no model, no training.

    python research/native/hazard_length_weighting.py

The training loss gives a row weight ~ 1 / (L - 1) (see e8_length_weighting.py).
At a context the loss optimum for "the value ends here" is then the WEIGHTED stop
rate of the rows sharing that context, and rows that stop are shorter than rows
that continue. An idealised model that conditions on its position is
approximated by strata of (absolute position // 4, position within the value):
    predicted h(pos) = sum_stratum w * stop / sum_stratum w.
The predicted model/corpus ratio per bucket, sum predicted h / sum stops, is set
against E7's MEASURED ratio (N2a-0's M1, the same rows: the 20,000-row training
sample, Q7's buckets and event definition).
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from n2a_reference import SAMPLE_SEED, STREAMS, TRAIN_ROWS, rows  # noqa: E402
from research.qry.q7_hazard import BUCKETS, bucket_of, field_spans, label  # noqa: E402
from uie.evaluation import FIELDS  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402


def main() -> None:
    _, tok, cfg = load_checkpoint(ROOT / "data" / "best_e7.pt", device=torch.device("cpu"))
    amp = tok.stoi.get("&")
    out = {"status": "DIAGNOSTIC, post hoc", "checkpoint": "best_e7.pt (tokenizer only)"}
    for cls in ("benign", "phishing"):
        train_stem = STREAMS[cls][0]
        train_q = [u for u in rows(train_stem) if FIELDS["QRY"](u)]
        train_q = random.Random(SAMPLE_SEED).sample(train_q, min(TRAIN_ROWS, len(train_q)))
        events = []                                   # (stratum, bucket, stop, w)
        for u in train_q:
            ids = tok.encode(u, cls, frame="domain", hide_fields=("PTH", "QRY"))
            if len(ids) > cfg.max_len:
                continue
            w = 1.0 / (len(ids) - 1)
            for s, e in field_spans(ids, tok, "QRY"):
                pos = s
                while pos < e:
                    end = pos
                    while end < e and ids[end] != amp:
                        end += 1
                    for i in range(pos, end):
                        within = i - pos
                        events.append(((i // 4, within), bucket_of(within),
                                       1 if i + 1 >= end else 0, w))
                    pos = end + 1
        num = collections.defaultdict(float)
        den = collections.defaultdict(float)
        for st, _, stop, w in events:
            num[st] += w * stop
            den[st] += w
        pred = collections.defaultdict(float)
        emp = collections.defaultdict(float)
        pooled_w = collections.defaultdict(lambda: [0.0, 0.0])
        pooled = collections.defaultdict(lambda: [0.0, 0.0])
        for st, bk, stop, w in events:
            pred[bk] += num[st] / den[st]
            emp[bk] += stop
            pooled_w[bk][0] += w * stop
            pooled_w[bk][1] += w
            pooled[bk][0] += stop
            pooled[bk][1] += 1
        measured = json.loads((ROOT / "data" / "native" / f"n2a_reference_{cls}.json")
                              .read_text(encoding="utf-8"))["train"]["M1_buckets"]
        res = {}
        print(f"== {cls}: {len(train_q):,} rows, {len(events):,} value positions")
        print(f"   {'bucket':<10}{'predicted':>10}{'pooled':>9}{'E7 measured':>13}")
        for bk in BUCKETS:
            lab = label(bk)
            if not emp[bk]:
                continue
            p = pred[bk] / emp[bk]
            q = (pooled_w[bk][0] / pooled_w[bk][1]) / (pooled[bk][0] / pooled[bk][1])
            m = measured.get(f"{bk[0]}-{bk[1]}") or measured.get(lab)
            mr = m.get("ratio") if isinstance(m, dict) else None
            res[lab] = {"predicted_stratified": p, "predicted_pooled": q, "e7_measured": mr,
                        "stop_events": emp[bk]}
            print(f"   {lab:<10}{p:>10.3f}{q:>9.3f}{(mr if mr is not None else float('nan')):>13.3f}")
        out[cls] = res
    dest = ROOT / "data" / "e8" / "hazard_length_weighting.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
