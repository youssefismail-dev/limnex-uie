"""E9 design evidence -- how much of the training signal each field receives now,
and under a length-neutral loss. READ-ONLY, corpus only; no model, no training.

    python research/native/e9_evidence_signal_share.py

The current loss gives every drawn row the same total weight whatever its length,
so a field's share of the signal is E_draw[field tokens / (L - 1)]. A
length-neutral loss gives every scored token the same weight, so the share
becomes E_draw[field tokens] / E_draw[L - 1]. Draws follow E7's sampler: class c
with probability proportional to its name count (class_alpha 1.0), rows uniform
within the class, both frames. Rows over max_len are dropped, as in training.
Field tokens are the field's marker plus its body (the same tokens `field_mask`
scores).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from research.qry.q7_hazard import field_spans  # noqa: E402
from uie.kaggle_train import STREAMS, load_streams  # noqa: E402
from uie.train import CORPUS_FRAME, load_checkpoint  # noqa: E402

FIELDS_OF_INTEREST = ("QRY", "PTH")


def main() -> None:
    _, tok, cfg = load_checkpoint(ROOT / "data" / "best_e7.pt", device=torch.device("cpu"))
    urls, labels, _, corpora, label_counts = load_streams(ROOT / "data" / "bundle2", STREAMS)
    total = sum(label_counts.values())
    pi = {c: n / total for c, n in label_counts.items()}
    acc = {c: collections.defaultdict(float) for c in pi}
    for u, c, corp in zip(urls, labels, corpora):
        ids = tok.encode(u, c, frame=CORPUS_FRAME[corp])
        if len(ids) > cfg.max_len:
            continue
        a = acc[c]
        scored = len(ids) - 1
        a["rows"] += 1
        a["scored"] += scored
        a["qrow"] += any(True for _ in field_spans(ids, tok, "QRY"))
        for f in FIELDS_OF_INTEREST:
            n = sum(e - s + 1 for s, e in field_spans(ids, tok, f))      # marker + body
            a[f"{f}_tokens"] += n
            a[f"{f}_old"] += n / scored
    out = {"status": "READ-ONLY design evidence for E9", "class_draw_probability": pi}
    for f in FIELDS_OF_INTEREST:
        old = sum(pi[c] * acc[c][f"{f}_old"] / acc[c]["rows"] for c in pi)
        new = (sum(pi[c] * acc[c][f"{f}_tokens"] / acc[c]["rows"] for c in pi)
               / sum(pi[c] * acc[c]["scored"] / acc[c]["rows"] for c in pi))
        out[f] = {"share_now": old, "share_length_neutral": new, "ratio": new / old}
        print(f"{f}: share of the training signal now {old:.4f}; length-neutral {new:.4f} "
              f"(x{new / old:.2f})")
    for c in pi:
        a = acc[c]
        out[c] = {"rows": a["rows"], "query_rows": a["qrow"],
                  "mean_scored_tokens_per_row": a["scored"] / a["rows"]}
        print(f"{c}: {a['rows']:,.0f} rows, {a['qrow']:,.0f} with a query "
              f"({a['qrow'] / a['rows']:.2%}); mean scored tokens per row "
              f"{a['scored'] / a['rows']:.2f}")
    mbar = sum(pi[c] * acc[c]["scored"] / acc[c]["rows"] for c in pi)
    out["expected_scored_tokens_per_draw"] = mbar
    print(f"expected scored tokens per draw (E7's sampler): {mbar:.3f}")
    dest = ROOT / "data" / "native" / "e9_signal_share.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
