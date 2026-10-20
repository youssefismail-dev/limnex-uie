"""Does the training objective itself pull presence decisions toward "absent"?
[DIAGNOSTIC], post hoc (after E8's K1b reading); measurement only, no training.

    python research/native/e8_length_weighting.py --arm e8_pfrs

Mechanism under test [HYPOTHESIS until this reads]:
- `URLTransformer.loss` divides the batch's summed cross-entropy by the batch's
  total scored-token count (uie/model.py);
- `uie.train.batches` sorts each window of rows by length before cutting batches,
  so a batch holds rows of nearly one length L.
- A row's targets therefore carry weight ~ 1 / (256 * (L - 1)). At a decision
  whose options lead to rows of different lengths, the loss optimum is the
  1/(L-1)-WEIGHTED rate, not the rate. Absent fields make shorter rows, so the
  weighted rate sits below the plain one.

The test: for each shipped suffix, the plan arm's own P(P:1 | s) and its free
Q (computed exactly as in e8_plan_calibration.py) set against the plain
training rate and the 1/(L-1)-weighted training rate, with L each row's token
length under the arm's own tokenizer, domain frame. If the model tracks the
weighted rate and not the plain one, the objective explains the miscalibration.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from e8_plan_calibration import plan_by_suffix, rows  # noqa: E402
from uie.factorization import parse  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402


def e7_mode(label_counts_only=False) -> dict:
    """E7 has no plan: its free-mode presence is decided implicitly. Here only the
    training side is computed -- plain and 1/(L-1)-weighted presence rates under
    E7's own tokenizer (lengths), shipped-weighted -- to set against E7's measured
    free emission (score_misc.json)."""
    from uie.evaluation import FIELDS
    _, tok, cfg = load_checkpoint(ROOT / "data" / "best_e7.pt", device=torch.device("cpu"))
    shipped = SuffixTable.load(ROOT / "data" / "bundle2" / "suffix_table.json").counts
    misc = json.loads((ROOT / "data" / "e8" / "score_misc.json").read_text(encoding="utf-8"))
    out = {}
    for label in ("benign", "phishing"):
        plain = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
        weighted = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
        for u in rows(label):
            ids = tok.encode(u, label, frame="domain")
            if len(ids) > cfg.max_len:
                continue
            s = parse(u).public_suffix
            P, Q = bool(FIELDS["PTH"](u)), bool(FIELDS["QRY"](u))
            w = 1.0 / (len(ids) - 1)
            for acc, x in ((plain[s], 1.0), (weighted[s], w)):
                acc[0] += x
                acc[1] += x * P
                acc[2] += x * Q
        weights = shipped[label]
        wsum = sum(weights.values())
        tot = {"train_P": 0.0, "weighted_P": 0.0, "train_Q": 0.0, "weighted_Q": 0.0}
        for s, c in weights.items():
            if plain[s][0]:
                tot["train_P"] += c / wsum * plain[s][1] / plain[s][0]
                tot["weighted_P"] += c / wsum * weighted[s][1] / weighted[s][0]
                tot["train_Q"] += c / wsum * plain[s][2] / plain[s][0]
                tot["weighted_Q"] += c / wsum * weighted[s][2] / weighted[s][0]
        em = misc["emission"]["e7"][label]
        tot["e7_free_P"], tot["e7_free_Q"] = em["PTH"]["rate"], em["QRY"]["rate"]
        out[label] = tot
        print(f"== E7 {label}: P train {tot['train_P']:.4f} weighted {tot['weighted_P']:.4f} "
              f"E7 free {tot['e7_free_P']:.4f} | Q train {tot['train_Q']:.4f} weighted "
              f"{tot['weighted_Q']:.4f} E7 free {tot['e7_free_Q']:.4f}")
    dest = ROOT / "data" / "e8" / "length_weighting_e7.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", dest)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="e8_pfrs")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--e7", action="store_true", help="E7: training side only")
    args = ap.parse_args()
    if args.e7:
        e7_mode()
        return
    model, tok, cfg = load_checkpoint(ROOT / "data" / "e8" / "train" / args.arm / "best.pt",
                                      device=torch.device("cpu"))
    model.eval()
    (p0, p1), (q0, q1) = tok.plan_slots()
    shipped = SuffixTable.load(ROOT / "data" / "bundle2" / "suffix_table.json").counts
    report = {"arm": args.arm, "status": "DIAGNOSTIC, post hoc"}
    for label in ("benign", "phishing"):
        plain = collections.defaultdict(lambda: [0.0, 0.0, 0.0])      # rows, P1, Q1
        weighted = collections.defaultdict(lambda: [0.0, 0.0, 0.0])   # sum w, w*P1, w*Q1
        for u in rows(label):
            ids = tok.encode(u, label, frame="domain")
            if len(ids) > cfg.max_len:
                continue                                  # dropped in training too
            s = parse(u).public_suffix
            k = next(j for j, t in enumerate(ids) if t in (p0, p1))
            P, Q = ids[k] == p1, ids[k + 1] == q1
            w = 1.0 / (len(ids) - 1)
            for acc, x in ((plain[s], 1.0), (weighted[s], w)):
                acc[0] += x
                acc[1] += x * P
                acc[2] += x * Q
        weights = shipped[label]
        wsum = sum(weights.values())
        model_plan = plan_by_suffix(model, tok, label, sorted(weights))
        tot = {k: 0.0 for k in ("train_P", "weighted_P", "model_P", "train_Q", "weighted_Q",
                                "model_Q")}
        for s, c in weights.items():
            if not plain[s][0]:
                continue
            share = c / wsum
            tot["train_P"] += share * plain[s][1] / plain[s][0]
            tot["weighted_P"] += share * weighted[s][1] / weighted[s][0]
            tot["model_P"] += share * model_plan[s]["p"]
            tot["train_Q"] += share * plain[s][2] / plain[s][0]
            tot["weighted_Q"] += share * weighted[s][2] / weighted[s][0]
            tot["model_Q"] += share * model_plan[s]["q"]
        top = sorted(weights, key=lambda s: -weights[s])[:args.top]
        per = {s: {"share": weights[s] / wsum, "rows": plain[s][0],
                   "train_P": plain[s][1] / plain[s][0],
                   "weighted_P": weighted[s][1] / weighted[s][0], "model_P": model_plan[s]["p"],
                   "train_Q": plain[s][2] / plain[s][0],
                   "weighted_Q": weighted[s][2] / weighted[s][0], "model_Q": model_plan[s]["q"]}
               for s in top if plain[s][0]}
        report[label] = {"shipped_weighted": tot, "top_suffixes": per}
        print(f"== {label} (shipped-weighted)")
        print(f"   P: train {tot['train_P']:.4f}   1/L-weighted {tot['weighted_P']:.4f}   "
              f"model {tot['model_P']:.4f}")
        print(f"   Q: train {tot['train_Q']:.4f}   1/L-weighted {tot['weighted_Q']:.4f}   "
              f"model {tot['model_Q']:.4f}")
        for s, v in per.items():
            print(f"     {s:<8} P train {v['train_P']:.3f} weighted {v['weighted_P']:.3f} model "
                  f"{v['model_P']:.3f}   Q train {v['train_Q']:.4f} weighted "
                  f"{v['weighted_Q']:.4f} model {v['model_Q']:.4f}")
    dest = ROOT / "data" / "e8" / f"length_weighting_{args.arm}.json"
    dest.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
