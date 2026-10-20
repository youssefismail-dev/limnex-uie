"""E8 -- where does free-mode emission come from? [DIAGNOSTIC], post hoc: written
after K1b's reading (score_misc.json) to check that the reading is not an
artefact. It changes no criterion and no registered number.

    python research/native/e8_plan_calibration.py --arm e8_pfrs

In FULL FREE a plan arm fills its plan right after the suffix, from its own
distribution renormalised over each slot's two legal tokens
(uie/model.py, `_generate_chunk`). That distribution depends only on the class,
the frame and the suffix, so for every suffix s of the shipped table it is ONE
number, computed here exactly with a forward pass:

    q_model(s) = sum_p P(p | s) * P(Q:1 | s, p)     (each slot renormalised)
    p_model(s) = P(P:1 | s)

Free mode's expected emission is then sum_s shipped(s) * q_model(s), to set
against K1b's target sum_s shipped(s) * P_train(QRY | s) and against what the
generated files actually carry. Also checked: that the plan LABELS the arm was
trained on (the tokenizer's presence) equal the gate's FIELDS presence on
name_train rows.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from uie.evaluation import FIELDS  # noqa: E402
from uie.factorization import parse  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402

STEM = {"benign": "name_train", "phishing": "phishing_name_train"}


def rows(label):
    with gzip.open(ROOT / "data" / "bundle2" / f"{STEM[label]}.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)["url"]


@torch.no_grad()
def last_probs(model, seqs):
    """Next-token distribution after each sequence (right-padded; causal)."""
    width = max(len(s) for s in seqs)
    x = torch.tensor([s + [0] * (width - len(s)) for s in seqs], dtype=torch.long)
    logits = model(x).float()
    idx = torch.tensor([len(s) - 1 for s in seqs])
    return F.softmax(logits[torch.arange(len(seqs)), idx], dim=-1)


def plan_by_suffix(model, tok, label, suffixes, batch=256):
    (p0, p1), (q0, q1) = tok.plan_slots()
    out = {}
    for b in range(0, len(suffixes), batch):
        chunk = suffixes[b:b + batch]
        prefixes = []
        for s in chunk:
            ids = tok.encode(f"x.{s}/", label, frame="domain")
            k = next(j for j, t in enumerate(ids) if t in (p0, p1))
            prefixes.append(ids[:k])                      # ends with the suffix
        pp = last_probs(model, prefixes)
        pq = {v: last_probs(model, [p + [v] for p in prefixes]) for v in (p0, p1)}
        for i, s in enumerate(chunk):
            mp = float(pp[i, p0] + pp[i, p1])
            pP1 = float(pp[i, p1]) / mp
            qc = {v: float(pq[v][i, q1]) / float(pq[v][i, q0] + pq[v][i, q1]) for v in (p0, p1)}
            out[s] = {"p": pP1, "q": (1 - pP1) * qc[p0] + pP1 * qc[p1], "legal_mass_P": mp,
                      "q_given_p0": qc[p0], "q_given_p1": qc[p1]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="e8_pfrs")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    model, tok, _ = load_checkpoint(ROOT / "data" / "e8" / "train" / args.arm / "best.pt",
                                    device=torch.device("cpu"))
    model.eval()
    assert tok.presence_plan, "not a plan arm"
    shipped = SuffixTable.load(ROOT / "data" / "bundle2" / "suffix_table.json").counts
    gen_dir = ROOT / "data" / "e8" / "gen"
    report = {"arm": args.arm, "status": "DIAGNOSTIC, post hoc"}
    for label in ("benign", "phishing"):
        # training rates per suffix (gate's FIELDS) and the plan labels (tokenizer)
        n = collections.Counter()
        has = {"PTH": collections.Counter(), "QRY": collections.Counter()}
        lab = {"PTH": 0, "QRY": 0}
        disagree = {"PTH": 0, "QRY": 0}
        (p0, p1), (q0, q1) = tok.plan_slots()
        total = 0
        for u in rows(label):
            s = parse(u).public_suffix
            n[s] += 1
            total += 1
            ids = tok.encode(u, label, frame="domain")
            k = next(j for j, t in enumerate(ids) if t in (p0, p1))
            plan = {"PTH": ids[k] == p1, "QRY": ids[k + 1] == q1}
            for f in ("PTH", "QRY"):
                g = bool(FIELDS[f](u))
                has[f][s] += g
                lab[f] += plan[f]
                disagree[f] += plan[f] != g
        weights = {s: c for s, c in shipped[label].items()}
        wsum = sum(weights.values())
        model_plan = plan_by_suffix(model, tok, label, sorted(weights))
        target = {f: sum(weights[s] * has[f][s] / n[s] for s in weights if n[s]) / wsum
                  for f in ("PTH", "QRY")}
        expected = {"PTH": sum(weights[s] * model_plan[s]["p"] for s in weights) / wsum,
                    "QRY": sum(weights[s] * model_plan[s]["q"] for s in weights) / wsum}
        legal = sum(weights[s] * model_plan[s]["legal_mass_P"] for s in weights) / wsum
        gen = [u for p in sorted(gen_dir.glob(f"free_{args.arm}_{label}_g*.json"))
               for u in json.loads(p.read_text(encoding="utf-8"))]
        realised = {f: sum(bool(FIELDS[f](u)) for u in gen) / len(gen) for f in ("PTH", "QRY")}
        top = sorted(weights, key=lambda s: -weights[s])[:12]
        report[label] = {
            "rows": total,
            "plan_labels_vs_gate": {f: {"label_rate": lab[f] / total,
                                        "gate_rate": sum(has[f].values()) / total,
                                        "rows_disagreeing": disagree[f]} for f in lab},
            "shipped_weighted": {"target_train": target, "model_expected": expected,
                                 "realised_in_generated_files": realised,
                                 "legal_mass_P_slot": legal, "generated_urls": len(gen)},
            "top_suffixes": {s: {"shipped_share": weights[s] / wsum,
                                 "train_P": has["PTH"][s] / n[s], "model_P": model_plan[s]["p"],
                                 "train_Q": has["QRY"][s] / n[s], "model_Q": model_plan[s]["q"],
                                 "legal_mass_P": model_plan[s]["legal_mass_P"]}
                             for s in top if n[s]},
        }
        r = report[label]
        print(f"== {label}: {total:,} name_train rows")
        for f in ("PTH", "QRY"):
            pl = r["plan_labels_vs_gate"][f]
            print(f"  {f}: plan labels {pl['label_rate']:.4f} vs gate {pl['gate_rate']:.4f} "
                  f"({pl['rows_disagreeing']:,} rows disagree) | shipped-weighted: target "
                  f"{target[f]:.4f}, model's own plan {expected[f]:.4f}, generated files "
                  f"{realised[f]:.4f}")
        print(f"  legal mass at the P slot (shipped-weighted): {legal:.4f}")
        for s, v in r["top_suffixes"].items():
            print(f"    {s:<10} share {v['shipped_share']:.3f}  P train {v['train_P']:.3f} model "
                  f"{v['model_P']:.3f}   Q train {v['train_Q']:.4f} model {v['model_Q']:.4f}   "
                  f"legal {v['legal_mass_P']:.3f}")
    dest = args.out or ROOT / "data" / "e8" / f"plan_calibration_{args.arm}.json"
    dest.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
