"""
research/v2/v2_p1_diag_bn.py

V2-P1, post hoc -- NOT a registered reading; it decides nothing.

Why it exists: p1_mem's training record shows its evaluation-mode loss tracking e16_cap's
for 6,000 steps, then swinging upward to 4.32 (benign) against e16_cap's 1.61, worst on
paths, with training rows as bad as validation rows. The hypothesis: the memory's query
BatchNorm. Training batches are grouped by length, so its per-batch statistics swing
between batch types (short host-only rows against long path rows). In evaluation it uses a
running average that matches neither.

    python research/v2/v2_p1_diag_bn.py      # -> data/v2/p1/diag_bn.json

The same validation rows (1,024 per class: half name frame, half page frame, seed
20261052), per-character NLL over content characters, by field, under:
  A. evaluation mode, as registered (BatchNorm's running statistics);
  B. BatchNorm on batch statistics, random batches of 64;
  C. BatchNorm on batch statistics, length-grouped batches of 64 (as training groups them);
  D. evaluation mode with the running statistics re-estimated on 4,096 random training-like
     rows (a global mean and variance);
and e16_cap on the same rows for reference. Everything else stays in evaluation mode (no
dropout); the BatchNorm buffers are restored after B, C and D.
"""
from __future__ import annotations

import copy
import gzip
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from v2_arm import P1  # noqa: E402  (V2_ARM: p1 or p1b)
import v2_p1_fit as F  # noqa: E402

from v2_arm import DATA as P1D  # noqa: E402
SEED = 20261052
PER = 512
BATCH = 64


def rows(label: str, rng: random.Random):
    out = []
    for corpus, stem in (("name", "name_val"), ("structure", "structure_val")):
        stem = stem if label == "benign" else f"phishing_{stem}"
        with gzip.open(ROOT / "data" / "corpus2" / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
            urls = [json.loads(line)["url"] for line in fh]
        out += [(corpus, u) for u in rng.sample(urls, PER)]
    return out


def nll(model, seqs, tok, chars, batches):
    """{field: [sum, n]} over the given batches (lists of row indices)."""
    import torch
    import torch.nn.functional as Fn
    acc: dict = {}
    with torch.no_grad():
        for idx in batches:
            s = [seqs[i] for i in idx]
            width = max(len(x) for x in s)
            ids = torch.tensor([x + [tok.pad_id] * (width - len(x)) for x in s])
            logp = Fn.log_softmax(model(ids[:, :-1]).float(), dim=-1)
            got = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            for r, i in enumerate(idx):
                for pos, f in enumerate(F.field_targets(seqs[i], tok, chars)):
                    if f is not None:
                        a = acc.setdefault(f, [0.0, 0])
                        a[0] -= float(got[r, pos])
                        a[1] += 1
    return acc


def summary(acc: dict) -> dict:
    tot = [sum(v[0] for v in acc.values()), sum(v[1] for v in acc.values())]
    out = {f: v[0] / v[1] for f, v in sorted(acc.items())}
    out["all"] = tot[0] / tot[1]
    return out


def main() -> None:
    import torch
    import v2_mem as M
    from uie.train import encode_corpus, load_checkpoint
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    t0 = time.time()
    p1, tok, cfg = M.load_checkpoint(P1D / "train" / P1.ARM / "final.pt")
    e16, _, _ = load_checkpoint(ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt")
    p1.eval()
    e16.eval()
    bn = p1.memory.query_bn
    saved = copy.deepcopy(bn.state_dict())
    chars = {tok.stoi[c] for c in tok._chars}
    rng = random.Random(SEED)
    out: dict = {"seed": SEED, "rows_per_class": 2 * PER, "batch": BATCH,
                 "running_mean_abs": float(saved["running_mean"].abs().mean()),
                 "running_var_mean": float(saved["running_var"].mean()), "classes": {}}
    for label in ("benign", "phishing"):
        data = rows(label, rng)
        enc, _ = encode_corpus([u for _, u in data], [label] * len(data), tok, cfg.max_len,
                               ["und"] * len(data), [c for c, _ in data])
        seqs = [ids for ids, _ in enc[label]]
        order = list(range(len(seqs)))
        rng.shuffle(order)
        random_b = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
        by_len = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        grouped_b = [by_len[i:i + BATCH] for i in range(0, len(by_len), BATCH)]
        node = {"rows": len(seqs), "e16_cap": summary(nll(e16, seqs, tok, chars, random_b)),
                "A_eval_running_stats": summary(nll(p1, seqs, tok, chars, random_b))}
        bn.train()                                    # only the BatchNorm uses batch statistics
        node["B_batch_stats_random"] = summary(nll(p1, seqs, tok, chars, random_b))
        bn.load_state_dict(saved)
        node["C_batch_stats_length_grouped"] = summary(nll(p1, seqs, tok, chars, grouped_b))
        bn.load_state_dict(saved)
        # D: re-estimate the running statistics as a plain average over random batches
        bn.reset_running_stats()
        bn.momentum = None                            # cumulative average
        nll(p1, seqs, tok, chars, random_b)
        bn.eval()
        node["D_eval_recalibrated"] = summary(nll(p1, seqs, tok, chars, random_b))
        bn.momentum = 0.1
        bn.load_state_dict(saved)
        out["classes"][label] = node
        print(f"  {label}: all-content NLL  e16_cap {node['e16_cap']['all']:.3f} | p1 A (eval) "
              f"{node['A_eval_running_stats']['all']:.3f} | B (batch stats, random) "
              f"{node['B_batch_stats_random']['all']:.3f} | C (batch stats, length-grouped) "
              f"{node['C_batch_stats_length_grouped']['all']:.3f} | D (recalibrated) "
              f"{node['D_eval_recalibrated']['all']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        for f in ("LBL", "SUB", "PTH", "QRY"):
            if f in node["e16_cap"]:
                print(f"     {f}: e16_cap {node['e16_cap'][f]:.3f}  A {node['A_eval_running_stats'][f]:.3f}  "
                      f"B {node['B_batch_stats_random'][f]:.3f}  C {node['C_batch_stats_length_grouped'][f]:.3f}  "
                      f"D {node['D_eval_recalibrated'][f]:.3f}", flush=True)
    (P1D / "diag_bn.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", P1D / "diag_bn.json")


if __name__ == "__main__":
    main()
