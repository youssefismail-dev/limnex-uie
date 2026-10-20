"""E11 amendment 2 -- the checkpoint-step check (research/native/PREREG_E11.md 3.2).
Post hoc, reported only; laptop.

    python research/native/e11_step_check.py

For each hidden-control arm, at best.pt and final.pt: the model's split at the end
of real hosts, teacher-forced in free mode's conditioning (domain frame, both
controls `?`). The host end is the first structural token after the host fields:
`<PTH>`, `<RS>`, `<QRY>`, `<FRG>` or the end. The split is renormalised over those
five, as QG2-D's snapshot was. The same 3,000 `name_train` rows per class (seeded,
rows that fit) are read by every checkpoint, so best against final is paired and
has no sampling error of its own. Writes data/e11/step_check.json.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "qgap"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import qg_common as Q  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402

N_ROWS = 3000
SEED = 20260930
ARMS = ("e11_rs", "e11_ref", "e11_joint")
ENDS = ("PTH", "RS", "QRY", "FRG", "EOS")


def end_ids(tok):
    return {"PTH": tok.marker_id("PTH"), "RS": tok.rs_id, "QRY": tok.marker_id("QRY"),
            "FRG": tok.marker_id("FRG"), "EOS": tok.eos_id}


def sample(tok, label, max_len):
    """Seeded rows of the domain-frame training stream that fit, with each row's
    host-end index (the first structural token after `<LBL>`)."""
    pool = list(Q.rows(Q.STREAMS[label][0]))
    order = random.Random(SEED).sample(range(len(pool)), len(pool))
    ends = set(end_ids(tok).values())
    lbl = tok.marker_id("LBL")
    out = []
    for i in order:
        seq = Q.encode_free(tok, pool[i], label)
        if seq is None or len(seq) > max_len or lbl not in seq:
            continue
        k = next((j for j in range(seq.index(lbl) + 1, len(seq)) if seq[j] in ends), None)
        if k is None:
            continue
        out.append((seq, k))
        if len(out) == N_ROWS:
            break
    return out


@torch.no_grad()
def shares(model, tok, rows, batch=64):
    ids = end_ids(tok)
    cols = torch.tensor([ids[e] for e in ENDS])
    model_sum = torch.zeros(len(ENDS), dtype=torch.float64)
    data = {e: 0 for e in ENDS}
    for b in range(0, len(rows), batch):
        chunk = rows[b:b + batch]
        width = max(k for _, k in chunk)                   # predict token k from k-1
        x = torch.tensor([s[:width] + [tok.pad_id] * (width - len(s[:width])) for s, _ in chunk])
        logits = model(x).float()
        at = torch.tensor([k - 1 for _, k in chunk])
        p = F.softmax(logits[torch.arange(len(chunk)), at], dim=-1)[:, cols].double()
        model_sum += (p / p.sum(dim=1, keepdim=True)).sum(dim=0)
        for s, k in chunk:
            data[next(e for e in ENDS if ids[e] == s[k])] += 1
    n = len(rows)
    return ({e: float(model_sum[i] / n) for i, e in enumerate(ENDS)},
            {e: data[e] / n for e in ENDS})


def main() -> None:
    torch.set_num_threads(4)
    report = {"amendment": "PREREG_E11.md 3.2", "rows_per_class": N_ROWS, "seed": SEED, "arms": {}}
    rows = {}
    for arm in ARMS:
        for ck in ("best", "final"):
            model, tok, cfg = load_checkpoint(ROOT / "data" / "e11" / "train" / arm / f"{ck}.pt",
                                              device=torch.device("cpu"))
            model.eval()
            for label in Q.LABELS:
                if label not in rows:
                    rows[label] = sample(tok, label, cfg.max_len)
                m, d = shares(model, tok, rows[label])
                report["arms"].setdefault(arm, {}).setdefault(ck, {})[label] = {"model": m, "data": d}
                print(f"  {arm:<10} {ck:<6} {label:<9} model PTH {100 * m['PTH']:.2f}%  RS "
                      f"{100 * m['RS']:.2f}%  QRY {100 * m['QRY']:.2f}%  EOS {100 * m['EOS']:.2f}%"
                      f"   | data PTH {100 * d['PTH']:.2f}%", flush=True)
    (ROOT / "data" / "e11" / "step_check.json").write_text(json.dumps(report, indent=1),
                                                          encoding="utf-8")
    print("wrote data/e11/step_check.json")


if __name__ == "__main__":
    main()
