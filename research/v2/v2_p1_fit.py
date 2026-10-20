"""
research/v2/v2_p1_fit.py

V2-P1 -- K-used (research/v2/PREREG_V2_P1_MEMORY.md sections 4.6 and 5.3): was the memory
used? Read at VERIFY_TRAIN, before any generation is spent.

    python research/v2/v2_p1_fit.py        # -> data/v2/p1/k_used.json

The rows: per class, 2,000 validation rows from the name frame and 2,000 from the page
frame (corpus2's *_val streams), drawn with seed 20261050 in the order benign name, benign
page, phishing name, phishing page. They are encoded exactly as training encodes rows
(`uie.train.encode_corpus`, with the frame); rows over max_len are dropped.

What is measured, per row and model: the NLL summed over its content characters (targets
that are characters, inside a field), by field.
- p1_mem against e16_cap, on the same rows: the pooled per-character difference with a
  paired row-bootstrap 95% interval (1,000 draws, seed 20261051).
- Slot usage: the share of the 262,144 slots chosen at least once by these rows, and the
  KL of the choice counts from uniform.
- The ablation, reported only: p1_mem with the memory's output at 0.

K-used holds iff benign's interval over every content character lies below 0 AND at least
10% of the slots are used.
"""
from __future__ import annotations

import gzip
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from v2_arm import P1  # noqa: E402  (V2_ARM: p1 or p1b)

from v2_arm import DATA as P1D  # noqa: E402
CK = P1D / "train" / P1.ARM / "final.pt"
E16_CAP = ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt"
STREAMS = (("benign", "name", "name_val"), ("benign", "structure", "structure_val"),
           ("phishing", "name", "phishing_name_val"), ("phishing", "structure", "phishing_structure_val"))
PER_STREAM = 2_000
BATCH = 32
DRAWS = 1_000
USAGE_MIN = 0.10
FIELDS = ("SUF", "LBL", "SUB", "PTH", "QRY", "FRG")


def rows():
    """[(label, corpus, url)] in the registered order, PER_STREAM per stream."""
    rng = random.Random(P1.P1_SEED)
    out = []
    for label, corpus, stem in STREAMS:
        with gzip.open(ROOT / "data" / "corpus2" / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
            urls = [json.loads(line)["url"] for line in fh]
        out += [(label, corpus, u) for u in rng.sample(urls, PER_STREAM)]
    return out


def field_targets(ids, tok, chars: set):
    """Per target position (aligned with ids[1:]): the field of a content character, else
    None. The field opens with its marker and runs until the next marker (uie.train.field_mask's
    rule)."""
    from uie.tokenizer import _MARKERS
    marker_of = {tok.marker_id(m): m for m in _MARKERS}
    out, current = [], None
    for target in ids[1:]:
        if target in marker_of:
            current = marker_of[target]
            out.append(None)
        elif target in chars and current is not None:
            out.append(current)
        else:
            out.append(None)
    return out


def per_row_nll(model, encoded, tok, chars, slot_counts=None):
    """[{field: (nll sum, n)}] per row, in order; `slot_counts` accumulates the memory's
    choices (p1_mem only)."""
    import torch
    import torch.nn.functional as F
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    res = [None] * len(encoded)
    model.eval()
    with torch.no_grad():
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            seqs = [encoded[i] for i in idx]
            width = max(len(s) for s in seqs)
            ids = torch.tensor([s + [tok.pad_id] * (width - len(s)) for s in seqs])
            if slot_counts is not None:
                model.memory.trace = []
            logp = F.log_softmax(model(ids[:, :-1]).float(), dim=-1)
            if slot_counts is not None:
                for t in model.memory.trace:
                    slot_counts += torch.bincount(t.flatten(), minlength=slot_counts.numel())
                model.memory.trace = None
            got = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            for r, i in enumerate(idx):
                fields = field_targets(encoded[i], tok, chars)
                node = {}
                for pos, f in enumerate(fields):
                    if f is None:
                        continue
                    s, n = node.get(f, (0.0, 0))
                    node[f] = (s - float(got[r, pos]), n + 1)
                res[i] = node
    return res


def pooled(rows_a, rows_b, keep, rng_seed):
    """(b - a) pooled per character over the rows, with a paired row-bootstrap 95% interval."""
    pairs = []
    for a, b in zip(rows_a, rows_b):
        sa = sum(v[0] for f, v in a.items() if keep(f))
        sb = sum(v[0] for f, v in b.items() if keep(f))
        n = sum(v[1] for f, v in a.items() if keep(f))
        if n:
            pairs.append((sb - sa, n, sa, sb))
    if not pairs:
        return None
    tot_n = sum(p[1] for p in pairs)
    diff = sum(p[0] for p in pairs) / tot_n
    rng = random.Random(rng_seed)
    boots = []
    for _ in range(DRAWS):
        pick = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        boots.append(sum(p[0] for p in pick) / sum(p[1] for p in pick))
    boots.sort()
    return {"diff": diff, "lo": boots[int(0.025 * DRAWS)], "hi": boots[int(0.975 * DRAWS) - 1],
            "a": sum(p[2] for p in pairs) / tot_n, "b": sum(p[3] for p in pairs) / tot_n,
            "rows": len(pairs), "chars": tot_n}


def main() -> None:
    import argparse
    import torch
    import v2_mem as M
    from uie.train import encode_corpus, load_checkpoint
    global CK, PER_STREAM, P1D
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, help="dry runs only: another p1_mem checkpoint")
    ap.add_argument("--rows", type=int, help="dry runs only: fewer rows per stream")
    args = ap.parse_args()
    if args.ckpt or args.rows:
        CK, PER_STREAM, P1D = args.ckpt or CK, args.rows or PER_STREAM, P1D / "dry"
        P1D.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    p1, tok, cfg = M.load_checkpoint(CK)
    e16, tok16, _ = load_checkpoint(E16_CAP)
    if tok16.state_dict() != tok.state_dict():
        raise SystemExit("STOP: p1_mem's tokenizer is not e16_cap's")
    chars = {tok.stoi[c] for c in tok._chars}
    data = rows()
    out = {"registration": P1.REGISTRATION, "rows_per_stream": PER_STREAM, "seed": P1.P1_SEED,
           "boot_seed": P1.BOOT_SEED, "classes": {}}
    slot_counts = torch.zeros(p1.memory.size, dtype=torch.long)
    for li, label in enumerate(("benign", "phishing")):
        urls = [u for lab, c, u in data if lab == label]
        corp = [c for lab, c, u in data if lab == label]
        enc, dropped = encode_corpus(urls, [label] * len(urls), tok, cfg.max_len, ["und"] * len(urls), corp)
        seqs = [ids for ids, _ in enc[label]]
        a = per_row_nll(e16, seqs, tok, chars)
        b = per_row_nll(p1, seqs, tok, chars, slot_counts)
        p1.memory.ablate = True
        c = per_row_nll(p1, seqs, tok, chars)
        p1.memory.ablate = False
        node = {"rows": len(seqs), "dropped_over_max_len": dropped,
                "all": pooled(a, b, lambda f: True, P1.BOOT_SEED + 100 * li),
                "by_field": {f: pooled(a, b, lambda g, f=f: g == f, P1.BOOT_SEED + 100 * li + 1 + k)
                             for k, f in enumerate(FIELDS)},
                "ablation_all": pooled(b, c, lambda f: True, P1.BOOT_SEED + 100 * li + 50)}
        out["classes"][label] = node
        x = node["all"]
        print(f"  {label}: {len(seqs):,} rows; p1_mem - e16_cap {x['diff']:+.4f} nats/char "
              f"[{x['lo']:+.4f}, {x['hi']:+.4f}] ({x['a']:.4f} -> {x['b']:.4f}); "
              f"ablation {node['ablation_all']['diff']:+.4f} ({time.time() - t0:.0f}s)", flush=True)
    used = int((slot_counts > 0).sum())
    p = slot_counts.double() / slot_counts.sum()
    nz = p[p > 0]
    kl = float((nz * (nz * p.numel()).log()).sum())
    out["usage"] = {"slots_used": used, "share": used / p1.memory.size, "kl_from_uniform": kl,
                    "selections": int(slot_counts.sum())}
    ben = out["classes"]["benign"]["all"]
    out["criteria"] = {"a_fit": ben["hi"] < 0, "b_usage": used / p1.memory.size >= USAGE_MIN}
    out["pass"] = all(out["criteria"].values())
    out["seconds"] = round(time.time() - t0, 1)
    (P1D / "k_used.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"  usage: {used:,} of {p1.memory.size:,} slots ({used / p1.memory.size:.1%}); KL from uniform {kl:.3f}")
    print(f"K-used: (a) benign fit {'PASS' if out['criteria']['a_fit'] else 'FAIL'}; "
          f"(b) usage {'PASS' if out['criteria']['b_usage'] else 'FAIL'} -> "
          f"{'PASS' if out['pass'] else 'FAIL: MEMORY UNUSED (no generation)'}")


if __name__ == "__main__":
    main()
