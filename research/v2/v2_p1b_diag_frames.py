"""
research/v2/v2_p1b_diag_frames.py

V2-P1b, after the stop (post hoc; it decides nothing): where K-used's benign label loss
comes from. Inference only, on the CPU.

K-used (v2_p1_fit) put p1b_mem 0.139 nats/char WORSE than e16_cap on benign labels. The
per-field table that training printed on Kaggle put it 0.044 BETTER. The two differ in three
ways: the rows (training: the first 3,000 name_val rows per class; K-used: 2,000 random rows
from each corpus), the frame (training: "domain" for every row; K-used: each corpus's own)
and the device (the Kaggle GPU; the laptop CPU). Three readings separate them:

    A  K-used's own rows, encoding and per-row NLL, split by corpus (name rows in the domain
       frame; structure rows in the page frame), with the ablation per field
    B  training's instrument (uie.evaluation.field_nll) on training's rows, on this CPU:
       if it reproduces the Kaggle table, the device is not the cause
    C  the page frame on TRAINING rows: 2,000 structure_train rows per class, against
       A's structure_val rows -- what the memory does on sites it trained on

    python research/v2/v2_p1b_diag_frames.py      # -> data/v2/p1b/diag_frames.json
"""
from __future__ import annotations

import gzip
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import v2_p1b_common as P1  # noqa: E402
import v2_p1_fit as FIT  # noqa: E402

DATA = ROOT / "data" / "v2" / "p1b"
CK = DATA / "train" / P1.ARM / "final.pt"
TRAIN_ROWS = 2_000
TRAIN_SEED = 20261054                  # C's draw; not used by any registered reading
DIAG_BOOT = 20261055
FIRST = 3_000                          # training's field_nll sample (uie.kaggle_train)
STREAMS = {"benign": ("name_val", "structure_val", "structure_train"),
           "phishing": ("phishing_name_val", "phishing_structure_val", "phishing_structure_train")}


def urls_of(stem: str) -> list:
    with gzip.open(ROOT / "data" / "corpus2" / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


def by_field(a, b, seed: int) -> dict:
    out = {"all": FIT.pooled(a, b, lambda f: True, seed)}
    for k, f in enumerate(FIT.FIELDS):
        out[f] = FIT.pooled(a, b, lambda g, f=f: g == f, seed + 1 + k)
    return {k: v for k, v in out.items() if v}


def show(title: str, node: dict) -> None:
    print(f"  {title}", flush=True)
    for f, v in node.items():
        print(f"    {f:<4} e16_cap {v['a']:.4f}  p1b_mem {v['b']:.4f}  diff {v['diff']:+.4f} "
              f"[{v['lo']:+.4f}, {v['hi']:+.4f}]  ({v['chars']:,} chars)", flush=True)


def main() -> None:
    import torch
    import v2_mem as M
    from uie.evaluation import field_nll
    from uie.train import encode_corpus, load_checkpoint
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    p1b, tok, cfg = M.load_checkpoint(CK)
    e16, tok16, _ = load_checkpoint(FIT.E16_CAP)
    if tok16.state_dict() != tok.state_dict():
        raise SystemExit("STOP: the tokenizers differ")
    chars = {tok.stoi[c] for c in tok._chars}
    out = {"post_hoc": True, "decides": "nothing", "A": {}, "B": {}, "C": {}}

    # A: K-used's rows (the registered draw), split by corpus
    data = FIT.rows()
    for li, label in enumerate(("benign", "phishing")):
        for ci, corpus in enumerate(("name", "structure")):
            urls = [u for lab, c, u in data if lab == label and c == corpus]
            enc, _ = encode_corpus(urls, [label] * len(urls), tok, cfg.max_len, ["und"] * len(urls),
                                   [corpus] * len(urls))
            seqs = [ids for ids, _ in enc[label]]
            a = FIT.per_row_nll(e16, seqs, tok, chars)
            b = FIT.per_row_nll(p1b, seqs, tok, chars)
            p1b.memory.ablate = True
            c = FIT.per_row_nll(p1b, seqs, tok, chars)
            p1b.memory.ablate = False
            seed = DIAG_BOOT + 1000 * li + 100 * ci
            node = {"rows": len(seqs), "p1b_minus_e16": by_field(a, b, seed),
                    "ablation": by_field(b, c, seed + 50)}
            out["A"][f"{label} {corpus}"] = node
            show(f"A {label} {corpus} ({len(seqs):,} rows; {time.time() - t0:.0f}s)", node["p1b_minus_e16"])
            print("    ablation (memory off minus on): "
                  + ", ".join(f"{f} {v['diff']:+.3f}" for f, v in node["ablation"].items()), flush=True)

    # B: training's instrument on training's rows, on this CPU
    cpu = torch.device("cpu")
    for label in ("benign", "phishing"):
        first = urls_of(STREAMS[label][0])[:FIRST]
        node = {"e16_cap": field_nll(e16, tok, first, label, cpu),
                "p1b_mem": field_nll(p1b, tok, first, label, cpu)}
        out["B"][label] = node
        print(f"  B {label}: " + ", ".join(f"{f} {node['p1b_mem'][f] - node['e16_cap'][f]:+.4f}"
                                          for f in node["p1b_mem"] if f in node["e16_cap"])
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    # C: the page frame on rows the models trained on
    for li, label in enumerate(("benign", "phishing")):
        pool = urls_of(STREAMS[label][2])
        urls = random.Random(TRAIN_SEED + li).sample(pool, TRAIN_ROWS)
        del pool
        enc, _ = encode_corpus(urls, [label] * len(urls), tok, cfg.max_len, ["und"] * len(urls),
                               ["structure"] * len(urls))
        seqs = [ids for ids, _ in enc[label]]
        a = FIT.per_row_nll(e16, seqs, tok, chars)
        b = FIT.per_row_nll(p1b, seqs, tok, chars)
        node = {"rows": len(seqs), "p1b_minus_e16": by_field(a, b, DIAG_BOOT + 5000 + 100 * li)}
        out["C"][label] = node
        show(f"C {label} structure_train ({len(seqs):,} rows; {time.time() - t0:.0f}s)", node["p1b_minus_e16"])

    out["seconds"] = round(time.time() - t0, 1)
    (DATA / "diag_frames.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {DATA / 'diag_frames.json'} ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
