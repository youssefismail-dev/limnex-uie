"""
research/v2/v2_p1b_audit.py

V2-P1b's audit at full size (PREREG_V2_P1B.md section 3), laptop CPU, inference only:
`v2_mem.batch_coupling_check` on real validation rows of six lengths, for
  - e16_cap (the baseline network: LayerNorm and dropout only),
  - p1_mem's final.pt (P1's query BatchNorm: the coupling that stopped P1),
  - P1b's start at full size (e16_cap's start + the memory with the query LayerNorm).

    python research/v2/v2_p1b_audit.py      # -> data/v2/p1b/audit.json
"""
from __future__ import annotations

import gc
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import v2_p1_common as P1  # noqa: E402

OUT = ROOT / "data" / "v2" / "p1b"


def main() -> None:
    import torch
    import v2_mem as M
    from uie.kaggle_train import new_model_e12
    from uie.model import ModelConfig
    from uie.train import load_checkpoint
    import e16_common as E16
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    e16, tok, _ = load_checkpoint(ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt")
    with gzip.open(ROOT / "data" / "corpus2" / "structure_val.jsonl.gz", "rt", encoding="utf-8") as fh:
        urls = [json.loads(next(fh))["url"] for _ in range(400)]
    enc = [tok.encode(u, "benign", frame="page") for u in urls]
    rows, seen = [], set()
    for want in (12, 30, 50, 80, 120, 200):                 # six lengths, as batches mix them
        r = min((x for x in enc if len(x) - 1 not in seen), key=lambda x: abs(len(x) - want))
        seen.add(len(r) - 1)
        rows.append(r[:-1])
    out = {"row_lengths": [len(r) for r in rows]}
    out["e16_cap"] = M.batch_coupling_check(e16, rows, tok.pad_id)
    del e16
    gc.collect()
    p1, _, _ = M.load_checkpoint(ROOT / "data" / "v2" / "p1" / "train" / P1.ARM / "final.pt")
    out["p1_mem_batchnorm"] = M.batch_coupling_check(p1, rows, tok.pad_id)
    del p1
    gc.collect()
    r = P1.recipe(Path(P1.STREAM_FILE))
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=r["d_model"], n_layers=r["n_layers"],
                      n_heads=r["n_heads"], d_ff=4 * r["d_model"], max_len=r["max_len"])
    base, _ = new_model_e12(cfg, tok, E16.SEED)
    fixed, _ = M.with_memory(base, P1.P1_SEED, tok.pad_id, mem_query_norm="layer", mem_branch_dropout=False)
    out["p1b_start_layernorm"] = M.batch_coupling_check(fixed, rows, tok.pad_id)
    (OUT / "audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
