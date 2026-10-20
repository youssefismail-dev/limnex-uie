"""
research/v2/v2_piece_hazard.py

Do word-piece tables memorise whole sites? A post-hoc check on R1's piece tables, made for
the owner's question about the host (2026-10-13). It decides nothing. CPU only, no model.

For each table:
- the pieces that equal a whole training string (a path/query/fragment body, or a host label
  or subdomain) of at least 4 characters;
- how many strings drawn 50+ times became a single piece;
- the pieces of 20+ characters.

The tables:
- R1's draw-weighted tables (`data/v2/r1/bpe_rest.json`, `bpe_host.json`): learned on 600,000
  rows drawn as training draws them, so a site with many pages counts many times;
- the same samples relearned with each distinct string counted once (v2_bpe.learn,
  deterministic).

    python research/v2/v2_piece_hazard.py     # -> data/v2/dr_probe/piece_hazard.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import v2_bpe as BPE  # noqa: E402

R1 = ROOT / "data" / "v2" / "r1"
OUT = ROOT / "data" / "v2" / "dr_probe" / "piece_hazard.json"
HEAVY = 50
MIN_LEN = 4
LONG = 20


def stats(merges, seqs: dict) -> dict:
    pieces = [a + b for a, b in merges]
    ps = set(pieces)
    whole = [p for p in pieces if p in seqs and len(p) >= MIN_LEN]
    heavy = [k for k, v in seqs.items() if v >= HEAVY]
    return {"merges": len(pieces), "whole_strings": len(whole), "whole_share": len(whole) / len(pieces),
            "heavy_strings": len(heavy), "heavy_single_piece": sum(1 for k in heavy if k in ps),
            "long_pieces": sum(1 for p in pieces if len(p) >= LONG),
            "longest": sorted(pieces, key=len)[-5:],
            "top_whole_by_weight": sorted(((p, seqs[p]) for p in whole), key=lambda x: -x[1])[:30]}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    sample = json.loads((R1 / "bpe_sample.json").read_text(encoding="utf-8"))["full"]
    out = {"post_hoc": True, "decides": "nothing", "tables": {}}
    for table in ("rest", "host"):
        seqs = sample[table]
        drawn = json.loads((R1 / f"bpe_{table}.json").read_text(encoding="utf-8"))
        n = len(drawn["merges"])
        out["tables"][f"{table}, draw-weighted (R1)"] = stats(drawn["merges"], seqs)
        once = BPE.learn({k: 1 for k in seqs}, n, log=None)
        out["tables"][f"{table}, each distinct string once"] = stats([tuple(m) for m in once.merges], seqs)
    for k, v in out["tables"].items():
        print(f"  {k}: {v['whole_strings']:,} of {v['merges']:,} pieces are whole training strings "
              f"({v['whole_share']:.1%}); {v['heavy_single_piece']}/{v['heavy_strings']} strings drawn "
              f"{HEAVY}+ times are one piece; {v['long_pieces']} pieces of {LONG}+ characters", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
