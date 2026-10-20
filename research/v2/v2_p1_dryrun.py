"""
research/v2/v2_p1_dryrun.py

V2-P1's laptop dry run at the registered size, on the CPU (PREREG_V2_P1_MEMORY.md section
10, step 2), before any Kaggle run:

    python research/v2/v2_p1_dryrun.py        # -> data/v2/p1/dryrun.json

1. The start, on this torch: E12's procedure at width 512, then the memory from seed
   20261050, built twice. e16_cap's part equals this torch's e16_cap start (Kaggle's torch
   rebuilds the recorded `caca0107...`; the laptop's differs, as E16's amendment 2 found),
   and the count is 161,795,072.
2. V-equiv at full size: with every value row at 0, the start computes exactly the logits
   of e16_cap's start on real rows.
3. The checkpoint's round trip at full size: saved by `uie.train.save_checkpoint`, loaded by
   `v2_mem.load_checkpoint`, identical weights and logits.
4. Generation at full size: e16_b5's decode on the start, twice from the same seed,
   identical.
5. The worst-case check's code path (on the CPU it measures nothing).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import v2_p1_common as P1  # noqa: E402

OUT = ROOT / "data" / "v2" / "p1"


def main() -> None:
    import torch
    from uie.controlled import StateSuffixTables  # noqa: F401  (the generation tables' module)
    from uie.evaluation import FIELDS
    from uie.free import weights_sha256
    from uie.kaggle_train import new_model_e12
    from uie.model import ModelConfig
    from uie.suffixes import SuffixTable
    from uie.tokenizer import URLTokenizer
    from uie.train import TrainConfig, save_checkpoint
    import e16_common as E16
    import v2_mem as M
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    rec: dict = {"torch": torch.__version__, "registration": P1.REGISTRATION}
    t0 = time.time()
    blob = torch.load(ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt", map_location="cpu",
                      weights_only=False)
    record = P1.e16_cap_record(blob)
    tok = URLTokenizer.from_state_dict(record["tokenizer"])
    assert tok.state_dict() == URLTokenizer.from_state_dict(blob["tokenizer"]).state_dict()
    del blob

    # 1. the start
    start512 = E16.start_fingerprints(tok, E16.D_MODEL)
    a, pa = P1.start(tok)
    del a
    b, pb = P1.start(tok)
    rec["start"] = {"e16_cap_start_this_torch": start512["init_weights_sha256"], "p1_mem": pa,
                    "deterministic": pa == pb,
                    "base_is_e16_caps_start": pa["base_init_sha256"] == start512["init_weights_sha256"],
                    "params": pa["params"], "params_registered": P1.PARAMS}
    print(f"1. start: deterministic {pa == pb}; e16_cap's part = this torch's e16_cap start "
          f"{rec['start']['base_is_e16_caps_start']}; {pa['params']:,} parameters ({time.time() - t0:.0f}s)",
          flush=True)

    # 2. V-equiv at full size, on real rows
    r = P1.recipe(Path(P1.STREAM_FILE))
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=r["d_model"], n_layers=r["n_layers"],
                      n_heads=r["n_heads"], d_ff=4 * r["d_model"], max_len=r["max_len"])
    base, _ = new_model_e12(cfg, tok, E16.SEED)
    import gzip
    with gzip.open(ROOT / "data" / "corpus2" / "structure_val.jsonl.gz", "rt", encoding="utf-8") as fh:
        urls = [json.loads(next(fh))["url"] for _ in range(20)]
    rows = [x for x in (tok.encode(u, "benign", frame="page") for u in urls) if len(x) <= r["max_len"]][:6]
    width = max(len(x) for x in rows)
    ids = torch.tensor([x + [tok.pad_id] * (width - len(x)) for x in rows])
    with torch.no_grad():
        b.memory.values.zero_()
        same = bool(torch.equal(base.eval()(ids), b.eval()(ids)))
    rec["v_equiv_full_size"] = same
    print(f"2. V-equiv at full size (values at 0 = e16_cap's start): {same}", flush=True)
    del b, base

    # 3. the checkpoint's round trip, and 4. generation, on the start as drawn
    model, _ = P1.start(tok)
    path = OUT / "dryrun_start.pt"
    save_checkpoint(model, tok, TrainConfig(), path)
    back, t2, c2 = M.load_checkpoint(path)
    weights = all(torch.equal(v, back.state_dict()[k]) for k, v in model.state_dict().items())
    with torch.no_grad():
        logits = bool(torch.equal(model.eval()(ids), back.eval()(ids)))
    rec["round_trip"] = {"weights": weights, "logits": logits, "size_mb": round(path.stat().st_size / 2 ** 20, 1),
                         "tokenizer": t2.state_dict() == tok.state_dict(), "memory": {k: getattr(c2, k) for k in P1.MEM}}
    print(f"3. round trip: weights {weights}, logits {logits}; checkpoint {rec['round_trip']['size_mb']} MB",
          flush=True)
    del back
    path.unlink()
    shipped = SuffixTable.load(ROOT / "data" / "bundle2" / "suffix_table.json")
    ps = json.loads((ROOT / "data" / "e12" / "presence_slash_table.json").read_text(encoding="utf-8"))["table"]
    t1 = time.time()
    g1 = P1.free_urls(model, tok, "benign", 0, shipped, ps, n=6)
    g2 = P1.free_urls(model, tok, "benign", 0, shipped, ps, n=6)
    rec["generation"] = {"identical": g1 == g2, "n": len(g1), "seconds": round(time.time() - t1, 1),
                         "decode": P1.decode_kwargs(), "examples": g1[:3],
                         "all_parse": all(FIELDS["FULL"](u) for u in g1)}
    print(f"4. generation twice from one seed identical: {g1 == g2}; decode {P1.decode_kwargs()} "
          f"({rec['generation']['seconds']}s for 12 URLs)", flush=True)
    del model

    # 5. the worst-case check's code path
    rec["preflight_cpu"] = M.memory_preflight(torch.device("cpu"), 2, vocab_size=tok.vocab_size,
                                              max_len=48)
    print("5. the worst-case check runs (CPU: nothing to measure)", flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    ok = (rec["start"]["deterministic"] and rec["start"]["base_is_e16_caps_start"]
          and rec["start"]["params"] == P1.PARAMS and rec["v_equiv_full_size"]
          and weights and logits and rec["round_trip"]["tokenizer"] and rec["generation"]["identical"])
    rec["pass"] = ok
    (OUT / "dryrun.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"DRY RUN: {'PASS' if ok else 'FAIL'} ({rec['seconds']:.0f}s) -> {OUT / 'dryrun.json'}")


if __name__ == "__main__":
    main()
