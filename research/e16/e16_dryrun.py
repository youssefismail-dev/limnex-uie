"""E16 -- the laptop dry run, before gate B (research/e16/PREREG_E16.md section 11, step 2).
CPU; nothing here is a reading.

    python research/e16/e16_dryrun.py --part start   # V-start': the width-512 start
    python research/e16/e16_dryrun.py --part train   # real rows, width 512, 2 parts, the stream

start  Rebuilds the start `uie.kaggle_train._run` builds for e15_new's recipe, first at
       width 384, where it must reproduce e12_final's recorded start exactly
       (c7e71edf... and E12's four fingerprints): that proves the procedure is
       training's own. Then at width 512, twice (it must be deterministic), with the
       parameter count. These are the fingerprints E16_train checks before training.
train  A few training steps at width 512 from that start, each batch in 2 parts, on
       real rows with real stream rows injected: the code path E16_train runs. Also
       times a step at 384 and at 512 on the same batch, for the cost estimate.

Each part prints PASS or raises. Writes data/e16/dryrun_<part>.json.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e15", ROOT / "research" / "e13", ROOT / "research" / "native"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e16_common as E16  # noqa: E402
import e15_common as E15  # noqa: E402

E15_NEW = ROOT / "data" / "e15" / "train" / "e15_new" / "final.pt"
DATA = ROOT / "data" / "bundle2"
STREAM = ROOT / "data" / "e15" / "qry_extra" / E15.STREAM_FILE
OUT = ROOT / "data" / "e16"


def baseline():
    import torch
    from uie.tokenizer import URLTokenizer
    if hashlib.sha256(E15_NEW.read_bytes()).hexdigest() != E16.E15_NEW_SHA256:
        raise SystemExit("STOP: the laptop's e15_new final.pt is not the adopted baseline")
    blob = torch.load(E15_NEW, map_location="cpu", weights_only=False)
    return blob, URLTokenizer.from_state_dict(blob["tokenizer"])


def part_start() -> dict:
    """Amendment 2 (section 14): the seeded initialisation depends on the torch build.
    The laptop's torch is not Kaggle's, so the laptop cannot record Kaggle's values. It
    proves what it can: that this rebuild IS E12's procedure (on the same torch, its
    width-384 fingerprints equal E12's own `start_fingerprints`), that the width-512
    start is deterministic, and the parameter count. Kaggle's own width-384 build must
    then reproduce e12_final's recorded start before training (E16_train, cell 3)."""
    import torch
    sys.path.insert(0, str(ROOT / "research" / "native"))
    import e12_common as K
    _, tok = baseline()
    ref = E16.start_fingerprints(tok, 384)
    e10, e11 = K.start_fingerprints()
    if (ref["e10_start_sha256"], ref["e11_start_sha256"]) != (e10, e11):
        raise SystemExit("STOP: at width 384 this rebuild is not E12's own procedure on this torch")
    on_record = {"init_weights_sha256": E15.E12_FINAL_INIT, **E15.E12_FINAL_START}
    same_torch = all(ref.get(k) == v for k, v in on_record.items())
    print(f"torch {torch.__version__}: width 384 equals E12's own start procedure on this torch "
          f"(e10 {e10[:12]}..., e11 {e11[:12]}...); equals e12_final's recorded start "
          f"(Kaggle's torch): {same_torch}")
    a = E16.start_fingerprints(tok, E16.D_MODEL)
    b = E16.start_fingerprints(tok, E16.D_MODEL)
    if a != b:
        raise SystemExit("STOP: the width-512 start is not deterministic")
    if a["params"] != E16.PARAMS:
        raise SystemExit(f"STOP: {a['params']:,} parameters, registered {E16.PARAMS:,}")
    print(f"width 512: {a['params']:,} parameters; deterministic (two builds identical); "
          f"this torch's values (informational):")
    for k, v in a.items():
        print(f"  {k}: {v}")
    print("PASS start")
    return {"torch": torch.__version__, "width_384": ref, "width_384_matches_record": same_torch,
            "width_512_this_torch": a}


def part_train() -> dict:
    import torch
    from uie.kaggle_train import new_model_e12
    from uie.model import ModelConfig
    from uie.train import TrainConfig, URLDataset, encode_corpus, train
    blob, tok = baseline()
    tc = dict(blob["train_config"])
    with gzip.open(DATA / "name_train.jsonl.gz", "rt", encoding="utf-8") as fh:
        urls = [json.loads(line)["url"] for _, line in zip(range(3000), fh)]
    enc, _ = encode_corpus(urls, ["benign"] * len(urls), tok, 384, corpora=["name"] * len(urls))
    data = URLDataset(enc)
    rows = [json.loads(x) for _, x in zip(range(400), open(STREAM, encoding="utf-8"))]
    out = {}
    with tempfile.TemporaryDirectory() as d:
        small = Path(d) / "stream.jsonl"
        small.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        for width, parts in ((E16.D_MODEL, E16.ACCUM_STEPS), (384, 1)):
            cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=width, n_layers=8, n_heads=8,
                              d_ff=4 * width, max_len=384)
            model, _ = new_model_e12(cfg, tok, E16.SEED)
            run_cfg = TrainConfig(**{**tc, "qry_extra": str(small), "steps": 3, "batch_size": 32,
                                     "eval_every": 3, "eval_batches": 1, "warmup": 1,
                                     "accum_steps": parts})
            t0 = time.time()
            res = train(model, data, data, tok, run_cfg, device=torch.device("cpu"),
                        out_dir=Path(d) / f"w{width}", log_every=1)
            secs = time.time() - t0
            q = res["qry_extra"]["strata"]
            out[f"width_{width}_parts_{parts}"] = {
                "seconds_3_steps": round(secs, 1), "final_val": res["final"],
                "appended_rows": {s: v["appended_rows"] for s, v in q.items()},
                "loss_constant": res["loss_constant"]}
            if not all(v["appended_rows"] > 0 for v in q.values()):
                raise SystemExit("STOP: no rows were injected")
            if res["loss_constant"] != E16.LOSS_CONSTANT:
                raise SystemExit("STOP: the loss constant is not pinned")
            saved = torch.load(Path(d) / f"w{width}" / "best.pt", weights_only=False) \
                if (Path(d) / f"w{width}" / "best.pt").exists() else None
            if saved is not None and parts != 1 and saved["train_config"].get("accum_steps") != parts:
                raise SystemExit("STOP: the checkpoint does not record the accumulation")
            print(f"width {width}, {parts} part(s): 3 steps in {secs:.0f}s; "
                  f"val {res['final']['mean']:.4f}", flush=True)
    a, b = out[f"width_{E16.D_MODEL}_parts_{E16.ACCUM_STEPS}"], out["width_384_parts_1"]
    out["cost_ratio_cpu"] = round(a["seconds_3_steps"] / b["seconds_3_steps"], 2)
    print(f"CPU cost ratio, width 512 in 2 parts against width 384 in 1: {out['cost_ratio_cpu']}")
    print("PASS train")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("start", "train"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    res = {"start": part_start, "train": part_train}[args.part]()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"dryrun_{args.part}.json").write_text(json.dumps(res, indent=1, default=float),
                                                  encoding="utf-8")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
