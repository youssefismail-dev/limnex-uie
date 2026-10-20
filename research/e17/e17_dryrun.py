"""E17 -- the laptop dry run, before the bundle (research/e17/PREREG_E17.md section 7, step 2).
CPU; nothing here is a reading.

    python research/e17/e17_dryrun.py --part b5       # B5 on e16_cap's own weights
    python research/e17/e17_dryrun.py --part inject   # the pooled stream on the full encoded corpus
    python research/e17/e17_dryrun.py --part train    # a few real steps of e17_a's recipe

b5      "host?query, no slash" plans (P0Q1S0), D1 + D1' with and without B5, on e16_cap:
        how many URLs keep their query; and a Q0 plan, identical with and without B5.
inject  The injector on the full encoded corpus with the pooled stream: the strata and
        counts are amendment 1's, the rate is 2, the old rows' batch digest over 400
        batches equals E15's replay, and the appended rows are twice the old draws.
train   Three steps of e17_a's recipe (width 512, 2 parts, E12's start) on real rows with
        pooled-stream rows injected.

Each part prints PASS or raises. Writes data/e17/dryrun_<part>.json.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e16", ROOT / "research" / "e15", ROOT / "research" / "e13",
          ROOT / "research" / "native"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e17_common as E17  # noqa: E402

E16_CAP = ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt"
STREAM = E17.STREAM2_DIR / E17.STREAM_FILE
OUT = ROOT / "data" / "e17"


def part_b5() -> dict:
    import torch
    from uie.model import generate
    from uie.train import load_checkpoint
    if hashlib.sha256(E16_CAP.read_bytes()).hexdigest() != E17.E16_CAP_SHA256:
        raise SystemExit("STOP: the laptop's e16_cap final.pt is not the adopted baseline")
    model, tok, _ = load_checkpoint(E16_CAP, device=torch.device("cpu"))
    model.eval()
    s0 = {"PTH": False, "QRY": True, "SLASH": False}
    out = {}
    for b5 in (False, True):
        urls = generate(model, tok, label="benign", n=48, suffix="com", seed=7, fields=s0,
                        **E17.decode_kwargs(b5))
        kept = sum("?" in u for u in urls)
        out[f"b5={b5}"] = {"n": len(urls), "kept_query": kept, "example": urls[:4]}
        print(f"P0Q1S0, B5 {b5}: {kept}/{len(urls)} keep their query; e.g. {urls[:3]}", flush=True)
    q0 = {"PTH": False, "QRY": False, "SLASH": True}
    a = generate(model, tok, label="phishing", n=24, suffix="com", seed=9, fields=q0, **E17.decode_kwargs(False))
    b = generate(model, tok, label="phishing", n=24, suffix="com", seed=9, fields=q0, **E17.decode_kwargs(True))
    out["q0_identical"] = a == b
    if not out["q0_identical"]:
        raise SystemExit("STOP: B5 changed a Q0 plan")
    if out["b5=True"]["kept_query"] != 48:
        raise SystemExit("STOP: B5 left a Q1 plan without its query")
    print("Q0 plan identical with and without B5\nPASS b5")
    return out


def part_inject() -> dict:
    from uie.train import QueryInjector, TrainConfig, batch_digest, batches
    from e15_smoke import corpus
    t0 = time.time()
    tok, data, blob = corpus()
    inj = QueryInjector.load(STREAM, tok, data, blob["model_config"]["max_len"], E17.E15.DRAW_SEED,
                             E17.E15.RATE_SEED)
    print(f"injector: {inj.summary()} ({time.time() - t0:.0f}s)", flush=True)
    assert {s: len(r) for s, r in inj.rows.items()} == E17.N_NEW, "N_new differs from amendment 1"
    assert {s: inj.old_counts[s] for s in E17.N_OLD} == E17.N_OLD, "N_old differs"
    assert all(r == E17.RATE for r in inj.rate.values()), inj.rate
    tc = blob["train_config"]
    cfg = TrainConfig(batch_size=tc["batch_size"], seed=tc["seed"], class_alpha=tc["class_alpha"],
                      bucket_window=tc["bucket_window"])
    steps = 400
    ref = batch_digest(data, cfg, tok.pad_id, steps)
    d = hashlib.sha256()
    s = batches(data, cfg, tok.pad_id, random.Random(cfg.seed), None, inj, d)
    for _ in range(steps):
        next(s)
    assert d.hexdigest() == ref, "injection changed the old rows' batches"
    rec = inj.record()["strata"]
    assert all(v["appended_rows"] == 2 * v["old_rows_seen"] for v in rec.values())
    app = sum(v["appended_rows"] for v in rec.values()) / steps
    pos = sum(v["appended_query_positions"] for v in rec.values()) / steps
    print(f"over {steps} batches: old-row digest equals the replay's; appended {app:.3f} rows/batch "
          f"(P8 {E17.APPENDED_PER_BATCH}), {pos:.1f} query positions/batch (P8 {E17.ADDED_QUERY_POSITIONS_PER_BATCH})")
    assert abs(app - E17.APPENDED_PER_BATCH) < 1.0 and abs(pos - E17.ADDED_QUERY_POSITIONS_PER_BATCH) < 40
    print("PASS inject")
    return {"rows_per_batch": app, "positions_per_batch": pos, "summary": inj.summary(),
            "source_sha256": inj.source_sha256}


def part_train() -> dict:
    import torch
    from uie.kaggle_train import new_model_e12
    from uie.model import ModelConfig
    from uie.train import TrainConfig, URLDataset, encode_corpus, train
    blob = torch.load(E16_CAP, map_location="cpu", weights_only=False)
    from uie.tokenizer import URLTokenizer
    tok = URLTokenizer.from_state_dict(blob["tokenizer"])
    tc = dict(blob["train_config"])
    with gzip.open(ROOT / "data" / "bundle2" / "name_train.jsonl.gz", "rt", encoding="utf-8") as fh:
        urls = [json.loads(line)["url"] for _, line in zip(range(3000), fh)]
    enc, _ = encode_corpus(urls, ["benign"] * len(urls), tok, 384, corpora=["name"] * len(urls))
    data = URLDataset(enc)
    lines = STREAM.read_text(encoding="utf-8").splitlines()
    sample = lines[:200] + lines[29726:29926]                     # P0Q1 rows of both streams
    with tempfile.TemporaryDirectory() as d:
        small = Path(d) / "stream.jsonl"
        small.write_text("\n".join(sample) + "\n", encoding="utf-8")
        cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=512, n_layers=8, n_heads=8, d_ff=2048, max_len=384)
        model, _ = new_model_e12(cfg, tok, 20260803)
        t0 = time.time()
        res = train(model, data, data, tok,
                    TrainConfig(**{**tc, "qry_extra": str(small), "steps": 3, "batch_size": 32, "eval_every": 3,
                                   "eval_batches": 1, "warmup": 1}),
                    device=torch.device("cpu"), out_dir=Path(d), log_every=1)
    q = res["qry_extra"]["strata"]
    print({s: {k: v[k] for k in ("N_new", "N_old", "rate", "old_rows_seen", "appended_rows")} for s, v in q.items()})
    assert res["loss_constant"] == E17.E16.LOSS_CONSTANT
    print(f"3 steps in {time.time() - t0:.0f}s\nPASS train")
    return {"strata": q, "final_val": res["final"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("b5", "inject", "train"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    res = {"b5": part_b5, "inject": part_inject, "train": part_train}[args.part]()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"dryrun_{args.part}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
