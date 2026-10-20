"""E15-B -- the laptop pre-flight on the real data, before the bundle (research/e15/
PREREG_E15.md section 9.12, step 3). CPU; nothing here is data.

    python research/e15/e15_smoke.py --part inject   # the injector on the full encoded corpus
    python research/e15/e15_smoke.py --part train    # train() a few steps with real rows injected
    python research/e15/e15_smoke.py --part decode   # e15's decode (D1 + D1') on e12_final

Each part prints PASS or raises.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e13", ROOT / "research" / "native"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e15_common as E  # noqa: E402

DATA = ROOT / "data" / "bundle2"
E12_FINAL = ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt"
STREAM = ROOT / "data" / "e15" / "qry_extra" / E.STREAM_FILE


def corpus():
    """The training set exactly as `kaggle_train._run` builds it (e15_replay's code)."""
    import torch
    from uie.kaggle_train import STREAMS, TOP_LANGUAGES, load_streams
    from uie.tokenizer import URLTokenizer
    from uie.train import URLDataset, encode_corpus
    blob = torch.load(E12_FINAL, map_location="cpu", weights_only=False)
    tok = URLTokenizer.from_state_dict(blob["tokenizer"])
    urls, labels, langs, corp, label_counts = load_streams(DATA, STREAMS)
    counts = {}
    for lang in langs:
        counts[lang] = counts.get(lang, 0) + 1
    keep = {lang for lang, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP_LANGUAGES]}
    keep.add("und")
    langs = [lang if lang in keep else "und" for lang in langs]
    enc, _ = encode_corpus(urls, labels, tok, blob["model_config"]["max_len"], langs, corp)
    return tok, URLDataset(enc, label_counts=label_counts), blob


def part_inject() -> None:
    from uie.train import QueryInjector, TrainConfig, batch_digest, batches
    t0 = time.time()
    tok, data, blob = corpus()
    inj = QueryInjector.load(STREAM, tok, data, blob["model_config"]["max_len"], E.DRAW_SEED, E.RATE_SEED)
    print(f"injector: {inj.summary()} ({time.time() - t0:.0f}s)", flush=True)
    assert {s: len(r) for s, r in inj.rows.items()} == E.N_NEW, "N_new differs from Amendment 3"
    assert {s: inj.old_counts[s] for s in E.N_OLD} == E.N_OLD, "N_old differs from Amendment 3"
    assert inj.p == E.P_INJECT
    assert inj.source_sha256 == hashlib.sha256(STREAM.read_bytes()).hexdigest()
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
    app = sum(inj.counts["appended_rows"].values()) / steps
    pos = sum(inj.counts["appended_query_positions"].values()) / steps
    print(f"over {steps} batches: old-row digest equals the replay's; appended {app:.3f} rows/batch "
          f"(P8 {E.APPENDED_PER_BATCH}), {pos:.1f} query positions/batch (P8 {E.ADDED_QUERY_POSITIONS_PER_BATCH})")
    assert abs(app - E.APPENDED_PER_BATCH) < 0.5 and abs(pos - E.ADDED_QUERY_POSITIONS_PER_BATCH) < 20
    print("PASS inject")


def part_train() -> None:
    import gzip
    import json
    import torch
    from uie.kaggle_train import new_model_e12
    from uie.train import TrainConfig, URLDataset, encode_corpus, load_checkpoint, train
    _, tok, cfg = load_checkpoint(E12_FINAL)
    tc = dict(torch.load(E12_FINAL, map_location="cpu", weights_only=False)["train_config"])
    with gzip.open(DATA / "name_train.jsonl.gz", "rt", encoding="utf-8") as fh:
        urls = [json.loads(line)["url"] for _, line in zip(range(3000), fh)]
    enc, _ = encode_corpus(urls, ["benign"] * len(urls), tok, cfg.max_len, corpora=["name"] * len(urls))
    data = URLDataset(enc)
    rows = [json.loads(l) for _, l in zip(range(400), open(STREAM, encoding="utf-8"))]
    with tempfile.TemporaryDirectory() as d:
        small = Path(d) / "stream.jsonl"
        small.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        model, _ = new_model_e12(cfg, tok, tc["seed"])
        res = train(model, data, data, tok, TrainConfig(**{**tc, "steps": 4, "batch_size": 32, "eval_every": 4,
                                                           "eval_batches": 1, "warmup": 1,
                                                           "qry_extra": str(small)}),
                    device=torch.device("cpu"), out_dir=Path(d), log_every=10)
    q = res["qry_extra"]
    print({s: {k: v for k, v in r.items() if k in ("N_new", "N_old", "p", "old_rows_seen", "appended_rows")}
           for s, r in q["strata"].items()})
    assert all(r["appended_rows"] > 0 for r in q["strata"].values())
    assert res["loss_constant"] == E.E12_FINAL_LOSS_CONSTANT
    print("PASS train")


def part_decode() -> None:
    import json
    from uie.controlled import StateSuffixTables
    from uie.factorization import parse
    from uie.suffixes import SuffixTable
    from uie.train import load_checkpoint
    model, tok, _ = load_checkpoint(E12_FINAL)
    model.eval()
    shipped = SuffixTable.load(DATA / "suffix_table.json")
    st = StateSuffixTables.load(ROOT / "data" / "state_suffix_tables.json")
    ps = json.loads((ROOT / "data" / "e12" / "presence_slash_table.json").read_text())["table"]
    for label in ("benign", "phishing"):
        urls = E.free_urls(model, tok, label, 0, shipped, ps, n=16)
        assert len(urls) == 16
    urls = E.control_urls(model, tok, "benign", "P1Q1", 0, st, ps, n=16)
    assert len(urls) == 16 and all(bool(parse(u).path) for u in urls)
    print("free and control generate with D1 + D1'; every P1Q1 host slashed")
    print("PASS decode")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("inject", "train", "decode"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    {"inject": part_inject, "train": part_train, "decode": part_decode}[args.part]()
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
