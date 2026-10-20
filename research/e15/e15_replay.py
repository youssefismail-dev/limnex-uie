"""E15-B V-batch's reference (research/e15/PREREG_E15.md sections 9.10, 10.7): the digest
of every batch the training sampler draws for e12_final's dataset and seed over 20,000
steps, with nothing injected. `e15_new`'s training record must carry the same digest for
its old rows. Laptop, CPU; nothing is trained.

    python research/e15/e15_replay.py

The dataset is built as `uie.kaggle_train._run` builds it (the same streams, language
map, tokenizer, frames and max_len); the tokenizer is e12_final's own, which E13 and E14
showed training rebuilds exactly. Writes data/e15/batch_digest_e12_final.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "bundle2"
E12_FINAL = ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt"
OUT = ROOT / "data" / "e15" / "batch_digest_e12_final.json"


def main() -> None:
    import torch
    from uie.kaggle_train import STREAMS, TOP_LANGUAGES, load_streams
    from uie.tokenizer import URLTokenizer
    from uie.train import TrainConfig, URLDataset, batch_digest, encode_corpus
    t0 = time.time()
    blob = torch.load(E12_FINAL, map_location="cpu", weights_only=False)
    tok = URLTokenizer.from_state_dict(blob["tokenizer"])
    tc = blob["train_config"]
    urls, labels, langs, corp, label_counts = load_streams(DATA, STREAMS)
    counts = {}
    for lang in langs:
        counts[lang] = counts.get(lang, 0) + 1
    keep = {lang for lang, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP_LANGUAGES]}
    keep.add("und")
    langs = [lang if lang in keep else "und" for lang in langs]
    enc, dropped = encode_corpus(urls, labels, tok, blob["model_config"]["max_len"], langs, corp)
    data = URLDataset(enc, label_counts=label_counts)
    print(f"encoded {data.counts()}, dropped {dropped:,} ({time.time() - t0:.0f}s)", flush=True)
    cfg = TrainConfig(batch_size=tc["batch_size"], seed=tc["seed"],
                      class_alpha=tc["class_alpha"], bucket_window=tc["bucket_window"])
    steps = tc["steps"]
    digest = batch_digest(data, cfg, tok.pad_id, steps)
    out = {"registration": "PREREG_E15 sections 9.10, 10.7 (V-batch)", "steps": steps,
           "seed": cfg.seed, "batch_size": cfg.batch_size, "bucket_window": cfg.bucket_window,
           "class_alpha": cfg.class_alpha, "counts": data.counts(), "dropped": dropped,
           "label_counts": label_counts, "digest": digest, "seconds": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"digest {digest} over {steps:,} steps ({time.time() - t0:.0f}s) -> {OUT}")


if __name__ == "__main__":
    main()
