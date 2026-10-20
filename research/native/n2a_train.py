"""N2a arms -- E7's recipe on query-bearing rows only (PREREG_N2A sections 2-3,
Amendments 2 and 3). DIAGNOSTIC: these models never generate data for use.

    python research/native/n2a_train.py --data <bundle> --e7 <best_e7.pt> --dry-run
    python research/native/n2a_train.py --data <bundle> --e7 <best_e7.pt> --arm A1 \
        --suffix-table <suffix_table.json> --out <dir>

What is E7's, by construction -- read from E7's own checkpoint, not restated:
- the tokenizer (vocabulary 107) and the model configuration;
- every training setting E7 recorded (batch 256, lr 3e-4, weight decay, clipping,
  seed 20260803, class_alpha 1.0, control dropout 0.5, ...), except the three
  below;
- `uie.train.train` runs unchanged, and validation rows are encoded with the
  controls hidden, as E7 was validated.

What differs, as registered:
- only query-bearing rows, in both frames and both splits;
- class weights = E7's per-class draws per row x kept query rows (Amendment 3):
  1.65888 benign, 3.76842 phishing -- so each query row gets E7's exposure;
- steps = that exposure / 256 for A1, four times A1's for A2;
- warm-up 2.5% of steps (E7: 500 / 20,000); evaluation every 5% (E7: 1,000).

Readings use final.pt (section 2): after training, each arm loads its final.pt
back and writes 5 x 1,500 free-mode URLs per class with the resting decode,
seeds 20261001 + (100*s + g)*1e9, s = 18 benign / 19 phishing (Amendment 2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import e8_common as K                                        # noqa: E402

#: E7's draws per training row, by class (PREREG_N2A Amendment 3).
EXPOSURE = {"benign": 1.65888, "phishing": 3.76842}
MULTIPLIER = {"A1": 1, "A2": 4}
WARMUP_SHARE = 0.025                 # E7: 500 / 20,000
EVAL_SHARE = 0.05                    # E7: 1,000 / 20,000
N_GEN, G = 1_500, 5
GEN_BLOCK = {"benign": 18, "phishing": 19}
#: E7's recorded training settings that N2a changes; every other one is E7's.
CHANGED = ("steps", "warmup", "eval_every")
#: Kept query rows, fixed from the counts before any arm (PREREG_N2A Amendment 4:
#: --dry-run on the data bundle). A run on other counts stops (section 6, V).
EXPECTED_KEPT = {"benign": 223_342, "phishing": 13_270}


def gen_seed(label: str, g: int) -> int:
    """Amendment 2's blocks s = 18 / 19, which no earlier experiment used."""
    return 20261001 + (100 * GEN_BLOCK[label] + g) * 1_000_000_000


def class_weights(kept: Dict[str, int]) -> Dict[str, float]:
    """`label_counts` for URLDataset. With class_alpha 1.0 the sampler draws class
    c with probability proportional to these, and rows uniformly within it, so a
    row of class c is drawn EXPOSURE[c] times per a1_steps(kept) * 256 draws."""
    return {c: EXPOSURE[c] * kept[c] for c in EXPOSURE}


def a1_steps(kept: Dict[str, int], batch_size: int = 256) -> int:
    return round(sum(class_weights(kept).values()) / batch_size)


def schedule(kept: Dict[str, int], arm: str, batch_size: int = 256) -> Dict[str, int]:
    """Steps, warm-up and evaluation spacing for an arm; A2 is exactly 4 x A1."""
    steps = MULTIPLIER[arm] * a1_steps(kept, batch_size)
    return {"steps": steps, "warmup": max(1, round(WARMUP_SHARE * steps)),
            "eval_every": max(1, round(EVAL_SHARE * steps))}


def has_query(url: str) -> bool:
    """The tokenizer's own test for opening <QRY> (a non-empty query), which is
    also `uie.evaluation.FIELDS["QRY"]` -- the gate's QRY field."""
    from uie.evaluation import FIELDS
    return bool(FIELDS["QRY"](url))


def query_rows(urls: Sequence[str], labels: Sequence[str], langs: Sequence[str],
               corpora: Sequence[str]):
    keep = [i for i, u in enumerate(urls) if has_query(u)]
    return ([urls[i] for i in keep], [labels[i] for i in keep],
            [langs[i] for i in keep], [corpora[i] for i in keep])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_e7(path: Path, expected_sha256: str = K.E7_SHA256):
    """E7's tokenizer, model configuration and recorded training settings."""
    import torch
    from uie.tokenizer import URLTokenizer
    from uie.model import ModelConfig
    if sha256(path) != expected_sha256:
        raise SystemExit(f"STOP: {path} is not E7's checkpoint ({expected_sha256[:12]}...)")
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    tok = URLTokenizer.from_state_dict(blob["tokenizer"])
    if not tok.include_fields or getattr(tok, "presence_plan", False):
        raise SystemExit("STOP: not an E7-type tokenizer (prefix controls, no plan)")
    return tok, ModelConfig(**blob["model_config"]), dict(blob["train_config"]), blob["tokenizer"]


def encode(data: Path, tok, max_len: int):
    """Query-bearing rows of both frames: training as E7 encoded them, validation
    with the controls hidden (E7's validation). Returns the encodings, the kept
    counts per class and the rows dropped over max_len."""
    from uie.kaggle_train import STREAMS, VAL_STREAMS, load_streams
    from uie.train import encode_corpus
    tr = query_rows(*load_streams(data, STREAMS)[:4])
    va = query_rows(*load_streams(data, VAL_STREAMS)[:4])
    tr_enc, tr_dropped = encode_corpus(tr[0], tr[1], tok, max_len, tr[2], tr[3])
    va_enc, va_dropped = encode_corpus(va[0], va[1], tok, max_len, va[2], va[3],
                                       hide_fields=("PTH", "QRY"))
    rows = {"train_query_rows": {c: tr[1].count(c) for c in EXPOSURE},
            "val_query_rows": {c: va[1].count(c) for c in EXPOSURE},
            "dropped_over_max_len": tr_dropped, "dropped_over_max_len_val": va_dropped}
    return tr_enc, va_enc, {c: len(tr_enc.get(c, [])) for c in EXPOSURE}, rows


def run(data: Path, e7: Path, arm: str, out: Path, suffix_table: Path, *,
        device=None, n_gen: int = N_GEN, g: int = G, steps: Optional[int] = None,
        expected_sha256: str = K.E7_SHA256,
        expect_kept: Optional[Dict[str, int]] = EXPECTED_KEPT) -> dict:
    """Train one arm, then generate from its final.pt. `steps`,
    `expected_sha256` and `expect_kept=None` exist for the tests' tiny model only."""
    import torch
    from uie.model import URLTransformer
    from uie.suffixes import SuffixTable, generate_with_suffixes
    from uie.train import (TrainConfig, URLDataset, load_checkpoint, save_checkpoint,
                           train)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, cfg, e7_train, e7_tok_state = load_e7(e7, expected_sha256)
    tr_enc, va_enc, kept, rows = encode(data, tok, cfg.max_len)
    if expect_kept is not None and kept != expect_kept:
        raise SystemExit(f"STOP (V): kept query rows {kept}, registered {expect_kept}")
    sched = schedule(kept, arm, e7_train["batch_size"])
    if steps is not None:
        sched = {"steps": steps, "warmup": max(1, round(WARMUP_SHARE * steps)),
                 "eval_every": max(1, round(EVAL_SHARE * steps))}
    train_cfg = TrainConfig(**{**e7_train, **sched})
    weights = class_weights(kept)
    train_set = URLDataset(tr_enc, label_counts=weights)
    val_set = URLDataset(va_enc, label_counts=weights)     # evaluate() ignores weights
    print(f"N2a {arm}: kept query rows {kept}, dropped {rows['dropped_over_max_len']} "
          f"over max_len; steps {sched['steps']}, warm-up {sched['warmup']}, eval every "
          f"{sched['eval_every']}; class weights {weights}", flush=True)

    # Initialisation is seeded too (E7's process-default init cannot be recovered).
    torch.manual_seed(train_cfg.seed)
    model = URLTransformer(cfg)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = train(model, train_set, val_set, tok, train_cfg, device=device,
                   out_dir=out, log_every=100)
    save_checkpoint(model, tok, train_cfg, out / "final.pt")
    train_minutes = round((time.time() - t0) / 60, 1)
    finite = all(math.isfinite(v) for v in result["final"].values())
    history = result["history"]
    best_step = min(history, key=lambda h: h["mean"])["step"] if history else None

    # Readings use final.pt: generate from the checkpoint as saved, exactly as a
    # later check would load it.
    del model
    final, ftok, _ = load_checkpoint(out / "final.pt", device=device)
    final.to(device).eval()
    assert ftok.state_dict() == e7_tok_state, "final.pt's tokenizer is not E7's"
    shipped = SuffixTable.load(suffix_table)
    t1 = time.time()
    files: Dict[str, dict] = {}
    for label in K.LABELS:
        for gi in range(g):
            urls: List[str] = generate_with_suffixes(
                final, ftok, label=label, n=n_gen, table=shipped, seed=gen_seed(label, gi),
                device=device, **K.resting(ftok, label))
            name = f"gen_{arm}_{label}_g{gi}.json"
            (out / name).write_text(json.dumps(urls), encoding="utf-8")
            files[name] = {"label": label, "g": gi, "seed": gen_seed(label, gi),
                           "n": len(urls), "with_query": sum(map(has_query, urls)),
                           "sha256": sha256(out / name)}
            print(f"  {name}: {len(urls)} URLs, {files[name]['with_query']} with a query",
                  flush=True)
    record = {
        "registration": "research/native/PREREG_N2A.md (Amendments 2 and 3)",
        "status": "DIAGNOSTIC", "arm": arm, "e7_sha256": expected_sha256,
        "kept_query_rows": kept, **rows, "class_weights": weights,
        "exposure_per_row": EXPOSURE, "schedule": sched,
        "train_config": {k: getattr(train_cfg, k) for k in e7_train},
        "changed_from_e7": {k: [e7_train[k], getattr(train_cfg, k)] for k in CHANGED},
        "vocab_size": tok.vocab_size, "final": result["final"], "finite": finite,
        "history": history, "best_step": best_step, "train_minutes": train_minutes,
        "best_sha256": sha256(out / "best.pt") if (out / "best.pt").exists() else None,
        "final_sha256": sha256(out / "final.pt"),
        "generation": {"from": "final.pt", "decode": "resting (temperature 1.0 + path delta)",
                       "delta": K.DELTA, "n": n_gen, "G": g, "files": files,
                       "minutes": round((time.time() - t1) / 60, 1)},
    }
    (out / "N2A_RECORD.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"N2a {arm}: trained in {train_minutes} min (final {result['final']}, finite "
          f"{finite}); generation {record['generation']['minutes']} min", flush=True)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--e7", type=Path, required=True)
    ap.add_argument("--arm", choices=tuple(MULTIPLIER))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--suffix-table", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="count and encode only: print both arms' schedules")
    args = ap.parse_args()
    if args.dry_run:
        tok, cfg, e7_train, _ = load_e7(args.e7)
        _, _, kept, rows = encode(args.data, tok, cfg.max_len)
        print(json.dumps({"kept_query_rows": kept, **rows, "class_weights": class_weights(kept),
                          "schedules": {a: schedule(kept, a, e7_train["batch_size"])
                                        for a in MULTIPLIER}}, indent=1))
        return
    if not (args.arm and args.out and args.suffix_table):
        ap.error("--arm, --out and --suffix-table are required unless --dry-run")
    run(args.data, args.e7, args.arm, args.out, args.suffix_table)


if __name__ == "__main__":
    main()
