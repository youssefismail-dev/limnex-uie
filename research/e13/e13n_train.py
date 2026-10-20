"""E13 N -- N2a's arms on e12_final's recipe (research/e13/PREREG_E13.md section 2.5).
DIAGNOSTIC: these models never generate data for use, and are never candidates.

    python research/e13/e13n_train.py --data <bundle> --e12 <e12_final final.pt> --dry-run
    python research/e13/e13n_train.py --data <bundle> --e12 <e12_final final.pt> --arm A1 \
        --tables <code dir> --out <dir>

What is e12_final's, read from its own checkpoint and record, not restated:
- the tokenizer (vocabulary 110), the model configuration (max_len 384) and every
  recorded training setting (the fixed loss constant, the clipping repair, the
  seed, class_alpha 1.0, ...), except the three below;
- E12's start (`uie.kaggle_train.new_model_e12`, the seed 20260803);
- `uie.train.train` runs unchanged; validation rows are encoded as E12 encoded them.

What differs, as registered (N2a's design, Amendments 2-4):
- only query-bearing rows, both frames, both splits;
- class weights = e12_final's draws per row x kept query rows, so each query row
  gets e12_final's exposure: EXPOSURE[c] = 20,000 x 256 x w_c / rows_c, with w_c
  the class's label share (class_alpha 1.0) and rows_c e12_final's encoded rows;
- steps = that exposure / 256 for A1, four times A1's for A2; warm-up 2.5% of
  steps, evaluation every 5% (N2a's).

After training, each arm loads its final.pt back and writes N's per-cell query
sets (e13_common.n_urls, the raw decode, blocks 20-22).
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
for p in (str(ROOT), str(HERE), str(ROOT / "research" / "native")):
    if p not in sys.path:
        sys.path.insert(0, p)

import e13_common as E13  # noqa: E402

MULTIPLIER = {"A1": 1, "A2": 4}
WARMUP_SHARE, EVAL_SHARE = 0.025, 0.05
CHANGED = ("steps", "warmup", "eval_every")
#: e12_final's recorded encoded training rows (data/e12/train/e12_final/train_result.json).
E12_ROWS = {"benign": 2_551_259, "phishing": 238_707}
E12_DRAWS = 20_000 * 256
#: Fixed by the dry run before any arm (PREREG_E13 section 2.5; recorded in Status):
#: A1 1,659 steps (warm-up 41, evaluation every 83), A2 6,636 (166, 332).
EXPECTED_KEPT: Optional[Dict[str, int]] = {"benign": 225_533, "phishing": 13_673}
E12_FINAL_SHA256 = "968ef622d0a6e9ba3d237778635d4930e0bbed02d240a88c0796567f7d686a17"
EXPECTED_SCHEDULE = {"A1": {"steps": 1_659, "warmup": 41, "eval_every": 83},
                     "A2": {"steps": 6_636, "warmup": 166, "eval_every": 332}}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def has_query(url: str) -> bool:
    from uie.evaluation import FIELDS
    return bool(FIELDS["QRY"](url))


def exposure(label_counts: Dict[str, int]) -> Dict[str, float]:
    """e12_final's draws per encoded row, by class: under class_alpha 1.0 the
    sampler draws class c with probability label_c / sum(labels)."""
    total = sum(label_counts.values())
    return {c: E12_DRAWS * label_counts[c] / total / E12_ROWS[c] for c in E12_ROWS}


def class_weights(kept: Dict[str, int], expo: Dict[str, float]) -> Dict[str, float]:
    return {c: expo[c] * kept[c] for c in expo}


def schedule(kept: Dict[str, int], expo: Dict[str, float], arm: str,
             batch_size: int = 256) -> Dict[str, int]:
    a1 = round(sum(class_weights(kept, expo).values()) / batch_size)
    steps = MULTIPLIER[arm] * a1
    return {"steps": steps, "warmup": max(1, round(WARMUP_SHARE * steps)),
            "eval_every": max(1, round(EVAL_SHARE * steps))}


def load_e12(path: Path, expected_sha256: Optional[str]):
    import torch
    from uie.model import ModelConfig
    from uie.tokenizer import URLTokenizer
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise SystemExit(f"STOP: {path} is not e12_final's final.pt ({expected_sha256[:12]}...)")
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    tok = URLTokenizer.from_state_dict(blob["tokenizer"])
    if not (tok.presence_plan and tok.slash_plan and tok.root_slash):
        raise SystemExit("STOP: not e12_final's tokenizer (presence plan, slash plan, root slash)")
    return tok, ModelConfig(**blob["model_config"]), dict(blob["train_config"]), blob["tokenizer"]


def query_rows(urls: Sequence[str], labels, langs, corpora):
    keep = [i for i, u in enumerate(urls) if has_query(u)]
    return ([urls[i] for i in keep], [labels[i] for i in keep], [langs[i] for i in keep],
            [corpora[i] for i in keep])


def encode(data: Path, tok, max_len: int):
    from uie.kaggle_train import STREAMS, VAL_STREAMS, load_streams
    from uie.train import encode_corpus
    tr_all = load_streams(data, STREAMS)
    label_counts = tr_all[4]
    tr = query_rows(*tr_all[:4])
    va = query_rows(*load_streams(data, VAL_STREAMS)[:4])
    tr_enc, tr_dropped = encode_corpus(tr[0], tr[1], tok, max_len, tr[2], tr[3])
    va_enc, va_dropped = encode_corpus(va[0], va[1], tok, max_len, va[2], va[3])
    kept = {c: len(tr_enc.get(c, [])) for c in E12_ROWS}
    rows = {"train_query_rows": {c: tr[1].count(c) for c in E12_ROWS},
            "val_query_rows": {c: va[1].count(c) for c in E12_ROWS},
            "dropped_over_max_len": tr_dropped, "dropped_over_max_len_val": va_dropped,
            "label_counts": label_counts}
    return tr_enc, va_enc, kept, rows


def run(data: Path, e12: Path, arm: str, out: Path, tables: Path, *, device=None,
        steps: Optional[int] = None, expected_sha256: Optional[str] = E12_FINAL_SHA256,
        expect_kept: Optional[Dict[str, int]] = EXPECTED_KEPT, n_gen: Optional[int] = None,
        g: int = E13.G) -> dict:
    """Train one arm from E12's start, then write N's query sets from its final.pt.
    `steps`, `expected_sha256=None`, `expect_kept=None` and `n_gen` exist for the
    tests' tiny model only."""
    import torch
    from uie.controlled import StateSuffixTables
    from uie.kaggle_train import new_model_e12
    from uie.suffixes import SuffixTable
    from uie.train import TrainConfig, URLDataset, load_checkpoint, save_checkpoint, train

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, cfg, e12_train, e12_tok = load_e12(e12, expected_sha256)
    tr_enc, va_enc, kept, rows = encode(data, tok, cfg.max_len)
    if expect_kept is None and steps is None:
        raise SystemExit("STOP: the kept rows are not registered yet (run --dry-run first)")
    if expect_kept is not None and kept != expect_kept:
        raise SystemExit(f"STOP (V): kept query rows {kept}, registered {expect_kept}")
    expo = exposure(rows["label_counts"])
    sched = schedule(kept, expo, arm, e12_train["batch_size"])
    if expect_kept is not None and sched != EXPECTED_SCHEDULE[arm]:
        raise SystemExit(f"STOP (V): schedule {sched}, registered {EXPECTED_SCHEDULE[arm]}")
    if steps is not None:
        sched = {"steps": steps, "warmup": max(1, round(WARMUP_SHARE * steps)),
                 "eval_every": max(1, round(EVAL_SHARE * steps))}
    train_cfg = TrainConfig(**{**e12_train, **sched})
    weights = class_weights(kept, expo)
    train_set = URLDataset(tr_enc, label_counts=weights)
    val_set = URLDataset(va_enc, label_counts=weights)
    print(f"E13 N {arm}: kept query rows {kept}; exposure per row {expo}; steps "
          f"{sched['steps']}, warm-up {sched['warmup']}, eval every {sched['eval_every']}",
          flush=True)
    model, prints = new_model_e12(cfg, tok, e12_train["seed"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = train(model, train_set, val_set, tok, train_cfg, device=device, out_dir=out,
                   log_every=100)
    save_checkpoint(model, tok, train_cfg, out / "final.pt")
    train_minutes = round((time.time() - t0) / 60, 1)
    finite = all(math.isfinite(v) for v in result["final"].values())
    history = result["history"]
    del model

    final, ftok, _ = load_checkpoint(out / "final.pt", device=device)
    final.to(device).eval()
    assert ftok.state_dict() == e12_tok, "final.pt's tokenizer is not e12_final's"
    shipped = SuffixTable.load(Path(tables) / "suffix_table.json")
    state_tables = StateSuffixTables.load(Path(tables) / "state_suffix_tables.json")
    presence_slash = json.loads((Path(tables) / "presence_slash_table.json").read_text())["table"]
    t1 = time.time()
    files: Dict[str, dict] = {}
    for j in E13.n_jobs((arm,)):
        if j["g"] >= g:
            continue
        urls: List[str] = E13.n_urls(final, ftok, j["cell"], j["g"], shipped, presence_slash,
                                     state_tables, device=device, n=n_gen)
        (out / j["file"]).write_text(json.dumps(urls), encoding="utf-8")
        files[j["file"]] = {**{k: j[k] for k in ("cell", "label", "state", "g", "seed")},
                            "n": len(urls), "with_query": sum(map(has_query, urls)),
                            "sha256": sha256(out / j["file"])}
        print(f"  {j['file']}: {len(urls)} URLs, {files[j['file']]['with_query']} with a query",
              flush=True)
    record = {
        "registration": E13.REGISTRATION + " (section 2.5)", "status": "DIAGNOSTIC", "arm": arm,
        "e12_final_sha256": sha256(e12), "kept_query_rows": kept, **rows,
        "exposure_per_row": expo, "class_weights": weights, "schedule": sched,
        "train_config": {k: getattr(train_cfg, k) for k in e12_train},
        "changed_from_e12": {k: [e12_train[k], getattr(train_cfg, k)] for k in CHANGED},
        "start": prints, "vocab_size": tok.vocab_size, "final": result["final"],
        "finite": finite, "history": history, "train_minutes": train_minutes,
        "best_sha256": sha256(out / "best.pt") if (out / "best.pt").exists() else None,
        "final_sha256": sha256(out / "final.pt"),
        "generation": {"from": "final.pt", "decode": "raw (temperature 1.0)", "files": files,
                       "minutes": round((time.time() - t1) / 60, 1)},
    }
    (out / "N_RECORD.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"E13 N {arm}: trained in {train_minutes} min (final {result['final']}, finite "
          f"{finite}); generation {record['generation']['minutes']} min", flush=True)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--e12", type=Path, required=True)
    ap.add_argument("--arm", choices=tuple(MULTIPLIER))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tables", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="count and encode only: print both arms' schedules")
    args = ap.parse_args()
    if args.dry_run:
        tok, cfg, e12_train, _ = load_e12(args.e12, None)
        _, _, kept, rows = encode(args.data, tok, cfg.max_len)
        expo = exposure(rows["label_counts"])
        print(json.dumps({"e12_final_sha256": sha256(args.e12), "kept_query_rows": kept, **rows,
                          "exposure_per_row": expo, "class_weights": class_weights(kept, expo),
                          "schedules": {a: schedule(kept, expo, a, e12_train["batch_size"])
                                        for a in MULTIPLIER}}, indent=1))
        return
    if not (args.arm and args.out and args.tables):
        ap.error("--arm, --out and --tables are required unless --dry-run")
    run(args.data, args.e12, args.arm, args.out, args.tables)


if __name__ == "__main__":
    main()
