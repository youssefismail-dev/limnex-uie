"""
uie/corpus/build_split2.py

Freeze the train/val/test split for the new corpus, and emit the per-population
held-out pools the evaluation needs.

One map, applied everywhere
---------------------------
The split is a single global `domain -> partition` assignment covering BOTH
classes and BOTH benign corpora. That is not a convenience — it is the only
thing preventing a contamination path the two-corpus design creates: split the
name and structure corpora independently and a domain lands in name-train and
structure-val, so the model is scored on paths belonging to a site whose name it
memorised.

Phishing domains are assigned by the same map at the same time. Building the
benign split first and the phishing split later would be two maps again.

Why this can run before phishing enrichment
-------------------------------------------
Enrichment adds PATHS to domains that already exist in the feed; it does not
introduce new domains. Every enriched URL therefore inherits the partition its
domain already has, and the frozen map stays valid.

What it emits
-------------
    split_random.json.gz            the map, hash-verified
    name_train / name_val           name corpus, partitioned
    structure_train / structure_val the structure corpus, partitioned
    phishing_train / phishing_val   phishing, partitioned

`run_eval` requires `name_val` and `structure_val` separately: each field is
scored against the population it was TRAINED on, so a single pooled val set
cannot be substituted.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from uie.factorization import parse
from uie.splits import Record, build_random_split, group_key, save_split

__all__ = ["build"]

SOURCES = {
    "name": ("name.jsonl.gz", "benign"),
    "structure": ("structure.jsonl.gz", "benign"),
}


def _iter(path: Path, key: str = "url") -> Iterable[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def build(corpus: Path, phishing: Path, out: Path, *,
          seed: int = 20260806) -> dict:
    out.mkdir(parents=True, exist_ok=True)

    # --- one record per DOMAIN, both classes ------------------------------
    seen: Dict[str, str] = {}
    for stem, (fname, label) in SOURCES.items():
        path = corpus / fname
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in _iter(path):
            seen.setdefault(group_key(row["url"]), label)
        print(f"  scanned {fname}: {len(seen):,} domains so far", flush=True)

    n_benign = len(seen)
    if phishing.is_file():
        for row in _iter(phishing):
            seen.setdefault(group_key(row["url"]), "phishing")
    print(f"  scanned phishing: {len(seen)-n_benign:,} new domains", flush=True)

    # `build_random_split` dedups by domain itself, but feeding it one record
    # per domain keeps 3.5M rows out of memory for no loss of behaviour.
    records = [Record(url=f"http://{d}/", label=lab, observed="")
               for d, lab in seen.items()]
    split = build_random_split(records, seed=seed,
                               name="random_domain_disjoint_corpus2")
    split_path = out / "split_random.json.gz"
    save_split(split, split_path)
    counts = split.counts()
    print(f"  split: {counts}  hash {split.content_hash()[:16]}...", flush=True)

    # --- partition every source through THE SAME map ----------------------
    manifest: Dict[str, object] = {
        "seed": seed, "split_hash": split.content_hash(),
        "domains": len(seen), "partitions": counts, "outputs": {},
    }
    jobs: List[Tuple[str, Path]] = [
        ("name", corpus / "name.jsonl.gz"),
        ("structure", corpus / "structure.jsonl.gz"),
    ]
    if phishing.is_file():
        jobs.append(("phishing", phishing))

    for stem, path in jobs:
        writers = {}
        n = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
        try:
            for part in ("train", "val"):
                writers[part] = gzip.open(out / f"{stem}_{part}.jsonl.gz", "wt",
                                          encoding="utf-8", newline="\n")
            for row in _iter(path):
                part = split.partition_of(row["url"])
                if part in writers:
                    row.setdefault("label",
                                   "phishing" if stem == "phishing" else "benign")
                    row["corpus"] = stem
                    writers[part].write(json.dumps(row, ensure_ascii=False) + "\n")
                n[part or "unassigned"] += 1
        finally:
            for w in writers.values():
                w.close()
        manifest["outputs"][stem] = n
        print(f"  {stem:<10} train {n['train']:>9,}  val {n['val']:>8,}  "
              f"test {n['test']:>8,}", flush=True)

    (out / "split.manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus2"))
    ap.add_argument("--phishing", type=Path,
                    default=Path("data/corpus/phishing.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus2"))
    args = ap.parse_args()
    m = build(args.corpus, args.phishing, args.out)
    print(f"\nsplit hash {m['split_hash'][:16]}...  "
          f"{m['domains']:,} domains  {m['partitions']}")
