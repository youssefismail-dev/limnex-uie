"""
uie/corpus/pack_bundle.py

Assemble the training bundle for the two-corpus, field-scoped architecture.

Replaces `pack_kaggle`, which packed a single subsampled URL list.

What goes in, and why each piece is load-bearing
------------------------------------------------
    name_train / name_val                  benign naming, 1 url per domain
    structure_train / structure_val        benign paths, multi-page sites only
    phishing_name_train / _val             phishing naming, 1 url per HOST
    phishing_structure_train / _val        phishing kits, hosts with >=3 urls

    split_random.json.gz                   benign split, DOMAIN-keyed
    split_phishing_host.json.gz            phishing split, HOST-keyed
    suffix_table.json                      per-class suffix frequencies
    known_urls.bloom.gz                    novelty check
    uie/                                   the package, so no network is needed

Two splits, not one, and that is deliberate. The invented name sits in a
different field per class — the label for benign, the subdomain for phishing —
so each class is split on the unit that carries its invention. Keying phishing
on the registrable domain put whole platforms in single partitions and made two
samples of REAL phishing separate at 0.9138 on full URLs; host-keying brought
that to 0.4791 against benign's 0.5020. Within each class both corpora share one
split, which is what prevents a domain landing in name-train and structure-val.

Why the Bloom filter is built from the SHARDS
---------------------------------------------
The bundle keeps ~2.5M of the 18.2M extracted URLs: the name corpus takes one
per domain and the structure corpus only multi-page sites. A novelty check built
from bundle rows alone would score a verbatim copy of any of the other 15.7M as
an invention. Checking against every training-partition URL under-reports
novelty slightly and can never overstate it, which is the only safe direction
for a pre-registered criterion.

Exposure arithmetic, for the record
-----------------------------------
With `class_alpha = 1` weighting by label count, and rows drawn uniformly within
a class, every ROW gets equal exposure. Each name is exactly one row, so at
20,000 steps x 256 each benign name is seen ~1.65 times. That low repetition is
the point: it is what stops the label field being memorised, which is the defect
that dominated E2, E2.1 and E2.2.
"""
from __future__ import annotations

import glob
import gzip
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

from uie.bloom import BloomFilter
from uie.gate import normalize_for_dedup
from uie.splits import load_split
from uie.suffixes import SuffixTable

__all__ = ["build"]

CORPORA = [
    "name_train", "name_val", "structure_train", "structure_val",
    "phishing_name_train", "phishing_name_val",
    "phishing_structure_train", "phishing_structure_val",
]
SPLITS = ["split_random.json.gz", "split_phishing_host.json.gz"]
BLOOM_ERROR_RATE = 0.001


def _count(path: Path) -> int:
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for _ in fh:
            n += 1
    return n


def build(corpus: Path, shards: Path, phishing: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {"corpora": {}, "splits": {}}

    for stem in CORPORA:
        src = corpus / f"{stem}.jsonl.gz"
        if not src.is_file():
            raise FileNotFoundError(f"{src} — run merge and merge_phishing first")
        dst = out / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        manifest["corpora"][stem] = {"rows": _count(dst),
                                     "bytes": dst.stat().st_size}
        print(f"  {stem:<28}{manifest['corpora'][stem]['rows']:>10,} rows", flush=True)

    for name in SPLITS:
        src = corpus / name
        if src.is_file():
            dst = out / name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            split = load_split(dst)
            manifest["splits"][name] = {"key_kind": split.key_kind,
                                        "hash": split.content_hash(),
                                        "counts": split.counts()}
            print(f"  {name:<28}{split.key_kind:>10}-keyed  "
                  f"{split.counts()}", flush=True)

    table = SuffixTable.from_files(
        [out / f"{s}.jsonl.gz" for s in CORPORA if s.endswith("_train")])
    table.save(out / "suffix_table.json")
    manifest["suffix_table"] = {cls: len(table.counts[cls])
                                for cls in table.classes()}
    for cls in table.classes():
        print(f"  suffix table  {table.summary(cls, top=6)}", flush=True)

    # Novelty against every TRAINING-partition URL, not just the bundled ones.
    print("  building novelty filter over the full training partition ...",
          flush=True)
    benign_split = load_split(out / "split_random.json.gz")
    phish_split = load_split(out / "split_phishing_host.json.gz")
    known = BloomFilter(capacity=20_000_000, error_rate=BLOOM_ERROR_RATE)
    n_seen = 0
    for path in sorted(glob.glob(str(shards / "*.jsonl.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                url = json.loads(line)["url"]
                if benign_split.partition_of(url) == "train":
                    known.add(normalize_for_dedup(url))
                    n_seen += 1
        print(f"    {Path(path).name}: {n_seen:,} so far", flush=True)
    if phishing.is_file():
        with open(phishing, encoding="utf-8") as fh:
            for line in fh:
                url = json.loads(line)["url"]
                if phish_split.partition_of(url) == "train":
                    known.add(normalize_for_dedup(url))
                    n_seen += 1
    bloom_bytes = known.save(out / "known_urls.bloom.gz")
    manifest["known_urls_bloom"] = {
        "n_added": known.n_added, "bytes": bloom_bytes,
        "expected_error_rate": round(known.expected_error_rate, 6)}
    print(f"  novelty filter {known.n_added:,} urls "
          f"({bloom_bytes/1048576:.1f} MB, "
          f"fp {known.expected_error_rate:.4%})", flush=True)

    package_src = Path(__file__).resolve().parent.parent
    package_dst = out / "uie"
    if package_dst.exists():
        shutil.rmtree(package_dst)
    shutil.copytree(package_src, package_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    (out / "bundle.manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    # Zip in the SAME step as the bundle. A zip built at a different time is how
    # a stale copy of the package gets uploaded, which fails only once it is
    # already running on the GPU.
    archive = out.parent / "uie-corpus2.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file() and "__pycache__" not in str(path):
                zf.write(path, path.relative_to(out))
    manifest["archive"] = str(archive)
    manifest["archive_bytes"] = archive.stat().st_size
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus2"))
    ap.add_argument("--shards", type=Path, default=Path("data/corpus2/shards"))
    ap.add_argument("--phishing", type=Path,
                    default=Path("data/corpus/phishing.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/bundle2"))
    args = ap.parse_args()
    m = build(args.corpus, args.shards, args.phishing, args.out)
    total = sum(v["rows"] for v in m["corpora"].values())
    print(f"\ntotal rows {total:,}")
    print(f"UPLOAD THIS FILE:  {m['archive']}  "
          f"({m['archive_bytes']/1048576:.1f} MB)")
