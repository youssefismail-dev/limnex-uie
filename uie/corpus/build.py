"""
uie/corpus/build.py

Build the benign corpus: era-matched, uniformly sampled Common Crawl.

Why era-matching
----------------
The phishing side spans two populations with different dates -- a live feed from
2024-10 onward and historical archives from roughly 2019-2022. Pairing both with
a single 2026 benign sample would teach the model that CURRENT-ERA STYLING IS
BENIGN and older styling is phishing: a temporal confound, and the same class of
artifact already found in the curated corpora (where the class<->path-depth
correlation flips direction between datasets). Common Crawl publishes monthly
back to 2008, so each phishing era is paired with contemporaneous benign data.

Why no cap on URLs per domain
-----------------------------
Benign runs ~58 URLs per registrable domain and phishing ~3.8. Capping benign to
match would "balance" the corpus by teaching a falsehood: the asymmetry is a real
property of the phenomena -- phishing domains are disposable and used once, real
sites are crawled deeply. Class imbalance belongs in the training sampler, not in
the corpus. The corpus records what is true.

Resumability
------------
One shard file per crawl, written atomically. A crash costs one crawl, not the
run. Re-running skips crawls whose output already exists, so the job can be
stopped and restarted freely.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from uie.corpus.commoncrawl import CrawlIndex, SampleStats

__all__ = ["Era", "ERA_A_LIVE", "ERA_B_HISTORICAL", "build_benign"]


@dataclass(frozen=True)
class Era:
    """A phishing population and the benign crawls contemporaneous with it."""

    name: str
    rationale: str
    crawls: Sequence[str]


#: Paired with the live OpenPhish feed (first_seen 2024-10 .. present).
ERA_A_LIVE = Era(
    name="live",
    rationale="contemporaneous with the OpenPhish live feed (2024-10 onward)",
    crawls=("CC-MAIN-2024-42", "CC-MAIN-2024-51", "CC-MAIN-2025-13",
            "CC-MAIN-2025-26", "CC-MAIN-2025-38", "CC-MAIN-2025-51",
            "CC-MAIN-2026-12", "CC-MAIN-2026-30"),
)

#: Paired with the historical archives (GramBeddings / Kaggle / Mendeley).
ERA_B_HISTORICAL = Era(
    name="historical",
    rationale="contemporaneous with the archived phishing corpora (~2019-2022)",
    crawls=("CC-MAIN-2019-04", "CC-MAIN-2019-30", "CC-MAIN-2019-51",
            "CC-MAIN-2020-24", "CC-MAIN-2020-50", "CC-MAIN-2021-25",
            "CC-MAIN-2021-49", "CC-MAIN-2022-49"),
)


def _shard_path(out_dir: Path, era: str, crawl: str) -> Path:
    return out_dir / "shards" / f"{era}__{crawl}.jsonl"


def _write_atomic(path: Path, lines: List[str]) -> None:
    """Write via a temp file and rename, so a partial file never looks complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(lines)
    tmp.replace(path)


def build_benign(out_dir: Path, *, eras: Sequence[Era] = (ERA_A_LIVE, ERA_B_HISTORICAL),
                 blocks_per_crawl: int = 200, seed: int = 20260803,
                 verbose: bool = True) -> dict:
    """Sample every crawl in every era and write one JSONL shard per crawl.

    Each crawl gets a distinct derived seed, so adding or removing a crawl leaves
    the others' draws untouched and previously built shards stay reproducible.
    """
    out_dir = Path(out_dir)
    per_crawl: List[dict] = []

    for era in eras:
        for i, crawl in enumerate(era.crawls):
            path = _shard_path(out_dir, era.name, crawl)
            if path.exists():
                if verbose:
                    print(f"  skip  {era.name}/{crawl} (already built)", flush=True)
                per_crawl.append(_shard_meta(path, era, crawl, skipped=True))
                continue

            started = time.time()
            stats = SampleStats()
            index = CrawlIndex(crawl)
            lines: List[str] = []
            try:
                for rec in index.sample(blocks_per_crawl,
                                        seed=seed + abs(hash(crawl)) % 100_003,
                                        stats=stats):
                    lines.append(json.dumps({
                        "url": rec.url,
                        "timestamp": rec.timestamp,
                        "languages": rec.languages,
                        "crawl": rec.crawl,
                        "era": era.name,
                    }, ensure_ascii=False) + "\n")
            except RuntimeError as exc:
                # A crawl that no longer serves its index must not kill the run.
                if verbose:
                    print(f"  FAIL  {era.name}/{crawl}: {exc}", flush=True)
                continue

            _write_atomic(path, lines)
            meta = _shard_meta(path, era, crawl, stats=stats,
                               seconds=time.time() - started)
            per_crawl.append(meta)
            if verbose:
                print(f"  done  {era.name}/{crawl}: {meta['urls']:,} URLs from "
                      f"{meta['blocks']} blocks in {meta['seconds']:.0f}s", flush=True)

    manifest = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "blocks_per_crawl": blocks_per_crawl,
        "eras": [{"name": e.name, "rationale": e.rationale,
                  "crawls": list(e.crawls)} for e in eras],
        "shards": per_crawl,
        "total_urls": sum(s["urls"] for s in per_crawl),
    }
    (out_dir / "benign.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _shard_meta(path: Path, era: Era, crawl: str, *,
                stats: Optional[SampleStats] = None,
                seconds: float = 0.0, skipped: bool = False) -> dict:
    """Per-shard provenance, including a content hash so a rebuild is verifiable."""
    digest = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            digest.update(line)
            n += 1
    return {
        "era": era.name,
        "crawl": crawl,
        "file": path.name,
        "urls": n,
        "blocks": stats.n_blocks if stats else None,
        "shards_hit": len(stats.shards_hit) if stats else None,
        "seconds": round(seconds, 1),
        "sha256": digest.hexdigest(),
        "reused": skipped,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the era-matched benign corpus.")
    ap.add_argument("--out", default="data/corpus", type=Path)
    ap.add_argument("--blocks", type=int, default=200,
                    help="blocks per crawl (~31.8 distinct domains each)")
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    man = build_benign(args.out, blocks_per_crawl=args.blocks, seed=args.seed)
    print(f"\nTotal: {man['total_urls']:,} benign URLs across "
          f"{len(man['shards'])} crawls -> {args.out}")
