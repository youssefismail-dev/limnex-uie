"""
uie/corpus/extract.py

Production extraction for the two-corpus design.

Target: 1,000,000 registrable labels, era-matched to the phishing feed.

Why only live-era crawls
------------------------
The phishing corpus is 100% OpenPhish spanning 2024-10-12 to 2026-08-03. E0's
benign corpus paired that with 8 historical crawls (2019-2022) as well, to match
archived phishing corpora that were never actually collected — leaving 47.1% of
benign in a time window containing no phishing at all, and with it a clean
"2019-era web => benign" signal. Measured: historical crawls are 76.6% https,
live crawls 92.5%, phishing 64.4%.

So only the 8 crawls covering the feed's window are used. This costs nothing:
the block budget is simply drawn from 8 crawls instead of 16.

Why blocks, and how many
------------------------
The pilot measured 29.4 labels per block under stratified sampling (uniform
sampling gave 31.0), and label yield is LINEAR in blocks — each of E0's 16
shards contributed ~99% previously-unseen labels with no decay. Heaps' law
fitted WITHIN the existing corpus predicts 220x the URLs for 1M labels, but that
describes sub-sampling a fixed corpus and saturates because only 99,303 labels
exist in it. Reading new blocks is the linear regime.

Two corpora from one extraction
-------------------------------
The label field and the path field want opposite things — names want flatness,
paths want concentration — and no single URL list satisfies both. E2.1 and E2.2
each measured that trade directly. So the stream is split:

  NAME       one URL per domain, every domain. Maximum naming diversity.
  STRUCTURE  up to 40 URLs for domains with >=32 pages. Real site structure.

Structure membership needs a PAGE COUNT, so it is decided at merge time, not
here. Among qualifying domains the choice is by HASH rather than arrival order
or a running reservoir: deterministic, reproducible by a stranger, and
uncorrelated with shard order (which is TLD order, so an arrival-order rule
would silently bias the structure corpus by alphabet).

The name corpus WILL have a shallow depth distribution — every site contributes
one page, so small shallow sites dominate. That is expected and is exactly why
training must apply FIELD-SCOPED LOSS: name-corpus rows supply gradient for
[SUF]/[LBL]/[SUB] only, never for the path. Without that, this design rebuilds
E2.1's bare-homepage artifact at larger scale.

Resumability
------------
Each crawl writes its own shard file and is skipped if already complete. A
7-hour extraction that dies at hour 6 must not start over.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from uie.corpus.commoncrawl import SampleStats
from uie.corpus.stratified import DEFAULT_ALPHA, ShardMap, sample_stratified
from uie.factorization import parse

__all__ = ["LIVE_CRAWLS", "extract_crawl", "extract_all"]

#: Era-matched to OpenPhish (2024-10-12 .. 2026-08-03).
LIVE_CRAWLS = [
    "CC-MAIN-2024-42", "CC-MAIN-2024-51", "CC-MAIN-2025-13", "CC-MAIN-2025-26",
    "CC-MAIN-2025-38", "CC-MAIN-2025-51", "CC-MAIN-2026-12", "CC-MAIN-2026-30",
]

#: Minimum pages a domain must contribute to qualify for the structure corpus.
#:
#: Selection was originally a flat 5% of ALL domains by hash. That is the right
#: rule for sampling NAMES and the wrong one for sampling STRUCTURE, because
#: path depth rises steeply with page count and most domains are tiny:
#:
#:     urls/domain   domains    mean depth   bare%
#:            >=1    266,798        0.40     72.8%
#:            >=6    154,914        1.44     11.5%
#:           >=32     58,827        1.92      4.0%
#:           >=48    203,802        2.12      2.5%
#:
#: A uniform 5% is therefore dominated by single-page sites that have no
#: structure to teach, and the resulting corpus measured mean depth 1.87 / bare
#: 6.5% against a population of 2.30 / 2.90% — shallower than the web it is
#: supposed to model. Qualifying on page count gives 2.09 / 2.8%.
STRUCTURE_MIN_PAGES = 32

#: Share of QUALIFYING domains kept, by hash. Still hash-based so the choice is
#: deterministic and uncorrelated with shard order (which is TLD order).
STRUCTURE_SHARE_PCT = 20
#: Max URLs kept per structure-corpus domain.
STRUCTURE_QUOTA = 40
#: Max URLs kept per domain in the raw per-crawl file. Bounds memory without
#: pre-judging the split, which happens in the merge.
RAW_QUOTA = 48

#: Fraction of requested blocks a crawl must actually deliver to count as done.
#: Below this the index is refusing or unreachable and the run must not be
#: recorded as complete — see the check in `extract_crawl`.
MIN_BLOCK_COMPLETION = 0.90


def in_structure(domain: str, n_pages: int = 0) -> bool:
    """Structure-corpus membership: enough pages to HAVE structure, then hashed.

    `n_pages` is the domain's total page count across the whole corpus, so this
    can only be decided at MERGE time — an extraction-time rule cannot know how
    many pages a domain will contribute across eight crawls.
    """
    if n_pages < STRUCTURE_MIN_PAGES:
        return False
    digest = hashlib.blake2b(domain.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 100 < STRUCTURE_SHARE_PCT


@dataclass
class CrawlResult:
    crawl: str
    blocks: int = 0
    records: int = 0
    kept: int = 0
    domains: int = 0
    seconds: float = 0.0
    path: str = ""
    # E15 (research/e15/PREREG_E15.md section 11): the network record of the run,
    # so a parallel acquisition is auditable. Defaults keep older done-files loading.
    n_draws: int = 0
    n_http_errors: int = 0
    n_lost_draws: int = 0
    n_failed_resync: int = 0


def extract_crawl(crawl: str, n_blocks: int, out_dir: Path, *,
                  seed: int, alpha: float = DEFAULT_ALPHA,
                  cache: Path = Path("data/cache"),
                  progress_every: int = 250_000) -> CrawlResult:
    """Sample one crawl and write its shard file. Returns collection stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{crawl}.jsonl.gz"
    done = out_dir / f"{crawl}.done.json"
    if done.is_file() and path.is_file():
        blob = json.loads(done.read_text(encoding="utf-8"))
        return CrawlResult(**blob)

    shard_map = ShardMap.load(cache / f"shardmap_{crawl}.json")   # refuses partial
    rng = random.Random(seed)
    stats = SampleStats()
    kept: Dict[str, List[dict]] = {}
    seen: Dict[str, int] = {}
    n_records = 0
    started = time.time()

    for rec in sample_stratified(crawl, n_blocks, seed=seed,
                                 shard_map=shard_map, alpha=alpha, stats=stats):
        n_records += 1
        p = parse(rec.url)
        if not p.registrable_label:
            continue
        domain = f"{p.registrable_label}.{p.public_suffix}"
        count = seen.get(domain, 0) + 1
        seen[domain] = count
        bucket = kept.setdefault(domain, [])
        row = {"url": rec.url,
               "lang": str(getattr(rec, "languages", None) or "und").split(",")[0].strip() or "und",
               "ts": getattr(rec, "timestamp", "")}
        if count <= RAW_QUOTA:
            bucket.append(row)
        else:
            # Reservoir: every URL of this domain stays equally likely to
            # survive, so the kept set is not biased toward whichever block
            # happened to be drawn first.
            j = rng.randrange(count)
            if j < RAW_QUOTA:
                bucket[j] = row
        if progress_every and n_records % progress_every == 0:
            print(f"    {crawl}: {n_records:,} records, {len(kept):,} domains, "
                  f"{stats.n_blocks}/{n_blocks} blocks, "
                  f"{(time.time()-started)/60:.1f} min", flush=True)

    # A short run is a FAILED run, and must never be marked done.
    #
    # The first production attempt was refused by the index partway through
    # crawl 1 (1,228 of 4,251 blocks) and got nothing at all for crawls 2-8.
    # Every one of them was written with a `.done.json`, so the resume logic
    # would have skipped seven empty 47-byte files forever and the merge would
    # have silently produced a one-crawl corpus. This is the same failure mode
    # as a partially-built ShardMap: an incomplete artifact that looks finished.
    #
    # The bar is deliberately below 1.0 — the sampler legitimately abandons a
    # shard whose offsets keep landing outside a valid gzip member — but far
    # above "we got something".
    got = stats.n_blocks / max(n_blocks, 1)
    if got < MIN_BLOCK_COMPLETION:
        raise RuntimeError(
            f"{crawl}: only {stats.n_blocks:,}/{n_blocks:,} blocks "
            f"({100*got:.1f}% < {100*MIN_BLOCK_COMPLETION:.0f}%) — "
            f"{stats.n_http_errors:,} HTTP errors, "
            f"{stats.n_failed_resync:,} resync failures. Not writing a done "
            f"marker; the index is refusing or unreachable, so retry later "
            f"rather than resuming past this crawl.")

    n_kept = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        for domain, rows in kept.items():
            for row in rows:
                row["crawl"] = crawl
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_kept += 1

    result = CrawlResult(crawl=crawl, blocks=stats.n_blocks, records=n_records,
                         kept=n_kept, domains=len(kept),
                         seconds=round(time.time() - started, 1), path=str(path),
                         n_draws=stats.n_draws, n_http_errors=stats.n_http_errors,
                         n_failed_resync=stats.n_failed_resync,
                         n_lost_draws=stats.n_lost_draws)
    done.write_text(json.dumps(result.__dict__, indent=1) + "\n", encoding="utf-8")
    return result


def extract_all(target_labels: int, out_dir: Path, *, seed: int = 20260804,
                alpha: float = DEFAULT_ALPHA, labels_per_block: float = 29.4,
                crawls: Optional[List[str]] = None,
                cache: Path = Path("data/cache")) -> dict:
    """Run every era-matched crawl, resuming any already finished."""
    crawls = crawls or LIVE_CRAWLS
    total_blocks = int(round(target_labels / labels_per_block))
    per_crawl = total_blocks // len(crawls)
    print(f"target {target_labels:,} labels -> {total_blocks:,} blocks "
          f"({per_crawl:,} per crawl across {len(crawls)} era-matched crawls)\n",
          flush=True)

    results = []
    started = time.time()
    for i, crawl in enumerate(crawls):
        print(f"[{i+1}/{len(crawls)}] {crawl}", flush=True)
        res = extract_crawl(crawl, per_crawl, out_dir, seed=seed + i * 7919,
                            alpha=alpha, cache=cache)
        results.append(res)
        print(f"  -> {res.kept:,} urls over {res.domains:,} domains "
              f"from {res.blocks:,} blocks in {res.seconds/60:.1f} min "
              f"[{(time.time()-started)/60:.0f} min total]\n", flush=True)

    manifest = {
        "seed": seed, "alpha": alpha, "target_labels": target_labels,
        "total_blocks": total_blocks, "crawls": [r.__dict__ for r in results],
        "structure_share_pct": STRUCTURE_SHARE_PCT,
        "structure_quota": STRUCTURE_QUOTA,
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "extract.manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Production corpus extraction.")
    ap.add_argument("--target-labels", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, default=Path("data/corpus2/shards"))
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    args = ap.parse_args()

    man = extract_all(args.target_labels, args.out, alpha=args.alpha,
                      cache=args.cache)
    urls = sum(c["kept"] for c in man["crawls"])
    doms = sum(c["domains"] for c in man["crawls"])
    print(f"DONE  {urls:,} urls, {doms:,} domain-slots (pre-merge), "
          f"{man['seconds']/3600:.1f} h")
