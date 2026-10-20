"""
uie/corpus/build_splits.py

Freeze the corpus splits.

Memory note — why domains, not URLs
-----------------------------------
The corpus is ~6.8M rows, which is too many to hold as objects just to compute a
split. But every split decision here is made at DOMAIN level, and the builders
consume only three facts per domain: its class, its earliest observation and its
latest one. So the corpus is streamed once into a per-domain summary (~150k
entries), and the builders are handed two synthetic rows per domain — one at its
first observation, one at its last.

That is EXACT, not an approximation: `build_temporal_split` derives nothing from
a domain's rows except their min and max timestamps, and `build_random_split`
uses only the first sighting's class.

Timestamp normalisation
-----------------------
Benign rows carry CDXJ timestamps (`20260701120000`) and phishing rows carry
ISO-8601 (`2026-08-01T00:00:00Z`). Those do NOT compare lexicographically with
each other, so a cutoff applied to the mixed corpus would silently mis-sort one
class entirely. Both are normalised to `YYYY-MM-DD` here before any comparison.
"""
from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from uie.splits import (Record, build_random_split, build_temporal_split,
                        group_key, save_split)

#: Splits the live era roughly 15 months train / 7 months eval, and places the
#: whole historical era in train. Chosen so eval sits entirely inside the window
#: where BOTH classes are observed — an eval period with no phishing data would
#: measure nothing.
DEFAULT_CUTOFF = "2026-01-01"


def _normalize_date(stamp: str) -> str:
    """Any supported timestamp -> `YYYY-MM-DD`. Empty stays empty."""
    if not stamp:
        return ""
    if len(stamp) >= 14 and stamp[:14].isdigit():        # CDXJ
        return f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"
    return stamp[:10]                                     # ISO-8601 prefix


def crawl_to_date(crawl: str) -> str:
    """`CC-MAIN-2026-30` -> the Monday of ISO week 30, 2026.

    A fallback for rows extracted before the CDXJ timestamp bug was fixed, and a
    legitimate observation date in its own right: every URL in a crawl was
    captured during that crawl's window. Precision is ~2 weeks, which is finer
    than any cutoff this project applies.
    """
    parts = crawl.split("-")
    if len(parts) < 4 or not (parts[2].isdigit() and parts[3].isdigit()):
        return ""
    try:
        return datetime.fromisocalendar(int(parts[2]), int(parts[3]), 1).strftime("%Y-%m-%d")
    except ValueError:
        return ""


def summarize_corpus(corpus_dir: Path) -> Dict[str, Tuple[str, str, str]]:
    """Stream the corpus into {domain: (label, first_seen, last_seen)}."""
    summary: Dict[str, Tuple[str, str, str]] = {}

    def observe(url: str, label: str, stamp: str) -> None:
        date = _normalize_date(stamp)
        if not date:
            return
        key = group_key(url)
        prev = summary.get(key)
        if prev is None:
            summary[key] = (label, date, date)
        else:
            # A domain's class is its FIRST sighting, matching build_random_split.
            summary[key] = (prev[0], min(prev[1], date), max(prev[2], date))

    for path in sorted(glob.glob(str(corpus_dir / "shards" / "*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                # Prefer the per-URL capture time; fall back to the crawl window
                # for rows extracted before the CDXJ timestamp fix.
                stamp = row.get("timestamp", "") or crawl_to_date(row.get("crawl", ""))
                observe(row["url"], "benign", stamp)

    phishing = corpus_dir / "phishing.jsonl"
    if phishing.exists():
        with open(phishing, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                observe(row["url"], "phishing", row.get("first_seen", ""))

    return summary


def _as_records(summary: Dict[str, Tuple[str, str, str]]) -> List[Record]:
    """Two synthetic rows per domain — see the module docstring on exactness."""
    out: List[Record] = []
    for domain, (label, first, last) in summary.items():
        url = f"https://{domain}/"
        out.append(Record(url=url, label=label, observed=first))
        if last != first:
            out.append(Record(url=url, label=label, observed=last))
    return out


def main(corpus_dir: Path, out_dir: Path, cutoff: str = DEFAULT_CUTOFF,
         seed: int = 20260803) -> None:
    summary = summarize_corpus(corpus_dir)
    print(f"{len(summary):,} distinct domains summarised")
    by_class: Dict[str, int] = {}
    for label, _, _ in summary.values():
        by_class[label] = by_class.get(label, 0) + 1
    print("  " + "  ".join(f"{k}={v:,}" for k, v in sorted(by_class.items())))

    records = _as_records(summary)

    random_split = build_random_split(records, seed=seed)
    man_r = save_split(random_split, out_dir / "split_random.json.gz")
    print(f"\nrandom domain-disjoint  -> {man_r['domains_per_partition']}")
    print(f"  sha256 {man_r['content_sha256'][:16]}...")

    temporal = build_temporal_split(records, cutoff=cutoff)
    man_t = save_split(temporal, out_dir / "split_temporal.json.gz")
    print(f"\ntemporal (cutoff {cutoff}) -> {man_t['domains_per_partition']}")
    print(f"  straddling domains kept in train: "
          f"{man_t['n_straddling_domains_kept_in_train']:,}")
    print(f"  sha256 {man_t['content_sha256'][:16]}...")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Freeze the corpus splits.")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus"))
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()
    main(args.corpus, args.out, args.cutoff, args.seed)
