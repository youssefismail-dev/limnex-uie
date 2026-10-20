"""
uie/corpus/merge.py

Merge the per-crawl shards into the NAME and STRUCTURE corpora.

The union is the number that matters
------------------------------------
The extraction reports 1,154,031 domain-SLOTS across 8 crawls, but a domain
appearing in five crawls is counted five times. Stratified sampling revisits the
same suffix ranges every crawl, so overlap is expected to be higher than E0's
~99% cross-shard novelty (which was measured under uniform sampling). The union
is measured here rather than extrapolated.

Two corpora, two passes
-----------------------
  NAME       one URL per domain, EVERY domain.       Maximum naming diversity.
  STRUCTURE  up to 40 URLs for domains with >=32     Real per-site structure.
             pages, then 20% of those by hash.

Structure membership needs a page count, and a domain's page count is only known
once every crawl has been read — hence pass one. The first version selected a
flat 5% of ALL domains by hash, which is correct for sampling names and wrong
for sampling structure: depth rises steeply with page count (0.40 at one page,
2.12 at 48+), most domains are tiny, and the resulting corpus came out at mean
depth 1.87 / bare 6.5% against a population of 2.30 / 2.90% — shallower than the
web it exists to model.

Among QUALIFYING domains the choice is still by hash: deterministic, reproducible
by a stranger, and uncorrelated with shard order (which is TLD order, so an
arrival-order rule would bias the structure corpus by alphabet).

The name corpus is chosen by RESERVOIR, not "first seen": shards are read in
crawl order, so keeping the first URL per domain would bias every name-corpus
row toward the earliest crawl and toward whichever path sorts first.

Why the name corpus may look wrong, and why that is correct
-----------------------------------------------------------
Its path-depth distribution will be shallow — every site contributes one page,
so small shallow sites dominate — and its bare-homepage rate will be far above
the web's 2.9%. That is unavoidable and is exactly why training must apply
FIELD-SCOPED LOSS: name rows supply gradient for [SUF]/[LBL]/[SUB] only, never
for the path. Without that, this rebuilds E2.1's bare-homepage artifact at
larger scale.
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

from uie.corpus.extract import (STRUCTURE_MIN_PAGES, STRUCTURE_QUOTA,
                                STRUCTURE_SHARE_PCT, in_structure)
from uie.factorization import parse

__all__ = ["merge"]


def _entropy(counter) -> float:
    n = sum(counter.values())
    if not n:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c)


def merge(shards: Path, out: Path, *, seed: int = 20260805) -> dict:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)

    # PASS 1: page counts. Structure membership depends on how many pages a
    # domain contributes across ALL crawls, which cannot be known while any
    # single crawl is still being read.
    files_pre = sorted(glob.glob(str(shards / "*.jsonl.gz")))
    pages: Dict[str, int] = collections.Counter()
    for path in files_pre:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                p = parse(json.loads(line)["url"])
                if p.registrable_label:
                    pages[f"{p.registrable_label}.{p.public_suffix}"] += 1
        print(f"  counted {Path(path).name} ({len(pages):,} domains)", flush=True)
    qualifying = {d for d, n in pages.items() if in_structure(d, n)}
    print(f"  structure domains: {len(qualifying):,} of {len(pages):,} "
          f"(>= {STRUCTURE_MIN_PAGES} pages, {STRUCTURE_SHARE_PCT}% of those)",
          flush=True)

    name_pick: Dict[str, dict] = {}          # domain -> one row (reservoir k=1)
    name_seen: Dict[str, int] = {}
    struct: Dict[str, List[dict]] = {}       # domain -> up to QUOTA rows
    struct_seen: Dict[str, int] = {}

    suffixes = collections.Counter()
    langs = collections.Counter()
    depths = collections.Counter()
    per_crawl_domains: Dict[str, set] = {}
    n_rows = 0

    files = sorted(glob.glob(str(shards / "*.jsonl.gz")))
    for path in files:
        crawl = Path(path).name.split(".jsonl")[0]
        here: set = set()
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                p = parse(row["url"])
                if not p.registrable_label:
                    continue
                domain = f"{p.registrable_label}.{p.public_suffix}"
                n_rows += 1
                here.add(domain)
                suffixes[p.public_suffix or "<none>"] += 1
                langs[row.get("lang", "und")] += 1
                depths[len([s for s in (p.path or "").split("/") if s])] += 1

                # NAME: reservoir of one, uniform over every sighting.
                c = name_seen.get(domain, 0) + 1
                name_seen[domain] = c
                if c == 1 or rng.randrange(c) == 0:
                    name_pick[domain] = row

                # STRUCTURE: qualifying domains only, reservoir up to the quota.
                if domain in qualifying:
                    k = struct_seen.get(domain, 0) + 1
                    struct_seen[domain] = k
                    bucket = struct.setdefault(domain, [])
                    if k <= STRUCTURE_QUOTA:
                        bucket.append(row)
                    else:
                        j = rng.randrange(k)
                        if j < STRUCTURE_QUOTA:
                            bucket[j] = row
        per_crawl_domains[crawl] = here
        print(f"  {crawl}: {len(here):,} domains "
              f"(union so far {len(name_seen):,})", flush=True)

    union = len(name_seen)
    slots = sum(len(v) for v in per_crawl_domains.values())

    def write(rows, path: Path) -> int:
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path.stat().st_size

    name_rows = [{**r, "label": "benign", "corpus": "name"}
                 for r in name_pick.values()]
    struct_rows = [{**r, "label": "benign", "corpus": "structure"}
                   for rows in struct.values() for r in rows]
    rng.shuffle(name_rows)
    rng.shuffle(struct_rows)
    name_bytes = write(name_rows, out / "name.jsonl.gz")
    struct_bytes = write(struct_rows, out / "structure.jsonl.gz")

    def depth_stats(rows) -> dict:
        c = collections.Counter(
            len([s for s in (parse(r["url"]).path or "").split("/") if s])
            for r in rows)
        t = sum(c.values()) or 1
        return {"mean": sum(d * n for d, n in c.items()) / t,
                "bare_pct": 100 * c[0] / t,
                "ge3_pct": 100 * sum(n for d, n in c.items() if d >= 3) / t}

    total_depth = sum(d * n for d, n in depths.items()) / max(n_rows, 1)
    manifest = {
        "seed": seed, "files": len(files), "rows_read": n_rows,
        "domain_slots": slots, "union_domains": union,
        "overlap_pct": 100 * (1 - union / max(slots, 1)),
        "name": {"rows": len(name_rows), "bytes": name_bytes,
                 "depth": depth_stats(name_rows)},
        "structure_min_pages": STRUCTURE_MIN_PAGES,
        "structure_share_pct": STRUCTURE_SHARE_PCT,
        "structure_qualifying_domains": len(qualifying),
        "structure": {"rows": len(struct_rows), "domains": len(struct),
                      "bytes": struct_bytes,
                      "urls_per_domain": len(struct_rows) / max(len(struct), 1),
                      "depth": depth_stats(struct_rows)},
        "as_collected": {"mean_depth": total_depth,
                         "bare_pct": 100 * depths[0] / max(n_rows, 1)},
        "n_suffixes": len(suffixes), "suffix_entropy": _entropy(suffixes),
        "top10_suffix_share": sum(c for _, c in suffixes.most_common(10)) / max(n_rows, 1),
        "top_suffixes": suffixes.most_common(20),
        "n_langs": len(langs), "lang_entropy": _entropy(langs),
        "top_langs": langs.most_common(15),
    }
    (out / "merge.manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("data/corpus2/shards"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus2"))
    args = ap.parse_args()

    m = merge(args.shards, args.out)
    print(f"\n{'UNION LABELS':<24}{m['union_domains']:>12,}")
    print(f"{'domain slots':<24}{m['domain_slots']:>12,}  "
          f"(overlap {m['overlap_pct']:.1f}%)")
    print(f"{'name corpus':<24}{m['name']['rows']:>12,} rows")
    print(f"{'structure corpus':<24}{m['structure']['rows']:>12,} rows over "
          f"{m['structure']['domains']:,} domains "
          f"({m['structure']['urls_per_domain']:.1f}/domain)")
    print(f"\n{'suffix entropy':<24}{m['suffix_entropy']:>12.3f}  (E0: 4.358)")
    print(f"{'distinct suffixes':<24}{m['n_suffixes']:>12,}  (E0: 251)")
    print(f"{'top-10 suffix share':<24}{100*m['top10_suffix_share']:>11.1f}%  (E0: 69.9%)")
    print(f"{'language entropy':<24}{m['lang_entropy']:>12.3f}  (E0: 3.541)")
    print(f"\nas collected: mean depth {m['as_collected']['mean_depth']:.2f}, "
          f"bare {m['as_collected']['bare_pct']:.2f}%   (E0: 2.30 / 2.90%)")
    print(f"name  depth {m['name']['depth']['mean']:.2f}, "
          f"bare {m['name']['depth']['bare_pct']:.1f}%  (expected shallow)")
    print(f"struct depth {m['structure']['depth']['mean']:.2f}, "
          f"bare {m['structure']['depth']['bare_pct']:.1f}%")
