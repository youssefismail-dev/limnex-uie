"""
uie/corpus/merge_phishing.py

Split the phishing feed into name and structure corpora — grouped by HOST.

Why the host, not the registrable domain
----------------------------------------
73% of phishing URLs sit on shared hosting, where each attacker gets a
SUBDOMAIN and the registrable domain belongs to the platform. `vercel.app` is
not an attacker's name; `bankx-verify.vercel.app` is. Grouping by registrable
domain therefore collapses thousands of distinct invented names into one:

    by registrable domain    94,234 groups   3.76 urls/group
    by HOST                 240,815 groups   1.47 urls/group

The host grouping recovers **2.5x more attacker-chosen names** for the name
corpus, and it is the same asymmetry the corpus audit recorded: benign names
things in the LABEL, phishing names things in the SUBDOMAIN.

Why there is no enrichment step
-------------------------------
Phishing's 47.4% bare-homepage rate was assumed to be a feed artifact — the
feed reporting a campaign's landing page and missing deeper ones — and a
Wayback scrape was scoped to repair it. Measurement killed that:

    within multi-url hosts, first sighting   mean depth 0.979   bare 50.9%
    ... later sightings                      mean depth 1.009   bare 44.9%
    observed <1 day                          mean depth 1.430
    observed 30+ days                        mean depth 0.919
    shared hosting (73% of urls)             mean depth 0.843   bare 51.3%
    own registrable domain                   mean depth 1.211   bare 37.0%

If the feed truncated at the landing page, later sightings would be deeper.
They are deeper by 0.03, and depth FALLS with observation span. The mechanism is
in the last two rows: a kit on shared hosting IS one page, so there is no deeper
path to recover. Shallowness is a property of how phishing is deployed, not a
gap in the data, and a generator that produced deep phishing paths would be
producing URLs that do not exist.

What the structure corpus is instead
------------------------------------
The multi-page kits were always in the corpus, drowned out by 231,000
single-page hosts. Selecting on page count surfaces them:

    hosts with >=3 urls    9,690 hosts   86,709 urls   mean depth 2.045   bare 4.5%

which sits alongside the benign structure corpus at 2.08 / 2.9%. Real examples:
session-ID credential harvesters (`/(S(...))/Main/Login`), IPFS-hosted kits,
community-page impersonation.

This is the same error the benign merge made — hash-sampling 5% of ALL domains
picked mostly single-page sites — and the same fix: select by page count, using
the unit that carries the invented name.
"""
from __future__ import annotations

import collections
import gzip
import json
import random
from pathlib import Path
from typing import Dict, List

from uie.factorization import parse
from uie.splits import (Record, build_random_split, group_key_host,
                        load_split, save_split)

__all__ = ["merge_phishing"]

#: Minimum URLs on a host for it to carry learnable path structure. At 3 the
#: corpus reads mean depth 2.045 / bare 4.5%; below it the single-page kits
#: dominate and drag the mean to 0.94 / 47.4%.
STRUCTURE_MIN_URLS = 3
#: Max URLs kept per structure host, matching the benign structure quota.
STRUCTURE_QUOTA = 40


def host_of(url: str) -> str:
    p = parse(url)
    if not p.registrable_label:
        return ""
    return ".".join(tuple(p.subdomains) + (p.registrable_label, p.public_suffix))


def build_phishing_split(by_host: Dict[str, List[dict]], out: Path, *,
                         seed: int = 20260806):
    """A host-keyed split, one record per host."""
    records = [Record(url=rows[0]["url"], label="phishing", observed="")
               for rows in by_host.values()]
    split = build_random_split(records, seed=seed, key=group_key_host,
                               name="random_host_disjoint_phishing")
    save_split(split, out / "split_phishing_host.json.gz")
    return split


def merge_phishing(phishing: Path, split_path: Path, out: Path, *,
                   seed: int = 20260806) -> dict:
    """`split_path` is IGNORED for phishing — see `build_phishing_split`."""
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)

    by_host: Dict[str, List[dict]] = collections.defaultdict(list)
    with open(phishing, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            host = host_of(row["url"])
            if host:
                by_host[host].append(row)

    # Phishing gets its OWN split, keyed on the HOST. Keying on the registrable
    # domain assigns whole platforms to single partitions, and two samples of
    # REAL phishing then separate at 0.9138 on full URLs against benign's 0.5020
    # through identical machinery. See `group_key_host`.
    split = build_phishing_split(by_host, out, seed=seed)

    name_rows: Dict[str, List[dict]] = collections.defaultdict(list)
    struct_rows: Dict[str, List[dict]] = collections.defaultdict(list)
    n_struct_hosts = 0

    for host, rows in by_host.items():
        part = split.partition_of(rows[0]["url"])
        if part not in ("train", "val"):
            continue

        pick = rng.choice(rows)
        name_rows[part].append({**pick, "label": "phishing", "corpus": "name"})

        if len(rows) >= STRUCTURE_MIN_URLS:
            n_struct_hosts += 1
            chosen = (rows if len(rows) <= STRUCTURE_QUOTA
                      else rng.sample(rows, STRUCTURE_QUOTA))
            for r in chosen:
                struct_rows[part].append(
                    {**r, "label": "phishing", "corpus": "structure"})

    def depth(u):
        return len([s for s in (parse(u).path or "").split("/") if s])

    def stats(rows):
        d = [depth(r["url"]) for r in rows]
        if not d:
            return {"rows": 0}
        return {"rows": len(d), "mean_depth": round(sum(d) / len(d), 3),
                "bare_pct": round(100 * sum(1 for k in d if k == 0) / len(d), 2)}

    manifest: Dict[str, object] = {
        "seed": seed, "hosts": len(by_host),
        "registrable_domains": len({".".join(h.split(".")[-2:]) for h in by_host}),
        "structure_min_urls": STRUCTURE_MIN_URLS,
        "structure_quota": STRUCTURE_QUOTA,
        "structure_hosts": n_struct_hosts,
        "split_hash": split.content_hash(),
        "outputs": {},
    }
    for stem, store in (("phishing_name", name_rows),
                        ("phishing_structure", struct_rows)):
        for part in ("train", "val"):
            rows = store.get(part, [])
            rng.shuffle(rows)
            path = out / f"{stem}_{part}.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            manifest["outputs"][f"{stem}_{part}"] = {
                **stats(rows), "bytes": path.stat().st_size}

    (out / "phishing_merge.manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--phishing", type=Path,
                    default=Path("data/corpus/phishing.jsonl"))
    ap.add_argument("--split", type=Path,
                    default=Path("data/corpus2/split_random.json.gz"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus2"))
    args = ap.parse_args()
    m = merge_phishing(args.phishing, args.split, args.out)
    print(f"hosts {m['hosts']:,} (vs {m['registrable_domains']:,} registrable "
          f"domains) — {m['hosts']/max(m['registrable_domains'],1):.1f}x more names")
    print(f"structure hosts (>= {m['structure_min_urls']} urls): "
          f"{m['structure_hosts']:,}\n")
    for k, v in m["outputs"].items():
        if v.get("rows"):
            print(f"  {k:<28}{v['rows']:>8,} rows  depth {v['mean_depth']:.3f}  "
                  f"bare {v['bare_pct']:.1f}%")
