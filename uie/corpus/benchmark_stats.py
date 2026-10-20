"""
uie/corpus/benchmark_stats.py

Compute the numbers published in BENCHMARK.md, so the document is generated from
the corpus rather than transcribed by hand.

Everything is reported per class and per stratum. Pooled statistics are what let
a model be excellent on English .com and useless on Japanese .jp while still
looking fine, which is precisely the failure an Internet World Model must not be
allowed to hide.
"""
from __future__ import annotations

import bisect
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from uie.factorization import URLParts, parse
from uie.metrics import BY_SUFFIX_KIND, URLStats
from uie.splits import group_key


def _iter_benign(corpus: Path) -> Iterator[Tuple[str, str, str]]:
    """(url, era, languages) for every benign row."""
    for path in sorted(glob.glob(str(corpus / "shards" / "*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield row["url"], row.get("era", ""), row.get("languages", "und")


def _iter_phishing(corpus: Path) -> Iterator[Tuple[str, str]]:
    """(url, first_seen) for every phishing row."""
    path = corpus / "phishing.jsonl"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield row["url"], row.get("first_seen", "")


def _depth_auc(phish_depths: List[int], benign_depths: List[int]) -> float:
    """AUC of a single trivial feature: 'shallower => phishing'.

    Reported because a high value is EVIDENCE OF A COLLECTION ARTIFACT, not of a
    model's skill. Feeds report the entry point a victim clicks; crawlers index
    pages found by following links. If one integer separates the classes, any
    model trained here can win without learning anything.
    """
    ordered = sorted(benign_depths)
    n = len(ordered)
    if not n or not phish_depths:
        return float("nan")
    wins = ties = 0
    for depth in phish_depths:
        lo = bisect.bisect_left(ordered, depth)
        hi = bisect.bisect_right(ordered, depth)
        wins += n - hi              # benign strictly deeper => correct ordering
        ties += hi - lo
    return (wins + 0.5 * ties) / (len(phish_depths) * n)


def collect(corpus: Path, *, depth_sample: int = 400_000) -> Dict[str, object]:
    classes: Dict[str, URLStats] = {"benign": URLStats(), "phishing": URLStats()}
    eras: Dict[str, URLStats] = {}
    strata: Dict[str, Dict[str, URLStats]] = {"benign": {}, "phishing": {}}
    domains: Dict[str, set] = {"benign": set(), "phishing": set()}
    languages: Counter = Counter()
    phish_month: Counter = Counter()
    depths: Dict[str, List[int]] = {"benign": [], "phishing": []}

    def absorb(url: str, label: str, parts: URLParts) -> None:
        classes[label].add(parts, url)
        strata[label].setdefault(BY_SUFFIX_KIND(parts), URLStats()).add(parts, url)
        domains[label].add(group_key(url))
        if len(depths[label]) < depth_sample:
            depths[label].append(parts.path_depth)

    for url, era, langs in _iter_benign(corpus):
        parts = parse(url)
        absorb(url, "benign", parts)
        if era:
            eras.setdefault(era, URLStats()).add(parts, url)
        for lang in str(langs).split(","):
            languages[lang.strip() or "und"] += 1

    for url, first_seen in _iter_phishing(corpus):
        absorb(url, "phishing", parse(url))
        if first_seen:
            phish_month[first_seen[:7]] += 1

    lang_total = sum(languages.values()) or 1
    return {
        "classes": {k: v.as_dict() for k, v in classes.items()},
        "domains": {k: len(v) for k, v in domains.items()},
        "eras": {k: v.as_dict() for k, v in sorted(eras.items())},
        "strata": {cls: {name: st.as_dict() for name, st in sorted(s.items())}
                   for cls, s in strata.items()},
        "languages_top": {k: round(v / lang_total, 5)
                          for k, v in languages.most_common(15)},
        "n_languages": len(languages),
        "phishing_by_month": dict(sorted(phish_month.items())),
        "depth_shortcut_auc": round(_depth_auc(depths["phishing"],
                                               depths["benign"]), 4),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus/benchmark_stats.json"))
    args = ap.parse_args()

    stats = collect(args.corpus)
    args.out.write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  depth shortcut AUC: {stats['depth_shortcut_auc']}")
    for cls, d in stats["classes"].items():
        print(f"  {cls:<9} n={d['n']:>9,}  domains={stats['domains'][cls]:>7,}  "
              f"bare={100*d['bare_homepage_rate']:.1f}%  depth={d['mean_path_depth']:.2f}")
