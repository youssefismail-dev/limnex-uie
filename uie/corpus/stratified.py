"""
uie/corpus/stratified.py

Suffix-stratified block sampling over the Common Crawl index.

Why this exists
---------------
E0 drew blocks by picking a uniformly random shard and a uniformly random byte
offset inside it. That is uniform over INDEX BYTES, and index bytes are not
distributed the way names are:

    measured on the E0 corpus:  top-10 suffixes = 69.9% of URLs, 251 suffixes,
                                suffix entropy 4.358 bits

The cause is visible directly in the index. CDXJ is SURT-sorted and SURT
reverses the host, so the public suffix is the LEADING sort key and every shard
covers a contiguous alphabetical run of suffixes. Probing CC-MAIN-2026-30:

    shard   0  ai        shard 160  de
    shard   2  ar        shard 200  ir
    shard  10  br        shard 240  org
    shard  20  ch        shard 280  si
    shard  40..140  com  shard 298  xn--j1amh
                         shard 299  za

`.com` spans roughly shards 30-155 — about 42% of the index. A uniform draw
therefore gives `.com` ~42% of the budget before a single URL is read. That is
an artifact of the sampling procedure, not a fact about how the Internet is
named.

What this module does instead
-----------------------------
Because a shard IS a suffix range, allocating blocks across shards IS allocating
across suffixes. The map is built once per crawl by probing each shard, and the
budget is then split across suffix strata with a TEMPERATURE-FLATTENED weight:

    weight(suffix) proportional to shards(suffix) ** alpha,   alpha = 0.5

Why not uniform over suffixes: `.com` really is the largest namespace on the
web, and a corpus that gives `.ai` and `.com` equal weight teaches a false
Internet. Why not proportional (alpha = 1): that reproduces the 42% and is what
we are fixing. Alpha interpolates between the two, and is the same device used
to balance high- and low-resource languages in multilingual LM training rather
than something invented here. At alpha = 0.5, `.com` stays the single largest
stratum while falling from ~42% to roughly 15%.

What this module deliberately does NOT do
-----------------------------------------
No hand-written per-suffix quotas, no CMS blocklist, no "interesting TLD" list.
Those would encode an author's belief about what the web looks like into the
data collected to find that out. The only input is the index's own structure.
"""
from __future__ import annotations

import collections
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (ClassVar, Dict, Iterator, List, Optional, Sequence,
                    Tuple)

from uie.corpus.commoncrawl import (CrawlIndex, CrawlRecord, SampleStats,
                                    _WINDOW_BYTES, _get)

__all__ = ["ShardMap", "shard_weights", "allocate", "sample_stratified"]

#: Flattening exponent. 1.0 reproduces the uniform-over-bytes bias, 0.0 makes
#: every suffix stratum equal regardless of real size. 0.5 keeps the ordering of
#: the web's namespaces while removing the long head.
DEFAULT_ALPHA = 0.5

#: Backoff after an index request fails, doubling per consecutive failure.
#: Sustained extraction gets rate-limited; the correct response is to slow
#: down, not to spend the retry budget in seconds.
_BACKOFF_BASE_S = 0.5
_BACKOFF_MAX_S = 30.0

#: Also pace SUCCESSFUL requests. The first production run sustained ~45
#: blocks/min and was refused partway through; a short pause keeps the rate
#: under whatever threshold that was, at a cost of a few percent of wall time.
_PACE_S = 0.05

#: Retries for the per-shard SIZE probe. A failure here costs the shard's whole
#: block allocation, so it is worth several attempts before giving up.
_SIZE_ATTEMPTS = 4
#: Reads of one draw's offset before the draw is given up (each read is itself _get's retries).
_DRAW_FETCHES = 5


@dataclass
class ShardMap:
    """Which public suffixes live in which index shards, for one crawl.

    Built by probing the MIDDLE of each shard rather than its start: a shard
    boundary can fall inside a suffix, so the first block of shard N may still
    be the tail of shard N-1's suffix. The midpoint is the single most
    representative cheap probe.
    """

    crawl: str
    #: shard index -> the suffixes observed in its probe block, most common first
    shards: Dict[int, List[str]] = field(default_factory=dict)
    n_shards: int = 0
    seconds: float = 0.0

    #: A map missing shards is not merely incomplete, it is BIASED, and silently
    #: so. Shards are probed in index order and the index is SURT-sorted, so a
    #: run that dies partway keeps a contiguous alphabetical PREFIX of the TLD
    #: space. One real occurrence: a build interrupted by network trouble saved
    #: 97/300 shards covering `ai`..`com` and nothing after it. That file loaded
    #: without complaint, reported 18 strata, and produced a plausible-looking
    #: allocation that would have collected a corpus with no `de`, `jp`, `org`,
    #: `ru` or `uk` in it at all.
    MIN_COVERAGE: ClassVar[float] = 0.95

    @property
    def coverage(self) -> float:
        return len(self.shards) / self.n_shards if self.n_shards else 0.0

    @property
    def complete(self) -> bool:
        return self.coverage >= self.MIN_COVERAGE

    # --- construction -----------------------------------------------------

    @classmethod
    def build(cls, crawl: str, *, lines: int = 400, progress: bool = True,
              attempts: int = 3, backoff: float = 5.0) -> "ShardMap":
        """Probe every shard, retrying transient failures.

        Retries exist because a dropped shard is a biased map, not a smaller
        one. `attempts` passes are made over whatever is still missing, with a
        growing pause between them, so a rate limit or a blip costs time rather
        than correctness.
        """
        index = CrawlIndex(crawl)
        obj = cls(crawl=crawl, n_shards=index.n_shards)
        started = time.time()

        def probe(shard: int) -> bool:
            size = index._size(shard)
            if size is None or size <= _WINDOW_BYTES:
                return False
            offset = size // 2
            try:
                blob = _get(index._shard_url(shard),
                            {"Range": f"bytes={offset}-{offset + _WINDOW_BYTES - 1}"})
                text = index._resync(blob)
            except Exception:
                return False
            if not text:
                return False
            counts: collections.Counter = collections.Counter()
            for line in text.splitlines()[:lines]:
                head = line.split(" ", 1)[0]
                if ")" in head:
                    counts[head.split(",", 1)[0]] += 1
            if counts:
                obj.shards[shard] = [s for s, _ in counts.most_common()]
                return True
            return False

        for attempt in range(attempts):
            missing = [s for s in range(obj.n_shards) if s not in obj.shards]
            if not missing:
                break
            if attempt and progress:
                print(f"  retry {attempt}: {len(missing)} shards still missing",
                      flush=True)
            if attempt:
                time.sleep(backoff * attempt)
            for i, shard in enumerate(missing):
                probe(shard)
                if progress and i % 25 == 0:
                    print(f"  pass {attempt}: {i}/{len(missing)} "
                          f"({len(obj.shards)}/{obj.n_shards} mapped)", flush=True)

        obj.seconds = round(time.time() - started, 1)
        return obj

    # --- strata -----------------------------------------------------------

    def strata(self) -> Dict[str, List[int]]:
        """suffix -> the shards it appears in.

        A shard is credited to every suffix present in its probe, so boundary
        shards belong to both neighbours. That over-counts slightly and is the
        safe direction: it never leaves a suffix unreachable.
        """
        out: Dict[str, List[int]] = collections.defaultdict(list)
        for shard, suffixes in self.shards.items():
            for suffix in suffixes:
                out[suffix].append(shard)
        return dict(out)

    def dominant(self) -> Dict[int, str]:
        return {shard: suffixes[0] for shard, suffixes in self.shards.items()}

    # --- persistence ------------------------------------------------------

    def save(self, path: Path, *, allow_partial: bool = False) -> None:
        """Refuses to write an incomplete map unless explicitly forced.

        The guard is here rather than only at load time because the file itself
        is the hazard: an incomplete map on disk looks exactly like a complete
        one to anything that later picks it up.
        """
        if not self.complete and not allow_partial:
            raise ValueError(
                f"{self.crawl}: only {len(self.shards)}/{self.n_shards} shards "
                f"mapped ({100*self.coverage:.1f}% < "
                f"{100*self.MIN_COVERAGE:.0f}%). Shards are probed in index "
                f"order and the index is SURT-sorted, so a partial map is a "
                f"contiguous SLICE OF THE TLD ALPHABET, not a random subset. "
                f"Rebuild it; do not sample from it.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "crawl": self.crawl, "n_shards": self.n_shards,
            "seconds": self.seconds, "coverage": round(self.coverage, 4),
            "complete": self.complete,
            "shards": {str(k): v for k, v in sorted(self.shards.items())},
        }, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, allow_partial: bool = False) -> "ShardMap":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(crawl=blob["crawl"], n_shards=blob["n_shards"],
                  seconds=blob.get("seconds", 0.0),
                  shards={int(k): v for k, v in blob["shards"].items()})
        if not obj.complete and not allow_partial:
            raise ValueError(
                f"{path}: incomplete shard map "
                f"({len(obj.shards)}/{obj.n_shards}, "
                f"{100*obj.coverage:.1f}%) — sampling from it would collect a "
                f"contiguous slice of the TLD alphabet. Rebuild it.")
        return obj


def shard_weights(shard_map: ShardMap, *,
                  alpha: float = DEFAULT_ALPHA) -> Dict[int, float]:
    """Per-shard sampling probabilities, flattened by suffix stratum.

    Two stages, because the quantity being flattened is the SUFFIX, not the
    shard: first give each suffix a share proportional to `shards ** alpha`,
    then spread that suffix's share evenly over the shards it occupies. A
    suffix holding 126 shards and one holding 1 therefore end up with block
    counts in a ratio of sqrt(126) rather than 126.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    strata = shard_map.strata()
    if not strata:
        return {}
    weights = {suffix: len(shards) ** alpha for suffix, shards in strata.items()}
    total = sum(weights.values())
    out: Dict[int, float] = collections.defaultdict(float)
    for suffix, shards in strata.items():
        share = weights[suffix] / total
        for shard in shards:
            out[shard] += share / len(shards)
    return dict(out)


def allocate(shard_map: ShardMap, n_blocks: int, *,
             alpha: float = DEFAULT_ALPHA, seed: int = 0) -> Dict[int, int]:
    """Draw `n_blocks` shard slots from the flattened distribution.

    This SAMPLES rather than partitions, and the difference is not cosmetic.

    The first version floored each shard's fractional quota and handed the
    remainder to the largest fractional parts. That silently excluded WIDE
    strata at small budgets: `.com` spans 126 shards, so at 200 blocks its
    per-shard quota is 200 x 0.101 / 126 = 0.16, which floors to zero, while
    every single-shard stratum sat at 0.86 and swept the entire remainder. The
    200-block pilot drew **zero** `.com` blocks against an intended 10.1%, and
    reported a suffix entropy inflated by its absence. Above ~2,000 blocks the
    same code was correct, so the defect would have survived into any run large
    enough to matter and corrupted every small diagnostic run forever.

    A multinomial draw is unbiased at EVERY budget: the expected count for each
    shard is exactly `n_blocks * p`, whether that is 0.16 or 1,638. Seeded, so
    the corpus stays reproducible — E0's gate requires a stranger to be able to
    regenerate the URL list, which cannot itself be redistributed.

    Coverage at small budgets is probabilistic, and that is arithmetic rather
    than a flaw: 40 blocks cannot cover 73 strata. Budget at least ~10x the
    stratum count when broad coverage matters.
    """
    weights = shard_weights(shard_map, alpha=alpha)
    if not weights or n_blocks <= 0:
        return {}

    shards = sorted(weights)
    probs = [weights[s] for s in shards]
    rng = random.Random(seed)
    out: Dict[int, int] = collections.Counter()
    for shard in rng.choices(shards, weights=probs, k=n_blocks):
        out[shard] += 1
    return dict(out)


def sample_stratified(crawl: str, n_blocks: int, *, seed: int,
                      shard_map: ShardMap,
                      alpha: float = DEFAULT_ALPHA,
                      stats: Optional[SampleStats] = None
                      ) -> Iterator[CrawlRecord]:
    """Yield records from `n_blocks` blocks drawn under the stratified plan.

    Deterministic for a given (crawl, n_blocks, seed, alpha, map), which E0's
    reproducibility gate requires: the URL list cannot be redistributed, so a
    stranger has to be able to regenerate it.
    """
    index = CrawlIndex(crawl)
    stats = stats if stats is not None else SampleStats()
    plan = allocate(shard_map, n_blocks, alpha=alpha, seed=seed)

    for shard in sorted(plan):
        rng = random.Random(seed + shard * 7919)

        # Retry the SIZE probe before abandoning the shard. One failure here
        # used to discard the shard's ENTIRE allocation — roughly 14 blocks —
        # while counting as a single error, so 76 transient failures cost
        # ~1,064 blocks and sank CC-MAIN-2026-12 to 66.5% completion. `_size`
        # caches successes but not failures, so a retry genuinely re-probes and
        # almost always succeeds.
        size = None
        for probe_attempt in range(_SIZE_ATTEMPTS):
            size = index._size(shard)
            if size is not None and size > _WINDOW_BYTES:
                break
            stats.n_http_errors += 1
            if probe_attempt < _SIZE_ATTEMPTS - 1:
                time.sleep(min(_BACKOFF_BASE_S * (2 ** probe_attempt),
                               _BACKOFF_MAX_S))
        if size is None or size <= _WINDOW_BYTES:
            continue
        drawn = 0
        attempts = 0
        consecutive = 0
        while drawn < plan[shard] and attempts < plan[shard] * 8 + 16:
            attempts += 1
            stats.n_draws += 1
            offset = rng.randrange(0, size - _WINDOW_BYTES)
            # A failed read retries the SAME offset. Drawing a fresh one instead
            # (as before) shifted every later draw of the shard, and through the
            # extractor's per-domain reservoir the whole crawl's kept URLs: a
            # clean-room rebuild that met 289 transient errors kept only 66% of
            # the original's URLs (D2 step 6). Now a run that loses no draw is
            # the same run, however many retries it needed.
            blob = None
            for _ in range(_DRAW_FETCHES):
                try:
                    blob = _get(index._shard_url(shard),
                                {"Range": f"bytes={offset}-{offset + _WINDOW_BYTES - 1}"})
                    break
                except RuntimeError:
                    stats.n_http_errors += 1
                    # Back off rather than hammer. A sustained run gets rate-limited
                    # by the index, and retrying immediately burns the attempt
                    # budget in seconds — which is how a production run "completed"
                    # seven crawls with zero blocks each. Failures are the signal to
                    # slow down, not to try harder.
                    consecutive += 1
                    time.sleep(min(_BACKOFF_BASE_S * (2 ** min(consecutive, 6)),
                                   _BACKOFF_MAX_S))
            if blob is None:
                stats.n_lost_draws += 1      # this draw is gone: the run is no longer the seed's
                continue
            consecutive = 0
            text = index._resync(blob)
            if not text:
                stats.n_failed_resync += 1
                continue
            drawn += 1
            stats.n_blocks += 1
            stats.shards_hit.add(shard)
            if _PACE_S:
                time.sleep(_PACE_S)
            yield from CrawlIndex._parse(text, crawl, stats)
