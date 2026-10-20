"""
uie/corpus/commoncrawl.py

Uniform sampling of the Common Crawl CDXJ index — the sole source of UIE's
learned benign distribution.

Why Common Crawl at all
-----------------------
Curated URL corpora are assembled to train CLASSIFIERS, which mostly care about
the domain, so their path structure is a collection artifact rather than a
property of the web. Measured against this module's own output, curated benign
corpora carry 12-17% bare homepages (two of them 98.6% and 100.0%) where the real
web carries 3.7%. A model trained on them would learn how a dataset was built,
not how the web names things.

Why the naive sampling method is wrong
--------------------------------------
CDXJ is sorted by SURT — the *reversed* domain — so the TLD is the LEADING sort
key and every shard is a contiguous TLD range. Reading the leading bytes of a few
shards therefore samples a handful of TLD neighbourhoods, not the web: an early
spike of this code produced ee=16.9% and mk=12.7%, which is Estonia and North
Macedonia standing in for the internet. Reading each shard's prefix also biases
toward alphabetically-early domains *within* each TLD.

The method used here
--------------------
Each shard is a concatenation of independently-gzipped blocks of ~3000 lines. We
seek to a RANDOM byte offset, resync forward to the next block boundary, and
decode that block. Blocks hold a near-constant number of lines, so uniform block
sampling is uniform line sampling — and it needs no `cluster.idx` download.

Resync is VALIDATED rather than assumed: the gzip magic can occur by chance
inside compressed data, so a candidate offset is accepted only if it both
decompresses and parses as CDXJ.

The statistical caveat that must travel with every result
---------------------------------------------------------
A block is ~3000 *contiguous* SURT lines, so URLs within a block are highly
correlated — they are mostly the same handful of hosts. The effective sample size
for TLD and language SHARES is the number of blocks drawn, not the number of URLs
returned. `SampleStats.n_blocks` is reported for exactly this reason. Path-depth
statistics are far better behaved (they vary within a host) and were stable
across two samples with entirely different TLD mixes.
"""
from __future__ import annotations

import http.client
import json
import random
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Set
from urllib.parse import urlsplit

__all__ = ["CrawlRecord", "SampleStats", "CrawlIndex", "list_crawls"]

_DATA_HOST = "https://data.commoncrawl.org"
_COLLINFO = "https://index.commoncrawl.org/collinfo.json"

#: Gzip member header: magic (1f 8b) plus DEFLATE compression method (08).
_GZIP_MAGIC = b"\x1f\x8b\x08"

#: Bytes fetched per draw. Must comfortably exceed one compressed block so that
#: a resync landing just past a boundary still captures a whole member.
_WINDOW_BYTES = 400_000

#: zlib window size that selects gzip framing rather than raw DEFLATE.
_ZLIB_GZIP = 31

_TIMEOUT_S = 180
_RETRIES = 3
_BACKOFF_S = 2.0     # pause before retry k: k * _BACKOFF_S seconds


@dataclass(frozen=True)
class CrawlRecord:
    """One indexed capture, reduced to the fields UIE actually consumes."""

    url: str
    timestamp: str            # CDXJ capture time, YYYYMMDDhhmmss
    languages: str            # comma-separated ISO-639-3, "und" when unknown
    crawl: str                # e.g. "CC-MAIN-2026-30" — provenance for §8

    @property
    def tld(self) -> Optional[str]:
        """Public-suffix-naive TLD. `None` for IP-literal hosts, which have no
        TLD — an early spike reported '.132' as a TLD by taking the last octet."""
        host = urlsplit(self.url).hostname or ""
        if "." not in host or host.replace(".", "").isdigit():
            return None
        return host.rsplit(".", 1)[-1]


@dataclass
class SampleStats:
    """Bookkeeping for a sampling run.

    `n_blocks` is not a curiosity: it is the honest denominator for any TLD or
    language share computed from `n_records` correlated URLs.
    """

    n_draws: int = 0              # random (shard, offset) attempts
    n_blocks: int = 0             # draws that resynced and parsed — effective n
    n_lines: int = 0              # index lines seen
    n_records: int = 0            # lines surviving the filters
    n_failed_resync: int = 0
    n_http_errors: int = 0
    n_lost_draws: int = 0          # draws given up after every retry of their offset
    shards_hit: Set[int] = field(default_factory=set)

    def summary(self) -> str:
        return (f"{self.n_records:,} records from {self.n_blocks} blocks "
                f"across {len(self.shards_hit)} shards "
                f"({self.n_lines:,} index lines, {self.n_failed_resync} resync "
                f"failures, {self.n_http_errors} HTTP errors)")


def _get(url: str, headers: Optional[dict] = None, method: str = "GET") -> bytes:
    """HTTP fetch with bounded retries. Raises on final failure."""
    last: Optional[Exception] = None
    for attempt in range(_RETRIES):
        if attempt:
            time.sleep(_BACKOFF_S * attempt)
        try:
            req = urllib.request.Request(url, headers=headers or {}, method=method)
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                return resp.read() if method == "GET" else bytes(
                    resp.headers.get("Content-Length", "0"), "ascii")
        # http.client.HTTPException covers a response cut short (IncompleteRead), which is not
        # an OSError: it escaped the retries and ended a whole crawl's extraction (D2 step 6)
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            last = exc
    raise RuntimeError(f"GET {url} failed after {_RETRIES} attempts: {last}")


def list_crawls() -> List[dict]:
    """All published crawls, newest first. Each has `id` and `name`.

    Era-matching depends on this: benign data must come from crawls
    contemporaneous with the phishing sources it is paired against, or the model
    learns that current-era styling is benign.
    """
    return json.loads(_get(_COLLINFO).decode("utf-8"))


class CrawlIndex:
    """Random-access sampler over one crawl's CDXJ index."""

    def __init__(self, crawl: str, *, max_shards: int = 300) -> None:
        self.crawl = crawl
        self._base = f"{_DATA_HOST}/cc-index/collections/{crawl}/indexes"
        self._max_shards = max_shards
        self._sizes: dict[int, int] = {}
        self._n_shards: Optional[int] = None
        #: HEAD failures that were NOT 404s — network, not absence.
        self._transient = 0

    # --- shard discovery ---------------------------------------------------

    def _shard_url(self, shard: int) -> str:
        return f"{self._base}/cdx-{shard:05d}.gz"

    def _size(self, shard: int) -> Optional[int]:
        """Byte length of a shard, or None if it does not exist.

        A 404 means the shard genuinely is not there; anything else — timeout,
        refused connection, 5xx, rate limit — means we could not find out. The
        two are recorded separately in `_transient` because conflating them
        turns a network problem into a false statement about the crawl.
        """
        if shard in self._sizes:
            return self._sizes[shard]
        try:
            req = urllib.request.Request(self._shard_url(shard), method="HEAD")
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                size = int(resp.headers["Content-Length"])
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                self._transient += 1
            return None
        except Exception:
            self._transient += 1
            return None
        self._sizes[shard] = size
        return size

    @property
    def n_shards(self) -> int:
        """Shard count, found by probing downward from `max_shards`.

        Crawls have varied in shard count over the years, so this is discovered
        rather than assumed — era-matching means old crawls will be sampled too.
        """
        if self._n_shards is None:
            self._transient = 0
            for probe in range(self._max_shards - 1, -1, -1):
                if self._size(probe) is not None:
                    self._n_shards = probe + 1
                    break
            else:
                # Distinguish "this crawl has no index" from "we could not
                # reach the index". Reporting the second as the first sent a
                # map build looking for a crawl that demonstrably exists, and
                # the misleading error cost more time than the outage did.
                if self._transient:
                    raise RuntimeError(
                        f"Could not reach the index for crawl {self.crawl!r}: "
                        f"{self._transient} of {self._max_shards} HEAD requests "
                        f"failed for reasons other than 404. This is a network "
                        f"or rate-limit problem, not a missing crawl.")
                raise RuntimeError(f"No index shards found for crawl {self.crawl!r}")
        return self._n_shards

    # --- block decoding ----------------------------------------------------

    @staticmethod
    def _resync(blob: bytes) -> str:
        """Decode the first VALID gzip member in `blob`.

        The gzip magic can appear by chance inside compressed bytes, so a
        candidate is accepted only if it decompresses AND its first line looks
        like CDXJ (a SURT key, which always contains ')', followed by JSON).
        Returns "" when no valid member is found.
        """
        pos = 0
        while True:
            idx = blob.find(_GZIP_MAGIC, pos)
            if idx < 0:
                return ""
            try:
                raw = zlib.decompressobj(_ZLIB_GZIP).decompress(blob[idx:])
                text = raw.decode("utf-8", "replace")
            except (zlib.error, MemoryError):
                pos = idx + 1
                continue
            head = text.split("\n", 1)[0]
            if "{" in head and ")" in head.split(" ", 1)[0]:
                return text
            pos = idx + 1

    @staticmethod
    def _parse(text: str, crawl: str, stats: SampleStats) -> Iterator[CrawlRecord]:
        """CDXJ lines -> records, keeping only real fetchable HTML pages.

        `status == 200` and an HTML mime are what make a URL a page that existed
        and served content, rather than a redirect, error, or asset.
        """
        for line in text.splitlines():
            stats.n_lines += 1
            brace = line.find("{")
            if brace < 0:
                continue
            try:
                rec = json.loads(line[brace:])
            except json.JSONDecodeError:
                continue

            # A CDXJ line is `<surt-key> <timestamp> {<json>}`: the capture time
            # is the SECOND FIELD ON THE LINE, and most records do not repeat it
            # inside the JSON. Reading only the JSON silently yielded an empty
            # timestamp for every row, which left the temporal split with no
            # benign data at all.
            head = line[:brace].split()
            line_timestamp = head[1] if len(head) > 1 else ""
            if rec.get("status") != "200":
                continue
            if not str(rec.get("mime-detected", "")).startswith("text/html"):
                continue
            url = rec.get("url")
            if not url:
                continue
            stats.n_records += 1
            yield CrawlRecord(
                url=url,
                timestamp=line_timestamp or str(rec.get("timestamp", "")),
                languages=str(rec.get("languages") or "und"),
                crawl=crawl,
            )

    # --- sampling ----------------------------------------------------------

    def sample(self, n_blocks: int, *, seed: int,
               stats: Optional[SampleStats] = None) -> Iterator[CrawlRecord]:
        """Yield records from `n_blocks` uniformly-drawn index blocks.

        Deterministic for a given (crawl, n_blocks, seed): the draw sequence is
        seeded, so a stranger reproduces the corpus exactly — which is what E0's
        gate requires, given that the URL list itself cannot be redistributed.
        """
        stats = stats if stats is not None else SampleStats()
        rng = random.Random(seed)
        n_shards = self.n_shards

        drawn = 0
        while drawn < n_blocks:
            stats.n_draws += 1
            shard = rng.randrange(n_shards)
            size = self._size(shard)
            if size is None or size <= _WINDOW_BYTES:
                stats.n_http_errors += 1
                continue
            offset = rng.randrange(0, size - _WINDOW_BYTES)
            try:
                blob = _get(self._shard_url(shard),
                            {"Range": f"bytes={offset}-{offset + _WINDOW_BYTES - 1}"})
            except RuntimeError:
                stats.n_http_errors += 1
                continue

            text = self._resync(blob)
            if not text:
                stats.n_failed_resync += 1
                continue

            drawn += 1
            stats.n_blocks += 1
            stats.shards_hit.add(shard)
            yield from self._parse(text, self.crawl, stats)


def sample_crawls(crawls: Sequence[str], blocks_per_crawl: int, *,
                  seed: int) -> Iterator[CrawlRecord]:
    """Sample several crawls — the mechanism behind era-matching.

    Each crawl gets a distinct derived seed so that adding a crawl does not
    perturb the draws of the others, keeping earlier corpora reproducible.
    """
    for i, crawl in enumerate(crawls):
        index = CrawlIndex(crawl)
        yield from index.sample(blocks_per_crawl, seed=seed + i * 7919)
