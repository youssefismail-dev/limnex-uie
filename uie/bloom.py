"""
uie/bloom.py

A dependency-free Bloom filter, used for one job: the novelty check.

Why this exists
---------------
`Gate` measures memorization by asking whether a generated URL appears in the
training corpus. That question must be asked against EVERY training URL, not
against whatever subset happened to be sampled into a training bundle.

The distinction became load-bearing when benign sampling moved from "random
1.2M of 5.28M" to a per-label cap. Both keep ~1M URLs, so both leave ~4.3M
training URLs outside the bundle — and a novelty check that only knows the
bundle would score a copy of any of those as an invention. Novelty is a
pre-registered criterion, so letting a sampling change quietly inflate it
would corrupt the result the criterion exists to protect.

Why a Bloom filter rather than the URLs
---------------------------------------
5.58M normalized URLs are ~350 MB of text and ~45 MB even as 64-bit hashes,
which does not compress (hashes are incompressible by construction). The same
membership test at a 0.1% false-positive rate costs ~10 MB.

The error direction is the safe one. A Bloom filter never reports "absent" for
something present, so it never misses a genuine copy. It only ever produces
false POSITIVES — a novel URL wrongly called memorized — which UNDER-states
novelty. A metric that can only understate the thing it is guarding is a metric
that cannot be gamed by the machinery reporting it.
"""
from __future__ import annotations

import gzip
import hashlib
import math
import struct
from pathlib import Path
from typing import Iterable, Optional

__all__ = ["BloomFilter"]

_MAGIC = b"UIEBLOOM1"


class BloomFilter:
    """Fixed-capacity approximate set membership.

    Sized at construction from the expected item count and the tolerated false
    positive rate; adding far more than `capacity` items degrades the rate
    silently, so `n_added` is stored and reported rather than left implicit.
    """

    def __init__(self, capacity: int, error_rate: float = 0.001) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 < error_rate < 1.0:
            raise ValueError("error_rate must lie in (0, 1)")
        self.capacity = int(capacity)
        self.error_rate = float(error_rate)
        # Standard sizing: m = -n ln p / (ln 2)^2, k = (m/n) ln 2.
        self.n_bits = max(8, int(math.ceil(
            -capacity * math.log(error_rate) / (math.log(2) ** 2))))
        self.n_bits += (-self.n_bits) % 8            # whole bytes
        self.n_hashes = max(1, int(round(self.n_bits / capacity * math.log(2))))
        self.bits = bytearray(self.n_bits // 8)
        self.n_added = 0

    # --- hashing ----------------------------------------------------------

    def _indices(self, item: str):
        """Kirsch-Mitzenmacher double hashing: h_i = h1 + i*h2.

        Two independent hashes generate k indices with the same asymptotic
        false-positive rate as k independent hashes, so the cost is one digest
        rather than k. `h2` is forced odd to keep it coprime with the (even)
        bit count, which stops the probe sequence from cycling early.
        """
        digest = hashlib.blake2b(item.encode("utf-8", "surrogatepass"),
                                 digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little") | 1
        for i in range(self.n_hashes):
            yield (h1 + i * h2) % self.n_bits

    # --- set operations ---------------------------------------------------

    def add(self, item: str) -> None:
        for index in self._indices(item):
            self.bits[index >> 3] |= 1 << (index & 7)
        self.n_added += 1

    def update(self, items: Iterable[str]) -> "BloomFilter":
        for item in items:
            self.add(item)
        return self

    def __contains__(self, item: str) -> bool:
        return all(self.bits[i >> 3] & (1 << (i & 7)) for i in self._indices(item))

    def __len__(self) -> int:
        return self.n_added

    @property
    def expected_error_rate(self) -> float:
        """False-positive rate at the CURRENT load, not the design load.

        Reported rather than assumed: a filter loaded past its capacity is
        still usable, but the novelty number it produces carries a different
        error bar and that has to be visible in the manifest.
        """
        k, m, n = self.n_hashes, self.n_bits, self.n_added
        if not n:
            return 0.0
        return (1.0 - math.exp(-k * n / m)) ** k

    # --- persistence ------------------------------------------------------

    def save(self, path: Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = _MAGIC + struct.pack("<QIQd", self.n_bits, self.n_hashes,
                                      self.n_added, self.error_rate)
        with gzip.open(path, "wb") as handle:
            handle.write(header)
            handle.write(bytes(self.bits))
        return path.stat().st_size

    @classmethod
    def load(cls, path: Path) -> "BloomFilter":
        # Sniff for the gzip magic rather than trusting the extension. Kaggle
        # DECOMPRESSES gzip members while processing an uploaded dataset, so a
        # bundle shipped as `known_urls.bloom.gz` arrives as `known_urls.bloom`
        # holding raw bytes. `resolve()` already handles this for the JSONL
        # files; reading the header covers the name AND the contents at once.
        raw = Path(path).read_bytes()
        blob = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        if not blob.startswith(_MAGIC):
            raise ValueError(f"{path} is not a uie bloom filter")
        offset = len(_MAGIC)
        n_bits, n_hashes, n_added, error_rate = struct.unpack_from(
            "<QIQd", blob, offset)
        offset += struct.calcsize("<QIQd")
        # Reconstruct without re-deriving the size from capacity: the stored
        # geometry is authoritative, because a filter loaded with a different
        # m or k would silently answer every query wrongly.
        obj = cls.__new__(cls)
        obj.capacity = n_added or 1
        obj.error_rate = error_rate
        obj.n_bits = n_bits
        obj.n_hashes = n_hashes
        obj.bits = bytearray(blob[offset:offset + n_bits // 8])
        obj.n_added = n_added
        if len(obj.bits) != n_bits // 8:
            raise ValueError("truncated bloom filter")
        return obj
