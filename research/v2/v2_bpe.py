"""
research/v2/v2_bpe.py

Greedy byte-pair encoding over characters: deterministic, pure Python, no dependency
(research/v2/PREREG_R1.md section 3, R1-4; DESIGN_V2.md section 4.3).

A table is learned from sequences with integer weights. Each sequence is one field body -- a
path body, a query, a fragment, a host label -- so a piece never crosses a sequence:

  - at each step the adjacent pair with the largest total weight becomes one piece;
  - ties go to the pair whose two pieces come first in code-point order;
  - the base alphabet is every character in the data, so nothing is unwritable.

Pieces are identified by their strings: a merge that produces a string some other path
already made reuses that piece. Encoding applies the learned merges by rank -- always the
lowest-ranked pair present next, every occurrence left to right -- so every string has
exactly one (canonical) encoding, and decoding is concatenation: lossless by construction.

In R1 these tables only COUNT how many real words pieces would hold whole. Nothing is trained
on them.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["BPETable", "learn", "covered_words"]


class BPETable:
    """A learned table: the base alphabet and the merges, in the order they were learned."""

    def __init__(self, alphabet: Sequence[str], merges: Sequence[Tuple[str, str]],
                 meta: Optional[dict] = None) -> None:
        self.alphabet = list(alphabet)
        self.merges = [(str(a), str(b)) for a, b in merges]
        self.rank: Dict[Tuple[str, str], int] = {}
        for i, m in enumerate(self.merges):
            self.rank.setdefault(m, i)
        self.meta = dict(meta or {})

    def __len__(self) -> int:
        return len(self.merges)

    def encode(self, s: str, limit: Optional[int] = None) -> List[str]:
        """The canonical encoding of `s` under the first `limit` merges (all by default).
        Characters outside the alphabet stay single pieces."""
        limit = len(self.merges) if limit is None else min(int(limit), len(self.merges))
        pieces = list(s)
        rank = self.rank
        while len(pieces) > 1:
            best_r, best = limit, None
            for i in range(len(pieces) - 1):
                r = rank.get((pieces[i], pieces[i + 1]))
                if r is not None and r < best_r:
                    best_r, best = r, (pieces[i], pieces[i + 1])
            if best is None:
                break
            a, b = best
            merged, out, i, n = a + b, [], 0, len(pieces)
            while i < n:
                if i < n - 1 and pieces[i] == a and pieces[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(pieces[i])
                    i += 1
            pieces = out
        return pieces

    def encode_at(self, s: str, limits: Sequence[int]) -> Dict[int, List[str]]:
        """`encode(s, limit=k)` for every k in `limits`, in one pass.

        The rank loop applies merges in strictly increasing rank: a pair can only contain a
        piece made by an earlier merge, so every pair a merge creates has a higher rank than
        the merge itself. The state just before the first merge of rank >= k is therefore
        exactly the k-merge encoding."""
        ks = sorted({min(int(k), len(self.merges)) for k in limits})
        want = {int(k): min(int(k), len(self.merges)) for k in limits}
        snap: Dict[int, List[str]] = {}
        pieces, li, rank = list(s), 0, self.rank
        while True:
            best_r, best = len(self.merges), None
            for i in range(len(pieces) - 1):
                r = rank.get((pieces[i], pieces[i + 1]))
                if r is not None and r < best_r:
                    best_r, best = r, (pieces[i], pieces[i + 1])
            while li < len(ks) and (best is None or ks[li] <= best_r):
                snap[ks[li]] = list(pieces)
                li += 1
            if best is None or li == len(ks):
                break
            a, b = best
            merged, out, i, n = a + b, [], 0, len(pieces)
            while i < n:
                if i < n - 1 and pieces[i] == a and pieces[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(pieces[i])
                    i += 1
            pieces = out
        return {k: snap[v] for k, v in want.items()}

    # --- persistence ----------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({"alphabet": self.alphabet, "merges": self.merges, "meta": self.meta},
                          ensure_ascii=False, sort_keys=True)

    def sha256(self) -> str:
        """The table's fingerprint: its alphabet and merges, not its meta."""
        core = json.dumps({"alphabet": self.alphabet, "merges": self.merges},
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(core.encode("utf-8")).hexdigest()

    def save(self, path) -> str:
        Path(path).write_text(self.to_json(), encoding="utf-8")
        return self.sha256()

    @classmethod
    def load(cls, path) -> "BPETable":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(blob["alphabet"], [tuple(m) for m in blob["merges"]], blob.get("meta"))


def learn(sequences: Mapping[str, int], n_merges: int, *, log_every: int = 1000,
          log: Optional[Callable[[str], None]] = print) -> BPETable:
    """Greedy BPE over `sequences` (string -> positive integer weight), up to `n_merges`
    distinct merges. Deterministic: the same input gives the same table on any machine."""
    alphabet = sorted({ch for s, w in sequences.items() if w > 0 for ch in s})
    sym: List[str] = list(alphabet)
    sid: Dict[str, int] = {c: i for i, c in enumerate(sym)}
    words: List[List[int]] = []
    weights: List[int] = []
    for s, w in sorted(sequences.items()):
        if s and int(w) > 0:
            words.append([sid[c] for c in s])
            weights.append(int(w))

    pairs: Dict[Tuple[int, int], int] = defaultdict(int)
    where: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for wi, w in enumerate(words):
        f = weights[wi]
        seen = set()
        for p in zip(w, w[1:]):
            pairs[p] += f
            if p not in seen:
                seen.add(p)
                where[p].append(wi)
    heap = [(-c, sym[a], sym[b], a, b) for (a, b), c in pairs.items()]
    heapq.heapify(heap)

    merges: List[Tuple[str, str]] = []
    done = set()
    started = time.time()
    while len(merges) < n_merges:
        best = None
        while heap:
            negc, sa, sb, a, b = heapq.heappop(heap)
            c = pairs.get((a, b), 0)
            if c <= 0:
                continue
            if c != -negc:                       # a stale count: put it back as it is now
                heapq.heappush(heap, (-c, sa, sb, a, b))
                continue
            best = (a, b, sa, sb)
            break
        if best is None:
            break                                # nothing left to merge
        a, b, sa, sb = best
        text = sa + sb
        new = sid.get(text)
        if new is None:
            new = len(sym)
            sym.append(text)
            sid[text] = new
        if (sa, sb) not in done:
            done.add((sa, sb))
            merges.append((sa, sb))

        delta: Dict[Tuple[int, int], int] = defaultdict(int)
        for wi in sorted(set(where.pop((a, b), ()))):
            w = words[wi]
            out, i, n, hit = [], 0, len(w), False
            while i < n:
                if i < n - 1 and w[i] == a and w[i + 1] == b:
                    out.append(new)
                    i += 2
                    hit = True
                else:
                    out.append(w[i])
                    i += 1
            if not hit:
                continue
            f = weights[wi]
            for p in zip(w, w[1:]):
                delta[p] -= f
            fresh = set()
            for p in zip(out, out[1:]):
                delta[p] += f
                if new in p and p not in fresh:
                    fresh.add(p)
                    where[p].append(wi)
            words[wi] = out
        pairs.pop((a, b), None)
        for p, d in delta.items():
            if p == (a, b) or d == 0:
                continue
            c = pairs.get(p, 0) + d
            if c > 0:
                pairs[p] = c
                if d > 0:
                    heapq.heappush(heap, (-c, sym[p[0]], sym[p[1]], p[0], p[1]))
            else:
                pairs.pop(p, None)
        if log and log_every and len(merges) % log_every == 0 and (sa, sb) == merges[-1]:
            log(f"    {len(merges):>6} merges  ({time.time() - started:.0f}s; "
                f"heap {len(heap):,}, pairs {len(pairs):,})")
    return BPETable(alphabet, merges, {"n_sequences": len(words), "weight": sum(weights),
                                       "seconds": round(time.time() - started, 1)})


def covered_words(text: str, pieces: Sequence[str], spans: Sequence[Tuple[int, int]]
                  ) -> List[bool]:
    """For each word span [start, end) in `text`, whether one piece holds the whole span.
    `pieces` must be an encoding of `text` (their concatenation equals it)."""
    bounds, pos = [], 0
    for p in pieces:
        bounds.append((pos, pos + len(p)))
        pos += len(p)
    if pos != len(text):
        raise ValueError("the pieces do not spell the text")
    out, j = [], 0
    for start, end in spans:
        while j < len(bounds) and bounds[j][1] <= start:
            j += 1
        out.append(j < len(bounds) and bounds[j][0] <= start and end <= bounds[j][1])
    return out
