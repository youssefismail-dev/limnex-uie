"""Deterministic transformations that each destroy ONE candidate signal.

The ablation runs on the discriminator's INPUT, not on the generator. Retraining
a generator per hypothesis would cost GPU hours, reintroduce the ~0.0141
retraining noise floor measured from w1-vs-E5, and move several properties at
once -- which is how S4 produced a calibration "fix" that turned out to be an
artifact of a mask bug.

Every function here maps a path body to a path body and is applied to REAL and
GENERATED strings by the same call, so the comparison stays paired and no
per-side branch can creep in.

Two directions are needed and neither is sufficient alone:

    destructive   remove family X, see whether excess collapses   -> NECESSITY
    isolating     keep only family X, see whether excess survives -> SUFFICIENCY

A family is named only when both agree, because destroying one signal often
damages another: `scramble` kills lexical identity but also any positional
regularity inside a segment, and only the isolating direction separates them.

RANDOMISED TRANSFORMS ARE SEEDED PER STRING, from the string's own content. The
same input therefore always yields the same output, across processes and across
the real/generated sides, so a difference in excess can never come from a
difference in random draws.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Callable, Dict, List

Transform = Callable[[str], str]


def _rng(text: str, salt: str) -> random.Random:
    """Deterministic per-string RNG. Same string -> same permutation, always."""
    digest = hashlib.sha256(f"{salt}\x00{text}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _segments(body: str) -> List[str]:
    return [s for s in body.split("/") if s]


# --------------------------------------------------------------------------- #
# Identity and null controls
# --------------------------------------------------------------------------- #

def t_identity(body: str) -> str:
    """E_0. Must reproduce the published excess or the harness is misplumbed."""
    return body


def t_null(body: str) -> str:
    """A no-op that is NOT the identity function, to measure the null.

    Round-trips through a split/join so it exercises the same code path as a
    real transform while provably changing nothing. Any excess difference
    between this and `t_identity` is pure measurement noise, and that spread is
    what the family-wise thresholds are built from -- 26.3 found |sigma| of 8.84
    where 2 had been assumed, and the structural statistic repeated the mistake
    at p90 = 2.11, so no threshold here is taken on faith.
    """
    return "/".join(body.split("/"))


# --------------------------------------------------------------------------- #
# F1 lexical identity
# --------------------------------------------------------------------------- #

def t_scramble(body: str) -> str:
    """Shuffle characters WITHIN each segment. Kills F1, preserves F3/F4/F5.

    Character multiset, segment lengths, depth and delimiters are all exactly
    conserved; only the ORDER inside a segment is destroyed. If excess survives
    this, the tell cannot be word identity or morphology.
    """
    out = []
    for seg in body.split("/"):
        if len(seg) > 1:
            chars = list(seg)
            _rng(seg, "scramble").shuffle(chars)
            seg = "".join(chars)
        out.append(seg)
    return "/".join(out)


# --------------------------------------------------------------------------- #
# F2 set diversity  (applied to the SET, see `apply_set` below)
# --------------------------------------------------------------------------- #
# Diversity is not a property of any single string, so it cannot be a per-string
# transform. TF-IDF is trained on a SET, and a generator that reuses a small
# segment vocabulary is detectable even when every individual string is
# plausible. Handled by `dedup_segments` in the set-level section.


# --------------------------------------------------------------------------- #
# F3 character composition
# --------------------------------------------------------------------------- #

def t_lower(body: str) -> str:
    """Kill case. Cheap, and case was a known E3 asymmetry."""
    return body.lower()


def t_digitmask(body: str) -> str:
    """All digits -> '0'. Kills digit IDENTITY, keeps digit POSITION and count."""
    return re.sub(r"\d", "0", body)


def t_alphamask(body: str) -> str:
    """All letters -> 'a'. Kills F1/F3-alphabet, keeps F4/F5 and digit layout."""
    return re.sub(r"[^\W\d_]", "a", body, flags=re.UNICODE)


# --------------------------------------------------------------------------- #
# F4 delimiters and structure
# --------------------------------------------------------------------------- #

def t_flatten(body: str) -> str:
    """Strip '/'. Kills depth and delimiter layout, keeps the character stream."""
    return "".join(_segments(body))


def t_punctmask(body: str) -> str:
    """Intra-segment '-' and '_' -> 'x'. Keeps '/' so depth is untouched."""
    return "".join("x" if c in "-_" else c for c in body)


# --------------------------------------------------------------------------- #
# F5 length
# --------------------------------------------------------------------------- #

def t_fixlen(body: str, width: int = 8) -> str:
    """Pad or truncate every segment to a constant width. Kills F5 only.

    Padding uses a character already common in paths rather than a novel one: a
    symbol the real corpus never contains would itself become a perfect tell and
    the measurement would collapse for the wrong reason.
    """
    out = []
    for seg in _segments(body):
        out.append(seg[:width] if len(seg) >= width
                   else seg + "a" * (width - len(seg)))
    return "/".join(out)


# --------------------------------------------------------------------------- #
# F6 edges
# --------------------------------------------------------------------------- #

def t_edges(body: str, k: int = 3) -> str:
    """ISOLATING: keep only the first and last k characters of the body.

    `char_wb` pads each document, so leading and trailing n-grams are visible to
    the discriminator; this asks whether they alone carry the tell.
    """
    return body[:k] + body[-k:] if len(body) > 2 * k else body


def t_noext(body: str) -> str:
    """Strip a trailing file extension. Kills the .html/.php signal specifically."""
    return re.sub(r"\.[A-Za-z0-9]{1,5}$", "", body)


# --------------------------------------------------------------------------- #
# Isolating projections
# --------------------------------------------------------------------------- #

def t_skeleton(body: str) -> str:
    """ISOLATING: letters -> 'a', digits -> '0', punctuation kept verbatim.

    Retains F3-shape, F4 and F5 and nothing else. High excess here means pure
    structure is SUFFICIENT; near-zero means the tell needs lexical content.
    """
    out = []
    for ch in body:
        if ch.isdigit():
            out.append("0")
        elif ch.isalpha():
            out.append("a")
        else:
            out.append(ch)
    return "".join(out)


def t_shape_only(body: str) -> str:
    """ISOLATING, coarsest: segment lengths and depth, nothing else.

    'docs/api/v2' -> 'aaaa/aaa/aa'. This is F4+F5 in their purest form and is
    the negative control: five models spanning 24-32 characters of path length
    never moved PTH excess outside noise, so a LARGE excess here would
    contradict established evidence and impugn the procedure rather than
    reveal a finding.
    """
    return "/".join("a" * len(s) for s in _segments(body))


#: Per-string transforms. Order is the order they are reported in.
TRANSFORMS: Dict[str, Transform] = {
    "identity": t_identity,
    "null": t_null,
    "scramble": t_scramble,
    "lower": t_lower,
    "digitmask": t_digitmask,
    "alphamask": t_alphamask,
    "flatten": t_flatten,
    "punctmask": t_punctmask,
    "fixlen": t_fixlen,
    "noext": t_noext,
    "edges": t_edges,
    "skeleton": t_skeleton,
    "shape_only": t_shape_only,
}

#: Which family each transform speaks to, and in which direction.
FAMILY = {
    "identity": ("--", "baseline"), "null": ("--", "null control"),
    "scramble": ("F1", "destructive"), "lower": ("F3", "destructive"),
    "digitmask": ("F3", "destructive"), "alphamask": ("F1/F3", "destructive"),
    "flatten": ("F4", "destructive"), "punctmask": ("F4", "destructive"),
    "fixlen": ("F5", "destructive"), "noext": ("F6", "destructive"),
    "edges": ("F6", "isolating"), "skeleton": ("F3/F4/F5", "isolating"),
    "shape_only": ("F4/F5", "isolating  NEGATIVE CONTROL"),
}


# --------------------------------------------------------------------------- #
# Set-level transforms (F2)
# --------------------------------------------------------------------------- #

def match_ttr(bodies: List[str], target_ttr: float,
              salt: str = "ttr") -> List[str]:
    """COLLAPSE a set's segment vocabulary until its type-token ratio hits target.

    THE DIRECTION IS FORCED, and an earlier version of this file had it
    backwards. Type-token ratio is bounded above by |vocab| / |tokens|, so a set
    that reuses a small segment vocabulary CANNOT be made more diverse out of
    its own types -- the missing vocabulary is precisely what is missing. The
    first attempt tried to raise the generated side's diversity and silently
    no-opped once the vocabulary ran out (0.075 -> 0.075), which would have read
    as "F2 is not the tell" while testing nothing at all.

    So the asymmetry is removed from the other end: the REAL set is degraded to
    the generated set's repetition profile. Real words, real segment lengths,
    real depth -- only the repetition changes. If excess collapses, repetition
    was carrying it.

    Each token maps to a kept type of the SAME LENGTH, chosen by hashing the
    token, so identical tokens stay identical and the length distribution is
    conserved exactly.
    """
    tokens = [s for b in bodies for s in _segments(b)]
    if not tokens:
        return list(bodies)
    current = len(set(tokens)) / len(tokens)
    if target_ttr >= current:
        return list(bodies)

    keep_fraction = max(target_ttr / current, 1e-6)
    by_len: Dict[int, List[str]] = {}
    for t in sorted(set(tokens)):
        by_len.setdefault(len(t), []).append(t)
    kept = {n: types[:max(1, round(len(types) * keep_fraction))]
            for n, types in by_len.items()}

    def swap(seg: str) -> str:
        pool = kept.get(len(seg)) or [seg]
        return pool[_rng(seg, salt).randrange(len(pool))]

    out = []
    for body in bodies:
        segs = [swap(s) for s in _segments(body)]
        out.append("/".join(segs) + ("/" if body.endswith("/") else ""))
    return out


def type_token_ratio(bodies: List[str]) -> float:
    segs = [s for b in bodies for s in _segments(b)]
    return len(set(segs)) / len(segs) if segs else 0.0


def novel_paths(bodies: List[str]) -> List[str]:
    """F2 NECESSITY, CORRECTED: remove repetition by DROPPING ROWS, never rewriting.

    The first attempt (`types_of` in ablate.py) emitted bare SEGMENTS as
    documents. That removed repetition but also collapsed depth to 1, deleted
    every delimiter, turned body length into segment length and turned the
    path's edges into a segment's edges -- so its R_T of 0.46 / 0.35 was a
    four-family ablation reported as one. It violated the conservation rule the
    rest of this file exists to enforce, and no test caught it because only the
    per-string transforms were covered.

    Here a whole path body is either KEPT VERBATIM or dropped. A row is kept
    only if every one of its segments is previously unseen and it repeats no
    segment internally. What survives therefore has, exactly:

        depth, delimiters, segment lengths, body length, character
        composition, edge patterns, and the original words

    and a segment type-token ratio of exactly 1.0 -- repetition and nothing
    else is gone.

    KNOWN BIAS, MEASURED RATHER THAN ASSUMED: a deep path has more chances to
    reuse a segment, so the filter preferentially keeps SHALLOW rows, and keeps
    more of them on the side that repeats less. If the two sides' depth
    distributions move apart under filtering, an F4 difference has been
    introduced and the result is confounded. `ablate.py` reports depth before
    and after for both sides so that is visible, not silent.
    """
    seen: set = set()
    out: List[str] = []
    for body in bodies:
        segs = [s for s in body.split("/") if s]
        if not segs or len(set(segs)) != len(segs):
            continue                      # repeats a segment within itself
        if any(s in seen for s in segs):
            continue                      # reuses a segment seen earlier
        seen.update(segs)
        out.append(body)
    return out
