"""
uie/splits.py

Frozen, leakage-safe corpus splits.

Two rules, both learned the hard way
------------------------------------
1. **Splits are ARTIFACTS, not computations.** A split recomputed at training
   time is a split that can silently change — a library version bump altering a
   grouping function is enough to invalidate every number ever published against
   it. So a split is built once, hashed, and thereafter only ever *loaded*.

2. **Grouping is by registrable domain, never by URL.** Benign runs ~58 URLs per
   domain: split at URL level and the same site lands on both sides, so the model
   scores well by having memorized `shopwave.io`'s path conventions rather than
   by having learned anything general. Domain-disjointness is what makes the
   evaluation mean "unseen site".

The temporal split and what "after the cutoff" must mean
--------------------------------------------------------
Section 8's freshness benchmark trains before date T and evaluates after it. The
subtlety is domains that STRADDLE the boundary. Assigning their later URLs to
eval would measure "can the model predict new pages on a site it already knows",
which is a much easier question than the one being asked.

So a straddling domain goes wholly to TRAIN (it was seen first), and its post-T
URLs are dropped rather than promoted. Eval therefore contains only URLs from
domains that did not exist in the training window at all — genuinely new sites,
which is what temporal generalization is supposed to measure. The dropped count
is reported, because it is a real cost and hiding it would flatter the split.

Class balance
-------------
Domains are split within each class, so proportions are preserved without
stratifying on anything derived from the URL itself.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from uie.factorization import parse

__all__ = ["Record", "SplitAssignment", "group_key", "build_random_split",
           "build_temporal_split", "apply_split", "save_split", "load_split"]

TRAIN, VAL, TEST = "train", "val", "test"
EVAL = "eval"                       # temporal splits are two-way


@dataclass(frozen=True)
class Record:
    """One corpus row: the URL, its class, and when it was observed."""

    url: str
    label: str                      # "benign" | "phishing"
    observed: str                   # ISO-8601 or CDXJ timestamp; "" if unknown


@dataclass
class SplitAssignment:
    """A frozen split, stored as domain -> partition.

    Domain-level storage is not merely compact (tens of thousands of entries
    rather than millions): it makes domain-disjointness a PROPERTY OF THE
    ARTIFACT rather than something a loader has to re-derive and could get wrong.
    """

    name: str
    rule: str                                   # human-readable, for the manifest
    assignment: Dict[str, str] = field(default_factory=dict)
    dropped_domains: List[str] = field(default_factory=list)
    cutoff: Optional[str] = None
    seed: Optional[int] = None
    #: Which key the assignment is stored under. A split BUILT on hosts and
    #: LOOKED UP by domain silently misses every host that has a subdomain and
    #: returns None, which callers read as "not in train or val" and drop. That
    #: happened: a host-keyed phishing split lost 78% of its rows and the
    #: survivors — the no-subdomain population — looked reassuringly balanced.
    #: The key kind travels WITH the artifact so the two can never disagree.
    key_kind: str = "domain"

    def _key(self, url: str) -> str:
        return group_key_host(url) if self.key_kind == "host" else group_key(url)

    def partition_of(self, url: str) -> Optional[str]:
        """Which partition a URL belongs to; None if its group is unassigned."""
        return self.assignment.get(self._key(url))

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for part in self.assignment.values():
            out[part] = out.get(part, 0) + 1
        return out

    def content_hash(self) -> str:
        """Stable hash of the assignment, so a reload can be verified."""
        digest = hashlib.sha256()
        for domain in sorted(self.assignment):
            digest.update(f"{domain}\t{self.assignment[domain]}\n".encode("utf-8"))
        return digest.hexdigest()


def group_key_host(url: str) -> str:
    """The unit of disjointness for PHISHING: the full host.

    73% of phishing URLs sit on shared hosting, where the registrable domain is
    the PLATFORM and the attacker's invention is the SUBDOMAIN. Keying the split
    on the registrable domain therefore assigns whole platforms to single
    partitions — measured on this corpus, `webflow.io` (18,729 hosts),
    `vercel.app` (17,493) and `github.io` (15,952) all fell wholly into train
    while `pages.dev` (16,963) fell wholly into val.

    The consequence is not subtle. Two samples of REAL phishing separate at
    0.9138 on full URLs and 0.7417 on invented names alone, against 0.5020 and
    0.5132 for benign through the identical machinery — a +0.2284 shift with no
    generator involved, which any metric would then charge to the model.

    Keying on the host puts every platform in BOTH partitions while guaranteeing
    that no attacker-chosen name appears in both. `vercel.app` is public
    infrastructure the model should know; `bankx-verify.vercel.app` is the
    invention that must be held out.
    """
    parts = parse(url)
    if not parts.registrable_label:
        return parts.host_raw.lower() or url
    return ".".join(tuple(parts.subdomains)
                    + (parts.registrable_label, parts.public_suffix)).lower()


def group_key(url: str) -> str:
    """The unit of disjointness: the registrable domain.

    IP-literal hosts have no registrable domain, so the host itself is the
    group — a URL on `192.168.1.1` must not straddle the split any more than one
    on `shopwave.io`. Falling back to the raw URL would silently disable
    grouping for exactly those rows.
    """
    parts = parse(url)
    return parts.registrable_domain or parts.host_raw.lower() or url


# --------------------------------------------------------------------------- #
# Random domain-disjoint split
# --------------------------------------------------------------------------- #

def build_random_split(records: Sequence[Record], *,
                       fractions: Mapping[str, float] = None,
                       seed: int = 20260803,
                       name: str = "random_domain_disjoint",
                       key: Callable[[str], str] = None) -> SplitAssignment:
    """Assign whole domains to partitions, preserving class proportions.

    Domains are shuffled per class and cut at the requested fractions, so a
    class that is 5% of the corpus is ~5% of every partition without any
    stratification on URL-derived features.
    """
    fractions = dict(fractions or {TRAIN: 0.8, VAL: 0.1, TEST: 0.1})
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    key_fn = key or group_key
    by_class: Dict[str, List[str]] = {}
    seen: set = set()
    for rec in records:
        k = key_fn(rec.url)
        if k in seen:
            continue                       # a group's class is its first sighting
        seen.add(k)
        by_class.setdefault(rec.label, []).append(k)

    assignment: Dict[str, str] = {}
    for label, domains in sorted(by_class.items()):
        ordered = sorted(domains)          # sort before shuffling: input order
        rng = random.Random(f"{seed}:{label}")   # must not affect the result
        rng.shuffle(ordered)

        start = 0
        parts = list(fractions.items())
        for i, (part, frac) in enumerate(parts):
            # Last partition takes the remainder, so nothing is lost to rounding.
            end = len(ordered) if i == len(parts) - 1 else start + int(len(ordered) * frac)
            for domain in ordered[start:end]:
                assignment[domain] = part
            start = end

    return SplitAssignment(
        name=name,
        rule=(f"whole {'hosts' if key is group_key_host else 'registrable domains'} "
              f"assigned to partitions; class proportions preserved; "
              f"fractions={fractions}"),
        assignment=assignment,
        seed=seed,
        key_kind="host" if key is group_key_host else "domain",
    )


# --------------------------------------------------------------------------- #
# Temporal split
# --------------------------------------------------------------------------- #

def build_temporal_split(records: Sequence[Record], *, cutoff: str,
                         name: str = "temporal") -> SplitAssignment:
    """Split by first observation, keeping eval strictly to unseen domains.

    `cutoff` is compared lexicographically against `observed`, which works for
    both ISO-8601 (`2026-01-01T00:00:00Z`) and CDXJ (`20260101000000`) as long as
    the cutoff is written in the same form as the data it is compared against.

    A domain first seen before the cutoff goes to TRAIN and STAYS there, even if
    it also appears afterwards — see the module docstring. Those later URLs are
    dropped, and the affected domains are recorded in `dropped_domains` so the
    cost is visible rather than silent.
    """
    first_seen: Dict[str, str] = {}
    for rec in records:
        if not rec.observed:
            continue
        key = group_key(rec.url)
        prev = first_seen.get(key)
        if prev is None or rec.observed < prev:
            first_seen[key] = rec.observed

    last_seen: Dict[str, str] = {}
    for rec in records:
        if not rec.observed:
            continue
        key = group_key(rec.url)
        prev = last_seen.get(key)
        if prev is None or rec.observed > prev:
            last_seen[key] = rec.observed

    assignment: Dict[str, str] = {}
    straddling: List[str] = []
    for domain, first in first_seen.items():
        if first < cutoff:
            assignment[domain] = TRAIN
            if last_seen[domain] >= cutoff:
                straddling.append(domain)
        else:
            assignment[domain] = EVAL

    return SplitAssignment(
        name=name,
        rule=("domains first observed before the cutoff -> train (and stay "
              "there, later URLs dropped); domains first observed at or after "
              "the cutoff -> eval, so eval contains only genuinely new sites"),
        assignment=assignment,
        dropped_domains=sorted(straddling),
        cutoff=cutoff,
    )


# --------------------------------------------------------------------------- #
# Application and persistence
# --------------------------------------------------------------------------- #

def apply_split(records: Sequence[Record],
                split: SplitAssignment) -> Dict[str, List[Record]]:
    """Partition records. Rows whose domain is unassigned are excluded.

    For a temporal split this is where straddling domains' post-cutoff URLs are
    dropped: the domain maps to TRAIN, so only its pre-cutoff rows survive the
    cutoff filter below.
    """
    out: Dict[str, List[Record]] = {}
    for rec in records:
        part = split.partition_of(rec.url)
        if part is None:
            continue
        if split.cutoff is not None:
            # Enforce the temporal boundary at ROW level too: a train domain's
            # later rows must not leak forward into the training window.
            if part == TRAIN and rec.observed and rec.observed >= split.cutoff:
                continue
            if part == EVAL and rec.observed and rec.observed < split.cutoff:
                continue
        out.setdefault(part, []).append(rec)
    return out


def _open_text(path: Path, mode: str):
    """Open plain or gzipped by extension.

    A split over ~200k domains is ~6 MB as JSON and roughly a tenth of that
    compressed. Since a frozen split MUST be committed — the guarantee is
    worthless if the artifact only exists on one machine — the size matters.
    """
    if str(path).endswith(".gz"):
        import gzip
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return open(path, mode, encoding="utf-8", newline="\n")


def save_split(split: SplitAssignment, path: Path) -> dict:
    """Freeze a split to disk with its hash. Returns the manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": split.name,
        "rule": split.rule,
        "cutoff": split.cutoff,
        "seed": split.seed,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "key_kind": split.key_kind,
        "n_domains": len(split.assignment),
        "domains_per_partition": split.counts(),
        "n_straddling_domains_kept_in_train": len(split.dropped_domains),
        "content_sha256": split.content_hash(),
    }
    payload = {"manifest": manifest,
               "assignment": dict(sorted(split.assignment.items())),
               "straddling_domains": split.dropped_domains}
    # Compact separators, not indented: this is a machine artifact whose readable
    # summary lives in `manifest`, and indentation nearly doubles it.
    with _open_text(path, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.write("\n")
    return manifest


def load_split(path: Path) -> SplitAssignment:
    """Load a frozen split, verifying its hash.

    The verification is the entire point of rule 1: a split whose contents no
    longer match its recorded hash has been edited or regenerated, and every
    result measured against it is suspect.
    """
    with _open_text(Path(path), "r") as handle:
        payload = json.load(handle)
    manifest = payload["manifest"]
    split = SplitAssignment(
        name=manifest["name"],
        rule=manifest["rule"],
        assignment=dict(payload["assignment"]),
        dropped_domains=list(payload.get("straddling_domains", [])),
        cutoff=manifest.get("cutoff"),
        seed=manifest.get("seed"),
        # Older splits predate the field and were all domain-keyed.
        key_kind=manifest.get("key_kind", "domain"),
    )
    actual = split.content_hash()
    if actual != manifest["content_sha256"]:
        raise ValueError(
            f"Split {path} failed hash verification: recorded "
            f"{manifest['content_sha256'][:16]}..., computed {actual[:16]}.... "
            "The file has been edited or regenerated; results measured against "
            "it cannot be compared to earlier ones.")
    return split
