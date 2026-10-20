"""
uie/gate.py

The validity, novelty and safety gate applied to every generated sample.

This is the ONLY place rules belong in the project: as FILTERS, never as
generators. A rule that decides what a phishing URL looks like caps the model at
its author's imagination; a rule that rejects an unparseable string does not.

Four checks, each answering a different question:

  1. VALIDITY   — could this URL exist at all? (parseable, IDNA-encodable,
                  a real public suffix, within protocol limits)
  2. NOVELTY    — did the model invent it, or copy it? Memorization is the
                  DEFAULT failure mode of generative models on short strings,
                  so it is measured on every run rather than assumed away.
  3. DUPLICATION— is the batch collapsing? Mode collapse shows up here first.
  4. SAFETY     — is this coincidentally a LIVE threat? The model samples from
                  the phishing distribution, concentrating exactly where real
                  phishing lives, and phishing kits are highly templated, so
                  independent regeneration of an active URL is far likelier than
                  the ~10^15 space suggests. Screened against an OFFLINE
                  blocklist snapshot; live DNS is never queried (it would break
                  the offline constraint and tell an attacker's nameserver that
                  someone is probing).

Rejection rates are a finding, not a nuisance
---------------------------------------------
`GateReport` breaks failures down by reason precisely so a bad model is
diagnosed rather than quietly filtered into looking good. A gate that discards
40% of samples is reporting something about the generator.

Collision rate is also a QUALITY metric
---------------------------------------
How often the model independently produces a domain already on a blocklist is
evidence it is generating in the right region of the space — a validation signal
available no other way. Report it; filter it from releases.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import (Container, Iterable, List, Optional, Sequence, Set, Tuple)

from uie.factorization import URLParts, parse

__all__ = ["Reason", "GateResult", "GateReport", "Gate", "normalize_for_dedup"]


class Reason:
    """Rejection reasons. Strings rather than an enum so reports stay readable
    when serialized to JSON alongside the samples they describe."""

    EMPTY = "empty"
    UNPARSEABLE_HOST = "unparseable_host"
    NO_PUBLIC_SUFFIX = "no_public_suffix"
    NOT_IDNA_ENCODABLE = "not_idna_encodable"
    LABEL_TOO_LONG = "label_too_long"
    HOST_TOO_LONG = "host_too_long"
    URL_TOO_LONG = "url_too_long"
    CONTROL_CHARACTERS = "control_characters"
    MEMORIZED = "memorized"
    DUPLICATE_IN_BATCH = "duplicate_in_batch"
    ACTIVE_THREAT = "active_threat"


#: Protocol limits. A DNS label is 63 octets and a full name 253 (RFC 1035).
_MAX_LABEL = 63
_MAX_HOST = 253
#: Not a protocol limit — no RFC caps URL length — but a practical ceiling that
#: catches degenerate generation without rejecting real long URLs (the longest
#: observed in ~2.35M real URLs is comfortably under this).
_MAX_URL = 8_192


@dataclass(frozen=True)
class GateResult:
    """Outcome for one candidate. `reasons` is empty iff it passed."""

    url: str
    reasons: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.reasons


@dataclass
class GateReport:
    """Aggregate outcome, broken down by reason so a failing model is diagnosed."""

    n_total: int = 0
    n_passed: int = 0
    by_reason: Counter = field(default_factory=Counter)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_total if self.n_total else 0.0

    @property
    def novelty_rate(self) -> float:
        """Share that were NOT memorized from the training corpus.

        Reported beside every quality number, never separately: the most
        realistic-looking samples are the ones the model copied.
        """
        if not self.n_total:
            return 0.0
        return 1.0 - self.by_reason[Reason.MEMORIZED] / self.n_total

    @property
    def collision_rate(self) -> float:
        """Share that independently landed on a live blocklisted domain.

        A safety filter AND a realism signal — see the module docstring.
        """
        if not self.n_total:
            return 0.0
        return self.by_reason[Reason.ACTIVE_THREAT] / self.n_total

    def summary(self) -> str:
        lines = [f"{self.n_passed:,}/{self.n_total:,} passed "
                 f"({100 * self.pass_rate:.1f}%)  "
                 f"novelty {100 * self.novelty_rate:.1f}%  "
                 f"collision {100 * self.collision_rate:.3f}%"]
        for reason, count in self.by_reason.most_common():
            lines.append(f"    {reason:<22} {count:>8,} "
                         f"({100 * count / self.n_total:.2f}%)")
        return "\n".join(lines)


def normalize_for_dedup(url: str) -> str:
    """A conservative normal form for memorization and duplicate detection.

    Deliberately shallow — lowercase, drop the scheme, drop a leading `www.`,
    drop a trailing slash. It catches the copies that matter (the same URL in
    different dress) without claiming to be near-duplicate detection: true
    near-duplicate search over millions of short strings needs MinHash or
    similar and is a separate piece of work. Scope is stated so the novelty
    number is not over-read.
    """
    text = url.strip().lower()
    for prefix in ("https://", "http://"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.startswith("www."):
        text = text[4:]
    return text.rstrip("/")


class Gate:
    """Applies the four checks. Stateful across a batch, for check 3."""

    def __init__(self, *,
                 known_urls: Optional[Iterable[str]] = None,
                 known_membership: Optional[Container] = None,
                 blocked_domains: Optional[Iterable[str]] = None,
                 max_url_length: int = _MAX_URL) -> None:
        """`known_urls` is the training corpus (for memorization) and
        `blocked_domains` an offline snapshot of live threat feeds, keyed on the
        REGISTRABLE DOMAIN — that is where the danger lives, and a generated path
        on a live phishing domain is just as unpublishable as an exact match.

        `known_membership` accepts a prebuilt container over ALREADY-NORMALIZED
        keys (a `BloomFilter`, or any object supporting `in`) for corpora too
        large to hold as a set. It exists because the training corpus and the
        training BUNDLE are not the same thing: benign URLs are subsampled, so
        a gate built only from the bundle would score a copy of any unsampled
        training URL as novel. Novelty is pre-registered, so it must be checked
        against the full corpus regardless of what was sampled for training.

        Both may be supplied; a URL is memorized if EITHER reports it.
        """
        self._known = {normalize_for_dedup(u) for u in (known_urls or ())}
        self._known_membership = known_membership
        self._blocked = {d.strip().lower() for d in (blocked_domains or ()) if d.strip()}
        self._max_url_length = max_url_length
        self._seen: Set[str] = set()

    def reset(self) -> None:
        """Clear per-batch duplicate state. Corpus and blocklist are unaffected."""
        self._seen.clear()

    # --- individual checks --------------------------------------------------

    def _validity_reasons(self, url: str, parts: URLParts) -> List[str]:
        reasons: List[str] = []
        if not url.strip():
            return [Reason.EMPTY]
        if len(url) > self._max_url_length:
            reasons.append(Reason.URL_TOO_LONG)
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url):
            reasons.append(Reason.CONTROL_CHARACTERS)

        host = parts.host_raw
        if not host:
            reasons.append(Reason.UNPARSEABLE_HOST)
            return reasons

        # IP literals are legitimate URLs with no suffix; exempt from suffix and
        # label rules rather than reported as malformed.
        if not parts.is_ip_host:
            if not parts.public_suffix:
                reasons.append(Reason.NO_PUBLIC_SUFFIX)
            if len(host.encode("utf-8", "ignore")) > _MAX_HOST:
                reasons.append(Reason.HOST_TOO_LONG)
            if any(len(lbl.encode("utf-8", "ignore")) > _MAX_LABEL
                   for lbl in host.split(".")):
                reasons.append(Reason.LABEL_TOO_LONG)
            if not _idna_encodable(host):
                reasons.append(Reason.NOT_IDNA_ENCODABLE)
        return reasons

    # --- batch application --------------------------------------------------

    def check(self, url: str) -> GateResult:
        """Evaluate one candidate. All applicable reasons are collected, not
        just the first — a sample failing three ways is worth knowing about."""
        parts = parse(url)
        reasons = self._validity_reasons(url, parts)

        key = normalize_for_dedup(url)
        if key in self._known or (self._known_membership is not None
                                  and key in self._known_membership):
            reasons.append(Reason.MEMORIZED)
        if key in self._seen:
            reasons.append(Reason.DUPLICATE_IN_BATCH)
        self._seen.add(key)

        domain = parts.registrable_domain.lower()
        if domain and domain in self._blocked:
            reasons.append(Reason.ACTIVE_THREAT)

        return GateResult(url, tuple(reasons))

    def filter(self, urls: Sequence[str], *,
               reset: bool = True) -> Tuple[List[str], GateReport]:
        """Apply the gate to a batch; return survivors and the diagnosis."""
        if reset:
            self.reset()
        report = GateReport()
        kept: List[str] = []
        for url in urls:
            result = self.check(url)
            report.n_total += 1
            if result.passed:
                report.n_passed += 1
                kept.append(url)
            else:
                for reason in result.reasons:
                    report.by_reason[reason] += 1
        return kept, report


def _idna_encodable(host: str) -> bool:
    """Can this host be represented on the wire?

    ASCII hosts are checked for the characters DNS actually permits; non-ASCII
    hosts must survive IDNA encoding. A host that cannot be encoded cannot be
    visited, so it is not a URL the web could contain.
    """
    if not host:
        return False
    if host.isascii():
        allowed = set("abcdefghijklmnopqrstuvwxyz"
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._[]:")
        return all(ch in allowed for ch in host)
    try:
        host.encode("idna")
        return True
    except (UnicodeError, UnicodeDecodeError):
        return False
