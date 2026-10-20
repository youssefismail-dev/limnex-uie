"""
uie/metrics.py

Distributional statistics over URL corpora, and the divergence between two of
them.

What this is for
----------------
Two jobs, one implementation:

  * E0 publishes these numbers for the frozen corpus, so the benchmark states
    what "the web" looks like before any model exists.
  * E3 computes the same numbers over GENERATED samples and compares them to
    held-out real URLs. A generator that matches the real distribution on every
    interpretable statistic has not proven it is good -- but one that does not
    has proven it is bad, cheaply and diagnosably.

Why stratified reporting is mandatory, not optional
---------------------------------------------------
Density estimation is dominated by what is common. A model can post excellent
aggregate numbers while being useless on rare-but-systematic phenomena: if
Japanese e-commerce is 0.1% of the data, getting it entirely wrong costs almost
nothing in a pooled metric. An Internet World Model is supposed to know how
Japanese e-commerce sites name things.

So every statistic here is computed PER STRATUM and never pooled. This is what
makes the world-model claim falsifiable rather than rhetorical, and it is also
what keeps benign genuinely first-class rather than first-class in the charter
and second-class in the numbers.

Why these particular statistics
-------------------------------
Each is interpretable on its own, so a mismatch points at a cause rather than
just lowering a score. "Mean path depth is 1.2 against the web's 2.4" tells you
the model is generating shallow URLs; a single distance number would not.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from uie.factorization import URLParts, parse

__all__ = ["URLStats", "Stratifier", "BY_CLASS", "BY_SUFFIX_KIND", "BY_LANGUAGE",
           "BY_PATH_DEPTH", "describe", "describe_stratified",
           "total_variation", "jensen_shannon", "compare"]


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #

@dataclass
class URLStats:
    """Interpretable summary of one corpus (or one stratum of one).

    Counts are kept alongside rates so strata can be merged and so a rate
    computed over 12 URLs is not mistaken for one computed over 12 million.
    """

    n: int = 0
    n_bare_homepage: int = 0          # path is "" or "/" — no path information
    n_has_query: int = 0
    n_has_fragment: int = 0
    n_has_port: int = 0
    n_has_userinfo: int = 0           # the `@` obfuscation vector
    n_ip_host: int = 0
    n_has_scheme: int = 0
    n_no_suffix: int = 0

    path_depth: Counter = field(default_factory=Counter)
    subdomain_depth: Counter = field(default_factory=Counter)
    suffix: Counter = field(default_factory=Counter)
    scheme: Counter = field(default_factory=Counter)

    _length_sum: int = 0
    _label_entropy_sum: float = 0.0
    _label_entropy_n: int = 0
    _lengths: List[int] = field(default_factory=list)

    # --- rates ------------------------------------------------------------

    def _rate(self, count: int) -> float:
        return count / self.n if self.n else 0.0

    @property
    def bare_homepage_rate(self) -> float:
        return self._rate(self.n_bare_homepage)

    @property
    def query_rate(self) -> float:
        return self._rate(self.n_has_query)

    @property
    def ip_host_rate(self) -> float:
        return self._rate(self.n_ip_host)

    @property
    def userinfo_rate(self) -> float:
        return self._rate(self.n_has_userinfo)

    @property
    def mean_path_depth(self) -> float:
        total = sum(d * c for d, c in self.path_depth.items())
        return total / self.n if self.n else 0.0

    @property
    def mean_subdomain_depth(self) -> float:
        total = sum(d * c for d, c in self.subdomain_depth.items())
        return total / self.n if self.n else 0.0

    @property
    def mean_url_length(self) -> float:
        return self._length_sum / self.n if self.n else 0.0

    @property
    def mean_label_entropy(self) -> float:
        """Character entropy of the registrable label.

        Algorithmically generated domains are high-entropy and human-chosen
        names are not, so this separates 'invented a plausible name' from
        'emitted noise that happens to be novel' -- the exact failure mode a
        novelty metric alone would reward.
        """
        return (self._label_entropy_sum / self._label_entropy_n
                if self._label_entropy_n else 0.0)

    def length_percentile(self, p: float) -> int:
        """URL length at percentile `p` in [0, 100]. 0 for an empty corpus."""
        if not self._lengths:
            return 0
        ordered = sorted(self._lengths)
        idx = min(len(ordered) - 1, max(0, int(round(p / 100 * (len(ordered) - 1)))))
        return ordered[idx]

    # --- accumulation -----------------------------------------------------

    def add(self, parts: URLParts, raw: str) -> None:
        self.n += 1
        if parts.path in ("", "/"):
            self.n_bare_homepage += 1
        self.n_has_query += parts.has_query
        self.n_has_fragment += parts.has_fragment
        self.n_has_port += bool(parts.port)
        self.n_has_userinfo += parts.has_userinfo
        self.n_has_scheme += parts.has_scheme

        if parts.is_ip_host:
            self.n_ip_host += 1
        elif not parts.public_suffix:
            self.n_no_suffix += 1
        else:
            self.suffix[parts.public_suffix] += 1

        self.path_depth[min(parts.path_depth, 10)] += 1
        self.subdomain_depth[min(parts.subdomain_depth, 6)] += 1
        self.scheme[parts.scheme if parts.has_scheme else ""] += 1

        self._length_sum += len(raw)
        self._lengths.append(len(raw))

        if parts.registrable_label:
            self._label_entropy_sum += _shannon(parts.registrable_label)
            self._label_entropy_n += 1

    def as_dict(self) -> Dict[str, object]:
        """Flat, JSON-serializable view for benchmark publication."""
        return {
            "n": self.n,
            "bare_homepage_rate": round(self.bare_homepage_rate, 5),
            "query_rate": round(self.query_rate, 5),
            "ip_host_rate": round(self.ip_host_rate, 5),
            "userinfo_rate": round(self.userinfo_rate, 6),
            "mean_path_depth": round(self.mean_path_depth, 4),
            "mean_subdomain_depth": round(self.mean_subdomain_depth, 4),
            "mean_url_length": round(self.mean_url_length, 2),
            "mean_label_entropy": round(self.mean_label_entropy, 4),
            "url_length_p50": self.length_percentile(50),
            "url_length_p90": self.length_percentile(90),
            "url_length_p99": self.length_percentile(99),
            "path_depth_dist": _normalized(self.path_depth),
            "suffix_top": dict(self.suffix.most_common(25)),
            "n_distinct_suffixes": len(self.suffix),
        }


def _shannon(text: str) -> float:
    """Character-level Shannon entropy in bits."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _normalized(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if not total:
        return {}
    return {str(k): round(v / total, 5) for k, v in sorted(counter.items())}


def describe(urls: Iterable[str]) -> URLStats:
    """Summarize a corpus."""
    stats = URLStats()
    for url in urls:
        stats.add(parse(url), url)
    return stats


# --------------------------------------------------------------------------- #
# Stratification
# --------------------------------------------------------------------------- #

#: A stratifier maps a URL's parts to a stratum name. Strata must be derivable
#: with NO hand-written semantics -- anything requiring a curated list (brands,
#: "is this a CMS") would smuggle rules back into a project whose premise is
#: that the model discovers structure itself.
Stratifier = Callable[[URLParts], str]


def BY_SUFFIX_KIND(parts: URLParts) -> str:
    """Legacy gTLD vs country-code vs new gTLD vs none.

    Structural, not editorial: ccTLDs are exactly the two-letter suffixes, and
    the legacy set is fixed by history. No judgement about which TLDs are
    "suspicious" enters here -- that would be a rule about phishing.
    """
    suffix = parts.public_suffix
    if not suffix:
        return "ip" if parts.is_ip_host else "none"
    last = suffix.rsplit(".", 1)[-1]
    if last in {"com", "org", "net", "edu", "gov", "mil", "int"}:
        return "legacy_gtld"
    if len(last) == 2:
        return "cctld"
    return "new_gtld"


def BY_PATH_DEPTH(parts: URLParts) -> str:
    depth = parts.path_depth
    if depth == 0:
        return "depth_0"
    if depth <= 2:
        return "depth_1_2"
    if depth <= 4:
        return "depth_3_4"
    return "depth_5plus"


def BY_CLASS(parts: URLParts) -> str:      # placeholder; class is external
    """Class is corpus metadata, not derivable from the URL. `describe_stratified`
    accepts an explicit label sequence instead -- this exists so the intent is
    documented rather than silently absent."""
    raise NotImplementedError(
        "Class is corpus metadata; pass labels to describe_stratified().")


def BY_LANGUAGE(language: str) -> Stratifier:
    """Language comes from the Common Crawl index, not from the URL."""
    raise NotImplementedError(
        "Language is corpus metadata; pass labels to describe_stratified().")


def describe_stratified(urls: Sequence[str], *,
                        stratifier: Optional[Stratifier] = None,
                        labels: Optional[Sequence[str]] = None) -> Dict[str, URLStats]:
    """Summarize per stratum.

    Either derive strata from the URL (`stratifier`) or supply them as corpus
    metadata (`labels`, index-aligned) -- class and language are not properties
    of the string.
    """
    if (stratifier is None) == (labels is None):
        raise ValueError("Pass exactly one of `stratifier` or `labels`.")
    if labels is not None and len(labels) != len(urls):
        raise ValueError(f"labels ({len(labels)}) must align with urls ({len(urls)}).")

    out: Dict[str, URLStats] = {}
    for i, url in enumerate(urls):
        parts = parse(url)
        key = labels[i] if labels is not None else stratifier(parts)
        out.setdefault(key, URLStats()).add(parts, url)
    return out


# --------------------------------------------------------------------------- #
# Divergence between two corpora
# --------------------------------------------------------------------------- #

def total_variation(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Total variation distance in [0, 1]: half the L1 distance.

    Chosen over KL because it is bounded, symmetric, and finite when one
    distribution has support the other lacks -- which happens constantly with
    TLDs, where a generator inventing an unseen suffix would send KL to
    infinity and destroy the metric.
    """
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def jensen_shannon(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Jensen-Shannon divergence in bits, in [0, 1]. Symmetric and always finite."""
    keys = set(p) | set(q)

    def kl(a: Mapping[str, float], m: Dict[str, float]) -> float:
        total = 0.0
        for k in keys:
            pa = a.get(k, 0.0)
            if pa > 0 and m[k] > 0:
                total += pa * math.log2(pa / m[k])
        return total

    mid = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    return 0.5 * kl(p, mid) + 0.5 * kl(q, mid)


def compare(real: URLStats, generated: URLStats) -> Dict[str, object]:
    """Diagnose how generated output differs from real URLs.

    Scalar statistics are reported as signed deltas rather than absolute
    differences: the SIGN is the diagnosis. A negative path-depth delta says the
    model generates shallow URLs; the magnitude alone would not say which way.
    """
    return {
        "n_real": real.n,
        "n_generated": generated.n,
        "delta_bare_homepage_rate":
            round(generated.bare_homepage_rate - real.bare_homepage_rate, 5),
        "delta_query_rate":
            round(generated.query_rate - real.query_rate, 5),
        "delta_mean_path_depth":
            round(generated.mean_path_depth - real.mean_path_depth, 4),
        "delta_mean_subdomain_depth":
            round(generated.mean_subdomain_depth - real.mean_subdomain_depth, 4),
        "delta_mean_url_length":
            round(generated.mean_url_length - real.mean_url_length, 3),
        "delta_mean_label_entropy":
            round(generated.mean_label_entropy - real.mean_label_entropy, 4),
        "tv_path_depth": round(total_variation(
            _normalized(real.path_depth), _normalized(generated.path_depth)), 5),
        "tv_suffix": round(total_variation(
            _normalized(real.suffix), _normalized(generated.suffix)), 5),
        "js_suffix": round(jensen_shannon(
            _normalized(real.suffix), _normalized(generated.suffix)), 5),
        "suffix_coverage": round(
            len(set(generated.suffix) & set(real.suffix)) / len(real.suffix), 5)
            if real.suffix else 0.0,
        "novel_suffixes": sorted(set(generated.suffix) - set(real.suffix))[:20],
    }
