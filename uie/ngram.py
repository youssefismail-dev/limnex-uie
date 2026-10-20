"""
uie/ngram.py

E1 — the deliberately dumb baseline.

Why a baseline is built before the model
----------------------------------------
Character n-grams are notoriously strong on short strings. A neural model that
cannot beat a 5-gram has not earned its complexity, and without a bar published
in advance "the samples look realistic" quietly becomes the standard again.

What makes the comparison fair
------------------------------
This model uses **the same factorization as E2** (see `uie.factorization`) and
differs only in how the components are learned: counting instead of gradient
descent. So any win E2 posts is attributable to LEARNING, not to a better
representation — which is the only way the comparison answers a real question.

Why the likelihoods are comparable
----------------------------------
`parse`/`render` is a lossless bijection, so factorizing a URL and multiplying
per-component probabilities yields a genuine distribution over URL STRINGS, not
over some other object. Each URL has exactly one parse, so no probability mass is
double-counted. An autoregressive model over the same factorization is therefore
directly comparable on held-out NLL.

Smoothing
---------
Witten-Bell interpolation: an n-gram's estimate is blended with its (n-1)-gram
backoff, weighted by how much evidence the context actually has. A context seen
once with one continuation trusts its backoff almost entirely; a context seen ten
thousand times trusts itself. The recursion bottoms out in a uniform distribution
over the alphabet plus an escape symbol, so **no string can ever receive zero
probability** — which matters because a single zero makes held-out NLL infinite
and destroys the metric.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from uie.factorization import URLParts, parse, render

__all__ = ["CharNgram", "NgramURLModel", "path_body", "path_from_body"]

#: Marks the end of a generated string, so LENGTH is modelled rather than being
#: imposed by a cutoff. Chosen outside the URL character set.
END = "\x00"

#: Guards against a pathological context steering generation forever.
_MAX_FIELD_LEN = 400

#: Printable ASCII, used as the uniform fallback when a component was never
#: fitted for a class. Ignorance, not impossibility.
_DEFAULT_ALPHABET_SIZE = 96

#: Bound on the generation distribution cache, so a long run cannot grow without
#: limit on adversarially varied contexts.
_DIST_CACHE_MAX = 200_000


def path_body(path: str) -> str:
    """Strip the structural leading "/" from a path.

    A parsed path is always "" or starts with "/", so that slash carries no
    information — modelling it wastes probability mass and, worse, lets the
    n-gram OMIT it. A generated path without its slash fuses into the host on
    render (`http://ie.support` + `h` -> `http://ie.supporth`), producing a
    different URL rather than a malformed one. The slash is structure, so the
    grammar supplies it and the model never sees it.
    """
    return path[1:] if path.startswith("/") else path


def path_from_body(body: str) -> str:
    """Inverse of `path_body`."""
    return "/" + body


class CharNgram:
    """Interpolated character n-gram with Witten-Bell smoothing."""

    def __init__(self, order: int = 5) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        # counts[k][context] -> Counter of next characters, for k-char contexts
        self._counts: List[Dict[str, Counter]] = [defaultdict(Counter)
                                                  for _ in range(order)]
        self._alphabet: set = set()
        self._n_fitted = 0
        self._alphabet_list: List[str] = []
        self._lam: List[Dict[str, float]] = [{} for _ in range(order)]
        self._total: List[Dict[str, int]] = [{} for _ in range(order)]
        self._uniform = 1.0
        self._dist_cache: Dict[str, List[float]] = {}

    # --- fitting ----------------------------------------------------------

    def fit(self, strings: Iterable[str]) -> "CharNgram":
        for text in strings:
            self._n_fitted += 1
            padded = text + END
            self._alphabet.update(padded)
            for i, char in enumerate(padded):
                for k in range(self.order):
                    if i - k < 0:
                        break
                    self._counts[k][padded[i - k:i]][char] += 1
        self._alphabet.discard("")
        self._alphabet_list = sorted(self._alphabet)
        self._dist_cache.clear()
        self._finalize()
        return self

    @property
    def alphabet(self) -> List[str]:
        return self._alphabet_list

    @property
    def is_fitted(self) -> bool:
        return self._n_fitted > 0

    # --- probability ------------------------------------------------------

    def _finalize(self) -> None:
        """Precompute per-context totals and Witten-Bell weights.

        `sum(table.values())` inside the scoring loop is O(alphabet) and runs
        once per character per order — around a billion operations to score
        20k URLs, which turns a minutes-long job into an hours-long one. The
        counts never change after fitting, so these are computed once.
        """
        self._lam = [{} for _ in range(self.order)]
        self._total = [{} for _ in range(self.order)]
        for k in range(self.order):
            for context, table in self._counts[k].items():
                total = sum(table.values())
                self._total[k][context] = total
                # Witten-Bell: weight on this order grows with evidence.
                self._lam[k][context] = total / (total + len(table))
        self._uniform = 1.0 / (len(self._alphabet) + 1)

    def _prob(self, context: str, char: str) -> float:
        """P(char | context), interpolated up from uniform.

        Order matters, and it is LOW-TO-HIGH. Each step mixes its own estimate
        with everything shorter as the backoff, so the LONGEST matching context
        must be applied LAST or it does not survive to the result.

        Running this loop the other way (high order first, backing off to k=0)
        looks equivalent and is not: the final iteration is then the empty
        context, whose Witten-Bell weight is ~1.0 because it has seen every
        character in the corpus. It therefore overwrites everything the longer
        contexts contributed, and the model silently degrades to a UNIGRAM at
        every order.

        That was this file's behaviour through E1 and E2. It was invisible in
        aggregate perplexity — a unigram over URL characters still scores
        plausibly — and invisible to the tests, which asserted that probabilities
        were finite, normalised and higher for corpus-like strings, all of which
        a unigram satisfies. What exposed it was the order sweep: orders 3, 5 and
        7 returned byte-identical numbers, which no correctly interpolated model
        can do. `test_higher_order_uses_longer_context` now asserts exactly that.
        """
        # Uniform base case. `+1` reserves mass for characters never seen at all,
        # so an unseen symbol at prediction time is improbable rather than fatal.
        prob = self._uniform
        for k in range(0, min(len(context), self.order - 1) + 1):
            key = context[len(context) - k:] if k else ""
            table = self._counts[k].get(key)
            if not table:
                continue
            lam = self._lam[k][key]
            prob = lam * (table.get(char, 0) / self._total[k][key]) + (1 - lam) * prob
        return prob

    def distribution(self, context: str) -> List[float]:
        """Interpolated distribution over the alphabet, cached by context.

        Generation asks for the whole distribution at every step, and contexts
        repeat heavily, so caching turns the inner loop from O(alphabet x order)
        per character into a dictionary hit.
        """
        cached = self._dist_cache.get(context)
        if cached is None:
            cached = [self._prob(context, ch) for ch in self.alphabet]
            if len(self._dist_cache) < _DIST_CACHE_MAX:
                self._dist_cache[context] = cached
        return cached

    def log_prob(self, text: str) -> float:
        """log P(text) in nats, including the end symbol.

        An UNFITTED component falls back to uniform rather than `-inf`. A class
        may legitimately never use a component — benign URLs with no fragments,
        say — and a single `-inf` makes the whole corpus NLL infinite, destroying
        the metric this phase exists to establish. The rarity of the component is
        already priced into the `shape` categorical; the content term should
        express ignorance, not impossibility.
        """
        if not self.is_fitted:
            return -(len(text) + 1) * math.log(_DEFAULT_ALPHABET_SIZE)
        total = 0.0
        padded = text + END
        for i, char in enumerate(padded):
            start = max(0, i - (self.order - 1))
            total += math.log(max(self._prob(padded[start:i], char), 1e-300))
        return total

    # --- generation -------------------------------------------------------

    def sample(self, rng: random.Random, *, max_len: int = _MAX_FIELD_LEN) -> str:
        """Draw a string, stopping at the end symbol."""
        if not self.is_fitted:
            return ""
        alphabet = self.alphabet
        out: List[str] = []
        while len(out) < max_len:
            context = "".join(out[-(self.order - 1):]) if self.order > 1 else ""
            weights = self.distribution(context)
            char = rng.choices(alphabet, weights=weights, k=1)[0]
            if char == END:
                break
            out.append(char)
        return "".join(out)


@dataclass
class _ClassModel:
    """The per-class bundle, one component per factorization field."""

    scheme: Counter = field(default_factory=Counter)
    suffix: Counter = field(default_factory=Counter)
    n_subdomains: Counter = field(default_factory=Counter)
    #: (has_path, has_query, has_fragment) — which optional components exist.
    shape: Counter = field(default_factory=Counter)
    label: CharNgram = None
    subdomain: CharNgram = None
    path: CharNgram = None            # fitted WITHOUT the leading "/" (see below)
    query: CharNgram = None
    fragment: CharNgram = None
    n: int = 0


def _categorical_log_prob(counter: Counter, key, *, floor: float = 0.5) -> float:
    """Add-`floor` smoothed log probability over a closed vocabulary.

    An unseen suffix must be improbable rather than impossible: a phishing corpus
    spans 719 suffixes and a benign one 250, so a held-out URL WILL arrive with a
    suffix its class never used.
    """
    total = sum(counter.values())
    if total <= 0:
        return math.log(1e-12)
    vocab = len(counter) + 1
    return math.log((counter.get(key, 0) + floor) / (total + floor * vocab))


def _sample_categorical(counter: Counter, rng: random.Random):
    keys = list(counter)
    if not keys:
        return None
    return rng.choices(keys, weights=[counter[k] for k in keys], k=1)[0]


class NgramURLModel:
    """Compositional n-gram baseline over the URL factorization.

    Components mirror `uie.factorization` exactly: a categorical over the public
    suffix (a closed vocabulary — never generated character by character), and
    character n-grams for the parts where invention actually happens.
    """

    def __init__(self, order: int = 5) -> None:
        self.order = order
        self._by_class: Dict[str, _ClassModel] = {}

    # --- fitting ----------------------------------------------------------

    def fit(self, urls: Sequence[str], labels: Sequence[str]) -> "NgramURLModel":
        if len(urls) != len(labels):
            raise ValueError(f"urls ({len(urls)}) and labels ({len(labels)}) "
                             "must be the same length")

        buckets: Dict[str, List[URLParts]] = defaultdict(list)
        for url, label in zip(urls, labels):
            buckets[label].append(parse(url))

        for label, parts_list in buckets.items():
            model = _ClassModel(
                label=CharNgram(self.order),
                subdomain=CharNgram(self.order),
                path=CharNgram(self.order),
                query=CharNgram(self.order),
                fragment=CharNgram(self.order),
            )
            model.n = len(parts_list)
            for parts in parts_list:
                model.scheme[(parts.has_scheme, parts.scheme)] += 1
                model.suffix[parts.public_suffix] += 1
                model.n_subdomains[len(parts.subdomains)] += 1
                model.shape[(bool(parts.path), parts.has_query,
                             parts.has_fragment)] += 1

            model.label.fit(p.registrable_label for p in parts_list)
            model.subdomain.fit(s for p in parts_list for s in p.subdomains)
            model.path.fit(path_body(p.path) for p in parts_list if p.path)
            model.query.fit(p.query for p in parts_list if p.has_query)
            model.fragment.fit(p.fragment for p in parts_list if p.has_fragment)
            self._by_class[label] = model
        return self

    @property
    def classes(self) -> List[str]:
        return sorted(self._by_class)

    # --- scoring ----------------------------------------------------------

    def log_prob(self, url: str, label: str) -> float:
        """log P(url | label) in nats, decomposed over the factorization."""
        model = self._by_class.get(label)
        if model is None:
            raise KeyError(f"unknown class {label!r}; known: {self.classes}")

        parts = parse(url)
        total = 0.0
        total += _categorical_log_prob(model.scheme, (parts.has_scheme, parts.scheme))
        total += _categorical_log_prob(model.suffix, parts.public_suffix)
        total += _categorical_log_prob(model.n_subdomains, len(parts.subdomains))
        total += _categorical_log_prob(
            model.shape, (bool(parts.path), parts.has_query, parts.has_fragment))
        total += model.label.log_prob(parts.registrable_label)
        for sub in parts.subdomains:
            total += model.subdomain.log_prob(sub)
        if parts.path:
            total += model.path.log_prob(path_body(parts.path))
        if parts.has_query:
            total += model.query.log_prob(parts.query)
        if parts.has_fragment:
            total += model.fragment.log_prob(parts.fragment)
        return total

    def log_prob_ratio(self, url: str, positive: str = "benign",
                       negative: str = "phishing") -> float:
        """log P(url|positive) - log P(url|negative).

        The generative-classifier signal. It falls out of the baseline for free,
        which makes it a bar for E2's version of the same capability rather than
        a novelty introduced later.
        """
        return self.log_prob(url, positive) - self.log_prob(url, negative)

    def perplexity(self, urls: Sequence[str], label: str) -> float:
        """Per-character perplexity — length-normalised so it does not simply
        reward whichever model generates shorter URLs."""
        if not urls:
            return float("nan")
        total_nats = sum(self.log_prob(u, label) for u in urls)
        total_chars = sum(len(u) for u in urls) or 1
        return math.exp(-total_nats / total_chars)

    # --- generation -------------------------------------------------------

    def generate(self, label: str, n: int = 1, *,
                 seed: Optional[int] = None) -> List[str]:
        """Sample `n` URLs from the class-conditional distribution."""
        model = self._by_class.get(label)
        if model is None:
            raise KeyError(f"unknown class {label!r}; known: {self.classes}")
        rng = random.Random(seed)

        out: List[str] = []
        for _ in range(n):
            has_scheme, scheme = _sample_categorical(model.scheme, rng)
            suffix = _sample_categorical(model.suffix, rng) or ""
            n_subs = _sample_categorical(model.n_subdomains, rng) or 0
            has_path, has_query, has_fragment = _sample_categorical(model.shape, rng)

            registrable = model.label.sample(rng)
            subdomains = tuple(model.subdomain.sample(rng) for _ in range(n_subs))
            host = ".".join([s for s in subdomains if s] +
                            ([registrable] if registrable else []) +
                            ([suffix] if suffix else []))

            parts = URLParts(
                scheme=scheme, has_scheme=has_scheme, userinfo="",
                subdomains=subdomains, registrable_label=registrable,
                public_suffix=suffix, host_raw=host, port="",
                # The slash is grammar, not model output — see `path_body`.
                path=path_from_body(model.path.sample(rng)) if has_path else "",
                query=model.query.sample(rng) if has_query else "",
                fragment=model.fragment.sample(rng) if has_fragment else "",
                has_query=has_query, has_fragment=has_fragment,
                has_userinfo=False,
            )
            out.append(render(parts))
        return out
