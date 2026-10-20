"""
uie/factorization.py

The structural factorization of a URL — UIE's representation.

A flat character model must rediscover URL grammar from data, spending capacity
on constraints that are known exactly. But enforcing grammar ALL THE WAY DOWN
would make the model unable to represent malformed URLs, and deliberate grammar
abuse is a real attack class: `http://brandx.com@evil.tk/`, doubled slashes,
backslash separators, percent-encoded delimiters. Both extremes lose something.

The balance struck here, per the design:

  * TOP-LEVEL structure is enforced. Path, query and fragment are separate
    components, so the `?` and `#` separators are supplied by the grammar and the
    model never generates them -- it cannot misplace what it does not emit.
  * INSIDE a component, generation is unconstrained. Slash, ampersand, equals,
    percent, backslash and anything else remain producible, so grammar abuse
    stays representable.

The public suffix is a closed vocabulary (the PSL), so it is a categorical
choice rather than characters -- validity for free, and no capacity spent
learning that URLs end in a real TLD.

Losslessness is the load-bearing property
-----------------------------------------
`render(parse(u)) == u` must hold for arbitrary input, including malformed input.
If it does not, the generator cannot produce URLs that the corpus contains, and
every statistic computed over parts describes something other than the data. The
test suite asserts this over the full curated corpora (~830k real URLs) rather
than over hand-written examples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import tldextract

__all__ = ["URLParts", "parse", "render", "PSL", "strip_scheme"]

#: Offline PSL snapshot: never fetches at runtime, so parsing is deterministic
#: and reproducible. A stale suffix list is a version-pinning problem, not a
#: silent network dependency inside a training pipeline.
PSL = tldextract.TLDExtract(suffix_list_urls=())

_SCHEME_SEP = "://"


@dataclass(frozen=True)
class URLParts:
    """A URL decomposed along the boundaries the model generates across.

    Every field is stored EXACTLY as it appeared. Nothing is lowercased,
    percent-decoded, punycode-decoded or otherwise normalized: consumers apply
    their own normalization, and the generator must be able to reproduce the
    input byte-for-byte.

    `has_scheme` distinguishes `http://a.com` from the scheme-less `a.com` that
    several corpora contain -- without it, rendering would invent a scheme the
    source never had.
    """

    scheme: str                       # "https", or "" when absent
    has_scheme: bool
    userinfo: str                     # "user:pass", or "" -- the `@` attack vector
    subdomains: Tuple[str, ...]       # ("www", "login"), outermost first
    registrable_label: str            # "shopwave"; "" for IP-literal hosts
    public_suffix: str                # "co.uk"; "" when not a PSL suffix
    host_raw: str                     # verbatim host, for lossless round-trip
    port: str                         # "8080", or ""
    path: str                         # leading "/" included when present
    query: str                        # without "?"
    fragment: str                     # without "#"
    has_query: bool                   # "?" present but empty is not "no query"
    has_fragment: bool
    has_userinfo: bool                # same distinction: `http://@evil.tk/` is
    #                                   an empty userinfo, not an absent one —
    #                                   and it is a real obfuscation form.

    # --- derived views the model and metrics consume ----------------------

    @property
    def registrable_domain(self) -> str:
        """`shopwave.io`. Empty for IP literals and suffix-less hosts."""
        if not self.registrable_label or not self.public_suffix:
            return ""
        return f"{self.registrable_label}.{self.public_suffix}"

    @property
    def is_ip_host(self) -> bool:
        """IP-literal hosts have no registrable label or suffix at all.

        Worth an explicit flag: an early spike reported '.132' as a TLD by
        taking the last octet, polluting the suffix distribution.
        """
        bare = self.host_raw.strip("[]")
        return bool(bare) and (bare.replace(".", "").isdigit() or ":" in bare)

    @property
    def path_depth(self) -> int:
        """Number of non-empty path segments. `/a/b/` is depth 2."""
        core = self.path.strip("/")
        return len(core.split("/")) if core else 0

    @property
    def subdomain_depth(self) -> int:
        return len(self.subdomains)


def _split_authority(authority: str) -> Tuple[str, bool, str, str]:
    """(userinfo, has_userinfo, host, port) from a raw authority component.

    The LAST `@` separates userinfo from host, which is what browsers do and
    what makes `http://brandx.com@evil.tk/` resolve to evil.tk. Splitting on the
    first `@` would mis-identify the host on exactly the URLs where it matters.
    """
    userinfo, has_userinfo = "", False
    rest = authority
    at = authority.rfind("@")
    if at >= 0:
        userinfo, has_userinfo, rest = authority[:at], True, authority[at + 1:]

    # IPv6 literals are bracketed and contain colons that are not port separators.
    if rest.startswith("["):
        close = rest.find("]")
        if close >= 0:
            host = rest[:close + 1]
            tail = rest[close + 1:]
            port = tail[1:] if tail.startswith(":") else ""
            return userinfo, has_userinfo, host, port

    colon = rest.rfind(":")
    if colon >= 0 and rest[colon + 1:].isdigit():
        return userinfo, has_userinfo, rest[:colon], rest[colon + 1:]
    return userinfo, has_userinfo, rest, ""


def _split_host(host: str) -> Tuple[Tuple[str, ...], str, str]:
    """(subdomains, registrable_label, public_suffix) via the offline PSL."""
    bare = host.strip("[]")
    if not bare or bare.replace(".", "").isdigit() or ":" in bare:
        return (), "", ""                       # IP literal: no domain structure
    ext = PSL(host)
    subs = tuple(p for p in ext.subdomain.split(".") if p) if ext.subdomain else ()
    return subs, ext.domain, ext.suffix


def parse(url: str) -> URLParts:
    """Decompose a URL. Never raises: malformed input is data, not an error.

    Deliberately hand-rolled rather than delegating to `urlsplit`, which
    normalizes in small ways (and rejects some malformed authorities) that would
    break losslessness on exactly the adversarial URLs this project cares about.
    """
    rest = url
    scheme, has_scheme = "", False

    sep = rest.find(_SCHEME_SEP)
    if sep >= 0 and _is_scheme(rest[:sep]):
        scheme, has_scheme = rest[:sep], True
        rest = rest[sep + len(_SCHEME_SEP):]

    # Fragment first, then query: `#` binds looser, so a `?` after a `#` is part
    # of the fragment, not a query string.
    fragment, has_fragment = "", False
    hash_at = rest.find("#")
    if hash_at >= 0:
        fragment, has_fragment = rest[hash_at + 1:], True
        rest = rest[:hash_at]

    query, has_query = "", False
    q_at = rest.find("?")
    if q_at >= 0:
        query, has_query = rest[q_at + 1:], True
        rest = rest[:q_at]

    # Authority ends at the first `/`. Backslash is NOT treated as a separator:
    # browsers differ, and preserving it verbatim keeps the abuse representable.
    slash = rest.find("/")
    authority, path = (rest[:slash], rest[slash:]) if slash >= 0 else (rest, "")

    userinfo, has_userinfo, host, port = _split_authority(authority)
    subs, label, suffix = _split_host(host)

    return URLParts(
        scheme=scheme, has_scheme=has_scheme, userinfo=userinfo,
        subdomains=subs, registrable_label=label, public_suffix=suffix,
        host_raw=host, port=port, path=path, query=query, fragment=fragment,
        has_query=has_query, has_fragment=has_fragment,
        has_userinfo=has_userinfo,
    )


def _is_scheme(candidate: str) -> bool:
    """RFC 3986 scheme: ALPHA *( ALPHA / DIGIT / "+" / "-" / "." ).

    Guards against treating the `://` inside a path or query as a scheme
    separator, e.g. `a.com/redirect?to=https://b.com`.
    """
    if not candidate or not candidate[0].isalpha():
        return False
    return all(c.isalnum() or c in "+-." for c in candidate)


def render(parts: URLParts) -> str:
    """Reassemble a URL. Inverse of `parse` for arbitrary input."""
    out = []
    if parts.has_scheme:
        out.append(parts.scheme)
        out.append(_SCHEME_SEP)
    if parts.has_userinfo:
        out.append(parts.userinfo)
        out.append("@")
    out.append(parts.host_raw)
    if parts.port:
        out.append(":")
        out.append(parts.port)
    out.append(parts.path)
    if parts.has_query:
        out.append("?")
        out.append(parts.query)
    if parts.has_fragment:
        out.append("#")
        out.append(parts.fragment)
    return "".join(out)


def strip_scheme(url: str) -> str:
    """Drop `http://` / `https://`, leaving everything else untouched.

    The canonical comparison form for this project since E2.3. The tokenizer no
    longer emits the scheme, so generated URLs never carry one; scoring them
    against real URLs that do would hand a discriminator a perfect separator
    that has nothing to do with generation quality.

    Deliberately narrower than `gate.normalize_for_dedup`, which also folds
    `www.` and trailing slashes. Those are real signal about how a URL is
    formed and must stay in the comparison.
    """
    for prefix in ("https://", "http://"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return url
