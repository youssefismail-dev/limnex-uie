"""
Tests for the structural factorization.

The load-bearing property is losslessness: `render(parse(u)) == u` for arbitrary
input, including malformed input. If that fails, the generator cannot reproduce
URLs the corpus contains and every part-derived statistic describes something
other than the data.

The adversarial cases below are not hypothetical. Deliberate grammar abuse is a
real attack class, and the design deliberately keeps it representable rather than
normalising it away.
"""
from __future__ import annotations

import pytest

from uie.factorization import URLParts, parse, render

ADVERSARIAL = [
    # --- the userinfo attack vector ---
    "http://brandx.com@evil.tk/login",          # last @ wins: host is evil.tk
    "http://user:pass@a.com/x",
    "http://@sokoyetu.co.ke/aol5/a000l.html",   # EMPTY userinfo — regression
    "http://@americanvalorfoundation.com",      # regression, no path
    "http://a@b@evil.tk/",                      # multiple @
    # --- grammar abuse ---
    "http://a.com//double//slash",
    "http://a.com/back\\slash",
    "http://a.com/%2e%2e/%2e%2e/etc",
    "http://a.com/path?a=1?b=2",
    "http://a.com/p#frag?notquery",             # ? after # belongs to fragment
    "http://a.com/p#a#b",
    # --- hosts ---
    "http://192.168.1.132/login",
    "http://[2001:db8::1]:8443/x",
    "http://a.com:8080/x",
    "https://xn--80ak6aa92e.com/",              # punycode
    "https://münchen.de/straße",                # raw unicode
    # --- shapes ---
    "a.com/no/scheme",                          # corpora contain these
    "example.com",
    "https://a.com",
    "https://a.com/",
    "https://a.com/?",                          # empty query, but present
    "https://a.com/#",                          # empty fragment, but present
    "ftp://a.com/f",
    "https://a.com/redirect?to=https://b.com",  # :// inside a query
    "",
]


@pytest.mark.parametrize("url", ADVERSARIAL)
def test_round_trip_is_lossless(url):
    assert render(parse(url)) == url


@pytest.mark.parametrize("url", ADVERSARIAL)
def test_parse_is_idempotent(url):
    """Re-parsing a rendered URL must yield identical parts."""
    assert parse(render(parse(url))) == parse(url)


# --- userinfo: the regression that 2.35M real URLs exposed -----------------

def test_empty_userinfo_is_distinguished_from_absent():
    """`http://@evil.tk/` has an EMPTY userinfo, not a missing one.

    Rendering on truthiness alone silently dropped the '@' — the only
    round-trip failure across the entire corpus, and a real obfuscation form.
    """
    with_at = parse("http://@evil.tk/")
    without = parse("http://evil.tk/")
    assert with_at.has_userinfo is True and with_at.userinfo == ""
    assert without.has_userinfo is False
    assert render(with_at) == "http://@evil.tk/"


def test_last_at_sign_determines_the_host():
    """Browsers resolve `http://brandx.com@evil.tk/` to evil.tk. Splitting on
    the FIRST '@' would name the wrong host on exactly the URLs that matter."""
    parts = parse("http://brandx.com@evil.tk/login")
    assert parts.host_raw == "evil.tk"
    assert parts.registrable_domain == "evil.tk"
    assert parts.userinfo == "brandx.com"


# --- host structure --------------------------------------------------------

def test_public_suffix_handles_multi_label_suffixes():
    parts = parse("https://shop.example.co.uk/a")
    assert parts.public_suffix == "co.uk"
    assert parts.registrable_label == "example"
    assert parts.registrable_domain == "example.co.uk"
    assert parts.subdomains == ("shop",)


def test_subdomains_are_ordered_outermost_first():
    parts = parse("https://login.secure.example.com/")
    assert parts.subdomains == ("login", "secure")
    assert parts.subdomain_depth == 2


@pytest.mark.parametrize("url", ["http://192.168.1.132/x", "http://[2001:db8::1]/x"])
def test_ip_hosts_have_no_domain_structure(url):
    """Regression: an early spike reported '.132' as a TLD by taking the last
    octet, polluting the suffix distribution with numeric 'suffixes'."""
    parts = parse(url)
    assert parts.is_ip_host
    assert parts.registrable_label == "" and parts.public_suffix == ""
    assert parts.registrable_domain == ""


def test_port_is_separated_from_host():
    parts = parse("http://a.com:8080/x")
    assert parts.host_raw == "a.com" and parts.port == "8080"


def test_ipv6_brackets_do_not_confuse_the_port_split():
    parts = parse("http://[2001:db8::1]:8443/x")
    assert parts.host_raw == "[2001:db8::1]" and parts.port == "8443"


# --- component boundaries --------------------------------------------------

def test_scheme_separator_inside_a_query_is_not_a_scheme():
    parts = parse("https://a.com/redirect?to=https://b.com")
    assert parts.scheme == "https"
    assert parts.host_raw == "a.com"
    assert parts.query == "to=https://b.com"


def test_question_mark_after_hash_belongs_to_the_fragment():
    """'#' binds looser than '?', so a later '?' is fragment content."""
    parts = parse("http://a.com/p#frag?notquery")
    assert parts.has_fragment and parts.fragment == "frag?notquery"
    assert parts.has_query is False and parts.query == ""


def test_empty_query_and_fragment_are_present_not_absent():
    q, f = parse("https://a.com/?"), parse("https://a.com/#")
    assert q.has_query and q.query == ""
    assert f.has_fragment and f.fragment == ""


def test_scheme_less_urls_are_preserved():
    """Several corpora store URLs without a scheme; inventing one would make
    the corpus unreproducible by the generator."""
    parts = parse("a.com/no/scheme")
    assert parts.has_scheme is False and parts.scheme == ""
    assert parts.host_raw == "a.com" and parts.path == "/no/scheme"
    assert render(parts) == "a.com/no/scheme"


def test_delimiters_inside_a_component_are_preserved():
    """Inside a component nothing is enforced, so abuse stays representable."""
    parts = parse("http://a.com//x//y?a=1&b=2%26c")
    assert parts.path == "//x//y"
    assert parts.query == "a=1&b=2%26c"


# --- derived views ---------------------------------------------------------

@pytest.mark.parametrize("url,depth", [
    ("https://a.com", 0),
    ("https://a.com/", 0),
    ("https://a.com/x", 1),
    ("https://a.com/x/", 1),
    ("https://a.com/x/y", 2),
    ("https://a.com/x/y/z/", 3),
])
def test_path_depth(url, depth):
    assert parse(url).path_depth == depth


def test_nothing_is_normalized():
    """Case, encoding and punycode are preserved verbatim; consumers normalize."""
    url = "HTTPS://WWW.Example.COM/Path%2Fx"
    parts = parse(url)
    assert parts.host_raw == "WWW.Example.COM"
    assert parts.path == "/Path%2Fx"
    assert render(parts) == url


def test_parse_never_raises_on_garbage():
    for junk in ["", "://", "http://", "@@@", "http://a.com:notaport/x", "?????"]:
        assert render(parse(junk)) == junk


def test_parts_are_hashable_and_comparable():
    """Frozen dataclass: usable as dict keys for dedup and grouping."""
    a, b = parse("https://a.com/x"), parse("https://a.com/x")
    assert a == b and len({a, b}) == 1
    assert isinstance(a, URLParts)
