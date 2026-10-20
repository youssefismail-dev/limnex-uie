"""
Tests for the validity / novelty / safety gate.

Two properties matter most and are easy to get subtly wrong:

  * The gate must not reject REAL URLs. A filter that discards legitimate web
    structure would silently teach us that the model is worse than it is, and
    would strip exactly the adversarial forms the project exists to model.
  * Rejection must be DIAGNOSED, not merely counted. `by_reason` is what turns
    a failing generator into a fixable one.
"""
from __future__ import annotations

import pytest

from uie.gate import Gate, GateReport, Reason, normalize_for_dedup


# --- validity: real URLs must survive --------------------------------------

REAL_SHAPES = [
    "https://example.com/",
    "https://sub.example.co.uk/a/b?c=1#frag",
    "http://192.168.1.1/admin",
    "http://[2001:db8::1]:8443/x",
    "https://xn--80ak6aa92e.com/path",
    "https://münchen.de/straße",
    "a.com/no/scheme",
    "http://a.com//double//slash",
    "http://a.com/back\\slash",
    "http://brandx.com@evil.tk/login",      # the @ trick is a real URL
    "http://@evil.tk/",                     # empty userinfo — also real
    "https://a.com/p?a=1&b=2%26c",
]


@pytest.mark.parametrize("url", REAL_SHAPES)
def test_real_urls_pass_validity(url):
    """Including the adversarial ones — deliberate grammar abuse is a real
    attack class the model must be able to represent."""
    assert Gate().check(url).passed, Gate().check(url).reasons


@pytest.mark.parametrize("url,reason", [
    ("", Reason.EMPTY),
    ("https://example.invalidtld/x", Reason.NO_PUBLIC_SUFFIX),
    ("https://" + "a" * 64 + ".com/", Reason.LABEL_TOO_LONG),
    ("https://a.com/" + "x" * 9000, Reason.URL_TOO_LONG),
    ("https://a\x00b.com/", Reason.CONTROL_CHARACTERS),
])
def test_invalid_urls_are_rejected_with_the_right_reason(url, reason):
    result = Gate().check(url)
    assert not result.passed
    assert reason in result.reasons


def test_host_length_limit():
    host = ".".join(["abcdefghij"] * 30)          # 329 chars, labels all legal
    result = Gate().check(f"https://{host}.com/")
    assert Reason.HOST_TOO_LONG in result.reasons


def test_ip_hosts_are_exempt_from_suffix_and_label_rules():
    """An IP literal is a legitimate URL with no public suffix; reporting it as
    malformed would be a bug, not a finding."""
    assert Gate().check("http://192.168.1.132/login").passed
    assert Gate().check("http://[2001:db8::1]/x").passed


def test_all_applicable_reasons_are_collected():
    """A sample failing several ways is worth knowing about in full."""
    result = Gate().check("https://" + "a" * 64 + ".invalidtld/\x01")
    assert {Reason.LABEL_TOO_LONG, Reason.NO_PUBLIC_SUFFIX,
            Reason.CONTROL_CHARACTERS} <= set(result.reasons)


# --- novelty ---------------------------------------------------------------

def test_memorized_urls_are_flagged():
    gate = Gate(known_urls=["https://known.com/page"])
    assert Reason.MEMORIZED in gate.check("https://known.com/page").reasons
    assert gate.check("https://unknown.com/page").passed


@pytest.mark.parametrize("variant", [
    "http://known.com/page",            # scheme differs
    "https://www.known.com/page",       # www differs
    "https://known.com/page/",          # trailing slash
    "HTTPS://KNOWN.COM/PAGE",           # case
])
def test_memorization_sees_through_shallow_disguises(variant):
    """The same URL in different dress is still a copy."""
    gate = Gate(known_urls=["https://known.com/page"])
    assert Reason.MEMORIZED in gate.check(variant).reasons


def test_novelty_rate_is_reported():
    gate = Gate(known_urls=["https://a.com/1", "https://a.com/2"])
    _, report = gate.filter(["https://a.com/1", "https://a.com/2",
                             "https://b.com/3", "https://b.com/4"])
    assert report.novelty_rate == 0.5


# --- duplication within a batch --------------------------------------------

def test_duplicates_within_a_batch_are_flagged():
    """Mode collapse shows up here before it shows up anywhere else."""
    kept, report = Gate().filter(["https://a.com/x", "https://a.com/x",
                                  "https://b.com/y"])
    assert kept == ["https://a.com/x", "https://b.com/y"]
    assert report.by_reason[Reason.DUPLICATE_IN_BATCH] == 1


def test_reset_clears_batch_state_only():
    gate = Gate(known_urls=["https://known.com/p"])
    gate.filter(["https://a.com/x"])
    gate.reset()
    assert gate.check("https://a.com/x").passed          # batch state cleared
    assert not gate.check("https://known.com/p").passed  # corpus state kept


def test_filter_resets_between_batches_by_default():
    gate = Gate()
    gate.filter(["https://a.com/x"])
    kept, _ = gate.filter(["https://a.com/x"])
    assert kept == ["https://a.com/x"]


# --- safety: the active-threat screen --------------------------------------

def test_live_blocklisted_domain_is_rejected():
    """The model samples from the phishing distribution, so independently
    regenerating an ACTIVE threat is far likelier than the space suggests."""
    gate = Gate(blocked_domains=["evil-live.tk"])
    assert Reason.ACTIVE_THREAT in gate.check("https://login.evil-live.tk/x").reasons


def test_blocklist_matches_the_registrable_domain_not_the_exact_url():
    """A generated path on a live phishing domain is just as unpublishable."""
    gate = Gate(blocked_domains=["evil-live.tk"])
    for url in ["https://evil-live.tk/", "http://a.b.evil-live.tk/never/seen"]:
        assert Reason.ACTIVE_THREAT in gate.check(url).reasons


def test_blocklist_does_not_over_match_similar_domains():
    gate = Gate(blocked_domains=["evil-live.tk"])
    assert gate.check("https://evil-live.tk.safe.com/x").passed
    assert gate.check("https://notevil-live.tk/x").passed


def test_collision_rate_is_reported_as_a_metric():
    """It is a realism signal, not only a safety filter."""
    gate = Gate(blocked_domains=["bad.tk"])
    _, report = gate.filter(["https://bad.tk/1", "https://good.com/2",
                             "https://good.com/3", "https://good.com/4"])
    assert report.collision_rate == 0.25


# --- reporting -------------------------------------------------------------

def test_report_breaks_failures_down_by_reason():
    """A gate that discards 40% of samples is reporting something about the
    generator; the breakdown is what makes it diagnosable."""
    gate = Gate(known_urls=["https://known.com/p"])
    _, report = gate.filter(["https://ok.com/1", "https://known.com/p",
                             "https://x.invalidtld/2", ""])
    assert report.n_total == 4 and report.n_passed == 1
    assert report.by_reason[Reason.MEMORIZED] == 1
    assert report.by_reason[Reason.NO_PUBLIC_SUFFIX] == 1
    assert report.by_reason[Reason.EMPTY] == 1
    assert "passed" in report.summary()


def test_empty_report_does_not_divide_by_zero():
    report = GateReport()
    assert report.pass_rate == 0.0
    assert report.novelty_rate == 0.0
    assert report.collision_rate == 0.0


def test_normalize_for_dedup_is_conservative():
    """Shallow by design — it catches copies in different dress without
    claiming to be near-duplicate detection."""
    assert normalize_for_dedup("HTTPS://WWW.A.com/X/") == "a.com/x"
    # Genuinely different URLs must NOT collapse together.
    assert normalize_for_dedup("https://a.com/x") != normalize_for_dedup("https://a.com/y")
