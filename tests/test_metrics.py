"""
Tests for corpus statistics and distributional comparison.

Two things these metrics must get right, because both are load-bearing:

  * Statistics must be computed PER STRATUM and must not silently pool. Pooling
    is how a model that is excellent on English .com and useless on Japanese
    .jp still looks fine.
  * Comparison must be DIAGNOSTIC. Signed deltas say which way a generator is
    wrong; a single distance number says only that it is.
"""
from __future__ import annotations

import math

import pytest

from uie.metrics import (BY_PATH_DEPTH, BY_SUFFIX_KIND, compare, describe,
                         describe_stratified, jensen_shannon, total_variation)


# --- summary statistics ----------------------------------------------------

def test_bare_homepage_rate():
    stats = describe(["https://a.com", "https://b.com/", "https://c.com/page"])
    assert stats.bare_homepage_rate == pytest.approx(2 / 3)


def test_mean_path_depth():
    stats = describe(["https://a.com/", "https://a.com/x", "https://a.com/x/y/z"])
    assert stats.mean_path_depth == pytest.approx((0 + 1 + 3) / 3)


def test_query_and_fragment_are_counted_separately():
    stats = describe(["https://a.com/p?x=1", "https://a.com/p#f", "https://a.com/p"])
    assert stats.n_has_query == 1 and stats.n_has_fragment == 1


def test_ip_hosts_do_not_pollute_the_suffix_distribution():
    """An IP literal has no suffix; counting one would invent a TLD."""
    stats = describe(["http://192.168.1.1/x", "https://a.com/y"])
    assert stats.n_ip_host == 1
    assert dict(stats.suffix) == {"com": 1}


def test_userinfo_rate_tracks_the_at_sign_vector():
    stats = describe(["http://brandx.com@evil.tk/", "http://@evil.tk/",
                      "https://ok.com/"])
    assert stats.userinfo_rate == pytest.approx(2 / 3)


def test_label_entropy_separates_plausible_names_from_noise():
    """High-entropy labels are algorithmically generated; low-entropy ones are
    human-chosen. This is what stops a novelty metric rewarding pure noise."""
    human = describe(["https://banana.com/", "https://london.com/"])
    noise = describe(["https://x7q2vz9k4m.com/", "https://p3w8dj5btr.com/"])
    assert noise.mean_label_entropy > human.mean_label_entropy


def test_length_percentiles_are_ordered():
    stats = describe([f"https://a.com/{'x' * i}" for i in range(100)])
    assert (stats.length_percentile(50) <= stats.length_percentile(90)
            <= stats.length_percentile(99))


def test_empty_corpus_does_not_divide_by_zero():
    stats = describe([])
    assert stats.n == 0
    assert stats.bare_homepage_rate == 0.0
    assert stats.mean_path_depth == 0.0
    assert stats.length_percentile(50) == 0


def test_as_dict_is_json_shaped():
    stats = describe(["https://a.co.uk/x/y?z=1"])
    d = stats.as_dict()
    assert d["n"] == 1 and d["n_distinct_suffixes"] == 1
    assert isinstance(d["path_depth_dist"], dict)
    assert all(isinstance(k, str) for k in d["path_depth_dist"])


# --- stratification --------------------------------------------------------

def test_suffix_kind_is_structural_not_editorial():
    """No judgement about which TLDs are 'suspicious' may enter a stratifier —
    that would be a rule about phishing, which the model must discover itself."""
    from uie.factorization import parse
    assert BY_SUFFIX_KIND(parse("https://a.com/")) == "legacy_gtld"
    assert BY_SUFFIX_KIND(parse("https://a.co.uk/")) == "cctld"
    assert BY_SUFFIX_KIND(parse("https://a.xyz/")) == "new_gtld"
    assert BY_SUFFIX_KIND(parse("http://192.168.1.1/")) == "ip"


def test_stratified_stats_are_not_pooled():
    urls = ["https://a.com/", "https://b.com/x/y/z", "https://c.co.uk/p"]
    strata = describe_stratified(urls, stratifier=BY_SUFFIX_KIND)
    assert set(strata) == {"legacy_gtld", "cctld"}
    assert strata["legacy_gtld"].n == 2 and strata["cctld"].n == 1
    # The pooled mean would hide that one stratum is entirely bare homepages.
    assert strata["cctld"].bare_homepage_rate == 0.0


def test_labels_carry_metadata_strata_the_url_cannot_provide():
    """Class and language are corpus metadata, not properties of the string."""
    urls = ["https://a.com/x", "https://b.tk/y", "https://c.com/z"]
    strata = describe_stratified(urls, labels=["benign", "phishing", "benign"])
    assert strata["benign"].n == 2 and strata["phishing"].n == 1


def test_stratifier_and_labels_are_mutually_exclusive():
    with pytest.raises(ValueError):
        describe_stratified(["https://a.com/"])
    with pytest.raises(ValueError):
        describe_stratified(["https://a.com/"], stratifier=BY_PATH_DEPTH,
                            labels=["x"])


def test_misaligned_labels_are_rejected_loudly():
    """Silently zipping mismatched lengths would mislabel every stratum."""
    with pytest.raises(ValueError):
        describe_stratified(["https://a.com/", "https://b.com/"], labels=["x"])


# --- divergence ------------------------------------------------------------

def test_total_variation_bounds():
    assert total_variation({"a": 1.0}, {"a": 1.0}) == 0.0
    assert total_variation({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)


def test_total_variation_is_symmetric():
    p, q = {"a": 0.7, "b": 0.3}, {"a": 0.2, "b": 0.8}
    assert total_variation(p, q) == pytest.approx(total_variation(q, p))


def test_total_variation_survives_disjoint_support():
    """A generator inventing an unseen suffix must not blow the metric up —
    KL would go to infinity here, which is why TV is used."""
    value = total_variation({"com": 1.0}, {"zzz": 1.0})
    assert math.isfinite(value) and value == pytest.approx(1.0)


def test_jensen_shannon_bounds_and_symmetry():
    assert jensen_shannon({"a": 1.0}, {"a": 1.0}) == pytest.approx(0.0)
    assert jensen_shannon({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    p, q = {"a": 0.6, "b": 0.4}, {"a": 0.1, "b": 0.9}
    assert jensen_shannon(p, q) == pytest.approx(jensen_shannon(q, p))


# --- comparison is diagnostic ----------------------------------------------

def test_delta_sign_says_which_way_the_generator_is_wrong():
    real = describe(["https://a.com/x/y/z", "https://b.com/p/q/r"])
    shallow = describe(["https://c.com/", "https://d.com/"])
    result = compare(real, shallow)
    assert result["delta_mean_path_depth"] < 0        # generates shallow URLs
    assert result["delta_bare_homepage_rate"] > 0


def test_suffix_coverage_detects_a_collapsed_generator():
    """Mode collapse onto one TLD shows up as low coverage even if every
    individual sample is perfectly valid."""
    real = describe([f"https://a{i}.{s}/x" for i, s in
                     enumerate(["com", "org", "net", "jp", "de"])])
    collapsed = describe([f"https://b{i}.com/x" for i in range(5)])
    result = compare(real, collapsed)
    assert result["suffix_coverage"] == pytest.approx(0.2)
    assert result["tv_suffix"] > 0.5


def test_novel_suffixes_are_surfaced():
    """A REAL suffix the reference sample happened not to contain — evidence
    the generator reached further than the sample it was compared against."""
    real = describe(["https://a.com/x"])
    generated = describe(["https://b.com/x", "https://c.jp/y"])
    assert compare(real, generated)["novel_suffixes"] == ["jp"]


def test_invented_suffixes_never_enter_the_suffix_distribution():
    """`.zzz` is not in the Public Suffix List, so it is not a suffix at all.

    A generator emitting one is caught by the gate (NO_PUBLIC_SUFFIX), not
    silently recorded here as a TLD the web contains — otherwise the published
    suffix distribution would include TLDs that do not exist.
    """
    stats = describe(["https://c.zzz/y", "https://a.com/x"])
    assert dict(stats.suffix) == {"com": 1}
    assert stats.n_no_suffix == 1


def test_identical_corpora_compare_as_identical():
    urls = ["https://a.com/x/y", "https://b.co.uk/p?q=1", "http://1.2.3.4/z"]
    result = compare(describe(urls), describe(urls))
    assert result["tv_path_depth"] == 0.0
    assert result["tv_suffix"] == 0.0
    assert result["delta_mean_path_depth"] == 0.0
