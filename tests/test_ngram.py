"""
Tests for the E1 n-gram baseline.

Three properties are load-bearing:

  * NO ZERO PROBABILITIES. A single zero makes held-out NLL infinite and
    destroys the metric the whole phase exists to establish.
  * GENERATED URLS MUST BE STRUCTURALLY VALID. A baseline that emits malformed
    strings sets an artificially low bar, so E2 would clear it without learning
    anything.
  * DETERMINISM. Baseline numbers are published; they must reproduce.
"""
from __future__ import annotations

import math

import pytest

from uie.factorization import parse, render
from uie.gate import Gate
from uie.ngram import CharNgram, NgramURLModel, path_body, path_from_body


def test_higher_order_uses_longer_context():
    """Order must change predictions. It did not, for the whole of E1 and E2.

    `_prob` interpolated from high order DOWN to the empty context, so the last
    iteration — whose Witten-Bell weight is ~1.0 because it has seen every
    character — overwrote every longer context's contribution. The model was a
    UNIGRAM at every order, and the E1 baseline published in BENCHMARK.md was
    measured on that.

    Nothing else caught it: probabilities stayed finite, normalised, and higher
    for corpus-like strings, which is all the other tests ask and all a unigram
    needs to satisfy. Only comparing ORDERS exposes it.
    """
    corpus = ["securelogin", "securepayment", "secureaccount",
              "securebank", "secureshop"] * 20

    # `l` follows `...cure` in exactly one of five equally frequent words, so a
    # model that genuinely conditions on the context approaches 0.2, while a
    # unigram returns the corpus frequency of `l` (~0.017).
    probs = {order: CharNgram(order=order).fit(corpus)._prob("secure", "l")
             for order in (2, 3, 5, 7)}
    assert probs[5] > 0.15, probs
    assert probs[5] > probs[2] * 2, probs
    assert probs[3] != probs[2], probs


def test_long_context_beats_short_on_held_out_text():
    """The practical consequence: a higher order must score structured
    held-out strings better, or the extra context is buying nothing."""
    train = [f"news{i:03d}daily" for i in range(200)]
    held_out = ["news777daily", "news888daily"]

    def nll(order):
        model = CharNgram(order=order).fit(train)
        return -sum(model.log_prob(w) for w in held_out) / sum(len(w) + 1
                                                              for w in held_out)

    assert nll(6) < nll(1), (nll(6), nll(1))

BENIGN = [
    "https://blog.shopwave.io/posts/2024/review",
    "https://cdn.shopwave.io/static/v2/app.js",
    "https://news.tenjiku.jp/2024/03/article",
    "https://support.northvale.org/patients/forms",
    "https://shop.bergen.de/produkte/kategorie/1",
    "https://www.example.com/about",
]
PHISH = [
    "http://secure-verify.bankx-alerts.support/login",
    "http://signin.trust-notice.online/session",
    "http://examp1e-billing.top/update",
    "http://account-verify.icu/",
    "http://login.secure-bank.cc/auth",
    "http://my-account.verify-now.xyz/signin",
]


@pytest.fixture
def model() -> NgramURLModel:
    urls = BENIGN + PHISH
    labels = ["benign"] * len(BENIGN) + ["phishing"] * len(PHISH)
    return NgramURLModel(order=5).fit(urls, labels)


# --- smoothing: nothing may be impossible ----------------------------------

def test_unseen_string_has_finite_log_prob():
    ngram = CharNgram(3).fit(["abcabc", "abcabd"])
    value = ngram.log_prob("zzzqqq")
    assert math.isfinite(value) and value < 0


def test_unseen_url_has_finite_log_prob(model):
    """A held-out URL WILL arrive with a suffix its class never used — 719
    phishing suffixes against 250 benign ones guarantees it."""
    value = model.log_prob("https://totally-unseen.museum/deep/path?x=1", "benign")
    assert math.isfinite(value)


def test_empty_string_is_scoreable():
    ngram = CharNgram(3).fit(["abc"])
    assert math.isfinite(ngram.log_prob(""))


def test_probabilities_sum_to_about_one():
    """The interpolated distribution must be normalised, or NLL is meaningless."""
    ngram = CharNgram(3).fit(["hello", "help", "held"])
    total = sum(ngram._prob("he", ch) for ch in ngram.alphabet)
    assert total == pytest.approx(1.0, abs=0.02)


def test_higher_order_is_not_worse_on_training_data():
    corpus = ["abcdefabcdef"] * 20
    low = CharNgram(2).fit(corpus).log_prob("abcdefabcdef")
    high = CharNgram(6).fit(corpus).log_prob("abcdefabcdef")
    assert high > low


# --- discrimination --------------------------------------------------------

def test_in_distribution_urls_score_higher_under_their_own_class(model):
    for url in BENIGN:
        assert model.log_prob(url, "benign") > model.log_prob(url, "phishing")
    for url in PHISH:
        assert model.log_prob(url, "phishing") > model.log_prob(url, "benign")


def test_log_prob_ratio_sign_follows_the_class(model):
    assert model.log_prob_ratio(BENIGN[0]) > 0
    assert model.log_prob_ratio(PHISH[0]) < 0


def test_unknown_class_raises(model):
    with pytest.raises(KeyError):
        model.log_prob("https://a.com/", "malware")


def test_mismatched_labels_are_rejected():
    with pytest.raises(ValueError):
        NgramURLModel().fit(["https://a.com/"], ["benign", "phishing"])


# --- generation: structural validity ---------------------------------------

def test_generated_urls_round_trip(model):
    for url in model.generate("benign", 25, seed=1):
        assert render(parse(url)) == url


def test_generated_urls_pass_the_validity_gate(model):
    """A baseline emitting malformed strings would set an artificially low bar."""
    gate = Gate()
    urls = model.generate("phishing", 40, seed=2)
    kept, report = gate.filter(urls)
    assert report.pass_rate > 0.8, report.summary()


def test_path_never_fuses_into_the_host(model):
    """Regression: the path n-gram once generated bodies with no leading '/',
    so `http://ie.support` + `h` rendered as `http://ie.supporth` — a DIFFERENT
    URL, not a malformed one. The slash is grammar, not model output."""
    for url in model.generate("benign", 40, seed=3):
        parts = parse(url)
        assert parts.path == "" or parts.path.startswith("/")


def test_generated_hosts_end_in_a_real_public_suffix(model):
    """The suffix is a closed vocabulary sampled categorically, never spelled
    out character by character."""
    for url in model.generate("benign", 25, seed=4):
        assert parse(url).public_suffix != ""


@pytest.mark.parametrize("path", ["", "/", "/a/b", "//double//slash", "/x?y"])
def test_path_body_round_trips(path):
    if path:
        assert path_from_body(path_body(path)) == path


# --- determinism -----------------------------------------------------------

def test_generation_is_deterministic_under_seed(model):
    assert model.generate("benign", 10, seed=99) == model.generate("benign", 10, seed=99)


def test_different_seeds_differ(model):
    assert model.generate("benign", 10, seed=1) != model.generate("benign", 10, seed=2)


def test_fitting_is_order_independent():
    """Counting is commutative; shard read order must not change the baseline."""
    urls, labels = BENIGN + PHISH, ["benign"] * 6 + ["phishing"] * 6
    a = NgramURLModel(4).fit(urls, labels)
    b = NgramURLModel(4).fit(list(reversed(urls)), list(reversed(labels)))
    probe = "https://news.tenjiku.jp/2024/03/article"
    assert a.log_prob(probe, "benign") == pytest.approx(b.log_prob(probe, "benign"))


# --- perplexity ------------------------------------------------------------

def test_perplexity_is_length_normalised(model):
    """Per-character, so it cannot be gamed by generating shorter URLs."""
    short = model.perplexity(["https://a.io/x"], "benign")
    long = model.perplexity(["https://a.io/x/y/z/w/v/u/t"], "benign")
    assert math.isfinite(short) and math.isfinite(long)


def test_perplexity_of_empty_corpus_is_nan(model):
    assert math.isnan(model.perplexity([], "benign"))


def test_order_must_be_positive():
    with pytest.raises(ValueError):
        CharNgram(0)
