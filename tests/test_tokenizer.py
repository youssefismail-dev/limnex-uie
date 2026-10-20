"""
Tests for the factorization tokenizer.

The scheme's whole value rests on three claims, and each is checked here:

  * a URL survives encode -> decode unchanged, including adversarial forms;
  * separators the GRAMMAR owns are never tokens, so the model cannot misplace
    a delimiter it never emits;
  * component presence is carried by the marker, so "has a query" is learned as
    part of the sequence rather than modelled separately.
"""
from __future__ import annotations

import pytest

from uie.factorization import parse
from uie.tokenizer import Fields, URLTokenizer, path_body

CORPUS = [
    "https://blog.shopwave.io/posts/2024/review?page=2#top",
    "http://secure-verify.bankx-alerts.support/login",
    "https://a.co.uk/",
    "http://brandx.com@evil.tk:8080/a/b?x=1#f",
    "http://@evil.tk/",
    "https://xn--80ak6aa92e.com/path",
    "http://a.com//double//slash",
    "http://a.com/back\\slash",
    "http://a.com/%2e%2e/etc",
    "a.com/no/scheme",
    "https://sub1.sub2.sub3.example.org/deep",
]


def _strip_scheme(url: str) -> str:
    for prefix in ("https://", "http://"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return url


@pytest.fixture
def tok() -> URLTokenizer:
    """The DEFAULT tokenizer, which drops the scheme (see `include_scheme`)."""
    return URLTokenizer().fit(CORPUS, min_count=1)


@pytest.fixture
def tok_scheme() -> URLTokenizer:
    """Scheme-preserving, for checking the field machinery is itself lossless
    and for loading pre-E2.3 checkpoints."""
    return URLTokenizer(include_scheme=True).fit(CORPUS, min_count=1)


# --- round trip ------------------------------------------------------------

def _expected(url: str) -> str:
    """The round trip, allowing for EMPTY structure fields being normalised away.

    A structure field opens only when its body is non-empty, so `a.com/` returns
    `a.com` and `a.com/x?` returns `a.com/x`. This is the ONE place the encoding
    is deliberately lossy, and it is worth the loss: see
    `test_an_empty_structure_field_does_not_open_a_marker` for the loss
    asymmetry it removes. Nothing downstream depends on the distinction —
    `normalize_for_dedup` already strips trailing slashes, so novelty and
    memorization are unaffected, and `example.com/` and `example.com` address
    one resource.
    """
    text = _strip_scheme(url)
    if text.endswith("#"):
        text = text[:-1]
    if text.endswith("?"):
        text = text[:-1]
    if text.endswith("/") and not path_body(parse(url).path or ""):
        text = text[:-1]
    return text


@pytest.mark.parametrize("url", CORPUS)
def test_round_trip_is_exact_when_the_scheme_is_kept(tok_scheme, url):
    """The factorization loses nothing except empty structure fields; dropping
    the scheme is a deliberate choice made one layer up, not a limitation."""
    scheme = url[:url.index("://") + 3] if "://" in url else ""
    assert (tok_scheme.decode(tok_scheme.encode(url, "benign"))
            == scheme + _expected(url))


@pytest.mark.parametrize("url", CORPUS)
def test_round_trip_is_exact_modulo_the_scheme(tok, url):
    """The default contract: everything except the scheme, and except empty
    structure fields, survives exactly.

    The scheme is dropped because it encodes WHEN a URL was observed rather than
    how it was named — benign historical crawls are 76.6% https, benign live
    92.5%, phishing 64.4% — so a model given it learns `http => phishing` from
    the web's TLS migration. A shortcut never presented cannot be learned.
    """
    assert tok.decode(tok.encode(url, "benign")) == _expected(url)


def test_scheme_is_absent_from_the_token_stream_entirely(tok):
    """Not merely dropped at decode time — never emitted. The model cannot
    condition on a token it never sees."""
    ids = tok.encode("https://blog.shopwave.io/posts/2024/review?page=2#top",
                     "benign")
    assert tok.marker_id("SCH") not in ids
    assert tok.decode_fields(ids).scheme is None


def test_round_trip_covers_userinfo_and_port(tok):
    """The `@` trick and explicit ports are real attack surface; a tokenizer
    that silently drops them makes them ungeneratable."""
    fields = tok.decode_fields(tok.encode("http://brandx.com@evil.tk:8080/x",
                                          "phishing"))
    assert fields.userinfo == "brandx.com"
    assert fields.port == "8080"
    assert fields.label == "evil" and fields.suffix == "tk"


def test_empty_userinfo_survives(tok):
    """`http://@evil.tk/` has an EMPTY userinfo, not an absent one."""
    fields = tok.decode_fields(tok.encode("http://@evil.tk/", "phishing"))
    assert fields.userinfo == ""
    # The trailing slash is an EMPTY path body and no longer opens <PTH>; the
    # empty USERINFO is a name field and is unaffected, which is the point here.
    assert tok.fields_to_url(fields) == "@evil.tk"       # scheme dropped


# --- grammar is never emitted ----------------------------------------------

def test_separators_are_not_tokens(tok):
    """`://`, the dots between host labels, the leading `/`, `?` and `#` are all
    supplied on reassembly. A model cannot misplace what it never produces."""
    ids = tok.encode("https://a.b.example.co.uk/p?q=1#f", "benign")
    text = "".join(tok.itos[i] for i in ids if len(tok.itos[i]) == 1)
    assert "?" not in text and "#" not in text
    assert ":" not in text and text.count(".") == 1     # only the "co.uk" dot


def test_path_slash_is_grammar_not_content(tok):
    fields = tok.decode_fields(tok.encode("https://a.co.uk/x/y", "benign"))
    assert fields.path == "x/y"                        # body only
    assert tok.fields_to_url(fields).endswith("/x/y")


def test_characters_inside_a_field_are_unconstrained(tok):
    """Grammar abuse must stay representable — it is a real attack class."""
    for url in ["http://a.com//double//slash", "http://a.com/back\\slash",
                "http://a.com/%2e%2e/etc"]:
        assert tok.decode(tok.encode(url, "benign")) == _strip_scheme(url)


# --- presence via markers --------------------------------------------------

def test_absent_components_have_no_marker(tok):
    """No `[QRY]` token means no query — presence is learned from the sequence,
    not from a separate distribution as the n-gram baseline required."""
    ids = tok.encode("https://a.co.uk/", "benign")
    assert tok.marker_id("QRY") not in ids
    assert tok.marker_id("FRG") not in ids
    assert tok.marker_id("SUF") in ids


def test_an_empty_structure_field_does_not_open_a_marker(tok):
    """REVERSES an earlier decision, deliberately.

    This test previously asserted the opposite — that a present-but-empty query
    keeps its marker, so `a.co.uk/?` round-tripped exactly. That reading of
    "presence" put the two branches of ONE decision under different loss
    weights. `field_mask` calls markers and EOS shape (weight 1.0 everywhere)
    and structure-field characters content (weight 0 outside the structure
    corpus), so on a name-corpus row:

        example.com/          <PTH> w=1.0   EOS w=1.0    <- taught
        example.com/login     <PTH> w=1.0   'l' w=0.0    <- masked

    The name corpus taught the empty branch and had the other masked out.
    Measured on the E3 checkpoint: <PTH> emission matched training exactly
    (100.0% vs 100.0%) while non-empty bodies ran 33.3 points under for phishing
    and 13.9 under for benign, phishing being worse because its name corpus is
    80.6% of rows against benign's 35.5%.

    Gating on the body makes presence a pure marker decision — both branches are
    shape, both trained on every row — at the cost of not distinguishing
    `a.co.uk/?` from `a.co.uk`. That distinction addresses one resource and is
    already erased by `normalize_for_dedup`.
    """
    tokenizer = URLTokenizer().fit(["https://a.co.uk/?"], min_count=1)
    ids = tokenizer.encode("https://a.co.uk/?", "benign")
    assert tokenizer.marker_id("QRY") not in ids
    assert tokenizer.marker_id("PTH") not in ids
    assert tokenizer.decode(ids) == "a.co.uk"


def test_a_non_empty_structure_field_still_opens_its_marker(tok):
    """The fix must not cost a real body: only EMPTY ones stop opening."""
    # Characters restricted to those the `tok` fixture was fit on, so an
    # out-of-vocabulary letter cannot be mistaken for a dropped field.
    ids = tok.encode("https://a.co.uk/x?x=1#f", "benign")
    for marker in ("PTH", "QRY", "FRG"):
        assert tok.marker_id(marker) in ids, marker
    assert tok.decode(ids) == "a.co.uk/x?x=1#f"


def test_both_branches_of_the_path_decision_carry_equal_weight(tok):
    """The property the fix exists to guarantee, asserted on the mask itself.

    After the change, "does this URL have a path" is a choice between <PTH> and
    EOS — two shape tokens — so a name-corpus row weights both at 1.0. Before,
    the alternative to EOS was a masked content character.
    """
    from uie.train import field_mask

    bare = tok.encode("https://a.co.uk/", "benign")
    withp = tok.encode("https://a.co.uk/login", "benign")
    w_bare = field_mask(bare, tok, "name")
    w_with = field_mask(withp, tok, "name")

    # The bare row ends at EOS with no PTH marker at all.
    assert tok.marker_id("PTH") not in bare
    assert w_bare[len(bare) - 2] == 1.0            # EOS is trained

    # The path row opens PTH, and that marker is trained on a NAME row too.
    i = withp.index(tok.marker_id("PTH"))
    assert w_with[i - 1] == 1.0                    # <PTH> is trained
    # Its contents remain scoped to the structure corpus.
    assert w_with[i] == 0.0
    assert field_mask(withp, tok, "structure")[i] == 1.0


def test_scheme_less_urls_have_no_scheme_marker(tok):
    ids = tok.encode("a.com/no/scheme", "benign")
    assert tok.marker_id("SCH") not in ids
    assert tok.decode(ids) == "a.com/no/scheme"


def test_each_subdomain_gets_its_own_marker(tok):
    ids = tok.encode("https://sub1.sub2.sub3.example.org/deep", "benign")
    assert ids.count(tok.marker_id("SUB")) == 3


# --- ordering --------------------------------------------------------------

def test_suffix_precedes_label(tok):
    """Reversed from the URL string on purpose: generation is causal, so the
    label is invented knowing the TLD. A .jp domain is named unlike a .gov one."""
    ids = tok.encode("https://shopwave.io/x", "benign")
    assert ids.index(tok.marker_id("SUF")) < ids.index(tok.marker_id("LBL"))


def test_control_tokens_lead_the_sequence(tok):
    """Only the CLASS token now. `<L:*>` was dropped: it was a class marker
    (100% `und` for phishing, 1.6% for benign) carrying information that is
    96% the TLD's, which the model already generates as the suffix."""
    ids = tok.encode("https://a.co.uk/", "phishing", language="und")
    assert ids[0] == tok.bos_id
    assert tok.itos[ids[1]] == "<C:phishing>"
    assert not any(tok.itos[i].startswith("<L:") for i in ids)
    assert ids[-1] == tok.eos_id


def test_legacy_tokenizer_still_emits_the_language_token(tok):
    """Pre-E2.3 checkpoints were trained with it and must still load."""
    legacy = URLTokenizer(include_language=True).fit(CORPUS, min_count=1)
    ids = legacy.encode("https://a.co.uk/", "phishing", language="und")
    assert legacy.itos[ids[2]] == "<L:und>"


def test_unknown_class_raises(tok):
    with pytest.raises(KeyError):
        tok.encode("https://a.co.uk/", "malware")


def test_unknown_language_falls_back_rather_than_failing(tok):
    """`language` is accepted and ignored now, but passing one must not raise —
    callers written against the old signature still work."""
    assert tok.decode(tok.encode("https://a.co.uk/", "benign",
                                 language="klingon")) == "a.co.uk"


# --- robustness to model output --------------------------------------------

def test_decode_tolerates_a_truncated_sequence(tok):
    """Sampling can stop anywhere; decode must not raise on a partial URL."""
    ids = tok.encode("https://blog.shopwave.io/posts/2024/review?page=2#top",
                     "benign")
    for cut in range(4, len(ids)):
        tok.decode(ids[:cut])          # must not raise


def test_decode_ignores_stray_control_tokens(tok):
    ids = tok.encode("https://a.co.uk/x", "benign")
    polluted = ids[:5] + [tok.stoi["<C:phishing>"]] + ids[5:]
    assert tok.decode(polluted) == "a.co.uk/x"


def test_decode_stops_at_eos(tok):
    ids = tok.encode("https://a.co.uk/x", "benign")
    assert tok.decode(ids + tok.encode("https://b.io/y", "benign")) == \
        "a.co.uk/x"


def test_padding_is_ignored(tok):
    ids = tok.encode("https://a.co.uk/x", "benign")
    assert tok.decode([tok.pad_id] * 3 + ids) == "a.co.uk/x"


def test_empty_fields_reassemble_without_crashing(tok):
    assert isinstance(tok.fields_to_url(Fields()), str)


# --- persistence -----------------------------------------------------------

def test_state_dict_round_trips(tok):
    restored = URLTokenizer.from_state_dict(tok.state_dict())
    assert restored.vocab_size == tok.vocab_size
    assert restored.itos == tok.itos
    for url in CORPUS:
        assert restored.decode(restored.encode(url, "benign")) == _expected(url)


def test_vocabulary_is_small(tok):
    """URLs are percent-encoded and punycode, so raw Unicode never appears:
    ~85 characters across the whole corpus. A character model is cheap here."""
    assert tok.vocab_size < 200
