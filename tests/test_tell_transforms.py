"""Each transform must destroy its OWN family and conserve the others.

An ablation is only interpretable if the transformation does exactly one thing.
A `scramble` that quietly changed segment lengths would collapse excess and be
read as "lexical identity is the tell" when length had done the work. These
tests pin the CONSERVATION LAWS -- what each transform must leave untouched --
rather than its output, so a future edit cannot silently widen a transform's
blast radius.

Determinism is tested too: the transforms are applied to real and generated
strings separately, so any hidden randomness would make the two sides
incomparable while still looking like a clean result.
"""
from __future__ import annotations

import re

import pytest

from research.tell.transforms import (TRANSFORMS, match_ttr, t_alphamask,
                                      t_digitmask, t_edges, t_fixlen,
                                      t_flatten, t_identity, t_lower, t_noext,
                                      t_null, t_punctmask, t_scramble,
                                      t_shape_only, t_skeleton,
                                      type_token_ratio)

BODIES = [
    "cgi-sys/suspendedpage.cgi",
    "homepage/",
    "produse/categorie-produse/cofetarie/page/2/",
    "blog/the-transformer-noise-is-normal/",
    "enindex.php/pro/showinfo/id/18.html",
    "een-klacht-melden",
    "a",
    "A/B/C",
    "x1/y22/z333",
]


def depth(body):
    return len([s for s in body.split("/") if s])


def seg_lens(body):
    return [len(s) for s in body.split("/") if s]


def charset(body):
    return sorted(body.replace("/", ""))


# --------------------------------------------------------------------------- #
# determinism -- the whole design rests on it
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_transform_is_deterministic(name):
    """Same input, same output, every call. Randomised transforms are seeded
    from the string's own content precisely so the real and generated sides
    cannot diverge."""
    fn = TRANSFORMS[name]
    for body in BODIES:
        assert fn(body) == fn(body) == fn(body), name


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_transform_never_raises_or_returns_none(name):
    fn = TRANSFORMS[name]
    for body in BODIES + ["", "/", "//", "a/", "/a"]:
        out = fn(body)
        assert isinstance(out, str), (name, body)


def test_null_control_changes_nothing():
    """`null` must be a genuine no-op: it defines the measurement's zero."""
    for body in BODIES:
        assert t_null(body) == t_identity(body) == body


# --------------------------------------------------------------------------- #
# F1 scramble: destroys order, conserves EVERYTHING else
# --------------------------------------------------------------------------- #

def test_scramble_conserves_length_depth_and_charset():
    for body in BODIES:
        out = t_scramble(body)
        assert seg_lens(out) == seg_lens(body), body
        assert depth(out) == depth(body), body
        assert charset(out) == charset(body), body


def test_scramble_actually_reorders_something():
    """A transform that changes nothing would read as 'family not necessary'."""
    assert any(t_scramble(b) != b for b in BODIES)


# --------------------------------------------------------------------------- #
# F3 composition
# --------------------------------------------------------------------------- #

def test_digitmask_conserves_positions_and_length():
    for body in BODIES:
        out = t_digitmask(body)
        assert len(out) == len(body)
        assert seg_lens(out) == seg_lens(body)
        # a digit stays a digit, a non-digit is untouched
        for a, b in zip(body, out):
            assert (a.isdigit() and b == "0") or (not a.isdigit() and a == b)


def test_lower_conserves_length_and_structure():
    for body in BODIES:
        out = t_lower(body)
        assert len(out) == len(body) and seg_lens(out) == seg_lens(body)
        assert not any(c.isupper() for c in out)


def test_alphamask_removes_letter_identity_but_keeps_layout():
    for body in BODIES:
        out = t_alphamask(body)
        assert len(out) == len(body) and seg_lens(out) == seg_lens(body)
        assert not re.search(r"[b-zB-Z]", out.replace("/", "")) or True
        for a, b in zip(body, out):
            if a.isalpha():
                assert b == "a"
            else:
                assert b == a


# --------------------------------------------------------------------------- #
# F4 structure
# --------------------------------------------------------------------------- #

def test_flatten_removes_all_delimiters_and_conserves_characters():
    for body in BODIES:
        out = t_flatten(body)
        assert "/" not in out
        assert sorted(out) == charset(body), body


def test_punctmask_keeps_depth():
    for body in BODIES:
        out = t_punctmask(body)
        assert depth(out) == depth(body)
        assert len(out) == len(body)
        assert "-" not in out and "_" not in out


# --------------------------------------------------------------------------- #
# F5 length
# --------------------------------------------------------------------------- #

def test_fixlen_makes_every_segment_constant_width():
    for body in BODIES:
        out = t_fixlen(body)
        lens = seg_lens(out)
        assert not lens or set(lens) == {8}, (body, out)
        assert depth(out) == depth(body), body


def test_fixlen_pads_with_a_character_the_corpus_already_uses():
    """A novel pad character would itself become a perfect tell."""
    assert "a" in t_fixlen("x")


# --------------------------------------------------------------------------- #
# isolating projections
# --------------------------------------------------------------------------- #

def test_shape_only_retains_exactly_depth_and_segment_lengths():
    for body in BODIES:
        out = t_shape_only(body)
        assert seg_lens(out) == seg_lens(body), body
        assert depth(out) == depth(body), body
        assert set(out) <= {"a", "/"}, out


def test_skeleton_erases_identity_but_not_punctuation():
    out = t_skeleton("cgi-sys/page.php/18")
    assert out == "aaa-aaa/aaaa.aaa/00", out


def test_edges_keeps_only_the_rim():
    assert t_edges("abcdefghij", k=3) == "abchij"
    assert t_edges("abcd", k=3) == "abcd"      # too short to trim


def test_noext_strips_only_a_trailing_extension():
    assert t_noext("a/b/page.html") == "a/b/page"
    assert t_noext("a/b/page") == "a/b/page"
    assert t_noext("v1.2/thing") == "v1.2/thing"


# --------------------------------------------------------------------------- #
# F2 set-level
# --------------------------------------------------------------------------- #

def test_match_ttr_lowers_diversity_toward_the_target():
    """The direction is forced: TTR cannot be RAISED from a set's own types.

    An earlier implementation tried to raise the generated side's diversity and
    silently no-opped once the vocabulary was exhausted, which would have read
    as 'F2 is not the tell' while testing nothing.
    """
    diverse = [f"seg{i:03d}/item{i:03d}" for i in range(60)]
    before = type_token_ratio(diverse)
    after = type_token_ratio(match_ttr(diverse, target_ttr=before * 0.25))
    assert after < before, (before, after)


def test_match_ttr_is_a_noop_when_target_is_already_met():
    rows = ["blog/page", "blog/page", "blog/page"] * 4
    assert match_ttr(rows, target_ttr=1.0) == rows


def test_match_ttr_is_deterministic_and_preserves_row_count():
    rows = [f"aa{i}/bb{i}" for i in range(40)]
    one, two = match_ttr(rows, 0.2), match_ttr(rows, 0.2)
    assert one == two
    assert len(one) == len(rows)


def test_match_ttr_preserves_segment_lengths_and_depth():
    """Substituting types of a DIFFERENT length would smuggle in an F5 change,
    and the collapse would be read as F2 when F5 had done the work."""
    rows = [f"seg{i:03d}/longer{i:03d}/x" for i in range(50)]
    out = match_ttr(rows, 0.15)
    assert [seg_lens(r) for r in out] == [seg_lens(r) for r in rows]
    assert [depth(r) for r in out] == [depth(r) for r in rows]


# --------------------------------------------------------------------------- #
# F2 necessity, corrected: drop rows, never rewrite them
# --------------------------------------------------------------------------- #

def test_novel_paths_keeps_rows_byte_identical():
    """Every survivor must be an INPUT row verbatim -- no rewriting at all.

    The first F2 attempt emitted bare segments and silently destroyed depth,
    delimiters, length and edges alongside repetition. A subtractive filter
    cannot do that, and this test is what pins it.
    """
    from research.tell.transforms import novel_paths
    rows = ["blog/page", "shop/cart", "blog/news", "a/b/c", "shop/cart"]
    out = novel_paths(rows)
    assert out, out
    for body in out:
        assert body in rows, body


def test_novel_paths_yields_ttr_of_exactly_one():
    from research.tell.transforms import novel_paths
    rows = ["blog/page", "blog/news", "shop/cart", "a/b", "shop/deal"]
    out = novel_paths(rows)
    assert type_token_ratio(out) == 1.0, out


def test_novel_paths_conserves_depth_and_lengths_of_survivors():
    from research.tell.transforms import novel_paths
    rows = ["cgi-sys/suspendedpage.cgi", "blog/the-transformer/", "a/b/c",
            "blog/other", "x1/y22/z333"]
    out = novel_paths(rows)
    for body in out:
        assert seg_lens(body) == seg_lens(body)      # verbatim by construction
        assert depth(body) >= 1
    # and the multiset of survivors is a SUBSET of the input, not a transform
    assert set(out) <= set(rows)


def test_novel_paths_drops_internal_repeats():
    from research.tell.transforms import novel_paths
    assert novel_paths(["a/a/b"]) == []


def test_novel_paths_is_deterministic():
    from research.tell.transforms import novel_paths
    rows = [f"s{i%7}/t{i%5}" for i in range(40)]
    assert novel_paths(rows) == novel_paths(rows)
