"""Each query transform must destroy its OWN property and conserve the others.

These pin CONSERVATION LAWS rather than outputs. The project has one withdrawn
headline caused by a transform that claimed to remove repetition and silently
destroyed depth, delimiters, length and edges too -- and no test caught it,
because only the per-string transforms were covered. These tests exist so the
query ladder cannot repeat that.
"""
from __future__ import annotations

import pytest

from research.qry.qtransforms import (TRANSFORMS, split_params, t_count_only,
                                      t_fixlen_values, t_identity,
                                      t_mask_alpha, t_mask_digits,
                                      t_mask_encoding, t_mask_names,
                                      t_mask_values, t_null, t_skeleton,
                                      t_sort_params, t_structure_only)

QUERIES = [
    "id=131&page=2",
    "language=complete_artistique&id=131&language=hello-world-cartier",
    "no=1765",
    "q=caf%C3%A9&lang=fr",
    "debug",
    "a=",
    "=orphan",
    "add-to-cart=4508&quantity=1",
    "lang=bong/weddingplay.shtml",
]


def count(q):
    return len(split_params(q))


def names(q):
    return [k for k, _, _ in split_params(q)]


def seps(q):
    return [s for _, s, _ in split_params(q)]


def val_lens(q):
    return [len(v) for _, _, v in split_params(q)]


# --------------------------------------------------------------------------- #
# determinism and totality
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_transform_is_deterministic(name):
    fn = TRANSFORMS[name]
    for q in QUERIES:
        assert fn(q) == fn(q) == fn(q), name


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_transform_is_total_and_returns_str(name):
    fn = TRANSFORMS[name]
    for q in QUERIES + ["", "&", "&&", "a&", "&a", "=", "%", "%A"]:
        assert isinstance(fn(q), str), (name, q)


def test_null_control_changes_nothing():
    """`null` defines the measurement's zero; if it moves, nothing else means
    anything."""
    for q in QUERIES:
        assert t_null(q) == t_identity(q) == q, q


# --------------------------------------------------------------------------- #
# parameter count survives everything except the count-only projection
# --------------------------------------------------------------------------- #

#: Transforms that change parameter COUNT BY DESIGN and are therefore exempt
#: from the conservation law below. Listed explicitly so that a future
#: transform cannot join them by accident.
COUNT_CHANGING = {"count_only", "first_param", "clip15_first"}


@pytest.mark.parametrize("name", sorted(set(TRANSFORMS) - COUNT_CHANGING))
def test_parameter_count_is_conserved(name):
    fn = TRANSFORMS[name]
    for q in QUERIES:
        assert count(fn(q)) == count(q), (name, q)


def test_count_only_conserves_count_and_nothing_else():
    for q in QUERIES:
        out = t_count_only(q)
        assert count(out) == count(q), q
        assert set(out) <= {"a", "=", "&"}, out


# --------------------------------------------------------------------------- #
# name identity
# --------------------------------------------------------------------------- #

def test_mask_names_kills_identity_but_keeps_lengths_values_and_seps():
    for q in QUERIES:
        out = t_mask_names(q)
        assert [len(k) for k in names(out)] == [len(k) for k in names(q)], q
        assert seps(out) == seps(q), q
        assert [v for _, _, v in split_params(out)] == \
               [v for _, _, v in split_params(q)], q
        assert all(set(k) <= {"a"} for k in names(out)), out


# --------------------------------------------------------------------------- #
# value composition vs value length -- the pair that isolates LENGTH
# --------------------------------------------------------------------------- #

def test_mask_values_keeps_value_lengths():
    """This is what makes `mask_values` vs `fixlen_values` an isolation of
    length: identical except that one conserves length and one does not."""
    for q in QUERIES:
        out = t_mask_values(q)
        assert val_lens(out) == val_lens(q), q
        assert names(out) == names(q), q
        assert seps(out) == seps(q), q


def test_fixlen_values_makes_every_present_value_constant_width():
    for q in QUERIES:
        out = t_fixlen_values(q)
        widths = {len(v) for _, s, v in split_params(out) if s}
        assert not widths or widths == {8}, (q, out)
        assert names(out) == names(q), q
        assert seps(out) == seps(q), q


def test_fixlen_pads_with_a_character_the_corpus_already_uses():
    """A novel pad character would itself become a perfect tell."""
    assert "a" in t_fixlen_values("k=x")


# --------------------------------------------------------------------------- #
# character classes -- length and position must survive
# --------------------------------------------------------------------------- #

def test_mask_digits_preserves_length_and_every_non_digit():
    for q in QUERIES:
        out = t_mask_digits(q)
        assert len(out) == len(q), q
        for a, b in zip(q, out):
            assert (a.isdigit() and b == "0") or (not a.isdigit() and a == b)


def test_mask_alpha_preserves_length_and_digit_layout():
    for q in QUERIES:
        out = t_mask_alpha(q)
        assert len(out) == len(q), q
        for a, b in zip(q, out):
            if a.isalpha():
                assert b == "a"
            else:
                assert b == a


# --------------------------------------------------------------------------- #
# percent-encoding: identity dies, PRESENCE and position survive
# --------------------------------------------------------------------------- #

def test_mask_encoding_keeps_presence_position_and_length():
    q = "q=caf%C3%A9&lang=fr"
    out = t_mask_encoding(q)
    assert len(out) == len(q)
    assert out.count("%") == q.count("%")
    assert [i for i, c in enumerate(out) if c == "%"] == \
           [i for i, c in enumerate(q) if c == "%"]
    assert "%00" in out and "%C3" not in out


def test_mask_encoding_leaves_unencoded_queries_alone():
    for q in ["id=131&page=2", "no=1765"]:
        assert t_mask_encoding(q) == q


# --------------------------------------------------------------------------- #
# ordering -- the one transform that must conserve the multiset exactly
# --------------------------------------------------------------------------- #

def test_sort_params_conserves_the_parameter_multiset():
    """Order is destroyed; nothing else may be. If a name, value or separator
    changed here, a collapse under `sort_params` would be read as 'ordering is
    the tell' when something else did the work."""
    for q in QUERIES:
        out = t_sort_params(q)
        assert sorted(split_params(out)) == sorted(split_params(q)), q


def test_sort_params_actually_reorders_something():
    assert t_sort_params("z=1&a=2") == "a=2&z=1"


def test_sort_params_preserves_repetition():
    """Repetition is a SEPARATE registered component; sorting must not remove
    it, or ordering and repetition become confounded."""
    q = "language=x&id=1&language=y"
    out = t_sort_params(q)
    assert names(out).count("language") == 2, out


# --------------------------------------------------------------------------- #
# isolating projections
# --------------------------------------------------------------------------- #

def test_skeleton_erases_identity_but_not_punctuation_or_length():
    assert t_skeleton("add-to-cart=4508&q=a1") == "aaa-aa-aaaa=0000&a=a0"
    for q in QUERIES:
        assert len(t_skeleton(q)) == len(q), q


def test_structure_only_retains_lengths_separators_and_count():
    for q in QUERIES:
        out = t_structure_only(q)
        assert [len(k) for k in names(out)] == [len(k) for k in names(q)], q
        assert val_lens(out) == val_lens(q), q
        assert seps(out) == seps(q), q
        assert set(out) <= {"a", "=", "&"}, out


# --------------------------------------------------------------------------- #
# the confound that `mask_names` hides, and the transform that avoids it
# --------------------------------------------------------------------------- #

def test_mask_names_is_documented_as_destroying_intra_name_punctuation():
    """`mask_names` is CONFOUNDED and this pins the confound so it cannot be
    forgotten: it deletes underscores, hyphens and digits inside a name along
    with the name's identity. Three of the ten commonest real parameter names
    carry an underscore, so this is not a corner case."""
    assert t_mask_names("utm_source=x") == "aaaaaaaaaa=x"
    assert "_" not in t_mask_names("page_id=1")


def test_mask_name_chars_kills_identity_but_keeps_intra_name_punctuation():
    from research.qry.qtransforms import t_mask_name_chars
    assert t_mask_name_chars("utm_source=x") == "aaa_aaaaaa=x"
    assert t_mask_name_chars("page_id=1&a2=b") == "aaaa_aa=1&a0=b"
    for q in QUERIES:
        out = t_mask_name_chars(q)
        assert [len(k) for k in names(out)] == [len(k) for k in names(q)], q
        assert seps(out) == seps(q), q
        assert [v for _, _, v in split_params(out)] == \
               [v for _, _, v in split_params(q)], q
        # every punctuation character inside a name survives, in place
        for a, b in zip(names(q), names(out)):
            for ca, cb in zip(a, b):
                if not ca.isalnum():
                    assert cb == ca, (q, a, b)


# =========================================================================== #
# Q3.1 -- finer shape decomposition
# =========================================================================== #

from research.qry.qtransforms import (t_alnum_flat, t_clip15_first,  # noqa: E402
                                      t_clip_values_10, t_clip_values_15,
                                      t_clip_values_20, t_first_param,
                                      t_fixlen_names, t_mask_pct,
                                      t_mask_punct, t_normalize_seps)


@pytest.mark.parametrize("fn,k", [(t_clip_values_20, 20), (t_clip_values_15, 15),
                                  (t_clip_values_10, 10)])
def test_clip_values_caps_length_and_changes_nothing_else(fn, k):
    """The cap is the ONLY thing that may move. Names, separators, count and
    every value already under the cap must come back byte-identical, or a
    collapse would be read as 'the long tail' when something else did it."""
    for q in QUERIES:
        out = fn(q)
        assert names(out) == names(q), q
        assert seps(out) == seps(q), q
        assert count(out) == count(q), q
        for (_, _, a), (_, _, b) in zip(split_params(q), split_params(out)):
            assert len(b) <= k, (q, b)
            if len(a) <= k:
                assert b == a, (q, a, b)      # untouched below the cap
            else:
                assert b == a[:k], (q, a, b)  # pure prefix, never a rewrite


def test_clip_series_is_monotone_in_aggressiveness():
    """A dose-response reading is only meaningful if the doses are ordered."""
    q = "a=" + "x" * 30
    assert len(split_params(t_clip_values_20(q))[0][2]) == 20
    assert len(split_params(t_clip_values_15(q))[0][2]) == 15
    assert len(split_params(t_clip_values_10(q))[0][2]) == 10


def test_first_param_keeps_parameter_one_byte_identical():
    """Survival under this transform is the clean reading, so parameter one
    must be untouched -- otherwise survival could be an artifact of rewriting."""
    for q in QUERIES:
        out = t_first_param(q)
        assert count(out) <= 1, (q, out)
        if split_params(q):
            assert split_params(out)[0] == split_params(q)[0], q


def test_fixlen_names_makes_every_name_constant_width():
    for q in QUERIES:
        out = t_fixlen_names(q)
        assert {len(k) for k in names(out)} == {8}, (q, out)
        assert seps(out) == seps(q), q
        assert [v for _, _, v in split_params(out)] == \
               [v for _, _, v in split_params(q)], q


def test_mask_pct_removes_presence_but_keeps_length():
    q = "q=caf%C3%A9&lang=fr"
    out = t_mask_pct(q)
    assert len(out) == len(q)
    assert "%" not in out
    assert t_mask_pct("id=131") == "id=131"      # no-op when unencoded


def test_normalize_seps_gives_every_parameter_an_equals():
    for q in QUERIES:
        out = t_normalize_seps(q)
        assert all(s == "=" for s in seps(out)), (q, out)
        assert names(out) == names(q), q
        assert [v for _, _, v in split_params(out)] == \
               [v for _, _, v in split_params(q)], q


def test_mask_punct_keeps_length_alnum_and_structural_separators():
    for q in QUERIES:
        out = t_mask_punct(q)
        assert len(out) == len(q), q
        assert count(out) == count(q), q
        for a, b in zip(q, out):
            if a.isalnum() or a in "&=":
                assert b == a, (q, a, b)
            else:
                assert b == "x", (q, a, b)


def test_alnum_flat_keeps_length_and_all_punctuation():
    for q in QUERIES:
        out = t_alnum_flat(q)
        assert len(out) == len(q), q
        for a, b in zip(q, out):
            assert b == ("a" if a.isalnum() else a), (q, a, b)


def test_alnum_flat_differs_from_skeleton_only_on_digits():
    """The PAIR is what isolates the digit-vs-letter distinction; if they ever
    agreed on a digit the isolation would be empty."""
    q = "id=131&page_id=2"
    assert t_skeleton(q) == "aa=000&aaaa_aa=0"
    assert t_alnum_flat(q) == "aa=aaa&aaaa_aa=a"
    for a, b in zip(t_skeleton(q), t_alnum_flat(q)):
        assert a == b or (a == "0" and b == "a")


def test_clip15_first_is_exactly_its_two_components():
    """The pre-specified interaction must be the composition of the two
    transforms it claims to combine, and nothing more."""
    for q in QUERIES:
        assert t_clip15_first(q) == t_first_param(t_clip_values_15(q)), q
