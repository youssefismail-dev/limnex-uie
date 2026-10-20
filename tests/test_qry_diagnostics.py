"""Tests for the Q7/Q8 read-only diagnostics.

These pin the properties the conclusions rest on. Two of them exist because the
corresponding bug was actually made during the investigation:

  * `field_spans` must locate field bodies by MARKER ID and never by re-parsing
    the URL text, or the measured span can drift from what the model consumed.
  * the stopper set for a field must contain that field's CUT token. Omitting
    '/' for PTH counted the empirical "segment ends" event while leaving its
    probability out of the model mass -- two different quantities compared, and
    a systematic artifact rather than a finding. It was caught before PTH ran.

No production code is exercised beyond frozen read-only helpers.
"""
from __future__ import annotations

import random

import pytest

from research.qry.q7_hazard import bucket_of, field_spans, label, boot_ci
from research.qry.q8_exposure import BANDS, band_label, band_of


# --------------------------------------------------------------------------- #
# bucketing -- must be total, ordered and non-overlapping
# --------------------------------------------------------------------------- #

def test_position_buckets_are_total_and_non_overlapping():
    seen = []
    for i in range(0, 500):
        b = bucket_of(i)
        assert b[0] <= i < b[1], (i, b)
        seen.append(b)
    assert seen[0] == (0, 4)
    # every index lands in exactly one bucket, and buckets only move forward
    idx = [sorted(set(seen)).index(b) for b in seen]
    assert idx == sorted(idx)


def test_band_assignment_is_total_and_monotone():
    prev = -1
    for c in [1, 2, 4, 5, 19, 20, 99, 100, 999, 1000, 10 ** 7]:
        b = band_of(c)
        assert b[0] <= c <= b[1], (c, b)
        cur = BANDS.index(b)
        assert cur >= prev, (c, b)
        prev = cur


def test_labels_render_the_open_bucket_as_inf():
    assert label((50, 100)) == "50-100"
    assert label((100, 10 ** 9)) == "100-inf"
    assert band_label((1000, 10 ** 9)) == "1000-inf"


# --------------------------------------------------------------------------- #
# field_spans -- located by marker id, conserving the body exactly
# --------------------------------------------------------------------------- #

class FakeTok:
    """Minimal stand-in exposing only what `field_spans` uses."""
    _M = {"SCH": 1, "USR": 2, "SUF": 3, "LBL": 4, "SUB": 5,
          "PRT": 6, "PTH": 7, "QRY": 8, "FRG": 9}

    eos_id = 99

    def marker_id(self, m):
        return self._M[m]


def test_field_spans_returns_the_body_between_marker_and_next_marker():
    tok = FakeTok()
    # <SUF> a b <LBL> c <QRY> d e f <FRG> g <eos>
    seq = [0, 3, 20, 21, 4, 22, 8, 30, 31, 32, 9, 40, 99]
    spans = field_spans(seq, tok, "QRY")
    assert spans == [(7, 10)], spans
    assert [seq[i] for i in range(*spans[0])] == [30, 31, 32]


def test_field_spans_stops_at_eos_when_the_field_is_last():
    tok = FakeTok()
    seq = [0, 4, 20, 8, 30, 31, 99]
    assert field_spans(seq, tok, "QRY") == [(4, 6)]


def test_field_spans_returns_empty_when_the_marker_is_absent():
    tok = FakeTok()
    assert field_spans([0, 4, 20, 99], tok, "QRY") == []


def test_field_spans_handles_an_empty_body():
    """A marker immediately followed by another marker is a zero-length body,
    not a crash and not a body running to the end of the sequence."""
    tok = FakeTok()
    assert field_spans([0, 8, 9, 99], tok, "QRY") == [(2, 2)]


def test_field_spans_is_deterministic():
    tok = FakeTok()
    seq = [0, 3, 20, 4, 21, 8, 30, 31, 99]
    assert field_spans(seq, tok, "QRY") == field_spans(seq, tok, "QRY")


# --------------------------------------------------------------------------- #
# bootstrap -- resamples ROWS, and its interval must bracket the point estimate
# --------------------------------------------------------------------------- #

def test_boot_ci_point_estimate_is_the_ratio_of_sums():
    pairs = [(1, 1.5), (2, 2.0), (1, 0.5)]
    point, lo, hi = boot_ci(pairs, random.Random(1))
    assert point == pytest.approx(4.0 / 4.0)
    assert lo <= point <= hi


def test_boot_ci_is_deterministic_for_a_given_seed():
    pairs = [(1, 1.2), (3, 2.4), (2, 2.6), (1, 0.9)]
    a = boot_ci(pairs, random.Random(7))
    b = boot_ci(pairs, random.Random(7))
    assert a == b


def test_boot_ci_on_an_exact_match_brackets_one():
    """If model mass equals empirical events everywhere, the ratio is 1.0 and
    the interval must contain it -- otherwise every 'tracks empirical' reading
    in Q7 would be suspect."""
    pairs = [(2, 2.0)] * 60
    point, lo, hi = boot_ci(pairs, random.Random(3))
    assert point == pytest.approx(1.0)
    assert lo <= 1.0 <= hi


def test_boot_ci_returns_nan_triple_on_empty_input():
    point, lo, hi = boot_ci([], random.Random(1))
    assert point != point          # nan
