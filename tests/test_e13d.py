"""E13-D -- the registered rules and the new instruments (research/e13/PREREG_E13D.md),
on synthetic inputs, proven before any real measurement."""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "e13"))

import e13d_common as D  # noqa: E402


@pytest.mark.parametrize("v,cls", [(None, "-"), ("", "e"), ("123", "D"), ("abc", "A"),
                                   ("8c5781c2-7b1e-4035-b1d7-990e5e2f3a1b", "U"),
                                   ("a@b.c", "E"), ("deadbeef12", "H"), ("ab-12", "X"), ("abcdef", "A")])
def test_value_classes(v, cls):
    assert D.value_class(v) == cls


def test_templates_keep_key_order_and_classify_values():
    assert D.template("m=1&email=x@y.z&naps") == "m=D&email=E&naps=-"
    assert D.template("page=2&lang=en") == "page=D&lang=A"
    assert D.template("a=") == "a=e"
    assert D.template("page=2&lang=en") != D.template("lang=en&page=2")     # order matters


def test_the_ngram_learner_is_seeded_and_writes_only_seen_characters():
    data = ["page=2&lang=en", "page=3&lang=fr", "id=7", "m=1&i=1"]
    m = D.CharNGram(order=6).fit(data)
    a = m.sample(50, random.Random(1))
    b = m.sample(50, random.Random(1))
    assert a == b
    seen = set("".join(data))
    assert all(set(s) <= seen for s in a)


def test_a_single_string_is_reproduced_exactly():
    m = D.CharNGram(order=6).fit(["email=3mail@b.c"])
    assert m.sample(3, random.Random(0)) == ["email=3mail@b.c"] * 3


def test_subsample_is_seeded_and_full_fraction_is_identity():
    xs = [str(i) for i in range(100)]
    assert D.subsample(xs, 1.0, 5) == xs
    assert D.subsample(xs, 0.25, 5) == D.subsample(xs, 0.25, 5)
    assert len(D.subsample(xs, 0.25, 5)) == 25


def test_dx1_shift_rule():
    gv = [0.10] * 12
    assert D.shift_reading([0.06 + 0.001 * i for i in range(12)], gv)["reading"] == "SHIFT-DOMINANT"
    assert D.shift_reading([0.03 + 0.001 * i for i in range(12)], gv)["reading"] == "SHIFT-PRESENT"
    noisy = [0.05, -0.05] * 6
    assert D.shift_reading(noisy, gv)["reading"] == "NO SHIFT"                 # not significant
    assert D.shift_reading([0.01 + 0.0001 * i for i in range(12)], gv)["reading"] == "NO SHIFT"   # small


def test_dx2_gap_rule_and_replication():
    gv = [0.10] * 12
    assert D.gap_reading([0.08 + 0.001 * i for i in range(12)], gv)["reading"] == "LEARNING-GAP"
    assert D.gap_reading([0.02 + 0.001 * i for i in range(12)], gv)["reading"] == "NO GAP"
    assert D.replicated("LEARNING-GAP", "LEARNING-GAP") == "LEARNING-GAP"
    assert D.replicated("LEARNING-GAP", "NO GAP") == "NOT REPLICATED"


@pytest.mark.parametrize("shift,gap,expect", [
    ("SHIFT-DOMINANT", "NO GAP", "DATA (population)"),
    ("NO SHIFT", "LEARNING-GAP", "LEARNING"),
    ("SHIFT-PRESENT", "LEARNING-GAP", "BOTH"),
    ("SHIFT-DOMINANT", "LEARNING-GAP", "BOTH"),
    ("NO SHIFT", "NO GAP", "UNRESOLVED"),
    ("SHIFT-PRESENT", "NO GAP", "UNRESOLVED"),
    ("NO SHIFT", "NOT REPLICATED", "UNRESOLVED"),
])
def test_the_attribution_table(shift, gap, expect):
    assert D.attribution(shift, gap) == expect


def test_dx3_needs_both_coverage_and_the_learner():
    half = [0.20 + 0.001 * i for i in range(12)]
    full = [0.18 + 0.001 * i + 0.0005 * (i % 2) for i in range(12)]
    assert D.quantity_reading(0.03, half, full)["reading"] == "DATA-LIMITED"
    assert D.quantity_reading(0.01, half, full)["reading"] == "NOT DATA-LIMITED"   # coverage flat
    assert D.quantity_reading(0.03, half, half)["reading"] == "NOT DATA-LIMITED"   # learner flat


def test_dx5_obedience_rule():
    uniform = [0.003] * 100
    assert D.obedience_reading(uniform, [100] * 100)["reading"] == "UNIFORM"
    conc = [0.0001] * 90 + [0.05] * 10
    support = [1000] * 90 + [5] * 10
    assert D.obedience_reading(conc, support)["reading"] == "CONCENTRATED"
    assert D.obedience_reading(conc, [1000] * 100)["reading"] == "MIXED"          # concentrated, not thin


def test_dx4_clear_probability():
    r = D.clear_probability(mean_x=0.0411, sd_run=0.0024, se=0.0032)
    assert abs(r["line"] - (0.05 - 2.7764 * 0.0032)) < 1e-12
    assert 0.4 < r["p_clear"] < 0.6                                               # at the line: a coin flip
    far = D.clear_probability(mean_x=0.11, sd_run=0.008, se=0.0047)
    assert far["p_clear"] < 1e-6
    assert far["x_for_0.8"] < far["line"]


def test_the_cells_are_the_registered_six_and_the_reference():
    assert [c[0] for c in D.CELLS] == ["free_phishing_QRY", "control_benign_P0Q1_QRY", "control_benign_P1Q1_QRY",
                                       "control_benign_P1Q1_PTH", "control_benign_P1Q1_SUB",
                                       "control_phishing_P1Q0_PTH"]
    assert D.REFERENCE[0] == "free_benign_QRY" and D.REFERENCE in D.QUERY_CELLS


# ---- amendment 2: DX7 ----------------------------------------------------------

def test_witten_bell_sums_to_one_and_scores_unseen_characters():
    m = D.WittenBell(order=4).fit(["page=2&lang=en", "page=3&lang=fr", "m=1"])
    for h in ("\x02\x02\x02", "pag", "zzz", "ng="):
        total = sum(m.prob(c, h) for c in m.alphabet + [m.END, "\x00"])
        assert abs(total - 1) < 1e-9
    assert m.prob("Q", "pag") > 0                         # never seen: still finite
    assert m.nll("page=2&lang=en") < m.nll("zq#k!")      # a seen string is likelier


def test_fit_rule_reads_all_three_outcomes():
    assert D.fit_reading([0.3 + 0.01 * (i % 5) for i in range(200)])["reading"] == "UNDER-FIT"
    assert D.fit_reading([-0.3 - 0.01 * (i % 5) for i in range(200)])["reading"] == "FITS"
    assert D.fit_reading([(-1) ** i * 0.5 for i in range(200)])["reading"] == "UNRESOLVED"
