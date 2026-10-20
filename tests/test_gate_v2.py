"""Gate v2 (research/qry/PREREG_GATE_V2.md): seeding, the section-26.1 verdict,
the unchanged pooled rule, and the generation-cache key.

All offline and fast; the data-level checks (v1 reproduces R4 exactly) live in
the registration's verification runs, not here.
"""
import math
import statistics
from pathlib import Path

import pytest

from uie.corpus.run_eval import (EXCESS_BAR, gen_cache_path, pooled_verdict,
                                 s261_verdict, split_seed, t_crit)

SEED = 20260806


# ---- seeding ----------------------------------------------------------------

def test_v1_seed_is_the_legacy_seed_for_every_generation():
    """v1 must be the pre-2026-09-23 gate: split-only seeding, shared by all
    generations -- the property that makes it reproduce published numbers."""
    for g in range(5):
        for d in range(3):
            assert split_seed(SEED, g, d, "v1") == SEED + d


def test_v2_gives_every_generation_split_its_own_seed():
    seeds = {split_seed(SEED, g, d, "v2") for g in range(5) for d in range(10)}
    assert len(seeds) == 50


def test_v2_seeds_are_disjoint_from_v1_and_generation_seeds():
    v2 = {split_seed(SEED, g, d, "v2") for g in range(100) for d in range(1000)}
    v1 = {SEED + d for d in range(1000)}
    gen = {SEED + 1 + 1000 * g for g in range(100)}
    assert not v2 & v1
    assert not v2 & gen


@pytest.mark.parametrize("g, d", [(100, 0), (0, 1000), (-1, 0), (0, -1)])
def test_v2_refuses_indices_outside_its_seed_block(g, d):
    with pytest.raises(ValueError):
        split_seed(SEED, g, d, "v2")


def test_an_unknown_estimator_is_refused():
    with pytest.raises(ValueError):
        split_seed(SEED, 0, 0, "v3")


# ---- the section-26.1 verdict --------------------------------------------------

def test_t_crit_is_the_two_sided_95_quantile():
    assert t_crit(4) == pytest.approx(2.7764, abs=1e-4)
    assert t_crit(2) == pytest.approx(4.3027, abs=1e-4)


def test_se_is_sample_sd_over_sqrt_G_not_sqrt_G_minus_1():
    """Section 26.1's form. The published t-values used sqrt(G - 1)."""
    gm = [0.06, 0.07, 0.08, 0.09, 0.10]
    a = s261_verdict([[m] for m in gm])
    assert a["x"] == pytest.approx(0.08)
    assert a["se"] == pytest.approx(statistics.stdev(gm) / math.sqrt(5), abs=1e-4)
    assert a["se"] != pytest.approx(statistics.stdev(gm) / math.sqrt(4), abs=1e-4)


def test_generation_mean_averages_its_splits_first():
    """D splits are averaged WITHIN a generation; the SE is across generations."""
    a = s261_verdict([[0.0, 0.2], [0.1, 0.1], [0.05, 0.15]])
    assert a["generation_means"] == [0.1, 0.1, 0.1]
    assert a["x"] == pytest.approx(0.1)


def test_convicted_cleared_and_undecided():
    convicted = s261_verdict([[m] for m in (0.06, 0.07, 0.08, 0.09, 0.10)])
    cleared = s261_verdict([[m] for m in (0.00, 0.01, 0.02, 0.01, 0.00)])
    undecided = s261_verdict([[m] for m in (0.02, 0.08, 0.04, 0.07, 0.05)])
    assert convicted["verdict"] == "CONVICTED" and convicted["t_vs_bar"] > 2.7764
    assert cleared["verdict"] == "CLEARED" and -cleared["t_vs_bar"] > 2.7764
    assert undecided["verdict"] == "UNDECIDED"


def test_the_bar_is_unchanged():
    assert EXCESS_BAR == 0.05


def test_a_generation_without_a_usable_draw_is_not_computable():
    """A missing generation mean is never invented, and G is never shrunk."""
    a = s261_verdict([[0.1], [], [0.1], [0.1], [0.1]])
    assert a["verdict"] == "NOT COMPUTABLE"
    assert a["generations_usable"] == 4 and a["generations"] == 5


def test_a_single_generation_is_not_computable():
    assert s261_verdict([[0.1, 0.2, 0.3]])["verdict"] == "NOT COMPUTABLE"


# ---- the pooled rule, unchanged ------------------------------------------------

def test_pooled_rule_convicts_or_abstains_and_never_acquits():
    assert pooled_verdict([0.06, 0.07, 0.08]) == "TELL"
    assert pooled_verdict([0.04, 0.07, 0.08]) == "UNDECIDED"
    assert pooled_verdict([0.00, 0.01, 0.02]) == "UNDECIDED"   # never "ok"
    assert pooled_verdict([]) == "UNDECIDED"


# ---- the generation cache key -----------------------------------------------------

def test_cache_key_changes_with_every_parameter_that_determines_a_generation():
    base = dict(cache_dir=Path("c"), ckpt_tag="best_e5-abc", label="benign",
                n_main=8000, decoder="temperature",
                decoder_kwargs={"temperature": 1.0}, delta=0.22,
                gen_seed=20260807)
    ref = gen_cache_path(**base)
    for key, other in (("ckpt_tag", "best_e5-xyz"), ("label", "phishing"),
                       ("n_main", 1200), ("decoder", "typical"),
                       ("decoder_kwargs", {"temperature": 0.9}),
                       ("delta", 0.30), ("gen_seed", 20261807)):
        assert gen_cache_path(**{**base, key: other}) != ref, key


def test_cache_key_ignores_kwarg_order():
    a = gen_cache_path(Path("c"), "t", "benign", 8000, "top_p",
                       {"p": 0.9, "temperature": 1.0}, 0.22, 1)
    b = gen_cache_path(Path("c"), "t", "benign", 8000, "top_p",
                       {"temperature": 1.0, "p": 0.9}, 0.22, 1)
    assert a == b
