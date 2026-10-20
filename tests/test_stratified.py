"""
Tests for suffix-stratified block allocation.

The allocator is pure arithmetic over a shard map, so it is tested without any
network. What matters is that it does the one thing it exists to do — stop a
suffix's share of the budget from tracking its share of the index — while still
respecting that `.com` really is the largest namespace on the web.
"""
from __future__ import annotations

import pytest

from uie.corpus.stratified import (DEFAULT_ALPHA, ShardMap, allocate,
                                   shard_weights)


def _share_of_weights(shard_map, suffix, *, alpha=DEFAULT_ALPHA):
    """The share the allocator is AIMING at, independent of any draw."""
    weights = shard_weights(shard_map, alpha=alpha)
    strata = shard_map.strata()
    return sum(weights[s] for s in strata[suffix]) / sum(weights.values())


def _map(**suffix_to_shards) -> ShardMap:
    """Build a map from {suffix: [shards]} for readability in tests."""
    shards = {}
    for suffix, indices in suffix_to_shards.items():
        for i in indices:
            shards.setdefault(i, []).append(suffix)
    return ShardMap(crawl="test", shards=shards, n_shards=max(shards) + 1)


def _share(plan, shard_map, suffix):
    strata = shard_map.strata()
    total = sum(plan.values())
    return sum(plan.get(s, 0) for s in strata[suffix]) / total


def test_allocation_totals_the_requested_blocks():
    """Rounding each shard independently drifts by tens of blocks over 300
    shards, so largest-remainder rounding must hit the total exactly."""
    m = _map(com=list(range(30, 155)), de=[160, 161], jp=[200], org=[240, 241])
    for n in (10, 100, 137, 1000):
        assert sum(allocate(m, n).values()) == n


def _realistic_map() -> ShardMap:
    """Shaped like the real CC-MAIN-2026-30 index: `.com` holds ~125 of 300
    shards and a long tail of ~200 suffixes holds one shard each.

    The shape matters. An earlier version of this test used six suffixes, where
    alpha=0.5 leaves `.com` at 52%, and asserted a magnitude drawn from that toy
    — on the real distribution the same alpha gives ~4%. Magnitudes depend on
    the whole stratum count, so the assertions below are about the MECHANISM.
    """
    suffixes = {"com": list(range(30, 155))}
    tail = [s for s in range(0, 30)] + [s for s in range(155, 300)]
    for i, shard in enumerate(tail):
        suffixes[f"t{i:03d}"] = [shard]
    return _map(**suffixes)


def test_flattening_cuts_com_share_and_keeps_it_largest():
    """`.com` spans ~42% of the real index, so proportional sampling hands it
    ~42% of the budget before a URL is read. That is the bias being removed —
    but `.com` must stay the LARGEST stratum, because a corpus weighting `.ai`
    and `.com` equally teaches an Internet that does not exist."""
    m = _realistic_map()
    proportional = _share(allocate(m, 20_000, alpha=1.0), m, "com")
    flattened = _share(allocate(m, 20_000, alpha=DEFAULT_ALPHA), m, "com")

    assert 0.35 < proportional < 0.50, proportional      # tracks index share
    assert flattened < proportional / 4, (proportional, flattened)
    assert flattened > _share(allocate(m, 20_000), m, "t000")


def test_alpha_monotonically_controls_concentration():
    """Alpha has to be a usable dial: higher alpha, more concentrated. It is
    tuned against the real shard map, so the ordering must be reliable."""
    m = _realistic_map()
    shares = [_share(allocate(m, 20_000, alpha=a), m, "com")
              for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert shares == sorted(shares), shares
    assert shares[0] < 0.02 and shares[-1] > 0.35, shares


def test_alpha_zero_equalises_suffixes_not_shards():
    """The unit being flattened is the SUFFIX. At alpha=0 a suffix holding 125
    shards and one holding 1 get the same total budget."""
    m = _map(com=list(range(0, 125)), ai=[200])
    plan = allocate(m, 4000, alpha=0.0)
    assert _share(plan, m, "com") == pytest.approx(_share(plan, m, "ai"), abs=0.02)


def test_alpha_one_reproduces_proportional_sampling():
    m = _map(com=list(range(0, 100)), ai=[200])
    plan = allocate(m, 4040, alpha=1.0)
    assert _share(plan, m, "com") == pytest.approx(100 / 101, abs=0.02)


def test_wide_strata_are_not_excluded_at_small_budgets():
    """The regression that the 200-block pilot exposed.

    Partitioning floored each shard's quota, so `.com` — spread over 126 shards
    at 0.16 blocks each — floored to zero while single-shard strata sat at 0.86
    and swept the whole remainder. The pilot drew ZERO `.com` blocks against an
    intended 10.1%, and above ~2,000 blocks the same code was correct, so the
    defect would have survived into production and quietly corrupted every small
    diagnostic run.

    Sampling is unbiased at every budget, so the expected share must hold at 200
    blocks just as it does at 16,000.
    """
    m = _realistic_map()
    intended = _share_of_weights(m, "com", alpha=DEFAULT_ALPHA)

    for budget in (200, 2_000, 16_000):
        # Average several seeds: one small draw is noisy, the ESTIMATOR is what
        # has to be unbiased.
        shares = [_share(allocate(m, budget, seed=s), m, "com") for s in range(12)]
        mean = sum(shares) / len(shares)
        assert abs(mean - intended) < 0.03, (budget, mean, intended)
        assert mean > 0.0


def test_allocation_is_reproducible_under_seed():
    """E0's gate needs a stranger to regenerate the corpus; the URL list itself
    cannot be redistributed."""
    m = _realistic_map()
    assert allocate(m, 500, seed=3) == allocate(m, 500, seed=3)
    assert allocate(m, 500, seed=3) != allocate(m, 500, seed=4)


def test_boundary_shards_belong_to_both_neighbours():
    """A shard whose probe shows two suffixes is credited to both. Over-counting
    is the safe direction: it can never leave a suffix unreachable."""
    m = ShardMap(crawl="test", n_shards=3,
                 shards={0: ["com"], 1: ["com", "de"], 2: ["de"]})
    strata = m.strata()
    assert strata["com"] == [0, 1]
    assert strata["de"] == [1, 2]


def test_empty_map_or_zero_budget_yields_nothing():
    assert allocate(ShardMap(crawl="x"), 100) == {}
    assert allocate(_map(com=[0]), 0) == {}


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_alpha_outside_the_unit_interval_raises(alpha):
    with pytest.raises(ValueError):
        allocate(_map(com=[0]), 10, alpha=alpha)


def test_round_trip_through_disk(tmp_path):
    m = _map(com=[0, 1], de=[2])
    path = tmp_path / "map.json"
    m.save(path)
    loaded = ShardMap.load(path)
    assert loaded.crawl == m.crawl
    assert loaded.shards == m.shards          # int keys survive JSON's strings
    assert loaded.strata() == m.strata()


# --- the partial-map hazard ------------------------------------------------

def test_saving_an_incomplete_map_is_refused(tmp_path):
    """A build interrupted by network trouble once saved 97/300 shards covering
    `ai`..`com` and nothing after it. The file loaded without complaint,
    reported 18 strata, and would have collected a corpus containing no `de`,
    `jp`, `org`, `ru` or `uk` at all.

    Shards are probed in index order and the index is SURT-sorted, so a partial
    map is a contiguous SLICE OF THE TLD ALPHABET — biased, not merely smaller.
    """
    partial = ShardMap(crawl="x", n_shards=300,
                       shards={i: ["com"] for i in range(97)})
    assert not partial.complete
    with pytest.raises(ValueError, match="ALPHABET|alphabet"):
        partial.save(tmp_path / "partial.json")


def test_loading_an_incomplete_map_is_refused(tmp_path):
    """Guarded at BOTH ends: a file written by an older version, or by hand,
    must not be sampled from either."""
    partial = ShardMap(crawl="x", n_shards=300,
                       shards={i: ["com"] for i in range(97)})
    path = tmp_path / "partial.json"
    partial.save(path, allow_partial=True)            # explicit escape hatch
    with pytest.raises(ValueError, match="incomplete"):
        ShardMap.load(path)
    assert ShardMap.load(path, allow_partial=True).coverage < 0.4


def test_a_complete_map_round_trips_without_the_escape_hatch(tmp_path):
    full = ShardMap(crawl="x", n_shards=100,
                    shards={i: ["com"] for i in range(100)})
    path = tmp_path / "full.json"
    full.save(path)
    assert ShardMap.load(path).complete


def _offsets_read(monkeypatch, fail_once_at=()):
    """The offsets a sampling run reads, through a fake index that fails the first read of
    each offset listed (a transient network error)."""
    import uie.corpus.stratified as st
    from uie.corpus.commoncrawl import CrawlIndex, SampleStats
    failed, read = set(), []

    def fake_get(url, headers=None, method="GET"):
        offset = int(headers["Range"].split("=")[1].split("-")[0])
        if offset in fail_once_at and offset not in failed:
            failed.add(offset)
            raise RuntimeError("transient")
        read.append(offset)
        return b"x"

    monkeypatch.setattr(st, "_get", fake_get)
    monkeypatch.setattr(st.time, "sleep", lambda s: None)
    monkeypatch.setattr(CrawlIndex, "_size", lambda self, shard: 50_000_000)
    monkeypatch.setattr(CrawlIndex, "_shard_url", lambda self, shard: f"shard-{shard}")
    monkeypatch.setattr(CrawlIndex, "_resync", staticmethod(lambda blob: "text"))
    monkeypatch.setattr(CrawlIndex, "_parse", staticmethod(lambda text, crawl, stats: iter(())))
    smap = ShardMap(crawl="CC-TEST", shards={0: ["com"], 1: ["org"], 2: ["de"]}, n_shards=3)
    stats = SampleStats()
    list(st.sample_stratified("CC-TEST", 9, seed=5, shard_map=smap, stats=stats))
    return read, stats


def test_a_transient_failure_rereads_the_same_offset_so_the_run_is_unchanged(monkeypatch):
    # D2 step 6: a failed read used to draw a NEW offset, so a rebuild that met network errors
    # sampled different blocks (66% URL overlap with the original crawl)
    clean, s0 = _offsets_read(monkeypatch)
    flaky, s1 = _offsets_read(monkeypatch, fail_once_at=(clean[1], clean[4]))
    assert flaky == clean and len(clean) == 9
    assert s0.n_http_errors == 0 and s1.n_http_errors == 2 and s1.n_lost_draws == 0
