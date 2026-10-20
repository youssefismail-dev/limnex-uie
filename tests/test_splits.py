"""
Tests for frozen, leakage-safe splits.

Three properties are load-bearing, and a failure in any one silently inflates
every number measured afterwards:

  * DOMAIN-DISJOINTNESS — no registrable domain may appear on both sides, or the
    model scores well by having memorised a site.
  * TEMPORAL HONESTY — eval must contain only domains unseen in the training
    window, otherwise it measures "new page on a known site" rather than
    generalization to new sites.
  * IMMUTABILITY — a split that can be silently regenerated is not a split. The
    hash check is what makes results comparable across time.
"""
from __future__ import annotations

import json

import pytest

from uie.splits import (EVAL, TRAIN, Record, apply_split, build_random_split,
                        build_temporal_split, group_key, load_split, save_split)


def _rec(url: str, label: str = "benign", observed: str = "") -> Record:
    return Record(url=url, label=label, observed=observed)


# --- grouping --------------------------------------------------------------

def test_group_key_is_the_registrable_domain():
    assert group_key("https://a.shopwave.io/x") == "shopwave.io"
    assert group_key("https://b.shopwave.io/y") == "shopwave.io"
    assert group_key("https://shop.example.co.uk/") == "example.co.uk"


def test_ip_hosts_group_by_host_not_by_url():
    """Falling back to the raw URL would silently disable grouping for exactly
    the rows that have no registrable domain."""
    assert group_key("http://192.168.1.1/a") == group_key("http://192.168.1.1/b")
    assert group_key("http://192.168.1.1/a") != group_key("http://10.0.0.1/a")


# --- random split: disjointness --------------------------------------------

def _corpus(n_domains: int = 100, per_domain: int = 5) -> list[Record]:
    return [_rec(f"https://site{d}.com/page{p}")
            for d in range(n_domains) for p in range(per_domain)]


def test_no_domain_appears_in_two_partitions():
    records = _corpus()
    split = build_random_split(records)
    parts = apply_split(records, split)
    seen: dict[str, str] = {}
    for name, rows in parts.items():
        for row in rows:
            key = group_key(row.url)
            assert seen.setdefault(key, name) == name, f"{key} leaked across splits"


def test_every_record_is_assigned():
    records = _corpus()
    parts = apply_split(records, build_random_split(records))
    assert sum(len(v) for v in parts.values()) == len(records)


def test_fractions_are_approximately_respected():
    split = build_random_split(_corpus(1000), fractions={TRAIN: 0.8, "val": 0.1,
                                                         "test": 0.1})
    counts = split.counts()
    assert counts[TRAIN] == pytest.approx(800, abs=5)
    assert counts["val"] == pytest.approx(100, abs=5)


def test_no_domains_are_lost_to_rounding():
    """The last partition absorbs the remainder, so the assignment is total."""
    split = build_random_split(_corpus(7), fractions={TRAIN: 0.7, "test": 0.3})
    assert sum(split.counts().values()) == 7


def test_class_proportions_are_preserved():
    records = ([_rec(f"https://b{i}.com/x", "benign") for i in range(900)] +
               [_rec(f"https://p{i}.tk/x", "phishing") for i in range(100)])
    parts = apply_split(records, build_random_split(records))
    for rows in parts.values():
        share = sum(r.label == "phishing" for r in rows) / len(rows)
        assert share == pytest.approx(0.1, abs=0.04)


def test_split_is_deterministic_under_seed():
    records = _corpus()
    a = build_random_split(records, seed=7)
    b = build_random_split(records, seed=7)
    assert a.assignment == b.assignment and a.content_hash() == b.content_hash()


def test_different_seeds_give_different_splits():
    records = _corpus()
    assert (build_random_split(records, seed=1).assignment
            != build_random_split(records, seed=2).assignment)


def test_input_order_does_not_change_the_split():
    """Domains are sorted before shuffling, so shard read order — which varies
    with the filesystem — cannot alter a frozen artifact."""
    records = _corpus()
    shuffled = list(reversed(records))
    assert (build_random_split(records, seed=3).assignment
            == build_random_split(shuffled, seed=3).assignment)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        build_random_split(_corpus(10), fractions={TRAIN: 0.5, "test": 0.2})


# --- temporal split --------------------------------------------------------

def test_eval_contains_only_domains_unseen_before_the_cutoff():
    records = [_rec("https://old.com/a", observed="2025-01-01"),
               _rec("https://new.com/a", observed="2026-05-01")]
    split = build_temporal_split(records, cutoff="2026-01-01")
    assert split.assignment == {"old.com": TRAIN, "new.com": EVAL}


def test_straddling_domain_stays_in_train_and_its_later_urls_are_dropped():
    """Promoting them to eval would measure 'new page on a known site', which is
    a much easier question than generalization to new sites."""
    records = [_rec("https://old.com/a", observed="2025-01-01"),
               _rec("https://old.com/b", observed="2026-06-01"),
               _rec("https://new.com/a", observed="2026-06-01")]
    split = build_temporal_split(records, cutoff="2026-01-01")
    parts = apply_split(records, split)

    assert split.assignment["old.com"] == TRAIN
    assert [r.url for r in parts[TRAIN]] == ["https://old.com/a"]
    assert [r.url for r in parts[EVAL]] == ["https://new.com/a"]


def test_straddling_domains_are_reported_not_hidden():
    records = [_rec("https://old.com/a", observed="2025-01-01"),
               _rec("https://old.com/b", observed="2026-06-01")]
    split = build_temporal_split(records, cutoff="2026-01-01")
    assert split.dropped_domains == ["old.com"]


def test_no_eval_row_predates_the_cutoff():
    records = [_rec(f"https://d{i}.com/x", observed=f"202{5 + i % 2}-06-01")
               for i in range(20)]
    split = build_temporal_split(records, cutoff="2026-01-01")
    parts = apply_split(records, split)
    assert all(r.observed >= "2026-01-01" for r in parts.get(EVAL, []))
    assert all(r.observed < "2026-01-01" for r in parts.get(TRAIN, []))


def test_cdxj_timestamps_compare_correctly():
    """Common Crawl uses YYYYMMDDhhmmss; lexicographic order is chronological."""
    records = [_rec("https://a.com/x", observed="20240101000000"),
               _rec("https://b.com/x", observed="20260701000000")]
    split = build_temporal_split(records, cutoff="20250101000000")
    assert split.assignment == {"a.com": TRAIN, "b.com": EVAL}


def test_records_without_a_timestamp_are_excluded_not_guessed():
    records = [_rec("https://known.com/x", observed="2025-01-01"),
               _rec("https://undated.com/x", observed="")]
    split = build_temporal_split(records, cutoff="2026-01-01")
    assert "undated.com" not in split.assignment
    assert apply_split(records, split)[TRAIN] == [records[0]]


# --- immutability ----------------------------------------------------------

def test_saved_split_round_trips(tmp_path):
    records = _corpus(50)
    split = build_random_split(records)
    path = tmp_path / "split.json"
    save_split(split, path)
    loaded = load_split(path)
    assert loaded.assignment == split.assignment
    assert loaded.content_hash() == split.content_hash()


def test_tampered_split_is_rejected_on_load(tmp_path):
    """A split that can be silently regenerated is not a split. Every result
    measured against an edited one is incomparable to earlier results."""
    path = tmp_path / "split.json"
    save_split(build_random_split(_corpus(20)), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    domain = next(iter(payload["assignment"]))
    payload["assignment"][domain] = "test" if payload["assignment"][domain] != "test" else TRAIN
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash verification"):
        load_split(path)


@pytest.mark.parametrize("filename", ["split.json", "split.json.gz"])
def test_save_and_load_agree_on_compression(tmp_path, filename):
    """Regression: an edit once left `save_split` writing plain text while
    `load_split` had already switched to gzip, so a `.gz` artifact was not
    actually compressed and would fail to reopen."""
    split = build_random_split(_corpus(40))
    path = tmp_path / filename
    save_split(split, path)
    assert load_split(path).assignment == split.assignment


def test_gzip_artifacts_are_actually_compressed(tmp_path):
    split = build_random_split(_corpus(500))
    plain, packed = tmp_path / "s.json", tmp_path / "s.json.gz"
    save_split(split, plain)
    save_split(split, packed)
    assert packed.read_bytes()[:2] == b"\x1f\x8b"
    assert packed.stat().st_size < plain.stat().st_size


def test_manifest_records_what_is_needed_to_reproduce(tmp_path):
    records = _corpus(30)
    manifest = save_split(build_random_split(records, seed=99), tmp_path / "s.json")
    assert manifest["seed"] == 99
    assert manifest["n_domains"] == 30
    assert set(manifest["domains_per_partition"]) == {TRAIN, "val", "test"}
    assert len(manifest["content_sha256"]) == 64


def test_temporal_manifest_records_the_cutoff_and_straddle_cost(tmp_path):
    records = [_rec("https://old.com/a", observed="2025-01-01"),
               _rec("https://old.com/b", observed="2026-06-01"),
               _rec("https://new.com/a", observed="2026-06-01")]
    manifest = save_split(build_temporal_split(records, cutoff="2026-01-01"),
                          tmp_path / "t.json")
    assert manifest["cutoff"] == "2026-01-01"
    assert manifest["n_straddling_domains_kept_in_train"] == 1


# --- the key-kind regression ------------------------------------------------

def test_host_keyed_split_resolves_hosts_not_domains():
    """A split BUILT on hosts and LOOKED UP by domain misses every host with a
    subdomain, returns None, and callers drop those rows as "not in train/val".

    That happened: a host-keyed phishing split silently lost 78% of its rows,
    and the survivors — the no-subdomain population — looked reassuringly
    balanced, so the corruption presented as an improvement.
    """
    from uie.splits import Record, build_random_split, group_key_host

    urls = [f"https://attacker{i}.vercel.app/login" for i in range(200)]
    records = [Record(url=u, label="phishing", observed="") for u in urls]
    split = build_random_split(records, key=group_key_host, seed=1)

    assert split.key_kind == "host"
    assigned = sum(1 for u in urls if split.partition_of(u) is not None)
    assert assigned == len(urls), f"only {assigned}/{len(urls)} resolved"
    # ...and every host under one platform must NOT collapse to one partition.
    parts = {split.partition_of(u) for u in urls}
    assert len(parts) > 1, parts


def test_domain_keyed_split_is_unchanged(tmp_path):
    from uie.splits import Record, build_random_split
    urls = [f"https://site{i}.com/x" for i in range(100)]
    split = build_random_split(
        [Record(url=u, label="benign", observed="") for u in urls], seed=1)
    assert split.key_kind == "domain"
    assert all(split.partition_of(u) is not None for u in urls)


def test_key_kind_survives_a_round_trip(tmp_path):
    """The key kind must travel WITH the artifact — a split whose lookup rule
    is inferred by the loader can disagree with how it was built."""
    from uie.splits import (Record, build_random_split, group_key_host,
                            load_split, save_split)
    urls = [f"https://a{i}.pages.dev/p" for i in range(120)]
    split = build_random_split(
        [Record(url=u, label="phishing", observed="") for u in urls],
        key=group_key_host, seed=2)
    path = tmp_path / "s.json.gz"
    save_split(split, path)
    loaded = load_split(path)
    assert loaded.key_kind == "host"
    assert all(loaded.partition_of(u) is not None for u in urls)
