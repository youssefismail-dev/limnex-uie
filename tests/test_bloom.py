"""
Tests for the novelty Bloom filter.

The property that matters is ONE-DIRECTIONAL error: everything added must be
reported present. If that ever fails, the novelty metric silently stops
catching memorization, which is the failure the metric exists to catch.
"""
from __future__ import annotations

import pytest

from uie.bloom import BloomFilter


def test_every_added_item_is_present():
    """No false negatives, ever. This is the guarantee the metric rests on."""
    bloom = BloomFilter(capacity=5_000, error_rate=0.01)
    items = [f"example{i}.com/path/{i}" for i in range(5_000)]
    bloom.update(items)
    assert all(item in bloom for item in items)


def test_false_positive_rate_is_near_the_design_rate():
    bloom = BloomFilter(capacity=20_000, error_rate=0.01)
    bloom.update(f"real-{i}.com" for i in range(20_000))
    absent = [f"never-seen-{i}.org" for i in range(20_000)]
    rate = sum(item in bloom for item in absent) / len(absent)
    # Generous bound: the point is the order of magnitude, not the constant.
    assert rate < 0.03, rate


def test_expected_error_rate_tracks_load():
    """A filter loaded past capacity still works but reports a worse rate, so
    the manifest cannot claim an accuracy the filter does not have."""
    bloom = BloomFilter(capacity=1_000, error_rate=0.001)
    bloom.update(f"a{i}" for i in range(1_000))
    at_capacity = bloom.expected_error_rate
    bloom.update(f"b{i}" for i in range(9_000))
    assert bloom.expected_error_rate > at_capacity


def test_round_trip_through_disk_preserves_membership(tmp_path):
    """A filter reloaded with different geometry would answer every query
    wrongly, so the stored m and k are authoritative rather than re-derived."""
    bloom = BloomFilter(capacity=2_000, error_rate=0.01)
    items = [f"site{i}.net/{i}" for i in range(2_000)]
    bloom.update(items)
    path = tmp_path / "known.bloom.gz"
    bloom.save(path)

    loaded = BloomFilter.load(path)
    assert loaded.n_bits == bloom.n_bits
    assert loaded.n_hashes == bloom.n_hashes
    assert loaded.n_added == bloom.n_added
    assert all(item in loaded for item in items)


def test_load_accepts_an_already_decompressed_file(tmp_path):
    """Kaggle DECOMPRESSES gzip members while processing an uploaded dataset,
    so a bundle shipped as `known_urls.bloom.gz` arrives as `known_urls.bloom`
    holding raw bytes. Loading must key on the header, not the extension."""
    import gzip

    bloom = BloomFilter(capacity=500, error_rate=0.01)
    items = [f"host{i}.org/p" for i in range(500)]
    bloom.update(items)
    gz_path = tmp_path / "known.bloom.gz"
    bloom.save(gz_path)

    raw_path = tmp_path / "known.bloom"
    raw_path.write_bytes(gzip.decompress(gz_path.read_bytes()))

    loaded = BloomFilter.load(raw_path)
    assert loaded.n_added == bloom.n_added
    assert all(item in loaded for item in items)


def test_load_rejects_a_foreign_file(tmp_path):
    import gzip
    path = tmp_path / "bogus.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(b"not a bloom filter at all")
    with pytest.raises(ValueError):
        BloomFilter.load(path)


def test_non_ascii_urls_survive():
    """Real corpora carry raw Unicode hosts; encoding must not raise."""
    bloom = BloomFilter(capacity=100, error_rate=0.01)
    bloom.add("münchen.de/straße")
    assert "münchen.de/straße" in bloom


@pytest.mark.parametrize("capacity,error", [(0, 0.01), (10, 0.0), (10, 1.0)])
def test_invalid_parameters_raise(capacity, error):
    with pytest.raises(ValueError):
        BloomFilter(capacity=capacity, error_rate=error)
