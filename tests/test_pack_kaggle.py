"""
Tests for E2.1's per-name capping.

The cap is the whole E2.1 intervention, so what is tested here is the two
properties the diagnosis depends on: that every name survives (coverage), and
that the survivors are an UNBIASED sample of that name's URLs (no crawl-order
or path-prefix bias). A cap that quietly kept the first K would reintroduce the
skew it was built to remove.
"""
from __future__ import annotations

import collections
import random

from uie.corpus.pack_kaggle import _Capped, _host_key, _label_key


def _rows(spec):
    """spec: {key: n} -> rows whose url encodes key and index."""
    return [{"url": f"https://{k}.com/page{i}", "label": "benign", "lang": "und"}
            for k, n in spec.items() for i in range(n)]


def test_cap_is_never_exceeded_and_every_key_survives():
    capped = _Capped(4, _label_key, random.Random(0))
    for row in _rows({"alpha": 100, "beta": 3, "gamma": 1}):
        capped.offer(row)

    kept = collections.Counter(_label_key(r["url"]) for r in capped.rows())
    assert kept["alpha"] == 4          # capped
    assert kept["beta"] == 3           # under the cap, untouched
    assert kept["gamma"] == 1
    assert set(kept) == {"alpha", "beta", "gamma"}


def test_capping_flattens_a_head_heavy_distribution():
    """The measured problem: 1.8% of labels held 38.4% of the label gradient."""
    spec = {"head": 200_000}
    spec.update({f"tail{i}": 1 for i in range(1_000)})
    capped = _Capped(32, _label_key, random.Random(1))
    for row in _rows(spec):
        capped.offer(row)

    kept = collections.Counter(_label_key(r["url"]) for r in capped.rows())
    total = sum(kept.values())
    assert kept["head"] == 32
    assert kept["head"] / total < 0.05          # was 200000/201000 = 99.5%
    assert len(kept) == 1_001                   # nothing dropped entirely


def test_survivors_are_not_the_first_k_seen():
    """Shards are ordered by crawl and by SURT, so keeping the first K would
    bias every label toward one crawl and one path prefix. Reservoir sampling
    must draw from the whole stream."""
    capped = _Capped(8, _label_key, random.Random(7))
    for row in _rows({"alpha": 500}):
        capped.offer(row)

    indices = sorted(int(r["url"].rsplit("page", 1)[1]) for r in capped.rows())
    assert len(indices) == 8
    assert max(indices) > 100, indices     # reached deep into the stream


def test_reservoir_sampling_is_uniform_over_the_stream():
    """Every row must be equally likely to survive. Run the cap repeatedly and
    check the survival distribution is flat rather than front- or back-loaded."""
    n, cap, trials = 50, 5, 4_000
    hits = collections.Counter()
    for trial in range(trials):
        capped = _Capped(cap, _label_key, random.Random(trial))
        for row in _rows({"alpha": n}):
            capped.offer(row)
        for row in capped.rows():
            hits[int(row["url"].rsplit("page", 1)[1])] += 1

    expected = trials * cap / n
    assert len(hits) == n
    # +/-25% of the expected rate; tight enough to catch "first K" or "last K",
    # loose enough not to be flaky.
    assert all(0.75 * expected < c < 1.25 * expected for c in hits.values()), \
        (min(hits.values()), max(hits.values()), expected)


def test_phishing_caps_on_host_so_platform_subdomains_survive():
    """`vercel` carries 26,773 phishing URLs across 17,494 DISTINCT subdomain
    prefixes: the attacker's invented name is the subdomain, not the label.
    Capping phishing by label would delete exactly that diversity."""
    rows = [{"url": f"https://attacker{i}.vercel.app/login", "label": "phishing",
             "lang": "und"} for i in range(500)]

    by_host = _Capped(4, _host_key, random.Random(0))
    by_label = _Capped(4, _label_key, random.Random(0))
    for row in rows:
        by_host.offer(row)
        by_label.offer(row)

    assert by_host.stats()["kept"] == 500      # every invented name survives
    assert by_label.stats()["kept"] == 4       # all but four names destroyed


def test_unparseable_rows_are_kept_rather_than_pooled():
    """A row with no registrable label must not fall into one giant bucket
    keyed on the empty string, where the cap would discard almost all of them."""
    capped = _Capped(2, _label_key, random.Random(0))
    for i in range(20):
        capped.offer({"url": f"http://192.168.0.{i}/x", "label": "benign",
                      "lang": "und"})
    assert capped.stats()["kept"] == 20


def test_cap_of_none_keeps_everything():
    """E2.2 reverts the phishing cap to None. That must mean 'deliberately
    uncapped' and keep every row — not a very large number that happens never
    to bind, which would read the same in the manifest but differently in an
    experiment log."""
    capped = _Capped(None, _host_key, random.Random(0))
    rows = [{"url": f"https://a{i%3}.vercel.app/p{i}", "label": "phishing",
             "lang": "und"} for i in range(300)]
    for row in rows:
        capped.offer(row)
    assert capped.stats() == {"keys": 3, "kept": 300, "seen": 300}


def test_stats_report_what_was_seen_and_what_was_kept():
    capped = _Capped(2, _label_key, random.Random(0))
    for row in _rows({"a": 10, "b": 1}):
        capped.offer(row)
    assert capped.stats() == {"keys": 2, "kept": 3, "seen": 11}
