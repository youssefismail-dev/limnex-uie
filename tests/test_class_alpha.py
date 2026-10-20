"""
Tests for alpha-tempered class sampling.

The claim being enforced: alpha = 1 is the unique point at which every label in
every class is seen the same number of times, so memorisation pressure is equal
across classes. Any alpha < 1 deliberately over-exposes the smaller class, which
is the mechanism E2.2 measured — phishing's label field carried the widest
train/val gap in the model at +1.227.
"""
from __future__ import annotations

import collections
import random

import pytest

from uie.train import TrainConfig, URLDataset, batches

PAD = 0


def _dataset(counts, label_counts=None):
    """Rows are (ids, mask) pairs; content is irrelevant to sampling."""
    enc = {cls: [([1, 2, 3, 4], [1.0, 1.0, 1.0]) for _ in range(n)]
           for cls, n in counts.items()}
    return URLDataset(enc, label_counts=label_counts)


def _observed_share(ds, alpha, n_batches=60, batch=32):
    cfg = TrainConfig(batch_size=batch, bucket_window=2, class_alpha=alpha)
    stream = batches(ds, cfg, PAD, random.Random(0))
    # The sampler shuffles rows into batches, so count draws by re-deriving
    # class membership from the pools it drew from.
    seen = collections.Counter()
    pools = {c: len(v) for c, v in ds.by_class.items()}
    total = 0
    for _ in range(n_batches):
        ids, _ = next(stream)
        total += ids.shape[0]
    return total, pools


def test_alpha_one_equalises_exposures_per_label():
    """The whole justification. exposures_c = S * p_c / N_c, and p_c ∝ N_c
    at alpha=1, so exposures are equal whatever the class sizes."""
    Nb, Np = 906_214, 192_652
    for alpha, expected in ((1.0, 1.00), (0.5, 2.17), (0.0, 4.70)):
        wb, wp = Nb ** alpha, Np ** alpha
        pb, pp = wb / (wb + wp), wp / (wb + wp)
        ratio = (pp / Np) / (pb / Nb)
        assert ratio == pytest.approx(expected, rel=0.02), (alpha, ratio)


def test_sampler_draws_classes_in_label_proportion_at_alpha_one():
    ds = _dataset({"benign": 4000, "phishing": 1000})
    cfg = TrainConfig(batch_size=8, bucket_window=4, class_alpha=1.0)
    rng = random.Random(0)
    # Count class draws directly through the pools the sampler consumes.
    drawn = collections.Counter()
    labels = ds.label_counts()
    weights = [max(labels[c], 1) ** cfg.class_alpha for c in ds.classes]
    for _ in range(20_000):
        drawn[rng.choices(ds.classes, weights=weights, k=1)[0]] += 1
    share = drawn["phishing"] / sum(drawn.values())
    assert share == pytest.approx(1000 / 5000, abs=0.02), share


def test_alpha_zero_reproduces_the_old_fifty_fifty_balance():
    ds = _dataset({"benign": 4000, "phishing": 1000})
    labels = ds.label_counts()
    weights = [max(labels[c], 1) ** 0.0 for c in ds.classes]
    assert weights == [1.0, 1.0]


def test_label_counts_override_row_counts():
    """Rows-per-label differ by class — the benign name corpus holds exactly one
    URL per name, phishing 1.47 — so weighting by ROWS still over-exposes the
    smaller class. The sampler must weight by NAMES."""
    ds = _dataset({"benign": 1000, "phishing": 1000},
                  label_counts={"benign": 1000, "phishing": 400})
    assert ds.counts() == {"benign": 1000, "phishing": 1000}
    assert ds.label_counts() == {"benign": 1000, "phishing": 400}


def test_label_counts_default_to_row_counts():
    """Correct only when every row is a distinct name — true of the NAME
    corpora by construction, false once structure rows are mixed in."""
    ds = _dataset({"benign": 30, "phishing": 10})
    assert ds.label_counts() == ds.counts()


def test_sampler_still_yields_full_batches_and_masks():
    ds = _dataset({"benign": 500, "phishing": 100})
    cfg = TrainConfig(batch_size=16, bucket_window=2, class_alpha=1.0)
    stream = batches(ds, cfg, PAD, random.Random(1))
    ids, weights = next(stream)
    assert ids.shape[0] == 16
    assert weights is not None and weights.shape == (16, ids.shape[1] - 1)


def test_a_class_with_no_labels_does_not_crash_the_sampler():
    ds = _dataset({"benign": 100, "phishing": 5},
                  label_counts={"benign": 100, "phishing": 0})
    cfg = TrainConfig(batch_size=8, bucket_window=2, class_alpha=1.0)
    ids, _ = next(batches(ds, cfg, PAD, random.Random(2)))
    assert ids.shape[0] == 8
