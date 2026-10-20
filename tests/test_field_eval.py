"""
Tests for field-scoped discrimination against measured floors.

The property that matters is that the FLOOR is measured, not assumed. A bar set
at an absolute AUC can sit below the floor of its own metric and be unreachable
by any generator — which is exactly what happened to the pre-registered 0.65,
against a real-vs-real floor measured at 0.741.
"""
from __future__ import annotations

import random

import pytest

from uie.evaluation import (FIELD_POPULATION, FIELD_POPULATION_BY_FRAME,
                            FIELDS, GENERATOR_FRAME, discriminate_with_floor,
                            field_report)


def _real(n, seed=0):
    """Realistic-ish URLs: real morphemes, varied suffixes and depths."""
    rng = random.Random(seed)
    head = ["news", "shop", "cloud", "data", "health", "tech", "green", "city",
            "media", "learn", "book", "food", "auto", "home", "travel"]
    tail = ["daily", "hub", "works", "group", "center", "labs", "store",
            "world", "point", "zone", "base", "line"]
    sufs = ["com", "org", "net", "de", "co.uk", "nl", "fr", "it"]
    out = []
    for _ in range(n):
        lab = rng.choice(head) + rng.choice(tail)
        depth = rng.choice([0, 1, 1, 2, 2, 3])
        path = "/".join(rng.choice(["about", "products", "2024", "news",
                                    "blog", "contact"]) for _ in range(depth))
        out.append(f"https://{lab}.{rng.choice(sufs)}/{path}")
    return out


def test_two_real_samples_score_near_the_floor():
    """The whole design rests on this: the same population scored against
    itself gives excess ~0, whatever its absolute AUC happens to be."""
    pool = _real(9000, seed=1)
    other = _real(3000, seed=2)          # same generator, disjoint draw
    r = discriminate_with_floor(pool, other, "LBL", n=800, seed=3,
                                dedup_by_host=False)
    assert abs(r["excess"]) < 0.08, r


def test_a_genuinely_different_distribution_shows_positive_excess():
    """Labels built from nonsense syllables must be caught."""
    rng = random.Random(4)
    syl = ["ka", "zu", "mi", "ro", "ta", "nu", "vi", "se"]
    junk = [f"https://{''.join(rng.choice(syl) for _ in range(4))}.com/x"
            for _ in range(2000)]
    r = discriminate_with_floor(_real(9000, seed=5), junk, "LBL",
                                n=800, seed=6, dedup_by_host=False)
    assert r["excess"] > 0.2, r


def test_floor_and_auc_use_disjoint_real_samples():
    """If the floor and the test shared real rows, the floor would be measured
    on data the comparison also uses and would drift toward it."""
    pool = _real(6000, seed=7)
    r = discriminate_with_floor(pool, _real(2000, seed=8), "LBL",
                                n=1000, seed=9, dedup_by_host=False)
    assert r["n"] == 1000
    assert not r.get("insufficient")


def test_insufficient_data_is_reported_not_faked():
    """Three disjoint real samples are needed. Too small a pool must say so
    rather than silently reuse rows."""
    r = discriminate_with_floor(_real(100), _real(100), "LBL", n=200,
                                dedup_by_host=False)
    assert r["insufficient"] is True


def test_every_field_is_scored_in_the_frame_the_generator_samples():
    """REVERSES the earlier routing of PTH/QRY to the structure corpus.

    The two corpora are not two halves of one distribution — they are two
    SAMPLING FRAMES: `name` takes one URL per domain over every domain, and
    `structure` takes up to 40 URLs per multi-page site. The generator emits ONE
    URL PER INVENTED HOST and never revisits a host, so it samples the domain
    frame and cannot produce page-frame output at all.

    Measured on E4 (benign PTH): against `structure`, AUC 0.6440 with 0.6013 of
    it from LENGTH ALONE (real 38.8 chars vs generated 28.5); against `name`,
    0.6290 with length at 0.5151 — chance. The model's conditional path length
    (27.9) matches the name frame (28.3). The length signal was the frame, not
    the model.

    This is `_one_per_host`'s argument applied to the choice of population
    instead of to repetition within it, so it is consistency rather than a
    relaxation — and the `FULL` pool is 93.9% name-frame hosts after dedup
    anyway, so both metrics were always asking for domain-frame output.
    """
    assert GENERATOR_FRAME == "domain"
    assert FIELD_POPULATION["LBL"] == "name"
    assert FIELD_POPULATION["SUF"] == "name"
    assert FIELD_POPULATION["PTH"] == "name"
    assert FIELD_POPULATION["QRY"] == "name"
    assert FIELD_POPULATION["FULL"] == "mixed"


def test_the_page_frame_is_still_defined_for_conditioned_generation():
    """Kept, not deleted. "Can the model produce multi-page-site paths" stays a
    live question; it just cannot be asked of an UNCONDITIONED generator. It
    becomes a real target once `<F:page>` conditioning exists, and until then
    run_eval reports it as a diagnostic that never gates."""
    page = FIELD_POPULATION_BY_FRAME["page"]
    assert page["PTH"] == "structure"
    assert page["QRY"] == "structure"
    assert page["LBL"] == "name"        # name fields are frame-independent


def test_an_unknown_frame_is_rejected():
    """Silently falling back to a default frame would reintroduce exactly the
    error this change fixes, without any signal that it had happened."""
    with pytest.raises(KeyError, match="frame"):
        field_report({"name": _real(100)}, _real(50), frame="nonsense")


def test_field_report_labels_each_row_with_the_population_it_used():
    pops = {"name": _real(6000, seed=10), "structure": _real(6000, seed=11),
            "mixed": _real(6000, seed=12)}
    out = field_report(pops, _real(1500, seed=13), n=200, seed=14)
    for name, r in out.items():
        assert r["population"] == FIELD_POPULATION[name]
        assert r["frame"] == GENERATOR_FRAME

    page = field_report(pops, _real(1500, seed=13), n=200, seed=14, frame="page")
    assert page["PTH"]["population"] == "structure"
    assert page["PTH"]["frame"] == "page"


def test_field_extractors_are_faithful():
    u = "https://blog.shopwave.io/posts/2024/review?page=2"
    assert FIELDS["LBL"](u) == "shopwave"
    assert FIELDS["SUF"](u) == "io"
    assert FIELDS["SUB"](u) == "blog"
    assert FIELDS["PTH"](u) == "posts/2024/review"
    assert FIELDS["QRY"](u) == "page=2"
    assert FIELDS["FULL"](u) == "blog.shopwave.io/posts/2024/review?page=2"


def test_empty_field_values_are_dropped_not_scored():
    """A URL with no query contributes nothing to the query metric. Scoring
    empty strings would compare 'has a query' rather than query CONTENT, which
    the gate and the marker already cover."""
    with_q = [f"https://a{i}.com/x?k={i}" for i in range(3000)]
    without = [f"https://b{i}.com/x" for i in range(1500)]
    r = discriminate_with_floor(with_q, without, "QRY", n=500, seed=15,
                                dedup_by_host=False)
    assert r["insufficient"] is True        # fake side has no queries at all


def test_real_pools_are_deduplicated_to_one_url_per_host():
    """The generator produces ONE url per invented host, so a real comparator
    holding six pages of the same site measures repetition, not realism.

    Measured on the phishing structure corpus: train and val separate at 0.7647
    with repetition present and 0.4798 without, against a control of 0.4990 —
    the entire signal was kits producing near-identical paths across their own
    pages, which no generated set can imitate because it never revisits a host.
    """
    # 40 pages each on 30 hosts, against one page per host.
    repeated = [f"https://kit{h}.example.com/step{i}/verify"
                for h in range(30) for i in range(40)]
    distinct = [f"https://kit{h}.example.com/step0/verify" for h in range(30)]

    with_dupes = discriminate_with_floor(repeated, distinct, "PTH", n=10,
                                         seed=1, dedup_by_host=False)
    deduped = discriminate_with_floor(repeated, distinct, "PTH", n=10, seed=1)

    # Deduplication collapses 1,200 rows to the 30 distinct hosts...
    assert deduped["field_coverage_real"] < 0.05
    assert with_dupes["field_coverage_real"] == 1.0
    # ...and most of the apparent tell goes with it, because it was repetition.
    assert deduped["excess"] < with_dupes["excess"] / 2, (deduped, with_dupes)


def test_each_class_is_scored_against_its_own_held_out_pool():
    """run_eval must build one pool set PER CLASS.

    Scoring phishing generations against benign held-out URLs measures the gap
    between the two classes — which is real, large, and not the model's doing.
    Here two real populations that differ only by class score far apart on the
    label; a per-class comparison of the same phishing pool against itself sits
    at the floor. Reusing the benign pool would report the first number and read
    as a catastrophic generator.
    """
    benign = _real(9000, seed=20)
    rng = random.Random(21)
    syl = ["secure", "login", "verify", "account", "update"]
    phish_pool = [f"https://{rng.choice(syl)}-{rng.randrange(10**6)}.tk/x"
                  for _ in range(9000)]
    phish_gen = [f"https://{rng.choice(syl)}-{rng.randrange(10**6)}.tk/x"
                 for _ in range(2000)]

    wrong = discriminate_with_floor(benign, phish_gen, "LBL", n=800, seed=22,
                                    dedup_by_host=False)
    right = discriminate_with_floor(phish_pool, phish_gen, "LBL", n=800,
                                    seed=22, dedup_by_host=False)

    assert wrong["excess"] > 0.3, wrong        # cross-class: looks catastrophic
    assert abs(right["excess"]) < 0.08, right  # same class: at the floor


def test_run_eval_refuses_a_class_with_no_pool_of_its_own(tmp_path):
    """Rather than silently falling back to another class's held-out set."""
    import gzip
    import json

    import torch

    from uie.corpus import run_eval
    from uie.model import ModelConfig, URLTransformer
    from uie.tokenizer import URLTokenizer

    for stem in ("name_val", "structure_val"):        # benign only, no phishing
        with gzip.open(tmp_path / f"{stem}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for u in _real(50):
                fh.write(json.dumps({"url": u}) + "\n")

    tok = URLTokenizer(include_scheme=False, include_language=False)
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_len=32, d_model=16,
                      n_layers=1, n_heads=2, d_ff=32)
    ckpt = tmp_path / "m.pt"
    torch.save({"model": URLTransformer(cfg).state_dict(),
                "model_config": cfg.__dict__ if not hasattr(cfg, "_asdict")
                else cfg._asdict(),
                "tokenizer": tok.state_dict(), "train_config": {}}, ckpt)

    with pytest.raises(FileNotFoundError, match="phishing"):
        run_eval.main(ckpt, tmp_path, tmp_path, n_main=10)


def test_a_small_pool_shrinks_n_instead_of_blanking_the_field():
    """The phishing structure pool holds 941 hosts against a requested 1200.

    Refusing outright would blank the phishing path and query fields, which are
    the ones tracking the shallowness artifact. Shrinking to what the pool
    supports is a weaker measurement, not a wrong one — and it must say so.
    """
    small = _real(1500, seed=30)          # ~500 usable per side, not 3x1200
    r = discriminate_with_floor(small, _real(1200, seed=31), "LBL", n=1200,
                                seed=32, dedup_by_host=False)
    assert not r.get("insufficient"), r
    assert r["reduced_n"] is True
    assert r["n"] == 500
    assert abs(r["excess"]) < 0.08, r     # still lands at the floor


def test_shrinking_stops_at_a_size_that_can_still_measure():
    """Below MIN_N the standard error approaches the bar itself, so the result
    is withheld rather than reported weakly."""
    from uie.evaluation import MIN_N

    tiny = _real(3 * (MIN_N - 40), seed=33)
    r = discriminate_with_floor(tiny, _real(400, seed=34), "LBL", n=1200,
                                seed=35, dedup_by_host=False)
    assert r["insufficient"] is True
    assert r["requested_n"] == 1200


def test_a_pool_that_already_fits_is_not_subject_to_the_minimum():
    """MIN_N guards SHRINKING. A caller that deliberately asks for a small n on
    a pool that supports it gets exactly what it asked for."""
    r = discriminate_with_floor(_real(90, seed=36), _real(30, seed=37), "LBL",
                                n=30, seed=38, dedup_by_host=False)
    assert not r.get("insufficient"), r
    assert r["n"] == 30 and r["reduced_n"] is False
