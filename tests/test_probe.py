"""The representation probe's machinery (research/probe/PROBE_PLAN.md), on a tiny model."""
import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_typicality import URLS, _scorer

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "research" / "probe" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load("run_probe")
build = _load("build_probe_sets")


def test_pooled_states_are_the_real_forward_pass_and_skip_prefix_and_padding():
    s = _scorer()
    s.model.eval()
    states, lp = run.pooled_states(s.model, s, URLS, batch=3)
    assert states.shape == (s.model.cfg.n_layers + 1, len(URLS), s.model.cfg.d_model)
    np.testing.assert_allclose(lp, s.log_prob(URLS, "benign"), atol=1e-5)
    # one URL alone (no padding) gives the same pooled state as inside a padded batch
    alone, _ = run.pooled_states(s.model, s, URLS[1:2], batch=1)
    np.testing.assert_allclose(alone[:, 0], states[:, 1], atol=1e-5)
    # by hand: the last block's output, averaged over positions PREFIX..n-1
    ids = torch.tensor([s.encode(URLS[1], "benign").ids])
    captured = []
    h = s.model.blocks[-1].register_forward_hook(lambda m, i, o: captured.append(o))
    with torch.no_grad():
        s.model(ids)
    h.remove()
    np.testing.assert_allclose(states[-1, 1], captured[0][0, run.PREFIX:].mean(0).numpy(), atol=1e-5)


def test_surface_text_is_what_uie_reads():
    assert run.surface_text("HTTPS://Shop.Example.COM/Path?Q=1") == "shop.example.com/Path?Q=1"
    assert run.surface_text("example.org/x") == "example.org/x"


def test_inner_folds_deal_each_language_round_robin():
    langs = ["a", "b", "a", "a", "b", "a", "a", "a"]
    assert run.inner_folds(langs).tolist() == [0, 0, 1, 2, 1, 3, 4, 0]


def test_tld_kind():
    assert build.tld_kind("shop.example.de") == "country"
    assert build.tld_kind("example.co.uk") == "country"
    assert build.tld_kind("example.xn--p1ai") == "country"
    assert build.tld_kind("example.com") == "generic" and build.tld_kind("example.online") == "generic"


def test_parts_are_stratified_twenty_percent_then_five_folds():
    rows = [{"lang": l} for l in ["eng"] * 500 + ["jpn"] * 500]
    build.assign_parts(rows, random.Random(1))
    for lang in ("eng", "jpn"):
        parts = [r["part"] for r in rows if r["lang"] == lang]
        assert parts.count("select") == 100
        assert all(parts.count(f"fold{k}") == 80 for k in range(5))


def test_bootstrap_paired_difference_is_zero_for_identical_readers():
    c = np.array([1.0, 0, 1, 1, 0, 1, 0, 1])
    r = run.bootstrap({"uie": c, "untrained": c, "surface": 1 - c}, n=200)
    assert r["uie_minus_untrained"]["diff"] == 0 and r["uie_minus_untrained"]["ci"] == [0.0, 0.0]
    assert r["uie"]["ci"][0] <= r["uie"]["acc"] <= r["uie"]["ci"][1]
