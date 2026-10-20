"""PC1's instrument (research/native/PREREG_PC1.md; pc1_measure.py, pc1_read.py).

Pins V4 (scored tokens = len - 3 = the field mask's sum) and V6 (the
frame-generalised plan equals `plan_by_suffix` in the domain frame); that a row's
suffix and plan labels are read exactly as training encoded them; that the batch
builder draws, sorts and chunks exactly as `uie.train.batches`; that the gradient
norm over micro-batches equals the full batch's; that the clip bins and weights
are as registered; and that the gap and query splits are exact identities.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from test_e7_fields import FIT, tiny_model
from uie import train as T
from uie.evaluation import FIELDS
from uie.factorization import parse
from uie.model import ModelConfig, URLTransformer
from uie.tokenizer import URLTokenizer
from uie.train import TrainConfig, URLDataset, encode_corpus, field_mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "native"))
import pc1_measure as M  # noqa: E402
import pc1_read as R  # noqa: E402
from e8_plan_calibration import plan_by_suffix  # noqa: E402


def plan_tok():
    return URLTokenizer(include_frame=True, include_fields=True, presence_plan=True).fit(FIT)


URLS = [u for u in FIT if u]


def test_a_row_is_read_as_training_encoded_it_and_scored_tokens_are_len_minus_3():
    tok = plan_tok()
    for u in URLS:
        for corpus in ("name", "structure"):
            ids = tok.encode(u, "phishing", frame=T.CORPUS_FRAME[corpus])
            suffix, path, query, m, usr = M.row_record(tok, ids)
            assert suffix == parse(u).public_suffix
            assert path == int(bool(FIELDS["PTH"](u))) and query == int(bool(FIELDS["QRY"](u)))
            assert m == len(ids) - 3 == sum(field_mask(ids, tok, corpus))          # V4
            if not usr:
                k = ids.index(tok.marker_id("SUF")) + 1 + len(suffix)
                assert ids[:k] == M.plan_prefix(tok, "phishing", T.CORPUS_FRAME[corpus], suffix)


def test_the_frame_generalised_plan_equals_plan_by_suffix_in_the_domain_frame():   # V6
    tok = plan_tok()
    model = tiny_model(tok)
    sufs = sorted({parse(u).public_suffix for u in URLS})
    want = plan_by_suffix(model, tok, "benign", sufs)
    got = M.plan_by_frame(model, tok, "benign", sufs, "domain")
    for s in sufs:
        for k in ("p", "q", "legal_mass_P", "q_given_p0", "q_given_p1"):
            assert got[s][k] == want[s][k]
    page = M.plan_by_frame(model, tok, "benign", sufs, "page")
    assert any(page[s]["p"] != got[s]["p"] for s in sufs)                  # the frame is read


def test_the_k10a_probe_misreads_se_and_the_training_prefix_does_not():
    """PREREG_PC1 amendment 1: `x.se` is itself a public suffix, so the probe
    `x.se/` encodes the suffix `x.se`; a real `.se` row carries `se`."""
    tok = URLTokenizer(include_frame=True, include_fields=True,
                       presence_plan=True).fit(URLS + ["http://example.se/"])
    assert parse("x.se/").public_suffix == "x.se"
    ids = tok.encode("http://example.se/", "benign", frame="domain")
    assert M.row_record(tok, ids)[0] == "se"
    k = ids.index(tok.marker_id("SUF")) + 1 + len("se")
    assert ids[:k] == M.plan_prefix(tok, "benign", "domain", "se")


def test_the_batch_builder_draws_sorts_and_chunks_exactly_as_training(monkeypatch):
    tok = plan_tok()
    labels = ["benign"] * len(URLS) + ["phishing"] * len(URLS)
    enc, _ = encode_corpus(URLS * 2, labels, tok, 256, corpora=["name"] * len(labels))
    ds = URLDataset(enc, label_counts={"benign": 30, "phishing": 7})
    cfg = TrainConfig(batch_size=4, bucket_window=3)
    ident = {id(r): (cls, i) for cls, rows in ds.by_class.items() for i, r in enumerate(rows)}
    monkeypatch.setattr(T, "_pad", lambda chunk, pad_id: chunk)
    stream = T.batches(ds, cfg, tok.pad_id, random.Random(11))
    want = [[tuple(ident[id(r)] for r in next(stream)) for _ in range(3)] for _ in range(4)]
    flat = {cls: list(range(len(rows))) for cls, rows in ds.by_class.items()}
    offset = {"benign": 0, "phishing": len(ds.by_class["benign"])}
    lengths = [len(r[0]) for cls in ("benign", "phishing") for r in ds.by_class[cls]]
    pools = {cls: [offset[cls] + i for i in ix] for cls, ix in flat.items()}
    back = {offset[c] + i: (c, i) for c, ix in flat.items() for i in ix}
    mine = M.window_batches(pools, lengths, {"benign": 30, "phishing": 7}, 4, 3, random.Random(11))
    for w in range(4):
        got = [tuple(back[i] for i in chunk) for chunk in next(mine)]
        assert sorted(got) == sorted(want[w])                 # the same batches, window by window


def test_the_micro_batched_norm_is_the_full_batch_norm():
    tok = plan_tok()
    torch.manual_seed(0)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, max_len=64, d_model=32,
                                       n_layers=1, n_heads=2, d_ff=64, dropout=0.0))
    rows = []
    for u in (URLS * 6)[:70]:
        ids = tok.encode(u, "benign", frame="domain")
        rows.append((ids, field_mask(ids, tok, "name")))
    norm, loss, mass = M.grad_norm(model, rows, tok.pad_id, seed=1)
    ids, w = T._pad(rows, tok.pad_id)
    model.zero_grad(set_to_none=True)
    full = model.loss(ids, tok.pad_id, w)
    full.backward()
    want = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in model.parameters())))
    assert norm == pytest.approx(want, rel=1e-5) and loss == pytest.approx(float(full), rel=1e-5)
    assert mass == sum(len(r[0]) - 3 for r in rows)


def test_clip_bins_are_four_equal_short_bins_then_the_long_bands():
    batches = [{"scored_per_row": 10 + i, "norm": 2.0 if i % 2 else 0.5,
                "omega": 0.5 if i % 2 else 1.0} for i in range(16)]           # 10..25: short
    batches += [{"scored_per_row": 45, "norm": 0.4, "omega": 1.0},
                {"scored_per_row": 90, "norm": 0.3, "omega": 1.0},
                {"scored_per_row": 200, "norm": 1.25, "omega": 0.8},
                {"scored_per_row": 12, "norm": float("nan"), "omega": None}]
    out = R.clip_bins(batches)
    bins = out["bins"]
    assert [b["batches"] for b in bins] == [4, 4, 4, 4, 1, 1, 1] and out["nonfinite"] == 1
    assert [b["lo"] for b in bins[:4]] == [0.0, 14, 18, 22] and bins[3]["hi"] == 30
    assert all(b["omega"] == 0.75 for b in bins[:4]) and bins[6]["omega"] == 0.8
    w = R.omega_of(np.array([5.0, 14.0, 29.0, 30.0, 150.0]), bins)
    assert list(w) == [0.75, 0.75, 0.75, 1.0, 0.8]


def _rows(spec):
    """rows.npz-like columns from (cls, frame, suf, path, query, m) tuples, all fitting."""
    cols = {k: np.array([r[i] for r in spec]) for i, k in
            enumerate(("cls", "frame", "suf", "path", "query", "m"))}
    cols["fits"] = np.ones(len(spec), dtype=np.int8)
    cols["usr"] = np.zeros(len(spec), dtype=np.int8)
    return cols


def test_the_gap_and_query_splits_are_exact():
    names = ["com", "com.br", "de"]
    rng = random.Random(3)
    spec = [(0, 0, rng.randrange(3), rng.random() < 0.7, rng.random() < 0.1, rng.randrange(10, 80))
            for _ in range(400)]
    rows = _rows(spec)
    omega = R.omega_of(rows["m"].astype(float), [{"lo": 0, "hi": 30, "omega": 0.8},
                                                  {"lo": 30, "hi": 10 ** 9, "omega": 1.0}])
    cells = R.Cells(R.aggregate(rows, 3, omega), names, 0, 0, 3)
    plan = {s: {"p": rng.random(), "q_given_p0": rng.random() / 5, "q_given_p1": rng.random() / 5}
            for s in names}
    for x in plan.values():
        x["q"] = (1 - x["p"]) * x["q_given_p0"] + x["p"] * x["q_given_p1"]
    t = R.gap_terms("benign", plan, cells, {"com": 5, "com.br": 2, "de": 3})
    assert t["DQ"] == pytest.approx(t["M"] + t["K0"] + t["K1"], abs=1e-12)
    assert t["Delta"] == pytest.approx(t["C"] + t["E"], abs=1e-12)
    assert sum(b["weight"] for b in t["bands"].values()) == pytest.approx(1.0)


def test_continuation_counts_longer_suffixes_that_start_with_the_suffix():
    names = ["com", "com.br", "community", "de"]
    spec = ([(0, 0, 0, 1, 0, 20)] * 6 + [(0, 0, 1, 1, 0, 20)] * 3 + [(0, 0, 2, 0, 0, 20)]
            + [(0, 0, 3, 0, 0, 20)] * 5)
    rows = _rows(spec)
    cells = R.Cells(R.aggregate(rows, 4, np.ones(len(spec))), names, 0, 0, 4)
    assert R.continuation(cells, "com", names) == pytest.approx(4 / 10)
    assert R.continuation(cells, "de", names) == 0.0


def test_the_readings_follow_the_registered_rules():
    t = {"Delta": 0.04, "C": 0.03, "E": 0.01, "bands": {"common": {"Delta": 0.021},
                                                      "rare": {"Delta": 0.0}},
         "DQ": -0.006, "M": -0.001, "K0": 0.0, "K1": -0.005}
    assert R.read_r1(t) == "CLIP-LED"
    assert R.read_r1({**t, "C": 0.02}) == "CLIP-PARTIAL" and R.read_r1({**t, "C": 0.01}) == "CLIP-MINOR"
    eps = {"domain": {"eps_residual": 0.3}, "page": {"eps_residual": 0.05}}
    assert R.read_r2({**t, "E": 0.012}, eps) == "FRAME-PULL"
    assert R.read_r2({**t, "E": 0.009}, eps) == "NOT-NEEDED"
    assert R.read_r2({**t, "E": 0.012}, {"domain": {"eps_residual": 0.3},
                                         "page": {"eps_residual": 0.25}}) == "OFFSET"
    assert R.read_r3(t) == "COMMON-LED"
    assert R.read_r5(t) == "Q-PATH"
