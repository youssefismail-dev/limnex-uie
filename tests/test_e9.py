"""E9 -- the length-neutral objective and the seeded start
(research/native/PREREG_E9.md; FINDING_LENGTH_WEIGHTING.md).

Pins, per the registration:
- flags off, training is byte-identical to the code before E9 (a fingerprint
  recorded on that code);
- "fixed" multiplies every clipped gradient by T_b / C, T_b being exactly the
  loss's own denominator, and nothing else changes; "batch" never rescales;
- unclipped, the scaled gradients sum to the gradient of (all scored-token
  losses) / C -- the length-neutral objective;
- C defaults to batch_size x the expected scored-token mass per draw under the
  sampler;
- a seeded start gives identical initial weights, and every run records its
  start's fingerprint and what clipping did by batch length.
"""
import hashlib
from dataclasses import asdict

import pytest
import torch

import uie.train as T
from test_e7_fields import FIT
from uie.free import weights_sha256
from uie.kaggle_train import new_model
from uie.model import ModelConfig, URLTransformer
from uie.tokenizer import URLTokenizer
from uie.train import (TrainConfig, URLDataset, batch_mass, encode_corpus,
                       expected_scored_tokens, train)

#: Recorded on commit 375e8a4 (before any E9 code) by the same tiny run below.
PRE_E9_FINGERPRINT = "bf32a1f564f260a4324646e1ad5e7f2fe805462ac07c6a86286e18ec0524bcc8"


def tiny_setup():
    urls = FIT[:40]
    tok = URLTokenizer(include_frame=True, include_fields=True).fit(FIT)
    labels = ["benign" if i % 3 else "phishing" for i in range(len(urls))]
    corpora = ["name" if i % 2 else "structure" for i in range(len(urls))]
    enc, _ = encode_corpus(urls, labels, tok, 64, corpora=corpora)
    return tok, URLDataset(enc)


def tiny_model(tok, seed=0):
    torch.manual_seed(seed)
    return URLTransformer(ModelConfig(vocab_size=tok.vocab_size, max_len=64, d_model=16,
                                      n_layers=1, n_heads=2, d_ff=32))


def fingerprint(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def cfg(**kw):
    base = dict(steps=6, batch_size=4, warmup=2, eval_every=3, eval_batches=1,
                field_dropout=0.5, bucket_window=2)
    base.update(kw)
    return TrainConfig(**base)


# ---- flags off: nothing moved --------------------------------------------------------

@pytest.mark.reference_machine
def test_flags_off_training_is_byte_identical_to_pre_e9():
    torch.set_num_threads(1)
    tok, ds = tiny_setup()
    model = tiny_model(tok)
    train(model, ds, ds, tok, cfg(), device=torch.device("cpu"), log_every=0)
    assert fingerprint(model) == PRE_E9_FINGERPRINT


def test_batch_mode_never_rescales(monkeypatch):
    calls = []
    monkeypatch.setattr(T, "_scale_gradients", lambda m, f: calls.append(f))
    tok, ds = tiny_setup()
    train(tiny_model(tok), ds, ds, tok, cfg(), device=torch.device("cpu"), log_every=0)
    assert calls == []


def test_older_train_configs_still_load():
    old = {k: v for k, v in asdict(TrainConfig()).items()
           if k not in ("loss_norm", "loss_constant")}
    c = TrainConfig(**old)
    assert c.loss_norm == "batch" and c.loss_constant is None


def test_an_unknown_objective_is_refused():
    tok, ds = tiny_setup()
    with pytest.raises(ValueError):
        train(tiny_model(tok), ds, ds, tok, cfg(loss_norm="sum"),
              device=torch.device("cpu"), log_every=0)


# ---- "fixed": the registered change and only it ---------------------------------------

def test_fixed_scales_every_clipped_gradient_by_mass_over_c(monkeypatch):
    factors, masses = [], []
    real_scale, real_mass = T._scale_gradients, T.batch_mass

    def scale(model, f):
        factors.append(f)
        real_scale(model, f)

    def mass(ids, weights, pad):
        m = real_mass(ids, weights, pad)
        masses.append(m)
        return m

    monkeypatch.setattr(T, "_scale_gradients", scale)
    monkeypatch.setattr(T, "batch_mass", mass)
    tok, ds = tiny_setup()
    res = train(tiny_model(tok), ds, ds, tok, cfg(loss_norm="fixed", loss_constant=50.0),
                device=torch.device("cpu"), log_every=0)
    assert len(factors) == 6 and len(masses) == 6
    assert factors == pytest.approx([m / 50.0 for m in masses])
    assert res["loss_norm"] == "fixed" and res["loss_constant"] == 50.0


def test_batch_mass_is_the_losss_own_denominator():
    tok, ds = tiny_setup()
    model = tiny_model(tok).eval()           # an identity check: no dropout draws
    rows = ds.by_class["benign"][:3] + ds.by_class["phishing"][:2]
    ids, w = T._pad(rows, tok.pad_id)
    per_token = torch.nn.functional.cross_entropy(
        model(ids[:, :-1]).reshape(-1, tok.vocab_size), ids[:, 1:].reshape(-1),
        ignore_index=tok.pad_id, reduction="none").view(ids[:, 1:].shape)
    manual = (per_token * w * (ids[:, 1:] != tok.pad_id)).sum() / batch_mass(ids, w, tok.pad_id)
    assert torch.allclose(model.loss(ids, tok.pad_id, w), manual, atol=1e-6)


def test_unclipped_the_scaled_gradients_are_the_token_sum_objective():
    """sum_b (T_b / C) grad(mean loss of b) == grad(sum of all scored-token losses) / C."""
    tok, ds = tiny_setup()
    model = tiny_model(tok).eval()           # an identity check: no dropout draws
    rows = ds.by_class["benign"]
    short = sorted(rows, key=lambda r: len(r[0]))[:4]
    long = sorted(rows, key=lambda r: len(r[0]))[-4:]
    C = 37.0
    acc = [torch.zeros_like(p) for p in model.parameters()]
    total = 0.0
    for chunk in (short, long):
        ids, w = T._pad(chunk, tok.pad_id)
        model.zero_grad()
        model.loss(ids, tok.pad_id, w).backward()
        s = batch_mass(ids, w, tok.pad_id) / C
        for a, p in zip(acc, model.parameters()):
            a += s * p.grad
        total = total + model.loss(ids, tok.pad_id, w).detach() * batch_mass(ids, w, tok.pad_id)
    model.zero_grad()
    token_sum = 0.0
    for chunk in (short, long):
        ids, w = T._pad(chunk, tok.pad_id)
        token_sum = token_sum + model.loss(ids, tok.pad_id, w) * batch_mass(ids, w, tok.pad_id)
    (token_sum / C).backward()
    for a, p in zip(acc, model.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-6)


def test_the_constant_defaults_to_the_expected_mass_per_draw():
    rows = {"a": [([1, 2, 3], [1.0, 1.0]), ([1, 2, 3, 4, 5], [1.0, 1.0, 1.0, 1.0])],
            "b": [([1] * 11, [1.0] * 10)]}
    ds = URLDataset(rows, label_counts={"a": 3, "b": 1})
    assert expected_scored_tokens(ds, TrainConfig(class_alpha=1.0)) == pytest.approx(
        0.75 * 3 + 0.25 * 10)
    tok, tds = tiny_setup()
    res = train(tiny_model(tok), tds, tds, tok, cfg(loss_norm="fixed"),
                device=torch.device("cpu"), log_every=0)
    assert res["loss_constant"] == pytest.approx(4 * expected_scored_tokens(tds, cfg()))


# ---- the seeded start and the records ---------------------------------------------------

def test_a_seeded_start_is_identical_and_an_unseeded_one_is_not():
    tok, _ = tiny_setup()
    c = ModelConfig(vocab_size=tok.vocab_size, max_len=64, d_model=16, n_layers=1,
                    n_heads=2, d_ff=32)
    torch.manual_seed(1)
    a = new_model(c, 20260803)
    torch.manual_seed(2)                                  # a different process state
    b = new_model(c, 20260803)
    assert weights_sha256(a) == weights_sha256(b)
    torch.manual_seed(1)
    u1 = new_model(c)
    torch.manual_seed(2)
    u2 = new_model(c)
    assert weights_sha256(u1) != weights_sha256(u2)


def test_every_run_records_its_start_and_what_clipping_did():
    tok, ds = tiny_setup()
    model = tiny_model(tok)
    start = weights_sha256(model)
    res = train(model, ds, ds, tok, cfg(), device=torch.device("cpu"), log_every=0)
    assert res["init_weights_sha256"] == start
    diag = res["grad_diagnostics"]
    assert sum(v["steps"] for k, v in diag.items() if k.startswith("all ")) == 6
    for v in diag.values():
        assert 0.0 <= v["clipped_share"] <= 1.0
