"""E16's library change (research/e16/PREREG_E16.md sections 3, 8, 11.1): gradient
accumulation. Each batch in K parts, each part's loss weighted by its share of the
batch's scored mass, gradients summed before the clip and the step -- the batch's own
gradient (V-accum). Off by default (K = 1), and off leaves every earlier path exactly
as it was.
"""
import gzip
import json
import random

import pytest
import torch

from test_e15 import NEW, OLD, OLD_LABELS, TINY_TRAIN, stream_file, tok
from uie.model import ModelConfig, URLTransformer
from uie.train import (TrainConfig, URLDataset, _accumulate, batches, encode_corpus,
                       load_checkpoint, save_checkpoint, train)


def model_for(t, seed=0, dropout=0.0, d_model=16):
    torch.manual_seed(seed)
    return URLTransformer(ModelConfig(vocab_size=t.vocab_size, max_len=128, d_model=d_model,
                                      n_layers=2, n_heads=2, d_ff=4 * d_model, dropout=dropout))


def dataset(t):
    enc, _ = encode_corpus(OLD, OLD_LABELS, t, 128, corpora=["name"] * len(OLD))
    return URLDataset(enc)


def a_batch(t, data, n=9):
    """One weighted batch exactly as training draws it (odd size: uneven parts)."""
    cfg = TrainConfig(batch_size=n, seed=5)
    return next(batches(data, cfg, t.pad_id, random.Random(cfg.seed)))


def grads(model):
    return [p.grad.detach().clone() for p in model.parameters()]


# ---- V-accum: the summed gradient is the batch's gradient ----------------------------------

@pytest.mark.parametrize("parts", [2, 3, 4])
def test_accumulated_gradient_equals_the_batch_gradient(parts):
    t = tok()
    data = dataset(t)
    ids, weights = a_batch(t, data)
    model = model_for(t).double()
    model.zero_grad(set_to_none=True)
    model.loss(ids, t.pad_id, weights.double()).backward()
    whole = grads(model)
    model.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = _accumulate(model, ids, weights.double(), t.pad_id, parts, scaler, False)
    split = grads(model)
    for a, b in zip(whole, split):
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-12)
    with torch.no_grad():
        assert abs(float(loss) - float(model.loss(ids, t.pad_id, weights.double()))) < 1e-10


def test_accumulation_also_holds_without_weights():
    t = tok()
    ids, _ = a_batch(t, dataset(t))
    model = model_for(t).double()
    model.zero_grad(set_to_none=True)
    model.loss(ids, t.pad_id, None).backward()
    whole = grads(model)
    model.zero_grad(set_to_none=True)
    _accumulate(model, ids, None, t.pad_id, 2, torch.amp.GradScaler("cuda", enabled=False), False)
    for a, b in zip(whole, grads(model)):
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-12)


def test_training_with_accumulation_matches_one_pass_per_batch(tmp_path):
    """The same batches, steps, clip and fixed objective: the weights agree to float32
    tolerance (only the order of additions differs), and so do the records."""
    t = tok()
    data = dataset(t)
    kw = dict(steps=5, batch_size=6, eval_every=5, eval_batches=1, warmup=1,
              loss_norm="fixed", loss_constant=60.0, grad_clip_late=5.0, grad_clip_switch=2)
    one, two = model_for(t), model_for(t)
    r1 = train(one, data, data, t, TrainConfig(**kw), device=torch.device("cpu"),
               out_dir=tmp_path / "one", log_every=100)
    r2 = train(two, data, data, t, TrainConfig(**kw, accum_steps=2), device=torch.device("cpu"),
               out_dir=tmp_path / "two", log_every=100)
    for (k, a), b in zip(one.state_dict().items(), two.state_dict().values()):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), k
    assert r1["init_weights_sha256"] == r2["init_weights_sha256"]
    steps = lambda r: {k: v["steps"] for k, v in r["grad_diagnostics"].items()}  # noqa: E731
    assert steps(r1) == steps(r2)                        # the same batches, the same masses
    assert abs(r1["final"]["mean"] - r2["final"]["mean"]) < 1e-4


def test_accumulation_with_injection_keeps_the_batches(tmp_path):
    t = tok()
    data = dataset(t)
    path = stream_file(tmp_path, NEW[:1] * 6)
    kw = dict(steps=6, batch_size=4, eval_every=6, eval_batches=1, warmup=1,
              loss_norm="fixed", loss_constant=60.0, qry_extra=str(path))
    a = train(model_for(t), data, data, t, TrainConfig(**kw), device=torch.device("cpu"),
              out_dir=tmp_path / "a", log_every=100)
    b = train(model_for(t), data, data, t, TrainConfig(**kw, accum_steps=2),
              device=torch.device("cpu"), out_dir=tmp_path / "b", log_every=100)
    assert a["qry_extra"] == b["qry_extra"]              # the same old rows, the same appended rows


# ---- off by default; recorded only when on ------------------------------------------------

def test_accumulation_is_off_by_default_and_written_only_when_on(tmp_path):
    t = tok()
    assert TrainConfig().accum_steps == 1
    save_checkpoint(model_for(t), t, TrainConfig(), tmp_path / "off.pt")
    save_checkpoint(model_for(t), t, TrainConfig(accum_steps=2), tmp_path / "on.pt")
    assert "accum_steps" not in torch.load(tmp_path / "off.pt", weights_only=False)["train_config"]
    assert torch.load(tmp_path / "on.pt", weights_only=False)["train_config"]["accum_steps"] == 2
    load_checkpoint(tmp_path / "off.pt")
    load_checkpoint(tmp_path / "on.pt")


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_accum_steps_must_be_a_positive_integer(tmp_path, bad):
    t = tok()
    data = dataset(t)
    with pytest.raises(ValueError):
        train(model_for(t), data, data, t, TrainConfig(steps=1, batch_size=4, accum_steps=bad),
              device=torch.device("cpu"), out_dir=tmp_path, log_every=100)


def test_accumulation_through_kaggle_run_end_to_end(tmp_path):
    from uie.kaggle_train import run
    d = tmp_path / "data"
    d.mkdir()

    def w(name, urls, label):
        with gzip.open(d / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for u in urls:
                fh.write(json.dumps({"url": u, "label": label}) + "\n")
    w("name_train", TINY_TRAIN * 3, "benign")
    w("structure_train", TINY_TRAIN[:2], "benign")
    w("phishing_name_train", TINY_TRAIN * 3, "phishing")
    w("phishing_structure_train", TINY_TRAIN[:2], "phishing")
    for v in ("name_val", "structure_val", "phishing_name_val", "phishing_structure_val"):
        w(v, TINY_TRAIN[:4], "phishing" if v.startswith("phishing") else "benign")
    kw = dict(steps=4, batch_size=4, d_model=16, n_layers=1, n_heads=2, eval_every=4,
              frame=True, fields=True, presence_plan=True, root_slash=True, slash_plan=True,
              seed_init=True, loss_norm="fixed", loss_constant=100.0, max_len=128)
    on = run(d, tmp_path / "on", accum_steps=2, **kw)
    off = run(d, tmp_path / "off", **kw)
    assert on["accum_steps"] == 2 and off["accum_steps"] == 1
    assert torch.load(tmp_path / "on" / "final.pt", weights_only=False)["train_config"]["accum_steps"] == 2
    assert "accum_steps" not in torch.load(tmp_path / "off" / "final.pt", weights_only=False)["train_config"]


# ---- the wider start: E12's procedure, at any width, deterministic ------------------------

def test_e12s_start_procedure_is_width_generic_and_deterministic():
    from uie.free import weights_sha256
    from uie.kaggle_train import new_model_e12
    t = tok()
    shas = {}
    for d in (32, 48):
        cfg = ModelConfig(vocab_size=t.vocab_size, max_len=384, d_model=d, n_layers=2,
                          n_heads=2, d_ff=4 * d)
        a, pa = new_model_e12(cfg, t, 20260803)
        b, pb = new_model_e12(cfg, t, 20260803)
        assert weights_sha256(a) == weights_sha256(b) and pa == pb
        assert a.cfg.d_model == d
        shas[d] = (weights_sha256(a), pa["e10_start_sha256"])
    assert shas[32] != shas[48]                          # the width is part of the start
