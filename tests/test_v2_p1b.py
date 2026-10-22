"""V2-P1b (research/v2/PREREG_V2_P1B.md, T2 and V-hook): the optional optimiser hook in
`uie.train.train`.

- Off by default, the training path is bit-identical to the code before the hook: the
  fingerprints below are the pre-hook uie/train.py's own (as at 904a79a and 18de524), from
  this exact 4-step CPU training on ONE thread (data/v2/p1b/fingerprint_old_vs_new.json:
  the old and the new code give identical weights and losses at 1 and at 6 threads). Weights,
  final validation losses, the result's fields and the saved checkpoint's settings must all
  reproduce. The thread count is pinned and restored by the test, as E9's byte-identity test
  pins it: CPU matrix sums round differently with the thread count, and test_e9 leaves it at
  1 (the full suite exposed exactly that).
- A memory model without `mem_values_lr` (P1) takes the default path.
- With `mem_values_lr`, the values' group is Adam exactly (AdamW at weight decay 0), with
  betas (0.9, 0.999), and its rate is the schedule x mem_values_lr / lr every step.
- A memory checkpoint written before these settings existed still loads.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

from test_e15 import tok
from test_e16 import dataset, model_for
from uie.free import weights_sha256
from uie.train import TrainConfig, _lr_at, save_checkpoint, train

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "v2"))

import v2_mem as M  # noqa: E402

TRAIN_CONFIG = {"batch_size": 4, "boundary_weight": 1.0, "bucket_window": 64, "class_alpha": 1.0,
                "eval_batches": 40, "eval_every": 2, "field_dropout": 0.0, "field_dropout_joint": False,
                "grad_clip": 1.0, "grad_clip_late": 5.0, "grad_clip_switch": 2, "loss_constant": None,
                "loss_norm": "batch", "lr": 0.0003, "seed": 7, "steps": 4, "warmup": 500,
                "weight_decay": 0.01}
#: the pre-hook code's own fingerprints (uie/train.py as at 904a79a), on one thread
BEFORE = {
    1: {"weights": "72eb59890090bac426fff82fc3655f8255345fecc40a3e45d1aefc0698c8238f",
        "final": {"benign": 4.049744447072347, "phishing": 4.0528818766276045, "mean": 4.051313161849976},
        "train_config": TRAIN_CONFIG},
    2: {"weights": "e881490b31f8feab26ce90ec9460bdbda1f0da21fadd5d22c7535765395358aa",
        "final": {"benign": 4.049746778276232, "phishing": 4.052883466084798, "mean": 4.051315122180515},
        "train_config": {**TRAIN_CONFIG, "accum_steps": 2}},
}
THREADS = 1
RESULT_KEYS = ["best_val_mean", "device", "final", "grad_diagnostics", "history", "init_weights_sha256",
               "loss_constant", "loss_norm", "seconds"]
MODEL_CONFIG = {"d_ff": 64, "d_model": 16, "dropout": 0.1, "max_len": 128, "n_heads": 2, "n_layers": 2,
                "struct_ids": [], "vocab_size": 57}


@pytest.mark.reference_machine
@pytest.mark.parametrize("accum", [1, 2])
def test_default_training_is_bit_identical_to_before_the_hook(accum, tmp_path):
    t = tok()
    data = dataset(t)
    model = model_for(t, seed=0, dropout=0.1)
    assert not hasattr(model, "optimizer_groups")            # no uie model defines the hook
    cfg = TrainConfig(steps=4, batch_size=4, eval_every=2, accum_steps=accum, grad_clip_late=5.0,
                      grad_clip_switch=2, seed=7)
    before = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    try:
        res = train(model, data, data, t, cfg, device=torch.device("cpu"), out_dir=tmp_path, log_every=0)
    finally:
        torch.set_num_threads(before)
    save_checkpoint(model, t, cfg, tmp_path / "final.pt")
    blob = torch.load(tmp_path / "final.pt", weights_only=False)
    want = BEFORE[accum]
    assert weights_sha256(model) == want["weights"]
    assert res["final"] == want["final"]
    assert sorted(res) == RESULT_KEYS
    assert json.loads(json.dumps(blob["train_config"])) == want["train_config"]
    assert json.loads(json.dumps(blob["model_config"])) == MODEL_CONFIG


def small_memory_model(t, **kw):
    torch.manual_seed(0)
    from uie.model import ModelConfig, URLTransformer
    base = URLTransformer(ModelConfig(vocab_size=t.vocab_size, max_len=64, d_model=16, n_layers=2, n_heads=2,
                                      d_ff=64, dropout=0.0))
    return M.with_memory(base, 3, t.pad_id, mem_after=1, mem_n_keys=16, mem_heads=2, mem_knn=4, mem_k_dim=8,
                         **kw)[0]


def test_a_memory_model_without_values_lr_takes_the_default_path():
    t = tok()
    assert small_memory_model(t).optimizer_groups(TrainConfig()) is None
    assert small_memory_model(t, mem_query_norm="layer").optimizer_groups(TrainConfig()) is None


def test_the_values_group_is_adam_exactly():
    t = tok()
    model = small_memory_model(t, mem_query_norm="layer", mem_values_lr=1e-3)
    cfg = TrainConfig(lr=3e-4, weight_decay=0.01)
    groups = model.optimizer_groups(cfg)
    values = model.memory.values
    assert groups[1]["params"] == [values] and groups[1]["weight_decay"] == 0.0
    assert groups[1]["betas"] == (0.9, 0.999) and groups[1]["lr_scale"] == pytest.approx(1e-3 / 3e-4)
    assert all(p is not values for p in groups[0]["params"])
    assert sum(p.numel() for g in groups for p in g["params"]) == sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    twin = torch.nn.Parameter(values.detach().clone())
    ref = torch.optim.Adam([twin], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    g = torch.Generator().manual_seed(1)
    for step in range(3):
        grad = torch.randn(values.shape, generator=g)
        for group in opt.param_groups:
            group["lr"] = 1e-3 if "lr_scale" in group else 3e-4
        values.grad, twin.grad = grad.clone(), grad.clone()
        for p in groups[0]["params"]:
            p.grad = None
        opt.step()
        ref.step()
        assert torch.equal(values.detach(), twin.detach()), step


def test_training_uses_the_values_group_with_its_scaled_rate(monkeypatch, tmp_path):
    t = tok()
    data = dataset(t)
    model = small_memory_model(t, mem_query_norm="layer", mem_values_lr=1e-3)
    made = []
    real = torch.optim.AdamW

    def spy(*a, **k):
        made.append(real(*a, **k))
        return made[-1]
    monkeypatch.setattr(torch.optim, "AdamW", spy)
    cfg = TrainConfig(steps=3, batch_size=4, eval_every=0, seed=7)
    train(model, data, data, t, cfg, device=torch.device("cpu"), out_dir=tmp_path, log_every=0)
    g0, g1 = made[0].param_groups
    assert g1["params"] == [model.memory.values] and g1["weight_decay"] == 0.0 and g1["betas"] == (0.9, 0.999)
    assert g0["weight_decay"] == 0.01 and g0["betas"] == (0.9, 0.95)
    assert g0["lr"] == _lr_at(2, cfg) and g1["lr"] == pytest.approx(_lr_at(2, cfg) * 1e-3 / 3e-4, rel=1e-12)


def test_a_memory_checkpoint_from_before_the_settings_still_loads(tmp_path):
    t = tok()
    model = small_memory_model(t)
    save_checkpoint(model, t, TrainConfig(), tmp_path / "old.pt")
    blob = torch.load(tmp_path / "old.pt", weights_only=False)
    for k in ("mem_query_norm", "mem_branch_dropout", "mem_values_lr"):
        blob["model_config"].pop(k)                          # P1's layout, written before them
    torch.save(blob, tmp_path / "old.pt")
    back, _, cfg = M.load_checkpoint(tmp_path / "old.pt")
    assert cfg.mem_query_norm == "batch" and cfg.mem_branch_dropout and cfg.mem_values_lr is None
    ids = torch.randint(4, t.vocab_size, (2, 9))
    assert torch.equal(model.eval()(ids), back.eval()(ids))


# ---- P1b's runner: T4's audit inside the run, T2's group through the real training path ----

def tiny_corpus(d):
    import gzip as _gz
    from test_e15 import TINY_TRAIN

    def w(name, urls, label):
        with _gz.open(d / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for u in urls:
                fh.write(json.dumps({"url": u, "label": label}) + "\n")
    w("name_train", TINY_TRAIN * 3, "benign")
    w("structure_train", TINY_TRAIN[:2], "benign")
    w("phishing_name_train", TINY_TRAIN * 3, "phishing")
    w("phishing_structure_train", TINY_TRAIN[:2], "phishing")
    for v in ("name_val", "structure_val", "phishing_name_val", "phishing_structure_val"):
        w(v, TINY_TRAIN[:4], "phishing" if v.startswith("phishing") else "benign")


TINY_RUN = dict(steps=3, batch_size=4, d_model=16, n_layers=2, n_heads=2, eval_every=3, frame=True,
                fields=True, presence_plan=True, root_slash=True, slash_plan=True, seed_init=True,
                e12_start=True, loss_norm="fixed", loss_constant=100.0, max_len=384, accum_steps=2)


def test_p1b_runner_audits_the_model_it_trains_and_trains_the_values_group(tmp_path, monkeypatch):
    import uie.kaggle_train as KT
    import v2_p1b_train as TB
    d = tmp_path / "data"
    d.mkdir()
    tiny_corpus(d)
    monkeypatch.setattr(M, "REGISTERED", {**M.REGISTERED, "mem_after": 1, "mem_n_keys": 16, "mem_heads": 2,
                                          "mem_knn": 4, "mem_k_dim": 8})
    monkeypatch.setattr(KT, "new_model_e12", TB.memory_builder(KT.new_model_e12))
    made = []
    real = torch.optim.AdamW

    def spy(*a, **k):
        made.append(real(*a, **k))
        return made[-1]
    monkeypatch.setattr(torch.optim, "AdamW", spy)
    res = KT.run(d, tmp_path / "out", **TINY_RUN)
    assert res["e12_start"]["audit_in_run"]["pass"] and TB.LAST_AUDIT["pass"]
    model, _, cfg = M.load_checkpoint(tmp_path / "out" / "final.pt")
    assert (cfg.mem_query_norm, cfg.mem_branch_dropout, cfg.mem_values_lr) == ("layer", False, 1e-3)
    g0, g1 = made[0].param_groups
    assert g1["weight_decay"] == 0.0 and g1["betas"] == (0.9, 0.999) and len(g1["params"]) == 1
    assert g1["params"][0].shape == model.memory.values.shape


def test_a_failing_audit_stops_the_run_before_training(tmp_path, monkeypatch):
    import uie.kaggle_train as KT
    import v2_p1b_common as P1B
    import v2_p1b_train as TB
    d = tmp_path / "data"
    d.mkdir()
    tiny_corpus(d)
    monkeypatch.setattr(M, "REGISTERED", {**M.REGISTERED, "mem_after": 1, "mem_n_keys": 16, "mem_heads": 2,
                                          "mem_knn": 4, "mem_k_dim": 8})
    monkeypatch.setattr(P1B, "audit", lambda *a, **k: {"pass": False})
    original = KT.new_model_e12
    KT.new_model_e12 = TB.memory_builder(original)
    try:
        with pytest.raises(TB.AuditFailed):
            KT.run(d, tmp_path / "out", **TINY_RUN)
    finally:
        KT.new_model_e12 = original
    assert not (tmp_path / "out" / "final.pt").exists()


def test_p1b_runner_takes_e16_caps_recipe_only_and_restores_the_builder(tmp_path):
    import uie.kaggle_train as KT
    import v2_p1b_common as P1B
    import v2_p1b_train as TB
    original = KT.new_model_e12
    with pytest.raises(ValueError):
        TB.run(tmp_path, tmp_path, **{**P1B.recipe(tmp_path / P1B.STREAM_FILE), "lr": 1e-3})
    with pytest.raises(FileNotFoundError):
        TB.run(tmp_path, tmp_path / "out", **P1B.recipe(tmp_path / P1B.STREAM_FILE))
    assert KT.new_model_e12 is original


def test_the_audit_bar_and_rows():
    import v2_p1b_common as P1B
    t = tok()
    good = small_memory_model(t, mem_query_norm="layer")
    rows = [list(range(4, 4 + n)) for n in (5, 9, 14, 20, 27, 33)]
    rows = [[x % t.vocab_size or 4 for x in r] for r in rows]
    assert P1B.audit(good, rows, t.pad_id)["pass"]
    bad = small_memory_model(t)                               # P1's BatchNorm
    bad.train()(torch.randint(4, t.vocab_size, (8, 12)))
    assert not P1B.audit(bad, rows, t.pad_id)["pass"]
    assert P1B.MEM_PARAMS == 136_318_976 and P1B.PARAMS == 161_792_000 and P1B.ARM == "p1b_mem"
