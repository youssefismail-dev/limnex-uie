"""V2-P1 (research/v2/PREREG_V2_P1_MEMORY.md sections 3 and 7): the memory layer, its start,
its checkpoints and its training path, checked on toy models before anything runs on a GPU.

- the product-key search is the exact top-k over every slot;
- padding is skipped and kept out of the BatchNorm's statistics;
- V-equiv: with every value row at 0 the model computes exactly the base model's logits;
- the start keeps the base's weights and draws the memory from its own seed;
- a checkpoint loads back exactly, and `uie`'s plain loader refuses it;
- gradients reach every part of the memory;
- V-iso: `uie/` is unchanged since the registration.
"""
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from test_e15 import TINY_TRAIN, tok
from uie.model import ModelConfig, URLTransformer, generate
from uie.train import TrainConfig, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "v2"))

import v2_mem as M  # noqa: E402

SMALL = {"mem_n_keys": 16, "mem_heads": 2, "mem_knn": 4, "mem_k_dim": 8}


def base_model(t, dropout=0.0, d_model=16, seed=0):
    torch.manual_seed(seed)
    return URLTransformer(ModelConfig(vocab_size=t.vocab_size, max_len=64, d_model=d_model,
                                      n_layers=2, n_heads=2, d_ff=4 * d_model, dropout=dropout))


def mem_model(t, dropout=0.0, seed=5, **kw):
    model, prints = M.with_memory(base_model(t, dropout), seed, t.pad_id,
                                  **{**SMALL, "mem_after": 1, **kw})
    return model, prints


def some_ids(t, rows=3, length=12, pad_tail=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(4, t.vocab_size, (rows, length), generator=g)
    if pad_tail:
        ids[-1, -pad_tail:] = t.pad_id
    return ids


# ---- the search ---------------------------------------------------------------------------

def test_select_is_the_exact_top_k_over_every_slot():
    torch.manual_seed(0)
    m = M.ProductKeyMemory(d_in=8, d_out=4, n_keys=16, heads=2, knn=5, k_dim=8)
    torch.nn.init.normal_(m.keys)
    q = torch.randn(10, 2, 8)
    slots, scores = m.select(q)
    for h in range(2):
        s1 = q[:, h, :4] @ m.keys[h, 0].t()
        s2 = q[:, h, 4:] @ m.keys[h, 1].t()
        full = (s1.unsqueeze(2) + s2.unsqueeze(1)).reshape(10, -1)      # all 256 slots
        best, idx = full.topk(5, dim=1)
        got = slots[:, h * 5:(h + 1) * 5]
        assert torch.equal(torch.sort(got, 1).values, torch.sort(idx, 1).values)
        assert torch.allclose(scores[:, h * 5:(h + 1) * 5], best)


def test_padding_is_skipped_and_kept_out_of_the_batch_statistics():
    torch.manual_seed(1)
    a = M.ProductKeyMemory(d_in=8, d_out=4, n_keys=8, heads=2, knn=3, k_dim=8)
    torch.nn.init.normal_(a.keys)
    torch.nn.init.normal_(a.values)
    b = M.ProductKeyMemory(d_in=8, d_out=4, n_keys=8, heads=2, knn=3, k_dim=8)
    b.load_state_dict(a.state_dict())
    u = torch.randn(2, 5, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[1, 3:] = False
    out = a.train()(u, valid)
    assert torch.equal(out[1, 3:], torch.zeros(2, 4))
    assert out[0].abs().sum() > 0
    b.train()(u[valid].unsqueeze(0), torch.ones(1, int(valid.sum()), dtype=torch.bool))
    assert torch.allclose(a.query_bn.running_mean, b.query_bn.running_mean)
    assert torch.allclose(a.query_bn.running_var, b.query_bn.running_var)


# ---- V-equiv ------------------------------------------------------------------------------

@pytest.mark.parametrize("train_mode", [False, True])
def test_zero_values_give_the_base_logits_exactly(train_mode):
    t = tok()
    base = base_model(t)
    model, _ = M.with_memory(base, 7, t.pad_id, **SMALL, mem_after=1)
    with torch.no_grad():
        model.memory.values.zero_()
    ids = some_ids(t, pad_tail=3)
    base.train(train_mode)
    model.train(train_mode)
    assert torch.equal(base(ids), model(ids))


def test_the_memory_changes_the_logits_once_its_values_are_drawn():
    t = tok()
    base = base_model(t)
    model, _ = M.with_memory(base, 7, t.pad_id, **SMALL, mem_after=1)
    ids = some_ids(t)
    assert not torch.equal(base.eval()(ids), model.eval()(ids))


# ---- the start ----------------------------------------------------------------------------

def test_the_start_keeps_the_base_and_draws_the_memory_from_its_seed():
    from uie.free import weights_sha256
    t = tok()
    base = base_model(t)
    a, pa = M.with_memory(base, 11, t.pad_id, **SMALL, mem_after=2)
    b, pb = M.with_memory(base, 11, t.pad_id, **SMALL, mem_after=2)
    c, pc = M.with_memory(base, 12, t.pad_id, **SMALL, mem_after=2)
    assert pa["base_init_sha256"] == weights_sha256(base) == pc["base_init_sha256"]
    assert M.part_sha256(a, lambda k: not M.is_memory_key(k)) == weights_sha256(base)
    assert pa == pb and weights_sha256(a) == weights_sha256(b)
    assert pa["memory_init_sha256"] != pc["memory_init_sha256"]
    m = a.memory
    bound = 1 / 8 ** 0.5
    assert m.keys.abs().max() <= bound and m.keys.abs().max() > 0.5 * bound
    assert torch.all(m.query_proj.bias == 0) and torch.all(m.query_bn.running_mean == 0)
    assert abs(float(m.values.detach().std()) - 16 ** -0.5) < 0.05
    assert a.cfg.mem_pad_id == t.pad_id and a.cfg.mem_after == 2
    with pytest.raises(ValueError):
        M.with_memory(base, 11, t.pad_id, mem_values=3)
    with pytest.raises(ValueError):                         # the registered block 4 needs 4 blocks
        M.with_memory(base, 11, t.pad_id, **SMALL)


def test_registered_size():
    assert M.memory_param_count(512, 512, 4, 512) == 136_322_048
    with torch.device("meta"):
        m = M.ProductKeyMemory(512, 512, **{"n_keys": 512, "heads": 4, "knn": 32, "k_dim": 512})
    assert sum(p.numel() for p in m.parameters()) + 2 * 512 == 136_322_048
    import v2_p1_common as P1
    assert P1.MEM_PARAMS == 136_322_048 and P1.PARAMS == 161_795_072


# ---- checkpoints --------------------------------------------------------------------------

def test_checkpoint_round_trip_and_the_plain_loader_refuses(tmp_path):
    from uie.train import load_checkpoint as plain
    t = tok()
    model, _ = mem_model(t)
    path = tmp_path / "final.pt"
    save_checkpoint(model, t, TrainConfig(), path)
    back, t2, cfg = M.load_checkpoint(path)
    assert isinstance(back, M.MemURLTransformer) and cfg.mem_n_keys == 16
    assert t2.state_dict() == t.state_dict()
    for k, v in model.state_dict().items():
        assert torch.equal(v, back.state_dict()[k]), k
    ids = some_ids(t)
    assert torch.equal(model.eval()(ids), back.eval()(ids))
    with pytest.raises(TypeError):
        plain(path)
    base = tmp_path / "base.pt"
    save_checkpoint(base_model(t), t, TrainConfig(), base)
    assert type(M.load_checkpoint(base)[0]) is URLTransformer     # a plain checkpoint still loads


# ---- training -----------------------------------------------------------------------------

def test_gradients_reach_every_part_of_the_memory():
    t = tok()
    model, _ = mem_model(t)
    model.train()
    model.memory.trace = []
    ids = some_ids(t, pad_tail=2)
    model.loss(ids, t.pad_id).backward()
    m = model.memory
    chosen = torch.unique(torch.cat(m.trace).flatten())
    rows = (m.values.grad.abs().sum(1) > 0).nonzero().flatten()
    assert len(rows) > 0 and set(rows.tolist()) <= set(chosen.tolist())
    for p in (m.keys, m.query_proj.weight, m.query_bn.weight, model.ln_mem.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_in_evaluation_a_row_does_not_depend_on_its_batch():
    t = tok()
    model, _ = mem_model(t)
    model.train()(some_ids(t, rows=6, seed=3))          # move the running statistics
    model.eval()
    ids = some_ids(t, rows=3, seed=4)
    alone = model(ids[1:2])
    together = model(ids)[1:2]
    assert torch.allclose(alone, together, atol=1e-5)


def test_generation_runs_and_is_deterministic():
    t = tok()
    model, _ = mem_model(t)
    a = generate(model, t, label="benign", n=3, seed=9, suffix="com")
    b = generate(model, t, label="benign", n=3, seed=9, suffix="com")
    assert a == b and len(a) == 3


def test_memory_preflight_runs_on_the_cpu():
    peak = M.memory_preflight(torch.device("cpu"), 2, vocab_size=40, d_model=16, n_layers=2,
                              n_heads=2, max_len=24, **SMALL, mem_after=1)
    assert peak == 0.0


def test_the_runner_takes_e16_caps_recipe_only_and_restores_the_builder(tmp_path):
    import uie.kaggle_train as KT
    import v2_p1_common as P1
    import v2_p1_train as T
    original = KT.new_model_e12
    stream = tmp_path / P1.STREAM_FILE
    with pytest.raises(ValueError):
        T.run(tmp_path, tmp_path, **{**P1.recipe(stream), "lr": 1e-3})
    with pytest.raises(FileNotFoundError):                 # the right recipe, but no data here
        T.run(tmp_path, tmp_path / "out", **P1.recipe(stream))
    assert KT.new_model_e12 is original


def test_the_recipe_is_e16_caps():
    import e16_common as E16
    import v2_p1_common as P1
    for parts in (2, 4):
        assert P1.recipe(Path("s.jsonl"), parts) == E16.recipe_cap(Path("s.jsonl"), parts)


def test_training_through_kaggle_run_end_to_end(tmp_path, monkeypatch):
    """The real training path at toy width: uie.kaggle_train.run with the memory builder,
    then the memory checkpoint loads and generates."""
    import uie.kaggle_train as KT
    import v2_p1_train as T
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
    monkeypatch.setattr(KT, "new_model_e12", T.memory_builder(KT.new_model_e12))
    monkeypatch.setattr(M, "REGISTERED", {**M.REGISTERED, **SMALL, "mem_after": 1})
    kw = dict(steps=3, batch_size=4, d_model=16, n_layers=2, n_heads=2, eval_every=3,
              frame=True, fields=True, presence_plan=True, root_slash=True, slash_plan=True,
              seed_init=True, e12_start=True, loss_norm="fixed", loss_constant=100.0,
              max_len=384, accum_steps=2)
    res = KT.run(d, tmp_path / "out", **kw)
    s = res["e12_start"]
    assert {"base_init_sha256", "memory_init_sha256", "memory_seed", "e10_start_sha256"} <= set(s)
    assert s["memory_seed"] == 20261050
    model, t, cfg = M.load_checkpoint(tmp_path / "out" / "final.pt")
    assert isinstance(model, M.MemURLTransformer) and cfg.mem_after == 1
    assert res["params"] == sum(p.numel() for p in model.parameters())
    assert all(v == v for v in res["final"].values())        # finite validation
    assert res["init_weights_sha256"] != s["base_init_sha256"]   # the whole start = base + memory
    assert len(generate(model, t, label="phishing", n=2, seed=3, suffix="com")) == 2


# ---- V-iso --------------------------------------------------------------------------------

def test_uie_was_unchanged_from_p1s_registration_to_the_code_p1_ran():
    """P1's V-iso, as history: uie/ did not change between P1's registration (904a79a) and the
    commit P1's bundle was built from (605f776), which V2P1_train ran. P1b's T2 changes uie/
    afterwards, on purpose (PREREG_V2_P1B.md), with its own V-iso (tests/test_v2_p1b.py)."""
    import v2_p1_common as P1
    for commit in (P1.REGISTRATION_COMMIT, "605f776962ea"):
        if subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT,
                          capture_output=True).returncode != 0:
            pytest.skip("needs the research history (the public repository is one commit)")
    diff = subprocess.run(["git", "diff", "--name-only", P1.REGISTRATION_COMMIT,
                           "605f776962ea", "--", "uie"], cwd=ROOT, capture_output=True, text=True)
    assert diff.returncode == 0 and diff.stdout.strip() == "", diff.stdout


# ---- V2-P1b: the query LayerNorm, and the batch-coupling audit ------------------------------

def coupled_rows(t, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(4, t.vocab_size, (n,), generator=g).tolist() for n in (5, 9, 14, 20, 27, 33)]


def test_the_layer_norm_query_has_no_batch_coupling():
    t = tok()
    model, _ = mem_model(t, dropout=0.1, mem_query_norm="layer")
    assert not hasattr(model.memory, "query_bn") and hasattr(model.memory, "query_ln")
    r = M.batch_coupling_check(model, coupled_rows(t), t.pad_id)
    assert r["buffers_changed_in_training"] == []
    assert max(r["batch_vs_alone"], r["length_grouped_vs_alone"], r["training_mode_vs_alone"]) < 1e-4, r


def test_the_audit_catches_p1s_batch_norm():
    t = tok()
    model, _ = mem_model(t, dropout=0.1)                     # P1's query BatchNorm
    model.train()(some_ids(t, rows=8, seed=2))               # running statistics move off (0, 1)
    r = M.batch_coupling_check(model, coupled_rows(t), t.pad_id)
    assert r["training_mode_vs_alone"] > 1e-3
    assert any("running" in k for k in r["buffers_changed_in_training"])


def test_p1b_settings_and_sizes(tmp_path):
    t = tok()
    model, _ = mem_model(t, mem_query_norm="layer", mem_branch_dropout=False)
    assert model.cfg.mem_query_norm == "layer" and model.cfg.mem_branch_dropout is False
    save_checkpoint(model, t, TrainConfig(), tmp_path / "p1b.pt")
    back, _, cfg = M.load_checkpoint(tmp_path / "p1b.pt")
    assert cfg.mem_query_norm == "layer" and torch.equal(model.eval()(some_ids(t)), back.eval()(some_ids(t)))
    assert M.memory_param_count(512, 512, 4, 512, "layer") == 136_322_048 - 4_096 + 1_024
    assert M.MemConfig().mem_query_norm == "batch"           # P1's checkpoint loads as trained
