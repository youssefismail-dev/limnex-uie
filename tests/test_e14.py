"""E14's library changes (research/e14/PREREG_E14.md sections 2.1-2.2, V3, V5, V6):
the query weight (Q) in training and D1' (the slash rule on plans with a path) in
generation. Both are off by default, and off leaves every earlier path exactly as
it was (the E7-E13 suites run unchanged).
"""
import gzip
import json

import pytest
import torch

from test_e7_fields import FIT
from uie.model import ModelConfig, URLTransformer, generate, slash_ends
from uie.tokenizer import URLTokenizer
from uie.train import (TrainConfig, URLDataset, encode_corpus, field_mask,
                       load_checkpoint, save_checkpoint, train)


def tok():
    return URLTokenizer(include_frame=True, include_fields=True, root_slash=True,
                        presence_plan=True, slash_plan=True).fit(FIT)


def model_for(t, max_len=128, seed=0):
    torch.manual_seed(seed)
    return URLTransformer(ModelConfig(vocab_size=t.vocab_size, max_len=max_len, d_model=16,
                                      n_layers=1, n_heads=2, d_ff=32))


def end_ids(t):
    return {"PTH": t.marker_id("PTH"), "RS": t.rs_id, "QRY": t.marker_id("QRY"),
            "FRG": t.marker_id("FRG"), "EOS": t.eos_id}


# ---- Q: the mask ------------------------------------------------------------------------

def _query_steps(t, ids):
    """Target positions (indices into ids) predicted while the query is open: from the
    first character after <QRY> through the step that predicts the query's end."""
    q = ids.index(t.marker_id("QRY"))
    ends = {t.marker_id("FRG"), t.eos_id}
    e = next(i for i in range(q + 1, len(ids)) if ids[i] in ends)
    return set(range(q + 1, e + 1))


@pytest.mark.parametrize("url", ["https://a.example.com/x?id=12#top",
                                 "https://a.example.com/?id=1&b=2",
                                 "https://b.example.com?x=1"])
def test_q_weights_exactly_the_steps_inside_the_query(url):
    t = tok()
    ids = t.encode(url, "phishing", frame="domain")
    base = field_mask(ids, t, "name")
    w4 = field_mask(ids, t, "name", query_weight=4.0)
    inside = _query_steps(t, ids)
    assert len(inside) >= 3                                  # not vacuous
    for pos in range(1, len(ids)):
        b, w = base[pos - 1], w4[pos - 1]
        if pos in inside:
            assert b == 1.0 and w == 4.0, (pos, t.itos[ids[pos]])
        else:
            assert w == b, (pos, t.itos[ids[pos]])
    q = ids.index(t.marker_id("QRY"))
    assert w4[q - 1] == 1.0                    # the <QRY> step (presence) is not weighted
    end = max(inside)
    assert ids[end] in (t.marker_id("FRG"), t.eos_id) and w4[end - 1] == 4.0


def test_q_default_is_unchanged_and_rows_without_a_query_are_untouched():
    t = tok()
    for url in ("https://a.example.com/x?id=12#top", "https://c.example.com/login",
                "https://d.example.com", "https://f.example.org/"):
        ids = t.encode(url, "benign", frame="page")
        assert field_mask(ids, t, "structure") == field_mask(ids, t, "structure", query_weight=1.0)
        if t.marker_id("QRY") not in ids:
            assert field_mask(ids, t, "structure", query_weight=4.0) == field_mask(ids, t, "structure")


def test_q_never_switches_a_position_on():
    """Without the frame token, the name corpus does not teach query content: those
    positions stay 0, and the query's end step is not weighted either."""
    t = URLTokenizer(include_fields=False).fit(FIT)
    ids = t.encode("https://a.example.com/x?id=12", "benign")
    assert field_mask(ids, t, "name", query_weight=4.0) == field_mask(ids, t, "name")
    s4 = field_mask(ids, t, "structure", query_weight=4.0)      # in scope there: weighted
    assert 4.0 in s4 and set(s4) <= {0.0, 1.0, 4.0}
    assert [w == 0.0 for w in s4] == [w == 0.0 for w in field_mask(ids, t, "structure")]


def test_q_encode_corpus_needs_field_masks():
    t = tok()
    with pytest.raises(ValueError):
        encode_corpus(["https://a.example.com/?q=1"], ["benign"], t, 128, query_weight=4.0)


# ---- Q: training ---------------------------------------------------------------------

URLS = ["https://a.example.com/x?id=1", "https://b.example.com/?q=2",
        "https://c.example.com/login", "https://d.example.com"] * 6
LABELS = ["phishing", "phishing", "benign", "benign"] * 6


def _train(t, tmp_path, qw, steps=6, **cfg):
    enc, _ = encode_corpus(URLS, LABELS, t, 128, corpora=["name"] * len(URLS),
                           query_weight=qw)
    data = URLDataset(enc)
    m = model_for(t)
    res = train(m, data, data, t, TrainConfig(steps=steps, batch_size=4, eval_every=steps,
                                              eval_batches=1, warmup=1, query_weight=qw, **cfg),
                device=torch.device("cpu"), out_dir=tmp_path / f"w{qw}", log_every=100)
    return m, res


def test_q_pins_the_loss_constant_under_the_fixed_objective(tmp_path):
    t = tok()
    with pytest.raises(ValueError):
        _train(t, tmp_path, 4.0, loss_norm="fixed")
    with pytest.raises(ValueError):
        _train(t, tmp_path, 0.0)


def test_q_same_batches_and_the_query_share_record(tmp_path):
    """V5's basis: the weight never touches the sampler, so the clip bands read on
    the w = 1 mass count the same steps as the unweighted run's."""
    t = tok()
    _, off = _train(t, tmp_path, 1.0, loss_norm="fixed", loss_constant=50.0)
    _, on = _train(t, tmp_path, 4.0, loss_norm="fixed", loss_constant=50.0)
    assert "query_weight" not in off
    steps = lambda r: {k: v["steps"] for k, v in r["grad_diagnostics"].items()}  # noqa: E731
    assert steps(on) == steps(off)
    q = on["query_weight"]
    assert q["w"] == 4.0 and q["weighted_mass"] > q["scored_positions"] > 0
    assert 0.0 < q["query_share"] < 1.0
    # exact: (weighted - scored) / (w - 1) is the number of query positions drawn
    assert abs(q["query_share"] * q["scored_positions"] * 3.0
               - (q["weighted_mass"] - q["scored_positions"])) < 1e-6
    assert on["loss_constant"] == off["loss_constant"] == 50.0


def test_q_checkpoint_records_the_weight_only_when_set(tmp_path):
    t = tok()
    m = model_for(t)
    save_checkpoint(m, t, TrainConfig(query_weight=4.0), tmp_path / "on.pt")
    save_checkpoint(m, t, TrainConfig(), tmp_path / "off.pt")
    on = torch.load(tmp_path / "on.pt", weights_only=False)["train_config"]
    off = torch.load(tmp_path / "off.pt", weights_only=False)["train_config"]
    assert on["query_weight"] == 4.0 and "query_weight" not in off
    load_checkpoint(tmp_path / "on.pt")
    load_checkpoint(tmp_path / "off.pt")


TINY_TRAIN = ["https://a.example.com/x?id=1", "https://b.example.org/?q=2",
              "https://c.example.net/login", "https://d.example.com",
              "https://e.example.com/p/q?r=1", "https://f.example.org/"]


def _write_bundle(d):
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
    return d


def test_q_through_kaggle_run_end_to_end(tmp_path):
    from uie.kaggle_train import run
    (tmp_path / "data").mkdir()
    data = _write_bundle(tmp_path / "data")
    kw = dict(steps=12, batch_size=4, d_model=16, n_layers=1, n_heads=2, eval_every=12,
              frame=True, fields=True, presence_plan=True, root_slash=True, slash_plan=True,
              seed_init=True, loss_norm="fixed", max_len=128)
    res = run(data, tmp_path / "on", loss_constant=100.0, query_weight=4.0, **kw)
    assert res["query_weight"]["w"] == 4.0 and res["loss_constant"] == 100.0
    assert res["query_weight"]["query_share"] > 0.0          # the weight reached the rows
    assert torch.load(tmp_path / "on" / "final.pt", weights_only=False)["train_config"][
        "query_weight"] == 4.0
    off = run(data, tmp_path / "off", loss_constant=100.0, **kw)
    assert off["query_weight"] is None
    # the same batches: the clip bands count the same steps
    assert ({k: v["steps"] for k, v in res["grad_diagnostics"].items()}
            == {k: v["steps"] for k, v in off["grad_diagnostics"].items()})
    with pytest.raises(ValueError):                      # C must be pinned
        run(data, tmp_path / "bad", query_weight=4.0, **kw)


# ---- D1': the slash rule on plans with a path ---------------------------------------------

P1_PLANS = [{"PTH": True, "QRY": False, "SLASH": True}, {"PTH": True, "QRY": True, "SLASH": True}]
P0_PLANS = [{"PTH": False, "QRY": False, "SLASH": True}, {"PTH": False, "QRY": True, "SLASH": True},
            {"PTH": False, "QRY": True, "SLASH": False}, {"PTH": False, "QRY": False, "SLASH": False}]


def test_d1p_gives_p1_plans_d1s_s1_rule_and_leaves_p0_plans_as_d1():
    t = tok()
    e = end_ids(t)
    for plan in P1_PLANS:
        assert slash_ends(t, plan) is None                        # D1 (E13) unchanged
        s = slash_ends(t, plan, paths=True)
        assert set(s["good"].tolist()) == {e["PTH"], e["RS"]}
        assert set(s["bad"].tolist()) == {e["QRY"], e["FRG"], e["EOS"]}
    for plan in P0_PLANS:
        a, b = slash_ends(t, plan), slash_ends(t, plan, paths=True)
        assert a["good"].tolist() == b["good"].tolist() and a["bad"].tolist() == b["bad"].tolist()


def _host_end(t, ids):
    lbl = t.marker_id("LBL")
    ends = set(end_ids(t).values())
    if lbl not in ids:
        return None
    k = ids.index(lbl)
    return next((x for x in ids[k + 1:] if x in ends), None)


@pytest.mark.parametrize("plan", P1_PLANS)
def test_d1p_every_p1_host_ends_with_a_slash(plan):
    t, m = tok(), model_for(tok())
    e = end_ids(t)
    ids, trace = [], []
    generate(m, t, label="phishing", n=32, suffix="com", seed=11, fields=plan, ids_out=ids,
             slash_rule=True, slash_paths=True, slash_trace=trace)
    seen = [x for x in (_host_end(t, row) for row in ids) if x is not None]
    assert len(seen) >= 5                                        # not vacuous
    assert all(x in (e["PTH"], e["RS"]) for x in seen)
    assert len(trace) == 32 and all(x >= 0.0 for x in trace)
    raw = []                                                     # the rule is what did it
    generate(m, t, label="phishing", n=32, suffix="com", seed=11, fields=plan, ids_out=raw)
    assert any(x not in (e["PTH"], e["RS"]) for x in (_host_end(t, r) for r in raw) if x is not None)


@pytest.mark.parametrize("temperature", [1.0, 0.8])
def test_d1p_p_honouring_can_only_rise(temperature):
    """At every host step of a P1 plan, <PTH>'s probability never falls and the
    host-end probability is unchanged: P honouring can only rise (section 2.2)."""
    import torch.nn.functional as F
    from uie.model import slash_redistribute
    t = tok()
    e = end_ids(t)
    s = slash_ends(t, P1_PLANS[1], paths=True)
    ends = list(e.values())
    g = torch.Generator().manual_seed(7)
    for _ in range(50):
        logits = torch.randn(t.vocab_size, generator=g) * 3
        before = F.softmax(logits / temperature, dim=-1)
        after = F.softmax(slash_redistribute(logits.clone(), s["good"], s["bad"],
                                             temperature)[0] / temperature, dim=-1)
        assert float(after[e["PTH"]]) >= float(before[e["PTH"]]) - 1e-7
        assert abs(float(after[ends].sum()) - float(before[ends].sum())) < 1e-6


def test_d1p_is_exactly_d1_on_p0_plans_and_off_by_default():
    t, m = tok(), model_for(tok())
    for plan in P0_PLANS:
        a = generate(m, t, label="benign", n=16, suffix="com", seed=5, fields=plan, slash_rule=True)
        b = generate(m, t, label="benign", n=16, suffix="com", seed=5, fields=plan, slash_rule=True,
                     slash_paths=True)
        assert a == b
    for plan in P1_PLANS:
        a = generate(m, t, label="phishing", n=16, suffix="com", seed=5, fields=plan, slash_rule=True)
        b = generate(m, t, label="phishing", n=16, suffix="com", seed=5, fields=plan,
                     slash_rule=True, slash_paths=False)
        c = generate(m, t, label="phishing", n=16, suffix="com", seed=5, fields=plan)
        assert a == b == c                                       # E13's D1 leaves P1 alone


def test_d1p_needs_the_slash_rule():
    t, m = tok(), model_for(tok())
    with pytest.raises(ValueError):
        generate(m, t, label="benign", n=2, suffix="com", seed=1, fields=P1_PLANS[0],
                 slash_paths=True)
