"""E17's library changes (research/e17/PREREG_E17.md sections 2, 3.2, V-B5, V3):
B5, the query-plan rule, in generation; and the injector's rate above 1. Both are off
by default, and off (or rate <= 1) leaves every earlier path exactly as it was.
"""
import json
import random

import pytest
import torch
import torch.nn.functional as F

from test_e13 import model_for, tok
from test_e15 import NEW, dataset, stream_file
from uie.model import generate, query_ends, query_redistribute
from uie.train import QueryInjector


def ended_rows_reaching_the_label(t, ids):
    lbl = t.marker_id("LBL")
    return [r for r in ids if lbl in r and r[-1] == t.eos_id]


# ---- B5: every finished Q = 1 row has its query -------------------------------------------

@pytest.mark.parametrize("plan", [{"PTH": False, "QRY": True, "SLASH": False},
                                  {"PTH": False, "QRY": True, "SLASH": True},
                                  {"PTH": True, "QRY": True, "SLASH": True}])
@pytest.mark.parametrize("with_d1", [False, True])
def test_b5_a_q1_plan_never_ends_before_its_query(plan, with_d1):
    t, m = tok(), model_for(tok())
    d1 = {"slash_rule": True, "slash_paths": True} if with_d1 else {}
    ids, trace = [], []
    generate(m, t, label="benign", n=24, suffix="com", seed=11, fields=plan, ids_out=ids,
             query_rule=True, query_trace=trace, **d1)
    q = t.marker_id("QRY")
    done = ended_rows_reaching_the_label(t, ids)
    assert len(done) >= 5                                   # the test is not vacuous
    assert all(q in r for r in done)
    assert len(trace) == 24 and all(x >= 0.0 for x in trace)
    raw = []                                                # and the rule is what did it
    generate(m, t, label="benign", n=24, suffix="com", seed=11, fields=plan, ids_out=raw, **d1)
    assert any(q not in r for r in ended_rows_reaching_the_label(t, raw))


def test_b5_off_and_q0_plans_are_bit_identical():
    t, m = tok(), model_for(tok())
    p1 = {"PTH": False, "QRY": True, "SLASH": False}
    a = generate(m, t, label="benign", n=16, suffix="com", seed=5, fields=p1)
    b = generate(m, t, label="benign", n=16, suffix="com", seed=5, fields=p1, query_rule=False)
    assert a == b
    for p0 in ({"PTH": False, "QRY": False, "SLASH": True}, {"PTH": True, "QRY": False, "SLASH": True}):
        c = generate(m, t, label="phishing", n=16, suffix="com", seed=5, fields=p0,
                     slash_rule=True, slash_paths=True)
        d = generate(m, t, label="phishing", n=16, suffix="com", seed=5, fields=p0,
                     slash_rule=True, slash_paths=True, query_rule=True)
        assert c == d
    assert query_ends(t, {"PTH": False, "QRY": False, "SLASH": True}) is None


def test_b5_refuses_what_it_was_not_registered_for():
    t, m = tok(), model_for(tok())
    with pytest.raises(ValueError):            # no supplied plan
        generate(m, t, label="benign", n=2, suffix="com", seed=1, query_rule=True)
    with pytest.raises(ValueError):            # not the raw decode
        generate(m, t, label="benign", n=2, suffix="com", seed=1, decoder="top_k",
                 fields={"PTH": False, "QRY": True, "SLASH": False}, query_rule=True)
    with pytest.raises(ValueError):            # a trace without the rule
        generate(m, t, label="benign", n=2, suffix="com", seed=1, query_trace=[],
                 fields={"PTH": False, "QRY": True, "SLASH": False})


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_b5_redistribution_is_exact(temperature):
    t = tok()
    rule = query_ends(t, {"PTH": False, "QRY": True, "SLASH": False})
    torch.manual_seed(3)
    logits = torch.randn(t.vocab_size, dtype=torch.float64) * 2
    before = F.softmax(logits / temperature, dim=-1)
    after_logits, moved = query_redistribute(logits.clone(), rule["good"], rule["bad"], temperature)
    after = F.softmax(after_logits / temperature, dim=-1)
    good, bad = rule["good"].tolist(), rule["bad"].tolist()
    assert all(float(after[b]) == 0.0 for b in bad)
    assert abs(float(after[good].sum()) - float(before[good].sum() + before[bad].sum())) < 1e-12
    assert abs(moved - float(before[bad].sum())) < 1e-12
    others = [k for k in range(t.vocab_size) if k not in good + bad]
    assert torch.allclose(after[others], before[others], rtol=0, atol=1e-12)


def test_b5_guards_when_nothing_can_move():
    t = tok()
    rule = query_ends(t, {"PTH": False, "QRY": True, "SLASH": True})
    logits = torch.randn(t.vocab_size, dtype=torch.float64)
    logits[rule["bad"]] = float("-inf")                    # D1 already removed every end
    out, moved = query_redistribute(logits.clone(), rule["good"], rule["bad"])
    assert moved == 0.0 and torch.equal(out, logits)
    logits = torch.randn(t.vocab_size, dtype=torch.float64)
    logits[rule["good"]] = float("-inf")                   # <QRY> itself forbidden
    out, moved = query_redistribute(logits.clone(), rule["good"], rule["bad"])
    assert moved > 0.0 and torch.isinf(out[rule["bad"]]).all() and torch.isfinite(F.softmax(out, -1)).all()


# ---- the injector above rate 1 -------------------------------------------------------------

def test_rate_two_appends_two_new_rows_per_old_draw_without_a_coin(tmp_path):
    t = tok()
    data = dataset(t)
    inj = QueryInjector.load(stream_file(tmp_path, NEW[:1] * 12), t, data, 128, 7, 8)   # 12 new / 6 old
    assert inj.rate["benign_P0Q1"] == 2.0 and inj.p["benign_P0Q1"] == 1.0
    coin = inj._rate.getstate()
    chunk = list(data.by_class["benign"][:4])
    before = [(list(i), list(m)) for i, m in chunk]
    out = inj(chunk)
    n_old = sum(inj.stratum_of(i) == "benign_P0Q1" for i, _ in chunk)
    assert n_old > 0 and len(out) == len(chunk) + 2 * n_old
    assert [(list(i), list(m)) for i, m in out[:len(chunk)]] == before      # old rows untouched
    assert inj._rate.getstate() == coin                                      # no coin at rate 2
    rec = inj.record()["strata"]["benign_P0Q1"]
    assert rec["appended_rows"] == 2 * rec["old_rows_seen"] and rec["rate"] == 2.0


def test_a_fractional_rate_above_one_draws_one_coin_per_old_draw(tmp_path):
    t = tok()
    data = dataset(t)
    inj = QueryInjector.load(stream_file(tmp_path, NEW[:1] * 9), t, data, 128, 7, 8)    # 9 new / 6 old
    assert inj.rate["benign_P0Q1"] == 1.5
    rng = random.Random(8)                                   # the injector's coin seed
    chunk = list(data.by_class["benign"][:4])
    n_old = sum(inj.stratum_of(i) == "benign_P0Q1" for i, _ in chunk)
    total = 0
    for _ in range(200):
        out = inj(chunk)
        want = sum(1 + (rng.random() < 0.5) for _ in range(n_old))
        assert len(out) - len(chunk) == want                 # the same coin, the same outcome
        total += want
    assert 1.4 < total / (200 * n_old) < 1.6


def test_a_short_stratum_is_still_injected_at_n_new_over_n_old(tmp_path):
    t = tok()
    data = dataset(t)
    inj = QueryInjector.load(stream_file(tmp_path, NEW[:1] * 3), t, data, 128, 7, 8)    # 3 new / 6 old
    rng = random.Random(8)
    chunk = list(data.by_class["benign"][:4])
    n_old = sum(inj.stratum_of(i) == "benign_P0Q1" for i, _ in chunk)
    for _ in range(50):
        out = inj(chunk)
        assert len(out) - len(chunk) == sum(rng.random() < 0.5 for _ in range(n_old))
