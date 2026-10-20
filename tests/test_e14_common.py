"""research/e14/e14_common.py (PREREG_E14 sections 2.1, 2.4, V2, V7): the raw arm is
E12's supplied-plan procedure byte for byte; D1' equals E13's D1 on every P0 plan, so
e12_d1p's P0-plan URLs are e12_d1's; the decode arguments and the recipe are exactly
the registered ones.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "research" / "e14", ROOT / "research" / "e13", ROOT / "research" / "native"):
    sys.path.insert(0, str(p))

import e12_common as E12  # noqa: E402
import e13_common as E13  # noqa: E402
import e14_common as E14  # noqa: E402
from test_e7_fields import FIT  # noqa: E402
from uie.model import ModelConfig, URLTransformer  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402
from uie.tokenizer import URLTokenizer  # noqa: E402


def tok():
    return URLTokenizer(include_frame=True, include_fields=True, root_slash=True,
                        presence_plan=True, slash_plan=True).fit(FIT)


def tiny(t):
    torch.manual_seed(0)
    return URLTransformer(ModelConfig(vocab_size=t.vocab_size, max_len=96, d_model=16,
                                      n_layers=1, n_heads=2, d_ff=32))


SHIPPED = SuffixTable({"benign": {"com": 3.0, "net": 1.0}, "phishing": {"com": 2.0, "net": 2.0}})
#: per suffix, counts of P0Q0S0, P0Q0S1, P0Q1S0, P0Q1S1, P1Q0S1, P1Q1S1
PRESENCE_SLASH = {"benign": {"com": [1, 2, 1, 2, 3, 3], "net": [0, 2, 1, 1, 2, 2]},
                  "phishing": {"com": [0, 2, 0, 2, 3, 3], "net": [0, 1, 0, 1, 2, 2]}}


def test_raw_arm_generation_is_e12s_procedure_byte_for_byte():
    t = tok()
    m = tiny(t)
    for label in ("benign", "phishing"):
        a = E12.free_urls(m, t, "e12_final", label, 0, SHIPPED, presence_slash=PRESENCE_SLASH,
                          n=40)
        b = E14.free_urls(m, t, "e14_q", label, 0, SHIPPED, PRESENCE_SLASH, n=40)
        assert a == b


def test_d1p_arm_equals_e13s_d1_on_every_p0_plan_url():
    """V7's basis: the supplied state of each position comes from the seed alone, and
    each state is its own generation call, so P0 positions match e12_d1's exactly."""
    from e12_score import state_positions
    t = tok()
    m = tiny(t)
    for label in ("benign", "phishing"):
        for g in (0, 1):
            d1 = E13.free_urls(m, t, "e12_d1", label, g, SHIPPED, PRESENCE_SLASH, n=60)
            trace = []
            d1p = E14.free_urls(m, t, "e12_d1p", label, g, SHIPPED, PRESENCE_SLASH, n=60,
                                slash_trace=trace)
            tables = E12.supplied_tables(PRESENCE_SLASH, SHIPPED.counts, label)
            states = state_positions(tables, 60, E14.seed_free(g))
            p0 = [i for i, k in enumerate(states) if k.startswith("P0")]
            assert p0 and all(d1[i] == d1p[i] for i in p0)
            assert trace and all(x >= 0.0 for x in trace)


def test_decode_arguments_are_exactly_each_arms():
    raw = E12.decode()
    assert E14.state_kwargs("e14_q") == raw
    assert E14.state_kwargs("e12_d1p") == {**raw, "slash_rule": True, "slash_paths": True}
    assert E14.state_kwargs("e14_qd") == E14.state_kwargs("e12_d1p")
    tr = []
    assert E14.state_kwargs("e14_qd", tr)["slash_trace"] is tr
    assert "slash_trace" not in E14.state_kwargs("e14_q", tr)


def test_recipe_is_e12_finals_plus_the_weight_only():
    r = E14.recipe_q()
    base = E12.recipe("e12_final")
    assert {k: v for k, v in r.items() if k != "query_weight"} == base
    assert r["query_weight"] == 4.0
    assert r["loss_norm"] == "fixed" and r["loss_constant"] == E14.E12_FINAL_LOSS_CONSTANT


def test_jobs_are_e12s_files_per_arm():
    j = E14.jobs(("e12_d1p",))
    assert len(j) == 50 and len({x["file"] for x in j}) == 50
    assert {x["seed"] for x in j if x["mode"] == "free"} == {E14.seed_free(g) for g in range(E14.G)}
    assert [x["seed"] for x in j] == [x["seed"] for x in E13.jobs(("e12_d1",))]
    assert len(E14.jobs(E14.GENERATE_ARMS)) == 100
    with pytest.raises(KeyError):
        E14.jobs(("e13_ctx",))


def test_the_recorded_e12_final_facts_match_its_files():
    """V5's constants are e12_final's own record, not transcriptions."""
    import json
    p = ROOT / "data" / "e12" / "train" / "e12_final" / "train_result.json"
    if not p.is_file():
        pytest.skip("e12_final's training record is not on this machine")
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["counts"] == E14.E12_FINAL_COUNTS
    assert r["dropped_over_max_len"] == E14.E12_FINAL_DROPPED
    assert r["loss_constant"] == E14.E12_FINAL_LOSS_CONSTANT
    assert r["init_weights_sha256"] == E14.E12_FINAL_INIT
    assert {k: v["steps"] for k, v in r["grad_diagnostics"].items()
            if k.startswith("all ")} == E14.E12_FINAL_BAND_STEPS
