"""research/e13/e13_common.py (PREREG_E13 section 2.4, V2): with no lever on, E13's
generation is E12's supplied-plan procedure byte for byte, so each decoder arm
differs from e12_final only by D1 and D2; and the decode arguments are exactly
the ones each arm registers.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "research" / "e13", ROOT / "research" / "native"):
    sys.path.insert(0, str(p))

import e12_common as E12  # noqa: E402
import e13_common as E13  # noqa: E402
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
        b = E13.free_urls(m, t, "e13_ctx", label, 0, SHIPPED, PRESENCE_SLASH, n=40)
        assert a == b


def test_d1_arms_differ_from_raw_only_by_the_rule():
    t = tok()
    m = tiny(t)
    raw = E13.free_urls(m, t, "e13_ctx", "benign", 1, SHIPPED, PRESENCE_SLASH, n=60)
    trace = []
    d1 = E13.free_urls(m, t, "e12_d1", "benign", 1, SHIPPED, PRESENCE_SLASH, n=60,
                       slash_trace=trace)
    assert len(d1) == len(raw) == 60
    assert trace and all(x >= 0.0 for x in trace)       # the rule ran on P0 states


SETTINGS = {"scopes": {"D2-Q": {"applied": True, "T": 0.9, "delta": -0.4},
                       "D2-P": {"applied": False, "T": 1.2, "delta": 0.1}}}


def test_decode_arguments_are_exactly_each_arms():
    raw = E12.decode()
    assert E13.state_kwargs("e13_ctx", "benign", "P0Q1S1") == raw
    assert E13.state_kwargs("e12_d1", "benign", "P0Q1S1") == {**raw, "slash_rule": True}
    assert E13.state_kwargs("e12_dec", "benign", "P0Q1S0", SETTINGS) == {
        **raw, "slash_rule": True, "field_temperature": {"QRY": 0.9},
        "field_stop_bias": {"QRY": -0.4}}
    # D2-P was not confirmed on validation: no change for its state
    assert E13.state_kwargs("e12_dec", "benign", "P1Q1S1", SETTINGS) == {**raw, "slash_rule": True}
    # D2 is benign-only: phishing's P0Q1 gets D1 alone
    assert E13.state_kwargs("e12_dec", "phishing", "P0Q1S1", SETTINGS) == {**raw, "slash_rule": True}
    with pytest.raises(ValueError):
        E13.state_kwargs("e12_dec", "benign", "P0Q1S1")       # D2 needs its settings


def test_jobs_are_e12s_files_per_arm_and_n_uses_new_blocks():
    j = E13.jobs(("e12_d1",))
    e12 = [x for x in E12.jobs() if x["arm"] == "e12_final"]
    assert len(j) == len(e12) == 50
    for a, b in zip(sorted(j, key=lambda x: x["file"]), sorted(e12, key=lambda x: x["file"])):
        assert (a["seed"], a["n"], a["label"], a["state"], a["g"]) == \
               (b["seed"], b["n"], b["label"], b["state"], b["g"])
    seeds = {x["seed"] for x in E13.n_jobs()}
    used = {x["seed"] for x in E12.jobs()}
    assert not seeds & used and len(E13.n_jobs()) == 3 * 3 * 5


def test_n_schedule_is_the_registered_one_from_the_dry_runs_counts():
    sys.path.insert(0, str(ROOT / "research" / "e13"))
    import e13n_train as N
    labels = {"benign": 906_214, "phishing": 192_652}          # the bundle's name rows
    expo = N.exposure(labels)
    assert abs(expo["benign"] - 1.6550131253652298) < 1e-12
    assert abs(expo["phishing"] - 3.7603960872275604) < 1e-12
    for arm in ("A1", "A2"):
        assert N.schedule(N.EXPECTED_KEPT, expo, arm) == N.EXPECTED_SCHEDULE[arm]
    assert N.EXPECTED_SCHEDULE["A2"]["steps"] == 4 * N.EXPECTED_SCHEDULE["A1"]["steps"]


def test_n_free_cell_uses_only_query_bearing_states():
    tables = E13.n_tables("free_phishing_QRY", SHIPPED, PRESENCE_SLASH, None)
    assert tables and all(k[3] == "1" for k in tables)
    full = E12.supplied_tables(PRESENCE_SLASH, SHIPPED.counts, "phishing")
    assert {k: v for k, v in full.items() if k[3] == "1" and v} == tables
