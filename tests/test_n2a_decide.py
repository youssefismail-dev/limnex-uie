"""N2a's registered decision (research/native/n2a_decide.py; PREREG_N2A Amendment
2's table with Amendment 4's clarifications), on synthetic arm readings against
E7's RECORDED references, so every row, the copy guard, the power rule and the
phishing agreement are proven before any arm exists."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "native"))
sys.path.insert(0, str(ROOT / "research" / "dc1"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "n2a_decide_under_test", ROOT / "research" / "native" / "n2a_decide.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load()


def cell(x, se=0.003):
    return {"x": x, "se": se}


def m4(rate, n=7_000):
    return {"queries": n, "copied": round(rate * n)}


# E7's references as recorded in PREREG_N2A before any arm
E7 = {"benign": {"q_val": cell(0.0957, 0.0038), "q_train": cell(0.0862, 0.0034),
                 "m4": {"queries": 8_616, "copied": 2_355}},
      "phishing": {"q_val": cell(0.1141, 0.0079), "q_train": cell(0.1170, 0.0114),
                   "m4": {"queries": 2_774, "copied": 733}}}


def arms(a1_val=0.090, a1_train=0.082, a2_val=0.088, a2_train=0.080, a1_copy=0.2733,
         a2_copy=0.2733, se=0.003):
    return {"A1": {"q_val": cell(a1_val, se), "q_train": cell(a1_train, se), "m4": m4(a1_copy)},
            "A2": {"q_val": cell(a2_val, se), "q_train": cell(a2_train, se), "m4": m4(a2_copy)}}


def test_the_power_rule_reproduces_the_recorded_table():
    assert D.usable(E7["benign"]["q_val"]) and D.usable(E7["benign"]["q_train"])
    assert D.usable(E7["phishing"]["q_val"])
    assert not D.usable(E7["phishing"]["q_train"])          # recorded: void


def test_closes_needs_half_the_gap_and_a_ci_excluding_e7():
    e7 = E7["benign"]["q_val"]
    assert D.closes(cell(0.030), e7)                          # closure 0.69, CI tight
    assert not D.closes(cell(0.050), e7)                      # closure 0.48
    assert not D.closes(cell(0.045, se=0.02), e7)             # closure 0.53, CI reaches E7
    assert D.little(cell(0.075), e7) and not D.little(cell(0.070), e7)


def test_allocation():
    r = D.reading(E7["benign"], arms(a1_val=0.030, a2_val=0.030, a2_train=0.025))
    assert r["reading"] == "ALLOCATION"
    assert r["holds"] == ["ALLOCATION", "EXPOSURE"]           # every row that holds, reported


def test_rows_are_read_in_order_and_every_holding_row_is_reported():
    # incoherent on purpose: A2 closes Q-VAL yet is little on Q-TRAIN
    r = D.reading(E7["benign"], arms(a1_val=0.080, a2_val=0.030, a2_train=0.080))
    assert r["holds"] == ["EXPOSURE", "CAPACITY-OR-REPRESENTATION"]
    assert r["reading"] == "EXPOSURE"


def test_exposure_is_a2_closing_when_a1_does_not():
    assert D.reading(E7["benign"], arms(a1_val=0.080, a2_val=0.030))["reading"] == "EXPOSURE"


def test_sparsity_by_fit_without_generalisation():
    r = D.reading(E7["benign"], arms(a2_train=0.020, a2_val=0.085))
    assert r["reading"] == "SPARSITY"


def test_sparsity_by_copying_and_the_guard_voids_the_validation_closure():
    # A2 would close Q-VAL, but its copying rose: the closure is void and counts
    # only for SPARSITY
    r = D.reading(E7["benign"], arms(a2_val=0.030, a2_copy=0.31))
    assert r["guarded"]["A2"] and not r["guarded"]["A1"]
    assert "EXPOSURE" not in r["holds"]
    assert r["reading"] == "SPARSITY"


def test_a_copy_rise_under_half_a_point_is_not_a_rise():
    r = D.reading(E7["benign"], arms(a2_copy=0.2773))
    assert not r["guarded"]["A2"]


def test_capacity_or_representation():
    r = D.reading(E7["benign"], arms(a2_train=0.080, a2_val=0.088))
    assert r["reading"] == "CAPACITY-OR-REPRESENTATION"


def test_mixed():
    # A2 moves Q-TRAIN part of the way: neither closes nor little
    r = D.reading(E7["benign"], arms(a2_train=0.055, a2_val=0.088))
    assert r["holds"] == [] and r["reading"] == "MIXED"


def test_phishing_q_train_void_leaves_the_fit_rows_unavailable():
    r = D.reading(E7["phishing"], arms(a1_val=0.11, a2_val=0.11, a2_train=0.110))
    assert r["reading"] == "UNAVAILABLE"
    # as recorded: SPARSITY is unavailable for phishing, copying included; a rise
    # is still reported and still voids that arm's validation closure
    r = D.reading(E7["phishing"], arms(a2_val=0.030, a2_copy=0.31))
    assert r["guarded"]["A2"] and r["holds"] == [] and r["reading"] == "UNAVAILABLE"
    # its available rows still read
    assert D.reading(E7["phishing"], arms(a1_val=0.030))["reading"] == "ALLOCATION"


@pytest.mark.parametrize("benign, phishing, agreement", [
    ("ALLOCATION", "ALLOCATION", "AGREES"),
    ("ALLOCATION", "EXPOSURE", "DISAGREES"),
    ("CAPACITY-OR-REPRESENTATION", "UNAVAILABLE", "UNAVAILABLE"),
    ("SPARSITY", "ALLOCATION", "DISAGREES"),
])
def test_benign_decides_and_only_agreement_extends_to_phishing(benign, phishing, agreement):
    d = D.decision({"benign": {"reading": benign}, "phishing": {"reading": phishing}})
    assert d["reading"] == benign and d["phishing"] == agreement
    assert d["extends_to_phishing"] == (agreement == "AGREES")


def test_the_expected_schedule_is_the_arithmetic_on_the_registered_counts():
    import n2a_train as T
    for arm in ("A1", "A2"):
        assert T.schedule(T.EXPECTED_KEPT, arm) == D.EXPECTED_SCHEDULE[arm]
