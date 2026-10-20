"""E8 (N1) -- constants and the generation calls shared by the Kaggle notebook
and the laptop verifier, so both make exactly the same calls
(research/native/PREREG_E8.md section 4).

Every arm and reference decodes with the RESTING decode: R4's temperature 1.0
and the frozen path delta (0.22 benign / 0.30 phishing). FULL CONTROL is
native: the suffix is drawn once from SC1's state table, the plan (or E7's
prefix controls) is supplied, and nothing is resampled, rejected or masked.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from uie.suffixes import generate_with_suffixes

ARMS = {
    "e8_base": {"presence_plan": False, "root_slash": False},
    "e8_rs": {"presence_plan": False, "root_slash": True},
    "e8_pf": {"presence_plan": True, "root_slash": False},
    "e8_pfrs": {"presence_plan": True, "root_slash": True},
}
LABELS = ("benign", "phishing")
STATES: Dict[str, Dict[str, bool]] = {
    "P1Q1": {"PTH": True, "QRY": True},
    "P1Q0": {"PTH": True, "QRY": False},
    "P0Q1": {"PTH": False, "QRY": True},
    "P0Q0": {"PTH": False, "QRY": False},
}
#: registered not measurable (CR1; NATIVE_REPAIR_PLAN section 2): honouring only
UNMEASURABLE = {("phishing", "P1Q1"), ("phishing", "P0Q1")}
G = 5
N_FREE = 8_000                      # the canonical n_main
N_CTRL = 3_000                      # CR1's per-state n_main
N_CTRL_UNMEASURABLE = 1_000         # honouring only
DELTA = {"benign": 0.22, "phishing": 0.30}
E7_SHA256 = "916d03857d7d3db4101d1b106293a83905f4b317676ee7e144b018170cb36065"
# CR1's joint-state blocks (research/dc1/dc1_common.py _CR1_JOINT)
_CR1_JOINT = {("benign", "P1Q1"): 8, ("benign", "P1Q0"): 9, ("benign", "P0Q1"): 10,
              ("benign", "P0Q0"): 11, ("phishing", "P1Q1"): 12, ("phishing", "P1Q0"): 13,
              ("phishing", "P0Q1"): 14, ("phishing", "P0Q0"): 15}


def seed_free(g: int) -> int:
    """The canonical gate's generation seed (uie/corpus/run_eval.py): E7's own
    canonical files (HC1 C2) used exactly these, so every arm pairs with E7."""
    return 20260806 + 1 + g * 1000


def seed_ctrl(label: str, state: str, g: int) -> int:
    """CR1's block for the state, offset 9e8: unused by CR1 (0), SC1 (3e8, 6e8,
    8e8), C3 (4.5e8) and DC1 (1.5e8)."""
    return 20261001 + (100 * _CR1_JOINT[(label, state)] + g) * 1_000_000_000 + 900_000_000


def n_ctrl(label: str, state: str) -> int:
    return N_CTRL_UNMEASURABLE if (label, state) in UNMEASURABLE else N_CTRL


def resting(tokenizer, label: str) -> dict:
    """R4's decoder with the frozen path delta -- every arm's decode."""
    return {"decoder": "temperature", "decoder_kwargs": {"temperature": 1.0},
            "field_bias": {"PTH": {tokenizer.eos_id: -DELTA[label]}}}


def free_urls(model, tokenizer, label: str, g: int, shipped, *, device=None,
              n: int = N_FREE, plan_trace: Optional[list] = None) -> List[str]:
    """FULL FREE: the shipped table's suffix; E7-conditioned arms run `?`,
    plan arms fill their own plan."""
    return generate_with_suffixes(model, tokenizer, label=label, n=n, table=shipped,
                                  seed=seed_free(g), device=device,
                                  plan_trace=plan_trace, **resting(tokenizer, label))


def control_urls(model, tokenizer, label: str, state: str, g: int, state_tables, *,
                 device=None, n: Optional[int] = None,
                 plan_trace: Optional[list] = None) -> List[str]:
    """FULL CONTROL, native: the suffix drawn once from the state's table, the
    request supplied (E7's prefix controls, or the plan), no resampling."""
    return generate_with_suffixes(model, tokenizer, label=label,
                                  n=n if n is not None else n_ctrl(label, state),
                                  table=state_tables.for_state(label, STATES[state]),
                                  seed=seed_ctrl(label, state, g), device=device,
                                  fields=dict(STATES[state]), plan_trace=plan_trace,
                                  **resting(tokenizer, label))


def jobs() -> List[dict]:
    """Every file the generation notebook writes (PREREG_E8 section 4)."""
    out: List[dict] = []
    for arm in ARMS:
        for label in LABELS:
            for g in range(G):
                out.append({"file": f"free_{arm}_{label}_g{g}.json", "who": arm,
                            "mode": "free", "label": label, "state": None, "g": g})
    for who in tuple(ARMS) + ("e7",):
        for label in LABELS:
            for state in STATES:
                for g in range(G):
                    out.append({"file": f"ctrl_{who}_{label}_{state}_g{g}.json", "who": who,
                                "mode": "control", "label": label, "state": state, "g": g})
    for label in LABELS:
        for g in range(G):
            out.append({"file": f"free_e7dc1_{label}_g{g}.json", "who": "e7dc1",
                        "mode": "free", "label": label, "state": None, "g": g})
    return out
