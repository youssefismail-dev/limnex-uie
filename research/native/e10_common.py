"""E10 -- constants and the generation calls shared by the Kaggle notebooks and the
laptop tools, so every one of them makes exactly the same calls
(research/native/PREREG_E10.md sections 1-2).

Two arms from one seeded start, trained in one session: `e10_base` is the adopted
baseline's recipe exactly (uie/baseline.py); `e10_plan` adds the presence plan --
the model writes its own plan after the suffix in free mode, the plan is supplied
in control mode -- and, because the plan replaces the hidden prefix controls, has
no control dropout. Both use the baseline's raw decode. Nothing is resampled,
rejected or masked.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from e8_common import (G, LABELS, N_FREE, STATES, UNMEASURABLE, n_ctrl,  # noqa: F401
                       seed_ctrl, seed_free)
from uie.baseline import E9_LN as BASELINE  # this arm is built on e9_ln, not the adopted engine
from uie.suffixes import generate_with_suffixes

#: D2: the baseline recipe, exactly.
BASE: Dict[str, object] = dict(BASELINE.recipe)
#: D3: the one change, and the dropout the plan implies.
ARMS: Dict[str, dict] = {
    "e10_base": {},
    "e10_plan": {"presence_plan": True, "field_dropout": 0.0},
}


def recipe(arm: str) -> dict:
    """The `uie.kaggle_train.run` arguments for one arm."""
    return {**BASE, **ARMS[arm]}


def decode() -> dict:
    """D4: the baseline's raw decode -- temperature 1.0 and nothing else."""
    return BASELINE.decode_kwargs()


def free_urls(model, tokenizer, label: str, g: int, shipped, *, device=None,
              n: int = N_FREE, plan_trace: Optional[list] = None) -> List[str]:
    """FULL FREE: the shipped table's suffix, canonical seeds; the baseline arm runs
    its controls `?`, the plan arm fills its own plan."""
    return generate_with_suffixes(model, tokenizer, label=label, n=n, table=shipped,
                                  seed=seed_free(g), device=device, plan_trace=plan_trace,
                                  **decode())


def control_urls(model, tokenizer, label: str, state: str, g: int, state_tables, *,
                 device=None, n: Optional[int] = None,
                 plan_trace: Optional[list] = None) -> List[str]:
    """FULL CONTROL, native: the state's suffix table, the request supplied (E7's
    prefix controls, or the plan), E8's seeds, nothing redrawn."""
    return generate_with_suffixes(model, tokenizer, label=label,
                                  n=n if n is not None else n_ctrl(label, state),
                                  table=state_tables.for_state(label, STATES[state]),
                                  seed=seed_ctrl(label, state, g), device=device,
                                  fields=dict(STATES[state]), plan_trace=plan_trace,
                                  **decode())


def jobs() -> List[dict]:
    """Every file the generation notebook writes (PREREG_E10 D5): 100."""
    out: List[dict] = []
    for arm in ARMS:
        for label in LABELS:
            for g in range(G):
                out.append({"file": f"free_{arm}_{label}_g{g}.json", "arm": arm,
                            "mode": "free", "label": label, "state": None, "g": g,
                            "n": N_FREE, "seed": seed_free(g)})
    for arm in ARMS:
        for label in LABELS:
            for state in STATES:
                for g in range(G):
                    out.append({"file": f"ctrl_{arm}_{label}_{state}_g{g}.json", "arm": arm,
                                "mode": "control", "label": label, "state": state, "g": g,
                                "n": n_ctrl(label, state), "seed": seed_ctrl(label, state, g)})
    return out
