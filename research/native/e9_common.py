"""E9 -- constants and the generation calls shared by the Kaggle notebooks and the
laptop tools, so every one of them makes exactly the same calls
(research/native/PREREG_E9.md sections 0, 2-3).

Two arms from one seeded start: `e9_base` (the objective every earlier run used)
and `e9_ln` (length-neutral). Free mode is E7's: presence controls shown as `?`.
Two decodes: RESTING (temperature 1.0 + the frozen PTH delta -- the decode of
every earlier result) and RAW (temperature 1.0, nothing else -- the native
candidate's decode). Nothing is resampled, rejected or masked.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from e8_common import (DELTA, G, LABELS, N_CTRL, N_CTRL_UNMEASURABLE, N_FREE,  # noqa: F401
                       STATES, UNMEASURABLE, n_ctrl, resting, seed_ctrl, seed_free)
from uie.suffixes import generate_with_suffixes

#: E[scored-token mass per drawn row] under E7's sampler, over-length rows
#: dropped (e9_evidence_signal_share.py, data/native/e9_signal_share.json).
EXPECTED_SCORED_TOKENS = 56.24174016390628
#: D3: C = 256 x that = 14,398 (registered, rounded).
LOSS_CONSTANT = 256 * EXPECTED_SCORED_TOKENS

#: E7's recipe, every argument stated, plus the seeded start (D1).
RECIPE = dict(steps=20_000, batch_size=256, lr=3e-4, d_model=384, n_layers=8, n_heads=8,
              eval_every=1_000, class_alpha=1.0, frame=True, factorized=False,
              boundary_weight=1.0, fields=True, field_dropout=0.5, seed_init=True)
ARMS: Dict[str, dict] = {
    "e9_base": {"loss_norm": "batch", "loss_constant": None},
    "e9_ln": {"loss_norm": "fixed", "loss_constant": LOSS_CONSTANT},
}
#: D5: (arm, decode) generated in each mode.
FREE_CONDITIONS = (("e9_base", "resting"), ("e9_base", "raw"),
                   ("e9_ln", "raw"), ("e9_ln", "resting"))
CTRL_CONDITIONS = (("e9_base", "resting"), ("e9_base", "raw"), ("e9_ln", "raw"))
DECODES = ("resting", "raw")


def decode(tokenizer, label: str, mode: str) -> dict:
    """RESTING = E8's `resting` (R4's temperature 1.0 + the frozen path delta);
    RAW = temperature 1.0 and nothing else."""
    if mode == "resting":
        return resting(tokenizer, label)
    if mode == "raw":
        return {"decoder": "temperature", "decoder_kwargs": {"temperature": 1.0}}
    raise ValueError(f"unknown decode {mode!r}")


def free_urls(model, tokenizer, label: str, g: int, shipped, mode: str, *,
              device=None, n: int = N_FREE) -> List[str]:
    """FULL FREE, E7's way: the shipped table's suffix, controls `?`, canonical seeds."""
    return generate_with_suffixes(model, tokenizer, label=label, n=n, table=shipped,
                                  seed=seed_free(g), device=device,
                                  **decode(tokenizer, label, mode))


def control_urls(model, tokenizer, label: str, state: str, g: int, state_tables, mode: str,
                 *, device=None, n: Optional[int] = None) -> List[str]:
    """FULL CONTROL, native: the state's suffix table, E7's prefix controls
    supplied, E8's seeds (so every file pairs with E8's E7 control files)."""
    return generate_with_suffixes(model, tokenizer, label=label,
                                  n=n if n is not None else n_ctrl(label, state),
                                  table=state_tables.for_state(label, STATES[state]),
                                  seed=seed_ctrl(label, state, g), device=device,
                                  fields=dict(STATES[state]), **decode(tokenizer, label, mode))


def jobs() -> List[dict]:
    """Every file the generation notebook writes (PREREG_E9 section 3): 160."""
    out: List[dict] = []
    for arm, mode in FREE_CONDITIONS:
        for label in LABELS:
            for g in range(G):
                out.append({"file": f"free_{arm}_{mode}_{label}_g{g}.json", "arm": arm,
                            "decode": mode, "mode": "free", "label": label, "state": None,
                            "g": g, "n": N_FREE, "seed": seed_free(g)})
    for arm, mode in CTRL_CONDITIONS:
        for label in LABELS:
            for state in STATES:
                for g in range(G):
                    out.append({"file": f"ctrl_{arm}_{mode}_{label}_{state}_g{g}.json",
                                "arm": arm, "decode": mode, "mode": "control", "label": label,
                                "state": state, "g": g, "n": n_ctrl(label, state),
                                "seed": seed_ctrl(label, state, g)})
    return out
