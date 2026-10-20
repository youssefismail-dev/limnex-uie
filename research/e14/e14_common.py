"""E14 -- constants and the generation calls shared by the Kaggle notebooks and the
laptop tools, so every one of them makes exactly the same calls
(research/e14/PREREG_E14.md section 2).

    e14_q     e12_final's recipe + the query weight w = 4 (Q), raw decode
    e12_d1p   e12_final's weights, raw decode + D1 + D1' (the slash rule on every plan)
    e14_qd    e14_q's weights, raw decode + D1 + D1'

Every arm's generation is E12's supplied-plan procedure at E12's seeds
(`e13_common.draw_and_generate`, whose hook left at E12's decode is E12's procedure
byte for byte), so every cell pairs with e12_final's own files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "native", ROOT / "research" / "e13"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as E12  # noqa: E402
import e13_common as E13  # noqa: E402
from e8_common import G, LABELS, N_FREE, STATES, n_ctrl, seed_ctrl, seed_free  # noqa: E402,F401
from uie.suffixes import SuffixTable  # noqa: E402

REGISTRATION = "research/e14/PREREG_E14.md"
BASELINE = "e12_final"

# --- Q: the query weight (section 2.1) -----------------------------------------------
#: Fixed at registration from N's A2 (4x exposure). One value; no sweep, no retry.
QUERY_WEIGHT = 4.0

# --- the arms (section 2.4) ----------------------------------------------------------
ARMS: Dict[str, dict] = {
    "e14_q": {"weights": "e14_q", "slash_rule": False, "slash_paths": False},
    "e12_d1p": {"weights": "e12_final", "slash_rule": True, "slash_paths": True},
    "e14_qd": {"weights": "e14_q", "slash_rule": True, "slash_paths": True},
}
#: Each arm is read against (section 3): Q against e12_final (retrained), D1' against
#: e12_final (same weights), the combination against e14_q (same weights).
REFERENCE = {"e14_q": "e12_final", "e12_d1p": "e12_final", "e14_qd": "e14_q"}
DECODE_ARMS = ("e12_d1p",)                  # E14_decode: needs no training
GENERATE_ARMS = ("e14_q", "e14_qd")         # E14_generate: after the laptop verifies e14_q
#: K14-1's target cells: the query cells e12_final does not clear. (mode, label, state)
TARGET_CELLS = {"free phishing QRY": ("free", "phishing", None),
                "control benign_P0Q1 QRY": ("control", "benign", "P0Q1"),
                "control benign_P1Q1 QRY": ("control", "benign", "P1Q1")}

# --- e12_final's record, which Q must reproduce in everything but the weight (V5) ----
#: data/e12/train/PROVENANCE.json and e12_final/train_result.json
E12_FINAL_SHA256 = "968ef622d0a6e9ba3d237778635d4930e0bbed02d240a88c0796567f7d686a17"
E12_FINAL_INIT = E13.E12_FINAL_INIT
E12_FINAL_START = E13.E12_FINAL_START
E12_FINAL_COUNTS = {"benign": 2_551_259, "phishing": 238_707}
E12_FINAL_DROPPED = 3_851
E12_FINAL_LOSS_CONSTANT = 14397.885481960007
#: The clip-record bands' step counts: on the w = 1 mass they depend only on the
#: batches drawn, so equal counts show the same batches.
E12_FINAL_BAND_STEPS = {"all 0-30": 3_454, "all 30-60": 9_973, "all 60-120": 5_637,
                        "all 120-inf": 936}

# --- K14-2 (section 3.1): N2a's bootstrap seed and draws ------------------------------
BOOT_SEED = 20260806
BOOT_DRAWS = 1_000


def recipe_q() -> dict:
    """The `uie.kaggle_train.run` arguments for e14_q: e12_final's recipe (PREREG_E12
    D1-D3, D5), the loss constant pinned at e12_final's, plus the query weight."""
    r = {**E12.recipe(BASELINE), "query_weight": QUERY_WEIGHT}
    if r.get("loss_constant") != E12_FINAL_LOSS_CONSTANT or r.get("loss_norm") != "fixed":
        raise ValueError("e12_final's recipe no longer pins its loss constant")
    return r


# --- decode arguments, per arm --------------------------------------------------------

def state_kwargs(arm: str, slash_trace: Optional[list] = None) -> dict:
    """The full decode arguments for one call: the raw decode (E12's), plus D1 and D1'
    as the arm registers them. The same for every supplied state."""
    kw = dict(E12.decode())
    spec = ARMS[arm]
    if spec["slash_rule"]:
        kw["slash_rule"] = True
        if spec["slash_paths"]:
            kw["slash_paths"] = True
        if slash_trace is not None:
            kw["slash_trace"] = slash_trace
    return kw


def free_urls(model, tokenizer, arm: str, label: str, g: int, shipped: SuffixTable,
              presence_slash, *, device=None, n: int = N_FREE,
              plan_trace: Optional[list] = None,
              slash_trace: Optional[list] = None) -> List[str]:
    """FULL FREE at the canonical seeds, the six-state plan supplied (E12 D2)."""
    tables = E12.supplied_tables(presence_slash, shipped.counts, label)
    return E13.draw_and_generate(model, tokenizer, label, n, seed_free(g), tables,
                                 lambda k: state_kwargs(arm, slash_trace),
                                 device=device, plan_trace=plan_trace)


def control_urls(model, tokenizer, arm: str, label: str, state: str, g: int, state_tables,
                 presence_slash, *, device=None, n: Optional[int] = None,
                 plan_trace: Optional[list] = None,
                 slash_trace: Optional[list] = None) -> List[str]:
    """FULL CONTROL at E8's seeds: the request (P, Q), S supplied (E12 D2)."""
    n = n if n is not None else n_ctrl(label, state)
    tables = E12.control_tables(presence_slash, state_tables, label, state)
    return E13.draw_and_generate(model, tokenizer, label, n, seed_ctrl(label, state, g), tables,
                                 lambda k: state_kwargs(arm, slash_trace),
                                 device=device, plan_trace=plan_trace)


def jobs(arms=tuple(ARMS)) -> List[dict]:
    """Every file an arm's generation writes: E12's 50 per arm, at E12's seeds."""
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise KeyError(f"not an E14 arm: {sorted(unknown)}")
    return E13.jobs(tuple(arms))
