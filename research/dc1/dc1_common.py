"""DC1 -- constants and the settings -> generate() translation (PREREG_DC1.md).

Shared by the fit (laptop), the Kaggle notebook and the scorers, so the three
cannot disagree about what a setting means.
"""
from typing import Dict, Optional

#: Fields DC1 calibrates: the gate's generated fields (SUF is supplied).
FIELDS_FIT = ("LBL", "SUB", "PTH", "QRY")

#: FULL CONTROL states -- both presence controls explicit (section 1).
STATES: Dict[str, Dict[str, bool]] = {
    "P1Q1": {"PTH": True, "QRY": True},
    "P1Q0": {"PTH": True, "QRY": False},
    "P0Q1": {"PTH": False, "QRY": True},
    "P0Q0": {"PTH": False, "QRY": False},
}
MODES = ("free",) + tuple(STATES)

#: Joint states with enough held-out URLs for the gate (CR1's registration).
MEASURABLE = {"benign": ("P1Q1", "P1Q0", "P0Q1", "P0Q0"),
              "phishing": ("P1Q0", "P0Q0")}

#: R4's frozen path-scoped end-token bias; DC1 sits on top of it.
FROZEN_DELTA = {"benign": 0.22, "phishing": 0.30}

#: Registered bounds (section 3).
BOUNDS = {"T": (0.5, 2.0), "delta": (-3.0, 3.0), "w": (0.0, 2.0)}

#: Registered sample sizes (section 3).
N_FIT = {"free": 20_000, "state": 10_000}
N_VAL = 5_000
FIT_SEED = 20261126
MAX_POSITIONS = 200_000          # per (class, mode, field): a seeded subsample if more

E7_SHA256 = "916d03857d7d3db4101d1b106293a83905f4b317676ee7e144b018170cb36065"

# --- evaluation generation (section 4) ------------------------------------------------
G = 5
#: URLs per free generation: enough for QRY to reach min(1,200, pool / 3).
N_FREE = {"benign": 40_000, "phishing": 25_000}
N_CTRL = 3_000                    # per controlled generation, as SC1 arm T

#: CR1's state order (research/cr1/cr1_score.py): the joint states are 8-15.
_CR1_JOINT = {("benign", "P1Q1"): 8, ("benign", "P1Q0"): 9, ("benign", "P0Q1"): 10,
              ("benign", "P0Q0"): 11, ("phishing", "P1Q1"): 12, ("phishing", "P1Q0"): 13,
              ("phishing", "P0Q1"): 14, ("phishing", "P0Q0"): 15}


def seed_free(label: str, g: int) -> int:
    """Blocks s = 16 (benign), 17 (phishing): never used by CR1, SC1 or C3.
    Both free arms use the same seeds, so they pair."""
    s = 16 + ("benign", "phishing").index(label)
    return 20261001 + (100 * s + g) * 1_000_000_000


def seed_ctrl(label: str, state: str, g: int) -> int:
    """CR1's block for the state, offset 1.5e8: disjoint from CR1 (0-1e8), SC1 arm C
    (3e8), C3 (4.5e8), SC1 arm T (6e8) and its replication (8e8)."""
    return 20261001 + (100 * _CR1_JOINT[(label, state)] + g) * 1_000_000_000 + 150_000_000


def mode_fields(mode: str) -> Optional[Dict[str, bool]]:
    """The presence request a mode generates with: None is FULL FREE."""
    return None if mode == "free" else dict(STATES[mode])


def generation_kwargs(settings: dict, label: str, mode: str) -> dict:
    """The generate() arguments DC1 adds for (label, mode). Only fields whose
    settings passed the validation confirmation are applied (section 3)."""
    cell = settings["classes"][label][mode]
    temperature, stop, blend = {}, {}, {}
    for field, r in cell.items():
        if not r.get("applied"):
            continue
        temperature[field] = r["T"]
        stop[field] = r["delta"]
        if mode != "free" and r.get("w") is not None:
            blend[field] = r["w"]
    out = {}
    if temperature:
        out["field_temperature"] = temperature
    if stop:
        out["field_stop_bias"] = stop
    if blend:
        out["blend"] = blend
    return out
