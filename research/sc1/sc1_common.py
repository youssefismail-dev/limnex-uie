"""SC1 -- the definitions every SC1 script and the Kaggle notebook share.

Registered in research/sc1/PREREG_SC1.md. The Kaggle code bundle ships this file
alone from research/, so CR1's state list is restated here rather than imported;
tests/test_sc1.py pins every restated value to research/cr1/cr1_score.py.
"""
from typing import Dict, List, Tuple

G, N_MAIN = 5, 3000                        # CR1's: generations x compliant URLs
MAX_ROUNDS = 100                           # HC2's registered redraw limit
DELTA = {"benign": 0.22, "phishing": 0.30}  # the adopted path-scoped EOS bias

#: CR1's 16 control states, in CR1's order: the index s fixes the seed block.
STATES = ([(label, {f: want}) for label in ("benign", "phishing")
           for f in ("PTH", "QRY") for want in (True, False)]
          + [(label, {"PTH": p, "QRY": q}) for label in ("benign", "phishing")
             for p in (True, False) for q in (True, False)])
#: Registered by CR1 as not measurable (too few held-out URLs); never generated.
NOT_MEASURABLE = {("phishing", (("PTH", True), ("QRY", True))),
                  ("phishing", (("PTH", False), ("QRY", True)))}


def tag_of(label: str, state: Dict[str, bool]) -> str:
    return label + "_" + "_".join(f"{f[0]}{int(v)}" for f, v in state.items())


#: (s, label, state) for the 14 measurable states, in CR1's order.
MEASURABLE: List[Tuple[int, str, Dict[str, bool]]] = [
    (s, label, state) for s, (label, state) in enumerate(STATES)
    if (label, tuple(sorted(state.items()))) not in NOT_MEASURABLE]

#: Section 3.4. CR1 used offsets 0 to about 1.0e8 inside each block of 1e9
#: (100 rounds x 1,000,003 plus the suffix groups); SC1's arms sit above it,
#: disjoint from CR1 and from each other.
ARMS = ("C", "T")
OFFSET = {"C": 300_000_000, "T": 600_000_000, "T_rep": 800_000_000}


def gen_seed(arm: str, s: int, g: int) -> int:
    return 20261001 + (100 * s + g) * 1_000_000_000 + OFFSET[arm]


def table_counts(tables: dict, arm: str, label: str,
                 state: Dict[str, bool]) -> Dict[str, int]:
    """The suffix counts an arm draws from: C the class marginal, T (and its
    replication) the requested state's."""
    if arm == "C":
        return tables["C"][label]
    return tables["T"][tag_of(label, state)][label]


#: Section 6, V3 and V4.
IDENTITY_STATES = ("benign_P1", "phishing_Q1", "benign_P1_Q0", "phishing_P0_Q0")
IDENTITY_MIN = 250
AGREEMENT_MIN = 0.99

#: The rows behind each table, as the design review counted them (section 2.3;
#: data/sc1/design_review/design_review.json). The builder must reproduce them.
REGISTERED_T_ROWS = {
    "benign_P1": 655215, "benign_P0": 250997, "benign_Q1": 57768,
    "benign_Q0": 848444, "phishing_P1": 75027, "phishing_P0": 117625,
    "phishing_Q1": 8272, "phishing_Q0": 184380, "benign_P1_Q1": 42862,
    "benign_P1_Q0": 612353, "benign_P0_Q1": 14906, "benign_P0_Q0": 236091,
    "phishing_P1_Q0": 69741, "phishing_P0_Q0": 114639}
REGISTERED_C_ROWS = {"benign": 906212, "phishing": 192652}
