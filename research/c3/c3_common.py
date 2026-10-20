"""C3-QRY -- the definitions the C3 scripts and the Kaggle notebook share.

Registered in research/c3/PREREG_C3_QRY.md. The Kaggle code bundle ships this
file, so nothing here imports from research/; tests/test_c3.py pins every value
restated from elsewhere (CR1's state order, SC1's delta and seed blocks, the
adopted tables' hash).
"""
from typing import Dict

G = 5                                        # generations per class
STATE = {"QRY": True}                        # arm F matches arm Q: the Q:1 state
DELTA = {"benign": 0.22, "phishing": 0.30}   # the adopted path-scoped EOS bias

#: Section 3: query-bearing URLs per generation -- what the QRY field is scored at
#: (min(1,200, real pool / 3): 1,200 benign; 356 phishing, the real-pool cap).
N = {"benign": 1200, "phishing": 400}
#: Round-0 draws per URL still needed; the batch doubles each round after.
M = {"benign": 40, "phishing": 80}
DOUBLING_CAP = 10                            # at most 2**10 x the round-0 batch
MAX_ROUNDS = 100                             # then the request fails (K1)
ROUND_STRIDE = 1_000_003                     # uie.presence's

#: CR1's state index of benign Q:1 and phishing Q:1 (its order) -- the seed blocks
#: arm Q's files were generated in; arm F sits at an offset of 4.5e8 inside them,
#: disjoint from CR1 (0), SC1's C (3e8), T (6e8) and T_rep (8e8).
STATE_INDEX = {"benign": 2, "phishing": 6}
OFFSET_F = 450_000_000


def tag(label: str) -> str:
    return f"{label}_Q1"


def gen_seed(label: str, g: int) -> int:
    return 20261001 + (100 * STATE_INDEX[label] + g) * 1_000_000_000 + OFFSET_F


#: The adopted state tables (uie/controlled.py, 3b3ae24): data/state_suffix_tables.json.
TABLES_SHA256 = "0c6f2760c033c532d271b360a14f2976139d5957e4aaf64977c8746504213cc9"

#: V3 and V4.
IDENTITY_MIN = 250
AGREEMENT_MIN = 0.99

#: V5: arm Q = SC1 arm T's Q:1 files and report, pinned by the registration.
Q_FILES_SHA256_PREFIX: Dict[str, str] = {
    "benign_Q1_g0.json": "f1f21d66cf3906ba", "benign_Q1_g1.json": "36901a2db7e81f89",
    "benign_Q1_g2.json": "b5136cbc037e097d", "benign_Q1_g3.json": "21a5e122b9e5210f",
    "benign_Q1_g4.json": "578c15dcb29743c0",
    "phishing_Q1_g0.json": "df87ea0b3e96bace", "phishing_Q1_g1.json": "dc74a3732b3b0a8f",
    "phishing_Q1_g2.json": "e11ce93f6a70d937", "phishing_Q1_g3.json": "5329dce57571d4e7",
    "phishing_Q1_g4.json": "451e9bef103838bd"}
T_REPORT_SHA256 = "7a9e6485164215a0abe569359c04042d6f987d01a2dd16ab755c9b576efef010"
