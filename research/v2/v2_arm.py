"""
research/v2/v2_arm.py

Which V2 arm the laptop's reading scripts read (verify, fit, score, words, compare, diag):

    python research/v2/v2_p1_verify.py --step train                  # P1 (the default)
    V2_ARM=p1b python research/v2/v2_p1_verify.py --step train       # P1b

`P1` is the arm's common module (v2_p1_common or v2_p1b_common: the same names, the arm's
values), `DATA` its data folder, `CODE` the V2 files its GPU ran (V1: unchanged since the
bundle's commit), `TESTS` the unit tests VERIFY_TRAIN reruns, and `ISO` the commit uie/ must
equal (V-iso).
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

NAME = os.environ.get("V2_ARM", "p1")
_MODULES = {"p1": "v2_p1_common", "p1b": "v2_p1b_common", "p2_a": "v2_p2_common", "p2_b": "v2_p2_common"}
if NAME not in _MODULES:
    raise SystemExit(f"V2_ARM must be one of {sorted(_MODULES)}; got {NAME!r}")
P1 = importlib.import_module(_MODULES[NAME])
if NAME.startswith("p2"):
    P1 = P1.arm_view(NAME)                 # V2-P2: one module, two arms; ARM is the arm's name
#: V2-P2's dry runs read stand-in files from another root (V2_DATA); nothing else does
DATA = Path(os.environ["V2_DATA"]) / NAME if os.environ.get("V2_DATA") else ROOT / "data" / "v2" / NAME
_P2_CODE = ["research/v2/v2_bpe.py", "research/v2/v2_mem.py", "research/v2/v2_p1_common.py",
            "research/v2/v2_p2_common.py", "research/v2/v2_p2_tokenizer.py", "research/v2/v2_p2_train.py"]
CODE = {"p1": ["research/v2/v2_mem.py", "research/v2/v2_p1_common.py", "research/v2/v2_p1_train.py"],
        "p1b": ["research/v2/v2_mem.py", "research/v2/v2_p1_common.py", "research/v2/v2_p1b_common.py",
                "research/v2/v2_p1b_train.py"],
        "p2_a": _P2_CODE, "p2_b": _P2_CODE}[NAME]
TESTS = {"p1": ["tests/test_v2_mem.py", "tests/test_e16.py", "tests/test_e17.py"],
         "p1b": ["tests/test_v2_mem.py", "tests/test_v2_p1b.py", "tests/test_e16.py", "tests/test_e17.py"],
         "p2_a": ["tests/test_v2_p2.py", "tests/test_e16.py", "tests/test_e17.py"],
         "p2_b": ["tests/test_v2_p2.py", "tests/test_e16.py", "tests/test_e17.py"]}[NAME]
ISO = getattr(P1, "ISO_COMMIT", None) or P1.REGISTRATION_COMMIT
