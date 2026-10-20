"""E15-B -- constants and the generation calls shared by the Kaggle notebooks and the
laptop tools (research/e15/PREREG_E15.md sections 10, 12, 13).

    e15_new   e12_final's recipe + the new benign query rows injected alongside the old
              ones (query-only targets, p = 1.0), decoded raw + D1 + D1'

Its 50 files are E12's jobs at E12's seeds, so every cell pairs with `e12_d1p`'s files
(E14's: e12_final + D1 + D1'), the frozen baseline.
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

REGISTRATION = "research/e15/PREREG_E15.md"
ARM = "e15_new"
BASELINE = "e12_d1p"                     # E14's files: e12_final + D1 + D1'
STREAM_FILE = "qry_extra.jsonl"          # plain JSON lines, never gzipped (Kaggle decompresses .gz)

# --- Amendment 3 (section 12), approved at gate B (section 13) -----------------------------
#: the admitted strata, as the injector counts them from the encoded training rows
N_NEW = {"benign_P0Q1": 14_863, "benign_P1Q1": 42_429}
N_OLD = {"benign_P0Q1": 14_863, "benign_P1Q1": 42_429}
P_INJECT = {"benign_P0Q1": 1.0, "benign_P1Q1": 1.0}
STRATUM_FILES = {"benign_P0Q1": ("benign_P0Q1.jsonl",
                                 "4ef1b466f1190e2705b07cfcd691dd48649bb9069e796be2a99987d78c30b746"),
                 "benign_P1Q1": ("benign_P1Q1.jsonl",
                                 "ce66a7db77b4fabb079a0500889c099e0daabf7f18c484461b7e70c7e77ee8cd")}
APPENDED_PER_BATCH = 4.741               # expected rows appended per batch (P8)
ADDED_QUERY_POSITIONS_PER_BATCH = 141.1  # expected (P8)
V_BATCH_DIGEST = "adbddc2ae4b85478395de46628cd7b5c1d1699511494ea6c84f836e19cb1d897"
DRAW_SEED, RATE_SEED = 20261007, 20261008

# --- e12_final's record (as E14's V5) ------------------------------------------------------
E12_FINAL_SHA256 = "968ef622d0a6e9ba3d237778635d4930e0bbed02d240a88c0796567f7d686a17"
E12_FINAL_INIT = E13.E12_FINAL_INIT
E12_FINAL_START = E13.E12_FINAL_START
E12_FINAL_COUNTS = {"benign": 2_551_259, "phishing": 238_707}
E12_FINAL_DROPPED = 3_851
E12_FINAL_LOSS_CONSTANT = 14397.885481960007


def recipe_new(stream: Path) -> dict:
    """The `uie.kaggle_train.run` arguments for e15_new: e12_final's recipe, unchanged,
    plus the stream. No query weight; the loss constant stays pinned."""
    r = {**E12.recipe("e12_final"), "qry_extra": str(stream)}
    if r.get("loss_constant") != E12_FINAL_LOSS_CONSTANT or r.get("loss_norm") != "fixed":
        raise ValueError("e12_final's recipe no longer pins its loss constant")
    if r.get("query_weight", 1.0) != 1.0:
        raise ValueError("E15-B uses no query loss weight")
    return r


def decode_kwargs() -> dict:
    """The adopted decode (DECISION_2026-10-05 section 2.5): raw + D1 + D1'."""
    return {**E12.decode(), "slash_rule": True, "slash_paths": True}


def free_urls(model, tokenizer, label: str, g: int, shipped: SuffixTable, presence_slash, *,
              device=None, n: int = N_FREE, plan_trace: Optional[list] = None) -> List[str]:
    """FULL FREE at the canonical seeds, the six-state plan supplied (E12 D2)."""
    tables = E12.supplied_tables(presence_slash, shipped.counts, label)
    return E13.draw_and_generate(model, tokenizer, label, n, seed_free(g), tables,
                                 lambda k: decode_kwargs(), device=device, plan_trace=plan_trace)


def control_urls(model, tokenizer, label: str, state: str, g: int, state_tables, presence_slash, *,
                 device=None, n: Optional[int] = None,
                 plan_trace: Optional[list] = None) -> List[str]:
    """FULL CONTROL at E8's seeds: the request (P, Q), S supplied (E12 D2)."""
    n = n if n is not None else n_ctrl(label, state)
    tables = E12.control_tables(presence_slash, state_tables, label, state)
    return E13.draw_and_generate(model, tokenizer, label, n, seed_ctrl(label, state, g), tables,
                                 lambda k: decode_kwargs(), device=device, plan_trace=plan_trace)


def jobs() -> List[dict]:
    """E12's 50 jobs for e15_new, at E12's seeds (E13's job builder, unchanged)."""
    return E13.jobs((ARM,))


def build_stream(src_dir: Path, out_path: Path) -> Dict[str, object]:
    """The training stream: the admitted strata's files, checked against Amendment 3's
    hashes, concatenated P0Q1 then P1Q1, each row reduced to what the injector reads."""
    import hashlib
    import json
    rows = []
    for stratum, (name, want) in STRATUM_FILES.items():
        data = (src_dir / name).read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise ValueError(f"{name} is not Amendment 3's file ({got[:12]} != {want[:12]})")
        for line in data.decode("utf-8").splitlines():
            r = json.loads(line)
            rows.append({"url": r["url"], "label": r["label"], "stratum": r["stratum"],
                         "lang": r.get("lang", "und")})
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out_path.write_bytes(text.encode("utf-8"))
    return {"rows": len(rows), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
