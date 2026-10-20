"""E17 -- constants and the calls shared by the Kaggle notebooks and the laptop tools
(research/e17/PREREG_E17.md sections 2, 3; amendment 1).

    e16_b5   e16_cap's weights, decoded raw + D1 + D1' + B5 (the query-plan rule).
             Generation only; judged against e16_cap by the decoder rules (R-a').
    e17_a    e16_cap's recipe (width 512, 2 parts, E12's start at width 512) with
             stream 1 AND stream 2 pooled: two new rows per old stratum draw.
             Decoded with B5 if B5 is adopted, else raw + D1 + D1'.

Every arm's 50 files are E12's jobs at E12's seeds, so every cell pairs with
e16_cap's files (data/e16/generate/gen), the official baseline.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "native", ROOT / "research" / "e13", ROOT / "research" / "e15",
          ROOT / "research" / "e16"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as E12  # noqa: E402
import e13_common as E13  # noqa: E402
import e15_common as E15  # noqa: E402
import e16_common as E16  # noqa: E402
from e8_common import G, LABELS, N_FREE, STATES, n_ctrl, seed_ctrl, seed_free  # noqa: E402,F401
from uie.suffixes import SuffixTable  # noqa: E402

REGISTRATION = "research/e17/PREREG_E17.md"
BASELINE = "e16_cap"
DECODE_ARM = "e16_b5"
TRAIN_ARM = "e17_a"
STREAM_FILE = "qry_extra_s12.jsonl"          # streams 1 + 2 pooled, never gzipped

# --- e16_cap, the baseline (DECISION_2026-10-07.md section 3) ------------------------------
E16_CAP_SHA256 = "98b10b4663ab85ea4af4368b6a09ded1bd5e809d077aaf4f8f1fea013037d65f"
#: V-start: e16_cap's start as E16_train recorded it on Kaggle before training (torch 2.11);
#: e17_a's must be the same start (the same procedure, width and seed).
E16_START = {
    "init_weights_sha256": "caca010736daeea728c9138253bb851a8feb20ababfad74332cee6ab39f003e4",
    "e10_start_sha256": "6496fd815d88119cff3c8410959d314108db52bd62c096ab4cd98131c085bf12",
    "e11_start_sha256": "85188237c68530468c3b6a1f71a904ef1070d424839e1a05059974816b5c6212",
    "new_token_rows_sha256": "7a4028cf4fa3431d4b0cfec3f73a9298bb48e30ce3e8a9dd7f578bbaa6f60817",
    "new_position_rows_sha256": "b5edfc270f64f98f9cdd7a5bb850cf634d051ee2cbc24a6ef5a56a59a40d7198",
}

# --- the pooled stream (amendment 1) --------------------------------------------------------
STREAM2_DIR = ROOT / "data" / "e17" / "qry_extra2"
STREAM2_FILES = {"benign_P0Q1": ("benign_P0Q1.jsonl",
                                 "23c64419096a04d4a35042094320ee57ae9392d547e6a3a1fd3168f0e069b698"),
                 "benign_P1Q1": ("benign_P1Q1.jsonl",
                                 "a153e75eefd348b107d3a89e32a1292df12674aaa73510bba4a90c9bf7a6b1de")}
N_OLD = dict(E15.N_OLD)
N_NEW = {s: 2 * n for s, n in N_OLD.items()}            # stream 1 + stream 2
RATE = 2.0
#: V-mass: the old rows drawn are e16_cap's (the same batches), so the appended rows are
#: exactly twice them; the query positions are checked against P8 within 3 SE.
OLD_ROWS_SEEN = {"benign_P0Q1": 24_600, "benign_P1Q1": 70_187}
APPENDED = {s: 2 * n for s, n in OLD_ROWS_SEEN.items()}
APPENDED_PER_BATCH = 9.482
ADDED_QUERY_POSITIONS_PER_BATCH = 282.2
V_BATCH_DIGEST = E15.V_BATCH_DIGEST


def build_stream(out_path: Path) -> Dict[str, object]:
    """The pooled training stream: stream 1's files then stream 2's, each checked against
    its registered hash, each row reduced to what the injector reads (url, label,
    stratum, lang) and tagged with its stream (1 or 2) for the uptake report."""
    rows = []
    for stream, (folder, files) in ((1, (ROOT / "data" / "e15" / "qry_extra", E15.STRATUM_FILES)),
                                    (2, (STREAM2_DIR, STREAM2_FILES))):
        for stratum, (name, want) in files.items():
            data = (folder / name).read_bytes()
            got = hashlib.sha256(data).hexdigest()
            if got != want:
                raise ValueError(f"stream {stream}: {name} is not the registered file ({got[:12]})")
            for line in data.decode("utf-8").splitlines():
                r = json.loads(line)
                rows.append({"url": r["url"], "label": r["label"], "stratum": r["stratum"],
                             "lang": r.get("lang", "und"), "stream": stream})
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out_path.write_bytes(text.encode("utf-8"))
    counts: Dict[str, int] = {}
    for r in rows:
        key = f"{r['label']}_{r['stratum']}" if "_" not in r["stratum"] else r["stratum"]
        counts[key] = counts.get(key, 0) + 1
    return {"rows": len(rows), "per_stratum": counts,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def recipe_a(stream: Path, accum_steps: int = E16.ACCUM_STEPS) -> dict:
    """The `uie.kaggle_train.run` arguments for e17_a: e16_cap's recipe, unchanged, with the
    pooled stream (rate 2 follows from its counts)."""
    base = E16.recipe_cap(Path(E15.STREAM_FILE), accum_steps)
    r = {**base, "qry_extra": str(stream)}
    changed = sorted(k for k in set(r) | set(base) if r.get(k) != base.get(k))
    if changed != ["qry_extra"]:
        raise ValueError(f"e17_a must differ from e16_cap only in its stream: {changed}")
    return r


def decode_kwargs(b5: bool) -> dict:
    """raw + D1 + D1' (e16_cap's), plus B5 when `b5`."""
    kw = E15.decode_kwargs()
    return {**kw, "query_rule": True} if b5 else kw


def _kwargs_for(b5: bool, query_trace: Optional[list]):
    kw = decode_kwargs(b5)
    if b5 and query_trace is not None:
        kw["query_trace"] = query_trace                 # B5's moved mass, per row
    return lambda k: kw


def free_urls(model, tokenizer, label: str, g: int, shipped: SuffixTable, presence_slash, *, b5: bool,
              device=None, n: int = N_FREE, plan_trace: Optional[list] = None,
              query_trace: Optional[list] = None) -> List[str]:
    """FULL FREE at the canonical seeds, the six-state plan supplied (E12 D2)."""
    tables = E12.supplied_tables(presence_slash, shipped.counts, label)
    return E13.draw_and_generate(model, tokenizer, label, n, seed_free(g), tables,
                                 _kwargs_for(b5, query_trace), device=device, plan_trace=plan_trace)


def control_urls(model, tokenizer, label: str, state: str, g: int, state_tables, presence_slash, *,
                 b5: bool, device=None, n: Optional[int] = None,
                 plan_trace: Optional[list] = None, query_trace: Optional[list] = None) -> List[str]:
    """FULL CONTROL at E8's seeds: the request (P, Q), S supplied (E12 D2)."""
    n = n if n is not None else n_ctrl(label, state)
    tables = E12.control_tables(presence_slash, state_tables, label, state)
    return E13.draw_and_generate(model, tokenizer, label, n, seed_ctrl(label, state, g), tables,
                                 _kwargs_for(b5, query_trace), device=device, plan_trace=plan_trace)


def jobs(arm: str) -> List[dict]:
    """E12's 50 jobs for one arm, at E12's seeds (E13's job builder, unchanged)."""
    return E13.jobs((arm,))
