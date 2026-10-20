"""E13 -- constants and the generation calls shared by the Kaggle notebooks and the
laptop tools, so every one of them makes exactly the same calls
(research/e13/PREREG_E13.md section 2).

    e13_ctx   e12_final's recipe + ContextSwap (C), raw decode
    e12_d1    e12_final's weights, raw decode + the slash rule (D1)
    e12_dec   e12_final's weights, raw decode + D1 + field calibration (D2)

Every arm's generation is E12's supplied-plan procedure (e12_common), at E12's
seeds, so every cell pairs with e12_final's own files. The only addition is a
per-state hook for the decode arguments: with it empty the procedure is E12's,
byte for byte (tests/test_e13_common.py; V2).

N (section 2.5) writes per-cell query sets from new seed blocks 20-22 for the
reference (e12_final) and N2a's two diagnostic arms.
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "native"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as E12  # noqa: E402
from e8_common import G, LABELS, N_FREE, STATES, n_ctrl, seed_ctrl, seed_free  # noqa: E402,F401
from uie.suffixes import SuffixTable, generate_with_suffixes  # noqa: E402

REGISTRATION = "research/e13/PREREG_E13.md"
BASELINE = "e12_final"

# --- C: ContextSwap (section 2.1) ----------------------------------------------------
CONTEXT_P = 0.5
CONTEXT_SEED = 20261009
PREFIX_K = 4
PREFIX_SEED = 20261006
#: Registered before any artifact existed: phishing domain-frame training rows that
#: carry a non-empty query (`name_train`), before the 384-token limit.
ELIGIBLE_ROWS = 8_272
#: The frozen laptop artifact is gzipped; the bundle carries its exact decompressed
#: bytes, because Kaggle decompresses .gz files inside an uploaded dataset.
PREFIX_ARTIFACT = "prefixes.json.gz"
PREFIX_FILE = "prefixes.json"

# --- D2: field calibration (section 2.3) ---------------------------------------------
FIT_SEED_TRAIN, FIT_SEED_VAL = 20261007, 20261008
N_FIT, N_VAL, MAX_POSITIONS = 10_000, 5_000, 200_000
BOUNDS = {"T": (0.5, 2.0), "delta": (-3.0, 3.0)}
#: scope -> (class, the supplied presence-slash states, field)
D2_SCOPES = {"D2-Q": ("benign", ("P0Q1S0", "P0Q1S1"), "QRY"),
             "D2-P": ("benign", ("P1Q1S1",), "PTH")}
SETTINGS_FILE = "d2_settings.json"

# --- the generated arms (section 2.4) ------------------------------------------------
ARMS: Dict[str, dict] = {
    "e13_ctx": {"weights": "e13_ctx", "slash_rule": False, "d2": False},
    "e12_d1": {"weights": "e12_final", "slash_rule": True, "d2": False},
    "e12_dec": {"weights": "e12_final", "slash_rule": True, "d2": True},
}
#: Each arm is read against the files of the same weights without its lever
#: (section 3): C against e12_final, D1 against e12_final, D2 against e12_d1.
REFERENCE = {"e13_ctx": "e12_final", "e12_d1": "e12_final", "e12_dec": "e12_d1"}
DECODE_ARMS = ("e12_d1", "e12_dec")

# --- N: N2a re-measured (section 2.5) ------------------------------------------------
#: cell -> (class, state or None for free mode, seed block)
N_CELLS = {"control_benign_P0Q1_QRY": ("benign", "P0Q1", 20),
           "control_benign_P1Q1_QRY": ("benign", "P1Q1", 21),
           "free_phishing_QRY": ("phishing", None, 22)}
N_MODELS = ("e12_final", "A1", "A2")
N_FREE_PER_GEN = 1_500


#: e12_final's recorded start (data/e12/train/PROVENANCE.json): e13_ctx trains the same
#: recipe from the same start, so its initial weights must be these bytes (V-id).
E12_FINAL_INIT = "c7e71edf983ff6802cc76a1e6a05ae76b863f8a4d48c4f0e011741d6b6eafd26"
E12_FINAL_START = {"e10_start_sha256": "26926388bb575e2c9f70f7a611f762b3e53062f58f6638c70d13f0e46f213ad1",
                   "e11_start_sha256": "5aa980f7c6ada9dacaef3d9a03646c23973a57e9901d622388935c69a5050aa0",
                   "new_token_rows_sha256": "3743d8971e06f7757e0c13da59f22bf41096e2c85597026ec19bb5f1858f55dc",
                   "new_position_rows_sha256": "4fe9099f54ae0e7bf535b2183f1018859cba544365998232ac391fa4050b50db"}


def recipe_ctx(prefixes: Path) -> dict:
    """The `uie.kaggle_train.run` arguments for e13_ctx: e12_final's recipe
    (PREREG_E12 D1-D3, D5) plus ContextSwap."""
    return {**E12.recipe(BASELINE), "context_prefixes": str(prefixes), "context_p": CONTEXT_P}


def seed_n(block: int, g: int) -> int:
    """N's blocks s = 20, 21, 22, which no earlier experiment used."""
    return 20261001 + (100 * block + g) * 1_000_000_000


# --- decode arguments, per arm and supplied state ------------------------------------

def d2_kwargs(settings: Optional[dict], label: str, state: str) -> dict:
    """The generate() arguments D2 adds for (class, presence-slash state): only a
    scope whose fit passed the validation confirmation applies (section 2.3)."""
    if not settings:
        return {}
    for name, (cls, states, field) in D2_SCOPES.items():
        if label == cls and state in states:
            s = settings["scopes"][name]
            if s.get("applied"):
                return {"field_temperature": {field: s["T"]},
                        "field_stop_bias": {field: s["delta"]}}
    return {}


def state_kwargs(arm: str, label: str, state: str, settings: Optional[dict] = None,
                 slash_trace: Optional[list] = None) -> dict:
    """The full decode arguments for one supplied state's call: the raw decode
    (E12's), plus D1 and D2 as the arm registers them."""
    kw = dict(E12.decode())
    spec = ARMS[arm]
    if spec["slash_rule"]:
        kw["slash_rule"] = True
        if slash_trace is not None:
            kw["slash_trace"] = slash_trace
    if spec["d2"]:
        if settings is None:
            raise ValueError(f"{arm} needs the D2 settings")
        kw.update(d2_kwargs(settings, label, state))
    return kw


# --- generation: E12's supplied-plan procedure, with the per-state hook --------------

def draw_and_generate(model, tokenizer, label: str, n: int, seed: int,
                      tables: Dict[str, Dict[str, float]],
                      kwargs_for: Callable[[str], dict], *, device=None,
                      plan_trace: Optional[list] = None) -> List[str]:
    """`e12_common._draw_and_generate` exactly, except that each state's call takes
    `kwargs_for(state)` as its decode arguments (E12's is `E12.decode()`)."""
    states = [k for k in tables]
    rng = random.Random(seed)
    counts = collections.Counter(rng.choices(states, weights=[sum(tables[k].values())
                                                              for k in states], k=n))
    urls: List[str] = []
    for i, k in enumerate(states):
        if not counts[k]:
            continue
        urls.extend(generate_with_suffixes(
            model, tokenizer, label=label, n=counts[k], table=SuffixTable({label: tables[k]}),
            seed=seed + E12.STATE_SEED * (i + 1), device=device, fields=E12.fields_of(k),
            plan_trace=plan_trace, **kwargs_for(k)))
    rng.shuffle(urls)
    return urls


def free_urls(model, tokenizer, arm: str, label: str, g: int, shipped: SuffixTable,
              presence_slash, *, settings: Optional[dict] = None, device=None,
              n: int = N_FREE, plan_trace: Optional[list] = None,
              slash_trace: Optional[list] = None) -> List[str]:
    """FULL FREE at the canonical seeds, the six-state plan supplied (E12 D2)."""
    tables = E12.supplied_tables(presence_slash, shipped.counts, label)
    return draw_and_generate(model, tokenizer, label, n, seed_free(g), tables,
                             lambda k: state_kwargs(arm, label, k, settings, slash_trace),
                             device=device, plan_trace=plan_trace)


def control_urls(model, tokenizer, arm: str, label: str, state: str, g: int, state_tables,
                 presence_slash, *, settings: Optional[dict] = None, device=None,
                 n: Optional[int] = None, plan_trace: Optional[list] = None,
                 slash_trace: Optional[list] = None) -> List[str]:
    """FULL CONTROL at E8's seeds: the request (P, Q), S supplied (E12 D2)."""
    n = n if n is not None else n_ctrl(label, state)
    tables = E12.control_tables(presence_slash, state_tables, label, state)
    return draw_and_generate(model, tokenizer, label, n, seed_ctrl(label, state, g), tables,
                             lambda k: state_kwargs(arm, label, k, settings, slash_trace),
                             device=device, plan_trace=plan_trace)


def jobs(arms=tuple(ARMS)) -> List[dict]:
    """Every file an arm's generation writes: E12's 50 per arm, at E12's seeds."""
    out: List[dict] = []
    for arm in arms:
        for label in LABELS:
            for g in range(G):
                out.append({"file": f"free_{arm}_{label}_g{g}.json", "arm": arm, "mode": "free",
                            "label": label, "state": None, "g": g, "n": N_FREE,
                            "seed": seed_free(g), "plan": "supplied"})
    for arm in arms:
        for label in LABELS:
            for state in STATES:
                for g in range(G):
                    out.append({"file": f"ctrl_{arm}_{label}_{state}_g{g}.json", "arm": arm,
                                "mode": "control", "label": label, "state": state, "g": g,
                                "n": n_ctrl(label, state), "seed": seed_ctrl(label, state, g),
                                "plan": "requested+S supplied"})
    return out


# --- N: the per-cell query sets -------------------------------------------------------

def n_tables(cell: str, shipped: SuffixTable, presence_slash, state_tables
             ) -> Dict[str, Dict[str, float]]:
    """A cell's supplied-state tables: the control state with S supplied (E12 D2), or
    free mode's six states restricted to the query-bearing ones."""
    label, state, _ = N_CELLS[cell]
    if state is not None:
        return E12.control_tables(presence_slash, state_tables, label, state)
    tables = E12.supplied_tables(presence_slash, shipped.counts, label)
    return {k: v for k, v in tables.items() if k[3] == "1" and v}


def n_urls(model, tokenizer, cell: str, g: int, shipped: SuffixTable, presence_slash,
           state_tables, *, device=None, n: Optional[int] = None) -> List[str]:
    """One generation of N's query set for a cell, with the raw decode. `n` other
    than the registered size is for smoke tests only."""
    label, state, block = N_CELLS[cell]
    if n is None:
        n = N_FREE_PER_GEN if state is None else n_ctrl(label, state)
    tables = n_tables(cell, shipped, presence_slash, state_tables)
    return draw_and_generate(model, tokenizer, label, n, seed_n(block, g), tables,
                             lambda k: dict(E12.decode()), device=device)


def n_jobs(models=N_MODELS) -> List[dict]:
    out = []
    for m in models:
        for cell, (label, state, block) in N_CELLS.items():
            for g in range(G):
                out.append({"file": f"n_{m}_{cell}_g{g}.json", "model": m, "cell": cell,
                            "label": label, "state": state, "g": g,
                            "n": N_FREE_PER_GEN if state is None else n_ctrl(label, state),
                            "seed": seed_n(block, g)})
    return out


def load_settings(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
