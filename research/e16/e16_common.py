"""E16 -- constants and the calls shared by the Kaggle notebooks and the laptop tools
(research/e16/PREREG_E16.md sections 3, 8, 11.1).

    e16_cap   e15_new's recipe -- corpus2 + the stream, the same batches, steps,
              objective, learning rate and decode -- at d_model 512 (d_ff 2048), each
              batch in ACCUM_STEPS parts (gradient accumulation: the same batches, the
              same summed gradient)

Its 50 files are E12's jobs at E12's seeds, so every cell pairs with `e15_new`'s files
(data/e15/generate/gen), the official baseline (DECISION_2026-10-06.md).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "native", ROOT / "research" / "e13", ROOT / "research" / "e15"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e13_common as E13  # noqa: E402
import e15_common as E15  # noqa: E402
from e8_common import G, LABELS, N_FREE, STATES, n_ctrl, seed_ctrl, seed_free  # noqa: E402,F401

REGISTRATION = "research/e16/PREREG_E16.md"
ARM = "e16_cap"
BASELINE = "e15_new"
STREAM_FILE = E15.STREAM_FILE

# --- the arm (section 3) --------------------------------------------------------------------
D_MODEL = 512                     # d_ff = 4 x d_model = 2048, as uie.kaggle_train builds it
ACCUM_STEPS = 2                   # section 3: each batch in 2 parts ...
ACCUM_FALLBACK = 4                # ... or 4 if the worst-case memory check fails at 2 (section 10)
PARAMS = 25_473_024               # 1.771 x e15_new's 14,386,176
SEED = 20260803                   # TrainConfig().seed: the seeded start and the sampler

# --- e15_new, the baseline (DECISION_2026-10-06.md section 3) -------------------------------
E15_NEW_SHA256 = "99a04fef259bbdd72ee6e3cc86273fceb705f95292076ba70b09bbf812c28b90"
#: V-mass': the injector is deterministic given the same old rows and seeds, so e16_cap's
#: record must equal e15_new's exactly.
E15_NEW_INJECTION = {
    "benign_P0Q1": {"old_rows_seen": 24_600, "appended_rows": 24_600, "appended_query_positions": 585_977},
    "benign_P1Q1": {"old_rows_seen": 70_187, "appended_rows": 70_187, "appended_query_positions": 2_231_497},
}
V_BATCH_DIGEST = E15.V_BATCH_DIGEST
COUNTS = E15.E12_FINAL_COUNTS
DROPPED = E15.E12_FINAL_DROPPED
LOSS_CONSTANT = E15.E12_FINAL_LOSS_CONSTANT

# --- V-start': the width-512 start, recorded by the dry run before any training -----------
#: research/e16/e16_dryrun.py; the same procedure at width 384 reproduces e12_final's
#: recorded start (c7e71edf...) exactly, which the dry run checks first.
START: Dict[str, str] = {}
INIT_SHA256 = ""


def recipe_cap(stream: Path, accum_steps: int = ACCUM_STEPS) -> dict:
    """The `uie.kaggle_train.run` arguments for e16_cap: e15_new's recipe, unchanged,
    with the width and the accumulation. Amendment 1 (section 13.2) keeps the learning
    rate and warmup."""
    base = E15.recipe_new(stream)
    r = {**base, "d_model": D_MODEL, "accum_steps": accum_steps}
    changed = sorted(k for k in set(r) | set(base) if r.get(k) != base.get(k))
    if changed != ["accum_steps", "d_model"]:
        raise ValueError(f"e16_cap must differ from e15_new only in width and accumulation: {changed}")
    if accum_steps not in (ACCUM_STEPS, ACCUM_FALLBACK):
        raise ValueError(f"registered accumulation is {ACCUM_STEPS} (fallback {ACCUM_FALLBACK})")
    return r


def start_fingerprints(tokenizer, d_model: int = D_MODEL) -> Dict[str, str]:
    """The start `uie.kaggle_train._run` builds for this recipe at `d_model`: E12's start
    procedure (`new_model_e12`), the recipe's architecture, the seeded initialisation.
    Built on the CPU, as training builds it. Returns its fingerprints and the start's own
    sha256 (what `train` records as init_weights_sha256)."""
    from uie.free import weights_sha256
    from uie.kaggle_train import new_model_e12
    from uie.model import ModelConfig
    r = E15.recipe_new(Path(STREAM_FILE))
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=d_model, n_layers=r["n_layers"],
                      n_heads=r["n_heads"], d_ff=4 * d_model, struct_ids=(),
                      max_len=r["max_len"], frame_every=r.get("frame_every", False))
    model, prints = new_model_e12(cfg, tokenizer, SEED)
    return {"init_weights_sha256": weights_sha256(model),
            "params": sum(p.numel() for p in model.parameters()), **prints}


def decode_kwargs() -> dict:
    """The adopted decode: raw + D1 + D1' (as e15_new's)."""
    return E15.decode_kwargs()


#: FULL FREE and FULL CONTROL at E12's seeds with the adopted decode (E15's calls).
free_urls = E15.free_urls
control_urls = E15.control_urls


def jobs() -> List[dict]:
    """E12's 50 jobs for e16_cap, at E12's seeds (E13's job builder, unchanged)."""
    return E13.jobs((ARM,))
