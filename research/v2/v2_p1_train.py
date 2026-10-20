"""
research/v2/v2_p1_train.py

V2-P1 training (research/v2/PREREG_V2_P1_MEMORY.md section 3.3): `uie.kaggle_train.run`,
unchanged, with e16_cap's recipe. The one difference is the model it starts from.

`uie.kaggle_train._run` builds its start with `new_model_e12` (E12's procedure: e16_cap's
start at width 512). For the length of one call, that function is replaced by a wrapper
that calls it, then adds the memory (`v2_mem.with_memory`, seed 20261050). The original is
restored afterwards, even if the run fails. Everything else -- the tokenizer, the encoding,
the batches, the stream injector, accumulation, the fixed objective, the optimiser, the
evaluations, the checkpoints and the training record -- is `uie`'s own code, untouched.

The training record's `e12_start` block then carries, beside E12's four fingerprints,
e16_cap's part of the start (`base_init_sha256`), the memory's (`memory_init_sha256`) and
its seed, and `init_weights_sha256` is the whole start's.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import v2_p1_common as P1  # noqa: E402


def memory_builder(original):
    """`new_model_e12`'s signature and return shape, plus the memory."""
    from v2_mem import with_memory

    def build(cfg, tokenizer, seed):
        base, prints = original(cfg, tokenizer, seed)
        model, mprints = with_memory(base, P1.P1_SEED, tokenizer.pad_id)
        return model, {**prints, **mprints}
    return build


def run(data: Path, out: Path, **recipe) -> dict:
    """Train p1_mem. `recipe` must be e16_cap's (`v2_p1_common.recipe`), except `steps`,
    which only a dry run lowers."""
    import uie.kaggle_train as KT
    ref = P1.recipe(Path(recipe.get("qry_extra") or P1.STREAM_FILE), recipe.get("accum_steps", P1.ACCUM_STEPS))
    off = sorted(k for k in set(ref) | set(recipe) if k != "steps" and ref.get(k) != recipe.get(k))
    if off:
        raise ValueError(f"p1_mem trains with e16_cap's recipe; these differ: {off}")
    if not recipe.get("e12_start"):
        raise ValueError("p1_mem's start extends E12's start procedure (e12_start=True)")
    original = KT.new_model_e12
    KT.new_model_e12 = memory_builder(original)
    try:
        return KT.run(Path(data), Path(out), **recipe)
    finally:
        KT.new_model_e12 = original
