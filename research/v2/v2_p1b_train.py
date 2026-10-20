"""
research/v2/v2_p1b_train.py

V2-P1b training (research/v2/PREREG_V2_P1B.md): P1's runner (`uie.kaggle_train.run`,
unchanged, with e16_cap's recipe) with P1b's start builder.

The builder calls E12's start procedure, adds the memory with P1b's settings (the query
LayerNorm, no branch dropout, the values' own optimiser group via `mem_values_lr`) and then,
strictly before the first training step, runs T4's batch-coupling audit on the exact model
object it hands to training (synthetic rows of six lengths). If the audit fails it raises,
and nothing is trained. The original builder is restored afterwards, even on failure.

T2's optimiser group needs no runner code: `uie.train.train` reads the model's
`optimizer_groups` (its hook) and builds the values' group from the model's settings.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import v2_p1b_common as P1B  # noqa: E402

#: the in-run audit's synthetic rows: six lengths, as batches mix them (section 3.2)
AUDIT_SEED = 20261053
LAST_AUDIT: dict = {}


class AuditFailed(RuntimeError):
    pass


def synthetic_rows(vocab_size: int):
    import torch
    g = torch.Generator().manual_seed(AUDIT_SEED)
    return [torch.randint(4, vocab_size, (n,), generator=g).tolist() for n in P1B.AUDIT_LENGTHS]


def memory_builder(original):
    """`new_model_e12`'s signature and return shape: e16_cap's start plus P1b's memory, audited."""
    from v2_mem import with_memory

    def build(cfg, tokenizer, seed):
        base, prints = original(cfg, tokenizer, seed)
        model, mprints = with_memory(base, P1B.P1_SEED, tokenizer.pad_id, **P1B._settings())
        result = P1B.audit(model, synthetic_rows(tokenizer.vocab_size), tokenizer.pad_id)
        LAST_AUDIT.clear()
        LAST_AUDIT.update(result)
        print(f"T4 audit (in run, the exact model about to train): {result}", flush=True)
        if not result["pass"]:
            raise AuditFailed(f"STOP. The batch-coupling audit failed before training: {result}")
        return model, {**prints, **mprints, "audit_in_run": result}
    return build


def run(data: Path, out: Path, **recipe) -> dict:
    """Train p1b_mem. `recipe` must be e16_cap's (`v2_p1_common.recipe`), except `steps`,
    which only a dry run lowers."""
    import uie.kaggle_train as KT
    ref = P1B.recipe(Path(recipe.get("qry_extra") or P1B.STREAM_FILE), recipe.get("accum_steps", P1B.ACCUM_STEPS))
    off = sorted(k for k in set(ref) | set(recipe) if k != "steps" and ref.get(k) != recipe.get(k))
    if off:
        raise ValueError(f"p1b_mem trains with e16_cap's recipe; these differ: {off}")
    if not recipe.get("e12_start"):
        raise ValueError("p1b_mem's start extends E12's start procedure (e12_start=True)")
    original = KT.new_model_e12
    KT.new_model_e12 = memory_builder(original)
    try:
        return KT.run(Path(data), Path(out), **recipe)
    finally:
        KT.new_model_e12 = original
