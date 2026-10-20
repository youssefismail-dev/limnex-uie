"""
research/v2/v2_p1_common.py

V2-P1 (research/v2/PREREG_V2_P1_MEMORY.md) -- the constants and the calls shared by the
Kaggle notebooks and the laptop tools.

    p1_mem   e16_cap's network plus one product-key memory layer after block 4
             (research/v2/v2_mem.py), trained with e16_cap's recipe exactly -- corpus2 +
             stream 1, the same batches, steps, objective and optimiser, 2 parts (4 if the
             worst-case check fails at 2) -- from e16_cap's recorded start plus a memory
             drawn from seed 20261050. Decoded as e16_b5: raw + D1 + D1' + B5.

Its 50 files are E12's jobs at E12's seeds, so every cell pairs with e16_b5's files
(data/e17/decode/gen), the frozen baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "native", ROOT / "research" / "e13",
          ROOT / "research" / "e15", ROOT / "research" / "e16", ROOT / "research" / "e17"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e15_common as E15  # noqa: E402
import e16_common as E16  # noqa: E402
import e17_common as E17  # noqa: E402
from e8_common import G, LABELS, N_FREE, STATES, n_ctrl, seed_ctrl, seed_free  # noqa: E402,F401
from v2_mem import REGISTERED as MEM, memory_param_count  # noqa: E402

REGISTRATION = "research/v2/PREREG_V2_P1_MEMORY.md"
#: V-iso: uie/ must stay byte-identical to its state at the registration's commit.
REGISTRATION_COMMIT = "904a79af1235f0c0b1b7bbff1c9e22636e1c3e08"
ARM = "p1_mem"
BASELINE = E17.DECODE_ARM                      # e16_b5: e16_cap's weights + D1 + D1' + B5
CODE_MANIFEST = "V2P1_CODE_MANIFEST.json"
E16_CAP_RECORD = "e16_cap_record.json"         # e16_cap's tokenizer and settings, in the bundle

# --- the memory (section 3.1) and its start (section 3.2) -------------------------------
P1_SEED = 20261050                             # the memory's own generator
BOOT_SEED = 20261051                           # section 4.6's paired bootstrap
E16_CAP_PARAMS = E16.PARAMS                    # 25,473,024
MEM_PARAMS = memory_param_count(E16.D_MODEL, MEM["mem_n_keys"], MEM["mem_heads"], MEM["mem_k_dim"])
PARAMS = E16_CAP_PARAMS + MEM_PARAMS           # 161,795,072
assert MEM_PARAMS == 136_322_048 and PARAMS == 161_795_072, (MEM_PARAMS, PARAMS)

# --- e16_cap's record: what the run must reproduce (sections 3.2, 3.3, 7) -----------------
E16_CAP_SHA256 = E17.E16_CAP_SHA256
E16_START = dict(E17.E16_START)                # caca0107... and E12's fingerprints at width 512
STREAM_FILE = E15.STREAM_FILE                  # qry_extra.jsonl: stream 1, as e16_cap
STREAM_ROWS = sum(E15.N_NEW.values())          # 57,292
STREAM_SHA256 = "637accab5622f14aa432849fe9f4bc6ede7b2d18188bc3ef141d9d3c71e69e91"
V_BATCH_DIGEST = E16.V_BATCH_DIGEST            # adbddc2a...
INJECTION = {k: dict(v) for k, v in E16.E15_NEW_INJECTION.items()}   # = e16_cap's, exactly
COUNTS = dict(E16.COUNTS)
DROPPED = E16.DROPPED
LOSS_CONSTANT = E16.LOSS_CONSTANT
ACCUM_STEPS, ACCUM_FALLBACK = E16.ACCUM_STEPS, E16.ACCUM_FALLBACK
#: e16_cap's final validation loss (E16_train), printed beside p1_mem's
E16_CAP_FINAL_VAL = {"benign": 1.6067327976226806, "phishing": 1.3713488847017288,
                     "mean": 1.4890408411622047}


def recipe(stream: Path, accum_steps: int = ACCUM_STEPS) -> dict:
    """The `uie.kaggle_train.run` arguments: e16_cap's recipe, unchanged (section 3.3).
    The memory is not an argument -- `v2_p1_train.run` adds it to the start."""
    return E16.recipe_cap(stream, accum_steps)


def start(tokenizer) -> tuple:
    """Section 3.2's start, built on the CPU as training builds it: E12's procedure at
    width 512 (e16_cap's start), then the memory from P1_SEED. Returns the model and its
    fingerprints: E12's four, e16_cap's part, the memory's, and the whole start's."""
    from uie.free import weights_sha256
    from uie.kaggle_train import new_model_e12
    from uie.model import ModelConfig
    from v2_mem import with_memory
    r = recipe(Path(STREAM_FILE))
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=r["d_model"], n_layers=r["n_layers"],
                      n_heads=r["n_heads"], d_ff=4 * r["d_model"], struct_ids=(),
                      max_len=r["max_len"], frame_every=r.get("frame_every", False))
    base, prints = new_model_e12(cfg, tokenizer, E16.SEED)
    model, mprints = with_memory(base, P1_SEED, tokenizer.pad_id)
    return model, {**prints, **mprints, "init_weights_sha256": weights_sha256(model),
                   "params": sum(p.numel() for p in model.parameters())}


def decode_kwargs() -> dict:
    """e16_b5's decode: raw + D1 + D1' + B5 (section 3.4)."""
    return E17.decode_kwargs(True)


def free_urls(model, tokenizer, label: str, g: int, shipped, presence_slash, *, device=None,
              n: int = N_FREE, plan_trace: Optional[list] = None,
              query_trace: Optional[list] = None) -> List[str]:
    """FULL FREE at E12's seeds with e16_b5's decode (E17's call, B5 on)."""
    return E17.free_urls(model, tokenizer, label, g, shipped, presence_slash, b5=True, device=device,
                         n=n, plan_trace=plan_trace, query_trace=query_trace)


def control_urls(model, tokenizer, label: str, state: str, g: int, state_tables, presence_slash, *,
                 device=None, n: Optional[int] = None, plan_trace: Optional[list] = None,
                 query_trace: Optional[list] = None) -> List[str]:
    """FULL CONTROL at E8's seeds with e16_b5's decode (E17's call, B5 on)."""
    return E17.control_urls(model, tokenizer, label, state, g, state_tables, presence_slash, b5=True,
                            device=device, n=n, plan_trace=plan_trace, query_trace=query_trace)


def jobs() -> List[dict]:
    """E12's 50 jobs for p1_mem, at E12's seeds (E13's job builder, unchanged)."""
    return E17.jobs(ARM)


def e16_cap_record(blob: Dict) -> Dict:
    """What the bundle carries of e16_cap's checkpoint: its tokenizer and settings, for the
    start check (the tokenizer) and the readback (the settings). JSON-safe."""
    import json
    rec = {"tokenizer": blob["tokenizer"], "model_config": dict(blob["model_config"]),
           "train_config": dict(blob["train_config"])}
    return json.loads(json.dumps(rec))
