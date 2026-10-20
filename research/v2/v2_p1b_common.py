"""
research/v2/v2_p1b_common.py

V2-P1b (research/v2/PREREG_V2_P1B.md) -- the constants and calls shared by P1b's Kaggle
notebooks and the laptop tools. Everything is P1's (`v2_p1_common`) except what P1b changes:

    p1b_mem   P1's arm with the query LayerNorm (the fix), no dropout on the memory branch
              (T1) and the values' own Adam group, peak 1e-3, no weight decay (T2), plus the
              batch-coupling audit before the first training step (T4).

Its 50 files are E12's jobs at E12's seeds, paired with e16_b5's, as P1's were.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v2_p1_common import *  # noqa: E402,F401,F403  (P1's constants and calls, unchanged)
import v2_p1_common as _P1  # noqa: E402
from v2_mem import REGISTERED, memory_param_count  # noqa: E402

REGISTRATION = "research/v2/PREREG_V2_P1B.md"
#: V-iso (P1b): uie/ must equal the commit that adds T2's hook (422cb23, set in the commit after
#: it: a commit cannot name its own hash).
ISO_COMMIT: Optional[str] = "422cb230658c1149aff7c10d51f0fa0f9d795ef0"
ARM = "p1b_mem"
CODE_MANIFEST = "V2P1B_CODE_MANIFEST.json"
MEM = {**REGISTERED, "mem_query_norm": "layer", "mem_branch_dropout": False, "mem_values_lr": 1e-3}
MEM_PARAMS = memory_param_count(_P1.E16.D_MODEL, MEM["mem_n_keys"], MEM["mem_heads"], MEM["mem_k_dim"],
                                MEM["mem_query_norm"])
PARAMS = _P1.E16_CAP_PARAMS + MEM_PARAMS
assert MEM_PARAMS == 136_318_976 and PARAMS == 161_792_000, (MEM_PARAMS, PARAMS)
#: T4's bar (section 3.2): every gap under this many logits, no stored statistic changed
AUDIT_MAX_GAP = 0.05
#: T4 on Kaggle: real validation rows of these lengths (tokens), from the attached corpus
AUDIT_LENGTHS = (20, 40, 60, 90, 130, 200)


def start(tokenizer) -> tuple:
    """Section 2.1's start: E12's procedure at width 512 (e16_cap's start), then the memory
    from P1's seed with P1b's settings. Returns the model and its fingerprints."""
    from uie.free import weights_sha256
    from uie.kaggle_train import new_model_e12
    from uie.model import ModelConfig
    from v2_mem import with_memory
    r = recipe(Path(STREAM_FILE))
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=r["d_model"], n_layers=r["n_layers"],
                      n_heads=r["n_heads"], d_ff=4 * r["d_model"], struct_ids=(),
                      max_len=r["max_len"], frame_every=r.get("frame_every", False))
    base, prints = new_model_e12(cfg, tokenizer, _P1.E16.SEED)
    model, mprints = with_memory(base, P1_SEED, tokenizer.pad_id, **_settings())
    return model, {**prints, **mprints, "init_weights_sha256": weights_sha256(model),
                   "params": sum(p.numel() for p in model.parameters())}


def _settings() -> dict:
    return {k: v for k, v in MEM.items() if k not in REGISTERED or REGISTERED[k] != v}


def audit(model, rows, pad_id: int) -> dict:
    """T4: v2_mem.batch_coupling_check against section 3.2's bar."""
    from v2_mem import batch_coupling_check
    r = batch_coupling_check(model, rows, pad_id)
    gaps = [r["batch_vs_alone"], r["length_grouped_vs_alone"], r["training_mode_vs_alone"]]
    return {**r, "pass": max(gaps) < AUDIT_MAX_GAP and not r["buffers_changed_in_training"],
            "bar": AUDIT_MAX_GAP}


def audit_rows(tokenizer, urls) -> List[List[int]]:
    """Real rows near AUDIT_LENGTHS (page frame), each without its last token."""
    enc = [tokenizer.encode(u, "benign", frame="page") for u in urls]
    rows, used = [], set()
    for want in AUDIT_LENGTHS:
        best = min((i for i in range(len(enc)) if i not in used), key=lambda i: abs(len(enc[i]) - want))
        used.add(best)
        rows.append(enc[best][:-1])
    return rows


def jobs() -> List[dict]:
    """E12's 50 jobs for p1b_mem, at E12's seeds."""
    return _P1.E17.jobs(ARM)
