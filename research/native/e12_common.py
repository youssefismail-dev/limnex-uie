"""E12 -- constants and the generation calls shared by the Kaggle notebooks and the
laptop tools, so every one of them makes exactly the same calls
(research/native/PREREG_E12.md sections 1-2).

Three arms from one start, trained in one session (QC1-9 read NO FRAME-LEAK, so
D4 is not used and PREREG_E12 D6's second arm set applies):

    e12_base    e11_plan's recipe exactly (D1): the control, and E11's replicate
    e12_final   + the slash plan (D2) + R1, max_len 384 (D3): the E3 candidate
    e12_no_r1   e12_final without R1 (leave-one-out)

All use the baseline's raw decode. Nothing is resampled, rejected or masked: a
supplied plan is control mode's native conditioning, fed the data's state for the
drawn suffix, exactly as the suffix itself is fed.
"""
from __future__ import annotations

import collections
import random
from typing import Dict, List, Mapping, Optional

import e11_common as E11
from e8_common import (G, LABELS, N_FREE, STATES, UNMEASURABLE, n_ctrl,  # noqa: F401
                       seed_ctrl, seed_free)
from uie.suffixes import SuffixTable, generate_with_suffixes

#: D1: e11_plan's recipe, with E12's start in place of E11's (the same start for
#: this arm: new_model_e12 adds no row to it).
BASE: Dict[str, object] = {**E11.recipe("e11_plan"), "shared_start": False, "e12_start": True}
SLASH_PLAN = {"slash_plan": True}          # D2
R1 = {"max_len": 384}                      # D3
ARMS: Dict[str, dict] = {
    "e12_base": {},
    "e12_final": {**SLASH_PLAN, **R1},
    "e12_no_r1": {**SLASH_PLAN},
}
CANDIDATE = "e12_final"
#: K12c / K12d: every arm is read against the same-session control.
REFERENCE = {"e12_final": "e12_base", "e12_no_r1": "e12_base"}
#: Leave-one-out: each lever's effect is the candidate against the arm without it.
LEAVE_ONE_OUT = {"R1": ("e12_final", "e12_no_r1")}
SLASH_ARMS = frozenset(a for a, r in ARMS.items() if r.get("slash_plan"))
#: The six presence-slash states, in the table's order (e12_targets.py).
PS_STATES = ("P0Q0S0", "P0Q0S1", "P0Q1S0", "P0Q1S1", "P1Q0S1", "P1Q1S1")
#: A supplied state's generation call: its seed offset, clear of generate's seed + i
#: (E11's, D6).
STATE_SEED = E11.STATE_SEED
#: E11's start (5aa980f7..., on Kaggle's torch 2.10), which every arm's start must
#: contain exactly (D5, V-id), and the E10 start inside it.
E11_START = "5aa980f7c6ada9dacaef3d9a03646c23973a57e9901d622388935c69a5050aa0"
E10_START = E11.E10_START


#: E11's vocabulary (e11_plan): the size E11's start was built for.
E11_VOCAB = 108


def recipe(arm: str) -> dict:
    """The `uie.kaggle_train.run` arguments for one arm."""
    return {**BASE, **ARMS[arm]}


def start_fingerprints(seed: int = 20260803) -> tuple:
    """D5, checked BEFORE any training: E10's and E11's start, rebuilt as E12's
    start builds them. The start depends only on the vocabulary size, `<RS>`'s
    index (fixed by the control tokens, which precede every character), the seed
    and the architecture, so a tokenizer without characters rebuilds it exactly.
    On a torch whose seeded initialisation differs, these differ from the recorded
    values, and the notebook stops before spending a session."""
    from uie.free import weights_sha256
    from uie.kaggle_train import new_model_shared
    from uie.model import ModelConfig
    from uie.tokenizer import URLTokenizer
    r = recipe("e12_base")
    tok = URLTokenizer(include_frame=True, include_fields=True, root_slash=True,
                       presence_plan=True)
    cfg = ModelConfig(vocab_size=E11_VOCAB, d_model=r["d_model"], n_layers=r["n_layers"],
                      n_heads=r["n_heads"], d_ff=4 * r["d_model"])
    model, e10 = new_model_shared(cfg, tok, seed)
    return e10, weights_sha256(model)


def decode() -> dict:
    """D7: the baseline's raw decode -- temperature 1.0 and nothing else."""
    return E11.decode()


def memory_preflight(device, batch_size: int = 256, vocab_size: Optional[int] = None,
                     **override) -> float:
    """D3, checked BEFORE any training: one training step's forward and backward on
    a full batch at the candidate's max_len (384), under training's autocast.
    Returns the peak memory in GB (0.0 on a CPU). If it does not fit, it raises, and
    the notebook stops before a session is spent."""
    import torch
    from uie.model import ModelConfig, URLTransformer
    r = {**recipe(CANDIDATE), **override}
    v = vocab_size or E11_VOCAB + 2
    cfg = ModelConfig(vocab_size=v, d_model=r["d_model"], n_layers=r["n_layers"],
                      n_heads=r["n_heads"], d_ff=4 * r["d_model"], max_len=r["max_len"])
    model = URLTransformer(cfg).to(device).train()
    x = torch.randint(4, v, (batch_size, cfg.max_len), device=device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    with torch.amp.autocast("cuda", enabled=cuda):
        loss = model.loss(x, 0)
    loss.backward()
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if cuda else 0.0
    del model, x, loss
    if cuda:
        torch.cuda.empty_cache()
    return peak


def fields_of(state: str) -> Dict[str, bool]:
    """A presence-slash state as a supplied plan, e.g. P0Q1S0 -> PTH, QRY, SLASH."""
    return {"PTH": state[1] == "1", "QRY": state[3] == "1", "SLASH": state[5] == "1"}


def supplied_tables(table: Mapping[str, Mapping[str, List[int]]],
                    shipped: Mapping[str, Mapping[str, float]], label: str
                    ) -> Dict[str, Dict[str, float]]:
    """D2, free mode: w_k(s) = shipped(s) * n(s, k) / n(s) for each presence-slash
    state k (E11's D6 with the six states)."""
    n = table[label]
    out: Dict[str, Dict[str, float]] = {k: {} for k in PS_STATES}
    for s, c in shipped[label].items():
        row = n.get(s)
        if not row:
            raise KeyError(f"{label} suffix {s!r} is in the shipped table but has no row")
        tot = sum(row)
        for i, k in enumerate(PS_STATES):
            if row[i]:
                out[k][s] = c * row[i] / tot
    return out


def _draw_and_generate(model, tokenizer, label: str, n: int, seed: int,
                       tables: Dict[str, Dict[str, float]], *, device=None,
                       plan_trace: Optional[list] = None) -> List[str]:
    """The supplied plan's procedure (E11 D6): the counts of each state from one
    multinomial draw at `seed`; each state generated by control mode's native
    call, its suffix drawn from its own table, at seed + STATE_SEED * (i + 1)
    (i = the state's index in `tables`' order); the URLs shuffled by the same
    generator."""
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
            seed=seed + STATE_SEED * (i + 1), device=device, fields=fields_of(k),
            plan_trace=plan_trace, **decode()))
    rng.shuffle(urls)
    return urls


def free_urls(model, tokenizer, arm: str, label: str, g: int, shipped: SuffixTable, *,
              presence=None, presence_slash=None, device=None, n: int = N_FREE,
              plan_trace: Optional[list] = None) -> List[str]:
    """FULL FREE at the canonical seeds, the plan supplied from the data: E11's D6
    (four states) for `e12_base`, the six presence-slash states for a slash arm."""
    if arm not in SLASH_ARMS:
        if presence is None:
            raise ValueError(f"{arm}'s free mode needs E11's presence table")
        return E11.free_urls_supplied(model, tokenizer, label, g, shipped, presence,
                                      device=device, n=n, plan_trace=plan_trace)
    if presence_slash is None:
        raise ValueError(f"{arm}'s free mode needs the presence-slash table (D2)")
    tables = supplied_tables(presence_slash, shipped.counts, label)
    return _draw_and_generate(model, tokenizer, label, n, seed_free(g), tables,
                              device=device, plan_trace=plan_trace)


def control_tables(presence_slash, state_tables, label: str, state: str
                   ) -> Dict[str, Dict[str, float]]:
    """D2, control mode: the state's suffix table (SC1's), split by S at each
    suffix's own rate within the state. P1 states carry S = 1 only."""
    base = state_tables.for_state(label, STATES[state]).counts[label]
    if state.startswith("P1"):
        return {f"{state}S1": dict(base)}
    i0, i1 = PS_STATES.index(f"{state}S0"), PS_STATES.index(f"{state}S1")
    out = {f"{state}S0": {}, f"{state}S1": {}}
    for s, c in base.items():
        row = presence_slash[label].get(s)
        if not row or not (row[i0] + row[i1]):
            raise KeyError(f"{label} {state} suffix {s!r} has no presence-slash rows")
        tot = row[i0] + row[i1]
        if row[i0]:
            out[f"{state}S0"][s] = c * row[i0] / tot
        if row[i1]:
            out[f"{state}S1"][s] = c * row[i1] / tot
    return {k: v for k, v in out.items() if v}


def control_urls(model, tokenizer, arm: str, label: str, state: str, g: int, state_tables, *,
                 presence_slash=None, device=None, n: Optional[int] = None,
                 plan_trace: Optional[list] = None) -> List[str]:
    """FULL CONTROL, native: E11's call for `e12_base`; for a slash arm, the request
    (P, Q) with S supplied from the data (D2), at E8's seeds."""
    n = n if n is not None else n_ctrl(label, state)
    if arm not in SLASH_ARMS:
        return E11.control_urls(model, tokenizer, label, state, g, state_tables,
                                device=device, n=n, plan_trace=plan_trace)
    if presence_slash is None:
        raise ValueError(f"{arm}'s control mode needs the presence-slash table (D2)")
    tables = control_tables(presence_slash, state_tables, label, state)
    return _draw_and_generate(model, tokenizer, label, n, seed_ctrl(label, state, g),
                              tables, device=device, plan_trace=plan_trace)


def jobs() -> List[dict]:
    """Every file the generation notebook writes (PREREG_E12 D7): 50 per arm."""
    out: List[dict] = []
    for arm in ARMS:
        for label in LABELS:
            for g in range(G):
                out.append({"file": f"free_{arm}_{label}_g{g}.json", "arm": arm,
                            "mode": "free", "label": label, "state": None, "g": g,
                            "n": N_FREE, "seed": seed_free(g), "plan": "supplied"})
    for arm in ARMS:
        for label in LABELS:
            for state in STATES:
                for g in range(G):
                    out.append({"file": f"ctrl_{arm}_{label}_{state}_g{g}.json", "arm": arm,
                                "mode": "control", "label": label, "state": state, "g": g,
                                "n": n_ctrl(label, state), "seed": seed_ctrl(label, state, g),
                                "plan": "requested" + ("+S supplied" if arm in SLASH_ARMS else "")})
    return out
