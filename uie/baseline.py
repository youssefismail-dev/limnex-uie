"""
uie/baseline.py

The adopted engine, and the earlier baseline the frozen experiments were built on.

`BASELINE` is `E16_B5`, adopted by the owner on 2026-10-08
(research/native/DECISION_2026-10-08.md), after E17 and the final E3 report
(research/E3_FINAL_REPORT.md):

  model      data/e16/train/e16_cap/final.pt, bound here by its weights
  recipe     E16's registered width-512 recipe (research/e16/PREREG_E16.md;
             `e16_common.recipe_cap`), trained with E15's query stream
             (sha256 637accab...) injected at rate 1
  decode     raw + D1 + D1' + B5 -- temperature 1.0, the slash rule, its path
             extension, and the query-plan rule

`E9_LN` is the previous baseline, adopted 2026-09-28
(research/native/DECISION_2026-09-28.md): E7's recipe trained with the
length-neutral objective from a seeded start, read with the RAW decode. It is
kept, unchanged and byte for byte, because the published E10, E11 and QG1
experiments derive their recipes and their decode from it. Repointing it would
silently redefine those arms, so it must not be touched.

Generation. `generate_free` and `generate_control` below are E9's own native
calls (research/native/e9_common.py, decode "raw"), and tests pin that. They
belong to `E9_LN` and default to it. **They cannot produce E16_B5's output**:
that engine writes to a SUPPLIED presence-slash plan, drawn per state, which
lives in the experiment layer (research/e17/e17_common.py `free_urls` and
`control_urls`, over research/native/e12_common.py). The library does not import
the experiment layer, so passing a rule-carrying baseline to these helpers is
refused rather than silently answered with the wrong decode.

What this does not change. `uie.free` (E7 + the frozen delta + DC1) stays bound
to E7's weights and reproduces the published runs. The library's defaults stay
as they were -- `TrainConfig.loss_norm` is still "batch" -- so every earlier
recipe re-runs byte for byte. Each recipe is spelled out here instead of reached
by changing a default.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import List, Mapping, Optional

from uie.free import weights_sha256
from uie.suffixes import SuffixTable, generate_with_suffixes

__all__ = ["BASELINE", "E9_LN", "E16_B5", "Baseline", "generate_control", "generate_free"]

#: PREREG_E9 D3: the expected scored-token mass per drawn row under E7's sampler.
EXPECTED_SCORED_TOKENS = 56.24174016390628
#: E15's query stream, injected into e16_cap's training (PREREG_E15 amendment 3).
E16_QRY_EXTRA_SHA256 = "637accab5622f14aa432849fe9f4bc6ede7b2d18188bc3ef141d9d3c71e69e91"


@dataclass(frozen=True)
class Baseline:
    """A model bound by its weights, the recipe that trained it, and its decode.

    `weights_sha256`, when set, is the only model the baseline may be used with;
    `file_sha256` names the checkpoint file it was read from. `recipe` holds the
    `uie.kaggle_train.run` arguments that trained it. `decode` holds the
    `uie.model.generate` arguments it is read with: `decoder` and
    `decoder_kwargs` always, plus any decode RULES the engine carries
    (`slash_rule`, `slash_paths`, `query_rule`). A rule forbids something real
    URLs never do and has no fitted parameter, so it is part of the engine's
    definition rather than a setting.
    """
    name: str
    weights_sha256: Optional[str]
    file_sha256: str
    recipe: Mapping[str, object]
    decode: Mapping[str, object]

    def __post_init__(self) -> None:
        # Read-only all the way down, as in uie.free: a module-level setting
        # must not be changed by accident through a caller's reference to it.
        # Every key is kept -- dropping an unrecognised one would quietly turn a
        # rule-carrying engine back into the raw decode.
        decode = dict(self.decode)
        decode["decoder_kwargs"] = MappingProxyType(dict(decode["decoder_kwargs"]))
        object.__setattr__(self, "recipe", MappingProxyType(dict(self.recipe)))
        object.__setattr__(self, "decode", MappingProxyType(decode))

    @property
    def rules(self) -> Mapping[str, object]:
        """The decode rules this engine carries, if any (empty for a raw decode)."""
        return {k: v for k, v in self.decode.items()
                if k not in ("decoder", "decoder_kwargs")}

    def decode_kwargs(self) -> dict:
        """The `generate` decoding arguments, as a fresh copy."""
        out = {k: v for k, v in self.decode.items() if k != "decoder_kwargs"}
        out["decoder_kwargs"] = dict(self.decode["decoder_kwargs"])
        return out

    def check_model(self, model) -> None:
        """Raise ValueError unless `model` is the baseline's model, by its weights."""
        if self.weights_sha256 is None:
            return
        got = weights_sha256(model)
        if got != self.weights_sha256:
            raise ValueError(
                f"the baseline is {self.name}, the model with weights "
                f"{self.weights_sha256[:12]}...; this one is {got[:12]}...")


#: The previous baseline (DECISION_2026-09-28). E10, E11 and QG1 are built on it;
#: its values must not change, or those published arms change with them.
E9_LN = Baseline(
    name="e9_ln",
    weights_sha256="7ac82c48d01a24de92c0ff7463fdf290bf5c1cd50312529eb2e765e86808a76f",
    file_sha256="60fcfb7dbe9aada599fde592d0732af307ac04f03a021025a206cf33babf257e",
    recipe=dict(steps=20_000, batch_size=256, lr=3e-4, d_model=384, n_layers=8, n_heads=8,
                eval_every=1_000, class_alpha=1.0, frame=True, factorized=False,
                boundary_weight=1.0, fields=True, field_dropout=0.5, seed_init=True,
                loss_norm="fixed", loss_constant=256 * EXPECTED_SCORED_TOKENS),
    decode={"decoder": "temperature", "decoder_kwargs": {"temperature": 1.0}},
)

#: The adopted engine (DECISION_2026-10-08): e16_cap's weights, decoded with
#: raw + D1 + D1' + B5. `qry_extra` names E15's stream by its registered file
#: name; its contents are pinned by E16_QRY_EXTRA_SHA256.
E16_B5 = Baseline(
    name="e16_b5",
    weights_sha256="14a27fc74630cf885e4702f2e34571237c9cd2cb5bc56dd3245d3c4530b301d4",
    file_sha256="98b10b4663ab85ea4af4368b6a09ded1bd5e809d077aaf4f8f1fea013037d65f",
    recipe=dict(steps=20_000, batch_size=256, lr=3e-4, d_model=512, n_layers=8, n_heads=8,
                eval_every=1_000, class_alpha=1.0, frame=True, factorized=False,
                boundary_weight=1.0, fields=True, field_dropout=0.0, seed_init=True,
                loss_norm="fixed", loss_constant=14397.885481960007,
                root_slash=True, shared_start=False, grad_clip_late=5.0,
                presence_plan=True, e12_start=True, slash_plan=True, max_len=384,
                qry_extra="qry_extra.jsonl", accum_steps=2),
    decode={"decoder": "temperature", "decoder_kwargs": {"temperature": 1.0},
            "slash_rule": True, "slash_paths": True, "query_rule": True},
)

#: The engine the project runs on.
BASELINE = E16_B5


def _require_controls(tokenizer) -> None:
    if not getattr(tokenizer, "include_fields", False):
        raise ValueError("these calls are for an E7-type model with presence controls; "
                         "this tokenizer has none")


def _require_native(baseline: Baseline) -> None:
    """These helpers implement E9's native calls. An engine carrying decode rules
    writes to a supplied plan drawn per state, which lives in the experiment layer,
    so answering here would return the right model under the wrong decode."""
    if baseline.rules:
        raise ValueError(
            f"{baseline.name} carries decode rules {sorted(baseline.rules)} and writes to a "
            "supplied presence-slash plan; these native helpers cannot reproduce it. "
            "Use research/e17/e17_common.py free_urls / control_urls.")


def generate_free(model, tokenizer, *, label: str, n: int, table: SuffixTable,
                  seed: int = 20260806, baseline: Baseline = E9_LN,
                  batch_size: int = 32, device=None) -> List[str]:
    """`n` free-mode URLs of class `label` from `baseline` (E9's `e9_ln`): the
    suffix from `table` (the shipped table), domain frame, controls `?`, the raw
    decode. Deterministic for a seed. Raises ValueError for any other model, and
    for an engine that carries decode rules."""
    _require_controls(tokenizer)
    _require_native(baseline)
    baseline.check_model(model)
    return generate_with_suffixes(model, tokenizer, label=label, n=n, table=table,
                                  seed=seed, frame="domain", batch_size=batch_size,
                                  device=device, **baseline.decode_kwargs())


def generate_control(model, tokenizer, *, label: str, n: int, fields: Mapping[str, bool],
                     state_tables, seed: int = 20260806, baseline: Baseline = E9_LN,
                     batch_size: int = 32, device=None) -> List[str]:
    """`n` control-mode URLs of class `label` for one joint state, natively: the
    suffix from the state's table (`uie.controlled.StateSuffixTables`), both
    controls supplied, the raw decode, and nothing redrawn. Raises ValueError for
    a partial state, any other model, or an engine that carries decode rules."""
    _require_controls(tokenizer)
    if set(fields) != {"PTH", "QRY"} or not all(isinstance(v, bool) for v in fields.values()):
        raise ValueError(f"a joint state sets PTH and QRY to True or False; got {dict(fields)}")
    _require_native(baseline)
    baseline.check_model(model)
    return generate_with_suffixes(model, tokenizer, label=label, n=n,
                                  table=state_tables.for_state(label, dict(fields)),
                                  seed=seed, frame="domain", fields=dict(fields),
                                  batch_size=batch_size, device=device,
                                  **baseline.decode_kwargs())
