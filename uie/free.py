"""
uie/free.py

Free-mode generation with E7, as adopted from DC1 (research/dc1/PREREG_DC1.md,
research/dc1/DC1_RESULTS.md; adopted by the owner 2026-09-26 for FULL FREE only).

Superseded as the baseline on 2026-09-28 (research/native/DECISION_2026-09-28.md)
by E9's `e9_ln`, and again on 2026-10-08 (DECISION_2026-10-08.md) by `e16_b5`;
both live in `uie.baseline`, where `BASELINE` is the adopted engine. This module
is kept exactly as adopted -- bound to E7's weights -- so the published runs
reproduce.

Free mode is E7 with no presence control: `<P:?> <Q:?>`. A URL is drawn exactly
as DC1's free arm drew it:

  1. the suffix from the shipped class-marginal table, as the gate draws it
     (`uie.suffixes.generate_with_suffixes`, domain frame);
  2. R4's decoder, temperature 1.0, with the frozen path delta on the path's
     end (benign 0.22, phishing 0.30);
  3. on top, DC1's per-field temperature and stop bias, for the fields whose
     fit was confirmed on validation: benign LBL, PTH, QRY; phishing LBL, SUB,
     PTH.

Why. At full n, against today's E7 on the same seeds, DC1 cut the benign query
tell by a quarter (+0.0957 -> +0.0718) and the phishing path tell by a sixth
(+0.0534 -> +0.0446): no regression, whole-URL novelty held, fewer queries
copied from training (27.2% -> 22.5%), 1.01x the compute. It does not clear
the query cells, and DC1_RESULTS.md says so.

What does not change. `generate`, `generate_with_suffixes` and
`uie.controlled.generate_controlled` keep their defaults -- DC1's options act
only when passed -- so the gate and every published run stay reproducible.
Controlled generation is untouched: DC1 was not adopted for control mode.

The settings are E7's own. They were fitted to `best_e7.pt`'s logits, so
`generate_free` fingerprints the model's weights and refuses any other model.
It also refuses a tokenizer without presence controls: a pre-E7 checkpoint has
no free mode to calibrate.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import List, Mapping, Optional

from uie.suffixes import SuffixTable, generate_with_suffixes

__all__ = ["ADOPTED", "FreeDecoding", "generate_free", "weights_sha256"]


def weights_sha256(model) -> str:
    """A fingerprint of a model's weights: sha256 over every state-dict entry in
    name order -- its name, dtype, shape and raw bytes. The same on any device;
    a change to any weight changes it."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().to("cpu").contiguous()
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(a.numpy().tobytes())
    return h.hexdigest()


def _frozen(by_label: Mapping[str, Mapping[str, float]]):
    return MappingProxyType({label: MappingProxyType(dict(by_field))
                             for label, by_field in by_label.items()})


@dataclass(frozen=True)
class FreeDecoding:
    """What free-mode decoding adds to `generate`, per class.

    `delta` is the frozen path delta (a bias against ending inside PTH);
    `temperature` and `stop_bias` are per-field settings; `weights_sha256`, when
    set, is the only model they may be used with.
    """
    delta: Mapping[str, float]
    temperature: Mapping[str, Mapping[str, float]]
    stop_bias: Mapping[str, Mapping[str, float]]
    weights_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        # Read-only all the way down: a module-level setting must not be
        # changed by accident through a caller's reference to it.
        object.__setattr__(self, "delta", MappingProxyType(dict(self.delta)))
        object.__setattr__(self, "temperature", _frozen(self.temperature))
        object.__setattr__(self, "stop_bias", _frozen(self.stop_bias))

    def kwargs(self, tokenizer, label: str) -> dict:
        """The `generate` arguments for one class, exactly as DC1's free arm
        passed them."""
        if label not in self.delta:
            raise ValueError(f"no free-mode decoding for class {label!r}; "
                             f"known: {sorted(self.delta)}")
        out = {"decoder": "temperature", "decoder_kwargs": {"temperature": 1.0},
               "field_bias": {"PTH": {tokenizer.eos_id: -self.delta[label]}}}
        if self.temperature.get(label):
            out["field_temperature"] = dict(self.temperature[label])
        if self.stop_bias.get(label):
            out["field_stop_bias"] = dict(self.stop_bias[label])
        return out


#: DC1's free-mode settings: the fields confirmed on validation, exactly as
#: fitted (data/dc1/settings.json, sha256 68850b79...5f35, research/dc1/dc1_fit.py
#: at 6851edd). Bound to best_e7.pt (file sha256 916d0385...6065) by its weights.
ADOPTED = FreeDecoding(
    delta={"benign": 0.22, "phishing": 0.30},
    temperature={
        "benign": {"LBL": 1.0129424258915505, "PTH": 1.0314776116131796,
                   "QRY": 1.0615582282895621},
        "phishing": {"LBL": 1.0341995219485105, "SUB": 1.037142009388801,
                     "PTH": 1.0462373803669767},
    },
    stop_bias={
        "benign": {"LBL": -0.00012506979037082543, "PTH": -0.10689410663667626,
                   "QRY": -0.2180800080685031},
        "phishing": {"LBL": 0.02954524922409636, "SUB": -0.029603135469293065,
                     "PTH": 0.08129225750036788},
    },
    weights_sha256="ae6a758ae816e786628bd7539ef2feda842558da214a0977355b83f2b8fb1fcd",
)


def generate_free(model, tokenizer, *, label: str, n: int, table: SuffixTable,
                  seed: int = 20260806, decoding: FreeDecoding = ADOPTED,
                  batch_size: int = 32, device=None) -> List[str]:
    """`n` free-mode URLs of class `label`: DC1's free arm, exactly.

    `table` is the shipped suffix table the gate draws from
    (`SuffixTable.load("data/bundle2/suffix_table.json")`). Deterministic for a seed.
    Raises ValueError for a tokenizer without presence controls, an unknown
    class, or a model other than the one `decoding` was fitted to.
    """
    if not getattr(tokenizer, "include_fields", False):
        raise ValueError("free mode is E7's `<P:?> <Q:?>`; this tokenizer has no "
                         "presence controls (a checkpoint from before E7)")
    kwargs = decoding.kwargs(tokenizer, label)
    if decoding.weights_sha256 is not None:
        got = weights_sha256(model)
        if got != decoding.weights_sha256:
            raise ValueError(
                f"these free-mode settings were fitted to the model with weights "
                f"{decoding.weights_sha256[:12]}...; this one is {got[:12]}... "
                "They would be untested there, so they are refused.")
    return generate_with_suffixes(model, tokenizer, label=label, n=n, table=table,
                                  seed=seed, frame="domain", batch_size=batch_size,
                                  device=device, **kwargs)
