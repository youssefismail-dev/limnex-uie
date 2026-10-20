"""
uie/model.py

E2 — the conditional autoregressive transformer.

Why autoregressive
------------------
It gives strong samples on short strings, trains stably, conditions trivially via
prefix tokens, and — decisively — yields an EXPLICIT LIKELIHOOD. Without
`log P(url)` there is no typicality signal, no principled quality metric and no
generative classifier: three quarters of the project's value. A GAN forfeits all
of that, and a latent-variable model risks posterior collapse on 30-character
strings (see DESIGN.md section 11.1).

Why it is small
---------------
The vocabulary is ~140 tokens: URLs are overwhelmingly percent-encoded and
punycode, so only ~110 distinct characters occur across 1.5M URLs (raw Unicode
does appear, but at 0.0002% of all characters). A character model over an
alphabet that small does not need scale, it needs the right structure — which
the tokenizer supplies.

Decoding is pluggable, deliberately
-----------------------------------
Sampling strategy is a RUNTIME choice, not an architectural one: strategies swap
without retraining a single weight. Freezing one into the model would confuse
"what the system is" with "how we happened to call it". The strategies live in
`DECODERS` and are benchmarked against one trained model in E3.

The reason decoding matters at all is that novelty is free and cliché is not.
The space of ~10-character labels is ~10^15 while the corpus is ~10^6, so a
well-fitted model produces unseen URLs essentially every time. What it will not
do by default is produce INTERESTING ones: maximum-likelihood training spreads
probability generously, and then low-temperature or top-k sampling sharpens back
toward the peak — which in URL space is `secure-login-verify`. The boring output
comes from the sampler, not the objective.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from uie.tokenizer import URLTokenizer

__all__ = ["ModelConfig", "URLTransformer", "DECODERS", "generate"]


@dataclass
class ModelConfig:
    """Architecture and the one sequence bound that is a data decision.

    `max_len=256` covers 99.0% of benign and 99.6% of phishing sequences.
    Sequences beyond it are DROPPED during training rather than truncated: a
    truncated URL teaches the model a URL that ends mid-path with no end token,
    which corrupts 1% of the data instead of merely losing it.
    """

    vocab_size: int = 100
    #: Token ids of the STRUCTURAL alphabet -- field markers and EOS. Empty
    #: means the classic single softmax over the whole vocabulary.
    #:
    #: When populated the output head FACTORISES into
    #:
    #:     P(token) = P(structural) * P(which structural)     if structural
    #:                (1 - P(structural)) * P(which char)     otherwise
    #:
    #: which is the fix for a mechanism measured on E5: EOS shares one softmax
    #: with 84 content characters, so whenever the model is uncertain about the
    #: next character the spread mass leaks into EOS. Binned by the model's own
    #: predictive entropy at positions where the path definitively continues:
    #:
    #:     entropy 0.0-0.5 nats -> EOS mass 0.111%
    #:     entropy 2.5-3.0 nats -> EOS mass 3.022%      (27x)
    #:
    #: URL paths run at ~1.69 nats/char, squarely in the leaky regime, and over
    #: a 28-character path a ~1.8% per-step leak compounds to 0.982**28 = 0.60
    #: survival -- a 40% chance of stopping early. A constant logit bias cannot
    #: correct it because the leak is PROPORTIONAL TO ENTROPY, not constant;
    #: measured, that correction fixed length only by inflating depth and host
    #: fields. Factorising removes the coupling by construction: character
    #: entropy lives entirely in the content head and cannot reach the gate.
    struct_ids: Tuple[int, ...] = ()
    max_len: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    #: E12 (research/native/PREREG_E12.md D4) -- the frame at every position: the
    #: frame token's own embedding (the token at `frame_index`) is added to the
    #: input at every position from it onward. No new parameters. Off by default,
    #: and then the forward pass is exactly what it was; a checkpoint writes these
    #: two fields only when on (uie.train.save_checkpoint).
    frame_every: bool = False
    frame_index: int = 2

    @property
    def approx_params(self) -> int:
        per_layer = 4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff
        return (self.vocab_size * self.d_model            # token embedding
                + self.max_len * self.d_model             # positions
                + self.n_layers * per_layer
                + self.d_model * self.vocab_size)         # output head


class _Block(nn.Module):
    """Pre-norm decoder block. Pre-norm because it trains stably without a
    learning-rate warmup schedule to babysit."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        shape = (b, t, self.n_heads, self.d_head)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        # is_causal handles the mask; SDPA picks a fused kernel where available.
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).reshape(b, t, c)
        x = x + self.drop(self.proj(att))
        return x + self.drop(self.ff(self.ln2(x)))


class URLTransformer(nn.Module):
    """Decoder-only transformer over the factorized token sequence."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: the input and output alphabets are identical, so two
        # separate tables would learn the same thing twice.
        self.head.weight = self.tok_emb.weight

        # Factorised head. `head` above is reused as the per-token scorer; the
        # gate is the only extra parameter, so the change costs d_model weights.
        self._struct = tuple(cfg.struct_ids)
        if self._struct:
            idx = torch.tensor(self._struct, dtype=torch.long)
            content = torch.tensor([i for i in range(cfg.vocab_size)
                                    if i not in set(self._struct)],
                                   dtype=torch.long)
            self.register_buffer("struct_idx", idx, persistent=False)
            self.register_buffer("content_idx", content, persistent=False)
            self.gate = nn.Linear(cfg.d_model, 1)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        _, t = ids.shape
        if t > self.cfg.max_len:
            raise ValueError(f"sequence of {t} exceeds max_len {self.cfg.max_len}")
        pos = torch.arange(t, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        if self.cfg.frame_every and t > self.cfg.frame_index:
            # E12 D4: the frame, read directly at every position from its own onward
            # (never before it, so no position sees a token that follows it).
            frame = self.tok_emb(ids[:, self.cfg.frame_index]).unsqueeze(1)
            after = (pos >= self.cfg.frame_index).to(x.dtype).view(1, t, 1)
            x = x + frame * after
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)
        logits = self.head(h)
        if not self._struct:
            return logits

        # Assemble full-vocabulary LOG-PROBABILITIES. Returning log-probs rather
        # than raw logits keeps every caller unchanged: softmax(log p) == p, so
        # decoders, `loss` and the probes all behave as before while the
        # factorisation is enforced underneath.
        #
        # FLOAT32 THROUGHOUT, explicitly. Under autocast `head` returns fp16
        # while `log_softmax` and `logsigmoid` are on the fp32 promotion list,
        # so an implicitly-typed buffer is Half and the sources are Float --
        # `index_copy_` then raises on the dtype mismatch. Assembling in fp32 is
        # also the numerically right choice: these are log-probabilities that
        # feed a loss autocast would compute in fp32 regardless.
        logits = logits.float()
        raw_gate = self.gate(h).float()
        gate = F.logsigmoid(raw_gate)                  # log P(structural)
        ngate = F.logsigmoid(-raw_gate)                # log P(content)
        out = torch.full_like(logits, float("-inf"))
        s_lp = F.log_softmax(logits.index_select(-1, self.struct_idx), dim=-1)
        c_lp = F.log_softmax(logits.index_select(-1, self.content_idx), dim=-1)
        out.index_copy_(-1, self.struct_idx, s_lp + gate)
        out.index_copy_(-1, self.content_idx, c_lp + ngate)
        return out

    def loss(self, ids: torch.Tensor, pad_id: int,
             weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Next-token cross-entropy, ignoring padding.

        `weights` is an optional per-TARGET mask, shape (batch, seq-1), used by
        field-scoped training: name-corpus rows supply gradient for the naming
        fields and structure-corpus rows for the path fields, because the two
        populations are sampled differently and neither is a valid teacher for
        the other's fields (see `uie.train.field_mask`).

        Weighted rows are averaged over their OWN weight mass, not over the
        batch: otherwise a row contributing eight scored tokens and one
        contributing eighty would be pooled as if equally informative, and the
        effective learning rate on a field would drift with whatever mix of
        rows a batch happened to contain.
        """
        logits = self(ids[:, :-1])
        targets = ids[:, 1:]
        if weights is None:
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=pad_id,
            )
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=pad_id, reduction="none",
        ).view(targets.shape)
        mask = weights * (targets != pad_id)
        total = mask.sum()
        if total == 0:
            return per_token.sum() * 0.0
        return (per_token * mask).sum() / total

    @torch.no_grad()
    def sequence_log_prob(self, ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        """log P(sequence) in nats, per row.

        Summed rather than averaged, so it is comparable with the n-gram
        baseline's `log_prob` and can be turned into per-character perplexity by
        the caller, which knows the URL lengths.
        """
        logits = self(ids[:, :-1])
        targets = ids[:, 1:]
        logp = F.log_softmax(logits, dim=-1)
        gathered = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return (gathered * (targets != pad_id)).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Decoding strategies — runtime choices, not architecture
# --------------------------------------------------------------------------- #

def _multinomial(probs: torch.Tensor, generator: torch.Generator) -> int:
    return int(torch.multinomial(probs, 1, generator=generator).item())


def decode_greedy(logits: torch.Tensor, generator: torch.Generator, **_) -> int:
    """Always the mode. Collapses immediately — included as the degenerate
    reference point, so 'diverse' has something to be measured against."""
    return int(torch.argmax(logits).item())


def decode_temperature(logits: torch.Tensor, generator: torch.Generator,
                       temperature: float = 1.0, **_) -> int:
    return _multinomial(F.softmax(logits / max(temperature, 1e-6), dim=-1),
                        generator)


def decode_top_k(logits: torch.Tensor, generator: torch.Generator,
                 k: int = 40, temperature: float = 1.0, **_) -> int:
    k = min(k, logits.numel())
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values / max(temperature, 1e-6), dim=-1)
    return int(indices[_multinomial(probs, generator)].item())


def decode_top_p(logits: torch.Tensor, generator: torch.Generator,
                 p: float = 0.9, temperature: float = 1.0, **_) -> int:
    """Nucleus sampling: the smallest set whose mass exceeds `p`."""
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    ordered, indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(ordered, dim=-1)
    keep = int(torch.searchsorted(cumulative, torch.tensor(p)).item()) + 1
    keep = min(max(keep, 1), ordered.numel())
    return int(indices[_multinomial(ordered[:keep] / ordered[:keep].sum(),
                                    generator)].item())


def decode_typical(logits: torch.Tensor, generator: torch.Generator,
                   mass: float = 0.95, temperature: float = 1.0, **_) -> int:
    """Locally typical sampling (Meister et al., TACL 2023).

    Keeps tokens whose information content sits NEAR the distribution's expected
    information content, rather than those with the highest probability. That
    avoids the high-probability cliché region and the low-probability garbage
    region simultaneously — temperature can only trade one against the other,
    which is why it is the initial default here.
    """
    logp = F.log_softmax(logits / max(temperature, 1e-6), dim=-1)
    probs = logp.exp()
    entropy = -(probs * logp).sum()
    deviation = (-logp - entropy).abs()
    _, order = torch.sort(deviation)                      # closest to typical
    cumulative = torch.cumsum(probs[order], dim=-1)
    keep = int(torch.searchsorted(cumulative, torch.tensor(mass)).item()) + 1
    keep = min(max(keep, 1), order.numel())
    chosen = order[:keep]
    return int(chosen[_multinomial(probs[chosen] / probs[chosen].sum(),
                                   generator)].item())


#: Registry. E3 benchmarks these against ONE trained model — nearly free, since
#: it is a single training run and N evaluations.
DECODERS: Dict[str, Callable[..., int]] = {
    "greedy": decode_greedy,
    "temperature": decode_temperature,
    "top_k": decode_top_k,
    "top_p": decode_top_p,
    "typical": decode_typical,
}


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

# HC1 (PREREG_HC1.md section 2). Presence is what the GATE reads from the URL
# STRING. The parser (uie.factorization.parse) starts the fragment at the first
# "#", the query at the first "?" before it, and the path at the first "/"
# before both; everything earlier is the host. So a marker alone does not make
# a field present or absent -- a delimiter CHARACTER inside another field can
# create one, or swallow one. The rules cover both.

#: Where the host sits in the sequence: every field rendered before the path.
_HOST = frozenset({None, "SCH", "USR", "SUF", "LBL", "SUB", "PRT"})

#: E8: placeholders in the forced queue for the plan slots of FULL FREE
#: (index = slot). Never real token ids, so they cannot collide with the
#: vocabulary; each is replaced by the model's own draw over the slot's pair.
#: E12 adds a third (the slash plan); a two-slot plan uses the first two, as before.
_PLAN_SLOTS = (-1, -2, -3)

#: A required field: the markers after it in grammar order (`... PRT, PTH, QRY,
#: FRG`, then EOS), the characters that may not open its content, and -- until
#: it is present -- the characters that would start a later part first.
_REQUIRE = {
    "PTH": {"later": ("QRY", "FRG"), "first": "?#",
            "earlier": ((_HOST, "?#"),)},
    "QRY": {"later": ("FRG",), "first": "#",
            "earlier": ((_HOST | {"PTH"}, "#"),)},
}

#: A forbidden field: where its delimiter would create it without its marker.
_FORBID = {
    "PTH": ((_HOST, "/"),),
    "QRY": ((_HOST | {"PTH"}, "?"),),
}


def _presence_rules(tokenizer: URLTokenizer, wanted: Dict[str, bool]) -> dict:
    """HC1's rules for one request, as token ids. A delimiter absent from the
    vocabulary cannot be sampled and is simply left out."""
    from uie.tokenizer import _MARKERS

    def ids(chars: str) -> List[int]:
        return [tokenizer.stoi[c] for c in chars if c in tokenizer.stoi]

    return {
        "require": {f: {"later": [tokenizer.marker_id(g) for g in _REQUIRE[f]["later"]],
                        "first": ids(_REQUIRE[f]["first"]),
                        "earlier": [(region, ids(chars))
                                    for region, chars in _REQUIRE[f]["earlier"]]}
                    for f, v in wanted.items() if v},
        "forbid_markers": [tokenizer.marker_id(f) for f, v in wanted.items() if not v],
        "forbid_regions": [(region, ids(chars)) for f, v in wanted.items() if not v
                           for region, chars in _FORBID[f]],
        "markers": [tokenizer.marker_id(m) for m in _MARKERS],
        "content": frozenset(tokenizer.stoi[c] for c in tokenizer._chars),
        "eos": tokenizer.eos_id,
    }


def _forbidden(rules: dict, field: Optional[str], filled: set) -> List[int]:
    """The token ids a row may not sample next, given the field it is in."""
    bad = set(rules["forbid_markers"])
    for region, ids in rules["forbid_regions"]:
        if field in region:
            bad.update(ids)
    for f, rule in rules["require"].items():
        if f in filled:
            continue
        bad.add(rules["eos"])
        if field == f:     # inside F, nothing yet: a marker or delimiter empties it
            bad.update(rules["markers"])
            bad.update(rule["first"])
        else:              # F not reached: a later marker skips it, a delimiter
            bad.update(rule["later"])      # starts a later part before it
            for region, ids in rule["earlier"]:
                if field in region:
                    bad.update(ids)
    return sorted(bad)


@torch.no_grad()
def generate(model: URLTransformer, tokenizer: URLTokenizer, *,
             label: str, n: int = 1, language: str = "und",
             frame: str = "domain", suffix: Optional[str] = None,
             logit_bias: Optional[Dict[int, float]] = None,
             field_bias: Optional[Dict[str, Dict[int, float]]] = None,
             # ADOPTED 2026-09-23, was "typical".
             # R4 (research/qry/R4_RESULTS.md): at the frozen checkpoint,
             # at gate scale, paired with matched floors, temperature beat
             # typical on every one of ten cells, five significantly, and
             # regressed none. decode_temperature defaults to temperature
             # 1.0, which is exactly what R4 gated.
             decoder: str = "temperature",
             decoder_kwargs: Optional[dict] = None,
             seed: Optional[int] = None, batch_size: int = 32,
             device: Optional[torch.device] = None,
             fields: Optional[Dict[str, Optional[bool]]] = None,
             enforce: Optional[str] = None,
             trace: Optional[list] = None,
             field_temperature: Optional[Dict[str, float]] = None,
             field_stop_bias: Optional[Dict[str, float]] = None,
             blend: Optional[Dict[str, float]] = None,
             logp: Optional[list] = None,
             ids_out: Optional[list] = None,
             plan_trace: Optional[list] = None,
             slash_rule: bool = False,
             slash_trace: Optional[list] = None,
             slash_paths: bool = False,
             query_rule: bool = False,
             query_trace: Optional[list] = None) -> List[str]:
    """Sample `n` URLs from the class-conditional distribution.

    E8 (research/native/PREREG_E8.md): with a presence-plan tokenizer, `fields`
    is None (FULL FREE: the model fills the plan after the suffix) or sets both
    fields (FULL CONTROL: the plan is supplied). `plan_trace`, when a list,
    receives per row the probability mass the model put outside each FREE plan
    slot's two legal tokens (0.0 for supplied plans).

    `suffix`, when given, is SUPPLIED rather than generated — sampled by the
    caller from an external frequency table. That is what lets a TLD registered
    after training still be used without retraining: the model has never seen it,
    but it reads the suffix as characters and can condition on its spelling.

    DC1 (research/dc1/PREREG_DC1.md), generation-time calibration -- both None
    by default, and then generation is bit-identical:
      `field_temperature` {field: T}: logits divided by T at every step inside
        that field (on top of the decoder's own temperature);
      `field_stop_bias` {field: d}: log-odds d added, at every step inside that
        field, to each token that ends it -- the end token and every field
        marker -- so it changes how long the field runs, not what follows it;
      `blend` {field: w}: inside that field, CONTENT characters use
        z_free + w * (z_control - z_free), where z_free is the same model's
        prediction with every presence control `?`; field markers and the end
        token always keep z_control, so the control alone decides structure.
    `logp`, when a list, receives each row's summed log-probability of its
    sampled tokens under the distribution the `temperature` decoder draws from;
    `ids_out`, when a list, receives each row's raw token ids (prefix included,
    and the end token when the row ended on it) -- for DC1's consistency tests.
    DC1 scopes all three with its OWN field tracker, which follows supplied
    markers too; the legacy tracker behind `field_bias` is left exactly as it
    was, because the frozen path delta was captured on it (96-case fixture).

    E13 D1 (research/e13/PREREG_E13.md section 2.2), the slash rule -- False by
    default, and then generation is bit-identical. It is a decoder-side
    intervention the owner authorised on 2026-10-02 for the diagnosed
    slash-obedience failures only, as an explicit amendment of the native-only
    rule; it changes no weight. With a SUPPLIED plan whose P is 0, at every
    sampled step from the label marker until the host ends, the host may only
    end as the plan's S says (`slash_ends`, `slash_redistribute`). P1 plans and
    every step outside the host are untouched. `slash_trace`, when a list,
    receives per row the end mass the rule moved.

    E14 D1' (research/e14/PREREG_E14.md section 2.2) -- `slash_paths`, False by
    default, and then the slash rule is E13's D1 bit for bit. True extends D1's
    own S = 1 rule to SUPPLIED plans with P = 1, where the path implies the slash:
    the host may not end without one (<QRY>, <FRG>, <EOS> get probability 0; their
    mass goes to <PTH> and <RS> in proportion). Needs `slash_rule=True`. P0 plans
    are exactly D1.

    E17 B5 (research/e17/PREREG_E17.md section 2), the query-plan rule --
    `query_rule`, False by default, and then generation is bit-identical. With a
    SUPPLIED plan whose Q is 1, at every sampled step from the label marker until
    <QRY> is sampled, the tokens that would end the URL or open a fragment before
    the query (<EOS>, <FRG>) get probability 0 and their mass goes to <QRY>
    (`query_ends`, `query_redistribute`): the URL starts its query where it would
    have ended. Q0 plans, every step after <QRY>, and every weight are untouched.
    Applied after D1, so both redistributions stay exact. `query_trace`, when a
    list, receives per row the mass the rule moved.
    """
    if decoder not in DECODERS:
        raise KeyError(f"unknown decoder {decoder!r}; known: {sorted(DECODERS)}")
    strategy = DECODERS[decoder]
    kwargs = decoder_kwargs or {}
    device = device or next(model.parameters()).device
    model.eval()

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    cls_token = f"<C:{label}>"
    if cls_token not in tokenizer.stoi:
        raise KeyError(f"unknown class {label!r}; known: {tokenizer.classes}")
    prefix = [tokenizer.bos_id, tokenizer.stoi[cls_token]]
    if getattr(tokenizer, "include_frame", False):
        # SUPPLIED, like the class token and the suffix. The generator emits one
        # URL per invented host, so `domain` is the frame it actually samples;
        # `page` asks for a URL as it would appear on a multi-page site.
        frame_token = f"<F:{frame}>"
        if frame_token not in tokenizer.stoi:
            raise KeyError(f"unknown frame {frame!r}; "
                           f"tokenizer has {sorted(t for t in tokenizer.stoi if t.startswith('<F:'))}")
        prefix.append(tokenizer.stoi[frame_token])
    if getattr(tokenizer, "include_language", True):
        # Pre-E2.3 tokenizers carry `<L:*>`; current ones do not (the token was
        # a class marker and its information was 96% the TLD's — see
        # `URLTokenizer.__init__`). `language` is accepted and ignored so a
        # caller written against the old signature still works.
        lang_token = f"<L:{language}>"
        if lang_token not in tokenizer.stoi:
            lang_token = "<L:und>"
        prefix.append(tokenizer.stoi[lang_token])

    # E8 (research/native/PREREG_E8.md section 3.2): a PRESENCE PLAN follows the
    # suffix instead of controls in the prefix. Supplied (FULL CONTROL: both
    # fields set) it is injected like the suffix; not supplied (FULL FREE) the
    # model fills each slot itself -- from its own distribution over that
    # slot's two legal tokens, with the mass it puts elsewhere recorded. There
    # are no partial plans, no masking and no second prediction in this mode.
    plan_mode = getattr(tokenizer, "presence_plan", False)
    plan: Optional[List[int]] = None
    if plan_mode:
        given = {f: v for f, v in (fields or {}).items() if v is not None}
        if given:
            plan = tokenizer.plan_ids(given)       # refuses a partial plan
            if suffix is None:
                raise ValueError("a supplied plan follows a supplied suffix; "
                                 "pass `suffix` (PREREG_E8 section 3.2)")
        if enforce is not None:
            raise ValueError("masking is not a native mechanism: `enforce` is "
                             "refused with a presence plan")
        if blend:
            raise ValueError("a second prediction is not a native mechanism: "
                             "`blend` is refused with a presence plan")
    # E7 presence controls: SUPPLIED like the class and frame. Missing or None
    # is `?`, the unconditional mode -- so a caller that never asks (the gate)
    # samples an E7 checkpoint exactly as it would any other. A checkpoint
    # without the controls REFUSES a request instead of ignoring it.
    elif getattr(tokenizer, "include_fields", False):
        prefix.extend(tokenizer.field_control_ids(fields))
    elif fields and any(v is not None for v in fields.values()):
        raise ValueError("this checkpoint has no field-presence controls "
                         "(trained before E7); `fields` cannot be honoured")

    # E13 D1 (the slash rule): validated here, after the plan is known.
    slash = None
    if slash_rule:
        if not (plan_mode and getattr(tokenizer, "slash_plan", False)):
            raise ValueError("the slash rule needs a slash-plan tokenizer (E12)")
        if plan is None:
            raise ValueError("the slash rule obeys a SUPPLIED plan: pass `fields` "
                             "with PTH, QRY and SLASH set")
        if decoder != "temperature":
            raise ValueError("the slash rule is registered for the raw decode "
                             "(decoder 'temperature') only")
        slash = slash_ends(tokenizer, {f: bool(v) for f, v in given.items()},
                           paths=slash_paths)
    elif slash_trace is not None:
        raise ValueError("`slash_trace` needs `slash_rule=True`")
    elif slash_paths:
        raise ValueError("`slash_paths` extends the slash rule: it needs `slash_rule=True`")

    # E17 B5 (the query-plan rule): validated here, after the plan is known.
    qrule = None
    if query_rule:
        if not plan_mode:
            raise ValueError("the query rule needs a presence-plan tokenizer (E8)")
        if plan is None:
            raise ValueError("the query rule obeys a SUPPLIED plan: pass `fields`")
        if decoder != "temperature":
            raise ValueError("the query rule is registered for the raw decode "
                             "(decoder 'temperature') only")
        qrule = query_ends(tokenizer, {f: bool(v) for f, v in given.items()})
    elif query_trace is not None:
        raise ValueError("`query_trace` needs `query_rule=True`")

    # HC1 (research/hc1/PREREG_HC1.md): ENFORCE the presence controls while
    # sampling. E7's controls are learned, and sampling can still end a URL
    # before a required field (E7 C1: 96.4% on benign <Q:1>). `enforce="mask"`
    # removes the tokens the request forbids before every draw. None, the
    # default, runs none of this: generation is bit-identical to the code
    # before HC1 (tests/fixtures_e7_generate.json).
    rules = None
    if enforce is not None:
        if enforce != "mask":
            raise ValueError(f"unknown enforce {enforce!r}; known: 'mask'")
        if not getattr(tokenizer, "include_fields", False):
            raise ValueError("enforce needs a checkpoint with field-presence "
                             "controls (E7 or later)")
        wanted = {f: bool(v) for f, v in (fields or {}).items() if v is not None}
        if not wanted:
            raise ValueError("enforce='mask' needs a presence control set to "
                             "True or False")
        rules = _presence_rules(tokenizer, wanted)

    # A supplied suffix is INJECTED WHEN THE MODEL REACHES THAT FIELD, never
    # prepended. `[SCH]` and `[USR]` precede `[SUF]` in the sequence, so seeding
    # the prefix with the suffix skips the scheme entirely — every sample comes
    # out scheme-less, and a discriminator then separates real from generated
    # perfectly on that alone (measured: AUC 1.0000). The model must be allowed
    # to produce everything before the suffix itself.
    forced = None
    if suffix is not None:
        forced = ([tokenizer.marker_id("SUF")] +
                  [tokenizer.stoi.get(ch, tokenizer.stoi["<unk>"]) for ch in suffix] +
                  ((plan if plan is not None else
                    list(_PLAN_SLOTS[:len(tokenizer.plan_slots())])) if plan_mode else []) +
                  [tokenizer.marker_id("LBL")])

    # DC1: validated here so a bad setting fails before any sampling.
    for f, t in (field_temperature or {}).items():
        if not (isinstance(t, (int, float)) and math.isfinite(t) and t > 0):
            raise ValueError(f"field_temperature[{f!r}] must be a positive number; got {t!r}")
    for f, d in (field_stop_bias or {}).items():
        if not (isinstance(d, (int, float)) and math.isfinite(d)):
            raise ValueError(f"field_stop_bias[{f!r}] must be a finite number; got {d!r}")
    dc1 = None
    if field_temperature or field_stop_bias or blend:
        dc1 = {"stop": torch.tensor(tokenizer.struct_ids(), dtype=torch.long)}
    if blend:
        if not getattr(tokenizer, "include_fields", False):
            raise ValueError("blend needs a checkpoint with field-presence controls "
                             "(E7 or later)")
        for f, w in blend.items():
            if not (isinstance(w, (int, float)) and math.isfinite(w)):
                raise ValueError(f"blend[{f!r}] must be a finite number; got {w!r}")
        n_ctrl = len(tokenizer.field_control_ids(None))
        dc1["free_prefix"] = prefix[:-n_ctrl] + tokenizer.field_control_ids(None)
        dc1["content"] = torch.tensor(sorted(tokenizer.stoi[c] for c in tokenizer._chars),
                                     dtype=torch.long)

    out: List[str] = []
    for start in range(0, n, batch_size):
        out.extend(_generate_chunk(model, tokenizer, prefix,
                                   min(batch_size, n - start), strategy,
                                   kwargs, generator, device, forced,
                                   logit_bias, field_bias, rules, trace,
                                   field_temperature, field_stop_bias, blend,
                                   dc1, logp, ids_out,
                                   tokenizer.plan_slots() if plan_mode else None,
                                   plan_trace, slash, slash_trace, qrule, query_trace))
    return out


def query_ends(tokenizer: URLTokenizer, plan: Dict[str, bool]) -> Optional[dict]:
    """E17 B5 (research/e17/PREREG_E17.md section 2): the tokens a supplied Q = 1
    plan forbids before its query (<EOS>, <FRG>) and the one their mass goes to
    (<QRY>), or None where B5 does not apply (Q = 0)."""
    if not plan.get("QRY"):
        return None
    return {"good": torch.tensor([tokenizer.marker_id("QRY")], dtype=torch.long),
            "bad": torch.tensor([tokenizer.eos_id, tokenizer.marker_id("FRG")], dtype=torch.long)}


def query_redistribute(logits: torch.Tensor, good: torch.Tensor, bad: torch.Tensor,
                       temperature: float = 1.0) -> Tuple[torch.Tensor, float]:
    """E17 B5 at one step, for one row's logits (in place, and returned).

    D1's exact redistribution (`slash_redistribute`): the forbidden tokens get
    probability 0 and their mass goes to <QRY>, so every other token keeps its
    probability exactly. Two guards D1 does not need: when the forbidden tokens
    carry no mass (D1 has already removed them at this host step) nothing moves,
    and when <QRY> itself is forbidden the forbidden tokens are removed without a
    target. Returns the moved mass."""
    t = max(float(temperature), 1e-6)
    z = logits / t
    lse_bad = torch.logsumexp(z[bad], dim=0)
    if not torch.isfinite(lse_bad):
        return logits, 0.0
    moved = float(torch.exp(lse_bad - torch.logsumexp(z, dim=0)))
    lse_good = torch.logsumexp(z[good], dim=0)
    if torch.isfinite(lse_good):
        shift = torch.logaddexp(lse_good, lse_bad) - lse_good
        logits[good] = logits[good] + t * shift
    logits[bad] = float("-inf")
    return logits, moved


def slash_ends(tokenizer: URLTokenizer, plan: Dict[str, bool],
               paths: bool = False) -> Optional[dict]:
    """E13 D1 (research/e13/PREREG_E13.md section 2.2): the host-end tokens a
    supplied plan allows and forbids, or None where D1 does not apply.

    D1 applies only to plans with P = 0, where S is the URL's own decision: a
    path implies the slash, so P1 plans are left exactly as they are. The five
    ways a host can end are <PTH>, <RS>, <QRY>, <FRG> and <EOS>. S = 1 forbids
    the slash-less ends (<QRY>, <FRG>, <EOS>); S = 0 forbids the slashed ones
    (<RS>, <PTH>).

    E14 D1' (`paths=True`, research/e14/PREREG_E14.md section 2.2): P1 plans get
    D1's S = 1 rule too. A supplied P1 plan always carries S = 1
    (`URLTokenizer.plan_ids` refuses anything else), so the same assignment below
    applies unchanged."""
    if plan["PTH"] and not paths:
        return None
    slashed = [tokenizer.marker_id("PTH"), tokenizer.rs_id]
    bare = [tokenizer.marker_id("QRY"), tokenizer.marker_id("FRG"), tokenizer.eos_id]
    good, bad = (slashed, bare) if plan["SLASH"] else (bare, slashed)
    return {"good": torch.tensor(good, dtype=torch.long),
            "bad": torch.tensor(bad, dtype=torch.long),
            "ends": frozenset(good + bad)}


def slash_redistribute(logits: torch.Tensor, good: torch.Tensor, bad: torch.Tensor,
                       temperature: float = 1.0) -> Tuple[torch.Tensor, float]:
    """E13 D1 at one host step, for one row's logits (in place, and returned).

    Under the decoder's temperature, the forbidden ends get probability 0 and
    their mass goes to the allowed ends in proportion to the allowed ends' own
    probabilities. So the probability of ending the host at this step is
    unchanged, and every token that is not an end keeps its probability exactly.
    Computed in log space: adding T * log((m_good + m_bad) / m_good) to the
    allowed ends' logits leaves the softmax normaliser unchanged. Returns the
    moved mass m_bad."""
    t = max(float(temperature), 1e-6)
    z = logits / t
    lse_good = torch.logsumexp(z[good], dim=0)
    lse_bad = torch.logsumexp(z[bad], dim=0)
    moved = float(torch.exp(lse_bad - torch.logsumexp(z, dim=0)))
    shift = torch.logaddexp(lse_good, lse_bad) - lse_good
    logits[good] = logits[good] + t * shift
    logits[bad] = float("-inf")
    return logits, moved


def blend_content(z_control: torch.Tensor, z_free: torch.Tensor, w: float,
                  content: torch.Tensor) -> torch.Tensor:
    """DC1's blend, for one row or many: content characters move to
    z_free + w * (z_control - z_free); every other token keeps z_control."""
    z = z_control.clone()
    zc, zu = z_control[..., content], z_free[..., content]
    z[..., content] = zu + w * (zc - zu)
    return z


@torch.no_grad()
def _generate_chunk(model: URLTransformer, tokenizer: URLTokenizer,
                    prefix: List[int], n: int, strategy, kwargs: dict,
                    generator: torch.Generator, device,
                    forced: Optional[List[int]] = None,
                    logit_bias: Optional[Dict[int, float]] = None,
                    field_bias: Optional[Dict[str, Dict[int, float]]] = None,
                    rules: Optional[dict] = None,
                    trace: Optional[list] = None,
                    field_temperature: Optional[Dict[str, float]] = None,
                    field_stop_bias: Optional[Dict[str, float]] = None,
                    blend: Optional[Dict[str, float]] = None,
                    dc1: Optional[dict] = None,
                    logp: Optional[list] = None,
                    ids_out: Optional[list] = None,
                    plan_slots: Optional[List[Tuple[int, int]]] = None,
                    plan_trace: Optional[list] = None,
                    slash: Optional[dict] = None,
                    slash_trace: Optional[list] = None,
                    qrule: Optional[dict] = None,
                    query_trace: Optional[list] = None
                    ) -> List[str]:
    """Step `n` sequences forward together.

    Only the FORWARD PASS is batched; the decoder is still applied per row. The
    forward pass is essentially all of the cost (a 4.8M-parameter model against
    ~150 logits), so this keeps decoding strategies pluggable and per-sample —
    they need no batched reimplementation — while removing the bottleneck that
    made a five-decoder ablation take hours on CPU.

    Rows that emit the end token are frozen: they are padded so the tensor stays
    rectangular, and their later tokens are discarded. Without that, a finished
    sequence would keep sampling and its text would grow past its own ending.
    """
    rows: List[List[int]] = [list(prefix) for _ in range(n)]
    active = list(range(n))                 # indices still generating
    # Which field each row is currently inside, for FIELD-SCOPED bias. A global
    # bias was measured to move host composition it has no business touching:
    # suppressing EOS everywhere made the model reluctant to stop anywhere, so
    # `host_len` rose 17.2 -> 18.4 and `lbl_len` 11.3 -> 12.6 while chasing path
    # length. The measured defect is mid-path leakage specifically, so the
    # correction belongs only at positions inside that field.
    from uie.tokenizer import _MARKERS as _FIELD_NAMES
    _marker_of = {tokenizer.marker_id(m): m for m in _FIELD_NAMES}
    in_field: List[Optional[str]] = [None] * n
    dc1_field: List[Optional[str]] = [None] * n    # DC1's tracker (see generate)
    injected = [False] * n                  # has the forced suffix been placed?
    queued: List[List[int]] = [[] for _ in range(n)]   # pending forced tokens
    suf_marker = tokenizer.marker_id("SUF")
    # HC1: which required fields each row has satisfied, and -- for `trace` --
    # the log of the probability the rules left it, sum_t log(1 - m_t).
    filled: List[set] = [set() for _ in range(n)]
    log_keep = [0.0] * n
    masked_steps = [0] * n
    temperature = max(float(kwargs.get("temperature", 1.0)), 1e-6)
    row_logp = [0.0] * n                    # DC1 `logp`: per-row sum, if asked
    ended = [False] * n                     # DC1 `ids_out`: ended on the end token
    plan_off = [0.0] * n                    # E8: mass outside a free plan slot's pair
    # E13 D1: the host is open once <LBL> is in the row, and ends with the first
    # host-end token sampled after it; the rule acts only in between.
    lbl_marker = tokenizer.marker_id("LBL")
    host_open = [False] * n
    host_done = [False] * n
    slash_moved = [0.0] * n
    # E17 B5: the rule acts from <LBL> until the row's <QRY> is sampled.
    qry_marker = tokenizer.marker_id("QRY")
    q_open = [False] * n
    q_done = [False] * n
    query_moved = [0.0] * n

    while active and len(rows[active[0]]) < model.cfg.max_len:
        # Finished rows are REMOVED from the batch, not merely masked. Keeping
        # them would make every step cost the full batch until the single
        # longest sequence terminated — which measured slower than generating
        # one at a time, since URL lengths are heavy-tailed (median 69 tokens,
        # p99 246).
        window = torch.tensor([rows[i] for i in active], dtype=torch.long,
                              device=device)
        logits = model(window)[:, -1].float().cpu()

        # DC1 blend: a second pass with every control `?`, for the rows whose
        # current field is blended (w != 1). w == 1 is skipped, not computed,
        # so an all-ones blend stays bit-identical.
        if blend:
            need = [slot for slot, i in enumerate(active)
                    if not queued[i] and blend.get(dc1_field[i] or "", 1.0) != 1.0]
            if need:
                cut = len(prefix)
                free = torch.tensor([dc1["free_prefix"] + rows[active[s]][cut:]
                                     for s in need], dtype=torch.long, device=device)
                z_free = model(free)[:, -1].float().cpu()
                for k, slot in enumerate(need):
                    logits[slot] = blend_content(logits[slot], z_free[k],
                                                 blend[dc1_field[active[slot]]],
                                                 dc1["content"])

        # A DECODE-TIME CALIBRATION CORRECTION, added in log-odds space so it
        # shifts every position's odds by the same factor rather than its
        # probability by the same amount. Measured on E5: mid-path EOS mass runs
        # ~1pp above the empirical hazard, and over a 28-step path that compounds
        # to `0.982**28 = 0.60` -- a 40% chance of stopping early. A uniform
        # log-odds shift suppresses the small (spurious) probabilities
        # proportionally more than the large (legitimate) ones, which is the
        # behaviour wanted here: at delta=0.5 a 1.8% leakage falls to 61% of
        # itself while a 58% legitimate stop falls only to 79%.
        #
        # Default None => not applied, and generation is bit-identical.
        if logit_bias:
            for token, delta in logit_bias.items():
                logits[:, token] += delta
        if field_bias:
            for slot, i in enumerate(active):
                b = field_bias.get(in_field[i] or "")
                if b:
                    for token, delta in b.items():
                        logits[slot, token] += delta
        # DC1 stop bias, then field temperature (so the bias is scaled with the
        # rest of that step). A zero bias and T == 1 are skipped.
        if field_stop_bias:
            for slot, i in enumerate(active):
                d = field_stop_bias.get(dc1_field[i] or "")
                if d:
                    logits[slot, dc1["stop"]] += d
        if field_temperature:
            for slot, i in enumerate(active):
                t = field_temperature.get(dc1_field[i] or "")
                if t is not None and t != 1.0:
                    logits[slot] = logits[slot] / t
        # E13 D1: after every other adjustment, so the redistribution is exact
        # under the distribution the decoder then samples.
        if slash is not None:
            for slot, i in enumerate(active):
                if queued[i] or not host_open[i] or host_done[i]:
                    continue
                _, m = slash_redistribute(logits[slot], slash["good"], slash["bad"],
                                          temperature)
                slash_moved[i] += m
        # E17 B5: after D1, so both redistributions are exact.
        if qrule is not None:
            for slot, i in enumerate(active):
                if queued[i] or not q_open[i] or q_done[i]:
                    continue
                _, m = query_redistribute(logits[slot], qrule["good"], qrule["bad"],
                                          temperature)
                query_moved[i] += m

        # HC1: remove what the request forbids at this row's position. The
        # decoder then samples the model's distribution renormalised over the
        # allowed tokens. m is the mass removed, under the temperature softmax
        # the `temperature` decoder samples from.
        if rules is not None:
            for slot, i in enumerate(active):
                if queued[i]:
                    continue                    # a supplied token is not sampled
                bad = _forbidden(rules, in_field[i], filled[i])
                if not bad:
                    continue
                if trace is not None:
                    m = float(F.softmax(logits[slot] / temperature, dim=-1)[bad].sum())
                    if m > 0.0:
                        masked_steps[i] += 1
                        log_keep[i] += math.log1p(-m) if m < 1.0 else -math.inf
                logits[slot, bad] = float("-inf")

        still: List[int] = []
        for slot, i in enumerate(active):
            # A forced field is fed ONE TOKEN PER STEP. Appending it all at once
            # would grow this row by len(forced) while its batch-mates grew by
            # one, so the rows stop being rectangular and the next tensor build
            # fails. (Invisible at batch size 1, which is how it survived a
            # manual check.)
            if queued[i]:
                token = queued[i].pop(0)
                if token < 0:
                    # E8 FULL FREE: a plan slot. The model fills it from its own
                    # distribution at this step (the decoder's temperature),
                    # renormalised over the slot's two legal tokens; the mass it
                    # put elsewhere is recorded, never hidden.
                    legal = list(plan_slots[_PLAN_SLOTS.index(token)])
                    probs = F.softmax(logits[slot] / temperature, dim=-1)
                    pair = probs[legal]
                    plan_off[i] += 1.0 - float(pair.sum())
                    pick = int(torch.multinomial(pair / pair.sum(), 1,
                                                 generator=generator))
                    token = legal[pick]
                    if logp is not None:
                        row_logp[i] += math.log(float(pair[pick] / pair.sum()))
                rows[i].append(token)
                # DC1 tracks a supplied marker like a sampled one; the legacy
                # `in_field` does not (the frozen path delta was defined, and
                # its 96-case fixture captured, on that behaviour).
                if token in _marker_of:
                    dc1_field[i] = _marker_of[token]
                if token == lbl_marker:
                    host_open[i] = True
                    if qrule is not None:
                        q_open[i] = True
                still.append(i)
                continue

            token = strategy(logits[slot], generator, **kwargs)
            if logp is not None:
                row_logp[i] += float(F.log_softmax(logits[slot] / temperature,
                                                   dim=-1)[token])
            if token == tokenizer.eos_id:
                ended[i] = True
                continue                    # row is complete; drop it
            if forced is not None and not injected[i] and token == suf_marker:
                # The model asked for a suffix; hand it the one we chose and
                # queue the move to the label field.
                injected[i] = True
                rows[i].append(forced[0])
                dc1_field[i] = "SUF"
                queued[i] = list(forced[1:])
            else:
                rows[i].append(token)
                if token in _marker_of:
                    in_field[i] = _marker_of[token]
                    dc1_field[i] = _marker_of[token]
                elif (rules is not None and token in rules["content"]
                      and in_field[i] in rules["require"]):
                    filled[i].add(in_field[i])
                if slash is not None:
                    if token == lbl_marker:
                        host_open[i] = True
                    elif host_open[i] and token in slash["ends"]:
                        host_done[i] = True
                if qrule is not None:
                    if token == lbl_marker:
                        q_open[i] = True
                    elif token == qry_marker:
                        q_done[i] = True
            still.append(i)
        active = still

    if logp is not None:
        logp.extend(row_logp)
    if plan_trace is not None:
        plan_trace.extend(plan_off)
    if slash_trace is not None:
        slash_trace.extend(slash_moved)
    if query_trace is not None:
        query_trace.extend(query_moved)
    if ids_out is not None:
        ids_out.extend(rows[i] + ([tokenizer.eos_id] if ended[i] else [])
                       for i in range(n))
    if trace is not None:
        # Rows still active here reached max_len without ending.
        unfinished = set(active)
        for i in range(n):
            trace.append({"log_keep": log_keep[i], "masked_steps": masked_steps[i],
                          "truncated": i in unfinished,
                          "satisfied": rules is None
                          or set(rules["require"]) <= filled[i]})
    return [tokenizer.decode(row) for row in rows]
