"""
research/v2/v2_mem.py

V2-P1 (research/v2/PREREG_V2_P1_MEMORY.md section 3): a product-key memory layer
(Lample et al., "Large Memory Layers with Product Keys", NeurIPS 2019) added to
e16_cap's network.

    x <- x + Dropout( M( LayerNorm(x) ) )        after block `mem_after` (4 of 8)

M, at every position that is not padding:
  1. q = BatchNorm(W_q u + b_q), 4 heads x 512 dimensions;
  2. per head, each half of q (256) scored against its own 512 sub-keys, top 32 kept;
  3. the 32 x 32 sums are the candidate slots (512 * i1 + i2); their top 32 kept --
     an EXACT top-32 over all 262,144 slots;
  4. a softmax over those 32 scores;
  5. the weighted sum of the chosen value rows, summed over the heads: 128 rows read.

Nothing in it is a word. The slots are learned vectors, trained by the engine's own
loss in the same run; generation still writes letter by letter (`uie.model.generate`
is called unchanged). Nothing under `uie/` changes: the model subclasses
`uie.model.URLTransformer`, saves through `uie.train.save_checkpoint` (its config is a
`ModelConfig` subclass, so `asdict` carries the memory's settings), and loads through
`load_checkpoint` below -- `uie.train.load_checkpoint` refuses a memory checkpoint
(an unexpected keyword), so it cannot be loaded as a plain model by mistake.

The memory runs in float32 outside autocast, skips padding (its output there is 0)
and takes the BatchNorm's batch statistics from real tokens only. In evaluation and
generation the BatchNorm uses its running statistics, so a URL's output never depends
on the other URLs in its batch.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from uie.model import ModelConfig, URLTransformer
from uie.tokenizer import URLTokenizer

__all__ = ["MemConfig", "ProductKeyMemory", "MemURLTransformer", "REGISTERED",
           "with_memory", "init_memory", "part_sha256", "is_memory_key",
           "load_checkpoint", "memory_preflight", "memory_param_count"]

#: The registered memory (PREREG_V2_P1_MEMORY.md section 3.1).
REGISTERED = {"mem_after": 4, "mem_n_keys": 512, "mem_heads": 4, "mem_knn": 32, "mem_k_dim": 512}
#: V2-P1b (T2): the reference implementation's values optimiser, `adam,lr=0.001` --
#: facebookresearch/XLM, xlm/optim.py: Adam, betas (0.9, 0.999), eps 1e-8, weight decay 0.
VALUES_BETAS = (0.9, 0.999)
#: State-dict prefixes of the memory's own entries; everything else is e16_cap's.
MEMORY_PREFIXES = ("memory.", "ln_mem.")


@dataclass
class MemConfig(ModelConfig):
    """`ModelConfig` plus the memory's settings. `mem_pad_id` is the tokenizer's padding
    id: the memory skips those positions."""

    mem_after: int = 4
    mem_n_keys: int = 512
    mem_heads: int = 4
    mem_knn: int = 32
    mem_k_dim: int = 512
    mem_pad_id: int = 0
    #: V2-P1b (PREREG_V2_P1B.md): how the query is normalised. "batch" is P1's BatchNorm
    #: (batch statistics in training, running ones in evaluation -- the cause of P1's stop);
    #: "layer" is a per-token LayerNorm over each head's query, identical in training and
    #: generation. The default keeps P1's checkpoint loading exactly as it was trained.
    mem_query_norm: str = "batch"
    #: V2-P1b: False takes the memory branch out of the network's dropout (Lample et al. use
    #: none on the memory). True is P1.
    mem_branch_dropout: bool = True
    #: V2-P1b (T2): None is P1 (the values train with every other weight). Set, the values
    #: get their own optimiser group (`optimizer_groups`): Adam with VALUES_BETAS, no weight
    #: decay, the network's schedule scaled to peak at this rate.
    mem_values_lr: Optional[float] = None


class ProductKeyMemory(nn.Module):
    """The memory itself: query network, sub-keys, values. Section 3.1, steps 1-5."""

    def __init__(self, d_in: int, d_out: int, n_keys: int, heads: int, knn: int,
                 k_dim: int, query_norm: str = "batch") -> None:
        super().__init__()
        if k_dim % 2:
            raise ValueError(f"k_dim must be even (two halves); got {k_dim}")
        if not 1 <= knn <= n_keys:
            raise ValueError(f"knn must be in 1..n_keys; got {knn} with {n_keys} keys")
        if query_norm not in ("batch", "layer"):
            raise ValueError(f"query_norm must be 'batch' or 'layer'; got {query_norm!r}")
        self.n_keys, self.heads, self.knn, self.k_dim = n_keys, heads, knn, k_dim
        self.query_norm = query_norm
        self.query_proj = nn.Linear(d_in, heads * k_dim)
        if query_norm == "batch":
            self.query_bn = nn.BatchNorm1d(heads * k_dim)
        else:
            self.query_ln = nn.LayerNorm(k_dim)       # per token and head: no batch statistics
        self.keys = nn.Parameter(torch.empty(heads, 2, n_keys, k_dim // 2))
        self.values = nn.Parameter(torch.empty(n_keys * n_keys, d_out))
        #: Diagnostics only, never saved: a list receives each call's chosen slots;
        #: `ablate` makes the output 0 (section 4.6's ablation).
        self.trace: Optional[list] = None
        self.ablate = False

    @property
    def size(self) -> int:
        return self.n_keys * self.n_keys

    def select(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """q: (N, heads, k_dim), float32. Returns the chosen slots (N, heads * knn) and
        their scores, head by head -- the exact top-knn of s1[i] + s2[j] over every
        (i, j), found through the top-knn of each half (product keys)."""
        half, knn, n = self.k_dim // 2, self.knn, self.n_keys
        slots, scores = [], []
        for h in range(self.heads):
            s1 = q[:, h, :half] @ self.keys[h, 0].t()               # (N, n_keys)
            s2 = q[:, h, half:] @ self.keys[h, 1].t()
            v1, i1 = s1.topk(knn, dim=1)
            v2, i2 = s2.topk(knn, dim=1)
            cand = (v1.unsqueeze(2) + v2.unsqueeze(1)).reshape(-1, knn * knn)
            best, pos = cand.topk(knn, dim=1)
            j1 = i1.gather(1, torch.div(pos, knn, rounding_mode="floor"))
            j2 = i2.gather(1, pos % knn)
            slots.append(j1 * n + j2)
            scores.append(best)
        return torch.cat(slots, dim=1), torch.cat(scores, dim=1)

    def forward(self, u: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """u: (B, T, d_in); valid: (B, T) bool, False at padding. Returns (B, T, d_out)
        in float32, 0 wherever `valid` is False."""
        out = torch.zeros(u.shape[:-1] + (self.values.shape[1],), dtype=torch.float32,
                          device=u.device)
        if self.ablate or not bool(valid.any()):
            return out
        with torch.autocast(device_type=u.device.type, enabled=False):
            flat = u[valid].float()
            q = self.query_proj(flat)
            if self.query_norm == "batch":
                q = self.query_bn(q).view(-1, self.heads, self.k_dim)
            else:
                q = self.query_ln(q.view(-1, self.heads, self.k_dim))
            slots, scores = self.select(q)
            w = F.softmax(scores.view(-1, self.heads, self.knn), dim=-1).view(-1, self.heads * self.knn)
            vals = F.embedding_bag(slots, self.values, mode="sum", per_sample_weights=w)
        if self.trace is not None:
            self.trace.append(slots.detach().cpu())
        out[valid] = vals
        return out


class MemURLTransformer(URLTransformer):
    """e16_cap's network with the memory after block `cfg.mem_after`. Every other module,
    name and computation is `URLTransformer`'s."""

    def __init__(self, cfg: MemConfig) -> None:
        super().__init__(cfg)
        if not 1 <= cfg.mem_after <= cfg.n_layers:
            raise ValueError(f"mem_after must be in 1..{cfg.n_layers}; got {cfg.mem_after}")
        self.ln_mem = nn.LayerNorm(cfg.d_model)
        self.memory = ProductKeyMemory(cfg.d_model, cfg.d_model, cfg.mem_n_keys, cfg.mem_heads,
                                       cfg.mem_knn, cfg.mem_k_dim, cfg.mem_query_norm)

    def optimizer_groups(self, train_cfg):
        """V2-P1b (T2), read by `uie.train.train`'s hook: None (P1, and the default) keeps
        the single optimiser group. With `mem_values_lr` set: every weight but the values in
        the first group (the optimiser's defaults: AdamW, weight decay and betas as the
        recipe sets them), the values in a second -- weight decay 0, which makes AdamW's
        update exactly Adam's, VALUES_BETAS, and `lr_scale` = mem_values_lr / the recipe's
        rate, so the network's warmup and cosine schedule peaks at mem_values_lr."""
        if self.cfg.mem_values_lr is None:
            return None
        values = self.memory.values
        rest = [p for p in self.parameters() if p is not values]
        return [{"params": rest},
                {"params": [values], "weight_decay": 0.0, "betas": VALUES_BETAS,
                 "lr_scale": self.cfg.mem_values_lr / train_cfg.lr}]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # URLTransformer.forward, line for line, with the memory branch after block
        # `mem_after` (V-equiv: with every value row at 0 the logits are exactly the base's).
        _, t = ids.shape
        if t > self.cfg.max_len:
            raise ValueError(f"sequence of {t} exceeds max_len {self.cfg.max_len}")
        pos = torch.arange(t, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        if self.cfg.frame_every and t > self.cfg.frame_index:
            frame = self.tok_emb(ids[:, self.cfg.frame_index]).unsqueeze(1)
            after = (pos >= self.cfg.frame_index).to(x.dtype).view(1, t, 1)
            x = x + frame * after
        x = self.drop(x)
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            if i == self.cfg.mem_after:
                branch = self.memory(self.ln_mem(x), ids != self.cfg.mem_pad_id)
                x = x + (self.drop(branch) if self.cfg.mem_branch_dropout else branch)
        h = self.ln_f(x)
        logits = self.head(h)
        if not self._struct:
            return logits
        logits = logits.float()
        raw_gate = self.gate(h).float()
        gate = F.logsigmoid(raw_gate)
        ngate = F.logsigmoid(-raw_gate)
        out = torch.full_like(logits, float("-inf"))
        s_lp = F.log_softmax(logits.index_select(-1, self.struct_idx), dim=-1)
        c_lp = F.log_softmax(logits.index_select(-1, self.content_idx), dim=-1)
        out.index_copy_(-1, self.struct_idx, s_lp + gate)
        out.index_copy_(-1, self.content_idx, c_lp + ngate)
        return out


# --------------------------------------------------------------------------- #
# the start (section 3.2)
# --------------------------------------------------------------------------- #

def is_memory_key(name: str) -> bool:
    return name.startswith(MEMORY_PREFIXES)


def part_sha256(model: nn.Module, keep: Callable[[str], bool]) -> str:
    """`uie.free.weights_sha256`'s fingerprint over the state-dict entries `keep` accepts:
    name order; name, dtype, shape and raw bytes. Over e16_cap's part it equals the base
    model's own `weights_sha256`."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not keep(name):
            continue
        a = tensor.detach().to("cpu").contiguous()
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(a.numpy().tobytes())
    return h.hexdigest()


def init_memory(model: MemURLTransformer, seed: int) -> None:
    """Section 3.2: the memory's weights from their own generator, in this order --
    sub-keys uniform(+-1/sqrt(k_dim)), values normal(0, 1/sqrt(d_value)), query weights
    normal(0, 0.02) with bias 0; BatchNorm and LayerNorm at their defaults."""
    g = torch.Generator().manual_seed(seed)
    m = model.memory
    with torch.no_grad():
        bound = 1.0 / math.sqrt(m.k_dim)
        m.keys.copy_(torch.rand(tuple(m.keys.shape), generator=g) * (2 * bound) - bound)
        m.values.copy_(torch.randn(tuple(m.values.shape), generator=g) * m.values.shape[1] ** -0.5)
        m.query_proj.weight.copy_(torch.randn(tuple(m.query_proj.weight.shape), generator=g) * 0.02)
        m.query_proj.bias.zero_()
    if m.query_norm == "batch":
        m.query_bn.reset_parameters()      # weight 1, bias 0; running mean 0, variance 1
    else:
        m.query_ln.reset_parameters()      # weight 1, bias 0
    model.ln_mem.reset_parameters()


def with_memory(base: URLTransformer, seed: int, pad_id: int,
                **settings) -> Tuple[MemURLTransformer, Dict[str, object]]:
    """`base`'s exact weights plus a freshly drawn memory (section 3.2). Returns the model
    and its fingerprints: e16_cap's part (equal to `weights_sha256(base)`, checked here),
    the memory's part, and the seed."""
    from uie.free import weights_sha256
    unknown = sorted(set(settings) - set(REGISTERED)
                     - {"mem_query_norm", "mem_branch_dropout", "mem_values_lr"})
    if unknown:
        raise ValueError(f"unknown memory settings: {unknown}")
    cfg = MemConfig(**asdict(base.cfg), mem_pad_id=pad_id, **{**REGISTERED, **settings})
    model = MemURLTransformer(cfg)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    if unexpected or not all(is_memory_key(k) for k in missing):
        raise ValueError(f"the base does not fit: missing {missing}, unexpected {unexpected}")
    init_memory(model, seed)
    base_sha = weights_sha256(base)
    if part_sha256(model, lambda k: not is_memory_key(k)) != base_sha:
        raise ValueError("e16_cap's part of the start is not the base's weights")
    return model, {"base_init_sha256": base_sha,
                   "memory_init_sha256": part_sha256(model, is_memory_key),
                   "memory_seed": seed, "memory": {**REGISTERED, **settings}}


# --------------------------------------------------------------------------- #
# loading, sizing, the worst-case check
# --------------------------------------------------------------------------- #

def load_checkpoint(path: Path, *, device: Optional[torch.device] = None
                    ) -> Tuple[URLTransformer, URLTokenizer, ModelConfig]:
    """`uie.train.load_checkpoint`, for a checkpoint with or without the memory."""
    blob = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    mc = dict(blob["model_config"])
    if "mem_n_keys" not in mc:
        from uie.train import load_checkpoint as plain
        return plain(path, device=device)
    cfg = MemConfig(**mc)
    model = MemURLTransformer(cfg)
    model.load_state_dict(blob["model"])
    return model, URLTokenizer.from_state_dict(blob["tokenizer"]), cfg


def memory_param_count(d_model: int, n_keys: int, heads: int, k_dim: int, query_norm: str = "batch",
                       **_) -> int:
    """The memory's parameters (section 3.1's table): values, sub-keys, query projection,
    BatchNorm, LayerNorm."""
    return (n_keys * n_keys * d_model + heads * 2 * n_keys * (k_dim // 2)
            + d_model * heads * k_dim + heads * k_dim
            + (2 * heads * k_dim if query_norm == "batch" else 2 * k_dim) + 2 * d_model)


def memory_preflight(device: torch.device, batch_size: int, *, vocab_size: int = 110,
                     d_model: int = 512, n_layers: int = 8, n_heads: int = 8,
                     max_len: int = 384, **settings) -> float:
    """Section 3.3's worst-case check, BEFORE any training: one full micro-batch at
    max_len, forward and backward under training's autocast, then a clip and an
    optimiser step (which allocates AdamW's state -- large here). Returns the peak
    memory in GB (0.0 on a CPU); raises if it does not fit."""
    cuda = device.type == "cuda"
    cfg = MemConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                    d_ff=4 * d_model, max_len=max_len, mem_pad_id=0, **{**REGISTERED, **settings})
    model = MemURLTransformer(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-12, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    x = torch.randint(4, vocab_size, (batch_size, max_len), device=device)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    with torch.amp.autocast("cuda", enabled=cuda):
        loss = model.loss(x, 0)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(opt)
    scaler.update()
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if cuda else 0.0
    del model, opt, x, loss
    if cuda:
        torch.cuda.empty_cache()
    return peak


# --------------------------------------------------------------------------- #
# V2-P1b's audit: no batch coupling, training mode = generation mode
# --------------------------------------------------------------------------- #

def batch_coupling_check(model: nn.Module, rows, pad_id: int, groups: int = 2) -> dict:
    """V2-P1b (PREREG_V2_P1B.md section 3): what broke P1, measured directly.

    `rows` are token-id lists of different lengths. Each row's logits at its own positions
    are computed (a) alone, (b) inside one batch with every other row, (c) inside batches of
    rows grouped by length, each in evaluation mode, and (d) in training mode with every
    dropout switched off (so only batch statistics could differ). Returns the largest
    absolute logit differences against (a), and whether any buffer (a running statistic)
    changed during the training-mode pass. A model with no batch coupling gives float noise
    everywhere and changes no buffer. Restores the model's mode and buffers."""
    import copy
    was_training = model.training
    buffers = copy.deepcopy({k: v for k, v in model.state_dict().items()
                             if k in dict(model.named_buffers())})

    def run(batch_rows):
        width = max(len(r) for r in batch_rows)
        ids = torch.tensor([r + [pad_id] * (width - len(r)) for r in batch_rows])
        with torch.no_grad():
            out = model(ids).float()
        return [out[i, :len(r)] for i, r in enumerate(batch_rows)]

    model.eval()
    alone = [run([r])[0] for r in rows]
    together = run(rows)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    grouped = [None] * len(rows)
    size = -(-len(rows) // groups)
    for g in range(0, len(order), size):
        idx = order[g:g + size]
        for i, out in zip(idx, run([rows[i] for i in idx])):
            grouped[i] = out
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    train_mode = run(rows)
    changed = [k for k, v in model.state_dict().items() if k in buffers and not torch.equal(v, buffers[k])]
    model.load_state_dict({**model.state_dict(), **buffers})
    model.train(was_training)

    def gap(xs):
        return max(float((a - b).abs().max()) for a, b in zip(xs, alone))
    return {"batch_vs_alone": gap(together), "length_grouped_vs_alone": gap(grouped),
            "training_mode_vs_alone": gap(train_mode), "buffers_changed_in_training": changed}
