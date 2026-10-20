"""
research/v2/v2_p2_train.py

V2-P2's training (research/v2/PREREG_V2_P2.md sections 3, 4.4, 4.5 and 5; Amendment 1):
`uie.kaggle_train.run` with `e16_cap`'s recipe, and these differences:

- **The tokenizer:** the piece tokenizer with the arm's tables, in place of `e16_cap`'s
  character tokenizer (which it extends without changing any id).
- **The start:** `e16_cap`'s start plus the piece rows, drawn N(0, 0.02) from seeded
  generators (rest rows, then host rows).
- **The rows and batches:**
  - rows are kept by their *character* length (`e16_cap`'s rows);
  - batches are sorted by character length;
  - the batch digest covers the character encodings, so V-batch is `e16_cap`'s `adbddc2a...`.
- **The name scope (section 4.5):** page rows train no content inside SUF, LBL, SUB or USR.
  This is `field_mask`'s own corpus scope for the structure corpus; name rows keep every
  field.
- **BPE-dropout** (p = 0.1) on every row of a training batch, stream rows included. Each
  row's generator is seeded from (20261071, the batch's index, the row's position).
- **Stream 1's query positions** are counted in characters (V-mass).
- **The per-field validation NLL** is in nats per character.
- **The vocabulary head is applied in chunks of positions** (a piece model only). The loss is
  the same per-target cross-entropy over the same mass. Only its memory differs: rows x 384
  x 64,110 logits never exist at once, which a T4 could not hold (Amendment 3). In training
  each chunk's logits are recomputed in the backward pass; validation and fit need no
  gradients.

Nothing in `uie/` changes: `run` patches these seams for the call and restores them.
With no tables, no dropout and no scope, the patched run is the unpatched one, bit for bit
(V-equiv, tests/test_v2_p2.py).
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import sys
from array import array
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

import uie.kaggle_train as KT  # noqa: E402
import uie.train as UT  # noqa: E402
from uie.model import URLTransformer  # noqa: E402
from uie.tokenizer import _MARKERS  # noqa: E402
from v2_p2_tokenizer import HybridTokenizer, rules_sha256, table_sha256  # noqa: E402

#: Amendment 1 (section 18.2) and Amendment 2's seeds
DROP_SEED = 20261071
REST_ROWS_SEED = 20261074
HOST_ROWS_SEED = 20261075
#: logits computed at once by the chunked head (about 0.7 GB of them in training)
CHUNK_LOGITS = 1 << 26

#: the run's settings while patched (one run at a time)
_STATE: Dict[str, object] = {}
_ORIG: Dict[str, object] = {}


# --------------------------------------------------------------------------- #
# the chunked vocabulary head (a piece model's loss, validation, fit and generation)
# --------------------------------------------------------------------------- #

def hidden_states(model, ids: torch.Tensor) -> torch.Tensor:
    """The model's final hidden states (`ln_f`'s output), with no vocabulary head applied:
    `URLTransformer.forward` itself, with the head standing aside for the call."""
    head = model.head
    model.head = torch.nn.Identity()
    try:
        return model(ids)
    finally:
        model.head = head


def chunked_ce(model, h: torch.Tensor, targets: torch.Tensor, pad_id: int,
               grad: bool) -> torch.Tensor:
    """Per-target cross-entropy, shape `targets.shape`: the head (`F.linear` with the tied
    table, as `nn.Linear` without bias computes it) and `F.cross_entropy` applied to a
    chunk of positions at a time. With `grad`, each chunk is checkpointed: its logits are
    dropped after the forward pass and recomputed in the backward pass."""
    from torch.utils.checkpoint import checkpoint
    w = model.head.weight
    flat_h, flat_t = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
    step = max(1, CHUNK_LOGITS // w.shape[0])

    def piece(hh, tt):
        return torch.nn.functional.cross_entropy(torch.nn.functional.linear(hh, w), tt,
                                                 ignore_index=pad_id, reduction="none")
    out = []
    for s in range(0, flat_t.numel(), step):
        hh, tt = flat_h[s:s + step], flat_t[s:s + step]
        out.append(checkpoint(piece, hh, tt, use_reentrant=False) if grad else piece(hh, tt))
    return torch.cat(out).view(targets.shape)


class PieceTransformer(URLTransformer):
    """`URLTransformer` whose `loss` applies the head in chunks. Its weights, state dict and
    forward are `URLTransformer`'s, so a checkpoint loads into either class."""

    def loss(self, ids: torch.Tensor, pad_id: int,
             weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`URLTransformer.loss`, the same arithmetic, with the head in chunks."""
        targets = ids[:, 1:]
        per_token = chunked_ce(self, hidden_states(self, ids[:, :-1]), targets, pad_id,
                               grad=torch.is_grad_enabled())
        if weights is None:
            return per_token.sum() / (targets != pad_id).sum()
        mask = weights * (targets != pad_id)
        total = mask.sum()
        if total == 0:
            return per_token.sum() * 0.0
        return (per_token * mask).sum() / total


# --------------------------------------------------------------------------- #
# the tokenizer factory and the start
# --------------------------------------------------------------------------- #

class _Factory:
    """Stands in for `URLTokenizer` inside `uie.kaggle_train`: builds e16_cap's character
    tokenizer exactly as the run would, then the piece tokenizer over it."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, urls, min_count: int = 1, **kw):
        base = _ORIG["URLTokenizer"](**self.kwargs).fit(urls, min_count=min_count, **kw)
        return HybridTokenizer(base_state=base.state_dict(), rest=_STATE.get("rest"),
                               host=_STATE.get("host"), dropout=float(_STATE.get("dropout") or 0.0))

    @staticmethod
    def from_state_dict(state: dict):
        return _ORIG["URLTokenizer"].from_state_dict(state)


def with_pieces(base_model, cfg, tokenizer: HybridTokenizer):
    """`base_model`'s weights plus the piece rows (section 3): a new model whose first
    `tokenizer.n_base` vocabulary rows are the base's, exactly."""
    from uie.free import weights_sha256
    model = PieceTransformer(cfg)
    old = base_model.state_dict()
    d = old["tok_emb.weight"].shape[1]
    extra = []
    prints = {}
    for name, seed in (("rest", REST_ROWS_SEED), ("host", HOST_ROWS_SEED)):
        n = len(tokenizer.ids_of_table(name))
        if not n:
            prints[f"{name}_rows_sha256"] = None
            continue
        gen = torch.Generator().manual_seed(seed)
        rows = (torch.randn((n, d), generator=gen) * 0.02).to(old["tok_emb.weight"].dtype)
        prints[f"{name}_rows_sha256"] = hashlib.sha256(rows.numpy().tobytes()).hexdigest()
        extra.append(rows)
    table = torch.cat([old["tok_emb.weight"]] + extra, dim=0) if extra else old["tok_emb.weight"].clone()
    state = {}
    for name, t in model.state_dict().items():
        if name in ("tok_emb.weight", "head.weight"):
            state[name] = table                      # tied: one table
        else:
            if old[name].shape != t.shape:
                raise ValueError(f"{name}: {tuple(old[name].shape)} -> {tuple(t.shape)}")
            state[name] = old[name].clone()
    model.load_state_dict(state)
    prints["base_init_sha256"] = weights_sha256(base_model)
    prints["pieces"] = {k: len(tokenizer.ids_of_table(k)) for k in ("rest", "host")}
    return model, prints


def _start(cfg, tokenizer, seed):
    base_tok = tokenizer.base if isinstance(tokenizer, HybridTokenizer) else tokenizer
    base_model, prints = _ORIG["new_model_e12"](replace(cfg, vocab_size=base_tok.vocab_size), base_tok, seed)
    if not isinstance(tokenizer, HybridTokenizer) or tokenizer.vocab_size == tokenizer.n_base:
        prints.update({"base_init_sha256": None, "pieces": {"rest": 0, "host": 0}})
        return base_model, prints
    model, piece_prints = with_pieces(base_model, cfg, tokenizer)
    return model, {**prints, **piece_prints}


# --------------------------------------------------------------------------- #
# masks, rows, batches
# --------------------------------------------------------------------------- #

class _NoFrame:
    """The tokenizer as `field_mask` sees an unframed one, so the corpus scope applies."""
    include_frame = False

    def __init__(self, tok) -> None:
        self._tok = tok

    def __getattr__(self, name):
        return getattr(self._tok, name)


def row_mask(ids: Sequence[int], tok, corpus: Optional[str]) -> List[float]:
    """The training mask: `field_mask` as e16_cap trains it, except that with the name scope
    a page row (structure corpus) trains no name-field content (section 4.5)."""
    if corpus is None:
        return [1.0] * (len(ids) - 1)
    if _STATE.get("scope") and corpus == "structure":
        return UT.field_mask(ids, _NoFrame(tok), "structure")
    return UT.field_mask(ids, tok, corpus)


def _encode_corpus(urls, labels, tokenizer, max_len, languages=None, corpora=None,
                   boundary_weight: float = 1.0, hide_fields=(), query_weight: float = 1.0):
    """`uie.train.encode_corpus`, keeping a row by its character length and carrying its
    character encoding (for the digest, the batches and the dropout)."""
    if boundary_weight != 1.0 or query_weight != 1.0:
        raise ValueError("V2-P2 trains at boundary and query weight 1.0")
    out: Dict[str, list] = {}
    dropped = 0
    for i, (url, label) in enumerate(zip(urls, labels)):
        lang = languages[i] if languages is not None else "und"
        corpus = corpora[i] if corpora else None
        frame = UT.CORPUS_FRAME.get(corpus, "domain") if corpora else "domain"
        chars = tokenizer.encode_chars(url, label, lang, frame=frame, hide_fields=hide_fields)
        if len(chars) > max_len:
            dropped += 1
            continue
        ids = tokenizer.pieces_from_chars(chars)
        out.setdefault(label, []).append((ids, row_mask(ids, tokenizer, corpus),
                                          (bytes(chars), corpus, False, 0)))
    return out, dropped


def _char_ids(row) -> Sequence[int]:
    return row[2][0] if isinstance(row, tuple) and len(row) > 2 else (row[0] if isinstance(row, tuple) else row)


def _digest_chars(digest, rows) -> None:
    """`uie.train._digest_rows` over the rows' character encodings."""
    for r in rows:
        digest.update(array("i", list(_char_ids(r))).tobytes())
        digest.update(b"|")
    digest.update(b"#")


def row_seed(batch_index: int, position: int) -> int:
    h = hashlib.blake2b(f"{DROP_SEED}|{batch_index}|{position}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little")


def redrop(row, tok: HybridTokenizer, batch_index: int, position: int):
    """The row re-split under BPE-dropout, its mask recomputed for the new split."""
    ids, mask, extra = row[0], row[1], row[2]
    chars, corpus, stream = list(extra[0]), extra[1], extra[2]
    new = tok.pieces_from_chars(chars, random.Random(row_seed(batch_index, position)), tok.dropout)
    new_mask = (UT.QueryInjector.query_only_mask(new, tok) if stream else row_mask(new, tok, corpus))
    return (new, new_mask, extra)


def _batches(dataset, cfg, pad_id: int, rng: random.Random, dropout=None, inject=None, digest=None):
    """`uie.train.batches`, sorted and digested on character encodings, with BPE-dropout
    applied to every row of a finished batch."""
    pools = {cls: list(seqs) for cls, seqs in dataset.by_class.items()}
    for seqs in pools.values():
        rng.shuffle(seqs)
    cursors = {cls: 0 for cls in pools}

    def draw(cls: str):
        seqs = pools[cls]
        if cursors[cls] >= len(seqs):
            rng.shuffle(seqs)
            cursors[cls] = 0
        cursors[cls] += 1
        return seqs[cursors[cls] - 1]

    classes = dataset.classes
    labels = dataset.label_counts()
    weights = [max(labels.get(c, 1), 1) ** cfg.class_alpha for c in classes]
    tok = _STATE.get("tokenizer")
    p = float(getattr(tok, "dropout", 0.0) or 0.0) if tok is not None else 0.0
    index = 0
    while True:
        window = []
        target = cfg.batch_size * cfg.bucket_window
        while len(window) < target:
            row = draw(rng.choices(classes, weights=weights, k=1)[0])
            window.append(dropout(row) if dropout is not None else row)
        window.sort(key=lambda r: len(_char_ids(r)))
        chunks = [window[i:i + cfg.batch_size] for i in range(0, len(window), cfg.batch_size)]
        rng.shuffle(chunks)
        for chunk in chunks:
            if digest is not None:
                _digest_chars(digest, chunk)
            if inject is not None:
                chunk = inject(chunk)
            if p:
                chunk = [redrop(r, tok, index, k) for k, r in enumerate(chunk)]
            index += 1
            yield UT._pad(chunk, pad_id)


class _Injector(UT.QueryInjector):
    """`uie.train.QueryInjector`, with stream rows carrying their character encoding and their
    query positions counted in characters (V-mass)."""

    @classmethod
    def load(cls, path, tokenizer, train_set, max_len, draw_seed, rate_seed):
        data = Path(path).read_bytes()
        rows: Dict[str, list] = collections.defaultdict(list)
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            chars = tokenizer.encode_chars(r["url"], r["label"], r.get("lang", "und"), frame="domain")
            if len(chars) > max_len:
                raise ValueError(f"a new row is over max_len: {r['url'][:80]}")
            ids = tokenizer.pieces_from_chars(chars)
            mask = cls.query_only_mask(ids, tokenizer)
            if not any(mask):
                raise ValueError(f"a new row has no query positions: {r['url'][:80]}")
            n_query_chars = int(sum(cls.query_only_mask(chars, tokenizer)))
            stratum = r["stratum"] if "_" in r["stratum"] else f"{r['label']}_{r['stratum']}"
            rows[stratum].append((ids, mask, (bytes(chars), "name", True, n_query_chars)))
        probe = cls(tokenizer, {}, {}, draw_seed, rate_seed)
        old: Dict[str, int] = collections.Counter()
        for seqs in train_set.by_class.values():
            for row in seqs:
                s = probe.stratum_of(row[0] if isinstance(row, tuple) else row)
                if s is not None:
                    old[s] += 1
        return cls(tokenizer, dict(rows), dict(old), draw_seed, rate_seed, hashlib.sha256(data).hexdigest())

    def __call__(self, chunk):
        extra = []
        for row in chunk:
            s = self.stratum_of(row[0] if isinstance(row, tuple) else row)
            if s is None or s not in self.p:
                continue
            self.counts["old_rows_seen"][s] += 1
            whole = int(self.rate[s])
            if self.rate[s] > whole and self._rate.random() < self.rate[s] - whole:
                whole += 1
            for _ in range(whole):
                new = self._next(s)
                extra.append(new)
                self.counts["appended_rows"][s] += 1
                self.counts["appended_query_positions"][s] += int(new[2][3])
        return chunk + extra if extra else chunk


# --------------------------------------------------------------------------- #
# fit in nats per character
# --------------------------------------------------------------------------- #

@torch.no_grad()
def field_nll_chars(model, tokenizer, urls, label: str, device, batch: int = 64,
                    min_tokens: int = 200, hide_fields=()) -> Dict[str, float]:
    """`uie.evaluation.field_nll`, with each target's loss divided over the characters it
    writes, so a piece engine's fit compares with a character engine's. For a character
    tokenizer it is `field_nll` exactly."""
    marker_of = {tokenizer.marker_id(m): m for m in _MARKERS}
    n_base = getattr(tokenizer, "n_base", None)
    piece_text = getattr(tokenizer, "piece_text", {})
    chunked = n_base is not None and tokenizer.vocab_size > n_base
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    model.eval()
    for i in range(0, len(urls), batch):
        chunk = urls[i:i + batch]
        encoded = [(tokenizer.encode(u, label, hide_fields=hide_fields) if hide_fields
                    else tokenizer.encode(u, label))[:model.cfg.max_len] for u in chunk]
        width = max(len(e) for e in encoded)
        ids = torch.tensor([e + [tokenizer.pad_id] * (width - len(e)) for e in encoded],
                           dtype=torch.long, device=device)
        if chunked:                     # a piece model: the head in chunks (no ignored target)
            losses = chunked_ce(model, hidden_states(model, ids)[:, :-1], ids[:, 1:], -100,
                                grad=False).float().cpu()
        else:
            logits = model(ids)
            losses = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                reduction="none").view(ids.shape[0], -1).float().cpu()
        for row, sequence in enumerate(encoded):
            current = None
            for t in range(len(sequence) - 1):
                target = sequence[t + 1]
                if target in marker_of:
                    current = marker_of[target]
                    continue
                if current is None or target == tokenizer.eos_id:
                    continue
                n = len(piece_text[target]) if n_base is not None and target >= n_base else 1
                totals[current] = totals.get(current, 0.0) + float(losses[row, t])
                counts[current] = counts.get(current, 0) + n
    return {f: totals[f] / counts[f] for f in sorted(counts) if counts[f] >= min_tokens}


@torch.no_grad()
def shape_nll_chunked(model, tokenizer, urls, label: str, device, batch: int = 64,
                      hide_fields=()) -> Dict[str, float]:
    """`uie.evaluation.shape_nll` (the NLL of the end token and of the field markers), the same
    numbers from the chunked head: the full log-softmax over 64 rows x 384 positions x the
    piece vocabulary would not fit a T4 after training."""
    from uie.tokenizer import _MARKERS
    markers = {tokenizer.marker_id(m) for m in _MARKERS}
    acc = {"eos": [0.0, 0], "marker": [0.0, 0]}
    model.eval()
    for i in range(0, len(urls), batch):
        encoded = [(tokenizer.encode(u, label, hide_fields=hide_fields) if hide_fields
                    else tokenizer.encode(u, label))[:model.cfg.max_len] for u in urls[i:i + batch]]
        if not encoded:
            continue
        width = max(len(e) for e in encoded)
        ids = torch.tensor([e + [tokenizer.pad_id] * (width - len(e)) for e in encoded],
                           dtype=torch.long, device=device)
        nll = chunked_ce(model, hidden_states(model, ids)[:, :-1], ids[:, 1:], -100, grad=False).float().cpu()
        for row, seq in enumerate(encoded):
            for t in range(1, len(seq)):
                key = ("eos" if seq[t] == tokenizer.eos_id else "marker" if seq[t] in markers else None)
                if key:
                    acc[key][0] += float(nll[row, t - 1])
                    acc[key][1] += 1
    return {f"{k}_nll": v[0] / v[1] for k, v in acc.items() if v[1]}


def boundary_nll_skipped(*args, **kwargs) -> Dict[str, str]:
    """S4's segment-start measure (`uie.evaluation.boundary_nll`) reads '/' as a token of its
    own; a piece model's pieces span it, so the measure is not defined there. Not a P2
    reading; recorded as skipped."""
    return {"skipped": "V2-P2: a character-level measure (pieces span '/')"}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def patch(*, rest: Optional[dict], host: Optional[dict], dropout: float, scope: bool) -> None:
    """Install the P2 seams (see the module docstring). `restore` undoes it."""
    if _ORIG:
        raise RuntimeError("already patched")
    _ORIG.update({"URLTokenizer": KT.URLTokenizer, "new_model_e12": KT.new_model_e12,
                  "encode_corpus": KT.encode_corpus, "field_nll": KT.field_nll,
                  "batches": UT.batches, "QueryInjector": UT.QueryInjector,
                  "shape_nll": KT.shape_nll, "boundary_nll": KT.boundary_nll})
    _STATE.clear()
    _STATE.update({"rest": rest, "host": host, "dropout": float(dropout), "scope": bool(scope)})
    KT.URLTokenizer = _Factory
    KT.new_model_e12 = _start
    KT.encode_corpus = _encode_corpus
    KT.field_nll = field_nll_chars
    UT.batches = _batches
    UT.QueryInjector = _Injector
    if rest is not None or host is not None:      # a piece model: the after-training diagnostics
        KT.shape_nll = shape_nll_chunked
        KT.boundary_nll = boundary_nll_skipped


def restore() -> None:
    if not _ORIG:
        return
    KT.URLTokenizer = _ORIG["URLTokenizer"]
    KT.new_model_e12 = _ORIG["new_model_e12"]
    KT.encode_corpus = _ORIG["encode_corpus"]
    KT.field_nll = _ORIG["field_nll"]
    UT.batches = _ORIG["batches"]
    UT.QueryInjector = _ORIG["QueryInjector"]
    KT.shape_nll = _ORIG["shape_nll"]
    KT.boundary_nll = _ORIG["boundary_nll"]
    _ORIG.clear()
    _STATE.clear()


class _Capture:
    """Records the tokenizer `_run` builds, so the batches can reach it."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, **kwargs):
        made = self.inner(**kwargs)
        orig_fit = made.fit

        def fit(*a, **k):
            tok = orig_fit(*a, **k)
            _STATE["tokenizer"] = tok
            return tok
        made.fit = fit
        return made

    def from_state_dict(self, state):
        return self.inner.from_state_dict(state)


def run(data, out, *, rest: Optional[dict], host: Optional[dict], dropout: float, scope: bool,
        record: Optional[dict] = None, **recipe) -> dict:
    """`uie.kaggle_train.run(data, out, **recipe)` under the P2 seams; the arm's settings and
    `record` are added to train_result.json."""
    patch(rest=rest, host=host, dropout=dropout, scope=scope)
    KT.URLTokenizer = _Capture(_Factory)
    try:
        result = KT.run(data, out, **recipe)
        tok = _STATE.get("tokenizer")
        extra = {"tables": {k: (table_sha256(v) if v is not None else None)
                            for k, v in (("rest", rest), ("host", host))},
                 "rules_sha256": rules_sha256(),
                 "pieces": {k: len(tok.ids_of_table(k)) for k in ("rest", "host")} if tok else None,
                 "vocab_size": getattr(tok, "vocab_size", None), "dropout": float(dropout),
                 "name_scope": bool(scope), **(record or {})}
    finally:
        restore()
    path = Path(out) / "train_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["p2"] = extra
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    result["p2"] = extra
    return result
