"""
uie/train.py

Training for the E2 transformer.

Runs unchanged on a laptop CPU (for correctness checks at toy scale) and on a
Kaggle GPU (for the real run). The only differences are device, batch size and
mixed precision, all resolved automatically.

Two data decisions worth stating
--------------------------------
**Over-long sequences are DROPPED, not truncated.** A truncated URL teaches the
model a URL that ends mid-path with no end token — corrupting 1% of the data
rather than merely losing it. `max_len=256` covers 99.0% of benign and 99.6% of
phishing sequences.

**Batches are bucketed by length.** URL lengths are heavy-tailed (median 69
tokens, p99 246), so naive batching pads most of a batch to the length of its
longest member and wastes the majority of every step on padding. Bucketing sorts
within a shuffled window, which keeps batches near-uniform in length while
preserving randomness.

Class balance
-------------
The real Internet is asymmetric and the corpus records that; the sampler decides
what the model sees. But "balanced" is the wrong target, and by the wrong unit.

Weighting must be by LABEL count, not row count. Rows-per-label differ by class
— the benign name corpus holds exactly one URL per name, phishing 1.47 — so
uniform row sampling still over-exposes phishing labels by that factor. And
50/50 sampling over-exposes them by the full class ratio:

    N_benign = 906,214 names      N_phishing = 192,652 names

    alpha  exposure ratio p:b   phishing gradient share
    0.00 (50/50)      4.70x            50.0%
    0.50              2.17x            31.5%
    1.00              1.00x            17.5%

alpha = 1 equalises exposures per label, so memorisation pressure is the same
for both classes and the ABSOLUTE level is then set by step count and early
stopping — two concerns cleanly separated instead of traded against each other.
"""
from __future__ import annotations

import collections
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from uie.model import ModelConfig, URLTransformer
from uie.tokenizer import URLTokenizer

__all__ = ["TrainConfig", "URLDataset", "train", "save_checkpoint",
           "load_checkpoint", "field_mask", "NAME_FIELDS",
           "STRUCTURE_FIELDS", "CORPUS_SCOPE", "CORPUS_FRAME"]


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 500
    grad_clip: float = 1.0
    eval_every: int = 1_000
    eval_batches: int = 40
    seed: int = 20260803
    #: Class sampling temperature over LABEL counts. See `batches`.
    #:
    #: p(class) proportional to n_labels ** class_alpha, so
    #:
    #:     exposures_c = S * N_c**(alpha-1) / sum(N**alpha)
    #:     exposures_p / exposures_b = (N_p / N_b) ** (alpha - 1)
    #:
    #: alpha = 1 is the UNIQUE exposure-neutral point: every label in every
    #: class is seen the same number of times, so memorisation pressure is
    #: identical across classes. Any alpha < 1 deliberately over-exposes the
    #: smaller class, which is the memorisation mechanism E2.2 measured
    #: (phishing's label field had the widest train/val gap in the model at
    #: +1.227). The old `balance_classes=True` is alpha = 0.
    class_alpha: float = 1.0
    #: Window (in batches) used for length bucketing.
    bucket_window: int = 64
    #: S4. Loss weight on path SEGMENT STARTS; 1.0 is exactly E5. Recorded in
    #: the checkpoint so a trained model always says which regime produced it.
    boundary_weight: float = 1.0
    #: E7. Probability that each presence control (`<P:*>`, `<Q:*>`) is shown
    #: as `?` when a row is batched, re-drawn every time. 0.0 is exactly the
    #: pre-E7 loop. Needs a tokenizer with field controls.
    field_dropout: float = 0.0
    #: E9 (research/native/PREREG_E9.md). "batch" is the objective every run
    #: before E9 trained on: each step is its batch's mean over scored tokens,
    #: and with length bucketing that weights every ROW about equally, so a
    #: token of a long row counts ~1/L as much (FINDING_LENGTH_WEIGHTING.md).
    #: "fixed" multiplies every CLIPPED gradient by T_b / C -- the batch's
    #: scored-token mass over a fixed constant -- so every scored token weighs
    #: the same in every batch. Clipping is unchanged and acts first (D4).
    loss_norm: str = "batch"
    #: C for "fixed". None: batch_size x the expected scored-token mass per draw
    #: under this sampler (`expected_scored_tokens`).
    loss_constant: Optional[float] = None
    #: E11 (research/native/PREREG_E11.md D3). None is every earlier run: the cap
    #: is `grad_clip` at every step. Set, the cap is `grad_clip` for the first
    #: `grad_clip_switch` steps and `grad_clip_late` after them. PC1 found that at
    #: 1.0 the cap binds on the shortest batches alone late in training -- every
    #: batch under 19 scored tokens per row, none over 23.5 -- which discounts the
    #: rows in them (mostly bare hosts) and tilts every presence decision toward
    #: "more follows" (research/native/PC1_RESULTS.md).
    grad_clip_late: Optional[float] = None
    grad_clip_switch: int = 1_000
    #: E11 (D4). False is every earlier run: each presence control is hidden
    #: independently with probability `field_dropout`. True: both are hidden
    #: together with that probability and otherwise both shown -- the only two
    #: states generation uses (free: both hidden; control: both shown).
    field_dropout_joint: bool = False
    #: E13 C (research/e13/PREREG_E13.md section 2.1). None is every earlier run.
    #: Set, the prefix artifact `ContextSwap` reads; `context_p` is the swap
    #: probability per draw and `context_seed` its own stream. Written to a
    #: checkpoint only when set, so earlier layouts stay exactly what they were.
    context_prefixes: Optional[str] = None
    context_p: float = 0.0
    context_seed: int = 20261009
    #: E14 Q (research/e14/PREREG_E14.md section 2.1). 1.0 is every earlier run.
    #: The weight itself lives in the training rows' masks (`field_mask`, applied
    #: when the corpus is encoded); this records it, and with it set `train`
    #: reports the clip bands on the w = 1 mass (scored positions) and the query
    #: share of the scored mass. Written to a checkpoint only when set.
    query_weight: float = 1.0
    #: E15-B (research/e15/PREREG_E15.md section 10.4). None is every earlier run. Set,
    #: the path of the new query-bearing rows (JSON lines: url, label, stratum) that
    #: `QueryInjector` appends alongside the old rows; the two seeds drive its draws and,
    #: where a stratum is short, its coin. Written to a checkpoint only when set.
    qry_extra: Optional[str] = None
    qry_extra_draw_seed: int = 20261007
    qry_extra_rate_seed: int = 20261008
    #: E16 (research/e16/PREREG_E16.md section 3). 1 is every earlier run. K > 1: each
    #: batch is split into K consecutive parts, each part's loss weighted by its share
    #: of the batch's scored mass, and the parts' gradients summed before the clip and
    #: the step -- the batch's own gradient, in less memory (`_accumulate`). The batches
    #: themselves are unchanged. Written to a checkpoint only when it is not 1.
    accum_steps: int = 1


class URLDataset(Dataset):
    """Pre-encoded (ids, field-mask) pairs, grouped by class.

    The mask travels WITH the row because it depends on which corpus the row
    came from, and that is not recoverable from the token sequence alone — a
    name-corpus URL and a structure-corpus URL can be textually identical.
    """

    def __init__(self, encoded: Dict[str, List[List[int]]],
                 label_counts: Optional[Dict[str, int]] = None) -> None:
        self.by_class = encoded
        self._label_counts = dict(label_counts) if label_counts else None
        self.classes = sorted(encoded)
        self.flat = [row for cls in self.classes for row in encoded[cls]]

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, idx: int) -> List[int]:
        return self.flat[idx]

    def counts(self) -> Dict[str, int]:
        return {cls: len(seqs) for cls, seqs in self.by_class.items()}

    def label_counts(self) -> Dict[str, int]:
        """Distinct NAMES per class — the unit class sampling weights by.

        Defaults to row counts when not supplied, which is correct only if every
        row is a distinct name. It is exactly true of the NAME corpora (one URL
        per name by construction) and false of the structure corpora, so the
        caller passes it explicitly whenever both are mixed.
        """
        return self._label_counts or self.counts()


def encode_corpus(urls: Sequence[str], labels: Sequence[str],
                  tokenizer: URLTokenizer, max_len: int,
                  languages: Optional[Sequence[str]] = None,
                  corpora: Optional[Sequence[str]] = None,
                  boundary_weight: float = 1.0,
                  hide_fields: Sequence[str] = (),
                  query_weight: float = 1.0
                  ) -> Tuple[Dict[str, List[Tuple[List[int], List[float]]]], int]:
    """Encode to (ids, field-mask) pairs, dropping anything over `max_len`.

    `corpora` names the source of each row ("name" or "structure"). Absent, every
    row is trained on every field, which is the pre-E2.3 behaviour.

    `boundary_weight` is S4; it reaches the loss only through the mask, so a row
    with no path is unaffected no matter what it is set to.

    `query_weight` is E14 Q (`field_mask`), the same way: a row without a query
    is unaffected. It needs field masks, so it needs `corpora`.
    """
    if query_weight != 1.0 and corpora is None:
        raise ValueError("query_weight acts through the field mask: it needs `corpora`")
    out: Dict[str, List[Tuple[List[int], List[float]]]] = {}
    dropped = 0
    for i, (url, label) in enumerate(zip(urls, labels)):
        lang = languages[i] if languages is not None else "und"
        # The frame is derived from the corpus, never guessed. A row encoded
        # under the wrong frame teaches the wrong conditional, which is the
        # failure the token exists to prevent.
        frame = CORPUS_FRAME.get(corpora[i], "domain") if corpora else "domain"
        ids = (tokenizer.encode(url, label, lang, frame=frame,
                                hide_fields=hide_fields)
               if hide_fields else tokenizer.encode(url, label, lang, frame=frame))
        if len(ids) > max_len:
            dropped += 1
            continue
        if corpora is None:
            mask = [1.0] * (len(ids) - 1)
        else:
            mask = field_mask(ids, tokenizer, corpora[i], boundary_weight, query_weight)
        out.setdefault(label, []).append((ids, mask))
    return out, dropped


# --------------------------------------------------------------------------- #
# Field-scoped loss
# --------------------------------------------------------------------------- #

#: Which fields each corpus is a valid teacher for.
#:
#: The two benign corpora are sampled differently on purpose — names want every
#: domain weighted equally, paths want many pages from the same site — and each
#: is therefore WRONG about the other's fields. The name corpus holds one URL
#: per domain, so single-page sites dominate and its path depth is 1.29 with
#: 27.7% bare against a population of 2.30 / 2.9%. Training paths on it would
#: rebuild E2.1's bare-homepage artifact at ten times the scale. Symmetrically,
#: the structure corpus covers only multi-page sites, a biased 4.6% of the
#: domain population, so its NAMES are not representative.
NAME_FIELDS = ("SUF", "LBL", "SUB", "USR")
STRUCTURE_FIELDS = ("PTH", "QRY", "PRT", "FRG")
CORPUS_SCOPE = {"name": NAME_FIELDS, "structure": STRUCTURE_FIELDS}

#: Which SAMPLING FRAME each corpus draws in. The corpora are not two halves of
#: one distribution: `name` takes one URL per registrable name over every name
#: (the domain frame), `structure` takes up to 40 URLs from one multi-page site
#: (the page frame). Measured on the benign corpus, they disagree systematically
#: about paths — 72.3% carry one at a conditional 28.3 characters in the domain
#: frame, 97.2% at 39.7 in the page frame.
#:
#: When the tokenizer carries `<F:*>` this is emitted as a control token and
#: `field_mask` trains every field on every row: the scoping above exists ONLY
#: because an unframed model cannot tell the two apart, so naming the frame
#: removes its reason to exist.
CORPUS_FRAME = {"name": "domain", "structure": "page"}


def field_mask(ids: Sequence[int], tokenizer: URLTokenizer,
               corpus: str, boundary_weight: float = 1.0,
               query_weight: float = 1.0) -> List[float]:
    """Per-target weights aligned with `ids[1:]`: 1 to train, 0 to ignore.

    S4 -- `boundary_weight` up-weights SEGMENT STARTS inside the path: the
    target immediately after `<PTH>` or after a `/`. Default 1.0 reproduces E5
    exactly, so the knob is reversible by configuration rather than by revert.

    E14 Q -- `query_weight` up-weights every step taken WHILE THE QUERY IS OPEN
    (research/e14/PREREG_E14.md section 2.1): from the first query character
    through the step that predicts the query's end (<FRG> or the end token), but
    not the step that predicts <QRY> itself -- that is the presence decision, and
    it belongs to the field before. The whole step is weighted, never particular
    targets (the S4b lesson below), so the data's own ratio of "another
    character" to "the query ends here" is untouched. Default 1.0 reproduces
    every earlier mask exactly.

    WHY THE LOSS AND NOT THE DECODER. S3 measured over-confidence of +0.329 at
    those positions against +0.032 elsewhere in the same segments, and a
    temperature sweep then showed only ~0.03 nats of that is recoverable by
    rescaling -- the distribution's SHAPE is wrong, not merely its sharpness.
    Cross-entropy is a STRICTLY PROPER scoring rule: its unique minimiser over
    distributions is the true conditional, so gradient pressure moves the whole
    distribution rather than its scale. That is the one mechanism available here
    that can reach a shape error, which is why S4 is a loss change.

    WHY WEIGHT RATHER THAN CAPACITY. Segment starts are ~6% of in-path targets,
    so under the mass-normalised loss in `URLTransformer.loss` they command ~6%
    of the in-path gradient. Capacity was separately ruled out as the binding
    constraint. Re-weighting changes allocation, not size.

    THE RISK THIS CREATES, stated because the guardrails exist for it: capacity
    is shared, so whatever the boundary gains may come out of mid-segment
    accuracy -- 94% of positions, currently well calibrated. `boundary_nll`
    reports both, and a mid-segment regression rejects the run.

    MARKERS AND THE END TOKEN ARE ALWAYS TRAINED, on rows from both corpora.
    They carry URL SHAPE — which fields exist at all — and shape is not a
    content question. Scoping them to the structure corpus would mean the model
    only ever learns "a path follows a label" from multi-page sites, which have
    a path essentially always, and generated URLs would come out with a bare
    rate near zero against a real 2.9% benign / 47.4% phishing. Shape must be
    learned from the whole corpus even though content is not.

    So only the CHARACTERS inside a field are scoped.

    UNLESS THE FRAME IS NAMED. Scoping exists for exactly one reason: the model
    cannot tell which sampling frame a row was drawn in, so name-corpus paths
    (benign depth 1.29 against a page-frame 2.30) would contaminate its idea of
    what a path looks like. A tokenizer carrying `<F:*>` removes that reason —
    the model conditions on the frame and learns P(path | frame) directly — so
    every row may teach every field.

    That is not merely permission, it is the point. Under scoping, path CONTENT
    trains on 1.65M benign rows and only 46k phishing rows, while path SHAPE
    trains on all of them; two teachers disagreeing about one decision is what
    produced the empty-path bug (E4) and the short-path bias measured after it.
    Naming the frame gives content 2.55M and 239k rows respectively and puts
    both under one teacher.
    """
    from uie.tokenizer import _MARKERS

    if getattr(tokenizer, "include_frame", False):
        scope = set(NAME_FIELDS) | set(STRUCTURE_FIELDS)
    else:
        scope = set(CORPUS_SCOPE.get(corpus, ()))
    marker_of = {tokenizer.marker_id(m): m for m in _MARKERS}
    # A MISSING "/" RAISES rather than degrading quietly. Silently dropping it
    # would leave the weight applying only after `<PTH>` -- one position per URL
    # instead of one per segment -- which still trains, still converges, and
    # would be read as "S4 barely helped" rather than "S4 never ran".
    after_boundary: set = set()
    if boundary_weight != 1.0:
        if "/" not in tokenizer.stoi:
            raise KeyError("boundary_weight needs '/' in the vocabulary; "
                           "this tokenizer has not been fitted on URLs")
        after_boundary = {tokenizer.marker_id("PTH"), tokenizer.stoi["/"]}

    # E8 (PREREG_E8 section 3): empty for every earlier tokenizer, so their
    # masks are unchanged.
    e8_shape: set = set()
    e8_ends_field: set = set()
    if getattr(tokenizer, "presence_plan", False):
        e8_shape.update(t for slot in tokenizer.plan_slots() for t in slot)
    if getattr(tokenizer, "root_slash", False):
        e8_shape.add(tokenizer.rs_id)
        e8_ends_field.add(tokenizer.rs_id)

    weights: List[float] = []
    current: Optional[str] = None
    for position, target in enumerate(ids[1:], start=1):
        # THE POSITION IS WEIGHTED, NOT THE TARGET (S4b).
        #
        # Decided BEFORE `current` is updated, and independently of what the
        # target turns out to be. S4 computed the boundary weight inside the
        # content branch, so a boundary whose target was EOS or a marker fell
        # through to weight 1.0 while a boundary whose target was a character
        # got the full weight. Measured on the corpus, 17.50% of boundary
        # positions are non-content -- 16.64% EOS, overwhelmingly the
        # trailing-slash form `example.com/docs/`.
        #
        # That made the loss a biased sample of the boundary decision: at w=4
        # the model saw four times the evidence for "a character follows" and
        # unamplified evidence for "the URL ends here". It learned that
        # segments START rather than what starts them. Measured consequences:
        # benign EOS NLL 1.261 -> 1.423 and path depth 1.769 -> 2.165 against a
        # target of 1.782, breaking the one structural metric E5 had right.
        #
        # Weighting the whole step leaves the content-versus-stop ratio at that
        # step exactly as the data says it is, and raises only the step's
        # importance -- which is what "the model is miscalibrated HERE" asks for.
        at_boundary = (current == "PTH" and "PTH" in scope
                       and ids[position - 1] in after_boundary)
        # E14 Q: the same rule for the query -- the open field decides, before
        # `current` moves on, whatever the target is.
        at_query = query_weight != 1.0 and current == "QRY" and "QRY" in scope

        if target in marker_of:
            current = marker_of[target]
            weight = 1.0                     # shape: always trained
        elif target in (tokenizer.eos_id, tokenizer.bos_id):
            weight = 1.0                     # shape: always trained
        elif target in e8_shape:
            # E8: the presence plan (a TARGET, unlike E7's supplied controls --
            # free mode samples it) and the root slash. Both are shape.
            weight = 1.0
            if target in e8_ends_field:
                current = None               # nothing follows <RS> inside a field
        elif target == tokenizer.pad_id:
            weight = 0.0
        elif current is None:
            # Control tokens, before any field marker. These are SUPPLIED at
            # generation and never sampled, so spending capacity on predicting
            # them buys nothing. Excluded deliberately, not by omission.
            weight = 0.0
        else:
            weight = 1.0 if current in scope else 0.0

        # Guarded on `weight` so padding and out-of-scope targets stay at zero:
        # S4b may re-weight a trained position, never switch one on.
        if weight and at_boundary:
            weight *= boundary_weight
        if weight and at_query:
            weight *= query_weight
        weights.append(weight)
    return weights


def _pad(batch, pad_id: int):
    """Pad a batch of (ids, mask) pairs. Returns (ids, weights).

    Plain id-sequences are still accepted so `evaluate` and older callers work.
    """
    if batch and isinstance(batch[0], tuple):
        seqs = [b[0] for b in batch]
        masks = [b[1] for b in batch]
    else:
        seqs, masks = list(batch), None
    width = max(len(s) for s in seqs)
    ids = torch.tensor([list(s) + [pad_id] * (width - len(s)) for s in seqs],
                       dtype=torch.long)
    if masks is None:
        return ids, None
    weights = torch.tensor(
        [list(m) + [0.0] * (width - 1 - len(m)) for m in masks],
        dtype=torch.float)
    return ids, weights


class FieldDropout:
    """E7 control dropout: hide each presence control with probability `p`.

    Applied as a row is BATCHED, not when it is encoded, so every row teaches
    both the conditional and the unconditional mode over the course of
    training. It uses its OWN random stream, so the batch composition is exactly
    what the same seed would draw without it. Rows are copied before a token is
    swapped -- the dataset's lists are shared across epochs and must stay
    intact. Lengths never change, so length bucketing is unaffected, and the
    field mask is untouched: control positions are weight 0 as targets anyway.
    """

    def __init__(self, tokenizer: URLTokenizer, p: float, seed: int,
                 joint: bool = False) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"field_dropout must be in [0, 1]; got {p}")
        self.p = p
        self.joint = bool(joint)
        self.positions = tokenizer.field_control_positions()
        self.hidden = {f: tokenizer.field_control_id(f, None)
                       for f in self.positions}
        self.rng = random.Random(seed)

    def __call__(self, row):
        ids, mask = row
        if self.joint:
            # E11 (PREREG_E11 D4): one draw per row -- every control hidden, or none.
            if self.rng.random() >= self.p:
                return row
            swapped = list(ids)
            for field, pos in self.positions.items():
                swapped[pos] = self.hidden[field]
            return (swapped, mask)
        swapped = None
        for field, pos in self.positions.items():
            if self.rng.random() < self.p:
                if swapped is None:
                    swapped = list(ids)
                swapped[pos] = self.hidden[field]
        return (swapped, mask) if swapped is not None else row


class QueryInjector:
    """E15-B (research/e15/PREREG_E15.md section 10.4): new query-bearing rows,
    injected alongside the old ones.

    Whenever a batch holds an old row of an admitted stratum -- (class, presence
    state) of a domain-frame row with a query, read from the row's own tokens --
    the next new row of that stratum is appended to the same batch with probability
    p_s = min(1, N_new,s / N_old,s), so every row, old or new, is drawn at the old
    rows' per-row rate. The draws cycle through a per-stratum shuffle (their own
    generator); the coin, used only where p_s < 1, has another.

    A new row supervises QUERY CONTENT ONLY, at weight 1.0: PREREG_E14 section 2.1's
    positions (exactly the ones `field_mask(query_weight=...)` scales), every other
    position 0, so its host, path and plan are context. Old rows are never touched.
    Under the fixed objective (C pinned) the appended targets ADD query mass and
    leave every old row's gradient contribution as it was.
    """

    def __init__(self, tokenizer: URLTokenizer, rows: Dict[str, List[tuple]],
                 old_counts: Dict[str, int], draw_seed: int, rate_seed: int,
                 source_sha256: Optional[str] = None) -> None:
        self.rows = {s: r for s, r in rows.items() if r}
        self.old_counts = dict(old_counts)
        missing = [s for s in self.rows if not self.old_counts.get(s)]
        if missing:
            raise ValueError(f"strata with new rows but no old rows: {missing}")
        # E17 (research/e17/PREREG_E17.md section 3.2): the rate r = N_new / N_old is no
        # longer capped at 1. Each old draw appends floor(r) new rows, plus one more with
        # probability r - floor(r) (the coin). At r <= 1 that is the E15 injector bit for
        # bit: the same coin, drawn at the same moments, with the same outcome.
        self.rate = {s: len(r) / self.old_counts[s] for s, r in self.rows.items()}
        self.p = {s: min(1.0, r) for s, r in self.rate.items()}
        self._draw = random.Random(draw_seed)
        self._rate = random.Random(rate_seed)
        self._order: Dict[str, List[int]] = {s: [] for s in self.rows}
        self.source_sha256 = source_sha256
        self.counts = {"old_rows_seen": collections.Counter(),
                       "appended_rows": collections.Counter(),
                       "appended_query_positions": collections.Counter()}
        self._domain = tokenizer.stoi.get("<F:domain>")
        self._cls = {tokenizer.stoi[f"<C:{c}>"]: c for c in tokenizer.classes}
        self._p1 = tokenizer.stoi["<P:1>"]
        self._p0 = tokenizer.stoi["<P:0>"]
        self._q1 = tokenizer.stoi["<Q:1>"]
        self._lbl = tokenizer.marker_id("LBL")

    @staticmethod
    def query_only_mask(ids: Sequence[int], tokenizer: URLTokenizer) -> List[float]:
        """1.0 exactly where `field_mask(query_weight=...)` scales, 0 elsewhere."""
        w = field_mask(ids, tokenizer, "name", 1.0, 2.0)
        return [1.0 if x == 2.0 else 0.0 for x in w]

    def stratum_of(self, ids: Sequence[int]) -> Optional[str]:
        """'<class>_P0Q1' or '<class>_P1Q1' for a domain-frame row with a query."""
        if len(ids) < 4 or ids[2] != self._domain:
            return None
        cls = self._cls.get(ids[1])
        p = q = None
        for t in ids[3:]:
            if t == self._lbl:
                break
            if t == self._p1 or t == self._p0:
                p = t == self._p1
            elif t == self._q1:
                q = True
        if cls is None or not q or p is None:
            return None
        return f"{cls}_{'P1Q1' if p else 'P0Q1'}"

    @classmethod
    def load(cls, path, tokenizer: URLTokenizer, train_set: "URLDataset", max_len: int,
             draw_seed: int, rate_seed: int) -> "QueryInjector":
        """Read the stream (JSON lines: url, label, stratum), encode it as training
        encodes a domain-frame row, and count the old rows of every stratum."""
        import hashlib
        data = Path(path).read_bytes()
        rows: Dict[str, List[tuple]] = collections.defaultdict(list)
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ids = tokenizer.encode(r["url"], r["label"], r.get("lang", "und"), frame="domain")
            if len(ids) > max_len:
                raise ValueError(f"a new row is over max_len: {r['url'][:80]}")
            mask = cls.query_only_mask(ids, tokenizer)
            if not any(mask):
                raise ValueError(f"a new row has no query positions: {r['url'][:80]}")
            stratum = r["stratum"] if "_" in r["stratum"] else f"{r['label']}_{r['stratum']}"
            rows[stratum].append((ids, mask))
        probe = cls(tokenizer, {}, {}, draw_seed, rate_seed)
        old: Dict[str, int] = collections.Counter()
        for seqs in train_set.by_class.values():
            for row in seqs:
                s = probe.stratum_of(row[0] if isinstance(row, tuple) else row)
                if s is not None:
                    old[s] += 1
        return cls(tokenizer, dict(rows), dict(old), draw_seed, rate_seed,
                   hashlib.sha256(data).hexdigest())

    def _next(self, s: str) -> tuple:
        order = self._order[s]
        if not order:
            order.extend(range(len(self.rows[s])))
            self._draw.shuffle(order)
        return self.rows[s][order.pop()]

    def __call__(self, chunk: List) -> List:
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
                self.counts["appended_query_positions"][s] += int(sum(new[1]))
        return chunk + extra if extra else chunk

    def summary(self) -> str:
        return "; ".join(f"{s}: {len(self.rows[s]):,} new / {self.old_counts[s]:,} old, "
                         f"p = {self.p[s]:.4f}"
                         + (f", rate {self.rate[s]:.4f} per old draw" if self.rate[s] > 1 else "")
                         for s in sorted(self.rows))

    def record(self) -> dict:
        return {"source_sha256": self.source_sha256,
                "strata": {s: {"N_new": len(self.rows[s]), "N_old": self.old_counts[s],
                               "p": self.p[s], "rate": self.rate[s],
                               "old_rows_seen": self.counts["old_rows_seen"][s],
                               "appended_rows": self.counts["appended_rows"][s],
                               "appended_query_positions": self.counts["appended_query_positions"][s]}
                           for s in sorted(self.rows)},
                "old_counts_all_strata": dict(self.old_counts)}


class ContextSwap:
    """E13 C (research/e13/PREREG_E13.md section 2.1): training on the model's own
    context, for the rows the artifact lists.

    Each time a listed row is drawn, with probability `p`, everything before its
    `<QRY>` is replaced by one of its model-written prefixes (chosen uniformly),
    which ends with the model's own `<QRY>`; the real query and everything after
    it are kept. Every target inside the model-written prefix -- including the
    prediction of its `<QRY>` -- gets weight 0, so the model is never taught to
    write those tokens: they are context only. Targets from the first query
    character on keep the real row's weights.

    Like `FieldDropout`, it has its OWN random stream, so training draws exactly
    the rows the same seed draws without it. Rows are matched by their exact
    encoded ids, so a row the artifact does not list -- every page-frame row,
    whose frame token differs -- is never touched. A swapped row longer than
    `max_len` is used real for that draw, and counted.
    """

    def __init__(self, table: Dict[Tuple[int, ...], List[List[int]]], qry_id: int,
                 p: float, seed: int, max_len: int) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"context_p must be in [0, 1]; got {p}")
        for key, prefixes in table.items():
            if qry_id not in key:
                raise ValueError("a listed row has no <QRY>")
            if any(not pre or pre[-1] != qry_id for pre in prefixes):
                raise ValueError("every model-written prefix must end with <QRY>")
        self.table = {k: v for k, v in table.items() if v}
        self.qry, self.p, self.max_len = qry_id, float(p), int(max_len)
        self.rng = random.Random(seed)
        self.counts = {"eligible_draws": 0, "swapped": 0, "too_long": 0}

    @classmethod
    def load(cls, path, tokenizer: URLTokenizer, p: float, seed: int,
             max_len: int) -> "ContextSwap":
        """The artifact `research/e13/e13_prefixes.py` writes: JSON, gzipped or
        plain, with the tokenizer it was built for and, per row, its ids and
        prefixes. A different tokenizer would match nothing silently, so it is
        refused."""
        import gzip
        data = Path(path).read_bytes()
        if data[:2] == bytes((0x1F, 0x8B)):  # gzipped; Kaggle decompresses .gz uploads
            data = gzip.decompress(data)
        blob = json.loads(data.decode("utf-8"))
        if blob["tokenizer"] != tokenizer.state_dict():
            raise ValueError("the prefix artifact was built for a different tokenizer")
        table = {tuple(r["ids"]): [list(x) for x in r["prefixes"]] for r in blob["rows"]}
        return cls(table, tokenizer.marker_id("QRY"), p, seed, max_len)

    def matched(self, dataset: "URLDataset") -> int:
        """How many distinct training rows the artifact lists (V5)."""
        seen = set()
        for row in dataset.flat:
            ids = row[0] if isinstance(row, tuple) else row
            key = tuple(ids)
            if key in self.table:
                seen.add(key)
        return len(seen)

    def __call__(self, row):
        ids, mask = row
        prefixes = self.table.get(tuple(ids))
        if prefixes is None:
            return row
        self.counts["eligible_draws"] += 1
        if self.rng.random() >= self.p:
            return row
        prefix = prefixes[self.rng.randrange(len(prefixes))]
        q = ids.index(self.qry)
        new_ids = list(prefix) + list(ids[q + 1:])
        if len(new_ids) > self.max_len:
            self.counts["too_long"] += 1
            return row
        self.counts["swapped"] += 1
        return (new_ids, [0.0] * (len(prefix) - 1) + list(mask[q:]))


def _compose(*transforms):
    """Row transforms applied in order; None entries are skipped."""
    active = [t for t in transforms if t is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def apply(row):
        for t in active:
            row = t(row)
        return row
    return apply


LOSS_NORMS = ("batch", "fixed")
#: Scored tokens per row at which the gradient diagnostics are banded.
_DIAG_BANDS = ((0, 30), (30, 60), (60, 120), (120, 10 ** 9))


def expected_scored_tokens(dataset: URLDataset, cfg: TrainConfig) -> float:
    """E[scored-token mass of one drawn row] under `batches`: class c with
    probability proportional to label_count_c ** class_alpha, rows uniform
    within the class. A row's mass is the sum of its target weights -- exactly
    what `URLTransformer.loss` divides by."""
    labels = dataset.label_counts()
    w = {c: max(labels.get(c, 1), 1) ** cfg.class_alpha for c in dataset.classes}
    total = sum(w.values())
    out = 0.0
    for c in dataset.classes:
        rows = dataset.by_class[c]
        mass = sum(float(sum(r[1])) if isinstance(r, tuple) else float(len(r) - 1)
                   for r in rows) / len(rows)
        out += w[c] / total * mass
    return out


def batch_mass(ids: torch.Tensor, weights: Optional[torch.Tensor], pad_id: int) -> float:
    """T_b: the batch's scored-token mass, the loss's own denominator."""
    live = ids[:, 1:] != pad_id
    return float((weights * live).sum() if weights is not None else live.sum())


def _scale_gradients(model: URLTransformer, factor: float) -> None:
    """E9 "fixed": the clipped gradient, reweighted to a fixed denominator."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(factor)


def _accumulate(model: URLTransformer, ids: torch.Tensor, weights: Optional[torch.Tensor],
                pad_id: int, parts: int, scaler, use_amp: bool) -> torch.Tensor:
    """E16: the batch's gradient, built from `parts` consecutive pieces of it.

    `model.loss` is the batch's scored loss over its scored mass, S / M. A piece's is
    S_k / M_k, so weighting each piece by M_k / M and summing the gradients gives the
    gradient of S / M -- the whole batch's. Clipping, the fixed objective's rescaling
    and the step then act on it exactly as on a single pass. Returns the batch's loss,
    detached (for the log)."""
    total = batch_mass(ids, weights, pad_id)
    loss = torch.zeros((), dtype=torch.float64, device=ids.device)
    w_parts = torch.tensor_split(weights, parts) if weights is not None else [None] * parts
    for p_ids, p_w in zip(torch.tensor_split(ids, parts), w_parts):
        share = batch_mass(p_ids, p_w, pad_id) / total if total else 0.0
        with torch.amp.autocast("cuda", enabled=use_amp):
            part = model.loss(p_ids, pad_id, p_w)
        scaler.scale(part * share).backward()
        loss += part.detach().double() * share
    return loss


def _diagnostics(records: List[Tuple[int, float, float, int, float]],
                 early_cap: Optional[float] = None) -> dict:
    """Per-step (step, grad norm before clipping, T_b, rows, that step's cap)
    summarised by the batch's scored tokens per row, and by phase (the first 1,000
    steps). `clipped_share` counts steps over their own cap. `early_cap` (E11, set
    only when the cap changes during the run) also reports the share of steps over
    the early cap: what the old cap would have clipped."""
    out: Dict[str, dict] = {}
    for name, keep in (("all", lambda s: True), ("first_1000", lambda s: s < 1000),
                       ("after_1000", lambda s: s >= 1000)):
        for lo, hi in _DIAG_BANDS:
            rows = [(n, c) for s, n, t, r, c in records if keep(s) and lo <= t / r < hi]
            if not rows:
                continue
            finite = [(n, c) for n, c in rows if math.isfinite(n)]
            entry = {
                "steps": len(rows),
                "mean_norm": sum(n for n, _ in finite) / len(finite) if finite else None,
                "clipped_share": sum(1 for n, c in finite if n > c) / len(rows),
                "nonfinite": len(rows) - len(finite)}
            if early_cap is not None:
                entry["over_early_cap_share"] = sum(1 for n, _ in finite
                                                    if n > early_cap) / len(rows)
            out[f"{name} {lo}-{hi if hi < 10 ** 9 else 'inf'}"] = entry
    return out


def batches(dataset: URLDataset, cfg: TrainConfig, pad_id: int,
            rng: random.Random,
            dropout: Optional["FieldDropout"] = None,
            inject: Optional["QueryInjector"] = None,
            digest=None) -> Iterator[torch.Tensor]:
    """Infinite stream of length-bucketed, optionally class-balanced batches.

    E15-B: `digest` (a hashlib object), when given, absorbs every batch's OLD rows
    before anything is appended, so a run can prove it drew the batches this seed
    draws without injection (V-batch). `inject` appends new rows to a finished
    batch; it never touches the old rows, their order or their masks, and it uses
    its own generators, so `rng` -- and with it every old batch -- is unchanged.
    """
    pools = {cls: list(seqs) for cls, seqs in dataset.by_class.items()}
    for seqs in pools.values():
        rng.shuffle(seqs)
    cursors = {cls: 0 for cls in pools}

    def draw(cls: str) -> List[int]:
        seqs = pools[cls]
        if cursors[cls] >= len(seqs):        # epoch boundary for this class
            rng.shuffle(seqs)
            cursors[cls] = 0
        cursors[cls] += 1
        return seqs[cursors[cls] - 1]

    classes = dataset.classes
    # p(class) proportional to n_labels ** alpha. At alpha = 1 every label in
    # every class is seen equally often; see `TrainConfig.class_alpha`.
    labels = dataset.label_counts()
    weights = [max(labels.get(c, 1), 1) ** cfg.class_alpha for c in classes]

    while True:
        window: List[List[int]] = []
        target = cfg.batch_size * cfg.bucket_window
        while len(window) < target:
            row = draw(rng.choices(classes, weights=weights, k=1)[0])
            window.append(dropout(row) if dropout is not None else row)
        # Bucket: sort the shuffled window so each batch is near-uniform in
        # length, then shuffle the batches so order stays random.
        window.sort(key=lambda r: len(r[0]) if isinstance(r, tuple) else len(r))
        chunks = [window[i:i + cfg.batch_size]
                  for i in range(0, len(window), cfg.batch_size)]
        rng.shuffle(chunks)
        for chunk in chunks:
            if digest is not None:
                _digest_rows(digest, chunk)
            if inject is not None:
                chunk = inject(chunk)
            yield _pad(chunk, pad_id)


def _digest_rows(digest, rows) -> None:
    """E15-B V-batch: one batch's rows (ids only), in order, into `digest`."""
    from array import array
    for r in rows:
        ids = r[0] if isinstance(r, tuple) else r
        digest.update(array("i", ids).tobytes())
        digest.update(b"|")
    digest.update(b"#")


def batch_digest(dataset: URLDataset, cfg: TrainConfig, pad_id: int, steps: int) -> str:
    """E15-B V-batch's reference: the digest of the first `steps` batches the
    sampler draws for this dataset and `cfg.seed`, nothing injected -- the laptop's
    replay of the old rows' batches."""
    import hashlib
    digest = hashlib.sha256()
    stream = batches(dataset, cfg, pad_id, random.Random(cfg.seed), None, None, digest)
    for _ in range(steps):
        next(stream)
    return digest.hexdigest()


def _lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(model: URLTransformer, dataset: URLDataset, cfg: TrainConfig,
             pad_id: int, device: torch.device, *, seed: int = 0) -> Dict[str, float]:
    """Held-out loss overall and per class — never pooled only."""
    model.eval()
    results: Dict[str, float] = {}
    for cls in dataset.classes:
        rng = random.Random(seed)
        seqs = list(dataset.by_class[cls])
        rng.shuffle(seqs)
        total = count = 0.0
        for i in range(0, min(len(seqs), cfg.eval_batches * cfg.batch_size),
                       cfg.batch_size):
            chunk = seqs[i:i + cfg.batch_size]
            if not chunk:
                break
            ids, weights = _pad(chunk, pad_id)
            ids = ids.to(device)
            weights = weights.to(device) if weights is not None else None
            # Weighted, to match the training objective: `best.pt` must be
            # selected on what is actually being optimised, not on a full-URL
            # loss the model is no longer trained to minimise.
            total += float(model.loss(ids, pad_id, weights)) * len(chunk)
            count += len(chunk)
        results[cls] = total / count if count else float("nan")
    results["mean"] = sum(results[c] for c in dataset.classes) / len(dataset.classes)
    model.train()
    return results


def train(model: URLTransformer, train_set: URLDataset, val_set: URLDataset,
          tokenizer: URLTokenizer, cfg: TrainConfig, *,
          device: Optional[torch.device] = None,
          out_dir: Optional[Path] = None,
          log_every: int = 100) -> Dict[str, object]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.loss_norm not in LOSS_NORMS:
        raise ValueError(f"loss_norm must be one of {LOSS_NORMS}, not {cfg.loss_norm!r}")
    fixed = cfg.loss_norm == "fixed"
    if not (math.isfinite(cfg.query_weight) and cfg.query_weight > 0):
        raise ValueError(f"query_weight must be a positive number; got {cfg.query_weight!r}")
    weighted_q = cfg.query_weight != 1.0
    if not (isinstance(cfg.accum_steps, int) and cfg.accum_steps >= 1):
        raise ValueError(f"accum_steps must be a positive integer; got {cfg.accum_steps!r}")
    if weighted_q and fixed and cfg.loss_constant is None:
        # E14 (PREREG_E14 section 2.1): C computed from the WEIGHTED mass would
        # shrink every other field's weight -- the trade-off would be a design
        # choice instead of a measurement. C must be pinned.
        raise ValueError("query_weight under the fixed objective needs a pinned loss_constant")
    loss_constant = None
    if fixed:
        loss_constant = float(cfg.loss_constant if cfg.loss_constant is not None else
                              cfg.batch_size * expected_scored_tokens(train_set, cfg))
    from uie.free import weights_sha256            # E9 V-id: the arms' shared start
    init_sha256 = weights_sha256(model)
    model.to(device)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    # V2-P1b (research/v2/PREREG_V2_P1B.md, T2): a model MAY define `optimizer_groups(cfg)`,
    # returning AdamW parameter groups -- dicts with "params" and optional overrides
    # ("weight_decay", "betas", and "lr_scale", which multiplies the schedule's rate for that
    # group every step). No uie model defines it; absent, or returning None, the optimiser is
    # built exactly as before (bit-identical: tests/test_v2_p1b.py).
    groups = model.optimizer_groups(cfg) if hasattr(model, "optimizer_groups") else None
    optimizer = torch.optim.AdamW(model.parameters() if groups is None else groups, lr=cfg.lr,
                                  weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    dropout = None
    if cfg.field_dropout:
        # A separate seed: the batches must be the ones this seed draws anyway.
        dropout = FieldDropout(tokenizer, cfg.field_dropout, seed=cfg.seed + 7919,
                               joint=cfg.field_dropout_joint)
    elif cfg.field_dropout_joint:
        raise ValueError("field_dropout_joint needs field_dropout > 0")
    swap = None
    context_record = None
    if cfg.context_prefixes:
        # E13 C: its own seed, like the dropout's.
        swap = ContextSwap.load(cfg.context_prefixes, tokenizer, cfg.context_p,
                                cfg.context_seed, model.cfg.max_len)
        context_record = {"listed_rows": len(swap.table),
                          "matched_rows": swap.matched(train_set), "p": cfg.context_p}
        print(f"E13 C: ContextSwap ON -> {context_record['matched_rows']:,} of "
              f"{context_record['listed_rows']:,} listed rows matched; p = {cfg.context_p}",
              flush=True)
    elif cfg.context_p:
        raise ValueError("context_p needs context_prefixes")
    inject = None
    digest = None
    if cfg.qry_extra:
        # E15-B: the new rows, appended alongside the old ones (section 10.4).
        import hashlib
        inject = QueryInjector.load(cfg.qry_extra, tokenizer, train_set, model.cfg.max_len,
                                    cfg.qry_extra_draw_seed, cfg.qry_extra_rate_seed)
        digest = hashlib.sha256()
        print(f"E15: query injection ON -> {inject.summary()}", flush=True)
    stream = batches(train_set, cfg, tokenizer.pad_id, rng, _compose(dropout, swap),
                     inject, digest)
    history: List[dict] = []
    best = float("inf")
    started = time.time()
    records: List[Tuple[int, float, float, int]] = []   # read-only diagnostics
    q_totals = [0.0, 0.0]          # E14: scored positions, weighted mass (over the run)
    model.train()

    for step in range(cfg.steps):
        for group in optimizer.param_groups:
            # V2-P1b: a group's own `lr_scale`, if it has one (only from `optimizer_groups`)
            group["lr"] = (_lr_at(step, cfg) * group["lr_scale"] if "lr_scale" in group
                           else _lr_at(step, cfg))

        ids, weights = next(stream)
        ids = ids.to(device)
        weights = weights.to(device) if weights is not None else None
        if cfg.accum_steps == 1:
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model.loss(ids, tokenizer.pad_id, weights)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
        else:
            # E16: the same batch's gradient in `accum_steps` pieces (`_accumulate`).
            optimizer.zero_grad(set_to_none=True)
            loss = _accumulate(model, ids, weights, tokenizer.pad_id, cfg.accum_steps,
                               scaler, use_amp)
        scaler.unscale_(optimizer)
        cap = (cfg.grad_clip if cfg.grad_clip_late is None or step < cfg.grad_clip_switch
               else cfg.grad_clip_late)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cap)
        mass = batch_mass(ids, weights, tokenizer.pad_id)
        if fixed:
            _scale_gradients(model, mass / loss_constant)
        band_mass = mass
        if weighted_q and weights is not None:
            # E14 V5: the clip bands are read on the w = 1 mass (the positions
            # scored at all), so they are comparable with the unweighted run's.
            band_mass = float(((weights > 0) & (ids[:, 1:] != tokenizer.pad_id)).sum())
            q_totals[0] += band_mass
            q_totals[1] += mass
        records.append((step, float(norm), band_mass, int(ids.shape[0]), cap))
        scaler.step(optimizer)
        scaler.update()

        if log_every and step % log_every == 0:
            print(f"  step {step:>6}  loss {float(loss):.4f}  "
                  f"lr {_lr_at(step, cfg):.2e}  {time.time()-started:.0f}s",
                  flush=True)

        if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            metrics = evaluate(model, val_set, cfg, tokenizer.pad_id, device)
            metrics["step"] = step + 1
            history.append(metrics)
            print(f"  eval  step {step+1:>6}  " +
                  "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                            if k != "step"), flush=True)
            if out_dir and metrics["mean"] < best:
                best = metrics["mean"]
                save_checkpoint(model, tokenizer, cfg, Path(out_dir) / "best.pt")

    final = evaluate(model, val_set, cfg, tokenizer.pad_id, device)
    return {"history": history, "final": final,
            "seconds": round(time.time() - started, 1),
            "device": str(device), "best_val_mean": min(best, final["mean"]),
            "loss_norm": cfg.loss_norm, "loss_constant": loss_constant,
            "init_weights_sha256": init_sha256,
            "grad_diagnostics": _diagnostics(
                records, cfg.grad_clip if cfg.grad_clip_late is not None else None),
            **({"context_swap": {**context_record, **swap.counts}} if swap is not None else {}),
            # E14 readback: the query positions' share of the scored positions the
            # run drew, from (weighted - scored) = (w - 1) x query positions.
            **({"query_weight": {
                "w": cfg.query_weight, "scored_positions": q_totals[0],
                "weighted_mass": q_totals[1],
                "query_share": ((q_totals[1] - q_totals[0]) / (cfg.query_weight - 1.0)
                                / q_totals[0]) if q_totals[0] else None}}
               if weighted_q else {}),
            # E15-B readback (V-batch, V-mass, V-stream): absent when the arm is off
            **({"qry_extra": {**inject.record(), "old_batch_digest": digest.hexdigest(),
                              "steps": cfg.steps}} if inject is not None else {})}


def save_checkpoint(model: URLTransformer, tokenizer: URLTokenizer,
                    cfg: TrainConfig, path: Path) -> None:
    """Weights, architecture AND tokenizer together.

    A checkpoint without its tokenizer is unusable — token ids are meaningless
    against a different vocabulary — and storing them apart invites exactly that
    mismatch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = asdict(model.cfg)
    if not model_config.get("frame_every"):
        # E12: written only when on, so a pre-E12 checkpoint's layout -- and every
        # readback that compares it -- stays exactly what it was.
        model_config.pop("frame_every", None)
        model_config.pop("frame_index", None)
    train_config = asdict(cfg)
    if not train_config.get("context_prefixes"):
        # E13: written only when on, as `frame_every` above.
        for k in ("context_prefixes", "context_p", "context_seed"):
            train_config.pop(k, None)
    if train_config.get("query_weight", 1.0) == 1.0:
        # E14: the same rule.
        train_config.pop("query_weight", None)
    if not train_config.get("qry_extra"):
        # E15-B: the same rule.
        for k in ("qry_extra", "qry_extra_draw_seed", "qry_extra_rate_seed"):
            train_config.pop(k, None)
    if train_config.get("accum_steps", 1) == 1:
        # E16: the same rule.
        train_config.pop("accum_steps", None)
    torch.save({"model": model.state_dict(),
                "model_config": model_config,
                "train_config": train_config,
                "tokenizer": tokenizer.state_dict()}, path)


def load_checkpoint(path: Path, *, device: Optional[torch.device] = None
                    ) -> Tuple[URLTransformer, URLTokenizer, ModelConfig]:
    blob = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    model = URLTransformer(cfg)
    model.load_state_dict(blob["model"])
    tokenizer = URLTokenizer.from_state_dict(blob["tokenizer"])
    return model, tokenizer, cfg
