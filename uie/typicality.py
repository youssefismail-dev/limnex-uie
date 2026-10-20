"""
uie/typicality.py

D1 — the typicality scorer (DESIGN.md sections 7 and 9; research/d1/D1_PLAN.md).

    from uie.typicality import TypicalityScorer
    scorer = TypicalityScorer.load("data/e16/train/e16_cap/final.pt")
    scorer.log_prob(urls, "benign")     # log P(url | benign), nats, one per URL
    scorer.llr(urls)                    # log P(url | phishing) - log P(url | benign)
    scorer.score(urls)                  # both, plus flags, as dicts

This is the public interface: a consumer imports this module and nothing else. It is the
training objective exposed: the probability the model assigns to the whole encoded URL,
control tokens and end token included, exactly as the URL was encoded in training.

Fixed choices, and why
----------------------
- **Frame `domain`**: the frame generation samples from, so the score describes the
  distribution the engine writes.
- **Host lowercased, nothing else changed.** DNS names are case-insensitive and the training
  hosts are lowercase (8 of 50,000 phishing rows had any capital); paths and queries are
  case-sensitive and stay as given. The scheme is read and dropped by the tokenizer, as in
  training.
- **Long URLs**: the model never saw more than `max_len` tokens, so only the first `max_len`
  are scored and the row is flagged `truncated`. Both classes score the same tokens, so the
  difference stays a fair comparison.
- **IDN hosts** (punycode `xn--` or non-ASCII) are scored and flagged `idn`; a consumer
  decides whether to use the score for them.

What the score is not
---------------------
It is a density, not a verdict: `llr` > 0 means the URL is more probable under the phishing
model than under the benign one. Its known limit is the engine's: real but rare words look
unusual to it (README, "Known limitations"), so it can find a legitimate URL atypical.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from uie.model import URLTransformer
from uie.tokenizer import URLTokenizer

__all__ = ["TypicalityScorer", "normalize_url", "is_idn"]

_AUTHORITY = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*://)?([^/?#]*)(.*)$", re.S)


def normalize_url(url: str) -> str:
    """Strips surrounding whitespace and lowercases the authority (host, and any userinfo
    or port with it). Everything after the host is returned unchanged."""
    m = _AUTHORITY.match(url.strip())
    scheme, authority, rest = m.group(1) or "", m.group(2), m.group(3)
    return scheme + authority.lower() + rest


def is_idn(url: str) -> bool:
    """True if the host is internationalised: punycode labels or non-ASCII characters."""
    authority = _AUTHORITY.match(url.strip()).group(2)
    host = authority.rsplit("@", 1)[-1]
    return "xn--" in host.lower() or any(ord(ch) > 127 for ch in host)


@dataclass(frozen=True)
class _Encoded:
    ids: List[int]
    truncated: bool


class TypicalityScorer:
    """log P(url | class) under a trained UIE checkpoint."""

    FRAME = "domain"

    def __init__(self, model: URLTransformer, tokenizer: URLTokenizer, *,
                 device: Optional[torch.device] = None, batch_size: int = 64) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.max_len = int(model.cfg.max_len)

    @classmethod
    def load(cls, path, *, device: Optional[torch.device] = None,
             batch_size: int = 64) -> "TypicalityScorer":
        from uie.train import load_checkpoint
        model, tokenizer, _ = load_checkpoint(Path(path), device=device or torch.device("cpu"))
        return cls(model, tokenizer, device=device, batch_size=batch_size)

    @property
    def classes(self) -> Tuple[str, ...]:
        return tuple(self.tokenizer.classes)

    def encode(self, url: str, label: str) -> _Encoded:
        ids = self.tokenizer.encode(normalize_url(url), label, frame=self.FRAME)
        if len(ids) > self.max_len:
            return _Encoded(ids[:self.max_len], True)
        return _Encoded(ids, False)

    @torch.no_grad()
    def log_prob(self, urls: Sequence[str], label: str) -> List[float]:
        """log P(url | label) in nats, one per URL, in the input order."""
        encoded = [self.encode(u, label).ids for u in urls]
        out = [0.0] * len(encoded)
        # Similar lengths together, so little padding is computed; the order is restored.
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        pad = self.tokenizer.pad_id
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            width = max(len(encoded[i]) for i in idx)
            ids = torch.tensor([encoded[i] + [pad] * (width - len(encoded[i])) for i in idx],
                               dtype=torch.long, device=self.device)
            lp = self.model.sequence_log_prob(ids, pad)
            for i, v in zip(idx, lp.tolist()):
                out[i] = float(v)
        return out

    def llr(self, urls: Sequence[str]) -> List[float]:
        """log P(url | phishing) - log P(url | benign): higher means more phishing-like."""
        p = self.log_prob(urls, "phishing")
        b = self.log_prob(urls, "benign")
        return [x - y for x, y in zip(p, b)]

    def score(self, urls: Sequence[str]) -> List[Dict[str, object]]:
        """Both log-probabilities, their difference and the flags, one dict per URL."""
        p = self.log_prob(urls, "phishing")
        b = self.log_prob(urls, "benign")
        out = []
        for u, lp, lb in zip(urls, p, b):
            enc = self.encode(u, "benign")
            out.append({"llr": lp - lb, "log_p_phishing": lp, "log_p_benign": lb,
                        "tokens": len(enc.ids), "truncated": enc.truncated, "idn": is_idn(u)})
        return out
