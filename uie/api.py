"""
uie/api.py

The public interface of the Limnex URL Intelligence Engine (D2 step 4, research/d2/D2_PLAN.md):
load the published model, generate URLs, score URLs.

    import uie
    engine = uie.load("limnex-uie-e16_b5.pt", known_urls="limnex-uie-known-urls.bloom.gz")
    engine.generate("phishing", n=10)
    engine.score(["example.com/login"])

READ THE README'S "RESPONSIBLE USE" AND "NO LIABILITY" SECTIONS BEFORE USING THIS. Generated URLs
are synthetic and exist for defensive research only: never register, resolve, visit, host or send
them, and never use them against anyone.

Every generated batch passes the library's gate (uie/gate.py) before it is returned -- there is no
way to switch it off here:
  validity     parseable, a real public suffix, IDNA-encodable, within protocol limits
  duplicates   no URL twice in one batch
  novelty      not a copy of a known training URL, using the published Bloom filter
               (`known_urls=`). REQUIRED for every generation: about 1 in 100 raw phishing
               samples copies a real phishing URL, and a rare benign copy can carry the personal
               data some real URLs hold (an e-mail address, a token)
  live threats not on a registrable domain of a current threat feed, when a snapshot is given
               (`blocked_domains=`); generation never queries DNS or the network
So `generate(..., n=10)` returns at most 10 URLs; `engine.last_report` says why any were removed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Union

from uie.baseline import E16_B5

__all__ = ["NOTICE", "PUBLISHED_WEIGHTS_SHA256", "PUBLISHED_KNOWN_URLS_SHA256", "Engine", "load",
           "load_blocklist"]

#: The published checkpoint file (e16_cap's weights, read as e16_b5), by sha256.
PUBLISHED_WEIGHTS_SHA256 = E16_B5.file_sha256
#: The published known-URL filter (data/bundle2/known_urls.bloom.gz), by the sha256 of its CONTENT:
#: the gzip container carries a time stamp, so a rebuild of the bundle changes the file's hash but
#: not this one, and a copy that was decompressed on the way (Kaggle does this) still matches.
PUBLISHED_KNOWN_URLS_SHA256 = "3ef34d06ff0e4735b1810fa3592d03ee182e748e05b08927c9eff53bde377eae"
ASSETS = Path(__file__).resolve().parent / "assets"
#: The generation tables the published runs used, by sha256 (research/e8, e13 verify records).
ASSET_SHA256 = {
    "suffix_table.json": "aaa6fb853ea538d41ba4c10661b89faad2a687b869208772c096c7e507ee866d",
    "presence_slash_table.json": "5a0eeac991b8343267d6f88f4c590bd6c208ace975dbf2218f8c5a277b71c050",
    # control mode's per-state suffix tables (`python -m uie.controlled` on the bundle), which the
    # gate's control-mode files used; read by REPRODUCE.md level 2, not by Engine
    "state_suffix_tables.json": "0c6f2760c033c532d271b360a14f2976139d5957e4aaf64977c8746504213cc9",
}
LABELS = ("benign", "phishing")
NOTICE = ("Limnex UIE: synthetic URLs for defensive research only. Never register, resolve, visit, "
          "host or send them, and never use them against anyone. Provided AS IS, without warranty "
          "or liability; see the README's Responsible Use and No Liability sections.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _content_sha256(path: Path) -> str:
    """sha256 of a filter file's content, gzip or not (as BloomFilter.load reads it)."""
    raw = Path(path).read_bytes()
    return hashlib.sha256(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).hexdigest()


def _asset(name: str) -> dict:
    path = ASSETS / name
    if _sha256(path) != ASSET_SHA256[name]:
        raise RuntimeError(f"the packaged table {name} is not the published one; reinstall the package")
    return json.loads(path.read_text(encoding="utf-8"))


def load_blocklist(path: Union[str, Path]) -> List[str]:
    """Registrable domains from a threat-feed snapshot: one domain, host or URL per line; blank
    lines and lines starting with '#' ignored. Every entry is reduced to its registrable domain,
    which is what the gate compares (a generated path on a listed domain is just as unsafe)."""
    from uie.factorization import parse
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        domain = parse(line).registrable_domain
        if domain:
            out.append(domain.lower())
    return out


class Engine:
    """The published engine: generate (free mode, e16_b5's decode, gated) and score (log P)."""

    def __init__(self, model, tokenizer, *, device=None, known_urls=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._known = known_urls
        shipped = _asset("suffix_table.json")
        self._shipped = {k: shipped[k] for k in LABELS}
        self._presence_slash = _asset("presence_slash_table.json")["table"]
        self.last_report = None
        self.last_seed: Optional[int] = None
        self._last_blocked = False

    @property
    def novelty_checked(self) -> bool:
        """True when the known-URL Bloom filter was given, so copies of training URLs are removed."""
        return self._known is not None

    def report_text(self) -> str:
        """The last batch's gate result in words, stating which checks did NOT run (the library's
        GateReport.summary reports novelty as 100% when it was never checked)."""
        r = self.last_report
        if r is None:
            return "no batch generated yet"
        removed = ", ".join(f"{k} {v}" for k, v in r.by_reason.most_common()) or "none"
        return (f"seed {self.last_seed}: {r.n_passed} of {r.n_total} kept; removed: {removed}; "
                f"novelty {'checked' if self.novelty_checked else 'NOT checked (no known-URL filter given)'}; "
                f"live threats {'checked' if self._last_blocked else 'NOT checked (no threat-feed snapshot given)'}")

    def generate(self, label: str, n: int = 10, *, seed: Optional[int] = None,
                 blocked_domains: Optional[Iterable[str]] = None) -> List[str]:
        """Up to `n` synthetic URLs of class `label` ("benign" or "phishing"), after the gate.
        Deterministic for a given `seed`; without one, a fresh seed is drawn and kept in
        `last_seed`. `blocked_domains`: registrable domains of a current threat-feed snapshot."""
        from uie.engine import generate_free
        from uie.gate import Gate
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, not {label!r}")
        if n < 1:
            raise ValueError("n must be at least 1")
        if self._known is None:
            # The model sometimes reproduces a training URL. Measured on the published model: about
            # 1 in 100 raw phishing samples is a real phishing URL (research/d2/ENGINE_VERIFY.json,
            # E4), and 545 of the gate's 80,000 free-mode URLs were known URLs. A benign copy can
            # carry personal data (about 1 in 400 real benign URLs holds an e-mail address or a
            # token). Only the known-URL filter removes copies, so nothing is generated without it.
            raise ValueError("generation needs the published known-URL filter "
                             "(known_urls= / --known-urls limnex-uie-known-urls.bloom.gz): the model "
                             "sometimes reproduces a training URL -- a real phishing URL, or a real "
                             "benign URL that may carry personal data -- and only the filter removes them")
        seed = random.SystemRandom().randrange(2 ** 31) if seed is None else int(seed)
        urls = generate_free(self.model, self.tokenizer, label=label, n=int(n), seed=seed,
                             presence_slash=self._presence_slash, shipped=self._shipped, device=self.device)
        blocked = list(blocked_domains) if blocked_domains is not None else None
        kept, report = Gate(known_membership=self._known, blocked_domains=blocked).filter(urls)
        self.last_report, self.last_seed, self._last_blocked = report, seed, bool(blocked)
        return kept

    def score(self, urls: Iterable[str]) -> List[dict]:
        """For each URL: log P(url | benign), log P(url | phishing), llr = their difference (higher
        = more typical of phishing), token count, and whether it was truncated or is an IDN host.
        A likelihood, not a validated phishing detector."""
        from uie.typicality import TypicalityScorer
        urls = list(urls)
        rows = TypicalityScorer(self.model, self.tokenizer, device=self.device).score(urls)
        return [{"url": u, **r} for u, r in zip(urls, rows)]


def load(weights: Union[str, Path], *, device: str = "cpu", verify: bool = True,
         known_urls: Union[str, Path, None] = None) -> Engine:
    """Load the published checkpoint. With `verify` (the default) the file must be the published
    one, by sha256, and its weights e16_b5's, and `known_urls` (when given) the published filter;
    pass `verify=False` only for a checkpoint and filter of your own.
    `known_urls`: the published Bloom filter of training URLs, to remove copies (novelty)."""
    import torch
    from uie.train import load_checkpoint
    path = Path(weights)
    known = None if known_urls is None else Path(known_urls)
    if verify and known is not None:
        # phishing generation relies on this filter to remove copies of real phishing URLs, so a
        # different file (a wrong download, an empty filter) must never pass for it
        got = _content_sha256(known)
        if got != PUBLISHED_KNOWN_URLS_SHA256:
            raise ValueError(f"{known} is not the published known-URL filter (content sha256 {got[:12]}..., "
                             f"expected {PUBLISHED_KNOWN_URLS_SHA256[:12]}...). Download it again, or pass "
                             "verify=False for a filter of your own.")
    if verify:
        got = _sha256(path)
        if got != PUBLISHED_WEIGHTS_SHA256:
            raise ValueError(f"{path} is not the published Limnex UIE checkpoint (sha256 {got[:12]}..., "
                             f"expected {PUBLISHED_WEIGHTS_SHA256[:12]}...). Download it again, or pass "
                             "verify=False for a checkpoint of your own.")
    dev = torch.device(device)
    model, tokenizer, _ = load_checkpoint(path, device=dev)
    model.to(dev).eval()          # load_checkpoint maps the tensors but builds the model on the CPU
    if verify:
        E16_B5.check_model(model)
    bloom = None
    if known is not None:
        from uie.bloom import BloomFilter
        bloom = BloomFilter.load(known)
    return Engine(model, tokenizer, device=dev, known_urls=bloom)
