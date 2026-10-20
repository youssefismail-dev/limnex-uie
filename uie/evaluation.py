"""
uie/evaluation.py

E3 — the measurements that decide whether E2's samples are as good as they look.

Four instruments, each answering a question the others cannot:

**Real-vs-Fake AUC.** Can a discriminator separate generated URLs from HELD-OUT
real ones? This is the text analogue of FID and the headline number. It must use
held-out real data: scored against training data it rewards memorisation, which
is the failure it exists to catch.

**Representation probe.** Can a LINEAR classifier read language or TLD category
off the model's internal states? Section 5 argues the model must represent such
concepts to predict URLs well, but that "the representation contains it" and
"we can use it" are different claims. This tests the first one cheaply, and a
positive result is a finding in its own right: an autoregressive URL model
learning web structure with no supervision.

**Generative precision and recall.** Precision asks whether samples land inside
the real distribution; recall asks whether they COVER it. A mode-collapsed model
scores high precision and low recall, and neither number alone would reveal it.
Computed k-NN style over the model's own hidden states.

**Human panel export.** The only instrument that tests the project's actual
claim — that a researcher shown a shuffled mix could not sort it. Everything
above is a proxy for this.

Why hidden states are the embedding
-----------------------------------
Precision/recall needs a metric space. Using the model's own representation is
circular for judging ITS samples in principle, but the alternative (hand-designed
URL features) would smuggle in exactly the rule-based assumptions this project
rejects. The circularity is bounded by the fact that real held-out URLs are
embedded by the same function, so the comparison is like-for-like.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from uie.factorization import parse as _parse, strip_scheme as _strip_scheme
from uie.ngram import path_body as _path_body

__all__ = ["embed", "field_nll", "real_vs_fake_auc", "probe_representation",
           "precision_recall", "diversity", "human_panel", "FIELDS",
           "FIELD_POPULATION", "discriminate_with_floor", "field_report"]


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

@torch.no_grad()
def embed(model, tokenizer, urls: Sequence[str], label: str,
          device: torch.device, batch: int = 64) -> np.ndarray:
    """Mean-pooled final-layer states, one vector per URL.

    Mean-pooled rather than last-token: the last position of a causal model is
    dominated by whatever field happens to end the URL, so two URLs differing
    only in whether they carry a fragment would sit far apart for a reason that
    has nothing to do with what they are.
    """
    model.eval()
    out: List[np.ndarray] = []
    for i in range(0, len(urls), batch):
        chunk = urls[i:i + batch]
        encoded = [tokenizer.encode(u, label)[:model.cfg.max_len] for u in chunk]
        width = max(len(e) for e in encoded)
        ids = torch.tensor([e + [tokenizer.pad_id] * (width - len(e))
                            for e in encoded], dtype=torch.long, device=device)

        x = model.drop(model.tok_emb(ids) +
                       model.pos_emb(torch.arange(ids.shape[1], device=device)))
        for block in model.blocks:
            x = block(x)
        x = model.ln_f(x)

        mask = (ids != tokenizer.pad_id).unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.cfg.d_model))


# --------------------------------------------------------------------------- #
# Per-field likelihood
# --------------------------------------------------------------------------- #

@torch.no_grad()
def field_nll(model, tokenizer, urls: Sequence[str], label: str,
              device: torch.device, batch: int = 64,
              min_tokens: int = 200,
        hide_fields: Sequence[str] = ()) -> Dict[str, float]:
    """Mean NLL in nats/character, broken down by URL field.

    Aggregate perplexity pools every field together and so cannot say WHICH
    part of a URL a model is bad at. Run on train and val rows separately, the
    per-field gap distinguishes the two failures that aggregate loss conflates:

      train high, val high, gap small  -> UNDERFIT; more capacity may help.
      train low,  val high, gap large  -> MEMORISED; more capacity makes it worse.

    This mattered concretely. E2's aggregate perplexity (6.285) looked like a
    model that had learned URLs well, while the breakdown showed the registrable
    label at train 1.201 / val 2.346 — a +1.144 gap, four times the path's. The
    model had memorised 99,303 label strings instead of learning how labels are
    built, which is why its invented labels were phonotactic mush. Capacity was
    the intuitive fix and would have been the wrong one.

    Field markers themselves are excluded: they are structure the grammar owns,
    they are predicted near-perfectly, and averaging them in dilutes exactly the
    content signal being measured.

    `min_tokens` suppresses fields too rare in the sample to mean anything — a
    port seen four times would otherwise report a number with no error bar
    beside the path's hundred thousand.
    """
    from uie.tokenizer import _MARKERS

    marker_of = {tokenizer.marker_id(m): m for m in _MARKERS}
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    model.eval()

    for i in range(0, len(urls), batch):
        chunk = urls[i:i + batch]
        encoded = [(tokenizer.encode(u, label, hide_fields=hide_fields) if hide_fields else tokenizer.encode(u, label))[:model.cfg.max_len] for u in chunk]
        width = max(len(e) for e in encoded)
        ids = torch.tensor([e + [tokenizer.pad_id] * (width - len(e))
                            for e in encoded], dtype=torch.long, device=device)

        logits = model(ids)
        losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1), reduction="none",
        ).view(ids.shape[0], -1).float().cpu()

        for row, sequence in enumerate(encoded):
            current: Optional[str] = None
            for t in range(len(sequence) - 1):
                target = sequence[t + 1]
                if target in marker_of:
                    current = marker_of[target]
                    continue
                if current is None or target == tokenizer.eos_id:
                    continue
                totals[current] = totals.get(current, 0.0) + float(losses[row, t])
                counts[current] = counts.get(current, 0) + 1

    return {f: totals[f] / counts[f]
            for f in sorted(counts) if counts[f] >= min_tokens}


@torch.no_grad()
def boundary_nll(model, tokenizer, urls: Sequence[str], label: str,
                 device: torch.device, batch: int = 64,
        hide_fields: Sequence[str] = ()) -> Dict[str, float]:
    """S4's PRIMARY OUTCOME: cross-entropy and over-confidence at segment starts.

    Splits in-path targets into the first character of each segment (right after
    `<PTH>` or a `/`) and everything after it, and reports for each:

        ce        E_real[-log p_model(x)]
        h         E[H(p_model)], the model's own expected surprise
        overconf  ce - h, confidence in excess of accuracy, in nats

    `overconf` IS NOT A KL DIVERGENCE, and was mislabelled as one in an earlier
    report. KL(p_real||p_model) = ce - H(p_REAL), and H(p_real) is not estimable
    from a corpus, which supplies one sample per context rather than a
    distribution. `overconf` is zero for a perfect model and can go NEGATIVE
    when a model is under-confident -- which a divergence cannot do.

    Measured on E5 before S4, benign held-out:

        boundary     ce 3.168   h 2.839   overconf +0.329
        mid-segment  ce 1.410   h 1.379   overconf +0.032

    Run on train and val rows separately, the pair also separates the two
    regimes that decide whether up-weighting is even coherent: a boundary
    train/val gap much wider than mid-segment's would mean the openings were
    MEMORISED, and weighting them harder would make over-confidence worse rather
    than better.
    """
    from uie.tokenizer import _MARKERS

    slash = tokenizer.stoi.get("/")
    if slash is None:
        raise KeyError("boundary_nll needs '/' in the vocabulary")
    pth = tokenizer.marker_id("PTH")
    ends = set(tokenizer.marker_id(m) for m in _MARKERS) | {tokenizer.eos_id}
    acc = {"boundary": [0.0, 0.0, 0], "mid": [0.0, 0.0, 0]}  # ce, h, n
    model.eval()

    for i in range(0, len(urls), batch):
        encoded = [(tokenizer.encode(u, label, hide_fields=hide_fields) if hide_fields else tokenizer.encode(u, label))[:model.cfg.max_len]
                   for u in urls[i:i + batch]]
        encoded = [e for e in encoded if pth in e]
        if not encoded:
            continue
        width = max(len(e) for e in encoded)
        ids = torch.tensor([e + [tokenizer.pad_id] * (width - len(e))
                            for e in encoded], dtype=torch.long, device=device)
        # float() BEFORE log_softmax: under AMP the head emits fp16, and an
        # fp16 log_softmax loses precision exactly where these numbers are read.
        logp = torch.log_softmax(model(ids).float(), dim=-1)
        # POSITION j PREDICTS TOKEN j+1. Both arrays are therefore indexed by the
        # PREDICTING position, and a target at t is scored from slot t-1. Getting
        # this off by one silently reports the model's confidence about a token
        # it has already been shown, which reads as excellent calibration.
        entropy = -(logp.exp() * logp).sum(-1)[:, :-1].cpu()
        nll = -logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).cpu()

        for row, sequence in enumerate(encoded):
            start = sequence.index(pth) + 1
            stop = start
            while stop < len(sequence) and sequence[stop] not in ends:
                stop += 1
            since = 0
            for t in range(start, stop):
                slot = acc["boundary" if since == 0 else "mid"]
                slot[0] += float(nll[row, t - 1])
                slot[1] += float(entropy[row, t - 1])
                slot[2] += 1
                since = 0 if sequence[t] == slash else since + 1

    out: Dict[str, float] = {}
    for kind, (ce, h, n) in acc.items():
        if n:
            out[f"{kind}_ce"] = ce / n
            out[f"{kind}_h"] = h / n
            out[f"{kind}_overconf"] = (ce - h) / n
            out[f"{kind}_n"] = float(n)
    if acc["boundary"][2] and acc["mid"][2]:
        out["overconf_ratio"] = (out["boundary_overconf"]
                                 / (out["mid_overconf"] or 1e-9))
    return out


@torch.no_grad()
def shape_nll(model, tokenizer, urls: Sequence[str], label: str,
              device: torch.device, batch: int = 64,
        hide_fields: Sequence[str] = ()) -> Dict[str, float]:
    """NLL of the STOP and STRUCTURE decisions, which content metrics hide.

    `field_nll` deliberately excludes markers and the end token, so a model that
    has gotten worse at deciding WHERE a URL stops shows nothing there. S4
    measured exactly that failure: benign EOS NLL rose 1.261 -> 1.423 while every
    content field moved by less than 0.04, and the visible consequence was path
    depth drifting 1.769 -> 2.165 against a target of 1.782.

    Reported separately rather than folded into an average, because stopping is
    a different decision from spelling and trades against it.
    """
    from uie.tokenizer import _MARKERS

    markers = {tokenizer.marker_id(m) for m in _MARKERS}
    acc = {"eos": [0.0, 0], "marker": [0.0, 0]}
    model.eval()

    for i in range(0, len(urls), batch):
        encoded = [(tokenizer.encode(u, label, hide_fields=hide_fields) if hide_fields else tokenizer.encode(u, label))[:model.cfg.max_len]
                   for u in urls[i:i + batch]]
        if not encoded:
            continue
        width = max(len(e) for e in encoded)
        ids = torch.tensor([e + [tokenizer.pad_id] * (width - len(e))
                            for e in encoded], dtype=torch.long, device=device)
        logp = torch.log_softmax(model(ids).float(), dim=-1)
        # position j predicts token j+1, so a target at t is scored from t-1
        nll = -logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).cpu()
        for row, seq in enumerate(encoded):
            for t in range(1, len(seq)):
                key = ("eos" if seq[t] == tokenizer.eos_id
                       else "marker" if seq[t] in markers else None)
                if key:
                    acc[key][0] += float(nll[row, t - 1])
                    acc[key][1] += 1

    return {f"{k}_nll": v[0] / v[1] for k, v in acc.items() if v[1]}


# --------------------------------------------------------------------------- #
# Real vs fake
# --------------------------------------------------------------------------- #

def real_vs_fake_auc(real: Sequence[str], fake: Sequence[str], *,
                     seed: int = 20260803) -> Dict[str, float]:
    """Train a character n-gram discriminator; report cross-validated AUC.

    0.50 means indistinguishable. Deliberately a SURFACE model (char 2-5 grams
    plus logistic regression) rather than a deep one: if a bag of character
    n-grams can separate the two sets, the generator has a surface-level tell
    and no subtler probe is needed to prove it.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    texts = list(real) + list(fake)
    y = np.array([0] * len(real) + [1] * len(fake))
    pipeline = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                        max_features=50_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    scores = cross_val_score(pipeline, texts, y, cv=5, scoring="roc_auc")
    return {"auc_mean": float(scores.mean()), "auc_std": float(scores.std()),
            "n_real": len(real), "n_fake": len(fake)}


# --------------------------------------------------------------------------- #
# Representation probe
# --------------------------------------------------------------------------- #

def probe_representation(vectors: np.ndarray, targets: Sequence[str], *,
                         seed: int = 20260803) -> Dict[str, float]:
    """Linear probe accuracy against the majority-class baseline.

    LINEAR on purpose. A non-linear probe can extract information the model does
    not make *available*, so a strong MLP would prove only that the information
    survives somewhere — not that the representation organises it. The lift over
    the majority baseline is the finding; raw accuracy is not.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    labels = np.array(list(targets))
    keep = np.array([np.sum(labels == v) >= 20 for v in labels])
    vectors, labels = vectors[keep], labels[keep]
    if len(set(labels)) < 2:
        return {"accuracy": float("nan"), "baseline": float("nan"),
                "lift": float("nan"), "n": int(len(labels))}

    _, counts = np.unique(labels, return_counts=True)
    baseline = float(counts.max() / counts.sum())
    scores = cross_val_score(
        LogisticRegression(max_iter=3000),
        vectors, labels, cv=5, scoring="accuracy")
    accuracy = float(scores.mean())
    return {"accuracy": accuracy, "baseline": baseline,
            "lift": accuracy - baseline, "n": int(len(labels)),
            "n_classes": int(len(set(labels)))}


# --------------------------------------------------------------------------- #
# Precision / recall
# --------------------------------------------------------------------------- #

def precision_recall(real: np.ndarray, fake: np.ndarray, *,
                     k: int = 5) -> Dict[str, float]:
    """Improved precision/recall (Kynkaanniemi et al., 2019).

    A point is "inside" a manifold if it lies within the k-th nearest-neighbour
    radius of some point of that manifold.

      precision — share of GENERATED points inside the real manifold: realism.
      recall    — share of REAL points inside the generated manifold: coverage.

    Both are needed. A model that emits one perfect URL a million times scores
    precision 1.0 and recall ~0, and precision alone would call that a success.
    """
    from sklearn.neighbors import NearestNeighbors

    def radii(points: np.ndarray) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(points))).fit(points)
        distances, _ = nn.kneighbors(points)
        return distances[:, -1]

    def coverage(query: np.ndarray, manifold: np.ndarray,
                 manifold_radii: np.ndarray) -> float:
        nn = NearestNeighbors(n_neighbors=1).fit(manifold)
        # A query point counts if it falls within ANY manifold point's radius,
        # so compare against every radius rather than only the nearest.
        inside = 0
        for i in range(0, len(query), 512):
            block = query[i:i + 512]
            dists = np.linalg.norm(block[:, None, :] - manifold[None, :, :], axis=2)
            inside += int((dists <= manifold_radii[None, :]).any(axis=1).sum())
        return inside / len(query) if len(query) else float("nan")

    real_r, fake_r = radii(real), radii(fake)
    return {"precision": coverage(fake, real, real_r),
            "recall": coverage(real, fake, fake_r),
            "k": k, "n_real": len(real), "n_fake": len(fake)}


# --------------------------------------------------------------------------- #
# Diversity
# --------------------------------------------------------------------------- #

def diversity(urls: Sequence[str]) -> Dict[str, float]:
    """Cheap collapse detectors, computed before anything expensive.

    `distinct_rate` near 1 is necessary but weak — the n-gram baseline achieved
    it while producing gibberish. `self_bleu`-style repetition is approximated
    by distinct character 4-gram ratio, which falls when a model reuses the same
    fragments across samples.
    """
    if not urls:
        return {"distinct_rate": float("nan"), "distinct_4gram_ratio": float("nan")}
    grams: set = set()
    total = 0
    for url in urls:
        for i in range(max(0, len(url) - 3)):
            grams.add(url[i:i + 4])
            total += 1
    return {"distinct_rate": len(set(urls)) / len(urls),
            "distinct_4gram_ratio": len(grams) / total if total else float("nan"),
            "n": len(urls)}


# --------------------------------------------------------------------------- #
# Human panel
# --------------------------------------------------------------------------- #

def human_panel(real: Sequence[str], fake: Sequence[str], *,
                seed: int = 20260803) -> Tuple[List[str], List[int]]:
    """Shuffled mix plus an answer key (1 = generated).

    The task is COMPARATIVE sorting, not "could this exist?". Humans judge URLs
    poorly in isolation — a plain n-gram passes that bar on most samples — so the
    weaker question cannot separate a world model from a Markov chain.
    """
    items = [(u, 0) for u in real] + [(u, 1) for u in fake]
    random.Random(seed).shuffle(items)
    return [u for u, _ in items], [k for _, k in items]


# --------------------------------------------------------------------------- #
# Field-scoped discrimination against measured floors
# --------------------------------------------------------------------------- #

#: How each field is extracted from a URL for scoring.
FIELDS = {
    "LBL": lambda u: _parse(u).registrable_label,
    "SUF": lambda u: _parse(u).public_suffix,
    "SUB": lambda u: ".".join(_parse(u).subdomains),
    "PTH": lambda u: _path_body(_parse(u).path or ""),
    "QRY": lambda u: _parse(u).query or "",
    "FULL": lambda u: _strip_scheme(u),
}

#: Which held-out population each field is scored against, keyed by the SAMPLING
#: FRAME the generator draws in.
#:
#: A field must be compared against a population drawn the SAME WAY the generator
#: draws. The two corpora are not two halves of one distribution; they are two
#: sampling frames over the same web:
#:
#:     name       one URL per domain, over every domain      "domain frame"
#:     structure  up to 40 URLs per multi-page site          "page frame"
#:
#: The generator emits ONE URL PER INVENTED HOST and never revisits a host, so it
#: samples the DOMAIN frame and cannot produce page-frame output at all. Routing
#: PTH to `structure` therefore scored it against a frame it cannot sample.
#: Measured on E4 (benign): against `structure` the discriminator gets AUC 0.6440
#: with 0.6013 of that from LENGTH ALONE (real 38.8 chars vs generated 28.5);
#: against `name` the same comparison gives 0.6290 with length at 0.5151 —
#: chance. The model's conditional path length (27.9) matches the name frame
#: (28.3) almost exactly. The length signal was an artifact of the frame, not a
#: defect.
#:
#: This is the same argument that already justifies `_one_per_host` on the real
#: side, applied to the choice of population rather than to repetition within it.
#: It is a consistency fix, not a relaxation: the whole-URL `FULL` pool is 93.9%
#: name-frame hosts after dedup anyway, so both metrics were always asking for
#: domain-frame output.
#:
#: `page` stays defined because it is the correct target ONCE the generator can
#: be conditioned on a frame (the proposed `<F:page>` token). Until then it is
#: reported as a DIAGNOSTIC, never as a gate — scoring an unconditioned generator
#: against it measures the frame gap, which is exactly the error above.
FIELD_POPULATION_BY_FRAME = {
    "domain": {
        "LBL": "name", "SUF": "name", "SUB": "name",
        "PTH": "name", "QRY": "name",
        "FULL": "mixed",
    },
    "page": {
        "LBL": "name", "SUF": "name", "SUB": "name",
        "PTH": "structure", "QRY": "structure",
        "FULL": "mixed",
    },
}

#: The frame the current generator samples. One URL per invented host.
GENERATOR_FRAME = "domain"

#: Backwards-compatible alias: the populations for the frame actually generated.
FIELD_POPULATION = FIELD_POPULATION_BY_FRAME[GENERATOR_FRAME]


def _one_per_host(urls: Sequence[str], seed: int) -> List[str]:
    """Keep one URL per host, chosen uniformly.

    The generator produces ONE URL per invented host, so a real comparator
    holding six pages of the same site measures repetition rather than realism.
    Measured on the phishing structure corpus: train and val separate at 0.7647
    with repetition present and 0.4798 without, against a control of 0.4990 —
    the entire signal was kits producing near-identical paths across their own
    pages, which no generated set can imitate because it never revisits a host.
    """
    rng = random.Random(seed)
    seen: Dict[str, str] = {}
    for url in rng.sample(list(urls), len(urls)):
        parts = _parse(url)
        host = ".".join(tuple(parts.subdomains) +
                        (parts.registrable_label, parts.public_suffix))
        seen.setdefault(host or url, url)
    return list(seen.values())


#: Smallest per-side sample the comparison may shrink to. At n=250 the
#: Hanley-McNeil standard error of a single AUC near 0.5 is ~0.026, so `excess`
#: — a difference of two AUCs — carries ~0.037. That is comparable to the 0.05
#: bar, which is the point at which shrinking further would stop being a
#: measurement. Below it the result is withheld rather than reported weakly.
MIN_N = 250


def discriminate_with_floor(real_pool: Sequence[str], fake: Sequence[str],
                            field: str, *, n: int = 1200,
                            seed: int = 20260806,
                            dedup_by_host: bool = True) -> Dict[str, float]:
    """AUC for one field, reported against the floor measured on real data.

    An AUC of 0.5 is only the right target when two samples of REAL data score
    0.5 against each other. They frequently do not: the split is domain-disjoint,
    the crawls span two years, and the corpus is suffix-stratified. Measured on
    real-vs-real, ordinary variation alone runs 0.51 (same crawl) to 0.79
    (different sampler), so a bar set at an absolute AUC can sit BELOW its own
    floor and be unreachable by any generator, including a perfect one. That is
    not hypothetical — it is what happened to the pre-registered 0.65.

    So three disjoint real samples are drawn. Two establish the floor; the third
    is what the generated set is scored against, at identical size. `excess` is
    the number that means anything:

        excess ~ 0     indistinguishable, given how variable real data is
        excess > 0     a real tell, over and above natural variation
    """
    extract = FIELDS[field]
    rng = random.Random(seed)

    source = _one_per_host(real_pool, seed) if dedup_by_host else real_pool
    pool = [x for x in (extract(u) for u in source) if x]
    fakes = [x for x in (extract(u) for u in fake) if x]

    # Shrink to what the pool supports rather than refusing outright. The
    # phishing structure pool holds 941 distinct hosts against a requested
    # n=1200, so an all-or-nothing rule would blank the phishing path and query
    # fields — the two this project most needs to watch. A reduced n is a weaker
    # measurement, not a wrong one, and it is flagged so it reads as such.
    requested = n
    n = min(n, len(pool) // 3, len(fakes))
    if n < requested and n < MIN_N or n < 1:
        return {"auc": float("nan"), "floor": float("nan"),
                "excess": float("nan"), "n_real": len(pool),
                "n_fake": len(fakes), "requested_n": requested,
                "insufficient": True}

    picked = rng.sample(pool, 3 * n)
    real_a, real_b, real_c = picked[:n], picked[n:2 * n], picked[2 * n:]
    fakes = rng.sample(fakes, n)

    floor = real_vs_fake_auc(real_a, real_b, seed=seed)["auc_mean"]
    auc = real_vs_fake_auc(real_c, fakes, seed=seed)["auc_mean"]
    # Hanley-McNeil standard error, evaluated at A=0.5 where Q1 = Q2 = 1/3 and
    # the expression collapses to sqrt(1/6n). `excess` is a difference of two
    # AUCs estimated on disjoint samples, so its variance is the sum of theirs.
    # Reported per row because a fixed error bar would understate the small
    # samples and overstate the large ones by the same factor.
    se = round(((0.25 + 2 * (n - 1) / 12) / n ** 2) ** 0.5 * 2 ** 0.5, 4)
    return {"auc": round(auc, 4), "floor": round(floor, 4),
            "excess": round(auc - floor, 4), "n": n, "se": se,
            "sigma": round((auc - floor) / se, 2) if se else float("nan"),
            "reduced_n": n < requested,
            "field_coverage_real": round(len(pool) / max(len(real_pool), 1), 4),
            "field_coverage_fake": round(len(fakes) / max(len(fake), 1), 4)}


def field_report(populations: Dict[str, Sequence[str]],
                 fake: Sequence[str], *, n: int = 1200,
                 seed: int = 20260806,
                 frame: str = GENERATOR_FRAME) -> Dict[str, dict]:
    """Score every field against the population drawn in the generator's FRAME.

    `populations` maps "name"/"structure"/"mixed" to held-out real URL lists.
    `frame` selects which mapping to use; it must match how `fake` was sampled,
    or the result measures the frame gap rather than generation quality.
    """
    if frame not in FIELD_POPULATION_BY_FRAME:
        raise KeyError(f"unknown frame {frame!r}; "
                       f"known: {sorted(FIELD_POPULATION_BY_FRAME)}")
    out: Dict[str, dict] = {}
    for field, pop in FIELD_POPULATION_BY_FRAME[frame].items():
        real = populations.get(pop)
        if not real:
            continue
        out[field] = discriminate_with_floor(real, fake, field, n=n, seed=seed)
        out[field]["population"] = pop
        out[field]["frame"] = frame
        out[field]["dedup_by_host"] = True
    return out
