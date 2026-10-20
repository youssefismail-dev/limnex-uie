"""
research/probe/run_probe.py — the representation probe, read once (research/probe/PROBE_PLAN.md).

    python research/probe/run_probe.py

  0. The sets are the frozen ones (sha256 = SETS.json).
  V1. The hidden-state pass is the model's real forward pass: log P(url | benign) computed from
      its logits matches TypicalityScorer.log_prob on 100 URLs within 1e-4 nats. Else STOP.
  1. Features. UIE (e16_cap/final.pt) and the untrained control (same architecture, random
     weights, seed 20261017): the state after the embedding and after each block (9 layers),
     averaged over every token after the fixed three-token prefix. Surface control: character
     1-4-gram counts (hashed, 2^18, log1p) of the URL text UIE reads (scheme dropped, host
     lower-cased).
  2. Layer choice on each set's 'select' part (5-fold inside it), separately for UIE and the control.
  3. The read: out-of-fold predictions on the 80% (the stored folds); accuracy; 2,000 bootstrap
     resamples over URLs for intervals and paired differences.
  V2. Positive control on the all-TLD set: country vs generic TLD from UIE's chosen layer,
      balanced accuracy >= 0.95. Else the reading is not made.
  4. The reading on the main set: REPRESENTS LANGUAGE / NOT SHOWN.
Writes research/probe/results.json. Changes nothing in uie/.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "data/e16/train/e16_cap/final.pt"
SETS = ROOT / "research/probe/SETS.json"
OUT = ROOT / "research/probe/results.json"
SEED = 20261017
PREFIX = 3                      # <bos>, class, frame
BOOT = 2_000
V1_TOL = 1e-4
V2_BAR = 0.95
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def log(t0, msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

def pooled_states(model, scorer, urls, batch=64):
    """(layers x urls x d_model) mean states over the tokens after PREFIX, and log P(url | benign)
    from the same forward pass (positions 0..n-2 predict 1..n-1, as sequence_log_prob)."""
    import torch
    import torch.nn.functional as F
    enc = [scorer.encode(u, "benign").ids for u in urls]
    pad = scorer.tokenizer.pad_id
    n_layers = model.cfg.n_layers + 1
    out = np.zeros((n_layers, len(urls), model.cfg.d_model), np.float32)
    logp = np.zeros(len(urls))
    captured = []
    hooks = [model.drop.register_forward_hook(lambda m, i, o: captured.append(o))]
    hooks += [b.register_forward_hook(lambda m, i, o: captured.append(o)) for b in model.blocks]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    try:
        with torch.no_grad():
            for start in range(0, len(order), batch):
                idx = order[start:start + batch]
                width = max(len(enc[i]) for i in idx)
                ids = torch.tensor([enc[i] + [pad] * (width - len(enc[i])) for i in idx], dtype=torch.long)
                captured.clear()
                logits = model(ids)
                if len(captured) != n_layers:
                    raise RuntimeError(f"captured {len(captured)} states, expected {n_layers}")
                lp = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                logp[idx] = (lp * (ids[:, 1:] != pad)).sum(-1).numpy()
                mask = torch.zeros(len(idx), width)
                for k, i in enumerate(idx):
                    mask[k, PREFIX:len(enc[i])] = 1.0
                denom = mask.sum(1, keepdim=True)
                for layer, h in enumerate(captured):
                    out[layer, idx] = ((h * mask.unsqueeze(-1)).sum(1) / denom).numpy()
    finally:
        for h in hooks:
            h.remove()
    return out, logp


def surface_text(url: str) -> str:
    """The URL text UIE reads: scheme dropped, host lower-cased (uie.typicality.normalize_url)."""
    from uie.typicality import normalize_url
    return _SCHEME.sub("", normalize_url(url))


def surface_features(urls):
    from sklearn.feature_extraction.text import HashingVectorizer
    vec = HashingVectorizer(analyzer="char", ngram_range=(1, 4), n_features=2 ** 18, alternate_sign=False,
                            norm=None, lowercase=False)
    X = vec.transform([surface_text(u) for u in urls]).astype(np.float64)
    X.data = np.log1p(X.data)
    return X


# --------------------------------------------------------------------------- #
# the reader
# --------------------------------------------------------------------------- #

def reader(dense: bool):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr = LogisticRegression(C=1.0, max_iter=3000)
    return make_pipeline(StandardScaler(), lr) if dense else lr


def out_of_fold(X, y, folds, dense=True):
    """Predictions for every row from a reader trained on the other folds (fold ids 0-4)."""
    pred = np.empty(len(y), dtype=object)
    for k in range(5):
        te = folds == k
        model = reader(dense).fit(X[~te], y[~te])
        pred[te] = model.predict(X[te])
    return pred


def inner_folds(langs):
    """Deterministic folds inside the select part: each language's rows dealt round-robin in file order."""
    seen, folds = {}, np.empty(len(langs), int)
    for i, l in enumerate(langs):
        folds[i] = seen.get(l, 0) % 5
        seen[l] = seen.get(l, 0) + 1
    return folds


def choose_layer(states, y, folds):
    acc = [float(np.mean(out_of_fold(states[l], y, folds) == y)) for l in range(states.shape[0])]
    return int(np.argmax(acc)), acc


def bootstrap(correct: dict, n=BOOT, seed=SEED):
    """Accuracy intervals per representation and paired differences, on the same resamples."""
    rng = np.random.default_rng(seed)
    names = list(correct)
    m = len(next(iter(correct.values())))
    boots = {k: np.empty(n) for k in names}
    for b in range(n):
        idx = rng.integers(0, m, m)
        for k in names:
            boots[k][b] = correct[k][idx].mean()
    res = {k: {"acc": float(correct[k].mean()), "ci": [float(x) for x in np.quantile(boots[k], [0.025, 0.975])]}
           for k in names}
    for a, b in (("uie", "untrained"), ("uie", "surface")):
        d = boots[a] - boots[b]
        res[f"{a}_minus_{b}"] = {"diff": float(correct[a].mean() - correct[b].mean()),
                                 "ci": [float(x) for x in np.quantile(d, [0.025, 0.975])]}
    return res


def balanced_accuracy(y, pred):
    classes = sorted(set(y))
    return float(np.mean([np.mean(pred[y == c] == c) for c in classes]))


# --------------------------------------------------------------------------- #

def main() -> None:
    import torch
    from uie.model import URLTransformer
    from uie.train import load_checkpoint
    from uie.typicality import TypicalityScorer

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    t0 = time.time()
    rec = json.loads(SETS.read_text(encoding="utf-8"))
    data = {}
    for name, info in rec["sets"].items():
        path = ROOT / info["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise SystemExit(f"STOP: {path} is not the frozen set")
        data[name] = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    chance = rec["chance"]
    res = {"plan": "research/probe/PROBE_PLAN.md", "languages": rec["languages"], "chance": chance}

    torch.manual_seed(0)
    model, tok, _ = load_checkpoint(CKPT, device=torch.device("cpu"))
    model.eval()
    scorer = TypicalityScorer(model, tok)
    torch.manual_seed(SEED)
    untrained = URLTransformer(model.cfg).eval()
    log(t0, f"models loaded: {model.cfg.n_layers} layers, d_model {model.cfg.d_model}; chance {chance:.3f}")

    # ---- V1: the extraction is the model's real forward pass ----------------------------
    probe_urls = [r["url"] for r in data["main"][:100]]
    _, lp = pooled_states(model, scorer, probe_urls)
    ref = np.asarray(scorer.log_prob(probe_urls, "benign"))
    v1 = float(np.abs(lp - ref).max())
    res["V1_forward_pass_max_diff_nats"] = v1
    log(t0, f"V1: hooked forward vs scorer log P, max |diff| {v1:.2e} nats ({'PASS' if v1 <= V1_TOL else 'FAIL'})")
    if v1 > V1_TOL:
        res["reading"] = "STOP (V1)"
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        raise SystemExit("STOP: V1")

    for name in ("all", "main"):
        rows = data[name]
        urls = [r["url"] for r in rows]
        y = np.array([r["lang"] for r in rows], dtype=object)
        part = np.array([r["part"] for r in rows])
        sel = part == "select"
        folds = np.array([int(p[4:]) if p != "select" else -1 for p in part])
        feats = {}
        for key, m in (("uie", model), ("untrained", untrained)):
            t1 = time.time()
            feats[key], _ = pooled_states(m, scorer, urls)
            log(t0, f"{name}: {key} states for {len(urls):,} URLs in {time.time() - t1:.0f}s")
        surface = surface_features(urls)

        # layer choice on the select part, separately per model
        ifold = inner_folds(y[sel])
        chosen, curves_sel = {}, {}
        for key in ("uie", "untrained"):
            chosen[key], curves_sel[key] = choose_layer(feats[key][:, sel], y[sel], ifold)
        log(t0, f"{name}: layers chosen on 'select': UIE {chosen['uie']}, untrained {chosen['untrained']}")

        # the read on the 80%
        rd = ~sel
        preds = {"uie": out_of_fold(feats["uie"][chosen["uie"]][rd], y[rd], folds[rd]),
                 "untrained": out_of_fold(feats["untrained"][chosen["untrained"]][rd], y[rd], folds[rd]),
                 "surface": out_of_fold(surface[np.flatnonzero(rd)], y[rd], folds[rd], dense=False)}
        correct = {k: (v == y[rd]).astype(float) for k, v in preds.items()}
        stats = bootstrap(correct)
        curve = {key: [float(np.mean(out_of_fold(feats[key][l][rd], y[rd], folds[rd]) == y[rd]))
                       for l in range(feats[key].shape[0])] for key in ("uie", "untrained")}
        per_lang = {l: float(np.mean(preds["uie"][y[rd] == l] == l)) for l in rec["languages"]}
        confusion = {t: {p: int(np.sum((y[rd] == t) & (preds["uie"] == p))) for p in rec["languages"]}
                     for t in rec["languages"]}
        res[name] = {"layers_chosen": chosen, "select_curves": curves_sel, "read_rows": int(rd.sum()),
                     "stats": stats, "layer_curves_80pct_reported": curve, "uie_per_language": per_lang,
                     "uie_confusion": confusion}
        s = stats
        log(t0, f"{name}: UIE {s['uie']['acc']:.3f} [{s['uie']['ci'][0]:.3f}, {s['uie']['ci'][1]:.3f}]  "
                f"untrained {s['untrained']['acc']:.3f}  surface {s['surface']['acc']:.3f}  chance {chance:.3f}")
        log(t0, f"{name}: UIE - untrained {s['uie_minus_untrained']['diff']:+.3f} {s['uie_minus_untrained']['ci']}; "
                f"UIE - surface {s['uie_minus_surface']['diff']:+.3f} {s['uie_minus_surface']['ci']}")

        if name == "all":   # V2: positive control
            kind = np.array([r["tld_kind"] for r in rows], dtype=object)
            pk = out_of_fold(feats["uie"][chosen["uie"]][rd], kind[rd], folds[rd])
            v2 = balanced_accuracy(kind[rd], pk)
            res["V2_tld_kind_balanced_accuracy"] = v2
            log(t0, f"V2: country vs generic TLD from UIE layer {chosen['uie']}: balanced accuracy {v2:.3f} "
                    f"({'PASS' if v2 >= V2_BAR else 'FAIL'})")
            if v2 < V2_BAR:
                res["reading"] = "STOP (V2: the reading pipeline fails its positive control)"
                OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
                raise SystemExit("STOP: V2")

    s = res["main"]["stats"]
    represents = s["uie"]["ci"][0] > chance and s["uie_minus_untrained"]["ci"][0] > 0
    lo, hi = s["uie_minus_surface"]["ci"]
    res["reading"] = "REPRESENTS LANGUAGE" if represents else "NOT SHOWN"
    res["against_surface"] = "beyond surface" if lo > 0 else ("below surface" if hi < 0 else "equal to surface")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(t0, f"READING: {res['reading']}; against the surface control: {res['against_surface']}")


if __name__ == "__main__":
    main()
