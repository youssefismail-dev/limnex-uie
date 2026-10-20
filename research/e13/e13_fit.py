"""E13 D2 -- fit the field calibration for the two field-led scopes, then confirm it
once on validation (research/e13/PREREG_E13.md section 2.3). DC1's objective and
rules (research/dc1/PREREG_DC1.md section 3), adapted to E12's tokenizer, where
the plan follows the supplied suffix:

  - per scope, (T, delta) minimises the teacher-forced cross-entropy of real
    TRAINING URLs of the scope's class and state at the steps the sampler SAMPLES
    inside the field, under exactly what `uie.model.generate` does there: delta
    on every token that ends a field (`struct_ids`: the markers, <RS>, <EOS>),
    then divide by T. The forced suffix, plan and <LBL> are never scored;
  - L-BFGS-B from (1, 0) within T in [0.5, 2], delta in [-3, 3];
  - the VALIDATION split is used once: the per-URL paired cross-entropy
    difference (fitted - default) over the field's positions must have a 95% CI
    below 0, or the scope stays at the defaults. Nothing here reads the gate.

    python research/e13/e13_fit.py --checkpoint data/e12/train/e12_final/final.pt \
        --bundle data/bundle2 --out data/e13/d2_settings.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e13_common as E13  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_urls(bundle: Path, label: str, split: str) -> List[str]:
    pre = "" if label == "benign" else f"{label}_"
    with gzip.open(Path(bundle) / f"{pre}name_{split}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


class Layout:
    """E12's tokenizer, as the sampler sees it: the prefix [BOS, class, frame], the
    first <SUF> sampled, then the suffix characters, the plan and <LBL> forced."""

    def __init__(self, tok):
        from uie.tokenizer import _MARKERS
        assert tok.presence_plan and tok.slash_plan and tok.include_frame
        assert not getattr(tok, "include_language", False)
        self.tok = tok
        self.markers = {tok.marker_id(m): m for m in _MARKERS}
        self.suf, self.lbl, self.pad = tok.marker_id("SUF"), tok.marker_id("LBL"), tok.pad_id
        self.n_prefix = 3
        self.V = tok.vocab_size
        self.stop_mask = torch.zeros(self.V)
        self.stop_mask[list(tok.struct_ids())] = 1.0
        self.names = {v: k for k, v in tok.stoi.items()}

    def state(self, ids: List[int]) -> str:
        """The presence-slash state the encoding's own plan states, e.g. P0Q1S1."""
        k = ids.index(self.lbl)
        p, q, s = (self.names[t] for t in ids[k - 3:k])
        return f"P{p[-2]}Q{q[-2]}S{s[-2]}"

    def steps(self, ids: List[int]):
        """(t, field, target) for every SAMPLED step: the logits at position t
        predict ids[t + 1]; `field` is the sampler's tracker (the last marker in
        ids[:t + 1]). The forced suffix, plan and <LBL> are skipped."""
        j0 = ids.index(self.suf, self.n_prefix)
        j1 = ids.index(self.lbl, j0)
        field = None
        for t in range(self.n_prefix - 1, len(ids) - 1):
            if ids[t] in self.markers:
                field = self.markers[ids[t]]
            j = t + 1
            if j0 < j <= j1:
                continue                                 # forced
            yield t, field, ids[j]


def transformed(z: torch.Tensor, layout: Layout, T, d) -> torch.Tensor:
    """What the sampler draws from inside the field: bias, then temperature."""
    return (z + d * layout.stop_mask) / T


@torch.no_grad()
def logits_for(model, seqs: List[List[int]], pad: int) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    x = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], dtype=torch.long)
    return model(x).float()


def select(layout: Layout, urls: List[str], label: str, states, max_len: int, quota: int,
           seed: int) -> List[List[int]]:
    order = list(range(len(urls)))
    random.Random(seed).shuffle(order)
    out = []
    for i in order:
        if len(out) >= quota:
            break
        try:
            ids = layout.tok.encode(urls[i], label, frame="domain")
        except Exception:
            continue
        if len(ids) <= max_len and layout.state(ids) in states:
            out.append(ids)
    return out


def collect(model, layout: Layout, seqs: List[List[int]], field: str, cap: int, seed: int,
            batch: int = 64) -> dict:
    picks = [(i, t, y) for i, ids in enumerate(seqs) for t, f, y in layout.steps(ids) if f == field]
    if len(picks) > cap:
        picks = random.Random(seed).sample(picks, cap)
    by_url: Dict[int, list] = {}
    for slot, (i, t, y) in enumerate(picks):
        by_url.setdefault(i, []).append((slot, t))
    z = torch.empty(len(picks), layout.V)
    for b0 in range(0, len(by_url), batch):
        idx = sorted(by_url)[b0:b0 + batch]
        zz = logits_for(model, [seqs[i] for i in idx], layout.pad)
        for r, i in enumerate(idx):
            for slot, t in by_url[i]:
                z[slot] = zz[r, t]
    return {"z": z, "y": torch.tensor([p[2] for p in picks], dtype=torch.long),
            "n_urls": len(by_url)}


def fit(data: dict, layout: Layout) -> dict:
    from scipy.optimize import minimize
    z, y = data["z"], data["y"]
    if len(y) == 0:
        return {"n_positions": 0}

    def objective(x):
        p = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        loss = F.cross_entropy(transformed(z, layout, p[0], p[1]), y)
        loss.backward()
        return float(loss.detach()), p.grad.double().numpy()

    x0 = [1.0, 0.0]
    res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                   bounds=[E13.BOUNDS["T"], E13.BOUNDS["delta"]])
    return {"T": float(res.x[0]), "delta": float(res.x[1]), "ce_default": objective(x0)[0],
            "ce_fit": float(res.fun), "converged": bool(res.success),
            "n_positions": int(len(y)), "n_urls": int(data["n_urls"])}


@torch.no_grad()
def confirm(model, layout: Layout, seqs: List[List[int]], field: str, g: dict,
            batch: int = 64) -> dict:
    """Per URL: the summed CE over the field's sampled positions, fitted minus
    default; mean and 95% CI across URLs (DC1's rule)."""
    diffs = []
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        zz = logits_for(model, [seqs[i] for i in idx], layout.pad)
        for r, i in enumerate(idx):
            rows = [(t, y) for t, f, y in layout.steps(seqs[i]) if f == field]
            if not rows:
                continue
            ts = torch.tensor([t for t, _ in rows])
            ys = torch.tensor([y for _, y in rows])
            a = zz[r, ts]
            ce0 = F.cross_entropy(transformed(a, layout, 1.0, 0.0), ys, reduction="sum")
            ce1 = F.cross_entropy(transformed(a, layout, g["T"], g["delta"]), ys, reduction="sum")
            diffs.append(float(ce1 - ce0))
    if len(diffs) < 2:
        return {"n_urls": len(diffs), "pass": False}
    m = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1))
    half = 1.96 * sd / math.sqrt(len(diffs))
    return {"n_urls": len(diffs), "mean": m, "lo": m - half, "hi": m + half,
            "pass": (m + half) < 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "e13" / E13.SETTINGS_FILE)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="smoke tests only: multiplies every registered sample size")
    args = ap.parse_args()
    from uie.train import load_checkpoint
    torch.manual_seed(0)
    model, tok, cfg = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    model.eval()
    layout = Layout(tok)
    n_fit, n_val = int(E13.N_FIT * args.scale), int(E13.N_VAL * args.scale)
    cap = int(E13.MAX_POSITIONS * args.scale)
    report = {"registration": E13.REGISTRATION, "purpose": "E13 D2 settings (section 2.3)",
              "checkpoint_sha256": sha256(args.checkpoint), "scale": args.scale,
              "bounds": E13.BOUNDS, "quotas": {"fit": n_fit, "val": n_val, "positions": cap},
              "seeds": {"train": E13.FIT_SEED_TRAIN, "val": E13.FIT_SEED_VAL}, "scopes": {}}
    cache: Dict[tuple, List[str]] = {}
    for k, (name, (label, states, field)) in enumerate(E13.D2_SCOPES.items()):
        t0 = time.time()
        for split in ("train", "val"):
            if (label, split) not in cache:
                cache[(label, split)] = read_urls(args.bundle, label, split)
        train = select(layout, cache[(label, "train")], label, states, cfg.max_len, n_fit,
                       E13.FIT_SEED_TRAIN + k)
        val = select(layout, cache[(label, "val")], label, states, cfg.max_len, n_val,
                     E13.FIT_SEED_VAL + k)
        data = collect(model, layout, train, field, cap, E13.FIT_SEED_TRAIN + 100 * (k + 1))
        r = fit(data, layout)
        r["val"] = confirm(model, layout, val, field, r) if "T" in r else {"n_urls": 0, "pass": False}
        r["applied"] = bool(r["val"].get("pass"))
        r.update({"class": label, "states": list(states), "field": field,
                  "n_train_urls": len(train), "n_val_urls": len(val),
                  "seconds": round(time.time() - t0, 1)})
        report["scopes"][name] = r
        print(f"{name} ({label} {'/'.join(states)} {field}): T {r.get('T', float('nan')):.3f} "
              f"delta {r.get('delta', float('nan')):+.3f}  CE {r.get('ce_default', float('nan')):.4f}"
              f" -> {r.get('ce_fit', float('nan')):.4f}  val dCE/url "
              f"{r['val'].get('mean', float('nan')):+.4f} [{r['val'].get('lo', float('nan')):+.4f}, "
              f"{r['val'].get('hi', float('nan')):+.4f}]  {'APPLIED' if r['applied'] else 'not applied'}"
              f"  ({r['seconds']:.0f}s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}  sha256 {sha256(args.out)}")


if __name__ == "__main__":
    main()
