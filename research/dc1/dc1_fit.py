"""DC1 -- fit calibrated per-field decoding for E7, then confirm it (PREREG_DC1.md s3).

For each class, mode (free; each FULL CONTROL state) and field f in LBL, SUB,
PTH, QRY, the settings (T_f, delta_f, and w_f in control modes) minimise the
teacher-forced cross-entropy of real TRAINING URLs under exactly what the
sampler does (uie/model.py):

  - blend (control modes): content characters z_free + w (z_control - z_free);
  - the frozen path delta on the end token where the LEGACY tracker is in PTH;
  - delta_f on every token that ends f, where DC1's tracker is in f;
  - divide by T_f;

scored only at the steps the sampler SAMPLES: with a supplied suffix, the
suffix characters and the <LBL> after them are forced, so never scored.

Then the VALIDATION split -- the gate's real data -- is used once per (class,
mode, field): the per-URL paired cross-entropy difference (fitted - default)
must have a 95% CI below 0, or that field stays at the defaults. Nothing here
reads the gate.

    python research/dc1/dc1_fit.py --checkpoint data/best_e7.pt \
        --bundle data/bundle2 --out data/dc1
"""
import argparse
import gzip
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from dc1_common import (BOUNDS, FIELDS_FIT, FIT_SEED, FROZEN_DELTA,  # noqa: E402
                        MAX_POSITIONS, MODES, N_FIT, N_VAL, STATES)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_urls(bundle: Path, label: str, split: str) -> List[str]:
    pre = "" if label == "benign" else f"{label}_"
    with gzip.open(Path(bundle) / f"{pre}name_{split}.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


class Layout:
    """One tokenizer's ids and prefix geometry, as the sampler sees them."""

    def __init__(self, tok):
        from uie.tokenizer import _MARKERS
        self.tok = tok
        self.markers = {tok.marker_id(m): m for m in _MARKERS}
        self.suf, self.lbl = tok.marker_id("SUF"), tok.marker_id("LBL")
        self.eos, self.pad = tok.eos_id, tok.pad_id
        pos = tok.field_control_positions()
        self.ctrl_pos = [pos["PTH"], pos["QRY"]]
        self.n_prefix = max(pos.values()) + 1
        self.free_ctrl = tok.field_control_ids(None)
        self.p1 = tok.field_control_id("PTH", True)
        self.q1 = tok.field_control_id("QRY", True)
        self.V = tok.vocab_size
        self.stop_mask = torch.zeros(self.V)
        self.stop_mask[list(tok.struct_ids())] = 1.0
        self.content_mask = torch.zeros(self.V)
        self.content_mask[sorted(tok.stoi[c] for c in tok._chars)] = 1.0

    def state(self, ids: List[int]) -> str:
        """The state a URL's own controls encode (the tokenizer's presence)."""
        return (f"P{int(ids[self.ctrl_pos[0]] == self.p1)}"
                f"Q{int(ids[self.ctrl_pos[1]] == self.q1)}")

    def free(self, ids: List[int]) -> List[int]:
        out = list(ids)
        for k, p in enumerate(self.ctrl_pos):
            out[p] = self.free_ctrl[k]
        return out

    def steps(self, ids: List[int]):
        """(t, dc1_field, legacy_field, target) for every SAMPLED step, as the
        sampler produces it with a supplied suffix: the first <SUF> is sampled
        and triggers injection; the suffix characters and the next <LBL> are
        forced. DC1's tracker follows every marker; the legacy one ignores the
        injected <SUF> and the forced markers (uie/model.py)."""
        first_suf = next((j for j in range(self.n_prefix, len(ids))
                          if ids[j] == self.suf), None)
        forced_end = None
        if first_suf is not None:
            forced_end = next((j for j in range(first_suf + 1, len(ids))
                               if ids[j] == self.lbl), len(ids) - 1)
        dc1_f = legacy_f = None
        out = []
        for j in range(self.n_prefix, len(ids)):
            forced = first_suf is not None and first_suf < j <= forced_end
            if not forced:
                out.append((j - 1, dc1_f, legacy_f, ids[j]))
            token = ids[j]
            if token in self.markers:
                dc1_f = self.markers[token]
                if j != first_suf and not forced:
                    legacy_f = self.markers[token]
        return out


def adjusted(zc: torch.Tensor, zu: Optional[torch.Tensor], legacy_pth: torch.Tensor,
             layout: Layout, frozen: float, T, d, w=None) -> torch.Tensor:
    """The logits the sampler draws from, for rows of positions of ONE field.
    T, d, w may be floats or 0-d tensors (for gradients)."""
    z = zc + d * layout.stop_mask
    if legacy_pth.any():
        z = z.clone()
        z[legacy_pth, layout.eos] -= frozen
    if w is not None and zu is not None:
        z = z + (w - 1.0) * (zc - zu) * layout.content_mask
    return z / T


@torch.no_grad()
def logits_for(model, seqs: List[List[int]], pad: int) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    x = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], dtype=torch.long)
    return model(x).float()


def sequence_logp(model, layout: Layout, ids: List[int], label: str,
                  settings: Dict[str, dict], blend_ok: bool) -> float:
    """Summed log-probability of one sequence's sampled tokens under DC1
    settings {field: {"T", "delta", "w"}} -- the sampler's `logp`, recomputed."""
    zc = logits_for(model, [ids], layout.pad)[0]
    zu = logits_for(model, [layout.free(ids)], layout.pad)[0] if blend_ok else None
    total = 0.0
    frozen = FROZEN_DELTA[label]
    for t, f, legacy, y in layout.steps(ids):
        s = settings.get(f or "", {})
        z = adjusted(zc[t:t + 1], None if zu is None else zu[t:t + 1],
                     torch.tensor([legacy == "PTH"]), layout, frozen,
                     s.get("T", 1.0), s.get("delta", 0.0),
                     s.get("w") if blend_ok and f in settings else None)
        total += float(F.log_softmax(z, dim=-1)[0, y])
    return total


def select(layout: Layout, urls: List[str], label: str, max_len: int,
           quotas: Dict[str, int], seed: int) -> Dict[str, List[List[int]]]:
    """Seeded, encodable samples: 'free' from every state, each state from its
    own URLs (as the tokenizer encodes their presence)."""
    order = list(range(len(urls)))
    random.Random(seed).shuffle(order)
    out = {m: [] for m in quotas}
    for i in order:
        if all(len(out[m]) >= q for m, q in quotas.items()):
            break
        try:
            ids = layout.tok.encode(urls[i], label, frame="domain")
        except Exception:
            continue
        if len(ids) > max_len:
            continue
        if "free" in quotas and len(out["free"]) < quotas["free"]:
            out["free"].append(layout.free(ids))
        st = layout.state(ids)
        if st in quotas and len(out[st]) < quotas[st]:
            out[st].append(ids)
    return out


def collect(model, layout: Layout, seqs: List[List[int]], blend_ok: bool,
            cap: int, seed: int, batch: int = 64) -> Dict[str, dict]:
    """Per field: the sampled positions' logits (control, and free when
    blending), targets, URL index and legacy-PTH flag -- a seeded subsample of
    at most `cap` positions."""
    picks = {f: [] for f in FIELDS_FIT}
    for i, ids in enumerate(seqs):
        for t, f, legacy, y in layout.steps(ids):
            if f in picks:
                picks[f].append((i, t, legacy == "PTH", y))
    for k, f in enumerate(FIELDS_FIT):
        if len(picks[f]) > cap:
            picks[f] = random.Random(seed + 101 * (k + 1)).sample(picks[f], cap)
    by_url: Dict[int, list] = {}
    for f in FIELDS_FIT:
        for slot, (i, t, lp, y) in enumerate(picks[f]):
            by_url.setdefault(i, []).append((f, slot, t))
    out = {f: {"zc": torch.empty(len(picks[f]), layout.V),
               "zu": torch.empty(len(picks[f]), layout.V) if blend_ok else None,
               "y": torch.tensor([p[3] for p in picks[f]], dtype=torch.long),
               "url": torch.tensor([p[0] for p in picks[f]], dtype=torch.long),
               "lpth": torch.tensor([p[2] for p in picks[f]], dtype=torch.bool)}
           for f in FIELDS_FIT}
    urls = sorted(by_url, key=lambda i: len(seqs[i]))
    for b0 in range(0, len(urls), batch):
        idx = urls[b0:b0 + batch]
        zc = logits_for(model, [seqs[i] for i in idx], layout.pad)
        zu = logits_for(model, [layout.free(seqs[i]) for i in idx], layout.pad) if blend_ok else None
        for r, i in enumerate(idx):
            for f, slot, t in by_url[i]:
                out[f]["zc"][slot] = zc[r, t]
                if blend_ok:
                    out[f]["zu"][slot] = zu[r, t]
    return out


def fit_field(data: dict, layout: Layout, frozen: float, blend_ok: bool) -> dict:
    """L-BFGS-B within the registered bounds, from the defaults."""
    from scipy.optimize import minimize
    zc, zu, y, lpth = data["zc"], data["zu"], data["y"], data["lpth"]
    if len(y) == 0:
        return {"n_positions": 0}

    def objective(x):
        p = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        z = adjusted(zc, zu, lpth, layout, frozen, p[0], p[1], p[2] if blend_ok else None)
        loss = F.cross_entropy(z, y)
        loss.backward()
        return float(loss.detach()), p.grad.double().numpy()

    x0 = [1.0, 0.0] + ([1.0] if blend_ok else [])
    bounds = [BOUNDS["T"], BOUNDS["delta"]] + ([BOUNDS["w"]] if blend_ok else [])
    res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    ce_default = objective(x0)[0]
    return {"T": float(res.x[0]), "delta": float(res.x[1]),
            "w": float(res.x[2]) if blend_ok else None,
            "ce_default": ce_default, "ce_fit": float(res.fun),
            "converged": bool(res.success), "n_positions": int(len(y)),
            "n_urls": int(len(torch.unique(data["url"])))}


@torch.no_grad()
def confirm(model, layout: Layout, seqs: List[List[int]], label: str, fits: dict,
            blend_ok: bool, batch: int = 64) -> Dict[str, dict]:
    """Per field: per-URL summed CE, fitted minus default, over every sampled
    position of the field; mean and 95% CI across URLs."""
    frozen = FROZEN_DELTA[label]
    diffs = {f: {} for f in FIELDS_FIT}
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        zc = logits_for(model, [seqs[i] for i in idx], layout.pad)
        zu = logits_for(model, [layout.free(seqs[i]) for i in idx], layout.pad) if blend_ok else None
        for r, i in enumerate(idx):
            per = {f: [] for f in FIELDS_FIT}
            for t, f, legacy, y in layout.steps(seqs[i]):
                if f in per:
                    per[f].append((t, legacy == "PTH", y))
            for f, rows in per.items():
                if not rows or "T" not in fits.get(f, {}):
                    continue
                ts = torch.tensor([t for t, _, _ in rows])
                lp = torch.tensor([b for _, b, _ in rows])
                ys = torch.tensor([y for _, _, y in rows])
                a, u = zc[r, ts], (zu[r, ts] if blend_ok else None)
                ce0 = F.cross_entropy(adjusted(a, u, lp, layout, frozen, 1.0, 0.0,
                                               1.0 if blend_ok else None), ys, reduction="sum")
                g = fits[f]
                ce1 = F.cross_entropy(adjusted(a, u, lp, layout, frozen, g["T"], g["delta"],
                                               g["w"] if blend_ok else None), ys, reduction="sum")
                diffs[f][i] = float(ce1 - ce0)
    out = {}
    for f, d in diffs.items():
        v = list(d.values())
        if len(v) < 2:
            out[f] = {"n_urls": len(v), "pass": False}
            continue
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
        half = 1.96 * sd / math.sqrt(len(v))
        out[f] = {"n_urls": len(v), "mean": m, "lo": m - half, "hi": m + half,
                  "pass": (m + half) < 0.0}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="smoke tests only: multiplies every registered sample size")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from uie.train import load_checkpoint
    torch.manual_seed(0)
    model, tok, _ = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    model.eval()
    layout = Layout(tok)
    max_len = model.cfg.max_len
    scale = args.scale
    q_fit = {"free": int(N_FIT["free"] * scale), **{s: int(N_FIT["state"] * scale) for s in STATES}}
    q_val = {m: int(N_VAL * scale) for m in MODES}

    report = {"registration": "research/dc1/PREREG_DC1.md",
              "checkpoint_sha256": sha256(args.checkpoint), "scale": scale,
              "bounds": BOUNDS, "frozen_delta": FROZEN_DELTA,
              "quotas": {"fit": q_fit, "val": q_val}, "classes": {}}
    for c, label in enumerate(("benign", "phishing")):
        t0 = time.time()
        train = select(layout, read_urls(args.bundle, label, "train"), label, max_len,
                       q_fit, FIT_SEED + c)
        val = select(layout, read_urls(args.bundle, label, "val"), label, max_len,
                     q_val, FIT_SEED + 1000 + c)
        print(f"{label}: selected train {{{', '.join(f'{m}: {len(v)}' for m, v in train.items())}}} "
              f"val {{{', '.join(f'{m}: {len(v)}' for m, v in val.items())}}} "
              f"in {time.time() - t0:.0f}s", flush=True)
        report["classes"][label] = {}
        for k, mode in enumerate(MODES):
            blend_ok = mode != "free"
            t1 = time.time()
            data = collect(model, layout, train[mode], blend_ok, MAX_POSITIONS,
                           FIT_SEED + 10 * k + c)
            fits = {f: fit_field(data[f], layout, FROZEN_DELTA[label], blend_ok)
                    for f in FIELDS_FIT}
            conf = confirm(model, layout, val[mode], label, fits, blend_ok)
            cell = {}
            for f in FIELDS_FIT:
                r = dict(fits[f])
                r["val"] = conf.get(f, {"n_urls": 0, "pass": False})
                r["applied"] = bool(r["val"].get("pass")) and "T" in r
                cell[f] = r
            report["classes"][label][mode] = cell
            print(f"  {label} {mode:<5} " + "  ".join(
                f"{f} T {r.get('T', float('nan')):.3f} d {r.get('delta', float('nan')):+.3f}"
                + (f" w {r['w']:.3f}" if r.get('w') is not None else "")
                + f" dCE/url {r['val'].get('mean', float('nan')):+.4f}"
                + (" APPLIED" if r["applied"] else " -")
                for f, r in cell.items()) + f"  ({time.time() - t1:.0f}s)", flush=True)
            (args.out / "settings.json").write_text(json.dumps(report, indent=1),
                                                    encoding="utf-8")
    print("wrote", args.out / "settings.json")


if __name__ == "__main__":
    main()
