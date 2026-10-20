"""PC1 -- why the plan over-predicts benign paths: the measurement
(research/native/PREREG_PC1.md section 3). Laptop only. It reads e10_plan's
checkpoints, the training bundle and e10_plan's training record, and nothing of
E10_generate, e10_base or E8.

    python research/native/pc1_measure.py --part data     # M0, the rows the model trained on
    python research/native/pc1_measure.py --part model    # M1 and V1, V6 (best.pt, final.pt)
    python research/native/pc1_measure.py --part clip     # M2, clip factors on best.pt
    python research/native/pc1_read.py                    # M3, validity, R1-R5, tables

Outputs go to data/pc1/.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from e8_plan_calibration import last_probs, plan_by_suffix  # noqa: E402
from uie.factorization import parse  # noqa: E402
from uie.train import CORPUS_FRAME, _pad, field_mask, load_checkpoint  # noqa: E402

OUT = ROOT / "data" / "pc1"
BUNDLE = ROOT / "data" / "bundle2"
TRAIN = ROOT / "data" / "e10" / "train"
LABELS = ("benign", "phishing")
FRAMES = ("domain", "page")
#: M2 (PREREG_PC1 section 3).
SEED = 20260929
WINDOWS = 12
SHORT = 30              # scored tokens per row: the band training clipped
LONG_EVERY = 8          # every 8th batch of the longer bands, in sorted window order
UNSCORED = 3            # len(ids) - scored tokens: BOS is no target; <C:*>, <F:*> weigh 0


# --------------------------------------------------------------------------- #
# M0 -- one training row, read from its ids exactly as the model saw it
# --------------------------------------------------------------------------- #

def row_record(tok, ids: Sequence[int]) -> Tuple[str, int, int, int, int]:
    """(suffix, path, query, scored tokens, has userinfo) of an encoded plan row.

    The suffix is the characters between `<SUF>` and the plan; the presence labels
    are the plan tokens themselves, which training used as targets."""
    (p0, p1), (q0, q1) = tok.plan_slots()
    suf = tok.marker_id("SUF")
    k = ids.index(suf)
    j = k + 1
    while ids[j] not in (p0, p1):
        j += 1
    suffix = "".join(tok.itos[t] for t in ids[k + 1:j])
    return (suffix, int(ids[j] == p1), int(ids[j + 1] == q1), len(ids) - UNSCORED,
            int(tok.marker_id("USR") in ids[:k]))


def plan_prefix(tok, label: str, frame: str, suffix: str) -> List[int]:
    """The ids the plan is predicted from: BOS, class, frame, `<SUF>`, the suffix."""
    return ([tok.bos_id, tok.stoi[f"<C:{label}>"], tok.stoi[f"<F:{frame}>"],
             tok.marker_id("SUF")] + tok._chars_to_ids(suffix))


def plan_by_frame(model, tok, label: str, suffixes: Sequence[str], frame: str,
                  batch: int = 256) -> Dict[str, dict]:
    """`plan_by_suffix` for any frame, from the prefix built directly (V6: equal to
    `plan_by_suffix` in the domain frame). Adds the full P-slot distribution's
    legal mass and the Q slot's."""
    (p0, p1), (q0, q1) = tok.plan_slots()
    out = {}
    for b in range(0, len(suffixes), batch):
        chunk = list(suffixes[b:b + batch])
        prefixes = [plan_prefix(tok, label, frame, s) for s in chunk]
        pp = last_probs(model, prefixes)
        pq = {v: last_probs(model, [p + [v] for p in prefixes]) for v in (p0, p1)}
        for i, s in enumerate(chunk):
            mp = float(pp[i, p0] + pp[i, p1])
            pP1 = float(pp[i, p1]) / mp
            qc = {v: float(pq[v][i, q1]) / float(pq[v][i, q0] + pq[v][i, q1]) for v in (p0, p1)}
            out[s] = {"p": pP1, "q": (1 - pP1) * qc[p0] + pP1 * qc[p1], "legal_mass_P": mp,
                      "legal_mass_Q": {str(v): float(pq[v][i, q0] + pq[v][i, q1]) for v in (p0, p1)},
                      "q_given_p0": qc[p0], "q_given_p1": qc[p1]}
    return out


def part_data(args) -> None:
    from uie.kaggle_train import STREAMS, load_streams
    _, tok, cfg = load_checkpoint(TRAIN / "e10_plan" / "best.pt", device=torch.device("cpu"))
    urls, labels, langs, corpora, label_counts = load_streams(BUNDLE, STREAMS)
    suffix_index: Dict[str, int] = {}
    n = len(urls)
    cols = {k: np.zeros(n, dtype=d) for k, d in
            (("cls", np.int8), ("frame", np.int8), ("suf", np.int32), ("path", np.int8),
             ("query", np.int8), ("m", np.int16), ("usr", np.int8), ("fits", np.int8))}
    t0 = time.time()
    for i, (u, lab, lang, corp) in enumerate(zip(urls, labels, langs, corpora)):
        frame = CORPUS_FRAME[corp]
        ids = tok.encode(u, lab, lang, frame=frame)
        fits = len(ids) <= cfg.max_len
        s, p, q, m, usr = row_record(tok, ids)
        cols["cls"][i] = LABELS.index(lab)
        cols["frame"][i] = FRAMES.index(frame)
        cols["suf"][i] = suffix_index.setdefault(s, len(suffix_index))
        cols["path"][i], cols["query"][i], cols["m"][i] = p, q, m
        cols["usr"][i], cols["fits"][i] = usr, int(fits)
        if i % 500_000 == 0:
            print(f"  {i:>9,} rows  {time.time() - t0:.0f}s", flush=True)
    # V4: scored tokens m = len - 3 against the field mask's own sum, on 1,000
    # random rows of each stream.
    rng = random.Random(SEED)
    by_stream: Dict[Tuple[str, str], List[int]] = {}
    for i, key in enumerate(zip(labels, corpora)):
        by_stream.setdefault(key, []).append(i)
    v4 = 0
    for key in sorted(by_stream):
        for i in rng.sample(by_stream[key], min(1000, len(by_stream[key]))):
            ids = tok.encode(urls[i], labels[i], langs[i], frame=CORPUS_FRAME[corpora[i]])
            v4 = max(v4, abs(int(cols["m"][i]) - int(sum(field_mask(ids, tok, corpora[i])))))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "rows.npz", **cols)
    meta = {"suffixes": sorted(suffix_index, key=suffix_index.get), "label_counts": label_counts,
            "max_len": cfg.max_len, "rows": n,
            "fits": {lab: int(((cols["cls"] == c) & (cols["fits"] == 1)).sum())
                     for c, lab in enumerate(LABELS)},
            "dropped": int((cols["fits"] == 0).sum()), "V4_max_abs_diff": v4,
            "V4_rows": {f"{k[0]}/{k[1]}": min(1000, len(v)) for k, v in sorted(by_stream.items())},
            "seconds": round(time.time() - t0)}
    (OUT / "rows_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("rows that fit", meta["fits"], "dropped", meta["dropped"], flush=True)


# --------------------------------------------------------------------------- #
# M1 -- the plan's distribution, per class, frame and suffix
# --------------------------------------------------------------------------- #

def part_model(args) -> None:
    from uie.suffixes import SuffixTable
    meta = json.loads((OUT / "rows_meta.json").read_text(encoding="utf-8"))
    rows = np.load(OUT / "rows.npz")
    shipped = SuffixTable.load(BUNDLE / "suffix_table.json").counts
    ks = json.loads((ROOT / "data" / "e10" / "score_plan.json").read_text(encoding="utf-8"))["plan"]
    names = meta["suffixes"]
    fits = rows["fits"] == 1
    out = {"registration": "research/native/PREREG_PC1.md", "checkpoints": {}}
    for ck in ("best", "final"):
        model, tok, _ = load_checkpoint(TRAIN / "e10_plan" / f"{ck}.pt", device=torch.device("cpu"))
        model.eval()
        rec = {}
        for c, label in enumerate(LABELS):
            weights = shipped[label]
            wsum = sum(weights.values())
            if ck == "best":
                # V1: K10a exactly as e10_score.part_plan computed it; V6 (as
                # amended, PREREG_PC1 amendment 1): the frame-generalised function
                # on the same suffixes and batches, wherever the probe `x.{s}/`
                # parses to s itself; the mismatches listed with both values.
                k10a = plan_by_suffix(model, tok, label, sorted(weights))
                mine = plan_by_frame(model, tok, label, sorted(weights), "domain")
                keys = (("P", "p", "P"), ("Q", "q", "Q"),
                        ("legal_mass_P", "legal_mass_P", "legal_mass_P"))
                rec.setdefault("V1", {})[label] = {
                    key: {"here": sum(cnt / wsum * k10a[s][f] for s, cnt in weights.items()),
                          "score_plan": ks[label][sk]} for key, f, sk in keys}
                probe_off = sorted(s for s in weights if parse(f"x.{s}/").public_suffix != s)
                fields = ("p", "q", "legal_mass_P", "q_given_p0", "q_given_p1")
                rec.setdefault("V6", {})[label] = max(
                    abs(k10a[s][f] - mine[s][f]) for s in weights if s not in probe_off
                    for f in fields)
                rec.setdefault("V6_probe_mismatch", {})[label] = {
                    s: {"probe_parses_as": parse(f"x.{s}/").public_suffix,
                        "shipped_share": weights[s] / wsum,
                        "probe": {f: k10a[s][f] for f in fields},
                        "training_prefix": {f: mine[s][f] for f in fields}} for s in probe_off}
                rec.setdefault("K10a_training_prefix", {})[label] = {
                    key: sum(cnt / wsum * mine[s][f] for s, cnt in weights.items())
                    for key, f, _ in keys}
            for f, frame in enumerate(FRAMES):
                present = np.unique(rows["suf"][fits & (rows["cls"] == c) & (rows["frame"] == f)])
                sufs = {names[i] for i in present}
                if frame == "domain":
                    sufs |= set(weights)
                rec.setdefault("plan", {}).setdefault(label, {})[frame] = plan_by_frame(
                    model, tok, label, sorted(sufs), frame)
                print(f"  {ck} {label:<9} {frame:<6} {len(sufs):>5} suffixes", flush=True)
        out["checkpoints"][ck] = rec
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps(out, indent=0), encoding="utf-8")
    for label, v in out["checkpoints"]["best"]["V1"].items():
        print(f"  V1 {label}: " + "  ".join(f"{k} {x['here']:.6f} / {x['score_plan']:.6f}"
                                            for k, x in v.items()), flush=True)
    print("  V6 max |diff|:", out["checkpoints"]["best"]["V6"], flush=True)


# --------------------------------------------------------------------------- #
# M2 -- clip factors: batches built as `uie.train.batches` builds them
# --------------------------------------------------------------------------- #

def window_batches(pools: Dict[str, List[int]], lengths: Sequence[int],
                   label_counts: Dict[str, int], batch_size: int, bucket_window: int,
                   rng: random.Random, class_alpha: float = 1.0) -> Iterator[List[List[int]]]:
    """`uie.train.batches` over row INDICES: the same draws, the same length sort,
    the same chunking, and the same RNG calls (the chunk shuffle included), so it
    consumes the stream exactly as training does. Yields each window's chunks in
    SORTED order (the order PREREG_PC1's sampling rule counts in)."""
    pools = {cls: list(ix) for cls, ix in pools.items()}
    for ix in pools.values():
        rng.shuffle(ix)
    cursors = {cls: 0 for cls in pools}

    def draw(cls: str) -> int:
        ix = pools[cls]
        if cursors[cls] >= len(ix):
            rng.shuffle(ix)
            cursors[cls] = 0
        cursors[cls] += 1
        return ix[cursors[cls] - 1]

    classes = sorted(pools)
    weights = [max(label_counts.get(c, 1), 1) ** class_alpha for c in classes]
    while True:
        window: List[int] = []
        while len(window) < batch_size * bucket_window:
            window.append(draw(rng.choices(classes, weights=weights, k=1)[0]))
        window.sort(key=lambda i: lengths[i])
        chunks = [window[i:i + batch_size] for i in range(0, len(window), batch_size)]
        ordered = list(chunks)
        rng.shuffle(chunks)
        yield ordered


MICRO = 32             # rows per micro-batch (memory), as e9_evidence_clipping.py


def grad_norm(model, batch: List[Tuple[List[int], List[float]]], pad_id: int, seed: int
              ) -> Tuple[float, float, float]:
    """The norm `clip_grad_norm_` sees for one batch: the gradient of the
    batch-mean loss over every parameter, training mode (dropout on), fixed seed.

    Accumulated over micro-batches exactly as `e9_evidence_clipping.py` does: the
    batch loss is sum / T_b, so a micro-batch contributes loss_mb * T_mb / T_b.
    Returns (norm, batch loss, T_b)."""
    torch.manual_seed(seed)
    parts = [_pad(batch[j:j + MICRO], pad_id) for j in range(0, len(batch), MICRO)]
    mass = [float((w * (ids[:, 1:] != pad_id)).sum()) for ids, w in parts]
    t = sum(mass)
    model.zero_grad(set_to_none=True)
    loss = 0.0
    for (ids, w), t_mb in zip(parts, mass):
        part = model.loss(ids, pad_id, w) * (t_mb / t)
        part.backward()
        loss += float(part.detach())
    sq = sum(float(p.grad.detach().double().pow(2).sum())
             for p in model.parameters() if p.grad is not None)
    return math.sqrt(sq), loss, t


def part_clip(args) -> None:
    from uie.kaggle_train import STREAMS, load_streams
    torch.set_num_threads(args.threads)
    model, tok, cfg = load_checkpoint(TRAIN / "e10_plan" / "best.pt", device=torch.device("cpu"))
    tcfg = torch.load(TRAIN / "e10_plan" / "best.pt", map_location="cpu",
                      weights_only=False)["train_config"]
    rows = np.load(OUT / "rows.npz")
    meta = json.loads((OUT / "rows_meta.json").read_text(encoding="utf-8"))
    urls, labels, langs, corpora, _ = load_streams(BUNDLE, STREAMS)
    fits = rows["fits"] == 1
    lengths = rows["m"].astype(np.int64) + UNSCORED
    pools = {lab: np.flatnonzero(fits & (rows["cls"] == c)).tolist() for c, lab in enumerate(LABELS)}
    rng = random.Random(SEED)
    stream = window_batches(pools, lengths, meta["label_counts"], tcfg["batch_size"],
                            tcfg["bucket_window"], rng, tcfg["class_alpha"])
    model.train()
    records = []
    t0 = time.time()
    for w in range(WINDOWS):
        chunks = next(stream)
        for j, chunk in enumerate(chunks):
            mean_m = float(rows["m"][chunk].mean())
            if mean_m >= SHORT and j % LONG_EVERY:
                continue
            batch = []
            for i in chunk:
                ids = tok.encode(urls[i], labels[i], langs[i], frame=CORPUS_FRAME[corpora[i]])
                batch.append((ids, field_mask(ids, tok, corpora[i])))
            norm, loss, mass = grad_norm(model, batch, tok.pad_id, seed=SEED + 1000 * w + j)
            records.append({"window": w, "j": j, "rows": len(chunk), "scored_per_row": mass / len(chunk),
                            "mean_m": mean_m, "norm": norm, "loss": loss,
                            "omega": min(1.0, 1.0 / norm) if math.isfinite(norm) else None})
        print(f"  window {w + 1}/{WINDOWS}: {len(records)} batches  {time.time() - t0:.0f}s",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clip.json").write_text(json.dumps({
        "registration": "research/native/PREREG_PC1.md", "seed": SEED, "windows": WINDOWS,
        "short": SHORT, "long_every": LONG_EVERY, "checkpoint": "e10_plan/best.pt",
        "threads": args.threads, "seconds": round(time.time() - t0), "batches": records},
        indent=0), encoding="utf-8")
    short = [r for r in records if r["scored_per_row"] < SHORT]
    print(f"  short batches {len(short)}; clipped {sum(r['norm'] > 1 for r in short)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("data", "model", "clip"), required=True)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    {"data": part_data, "model": part_model, "clip": part_clip}[args.part](args)


if __name__ == "__main__":
    main()
