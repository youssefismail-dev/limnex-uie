"""QG1 -- the per-URL measurements (research/qgap/QG_PLAN.md section 5): the query
hazard and y on each route, teacher-forced in free mode's exact conditioning, for
one model over real training rows, real validation rows and its own raw free URLs.
Diagnostic; laptop; no training, no generation.

    python research/qgap/qg1_measure.py --model e9_ln                  # the baseline
    python research/qgap/qg1_measure.py --model e9_base                # the old objective
    python research/qgap/qg1_measure.py --model e7 --sets train,val    # T / D only
    python research/qgap/qg1_measure.py --model e10_base               # E10's reported reading

Writes data/qgap/qg1_<model>_<set>_<label>.json.gz. VQ1 (fingerprints and file
hashes) is checked before anything is measured.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research" / "native"))

import torch  # noqa: E402

import qg_common as Q  # noqa: E402
from uie.baseline import E9_LN as BASELINE  # built on e9_ln  # noqa: E402
from uie.evaluation import FIELDS  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402

E7_SHA256 = "916d03857d7d3db4101d1b106293a83905f4b317676ee7e144b018170cb36065"
CKPT = {"e9_ln": ROOT / "data" / "e9" / "train" / "e9_ln" / "best.pt",
        "e9_base": ROOT / "data" / "e9" / "train" / "e9_base" / "best.pt",
        "e7": ROOT / "data" / "best_e7.pt",
        "e10_base": ROOT / "data" / "e10" / "train" / "e10_base" / "best.pt"}
GEN = ROOT / "data" / "e9" / "gen"
#: E10's raw free files: `e10_base` is the baseline recipe's other run, whose QG1
#: decomposition E10 reports (PREREG_E10 section 2, "reported, not criteria").
GEN_E10 = ROOT / "data" / "e10" / "gen"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def vq1_model(name: str):
    """Load and verify: e9_ln by the baseline's weights and file; e9_base by E9's
    provenance; E7 by its registered file hash. Refuses anything else."""
    path = CKPT[name]
    got = sha256(path)
    if name == "e9_ln":
        want = BASELINE.file_sha256
    elif name == "e9_base":
        prov = json.loads((ROOT / "data" / "e9" / "train" / "PROVENANCE.json")
                          .read_text(encoding="utf-8"))
        want = prov["arms"]["e9_base"]["best_sha256"]
    elif name == "e10_base":
        prov = json.loads((ROOT / "data" / "e10" / "train" / "PROVENANCE.json")
                          .read_text(encoding="utf-8"))
        want = prov["arms"]["e10_base"]["best_sha256"]
    else:
        want = E7_SHA256
    if got != want:
        raise SystemExit(f"VQ1: {path} is {got[:12]}..., registered {want[:12]}...")
    model, tok, cfg = load_checkpoint(path, device=torch.device("cpu"))
    model.eval()
    if name == "e9_ln":
        BASELINE.check_model(model)
    return model, tok, cfg


def gen_files(name: str, label: str):
    """The model's raw free files for this class -- E9's, or E10's for `e10_base` --
    each checked against its generation provenance."""
    d, pattern = ((GEN_E10, "free_e10_base_{label}_g{g}.json") if name == "e10_base"
                  else (GEN, "free_{name}_raw_{label}_g{g}.json"))
    prov = json.loads((d / "PROVENANCE.json").read_text(encoding="utf-8"))["files"]
    out = []
    for g in range(5):
        f = pattern.format(name=name, label=label, g=g)
        if sha256(d / f) != prov[f]["sha256"]:
            raise SystemExit(f"VQ1: {f} differs from its generation provenance")
        out.append(json.loads((d / f).read_text(encoding="utf-8")))
    return out


def sample_rows(stem: str, n: int, tok, label: str, max_len: int, limit=None):
    """A uniform sample of `n` rows that fit max_len: the pool in a seeded random
    order, kept while they fit. Returns (urls, their sequences, pool size, dropped)."""
    pool = list(Q.rows(stem))
    order = random.Random(Q.SEED).sample(range(len(pool)), len(pool))
    want = min(n, limit or n)
    urls, seqs, dropped = [], [], 0
    for i in order:
        seq = Q.encode_free(tok, pool[i], label)
        if seq is None or len(seq) > max_len:
            dropped += 1
            continue
        urls.append(pool[i])
        seqs.append(seq)
        if len(urls) == want:
            break
    return urls, seqs, len(pool), dropped


def suffix_of(url: str) -> str:
    return FIELDS["SUF"](url) or ""


def measure(model, tok, cfg, name: str, sets, limit=None) -> None:
    Q.DATA.mkdir(parents=True, exist_ok=True)
    for label in Q.LABELS:
        for which in sets:
            t0 = time.time()
            payload = {"registration": "research/qgap/QG_PLAN.md (QG1)", "model": name,
                       "set": which, "label": label, "limit": limit}
            if which in ("train", "val"):
                stem = Q.STREAMS[label][0 if which == "train" else 1]
                n = Q.N_TRAIN if which == "train" else Q.N_VAL
                urls, seqs, pool, dropped = sample_rows(stem, n, tok, label, cfg.max_len, limit)
                payload.update({"stem": stem, "pool": pool, "skipped_too_long": dropped})
            else:
                files = gen_files(name, label)
                every = [u for f in files for u in f]
                payload["all"] = {                           # string-level, all 40,000
                    "n": len(every),
                    "by_generation": [[len(f), sum(bool(FIELDS["QRY"](u)) for u in f)]
                                      for f in files],
                    "suffix": [suffix_of(u) for u in every],
                    "query": [int(bool(FIELDS["QRY"](u))) for u in every],
                    "empty_query": [int("?" in u and not FIELDS["QRY"](u)) for u in every]}
                idx = sorted(random.Random(Q.SEED).sample(range(len(every)),
                                                          min(Q.N_GEN, limit or Q.N_GEN)))
                urls = [every[i] for i in idx]
                seqs = [Q.encode_free(tok, u, label) for u in urls]
                payload["sample_index"] = idx
            recs = Q.hazards(model, tok, seqs, max_len=cfg.max_len)
            payload.update({
                "urls": urls, "suffix": [suffix_of(u) for u in urls],
                "query_gate": [int(bool(FIELDS["QRY"](u))) for u in urls],
                "unreadable": [int(r is None) for r in recs],
                "too_long": [int(s is not None and len(s) > cfg.max_len) for s in seqs],
                "records": recs})
            dest = Q.DATA / f"qg1_{name}_{which}_{label}.json.gz"
            with gzip.open(dest, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh)
            ok = [r for r in recs if r]
            ev = sum(r["y_host"] + r["y_path"] for r in ok)
            hz = sum(r["h_host"] + r["h_path"] for r in ok)
            print(f"  {name:<8} {which:<5} {label:<9} rows {len(ok):>6,} (unreadable "
                  f"{len(recs) - len(ok)})  queries {ev:>5,}  hazard {hz:9.1f}  "
                  f"ratio {hz / ev if ev else float('nan'):.3f}  {time.time() - t0:6.0f}s",
                  flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(CKPT), required=True)
    ap.add_argument("--sets", default="train,val,gen")
    ap.add_argument("--limit", type=int, help="dry runs only: rows per set")
    args = ap.parse_args()
    sets = args.sets.split(",")
    if args.model == "e7" and "gen" in sets:
        raise SystemExit("E7 has no raw free files (QG_PLAN 5.1): T / D only")
    try:                                    # the laptop stays usable (UIE practice)
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    model, tok, cfg = vq1_model(args.model)
    print(f"VQ1: {args.model} verified", flush=True)
    measure(model, tok, cfg, args.model, sets, args.limit)


if __name__ == "__main__":
    main()
