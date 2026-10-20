"""E14 -- K14-2 (held-out query fit) and K14-3 (the band profile), with N2a's
instruments (research/e14/PREREG_E14.md section 3.1). Laptop, CPU.

    python research/e14/e14_fit.py --part m3       # K14-2: needs only the two checkpoints
    python research/e14/e14_fit.py --part n2a      # N2a's M1-M4 on each model's free files:
                                                   #   K14-3 (M2 on e14_q), the rest reported

K14-2. N2a's M3 (research/native/n2a_reference.py `teacher_forced`): teacher-forced
negative log-likelihood per query character -- the plan the tokenizer scores, the
domain frame, every validation query row of the class -- computed PER ROW for
e12_final and e14_q on the same rows. "Falls": the pooled difference (e14_q -
e12_final) is below 0 and its paired row-bootstrap 95% interval (1,000 draws, seed
20260806; N2a's sorted [25] and [974]) excludes 0. Before any reading, the per-row
sums are checked against N2a's own pooled M3 on the reference model (`--part m3`
stops if they differ). M3 on N2a's training sample is reported (the train/validation
gap, A2's memorisation signature).

K14-3. N2a's M2 (`band_profile`) on e14_q's free-mode files, all 5 generations:
OVERSHOOT if the observed rare-band share's 95% interval lies wholly above the
faithful-sampler null's share.

Writes data/e14/k14_2.json and data/e14/n2a_<model>_<label>.json.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "native", ROOT / "research" / "dc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e14_common as E14  # noqa: E402

E14D = ROOT / "data" / "e14"
CHECKPOINTS = {"e12_final": ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt",
               "e14_q": E14D / "train" / "e14_q" / "final.pt"}
FREE_FILES = {"e12_final": ROOT / "data" / "e12" / "gen", "e14_q": E14D / "generate" / "gen"}
SELF_CHECK_ROWS = 480


def per_row_m3(model, tok, urls, label):
    """[(nll_sum, chars)] per kept row: N2a's M3 positions (every query character,
    s..e-1 of each QRY span; the target at i predicted from i-1), batched exactly as
    `teacher_forced` batches, so the floats are its floats."""
    import torch
    import torch.nn.functional as F
    from n2a_reference import encode
    from research.qry.q7_hazard import BATCH, field_spans
    out = []
    for start in range(0, len(urls), BATCH):
        enc = [encode(tok, u, label) for u in urls[start:start + BATCH]]
        enc = [e for e in enc if len(e) <= model.cfg.max_len]
        if not enc:
            continue
        width = max(len(e) for e in enc)
        ids = torch.tensor([e + [tok.pad_id] * (width - len(e)) for e in enc], dtype=torch.long)
        with torch.no_grad():
            logp = F.log_softmax(model(ids).float(), dim=-1)
        for b, seq in enumerate(enc):
            s_nll, chars = 0.0, 0
            for s, e in field_spans(seq, tok, "QRY"):
                for i in range(s, e):
                    s_nll -= float(logp[b, i - 1, seq[i]])
                    chars += 1
            out.append((s_nll, chars))
    return out


def paired(ref, arm, seed=E14.BOOT_SEED, draws=E14.BOOT_DRAWS):
    """The pooled per-character difference (arm - ref) on the same rows, and its
    paired row bootstrap (N2a's convention: sorted draws [25] and [974])."""
    import numpy as np
    d = np.array([a[0] - r[0] for r, a in zip(ref, arm)], dtype=np.float64)
    c = np.array([r[1] for r in ref], dtype=np.float64)
    assert all(r[1] == a[1] for r, a in zip(ref, arm)), "the two models scored different characters"
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(draws):
        k = rng.integers(0, len(d), len(d))
        boots.append(float(d[k].sum() / c[k].sum()))
    boots.sort()
    point = float(d.sum() / c.sum())
    lo, hi = boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1]
    return {"rows": len(d), "chars": int(c.sum()), "diff": point, "ci": [lo, hi],
            "falls": point < 0 and hi < 0}


def part_m3(args) -> None:
    import torch
    from n2a_reference import (SAMPLE_SEED, STREAMS, TRAIN_ROWS, rows, teacher_forced)
    from uie.evaluation import FIELDS
    from uie.train import load_checkpoint
    models = {}
    for name, path in CHECKPOINTS.items():
        m, t, _ = load_checkpoint(path, device=torch.device("cpu"))
        m.eval()
        models[name] = (m, t)
    if models["e12_final"][1].state_dict() != models["e14_q"][1].state_dict():
        raise SystemExit("STOP: e14_q's tokenizer is not e12_final's (V5); nothing is read")
    out = {"registration": E14.REGISTRATION + " section 3.1 (K14-2)",
           "instrument": "research/native/n2a_reference.py teacher_forced (M3), per row",
           "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()}}
    for label in E14.LABELS:
        train_stem, val_stem, _ = STREAMS[label]
        val_q = [u for u in rows(val_stem) if FIELDS["QRY"](u)]
        train_q = [u for u in rows(train_stem) if FIELDS["QRY"](u)]
        train_q = random.Random(SAMPLE_SEED).sample(train_q, min(TRAIN_ROWS, len(train_q)))
        # the self-check: these per-row sums ARE N2a's M3 (pooled over its bands)
        m, t = models["e12_final"]
        probe = val_q[:SELF_CHECK_ROWS]
        mine = per_row_m3(m, t, probe, label)
        _, _, theirs = teacher_forced(m, t, probe, label, {})
        a = sum(x[0] for x in mine) / sum(x[1] for x in mine)
        b = sum(v[0] for v in theirs.values()) / sum(v[1] for v in theirs.values())
        if abs(a - b) > 1e-9 * max(1.0, abs(b)) or sum(x[1] for x in mine) != sum(
                v[1] for v in theirs.values()):
            raise SystemExit(f"STOP: per-row M3 {a!r} is not N2a's M3 {b!r} ({label}); nothing read")
        node = {"self_check": {"rows": len(probe), "per_row": a, "n2a": b}}
        for split, urls in (("validation", val_q), ("train", train_q)):
            per = {name: per_row_m3(m_, t_, urls, label) for name, (m_, t_) in models.items()}
            res = paired(per["e12_final"], per["e14_q"])
            pooled = {name: sum(x[0] for x in v) / sum(x[1] for x in v) for name, v in per.items()}
            node[split] = {**res, "nll_per_char": pooled}
            print(f"  {label:<9} {split:<10} e12_final {pooled['e12_final']:.4f}  e14_q "
                  f"{pooled['e14_q']:.4f}  diff {res['diff']:+.4f} [{res['ci'][0]:+.4f}, "
                  f"{res['ci'][1]:+.4f}]  rows {res['rows']:,}", flush=True)
        node["gap"] = {name: node["validation"]["nll_per_char"][name] - node["train"]["nll_per_char"][name]
                       for name in CHECKPOINTS}
        node["pass"] = node["validation"]["falls"]
        out[label] = node
    out["pass"] = all(out[label]["pass"] for label in E14.LABELS)
    E14D.mkdir(parents=True, exist_ok=True)
    (E14D / "k14_2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("K14-2:", "PASS" if out["pass"] else "FAIL", "->", E14D / "k14_2.json")


def part_n2a(args) -> None:
    """N2a's own script, unchanged, once per model and class, on that model's free
    files (e12_final's from E12; e14_q's from E14_generate)."""
    for name in CHECKPOINTS:
        for label in E14.LABELS:
            files = sorted(FREE_FILES[name].glob(f"free_{name}_{label}_g*.json"))
            if len(files) != E14.G:
                raise SystemExit(f"STOP: {name} {label}: {len(files)} free files, not {E14.G}")
            dest = E14D / f"n2a_{name}_{label}.json"
            cmd = [sys.executable, str(ROOT / "research" / "native" / "n2a_reference.py"),
                   "--label", label, "--checkpoint", str(CHECKPOINTS[name]),
                   "--queries", *map(str, files), "--out", str(dest)]
            print(" ".join(cmd[1:3]), name, label, flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
    out = {"registration": E14.REGISTRATION + " section 3.1 (K14-3)"}
    for label in E14.LABELS:
        m2 = json.loads((E14D / f"n2a_e14_q_{label}.json").read_text(encoding="utf-8"))["M2"]
        over = m2["observed_ci"][0] > m2["null_rare"]
        out[label] = {"observed_rare": m2["observed_rare"], "observed_ci": m2["observed_ci"],
                      "null_rare": m2["null_rare"], "overshoot": over}
    out["pass"] = not any(out[label]["overshoot"] for label in E14.LABELS)
    (E14D / "k14_3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("K14-3:", "PASS" if out["pass"] else "FAIL", "->", E14D / "k14_3.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("m3", "n2a"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"m3": part_m3, "n2a": part_n2a}[args.part](args)


if __name__ == "__main__":
    main()
