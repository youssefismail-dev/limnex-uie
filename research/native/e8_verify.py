"""E8 (N1) -- validity checks before anything is scored (PREREG_E8 section 5, V1-V4).

    python research/native/e8_verify.py --e8 data/e8 --bundle data/bundle2

Expects the Kaggle outputs unpacked as data/e8/train/ (the E8_train output: the
four e8_* folders and PROVENANCE.json) and data/e8/gen/ (the E8_generate output:
250 files and PROVENANCE.json). Writes data/e8/VERIFY.json and, from V4,
data/e8/e7_free_fields.json (E7's canonical free record, reused by the scorer).

V1  hashes and completeness: checkpoints (local == training record == generation
    record), E7, tables, the code the GPU ran (unchanged under uie/ and
    e8_common.py since), all 250 files present with the registered n and seeds.
V2  byte identity on this CPU: leading suffix groups of one file per checkpoint
    and mode, at least 250 URLs each, regenerated with e8_common's exact calls.
V3  flags off reproduce E7: a code property, proven by the suite at 36e1b26
    (the E7/HC1/DC1 fixtures); here, that no uie/ file changed since.
V4  the free scorer reproduces HC1 C2's recorded E7 canonical cells from E7's
    own files, exactly.
V-base (e8_base against E7) needs scores; it is read by e8_compare.py.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cr1"))

import e8_common as K  # noqa: E402

IDENTITY_MIN = 250
TABLES = {"suffix_table.json": ("data/bundle2/suffix_table.json",
                                "aaa6fb853ea538d41ba4c10661b89faad2a687b869208772c096c7e507ee866d"),
          "state_suffix_tables.json": ("data/state_suffix_tables.json",
                                       "0c6f2760c033c532d271b360a14f2976139d5957e4aaf64977c8746504213cc9")}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def placement(table, label: str, n: int, seed: int):
    """The suffix groups and the final shuffle generate_with_suffixes applies."""
    rng = random.Random(seed)
    drawn = table.sample(label, n, rng)
    perm = list(range(n))
    rng.shuffle(perm)
    where = {orig: pos for pos, orig in enumerate(perm)}
    groups = sorted(collections.Counter(zip(["domain"] * n, drawn)).items())
    return groups, where


def identity(model, tok, table, urls, label, seed, extra, budget):
    """Regenerate leading suffix groups on this CPU; every URL of a group sits
    in the group's block of the (shuffled) file, since nothing is resampled."""
    import torch
    from uie.model import generate
    groups, where = placement(table, label, len(urls), seed)
    start = same = compared = 0
    for i, ((frame, suffix), count) in enumerate(groups):
        if compared >= budget:
            break
        got = generate(model, tok, label=label, n=count, suffix=suffix, seed=seed + i,
                       frame=frame, batch_size=32, device=torch.device("cpu"), **extra)
        for rank, u in enumerate(got):
            same += urls[where[start + rank]] == u
            compared += 1
        start += count
    return same, compared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e8", type=Path, default=ROOT / "data" / "e8")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--e7", type=Path, default=ROOT / "data" / "best_e7.pt")
    ap.add_argument("--skip-v2", action="store_true", help="dry runs only")
    ap.add_argument("--skip-v4", action="store_true", help="dry runs only")
    args = ap.parse_args()

    import torch
    from uie.controlled import StateSuffixTables
    from uie.free import ADOPTED
    from uie.suffixes import SuffixTable
    from uie.train import load_checkpoint

    gens, train = args.e8 / "gen", args.e8 / "train"
    gprov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    tprov = json.loads((train / "PROVENANCE.json").read_text(encoding="utf-8"))
    by_file = {r["file"]: r for r in gprov["files"]}
    failures = []

    # V1 -- checkpoints, tables, code
    ckpts = {"e7": args.e7, **{a: train / a / "best.pt" for a in K.ARMS}}
    for who, path in ckpts.items():
        local = sha256(path) if path.is_file() else None
        if local is None:
            failures.append(f"V1: {path} missing")
            continue
        if gprov["checkpoints"].get(who) != local:
            failures.append(f"V1: {who} differs from the checkpoint generation used")
        if who == "e7" and local != K.E7_SHA256:
            failures.append("V1: data/best_e7.pt is not E7")
        if who != "e7" and tprov["arms"].get(who, {}).get("best_sha256") != local:
            failures.append(f"V1: {who} differs from the checkpoint training recorded")
    for name, (rel, want) in TABLES.items():
        if sha256(ROOT / rel) != want:
            failures.append(f"V1: {rel} is not the registered {name}")
    commit = gprov.get("code_commit", "")
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT).returncode:
        failures.append(f"V1: code commit {commit!r} is not in this history")
    changed = git("diff", "--name-only", commit, "HEAD", "--", "uie",
                  "research/native/e8_common.py")
    if changed:                                                            # V3 as well
        failures.append(f"V1/V3: code changed since the GPU ran it: {changed.split()}")
    if tprov.get("code_commit") != commit:
        failures.append("V1: training and generation ran different code commits")

    # V1 -- the 250 registered files
    jobs = K.jobs()
    names = {j["file"] for j in jobs}
    extra = sorted(p.name for p in gens.glob("*.json") if p.name != "PROVENANCE.json"
                   and p.name not in names)
    if extra:
        failures.append(f"V1: unregistered files {extra}")
    files = {}
    for j in jobs:
        p, rec = gens / j["file"], by_file.get(j["file"]) or {}
        if not p.is_file():
            failures.append(f"V1: {j['file']} missing ({rec.get('error')})")
            continue
        urls = json.loads(p.read_text(encoding="utf-8"))
        want_n = K.N_FREE if j["mode"] == "free" else K.n_ctrl(j["label"], j["state"])
        want_seed = (K.seed_free(j["g"]) if j["mode"] == "free"
                     else K.seed_ctrl(j["label"], j["state"], j["g"]))
        if len(urls) != want_n:
            failures.append(f"V1: {j['file']} has {len(urls)} urls, registered {want_n}")
        if rec.get("sha256") != sha256(p) or rec.get("seed") != want_seed:
            failures.append(f"V1: {j['file']} sha256 or seed differs from PROVENANCE")
        files[j["file"]] = (j, urls)
    print(f"  V1: {len(files)} of {len(jobs)} registered files present", flush=True)

    # V2 -- byte identity on this CPU
    ident = {}
    if not args.skip_v2:
        shipped = SuffixTable.load(args.bundle / "suffix_table.json")
        tables = StateSuffixTables.load(ROOT / "data" / "state_suffix_tables.json")
        checks = [(who, "free", [f"free_{who}_benign_g0.json", f"free_{who}_phishing_g0.json"])
                  for who in K.ARMS]
        checks += [(who, "control", [f"ctrl_{who}_benign_P1Q1_g0.json",
                                     f"ctrl_{who}_phishing_P0Q1_g0.json"])
                   for who in tuple(K.ARMS) + ("e7",)]
        checks.append(("e7dc1", "free", ["free_e7dc1_benign_g0.json",
                                         "free_e7dc1_phishing_g0.json"]))
        loaded = {}
        for who, mode, targets in checks:
            key = "e7" if who == "e7dc1" else who
            if key not in loaded:
                loaded.clear()
                m, t, _ = load_checkpoint(ckpts[key], device=torch.device("cpu"))
                loaded[key] = (m.eval(), t)
            model, tok = loaded[key]
            same = compared = 0
            for f in targets:
                if f not in files:
                    continue
                j, urls = files[f]
                label, st = j["label"], j["state"]
                if who == "e7dc1":
                    extra = ADOPTED.kwargs(tok, label)
                    table, seed = shipped, K.seed_free(j["g"])
                elif mode == "free":
                    extra, table, seed = K.resting(tok, label), shipped, K.seed_free(j["g"])
                else:
                    extra = {**K.resting(tok, label), "fields": dict(K.STATES[st])}
                    table = tables.for_state(label, K.STATES[st])
                    seed = K.seed_ctrl(label, st, j["g"])
                s, c = identity(model, tok, table, urls, label, seed, extra,
                                -(-IDENTITY_MIN // len(targets)))
                same, compared = same + s, compared + c
                print(f"  V2 identity {f:<36} {s}/{c}", flush=True)
            ident[f"{who}/{mode}"] = {"compared": compared, "identical": same}
            if compared < IDENTITY_MIN or same != compared:
                failures.append(f"V2: {who} {mode} identity {same}/{compared}")

    # V4 -- the free scorer reproduces HC1 C2's recorded E7 cells, exactly
    v4 = {}
    if not args.skip_v4:
        from cr1_score import pools, score
        e7_free = {}
        for label in K.LABELS:
            published = json.loads((ROOT / "data" / f"hc1_c2_e7_{label}" / "eval_report.json")
                                   .read_text(encoding="utf-8"))[label]["fields"]
            src = [ROOT / "data" / "gen_cache_e7_gpu" / p for p in sorted(
                p.name for p in (ROOT / "data" / "gen_cache_e7_gpu").glob(f"*_{label}_n8000_*.json"))]
            seeds = [int(p.stem.rsplit("_s", 1)[1]) for p in src]
            if seeds != [K.seed_free(g) for g in range(K.G)]:
                failures.append(f"V4: E7's canonical {label} files are not at the registered seeds")
            fields, pooled = score(pools(args.bundle, label),
                                   [json.loads(p.read_text(encoding="utf-8")) for p in src], K.N_FREE)
            e7_free[label] = {"fields": fields, "pooled_fields": pooled}

            def x_of(row):
                x = (row or {}).get("acceptance", {}).get("x")
                return None if x is None or x != x else float(x)

            diffs = {f: (0.0 if x_of(fields.get(f)) is None and x_of(published[f]) is None else
                         float("inf") if x_of(fields.get(f)) is None or x_of(published[f]) is None
                         else abs(x_of(fields[f]) - x_of(published[f]))) for f in published}
            v4[label] = diffs
            worst = max(diffs.values())
            print(f"  V4 scorer {label:<9} max |x - published| {worst:.2e}", flush=True)
            if worst > 1e-9:
                failures.append(f"V4: {label} scorer differs from HC1 C2's record by {worst:.2e}")
        (args.e8 / "e7_free_fields.json").write_text(json.dumps(e7_free, indent=1, default=float),
                                                     encoding="utf-8")

    verdict = "PASS" if not failures else "FAIL"
    (args.e8 / "VERIFY.json").write_text(json.dumps(
        {"registration": "research/native/PREREG_E8.md", "identity": ident, "v4": v4,
         "failures": failures, "verdict": verdict}, indent=1), encoding="utf-8")
    print("failures:", failures or "none")
    print("VERIFY:", verdict)


if __name__ == "__main__":
    main()
