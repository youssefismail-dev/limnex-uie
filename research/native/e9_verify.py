"""E9 -- validity checks before anything is scored (PREREG_E9 section 5).

    python research/native/e9_verify.py                 # everything
    python research/native/e9_verify.py --train-only    # after E9_train, before generation:
                                                        # V-id and the training records, so
                                                        # R1 (teacher-forced) can be read early

Expects the Kaggle outputs unpacked as data/e9/train/ (E9_train: e9_base/, e9_ln/,
PROVENANCE.json) and data/e9/gen/ (E9_generate: 160 files and PROVENANCE.json).
Writes data/e9/VERIFY.json and, from V4, data/e9/e7_free_fields.json.

V1    hashes and completeness: each checkpoint (local == training record ==
      generation record), the tables, the code the GPU ran (no change since under
      uie/, e8_common.py, e9_common.py), all 160 files with their registered n and
      seeds.
V2    byte identity on this CPU: leading suffix groups of two files per condition
      (arm x decode x mode), at least 250 URLs per condition, regenerated with
      e9_common's exact calls.
V3    flags off reproduce the pre-E9 training byte for byte: tests/test_e9.py at
      3b29288; here, that no uie/ file changed since the code the GPU ran.
V4    the free scorer reproduces HC1 C2's recorded E7 canonical cells exactly.
V-id  the arms' initial weights are identical; tokenizer, model settings and
      training settings equal E7's and each other's except the registered flag
      (loss_norm, and C for e9_ln); both seeded; data counts equal E7's record.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cr1"))

import e9_common as K  # noqa: E402
from e8_common import E7_SHA256  # noqa: E402
from e8_verify import IDENTITY_MIN, TABLES, git, identity, sha256  # noqa: E402

E7_COUNTS = {"benign": 2_545_305, "phishing": 238_199}
E7_DROPPED = 10_313


def v_id(train: Path, e7: Path) -> list:
    import torch
    fails = []
    e7b = torch.load(e7, map_location="cpu", weights_only=False)
    starts = {}
    for arm, flags in K.ARMS.items():
        blob = torch.load(train / arm / "best.pt", map_location="cpu", weights_only=False)
        res = json.loads((train / arm / "train_result.json").read_text(encoding="utf-8"))
        starts[arm] = res.get("init_weights_sha256")
        if blob["tokenizer"] != e7b["tokenizer"]:
            fails.append(f"V-id: {arm} tokenizer is not E7's")
        if blob["model_config"] != e7b["model_config"]:
            fails.append(f"V-id: {arm} model settings differ from E7's")
        tc = blob["train_config"]
        for k, v in e7b["train_config"].items():
            if tc.get(k) != v:
                fails.append(f"V-id: {arm} train_config {k} = {tc.get(k)}, E7 {v}")
        if tc.get("loss_norm") != flags["loss_norm"]:
            fails.append(f"V-id: {arm} loss_norm {tc.get('loss_norm')}, registered "
                         f"{flags['loss_norm']}")
        want_c = flags["loss_constant"]
        got_c = res.get("loss_constant")
        if (want_c is None) != (got_c is None) or (
                want_c is not None and abs(got_c - want_c) > 1e-6 * want_c):
            fails.append(f"V-id: {arm} loss_constant {got_c}, registered {want_c}")
        if res.get("seed_init") is not True:
            fails.append(f"V-id: {arm} was not seeded")
        if res.get("counts") != E7_COUNTS or res.get("dropped_over_max_len") != E7_DROPPED:
            fails.append(f"V-id: {arm} data {res.get('counts')} / {res.get('dropped_over_max_len')}"
                         f" differ from E7's")
    if None in starts.values() or len(set(starts.values())) != 1:
        fails.append(f"V-id: the arms' initial weights differ: {starts}")
    return fails


def train_only(args) -> None:
    """The training side alone: checkpoints against the training record, the code
    the GPU ran, and V-id. Enough to read R1, which uses no generated file."""
    train = args.e9 / "train"
    tprov = json.loads((train / "PROVENANCE.json").read_text(encoding="utf-8"))
    failures = []
    for arm in K.ARMS:
        path = train / arm / "best.pt"
        if not path.is_file() or tprov["arms"].get(arm, {}).get("best_sha256") != sha256(path):
            failures.append(f"V1: {arm} best.pt differs from the checkpoint training recorded")
    if sha256(args.e7) != E7_SHA256:
        failures.append("V1: data/best_e7.pt is not E7")
    commit = tprov.get("code_commit", "")
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT).returncode:
        failures.append(f"V1: code commit {commit!r} is not in this history")
    changed = git("diff", "--name-only", commit, "HEAD", "--", "uie",
                  "research/native/e8_common.py", "research/native/e9_common.py")
    if changed:
        failures.append(f"V1/V3: code changed since the GPU ran it: {changed.split()}")
    failures += v_id(train, args.e7)
    verdict = "PASS" if not failures else "FAIL"
    (args.e9 / "VERIFY_TRAIN.json").write_text(json.dumps(
        {"registration": "research/native/PREREG_E9.md", "scope": "training only",
         "failures": failures, "verdict": verdict}, indent=1), encoding="utf-8")
    print("failures:", failures or "none")
    print("VERIFY (training side):", verdict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e9", type=Path, default=ROOT / "data" / "e9")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--e7", type=Path, default=ROOT / "data" / "best_e7.pt")
    ap.add_argument("--skip-v2", action="store_true", help="dry runs only")
    ap.add_argument("--skip-v4", action="store_true", help="dry runs only")
    ap.add_argument("--train-only", action="store_true",
                    help="V-id and the training checkpoints only (writes VERIFY_TRAIN.json)")
    args = ap.parse_args()
    if args.train_only:
        train_only(args)
        return

    import torch
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    from uie.train import load_checkpoint

    gens, train = args.e9 / "gen", args.e9 / "train"
    gprov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    tprov = json.loads((train / "PROVENANCE.json").read_text(encoding="utf-8"))
    failures = []

    # V1 -- checkpoints, tables, code
    ckpts = {a: train / a / "best.pt" for a in K.ARMS}
    for arm, path in ckpts.items():
        local = sha256(path) if path.is_file() else None
        if local is None:
            failures.append(f"V1: {path} missing")
            continue
        if gprov["checkpoints"].get(arm) != local:
            failures.append(f"V1: {arm} differs from the checkpoint generation used")
        if tprov["arms"].get(arm, {}).get("best_sha256") != local:
            failures.append(f"V1: {arm} differs from the checkpoint training recorded")
    if sha256(args.e7) != E7_SHA256:
        failures.append("V1: data/best_e7.pt is not E7")
    for name, (rel, want) in TABLES.items():
        if sha256(ROOT / rel) != want:
            failures.append(f"V1: {rel} is not the registered {name}")
    commit = gprov.get("code_commit", "")
    if tprov.get("code_commit") != commit:
        failures.append("V1: training and generation ran different code commits")
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT).returncode:
        failures.append(f"V1: code commit {commit!r} is not in this history")
    changed = git("diff", "--name-only", commit, "HEAD", "--", "uie",
                  "research/native/e8_common.py", "research/native/e9_common.py")
    if changed:                                                            # V3 as well
        failures.append(f"V1/V3: code changed since the GPU ran it: {changed.split()}")

    # V1 -- the 160 registered files
    jobs = K.jobs()
    names = {j["file"] for j in jobs}
    extra = sorted(p.name for p in gens.glob("*.json") if p.name != "PROVENANCE.json"
                   and p.name not in names)
    if extra:
        failures.append(f"V1: unregistered files {extra}")
    files = {}
    for j in jobs:
        p, rec = gens / j["file"], gprov["files"].get(j["file"]) or {}
        if not p.is_file():
            failures.append(f"V1: {j['file']} missing")
            continue
        urls = json.loads(p.read_text(encoding="utf-8"))
        if len(urls) != j["n"]:
            failures.append(f"V1: {j['file']} has {len(urls)} urls, registered {j['n']}")
        if rec.get("sha256") != sha256(p) or rec.get("seed") != j["seed"]:
            failures.append(f"V1: {j['file']} sha256 or seed differs from PROVENANCE")
        files[j["file"]] = (j, urls)
    print(f"  V1: {len(files)} of {len(jobs)} registered files present", flush=True)

    # V-id -- the shared start and the one registered difference
    vid = v_id(train, args.e7)
    failures += vid
    print(f"  V-id: {'identical start; settings as registered' if not vid else vid}", flush=True)

    # V2 -- byte identity on this CPU
    ident = {}
    if not args.skip_v2:
        shipped = SuffixTable.load(args.bundle / "suffix_table.json")
        tables = StateSuffixTables.load(ROOT / "data" / "state_suffix_tables.json")
        checks = [(arm, dec, "free", [f"free_{arm}_{dec}_benign_g0.json",
                                      f"free_{arm}_{dec}_phishing_g0.json"])
                  for arm, dec in K.FREE_CONDITIONS]
        checks += [(arm, dec, "control", [f"ctrl_{arm}_{dec}_benign_P1Q1_g0.json",
                                          f"ctrl_{arm}_{dec}_phishing_P0Q1_g0.json"])
                   for arm, dec in K.CTRL_CONDITIONS]
        loaded = {}
        for arm, dec, mode, targets in checks:
            if arm not in loaded:
                loaded.clear()
                m, t, _ = load_checkpoint(ckpts[arm], device=torch.device("cpu"))
                loaded[arm] = (m.eval(), t)
            model, tok = loaded[arm]
            same = compared = 0
            for f in targets:
                if f not in files:
                    continue
                j, urls = files[f]
                label, st = j["label"], j["state"]
                extra_kw = K.decode(tok, label, dec)
                if mode == "free":
                    table = shipped
                else:
                    extra_kw = {**extra_kw, "fields": dict(K.STATES[st])}
                    table = tables.for_state(label, K.STATES[st])
                s, c = identity(model, tok, table, urls, label, j["seed"], extra_kw,
                                -(-IDENTITY_MIN // len(targets)))
                same, compared = same + s, compared + c
                print(f"  V2 identity {f:<46} {s}/{c}", flush=True)
            ident[f"{arm}/{dec}/{mode}"] = {"compared": compared, "identical": same}
            if compared < IDENTITY_MIN or same != compared:
                failures.append(f"V2: {arm} {dec} {mode} identity {same}/{compared}")

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
            if [int(p.stem.rsplit("_s", 1)[1]) for p in src] != [K.seed_free(g) for g in range(K.G)]:
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
        (args.e9 / "e7_free_fields.json").write_text(json.dumps(e7_free, indent=1, default=float),
                                                     encoding="utf-8")

    verdict = "PASS" if not failures else "FAIL"
    (args.e9 / "VERIFY.json").write_text(json.dumps(
        {"registration": "research/native/PREREG_E9.md", "identity": ident, "v4": v4,
         "v_id": vid, "failures": failures, "verdict": verdict}, indent=1), encoding="utf-8")
    print("failures:", failures or "none")
    print("VERIFY:", verdict)


if __name__ == "__main__":
    main()
