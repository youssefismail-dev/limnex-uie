"""E10 -- validity (research/native/PREREG_E10.md section 4): V1-V4 and V-id,
before any reading.

    python research/native/e10_verify.py --train-only    # V-id + checkpoints (K10a can be read)
    python research/native/e10_verify.py                 # everything, after generation

Put the Kaggle outputs in data/e10/train (E10_train: e10_base/, e10_plan/,
PROVENANCE.json) and data/e10/gen (E10_generate: the 100 files + PROVENANCE.json).
Writes data/e10/VERIFY_TRAIN.json or data/e10/VERIFY.json.
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

import e10_common as K  # noqa: E402
from e8_verify import IDENTITY_MIN, TABLES, git, identity, sha256  # noqa: E402

E9_LN = ROOT / "data" / "e9" / "train" / "e9_ln" / "best.pt"
E9_START = "26926388bb575e2c9f70f7a611f762b3e53062f58f6638c70d13f0e46f213ad1"
E9_COUNTS = {"benign": 2_545_305, "phishing": 238_199}
E9_DROPPED = 10_313


def v_id(train: Path) -> tuple:
    """The shared start and the registered difference, against e9_ln's recorded
    tokenizer, model and training settings (the baseline recipe's own run)."""
    import torch
    fails, starts = [], {}
    ref = torch.load(E9_LN, map_location="cpu", weights_only=False)
    for arm in K.ARMS:
        rec = K.recipe(arm)
        blob = torch.load(train / arm / "best.pt", map_location="cpu", weights_only=False)
        res = json.loads((train / arm / "train_result.json").read_text(encoding="utf-8"))
        starts[arm] = res.get("init_weights_sha256")
        tk = dict(blob["tokenizer"])
        plan = bool(tk.pop("presence_plan", False))
        ref_tk = {k: v for k, v in ref["tokenizer"].items() if k != "presence_plan"}
        if tk != ref_tk:
            fails.append(f"V-id: {arm} tokenizer differs from the baseline's beyond the plan")
        if plan != bool(rec.get("presence_plan", False)):
            fails.append(f"V-id: {arm} presence_plan {plan}, registered {rec.get('presence_plan')}")
        if blob["model_config"] != ref["model_config"]:
            fails.append(f"V-id: {arm} model settings differ from the baseline's")
        tc = blob["train_config"]
        for k, v in ref["train_config"].items():
            want = rec["field_dropout"] if k == "field_dropout" else v
            if tc.get(k) != want:
                fails.append(f"V-id: {arm} train_config {k} = {tc.get(k)}, registered {want}")
        if abs((res.get("loss_constant") or 0) - rec["loss_constant"]) > 1e-6 * rec["loss_constant"]:
            fails.append(f"V-id: {arm} loss_constant {res.get('loss_constant')}")
        if res.get("seed_init") is not True:
            fails.append(f"V-id: {arm} was not seeded")
        if res.get("counts") != E9_COUNTS or res.get("dropped_over_max_len") != E9_DROPPED:
            fails.append(f"V-id: {arm} data {res.get('counts')} / {res.get('dropped_over_max_len')}"
                         f" differ from E9's")
    if None in starts.values() or len(set(starts.values())) != 1:
        fails.append(f"V-id: the arms' initial weights differ: {starts}")
    same_as_e9 = len(set(starts.values())) == 1 and next(iter(starts.values())) == E9_START
    return fails, {"starts": starts, "start_equals_e9_ln": same_as_e9}


def code_checks(prov_commit: str) -> list:
    fails = []
    if subprocess.run(["git", "merge-base", "--is-ancestor", prov_commit, "HEAD"],
                      cwd=ROOT).returncode:
        fails.append(f"V1: code commit {prov_commit!r} is not in this history")
    changed = git("diff", "--name-only", prov_commit, "HEAD", "--", "uie",
                  "research/native/e8_common.py", "research/native/e10_common.py")
    if changed:
        fails.append(f"V1/V3: code changed since the GPU ran it: {changed.split()}")
    return fails


def checkpoints(train: Path, tprov: dict) -> list:
    return [f"V1: {arm} best.pt differs from the checkpoint training recorded"
            for arm in K.ARMS
            if not (train / arm / "best.pt").is_file()
            or tprov["arms"].get(arm, {}).get("best_sha256") != sha256(train / arm / "best.pt")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e10", type=Path, default=ROOT / "data" / "e10")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--skip-v2", action="store_true", help="dry runs only")
    ap.add_argument("--skip-v4", action="store_true", help="dry runs only")
    args = ap.parse_args()
    train, gens = args.e10 / "train", args.e10 / "gen"
    tprov = json.loads((train / "PROVENANCE.json").read_text(encoding="utf-8"))
    failures = checkpoints(train, tprov) + code_checks(tprov.get("code_commit", ""))
    vid_fails, vid = v_id(train)
    failures += vid_fails
    print(f"  V-id: {'identical start; settings as registered' if not vid_fails else vid_fails}"
          f"  (start equals e9_ln's: {vid['start_equals_e9_ln']})", flush=True)
    if args.train_only:
        verdict = "PASS" if not failures else "FAIL"
        (args.e10 / "VERIFY_TRAIN.json").write_text(json.dumps(
            {"registration": "research/native/PREREG_E10.md", "scope": "training only",
             "v_id": vid, "failures": failures, "verdict": verdict}, indent=1), encoding="utf-8")
        print("failures:", failures or "none")
        print("VERIFY (training side):", verdict)
        return

    import torch
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    from uie.train import load_checkpoint

    gprov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    for arm in K.ARMS:
        if gprov["checkpoints"].get(arm) != sha256(train / arm / "best.pt"):
            failures.append(f"V1: {arm} differs from the checkpoint generation used")
    if gprov.get("code_commit") != tprov.get("code_commit"):
        failures.append("V1: training and generation ran different code commits")
    for name, (rel, want) in TABLES.items():
        if sha256(ROOT / rel) != want:
            failures.append(f"V1: {rel} is not the registered {name}")
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

    ident = {}
    if not args.skip_v2:
        shipped = SuffixTable.load(args.bundle / "suffix_table.json")
        tables = StateSuffixTables.load(ROOT / "data" / "state_suffix_tables.json")
        for arm in K.ARMS:
            model, tok, _ = load_checkpoint(train / arm / "best.pt", device=torch.device("cpu"))
            model.eval()
            for mode, targets in (("free", [f"free_{arm}_benign_g0.json",
                                            f"free_{arm}_phishing_g0.json"]),
                                  ("control", [f"ctrl_{arm}_benign_P1Q1_g0.json",
                                               f"ctrl_{arm}_phishing_P0Q1_g0.json"])):
                same = compared = 0
                for f in targets:
                    if f not in files:
                        continue
                    j, urls = files[f]
                    extra_kw = K.decode()
                    if mode == "free":
                        table = shipped
                    else:
                        extra_kw = {**extra_kw, "fields": dict(K.STATES[j["state"]])}
                        table = tables.for_state(j["label"], K.STATES[j["state"]])
                    s, c = identity(model, tok, table, urls, j["label"], j["seed"], extra_kw,
                                    -(-IDENTITY_MIN // len(targets)))
                    same, compared = same + s, compared + c
                    print(f"  V2 identity {f:<40} {s}/{c}", flush=True)
                ident[f"{arm}/{mode}"] = {"compared": compared, "identical": same}
                if compared < IDENTITY_MIN or same != compared:
                    failures.append(f"V2: {arm} {mode} identity {same}/{compared}")

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
        (args.e10 / "e7_free_fields.json").write_text(json.dumps(e7_free, indent=1, default=float),
                                                      encoding="utf-8")

    verdict = "PASS" if not failures else "FAIL"
    (args.e10 / "VERIFY.json").write_text(json.dumps(
        {"registration": "research/native/PREREG_E10.md", "identity": ident, "v4": v4,
         "v_id": vid, "failures": failures, "verdict": verdict}, indent=1), encoding="utf-8")
    print("failures:", failures or "none")
    print("VERIFY:", verdict)


if __name__ == "__main__":
    main()
