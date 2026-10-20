"""E11 -- validity (research/native/PREREG_E11.md section 4): V1-V4 and V-id, before
any reading.

    python research/native/e11_verify.py --train-only    # V-id + checkpoints
    python research/native/e11_verify.py                 # everything, after generation

Put the Kaggle outputs in data/e11/train (E11_train: the four arm folders and
PROVENANCE.json) and data/e11/gen (E11_generate: the 200 files + PROVENANCE.json).
Writes data/e11/VERIFY_TRAIN.json or data/e11/VERIFY.json.
"""
from __future__ import annotations

import argparse
import collections
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

import e11_common as K  # noqa: E402
from e8_verify import IDENTITY_MIN, TABLES, git, identity, placement, sha256  # noqa: E402

E9_LN = ROOT / "data" / "e9" / "train" / "e9_ln" / "best.pt"
PRESENCE = ("data/e11/presence_table.json",
            "7a74345241af3b8acaaa15e6229920ec43dac2497c3aa2ee6dbfd849b62783c6")


def v_id(train: Path, ck: str = "best") -> tuple:
    """The shared start (identical across the arms; its part without <RS> is E10's)
    and each arm's registered settings, against e9_ln's recorded tokenizer, model
    and training settings (the baseline recipe's own run)."""
    import torch
    fails, starts, bases, counts = [], {}, {}, {}
    ref = torch.load(E9_LN, map_location="cpu", weights_only=False)
    for arm in K.ARMS:
        rec = K.recipe(arm)
        blob = torch.load(train / arm / f"{ck}.pt", map_location="cpu", weights_only=False)
        res = json.loads((train / arm / "train_result.json").read_text(encoding="utf-8"))
        starts[arm] = res.get("init_weights_sha256")
        bases[arm] = (res.get("shared_start") or {}).get("base_sha256")
        counts[arm] = {"counts": res.get("counts"), "dropped": res.get("dropped_over_max_len")}
        tk = dict(blob["tokenizer"])
        plan = bool(tk.pop("presence_plan", False))
        rs = bool(tk.pop("root_slash", False))
        ref_tk = {k: v for k, v in ref["tokenizer"].items() if k not in ("presence_plan", "root_slash")}
        if tk != ref_tk:
            fails.append(f"V-id: {arm} tokenizer differs from the baseline's beyond the plan and <RS>")
        if not rs or plan != bool(rec.get("presence_plan", False)):
            fails.append(f"V-id: {arm} root_slash {rs}, presence_plan {plan}")
        mc = dict(blob["model_config"])
        if mc.pop("vocab_size") != ref["model_config"]["vocab_size"] + 1 or \
                mc != {k: v for k, v in ref["model_config"].items() if k != "vocab_size"}:
            fails.append(f"V-id: {arm} model settings differ from the baseline's beyond <RS>")
        tc = blob["train_config"]
        for k, v in ref["train_config"].items():
            want = rec["field_dropout"] if k == "field_dropout" else v
            if tc.get(k) != want:
                fails.append(f"V-id: {arm} train_config {k} = {tc.get(k)}, registered {want}")
        for k, want in (("grad_clip_late", rec.get("grad_clip_late")), ("grad_clip_switch", 1000),
                        ("field_dropout_joint", rec.get("field_dropout_joint", False))):
            if tc.get(k) != want:
                fails.append(f"V-id: {arm} train_config {k} = {tc.get(k)}, registered {want}")
        if abs((res.get("loss_constant") or 0) - rec["loss_constant"]) > 1e-6 * rec["loss_constant"]:
            fails.append(f"V-id: {arm} loss_constant {res.get('loss_constant')}")
        if res.get("seed_init") is not True or not (res.get("shared_start") or {}).get("on"):
            fails.append(f"V-id: {arm} was not built from the shared start")
    if None in starts.values() or len(set(starts.values())) != 1:
        fails.append(f"V-id: the arms' initial weights differ: {starts}")
    if set(bases.values()) != {K.E10_START}:
        fails.append(f"V-id: the start without <RS> is not E10's: {bases}")
    return fails, {"starts": starts, "base_starts": bases, "base_equals_e10": set(bases.values()) == {K.E10_START},
                   "data": counts}


def code_checks(prov_commit: str) -> list:
    fails = []
    if subprocess.run(["git", "merge-base", "--is-ancestor", prov_commit, "HEAD"],
                      cwd=ROOT).returncode:
        fails.append(f"V1: code commit {prov_commit!r} is not in this history")
    changed = git("diff", "--name-only", prov_commit, "HEAD", "--", "uie",
                  "research/native/e8_common.py", "research/native/e11_common.py")
    if changed:
        fails.append(f"V1/V3: code changed since the GPU ran it: {changed.split()}")
    return fails


def supplied_identity(model, tok, shipped, presence, urls, label, g, budget):
    """V2 for D6's free mode: the state draw and the outer shuffle depend only on
    the seed and n, so every state's block lands at known positions; within a
    state, E8's placement locates each suffix group. Regenerates leading groups
    of the states in order until `budget` URLs are compared."""
    import torch
    from uie.model import generate
    from uie.suffixes import SuffixTable
    n = len(urls)
    tables = K.supplied_tables(presence, shipped.counts, label)
    rng = random.Random(K.seed_free(g))
    counts = collections.Counter(rng.choices(K.PRESENCE, weights=[sum(tables[k].values())
                                                                 for k in K.PRESENCE], k=n))
    perm = list(range(n))
    rng.shuffle(perm)
    outer = {orig: pos for pos, orig in enumerate(perm)}
    same = compared = block = 0
    for i, k in enumerate(K.PRESENCE):
        nk = counts[k]
        if not nk:
            continue
        seed = K.seed_free(g) + K.STATE_SEED * (i + 1)
        table = SuffixTable({label: tables[k]})
        groups, inner = placement(table, label, nk, seed)
        start = 0
        for j, ((frame, suffix), count) in enumerate(groups):
            if compared >= budget:
                break
            got = generate(model, tok, label=label, n=count, suffix=suffix, seed=seed + j,
                           frame=frame, batch_size=32, device=torch.device("cpu"),
                           fields=dict(K.STATES[k]), **K.decode())
            for rank, u in enumerate(got):
                same += urls[outer[block + inner[start + rank]]] == u
                compared += 1
            start += count
        block += nk
        if compared >= budget:
            break
    return same, compared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e11", type=Path, default=ROOT / "data" / "e11")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--train", type=Path, help="the training output (default: <e11>/train)")
    ap.add_argument("--checkpoint", choices=("best", "final"), default="best",
                    help="amendment 3: final reads each arm's final.pt (step 20,000)")
    ap.add_argument("--skip-v2", action="store_true", help="dry runs only")
    ap.add_argument("--skip-v4", action="store_true", help="dry runs only")
    args = ap.parse_args()
    train, gens = args.train or args.e11 / "train", args.e11 / "gen"
    ck = args.checkpoint
    tprov = json.loads((train / "PROVENANCE.json").read_text(encoding="utf-8"))
    failures = [f"V1: {arm} best.pt differs from the checkpoint training recorded"
                for arm in K.ARMS
                if not (train / arm / f"{ck}.pt").is_file()
                or tprov["arms"].get(arm, {}).get(f"{ck}_sha256") != sha256(train / arm / f"{ck}.pt")]
    failures += code_checks(tprov.get("code_commit", ""))
    vid_fails, vid = v_id(train, ck)
    failures += vid_fails
    print(f"  V-id: {'one shared start; settings as registered' if not vid_fails else vid_fails}"
          f"  (start without <RS> equals E10's: {vid['base_equals_e10']})", flush=True)
    if args.train_only:
        verdict = "PASS" if not failures else "FAIL"
        (args.e11 / "VERIFY_TRAIN.json").write_text(json.dumps(
            {"registration": "research/native/PREREG_E11.md", "scope": "training only",
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
        if gprov["checkpoints"].get(arm) != sha256(train / arm / f"{ck}.pt"):
            failures.append(f"V1: {arm} differs from the checkpoint generation used")
    if gprov.get("code_commit") != tprov.get("code_commit"):
        failures.append("V1: training and generation ran different code commits")
    for name, (rel, want) in {**TABLES, "presence_table.json": PRESENCE}.items():
        if sha256(ROOT / rel) != want:
            failures.append(f"V1: {rel} is not the registered {name}")
    if gprov.get("checkpoint_file", "best.pt") != f"{ck}.pt":
        failures.append(f"V1: generation read {gprov.get('checkpoint_file', 'best.pt')}, "
                        f"this reading is {ck}.pt")
    if gprov.get("presence_table") != PRESENCE[1]:
        failures.append("V1: generation used another presence table")
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
        presence = json.loads((ROOT / PRESENCE[0]).read_text(encoding="utf-8"))
        for arm in K.ARMS:
            model, tok, _ = load_checkpoint(train / arm / f"{ck}.pt", device=torch.device("cpu"))
            model.eval()
            for mode, targets in (("free", [f"free_{arm}_benign_g0.json",
                                            f"free_{arm}_phishing_g0.json"]),
                                  ("control", [f"ctrl_{arm}_benign_P1Q1_g0.json",
                                               f"ctrl_{arm}_phishing_P0Q1_g0.json"])):
                same = compared = 0
                per = -(-IDENTITY_MIN // len(targets))
                for f in targets:
                    if f not in files:
                        continue
                    j, urls = files[f]
                    if mode == "free" and arm in K.SUPPLIED:
                        s, c = supplied_identity(model, tok, shipped, presence, urls, j["label"],
                                                 j["g"], per)
                    elif mode == "free":
                        s, c = identity(model, tok, shipped, urls, j["label"], j["seed"],
                                        K.decode(), per)
                    else:
                        s, c = identity(model, tok, tables.for_state(j["label"], K.STATES[j["state"]]),
                                        urls, j["label"], j["seed"],
                                        {**K.decode(), "fields": dict(K.STATES[j["state"]])}, per)
                    same, compared = same + s, compared + c
                    print(f"  V2 identity {f:<42} {s}/{c}", flush=True)
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
        (args.e11 / "e7_free_fields.json").write_text(json.dumps(e7_free, indent=1, default=float),
                                                      encoding="utf-8")

    verdict = "PASS" if not failures else "FAIL"
    (args.e11 / "VERIFY.json").write_text(json.dumps(
        {"registration": "research/native/PREREG_E11.md", "identity": ident, "v4": v4,
         "v_id": vid, "failures": failures, "verdict": verdict}, indent=1), encoding="utf-8")
    print("failures:", failures or "none")
    print("VERIFY:", verdict)


if __name__ == "__main__":
    main()
