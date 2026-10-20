"""E12 -- validity (research/native/PREREG_E12.md section 4): V1-V4 and V-id, before
any reading.

    python research/native/e12_verify.py --train-only    # V-id + checkpoints
    python research/native/e12_verify.py                 # everything, after generation

Put the Kaggle outputs in data/e12/train (E12_train: the three arm folders and
PROVENANCE.json) and data/e12/gen (E12_generate: the 150 files + PROVENANCE.json).
Every checkpoint is `final.pt` (the owner's decision 2). Writes
data/e12/VERIFY_TRAIN.json or data/e12/VERIFY.json.
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

import e12_common as K  # noqa: E402
from e8_verify import IDENTITY_MIN, TABLES, git, identity, placement, sha256  # noqa: E402
from e11_verify import supplied_identity  # noqa: E402

E12 = ROOT / "data" / "e12"
PRESENCE = ("data/e11/presence_table.json",
            "7a74345241af3b8acaaa15e6229920ec43dac2497c3aa2ee6dbfd849b62783c6")
PRESENCE_SLASH = ("data/e12/presence_slash_table.json",
                  "5a0eeac991b8343267d6f88f4c590bd6c208ace975dbf2218f8c5a277b71c050")


def v_id(train: Path) -> tuple:
    """D5: every arm's start contains E10's and E11's exactly; the slash arms share
    their new token rows; each arm's settings are the registered recipe."""
    import torch
    fails, starts = [], {}
    for arm in K.ARMS:
        rec = K.recipe(arm)
        blob = torch.load(train / arm / "final.pt", map_location="cpu", weights_only=False)
        res = json.loads((train / arm / "train_result.json").read_text(encoding="utf-8"))
        s = res.get("e12_start") or {}
        starts[arm] = {k: s.get(k) for k in ("e10_start_sha256", "e11_start_sha256",
                                             "new_token_rows_sha256", "new_position_rows_sha256")}
        if s.get("e10_start_sha256") != K.E10_START or s.get("e11_start_sha256") != K.E11_START:
            fails.append(f"V-id: {arm}'s start does not contain E10's and E11's")
        tk, mc, tc = blob["tokenizer"], blob["model_config"], blob["train_config"]
        slash = bool(rec.get("slash_plan"))
        want = [("presence_plan", tk.get("presence_plan", False), True),
                ("root_slash", tk.get("root_slash", False), True),
                ("slash_plan", tk.get("slash_plan", False), slash),
                ("vocab_size", mc.get("vocab_size"), K.E11_VOCAB + (2 if slash else 0)),
                ("max_len", mc.get("max_len"), rec.get("max_len", 256)),
                ("frame_every", mc.get("frame_every", False), False),
                ("d_model", mc.get("d_model"), rec["d_model"]),
                ("n_layers", mc.get("n_layers"), rec["n_layers"]),
                ("steps", tc.get("steps"), rec["steps"]), ("lr", tc.get("lr"), rec["lr"]),
                ("field_dropout", tc.get("field_dropout"), 0.0),
                ("loss_norm", tc.get("loss_norm"), "fixed"),
                ("grad_clip_late", tc.get("grad_clip_late"), 5.0),
                ("grad_clip_switch", tc.get("grad_clip_switch"), 1000),
                ("seed_init", res.get("seed_init"), True)]
        fails += [f"V-id: {arm} {k} = {got}, registered {exp}" for k, got, exp in want if got != exp]
        if abs((res.get("loss_constant") or 0) - rec["loss_constant"]) > 1e-6 * rec["loss_constant"]:
            fails.append(f"V-id: {arm} loss_constant {res.get('loss_constant')}")
    rows = {starts[a]["new_token_rows_sha256"] for a in K.SLASH_ARMS}
    if len(rows) != 1 or None in rows:
        fails.append(f"V-id: the slash arms' new token rows differ: {rows}")
    return fails, starts


def drawn_identity(model, tok, tables, urls, label: str, seed: int, budget: int):
    """V2 for a supplied-plan file (e12_common._draw_and_generate): the state counts
    and the outer shuffle depend only on the seed and n; within a state, E8's
    placement locates each suffix group. Regenerates leading groups, state by state,
    until `budget` URLs are compared."""
    import torch
    from uie.model import generate
    from uie.suffixes import SuffixTable
    n = len(urls)
    states = list(tables)
    rng = random.Random(seed)
    counts = collections.Counter(rng.choices(states, weights=[sum(tables[k].values())
                                                              for k in states], k=n))
    perm = list(range(n))
    rng.shuffle(perm)
    outer = {orig: pos for pos, orig in enumerate(perm)}
    same = compared = block = 0
    for i, k in enumerate(states):
        nk = counts[k]
        if not nk:
            continue
        s = seed + K.STATE_SEED * (i + 1)
        groups, inner = placement(SuffixTable({label: tables[k]}), label, nk, s)
        start = 0
        for j, ((frame, suffix), count) in enumerate(groups):
            if compared >= budget:
                break
            got = generate(model, tok, label=label, n=count, suffix=suffix, seed=s + j,
                           frame=frame, batch_size=32, device=torch.device("cpu"),
                           fields=K.fields_of(k), **K.decode())
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
    ap.add_argument("--e12", type=Path, default=E12)
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--skip-v2", action="store_true", help="dry runs only")
    ap.add_argument("--skip-v4", action="store_true", help="dry runs only")
    args = ap.parse_args()
    train, gens = args.e12 / "train", args.e12 / "gen"
    tprov = json.loads((train / "PROVENANCE.json").read_text(encoding="utf-8"))
    failures = [f"V1: {arm} final.pt differs from the checkpoint training recorded"
                for arm in K.ARMS
                if not (train / arm / "final.pt").is_file()
                or tprov["arms"].get(arm, {}).get("final_sha256") != sha256(train / arm / "final.pt")]
    commit = tprov.get("code_commit", "")
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT).returncode:
        failures.append(f"V1: code commit {commit!r} is not in this history")
    changed = git("diff", "--name-only", commit, "HEAD", "--", "uie",
                  "research/native/e8_common.py", "research/native/e11_common.py",
                  "research/native/e12_common.py")
    if changed:
        failures.append(f"V1/V3: code changed since the GPU ran it: {changed.split()}")
    vid_fails, vid = v_id(train)
    failures += vid_fails
    print(f"  V-id: {'as registered' if not vid_fails else vid_fails}", flush=True)
    if args.train_only:
        verdict = "PASS" if not failures else "FAIL"
        (args.e12 / "VERIFY_TRAIN.json").write_text(json.dumps(
            {"registration": "research/native/PREREG_E12.md", "scope": "training only",
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
        if gprov["checkpoints"].get(arm) != sha256(train / arm / "final.pt"):
            failures.append(f"V1: {arm} differs from the checkpoint generation used")
    if gprov.get("checkpoint_file") != "final.pt":
        failures.append(f"V1: generation read {gprov.get('checkpoint_file')}, registered final.pt")
    if gprov.get("code_commit") != commit:
        failures.append("V1: training and generation ran different code commits")
    for name, (rel, want) in {**TABLES, "presence_table.json": PRESENCE,
                              "presence_slash_table.json": PRESENCE_SLASH}.items():
        if sha256(ROOT / rel) != want:
            failures.append(f"V1: {rel} is not the registered {name}")
    if gprov.get("presence_table") != PRESENCE[1] or gprov.get("presence_slash_table") != PRESENCE_SLASH[1]:
        failures.append("V1: generation used other presence tables")
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
        ps = json.loads((ROOT / PRESENCE_SLASH[0]).read_text(encoding="utf-8"))["table"]
        for arm in K.ARMS:
            model, tok, _ = load_checkpoint(train / arm / "final.pt", device=torch.device("cpu"))
            model.eval()
            slash = arm in K.SLASH_ARMS
            for mode, targets in (("free", [f"free_{arm}_benign_g0.json", f"free_{arm}_phishing_g0.json"]),
                                  ("control", [f"ctrl_{arm}_benign_P1Q1_g0.json",
                                               f"ctrl_{arm}_phishing_P0Q0_g0.json"])):
                same = compared = 0
                per = -(-IDENTITY_MIN // len(targets))
                for f in targets:
                    if f not in files:
                        continue
                    j, urls = files[f]
                    if mode == "free" and slash:
                        s, c = drawn_identity(model, tok, K.supplied_tables(ps, shipped.counts, j["label"]),
                                              urls, j["label"], j["seed"], per)
                    elif mode == "free":
                        s, c = supplied_identity(model, tok, shipped, presence, urls, j["label"],
                                                 j["g"], per)
                    elif slash:
                        s, c = drawn_identity(model, tok, K.control_tables(ps, tables, j["label"], j["state"]),
                                              urls, j["label"], j["seed"], per)
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
        (args.e12 / "e7_free_fields.json").write_text(json.dumps(e7_free, indent=1, default=float),
                                                      encoding="utf-8")

    verdict = "PASS" if not failures else "FAIL"
    (args.e12 / "VERIFY.json").write_text(json.dumps(
        {"registration": "research/native/PREREG_E12.md", "identity": ident, "v4": v4,
         "v_id": vid, "failures": failures, "verdict": verdict}, indent=1), encoding="utf-8")
    print("failures:", failures or "none")
    print("VERIFY:", verdict)


if __name__ == "__main__":
    main()
