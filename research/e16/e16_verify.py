"""E16 -- validity (research/e16/PREREG_E16.md section 8; amendment 2), before anything is
read.

    python research/e16/e16_verify.py --step train      # after E16_train:    V1, V-start', V-batch,
                                                        #   V-mass', V-stream, V-accum, the recipe
    python research/e16/e16_verify.py --step generate   # after E16_generate: V1, V2

Layout (the Kaggle outputs, extracted):
    data/e16/train         E16_train:    e16_cap/, PROVENANCE.json
    data/e16/generate/gen  E16_generate: e16_cap's 50 files + PROVENANCE.json
Writes data/e16/VERIFY_<STEP>.json. V2 regenerates leading suffix groups on this CPU with
e16_cap's decode (raw + D1 + D1') and compares bytes, as E12-E15 did.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e15", ROOT / "research" / "e14", ROOT / "research" / "e13",
          ROOT / "research" / "native", ROOT / "research" / "cr1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as K  # noqa: E402
import e15_common as E15  # noqa: E402
import e16_common as E16  # noqa: E402
from e8_verify import IDENTITY_MIN, git, sha256  # noqa: E402
from e13_verify import drawn_identity, files_check, tables_check  # noqa: E402

E16D = ROOT / "data" / "e16"
E15_NEW = ROOT / "data" / "e15" / "train" / "e15_new" / "final.pt"
STREAM = ROOT / "data" / "e15" / "qry_extra" / E16.STREAM_FILE
STEPS = 20_000
CODE = ["uie", "research/native/e8_common.py", "research/native/e11_common.py",
        "research/native/e12_common.py", "research/e13/e13_common.py",
        "research/e15/e15_common.py", "research/e16/e16_common.py"]
PRINTS = ("e10_start_sha256", "e11_start_sha256", "new_token_rows_sha256", "new_position_rows_sha256")


def code_unchanged(commit: str) -> list:
    out = []
    if not commit or subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                    cwd=ROOT).returncode:
        return [f"V1: code commit {commit!r} is not in this history"]
    changed = git("diff", "--name-only", commit, "HEAD", "--", *CODE)
    if changed:
        out.append(f"V1/V3: code changed since the GPU ran it: {changed.split()}")
    return out


def v_accum() -> tuple:
    """V-accum: the unit tests at this commit (tests/test_e16.py)."""
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/test_e16.py"], cwd=ROOT, capture_output=True, text=True)
    return ([] if r.returncode == 0 else ["V-accum: tests/test_e16.py failed"]), \
        f"tests/test_e16.py exit {r.returncode}"


def step_train(args) -> dict:
    import torch
    train = E16D / "train"
    prov = json.loads((train / "PROVENANCE.json").read_text(encoding="utf-8"))
    fails = code_unchanged(prov.get("code_commit", ""))
    arm = prov["arms"].get(E16.ARM, {})
    ck = train / E16.ARM / "final.pt"
    # V1
    if not ck.is_file() or arm.get("final_sha256") != sha256(ck):
        fails.append("V1: e16_cap final.pt differs from the checkpoint training recorded")
    if sha256(E15_NEW) != E16.E15_NEW_SHA256:
        fails.append("V1: the laptop's e15_new final.pt is not the adopted baseline")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    base = torch.load(E15_NEW, map_location="cpu", weights_only=False)
    tc, mc, tk = blob["train_config"], blob["model_config"], blob["tokenizer"]
    r = json.loads((train / E16.ARM / "train_result.json").read_text(encoding="utf-8"))
    accum = (prov.get("memory") or {}).get("accum_steps")
    # the recipe: e15_new's in every respect but the width and the accumulation
    for k, v in base["train_config"].items():
        got = Path(tc.get(k, "")).name if k == "qry_extra" else tc.get(k)
        want = E16.STREAM_FILE if k == "qry_extra" else v
        if got != want:
            fails.append(f"recipe: train_config {k} = {got!r}, e15_new's {want!r}")
    if sorted(set(tc) - set(base["train_config"])) != ["accum_steps"]:
        fails.append(f"recipe: extra settings {sorted(set(tc) - set(base['train_config']))}")
    if accum not in (E16.ACCUM_STEPS, E16.ACCUM_FALLBACK) or tc.get("accum_steps") != accum \
            or r.get("accum_steps") != accum:
        fails.append(f"recipe: accum_steps {tc.get('accum_steps')} / {r.get('accum_steps')}, "
                     f"the memory check chose {accum}")
    for k, v in base["model_config"].items():
        want = {"d_model": E16.D_MODEL, "d_ff": 4 * E16.D_MODEL}.get(k, v)
        if mc.get(k) != want:
            fails.append(f"recipe: model_config {k} = {mc.get(k)}, registered {want}")
    if tk != base["tokenizer"]:
        fails.append("recipe: e16_cap's tokenizer is not e15_new's")
    if r.get("params") != E16.PARAMS:
        fails.append(f"recipe: {r.get('params')} parameters, registered {E16.PARAMS}")
    if r.get("counts") != E16.COUNTS or r.get("dropped_over_max_len") != E16.DROPPED:
        fails.append(f"V1: rows {r.get('counts')} / dropped {r.get('dropped_over_max_len')}")
    if r.get("loss_constant") != E16.LOSS_CONSTANT:
        fails.append(f"V1: C = {r.get('loss_constant')}")
    if not all(isinstance(x, float) and math.isfinite(x) for x in (r.get("final") or {}).values()):
        fails.append("V1: non-finite validation loss")
    # V-start' (amendment 2): Kaggle's width-384 rebuild equalled e12_final's recorded start,
    # and training started from the width-512 start the notebook recorded before training
    sc = prov.get("start_check") or {}
    w384, w512 = sc.get("width_384") or {}, sc.get("width_512") or {}
    rec = {"init_weights_sha256": E15.E12_FINAL_INIT, **E15.E12_FINAL_START}
    if any(w384.get(k) != v for k, v in rec.items()):
        fails.append("V-start': the notebook's width-384 rebuild is not e12_final's recorded start")
    if r.get("init_weights_sha256") != w512.get("init_weights_sha256"):
        fails.append("V-start': training did not start from the recorded width-512 start")
    s = r.get("e12_start") or {}
    fails += [f"V-start': {k} differs from the recorded width-512 start" for k in PRINTS
              if s.get(k) != w512.get(k)]
    # V-batch, V-mass', V-stream
    q = r.get("qry_extra") or {}
    if q.get("old_batch_digest") != E16.V_BATCH_DIGEST or q.get("steps") != STEPS:
        fails.append(f"V-batch: digest {str(q.get('old_batch_digest'))[:12]} over {q.get('steps')} steps")
    st = q.get("strata") or {}
    got = {k: {f: v.get(f) for f in ("old_rows_seen", "appended_rows", "appended_query_positions")}
           for k, v in st.items()}
    if got != E16.E15_NEW_INJECTION:
        fails.append(f"V-mass': injection {got}, e15_new's {E16.E15_NEW_INJECTION}")
    local = hashlib.sha256(STREAM.read_bytes()).hexdigest()
    if not (q.get("source_sha256") == local == prov.get("stream_sha256")):
        fails.append("V-stream: the stream's hash differs")
    for k in E15.N_NEW:
        v = st.get(k, {})
        if (v.get("N_new"), v.get("N_old"), v.get("p")) != (E15.N_NEW[k], E15.N_OLD[k], E15.P_INJECT[k]):
            fails.append(f"V-stream: {k} N_new {v.get('N_new')}, N_old {v.get('N_old')}, p {v.get('p')}")
    f_acc, tests = ([], "skipped") if args.skip_tests else v_accum()
    fails += f_acc
    clip = {k: v["clipped_share"] for k, v in (r.get("grad_diagnostics") or {}).items()
            if k.startswith("after_1000")}
    return {"accum_steps": accum, "memory": prov.get("memory"), "torch": prov.get("torch"),
            "start_width_512": w512, "v_accum_tests": tests, "clipped_after_1000": clip,
            "final_val": r.get("final"), "final_val_e15_new": E15_NEW_FINAL,
            "wall_minutes": arm.get("wall_minutes"), "failures": fails}


#: e15_new's training record (data/e15/train/e15_new/train_result.json), for the report
E15_NEW_FINAL = {"benign": 1.5607721298933028, "phishing": 1.4003939181566238, "mean": 1.4805830240249633}


def identity_for(model, tok, files: dict, targets: list, budget=IDENTITY_MIN):
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    from e8_verify import TABLES
    from e13_verify import PS
    shipped = SuffixTable.load(ROOT / TABLES["suffix_table.json"][0])
    st = StateSuffixTables.load(ROOT / TABLES["state_suffix_tables.json"][0])
    ps = json.loads((ROOT / PS[0]).read_text(encoding="utf-8"))["table"]
    same = compared = 0
    per = -(-budget // len(targets))
    for f in targets:
        if f not in files:
            continue
        j, urls = files[f]
        tables = (K.supplied_tables(ps, shipped.counts, j["label"]) if j["mode"] == "free"
                  else K.control_tables(ps, st, j["label"], j["state"]))
        s, c = drawn_identity(model, tok, tables, urls, j["label"], j["seed"], per,
                              lambda k: E16.decode_kwargs())
        same, compared = same + s, compared + c
        print(f"  V2 identity {f:<46} {s}/{c}", flush=True)
    return same, compared


def step_generate(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    gens, train = E16D / "generate" / "gen", E16D / "train"
    gprov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    fails = code_unchanged(gprov.get("code_commit", "")) + tables_check(gprov)
    ck = train / E16.ARM / "final.pt"
    if gprov.get("checkpoints") != {E16.ARM: sha256(ck)}:
        fails.append("V1: generation did not read e16_cap's verified final.pt")
    if json.loads((E16D / "VERIFY_TRAIN.json").read_text(encoding="utf-8")).get("verdict") != "PASS":
        fails.append("V1: VERIFY_TRAIN has not passed")
    f2, files = files_check(gens, gprov, E16.jobs())
    fails += f2
    ident = {}
    if not args.skip_v2:
        model, tok, _ = load_checkpoint(ck, device=torch.device("cpu"))
        model.eval()
        arm = E16.ARM
        for mode, targets in (("free", [f"free_{arm}_benign_g0.json", f"free_{arm}_phishing_g0.json"]),
                              ("control", [f"ctrl_{arm}_benign_P1Q1_g0.json",
                                           f"ctrl_{arm}_benign_P0Q1_g0.json"])):
            s, c = identity_for(model, tok, files, targets)
            ident[f"{arm}/{mode}"] = {"compared": c, "identical": s}
            if c < IDENTITY_MIN or s != c:
                fails.append(f"V2: {arm} {mode} identity {s}/{c}")
    return {"identity": ident, "failures": fails}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("train", "generate"))
    ap.add_argument("--skip-v2", action="store_true", help="dry runs only")
    ap.add_argument("--skip-tests", action="store_true", help="dry runs only")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rep = {"train": step_train, "generate": step_generate}[args.step](args)
    rep.update({"registration": E16.REGISTRATION, "step": args.step,
                "verdict": "PASS" if not rep["failures"] else "FAIL"})
    E16D.mkdir(parents=True, exist_ok=True)
    (E16D / f"VERIFY_{args.step.upper()}.json").write_text(json.dumps(rep, indent=1, default=float),
                                                          encoding="utf-8")
    if args.step == "train":
        print(f"accumulation: {rep['accum_steps']} parts; memory {rep['memory']}; torch {rep['torch']}")
        print(f"clipped after step 1,000: {rep['clipped_after_1000']}")
        print(f"final val: {rep['final_val']}  (e15_new {rep['final_val_e15_new']})")
        print(f"V-accum tests: {rep['v_accum_tests']}")
    print("failures:", rep["failures"] or "none")
    print(f"VERIFY ({args.step}):", rep["verdict"])


if __name__ == "__main__":
    main()
