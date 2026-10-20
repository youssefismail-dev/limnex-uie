"""E14 -- validity (research/e14/PREREG_E14.md section 5), before anything is read.

    python research/e14/e14_verify.py --step train      # after E14_train:    V1, V5
    python research/e14/e14_verify.py --step decode     # after E14_decode:   V1, V2, V7
    python research/e14/e14_verify.py --step generate   # after E14_generate: V1, V2

Layout (the Kaggle outputs, extracted):
    data/e14/train         E14_train:    e14_q/, PROVENANCE.json
    data/e14/decode/gen    E14_decode:   e12_d1p's 50 files + PROVENANCE.json
    data/e14/generate/gen  E14_generate: e14_q's and e14_qd's 100 files + PROVENANCE.json
(the `<base>/gen` layout E12's scorer reads). Writes data/e14/VERIFY_<STEP>.json.
V2 regenerates leading suffix groups on this CPU with each file's own decode
arguments (D1' included) and compares bytes (E12's and E13's method).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e13", ROOT / "research" / "native",
          ROOT / "research" / "cr1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as K  # noqa: E402
import e14_common as E14  # noqa: E402
from e8_verify import IDENTITY_MIN, git, sha256  # noqa: E402
from e13_verify import drawn_identity, files_check, tables_check  # noqa: E402

E14D = ROOT / "data" / "e14"
E12_FINAL = ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt"
E12_FINAL_RESULT = ROOT / "data" / "e12" / "train" / "e12_final" / "train_result.json"
E13_D1 = ROOT / "data" / "e13" / "decode" / "gen"            # e12_d1's files (V7)
CODE = ["uie", "research/native/e8_common.py", "research/native/e11_common.py",
        "research/native/e12_common.py", "research/e13/e13_common.py",
        "research/e14/e14_common.py"]


def code_unchanged(commit: str) -> list:
    out = []
    if not commit or subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                    cwd=ROOT).returncode:
        out.append(f"V1: code commit {commit!r} is not in this history")
        return out
    changed = git("diff", "--name-only", commit, "HEAD", "--", *CODE)
    if changed:
        out.append(f"V1/V3: code changed since the GPU ran it: {changed.split()}")
    return out


def identity_for(model, tok, files: dict, targets: list, arm: str, budget=IDENTITY_MIN):
    """V2 over the target files of one arm and mode, with the arm's own decode."""
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
                              lambda k: E14.state_kwargs(arm))
        same, compared = same + s, compared + c
        print(f"  V2 identity {f:<46} {s}/{c}", flush=True)
    return same, compared


def v2(model, tok, files, arm, fails, ident, control=("benign_P1Q1", "phishing_P1Q0")):
    """V2 per mode; the control files chosen where D1' acts (P1 states)."""
    for mode, targets in (("free", [f"free_{arm}_benign_g0.json", f"free_{arm}_phishing_g0.json"]),
                          ("control", [f"ctrl_{arm}_{c}_g0.json" for c in control])):
        s, c = identity_for(model, tok, files, targets, arm)
        ident[f"{arm}/{mode}"] = {"compared": c, "identical": s}
        if c < IDENTITY_MIN or s != c:
            fails.append(f"V2: {arm} {mode} identity {s}/{c}")


def step_train(args) -> dict:
    import torch
    train = E14D / "train"
    tprov = json.loads((train / "PROVENANCE.json").read_text(encoding="utf-8"))
    fails = code_unchanged(tprov.get("code_commit", ""))
    ck = train / "e14_q" / "final.pt"
    if not ck.is_file() or tprov["arms"].get("e14_q", {}).get("final_sha256") != sha256(ck):
        fails.append("V1: e14_q final.pt differs from the checkpoint training recorded")
    if sha256(E12_FINAL) != E14.E12_FINAL_SHA256:
        fails.append("V1: the laptop's e12_final.pt is not E12's")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    tc, mc, tk = blob["train_config"], blob["model_config"], blob["tokenizer"]
    e12 = torch.load(E12_FINAL, map_location="cpu", weights_only=False)
    r = json.loads((train / "e14_q" / "train_result.json").read_text(encoding="utf-8"))
    r12 = json.loads(E12_FINAL_RESULT.read_text(encoding="utf-8"))
    s = r.get("e12_start") or {}
    # V5: e12_final's recipe in every respect but the weight
    for k, v in e12["train_config"].items():
        if tc.get(k) != v:
            fails.append(f"V5: e14_q train_config {k} = {tc.get(k)}, e12_final's {v}")
    extra = set(tc) - set(e12["train_config"])
    if extra != {"query_weight"} or tc.get("query_weight") != E14.QUERY_WEIGHT:
        fails.append(f"V5: train_config adds {sorted(extra)}, query_weight {tc.get('query_weight')}")
    for k, v in e12["model_config"].items():
        if mc.get(k) != v:
            fails.append(f"V5: e14_q model_config {k} = {mc.get(k)}, e12_final's {v}")
    if tk != e12["tokenizer"]:
        fails.append("V5: e14_q's tokenizer is not e12_final's")
    if r.get("init_weights_sha256") != E14.E12_FINAL_INIT:
        fails.append("V5: e14_q did not start from E12's start")
    fails += [f"V5: start {k} differs" for k, v in E14.E12_FINAL_START.items() if s.get(k) != v]
    if r.get("counts") != E14.E12_FINAL_COUNTS or r.get("counts") != r12["counts"]:
        fails.append(f"V5: row counts {r.get('counts')}, e12_final's {r12['counts']}")
    if r.get("dropped_over_max_len") != E14.E12_FINAL_DROPPED:
        fails.append(f"V5: dropped {r.get('dropped_over_max_len')}, e12_final's {E14.E12_FINAL_DROPPED}")
    if r.get("loss_constant") != E14.E12_FINAL_LOSS_CONSTANT:
        fails.append(f"V5: C = {r.get('loss_constant')}, e12_final's {E14.E12_FINAL_LOSS_CONSTANT}")
    bands = {k: v["steps"] for k, v in (r.get("grad_diagnostics") or {}).items() if k.startswith("all ")}
    bands12 = {k: v["steps"] for k, v in r12["grad_diagnostics"].items() if k.startswith("all ")}
    if bands != E14.E12_FINAL_BAND_STEPS or bands != bands12:
        fails.append(f"V5: band steps {bands} (w = 1 mass), e12_final's {bands12}: not the same batches")
    q = r.get("query_weight") or {}
    if q.get("w") != E14.QUERY_WEIGHT or not (0.0 < (q.get("query_share") or 0.0) < 1.0):
        fails.append(f"V5: the query-weight record is {q}")
    if not all(x == x and abs(x) != float("inf") for x in (r.get("final") or {}).values()):
        fails.append("V5: non-finite validation loss")
    return {"query_weight": q, "band_steps": bands, "final_val": r.get("final"),
            "final_val_e12_final": r12["final"], "failures": fails}


def v7(files: dict) -> tuple:
    """V7: every URL of a P0 plan in e12_d1p equals the URL at the same position in
    e12_d1's E13 files. Free mode: each position's supplied state from the seed
    (e12_score.state_positions); control: the P0 states' files whole."""
    from uie.suffixes import SuffixTable
    from e8_verify import TABLES
    from e12_score import state_positions
    from e13_verify import PS
    shipped = SuffixTable.load(ROOT / TABLES["suffix_table.json"][0])
    ps = json.loads((ROOT / PS[0]).read_text(encoding="utf-8"))["table"]
    fails, rec = [], {"p0": [0, 0], "p1_changed": [0, 0]}
    for name, (j, urls) in files.items():
        ref = json.loads((E13_D1 / name.replace("e12_d1p", "e12_d1")).read_text(encoding="utf-8"))
        if j["mode"] == "free":
            tables = K.supplied_tables(ps, shipped.counts, j["label"])
            p0 = [k.startswith("P0") for k in state_positions(tables, len(urls), j["seed"])]
        else:
            p0 = [j["state"].startswith("P0")] * len(urls)
        for a, b, is_p0 in zip(urls, ref, p0):
            if is_p0:
                rec["p0"][0] += a == b
                rec["p0"][1] += 1
            else:
                rec["p1_changed"][0] += a != b
                rec["p1_changed"][1] += 1
    if rec["p0"][0] != rec["p0"][1] or not rec["p0"][1]:
        fails.append(f"V7: {rec['p0'][1] - rec['p0'][0]} of {rec['p0'][1]} P0-plan URLs differ "
                     f"from e12_d1's -- the decoder touched URLs outside its scope")
    return fails, rec


def step_decode(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    gens = E14D / "decode" / "gen"
    gprov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    fails = code_unchanged(gprov.get("code_commit", "")) + tables_check(gprov)
    if gprov.get("checkpoints") != {"e12_final": sha256(E12_FINAL)}:
        fails.append("V1: e12_d1p did not read e12_final's final.pt")
    e13_gpu = json.loads((E13_D1 / "PROVENANCE.json").read_text(encoding="utf-8"))["environment"]["gpu"]
    if gprov.get("environment", {}).get("gpu") != e13_gpu:
        fails.append(f"V7: GPU {gprov.get('environment', {}).get('gpu')!r}, E13_decode's {e13_gpu!r}")
    f2, files = files_check(gens, gprov, E14.jobs(E14.DECODE_ARMS))
    fails += f2
    f7, rec7 = v7(files) if not f2 else ([], {})
    fails += f7
    ident = {}
    if not args.skip_v2:
        model, tok, _ = load_checkpoint(E12_FINAL, device=torch.device("cpu"))
        model.eval()
        v2(model, tok, files, "e12_d1p", fails, ident)
    return {"identity": ident, "v7": rec7, "failures": fails}


def step_generate(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    gens, train = E14D / "generate" / "gen", E14D / "train"
    gprov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    fails = code_unchanged(gprov.get("code_commit", "")) + tables_check(gprov)
    ck = train / "e14_q" / "final.pt"
    if gprov.get("checkpoints") != {"e14_q": sha256(ck)}:
        fails.append("V1: generation did not read e14_q's verified final.pt")
    if json.loads((E14D / "VERIFY_TRAIN.json").read_text(encoding="utf-8")).get("verdict") != "PASS":
        fails.append("V1: VERIFY_TRAIN has not passed")
    f2, files = files_check(gens, gprov, E14.jobs(E14.GENERATE_ARMS))
    fails += f2
    ident = {}
    if not args.skip_v2:
        model, tok, _ = load_checkpoint(ck, device=torch.device("cpu"))
        model.eval()
        for arm in E14.GENERATE_ARMS:
            v2(model, tok, {k: v for k, v in files.items() if f"_{arm}_" in k}, arm, fails, ident)
    return {"identity": ident, "failures": fails}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("train", "decode", "generate"))
    ap.add_argument("--skip-v2", action="store_true", help="dry runs only")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rep = {"train": step_train, "decode": step_decode, "generate": step_generate}[args.step](args)
    rep.update({"registration": E14.REGISTRATION, "step": args.step,
                "verdict": "PASS" if not rep["failures"] else "FAIL"})
    E14D.mkdir(parents=True, exist_ok=True)
    (E14D / f"VERIFY_{args.step.upper()}.json").write_text(json.dumps(rep, indent=1, default=float),
                                                          encoding="utf-8")
    print("failures:", rep["failures"] or "none")
    print(f"VERIFY ({args.step}):", rep["verdict"])


if __name__ == "__main__":
    main()
