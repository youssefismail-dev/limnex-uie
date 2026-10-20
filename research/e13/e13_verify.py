"""E13 -- validity (research/e13/PREREG_E13.md section 5), before anything is read.

    python research/e13/e13_verify.py --step train      # after E13_train: V1, V-id, V5, V-N
    python research/e13/e13_verify.py --step decode     # after E13_decode: V1, V2
    python research/e13/e13_verify.py --step generate   # after E13_generate: V1, V2

Layout (the Kaggle outputs, extracted):
    data/e13/train       E13_train:    e13_ctx/, n_A1/, n_A2/, PROVENANCE.json
    data/e13/decode/gen  E13_decode:   the decoder arms' and N's reference files + PROVENANCE.json
    data/e13/ctx/gen     E13_generate: e13_ctx's 50 files + PROVENANCE.json
(the `<base>/gen` layout E12's scorer reads, research/native/e12_score.py)
Writes data/e13/VERIFY_<STEP>.json. V2 regenerates leading suffix groups on this CPU
with each file's own decode arguments (D1 and D2 included) and compares bytes
(E12's method, research/native/e12_verify.py).
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
for p in (ROOT, HERE, ROOT / "research" / "native", ROOT / "research" / "cr1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as K  # noqa: E402
import e13_common as E13  # noqa: E402
import e13n_train as N  # noqa: E402
from e8_verify import IDENTITY_MIN, TABLES, git, placement, sha256  # noqa: E402

E13D = ROOT / "data" / "e13"
PS = ("data/e12/presence_slash_table.json",
      "5a0eeac991b8343267d6f88f4c590bd6c208ace975dbf2218f8c5a277b71c050")
E12_FINAL = ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt"
CODE = ["uie", "research/native/e8_common.py", "research/native/e11_common.py",
        "research/native/e12_common.py", "research/e13/e13_common.py", "research/e13/e13n_train.py"]


def code_unchanged(commit: str) -> list:
    out = []
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT).returncode:
        out.append(f"V1: code commit {commit!r} is not in this history")
    changed = git("diff", "--name-only", commit, "HEAD", "--", *CODE)
    if changed:
        out.append(f"V1/V3: code changed since the GPU ran it: {changed.split()}")
    return out


def drawn_identity(model, tok, tables, urls, label: str, seed: int, budget: int, kwargs_for):
    """V2 for a supplied-plan file (e13_common.draw_and_generate): the state counts
    and the outer shuffle depend only on the seed and n; within a state, E8's
    placement locates each suffix group. Regenerates leading groups with the
    state's own decode arguments until `budget` URLs are compared."""
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
                           fields=K.fields_of(k), **kwargs_for(k))
            for rank, u in enumerate(got):
                same += urls[outer[block + inner[start + rank]]] == u
                compared += 1
            start += count
        block += nk
        if compared >= budget:
            break
    return same, compared


def files_check(gens: Path, gprov: dict, jobs: list) -> tuple:
    fails, files = [], {}
    names = {j["file"] for j in jobs}
    extra = sorted(p.name for p in gens.glob("*.json") if p.name != "PROVENANCE.json"
                   and p.name not in names)
    if extra:
        fails.append(f"V1: unregistered files {extra}")
    for j in jobs:
        p, rec = gens / j["file"], gprov["files"].get(j["file"]) or {}
        if not p.is_file():
            fails.append(f"V1: {j['file']} missing")
            continue
        urls = json.loads(p.read_text(encoding="utf-8"))
        if len(urls) != j["n"]:
            fails.append(f"V1: {j['file']} has {len(urls)} urls, registered {j['n']}")
        if rec.get("sha256") != sha256(p) or rec.get("seed") != j["seed"]:
            fails.append(f"V1: {j['file']} sha256 or seed differs from PROVENANCE")
        files[j["file"]] = (j, urls)
    return fails, files


def tables_check(gprov: dict) -> list:
    fails = []
    for name, (rel, want) in {**TABLES, "presence_slash_table.json": PS}.items():
        if sha256(ROOT / rel) != want:
            fails.append(f"V1: {rel} is not the registered {name}")
        if gprov.get("tables", {}).get(name) != want:
            fails.append(f"V1: generation used another {name}")
    return fails


def identity_for(model, tok, files: dict, targets: list, kwargs_arm, budget=IDENTITY_MIN,
                 settings=None):
    """V2 over the target files of one arm and mode."""
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    shipped = SuffixTable.load(ROOT / TABLES["suffix_table.json"][0])
    st = StateSuffixTables.load(ROOT / TABLES["state_suffix_tables.json"][0])
    ps = json.loads((ROOT / PS[0]).read_text(encoding="utf-8"))["table"]
    same = compared = 0
    per = -(-budget // len(targets))
    for f in targets:
        if f not in files:
            continue
        j, urls = files[f]
        if "cell" in j:
            tables = E13.n_tables(j["cell"], shipped, ps, st)
            kw = lambda k: dict(K.decode())                       # noqa: E731
        elif j["mode"] == "free":
            tables = K.supplied_tables(ps, shipped.counts, j["label"])
            kw = lambda k, j=j: E13.state_kwargs(kwargs_arm, j["label"], k, settings)  # noqa: E731
        else:
            tables = K.control_tables(ps, st, j["label"], j["state"])
            kw = lambda k, j=j: E13.state_kwargs(kwargs_arm, j["label"], k, settings)  # noqa: E731
        s, c = drawn_identity(model, tok, tables, urls, j["label"], j["seed"], per, kw)
        same, compared = same + s, compared + c
        print(f"  V2 identity {f:<46} {s}/{c}", flush=True)
    return same, compared


def step_train(args) -> dict:
    import torch
    train = E13D / "train"
    tprov = json.loads((train / "PROVENANCE.json").read_text(encoding="utf-8"))
    fails = code_unchanged(tprov.get("code_commit", ""))
    ck = train / "e13_ctx" / "final.pt"
    if not ck.is_file() or tprov["arms"].get("e13_ctx", {}).get("final_sha256") != sha256(ck):
        fails.append("V1: e13_ctx final.pt differs from the checkpoint training recorded")
    if tprov.get("prefixes_sha256") != json.loads(
            (E13D / "registered_hashes.json").read_text(encoding="utf-8"))[E13.PREFIX_FILE]:
        fails.append("V1: training used another prefix artifact")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    tc, mc, tk = blob["train_config"], blob["model_config"], blob["tokenizer"]
    e12 = torch.load(E12_FINAL, map_location="cpu", weights_only=False)
    r = json.loads((train / "e13_ctx" / "train_result.json").read_text(encoding="utf-8"))
    s, cs = r.get("e12_start") or {}, r.get("context_swap") or {}
    for k, v in e12["train_config"].items():
        if tc.get(k) != v:
            fails.append(f"V-id: e13_ctx train_config {k} = {tc.get(k)}, e12_final's {v}")
    for k, v in e12["model_config"].items():
        if mc.get(k) != v:
            fails.append(f"V-id: e13_ctx model_config {k} = {mc.get(k)}, e12_final's {v}")
    if tk != e12["tokenizer"]:
        fails.append("V-id: e13_ctx's tokenizer is not e12_final's")
    if (tc.get("context_p"), tc.get("context_seed")) != (E13.CONTEXT_P, E13.CONTEXT_SEED):
        fails.append("V-id: e13_ctx's ContextSwap settings are not the registered ones")
    if r.get("init_weights_sha256") != E13.E12_FINAL_INIT:
        fails.append("V-id: e13_ctx did not start from e12_final's start")
    fails += [f"V-id: start {k} differs" for k, v in E13.E12_FINAL_START.items() if s.get(k) != v]
    drawn = cs.get("eligible_draws") or 0
    share = ((cs.get("swapped") or 0) + (cs.get("too_long") or 0)) / drawn if drawn else float("nan")
    band = 3 * (0.25 / drawn) ** 0.5 if drawn else float("nan")
    if cs.get("matched_rows") != cs.get("listed_rows"):
        fails.append(f"V5: matched {cs.get('matched_rows')} of {cs.get('listed_rows')} listed rows")
    if not abs(share - E13.CONTEXT_P) <= band:
        fails.append(f"V5: swap share {share:.4f}, registered {E13.CONTEXT_P} +- {band:.4f}")
    # V-N
    for arm in ("A1", "A2"):
        d = train / f"n_{arm}"
        rec = json.loads((d / "N_RECORD.json").read_text(encoding="utf-8"))
        if sha256(d / "final.pt") != rec["final_sha256"]:
            fails.append(f"V-N {arm}: final.pt differs from its record")
        nb = torch.load(d / "final.pt", map_location="cpu", weights_only=False)
        if nb["tokenizer"] != e12["tokenizer"]:
            fails.append(f"V-N {arm}: the tokenizer is not e12_final's")
        for k, v in e12["train_config"].items():
            want = N.EXPECTED_SCHEDULE[arm][k] if k in N.CHANGED else v
            if nb["train_config"].get(k) != want:
                fails.append(f"V-N {arm}: train_config {k} = {nb['train_config'].get(k)}, registered {want}")
        if rec["kept_query_rows"] != N.EXPECTED_KEPT:
            fails.append(f"V-N {arm}: kept query rows {rec['kept_query_rows']}")
        if not rec["finite"]:
            fails.append(f"K {arm}: non-finite validation loss -- the arm is void")
        jobs = E13.n_jobs((arm,))
        got = rec["generation"]["files"]
        for j in jobs:
            meta = got.get(j["file"], {})
            if (meta.get("seed") != j["seed"] or meta.get("n") != j["n"] or not (d / j["file"]).is_file()
                    or sha256(d / j["file"]) != meta.get("sha256")):
                fails.append(f"V-N {arm}: {j['file']} differs from its record")
    return {"swap_share": share, "band": band, "context_swap": cs, "failures": fails}


def step_decode(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    gens = E13D / "decode" / "gen"
    gprov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    fails = code_unchanged(gprov.get("code_commit", "")) + tables_check(gprov)
    if gprov.get("checkpoints") != {"e12_final": sha256(E12_FINAL)}:
        fails.append("V1: the decoder arms did not read e12_final's final.pt")
    settings = E13.load_settings(E13D / E13.SETTINGS_FILE)
    applied = [s for s, r in settings["scopes"].items() if r.get("applied")]
    arms = ("e12_d1",) + (("e12_dec",) if applied else ())
    jobs = E13.jobs(arms) + E13.n_jobs(("e12_final",))
    f2, files = files_check(gens, gprov, jobs)
    fails += f2
    ident = {}
    if not args.skip_v2:
        model, tok, _ = load_checkpoint(E12_FINAL, device=torch.device("cpu"))
        model.eval()
        for arm in arms:
            for mode, targets in (("free", [f"free_{arm}_benign_g0.json", f"free_{arm}_phishing_g0.json"]),
                                  ("control", [f"ctrl_{arm}_benign_P0Q1_g0.json",
                                               f"ctrl_{arm}_phishing_P0Q0_g0.json"])):
                s, c = identity_for(model, tok, files, targets, arm, settings=settings)
                ident[f"{arm}/{mode}"] = {"compared": c, "identical": s}
                if c < IDENTITY_MIN or s != c:
                    fails.append(f"V2: {arm} {mode} identity {s}/{c}")
        s, c = identity_for(model, tok, files, [f"n_e12_final_{cell}_g0.json" for cell in E13.N_CELLS],
                            "e13_ctx")
        ident["N reference"] = {"compared": c, "identical": s}
        if c < IDENTITY_MIN or s != c:
            fails.append(f"V2: N reference identity {s}/{c}")
    return {"arms": list(arms), "identity": ident, "failures": fails}


def step_generate(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    gens, train = E13D / "ctx" / "gen", E13D / "train"
    gprov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    fails = code_unchanged(gprov.get("code_commit", "")) + tables_check(gprov)
    ck = train / "e13_ctx" / "final.pt"
    if gprov.get("checkpoints") != {"e13_ctx": sha256(ck)}:
        fails.append("V1: generation did not read e13_ctx's verified final.pt")
    f2, files = files_check(gens, gprov, E13.jobs(("e13_ctx",)))
    fails += f2
    ident = {}
    if not args.skip_v2:
        model, tok, _ = load_checkpoint(ck, device=torch.device("cpu"))
        model.eval()
        for mode, targets in (("free", ["free_e13_ctx_benign_g0.json", "free_e13_ctx_phishing_g0.json"]),
                              ("control", ["ctrl_e13_ctx_benign_P1Q1_g0.json",
                                           "ctrl_e13_ctx_phishing_P0Q0_g0.json"])):
            s, c = identity_for(model, tok, files, targets, "e13_ctx")
            ident[f"e13_ctx/{mode}"] = {"compared": c, "identical": s}
            if c < IDENTITY_MIN or s != c:
                fails.append(f"V2: e13_ctx {mode} identity {s}/{c}")
    return {"identity": ident, "failures": fails}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("train", "decode", "generate"))
    ap.add_argument("--skip-v2", action="store_true", help="dry runs only")
    args = ap.parse_args()
    rep = {"train": step_train, "decode": step_decode, "generate": step_generate}[args.step](args)
    rep.update({"registration": E13.REGISTRATION, "step": args.step,
                "verdict": "PASS" if not rep["failures"] else "FAIL"})
    (E13D / f"VERIFY_{args.step.upper()}.json").write_text(json.dumps(rep, indent=1, default=float),
                                                          encoding="utf-8")
    print("failures:", rep["failures"] or "none")
    print(f"VERIFY ({args.step}):", rep["verdict"])


if __name__ == "__main__":
    main()
