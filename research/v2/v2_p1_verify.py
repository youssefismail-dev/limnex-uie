"""
research/v2/v2_p1_verify.py

V2-P1 -- validity (research/v2/PREREG_V2_P1_MEMORY.md section 7), before anything is read.

    python research/v2/v2_p1_verify.py --step train      # after V2P1_train:    V1, V-iso, V-recipe,
                                                         #   V-start, V-batch, V-stream, V-mass,
                                                         #   K12e, the unit tests (V-equiv)
    python research/v2/v2_p1_verify.py --step generate   # after V2P1_generate: V1, V-gen, V-rules

Layout (the Kaggle outputs, extracted):
    data/v2/p1/train          V2P1_train:    p1_mem/final.pt, p1_mem/train_result.json,
                                             PROVENANCE.json
    data/v2/p1/generate/gen   V2P1_generate: the 50 files + PROVENANCE.json
Writes data/v2/p1/VERIFY_<STEP>.json.

V-gen regenerates leading suffix groups of four files (two per mode) on this CPU with
e16_b5's decode and compares bytes, as E12-E17 did. Its bar is the registered one: at least
1,000 URLs compared and at least 98% identical. A memory layer makes a discrete choice of
slots at every position, and a floating-point near-tie at the edge of that choice can fall
differently on the GPU and the CPU (section 7). Every mismatch is listed with the position
of its first differing character.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e17", ROOT / "research" / "e16", ROOT / "research" / "e15",
          ROOT / "research" / "e14", ROOT / "research" / "e13", ROOT / "research" / "native",
          ROOT / "research" / "cr1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as K  # noqa: E402
import e15_common as E15  # noqa: E402
from v2_arm import P1  # noqa: E402  (V2_ARM: p1 or p1b)
from v2_arm import CODE as ARM_CODE, ISO, TESTS as ARM_TESTS  # noqa: E402
from e8_verify import TABLES, git, placement, sha256  # noqa: E402
from e13_verify import files_check, tables_check  # noqa: E402

from v2_arm import DATA as P1D  # noqa: E402
TRAIN = P1D / "train"
GEN = P1D / "generate" / "gen"
E16_CAP = ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt"
STREAM = ROOT / "data" / "e15" / "qry_extra" / P1.STREAM_FILE
STEPS = 20_000
CODE = ["uie", "research/native/e8_common.py", "research/native/e11_common.py",
        "research/native/e12_common.py", "research/e13/e13_common.py", "research/e15/e15_common.py",
        "research/e16/e16_common.py", "research/e17/e17_common.py"] + ARM_CODE
PRINTS = ("e10_start_sha256", "e11_start_sha256", "new_token_rows_sha256", "new_position_rows_sha256")
TESTS = ARM_TESTS
#: V-gen's files: both labels in free mode; in control, two Q = 1 states, so B5 acts in them
V_GEN_TARGETS = {"free": [f"free_{P1.ARM}_benign_g0.json", f"free_{P1.ARM}_phishing_g0.json"],
                 "control": [f"ctrl_{P1.ARM}_benign_P1Q1_g0.json", f"ctrl_{P1.ARM}_benign_P0Q1_g0.json"]}
V_GEN_PER_MODE = 500                 # 1,000 in all
V_GEN_MIN_SHARE = 0.98
K12E_MAX = 0.01


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def code_unchanged(commit: str) -> list:
    out = []
    if not commit or subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                    cwd=ROOT).returncode:
        return [f"V1: code commit {commit!r} is not in this history"]
    changed = git("diff", "--name-only", commit, "HEAD", "--", *CODE)
    if changed:
        out.append(f"V1: code changed since the GPU ran it: {changed.split()}")
    return out


def v_iso() -> list:
    """V-iso: uie/ byte-identical to the registration commit, committed and in the tree."""
    out = []
    if not ISO or git("diff", "--name-only", ISO, "HEAD", "--", "uie"):
        out.append(f"V-iso: uie/ differs from {ISO}")
    if git("status", "--porcelain", "--", "uie"):
        out.append("V-iso: uie/ has uncommitted changes")
    return out


def tests() -> tuple:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return ([] if r.returncode == 0 else [f"tests: {' '.join(TESTS)} failed"]), f"exit {r.returncode}: {tail}"


def norm(x):
    return json.loads(json.dumps(x))


def step_train(args) -> dict:
    import torch
    prov = load(TRAIN / "PROVENANCE.json")
    fails = code_unchanged(prov.get("code_commit", "")) + v_iso()
    arm = prov.get("arms", {}).get(P1.ARM, {})
    ck = TRAIN / P1.ARM / "final.pt"
    # V1
    if not ck.is_file() or arm.get("final_sha256") != sha256(ck):
        fails.append("V1: p1_mem final.pt differs from the checkpoint training recorded")
    if sha256(E16_CAP) != P1.E16_CAP_SHA256:
        fails.append("V1: the laptop's e16_cap final.pt is not the adopted one")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    base = torch.load(E16_CAP, map_location="cpu", weights_only=False)
    tc, mc, tk = norm(blob["train_config"]), norm(blob["model_config"]), norm(blob["tokenizer"])
    btc, bmc = norm(base["train_config"]), norm(base["model_config"])
    raw_tok, base_tok = blob["tokenizer"], norm(base["tokenizer"])
    del blob, base                     # about 0.7 GB: freed before the generation check loads the model
    import gc
    gc.collect()
    r = load(TRAIN / P1.ARM / "train_result.json")
    accum = (prov.get("memory") or {}).get("accum_steps")
    # V-recipe: e16_cap's in every respect but the memory (and the parts, if the check fell back)
    for k, v in btc.items():
        if k == "qry_extra":
            got, want = Path(tc.get(k, "")).name, P1.STREAM_FILE
        elif k == "accum_steps":
            got, want = tc.get(k), accum
        else:
            got, want = tc.get(k), v
        if got != want:
            fails.append(f"V-recipe: train_config {k} = {got!r}, e16_cap's {want!r}")
    if sorted(set(tc) - set(btc)):
        fails.append(f"V-recipe: extra train settings {sorted(set(tc) - set(btc))}")
    if accum not in (P1.ACCUM_STEPS, P1.ACCUM_FALLBACK) or r.get("accum_steps") != accum:
        fails.append(f"V-recipe: accum_steps {r.get('accum_steps')}, the memory check chose {accum}")
    for k, v in bmc.items():
        if mc.get(k) != v:
            fails.append(f"V-recipe: model_config {k} = {mc.get(k)}, e16_cap's {v}")
    if {k: mc.get(k) for k in P1.MEM} != P1.MEM:
        fails.append(f"V-recipe: memory {({k: mc.get(k) for k in P1.MEM})}, registered {P1.MEM}")
    from uie.tokenizer import URLTokenizer
    tok = URLTokenizer.from_state_dict(raw_tok)
    if mc.get("mem_pad_id") != tok.pad_id:
        fails.append(f"V-recipe: the memory skips id {mc.get('mem_pad_id')}, padding is {tok.pad_id}")
    extra = sorted(set(mc) - set(bmc) - set(P1.MEM) - {"mem_pad_id"})
    if extra:
        fails.append(f"V-recipe: extra model settings {extra}")
    if tk != base_tok:
        fails.append("V-recipe: p1_mem's tokenizer is not e16_cap's")
    if r.get("params") != P1.PARAMS:                  # model.parameters(): tied weights once
        fails.append(f"V-recipe: {r.get('params')} parameters, registered {P1.PARAMS}")
    if r.get("counts") != P1.COUNTS or r.get("dropped_over_max_len") != P1.DROPPED:
        fails.append(f"V1: rows {r.get('counts')} / dropped {r.get('dropped_over_max_len')}")
    if r.get("loss_constant") != P1.LOSS_CONSTANT:
        fails.append(f"V1: C = {r.get('loss_constant')}")
    if not all(isinstance(x, float) and math.isfinite(x) for x in (r.get("final") or {}).values()):
        fails.append("V1: non-finite validation loss")
    # V-start
    sc = prov.get("start_check") or {}
    w384, w512, pm = sc.get("width_384") or {}, sc.get("width_512") or {}, sc.get(P1.ARM) or {}
    want384 = {"init_weights_sha256": E15.E12_FINAL_INIT, **E15.E12_FINAL_START}
    if any(w384.get(k) != v for k, v in want384.items()):
        fails.append("V-start: the notebook's width-384 rebuild is not e12_final's recorded start")
    if any(w512.get(k) != v for k, v in P1.E16_START.items()):
        fails.append("V-start: the notebook's width-512 start is not e16_cap's recorded start")
    if pm.get("base_init_sha256") != P1.E16_START["init_weights_sha256"] or pm.get("params") != P1.PARAMS \
            or pm.get("memory_seed") != P1.P1_SEED:
        fails.append("V-start: the notebook's p1_mem start is not e16_cap's start plus the registered memory")
    s = r.get("e12_start") or {}
    if s.get("base_init_sha256") != P1.E16_START["init_weights_sha256"]:
        fails.append("V-start: training's e16_cap part is not e16_cap's recorded start")
    if s.get("memory_init_sha256") != pm.get("memory_init_sha256") or s.get("memory_seed") != P1.P1_SEED:
        fails.append("V-start: training's memory start is not the one recorded before training")
    if r.get("init_weights_sha256") != pm.get("init_weights_sha256"):
        fails.append("V-start: training did not start from the start recorded before training")
    fails += [f"V-start: {k} differs from e16_cap's recorded start" for k in PRINTS
              if s.get(k) != P1.E16_START[k]]
    # V-batch, V-stream, V-mass
    q = r.get("qry_extra") or {}
    if q.get("old_batch_digest") != P1.V_BATCH_DIGEST or q.get("steps") != STEPS:
        fails.append(f"V-batch: digest {str(q.get('old_batch_digest'))[:12]} over {q.get('steps')} steps")
    local = hashlib.sha256(STREAM.read_bytes()).hexdigest()
    if not (q.get("source_sha256") == local == prov.get("stream_sha256") == P1.STREAM_SHA256):
        fails.append("V-stream: stream 1's hash differs")
    st = q.get("strata") or {}
    inj = ("old_rows_seen", "appended_rows", "appended_query_positions")
    got_mass = {k: {f: v.get(f) for f in inj} for k, v in st.items()}
    if got_mass != {k: {f: v[f] for f in inj} for k, v in P1.INJECTION.items()}:
        fails.append(f"V-mass: {got_mass}, e16_cap's {P1.INJECTION}")
    for k, n in E15.N_NEW.items():
        if (st.get(k, {}).get("N_new"), st.get(k, {}).get("N_old")) != (n, E15.N_OLD[k]):
            fails.append(f"V-stream: {k} N_new/N_old {st.get(k)}")
    # K12e
    clip = {k: v["clipped_share"] for k, v in (r.get("grad_diagnostics") or {}).items()
            if k.startswith("after_1000")}
    if not clip or any(v > K12E_MAX for v in clip.values()):
        fails.append(f"K12e: clipped after step 1,000 {clip}")
    f_t, test_tail = ([], "skipped") if args.skip_tests else tests()
    fails += f_t
    # the generation check (Status, 2026-10-12): before V2P1_generate runs, the trained
    # checkpoint generates on this CPU with the notebook's own calls -- inference only
    gen_check, f_g = ({}, []) if args.skip_gen_check else generation_check(ck)
    fails += f_g
    # P1b's V-audit (PREREG_V2_P1B.md section 4): both audits on record passed, and the
    # trained final.pt passes it again on this CPU, on real rows
    audit = {}
    if hasattr(P1, "audit"):
        audit = {"kaggle": prov.get("audit_kaggle"), "in_run": (r.get("e12_start") or {}).get("audit_in_run"),
                 "laptop": None if args.skip_gen_check else laptop_audit(ck)}
        for k, v in audit.items():
            if v is not None and not v.get("pass"):
                fails.append(f"V-audit: the {k} audit did not pass")
        if not (audit["kaggle"] and audit["in_run"]):
            fails.append("V-audit: an audit record is missing")
    # reported: fit by field against e16_cap's record, and the train/validation gap
    e16r = load(ROOT / "data" / "e16" / "train" / "e16_cap" / "train_result.json")
    fit = {}
    for label in ("benign", "phishing"):
        a, b = r["field_nll"][label], e16r["field_nll"][label]
        fit[label] = {f: {"val": a["val"][f], "val_e16_cap": b["val"][f], "delta_val": a["val"][f] - b["val"][f],
                          "gap": a["gap"][f], "gap_e16_cap": b["gap"][f]}
                      for f in a["val"] if f in b["val"]}
    return {"accum_steps": accum, "memory_check": prov.get("memory"), "torch": prov.get("torch"),
            "gpu": prov.get("gpu"), "start": pm, "tests": test_tail, "clipped_after_1000": clip,
            "final_val": r.get("final"), "final_val_e16_cap": P1.E16_CAP_FINAL_VAL,
            "train_hours": (r.get("seconds") or 0) / 3600, "wall_minutes": arm.get("wall_minutes"),
            "peak_gb_training": arm.get("peak_gb_training"), "field_fit": fit,
            "generation_check": gen_check, "audit": audit, "failures": fails}


def laptop_audit(ck: Path) -> dict:
    """P1b's V-audit on the trained final.pt (inference only): real page-frame rows of
    P1B.AUDIT_LENGTHS from corpus2's structure_val, against the registered bar."""
    import gzip
    import torch
    import v2_mem as M
    model, tok, _ = M.load_checkpoint(ck, device=torch.device("cpu"))
    with gzip.open(ROOT / "data" / "corpus2" / "structure_val.jsonl.gz", "rt", encoding="utf-8") as fh:
        urls = [json.loads(next(fh))["url"] for _ in range(400)]
    return P1.audit(model, P1.audit_rows(tok, urls), tok.pad_id)


def generation_check(ck: Path) -> tuple:
    """Three URLs per call, both labels in free mode and benign P1Q1 in control, with the
    notebook's own calls (v2_p1_common.free_urls / control_urls, e16_b5's decode) on this
    CPU. Each must parse."""
    import torch
    import v2_mem as M
    from uie.controlled import StateSuffixTables
    from uie.evaluation import FIELDS
    from uie.suffixes import SuffixTable
    shipped = SuffixTable.load(ROOT / TABLES["suffix_table.json"][0])
    st = StateSuffixTables.load(ROOT / TABLES["state_suffix_tables.json"][0])
    ps = load(ROOT / "data" / "e12" / "presence_slash_table.json")["table"]
    model, tok, _ = M.load_checkpoint(ck, device=torch.device("cpu"))
    model.eval()
    out = {f"free {label}": P1.free_urls(model, tok, label, 0, shipped, ps, n=3) for label in K.LABELS}
    out["control benign P1Q1"] = P1.control_urls(model, tok, "benign", "P1Q1", 0, st, ps, n=3)
    fails = [f"generation check: {k} gave {len(v)} URLs, or one that does not parse"
             for k, v in out.items() if len(v) != 3 or not all(FIELDS["FULL"](u) for u in v)]
    return out, fails


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #

def first_difference(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def drawn_identity(model, tok, tables, urls, label: str, seed: int, budget: int) -> tuple:
    """E13's V2 (e13_verify.drawn_identity), unchanged in what it regenerates, with every
    mismatch kept: (same, compared, mismatches)."""
    import torch
    from uie.model import generate
    from uie.suffixes import SuffixTable
    n = len(urls)
    states = list(tables)
    rng = random.Random(seed)
    counts = collections.Counter(rng.choices(states, weights=[sum(tables[k].values()) for k in states], k=n))
    perm = list(range(n))
    rng.shuffle(perm)
    outer = {orig: pos for pos, orig in enumerate(perm)}
    same = compared = block = 0
    mismatches = []
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
            got = generate(model, tok, label=label, n=count, suffix=suffix, seed=s + j, frame=frame,
                           batch_size=32, device=torch.device("cpu"), fields=K.fields_of(k),
                           **P1.decode_kwargs())
            for rank, u in enumerate(got):
                want = urls[outer[block + inner[start + rank]]]
                same += want == u
                compared += 1
                if want != u:
                    mismatches.append({"state": k, "at": first_difference(want, u), "len": len(want),
                                       "gpu": want, "cpu": u})
            start += count
        block += nk
        if compared >= budget:
            break
    return same, compared, mismatches


def v_gen(model, tok, files: dict) -> tuple:
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    shipped = SuffixTable.load(ROOT / TABLES["suffix_table.json"][0])
    st = StateSuffixTables.load(ROOT / TABLES["state_suffix_tables.json"][0])
    ps = load(ROOT / "data" / "e12" / "presence_slash_table.json")["table"]
    out, total, same, mism = {}, 0, 0, []
    for mode, targets in V_GEN_TARGETS.items():
        per = -(-V_GEN_PER_MODE // len(targets))
        for f in targets:
            j, urls = files[f]
            tables = (K.supplied_tables(ps, shipped.counts, j["label"]) if j["mode"] == "free"
                      else K.control_tables(ps, st, j["label"], j["state"]))
            s, c, m = drawn_identity(model, tok, tables, urls, j["label"], j["seed"], per)
            out[f] = {"compared": c, "identical": s}
            total, same = total + c, same + s
            mism += [{"file": f, **x} for x in m]
            print(f"  V-gen {f:<46} {s}/{c}", flush=True)
    fails = []
    if total < 2 * V_GEN_PER_MODE or same < V_GEN_MIN_SHARE * total:
        fails.append(f"V-gen: {same}/{total} identical (bar: >= {2 * V_GEN_PER_MODE} compared, "
                     f">= {V_GEN_MIN_SHARE:.0%} identical)")
    return {"per_file": out, "compared": total, "identical": same,
            "share": same / total if total else None, "mismatches": mism}, fails


def v_rules() -> tuple:
    """V-rules (B5): no URL under a Q = 1 plan ends, or starts a fragment, before its query
    (E17's K17-B2 clause as amended, from the files; cut-offs at max_len counted apart)."""
    import e13_compare as C13
    import e17_compare as C17
    C13.DIRS[P1.ARM] = GEN
    qh = C17.q_honouring(P1.ARM)
    fails = [] if qh["total"]["ended_without"] == 0 else [f"V-rules: {qh['total']['ended_without']} Q = 1 "
                                                           "URLs ended before their query"]
    return {"total": qh["total"]}, fails


def step_generate(args) -> dict:
    import torch
    import v2_mem as M
    gprov = load(GEN / "PROVENANCE.json")
    fails = code_unchanged(gprov.get("code_commit", "")) + tables_check(gprov) + v_iso()
    ck = TRAIN / P1.ARM / "final.pt"
    if gprov.get("checkpoints") != {P1.ARM: sha256(ck)}:
        fails.append("V1: generation did not read p1_mem's verified final.pt")
    if load(P1D / "VERIFY_TRAIN.json").get("verdict") != "PASS":
        fails.append("V1: VERIFY_TRAIN has not passed")
    if not load(P1D / "k_used.json").get("pass"):
        fails.append("V1: K-used has not passed (section 5.3): generation should not have run")
    if gprov.get("decode") != P1.decode_kwargs():
        fails.append(f"V1: decode {gprov.get('decode')}, registered {P1.decode_kwargs()}")
    f2, files = files_check(GEN, gprov, P1.jobs())
    fails += f2
    gen = {}
    if not args.skip_v_gen and not f2:
        model, tok, _ = M.load_checkpoint(ck, device=torch.device("cpu"))
        model.eval()
        gen, f3 = v_gen(model, tok, files)
        fails += f3
    rules, f4 = v_rules()
    fails += f4
    return {"v_gen": gen, "v_rules": rules, "environment": gprov.get("environment"), "failures": fails}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("train", "generate"))
    ap.add_argument("--skip-v-gen", action="store_true", help="dry runs only")
    ap.add_argument("--skip-tests", action="store_true", help="dry runs only")
    ap.add_argument("--skip-gen-check", action="store_true", help="dry runs only")
    ap.add_argument("--train-dir", type=Path, help="dry runs only: read another training output")
    ap.add_argument("--gen-dir", type=Path, help="dry runs only: read another generation output")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    global TRAIN, GEN, P1D
    if args.train_dir or args.gen_dir:
        TRAIN, GEN = args.train_dir or TRAIN, args.gen_dir or GEN
        P1D = P1D / "dry"
    rep = {"train": step_train, "generate": step_generate}[args.step](args)
    rep.update({"registration": P1.REGISTRATION, "step": args.step,
                "verdict": "PASS" if not rep["failures"] else "FAIL"})
    P1D.mkdir(parents=True, exist_ok=True)
    (P1D / f"VERIFY_{args.step.upper()}.json").write_text(json.dumps(rep, indent=1, default=float),
                                                          encoding="utf-8")
    if args.step == "train":
        print(f"accumulation: {rep['accum_steps']} parts; memory check {rep['memory_check']}; "
              f"torch {rep['torch']} on {rep['gpu']}")
        print(f"trained in {rep['train_hours']:.2f} h (e16_cap 2.73 h); peak {rep['peak_gb_training']} GB")
        print(f"clipped after step 1,000: {rep['clipped_after_1000']}")
        print(f"final val: {rep['final_val']}  (e16_cap {rep['final_val_e16_cap']})")
        print(f"tests: {rep['tests']}")
        for k, v in (rep.get("generation_check") or {}).items():
            print(f"generation check, {k}: {v}")
    else:
        g = rep.get("v_gen") or {}
        if g:
            print(f"V-gen: {g['identical']}/{g['compared']} identical ({g['share']:.2%}); "
                  f"first divergences at {[m['at'] for m in g['mismatches']]}")
        print(f"V-rules: {rep['v_rules']['total']}")
    print("failures:", rep["failures"] or "none")
    print(f"VERIFY ({args.step}):", rep["verdict"])


if __name__ == "__main__":
    main()
