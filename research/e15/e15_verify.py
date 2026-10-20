"""E15-B -- validity (research/e15/PREREG_E15.md sections 9.10, 10.7, 13.3), before
anything is read.

    python research/e15/e15_verify.py --step train      # after E15B_train:    V1, V-start, V-batch,
                                                        #   V-mask, V-mass, V-stream
    python research/e15/e15_verify.py --step generate   # after E15B_generate: V1, V2

Layout (the Kaggle outputs, extracted):
    data/e15/train         E15B_train:    e15_new/, PROVENANCE.json
    data/e15/generate/gen  E15B_generate: e15_new's 50 files + PROVENANCE.json
(the `<base>/gen` layout E12's scorer reads). Writes data/e15/VERIFY_<STEP>.json.
V2 regenerates leading suffix groups on this CPU with e15_new's decode (raw + D1 + D1')
and compares bytes (E12's, E13's and E14's method).
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
for p in (ROOT, HERE, ROOT / "research" / "e14", ROOT / "research" / "e13",
          ROOT / "research" / "native", ROOT / "research" / "cr1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as K  # noqa: E402
import e15_common as E15  # noqa: E402
from e8_verify import IDENTITY_MIN, git, sha256  # noqa: E402
from e13_verify import drawn_identity, files_check, tables_check  # noqa: E402

E15D = ROOT / "data" / "e15"
E12_FINAL = ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt"
E12_FINAL_RESULT = ROOT / "data" / "e12" / "train" / "e12_final" / "train_result.json"
STREAM = E15D / "qry_extra" / E15.STREAM_FILE
STEPS = 20_000
QRY_KEYS = {"qry_extra", "qry_extra_draw_seed", "qry_extra_rate_seed"}
CODE = ["uie", "research/native/e8_common.py", "research/native/e11_common.py",
        "research/native/e12_common.py", "research/e13/e13_common.py",
        "research/e15/e15_common.py"]


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


def stream_lengths(tok, max_len: int) -> dict:
    """Per stratum, each new row's query positions as the injector counts them
    (QueryInjector.load's encoding and mask): the mean and variance V-mass needs."""
    from uie.train import QueryInjector
    per: dict = {}
    for line in STREAM.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ids = tok.encode(r["url"], r["label"], r.get("lang", "und"), frame="domain")
        s = r["stratum"] if "_" in r["stratum"] else f"{r['label']}_{r['stratum']}"
        per.setdefault(s, []).append(sum(QueryInjector.query_only_mask(ids, tok)))
    out = {}
    for s, xs in per.items():
        m = sum(xs) / len(xs)
        out[s] = {"rows": len(xs), "mean": m, "var": sum((x - m) ** 2 for x in xs) / (len(xs) - 1)}
    return out


def v_mass(rec: dict, lengths: dict) -> tuple:
    """V-mass (sections 10.7, 13.3): appended rows and added query positions within 3 SE
    of amendment 3's expectations (P8: 4.741 rows and 141.1 positions per batch).
    Rows: an appended count per old draw is ~Poisson in the step count, SE = sqrt(n)
    (section 13.3's +-915 at 3 SE). Positions: SE^2 = sum_s n_s (Var L_s + L_s^2)."""
    st = rec["strata"]
    rows = sum(v["appended_rows"] for v in st.values())
    pos = sum(v["appended_query_positions"] for v in st.values())
    want_rows = E15.APPENDED_PER_BATCH * STEPS
    want_pos = E15.ADDED_QUERY_POSITIONS_PER_BATCH * STEPS
    se_rows = math.sqrt(want_rows)
    n_s = {s: want_rows * E15.N_OLD[s] / sum(E15.N_OLD.values()) for s in E15.N_OLD}
    se_pos = math.sqrt(sum(n_s[s] * (lengths[s]["var"] + lengths[s]["mean"] ** 2) for s in n_s))
    out = {"appended_rows": rows, "expected_rows": want_rows, "se_rows": se_rows,
           "z_rows": (rows - want_rows) / se_rows,
           "appended_query_positions": pos, "expected_positions": want_pos, "se_positions": se_pos,
           "z_positions": (pos - want_pos) / se_pos,
           "per_batch": {"rows": rows / STEPS, "positions": pos / STEPS},
           "positions_per_row": {s: v["appended_query_positions"] / v["appended_rows"]
                                 for s, v in st.items() if v["appended_rows"]},
           "stream_L": {s: v["mean"] for s, v in lengths.items()},
           "query_mass_added_vs_e12_final": pos / STEPS / 750.0}
    fails = []
    if abs(out["z_rows"]) > 3:
        fails.append(f"V-mass: {rows:,} rows appended, expected {want_rows:,.0f} +- {3 * se_rows:,.0f}")
    if abs(out["z_positions"]) > 3:
        fails.append(f"V-mass: {pos:,} query positions, expected {want_pos:,.0f} +- {3 * se_pos:,.0f}")
    for s, v in st.items():
        if E15.P_INJECT[s] == 1.0 and v["appended_rows"] != v["old_rows_seen"]:
            fails.append(f"V-mass: {s} appended {v['appended_rows']} rows for {v['old_rows_seen']} old draws at p = 1")
    return fails, out


def v_mask() -> tuple:
    """V-mask: the unit tests (tests/test_e15.py: query-only masks, old rows untouched)
    at this commit; the dry run's record is in PREREG_E15.md's Status."""
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_e15.py"], cwd=ROOT,
                       capture_output=True, text=True)
    tail = f"tests/test_e15.py exit {r.returncode}" + "".join(
        f"; {ln.strip()}" for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln)
    return ([] if r.returncode == 0 else [f"V-mask: tests/test_e15.py failed: {tail}"]), tail


def step_train(args) -> dict:
    import torch
    from uie.tokenizer import URLTokenizer
    train = E15D / "train"
    tprov = json.loads((train / "PROVENANCE.json").read_text(encoding="utf-8"))
    fails = code_unchanged(tprov.get("code_commit", ""))
    arm = tprov["arms"].get(E15.ARM, {})
    ck = train / E15.ARM / "final.pt"
    # V1: the checkpoint is the one training recorded; the reference is E12's
    if not ck.is_file() or arm.get("final_sha256") != sha256(ck):
        fails.append("V1: e15_new final.pt differs from the checkpoint training recorded")
    if sha256(E12_FINAL) != E15.E12_FINAL_SHA256:
        fails.append("V1: the laptop's e12_final.pt is not E12's")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    tc, mc, tk = blob["train_config"], blob["model_config"], blob["tokenizer"]
    e12 = torch.load(E12_FINAL, map_location="cpu", weights_only=False)
    r = json.loads((train / E15.ARM / "train_result.json").read_text(encoding="utf-8"))
    r12 = json.loads(E12_FINAL_RESULT.read_text(encoding="utf-8"))
    # the recipe: e12_final's in every respect but the stream (no query weight)
    for k, v in e12["train_config"].items():
        if tc.get(k) != v:
            fails.append(f"V1: e15_new train_config {k} = {tc.get(k)}, e12_final's {v}")
    extra = set(tc) - set(e12["train_config"])
    if extra != QRY_KEYS or not str(tc.get("qry_extra", "")).endswith(E15.STREAM_FILE) \
            or (tc.get("qry_extra_draw_seed"), tc.get("qry_extra_rate_seed")) != (E15.DRAW_SEED, E15.RATE_SEED):
        fails.append(f"V1: train_config adds {sorted(extra)}: {({k: tc.get(k) for k in extra})}")
    if tc.get("query_weight", 1.0) != 1.0 or r.get("query_weight") is not None:
        fails.append(f"V1: a query weight is set ({tc.get('query_weight')}, {r.get('query_weight')})")
    for k, v in e12["model_config"].items():
        if mc.get(k) != v:
            fails.append(f"V1: e15_new model_config {k} = {mc.get(k)}, e12_final's {v}")
    if tk != e12["tokenizer"]:
        fails.append("V1: e15_new's tokenizer is not e12_final's")
    if r.get("counts") != E15.E12_FINAL_COUNTS or r.get("counts") != r12["counts"]:
        fails.append(f"V1: row counts {r.get('counts')}, e12_final's {r12['counts']}")
    if r.get("dropped_over_max_len") != E15.E12_FINAL_DROPPED:
        fails.append(f"V1: dropped {r.get('dropped_over_max_len')}, e12_final's {E15.E12_FINAL_DROPPED}")
    if r.get("loss_constant") != E15.E12_FINAL_LOSS_CONSTANT:
        fails.append(f"V1: C = {r.get('loss_constant')}, e12_final's {E15.E12_FINAL_LOSS_CONSTANT}")
    if not all(x == x and abs(x) != float("inf") for x in (r.get("final") or {}).values()):
        fails.append("V1: non-finite validation loss")
    # V-start: E12's start (c7e71edf...; the E10 and E11 fingerprints)
    s = r.get("e12_start") or {}
    if r.get("init_weights_sha256") != E15.E12_FINAL_INIT:
        fails.append("V-start: e15_new did not start from E12's start")
    fails += [f"V-start: start {k} differs" for k, v in E15.E12_FINAL_START.items() if s.get(k) != v]
    # V-batch: the old rows' batches are e12_final's
    q = r.get("qry_extra") or {}
    if q.get("old_batch_digest") != E15.V_BATCH_DIGEST or q.get("steps") != STEPS:
        fails.append(f"V-batch: old-row digest {str(q.get('old_batch_digest'))[:12]} over "
                     f"{q.get('steps')} steps, the replay's {E15.V_BATCH_DIGEST[:12]} over {STEPS}")
    if (tprov["arms"][E15.ARM].get("qry_extra") or {}).get("old_batch_digest") != q.get("old_batch_digest"):
        fails.append("V-batch: PROVENANCE and train_result disagree")
    # V-stream: the hash, the strata, the counts and the rate are amendment 3's
    local = hashlib.sha256(STREAM.read_bytes()).hexdigest()
    if not (q.get("source_sha256") == local == tprov.get("stream_sha256")):
        fails.append(f"V-stream: stream sha {str(q.get('source_sha256'))[:12]}, the laptop's {local[:12]}")
    st = q.get("strata") or {}
    if set(st) != set(E15.N_NEW):
        fails.append(f"V-stream: strata {sorted(st)}, amendment 3's {sorted(E15.N_NEW)}")
    for k in E15.N_NEW:
        v = st.get(k, {})
        if (v.get("N_new"), v.get("N_old"), v.get("p")) != (E15.N_NEW[k], E15.N_OLD[k], E15.P_INJECT[k]):
            fails.append(f"V-stream: {k} N_new {v.get('N_new')}, N_old {v.get('N_old')}, p {v.get('p')}")
    # V-mass and V-mask
    tok = URLTokenizer.from_state_dict(tk)
    f_mass, mass = v_mass(q, stream_lengths(tok, mc["max_len"]))
    fails += f_mass
    f_mask, tests = ([], "skipped") if args.skip_tests else v_mask()
    fails += f_mask
    bands = {k: v["steps"] for k, v in (r.get("grad_diagnostics") or {}).items() if k.startswith("all ")}
    bands12 = {k: v["steps"] for k, v in r12["grad_diagnostics"].items() if k.startswith("all ")}
    return {"qry_extra": q, "v_mass": mass, "v_mask_tests": tests,
            "band_steps": bands, "band_steps_e12_final": bands12,
            "final_val": r.get("final"), "final_val_e12_final": r12["final"],
            "wall_minutes": arm.get("wall_minutes"), "failures": fails}


def identity_for(model, tok, files: dict, targets: list, budget=IDENTITY_MIN):
    """V2 over the target files of one mode, with e15_new's decode."""
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
                              lambda k: E15.decode_kwargs())
        same, compared = same + s, compared + c
        print(f"  V2 identity {f:<46} {s}/{c}", flush=True)
    return same, compared


def step_generate(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    gens, train = E15D / "generate" / "gen", E15D / "train"
    gprov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    fails = code_unchanged(gprov.get("code_commit", "")) + tables_check(gprov)
    ck = train / E15.ARM / "final.pt"
    if gprov.get("checkpoints") != {E15.ARM: sha256(ck)}:
        fails.append("V1: generation did not read e15_new's verified final.pt")
    if json.loads((E15D / "VERIFY_TRAIN.json").read_text(encoding="utf-8")).get("verdict") != "PASS":
        fails.append("V1: VERIFY_TRAIN has not passed")
    f2, files = files_check(gens, gprov, E15.jobs())
    fails += f2
    ident = {}
    if not args.skip_v2:
        model, tok, _ = load_checkpoint(ck, device=torch.device("cpu"))
        model.eval()
        arm = E15.ARM
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
    rep.update({"registration": E15.REGISTRATION, "step": args.step,
                "verdict": "PASS" if not rep["failures"] else "FAIL"})
    E15D.mkdir(parents=True, exist_ok=True)
    (E15D / f"VERIFY_{args.step.upper()}.json").write_text(json.dumps(rep, indent=1, default=float),
                                                          encoding="utf-8")
    if args.step == "train":
        m = rep["v_mass"]
        print(f"V-mass: {m['appended_rows']:,} rows (expected {m['expected_rows']:,.0f}, z {m['z_rows']:+.2f}); "
              f"{m['appended_query_positions']:,} query positions (expected {m['expected_positions']:,.0f}, "
              f"z {m['z_positions']:+.2f}); query mass +{100 * m['query_mass_added_vs_e12_final']:.1f}%")
        print(f"V-mask tests: {rep['v_mask_tests']}")
        print(f"final val: {rep['final_val']}  (e12_final {rep['final_val_e12_final']})")
    print("failures:", rep["failures"] or "none")
    print(f"VERIFY ({args.step}):", rep["verdict"])


if __name__ == "__main__":
    main()
