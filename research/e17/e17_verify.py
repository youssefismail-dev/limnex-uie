"""E17 -- validity (research/e17/PREREG_E17.md section 5; amendment 1), before anything is
read.

    python research/e17/e17_verify.py --step decode     # after E17_decode:   V1, V2 (e16_b5: B5 on)
    python research/e17/e17_verify.py --step train      # after E17_train:    V1, V-start, V-batch,
                                                        #   V-mass, V-stream, the recipe, V3/V-B5
    python research/e17/e17_verify.py --step generate   # after E17_generate: V1, V2 (e17_a, its decode)

Layout (the Kaggle outputs, extracted):
    data/e17/decode/gen     E17_decode:   e16_b5's 50 files + PROVENANCE.json
    data/e17/train          E17_train:    e17_a/, PROVENANCE.json
    data/e17/generate/gen   E17_generate: e17_a's 50 files + PROVENANCE.json
Writes data/e17/VERIFY_<STEP>.json. V2 regenerates leading suffix groups on this CPU with
the arm's own decode (B5 included where the arm carries it) and compares bytes, as E12-E16
did. The constants fixed after the bundle was built (the pooled stream's hash) live here,
not in e17_common.py, which the GPU ran and which must not change (V1/V3).
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
for p in (ROOT, HERE, ROOT / "research" / "e16", ROOT / "research" / "e15", ROOT / "research" / "e14",
          ROOT / "research" / "e13", ROOT / "research" / "native", ROOT / "research" / "cr1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as K  # noqa: E402
import e15_common as E15  # noqa: E402
import e16_common as E16  # noqa: E402
import e17_common as E17  # noqa: E402
from e8_verify import IDENTITY_MIN, git, sha256  # noqa: E402
from e13_verify import drawn_identity, files_check, tables_check  # noqa: E402

E17D = ROOT / "data" / "e17"
E16_CAP = ROOT / "data" / "e16" / "train" / E17.BASELINE / "final.pt"
STREAM = E17.STREAM2_DIR / E17.STREAM_FILE
#: the pooled stream as built at step 2 (make_e17_bundle.py, commit 04d5c00; V-stream)
STREAM_SHA256 = "128b7d69f029e05554cfc9cfce75545c94a19c4f40f013f398a5d580706b4855"
STREAM_ROWS = 114_584
STEPS = 20_000
CODE = ["uie", "research/native/e8_common.py", "research/native/e11_common.py",
        "research/native/e12_common.py", "research/e13/e13_common.py",
        "research/e15/e15_common.py", "research/e16/e16_common.py", "research/e17/e17_common.py"]
PRINTS = ("e10_start_sha256", "e11_start_sha256", "new_token_rows_sha256", "new_position_rows_sha256")
TESTS = ["tests/test_e15.py", "tests/test_e16.py", "tests/test_e17.py"]
#: V2's files: both labels in free mode; in control, two Q = 1 states, so B5 acts in them
V2_TARGETS = {"free": ["free_{arm}_benign_g0.json", "free_{arm}_phishing_g0.json"],
              "control": ["ctrl_{arm}_benign_P1Q1_g0.json", "ctrl_{arm}_benign_P0Q1_g0.json"]}


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def code_unchanged(commit: str) -> list:
    out = []
    if not commit or subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                    cwd=ROOT).returncode:
        return [f"V1: code commit {commit!r} is not in this history"]
    changed = git("diff", "--name-only", commit, "HEAD", "--", *CODE)
    if changed:
        out.append(f"V1/V3: code changed since the GPU ran it: {changed.split()}")
    return out


def v3_tests() -> tuple:
    """V3 and V-B5: the unit tests at this commit (options off bit for bit, B5, rate 2)."""
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return ([] if r.returncode == 0 else [f"V3/V-B5: {' '.join(TESTS)} failed"]), \
        f"exit {r.returncode}: {tail}"


def identity(model, tok, files: dict, arm: str, b5: bool) -> tuple:
    """V2 for one arm, both modes, with the arm's own decode."""
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    from e8_verify import TABLES
    from e13_verify import PS
    shipped = SuffixTable.load(ROOT / TABLES["suffix_table.json"][0])
    st = StateSuffixTables.load(ROOT / TABLES["state_suffix_tables.json"][0])
    ps = json.loads((ROOT / PS[0]).read_text(encoding="utf-8"))["table"]
    out, fails = {}, []
    for mode, names in V2_TARGETS.items():
        targets = [n.format(arm=arm) for n in names]
        per = -(-IDENTITY_MIN // len(targets))
        same = compared = 0
        for f in targets:
            if f not in files:
                continue
            j, urls = files[f]
            tables = (K.supplied_tables(ps, shipped.counts, j["label"]) if j["mode"] == "free"
                      else K.control_tables(ps, st, j["label"], j["state"]))
            s, c = drawn_identity(model, tok, tables, urls, j["label"], j["seed"], per,
                                  lambda k: E17.decode_kwargs(b5))
            same, compared = same + s, compared + c
            print(f"  V2 identity {f:<46} {s}/{c}", flush=True)
        out[f"{arm}/{mode}"] = {"compared": compared, "identical": same}
        if compared < IDENTITY_MIN or same != compared:
            fails.append(f"V2: {arm} {mode} identity {same}/{compared}")
    return out, fails


def b5_trace(gprov: dict) -> dict:
    """What B5 did, from the notebook's per-file record (reported), per mode. `acted_on`:
    the Q = 1 rows (every one has some end mass moved, however small). `moved_total`: the
    end mass moved, summed over rows -- about the number of rows whose query the rule
    started (the files' own count, against e16_cap's, is in the comparison)."""
    out = {}
    for name, r in gprov["files"].items():
        node = out.setdefault(r["mode"], {"files": 0, "rows": 0, "acted_on": 0, "moved_total": 0.0})
        node["files"] += 1
        node["rows"] += r["n"]
        node["acted_on"] += r.get("query_rows_moved") or 0
        node["moved_total"] += (r.get("query_moved_mean") or 0.0) * r["n"]
    return out


def step_decode(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    gens = E17D / "decode" / "gen"
    gprov = load(gens / "PROVENANCE.json")
    fails = code_unchanged(gprov.get("code_commit", "")) + tables_check(gprov)
    if sha256(E16_CAP) != E17.E16_CAP_SHA256:
        fails.append("V1: the laptop's e16_cap final.pt is not the adopted baseline")
    if gprov.get("checkpoints") != {E17.BASELINE: E17.E16_CAP_SHA256}:
        fails.append("V1: the decode arm did not read e16_cap's final.pt")
    if gprov.get("decode") != E17.decode_kwargs(True):
        fails.append(f"V1: decode {gprov.get('decode')}, registered {E17.decode_kwargs(True)}")
    f2, files = files_check(gens, gprov, E17.jobs(E17.DECODE_ARM))
    fails += f2
    ident = {}
    if not args.skip_v2:
        model, tok, _ = load_checkpoint(E16_CAP, device=torch.device("cpu"))
        model.eval()
        ident, f3 = identity(model, tok, files, E17.DECODE_ARM, True)
        fails += f3
    return {"identity": ident, "b5_trace": b5_trace(gprov), "environment": gprov.get("environment"),
            "failures": fails}


def stream_lengths(tok) -> dict:
    """Per stratum, each pooled row's query positions as the injector counts them
    (QueryInjector.load's encoding and mask; e15_verify.stream_lengths on this stream)."""
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


def v_mass(st: dict, lengths: dict) -> tuple:
    """V-mass (section 5): per stratum, appended rows = 2 x the old rows drawn, exactly, and
    the old rows drawn are e16_cap's (the same batches); added query positions within 3 SE
    of P8 (282.2 per batch). At rate 2 the row counts are exact, so the positions vary
    only through which pooled rows are drawn: SE^2 = sum_s n_s Var L_s (with replacement;
    the injector cycles through a shuffle, so this is conservative)."""
    fails = []
    for s in E17.N_NEW:
        v = st.get(s, {})
        if v.get("old_rows_seen") != E17.OLD_ROWS_SEEN[s]:
            fails.append(f"V-mass: {s} drew {v.get('old_rows_seen')} old rows, e16_cap's {E17.OLD_ROWS_SEEN[s]}")
        if v.get("appended_rows") != E17.APPENDED[s] or v.get("appended_rows") != 2 * (v.get("old_rows_seen") or -1):
            fails.append(f"V-mass: {s} appended {v.get('appended_rows')} rows, registered {E17.APPENDED[s]} "
                         f"(2 x {v.get('old_rows_seen')})")
    pos = sum(v["appended_query_positions"] for v in st.values())
    want = E17.ADDED_QUERY_POSITIONS_PER_BATCH * STEPS
    n_s = {s: st[s]["appended_rows"] for s in E17.N_NEW if s in st}
    se = math.sqrt(sum(n_s[s] * lengths[s]["var"] for s in n_s))
    exact = sum(n_s[s] * lengths[s]["mean"] for s in n_s)
    out = {"appended_rows": {s: st[s]["appended_rows"] for s in n_s}, "old_rows_seen": E17.OLD_ROWS_SEEN,
           "appended_query_positions": pos, "expected_positions_p8": want, "se_positions": se,
           "z_positions": (pos - want) / se,
           "expected_positions_given_draws": exact, "z_given_draws": (pos - exact) / se,
           "per_batch": {"rows": sum(n_s.values()) / STEPS, "positions": pos / STEPS},
           "positions_per_row": {s: st[s]["appended_query_positions"] / st[s]["appended_rows"]
                                 for s in n_s if st[s]["appended_rows"]},
           "stream_L": {s: v["mean"] for s, v in lengths.items()},
           "added_vs_e16_cap_added": pos / STEPS / E16_ADDED_PER_BATCH - 1,
           "query_mass_vs_e16_cap": (E12_FINAL_QUERY_POSITIONS + pos / STEPS)
                                    / (E12_FINAL_QUERY_POSITIONS + E16_ADDED_PER_BATCH) - 1}
    if abs(out["z_positions"]) > 3:
        fails.append(f"V-mass: {pos:,} query positions, P8 {want:,.0f} +- {3 * se:,.0f}")
    return fails, out


#: amendment 1 section 9.3's references: e16_cap's added query positions per batch (141.1)
#: and e12_final's own query positions per batch (749.8); P8's +15.8% is their ratio's change
E16_ADDED_PER_BATCH = 141.1
E12_FINAL_QUERY_POSITIONS = 749.8


def step_train(args) -> dict:
    import torch
    train = E17D / "train"
    prov = load(train / "PROVENANCE.json")
    fails = code_unchanged(prov.get("code_commit", ""))
    arm = prov["arms"].get(E17.TRAIN_ARM, {})
    ck = train / E17.TRAIN_ARM / "final.pt"
    # V1
    if not ck.is_file() or arm.get("final_sha256") != sha256(ck):
        fails.append("V1: e17_a final.pt differs from the checkpoint training recorded")
    if sha256(E16_CAP) != E17.E16_CAP_SHA256:
        fails.append("V1: the laptop's e16_cap final.pt is not the adopted baseline")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    base = torch.load(E16_CAP, map_location="cpu", weights_only=False)
    tc, mc, tk = blob["train_config"], blob["model_config"], blob["tokenizer"]
    btc, bmc = base["train_config"], base["model_config"]
    r = load(train / E17.TRAIN_ARM / "train_result.json")
    accum = (prov.get("memory") or {}).get("accum_steps")
    # the recipe: e16_cap's in every respect but the stream (and the parts, if the memory
    # check fell back)
    for k, v in btc.items():
        if k == "qry_extra":
            got, want = Path(tc.get(k, "")).name, E17.STREAM_FILE
        elif k == "accum_steps":
            got, want = tc.get(k), accum
        else:
            got, want = tc.get(k), v
        if got != want:
            fails.append(f"recipe: train_config {k} = {got!r}, registered {want!r}")
    if sorted(set(tc) - set(btc)):
        fails.append(f"recipe: extra settings {sorted(set(tc) - set(btc))}")
    if accum not in (E16.ACCUM_STEPS, E16.ACCUM_FALLBACK) or r.get("accum_steps") != accum:
        fails.append(f"recipe: accum_steps {r.get('accum_steps')}, the memory check chose {accum}")
    for k, v in bmc.items():
        if mc.get(k) != v:
            fails.append(f"recipe: model_config {k} = {mc.get(k)}, e16_cap's {v}")
    if tk != base["tokenizer"]:
        fails.append("recipe: e17_a's tokenizer is not e16_cap's")
    if r.get("params") != E16.PARAMS:
        fails.append(f"recipe: {r.get('params')} parameters, registered {E16.PARAMS}")
    if r.get("counts") != E16.COUNTS or r.get("dropped_over_max_len") != E16.DROPPED:
        fails.append(f"V1: rows {r.get('counts')} / dropped {r.get('dropped_over_max_len')}")
    if r.get("loss_constant") != E16.LOSS_CONSTANT:
        fails.append(f"V1: C = {r.get('loss_constant')}")
    if not all(isinstance(x, float) and math.isfinite(x) for x in (r.get("final") or {}).values()):
        fails.append("V1: non-finite validation loss")
    # V-start: Kaggle's width-384 rebuild equals e12_final's recorded start; the width-512
    # start equals e16_cap's recorded start; training started from it
    sc = prov.get("start_check") or {}
    w384, w512 = sc.get("width_384") or {}, sc.get("width_512") or {}
    rec = {"init_weights_sha256": E15.E12_FINAL_INIT, **E15.E12_FINAL_START}
    if any(w384.get(k) != v for k, v in rec.items()):
        fails.append("V-start: the notebook's width-384 rebuild is not e12_final's recorded start")
    if any(w512.get(k) != v for k, v in E17.E16_START.items()):
        fails.append("V-start: the notebook's width-512 start is not e16_cap's recorded start")
    if r.get("init_weights_sha256") != E17.E16_START["init_weights_sha256"]:
        fails.append("V-start: training did not start from e16_cap's start")
    s = r.get("e12_start") or {}
    fails += [f"V-start: {k} differs from e16_cap's recorded start" for k in PRINTS
              if s.get(k) != E17.E16_START[k]]
    # V-batch
    q = r.get("qry_extra") or {}
    if q.get("old_batch_digest") != E17.V_BATCH_DIGEST or q.get("steps") != STEPS:
        fails.append(f"V-batch: digest {str(q.get('old_batch_digest'))[:12]} over {q.get('steps')} steps")
    # V-stream
    local = hashlib.sha256(STREAM.read_bytes()).hexdigest()
    if not (q.get("source_sha256") == local == prov.get("stream_sha256") == STREAM_SHA256):
        fails.append("V-stream: the pooled stream's hash differs")
    if sum(1 for _ in STREAM.read_text(encoding="utf-8").splitlines()) != STREAM_ROWS:
        fails.append("V-stream: the pooled stream's row count differs")
    st = q.get("strata") or {}
    if sorted(st) != sorted(E17.N_NEW):
        fails.append(f"V-stream: strata {sorted(st)}, registered {sorted(E17.N_NEW)}")
    for k in E17.N_NEW:
        v = st.get(k, {})
        if (v.get("N_new"), v.get("N_old"), v.get("rate")) != (E17.N_NEW[k], E17.N_OLD[k], E17.RATE):
            fails.append(f"V-stream: {k} N_new {v.get('N_new')}, N_old {v.get('N_old')}, rate {v.get('rate')}")
    # V-mass
    from uie.tokenizer import URLTokenizer
    f_mass, mass = v_mass(st, stream_lengths(URLTokenizer.from_state_dict(tk)))
    fails += f_mass
    f_t, tests = ([], "skipped") if args.skip_tests else v3_tests()
    fails += f_t
    clip = {k: v["clipped_share"] for k, v in (r.get("grad_diagnostics") or {}).items()
            if k.startswith("after_1000")}
    return {"accum_steps": accum, "memory": prov.get("memory"), "torch": prov.get("torch"),
            "gpu": prov.get("gpu"), "start_width_512": w512, "v3_tests": tests, "v_mass": mass,
            "clipped_after_1000": clip, "final_val": r.get("final"), "final_val_e16_cap": E16_CAP_FINAL,
            "wall_minutes": arm.get("wall_minutes"), "failures": fails}


#: e16_cap's training record (data/e16/train/e16_cap/train_result.json), for the report
E16_CAP_FINAL = {"benign": 1.6067, "phishing": 1.3713, "mean": 1.4890}


def step_generate(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    gens, train = E17D / "generate" / "gen", E17D / "train"
    gprov = load(gens / "PROVENANCE.json")
    fails = code_unchanged(gprov.get("code_commit", "")) + tables_check(gprov)
    ck = train / E17.TRAIN_ARM / "final.pt"
    if gprov.get("checkpoints") != {E17.TRAIN_ARM: sha256(ck)}:
        fails.append("V1: generation did not read e17_a's verified final.pt")
    if load(E17D / "VERIFY_TRAIN.json").get("verdict") != "PASS":
        fails.append("V1: VERIFY_TRAIN has not passed")
    b5_path = E17D / "compare_b5.json"
    if not b5_path.is_file():
        raise SystemExit("STOP: B5 has not been read (data/e17/compare_b5.json); e17_a's decode is undecided")
    b5 = bool(load(b5_path).get("adopted"))
    if gprov.get("decode") != E17.decode_kwargs(b5):
        fails.append(f"V1: decode {gprov.get('decode')}; B5's reading leaves {E17.decode_kwargs(b5)}")
    f2, files = files_check(gens, gprov, E17.jobs(E17.TRAIN_ARM))
    fails += f2
    ident = {}
    if not args.skip_v2:
        model, tok, _ = load_checkpoint(ck, device=torch.device("cpu"))
        model.eval()
        ident, f3 = identity(model, tok, files, E17.TRAIN_ARM, b5)
        fails += f3
    return {"b5_adopted": b5, "identity": ident, "b5_trace": b5_trace(gprov) if b5 else None,
            "environment": gprov.get("environment"), "failures": fails}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("decode", "train", "generate"))
    ap.add_argument("--skip-v2", action="store_true", help="dry runs only")
    ap.add_argument("--skip-tests", action="store_true", help="dry runs only")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rep = {"decode": step_decode, "train": step_train, "generate": step_generate}[args.step](args)
    rep.update({"registration": E17.REGISTRATION, "step": args.step,
                "verdict": "PASS" if not rep["failures"] else "FAIL"})
    E17D.mkdir(parents=True, exist_ok=True)
    (E17D / f"VERIFY_{args.step.upper()}.json").write_text(json.dumps(rep, indent=1, default=float),
                                                          encoding="utf-8")
    if args.step == "train":
        m = rep["v_mass"]
        print(f"accumulation: {rep['accum_steps']} parts; memory {rep['memory']}; torch {rep['torch']}")
        print(f"V-mass: appended {m['appended_rows']}; {m['appended_query_positions']:,} query positions "
              f"(P8 {m['expected_positions_p8']:,.0f}, z {m['z_positions']:+.2f}); "
              f"query mass {m['query_mass_vs_e16_cap']:+.1%} over e16_cap's (P8 +15.8%)")
        print(f"clipped after step 1,000: {rep['clipped_after_1000']}")
        print(f"final val: {rep['final_val']}  (e16_cap {rep['final_val_e16_cap']})")
        print(f"V3/V-B5 tests: {rep['v3_tests']}")
    elif rep.get("b5_trace"):
        for mode, t in rep["b5_trace"].items():
            print(f"B5 {mode}: acted on {t['acted_on']:,} of {t['rows']:,} rows (the Q = 1 plans); "
                  f"end mass moved {t['moved_total']:.1f} in all (about the rows whose query it started)")
    print("failures:", rep["failures"] or "none")
    print(f"VERIFY ({args.step}):", rep["verdict"])


if __name__ == "__main__":
    main()
