"""
research/v2/v2_p2_verify.py

V2-P2 -- validity (research/v2/PREREG_V2_P2.md section 12), before anything is read.

    python research/v2/v2_p2_verify.py --step gen-equiv               # before any Kaggle run
    V2_ARM=p2_a python research/v2/v2_p2_verify.py --step train      # after V2P2_train_a
    V2_ARM=p2_a python research/v2/v2_p2_verify.py --step generate   # after V2P2_generate_a
    (V2_ARM=p2_b for arm B)

- **gen-equiv (V-gen-equiv):** P2's generator -- `e16_cap`'s weights through the piece tokenizer
  with no tables and the piece rule, `e16_b5`'s decode -- regenerates the start of `e16_b5`'s
  files on this CPU byte for byte (1,000 URLs: both labels free, benign P1Q1 and P0Q1).
- **train (VERIFY_TRAIN):** V1, V-iso, V-recipe (`e16_cap`'s in every setting but the arm's:
  the tables, C, dropout, name scope, vocabulary and, if the check fell back, 4 parts),
  V-start (`caca0107...`, E12's prints, the piece rows as recorded before training), V-batch,
  V-mass, V-stream, the P2 record (V-T3), K12e, the unit tests, and a generation check on this
  CPU with the notebook's own calls. Reported: fit per character against `e16_cap`, and the
  page-row name gap (section 5), the arm's against `e16_cap`'s on the same rows.
- **generate (VERIFY_GENERATE):** V1 (the verified checkpoint, the registered decode, tables and
  rules), the 50 files, V-gen (at least 1,000 URLs regenerated on this CPU, at least 99%
  identical, every mismatch listed), V-rules from the files: S honoured on every plan D1 and
  D1' cover, B5 on every Q = 1 plan (a URL that reached max_len without its end token, as the
  generation recorded, is excused), and no piece outside its fields.

Layout (per arm, the Kaggle outputs extracted):
    data/v2/<arm>/train          V2P2_train_<a|b>:    <arm>/final.pt, <arm>/train_result.json,
                                                     PROVENANCE.json
    data/v2/<arm>/generate/gen   V2P2_generate_<a|b>: the 50 files + PROVENANCE.json
Writes data/v2/<arm>/VERIFY_<STEP>.json; gen-equiv writes data/v2/p2/VERIFY_GEN_EQUIV.json.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import random
import subprocess
import sys
import time
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
import v2_p2_common as C  # noqa: E402
from e8_verify import TABLES, git, placement, sha256  # noqa: E402
from e13_verify import files_check, tables_check  # noqa: E402

E16_CAP = ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt"
E16_B5_GEN = ROOT / "data" / "e17" / "decode" / "gen"
STREAM = ROOT / "data" / "e15" / "qry_extra" / C.STREAM_FILE
P2D = ROOT / "data" / "v2" / "p2"
STEPS = 20_000
CODE = ["uie", "research/native/e8_common.py", "research/native/e11_common.py",
        "research/native/e12_common.py", "research/e13/e13_common.py", "research/e15/e15_common.py",
        "research/e16/e16_common.py", "research/e17/e17_common.py", "research/v2/v2_bpe.py",
        "research/v2/v2_mem.py", "research/v2/v2_p1_common.py", "research/v2/v2_p2_common.py",
        "research/v2/v2_p2_tokenizer.py", "research/v2/v2_p2_train.py"]
PRINTS = ("e10_start_sha256", "e11_start_sha256", "new_token_rows_sha256", "new_position_rows_sha256")
TESTS = ["tests/test_v2_p2.py", "tests/test_e16.py", "tests/test_e17.py"]
V_GEN_PER_MODE = 500                 # 1,000 in all
V_GEN_MIN_SHARE = 0.99               # section 12
K12E_MAX = 0.01
#: the page-row name gap (section 5): P1b's rows (v2_p1b_diag_frames: reading C's trained-site
#: draw, and reading A's new-site rows, which are K-used's draw in v2_p1_fit)
GAP_ROWS = 2_000
GAP_SEED = 20261054
K_USED_SEED = 20261050                 # v2_p1_common.P1_SEED, which v2_p1_fit.rows draws with
K_USED_STREAMS = (("benign", "name", "name_val"), ("benign", "structure", "structure_val"),
                  ("phishing", "name", "phishing_name_val"), ("phishing", "structure", "phishing_structure_val"))
#: the arm being verified (V2_ARM), set by main
ARM = ""


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def norm(x):
    return json.loads(json.dumps(x))


def arm_paths(arm: str) -> dict:
    root = Path(os.environ["V2_DATA"]) / arm if os.environ.get("V2_DATA") else ROOT / "data" / "v2" / arm
    return {"root": root, "train": root / "train", "gen": root / "generate" / "gen"}


def code_unchanged(commit: str) -> list:
    if not commit or subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                    cwd=ROOT).returncode:
        return [f"V1: code commit {commit!r} is not in this history"]
    changed = git("diff", "--name-only", commit, "HEAD", "--", *CODE)
    return [f"V1: code changed since the GPU ran it: {changed.split()}"] if changed else []


def v_iso() -> list:
    out = []
    if git("diff", "--name-only", C.ISO_COMMIT, "HEAD", "--", "uie"):
        out.append(f"V-iso: uie/ differs from {C.ISO_COMMIT[:12]}")
    if git("status", "--porcelain", "--", "uie"):
        out.append("V-iso: uie/ has uncommitted changes")
    return out


def tests() -> tuple:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return ([] if r.returncode == 0 else [f"tests: {' '.join(TESTS)} failed"]), f"exit {r.returncode}: {tail}"


def generation_tables():
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    shipped = SuffixTable.load(ROOT / TABLES["suffix_table.json"][0])
    st = StateSuffixTables.load(ROOT / TABLES["state_suffix_tables.json"][0])
    ps = load(ROOT / "data" / "e12" / "presence_slash_table.json")["table"]
    return shipped, st, ps


# --------------------------------------------------------------------------- #
# regeneration (V-gen-equiv, V-gen)
# --------------------------------------------------------------------------- #

def first_difference(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def drawn_identity(model, tok, tables, urls, label: str, seed: int, budget: int) -> tuple:
    """E13's V2 (as v2_p1_verify.drawn_identity): leading suffix groups regenerated on this CPU
    with the registered decode, every mismatch kept: (same, compared, mismatches). `model` is
    the piece rule around the model, as the notebook generates."""
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
                           **C.decode_kwargs())
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


def regenerate(rule, tok, files: dict, targets: dict, per_mode: int) -> dict:
    shipped, st, ps = generation_tables()
    out, total, same, mism = {}, 0, 0, []
    for mode, names in targets.items():
        per = -(-per_mode // len(names))
        for f in names:
            j, urls = files[f]
            tables = (K.supplied_tables(ps, shipped.counts, j["label"]) if j["mode"] == "free"
                      else K.control_tables(ps, st, j["label"], j["state"]))
            s, c, m = drawn_identity(rule, tok, tables, urls, j["label"], j["seed"], per)
            out[f] = {"compared": c, "identical": s}
            total, same = total + c, same + s
            mism += [{"file": f, **x} for x in m]
            print(f"  {f:<46} {s}/{c}", flush=True)
    return {"per_file": out, "compared": total, "identical": same,
            "share": same / total if total else None, "mismatches": mism}


def targets_for(arm: str) -> dict:
    return {"free": [f"free_{arm}_benign_g0.json", f"free_{arm}_phishing_g0.json"],
            "control": [f"ctrl_{arm}_benign_P1Q1_g0.json", f"ctrl_{arm}_benign_P0Q1_g0.json"]}


def step_gen_equiv(args) -> dict:
    """V-gen-equiv (section 12): e16_cap's weights, P2's tokenizer with no tables, the piece
    rule and e16_b5's decode reproduce e16_b5's files byte for byte."""
    import torch
    from uie.train import load_checkpoint
    from v2_p2_tokenizer import HybridTokenizer
    fails = []
    if sha256(E16_CAP) != C.E16_CAP_SHA256:
        fails.append("V1: the laptop's e16_cap final.pt is not the adopted one")
    gprov = load(E16_B5_GEN / "PROVENANCE.json")
    if gprov.get("decode") != C.decode_kwargs():
        fails.append(f"V-gen-equiv: e16_b5's decode {gprov.get('decode')} is not P2's {C.decode_kwargs()}")
    f2, files = files_check(E16_B5_GEN, gprov, C.jobs(C.BASELINE))
    fails += f2
    model, tok16, _ = load_checkpoint(E16_CAP, device=torch.device("cpu"))
    model.eval()
    tok = HybridTokenizer(base_state=tok16.state_dict())
    rule = C.PieceRule(model, tok).eval()
    rep = regenerate(rule, tok, files, targets_for(C.BASELINE), args.per_mode)
    if rep["compared"] < 2 * args.per_mode or rep["identical"] != rep["compared"]:
        fails.append(f"V-gen-equiv: {rep['identical']}/{rep['compared']} identical (bar: every URL)")
    return {"regenerated": rep, "failures": fails}


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #

def step_train(args) -> dict:
    import torch
    from v2_p2_tokenizer import HybridTokenizer
    arm = ARM
    paths = arm_paths(arm)
    prov = load(paths["train"] / "PROVENANCE.json")
    fails = code_unchanged(prov.get("code_commit", "")) + v_iso()
    rec_arm = prov.get("arms", {}).get(arm, {})
    ck = paths["train"] / arm / "final.pt"
    if not ck.is_file() or rec_arm.get("final_sha256") != sha256(ck):
        fails.append(f"V1: {arm} final.pt differs from the checkpoint training recorded")
    if sha256(E16_CAP) != C.E16_CAP_SHA256:
        fails.append("V1: the laptop's e16_cap final.pt is not the adopted one")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    base = torch.load(E16_CAP, map_location="cpu", weights_only=False)
    tc, mc, tk = norm(blob["train_config"]), norm(blob["model_config"]), norm(blob["tokenizer"])
    btc, bmc, base_tok = norm(base["train_config"]), norm(base["model_config"]), norm(base["tokenizer"])
    del blob, base
    import gc
    gc.collect()
    r = load(paths["train"] / arm / "train_result.json")
    accum = (prov.get("memory") or {}).get("accum_steps")
    tok = HybridTokenizer.from_state_dict(tk) if tk.get("hybrid") == "V2-P2" else None
    # V-recipe: e16_cap's in every respect but the arm's own settings
    for k, v in btc.items():
        if k == "qry_extra":
            got, want = Path(tc.get(k, "")).name, C.STREAM_FILE
        elif k == "accum_steps":
            got, want = tc.get(k), accum
        elif k == "loss_constant":
            got, want = tc.get(k), C.LOSS_CONSTANT_P2[arm]
        else:
            got, want = tc.get(k), v
        if got != want:
            fails.append(f"V-recipe: train_config {k} = {got!r}, registered {want!r}")
    if sorted(set(tc) - set(btc)):
        fails.append(f"V-recipe: extra train settings {sorted(set(tc) - set(btc))}")
    if accum not in (C.ACCUM_STEPS, C.ACCUM_FALLBACK) or r.get("accum_steps") != accum:
        fails.append(f"V-recipe: accum_steps {r.get('accum_steps')}, the memory check chose {accum}")
    for k, v in bmc.items():
        want = tok.vocab_size if (k == "vocab_size" and tok is not None) else v
        if mc.get(k) != want:
            fails.append(f"V-recipe: model_config {k} = {mc.get(k)!r}, registered {want!r}")
    if sorted(set(mc) - set(bmc)):
        fails.append(f"V-recipe: extra model settings {sorted(set(mc) - set(bmc))}")
    # the tokenizer: e16_cap's, plus exactly the arm's registered tables (V-T3)
    if tok is None or tk.get("base") != base_tok:
        fails.append("V-recipe: the tokenizer is not e16_cap's plus pieces")
    else:
        got_tables = {k: C.table_sha256(v) for k, v in (tk.get("tables") or {}).items()}
        if got_tables != {k: C.TABLE_SHA256[k] for k in C.ARMS[arm]}:
            fails.append(f"V-T3: the checkpoint's tables {got_tables} are not the arm's registered ones")
        if tk.get("dropout") != C.DROPOUT:
            fails.append(f"V-recipe: tokenizer dropout {tk.get('dropout')}")
        if r.get("params") != C.params(tok):
            fails.append(f"V-recipe: {r.get('params')} parameters, registered {C.params(tok)}")
    if r.get("counts") != C.COUNTS or r.get("dropped_over_max_len") != C.DROPPED:
        fails.append(f"V1: rows {r.get('counts')} / dropped {r.get('dropped_over_max_len')}")
    if r.get("loss_constant") != C.LOSS_CONSTANT_P2[arm]:
        fails.append(f"V1: C = {r.get('loss_constant')}, registered {C.LOSS_CONSTANT_P2[arm]}")
    if not all(isinstance(x, float) and math.isfinite(x) for x in (r.get("final") or {}).values()):
        fails.append("V1: non-finite validation loss")
    # the P2 record (V-T3, dropout, scope)
    p2 = r.get("p2") or {}
    want_tables = {k: (C.TABLE_SHA256[k] if k in C.ARMS[arm] else None) for k in ("rest", "host")}
    for name, got, want in (("tables", p2.get("tables"), want_tables), ("rules", p2.get("rules_sha256"), C.rules_sha256()),
                            ("dropout", p2.get("dropout"), C.DROPOUT), ("name scope", p2.get("name_scope"), C.NAME_SCOPE)):
        if got != want:
            fails.append(f"V-T3/V-recipe: the run's P2 {name} {got!r}, registered {want!r}")
    # V-start
    sc = prov.get("start_check") or {}
    w384, w512, ps_ = sc.get("width_384") or {}, sc.get("width_512") or {}, sc.get(arm) or {}
    want384 = {"init_weights_sha256": E15.E12_FINAL_INIT, **E15.E12_FINAL_START}
    if any(w384.get(k) != v for k, v in want384.items()):
        fails.append("V-start: the notebook's width-384 rebuild is not e12_final's recorded start")
    if any(w512.get(k) != v for k, v in C.E16_START.items()):
        fails.append("V-start: the notebook's width-512 start is not e16_cap's recorded start")
    caca = C.E16_START["init_weights_sha256"]
    if ps_.get("base_init_sha256") != caca or sc.get("peeled") != caca:
        fails.append("V-start: the notebook's arm start does not peel off to e16_cap's start")
    s = r.get("e12_start") or {}
    if s.get("base_init_sha256") != caca:
        fails.append("V-start: training's e16_cap part is not e16_cap's recorded start")
    for k in ("rest_rows_sha256", "host_rows_sha256"):
        if s.get(k) != ps_.get(k):
            fails.append(f"V-start: training's {k} is not the one recorded before training")
    if r.get("init_weights_sha256") != ps_.get("init_weights_sha256"):
        fails.append("V-start: training did not start from the start recorded before training")
    fails += [f"V-start: {k} differs from e16_cap's recorded start" for k in PRINTS if s.get(k) != C.E16_START[k]]
    # V-batch, V-stream, V-mass
    q = r.get("qry_extra") or {}
    if q.get("old_batch_digest") != C.V_BATCH_DIGEST or q.get("steps") != STEPS:
        fails.append(f"V-batch: digest {str(q.get('old_batch_digest'))[:12]} over {q.get('steps')} steps")
    local = sha256(STREAM)
    if not (q.get("source_sha256") == local == prov.get("stream_sha256") == C.STREAM_SHA256):
        fails.append("V-stream: stream 1's hash differs")
    st = q.get("strata") or {}
    inj = ("old_rows_seen", "appended_rows", "appended_query_positions")
    got_mass = {k: {f: v.get(f) for f in inj} for k, v in st.items()}
    if got_mass != {k: {f: v[f] for f in inj} for k, v in C.INJECTION.items()}:
        fails.append(f"V-mass: {got_mass}, e16_cap's {C.INJECTION}")
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
    gen_check, f_g = ({}, []) if args.skip_gen_check else generation_check(ck)
    fails += f_g
    # reported: fit per character against e16_cap's record, and the page-row name gap
    e16r = load(ROOT / "data" / "e16" / "train" / "e16_cap" / "train_result.json")
    fit = {}
    for label in ("benign", "phishing"):
        a, b = r["field_nll"][label], e16r["field_nll"][label]
        fit[label] = {f: {"val": a["val"][f], "val_e16_cap": b["val"][f], "delta_val": a["val"][f] - b["val"][f],
                          "gap": a["gap"][f], "gap_e16_cap": b["gap"][f]}
                      for f in a["val"] if f in b["val"]}
    gap = {} if args.skip_gap else name_gap(ck)
    return {"arm": arm, "accum_steps": accum, "memory_check": prov.get("memory"), "torch": prov.get("torch"),
            "gpu": prov.get("gpu"), "start": ps_, "tests": test_tail, "clipped_after_1000": clip,
            "final_val": r.get("final"), "train_hours": (r.get("seconds") or 0) / 3600,
            "wall_minutes": rec_arm.get("wall_minutes"), "peak_gb_training": rec_arm.get("peak_gb_training"),
            "field_fit_per_character": fit, "page_row_name_gap": gap, "generation_check": gen_check,
            "failures": fails}


def generation_check(ck: Path) -> tuple:
    """Three URLs per call, both labels free and benign P1Q1 in control, with the notebook's
    own calls and the piece rule, on this CPU. Each must parse; no piece outside its fields."""
    import torch
    from uie.evaluation import FIELDS
    shipped, st, ps = generation_tables()
    model, tok, _ = C.load_checkpoint(ck, device=torch.device("cpu"))
    model.eval()
    rule = C.PieceRule(model, tok).eval()
    out = {f"free {label}": C.free_urls(model, tok, label, 0, shipped, ps, n=3, rule=rule) for label in K.LABELS}
    out["control benign P1Q1"] = C.control_urls(model, tok, "benign", "P1Q1", 0, st, ps, n=3, rule=rule)
    fails = [f"generation check: {k} gave {len(v)} URLs, or one that does not parse"
             for k, v in out.items() if len(v) != 3 or not all(FIELDS["FULL"](u) for u in v)]
    if rule.record()["violations"]:
        fails.append(f"generation check: {rule.record()['violations']} pieces outside their fields")
    return {**out, "piece_rule": rule.record()}, fails


def per_row_field_nll(model, tok, seqs, batch: int = 32) -> list:
    """[{field: (nll sum, characters)}] per row: each content target's loss, a piece's counted
    over the characters it writes (the chunked head, no gradients)."""
    import torch
    from uie.tokenizer import _MARKERS
    from v2_p2_train import chunked_ce, hidden_states
    marker_of = {tok.marker_id(m): m for m in _MARKERS}
    text_of = getattr(tok, "token_text", None)
    chars = {tok.stoi[c] for c in tok._chars}
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [None] * len(seqs)
    model.eval()
    with torch.no_grad():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            width = max(len(seqs[i]) for i in idx)
            ids = torch.tensor([seqs[i] + [tok.pad_id] * (width - len(seqs[i])) for i in idx])
            nll = chunked_ce(model, hidden_states(model, ids[:, :-1]), ids[:, 1:], -100, grad=False)
            for r, i in enumerate(idx):
                node, current = {}, None
                for pos, t in enumerate(seqs[i][1:]):
                    if t in marker_of:
                        current = marker_of[t]
                        continue
                    text = text_of(t) if text_of else (tok.itos[t] if t in chars else None)
                    if current is None or text is None or t == tok.eos_id:
                        continue
                    s, n = node.get(current, (0.0, 0))
                    node[current] = (s + float(nll[r, pos]), n + len(text))
                out[i] = node
    return out


def _corpus_urls(stem: str) -> list:
    with gzip.open(ROOT / "data" / "corpus2" / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


def p1b_rows() -> dict:
    """P1b's rows, exactly (v2_p1b_diag_frames): per class, the new-site page rows of its
    reading A (K-used's draw, v2_p1_fit.rows: one generator seeded 20261050 sampling 2,000 rows
    from each validation stream in its order) and the trained-site page rows of its reading C
    (2,000 structure_train rows, seed 20261054 + the class's index)."""
    rng = random.Random(K_USED_SEED)
    new = {}
    for label, corpus, stem in K_USED_STREAMS:
        pick = rng.sample(_corpus_urls(stem), GAP_ROWS)
        if corpus == "structure":
            new[label] = pick
    out = {}
    for li, label in enumerate(("benign", "phishing")):
        pool = _corpus_urls(("" if label == "benign" else "phishing_") + "structure_train")
        out[label] = {"trained sites": random.Random(GAP_SEED + li).sample(pool, GAP_ROWS),
                      "new sites": new[label]}
    return out


def name_gap(ck: Path) -> dict:
    """Section 5, reported: P1b's measure on P1b's rows (`p1b_rows`), each in the page frame,
    rows over max_len dropped as `encode_corpus` drops them: the name fields' NLL per character,
    the arm's and e16_cap's on the same rows."""
    import torch
    from uie.train import load_checkpoint
    model, tok, _ = C.load_checkpoint(ck, device=torch.device("cpu"))
    e16, tok16, _ = load_checkpoint(E16_CAP, device=torch.device("cpu"))
    rows_of = p1b_rows()
    out = {}
    for label in ("benign", "phishing"):
        node = {}
        for kind in ("trained sites", "new sites"):
            urls = [u for u in rows_of[label][kind] if len(tok16.encode(u, label, frame="page")) <= model.cfg.max_len]
            for who, m, t in ((ARM, model, tok), ("e16_cap", e16, tok16)):
                rows = per_row_field_nll(m, t, [t.encode(u, label, frame="page") for u in urls])
                for f in ("LBL", "SUB"):
                    s = sum(r.get(f, (0.0, 0))[0] for r in rows)
                    n = sum(r.get(f, (0.0, 0))[1] for r in rows)
                    node.setdefault(who, {}).setdefault(f, {})[kind] = s / n if n else None
        for who, fields in node.items():
            for f, v in fields.items():
                if v.get("trained sites") is not None and v.get("new sites") is not None:
                    v["gap (new - trained)"] = v["new sites"] - v["trained sites"]
        out[label] = node
        print(f"  name gap {label}: " + "; ".join(
            f"{who} LBL {v['LBL'].get('trained sites', 0):.3f} trained / {v['LBL'].get('new sites', 0):.3f} new"
            for who, v in node.items()), flush=True)
    return out


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #

def begun(u: str) -> bool:
    """The query has begun: a '?' before any '#' (e17_compare.begun)."""
    h, q = u.find("#"), u.find("?")
    return q >= 0 and (h < 0 or q < h)


def v_rules(files: dict, gprov: dict) -> tuple:
    """V-rules (section 12; Amendment 3), from the generation's output -- each file's record:
    - D1 and D1' (the host ended on a token its plan allows), B5 (a Q = 1 row's <QRY> before
      any <EOS> or <FRG>) and the piece rule (no piece outside its fields), counted on the
      generated tokens as they were sampled (v2_p2_common.token_rules, PieceRule): zero
      violations, and the rows' plans as supplied;
    - reported beside them, the same rules read from the URL strings, each URL's plan rebuilt
      from its file's seed (e12_score.state_positions; S by e12_score's rule; B5 by
      e17_compare's): a delimiter written as a character inside a host field reads as a host
      end there, so the strings are not the exact record (e16_b5: 5 of 180,000)."""
    import e12_score as S12
    from uie.factorization import parse
    shipped, st, ps = generation_tables()
    keys = ("d1_checked", "d1_violations", "d1_cut", "b5_checked", "b5_violations", "b5_cut",
            "b5_closed_before_host", "cut_before_host", "ended_before_host", "no_plan", "plan_mismatch", "rows")
    tokens = dict.fromkeys(keys, 0)
    no_record = []
    for name in files:
        tr = gprov["files"][name].get("token_rules")
        if tr is None:
            no_record.append(name)
            continue
        for k in keys:
            tokens[k] += tr.get(k, 0)
    s_bad, q_bad, cut_n, n = [], [], 0, 0
    for name, (j, urls) in files.items():
        if j["mode"] == "free":
            tables = K.supplied_tables(ps, shipped.counts, j["label"])
        else:
            tables = K.control_tables(ps, st, j["label"], j["state"])
        cut = collections.Counter(gprov["files"][name].get("cut_at_max_len") or [])
        for u, k in zip(urls, S12.state_positions(tables, len(urls), j["seed"])):
            n += 1
            if cut[u]:
                cut[u] -= 1
                cut_n += 1
                continue
            if bool(parse(u).path) != (k[-1] == "1"):
                s_bad.append({"file": name, "plan": k, "url": u})
            if k[3] == "1" and not begun(u):
                q_bad.append({"file": name, "plan": k, "url": u})
    viol = {f: r.get("piece_rule", {}).get("violations") for f, r in gprov["files"].items()}
    missing = sorted(f for f, v in viol.items() if v is None)
    pieces_out = sum(v for v in viol.values() if v)
    fails = []
    if no_record:
        fails.append(f"V-rules: files without a token-level record {no_record[:5]}")
    if tokens["d1_violations"] or tokens["b5_violations"] or tokens["plan_mismatch"]:
        fails.append(f"V-rules: D1/D1' {tokens['d1_violations']}, B5 {tokens['b5_violations']} violations; "
                     f"{tokens['plan_mismatch']} rows against their supplied plan")
    if pieces_out or missing:
        fails.append(f"V-rules: {pieces_out} pieces outside their fields; files without a record {missing}")
    return {"tokens (deciding)": tokens, "pieces_outside_fields": pieces_out,
            "piece_rule_mass_removed": _mass(gprov),
            "strings (reported)": {"urls": n, "cut_at_max_len": cut_n,
                                   "s_unhonoured": len(s_bad), "s_examples": s_bad[:20],
                                   "b5_unbegun": len(q_bad), "b5_examples": q_bad[:20]}}, fails


def _mass(gprov: dict) -> dict:
    out = collections.defaultdict(lambda: [0.0, 0])
    for r in gprov["files"].values():
        for f, v in ((r.get("piece_rule") or {}).get("fields") or {}).items():
            out[f][0] += v["mass_removed"]
            out[f][1] += v["steps"]
    return {f: {"mass_removed_per_step": m / s if s else None, "steps": s} for f, (m, s) in out.items()}


def step_generate(args) -> dict:
    import torch
    arm = ARM
    paths = arm_paths(arm)
    gprov = load(paths["gen"] / "PROVENANCE.json")
    fails = code_unchanged(gprov.get("code_commit", "")) + tables_check(gprov) + v_iso()
    ck = paths["train"] / arm / "final.pt"
    if gprov.get("checkpoints") != {arm: sha256(ck)}:
        fails.append(f"V1: generation did not read {arm}'s verified final.pt")
    if load(paths["root"] / "VERIFY_TRAIN.json").get("verdict") != "PASS":
        fails.append("V1: VERIFY_TRAIN has not passed")
    if gprov.get("decode") != C.decode_kwargs():
        fails.append(f"V1: decode {gprov.get('decode')}, registered {C.decode_kwargs()}")
    if gprov.get("piece_tables") != C.TABLE_SHA256 or gprov.get("rules_sha256") != C.rules_sha256():
        fails.append("V-T3: generation read other piece tables or rules")
    f2, files = files_check(paths["gen"], gprov, C.jobs(arm))
    fails += f2
    gen = {}
    if not args.skip_v_gen and not f2:
        model, tok, _ = C.load_checkpoint(ck, device=torch.device("cpu"))
        model.eval()
        rule = C.PieceRule(model, tok).eval()
        gen = regenerate(rule, tok, files, targets_for(arm), V_GEN_PER_MODE)
        if gen["compared"] < 2 * V_GEN_PER_MODE or gen["identical"] < V_GEN_MIN_SHARE * gen["compared"]:
            fails.append(f"V-gen: {gen['identical']}/{gen['compared']} identical (bar: >= {2 * V_GEN_PER_MODE} "
                         f"compared, >= {V_GEN_MIN_SHARE:.0%} identical)")
    rules, f4 = v_rules(files, gprov) if not f2 else ({}, ["V-rules: not checked (the files failed)"])
    fails += f4
    return {"arm": arm, "v_gen": gen, "v_rules": rules, "environment": gprov.get("environment"),
            "failures": fails}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("gen-equiv", "train", "generate"))
    ap.add_argument("--per-mode", type=int, default=V_GEN_PER_MODE, help="gen-equiv: URLs per mode")
    ap.add_argument("--skip-v-gen", action="store_true", help="dry runs only")
    ap.add_argument("--skip-tests", action="store_true", help="dry runs only")
    ap.add_argument("--skip-gen-check", action="store_true", help="dry runs only")
    ap.add_argument("--skip-gap", action="store_true", help="dry runs only")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    if args.step == "gen-equiv":
        rep = step_gen_equiv(args)
        dest = P2D / "VERIFY_GEN_EQUIV.json"
    else:
        arm = os.environ.get("V2_ARM")
        if arm not in C.ARMS:
            raise SystemExit(f"set V2_ARM to one of {sorted(C.ARMS)}")
        global ARM
        ARM = arm
        rep = {"train": step_train, "generate": step_generate}[args.step](args)
        dest = arm_paths(arm)["root"] / f"VERIFY_{args.step.upper()}.json"
    rep.update({"registration": C.REGISTRATION, "step": args.step, "seconds": round(time.time() - t0, 1),
                "verdict": "PASS" if not rep["failures"] else "FAIL"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    if args.step == "gen-equiv" or args.step == "generate":
        g = rep.get("regenerated") or rep.get("v_gen") or {}
        if g:
            print(f"regenerated: {g['identical']}/{g['compared']} identical ({g['share']:.2%}); "
                  f"first divergences at {[m['at'] for m in g['mismatches']][:20]}")
        if rep.get("v_rules"):
            v = rep["v_rules"]
            t, s = v["tokens (deciding)"], v["strings (reported)"]
            print(f"V-rules (tokens): D1/D1' {t['d1_violations']} of {t['d1_checked']:,}, B5 {t['b5_violations']} "
                  f"of {t['b5_checked']:,}, plan mismatches {t['plan_mismatch']}, pieces outside their fields "
                  f"{v['pieces_outside_fields']}")
            print(f"  read from the strings (reported): {s['urls']:,} URLs, {s['cut_at_max_len']} cut; "
                  f"S unhonoured {s['s_unhonoured']}, Q = 1 without a begun query {s['b5_unbegun']}")
    else:
        print(f"accumulation: {rep['accum_steps']} parts; memory check {rep['memory_check']}; "
              f"torch {rep['torch']} on {rep['gpu']}")
        print(f"trained in {rep['train_hours']:.2f} h; peak {rep['peak_gb_training']} GB")
        print(f"clipped after step 1,000: {rep['clipped_after_1000']}")
        print(f"final val: {rep['final_val']}")
        print(f"tests: {rep['tests']}")
    print("failures:", rep["failures"] or "none")
    print(f"VERIFY ({args.step}):", rep["verdict"], f"-> {dest}")


if __name__ == "__main__":
    main()
