"""Build research/v2/V2P2_train_a.ipynb, V2P2_train_b.ipynb, V2P2_generate_a.ipynb and
V2P2_generate_b.ipynb (research/v2/PREREG_V2_P2.md sections 5, 6 and 15).

    python research/v2/make_v2p2_notebooks.py

All four run the `uie-v2p2-code` bundle (make_v2p2_bundle.py); every constant and call comes
from research/v2/v2_p2_common.py, so the laptop repeats them exactly. The structure is
V2-P1's notebooks' (make_v2p1_notebooks.py), which ran on Kaggle.

- V2P2_train_<a|b>     the arm: e16_cap's recipe with the arm's tables, BPE-dropout and the
                       name scope. Before training (section 5): the bundle's manifest; torch
                       and the GPU; V-start' (the width-384 rebuild against e12_final's record,
                       the width-512 start against caca0107..., the arm's start built twice,
                       and its piece rows peeled off back to caca0107...); the worst-case
                       memory check at max_len 384 (2 parts, else 4). The readback compares
                       the run with e16_cap's settings and injection, the arm's tables, C,
                       dropout and scope.
- V2P2_generate_<a|b>  the arm's 50 files with e16_b5's decode plus the piece rule, after the
                       laptop has verified training. Each file records the piece rule's
                       removed mass and its violation count.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "e15"))
import make_v2p1_notebooks as BASE  # noqa: E402
from make_e15_notebooks import code, md  # noqa: E402

LOCATE_CODE = BASE.LOCATE_CODE.replace("V2P1_CODE_MANIFEST.json", "V2P2_CODE_MANIFEST.json").replace(
    "uie-v2p1-code", "uie-v2p2-code")

ENV = r'''
# --- 2. environment: install first, then run THIS code and prove it ------------
import importlib, subprocess, sys
from importlib import metadata

TLDEXTRACT = "5.3.1"          # the version the local evaluation parses with


def installed(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


if installed("tldextract") != TLDEXTRACT:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        f"tldextract=={TLDEXTRACT}"], check=True)
    except subprocess.CalledProcessError:
        raise SystemExit("pip could not install tldextract: Settings -> Internet -> On, rerun.")
    importlib.invalidate_caches()
assert installed("tldextract") == TLDEXTRACT, installed("tldextract")

# CODE ONLY on the path: the data bundle carries an older `uie`.
for p in (CODE / "research" / "v2", CODE / "research" / "e17", CODE / "research" / "e16",
          CODE / "research" / "e15", CODE / "research" / "e13", CODE / "research" / "native", CODE):
    sys.path.insert(0, str(p))
for name in [m for m in sys.modules if m == "uie" or m.startswith("uie.")
             or m in ("e8_common", "e11_common", "e12_common", "e13_common", "e15_common",
                      "e16_common", "e17_common", "v2_mem", "v2_bpe", "v2_p1_common", "v2_p2_common",
                      "v2_p2_tokenizer", "v2_p2_train")]:
    del sys.modules[name]
import uie
import e12_common as K
import e15_common as E15
import e16_common as E16
import e17_common as E17
import v2_bpe
import v2_p2_common as C
import v2_p2_tokenizer as TK
import v2_p2_train as T
manifest = json.loads((CODE / C.CODE_MANIFEST).read_text())
for mod in (uie, K, E15, E16, E17, v2_bpe, C, TK, T):
    assert str(CODE) in mod.__file__, f"{mod.__name__} is SHADOWED -- check sys.path"
bad = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
bad += [t for t, h in manifest["tables"].items() if sha256(CODE / t) != h]
bad += [t for t, h in manifest["piece_tables"].items() if sha256(CODE / t) != h]
if sha256(CODE / C.STREAM_FILE) != manifest["stream"]["sha256"] or manifest["stream"]["sha256"] != C.STREAM_SHA256:
    bad.append(C.STREAM_FILE)
if sha256(CODE / C.E16_CAP_RECORD) != manifest["e16_cap_record"]:
    bad.append(C.E16_CAP_RECORD)
if manifest["piece_tables_registered"] != C.TABLE_SHA256 or manifest["rules_sha256"] != C.rules_sha256():
    bad.append("the registered tables or rules")
if manifest["loss_constants"] != C.LOSS_CONSTANT_P2:
    bad.append("the registered C")
assert not bad, f"STOP. files differ from the manifest: {bad}"
PIECES = C.load_tables(ARM, CODE)            # raises unless each table is the registered one (V-T3)
print(f"code: commit {manifest['commit'][:12]}; {len(manifest['files'])} files, "
      f"{len(manifest['tables'])} tables, the piece tables {[h[:12] for h in C.TABLE_SHA256.values()]}, "
      f"stream 1 ({manifest['stream']['rows']:,} rows) and e16_cap's record verified")

import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4, then rerun.")
GPU = torch.cuda.get_device_name(0)
print("torch:", torch.__version__, "| gpu:", GPU)
'''


def train_cells(arm: str):
    letter = arm[-1].upper()
    what = ("the path, query and fragment written in pieces (the rest table)" if arm == "p2_a" else
            "the path, query and fragment in rest pieces, and the name and subdomains in host pieces")
    return [
md(rf'''
# V2-P2 — training `{arm}`: {what}

Registration: `research/v2/PREREG_V2_P2.md` (§3–§5, Amendments 1–3). Everything is `e16_cap`'s:
the width-512 network and its start, the data and stream 1, the seed, the rows and batches,
20,000 steps, the learning rate and the fixed objective, the two-part batches. **The arm's
changes:** {what}; BPE-dropout at p = 0.1 in training; page rows teach no name-field content
(the name scope); C re-derived for the arm. No word list: every piece was learned from how
many independent sites share it.

**Before any training, cell 3 checks the start** (the registered procedure must rebuild
`e12_final`'s and `e16_cap`'s recorded starts; the arm's start is built twice, and removing its
piece rows must give back `e16_cap`'s start exactly) **and that the worst-case batch fits.**
**Cell 5 reads the run back:** it must print `MISMATCHES: none -- {arm} is as registered`.

**Attach two inputs:** **`uie-corpus2`** and **`uie-v2p2-code`**. **Settings:** GPU T4,
Internet **On**. **Save Version → Save & Run All.** About 3.5–4.5 h. **A rerun skips a
finished arm, and never resumes a half-finished one.** When done, download
`/kaggle/working/{arm}/final.pt`, `/kaggle/working/{arm}/train_result.json` and
`/kaggle/working/PROVENANCE.json` (`best.pt` is not needed).
'''),
code(f'ARM = "{arm}"\n' + LOCATE_CODE + r'''
data_hits = sorted(root.rglob("bundle.manifest.json")) if root.exists() else []
if not data_hits:
    raise SystemExit("Attach the uie-corpus2 dataset too, then rerun.")
DATA = data_hits[0].parent
print("data:", DATA)
'''),
code(ENV + r'''
need = ["name_train", "structure_train", "phishing_name_train", "phishing_structure_train",
        "name_val", "structure_val", "phishing_name_val", "phishing_structure_val"]
present = {p.name for p in DATA.iterdir()}
missing = [s for s in need if f"{s}.jsonl" not in present and f"{s}.jsonl.gz" not in present]
if missing:
    raise SystemExit(f"STOP. missing {missing}: this is not the uie-corpus2 bundle.")
import inspect
from uie.kaggle_train import run
assert {"accum_steps", "qry_extra", "d_model", "e12_start", "loss_constant"} <= set(inspect.signature(run).parameters)
REC = json.loads((CODE / C.E16_CAP_RECORD).read_text())
print("data bundle: all 8 streams present; e16_cap's record loaded")
'''),
code(r'''
# --- 3. THE START CHECK and the worst-case memory check, before training ---------------
from uie.free import weights_sha256
from uie.tokenizer import URLTokenizer
tok = URLTokenizer.from_state_dict(REC["tokenizer"])
ref = E16.start_fingerprints(tok, 384)
want = {"init_weights_sha256": E15.E12_FINAL_INIT, **E15.E12_FINAL_START}
off = {k: (ref.get(k), v) for k, v in want.items() if ref.get(k) != v}
if off:
    raise SystemExit(f"STOP. This torch does not rebuild e12_final's recorded start: {off}. Nothing was trained.")
START = E16.start_fingerprints(tok, E16.D_MODEL)
off = {k: (START.get(k), v) for k, v in C.E16_START.items() if START.get(k) != v}
if off:
    raise SystemExit(f"STOP. The width-512 start is not e16_cap's recorded start: {off}. Nothing was trained.")
ptok = C.tokenizer(REC["tokenizer"], ARM, CODE)
m1, P2_START = C.start(ptok)
PEELED = weights_sha256(C.peel(m1, ptok.n_base))
del m1
gc.collect()
m2, again = C.start(ptok)
del m2
gc.collect()
if again != P2_START:
    raise SystemExit("STOP. The arm's start is not deterministic. Nothing was trained.")
if not (P2_START["base_init_sha256"] == PEELED == C.E16_START["init_weights_sha256"]) or P2_START["params"] != C.params(ptok):
    raise SystemExit(f"STOP. The arm's start is not e16_cap's start plus the piece rows: {P2_START}, "
                     f"peeled {PEELED}. Nothing was trained.")
print(f"width 384 rebuilds e12_final's recorded start; width 512 rebuilds e16_cap's ({START['init_weights_sha256'][:12]}...)")
print(f"{ARM}'s start: e16_cap's part {P2_START['base_init_sha256'][:12]}..., the piece rows peeled off give "
      f"{PEELED[:12]}...; pieces {P2_START['pieces']}; whole {P2_START['init_weights_sha256'][:12]}...; "
      f"{P2_START['params']:,} parameters; vocab {ptok.vocab_size:,}")
total = torch.cuda.get_device_properties(0).total_memory / 2**30
ACCUM, peak = None, None
for parts in (C.ACCUM_STEPS, C.ACCUM_FALLBACK):
    fits = True
    try:
        # 256 rows plus a margin for the appended stream rows, split; every row at max_len
        peak = C.memory_preflight(torch.device("cuda"), math.ceil(C.PREFLIGHT_ROWS / parts), ptok.vocab_size)
    except torch.cuda.OutOfMemoryError:
        fits = False
    gc.collect()
    torch.cuda.empty_cache()
    if fits:
        ACCUM = parts
        break
    print(f"  {parts} parts: the worst-case step does not fit")
if ACCUM is None:
    raise SystemExit("STOP. The worst-case step does not fit even in 4 parts. Nothing was trained.")
RECIPE = C.recipe(CODE / C.STREAM_FILE, ARM, accum_steps=ACCUM)
SETTINGS = C.train_settings(ARM, CODE)
print(f"start check PASSED; memory check PASSED in {ACCUM} parts ({peak:.1f} of {total:.1f} GB)")
print(f"{ARM}:", RECIPE, "| dropout", SETTINGS["dropout"], "| name scope", SETTINGS["scope"])
'''),
code(r'''
# --- 4. train the arm (skipped only if finished) ----------------------------------------
import time

PROV = WORK / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"arms": {}}
prov.update({"registration": C.REGISTRATION, "code_commit": manifest["commit"], "gpu": GPU,
             "torch": torch.__version__, "stream_sha256": manifest["stream"]["sha256"],
             "start_check": {"width_384": ref, "width_512": START, ARM: P2_START, "peeled": PEELED},
             "memory": {"accum_steps": ACCUM, "peak_gb": round(peak, 2), "total_gb": round(total, 2)}})
PROV.write_text(json.dumps(prov, indent=1))
out = WORK / ARM
if (out / "train_result.json").exists() and (out / "final.pt").exists():
    print(f"{ARM}: finished earlier -- skipped")
else:
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    T.run(DATA, out, **SETTINGS, **RECIPE)
    prov["arms"][ARM] = {"wall_minutes": round((time.time() - t0) / 60, 1),
                         "peak_gb_training": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    PROV.write_text(json.dumps(prov, indent=1))
'''),
code(r'''
# --- 5. read the arm back against e16_cap's record and the registration -----------------
ck = torch.load(WORK / ARM / "final.pt", map_location="cpu", weights_only=False)
tc, mc = json.loads(json.dumps(ck["train_config"])), json.loads(json.dumps(ck["model_config"]))
tk = json.loads(json.dumps(ck["tokenizer"]))
btc, bmc = REC["train_config"], REC["model_config"]
r = json.loads((WORK / ARM / "train_result.json").read_text())
s = r.get("e12_start") or {}
q = r.get("qry_extra") or {}
p2 = r.get("p2") or {}
strata = q.get("strata", {})
inj = ("old_rows_seen", "appended_rows", "appended_query_positions")
want = []
for k, v in btc.items():
    if k == "qry_extra":
        want.append(("train_config qry_extra (file)", Path(tc.get(k, "")).name, C.STREAM_FILE))
    elif k == "accum_steps":
        want.append(("train_config accum_steps", tc.get(k), ACCUM))
    elif k == "loss_constant":
        want.append(("train_config loss_constant (the arm's C)", tc.get(k), C.LOSS_CONSTANT_P2[ARM]))
    else:
        want.append((f"train_config {k}", tc.get(k), v))
want += [("extra train settings", sorted(set(tc) - set(btc)), [])]
want += [(f"model_config {k}", mc.get(k), ptok.vocab_size if k == "vocab_size" else v) for k, v in bmc.items()]
want += [("extra model settings", sorted(set(mc) - set(bmc)), [])]
want += [("tokenizer: e16_cap's plus the arm's tables", tk == json.loads(json.dumps(ptok.state_dict())), True),
         ("tokenizer: its base is e16_cap's", tk.get("base") == REC["tokenizer"], True),
         ("parameters", r.get("params"), C.params(ptok)),
         ("start: e16_cap's part (V-start)", s.get("base_init_sha256"), C.E16_START["init_weights_sha256"]),
         ("start: the rest rows", s.get("rest_rows_sha256"), P2_START["rest_rows_sha256"]),
         ("start: the host rows", s.get("host_rows_sha256"), P2_START["host_rows_sha256"]),
         ("start: the whole (as training recorded it)", r.get("init_weights_sha256"), P2_START["init_weights_sha256"]),
         ("row counts", r.get("counts"), C.COUNTS),
         ("dropped over max_len", r.get("dropped_over_max_len"), C.DROPPED),
         ("loss constant", r.get("loss_constant"), C.LOSS_CONSTANT_P2[ARM]),
         ("stream sha256 (V-stream)", q.get("source_sha256"), C.STREAM_SHA256),
         ("old rows' batch digest (V-batch)", q.get("old_batch_digest"), C.V_BATCH_DIGEST),
         ("injection (V-mass: e16_cap's exactly, in characters)", {k: {f: v[f] for f in inj} for k, v in strata.items()},
          {k: {f: v[f] for f in inj} for k, v in C.INJECTION.items()}),
         ("P2: the tables (V-T3)", p2.get("tables"), {k: (C.TABLE_SHA256[k] if k in C.ARMS[ARM] else None) for k in ("rest", "host")}),
         ("P2: the per-field rules (V-T3)", p2.get("rules_sha256"), C.rules_sha256()),
         ("P2: dropout", p2.get("dropout"), C.DROPOUT),
         ("P2: the name scope", p2.get("name_scope"), C.NAME_SCOPE),
         ("finite validation", all(math.isfinite(x) for x in (r.get("final") or {}).values()), True)]
want += [(f"start: {k}", s.get(k), C.E16_START[k]) for k in
         ("e10_start_sha256", "e11_start_sha256", "new_token_rows_sha256", "new_position_rows_sha256")]
problems = [w for w in want if w[1] != w[2]]
clip = {k: v["clipped_share"] for k, v in (r.get("grad_diagnostics") or {}).items() if k.startswith("after_1000")}
prov["arms"].setdefault(ARM, {})
prov["arms"][ARM].update({"final_sha256": sha256(WORK / ARM / "final.pt"),
                          "init_weights_sha256": r.get("init_weights_sha256"),
                          "e12_start": s, "qry_extra": q, "p2": p2, "final_val": r.get("final"),
                          "counts": r.get("counts"), "params": r.get("params"),
                          "clipped_after_1000": clip, "train_seconds": r.get("seconds")})
PROV.write_text(json.dumps(prov, indent=1))
print(f"{ARM} val {r.get('final')}  (e16_cap's, per token of a different vocabulary: {C.E16_CAP_FINAL_VAL})")
print(f"trained in {r.get('seconds', 0) / 3600:.2f} h (e16_cap's loop: 2.73 h)")
print(f"clipped after step 1,000 (K12e: at most 0.01 in every band): {clip}")
print("MISMATCHES:", problems if problems else f"none -- {ARM} is as registered")
print(f"\nNow download /kaggle/working/{ARM}/final.pt, /kaggle/working/{ARM}/train_result.json "
      "and /kaggle/working/PROVENANCE.json (best.pt is not needed).")
'''),
    ]


def generate_cells(arm: str):
    gen = f"v2p2_gen_{arm[-1]}"
    return [
md(rf'''
# V2-P2 — `{arm}`'s generation: 50 files with `e16_b5`'s decode and the piece rule

Registration: `research/v2/PREREG_V2_P2.md` §6. Run **only after the laptop has verified
`{arm}`'s training** (VERIFY_TRAIN). E12's 50 jobs at E12's seeds, decoded raw + D1 + D1′ + B5,
exactly as `e16_b5`, plus the piece rule: a rest piece only in the path, query and fragment, a
host piece only in the name and subdomains. Every file records the probability mass the rule
removed and how many pieces were written outside their fields (registered: none).

**Attach two inputs:** **`uie-v2p2-code`**, and the **V2P2_train_{arm[-1]}** notebook's **output**.
**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About 1.5–2.5 h.
A rerun keeps every finished file. When done, download `/kaggle/working/{gen}`.
'''),
code(f'ARM = "{arm}"\n' + LOCATE_CODE + r'''
SRC = output_of(ARM)
if SRC is None:
    raise SystemExit(f"Attach the V2P2_train_{ARM[-1]} notebook's output too, then rerun.")
tprov = json.loads((SRC / "PROVENANCE.json").read_text())
CKPT = SRC / ARM / "final.pt"
assert CKPT.is_file() and sha256(CKPT) == tprov["arms"][ARM]["final_sha256"], \
    f"{ARM}: final.pt is not the one V2P2_train recorded"
print("checkpoint verified:", CKPT)
'''),
code(ENV),
code(rf'''
# --- 3. tables, the registered jobs, the model and the piece rule ---------------------
from uie.controlled import StateSuffixTables
from uie.suffixes import SuffixTable

shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
presence_slash = json.loads((CODE / "presence_slash_table.json").read_text())["table"]
device = torch.device("cuda")
OUT = WORK / "{gen}"
OUT.mkdir(parents=True, exist_ok=True)
jobs = sorted(C.jobs(ARM), key=lambda j: (0 if j["mode"] == "free" else 1, j["label"], j["state"] or "", j["g"]))
print(f"{{len(jobs)}} files, {{sum(j['n'] for j in jobs):,}} URLs; decode {{C.decode_kwargs()}} + the piece rule")
CKPT_SHA = {{ARM: sha256(CKPT)}}
model, tok, _ = C.load_checkpoint(CKPT, device=device)
model.to(device).eval()
assert all(p.device.type == "cuda" for p in model.parameters()), "STOP. the model is not on the GPU"
assert tok.presence_plan and tok.root_slash and tok.slash_plan and tok.n_base == 110
assert {{k: C.table_sha256(v) for k, v in tok._table_state.items()}} == {{k: C.TABLE_SHA256[k] for k in C.ARMS[ARM]}}, \
    "STOP. the checkpoint's tables are not the registered ones"
assert model.cfg.d_model == E16.D_MODEL and sum(p.numel() for p in model.parameters()) == C.params(tok)
rule = C.PieceRule(model, tok).to(device).eval()
print(f"loaded {{ARM}}: width {{model.cfg.d_model}}, vocab {{tok.vocab_size:,}} "
      f"({{ {{k: len(tok.ids_of_table(k)) for k in ('rest', 'host')}} }} pieces), max_len {{model.cfg.max_len}}; "
      f"the piece rule: {{tok.field_tables()}}")
'''),
code(r'''
# --- 4. generate -----------------------------------------------------------------------
# PROVENANCE.json is rewritten after EVERY file; a rerun keeps a file only with a
# matching record -- anything else is made again: same seed, same bytes.
import platform
import time

PROV = OUT / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"files": {}}
prov.update({"purpose": f"V2-P2 (research/v2/PREREG_V2_P2.md): {ARM}, decoded as e16_b5 plus the piece rule",
             "checkpoint_file": "final.pt", "code_commit": manifest["commit"],
             "checkpoints": CKPT_SHA, "tables": manifest["tables"], "piece_tables": C.TABLE_SHA256,
             "rules_sha256": C.rules_sha256(), "decode": C.decode_kwargs(),
             "environment": {"torch": torch.__version__, "python": platform.python_version(),
                             "gpu": GPU, "tldextract": metadata.version("tldextract")}})
t_start = time.time()
done = 0
for j in jobs:
    path = OUT / j["file"]
    rec = prov["files"].get(j["file"])
    if rec and path.is_file() and sha256(path) == rec["sha256"]:
        done += 1
        continue
    t0 = time.time()
    trace, qtrace, grec = [], [], {}
    before = rule.snapshot()
    if j["mode"] == "free":
        urls = C.free_urls(model, tok, j["label"], j["g"], shipped, presence_slash, device=device,
                           plan_trace=trace, query_trace=qtrace, rule=rule, record=grec)
    else:
        urls = C.control_urls(model, tok, j["label"], j["state"], j["g"], tables, presence_slash,
                              device=device, plan_trace=trace, query_trace=qtrace, rule=rule, record=grec)
    assert len(urls) == j["n"], (j["file"], len(urls))
    piece, rules = rule.record(since=before), grec["token_rules"]
    path.write_text(json.dumps(urls))
    prov["files"][j["file"]] = {**{k: v for k, v in j.items() if k != "file"},
                                "sha256": sha256(path), "seconds": round(time.time() - t0, 1),
                                "plan_trace_mean": sum(trace) / len(trace) if trace else None,
                                "query_moved_mean": sum(qtrace) / len(qtrace) if qtrace else None,
                                "query_rows_moved": sum(x > 0 for x in qtrace) if qtrace else None,
                                "slash_moved_mean": grec["slash_moved_mean"],
                                "slash_rows_moved": grec["slash_rows_moved"],
                                "piece_rule": piece, "token_rules": rules,
                                "cut_at_max_len": grec["cut_at_max_len"]}
    PROV.write_text(json.dumps(prov, indent=1))
    done += 1
    el = (time.time() - t_start) / 60
    print(f"  {done:>3}/{len(jobs)}  {j['file']:<46} {time.time() - t0:6.0f}s  ({el:.0f} min so far)"
          f"  violations: pieces {piece['violations']}, D1 {rules['d1_violations']}, B5 {rules['b5_violations']}",
          flush=True)
bad = {f: (r["piece_rule"]["violations"], r["token_rules"]["d1_violations"], r["token_rules"]["b5_violations"])
       for f, r in prov["files"].items()
       if r["piece_rule"]["violations"] or r["token_rules"]["d1_violations"] or r["token_rules"]["b5_violations"]}
print(f"\nall {len(jobs)} files written; violations (pieces, D1, B5): {bad or 'none'}. Now download {OUT}.")
'''),
    ]


def main() -> None:
    for arm in ("p2_a", "p2_b"):
        BASE.write(f"V2P2_train_{arm[-1]}.ipynb", train_cells(arm))
        BASE.write(f"V2P2_generate_{arm[-1]}.ipynb", generate_cells(arm))


if __name__ == "__main__":
    main()
