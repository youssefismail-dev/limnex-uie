"""Build research/e17/E17_decode.ipynb, E17_train.ipynb, E17_generate_b5.ipynb and
E17_generate_d1p.ipynb (research/e17/PREREG_E17.md section 7, step 2).

    python research/e17/make_e17_notebooks.py

All four run the `uie-e17-code` bundle (make_e17_bundle.py); every constant and call
comes from research/e17/e17_common.py, so the laptop repeats them exactly. The
structure is E16's notebooks', which ran on Kaggle.

- E17_decode        e16_b5: e16_cap's weights (E16_train's output) + D1 + D1' + B5;
                    50 files. Generation only.
- E17_train         e17_a: e16_cap's recipe with streams 1 + 2 pooled (rate 2). Before
                    training: the start check (the width-384 rebuild against e12_final's
                    record; the width-512 start against e16_cap's record) and the
                    worst-case memory check. The readback compares the run with
                    e16_cap's checkpoint and the registered injection, exactly.
- E17_generate_b5   e17_a's 50 files with B5 (if B5 is adopted) ...
- E17_generate_d1p  ... or with raw + D1 + D1' (if it is not). Run ONE of the two, as
                    the laptop's reading of e16_b5 says.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e15"))
from make_e15_notebooks import code, md, write as _write  # noqa: E402

LOCATE_CODE = r'''
# --- 1. locate the code bundle (and the inputs) --------------------------------------
import hashlib, json, math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


root = Path("/kaggle/input")
code_hits = sorted(root.rglob("E17_CODE_MANIFEST.json")) if root.exists() else []
if not code_hits:
    raise SystemExit("Attach the uie-e17-code dataset (right panel -> Add Input), then rerun.")
CODE = code_hits[0].parent
WORK = Path("/kaggle/working")
print("code:", CODE)


def output_of(arm):
    hits = sorted(p for p in root.rglob("PROVENANCE.json") if (p.parent / arm).is_dir()) if root.exists() else []
    return hits[0].parent if hits else None
'''

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
for p in (CODE / "research" / "e17", CODE / "research" / "e16", CODE / "research" / "e15",
          CODE / "research" / "e13", CODE / "research" / "native", CODE):
    sys.path.insert(0, str(p))
for name in [m for m in sys.modules if m == "uie" or m.startswith("uie.")
             or m in ("e8_common", "e11_common", "e12_common", "e13_common", "e15_common",
                      "e16_common", "e17_common")]:
    del sys.modules[name]
import uie
import e12_common as K
import e15_common as E15
import e16_common as E16
import e17_common as E17
manifest = json.loads((CODE / "E17_CODE_MANIFEST.json").read_text())
for mod in (uie, K, E15, E16, E17):
    assert str(CODE) in mod.__file__, f"{mod.__name__} is SHADOWED -- check sys.path"
bad = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
bad += [t for t, h in manifest["tables"].items() if sha256(CODE / t) != h]
if sha256(CODE / E17.STREAM_FILE) != manifest["stream"]["sha256"]:
    bad.append(E17.STREAM_FILE)
assert not bad, f"STOP. files differ from the manifest: {bad}"
print(f"code: commit {manifest['commit'][:12]}; {len(manifest['files'])} files, "
      f"{len(manifest['tables'])} tables and the pooled stream ({manifest['stream']['rows']:,} rows) verified")

import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4, then rerun.")
GPU = torch.cuda.get_device_name(0)
print("torch:", torch.__version__, "| gpu:", GPU)
'''


def generate_loop(arm_expr: str, b5: bool) -> str:
    return r'''
# --- 4. generate -----------------------------------------------------------------------
# PROVENANCE.json is rewritten after EVERY file; a rerun keeps a file only with a
# matching record -- anything else is made again: same seed, same bytes.
import platform
import time

B5 = ''' + repr(b5) + r'''
ARM = ''' + arm_expr + r'''
PROV = OUT / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"files": {}}
prov.update({"purpose": PURPOSE, "checkpoint_file": "final.pt", "code_commit": manifest["commit"],
             "checkpoints": CKPT_SHA, "tables": manifest["tables"], "decode": E17.decode_kwargs(B5),
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
    trace, qtrace = [], []
    if j["mode"] == "free":
        urls = E17.free_urls(model, tok, j["label"], j["g"], shipped, presence_slash, b5=B5,
                             device=device, plan_trace=trace, query_trace=qtrace if B5 else None)
    else:
        urls = E17.control_urls(model, tok, j["label"], j["state"], j["g"], tables, presence_slash,
                                b5=B5, device=device, plan_trace=trace,
                                query_trace=qtrace if B5 else None)
    assert len(urls) == j["n"], (j["file"], len(urls))
    path.write_text(json.dumps(urls))
    prov["files"][j["file"]] = {**{k: v for k, v in j.items() if k != "file"},
                                "sha256": sha256(path), "seconds": round(time.time() - t0, 1),
                                "plan_trace_mean": sum(trace) / len(trace) if trace else None,
                                "query_moved_mean": sum(qtrace) / len(qtrace) if qtrace else None,
                                "query_rows_moved": sum(x > 0 for x in qtrace) if qtrace else None}
    PROV.write_text(json.dumps(prov, indent=1))
    done += 1
    el = (time.time() - t_start) / 60
    print(f"  {done:>3}/{len(jobs)}  {j['file']:<46} {time.time() - t0:6.0f}s  ({el:.0f} min so far)",
          flush=True)
print(f"\nall {len(jobs)} files written. Now download {OUT}.")
'''


SETUP_GEN = r'''
# --- 3. tables, the registered jobs, the model ---------------------------------------
from uie.controlled import StateSuffixTables
from uie.suffixes import SuffixTable
from uie.train import load_checkpoint

shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
presence_slash = json.loads((CODE / "presence_slash_table.json").read_text())["table"]
device = torch.device("cuda")
OUT = WORK / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)
jobs = sorted(E17.jobs(ARM_NAME), key=lambda j: (0 if j["mode"] == "free" else 1, j["label"], j["state"] or "", j["g"]))
print(f"{len(jobs)} files, {sum(j['n'] for j in jobs):,} URLs; decode {E17.decode_kwargs(B5_DECODE)}")
CKPT_SHA = {MODEL_NAME: sha256(CKPT)}
model, tok, _ = load_checkpoint(CKPT, device=device)
model.to(device).eval()
assert tok.presence_plan and tok.root_slash and tok.slash_plan and tok.vocab_size == 110
assert model.cfg.d_model == E16.D_MODEL, model.cfg.d_model
print(f"loaded {MODEL_NAME}: width {model.cfg.d_model}, vocab {tok.vocab_size}, max_len {model.cfg.max_len}")
'''


def decode_cells():
    return [
md(r'''
# E17 — `e16_b5`: the adopted model with B5, the query-plan rule (generation only)

Registration: `research/e17/PREREG_E17.md` §2. `e16_cap`'s own weights, decoded as
always (raw + D1 + D1′) **plus B5**: when a URL's plan says it has a query, the URL may
not end before the query starts. 50 files at E12's seeds, each paired with `e16_cap`'s.

**Attach two inputs:** **`uie-e17-code`**, and the **E16_train** notebook's **output**
(it holds `e16_cap/final.pt`). **Settings:** GPU T4, Internet **On**.
**Save Version → Save & Run All.** About 2.5 h. A rerun keeps every finished file.
When done, download `/kaggle/working/e17_decode_gen`.
'''),
code(LOCATE_CODE + r'''
SRC = output_of("e16_cap")
if SRC is None:
    raise SystemExit("Attach the E16_train notebook's output too, then rerun.")
CKPT = SRC / "e16_cap" / "final.pt"
'''),
code(ENV + r'''
if sha256(CKPT) != E17.E16_CAP_SHA256:
    raise SystemExit("STOP. The attached e16_cap final.pt is not the adopted baseline.")
print("checkpoint verified:", CKPT)
'''),
code("OUT_NAME, ARM_NAME, MODEL_NAME, B5_DECODE = 'e17_decode_gen', E17.DECODE_ARM, 'e16_cap', True\n"
     "PURPOSE = \"E17 (research/e17/PREREG_E17.md): e16_b5 = e16_cap + D1 + D1' + B5; generation only\"\n"
     + SETUP_GEN),
code(generate_loop("E17.DECODE_ARM", True)),
    ]


def generate_cells(b5: bool):
    tag = "b5" if b5 else "d1p"
    title = ("with B5 (raw + D1 + D1′ + B5) — run this one **only if the laptop says B5 was adopted**"
             if b5 else "without B5 (raw + D1 + D1′) — run this one **only if the laptop says B5 was not adopted**")
    return [
md(rf'''
# E17 — `e17_a`'s generation {title}

Registration: `research/e17/PREREG_E17.md` §3. Run **after the laptop has verified
`e17_a`'s training**. 50 files at E12's seeds.

**Attach two inputs:** **`uie-e17-code`**, and the **E17_train** notebook's **output**.
**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About 2.5 h.
When done, download `/kaggle/working/e17_gen`.
'''),
code(LOCATE_CODE + r'''
SRC = output_of("e17_a")
if SRC is None:
    raise SystemExit("Attach the E17_train notebook's output too, then rerun.")
tprov = json.loads((SRC / "PROVENANCE.json").read_text())
CKPT = SRC / "e17_a" / "final.pt"
assert CKPT.is_file() and sha256(CKPT) == tprov["arms"]["e17_a"]["final_sha256"], \
    "e17_a: final.pt is not the one E17_train recorded"
print("checkpoint verified:", CKPT)
'''),
code(ENV),
code(f"OUT_NAME, ARM_NAME, MODEL_NAME, B5_DECODE = 'e17_gen', E17.TRAIN_ARM, 'e17_a', {b5!r}\n"
     f"PURPOSE = \"E17 (research/e17/PREREG_E17.md): e17_a, decode {tag}; generation only\"\n"
     + SETUP_GEN),
code(generate_loop("E17.TRAIN_ARM", b5)),
    ]


def train_cells():
    return [
md(r'''
# E17 — training `e17_a`: the adopted recipe with both new query streams (option A)

Registration: `research/e17/PREREG_E17.md` §3 (amendment 1). Everything is `e16_cap`'s:
the width-512 model and its start, the data, the seed and batches, 20,000 steps, the
learning rate and objective, the two-part batches. **One change: stream 2 joins stream 1**,
so each old query row of the two benign strata now brings **two** new rows (each
teaching only its query) instead of one.

**Before any training, cell 3 checks the start** (the registered procedure must rebuild
`e12_final`'s recorded start, and `e16_cap`'s recorded width-512 start) **and that the
worst-case batch fits.** **Cell 5 reads the run back:** it must print
`MISMATCHES: none -- e17_a is as registered`.

**Attach three inputs:** **`uie-corpus2`**, **`uie-e17-code`**, and the **E16_train**
notebook's **output**. **Settings:** GPU T4, Internet **On**.
**Save Version → Save & Run All.** About 3.5 h. **A rerun skips a finished arm, and never
resumes a half-finished one.** When done, download `/kaggle/working`.
'''),
code(LOCATE_CODE + r'''
data_hits = sorted(root.rglob("bundle.manifest.json")) if root.exists() else []
if not data_hits:
    raise SystemExit("Attach the uie-corpus2 dataset too, then rerun.")
DATA = data_hits[0].parent
SRC = output_of("e16_cap")
if SRC is None:
    raise SystemExit("Attach the E16_train notebook's output too, then rerun.")
print("data:", DATA, "| e16_cap from:", SRC)
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
if sha256(SRC / "e16_cap" / "final.pt") != E17.E16_CAP_SHA256:
    raise SystemExit("STOP. The attached e16_cap final.pt is not the adopted baseline.")
print("data bundle: all 8 streams present; E17 options present; e16_cap verified")
'''),
code(r'''
# --- 3. THE START CHECK and the worst-case memory check, before training ---------------
from uie.tokenizer import URLTokenizer
base = torch.load(SRC / "e16_cap" / "final.pt", map_location="cpu", weights_only=False)
tok = URLTokenizer.from_state_dict(base["tokenizer"])
ref = E16.start_fingerprints(tok, 384)
want = {"init_weights_sha256": E15.E12_FINAL_INIT, **E15.E12_FINAL_START}
off = {k: (ref.get(k), v) for k, v in want.items() if ref.get(k) != v}
if off:
    raise SystemExit(f"STOP. This torch does not rebuild e12_final's recorded start: {off}. Nothing was trained.")
START = E16.start_fingerprints(tok, E16.D_MODEL)
off = {k: (START.get(k), v) for k, v in E17.E16_START.items() if START.get(k) != v}
if off or START != E16.start_fingerprints(tok, E16.D_MODEL):
    raise SystemExit(f"STOP. The width-512 start is not e16_cap's recorded start: {off}. Nothing was trained.")
print(f"width 384 rebuilds e12_final's recorded start; width 512 rebuilds e16_cap's ({START['init_weights_sha256'][:12]}...)")
total = torch.cuda.get_device_properties(0).total_memory / 2**30
ACCUM, peak = None, None
for parts in (E16.ACCUM_STEPS, E16.ACCUM_FALLBACK):
    fits = True
    try:
        # 256 rows plus a margin for the appended new rows (about 9.5 per batch), split
        peak = K.memory_preflight(torch.device("cuda"), batch_size=math.ceil(288 / parts),
                                  d_model=E16.D_MODEL)
    except torch.cuda.OutOfMemoryError:
        fits = False
    if fits:
        ACCUM = parts
        break
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {parts} parts: the worst-case step does not fit")
if ACCUM is None:
    raise SystemExit("STOP. The worst-case step does not fit even in 4 parts. Nothing was trained.")
RECIPE = E17.recipe_a(CODE / E17.STREAM_FILE, accum_steps=ACCUM)
print(f"start check PASSED; memory check PASSED in {ACCUM} parts ({peak:.1f} of {total:.1f} GB)")
print("e17_a:", RECIPE)
'''),
code(r'''
# --- 4. train e17_a (skipped only if finished) ---------------------------------------
import time

PROV = WORK / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"arms": {}}
prov.update({"registration": E17.REGISTRATION, "code_commit": manifest["commit"], "gpu": GPU,
             "torch": torch.__version__, "stream_sha256": manifest["stream"]["sha256"],
             "start_check": {"width_384": ref, "width_512": START},
             "memory": {"accum_steps": ACCUM, "peak_gb": round(peak, 2), "total_gb": round(total, 2)}})
PROV.write_text(json.dumps(prov, indent=1))
out = WORK / E17.TRAIN_ARM
if (out / "train_result.json").exists() and (out / "final.pt").exists():
    print(f"{E17.TRAIN_ARM}: finished earlier -- skipped")
else:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run(DATA, out, **RECIPE)
    prov["arms"][E17.TRAIN_ARM] = {"wall_minutes": round((time.time() - t0) / 60, 1)}
    PROV.write_text(json.dumps(prov, indent=1))
'''),
code(r'''
# --- 5. read e17_a back against e16_cap's checkpoint and amendment 1 -------------------
ck = torch.load(WORK / E17.TRAIN_ARM / "final.pt", map_location="cpu", weights_only=False)
tc, mc, tk = ck["train_config"], ck["model_config"], ck["tokenizer"]
btc, bmc = base["train_config"], base["model_config"]
r = json.loads((WORK / E17.TRAIN_ARM / "train_result.json").read_text())
s = r.get("e12_start") or {}
q = r.get("qry_extra") or {}
strata = q.get("strata", {})
want = []
for k, v in btc.items():
    if k == "qry_extra":
        want.append(("train_config qry_extra (file)", Path(tc.get(k, "")).name, E17.STREAM_FILE))
    elif k == "accum_steps":
        want.append(("train_config accum_steps", tc.get(k), ACCUM))
    else:
        want.append((f"train_config {k}", tc.get(k), v))
want += [("extra settings", sorted(set(tc) - set(btc)), [])]
want += [(f"model_config {k}", mc.get(k), v) for k, v in bmc.items()]
want += [("tokenizer", tk == base["tokenizer"], True),
         ("parameters", r.get("params"), E16.PARAMS),
         ("init weights (V-start)", r.get("init_weights_sha256"), E17.E16_START["init_weights_sha256"]),
         ("row counts", r.get("counts"), E16.COUNTS),
         ("dropped over max_len", r.get("dropped_over_max_len"), E16.DROPPED),
         ("loss constant", r.get("loss_constant"), E16.LOSS_CONSTANT),
         ("stream sha256", q.get("source_sha256"), manifest["stream"]["sha256"]),
         ("N_new", {k: v["N_new"] for k, v in strata.items()}, E17.N_NEW),
         ("N_old", {k: v["N_old"] for k, v in strata.items()}, E17.N_OLD),
         ("rate", {k: v.get("rate") for k, v in strata.items()}, {k: E17.RATE for k in E17.N_NEW}),
         ("old rows' batch digest (V-batch)", q.get("old_batch_digest"), E17.V_BATCH_DIGEST),
         ("old rows drawn (e16_cap's)", {k: v["old_rows_seen"] for k, v in strata.items()}, E17.OLD_ROWS_SEEN),
         ("appended rows (V-mass: twice the old)", {k: v["appended_rows"] for k, v in strata.items()}, E17.APPENDED),
         ("finite validation", all(math.isfinite(x) for x in (r.get("final") or {}).values()), True)]
want += [(f"start: {k}", s.get(k), E17.E16_START[k]) for k in
         ("e10_start_sha256", "e11_start_sha256", "new_token_rows_sha256", "new_position_rows_sha256")]
problems = [w for w in want if w[1] != w[2]]
positions = sum(v["appended_query_positions"] for v in strata.values())
clip = {k: v["clipped_share"] for k, v in (r.get("grad_diagnostics") or {}).items() if k.startswith("after_1000")}
prov["arms"].setdefault(E17.TRAIN_ARM, {})
prov["arms"][E17.TRAIN_ARM].update({"best_sha256": sha256(WORK / E17.TRAIN_ARM / "best.pt"),
                                    "final_sha256": sha256(WORK / E17.TRAIN_ARM / "final.pt"),
                                    "init_weights_sha256": r.get("init_weights_sha256"),
                                    "qry_extra": q, "final_val": r.get("final"), "counts": r.get("counts"),
                                    "clipped_after_1000": clip})
PROV.write_text(json.dumps(prov, indent=1))
print(f"{E17.TRAIN_ARM} val {r.get('final')}  (e16_cap's: benign 1.6067, phishing 1.3713, mean 1.4890)")
print(f"appended query positions {positions:,} (P8: about {E17.ADDED_QUERY_POSITIONS_PER_BATCH * 20000:,.0f})")
print(f"clipped after step 1,000 (K12e: at most 0.01 in every band): {clip}")
print("MISMATCHES:", problems if problems else f"none -- {E17.TRAIN_ARM} is as registered")
print("\nNow download /kaggle/working (e17_a/, PROVENANCE.json).")
'''),
    ]


def write(name, cells):
    """E15's writer, aimed at this folder."""
    import make_e15_notebooks as M
    old = M.HERE
    try:
        M.HERE = HERE
        _write(name, cells)
    finally:
        M.HERE = old


def main() -> None:
    write("E17_decode.ipynb", decode_cells())
    write("E17_train.ipynb", train_cells())
    write("E17_generate_b5.ipynb", generate_cells(True))
    write("E17_generate_d1p.ipynb", generate_cells(False))


if __name__ == "__main__":
    main()
