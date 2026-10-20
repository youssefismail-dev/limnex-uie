"""Build research/e16/E16_train.ipynb and E16_generate.ipynb
(research/e16/PREREG_E16.md section 11, step 2; amendments 1-2).

    python research/e16/make_e16_notebooks.py

Both run the `uie-e16-code` bundle (make_e16_bundle.py). Every constant and every call
comes from research/e16/e16_common.py, so the laptop repeats them exactly. The structure
is E15's notebooks', which ran on Kaggle once each.

- E16_train     e16_cap: e15_new's recipe at d_model 512, each batch in 2 parts. Before
                training: the start check (amendment 2) and the worst-case memory check
                (with the registered fallback to 4 parts). The readback compares the run
                with e15_new's record and the injection with e15_new's, exactly.
- E16_generate  e16_cap's 50 files (raw + D1 + D1'), after the laptop verifies training.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e15"))
from make_e15_notebooks import GENERATE_LOOP, code, md, write as _write  # noqa: E402

LOCATE_CODE = r'''
# --- 1. locate the code bundle -----------------------------------------------------
import hashlib, json, math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


root = Path("/kaggle/input")
code_hits = sorted(root.rglob("E16_CODE_MANIFEST.json")) if root.exists() else []
if not code_hits:
    raise SystemExit("Attach the uie-e16-code dataset (right panel -> Add Input), then rerun.")
CODE = code_hits[0].parent
WORK = Path("/kaggle/working")
print("code:", CODE)
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
for p in (CODE / "research" / "e16", CODE / "research" / "e15", CODE / "research" / "e13",
          CODE / "research" / "native", CODE):
    sys.path.insert(0, str(p))
for name in [m for m in sys.modules if m == "uie" or m.startswith("uie.")
             or m in ("e8_common", "e11_common", "e12_common", "e13_common", "e15_common",
                      "e16_common")]:
    del sys.modules[name]
import uie
import e12_common as K
import e15_common as E15
import e16_common as E16
manifest = json.loads((CODE / "E16_CODE_MANIFEST.json").read_text())
for mod in (uie, K, E15, E16):
    assert str(CODE) in mod.__file__, f"{mod.__name__} is SHADOWED -- check sys.path"
bad = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
bad += [t for t, h in manifest["tables"].items() if sha256(CODE / t) != h]
if sha256(CODE / "e15_new.pt") != manifest["e15_new"]:
    bad.append("e15_new.pt")
if sha256(CODE / E16.STREAM_FILE) != manifest["stream"]["sha256"]:
    bad.append(E16.STREAM_FILE)
assert not bad, f"STOP. files differ from the manifest: {bad}"
print(f"code: commit {manifest['commit'][:12]}; {len(manifest['files'])} files, "
      f"{len(manifest['tables'])} tables, e15_new.pt and the stream "
      f"({manifest['stream']['rows']:,} rows) verified")

import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4, then rerun.")
GPU = torch.cuda.get_device_name(0)
print("torch:", torch.__version__, "| gpu:", GPU)
'''


def train_cells():
    return [
md(r'''
# E16 — training `e16_cap`: `e15_new`'s recipe in a wider model (the capacity probe)

Registration: `research/e16/PREREG_E16.md` (amendments 1–2). Everything is `e15_new`'s:
the data and the E15 stream, the seed and batches, 20,000 steps, the learning rate,
the objective and its pinned constant. **One change: the width,** 512 instead of 384
(25.5M parameters instead of 14.4M). To fit the GPU's memory, each batch is processed
in two halves whose gradients are added before the step: the same batch, the same
gradient.

**Before any training, cell 3 checks the start** (the registered procedure must rebuild
`e12_final`'s recorded start on this torch, and the wider start must build identically
twice) **and that the worst-case batch fits.** **Cell 5 reads the run back:** it must
print `MISMATCHES: none -- e16_cap is as registered`.

**Attach two inputs** (right panel → *Add Input*):
1. **`uie-corpus2`**, the data bundle E7–E15 used;
2. **`uie-e16-code`**.

**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About 3.5 h.
**A rerun skips a finished arm, and never resumes a half-finished one.** When done,
download `/kaggle/working`.
'''),
code(LOCATE_CODE + r'''
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
assert {"accum_steps", "qry_extra", "slash_plan", "max_len", "e12_start", "loss_constant",
        "d_model"} <= set(inspect.signature(run).parameters)
print("data bundle: all 8 streams present; E16 option present")
'''),
code(r'''
# --- 3. THE START CHECK (amendment 2) and the worst-case memory check, before training ---
from uie.tokenizer import URLTokenizer
base = torch.load(CODE / "e15_new.pt", map_location="cpu", weights_only=False)
tok = URLTokenizer.from_state_dict(base["tokenizer"])
ref = E16.start_fingerprints(tok, 384)
want = {"init_weights_sha256": E15.E12_FINAL_INIT, **E15.E12_FINAL_START}
off = {k: (ref.get(k), v) for k, v in want.items() if ref.get(k) != v}
if off:
    raise SystemExit(f"STOP. This torch does not rebuild e12_final's recorded start: {off}. "
                     "Nothing was trained.")
print(f"width 384 rebuilds e12_final's recorded start exactly ({E15.E12_FINAL_INIT[:12]}...)")
START = E16.start_fingerprints(tok, E16.D_MODEL)
if START != E16.start_fingerprints(tok, E16.D_MODEL) or START["params"] != E16.PARAMS:
    raise SystemExit("STOP. The width-512 start is not deterministic, or not 25,473,024 "
                     "parameters. Nothing was trained.")
print(f"width 512 start: {START['init_weights_sha256'][:12]}... ({START['params']:,} parameters), "
      "built twice, identical")
total = torch.cuda.get_device_properties(0).total_memory / 2**30
ACCUM, peak = None, None
for parts in (E16.ACCUM_STEPS, E16.ACCUM_FALLBACK):
    fits = True
    try:
        # 256 rows plus a margin for the appended new rows, split into `parts`
        peak = K.memory_preflight(torch.device("cuda"), batch_size=math.ceil(272 / parts),
                                  d_model=E16.D_MODEL)
    except torch.cuda.OutOfMemoryError:
        fits = False
    if fits:
        ACCUM = parts
        break
    import gc
    gc.collect()                    # outside the handler, so the failed step's tensors are free
    torch.cuda.empty_cache()
    print(f"  {parts} parts: the worst-case step does not fit")
if ACCUM is None:
    raise SystemExit("STOP. The worst-case step does not fit even in 4 parts. Nothing was trained.")
RECIPE = E16.recipe_cap(CODE / E16.STREAM_FILE, accum_steps=ACCUM)
print(f"start check PASSED; memory check PASSED in {ACCUM} parts ({peak:.1f} of {total:.1f} GB)")
print("e16_cap:", RECIPE)
'''),
code(r'''
# --- 4. train e16_cap (skipped only if finished) -------------------------------------
import time

PROV = WORK / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"arms": {}}
prov.update({"registration": E16.REGISTRATION, "code_commit": manifest["commit"], "gpu": GPU,
             "torch": torch.__version__, "stream_sha256": manifest["stream"]["sha256"],
             "start_check": {"width_384": ref, "width_512": START},
             "memory": {"accum_steps": ACCUM, "peak_gb": round(peak, 2), "total_gb": round(total, 2)}})
PROV.write_text(json.dumps(prov, indent=1))
out = WORK / E16.ARM
if (out / "train_result.json").exists() and (out / "final.pt").exists():
    print(f"{E16.ARM}: finished earlier -- skipped")
else:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run(DATA, out, **RECIPE)
    prov["arms"][E16.ARM] = {"wall_minutes": round((time.time() - t0) / 60, 1)}
    PROV.write_text(json.dumps(prov, indent=1))
'''),
code(r'''
# --- 5. read e16_cap back against e15_new's record ------------------------------------
ck = torch.load(WORK / E16.ARM / "final.pt", map_location="cpu", weights_only=False)
tc, mc, tk = ck["train_config"], ck["model_config"], ck["tokenizer"]
btc, bmc = base["train_config"], base["model_config"]
r = json.loads((WORK / E16.ARM / "train_result.json").read_text())
s = r.get("e12_start") or {}
q = r.get("qry_extra") or {}
strata = q.get("strata", {})
want = []
for k, v in btc.items():
    if k == "qry_extra":                       # the stream's path differs between bundles
        want.append(("train_config qry_extra (file)", Path(tc.get(k, "")).name, E16.STREAM_FILE))
    else:
        want.append((f"train_config {k}", tc.get(k), v))
want += [("extra settings", sorted(set(tc) - set(btc)), ["accum_steps"]),
         ("accum_steps", (tc.get("accum_steps"), r.get("accum_steps")), (ACCUM, ACCUM))]
for k, v in bmc.items():
    target = {"d_model": E16.D_MODEL, "d_ff": 4 * E16.D_MODEL}.get(k, v)
    want.append((f"model_config {k}", mc.get(k), target))
want += [("tokenizer", tk == base["tokenizer"], True),
         ("parameters", r.get("params"), E16.PARAMS),
         ("init weights (V-start')", r.get("init_weights_sha256"), START["init_weights_sha256"]),
         ("row counts", r.get("counts"), E16.COUNTS),
         ("dropped over max_len", r.get("dropped_over_max_len"), E16.DROPPED),
         ("loss constant", r.get("loss_constant"), E16.LOSS_CONSTANT),
         ("stream sha256", q.get("source_sha256"), manifest["stream"]["sha256"]),
         ("N_new", {k: v["N_new"] for k, v in strata.items()}, E15.N_NEW),
         ("N_old", {k: v["N_old"] for k, v in strata.items()}, E15.N_OLD),
         ("p", {k: v["p"] for k, v in strata.items()}, E15.P_INJECT),
         ("old rows' batch digest (V-batch)", q.get("old_batch_digest"), E16.V_BATCH_DIGEST),
         ("injection, exactly e15_new's (V-mass')",
          {k: {f: v[f] for f in ("old_rows_seen", "appended_rows", "appended_query_positions")}
           for k, v in strata.items()}, E16.E15_NEW_INJECTION),
         ("finite validation", all(math.isfinite(x) for x in (r.get("final") or {}).values()), True)]
want += [(f"start: {k}", s.get(k), START[k]) for k in
         ("e10_start_sha256", "e11_start_sha256", "new_token_rows_sha256", "new_position_rows_sha256")]
problems = [w for w in want if w[1] != w[2]]
clip = {k: v["clipped_share"] for k, v in (r.get("grad_diagnostics") or {}).items()
        if k.startswith("after_1000")}
prov["arms"].setdefault(E16.ARM, {})
prov["arms"][E16.ARM].update({"best_sha256": sha256(WORK / E16.ARM / "best.pt"),
                              "final_sha256": sha256(WORK / E16.ARM / "final.pt"),
                              "init_weights_sha256": r.get("init_weights_sha256"),
                              "qry_extra": q, "final_val": r.get("final"), "counts": r.get("counts"),
                              "clipped_after_1000": clip})
PROV.write_text(json.dumps(prov, indent=1))
print(f"{E16.ARM} val {r.get('final')}  (e15_new's: benign 1.5608, phishing 1.4004, mean 1.4806)")
print(f"clipped after step 1,000 (K12e: at most 0.01 in every band): {clip}")
print("MISMATCHES:", problems if problems else f"none -- {E16.ARM} is as registered")
print("\nNow download /kaggle/working (e16_cap/, PROVENANCE.json).")
'''),
    ]


def generate_cells():
    loop = GENERATE_LOOP.replace("E15.free_urls", "E16.free_urls").replace("E15.control_urls", "E16.control_urls")
    return [
md(r'''
# E16 — `e16_cap`'s generation, from `final.pt` (nothing is scored here)

Registration: `research/e16/PREREG_E16.md` §§3, 11. Run this **after the laptop has
verified the training**. It writes 50 files at E12's seeds with the adopted decode
(raw + D1 + D1′), so each cell pairs with `e15_new`'s files, the official baseline.

**Attach two inputs:** **`uie-e16-code`**, and the **E16_train** notebook's **output**.
**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About 3–3.5 h.
A rerun keeps every finished file with its record. When done, download
`/kaggle/working/e16_gen`.
'''),
code(LOCATE_CODE + r'''
train_prov = sorted(p for p in root.rglob("PROVENANCE.json")
                    if (p.parent / "e16_cap").is_dir()) if root.exists() else []
if not train_prov:
    raise SystemExit("Attach the E16_train notebook's output too, then rerun.")
TRAIN = train_prov[0].parent
tprov = json.loads(train_prov[0].read_text())
CKPT = TRAIN / "e16_cap" / "final.pt"
assert CKPT.is_file() and sha256(CKPT) == tprov["arms"]["e16_cap"]["final_sha256"], \
    "e16_cap: final.pt is not the one E16_train recorded"
print("checkpoint verified:", CKPT)
'''),
code(ENV),
code(r'''
# --- 3. tables, the registered jobs, the model ---------------------------------------
from uie.controlled import StateSuffixTables
from uie.suffixes import SuffixTable
from uie.train import load_checkpoint

shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
presence_slash = json.loads((CODE / "presence_slash_table.json").read_text())["table"]
device = torch.device("cuda")
OUT = WORK / "e16_gen"
OUT.mkdir(parents=True, exist_ok=True)
jobs = sorted(E16.jobs(), key=lambda j: (0 if j["mode"] == "free" else 1, j["label"], j["state"] or "", j["g"]))
print(f"{len(jobs)} files, {sum(j['n'] for j in jobs):,} URLs; decode {E16.decode_kwargs()}")
CKPT_SHA = {"e16_cap": sha256(CKPT)}
PURPOSE = "E16 (research/e16/PREREG_E16.md): e16_cap, raw + D1 + D1'; generation only"
model, tok, _ = load_checkpoint(CKPT, device=device)
model.to(device).eval()
assert tok.presence_plan and tok.root_slash and tok.slash_plan and tok.vocab_size == 110
assert model.cfg.d_model == E16.D_MODEL, model.cfg.d_model
print(f"loaded e16_cap: width {model.cfg.d_model}, vocab {tok.vocab_size}, max_len {model.cfg.max_len}")
'''),
code(loop),
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
    write("E16_train.ipynb", train_cells())
    write("E16_generate.ipynb", generate_cells())


if __name__ == "__main__":
    main()
