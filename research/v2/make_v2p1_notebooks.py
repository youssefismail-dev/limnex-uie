"""Build research/v2/V2P1_train.ipynb and research/v2/V2P1_generate.ipynb
(research/v2/PREREG_V2_P1_MEMORY.md section 10, step 2).

    python research/v2/make_v2p1_notebooks.py

Both run the `uie-v2p1-code` bundle (make_v2p1_bundle.py); every constant and call comes
from research/v2/v2_p1_common.py, so the laptop repeats them exactly. The structure is
E17's notebooks', which ran on Kaggle.

- V2P1_train      p1_mem: e16_cap's recipe plus the memory layer. Before training: the
                  start check (the width-384 rebuild against e12_final's record, the
                  width-512 rebuild against e16_cap's, the memory's start built twice) and
                  the worst-case memory check. The readback compares the run with
                  e16_cap's settings (carried in the bundle) and its injection, exactly.
- V2P1_generate   p1_mem's 50 files with e16_b5's decode, after the laptop has verified
                  training.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "e15"))
from make_e15_notebooks import code, md, write as _write  # noqa: E402

LOCATE_CODE = r'''
# --- 1. locate the code bundle (and the inputs) --------------------------------------
import gc, hashlib, json, math
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


root = Path("/kaggle/input")
code_hits = sorted(root.rglob("V2P1_CODE_MANIFEST.json")) if root.exists() else []
if not code_hits:
    raise SystemExit("Attach the uie-v2p1-code dataset (right panel -> Add Input), then rerun.")
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
for p in (CODE / "research" / "v2", CODE / "research" / "e17", CODE / "research" / "e16",
          CODE / "research" / "e15", CODE / "research" / "e13", CODE / "research" / "native", CODE):
    sys.path.insert(0, str(p))
for name in [m for m in sys.modules if m == "uie" or m.startswith("uie.")
             or m in ("e8_common", "e11_common", "e12_common", "e13_common", "e15_common",
                      "e16_common", "e17_common", "v2_mem", "v2_p1_common", "v2_p1_train")]:
    del sys.modules[name]
import uie
import e12_common as K
import e15_common as E15
import e16_common as E16
import e17_common as E17
import v2_mem as M
import v2_p1_common as P1
import v2_p1_train as T
manifest = json.loads((CODE / P1.CODE_MANIFEST).read_text())
for mod in (uie, K, E15, E16, E17, M, P1, T):
    assert str(CODE) in mod.__file__, f"{mod.__name__} is SHADOWED -- check sys.path"
bad = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
bad += [t for t, h in manifest["tables"].items() if sha256(CODE / t) != h]
if sha256(CODE / P1.STREAM_FILE) != manifest["stream"]["sha256"] or manifest["stream"]["sha256"] != P1.STREAM_SHA256:
    bad.append(P1.STREAM_FILE)
if sha256(CODE / P1.E16_CAP_RECORD) != manifest["e16_cap_record"]:
    bad.append(P1.E16_CAP_RECORD)
assert not bad, f"STOP. files differ from the manifest: {bad}"
print(f"code: commit {manifest['commit'][:12]}; {len(manifest['files'])} files, "
      f"{len(manifest['tables'])} tables, stream 1 ({manifest['stream']['rows']:,} rows) and "
      f"e16_cap's record verified")

import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4, then rerun.")
GPU = torch.cuda.get_device_name(0)
print("torch:", torch.__version__, "| gpu:", GPU)
'''


def train_cells():
    return [
md(r'''
# V2-P1 — training `p1_mem`: `e16_cap` plus a learned memory layer

Registration: `research/v2/PREREG_V2_P1_MEMORY.md` (§3). Everything is `e16_cap`'s: the
width-512 network and its start, the data and stream 1, the seed and batches, 20,000
steps, the learning rate and objective, the two-part batches. **One change: a product-key
memory layer after block 4** (262,144 learned slots; each letter reads 128 of them). No
word list, no lookup of strings: the slots are learned vectors, and the engine still writes
letter by letter.

**Before any training, cell 3 checks the start** (the registered procedure must rebuild
`e12_final`'s and `e16_cap`'s recorded starts; the memory's start is built twice and must
be identical) **and that the worst-case batch fits.** **Cell 5 reads the run back:** it must
print `MISMATCHES: none -- p1_mem is as registered`.

**Attach two inputs:** **`uie-corpus2`** and **`uie-v2p1-code`**. **Settings:** GPU T4,
Internet **On**. **Save Version → Save & Run All.** About 4–4.5 h. **A rerun skips a
finished arm, and never resumes a half-finished one.** When done, download
`/kaggle/working/p1_mem/final.pt`, `/kaggle/working/p1_mem/train_result.json` and
`/kaggle/working/PROVENANCE.json` (`best.pt` is not needed).
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
assert {"accum_steps", "qry_extra", "d_model", "e12_start", "loss_constant"} <= set(inspect.signature(run).parameters)
REC = json.loads((CODE / P1.E16_CAP_RECORD).read_text())
print("data bundle: all 8 streams present; e16_cap's record loaded")
'''),
code(r'''
# --- 3. THE START CHECK and the worst-case memory check, before training ---------------
from uie.tokenizer import URLTokenizer
tok = URLTokenizer.from_state_dict(REC["tokenizer"])
ref = E16.start_fingerprints(tok, 384)
want = {"init_weights_sha256": E15.E12_FINAL_INIT, **E15.E12_FINAL_START}
off = {k: (ref.get(k), v) for k, v in want.items() if ref.get(k) != v}
if off:
    raise SystemExit(f"STOP. This torch does not rebuild e12_final's recorded start: {off}. Nothing was trained.")
START = E16.start_fingerprints(tok, E16.D_MODEL)
off = {k: (START.get(k), v) for k, v in P1.E16_START.items() if START.get(k) != v}
if off:
    raise SystemExit(f"STOP. The width-512 start is not e16_cap's recorded start: {off}. Nothing was trained.")
m1, MEM_START = P1.start(tok)
del m1
gc.collect()
m2, again = P1.start(tok)
del m2
gc.collect()
if again != MEM_START:
    raise SystemExit("STOP. The memory's start is not deterministic. Nothing was trained.")
if MEM_START["base_init_sha256"] != P1.E16_START["init_weights_sha256"] or MEM_START["params"] != P1.PARAMS:
    raise SystemExit(f"STOP. p1_mem's start is not e16_cap's start plus the memory: {MEM_START}. Nothing was trained.")
print(f"width 384 rebuilds e12_final's recorded start; width 512 rebuilds e16_cap's ({START['init_weights_sha256'][:12]}...)")
print(f"p1_mem's start: e16_cap's part {MEM_START['base_init_sha256'][:12]}..., memory {MEM_START['memory_init_sha256'][:12]}... "
      f"(seed {MEM_START['memory_seed']}), whole {MEM_START['init_weights_sha256'][:12]}...; {MEM_START['params']:,} parameters")
total = torch.cuda.get_device_properties(0).total_memory / 2**30
ACCUM, peak = None, None
for parts in (P1.ACCUM_STEPS, P1.ACCUM_FALLBACK):
    fits = True
    try:
        # 256 rows plus a margin for the appended stream rows (about 4.7 per batch), split
        peak = M.memory_preflight(torch.device("cuda"), math.ceil(288 / parts))
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
RECIPE = P1.recipe(CODE / P1.STREAM_FILE, accum_steps=ACCUM)
print(f"start check PASSED; memory check PASSED in {ACCUM} parts ({peak:.1f} of {total:.1f} GB)")
print("p1_mem:", RECIPE, "+ memory", P1.MEM)
'''),
code(r'''
# --- 4. train p1_mem (skipped only if finished) ----------------------------------------
import time

PROV = WORK / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"arms": {}}
prov.update({"registration": P1.REGISTRATION, "code_commit": manifest["commit"], "gpu": GPU,
             "torch": torch.__version__, "stream_sha256": manifest["stream"]["sha256"],
             "start_check": {"width_384": ref, "width_512": START, "p1_mem": MEM_START},
             "memory": {"accum_steps": ACCUM, "peak_gb": round(peak, 2), "total_gb": round(total, 2)}})
PROV.write_text(json.dumps(prov, indent=1))
out = WORK / P1.ARM
if (out / "train_result.json").exists() and (out / "final.pt").exists():
    print(f"{P1.ARM}: finished earlier -- skipped")
else:
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    T.run(DATA, out, **RECIPE)
    prov["arms"][P1.ARM] = {"wall_minutes": round((time.time() - t0) / 60, 1),
                            "peak_gb_training": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    PROV.write_text(json.dumps(prov, indent=1))
'''),
code(r'''
# --- 5. read p1_mem back against e16_cap's record and the registration ------------------
ck = torch.load(WORK / P1.ARM / "final.pt", map_location="cpu", weights_only=False)
tc, mc = json.loads(json.dumps(ck["train_config"])), json.loads(json.dumps(ck["model_config"]))
tk = json.loads(json.dumps(ck["tokenizer"]))
btc, bmc = REC["train_config"], REC["model_config"]
r = json.loads((WORK / P1.ARM / "train_result.json").read_text())
s = r.get("e12_start") or {}
q = r.get("qry_extra") or {}
strata = q.get("strata", {})
inj = ("old_rows_seen", "appended_rows", "appended_query_positions")
want = []
for k, v in btc.items():
    if k == "qry_extra":
        want.append(("train_config qry_extra (file)", Path(tc.get(k, "")).name, P1.STREAM_FILE))
    elif k == "accum_steps":
        want.append(("train_config accum_steps", tc.get(k), ACCUM))
    else:
        want.append((f"train_config {k}", tc.get(k), v))
want += [("extra train settings", sorted(set(tc) - set(btc)), [])]
want += [(f"model_config {k}", mc.get(k), v) for k, v in bmc.items()]
want += [("memory settings", {k: mc.get(k) for k in P1.MEM}, P1.MEM),
         ("memory pad id", mc.get("mem_pad_id"), tok.pad_id),
         ("extra model settings", sorted(set(mc) - set(bmc) - set(P1.MEM) - {"mem_pad_id"}), [])]
want += [("tokenizer", tk == REC["tokenizer"], True),
         ("parameters", r.get("params"), P1.PARAMS),
         ("start: e16_cap's part (V-start)", s.get("base_init_sha256"), P1.E16_START["init_weights_sha256"]),
         ("start: the memory's part", s.get("memory_init_sha256"), MEM_START["memory_init_sha256"]),
         ("start: the memory's seed", s.get("memory_seed"), P1.P1_SEED),
         ("start: the whole (as training recorded it)", r.get("init_weights_sha256"), MEM_START["init_weights_sha256"]),
         ("row counts", r.get("counts"), P1.COUNTS),
         ("dropped over max_len", r.get("dropped_over_max_len"), P1.DROPPED),
         ("loss constant", r.get("loss_constant"), P1.LOSS_CONSTANT),
         ("stream sha256 (V-stream)", q.get("source_sha256"), P1.STREAM_SHA256),
         ("old rows' batch digest (V-batch)", q.get("old_batch_digest"), P1.V_BATCH_DIGEST),
         ("injection (V-mass: e16_cap's exactly)", {k: {f: v[f] for f in inj} for k, v in strata.items()},
          {k: {f: v[f] for f in inj} for k, v in P1.INJECTION.items()}),
         ("finite validation", all(math.isfinite(x) for x in (r.get("final") or {}).values()), True)]
want += [(f"start: {k}", s.get(k), P1.E16_START[k]) for k in
         ("e10_start_sha256", "e11_start_sha256", "new_token_rows_sha256", "new_position_rows_sha256")]
problems = [w for w in want if w[1] != w[2]]
clip = {k: v["clipped_share"] for k, v in (r.get("grad_diagnostics") or {}).items() if k.startswith("after_1000")}
prov["arms"].setdefault(P1.ARM, {})
prov["arms"][P1.ARM].update({"final_sha256": sha256(WORK / P1.ARM / "final.pt"),
                             "init_weights_sha256": r.get("init_weights_sha256"),
                             "e12_start": s, "qry_extra": q, "final_val": r.get("final"),
                             "counts": r.get("counts"), "params": r.get("params"),
                             "clipped_after_1000": clip, "train_seconds": r.get("seconds")})
PROV.write_text(json.dumps(prov, indent=1))
print(f"{P1.ARM} val {r.get('final')}  (e16_cap's: {P1.E16_CAP_FINAL_VAL})")
print(f"trained in {r.get('seconds', 0) / 3600:.2f} h (e16_cap's loop: 2.73 h)")
print(f"clipped after step 1,000 (K12e: at most 0.01 in every band): {clip}")
print("MISMATCHES:", problems if problems else f"none -- {P1.ARM} is as registered")
print("\nNow download /kaggle/working/p1_mem/final.pt, /kaggle/working/p1_mem/train_result.json "
      "and /kaggle/working/PROVENANCE.json (best.pt is not needed).")
'''),
    ]


def generate_cells():
    return [
md(r'''
# V2-P1 — `p1_mem`'s generation: 50 files with `e16_b5`'s decode

Registration: `research/v2/PREREG_V2_P1_MEMORY.md` §3.4. Run **only after the laptop has
verified `p1_mem`'s training** (VERIFY_TRAIN and K-used). E12's 50 jobs at E12's seeds,
decoded raw + D1 + D1′ + B5, exactly as `e16_b5`, so every file pairs with `e16_b5`'s.

**Attach two inputs:** **`uie-v2p1-code`**, and the **V2P1_train** notebook's **output**.
**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About 3–3.5 h.
A rerun keeps every finished file. When done, download `/kaggle/working/v2p1_gen`.
'''),
code(LOCATE_CODE + r'''
SRC = output_of("p1_mem")
if SRC is None:
    raise SystemExit("Attach the V2P1_train notebook's output too, then rerun.")
tprov = json.loads((SRC / "PROVENANCE.json").read_text())
CKPT = SRC / "p1_mem" / "final.pt"
assert CKPT.is_file() and sha256(CKPT) == tprov["arms"]["p1_mem"]["final_sha256"], \
    "p1_mem: final.pt is not the one V2P1_train recorded"
print("checkpoint verified:", CKPT)
'''),
code(ENV),
code(r'''
# --- 3. tables, the registered jobs, the model ---------------------------------------
from uie.controlled import StateSuffixTables
from uie.suffixes import SuffixTable

shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
presence_slash = json.loads((CODE / "presence_slash_table.json").read_text())["table"]
device = torch.device("cuda")
OUT = WORK / "v2p1_gen"
OUT.mkdir(parents=True, exist_ok=True)
jobs = sorted(P1.jobs(), key=lambda j: (0 if j["mode"] == "free" else 1, j["label"], j["state"] or "", j["g"]))
print(f"{len(jobs)} files, {sum(j['n'] for j in jobs):,} URLs; decode {P1.decode_kwargs()}")
CKPT_SHA = {"p1_mem": sha256(CKPT)}
model, tok, _ = M.load_checkpoint(CKPT, device=device)
model.to(device).eval()
# load_checkpoint builds the model on the CPU: prove the weights (the memory's too) are on the GPU
assert all(p.device.type == "cuda" for p in model.parameters()), "STOP. the model is not on the GPU"
assert isinstance(model, M.MemURLTransformer)
assert tok.presence_plan and tok.root_slash and tok.slash_plan and tok.vocab_size == 110
assert model.cfg.d_model == E16.D_MODEL and sum(p.numel() for p in model.parameters()) == P1.PARAMS
print(f"loaded p1_mem: width {model.cfg.d_model}, memory {P1.MEM}, vocab {tok.vocab_size}, max_len {model.cfg.max_len}")
'''),
code(r'''
# --- 4. generate -----------------------------------------------------------------------
# PROVENANCE.json is rewritten after EVERY file; a rerun keeps a file only with a
# matching record -- anything else is made again: same seed, same bytes.
import platform
import time

PROV = OUT / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"files": {}}
prov.update({"purpose": "V2-P1 (research/v2/PREREG_V2_P1_MEMORY.md): p1_mem, decoded as e16_b5",
             "checkpoint_file": "final.pt", "code_commit": manifest["commit"],
             "checkpoints": CKPT_SHA, "tables": manifest["tables"], "decode": P1.decode_kwargs(),
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
        urls = P1.free_urls(model, tok, j["label"], j["g"], shipped, presence_slash, device=device,
                            plan_trace=trace, query_trace=qtrace)
    else:
        urls = P1.control_urls(model, tok, j["label"], j["state"], j["g"], tables, presence_slash,
                               device=device, plan_trace=trace, query_trace=qtrace)
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
'''),
    ]


def write(name, cells):
    """E15's writer, aimed at this folder."""
    import make_e15_notebooks as W
    old = W.HERE
    try:
        W.HERE = HERE
        _write(name, cells)
    finally:
        W.HERE = old


def main() -> None:
    write("V2P1_train.ipynb", train_cells())
    write("V2P1_generate.ipynb", generate_cells())


if __name__ == "__main__":
    main()
