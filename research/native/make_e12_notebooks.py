"""Build research/native/E12_train.ipynb and E12_generate.ipynb
(research/native/PREREG_E12.md sections 1-2).

    python research/native/make_e12_notebooks.py

Both run the `uie-e12-code` bundle (make_e12_bundle.py). Every constant and every
generation call comes from research/native/e12_common.py, so the laptop repeats
them exactly. The structure is E11's notebooks', which ran on Kaggle twice; E12
adds a start check that runs BEFORE any training.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


ENV = r'''
# --- 2. environment: install first, then run THIS code and prove it ------------
import hashlib, importlib, json, subprocess, sys
from importlib import metadata

TLDEXTRACT = "5.3.1"          # the version the local evaluation parses with


def installed(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if installed("tldextract") != TLDEXTRACT:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        f"tldextract=={TLDEXTRACT}"], check=True)
    except subprocess.CalledProcessError:
        raise SystemExit("pip could not install tldextract: Settings -> Internet -> On, rerun.")
    importlib.invalidate_caches()
assert installed("tldextract") == TLDEXTRACT, installed("tldextract")

# CODE ONLY on the path: the data bundle carries an older `uie`.
sys.path.insert(0, str(CODE / "research" / "native"))
sys.path.insert(0, str(CODE))
for name in [m for m in sys.modules
             if m == "uie" or m.startswith("uie.") or m in ("e8_common", "e11_common", "e12_common")]:
    del sys.modules[name]
import uie
import e12_common as K
manifest = json.loads((CODE / "E12_CODE_MANIFEST.json").read_text())
for mod in (uie, K):
    assert str(CODE) in mod.__file__, f"{mod.__name__} is SHADOWED -- check sys.path"
bad = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
bad += [t for t, h in manifest["tables"].items() if sha256(CODE / t) != h]
assert not bad, f"STOP. files differ from the manifest: {bad}"
print(f"code: commit {manifest['commit'][:12]}; {len(manifest['files'])} files and "
      f"{len(manifest['tables'])} tables verified")

import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4, then rerun.")
print("torch:", torch.__version__, "| gpu:", torch.cuda.get_device_name(0))
'''


def train_cells():
    return [
md(r'''
# E12 — the final refinement pass: training the three registered arms

Registration: `research/native/PREREG_E12.md`. All arms start from **E11's shared
start**; the only new rows (the slash-plan tokens, the positions 256–383) come from
their own seeded generators.

| arm | recipe |
|---|---|
| `e12_base` | E11's `e11_plan` recipe exactly: the control, and E11's replicate |
| `e12_final` | + the slash in the plan + a 384-token length limit: **the E3 candidate** |
| `e12_no_r1` | `e12_final` without the longer limit (leave-one-out) |

**Before any training, cell 3 rebuilds E11's start and checks it against the recorded
hash.** If Kaggle's torch has changed, it stops there, before a session is spent.

**Attach two inputs** (right panel → *Add Input*):
1. **`uie-corpus2`**, the data bundle, the same one E7–E11 used;
2. **`uie-e12-code`**, the new code dataset.

**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About
3.3 h (three arms, about 65 min each). **A rerun skips a finished arm.** When done,
download `/kaggle/working` (the three arm folders and `PROVENANCE.json`).
'''),
code(r'''
# --- 1. locate both datasets ---------------------------------------------------
from pathlib import Path

root = Path("/kaggle/input")
data_hits = sorted(root.rglob("bundle.manifest.json")) if root.exists() else []
code_hits = sorted(root.rglob("E12_CODE_MANIFEST.json")) if root.exists() else []
if not data_hits or not code_hits:
    raise SystemExit("Attach BOTH datasets (right panel -> Add Input):\n"
                     "  uie-corpus2   -- the data, has bundle.manifest.json\n"
                     "  uie-e12-code  -- the E12 code, has E12_CODE_MANIFEST.json\n"
                     "then re-run.")
DATA, CODE = data_hits[0].parent, code_hits[0].parent
WORK = Path("/kaggle/working")
print("data:", DATA)
print("code:", CODE)
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
assert {"loss_norm", "loss_constant", "seed_init", "presence_plan", "root_slash",
        "grad_clip_late", "slash_plan", "max_len", "frame_every", "e12_start"} <= set(
    inspect.signature(run).parameters)
print("data bundle: all 8 streams present; E12 flags present")
for arm in K.ARMS:
    print(f"  {arm}: {K.recipe(arm)}")
'''),
code(r'''
# --- 3. THE START CHECK, before any training (PREREG_E12 D5) ---------------------
e10, e11 = K.start_fingerprints()
print("E10 start:", e10, "(recorded", K.E10_START + ")")
print("E11 start:", e11, "(recorded", K.E11_START + ")")
if (e10, e11) != (K.E10_START, K.E11_START):
    raise SystemExit("STOP. This torch rebuilds a different start than E10/E11 recorded, "
                     "so E12 would not start where E11 did. Nothing was trained.")
print("start check PASSED: E12 starts exactly where E11 did")

# --- 3b. MEMORY PREFLIGHT for the 384-token arm, before any training (D3) --------
try:
    peak = K.memory_preflight(torch.device("cuda"))
except torch.cuda.OutOfMemoryError:
    raise SystemExit("STOP. A full 256 x 384 training step does not fit on this GPU. "
                     "Nothing was trained; report this line.")
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"memory preflight PASSED: a full 384-token step peaks at {peak:.1f} GB of {total:.1f} GB")
'''),
code(r'''
# --- 4. train the three arms (a finished arm is skipped) --------------------------
import time

PROV = WORK / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"arms": {}}
prov.update({"registration": "research/native/PREREG_E12.md",
             "code_commit": manifest["commit"], "base_recipe": K.BASE,
             "start_check": {"e10": e10, "e11": e11}})
for arm in K.ARMS:
    out = WORK / arm
    if (out / "train_result.json").exists() and (out / "final.pt").exists():
        print(f"{arm}: finished earlier -- skipped")
        continue
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n==================== {arm} {K.ARMS[arm]} ====================", flush=True)
    t0 = time.time()
    run(DATA, out, **K.recipe(arm))
    prov["arms"][arm] = {"changes": K.ARMS[arm], "wall_minutes": round((time.time() - t0) / 60, 1)}
    PROV.write_text(json.dumps(prov, indent=1))
print("\nall three arms trained")
'''),
code(r'''
# --- 5. read the three arms back from their CHECKPOINTS and records ----------------
problems = []
res = {}
for arm in K.ARMS:
    rec = K.recipe(arm)
    ck = torch.load(WORK / arm / "final.pt", map_location="cpu", weights_only=False)
    tc, mc, tk = ck["train_config"], ck["model_config"], ck["tokenizer"]
    r = json.loads((WORK / arm / "train_result.json").read_text())
    res[arm] = r
    s = r.get("e12_start") or {}
    slash = bool(rec.get("slash_plan"))
    want = [("steps", tc.get("steps"), 20000), ("batch_size", tc.get("batch_size"), 256),
            ("lr", tc.get("lr"), 3e-4), ("seed", tc.get("seed"), 20260803),
            ("field_dropout", tc.get("field_dropout"), 0.0),
            ("class_alpha", tc.get("class_alpha"), 1.0),
            ("loss_norm", tc.get("loss_norm"), "fixed"),
            ("loss_constant", round(r.get("loss_constant") or 0, 3), round(rec["loss_constant"], 3)),
            ("grad_clip", tc.get("grad_clip"), 1.0), ("grad_clip_late", tc.get("grad_clip_late"), 5.0),
            ("grad_clip_switch", tc.get("grad_clip_switch"), 1000),
            ("d_model", mc.get("d_model"), 384), ("n_layers", mc.get("n_layers"), 8),
            ("max_len", mc.get("max_len"), rec.get("max_len", 256)),
            ("frame_every", mc.get("frame_every", False), False),
            ("vocab_size", mc.get("vocab_size"), 110 if slash else 108),
            ("presence_plan", tk.get("presence_plan", False), True),
            ("root_slash", tk.get("root_slash", False), True),
            ("slash_plan", tk.get("slash_plan", False), slash),
            ("seed_init", r.get("seed_init"), True),
            ("e12_start", s.get("on"), True),
            ("start: E10 part", s.get("e10_start_sha256"), K.E10_START),
            ("start: E11 part", s.get("e11_start_sha256"), K.E11_START)]
    wrong = [w for w in want if w[1] != w[2]]
    problems += [(arm,) + w for w in wrong]
    prov["arms"].setdefault(arm, {"changes": K.ARMS[arm]})
    prov["arms"][arm].update({
        "best_sha256": sha256(WORK / arm / "best.pt"),
        "final_sha256": sha256(WORK / arm / "final.pt"),
        "init_weights_sha256": r.get("init_weights_sha256"),
        "e12_start": s, "loss_constant": r.get("loss_constant"), "final_val": r.get("final"),
        "dropped_over_max_len": r.get("dropped_over_max_len"), "counts": r.get("counts"),
        "honoring_quick": r.get("honoring_quick")})
    late = {k: v for k, v in (r.get("grad_diagnostics") or {}).items() if k.startswith("after_1000")}
    print(f"{arm:<10} val {r.get('final')}  dropped {r.get('dropped_over_max_len')}  "
          f"{'OK' if not wrong else 'WRONG ' + str(wrong)}")
    for k, v in late.items():
        print(f"    clipping {k:<22} steps {v['steps']:>6}  clipped {100 * v['clipped_share']:5.1f}%")
rows = {a: (res[a].get("e12_start") or {}).get("new_token_rows_sha256") for a in K.SLASH_ARMS}
if len(set(rows.values())) != 1:
    problems.append(("V-id", "the slash arms' new token rows differ", rows))
PROV.write_text(json.dumps(prov, indent=1))
print("\nslash-plan token rows:", "IDENTICAL across the slash arms" if len(set(rows.values())) == 1 else rows)
print("MISMATCHES:", problems if problems else "none -- all three arms are as registered")
print("Now download /kaggle/working (the three arm folders and PROVENANCE.json).")
'''),
    ]


def generate_cells():
    return [
md(r'''
# E12 — generation on the GPU, from `final.pt` (nothing is scored here)

Registered in `research/native/PREREG_E12.md`, D2 and D7. It writes 150 files,
540,000 URLs, all with the baseline's raw decode (temperature 1.0, nothing else),
from each arm's **`final.pt`** (step 20,000; the owner's decision):
- **free:** 5 × 8,000 per class at the canonical seeds. The plan is supplied from the
  data, and the model writes the URL. For `e12_base` that is E11's plan (path and
  query); the slash arms also get the slash from the data;
- **control:** native, SC1's state tables, E8's seeds, the request supplied (the
  slash arms get S from the data).

The free files come first.

**Attach two inputs:**
1. **`uie-e12-code`**, the same dataset the training used;
2. the **E12_train** notebook's **output** (the three arm folders and
   `PROVENANCE.json`).

**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About
5 h (E11's 200 files took 6.4 h). A rerun keeps every finished file with its
record. When done, download `/kaggle/working/e12_gen`.
'''),
code(r'''
# --- 1. locate the inputs and verify the three checkpoints ------------------------
import hashlib, json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ARMS = ("e12_base", "e12_final", "e12_no_r1")
root = Path("/kaggle/input")
code_hits = sorted(root.rglob("E12_CODE_MANIFEST.json")) if root.exists() else []
train_prov = sorted(p for p in root.rglob("PROVENANCE.json")
                    if all((p.parent / a).is_dir() for a in ARMS)) if root.exists() else []
if not code_hits or not train_prov:
    raise SystemExit("Attach both inputs (right panel -> Add Input):\n"
                     "  uie-e12-code          -- has E12_CODE_MANIFEST.json\n"
                     "  the E12_train OUTPUT  -- the three arm folders and PROVENANCE.json\n"
                     "Inputs mount when the session starts; rerun after attaching.")
CODE, TRAIN = code_hits[0].parent, train_prov[0].parent
tprov = json.loads(train_prov[0].read_text())
CKPTS = {}
for arm in ARMS:
    p = TRAIN / arm / "final.pt"
    assert p.is_file() and sha256(p) == tprov["arms"][arm]["final_sha256"], \
        f"{arm}: final.pt is not the one E12_train recorded"
    CKPTS[arm] = p
print("code:", CODE)
print("checkpoints verified (final.pt):", {k: str(v) for k, v in CKPTS.items()})
'''),
code(ENV),
code(r'''
# --- 3. tables, the registered jobs, the models ------------------------------------
from uie.controlled import StateSuffixTables
from uie.suffixes import SuffixTable
from uie.train import load_checkpoint

assert tuple(K.ARMS) == ARMS
OUT = Path("/kaggle/working/e12_gen")
OUT.mkdir(parents=True, exist_ok=True)
shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
presence = json.loads((CODE / "presence_table.json").read_text())                 # e12_base
presence_slash = json.loads((CODE / "presence_slash_table.json").read_text())["table"]  # D2
device = torch.device("cuda")
ORDER = [(a, "free") for a in ARMS] + [(a, "control") for a in ARMS]
jobs = sorted(K.jobs(), key=lambda j: (ORDER.index((j["arm"], j["mode"])), j["label"],
                                       j["state"] or "", j["g"]))
print(f"{len(jobs)} files, {sum(j['n'] for j in jobs):,} URLs, free first")
_loaded = {}


def model_for(arm):
    if arm not in _loaded:
        _loaded.clear()
        torch.cuda.empty_cache()
        model, tok, _ = load_checkpoint(CKPTS[arm], device=device)
        model.to(device).eval()
        assert next(model.parameters()).device.type == "cuda"
        slash = arm in K.SLASH_ARMS
        assert tok.include_fields and tok.root_slash and tok.presence_plan
        assert bool(getattr(tok, "slash_plan", False)) == slash
        assert tok.vocab_size == (110 if slash else 108)
        _loaded[arm] = (model, tok)
        print(f"   loaded {arm}: vocab {tok.vocab_size}, max_len {model.cfg.max_len}", flush=True)
    return _loaded[arm]
'''),
code(r'''
# --- 4. generate ----------------------------------------------------------------------
# PROVENANCE.json is rewritten after EVERY file; a rerun keeps a file only with a
# matching record -- anything else is made again: same seed, same bytes.
import platform
import time

PROV = OUT / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"files": {}}
prov.update({"purpose": "E12 (research/native/PREREG_E12.md); generation only",
             "checkpoint_file": "final.pt",
             "code_commit": manifest["commit"],
             "checkpoints": {a: sha256(p) for a, p in CKPTS.items()},
             "presence_table": sha256(CODE / "presence_table.json"),
             "presence_slash_table": sha256(CODE / "presence_slash_table.json"),
             "environment": {"torch": torch.__version__, "python": platform.python_version(),
                             "gpu": torch.cuda.get_device_name(0),
                             "tldextract": metadata.version("tldextract")},
             "decode": K.decode()})
t_start = time.time()
done = 0
for j in jobs:
    path = OUT / j["file"]
    rec = prov["files"].get(j["file"])
    if rec and path.is_file() and sha256(path) == rec["sha256"]:
        done += 1
        continue
    model, tok = model_for(j["arm"])
    t0 = time.time()
    trace = []
    if j["mode"] == "free":
        urls = K.free_urls(model, tok, j["arm"], j["label"], j["g"], shipped, presence=presence,
                           presence_slash=presence_slash, device=device, plan_trace=trace)
    else:
        urls = K.control_urls(model, tok, j["arm"], j["label"], j["state"], j["g"], tables,
                              presence_slash=presence_slash, device=device, plan_trace=trace)
    assert len(urls) == j["n"], (j["file"], len(urls))
    path.write_text(json.dumps(urls))
    prov["files"][j["file"]] = {**{k: j[k] for k in ("arm", "mode", "label", "state", "g",
                                                    "n", "seed", "plan")},
                                "sha256": sha256(path), "seconds": round(time.time() - t0, 1),
                                "plan_trace_mean": sum(trace) / len(trace) if trace else None}
    PROV.write_text(json.dumps(prov, indent=1))
    done += 1
    el = (time.time() - t_start) / 60
    print(f"  {done:>3}/{len(jobs)}  {j['file']:<42} {time.time() - t0:6.0f}s  "
          f"({el:.0f} min so far)", flush=True)
print(f"\nall {len(jobs)} files written. Now download /kaggle/working/e12_gen.")
'''),
    ]


def write(name, cells):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (HERE / name).write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("wrote", HERE / name)


def main() -> None:
    write("E12_train.ipynb", train_cells())
    write("E12_generate.ipynb", generate_cells())


if __name__ == "__main__":
    main()
