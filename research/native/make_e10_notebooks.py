"""Build research/native/E10_train.ipynb and E10_generate.ipynb
(research/native/PREREG_E10.md sections 1-2).

    python research/native/make_e10_notebooks.py

Both run the `uie-e10-code` bundle (make_e10_bundle.py). Every constant and every
generation call comes from research/native/e10_common.py, so the laptop repeats
them exactly.
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
             if m == "uie" or m.startswith("uie.") or m in ("e8_common", "e10_common")]:
    del sys.modules[name]
import uie
import e10_common as K
manifest = json.loads((CODE / "E10_CODE_MANIFEST.json").read_text())
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
# E10 — training the two registered arms (the explicit presence decision)

Registration: `research/native/PREREG_E10.md`. Both arms are **the adopted
baseline's recipe from one seeded start**, so they begin with identical weights.
They differ in one thing:

| arm | presence |
|---|---|
| `e10_base` | the baseline: hidden prefix controls (`?` half the time) |
| `e10_plan` | a presence plan after the suffix, written by the model (so no control dropout) |

**Attach two inputs** (right panel → *Add Input*):
1. **`uie-corpus2`**, the data bundle, the same one E7, E8, E9 and N2a used;
2. **`uie-e10-code`**, the new code dataset.

**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About
2.5 h. **A rerun skips a finished arm.** When done, download `/kaggle/working`
(both arm folders and `PROVENANCE.json`).
'''),
code(r'''
# --- 1. locate both datasets ---------------------------------------------------
from pathlib import Path

root = Path("/kaggle/input")
data_hits = sorted(root.rglob("bundle.manifest.json")) if root.exists() else []
code_hits = sorted(root.rglob("E10_CODE_MANIFEST.json")) if root.exists() else []
if not data_hits or not code_hits:
    raise SystemExit("Attach BOTH datasets (right panel -> Add Input):\n"
                     "  uie-corpus2   -- the data, has bundle.manifest.json\n"
                     "  uie-e10-code  -- the E10 code, has E10_CODE_MANIFEST.json\n"
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
assert {"loss_norm", "loss_constant", "seed_init", "presence_plan"} <= set(
    inspect.signature(run).parameters)
print("data bundle: all 8 streams present; E10 flags present")
for arm in K.ARMS:
    print(f"  {arm}: {K.recipe(arm)}")
'''),
code(r'''
# --- 3. train both arms (a finished arm is skipped) ------------------------------
import time

PROV = WORK / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"arms": {}}
prov.update({"registration": "research/native/PREREG_E10.md",
             "code_commit": manifest["commit"], "base_recipe": K.BASE})
for arm in K.ARMS:
    out = WORK / arm
    if (out / "train_result.json").exists() and (out / "best.pt").exists():
        print(f"{arm}: finished earlier -- skipped")
        continue
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n==================== {arm} {K.ARMS[arm]} ====================", flush=True)
    t0 = time.time()
    run(DATA, out, **K.recipe(arm))
    prov["arms"][arm] = {"changes": K.ARMS[arm], "wall_minutes": round((time.time() - t0) / 60, 1)}
    PROV.write_text(json.dumps(prov, indent=1))
print("\nboth arms trained")
'''),
code(r'''
# --- 4. read both arms back from their CHECKPOINTS and records --------------------
problems = []
res = {}
for arm in K.ARMS:
    rec = K.recipe(arm)
    ck = torch.load(WORK / arm / "best.pt", map_location="cpu", weights_only=False)
    tc, mc, tk = ck["train_config"], ck["model_config"], ck["tokenizer"]
    r = json.loads((WORK / arm / "train_result.json").read_text())
    res[arm] = r
    want = [("steps", tc.get("steps"), 20000), ("batch_size", tc.get("batch_size"), 256),
            ("lr", tc.get("lr"), 3e-4), ("seed", tc.get("seed"), 20260803),
            ("field_dropout", tc.get("field_dropout"), rec["field_dropout"]),
            ("class_alpha", tc.get("class_alpha"), 1.0),
            ("loss_norm", tc.get("loss_norm"), "fixed"),
            ("loss_constant", round(r.get("loss_constant") or 0, 3),
             round(rec["loss_constant"], 3)),
            ("d_model", mc.get("d_model"), 384), ("n_layers", mc.get("n_layers"), 8),
            ("vocab_size", mc.get("vocab_size"), 107),
            ("include_fields", tk.get("include_fields"), True),
            ("presence_plan", tk.get("presence_plan", False), rec.get("presence_plan", False)),
            ("root_slash", tk.get("root_slash", False), False),
            ("seed_init", r.get("seed_init"), True)]
    wrong = [w for w in want if w[1] != w[2]]
    problems += [(arm,) + w for w in wrong]
    prov["arms"].setdefault(arm, {"changes": K.ARMS[arm]})
    prov["arms"][arm].update({
        "best_sha256": sha256(WORK / arm / "best.pt"),
        "final_sha256": sha256(WORK / arm / "final.pt"),
        "init_weights_sha256": r.get("init_weights_sha256"),
        "loss_constant": r.get("loss_constant"), "final_val": r.get("final"),
        "dropped_over_max_len": r.get("dropped_over_max_len"), "counts": r.get("counts")})
    print(f"{arm:<9} val {r.get('final')}  dropped {r.get('dropped_over_max_len')}  "
          f"{'OK' if not wrong else 'WRONG ' + str(wrong)}")
starts = {res[a].get("init_weights_sha256") for a in K.ARMS}
if len(starts) != 1:
    problems.append(("V-id", "initial weights differ between the arms", starts))
PROV.write_text(json.dumps(prov, indent=1))
print("\nshared start:", "IDENTICAL" if len(starts) == 1 else "DIFFERENT", starts)
print("MISMATCHES:", problems if problems else "none -- both arms are as registered")
print("Now download /kaggle/working (e10_base/, e10_plan/ and PROVENANCE.json).")
'''),
    ]


def generate_cells():
    return [
md(r'''
# E10 — generation on the GPU (nothing is scored here)

Registered in `research/native/PREREG_E10.md`, D5. It writes 100 files, 360,000
URLs, all with the baseline's RAW decode (temperature 1.0, nothing else):
- **free:** 5 × 8,000 per class at the canonical seeds. `e10_base` runs its
  controls `?`; `e10_plan` writes its own plan;
- **control:** native, SC1's state tables, E8's seeds. The request is supplied as
  prefix controls (`e10_base`) or as the plan (`e10_plan`).

The free files come first, so a stopped session still leaves the main reading.

**Attach two inputs:**
1. **`uie-e10-code`**, the same dataset the training used;
2. the **E10_train** notebook's **output** (`e10_base/`, `e10_plan/`,
   `PROVENANCE.json`).

**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About
2.5 h. A rerun keeps every finished file with its record. When done, download
`/kaggle/working/e10_gen`.
'''),
code(r'''
# --- 1. locate the inputs and verify both checkpoints -----------------------------
import hashlib, json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


root = Path("/kaggle/input")
code_hits = sorted(root.rglob("E10_CODE_MANIFEST.json")) if root.exists() else []
train_prov = sorted(p for p in root.rglob("PROVENANCE.json")
                    if (p.parent / "e10_plan").is_dir()) if root.exists() else []
if not code_hits or not train_prov:
    raise SystemExit("Attach both inputs (right panel -> Add Input):\n"
                     "  uie-e10-code          -- has E10_CODE_MANIFEST.json\n"
                     "  the E10_train OUTPUT  -- e10_base/, e10_plan/ and PROVENANCE.json\n"
                     "Inputs mount when the session starts; rerun after attaching.")
CODE, TRAIN = code_hits[0].parent, train_prov[0].parent
tprov = json.loads(train_prov[0].read_text())
CKPTS = {}
for arm in ("e10_base", "e10_plan"):
    p = TRAIN / arm / "best.pt"
    assert p.is_file() and sha256(p) == tprov["arms"][arm]["best_sha256"], \
        f"{arm}: best.pt is not the one E10_train recorded"
    CKPTS[arm] = p
print("code:", CODE)
print("checkpoints verified:", {k: str(v) for k, v in CKPTS.items()})
'''),
code(ENV.replace("# --- 2. environment", "# --- 2. environment")),
code(r'''
# --- 3. tables, the registered jobs, the models ------------------------------------
from uie.controlled import StateSuffixTables
from uie.suffixes import SuffixTable
from uie.train import load_checkpoint

OUT = Path("/kaggle/working/e10_gen")
OUT.mkdir(parents=True, exist_ok=True)
shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
device = torch.device("cuda")
ORDER = [("e10_plan", "free"), ("e10_base", "free"), ("e10_plan", "control"),
         ("e10_base", "control")]
jobs = sorted(K.jobs(), key=lambda j: (ORDER.index((j["arm"], j["mode"])), j["label"],
                                       j["state"] or "", j["g"]))
print(f"{len(jobs)} files, {sum(j['n'] for j in jobs):,} URLs, in this order: {ORDER}")
_loaded = {}


def model_for(arm):
    if arm not in _loaded:
        _loaded.clear()
        torch.cuda.empty_cache()
        model, tok, _ = load_checkpoint(CKPTS[arm], device=device)
        model.to(device).eval()
        assert tok.include_fields and not getattr(tok, "root_slash", False)
        assert bool(getattr(tok, "presence_plan", False)) == (arm == "e10_plan")
        _loaded[arm] = (model, tok)
        print(f"   loaded {arm}: vocab {tok.vocab_size}", flush=True)
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
prov.update({"purpose": "E10 (research/native/PREREG_E10.md); generation only",
             "code_commit": manifest["commit"],
             "checkpoints": {a: sha256(p) for a, p in CKPTS.items()},
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
        urls = K.free_urls(model, tok, j["label"], j["g"], shipped, device=device,
                           plan_trace=trace)
    else:
        urls = K.control_urls(model, tok, j["label"], j["state"], j["g"], tables,
                              device=device, plan_trace=trace)
    assert len(urls) == j["n"], (j["file"], len(urls))
    path.write_text(json.dumps(urls))
    prov["files"][j["file"]] = {**{k: j[k] for k in ("arm", "mode", "label", "state", "g",
                                                    "n", "seed")},
                                "sha256": sha256(path), "seconds": round(time.time() - t0, 1),
                                "plan_trace_mean": sum(trace) / len(trace) if trace else None}
    PROV.write_text(json.dumps(prov, indent=1))
    done += 1
    el = (time.time() - t_start) / 60
    print(f"  {done:>3}/{len(jobs)}  {j['file']:<40} {time.time() - t0:6.0f}s  "
          f"({el:.0f} min so far)", flush=True)
print(f"\nall {len(jobs)} files written. Now download /kaggle/working/e10_gen.")
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
    write("E10_train.ipynb", train_cells())
    write("E10_generate.ipynb", generate_cells())


if __name__ == "__main__":
    main()
