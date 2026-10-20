"""Build research/native/E9_train.ipynb and E9_generate.ipynb
(research/native/PREREG_E9.md sections 2-3).

    python research/native/make_e9_notebooks.py

Both run the `uie-e9-code` bundle (make_e9_bundle.py). Every constant and every
generation call comes from research/native/e9_common.py, so the laptop repeats
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
             if m == "uie" or m.startswith("uie.") or m in ("e8_common", "e9_common")]:
    del sys.modules[name]
import uie
import e9_common as K
manifest = json.loads((CODE / "E9_CODE_MANIFEST.json").read_text())
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
# E9 — training the two registered arms (the length-neutral objective)

Registration: `research/native/PREREG_E9.md`. Both arms are **E7's recipe from one
seeded start**, so they begin with identical weights. They differ in one thing,
the objective:

| arm | objective |
|---|---|
| `e9_base` | today's: each batch's loss is its mean over its scored tokens |
| `e9_ln` | length-neutral: every clipped gradient × T_b / C, C = 14,398 |

**Attach two inputs** (right panel → *Add Input*):
1. **`uie-corpus2`**, the data bundle, the same one E7, E8 and N2a used;
2. **`uie-e9-code`**, the new code dataset.

**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About
2.5 h (E7's arm took 1.2 h). **A rerun skips a finished arm.** When done, download
`/kaggle/working` (both arm folders and `PROVENANCE.json`).
'''),
code(r'''
# --- 1. locate both datasets ---------------------------------------------------
from pathlib import Path

root = Path("/kaggle/input")
data_hits = sorted(root.rglob("bundle.manifest.json")) if root.exists() else []
code_hits = sorted(root.rglob("E9_CODE_MANIFEST.json")) if root.exists() else []
if not data_hits or not code_hits:
    raise SystemExit("Attach BOTH datasets (right panel -> Add Input):\n"
                     "  uie-corpus2  -- the data, has bundle.manifest.json\n"
                     "  uie-e9-code  -- the E9 code, has E9_CODE_MANIFEST.json\n"
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
assert {"loss_norm", "loss_constant", "seed_init"} <= set(inspect.signature(run).parameters)
print("data bundle: all 8 streams present; E9 flags present")
'''),
code(r'''
# --- 3. train both arms (a finished arm is skipped) ------------------------------
import time

PROV = WORK / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"arms": {}}
prov.update({"registration": "research/native/PREREG_E9.md",
             "code_commit": manifest["commit"], "recipe": K.RECIPE})
for arm, flags in K.ARMS.items():
    out = WORK / arm
    if (out / "train_result.json").exists() and (out / "best.pt").exists():
        print(f"{arm}: finished earlier -- skipped")
        continue
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n==================== {arm} {flags} ====================", flush=True)
    t0 = time.time()
    run(DATA, out, **K.RECIPE, **flags)
    prov["arms"][arm] = {"flags": flags, "wall_minutes": round((time.time() - t0) / 60, 1)}
    PROV.write_text(json.dumps(prov, indent=1))
print("\nboth arms trained")
'''),
code(r'''
# --- 4. read both arms back from their CHECKPOINTS and records --------------------
problems = []
res = {}
for arm, flags in K.ARMS.items():
    ck = torch.load(WORK / arm / "best.pt", map_location="cpu", weights_only=False)
    tc, mc, tk = ck["train_config"], ck["model_config"], ck["tokenizer"]
    r = json.loads((WORK / arm / "train_result.json").read_text())
    res[arm] = r
    want = [("steps", tc.get("steps"), 20000), ("batch_size", tc.get("batch_size"), 256),
            ("lr", tc.get("lr"), 3e-4), ("seed", tc.get("seed"), 20260803),
            ("field_dropout", tc.get("field_dropout"), 0.5),
            ("class_alpha", tc.get("class_alpha"), 1.0),
            ("loss_norm", tc.get("loss_norm"), flags["loss_norm"]),
            ("d_model", mc.get("d_model"), 384), ("n_layers", mc.get("n_layers"), 8),
            ("vocab_size", mc.get("vocab_size"), 107),
            ("include_fields", tk.get("include_fields"), True),
            ("presence_plan", tk.get("presence_plan", False), False),
            ("root_slash", tk.get("root_slash", False), False),
            ("seed_init", r.get("seed_init"), True)]
    if flags["loss_norm"] == "fixed":
        want.append(("loss_constant", round(r.get("loss_constant") or 0, 3),
                     round(K.LOSS_CONSTANT, 3)))
    wrong = [w for w in want if w[1] != w[2]]
    problems += [(arm,) + w for w in wrong]
    prov["arms"].setdefault(arm, {"flags": flags})
    prov["arms"][arm].update({
        "best_sha256": sha256(WORK / arm / "best.pt"),
        "final_sha256": sha256(WORK / arm / "final.pt"),
        "init_weights_sha256": r.get("init_weights_sha256"),
        "loss_constant": r.get("loss_constant"), "final_val": r.get("final"),
        "dropped_over_max_len": r.get("dropped_over_max_len"), "counts": r.get("counts")})
    print(f"{arm:<8} val {r.get('final')}  dropped {r.get('dropped_over_max_len')}  "
          f"{'OK' if not wrong else 'WRONG ' + str(wrong)}")
    for band, d in sorted((r.get("grad_diagnostics") or {}).items()):
        mn = d["mean_norm"] if d["mean_norm"] is not None else float("nan")
        print(f"    {band:<22} steps {d['steps']:>6}  mean norm {mn:.3f}  "
              f"clipped {100 * d['clipped_share']:5.1f}%  non-finite {d['nonfinite']}")
starts = {res[a].get("init_weights_sha256") for a in K.ARMS}
if len(starts) != 1:
    problems.append(("V-id", "initial weights differ between the arms", starts))
PROV.write_text(json.dumps(prov, indent=1))
print("\nshared start:", "IDENTICAL" if len(starts) == 1 else "DIFFERENT")
print("MISMATCHES:", problems if problems else "none -- both arms are as registered")
print("Now download /kaggle/working (e9_base/, e9_ln/ and PROVENANCE.json).")
'''),
    ]


def generate_cells():
    return [
md(r'''
# E9 — generation on the GPU (nothing is scored here)

Registered in `research/native/PREREG_E9.md`, section 3. It writes 160 files,
620,000 URLs:
- **free:** E7's free mode, 5 × 8,000 per class at the canonical seeds;
- **control:** native, SC1's state tables, E8's seeds.

| arm / decode | free | control |
|---|---|---|
| `e9_base` / resting (today's system) | yes | yes |
| `e9_base` / raw | yes | yes |
| `e9_ln` / raw (the native candidate) | yes | yes |
| `e9_ln` / resting | yes | — |

RESTING = temperature 1.0 + the frozen path delta. RAW = temperature 1.0 and
nothing else.

**Attach two inputs:**
1. **`uie-e9-code`**, the same dataset the training used;
2. the **E9_train** notebook's **output** (`e9_base/`, `e9_ln/`,
   `PROVENANCE.json`).

**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About
4–4.5 h. A rerun keeps every finished file with its record. When done, download
`/kaggle/working/e9_gen`.
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
code_hits = sorted(root.rglob("E9_CODE_MANIFEST.json")) if root.exists() else []
train_prov = sorted(p for p in root.rglob("PROVENANCE.json")
                    if (p.parent / "e9_ln").is_dir()) if root.exists() else []
if not code_hits or not train_prov:
    raise SystemExit("Attach both inputs (right panel -> Add Input):\n"
                     "  uie-e9-code          -- has E9_CODE_MANIFEST.json\n"
                     "  the E9_train OUTPUT  -- e9_base/, e9_ln/ and PROVENANCE.json\n"
                     "Inputs mount when the session starts; rerun after attaching.")
CODE, TRAIN = code_hits[0].parent, train_prov[0].parent
tprov = json.loads(train_prov[0].read_text())
CKPTS = {}
for arm in ("e9_base", "e9_ln"):
    p = TRAIN / arm / "best.pt"
    assert p.is_file() and sha256(p) == tprov["arms"][arm]["best_sha256"], \
        f"{arm}: best.pt is not the one E9_train recorded"
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

OUT = Path("/kaggle/working/e9_gen")
OUT.mkdir(parents=True, exist_ok=True)
shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
device = torch.device("cuda")
ORDER = [("e9_ln", "raw"), ("e9_base", "resting"), ("e9_base", "raw"), ("e9_ln", "resting")]
jobs = sorted(K.jobs(), key=lambda j: (ORDER.index((j["arm"], j["decode"])),
                                       j["mode"] != "free", j["label"], j["state"] or "", j["g"]))
print(f"{len(jobs)} files, {sum(j['n'] for j in jobs):,} URLs, in this order: {ORDER}")
_loaded = {}


def model_for(arm):
    if arm not in _loaded:
        _loaded.clear()
        torch.cuda.empty_cache()
        model, tok, _ = load_checkpoint(CKPTS[arm], device=device)
        model.to(device).eval()
        assert tok.include_fields and not getattr(tok, "presence_plan", False)
        assert not getattr(tok, "root_slash", False)
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
prov.update({"purpose": "E9 (research/native/PREREG_E9.md); generation only",
             "code_commit": manifest["commit"],
             "checkpoints": {a: sha256(p) for a, p in CKPTS.items()},
             "environment": {"torch": torch.__version__, "python": platform.python_version(),
                             "gpu": torch.cuda.get_device_name(0),
                             "tldextract": metadata.version("tldextract")},
             "delta": K.DELTA})
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
    if j["mode"] == "free":
        urls = K.free_urls(model, tok, j["label"], j["g"], shipped, j["decode"], device=device)
    else:
        urls = K.control_urls(model, tok, j["label"], j["state"], j["g"], tables, j["decode"],
                              device=device)
    assert len(urls) == j["n"], (j["file"], len(urls))
    path.write_text(json.dumps(urls))
    prov["files"][j["file"]] = {**{k: j[k] for k in ("arm", "decode", "mode", "label", "state",
                                                    "g", "n", "seed")},
                                "sha256": sha256(path), "seconds": round(time.time() - t0, 1)}
    PROV.write_text(json.dumps(prov, indent=1))
    done += 1
    el = (time.time() - t_start) / 60
    print(f"  {done:>3}/{len(jobs)}  {j['file']:<44} {time.time() - t0:6.0f}s  "
          f"({el:.0f} min so far)", flush=True)
print(f"\nall {len(jobs)} files written. Now download /kaggle/working/e9_gen.")
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
    write("E9_train.ipynb", train_cells())
    write("E9_generate.ipynb", generate_cells())


if __name__ == "__main__":
    main()
