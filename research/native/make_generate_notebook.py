"""Build research/native/E8_generate.ipynb -- E8 (N1) generation on the Kaggle
GPU (research/native/PREREG_E8.md section 4). Nothing is scored here.

    python research/native/make_generate_notebook.py

Inputs: `uie-e8-code` (the same dataset E8_train used), the E8_train notebook's
OUTPUT (the four arms' best.pt and its PROVENANCE.json), and the E7 training
notebook's output (best.pt, sha256 916d0385...). Every call is
`research/native/e8_common.py`, so the laptop verifier repeats it exactly.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "E8_generate.ipynb"
E7_SHA256 = "916d03857d7d3db4101d1b106293a83905f4b317676ee7e144b018170cb36065"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def cells():
    return [
md(r'''
# E8 (N1) — generation on the GPU (nothing is scored here)

Registered in `research/native/PREREG_E8.md`, section 4. Every file uses the
resting decode (temperature 1.0 + the frozen path delta). FULL CONTROL is
**native**: the suffix is drawn once from SC1's state table and the request is
supplied -- nothing is resampled, rejected or masked.

| who | FULL FREE (5 × 8,000 per class) | FULL CONTROL (8 joint states × 5) |
|---|---|---|
| `e8_base`, `e8_pfrs`, `e8_pf`, `e8_rs` | yes | yes |
| `e7` (reference) | reused from HC1 C2, same seeds | yes |
| `e7dc1` (adopted free baseline) | yes | — |

**Attach three inputs** (right panel → *Add Input*):
1. **`uie-e8-code`** — the same dataset the training used;
2. the **E8_train** notebook's **output** (the four `e8_*` folders);
3. the **E7 training** notebook's output (`best.pt`).

**Settings:** GPU T4, Internet **On**. **Save & Run All.** About 6–7 h. A rerun keeps
every finished file with its record.
'''),
code(r'''
# --- 1. locate the inputs and verify every checkpoint ----------------------------
import hashlib, json
from pathlib import Path

E7_SHA256 = "@@E7@@"
ARM_NAMES = ("e8_base", "e8_rs", "e8_pf", "e8_pfrs")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


root = Path("/kaggle/input")
code_hits = sorted(root.rglob("E8_CODE_MANIFEST.json")) if root.exists() else []
train_prov = sorted(p for p in root.rglob("PROVENANCE.json")
                    if (p.parent / "e8_base").is_dir()) if root.exists() else []
pts = {p: sha256(p) for p in sorted(root.rglob("*.pt"))} if root.exists() else {}
for p, h in pts.items():
    print(f"   checkpoint seen: {p}  sha256 {h[:16]}")
e7_hits = [p for p, h in pts.items() if h == E7_SHA256]
if not code_hits or not train_prov or not e7_hits:
    raise SystemExit(
        "Attach all three inputs (right panel -> Add Input):\n"
        "  uie-e8-code             -- has E8_CODE_MANIFEST.json\n"
        "  the E8_train OUTPUT     -- e8_base/, e8_rs/, e8_pf/, e8_pfrs/ and PROVENANCE.json\n"
        f"  the E7 training OUTPUT  -- best.pt, sha256 {E7_SHA256[:16]}...\n"
        "Inputs mount when the session starts; rerun after attaching.")
CODE, E7_CKPT = code_hits[0].parent, e7_hits[0]
TRAIN = train_prov[0].parent
tprov = json.loads(train_prov[0].read_text())
CKPTS = {"e7": E7_CKPT}
for arm in ARM_NAMES:
    p = TRAIN / arm / "best.pt"
    want = tprov["arms"][arm]["best_sha256"]
    assert p.is_file() and pts.get(p) == want, f"{arm}: best.pt is not the one E8_train recorded"
    CKPTS[arm] = p
print("code:", CODE)
print("checkpoints verified:", {k: str(v) for k, v in CKPTS.items()})
'''.replace("@@E7@@", E7_SHA256)),
code(r'''
# --- 2. environment: install first, then run THIS code and prove it ------------
import importlib, subprocess, sys
from importlib import metadata

TLDEXTRACT = "5.3.1"


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
        raise SystemExit("pip could not install tldextract: Settings -> Internet -> On.")
    importlib.invalidate_caches()
assert installed("tldextract") == TLDEXTRACT, installed("tldextract")

sys.path.insert(0, str(CODE / "research" / "native"))
sys.path.insert(0, str(CODE))
for name in [m for m in sys.modules if m == "uie" or m.startswith("uie.") or m == "e8_common"]:
    del sys.modules[name]
import uie
import e8_common as K
manifest = json.loads((CODE / "E8_CODE_MANIFEST.json").read_text())
for mod in (uie, K):
    assert str(CODE) in mod.__file__, f"{mod.__name__} is SHADOWED -- check sys.path"
mismatched = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
mismatched += [t for t, h in manifest["tables"].items() if sha256(CODE / t) != h]
assert not mismatched, f"files differ from the manifest: {mismatched}"
print(f"code: commit {manifest['commit'][:12]}; {len(manifest['files'])} files and "
      f"{len(manifest['tables'])} tables verified")

import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4, then rerun.")
print("torch:", torch.__version__, "| gpu:", torch.cuda.get_device_name(0))
'''),
code(r'''
# --- 3. tables and the registered jobs ----------------------------------------------
from uie.controlled import StateSuffixTables
from uie.free import generate_free
from uie.suffixes import SuffixTable
from uie.train import load_checkpoint

OUT = Path("/kaggle/working/e8_gen")
shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
device = torch.device("cuda")

ORDER = ("e8_base", "e7", "e8_pfrs", "e8_pf", "e8_rs", "e7dc1")   # validity first
jobs = sorted(K.jobs(), key=lambda j: (ORDER.index(j["who"]), j["mode"] != "free",
                                       j["label"], j["state"] or "", j["g"]))
print(f"{len(jobs)} files to make, in this order: {ORDER}")

_loaded = {}


def model_for(who):
    key = "e7" if who == "e7dc1" else who
    if key not in _loaded:
        _loaded.clear()
        torch.cuda.empty_cache()
        model, tok, _ = load_checkpoint(CKPTS[key], device=device)
        model.to(device).eval()
        flags = K.ARMS.get(key, {"presence_plan": False, "root_slash": False})
        assert bool(getattr(tok, "presence_plan", False)) == flags["presence_plan"], key
        assert bool(getattr(tok, "root_slash", False)) == flags["root_slash"], key
        assert next(model.parameters()).device.type == "cuda"
        _loaded[key] = (model, tok)
        print(f"   loaded {key}: vocab {tok.vocab_size}, plan {tok.presence_plan}, "
              f"root slash {tok.root_slash}", flush=True)
    return _loaded[key]
'''),
code(r'''
# --- 4. generate ------------------------------------------------------------------
# PROVENANCE.json is rewritten after EVERY file; a rerun keeps an earlier file
# only with its own record (a matching sha256) -- anything else is made again:
# same seed, same bytes.
import platform
import time

PROV = OUT / "PROVENANCE.json"
CKPT_SHA = {k: sha256(v) for k, v in CKPTS.items()}


def provenance(records, kept, minutes):
    return {"purpose": "E8 (N1) (research/native/PREREG_E8.md); generation only",
            "decode": "temperature 1.0 + frozen path delta", "delta": K.DELTA, "frame": "domain",
            "G": K.G, "N_FREE": K.N_FREE, "N_CTRL": K.N_CTRL,
            "N_CTRL_UNMEASURABLE": K.N_CTRL_UNMEASURABLE,
            "device": {"forward": str(device), "gpu": torch.cuda.get_device_name(0),
                       "sampling": "cpu (seeded torch.Generator, as in generate())"},
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "tldextract": metadata.version("tldextract")},
            "code_commit": manifest["commit"], "checkpoints": CKPT_SHA,
            "files": records, "kept_from_an_earlier_run": kept, "total_minutes": minutes}


OUT.mkdir(parents=True, exist_ok=True)
earlier = ({r["file"]: r for r in json.loads(PROV.read_text(encoding="utf-8"))["files"]}
           if PROV.is_file() else {})
started, records, kept = time.time(), [], []
for k, job in enumerate(jobs, 1):
    path = OUT / job["file"]
    old = earlier.get(path.name)
    if old is not None and old["sha256"] == (sha256(path) if path.is_file() else None):
        records.append(old)
        kept.append(path.name)
        print(f"[{k}/{len(jobs)}] {path.name}: kept from an earlier run", flush=True)
        continue
    model, tok = model_for(job["who"])
    lab, st, g = job["label"], job["state"], job["g"]
    trace = [] if getattr(tok, "presence_plan", False) else None
    t0, error = time.time(), None
    try:
        if job["who"] == "e7dc1":
            urls = generate_free(model, tok, label=lab, n=K.N_FREE, table=shipped,
                                 seed=K.seed_free(g), device=device)
        elif job["mode"] == "free":
            urls = K.free_urls(model, tok, lab, g, shipped, device=device, plan_trace=trace)
        else:
            urls = K.control_urls(model, tok, lab, st, g, tables, device=device, plan_trace=trace)
        path.write_text(json.dumps(urls), encoding="utf-8")
    except Exception as exc:                                    # recorded, never hidden
        urls, error = [], f"{type(exc).__name__}: {exc}"
        path.unlink(missing_ok=True)
    records.append({**{key: job[key] for key in ("file", "who", "mode", "label", "state", "g")},
                    "seed": K.seed_free(g) if job["mode"] == "free" else K.seed_ctrl(lab, st, g),
                    "n": len(urls), "error": error,
                    "plan_off_mass": ({"mean": sum(trace) / len(trace), "max": max(trace)}
                                      if trace else None),
                    "sha256": sha256(path) if path.is_file() else None,
                    "seconds": round(time.time() - t0, 1)})
    PROV.write_text(json.dumps(provenance(records, kept, round((time.time() - started) / 60, 1)),
                               indent=1), encoding="utf-8")
    left = (time.time() - started) / k * (len(jobs) - k)
    print(f"[{k}/{len(jobs)}] {path.name}: {len(urls)} urls in {(time.time() - t0) / 60:.1f} min "
          f"(about {left / 60:.0f} min left)" + (f"  ERROR {error}" if error else ""), flush=True)
'''),
code(r'''
# --- 5. completeness ----------------------------------------------------------------
PROV.write_text(json.dumps(provenance(records, kept, round((time.time() - started) / 60, 1)),
                           indent=1), encoding="utf-8")


def want_n(r):
    return K.N_FREE if r["mode"] == "free" else K.n_ctrl(r["label"], r["state"])


ok = len(records) == len(jobs) and all(r["error"] is None and r["n"] == want_n(r) for r in records)
print(f"files {len(records)} / {len(jobs)}; all complete: {ok}")
print("DONE" if ok else "INCOMPLETE -- see the ERROR lines above")
'''),
md(r'''
---

**Download** the *Output* tab (`e8_gen/`: 250 files and `PROVENANCE.json`) into
`Downloads`. On the laptop it goes to `data/e8/gen`, and `research/native/e8_verify.py`
checks every file, seed and hash and regenerates a sample on the CPU for byte
identity before anything is scored.
'''),
    ]


def main() -> None:
    nb = {"cells": cells(),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
