"""Write research/sc1/SC1_generate.ipynb (nbformat 4) for the Kaggle GPU.

The notebook takes every registered constant -- states, arms, seeds, delta,
max rounds -- from the bundled `sc1_common.py`, so the laptop's verifier and the
GPU run cannot disagree about them. It embeds the sha256 of the tables it must
be given (data/sc1/tables.json, built by build_tables.py), so run this after the
tables exist.

    python research/sc1/make_notebook.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "SC1_generate.ipynb"
TABLES = ROOT / "data" / "sc1" / "tables.json"
E7_SHA256 = "916d03857d7d3db4101d1b106293a83905f4b317676ee7e144b018170cb36065"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def cells(tables_sha: str):
    return [
md(r'''
# SC1 — generation on the GPU (nothing is scored here)

Registered in `research/sc1/PREREG_SC1.md`. **Two arms × 14 control states ×
5 generations × 3,000 compliant URLs** from E7, presence enforced by
**held-suffix** resampling (`uie.presence.generate_with_presence`):

| arm | the suffix is drawn from |
|---|---|
| **C** (matched control) | the class-marginal table of `name_train` |
| **T** (intervention) | the requested state's table of `name_train` |

The two phishing states registered as not measurable are not generated. Scoring
happens on the laptop, with CR1's scorer unchanged.

**Attach two inputs** (right panel → *Add Input*):

| input | what | found by |
|---|---|---|
| **`uie-sc1-code`** — NEW | `uie/`, `sc1_common.py`, `tables.json` | `SC1_CODE_MANIFEST.json` |
| your E7 training notebook's output | `best.pt` | its SHA-256 |

**Settings:** GPU T4 x2, Internet **On**. **Save & Run All.** About 4 h.
'''),
code(r'''
# --- 1. locate the two inputs ---------------------------------------------------
import hashlib
from pathlib import Path

E7_SHA256 = "@@E7@@"
TABLES_SHA256 = "@@TABLES@@"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


root = Path("/kaggle/input")
code_hits = sorted(root.rglob("SC1_CODE_MANIFEST.json")) if root.exists() else []
pts = {p: sha256(p) for p in sorted(root.rglob("*.pt"))} if root.exists() else {}
ckpt_hits = [p for p, h in pts.items() if h == E7_SHA256]
for p, h in pts.items():
    print(f"   checkpoint seen: {p}  sha256 {h[:16]}")
if not code_hits or not ckpt_hits:
    raise SystemExit(
        "Attach both inputs (right panel -> Add Input):\n"
        "  uie-sc1-code  -- has SC1_CODE_MANIFEST.json (the NEW code dataset)\n"
        "  the E7 training notebook's OUTPUT -- best.pt, sha256 "
        f"{E7_SHA256[:16]}...\nInputs mount when the session starts; rerun after attaching.")
CODE, CKPT = code_hits[0].parent, ckpt_hits[0]
print("code:", CODE)
print("ckpt:", CKPT, "(sha256 verified: best_e7)")
'''.replace("@@E7@@", E7_SHA256).replace("@@TABLES@@", tables_sha)),
code(r'''
# --- 2. environment: install first, then run THIS code and prove it ------------
import importlib, json, subprocess, sys
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
print("tldextract:", TLDEXTRACT)

sys.path.insert(0, str(CODE))
for name in [m for m in sys.modules
             if m == "uie" or m.startswith("uie.") or m == "sc1_common"]:
    del sys.modules[name]
import uie
import sc1_common
manifest = json.loads((CODE / "SC1_CODE_MANIFEST.json").read_text())
print("uie imported from:", uie.__file__)
assert str(CODE) in uie.__file__, "the SC1 code is SHADOWED -- check sys.path"
assert str(CODE) in sc1_common.__file__, "sc1_common is SHADOWED -- check sys.path"
mismatched = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
print(f"code integrity: {len(manifest['files']) - len(mismatched)}/"
      f"{len(manifest['files'])} files match")
assert not mismatched, f"files differ from the manifest: {mismatched}"
assert manifest["files"]["tables.json"] == TABLES_SHA256, "tables differ -- STOP"
from uie.presence import PresenceUnmet, generate_with_presence
print(f"code: commit {manifest['commit'][:12]}; tables sha256 {TABLES_SHA256[:16]} verified")

import torch
print("torch:", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then rerun.")
print("gpu  :", torch.cuda.get_device_name(0))
'''),
code(r'''
# --- 3. the registered arms, states and seeds; the model -----------------------
from sc1_common import (ARMS, DELTA, G, MAX_ROUNDS, MEASURABLE, N_MAIN, OFFSET,
                        gen_seed, table_counts, tag_of)
from uie.suffixes import SuffixTable
from uie.train import load_checkpoint

RUN_ARMS = list(ARMS)     # ["C", "T"]. After a timeout: the arm still missing.
REPLICATE = []            # PREREG section 7.4 ONLY: flagged tags, e.g. ["phishing_P1"]
if REPLICATE:
    RUN_ARMS = ["T_rep"]
OUT = Path("/kaggle/working/sc1_gen")
tables = json.loads((CODE / "tables.json").read_text(encoding="utf-8"))

device = torch.device("cuda")
model, tokenizer, _ = load_checkpoint(CKPT, device=device)
model.to(device).eval()
where = next(model.parameters()).device
assert where.type == "cuda", f"model is on {where}"
assert getattr(tokenizer, "include_fields", False) and tokenizer.vocab_size == 107
jobs = [(arm, s, label, state, g) for arm in RUN_ARMS for s, label, state in MEASURABLE
        if not REPLICATE or tag_of(label, state) in REPLICATE for g in range(G)]
print(f"model parameters on {where}; arms {RUN_ARMS}; mode held; {len(jobs)} files to make "
      f"({len(jobs) // G} arm-states x {G} generations x {N_MAIN} compliant URLs)")
'''),
code(r'''
# --- 4. generate: held-suffix resampling until every URL honours its state -----
import time

started, records = time.time(), {arm: [] for arm in RUN_ARMS}
for k, (arm, s, label, state, g) in enumerate(jobs, 1):
    out = OUT / arm / "gen"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag_of(label, state)}_g{g}.json"
    t0, stats, error = time.time(), {}, None
    if path.is_file():
        urls = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            urls = generate_with_presence(
                model, tokenizer, label=label, n=N_MAIN,
                table=SuffixTable({label: table_counts(tables, arm, label, state)}),
                fields=state, seed=gen_seed(arm, s, g), hold_suffix=True,
                max_rounds=MAX_ROUNDS, stats=stats,
                decoder="temperature", decoder_kwargs={"temperature": 1.0},
                field_bias={"PTH": {tokenizer.eos_id: -DELTA[label]}}, device=device)
            path.write_text(json.dumps(urls), encoding="utf-8")
        except PresenceUnmet as exc:
            urls, error = [], f"PresenceUnmet: {exc}"
    records[arm].append({"file": path.name, "arm": arm, "state_index": s, "label": label,
                         "state": state, "generation": g, "seed": gen_seed(arm, s, g),
                         "n": len(urls), "stats": stats, "error": error,
                         "sha256": sha256(path) if path.is_file() else None,
                         "seconds": round(time.time() - t0, 1)})
    left = (time.time() - started) / k * (len(jobs) - k)
    print(f"[{k}/{len(jobs)}] {arm}/{path.name}: {len(urls)} urls, draws {stats.get('draws')} "
          f"in {(time.time() - t0) / 60:.1f} min (about {left / 60:.0f} min left)"
          + (f"  ERROR {error}" if error else ""), flush=True)
'''),
code(r'''
# --- 5. provenance, one file per arm --------------------------------------------
import platform

for arm in RUN_ARMS:
    prov = {"purpose": "SC1 generations (research/sc1/PREREG_SC1.md); generation only",
            "arm": arm, "mode": "held", "G": G, "n_main": N_MAIN, "max_rounds": MAX_ROUNDS,
            "table": ("name_train, class marginal" if arm == "C"
                      else "name_train, the requested state"),
            "generation": {"decoder": "temperature", "temperature": 1.0, "delta": DELTA,
                           "frame": "domain",
                           "seed_rule": f"20261001 + (100*s + g) * 1e9 + {OFFSET[arm]}"},
            "device": {"forward": str(device), "gpu": torch.cuda.get_device_name(0),
                       "sampling": "cpu (seeded torch.Generator, as in generate())"},
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "tldextract": metadata.version("tldextract")},
            "code_commit": manifest["commit"],
            "code_integrity": {"files": len(manifest["files"]), "mismatched": mismatched},
            "checkpoint": {"source": str(CKPT), "sha256": sha256(CKPT)},
            "tables_sha256": TABLES_SHA256, "files": records[arm],
            "total_minutes": round((time.time() - started) / 60, 1)}
    (OUT / arm / "gen" / "PROVENANCE.json").write_text(json.dumps(prov, indent=1),
                                                       encoding="utf-8")
ok = all(r["n"] == N_MAIN and r["error"] is None for arm in RUN_ARMS for r in records[arm])
print(f"files {sum(len(v) for v in records.values())}; all complete: {ok}")
print("DONE" if ok else "INCOMPLETE -- see the ERROR lines above")
'''),
md(r'''
---

**Download** the *Output* tab (`sc1_gen/`: `C/gen/` and `T/gen/`, 70 files and a
`PROVENANCE.json` each) into `Downloads`. On the laptop they go to `data/sc1/C/gen`
and `data/sc1/T/gen`, and `research/sc1/sc1_verify.py` checks every file, seed,
hash, state and suffix, and regenerates a sample on the CPU for byte identity,
before the scorer reads anything.
'''),
    ]


def main() -> None:
    tables_sha = hashlib.sha256(TABLES.read_bytes()).hexdigest()
    nb = {"cells": cells(tables_sha),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"wrote {OUT}  (tables sha256 {tables_sha[:16]})")


if __name__ == "__main__":
    main()
