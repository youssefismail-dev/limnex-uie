"""Write research/c3/C3_generate.ipynb (nbformat 4) for the Kaggle GPU.

Arm F only (arm Q is SC1 arm T's existing files). Every registered constant --
sizes, M, doubling, rounds, seeds, delta -- comes from the bundled `c3_common.py`,
and the generator is the bundled `free_matched.py`, so the laptop's tests, its
verifier and the GPU run cannot disagree.

    python research/c3/make_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "C3_generate.ipynb"
E7_SHA256 = "916d03857d7d3db4101d1b106293a83905f4b317676ee7e144b018170cb36065"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def cells():
    return [
md(r'''
# C3-QRY — arm F on the GPU (nothing is scored here)

Registered in `research/c3/PREREG_C3_QRY.md`. **Arm F, matched free generation:**
E7 writes with every control token `?` — nothing asks for a query — while the
suffix is drawn from the Q:1 state table and held, exactly as arm Q's was; a URL
is kept only if it carries a query. **Benign 5 × 1,200, phishing 5 × 400**
query-bearing URLs. Arm Q is SC1's existing arm-T files; nothing of it runs here.

**Attach two inputs** (right panel → *Add Input*):

| input | what | found by |
|---|---|---|
| **`uie-c3-code`** — NEW | `uie/`, `c3_common.py`, `free_matched.py`, the adopted tables | `C3_CODE_MANIFEST.json` |
| your E7 training notebook's output | `best.pt` | its SHA-256 |

**Settings:** GPU T4 x2, Internet **On**. **Save & Run All.** About 4–5 h (4.1 h median,
4.8 h at the 99th percentile, in a simulation of this exact run).
'''),
code(r'''
# --- 1. locate the two inputs ---------------------------------------------------
import hashlib
from pathlib import Path

E7_SHA256 = "@@E7@@"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


root = Path("/kaggle/input")
code_hits = sorted(root.rglob("C3_CODE_MANIFEST.json")) if root.exists() else []
pts = {p: sha256(p) for p in sorted(root.rglob("*.pt"))} if root.exists() else {}
ckpt_hits = [p for p, h in pts.items() if h == E7_SHA256]
for p, h in pts.items():
    print(f"   checkpoint seen: {p}  sha256 {h[:16]}")
if not code_hits or not ckpt_hits:
    raise SystemExit(
        "Attach both inputs (right panel -> Add Input):\n"
        "  uie-c3-code  -- has C3_CODE_MANIFEST.json (the NEW code dataset)\n"
        "  the E7 training notebook's OUTPUT -- best.pt, sha256 "
        f"{E7_SHA256[:16]}...\nInputs mount when the session starts; rerun after attaching.")
CODE, CKPT = code_hits[0].parent, ckpt_hits[0]
print("code:", CODE)
print("ckpt:", CKPT, "(sha256 verified: best_e7)")
'''.replace("@@E7@@", E7_SHA256)),
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
             if m == "uie" or m.startswith("uie.") or m in ("c3_common", "free_matched")]:
    del sys.modules[name]
import uie
import c3_common
import free_matched
manifest = json.loads((CODE / "C3_CODE_MANIFEST.json").read_text())
print("uie imported from:", uie.__file__)
for mod in (uie, c3_common, free_matched):
    assert str(CODE) in mod.__file__, f"{mod.__name__} is SHADOWED -- check sys.path"
mismatched = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
print(f"code integrity: {len(manifest['files']) - len(mismatched)}/"
      f"{len(manifest['files'])} files match")
assert not mismatched, f"files differ from the manifest: {mismatched}"
TABLES_SHA256 = c3_common.TABLES_SHA256
assert manifest["files"]["state_suffix_tables.json"] == TABLES_SHA256, "tables differ -- STOP"
print(f"code: commit {manifest['commit'][:12]}; tables sha256 {TABLES_SHA256[:16]} verified")

import torch
print("torch:", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then rerun.")
print("gpu  :", torch.cuda.get_device_name(0))
'''),
code(r'''
# --- 3. the registered arm, sizes and seeds; the model --------------------------
from c3_common import (DELTA, DOUBLING_CAP, G, M, MAX_ROUNDS, N, OFFSET_F, ROUND_STRIDE,
                       STATE, gen_seed, tag)
from free_matched import FreeUnmet, generate_free_matched
from uie.controlled import StateSuffixTables
from uie.train import load_checkpoint

OUT = Path("/kaggle/working/c3_gen/F/gen")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")

device = torch.device("cuda")
model, tokenizer, _ = load_checkpoint(CKPT, device=device)
model.to(device).eval()
where = next(model.parameters()).device
assert where.type == "cuda", f"model is on {where}"
assert getattr(tokenizer, "include_fields", False) and tokenizer.vocab_size == 107
jobs = [(label, g) for label in ("benign", "phishing") for g in range(G)]
print(f"model parameters on {where}; arm F (matched free); {len(jobs)} files to make "
      f"(benign {G} x {N['benign']}, phishing {G} x {N['phishing']} query-bearing URLs)")
'''),
code(r'''
# --- 4. generate: free URLs, Q:1 suffix held, kept only with a query ------------
# PROVENANCE.json is rewritten after EVERY file, so no finished file loses its
# record. A rerun in this session keeps an earlier outcome only with its own
# record (a file whose sha256 matches, or a FreeUnmet with no file) -- draw counts
# included; anything else is made again: same seed, same bytes.
import platform
import time

CKPT_SHA256 = sha256(CKPT)
PROV = OUT / "PROVENANCE.json"


def provenance(records, kept, minutes):
    return {"purpose": "C3-QRY arm F (research/c3/PREREG_C3_QRY.md); generation only",
            "arm": "F", "mode": "matched free: <Q:?>, the Q:1 state table held, kept if a query",
            "G": G, "N": N, "M": M, "doubling_cap": DOUBLING_CAP, "max_rounds": MAX_ROUNDS,
            "generation": {"decoder": "temperature", "temperature": 1.0, "delta": DELTA,
                           "frame": "domain", "fields": None,
                           "seed_rule": f"20261001 + (100*s + g) * 1e9 + {OFFSET_F}; "
                                        "s = 2 benign, 6 phishing"},
            "device": {"forward": str(device), "gpu": torch.cuda.get_device_name(0),
                       "sampling": "cpu (seeded torch.Generator, as in generate())"},
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "tldextract": metadata.version("tldextract")},
            "code_commit": manifest["commit"],
            "code_integrity": {"files": len(manifest["files"]), "mismatched": mismatched},
            "checkpoint": {"source": str(CKPT), "sha256": CKPT_SHA256},
            "tables_sha256": TABLES_SHA256, "files": records,
            "kept_from_an_earlier_run": kept, "total_minutes": minutes}


OUT.mkdir(parents=True, exist_ok=True)
earlier = ({r["file"]: r for r in json.loads(PROV.read_text(encoding="utf-8"))["files"]}
           if PROV.is_file() else {})
started, records, kept = time.time(), [], []
for k, (label, g) in enumerate(jobs, 1):
    path = OUT / f"{tag(label)}_g{g}.json"
    old = earlier.get(path.name)
    if old is not None and old["sha256"] == (sha256(path) if path.is_file() else None):
        records.append(old)
        kept.append(path.name)
        print(f"[{k}/{len(jobs)}] {path.name}: kept from an earlier run, with its record "
              f"({old['n']} urls, draws {old['stats'].get('draws')})"
              + (f"  ERROR {old['error']}" if old["error"] else ""), flush=True)
        continue
    t0, stats, error = time.time(), {}, None
    try:
        urls = generate_free_matched(
            model, tokenizer, label=label, n=N[label],
            table=tables.for_state(label, STATE), seed=gen_seed(label, g),
            m=M[label], max_rounds=MAX_ROUNDS, doubling_cap=DOUBLING_CAP,
            stride=ROUND_STRIDE, stats=stats,
            decoder="temperature", decoder_kwargs={"temperature": 1.0},
            field_bias={"PTH": {tokenizer.eos_id: -DELTA[label]}}, device=device)
        path.write_text(json.dumps(urls), encoding="utf-8")
    except FreeUnmet as exc:
        urls, error = [], f"FreeUnmet: {exc}"
        path.unlink(missing_ok=True)
    records.append({"file": path.name, "label": label, "state": STATE, "generation": g,
                    "seed": gen_seed(label, g), "n": len(urls), "stats": stats,
                    "error": error, "sha256": sha256(path) if path.is_file() else None,
                    "seconds": round(time.time() - t0, 1)})
    PROV.write_text(json.dumps(provenance(records, kept, round((time.time() - started) / 60, 1)),
                               indent=1), encoding="utf-8")
    left = (time.time() - started) / k * (len(jobs) - k)
    rate = (stats["round0_with_query"] / stats["round0_draws"]) if stats.get("round0_draws") else None
    print(f"[{k}/{len(jobs)}] {path.name}: {len(urls)} urls, draws {stats.get('draws')}, "
          f"query rate {rate if rate is None else round(rate, 4)} "
          f"in {(time.time() - t0) / 60:.1f} min (about {left / 60:.0f} min left)"
          + (f"  ERROR {error}" if error else ""), flush=True)
'''),
code(r'''
# --- 5. provenance ---------------------------------------------------------------
PROV.write_text(json.dumps(provenance(records, kept, round((time.time() - started) / 60, 1)),
                           indent=1), encoding="utf-8")
ok = all(r["n"] == N[r["label"]] and r["error"] is None for r in records)
print(f"files {len(records)}; all complete: {ok}")
print("DONE" if ok else "INCOMPLETE -- see the ERROR lines above")
'''),
md(r'''
---

**Download** the *Output* tab (`c3_gen/F/gen/`: 10 files and `PROVENANCE.json`)
into `Downloads`. On the laptop it goes to `data/c3/F/gen`, and
`research/c3/c3_verify.py` checks every file, seed, hash and query, arm Q's
integrity and the scorer's, and regenerates a sample on the CPU for byte
identity, before anything is scored.
'''),
    ]


def main() -> None:
    nb = {"cells": cells(),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
