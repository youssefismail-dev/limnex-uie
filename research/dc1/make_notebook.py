"""Write research/dc1/DC1_generate.ipynb (nbformat 4) for the Kaggle GPU.

Generation only (PREREG_DC1.md section 4); nothing is scored here. Every
constant, seed and setting comes from the bundled `dc1_common.py` and the
bundled `settings.json` (the fit's output, hash-pinned by the manifest).

    python research/dc1/make_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "DC1_generate.ipynb"
E7_SHA256 = "916d03857d7d3db4101d1b106293a83905f4b317676ee7e144b018170cb36065"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def cells():
    return [
md(r'''
# DC1 — calibrated per-field decoding for E7, on the GPU (nothing is scored here)

Registered in `research/dc1/PREREG_DC1.md`. E7's weights are untouched; only the
sampler's settings differ.

| arm | what | files |
|---|---|---|
| **free, base** | today's E7, FULL FREE | `free_base_{class}_g{g}.json` |
| **free, DC1** | the same call and seeds + DC1's free settings | `free_dc1_{class}_g{g}.json` |
| **control, DC1** | FULL CONTROL, six measurable states + DC1's settings | `ctrl_dc1_{class}_{state}_g{g}.json` |
| **control, cost reference** | the same, without DC1, generation 0 only | `ctrl_base_{class}_{state}_g0.json` |

**Attach two inputs** (right panel → *Add Input*): **`uie-dc1-code`** (NEW) and your
E7 training notebook's output (`best.pt`). **Settings:** GPU T4 x2, Internet **On**.
**Save & Run All.** About 5–6 h.
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
code_hits = sorted(root.rglob("DC1_CODE_MANIFEST.json")) if root.exists() else []
pts = {p: sha256(p) for p in sorted(root.rglob("*.pt"))} if root.exists() else {}
ckpt_hits = [p for p, h in pts.items() if h == E7_SHA256]
for p, h in pts.items():
    print(f"   checkpoint seen: {p}  sha256 {h[:16]}")
if not code_hits or not ckpt_hits:
    raise SystemExit(
        "Attach both inputs (right panel -> Add Input):\n"
        "  uie-dc1-code  -- has DC1_CODE_MANIFEST.json (the NEW code dataset)\n"
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
for name in [m for m in sys.modules if m == "uie" or m.startswith("uie.") or m == "dc1_common"]:
    del sys.modules[name]
import uie
import dc1_common
manifest = json.loads((CODE / "DC1_CODE_MANIFEST.json").read_text())
print("uie imported from:", uie.__file__)
for mod in (uie, dc1_common):
    assert str(CODE) in mod.__file__, f"{mod.__name__} is SHADOWED -- check sys.path"
mismatched = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
print(f"code integrity: {len(manifest['files']) - len(mismatched)}/"
      f"{len(manifest['files'])} files match")
assert not mismatched, f"files differ from the manifest: {mismatched}"
SETTINGS_SHA256 = manifest["files"]["settings.json"]
print(f"code: commit {manifest['commit'][:12]}; settings sha256 {SETTINGS_SHA256[:16]} verified")

import torch
print("torch:", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then rerun.")
print("gpu  :", torch.cuda.get_device_name(0))
'''),
code(r'''
# --- 3. settings, tables, model and the registered jobs ---------------------------
from dc1_common import (FROZEN_DELTA, G, MEASURABLE, N_CTRL, N_FREE, STATES,
                        generation_kwargs, seed_ctrl, seed_free)
from uie.controlled import StateSuffixTables, generate_controlled
from uie.presence import PresenceUnmet
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.train import load_checkpoint

OUT = Path("/kaggle/working/dc1_gen")
settings = json.loads((CODE / "settings.json").read_text(encoding="utf-8"))
shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")

device = torch.device("cuda")
model, tokenizer, _ = load_checkpoint(CKPT, device=device)
model.to(device).eval()
where = next(model.parameters()).device
assert where.type == "cuda", f"model is on {where}"
assert getattr(tokenizer, "include_fields", False) and tokenizer.vocab_size == 107


class Counting(torch.nn.Module):
    """Counts next-token predictions (rows per forward), for the cost criterion."""

    def __init__(self, m):
        super().__init__()
        self.m, self.cfg, self.rows = m, m.cfg, 0

    def forward(self, ids):
        self.rows += int(ids.shape[0])
        return self.m(ids)


counted = Counting(model).eval()
free_on = any(generation_kwargs(settings, lab, "free") for lab in ("benign", "phishing"))
ctrl_on = any(generation_kwargs(settings, lab, st) for lab, sts in MEASURABLE.items() for st in sts)
jobs = []
if free_on:
    for arm in ("base", "dc1"):
        jobs += [dict(mode="free", arm=arm, label=lab, state=None, g=g, seed=seed_free(lab, g),
                      file=f"free_{arm}_{lab}_g{g}.json")
                 for lab in ("benign", "phishing") for g in range(G)]
if ctrl_on:
    jobs += [dict(mode="control", arm="dc1", label=lab, state=st, g=g, seed=seed_ctrl(lab, st, g),
                  file=f"ctrl_dc1_{lab}_{st}_g{g}.json")
             for lab, sts in MEASURABLE.items() for st in sts for g in range(G)]
    jobs += [dict(mode="control", arm="base", label=lab, state=st, g=0, seed=seed_ctrl(lab, st, 0),
                  file=f"ctrl_base_{lab}_{st}_g0.json")
             for lab, sts in MEASURABLE.items() for st in sts]
print(f"model parameters on {where}; FULL FREE {'ON' if free_on else 'OFF (K1)'}, "
      f"FULL CONTROL {'ON' if ctrl_on else 'OFF (K1)'}; {len(jobs)} files to make")
'''),
code(r'''
# --- 4. generate ------------------------------------------------------------------
# PROVENANCE.json is rewritten after EVERY file; a rerun keeps an earlier outcome
# only with its own record (a file whose sha256 matches it, or an error with no
# file) -- anything else is made again: same seed, same bytes.
import platform
import time

CKPT_SHA256 = sha256(CKPT)
PROV = OUT / "PROVENANCE.json"


def provenance(records, kept, minutes):
    return {"purpose": "DC1 (research/dc1/PREREG_DC1.md); generation only",
            "decoder": "temperature 1.0", "frozen_delta": FROZEN_DELTA, "frame": "domain",
            "G": G, "N_FREE": N_FREE, "N_CTRL": N_CTRL,
            "device": {"forward": str(device), "gpu": torch.cuda.get_device_name(0),
                       "sampling": "cpu (seeded torch.Generator, as in generate())"},
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "tldextract": metadata.version("tldextract")},
            "code_commit": manifest["commit"],
            "code_integrity": {"files": len(manifest["files"]), "mismatched": mismatched},
            "checkpoint": {"source": str(CKPT), "sha256": CKPT_SHA256},
            "settings_sha256": SETTINGS_SHA256, "files": records,
            "kept_from_an_earlier_run": kept, "total_minutes": minutes}


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
        print(f"[{k}/{len(jobs)}] {path.name}: kept from an earlier run, with its record", flush=True)
        continue
    lab, st = job["label"], job["state"]
    extra = generation_kwargs(settings, lab, st or "free") if job["arm"] == "dc1" else {}
    common = dict(label=lab, seed=job["seed"], decoder="temperature",
                  decoder_kwargs={"temperature": 1.0},
                  field_bias={"PTH": {tokenizer.eos_id: -FROZEN_DELTA[lab]}},
                  device=device, **extra)
    t0, stats, error, rows0 = time.time(), {}, None, counted.rows
    try:
        if job["mode"] == "free":
            urls = generate_with_suffixes(counted, tokenizer, n=N_FREE[lab], table=shipped, **common)
        else:
            urls = generate_controlled(counted, tokenizer, n=N_CTRL, state_tables=tables,
                                       fields=STATES[st], stats=stats, **common)
        path.write_text(json.dumps(urls), encoding="utf-8")
    except PresenceUnmet as exc:
        urls, error = [], f"PresenceUnmet: {exc}"
        path.unlink(missing_ok=True)
    records.append({**{key: job[key] for key in ("file", "mode", "arm", "label", "state", "g", "seed")},
                    "n": len(urls), "settings": extra, "stats": stats,
                    "forward_rows": counted.rows - rows0, "error": error,
                    "sha256": sha256(path) if path.is_file() else None,
                    "seconds": round(time.time() - t0, 1)})
    PROV.write_text(json.dumps(provenance(records, kept, round((time.time() - started) / 60, 1)),
                               indent=1), encoding="utf-8")
    left = (time.time() - started) / k * (len(jobs) - k)
    print(f"[{k}/{len(jobs)}] {path.name}: {len(urls)} urls, "
          f"{(counted.rows - rows0) / max(1, len(urls)):.1f} predictions/url, "
          f"in {(time.time() - t0) / 60:.1f} min (about {left / 60:.0f} min left)"
          + (f"  ERROR {error}" if error else ""), flush=True)
'''),
code(r'''
# --- 5. provenance ---------------------------------------------------------------
PROV.write_text(json.dumps(provenance(records, kept, round((time.time() - started) / 60, 1)),
                           indent=1), encoding="utf-8")
ok = all(r["error"] is None and r["n"] == (N_FREE[r["label"]] if r["mode"] == "free" else N_CTRL)
         for r in records)
print(f"files {len(records)}; all complete: {ok}")
print("DONE" if ok else "INCOMPLETE -- see the ERROR lines above")
'''),
md(r'''
---

**Download** the *Output* tab (`dc1_gen/`: the files and `PROVENANCE.json`) into
`Downloads`. On the laptop it goes to `data/dc1/gen`, and `research/dc1/dc1_verify.py`
checks every file, seed, hash and state and regenerates a sample on the CPU for
byte identity before anything is scored.
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
