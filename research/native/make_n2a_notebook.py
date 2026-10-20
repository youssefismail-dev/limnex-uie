"""Build research/native/N2a_train.ipynb -- N2a's two arms and their generated
queries on the Kaggle GPU (research/native/PREREG_N2A.md, Amendments 2-4).
Nothing is scored here.

    python research/native/make_n2a_notebook.py

Inputs: `uie-corpus2` (the data bundle), `uie-n2a-code` (make_n2a_bundle.py) and
the E7 training notebook's output (best.pt, sha256 916d0385...). Every call is
research/native/n2a_train.py, so the laptop repeats it exactly.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "N2a_train.ipynb"
E7_SHA256 = "916d03857d7d3db4101d1b106293a83905f4b317676ee7e144b018170cb36065"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def cells():
    return [
md(r'''
# N2a — two diagnostic arms on query rows only (nothing is scored here)

Registered in `research/native/PREREG_N2A.md` (Amendments 2–4). Each arm is **E7's
recipe on query-bearing rows only**: E7's own tokenizer, model configuration and
training settings, read from E7's checkpoint. Only three things differ.

| arm | query-row exposure | steps | warm-up |
|---|---|---|---|
| **A1** | E7's own, per class | 1,643 | 41 |
| **A2** | four times A1's | 6,572 | 164 |

After training, each arm loads its `final.pt` back and writes 5 × 1,500 free-mode
URLs per class with the resting decode. These models are **diagnostic only**: they
answer why E7's queries are compressed, and nothing they write is used as data.

**Attach three inputs** (right panel → *Add Input*):
1. **`uie-corpus2`**, the data bundle E7 and E8 trained on;
2. **`uie-n2a-code`**, the new code dataset;
3. the **E7 training** notebook's output (`best.pt`), the same one E8_generate used.

**Settings:** GPU T4, Internet **On**. **Save & Run All.** About 2 h, as an
estimate: E7 trained at 0.2 s per step, but query rows are longer, so both training
and generation run slower than E7's. **A rerun skips a finished arm.** When done,
download `/kaggle/working/n2a`.
'''),
code(r'''
# --- 1. locate the inputs ---------------------------------------------------------
import hashlib, json
from pathlib import Path

E7_SHA256 = "@@E7@@"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


root = Path("/kaggle/input")
data_hits = sorted(root.rglob("bundle.manifest.json")) if root.exists() else []
code_hits = sorted(root.rglob("N2A_CODE_MANIFEST.json")) if root.exists() else []
e7_hits = [p for p in (sorted(root.rglob("*.pt")) if root.exists() else [])
           if sha256(p) == E7_SHA256]
if not data_hits or not code_hits or not e7_hits:
    raise SystemExit(
        "Attach all three inputs (right panel -> Add Input):\n"
        "  uie-corpus2             -- has bundle.manifest.json\n"
        "  uie-n2a-code            -- has N2A_CODE_MANIFEST.json\n"
        f"  the E7 training OUTPUT  -- best.pt, sha256 {E7_SHA256[:16]}...\n"
        "Inputs mount when the session starts; rerun after attaching.")
DATA, CODE, E7_CKPT = data_hits[0].parent, code_hits[0].parent, e7_hits[0]
WORK = Path("/kaggle/working/n2a")
print("data:", DATA)
print("code:", CODE)
print("E7:  ", E7_CKPT)
'''.replace("@@E7@@", E7_SHA256)),
code(r'''
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
sys.path.insert(0, str(CODE / "research" / "native"))
sys.path.insert(0, str(CODE))
for name in [m for m in sys.modules
             if m == "uie" or m.startswith("uie.") or m in ("e8_common", "n2a_train")]:
    del sys.modules[name]
import uie
import e8_common
import n2a_train as T
manifest = json.loads((CODE / "N2A_CODE_MANIFEST.json").read_text())
for mod in (uie, e8_common, T):
    assert str(CODE) in mod.__file__, f"{mod.__name__} is SHADOWED -- check sys.path"
bad = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
bad += [t for t, h in manifest["tables"].items() if sha256(CODE / t) != h]
assert not bad, f"STOP. files differ from the manifest: {bad}"
print(f"code: commit {manifest['commit'][:12]}; {len(manifest['files'])} files and "
      f"{len(manifest['tables'])} table verified")

import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4, then rerun.")
print("torch:", torch.__version__, "| gpu:", torch.cuda.get_device_name(0))

need = ["name_train", "structure_train", "phishing_name_train", "phishing_structure_train",
        "name_val", "structure_val", "phishing_name_val", "phishing_structure_val"]
present = {p.name for p in DATA.iterdir()}
missing = [s for s in need if f"{s}.jsonl" not in present and f"{s}.jsonl.gz" not in present]
if missing:
    raise SystemExit(f"STOP. missing {missing}: this is not the uie-corpus2 bundle.")
print("data bundle: all 8 streams present")
'''),
code(r'''
# --- 3. train and generate A1, then A2 (a finished arm is skipped) ---------------
# run() stops before training if the kept query rows are not the registered
# 223,342 / 13,270 (Amendment 4), so the schedule cannot drift.
import time


def finished(arm):
    rec_path = WORK / arm / "N2A_RECORD.json"
    if not rec_path.exists():
        return False
    rec = json.loads(rec_path.read_text())
    ok = sha256(WORK / arm / "final.pt") == rec["final_sha256"]
    for name, meta in rec["generation"]["files"].items():
        ok = ok and (WORK / arm / name).is_file() and sha256(WORK / arm / name) == meta["sha256"]
    return ok


device = torch.device("cuda")
for arm in ("A1", "A2"):
    if finished(arm):
        print(f"{arm}: finished earlier -- skipped")
        continue
    print(f"\n==================== {arm} ====================", flush=True)
    t0 = time.time()
    T.run(DATA, E7_CKPT, arm, WORK / arm, CODE / "suffix_table.json", device=device)
    print(f"{arm}: {round((time.time() - t0) / 60, 1)} min wall", flush=True)
print("\nboth arms done")
'''),
code(r'''
# --- 4. read every arm back and write the provenance --------------------------------
import platform

prov = {"registration": "research/native/PREREG_N2A.md (Amendments 2-4)",
        "status": "DIAGNOSTIC", "code_commit": manifest["commit"], "e7_sha256": E7_SHA256,
        "environment": {"torch": torch.__version__, "python": platform.python_version(),
                        "gpu": torch.cuda.get_device_name(0),
                        "tldextract": metadata.version("tldextract")},
        "arms": {}}
for arm in ("A1", "A2"):
    rec = json.loads((WORK / arm / "N2A_RECORD.json").read_text())
    assert finished(arm), f"{arm}: files differ from its record"
    assert rec["schedule"]["steps"] == {"A1": 1643, "A2": 6572}[arm], rec["schedule"]
    gen = rec["generation"]["files"]
    prov["arms"][arm] = {k: rec[k] for k in ("kept_query_rows", "schedule", "final", "finite",
                                            "best_step", "train_minutes", "final_sha256",
                                            "best_sha256")}
    prov["arms"][arm]["generated"] = {n: {"n": m["n"], "with_query": m["with_query"]}
                                      for n, m in gen.items()}
    print(f"{arm}: steps {rec['schedule']['steps']}, final val {rec['final']}, finite "
          f"{rec['finite']}, best at step {rec['best_step']}, {rec['train_minutes']} min")
    for n, m in sorted(gen.items()):
        print(f"    {n}: {m['n']} URLs, {m['with_query']} with a query")
(WORK / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))
print("\nwrote", WORK / "PROVENANCE.json")
print("Now download /kaggle/working/n2a (A1/, A2/ and PROVENANCE.json).")
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
