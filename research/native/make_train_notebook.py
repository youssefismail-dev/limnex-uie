"""Build research/native/E8_train.ipynb -- E8 (N1) training, four arms
(research/native/PREREG_E8.md section 2).

    python research/native/make_train_notebook.py

Inputs on Kaggle: `uie-corpus2` (the data bundle) and `uie-e8-code` (built by
make_code_bundle.py). GPU T4 or P100, Internet on (for one pip install).
Output: /kaggle/working/<arm>/{best.pt, final.pt, train_result.json} and
/kaggle/working/PROVENANCE.json. Rerunning skips every finished arm.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "E8_train.ipynb"

INTRO = """# E8 (N1) -- training the four registered arms

Registration: `research/native/PREREG_E8.md`, section 2. Every arm is E7's
recipe with **E7's seed**; the arms differ only in the two registered flags:

| arm | presence conditioning | root slash |
|---|---|---|
| `e8_base` | E7's (prefix controls, dropout 0.5) | no |
| `e8_pfrs` | presence plan | yes -- the candidate |
| `e8_pf` | presence plan | no |
| `e8_rs` | E7's | yes |

About 1.1 h per arm on a T4 or P100 (E7's record). **Rerunning the notebook
skips every arm that already finished**, so a stopped session loses at most the
arm in progress. When all four are done, download `/kaggle/working` as a zip."""

CELL_LOCATE = r'''# --- 1. locate both datasets ---------------------------------------------------
from pathlib import Path

root = Path("/kaggle/input")
data_hits = sorted(root.rglob("bundle.manifest.json")) if root.exists() else []
code_hits = sorted(root.rglob("E8_CODE_MANIFEST.json")) if root.exists() else []
if not data_hits or not code_hits:
    for p in (sorted(root.rglob("*"))[:80] if root.exists() else []):
        print("   ", p)
    raise SystemExit("Attach BOTH datasets (right panel -> Add Input):\n"
                     "  uie-corpus2  -- the data, has bundle.manifest.json\n"
                     "  uie-e8-code  -- the E8 code, has E8_CODE_MANIFEST.json\n"
                     "then re-run.")
DATA = data_hits[0].parent
CODE = code_hits[0].parent
WORK = Path("/kaggle/working")
print("data:", DATA)
print("code:", CODE)'''

CELL_ENV = r'''# --- 2. environment: install, then run the E8 code and PROVE it ----------------
import hashlib, importlib, json, subprocess, sys
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

# CODE ONLY on the path: the data bundle carries an older `uie` without E8.
sys.path.insert(0, str(CODE))
for name in [m for m in sys.modules if m == "uie" or m.startswith("uie.")]:
    del sys.modules[name]

manifest = json.loads((CODE / "E8_CODE_MANIFEST.json").read_text())
bad = [f for f, h in manifest["files"].items()
       if hashlib.sha256((CODE / f).read_bytes()).hexdigest() != h]
if bad:
    raise SystemExit(f"STOP. code files differ from the manifest: {bad}")

import uie.tokenizer
from uie.tokenizer import URLTokenizer
assert str(CODE) in uie.tokenizer.__file__, uie.tokenizer.__file__
t = URLTokenizer(include_frame=True, include_fields=True, root_slash=True, presence_plan=True)
assert "<RS>" in t.stoi and t.presence_plan
print("E8 code OK:", len(manifest["files"]), "files at commit", manifest["commit"][:12])'''

CELL_DATA = r'''# --- 3. the data bundle is the two-corpus one ------------------------------------
need = ["name_train", "structure_train", "phishing_name_train", "phishing_structure_train",
        "name_val", "structure_val", "phishing_name_val", "phishing_structure_val"]
present = {p.name for p in DATA.iterdir()}
missing = [s for s in need if f"{s}.jsonl" not in present and f"{s}.jsonl.gz" not in present]
for s in need:
    print(("OK   " if s not in missing else "MISS "), s)
if missing:
    raise SystemExit(f"STOP. missing {missing}: this is not the uie-corpus2 bundle.")
print("ready")'''

CELL_TRAIN = r'''# --- 4. train the four arms (a finished arm is skipped) -----------------------------
# E7'S RECIPE, EVERY ARGUMENT STATED (run()'s own defaults are a smaller model).
# The seed is TrainConfig's default, 20260803 -- E7's -- for every arm.
import time
from uie.kaggle_train import run

RECIPE = dict(steps=20000, batch_size=256, lr=3e-4, d_model=384, n_layers=8, n_heads=8,
              eval_every=1000, class_alpha=1.0, frame=True, factorized=False,
              boundary_weight=1.0, fields=True)
ARMS = {
    "e8_base": dict(field_dropout=0.5, presence_plan=False, root_slash=False),
    "e8_pfrs": dict(field_dropout=0.0, presence_plan=True, root_slash=True),
    "e8_pf":   dict(field_dropout=0.0, presence_plan=True, root_slash=False),
    "e8_rs":   dict(field_dropout=0.5, presence_plan=False, root_slash=True),
}
PROV = WORK / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"arms": {}}
prov.update({"registration": "research/native/PREREG_E8.md",
             "code_commit": manifest["commit"], "recipe": RECIPE})
for arm, flags in ARMS.items():
    out = WORK / arm
    if (out / "train_result.json").exists() and (out / "best.pt").exists():
        print(f"{arm}: finished earlier -- skipped")
        continue
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n==================== {arm} {flags} ====================")
    t0 = time.time()
    run(DATA, out, **RECIPE, **flags)
    prov["arms"][arm] = {"flags": flags, "wall_minutes": round((time.time() - t0) / 60, 1)}
    PROV.write_text(json.dumps(prov, indent=1))
print("\nall arms trained")'''

CELL_READBACK = r'''# --- 5. read every arm's config back from its CHECKPOINT ----------------------------
import torch


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


problems = []
for arm, flags in ARMS.items():
    ck = torch.load(WORK / arm / "best.pt", map_location="cpu", weights_only=False)
    tc, mc, tk = ck["train_config"], ck["model_config"], ck["tokenizer"]
    want = [("steps", tc.get("steps"), 20000), ("batch_size", tc.get("batch_size"), 256),
            ("lr", tc.get("lr"), 3e-4), ("class_alpha", tc.get("class_alpha"), 1.0),
            ("boundary_weight", tc.get("boundary_weight"), 1.0),
            ("field_dropout", tc.get("field_dropout"), flags["field_dropout"]),
            ("seed", tc.get("seed"), 20260803),
            ("d_model", mc.get("d_model"), 384), ("n_layers", mc.get("n_layers"), 8),
            ("n_heads", mc.get("n_heads"), 8), ("max_len", mc.get("max_len"), 256),
            ("include_frame", tk.get("include_frame"), True),
            ("include_fields", tk.get("include_fields"), True),
            ("presence_plan", tk.get("presence_plan", False), flags["presence_plan"]),
            ("root_slash", tk.get("root_slash", False), flags["root_slash"])]
    wrong = [w for w in want if w[1] != w[2]]
    problems += [(arm,) + w for w in wrong]
    res = json.loads((WORK / arm / "train_result.json").read_text())
    prov["arms"].setdefault(arm, {"flags": flags})
    prov["arms"][arm].update({"best_sha256": sha256(WORK / arm / "best.pt"),
                              "final_sha256": sha256(WORK / arm / "final.pt"),
                              "vocab_size": mc.get("vocab_size"),
                              "final_val": res.get("final"),
                              "dropped_over_max_len": res.get("dropped_over_max_len"),
                              "honoring_quick": res.get("honoring_quick")})
    print(f"{arm:<8} vocab {mc.get('vocab_size')}  val {res.get('final')}  "
          f"{'OK' if not wrong else 'WRONG ' + str(wrong)}")
PROV.write_text(json.dumps(prov, indent=1))
print("\nMISMATCHES:", problems if problems else "none -- every arm is as registered")
print("Now download /kaggle/working (all four arm folders and PROVENANCE.json).")'''


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def main() -> None:
    nb = {"cells": [cell("markdown", INTRO), cell("code", CELL_LOCATE), cell("code", CELL_ENV),
                    cell("code", CELL_DATA), cell("code", CELL_TRAIN),
                    cell("code", CELL_READBACK)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
