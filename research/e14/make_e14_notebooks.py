"""Build research/e14/E14_decode.ipynb, E14_train.ipynb and E14_generate.ipynb
(research/e14/PREREG_E14.md section 6).

    python research/e14/make_e14_notebooks.py

All three run the `uie-e14-code` bundle (make_e14_bundle.py). Every constant and
every generation call comes from research/e14/e14_common.py, so the laptop repeats
them exactly. The structure is E13's notebooks', which ran on Kaggle once each.

- E14_decode   e12_d1p (e12_final + D1'); needs no training, so it can run first.
- E14_train    e14_q (e12_final's recipe + the query weight).
- E14_generate e14_q and e14_qd, 50 files each, after the laptop has verified the training.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


LOCATE_CODE = r'''
# --- 1. locate the code bundle -----------------------------------------------------
import hashlib, json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


root = Path("/kaggle/input")
code_hits = sorted(root.rglob("E14_CODE_MANIFEST.json")) if root.exists() else []
if not code_hits:
    raise SystemExit("Attach the uie-e14-code dataset (right panel -> Add Input), then rerun.")
CODE = code_hits[0].parent
WORK = Path("/kaggle/working")
print("code:", CODE)
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
for p in (CODE / "research" / "e14", CODE / "research" / "e13", CODE / "research" / "native", CODE):
    sys.path.insert(0, str(p))
for name in [m for m in sys.modules if m == "uie" or m.startswith("uie.")
             or m in ("e8_common", "e11_common", "e12_common", "e13_common", "e14_common")]:
    del sys.modules[name]
import uie
import e12_common as K
import e14_common as E14
manifest = json.loads((CODE / "E14_CODE_MANIFEST.json").read_text())
for mod in (uie, K, E14):
    assert str(CODE) in mod.__file__, f"{mod.__name__} is SHADOWED -- check sys.path"
bad = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
bad += [t for t, h in manifest["tables"].items() if sha256(CODE / t) != h]
if sha256(CODE / "e12_final.pt") != manifest["e12_final"]:
    bad.append("e12_final.pt")
assert not bad, f"STOP. files differ from the manifest: {bad}"
print(f"code: commit {manifest['commit'][:12]}; {len(manifest['files'])} files, "
      f"{len(manifest['tables'])} tables and e12_final.pt verified")

import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4, then rerun.")
GPU = torch.cuda.get_device_name(0)
print("torch:", torch.__version__, "| gpu:", GPU)
'''

GENERATE_LOOP = r'''
# --- 4. generate -----------------------------------------------------------------------
# PROVENANCE.json is rewritten after EVERY file; a rerun keeps a file only with a
# matching record -- anything else is made again: same seed, same bytes.
import platform
import time

PROV = OUT / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"files": {}}
prov.update({"purpose": PURPOSE, "checkpoint_file": "final.pt",
             "code_commit": manifest["commit"], "checkpoints": CKPT_SHA,
             "tables": manifest["tables"],
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
    trace, slash = [], []
    model, tok = model_for(E14.ARMS[j["arm"]]["weights"])
    st = slash if E14.ARMS[j["arm"]]["slash_rule"] else None
    if j["mode"] == "free":
        urls = E14.free_urls(model, tok, j["arm"], j["label"], j["g"], shipped, presence_slash,
                             device=device, plan_trace=trace, slash_trace=st)
    else:
        urls = E14.control_urls(model, tok, j["arm"], j["label"], j["state"], j["g"], tables,
                                presence_slash, device=device, plan_trace=trace, slash_trace=st)
    assert len(urls) == j["n"], (j["file"], len(urls))
    path.write_text(json.dumps(urls))
    prov["files"][j["file"]] = {**{k: v for k, v in j.items() if k != "file"},
                                "sha256": sha256(path), "seconds": round(time.time() - t0, 1),
                                "plan_trace_mean": sum(trace) / len(trace) if trace else None,
                                "slash_moved_mean": sum(slash) / len(slash) if slash else None}
    PROV.write_text(json.dumps(prov, indent=1))
    done += 1
    el = (time.time() - t_start) / 60
    print(f"  {done:>3}/{len(jobs)}  {j['file']:<46} {time.time() - t0:6.0f}s  "
          f"({el:.0f} min so far)", flush=True)
print(f"\nall {len(jobs)} files written. Now download {OUT}.")
'''

TABLES_CELL = r'''
from uie.controlled import StateSuffixTables
from uie.suffixes import SuffixTable
from uie.train import load_checkpoint

shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
presence_slash = json.loads((CODE / "presence_slash_table.json").read_text())["table"]
device = torch.device("cuda")
_loaded = {}
'''


def decode_cells():
    return [
md(r'''
# E14 — D1′, the slash rule on plans with a path (generation only; nothing is trained)

Registration: `research/e14/PREREG_E14.md` §2.2. The weights are **`e12_final`'s,
unchanged** (carried in the bundle, checked against E12's records).

| arm | what | files |
|---|---|---|
| `e12_d1p` | `e12_final` + D1 + **D1′**: in every supplied plan, the host ends as the plan's slash says — plans with a path included | 50 |

It uses **E12's seeds**, so each cell pairs with `e12_final`'s own files. On plans
without a path, D1′ *is* E13's D1, so those URLs must come out byte-identical to
E13's `e12_d1` files (the laptop checks this, V7). **That needs the same GPU type
E13 used: Tesla T4.** Cell 3 stops on any other.

**Attach one input:** **`uie-e14-code`**. **Settings:** GPU **T4**, Internet **On**.
**Save Version → Save & Run All.** About 1.5 h. A rerun keeps every finished file with
its record. When done, download `/kaggle/working/e14_decode`.
'''),
code(LOCATE_CODE),
code(ENV),
code(r'''
# --- 3. tables, the registered jobs, the model ---------------------------------------
if GPU != "Tesla T4":
    raise SystemExit(f"STOP. GPU is {GPU!r}: V7 compares with E13's T4 run. "
                     "Settings -> Accelerator -> GPU T4 x2 or T4, then rerun.")
''' + TABLES_CELL + r'''
OUT = WORK / "e14_decode"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = [("e12_d1p", "free"), ("e12_d1p", "control")]
jobs = sorted(E14.jobs(E14.DECODE_ARMS), key=lambda j: (ORDER.index((j["arm"], j["mode"])),
                                                        j["label"], j["state"] or "", j["g"]))
print(f"{len(jobs)} files, {sum(j['n'] for j in jobs):,} URLs")
CKPT_SHA = {"e12_final": manifest["e12_final"]}
PURPOSE = "E14 (research/e14/PREREG_E14.md): e12_d1p (D1'); generation only"


def model_for(weights):
    assert weights == "e12_final", weights
    if weights not in _loaded:
        model, tok, _ = load_checkpoint(CODE / "e12_final.pt", device=device)
        model.to(device).eval()
        assert tok.presence_plan and tok.root_slash and tok.slash_plan and tok.vocab_size == 110
        _loaded[weights] = (model, tok)
        print(f"   loaded {weights}: vocab {tok.vocab_size}, max_len {model.cfg.max_len}", flush=True)
    return _loaded[weights]
'''),
code(GENERATE_LOOP),
    ]


def train_cells():
    return [
md(r'''
# E14 — training `e14_q`: `e12_final`'s recipe with the query weight

Registration: `research/e14/PREREG_E14.md` §2.1. Everything is `e12_final`'s: the same
start, data, seed, batches, 20,000 steps, objective and pinned loss constant. **One
change:** every training step inside a query counts **4×** in the loss. Every other
step counts exactly what it did, and validation is unweighted.

**Before any training, cell 3 rebuilds E11's start and checks it**, and checks that a
384-token step fits. If either fails it stops before a session is spent. **Cell 5
reads the run back** against `e12_final`'s record. It must print
`MISMATCHES: none -- e14_q is as registered`.

**Attach two inputs** (right panel → *Add Input*):
1. **`uie-corpus2`**, the data bundle E7–E13 used;
2. **`uie-e14-code`**.

**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About 1.5 h.
**A rerun skips a finished arm.** When done, download `/kaggle/working`.
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
assert {"query_weight", "slash_plan", "max_len", "e12_start", "loss_constant"} <= set(
    inspect.signature(run).parameters)
print("data bundle: all 8 streams present; E14 option present")
print("e14_q:", E14.recipe_q())
'''),
code(r'''
# --- 3. THE START CHECK and the memory preflight, before any training -----------------
e10, e11 = K.start_fingerprints()
print("E10 start:", e10, "(recorded", K.E10_START + ")")
print("E11 start:", e11, "(recorded", K.E11_START + ")")
if (e10, e11) != (K.E10_START, K.E11_START):
    raise SystemExit("STOP. This torch rebuilds a different start than E10/E11 recorded. "
                     "Nothing was trained.")
try:
    peak = K.memory_preflight(torch.device("cuda"))
except torch.cuda.OutOfMemoryError:
    raise SystemExit("STOP. A full 256 x 384 training step does not fit on this GPU.")
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"start check PASSED; memory preflight PASSED ({peak:.1f} of {total:.1f} GB)")
'''),
code(r'''
# --- 4. train e14_q (skipped if finished) -------------------------------------------
import time

PROV = WORK / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"arms": {}}
prov.update({"registration": E14.REGISTRATION, "code_commit": manifest["commit"],
             "start_check": {"e10": e10, "e11": e11}, "gpu": GPU})
out = WORK / "e14_q"
if (out / "train_result.json").exists() and (out / "final.pt").exists():
    print("e14_q: finished earlier -- skipped")
else:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run(DATA, out, **E14.recipe_q())
    prov["arms"]["e14_q"] = {"wall_minutes": round((time.time() - t0) / 60, 1)}
    PROV.write_text(json.dumps(prov, indent=1))
'''),
code(r'''
# --- 5. read e14_q back against e12_final's record (V5) -------------------------------
problems = []
ck = torch.load(WORK / "e14_q" / "final.pt", map_location="cpu", weights_only=False)
e12 = torch.load(CODE / "e12_final.pt", map_location="cpu", weights_only=False)
tc, mc, tk = ck["train_config"], ck["model_config"], ck["tokenizer"]
r = json.loads((WORK / "e14_q" / "train_result.json").read_text())
s = r.get("e12_start") or {}
q = r.get("query_weight") or {}
want = [(f"train_config {k}", tc.get(k), v) for k, v in e12["train_config"].items()]
want += [(f"model_config {k}", mc.get(k), v) for k, v in e12["model_config"].items()]
want += [("query_weight", tc.get("query_weight"), E14.QUERY_WEIGHT),
         ("extra settings", sorted(set(tc) - set(e12["train_config"])), ["query_weight"]),
         ("tokenizer", tk == e12["tokenizer"], True),
         ("init weights", r.get("init_weights_sha256"), E14.E12_FINAL_INIT),
         ("row counts", r.get("counts"), E14.E12_FINAL_COUNTS),
         ("dropped over max_len", r.get("dropped_over_max_len"), E14.E12_FINAL_DROPPED),
         ("loss constant", r.get("loss_constant"), E14.E12_FINAL_LOSS_CONSTANT),
         ("band steps (same batches)",
          {k: v["steps"] for k, v in (r.get("grad_diagnostics") or {}).items() if k.startswith("all ")},
          E14.E12_FINAL_BAND_STEPS),
         ("query-weight record", q.get("w"), E14.QUERY_WEIGHT)]
want += [(f"start: {k}", s.get(k), v) for k, v in E14.E12_FINAL_START.items()]
problems = [w for w in want if w[1] != w[2]]
prov["arms"].setdefault("e14_q", {})
prov["arms"]["e14_q"].update({"best_sha256": sha256(WORK / "e14_q" / "best.pt"),
                              "final_sha256": sha256(WORK / "e14_q" / "final.pt"),
                              "init_weights_sha256": r.get("init_weights_sha256"),
                              "query_weight": q, "final_val": r.get("final"),
                              "counts": r.get("counts")})
PROV.write_text(json.dumps(prov, indent=1))
print(f"e14_q val {r.get('final')}; query weight record {q}")
print("MISMATCHES:", problems if problems else "none -- e14_q is as registered")
print("\nNow download /kaggle/working (e14_q/, PROVENANCE.json).")
'''),
    ]


def generate_cells():
    return [
md(r'''
# E14 — `e14_q` and `e14_qd`'s generation, from `final.pt` (nothing is scored here)

Registration: `research/e14/PREREG_E14.md` §2.4. Run this **after the laptop has
verified the training**. It writes 100 files at E12's seeds, so each cell pairs with
`e12_final`'s own files:

| arm | weights | decode | files |
|---|---|---|---|
| `e14_q` | `e14_q` | raw | 50 |
| `e14_qd` | `e14_q` | raw + D1 + D1′ | 50 |

**Attach two inputs:** **`uie-e14-code`**, and the **E14_train** notebook's **output**.
**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About 3.2 h.
A rerun keeps every finished file with its record. When done, download
`/kaggle/working/e14_gen`.
'''),
code(LOCATE_CODE + r'''
train_prov = sorted(p for p in root.rglob("PROVENANCE.json")
                    if (p.parent / "e14_q").is_dir()) if root.exists() else []
if not train_prov:
    raise SystemExit("Attach the E14_train notebook's output too, then rerun.")
TRAIN = train_prov[0].parent
tprov = json.loads(train_prov[0].read_text())
CKPT = TRAIN / "e14_q" / "final.pt"
assert CKPT.is_file() and sha256(CKPT) == tprov["arms"]["e14_q"]["final_sha256"], \
    "e14_q: final.pt is not the one E14_train recorded"
print("checkpoint verified:", CKPT)
'''),
code(ENV),
code(r'''
# --- 3. tables, the registered jobs, the model ---------------------------------------
''' + TABLES_CELL + r'''
OUT = WORK / "e14_gen"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = [(a, m) for a in E14.GENERATE_ARMS for m in ("free", "control")]
jobs = sorted(E14.jobs(E14.GENERATE_ARMS), key=lambda j: (ORDER.index((j["arm"], j["mode"])),
                                                          j["label"], j["state"] or "", j["g"]))
print(f"{len(jobs)} files, {sum(j['n'] for j in jobs):,} URLs")
CKPT_SHA = {"e14_q": sha256(CKPT)}
PURPOSE = "E14 (research/e14/PREREG_E14.md): e14_q and e14_qd; generation only"


def model_for(weights):
    assert weights == "e14_q", weights
    if weights not in _loaded:
        model, tok, _ = load_checkpoint(CKPT, device=device)
        model.to(device).eval()
        assert tok.presence_plan and tok.root_slash and tok.slash_plan and tok.vocab_size == 110
        _loaded[weights] = (model, tok)
        print(f"   loaded {weights}: vocab {tok.vocab_size}, max_len {model.cfg.max_len}", flush=True)
    return _loaded[weights]
'''),
code(GENERATE_LOOP),
    ]


def write(name, cells):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (HERE / name).write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", HERE / name)


def main() -> None:
    write("E14_decode.ipynb", decode_cells())
    write("E14_train.ipynb", train_cells())
    write("E14_generate.ipynb", generate_cells())


if __name__ == "__main__":
    main()
