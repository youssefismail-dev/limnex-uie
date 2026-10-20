"""Build research/e13/E13_decode.ipynb, E13_train.ipynb and E13_generate.ipynb
(research/e13/PREREG_E13.md section 6).

    python research/e13/make_e13_notebooks.py

All three run the `uie-e13-code` bundle (make_e13_bundle.py). Every constant and
every generation call comes from research/e13/e13_common.py, so the laptop repeats
them exactly. The structure is E12's notebooks', which ran on Kaggle once each.

- E13_decode   the decoder arms (e12_d1, e12_dec) and N's e12_final reference
               sets; needs no training, so it can run first.
- E13_train    e13_ctx (C), then N's A1 and A2 with their query sets.
- E13_generate e13_ctx's 50 files, after the laptop has verified the training.
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
code_hits = sorted(root.rglob("E13_CODE_MANIFEST.json")) if root.exists() else []
if not code_hits:
    raise SystemExit("Attach the uie-e13-code dataset (right panel -> Add Input), then rerun.")
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
for p in (CODE / "research" / "e13", CODE / "research" / "native", CODE):
    sys.path.insert(0, str(p))
for name in [m for m in sys.modules if m == "uie" or m.startswith("uie.")
             or m in ("e8_common", "e11_common", "e12_common", "e13_common", "e13n_train")]:
    del sys.modules[name]
import uie
import e12_common as K
import e13_common as E13
manifest = json.loads((CODE / "E13_CODE_MANIFEST.json").read_text())
for mod in (uie, K, E13):
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
print("torch:", torch.__version__, "| gpu:", torch.cuda.get_device_name(0))
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
                             "gpu": torch.cuda.get_device_name(0),
                             "tldextract": metadata.version("tldextract")}})
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
    if "cell" in j:                                   # N's per-cell query set
        model, tok = model_for(j["model"])
        urls = E13.n_urls(model, tok, j["cell"], j["g"], shipped, presence_slash, tables,
                          device=device)
    elif j["mode"] == "free":
        model, tok = model_for(E13.ARMS[j["arm"]]["weights"])
        urls = E13.free_urls(model, tok, j["arm"], j["label"], j["g"], shipped, presence_slash,
                             settings=settings, device=device, plan_trace=trace,
                             slash_trace=slash if E13.ARMS[j["arm"]]["slash_rule"] else None)
    else:
        model, tok = model_for(E13.ARMS[j["arm"]]["weights"])
        urls = E13.control_urls(model, tok, j["arm"], j["label"], j["state"], j["g"], tables,
                                presence_slash, settings=settings, device=device,
                                plan_trace=trace,
                                slash_trace=slash if E13.ARMS[j["arm"]]["slash_rule"] else None)
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


def decode_cells():
    return [
md(r'''
# E13 — the decoder arms and N's reference sets (generation only; nothing is trained)

Registration: `research/e13/PREREG_E13.md` §§2.2–2.5. The weights are **`e12_final`'s,
unchanged** (carried in the bundle, checked against E12's records).

| set | what | files |
|---|---|---|
| `e12_d1` | `e12_final` + **D1**, the slash rule (owner-authorised decoder override) | 50 |
| `e12_dec` | `e12_final` + D1 + **D2**, the fitted field calibration (only if the fit applied a scope) | 50 |
| N reference | `e12_final`'s query sets for N's three cells, raw decode | 15 |

Every arm uses **E12's seeds**, so each cell pairs with `e12_final`'s own files.
Neither rule changes a weight.

**Attach one input:** **`uie-e13-code`**. **Settings:** GPU T4, Internet **On**.
**Save Version → Save & Run All.** About 3.5 h. A rerun keeps every finished file with
its record. When done, download `/kaggle/working/e13_decode`.
'''),
code(LOCATE_CODE),
code(ENV),
code(r'''
# --- 3. tables, the D2 settings, the registered jobs, the model ----------------------
from uie.controlled import StateSuffixTables
from uie.suffixes import SuffixTable
from uie.train import load_checkpoint

OUT = WORK / "e13_decode"
OUT.mkdir(parents=True, exist_ok=True)
shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
presence_slash = json.loads((CODE / "presence_slash_table.json").read_text())["table"]
settings = E13.load_settings(CODE / E13.SETTINGS_FILE)
applied = [s for s, r in settings["scopes"].items() if r.get("applied")]
print("D2 scopes applied:", applied or "none -> e12_dec is NOT APPLICABLE and not generated")
device = torch.device("cuda")
arms = ["e12_d1"] + (["e12_dec"] if applied else [])
ORDER = [(a, "free") for a in arms] + [(a, "control") for a in arms]
jobs = sorted(E13.jobs(tuple(arms)), key=lambda j: (ORDER.index((j["arm"], j["mode"])),
                                                    j["label"], j["state"] or "", j["g"]))
jobs += E13.n_jobs(("e12_final",))
print(f"{len(jobs)} files, {sum(j['n'] for j in jobs):,} URLs")
CKPT_SHA = {"e12_final": manifest["e12_final"]}
PURPOSE = "E13 (research/e13/PREREG_E13.md): decoder arms and N's reference; generation only"
_loaded = {}


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
# E13 — training: the context arm, then N2a's two diagnostic arms

Registration: `research/e13/PREREG_E13.md` §§2.1, 2.5.

| arm | what |
|---|---|
| `e13_ctx` | `e12_final`'s recipe + **ContextSwap**: for the phishing query rows, half the draws put a host and path `e12_final` wrote before the real query; the model-written part is input only, never a target |
| N `A1` | `e12_final`'s recipe on query rows only, each row at `e12_final`'s exposure (diagnostic) |
| N `A2` | the same at four times the exposure (diagnostic) |

**Before any training, cell 3 rebuilds E11's start and checks it**, and checks that a
384-token step fits. If either fails it stops before a session is spent.

**Attach two inputs** (right panel → *Add Input*):
1. **`uie-corpus2`**, the data bundle E7–E12 used;
2. **`uie-e13-code`**.

**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About 2.5 h.
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
assert {"context_prefixes", "context_p", "slash_plan", "max_len", "e12_start"} <= set(
    inspect.signature(run).parameters)
import e13n_train as N
assert N.EXPECTED_KEPT and N.E12_FINAL_SHA256 == manifest["e12_final"], "N's counts not registered"
print("data bundle: all 8 streams present; E13 options present")
print("e13_ctx:", E13.recipe_ctx(CODE / E13.PREFIX_FILE))
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
# --- 4. train e13_ctx (skipped if finished) -----------------------------------------
import time

PROV = WORK / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"arms": {}}
prov.update({"registration": E13.REGISTRATION, "code_commit": manifest["commit"],
             "start_check": {"e10": e10, "e11": e11},
             "prefixes_sha256": manifest["tables"][E13.PREFIX_FILE]})
out = WORK / "e13_ctx"
if (out / "train_result.json").exists() and (out / "final.pt").exists():
    print("e13_ctx: finished earlier -- skipped")
else:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run(DATA, out, **E13.recipe_ctx(CODE / E13.PREFIX_FILE))
    prov["arms"]["e13_ctx"] = {"wall_minutes": round((time.time() - t0) / 60, 1)}
    PROV.write_text(json.dumps(prov, indent=1))
'''),
code(r'''
# --- 5. read e13_ctx back: settings, the start (V-id) and ContextSwap (V5) ------------
problems = []
ck = torch.load(WORK / "e13_ctx" / "final.pt", map_location="cpu", weights_only=False)
tc, mc, tk = ck["train_config"], ck["model_config"], ck["tokenizer"]
r = json.loads((WORK / "e13_ctx" / "train_result.json").read_text())
rec = E13.recipe_ctx(CODE / E13.PREFIX_FILE)
s = r.get("e12_start") or {}
cs = r.get("context_swap") or {}
want = [("steps", tc.get("steps"), 20000), ("batch_size", tc.get("batch_size"), 256),
        ("seed", tc.get("seed"), 20260803), ("loss_norm", tc.get("loss_norm"), "fixed"),
        ("loss_constant", round(r.get("loss_constant") or 0, 3), round(rec["loss_constant"], 3)),
        ("grad_clip_late", tc.get("grad_clip_late"), 5.0), ("max_len", mc.get("max_len"), 384),
        ("vocab_size", mc.get("vocab_size"), 110), ("slash_plan", tk.get("slash_plan"), True),
        ("context_p", tc.get("context_p"), E13.CONTEXT_P),
        ("context_seed", tc.get("context_seed"), E13.CONTEXT_SEED),
        ("init weights (V-id)", r.get("init_weights_sha256"), E13.E12_FINAL_INIT)]
want += [(f"start: {k}", s.get(k), v) for k, v in E13.E12_FINAL_START.items()]
want += [("matched rows (V5)", cs.get("matched_rows"), cs.get("listed_rows"))]
wrong = [w for w in want if w[1] != w[2]]
problems += wrong
drawn = cs.get("eligible_draws") or 0
decided = (cs.get("swapped") or 0) + (cs.get("too_long") or 0)
share = decided / drawn if drawn else float("nan")
band = 3 * (0.25 / drawn) ** 0.5 if drawn else float("nan")
if not abs(share - E13.CONTEXT_P) <= band:
    problems.append(("swap share (V5)", share, f"{E13.CONTEXT_P} +- {band:.4f}"))
prov["arms"].setdefault("e13_ctx", {})
prov["arms"]["e13_ctx"].update({"best_sha256": sha256(WORK / "e13_ctx" / "best.pt"),
                                "final_sha256": sha256(WORK / "e13_ctx" / "final.pt"),
                                "init_weights_sha256": r.get("init_weights_sha256"),
                                "context_swap": cs, "final_val": r.get("final"),
                                "counts": r.get("counts")})
PROV.write_text(json.dumps(prov, indent=1))
print(f"e13_ctx val {r.get('final')}; ContextSwap {cs}; swap share {share:.4f} (band +- {band:.4f})")
print("MISMATCHES:", problems if problems else "none -- e13_ctx is as registered")
'''),
code(r'''
# --- 6. N: A1 then A2, each trained from E12's start, then its query sets ------------
for arm in ("A1", "A2"):
    out = WORK / f"n_{arm}"
    if (out / "N_RECORD.json").exists():
        print(f"N {arm}: finished earlier -- skipped")
        continue
    t0 = time.time()
    rec = N.run(DATA, CODE / "e12_final.pt", arm, out, CODE)
    prov["arms"][f"n_{arm}"] = {"final_sha256": rec["final_sha256"], "schedule": rec["schedule"],
                                "kept_query_rows": rec["kept_query_rows"],
                                "wall_minutes": round((time.time() - t0) / 60, 1)}
    PROV.write_text(json.dumps(prov, indent=1))
print("\nall arms trained. Now download /kaggle/working (e13_ctx, n_A1, n_A2, PROVENANCE.json).")
'''),
    ]


def generate_cells():
    return [
md(r'''
# E13 — `e13_ctx`'s generation, from `final.pt` (nothing is scored here)

Registration: `research/e13/PREREG_E13.md` §2.4. Run this **after the laptop has
verified the training**. It writes `e13_ctx`'s 50 files with the raw decode at E12's
seeds, so each cell pairs with `e12_final`'s own files.

**Attach two inputs:** **`uie-e13-code`**, and the **E13_train** notebook's **output**.
**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About 1.6 h.
When done, download `/kaggle/working/e13_gen`.
'''),
code(LOCATE_CODE + r'''
train_prov = sorted(p for p in root.rglob("PROVENANCE.json")
                    if (p.parent / "e13_ctx").is_dir()) if root.exists() else []
if not train_prov:
    raise SystemExit("Attach the E13_train notebook's output too, then rerun.")
TRAIN = train_prov[0].parent
tprov = json.loads(train_prov[0].read_text())
CKPT = TRAIN / "e13_ctx" / "final.pt"
assert CKPT.is_file() and sha256(CKPT) == tprov["arms"]["e13_ctx"]["final_sha256"], \
    "e13_ctx: final.pt is not the one E13_train recorded"
print("checkpoint verified:", CKPT)
'''),
code(ENV),
code(r'''
# --- 3. tables, the registered jobs, the model ---------------------------------------
from uie.controlled import StateSuffixTables
from uie.suffixes import SuffixTable
from uie.train import load_checkpoint

OUT = WORK / "e13_gen"
OUT.mkdir(parents=True, exist_ok=True)
shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
presence_slash = json.loads((CODE / "presence_slash_table.json").read_text())["table"]
settings = None                                   # e13_ctx: the raw decode
device = torch.device("cuda")
jobs = E13.jobs(("e13_ctx",))
print(f"{len(jobs)} files, {sum(j['n'] for j in jobs):,} URLs")
CKPT_SHA = {"e13_ctx": sha256(CKPT)}
PURPOSE = "E13 (research/e13/PREREG_E13.md): e13_ctx; generation only"
_loaded = {}


def model_for(weights):
    assert weights == "e13_ctx", weights
    if weights not in _loaded:
        model, tok, _ = load_checkpoint(CKPT, device=device)
        model.to(device).eval()
        assert tok.presence_plan and tok.root_slash and tok.slash_plan and tok.vocab_size == 110
        _loaded[weights] = (model, tok)
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
    (HERE / name).write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("wrote", HERE / name)


def main() -> None:
    write("E13_decode.ipynb", decode_cells())
    write("E13_train.ipynb", train_cells())
    write("E13_generate.ipynb", generate_cells())


if __name__ == "__main__":
    main()
