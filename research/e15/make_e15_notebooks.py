"""Build research/e15/E15B_train.ipynb and E15B_generate.ipynb
(research/e15/PREREG_E15.md section 9.12, step 3).

    python research/e15/make_e15_notebooks.py

Both run the `uie-e15-code` bundle (make_e15_bundle.py). Every constant and every
generation call comes from research/e15/e15_common.py, so the laptop repeats them
exactly. The structure is E14's notebooks', which ran on Kaggle once each.

- E15B_train     e15_new: e12_final's recipe + the injected stream; the readback checks
                 the start, the settings, the injection record and the old rows' batch
                 digest against the laptop's replay.
- E15B_generate  e15_new's 50 files (raw + D1 + D1'), after the laptop verifies training.
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
code_hits = sorted(root.rglob("E15_CODE_MANIFEST.json")) if root.exists() else []
if not code_hits:
    raise SystemExit("Attach the uie-e15-code dataset (right panel -> Add Input), then rerun.")
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
for p in (CODE / "research" / "e15", CODE / "research" / "e13", CODE / "research" / "native", CODE):
    sys.path.insert(0, str(p))
for name in [m for m in sys.modules if m == "uie" or m.startswith("uie.")
             or m in ("e8_common", "e11_common", "e12_common", "e13_common", "e15_common")]:
    del sys.modules[name]
import uie
import e12_common as K
import e15_common as E15
manifest = json.loads((CODE / "E15_CODE_MANIFEST.json").read_text())
for mod in (uie, K, E15):
    assert str(CODE) in mod.__file__, f"{mod.__name__} is SHADOWED -- check sys.path"
bad = [f for f, h in manifest["files"].items() if sha256(CODE / f) != h]
bad += [t for t, h in manifest["tables"].items() if sha256(CODE / t) != h]
if sha256(CODE / "e12_final.pt") != manifest["e12_final"]:
    bad.append("e12_final.pt")
if sha256(CODE / E15.STREAM_FILE) != manifest["stream"]["sha256"]:
    bad.append(E15.STREAM_FILE)
assert not bad, f"STOP. files differ from the manifest: {bad}"
print(f"code: commit {manifest['commit'][:12]}; {len(manifest['files'])} files, "
      f"{len(manifest['tables'])} tables, e12_final.pt and the stream ({manifest['stream']['rows']:,} rows) verified")

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
    trace = []
    if j["mode"] == "free":
        urls = E15.free_urls(model, tok, j["label"], j["g"], shipped, presence_slash,
                             device=device, plan_trace=trace)
    else:
        urls = E15.control_urls(model, tok, j["label"], j["state"], j["g"], tables, presence_slash,
                                device=device, plan_trace=trace)
    assert len(urls) == j["n"], (j["file"], len(urls))
    path.write_text(json.dumps(urls))
    prov["files"][j["file"]] = {**{k: v for k, v in j.items() if k != "file"},
                                "sha256": sha256(path), "seconds": round(time.time() - t0, 1),
                                "plan_trace_mean": sum(trace) / len(trace) if trace else None}
    PROV.write_text(json.dumps(prov, indent=1))
    done += 1
    el = (time.time() - t_start) / 60
    print(f"  {done:>3}/{len(jobs)}  {j['file']:<46} {time.time() - t0:6.0f}s  "
          f"({el:.0f} min so far)", flush=True)
print(f"\nall {len(jobs)} files written. Now download {OUT}.")
'''


def train_cells():
    return [
md(r'''
# E15-B — training `e15_new`: `e12_final`'s recipe, with new benign query rows injected

Registration: `research/e15/PREREG_E15.md` §§10, 12, 13. Everything is `e12_final`'s:
the same start, data, seed, batches, 20,000 steps, objective and pinned loss constant.
**One change:** 57,292 new same-population benign URLs with queries are added alongside
the old rows. Whenever a batch holds an old benign domain-frame row with a query, one
new row of the same kind joins the batch. The new rows teach **only their query**;
everything else in them is input. The old rows train exactly as before.

**Before any training, cell 3 rebuilds E11's start and checks it**, and checks that a
slightly larger batch at 384 tokens fits. **Cell 5 reads the run back:** it must print
`MISMATCHES: none -- e15_new is as registered`, including the old rows' batch digest
equal to the laptop's replay.

**Attach two inputs** (right panel → *Add Input*):
1. **`uie-corpus2`**, the data bundle E7–E14 used;
2. **`uie-e15-code`**.

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
assert {"qry_extra", "slash_plan", "max_len", "e12_start", "loss_constant"} <= set(
    inspect.signature(run).parameters)
print("data bundle: all 8 streams present; E15 option present")
RECIPE = E15.recipe_new(CODE / E15.STREAM_FILE)
print("e15_new:", RECIPE)
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
    # 256 rows plus a margin for the appended new rows (about 4.7 per batch on average)
    peak = K.memory_preflight(torch.device("cuda"), batch_size=272)
except torch.cuda.OutOfMemoryError:
    raise SystemExit("STOP. A 272 x 384 training step does not fit on this GPU.")
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"start check PASSED; memory preflight PASSED ({peak:.1f} of {total:.1f} GB)")
'''),
code(r'''
# --- 4. train e15_new (skipped if finished) -----------------------------------------
import time

PROV = WORK / "PROVENANCE.json"
prov = json.loads(PROV.read_text()) if PROV.exists() else {"arms": {}}
prov.update({"registration": E15.REGISTRATION, "code_commit": manifest["commit"],
             "start_check": {"e10": e10, "e11": e11}, "gpu": GPU,
             "stream_sha256": manifest["stream"]["sha256"]})
out = WORK / E15.ARM
if (out / "train_result.json").exists() and (out / "final.pt").exists():
    print(f"{E15.ARM}: finished earlier -- skipped")
else:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run(DATA, out, **RECIPE)
    prov["arms"][E15.ARM] = {"wall_minutes": round((time.time() - t0) / 60, 1)}
    PROV.write_text(json.dumps(prov, indent=1))
'''),
code(r'''
# --- 5. read e15_new back against e12_final's record and Amendment 3 ------------------
ck = torch.load(WORK / E15.ARM / "final.pt", map_location="cpu", weights_only=False)
e12 = torch.load(CODE / "e12_final.pt", map_location="cpu", weights_only=False)
tc, mc, tk = ck["train_config"], ck["model_config"], ck["tokenizer"]
r = json.loads((WORK / E15.ARM / "train_result.json").read_text())
s = r.get("e12_start") or {}
q = r.get("qry_extra") or {}
strata = q.get("strata", {})
want = [(f"train_config {k}", tc.get(k), v) for k, v in e12["train_config"].items()]
want += [(f"model_config {k}", mc.get(k), v) for k, v in e12["model_config"].items()]
want += [("extra settings", sorted(set(tc) - set(e12["train_config"])),
          ["qry_extra", "qry_extra_draw_seed", "qry_extra_rate_seed"]),
         ("draw / rate seeds", (tc.get("qry_extra_draw_seed"), tc.get("qry_extra_rate_seed")),
          (E15.DRAW_SEED, E15.RATE_SEED)),
         ("tokenizer", tk == e12["tokenizer"], True),
         ("init weights", r.get("init_weights_sha256"), E15.E12_FINAL_INIT),
         ("row counts", r.get("counts"), E15.E12_FINAL_COUNTS),
         ("dropped over max_len", r.get("dropped_over_max_len"), E15.E12_FINAL_DROPPED),
         ("loss constant", r.get("loss_constant"), E15.E12_FINAL_LOSS_CONSTANT),
         ("stream sha256", q.get("source_sha256"), manifest["stream"]["sha256"]),
         ("N_new", {k: v["N_new"] for k, v in strata.items()}, E15.N_NEW),
         ("N_old", {k: v["N_old"] for k, v in strata.items()}, E15.N_OLD),
         ("p", {k: v["p"] for k, v in strata.items()}, E15.P_INJECT),
         ("old rows' batch digest (V-batch)", q.get("old_batch_digest"), E15.V_BATCH_DIGEST)]
want += [(f"start: {k}", s.get(k), v) for k, v in E15.E12_FINAL_START.items()]
problems = [w for w in want if w[1] != w[2]]
appended = sum(v["appended_rows"] for v in strata.values())
positions = sum(v["appended_query_positions"] for v in strata.values())
prov["arms"].setdefault(E15.ARM, {})
prov["arms"][E15.ARM].update({"best_sha256": sha256(WORK / E15.ARM / "best.pt"),
                              "final_sha256": sha256(WORK / E15.ARM / "final.pt"),
                              "init_weights_sha256": r.get("init_weights_sha256"),
                              "qry_extra": q, "final_val": r.get("final"), "counts": r.get("counts")})
PROV.write_text(json.dumps(prov, indent=1))
print(f"{E15.ARM} val {r.get('final')}")
print(f"injected: {appended:,} rows ({appended / max(q.get('steps') or 1, 1):.3f} per batch; "
      f"registered {E15.APPENDED_PER_BATCH}), {positions:,} query positions")
print("MISMATCHES:", problems if problems else f"none -- {E15.ARM} is as registered")
print("\nNow download /kaggle/working (e15_new/, PROVENANCE.json).")
'''),
    ]


def generate_cells():
    return [
md(r'''
# E15-B — `e15_new`'s generation, from `final.pt` (nothing is scored here)

Registration: `research/e15/PREREG_E15.md` §§9.7, 13. Run this **after the laptop has
verified the training**. It writes 50 files at E12's seeds with the adopted decode
(raw + D1 + D1′), so each cell pairs with `e12_d1p`'s files, the frozen baseline.

**Attach two inputs:** **`uie-e15-code`**, and the **E15B_train** notebook's **output**.
**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About 2 h.
A rerun keeps every finished file with its record. When done, download
`/kaggle/working/e15_gen`.
'''),
code(LOCATE_CODE + r'''
train_prov = sorted(p for p in root.rglob("PROVENANCE.json")
                    if (p.parent / "e15_new").is_dir()) if root.exists() else []
if not train_prov:
    raise SystemExit("Attach the E15B_train notebook's output too, then rerun.")
TRAIN = train_prov[0].parent
tprov = json.loads(train_prov[0].read_text())
CKPT = TRAIN / "e15_new" / "final.pt"
assert CKPT.is_file() and sha256(CKPT) == tprov["arms"]["e15_new"]["final_sha256"], \
    "e15_new: final.pt is not the one E15B_train recorded"
print("checkpoint verified:", CKPT)
'''),
code(ENV),
code(r'''
# --- 3. tables, the registered jobs, the model ---------------------------------------
from uie.controlled import StateSuffixTables
from uie.suffixes import SuffixTable
from uie.train import load_checkpoint

shipped = SuffixTable.load(CODE / "suffix_table.json")
tables = StateSuffixTables.load(CODE / "state_suffix_tables.json")
presence_slash = json.loads((CODE / "presence_slash_table.json").read_text())["table"]
device = torch.device("cuda")
OUT = WORK / "e15_gen"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = [("free",), ("control",)]
jobs = sorted(E15.jobs(), key=lambda j: (0 if j["mode"] == "free" else 1, j["label"], j["state"] or "", j["g"]))
print(f"{len(jobs)} files, {sum(j['n'] for j in jobs):,} URLs; decode {E15.decode_kwargs()}")
CKPT_SHA = {"e15_new": sha256(CKPT)}
PURPOSE = "E15-B (research/e15/PREREG_E15.md): e15_new, raw + D1 + D1'; generation only"
model, tok, _ = load_checkpoint(CKPT, device=device)
model.to(device).eval()
assert tok.presence_plan and tok.root_slash and tok.slash_plan and tok.vocab_size == 110
print(f"loaded e15_new: vocab {tok.vocab_size}, max_len {model.cfg.max_len}")
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
    write("E15B_train.ipynb", train_cells())
    write("E15B_generate.ipynb", generate_cells())


if __name__ == "__main__":
    main()
