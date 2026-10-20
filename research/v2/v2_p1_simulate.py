"""
research/v2/v2_p1_simulate.py

V2-P1, section 10 step 2: run V2P1_train and V2P1_generate end to end on the laptop CPU,
from the unzipped bundle exactly as Kaggle will see it, before the owner spends a session.
Nothing here is a reading.

    python research/v2/v2_p1_simulate.py        # -> data/v2/p1/sim/simulate.json

What is patched, and only for the simulation (E16 did the same):
- Kaggle's paths -> data/v2/p1/sim/;
- the data: a small copy of corpus2's eight streams (the first rows of each), with a
  `bundle.manifest.json` marker as uie-corpus2 has;
- the GPU: the device is the CPU; the worst-case check runs its code path at a toy size;
- the start check: this torch builds a different start from Kaggle's (E16 amendment 2), so
  the comparisons with the recorded fingerprints are skipped; the start is still built
  twice and must be identical;
- training: 2 steps; generation: three tiny jobs.

The training readback then reports mismatches that the small data must cause (steps, row
counts, the batch digest, the injection, the tokenizer fitted on the small data and the
start it implies). Any OTHER mismatch, or any error, fails the simulation.

RETIRED 2026-10-12 -- it refuses to run. Its one run passed V2P1_train's cells 1-3 from the
unzipped bundle, then froze the laptop in cell 4:
- training the full-size engine on the CPU needs about 3 GB on its own;
- the small data copy made the stream injector add about 780 new rows per old row (the
  real run's rate is exactly 1.0), and one batch then asked for 2.65 GB more.
The owner shut the laptop down. On the owner's decision, nothing trains on the laptop:
training runs only on Kaggle (PREREG_V2_P1_MEMORY.md, Status).
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIM = ROOT / "data" / "v2" / "p1" / "sim"
INPUT = SIM / "input"
WORK_TRAIN = INPUT / "v2p1-train-output"        # V2P1_generate finds it as an attached output
WORK_GEN = SIM / "work_generate"
BUNDLE = ROOT / "data" / "kaggle" / "uie-v2p1-code.zip"
ROWS = 1_500                                     # per training stream; 300 per validation stream
#: The readback mismatches the small data and the laptop's torch must cause in a simulation.
EXPECTED = ("train_config steps", "model_config vocab_size", "row counts", "dropped over max_len", "old rows' batch digest",
            "injection", "tokenizer", "parameters", "start:")


def cells(nb: Path) -> list:
    data = json.loads(nb.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code"]


def patch(src: str, work: Path, pairs) -> str:
    src = src.replace('Path("/kaggle/input")', f'Path(r"{INPUT}")')
    src = src.replace('Path("/kaggle/working")', f'Path(r"{work}")')
    src = src.replace('if not torch.cuda.is_available():\n    raise SystemExit("No GPU. Settings -> '
                      'Accelerator -> GPU T4, then rerun.")\nGPU = torch.cuda.get_device_name(0)',
                      'GPU = "cpu (simulation)"')
    for old, new in pairs:
        if old not in src:
            raise SystemExit(f"simulation patch no longer matches the notebook: {old[:60]!r}")
        src = src.replace(old, new)
    return src


TRAIN_PATCHES = {
    2: [('if off:\n    raise SystemExit(f"STOP. This torch does not rebuild', 'if False:\n    raise SystemExit(f"'),
        ('if off:\n    raise SystemExit(f"STOP. The width-512 start', 'if False:\n    raise SystemExit(f"'),
        ('if MEM_START["base_init_sha256"] != P1.E16_START["init_weights_sha256"] or',
         'if MEM_START["base_init_sha256"] != START["init_weights_sha256"] or'),
        ("total = torch.cuda.get_device_properties(0).total_memory / 2**30", "total = 0.0"),
        ("peak = M.memory_preflight(torch.device(\"cuda\"), math.ceil(288 / parts))",
         "peak = M.memory_preflight(torch.device(\"cpu\"), 2, vocab_size=tok.vocab_size, max_len=48)"),
        ("    torch.cuda.empty_cache()\n    if fits:", "    if fits:")],
    3: [("    torch.cuda.reset_peak_memory_stats()\n", ""),
        # the training record's own small samples and honouring check (2,520 URLs) at 2
        # per call: about 5 s per URL on this CPU at full size
        ("    t0 = time.time()\n    T.run(DATA, out,",
         "    import uie.kaggle_train as _KT\n    _gen = _KT.generate\n"
         "    _KT.generate = lambda *a, **k: _gen(*a, **{**k, 'n': min(k.get('n', 1), 2)})\n"
         "    t0 = time.time()\n    T.run(DATA, out,"),
        ("T.run(DATA, out, **RECIPE)", 'T.run(DATA, out, **{**RECIPE, "steps": 2})'),
        ("torch.cuda.max_memory_allocated() / 2**30", "0.0")],
}
GEN_PATCHES = {
    2: [('device = torch.device("cuda")', 'device = torch.device("cpu")'),
        ('jobs = sorted(P1.jobs(), key=lambda j: (0 if j["mode"] == "free" else 1, j["label"], j["state"] or "", j["g"]))',
         'jobs = [dict(j, n=3) for j in sorted(P1.jobs(), key=lambda j: (0 if j["mode"] == "free" else 1, '
         'j["label"], j["state"] or "", j["g"])) if (j["mode"], j["g"]) == ("free", 0) or '
         '(j["mode"], j["label"], j["state"], j["g"]) == ("control", "benign", "P1Q1", 0)]'),
        ("assert tok.presence_plan and tok.root_slash and tok.slash_plan and tok.vocab_size == 110",
         "assert tok.presence_plan and tok.root_slash and tok.slash_plan"),
        ("and sum(p.numel() for p in model.parameters()) == P1.PARAMS", "")],
    3: [('shipped, presence_slash, device=device,\n                            plan_trace',
         'shipped, presence_slash, device=device, n=j["n"],\n                            plan_trace'),
        ('tables, presence_slash,\n                               device=device, plan_trace',
         'tables, presence_slash,\n                               device=device, n=j["n"], plan_trace')],
}


def small_corpus(dest: Path) -> None:
    src = ROOT / "data" / "corpus2"
    dest.mkdir(parents=True, exist_ok=True)
    for stem in ("name_train", "structure_train", "phishing_name_train", "phishing_structure_train",
                 "name_val", "structure_val", "phishing_name_val", "phishing_structure_val"):
        n = ROWS if stem.endswith("_train") else 300
        with gzip.open(src / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
            lines = [next(fh) for _ in range(n)]
        (dest / f"{stem}.jsonl").write_text("".join(lines), encoding="utf-8")
    (dest / "bundle.manifest.json").write_text(json.dumps({"simulation": True}), encoding="utf-8")


def run_cells(srcs, ns: dict, label: str) -> None:
    for i, src in enumerate(srcs, start=1):
        t0 = time.time()
        print(f"--- {label} cell {i}", flush=True)
        exec(compile(src, f"<{label} cell {i}>", "exec"), ns)
        print(f"--- {label} cell {i} done ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    raise SystemExit("v2_p1_simulate.py is retired: training the full-size engine on this laptop's CPU "
                     "froze it on 2026-10-12. Nothing trains on the laptop (see the module docstring).")
    sys.stdout.reconfigure(encoding="utf-8")
    if SIM.exists():
        shutil.rmtree(SIM)
    (INPUT / "uie-v2p1-code").mkdir(parents=True)
    with zipfile.ZipFile(BUNDLE) as z:
        z.extractall(INPUT / "uie-v2p1-code")
    small_corpus(INPUT / "uie-corpus2")
    WORK_TRAIN.mkdir(parents=True)
    WORK_GEN.mkdir(parents=True)
    rec: dict = {}
    t0 = time.time()
    train = cells(HERE / "V2P1_train.ipynb")
    srcs = [patch(s, WORK_TRAIN, TRAIN_PATCHES.get(i, [])) for i, s in enumerate(train)]
    ns: dict = {"__name__": "__main__"}
    run_cells(srcs, ns, "V2P1_train")
    names = [w[0] for w in ns["problems"]]
    unexpected = [n for n in names if not n.startswith(EXPECTED)]
    rec["train"] = {"mismatches": names, "unexpected": unexpected, "accum": ns["ACCUM"],
                    "final_val": ns["r"].get("final"), "params": ns["r"].get("params"),
                    "memory_start": {k: ns["MEM_START"][k] for k in ("base_init_sha256", "memory_init_sha256",
                                                                     "memory_seed", "params")}}
    for mod in [m for m in sys.modules if m == "uie" or m.startswith("uie.") or m.endswith("_common")
                or m.startswith("v2_")]:
        del sys.modules[mod]
    gen = cells(HERE / "V2P1_generate.ipynb")
    srcs = [patch(s, WORK_GEN, GEN_PATCHES.get(i, [])) for i, s in enumerate(gen)]
    ns = {"__name__": "__main__"}
    run_cells(srcs, ns, "V2P1_generate")
    prov = json.loads((WORK_GEN / "v2p1_gen" / "PROVENANCE.json").read_text(encoding="utf-8"))
    rec["generate"] = {"files": sorted(prov["files"]), "decode": prov["decode"],
                       "examples": json.loads((WORK_GEN / "v2p1_gen" / sorted(prov["files"])[0]).read_text())}
    rec["seconds"] = round(time.time() - t0, 1)
    rec["pass"] = not unexpected and len(prov["files"]) == 3
    (SIM / "simulate.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    print(f"SIMULATION: {'PASS' if rec['pass'] else 'FAIL'} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
