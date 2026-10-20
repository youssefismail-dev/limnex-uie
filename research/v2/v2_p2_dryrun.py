"""
research/v2/v2_p2_dryrun.py

V2-P2's laptop dry run of the reading pipeline (research/v2/PREREG_V2_P2.md section 15,
step 1: "the reading scripts"), before any Kaggle run. It decides nothing.

`e16_b5`'s own 50 files stand in for both arms, so every reading script runs end to end on
real files:

    python research/v2/v2_p2_dryrun.py --part setup     # -> data/v2/p2_dry/{p2_a,p2_b}/...
    python research/v2/v2_p2_dryrun.py --part rules     # V-rules on the stand-ins
    then, with V2_DATA=data/v2/p2_dry:
      V2_ARM=p2_a / p2_b  research/v2/v2_p1_score.py --v4, --part free|control|misc|copy, --merge
      research/v2/v2_p2_words.py --part samples|run|metrics
      research/v2/v2_p2_compare.py

- **The stand-ins:** each arm's files are `e16_b5`'s, renamed; their PROVENANCE is `e16_b5`'s with
  a piece-rule record of none removed and none outside, and the URLs cut at max_len by the
  character rule (exact for a character model: the encoding with its end token exceeds 384).
  The training record is `e16_cap`'s. VERIFY_TRAIN and VERIFY_GENERATE are written PASS and
  marked dry: what is being checked is the reading, not the run.
- **What a correct pipeline shows:** V-D1b reproduces R2 Part A exactly; both arms read like
  `e16_b5` (NO EFFECT); every guard that compares the arm's files with `e16_b5`'s, which are
  the same files, passes; the word readings differ from `e16_b5`'s only by their own samples.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e17", ROOT / "research" / "e16", ROOT / "research" / "e15",
          ROOT / "research" / "e13", ROOT / "research" / "native"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import v2_p2_common as C  # noqa: E402

DRY = ROOT / "data" / "v2" / "p2_dry"
SRC = ROOT / "data" / "e17" / "decode" / "gen"
E16_TRAIN = ROOT / "data" / "e16" / "train" / "e16_cap"


def part_setup() -> None:
    import torch
    from e8_verify import sha256
    from uie.tokenizer import URLTokenizer
    blob = torch.load(E16_TRAIN / "final.pt", map_location="cpu", weights_only=False)
    tok, max_len = URLTokenizer.from_state_dict(blob["tokenizer"]), blob["model_config"]["max_len"]
    del blob
    gprov = json.loads((SRC / "PROVENANCE.json").read_text(encoding="utf-8"))
    for arm in C.ARMS:
        root = DRY / arm
        gen = root / "generate" / "gen"
        if root.exists():
            shutil.rmtree(root)
        gen.mkdir(parents=True)
        files = {}
        for name, rec in gprov["files"].items():
            new = name.replace(f"_{C.BASELINE}_", f"_{arm}_")
            shutil.copyfile(SRC / name, gen / new)
            urls = json.loads((gen / new).read_text(encoding="utf-8"))
            label = rec["label"]
            cut = [u for u in urls if len(tok.encode(u, label)) + 1 > max_len]
            files[new] = {**rec, "arm": arm, "sha256": sha256(gen / new),
                          "piece_rule": {"fields": {}, "tokens_checked": 0, "violations": 0},
                          "cut_at_max_len": cut}
        prov = {**{k: v for k, v in gprov.items() if k != "files"}, "dry_run": True,
                "piece_tables": C.TABLE_SHA256, "rules_sha256": C.rules_sha256(), "files": files}
        (gen / "PROVENANCE.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
        (root / "train" / arm).mkdir(parents=True)
        shutil.copyfile(E16_TRAIN / "train_result.json", root / "train" / arm / "train_result.json")
        for step in ("TRAIN", "GENERATE"):
            (root / f"VERIFY_{step}.json").write_text(json.dumps(
                {"verdict": "PASS", "dry_run": "stand-in: e16_b5's files and e16_cap's record"}, indent=1),
                encoding="utf-8")
        print(f"{arm}: {len(files)} stand-in files in {gen}; cut at max_len "
              f"{sum(len(f['cut_at_max_len']) for f in files.values())}", flush=True)


def part_standin_scores() -> None:
    """Each stand-in arm's score report and V4 record, taken from e16_b5's published report
    (data/e17/score_report_e16_b5.json) under the arm's name: the stand-ins ARE e16_b5's files,
    so the scorer gives exactly these numbers (the dry run's own `v2_p1_score.py` runs check
    that, file by file). This lets the comparison be dry-run without waiting for them."""
    rr = json.loads((ROOT / "data" / "e17" / "score_report_e16_b5.json").read_text(encoding="utf-8"))
    import e17_compare as C17
    for arm in C.ARMS:
        rep = {"registration": C.REGISTRATION, "arm": arm, "dry_run": "e16_b5's published report, renamed"}
        for block in C17.BLOCKS:
            rep[block] = {arm: rr[block][C.BASELINE]}
        rep["copy"] = {cell: {"real_stream": node["real_stream"], arm: node[C.BASELINE]}
                       for cell, node in rr["copy"].items()}
        (DRY / arm / f"score_report_{arm}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        (DRY / arm / "score").mkdir(parents=True, exist_ok=True)
        v4 = ROOT / "data" / "v2" / "p2_dry" / "p2_a" / "score" / "v4.json"
        if v4.is_file():
            (DRY / arm / "score" / "v4.json").write_text(v4.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"{arm}: stand-in score report written ({len(rep['copy'])} copy cells)", flush=True)


def part_rules() -> None:
    """V-rules' string reading (v2_p2_verify.v_rules) on the stand-ins. e16_b5's generation kept
    no token-level record, so that half reports the records missing (expected here); the
    string half must reproduce e16_b5's published S honouring (a delimiter written inside a
    host field reads as a host end: a handful of URLs, never a decode failure)."""
    import v2_p2_verify as V
    from e13_verify import files_check
    for arm in C.ARMS:
        gen = DRY / arm / "generate" / "gen"
        gprov = json.loads((gen / "PROVENANCE.json").read_text(encoding="utf-8"))
        fails, files = files_check(gen, gprov, C.jobs(arm))
        rules, f2 = V.v_rules(files, gprov)
        s = rules["strings (reported)"]
        print(f"{arm}: files {'OK' if not fails else fails}; strings: {s['urls']:,} URLs, {s['cut_at_max_len']} cut; "
              f"S unhonoured {s['s_unhonoured']}, Q = 1 unbegun {s['b5_unbegun']}; pieces outside "
              f"{rules['pieces_outside_fields']}; record failures (expected: no token record) {f2}", flush=True)
        if s["s_examples"] or s["b5_examples"]:
            print("  examples:", (s["s_examples"] + s["b5_examples"])[:5])


#: the notebooks' Kaggle-only lines, and what the laptop runs instead (part notebooks)
LAPTOP = [
    ('root = Path("/kaggle/input")', 'root = Path(r"{input}")'),
    ('WORK = Path("/kaggle/working")', 'WORK = Path(r"{work}")'),
    ('if not torch.cuda.is_available():\n    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4, then rerun.")\n'
     'GPU = torch.cuda.get_device_name(0)', 'GPU = "cpu (laptop pre-flight)"'),
]


def _cells(name: str) -> list:
    nb = json.loads((HERE / name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _run(src: str, env: dict, subs: list, label: str, required: tuple = ()) -> None:
    """Run one cell with the substitutions that match it; each of `required` must match."""
    applied = []
    for old, new in subs:
        if old in src:
            src = src.replace(old, new)
            applied.append(old)
    missing = [r[:60] for r in required if r not in applied]
    if missing:
        raise SystemExit(f"{label}: a laptop substitution did not match: {missing}")
    print(f"--- {label} ({len(applied)} laptop substitutions)", flush=True)
    exec(compile(src, label, "exec"), env)


#: the pre-flight's stand-in C while the real values are still being computed (a dry bundle only)
PROVISIONAL_C = {"p2_a": 1.0e4, "p2_b": 1.0e4}


def part_notebooks(zip_name: str = "uie-v2p2-code.zip") -> None:
    """The four notebooks' own cells, run on this CPU from the extracted bundle, with only their
    Kaggle-only lines replaced: the input and working folders, the GPU query, and -- where the
    laptop cannot follow -- the registered start values (this torch builds its own start), the
    memory check (one row, on the CPU) and the job sizes (two jobs of 4 URLs). Nothing trains:
    the training and readback cells are not run; generation reads an untrained full-size start
    saved as a checkpoint."""
    import tempfile
    import zipfile
    import torch
    from uie.train import save_checkpoint, TrainConfig
    zpath = ROOT / "data" / "kaggle" / zip_name
    tmp = Path(tempfile.mkdtemp(prefix="v2p2_preflight_", dir=str(DRY.parent)))
    inp, work = tmp / "input", tmp / "working"
    code_dir, data_dir = inp / "uie-v2p2-code", inp / "uie-corpus2"
    with zipfile.ZipFile(zpath) as z:
        z.extractall(code_dir)
    data_dir.mkdir(parents=True)
    (data_dir / "bundle.manifest.json").write_text("{}", encoding="utf-8")
    for stem in ("name_train", "structure_train", "phishing_name_train", "phishing_structure_train",
                 "name_val", "structure_val", "phishing_name_val", "phishing_structure_val"):
        (data_dir / f"{stem}.jsonl").write_text("", encoding="utf-8")
    work.mkdir()
    fmt = {"input": str(inp), "work": str(work)}
    base = [(a, b.format(**fmt)) for a, b in LAPTOP]
    for arm in C.ARMS:
        letter = arm[-1]
        # the training notebook: locate, environment, the start check (this torch's own start
        # values in place of Kaggle's recorded ones) and the memory check's code path
        env: dict = {"__name__": f"preflight_train_{letter}"}
        train = _cells(f"V2P2_train_{letter}.ipynb")
        start_subs = base + [
            ("want = {\"init_weights_sha256\": E15.E12_FINAL_INIT, **E15.E12_FINAL_START}",
             "want = {k: ref[k] for k in (\"init_weights_sha256\",) + tuple(E15.E12_FINAL_START)}"),
            ("off = {k: (START.get(k), v) for k, v in C.E16_START.items() if START.get(k) != v}",
             "off = {}; C.E16_START = {k: START[k] for k in C.E16_START}"),
            ("total = torch.cuda.get_device_properties(0).total_memory / 2**30", "total = 0.0"),
            ("peak = C.memory_preflight(torch.device(\"cuda\"), math.ceil(C.PREFLIGHT_ROWS / parts), ptok.vocab_size)",
             "peak = C.memory_preflight(torch.device(\"cpu\"), 1, ptok.vocab_size)"),
            ("except torch.cuda.OutOfMemoryError:", "except MemoryError:"),
            ("    torch.cuda.empty_cache()\n", "    pass\n")]
        for k, src in enumerate(train[:3]):
            if k == 2 and any(v is None for v in env["C"].LOSS_CONSTANT_P2.values()):
                env["C"].LOSS_CONSTANT_P2 = dict(PROVISIONAL_C)      # a dry bundle: C not pinned yet
                print(f"  (dry bundle: provisional C {PROVISIONAL_C})", flush=True)
            _run(src, env, start_subs, f"V2P2_train_{letter} cell {k + 1}",
                 required=tuple(s[0] for s in start_subs[3:]) if k == 2 else ())
        # a stand-in training output: the untrained full-size start, saved as training saves it
        tout = inp / f"v2p2-train-{letter}"
        (tout / arm).mkdir(parents=True)
        model, _ = C.start(env["ptok"])
        save_checkpoint(model, env["ptok"], TrainConfig(), tout / arm / "final.pt")
        del model
        from e8_verify import sha256
        (tout / "PROVENANCE.json").write_text(json.dumps(
            {"arms": {arm: {"final_sha256": sha256(tout / arm / "final.pt")}}}), encoding="utf-8")
        # the generation notebook: everything, on two jobs of 4 URLs, on the CPU
        genv: dict = {"__name__": f"preflight_generate_{letter}"}
        gen = _cells(f"V2P2_generate_{letter}.ipynb")
        gen_subs = base + [
            ('device = torch.device("cuda")', 'device = torch.device("cpu")'),
            ('assert all(p.device.type == "cuda" for p in model.parameters()), "STOP. the model is not on the GPU"', ''),
            ('jobs = sorted(C.jobs(ARM), key=lambda j: (0 if j["mode"] == "free" else 1, j["label"], j["state"] or "", j["g"]))',
             'jobs = [dict(j, n=2) for j in sorted(C.jobs(ARM), key=lambda j: (0 if j["mode"] == "free" else 1, '
             'j["label"], j["state"] or "", j["g"]))[::25]]'),
            ('shipped, presence_slash, device=device,', 'shipped, presence_slash, device=device, n=j["n"],'),
            ('tables, presence_slash,\n                              device=device,',
             'tables, presence_slash,\n                              device=device, n=j["n"],')]
        need = {2: tuple(s[0] for s in gen_subs[3:6]), 3: tuple(s[0] for s in gen_subs[6:8])}
        for k, src in enumerate(gen):
            _run(src, genv, gen_subs, f"V2P2_generate_{letter} cell {k + 1}", required=need.get(k, ()))
        prov = json.loads((work / f"v2p2_gen_{letter}" / "PROVENANCE.json").read_text(encoding="utf-8"))
        for f, r in prov["files"].items():
            print(f"  {f}: {r['n']} URLs, piece rule {r['piece_rule']['violations']} violations, token rules "
                  f"D1 {r['token_rules']['d1_violations']} / B5 {r['token_rules']['b5_violations']}", flush=True)
        shutil.rmtree(tout)
    print(f"notebook pre-flight done in {tmp} (delete it when read)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("setup", "rules", "notebooks", "standin_scores"))
    ap.add_argument("--zip", default="uie-v2p2-code.zip", help="notebooks: the bundle in data/kaggle")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.part == "notebooks":
        part_notebooks(args.zip)
    else:
        {"setup": part_setup, "rules": part_rules, "standin_scores": part_standin_scores}[args.part]()


if __name__ == "__main__":
    main()
