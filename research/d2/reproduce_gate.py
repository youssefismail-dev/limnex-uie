"""
research/d2/reproduce_gate.py — REPRODUCE.md level 2: regenerate the E3 gate's 50 files, then read them.

    python research/d2/reproduce_gate.py generate --weights limnex-uie-e16_b5.pt --device cuda
    python research/d2/reproduce_gate.py score

generate  The 50 files of the published decode run (E17_decode's jobs: 10 free-mode files of 8,000
          URLs, 40 control-mode files of 3,000; e16_b5's decode), written exactly as that run wrote
          them, with a PROVENANCE.json, into data/repro/gen. Each file's sha256 is compared with the
          one the published run recorded (research/d2/GATE_FINGERPRINTS.json) as it is written. It
          resumes: a file already matching its fingerprint is kept. Needs only the release: the
          package (its tables in uie/assets), the experiment code in research/, and the published
          model, which uie.load verifies. About 2.5 hours on a Kaggle T4
          (research/d2/reproduce_gate.ipynb runs it); a CPU takes days.
score     The gate's instruments (research/native/e12_score.py, unchanged) on those 50 files against
          the corpus rebuilt in level 2a: free mode, control mode, and the plan readings (honouring,
          emission, slash, path honouring, plan trace). Every number is then compared with the
          published score report (research/d2/score_report_e16_b5.json), the source of every table
          in research/E3_FINAL_REPORT.md section 4. Refuses files that are not the published ones.
          About 2.5 hours on a laptop.

Novelty and query copying are not part of level 2: E3 measured them against the model's added
training stream (E15), which holds real phishing URLs and is not published; level 3 rebuilds it.
Generated URLs are written only into data/repro/ (git-ignored) and never printed: they include
phishing-style URLs, which the README's Responsible Use section covers.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "research" / "e17", ROOT / "research" / "e16", ROOT / "research" / "e15",
          ROOT / "research" / "e13", ROOT / "research" / "native", ROOT / "research" / "cr1",
          ROOT / "research" / "dc1", ROOT / "research" / "qc1", ROOT / "research" / "sc1",
          ROOT / "research" / "pc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from uie.api import ASSET_SHA256, ASSETS, _sha256  # noqa: E402

D2 = ROOT / "research" / "d2"
FINGERPRINTS = D2 / "GATE_FINGERPRINTS.json"
PUBLISHED_REPORT = D2 / "score_report_e16_b5.json"
#: the same 50 files read against our own clean-room rebuild of the corpus (D2 step 6)
CLEANROOM_REPORT = D2 / "score_report_e16_b5_cleanroom.json"
DEFAULT_GEN = ROOT / "data" / "repro" / "gen"
ARM = "e16_b5"
#: the sections of the published report that level 2 reproduces (see the module docstring)
COMPARED = ("free", "control", "honouring", "emission", "slash", "s_honouring", "plan_trace")


def asset(name: str) -> Path:
    path = ASSETS / name
    if _sha256(path) != ASSET_SHA256[name]:
        raise SystemExit(f"STOP: the packaged {name} is not the published one; reinstall the package")
    return path


def fingerprints() -> dict:
    return json.loads(FINGERPRINTS.read_text(encoding="utf-8"))["files"]


def cmd_generate(args) -> int:
    import torch
    import uie
    import e17_common as E17
    from importlib import metadata
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable

    fp = fingerprints()
    engine = uie.load(args.weights, device=args.device)     # refuses anything but the published model
    model, tok, device = engine.model, engine.tokenizer, engine.device
    shipped = SuffixTable.load(asset("suffix_table.json"))
    tables = StateSuffixTables.load(asset("state_suffix_tables.json"))
    presence_slash = json.loads(asset("presence_slash_table.json").read_text(encoding="utf-8"))["table"]
    jobs = sorted(E17.jobs(E17.DECODE_ARM),
                  key=lambda j: (0 if j["mode"] == "free" else 1, j["label"], j["state"] or "", j["g"]))
    if sorted(j["file"] for j in jobs) != sorted(fp):
        raise SystemExit("STOP: the job list is not the published run's 50 files")
    if args.only:
        unknown = sorted(set(args.only) - set(fp))
        if unknown:
            raise SystemExit(f"unknown gate file(s): {unknown}")
        jobs = [j for j in jobs if j["file"] in args.only]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prov_path = out / "PROVENANCE.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {"files": {}}
    prov.update({"purpose": "REPRODUCE.md level 2: the published decode run, regenerated",
                 "checkpoint_sha256": _sha256(Path(args.weights)), "decode": E17.decode_kwargs(True),
                 "environment": {"torch": torch.__version__, "python": platform.python_version(),
                                 "device": str(device), "tldextract": metadata.version("tldextract")}})
    t_start, matched = time.time(), 0
    total = len(jobs)
    for i, j in enumerate(jobs, 1):
        path, want = out / j["file"], fp[j["file"]]["sha256"]
        if path.is_file() and _sha256(path) == want:
            matched += 1
            print(f"  {i:>2}/{total}  {j['file']:<36} already here, matches", flush=True)
            continue
        t0 = time.time()
        trace, qtrace = [], []
        if j["mode"] == "free":
            urls = E17.free_urls(model, tok, j["label"], j["g"], shipped, presence_slash, b5=True,
                                 device=device, plan_trace=trace, query_trace=qtrace)
        else:
            urls = E17.control_urls(model, tok, j["label"], j["state"], j["g"], tables, presence_slash,
                                    b5=True, device=device, plan_trace=trace, query_trace=qtrace)
        if len(urls) != j["n"]:
            raise SystemExit(f"STOP: {j['file']} has {len(urls)} URLs, not {j['n']}")
        path.write_text(json.dumps(urls), encoding="utf-8")
        got = _sha256(path)
        matched += got == want
        prov["files"][j["file"]] = {**{k: v for k, v in j.items() if k != "file"},
                                    "sha256": got, "seconds": round(time.time() - t0, 1),
                                    "plan_trace_mean": sum(trace) / len(trace) if trace else None,
                                    "query_moved_mean": sum(qtrace) / len(qtrace) if qtrace else None,
                                    "query_rows_moved": sum(x > 0 for x in qtrace) if qtrace else None}
        prov_path.write_text(json.dumps(prov, indent=1), encoding="utf-8")
        print(f"  {i:>2}/{total}  {j['file']:<36} {time.time() - t0:6.0f}s  "
              f"{'matches' if got == want else 'DIFFERS from'} the published file  "
              f"({(time.time() - t_start) / 60:.0f} min so far)", flush=True)
    print(f"{matched} of {total} files identical to the published run")
    return 0 if matched == total else 1


def same(a, b, where="", diffs=None):
    """Every key and number of the published report, recursively (floats to 1e-9)."""
    diffs = [] if diffs is None else diffs
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append(f"{where}/{k}: only in {'published' if k in a else 'yours'}")
            else:
                same(a[k], b[k], f"{where}/{k}", diffs)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{where}[{i}]", diffs)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if math.isnan(a) and math.isnan(b):      # an unmeasurable cell is NaN on both sides
            return diffs
        if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
            diffs.append(f"{where}: published {a!r}, yours {b!r}")
    elif a != b:
        diffs.append(f"{where}: published {a!r}, yours {b!r}")
    return diffs


def gated_cells(report) -> dict:
    """(mode, cell, field) -> (verdict, x) for the 32 gated cells: CLEARED, CONVICTED or UNDECIDED
    (auxiliary and not-computable entries are not gated)."""
    out = {}
    for mode in ("free", "control"):
        for cell, node in report[mode][ARM].items():
            for field, r in node["fields"].items():
                a = r.get("acceptance") or {}
                if a.get("verdict") in ("CLEARED", "CONVICTED", "UNDECIDED"):
                    out[(mode, cell, field)] = (a["verdict"], a["x"])
    return out


def tally(cells: dict) -> str:
    v = [r[0] for r in cells.values()]
    return f"{v.count('CLEARED')}/{v.count('CONVICTED')}/{v.count('UNDECIDED')}"


def cmd_score(args) -> int:
    gen, bundle = Path(args.gen), Path(args.bundle)
    if gen.name != "gen":
        raise SystemExit("--gen must be the folder named gen that `generate` wrote")
    fp = fingerprints()
    bad = [f for f, r in fp.items() if not (gen / f).is_file() or _sha256(gen / f) != r["sha256"]]
    if bad:
        raise SystemExit(f"STOP: {len(bad)} of 50 files are missing or not the published ones "
                         f"(first: {bad[0]}); nothing is read")
    import e12_score as S12
    S12.K.SLASH_ARMS = frozenset(set(S12.K.SLASH_ARMS) | {ARM})
    S12.PS_PATH = asset("presence_slash_table.json")
    # the plan readings read the state tables from data/; they must be the published ones
    tables = ROOT / "data" / "state_suffix_tables.json"
    if not tables.is_file():
        tables.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(asset("state_suffix_tables.json"), tables)
    elif _sha256(tables) != ASSET_SHA256["state_suffix_tables.json"]:
        raise SystemExit(f"STOP: {tables} is not the published state table")
    ns = argparse.Namespace(e12=gen.parent, bundle=bundle, label=None)
    # The plan readings rebuild each URL's supplied plan from the suffix table generation used,
    # which is the packaged one; part_misc reads `bundle/suffix_table.json`, which on a rebuilt
    # corpus is a different sample's table (found by the clean room: a suffix the published plans
    # never had). So that one table is always the packaged one; the real URLs stay the bundle's.
    import uie.suffixes as SUF
    real_load, published_table = SUF.SuffixTable.load, asset("suffix_table.json")
    SUF.SuffixTable.load = staticmethod(lambda path, *a, **k: real_load(
        published_table if Path(path).name == "suffix_table.json" else path, *a, **k))
    report = {}
    try:
        for name, part in (("free", S12.part_free), ("control", S12.part_control), ("misc", S12.part_misc)):
            saved = gen.parent / f"score_part_{name}.json"     # a finished part survives a later crash
            if saved.is_file():
                report.update(json.loads(saved.read_text(encoding="utf-8")))
                print(f"[{name}] kept from {saved.name}", flush=True)
                continue
            t0 = time.time()
            print(f"[{name}]", flush=True)
            out = part(ns, (ARM,))
            saved.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
            report.update(out)
            print(f"[{name}] done in {(time.time() - t0) / 60:.0f} min", flush=True)
    finally:
        SUF.SuffixTable.load = real_load
    mine = gen.parent / f"score_report_{ARM}.json"
    mine.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    # one JSON round trip on both sides, so the comparison sees what the files hold
    yours = json.loads(mine.read_text(encoding="utf-8"))
    published = json.loads(PUBLISHED_REPORT.read_text(encoding="utf-8"))
    diffs = []
    for sec in COMPARED:
        same(published[sec][ARM], yours[sec][ARM], sec, diffs)
    print(f"wrote {mine}")
    print(f"compared with the published report: sections {', '.join(COMPARED)}")
    if not diffs:
        print("PASS (identical): every number identical to the published score report, so the E3 "
              "tables (research/E3_FINAL_REPORT.md section 4) are reproduced exactly")
        return 0
    # A rebuilt corpus is the same method, not the same sample (REPRODUCE.md 2a), and the gate's
    # published uncertainty counts only the variation between generations, not the variation of
    # the real reference sample. So there is no pass/fail rule to apply here: the readings are
    # shown beside the two measured ones, the published and our own clean-room rebuild's.
    print(f"not identical: {len(diffs)} numbers differ, as expected on a rebuilt corpus "
          f"(REPRODUCE.md 2a and 2c)")
    clean = json.loads(CLEANROOM_REPORT.read_text(encoding="utf-8"))
    for line in describe(published, clean, yours):
        print(line)
    print("COMPARED: REPRODUCE.md 2c says what a rebuilt corpus is expected to show")
    return 0


def describe(published, clean, yours) -> list:
    """Your gated cells beside the published reading and our clean-room rebuild's."""
    p, c, y = gated_cells(published), gated_cells(clean), gated_cells(yours)
    free = [k for k in p if k[0] == "free"]
    out = [f"the 32 gated cells (CLEARED / CONVICTED / UNDECIDED):  published {tally(p)}   "
           f"our clean room {tally(c)}   yours {tally(y)}",
           f"free mode: {sum(y[k][0] == p[k][0] for k in free)} of {len(free)} verdicts as published "
           f"(our clean room: {sum(c[k][0] == p[k][0] for k in free)} of {len(free)})"]
    moved = [k for k in sorted(p) if y[k][0] != p[k][0] or c[k][0] != p[k][0]]
    if moved:
        out.append("cells whose verdict differs from the published one, here or in our clean room (x):")
        out += [f"  {'/'.join(k):<34} published {p[k][0]} {p[k][1]:.4f}   clean room {c[k][0]} "
                f"{c[k][1]:.4f}   yours {y[k][0]} {y[k][1]:.4f}" for k in moved]
    out.append(f"largest shift of a cell's x from the published value: yours "
               f"{max(abs(y[k][1] - p[k][1]) for k in p):.4f}, our clean room "
               f"{max(abs(c[k][1] - p[k][1]) for k in p):.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="REPRODUCE.md level 2: the E3 gate, regenerated and read")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="the gate's 50 files (GPU)")
    g.add_argument("--weights", required=True, help="the published model file, limnex-uie-e16_b5.pt")
    g.add_argument("--device", default="cuda", help="cuda (default) or cpu (days)")
    g.add_argument("--out", default=str(DEFAULT_GEN))
    g.add_argument("--only", nargs="+", help="just these gate files (e.g. one, to check a CPU)")
    s = sub.add_parser("score", help="read the 50 files against the rebuilt corpus (laptop)")
    s.add_argument("--gen", default=str(DEFAULT_GEN), help="the folder `generate` wrote")
    s.add_argument("--bundle", default=str(ROOT / "data" / "bundle2"), help="the corpus bundle from level 2a")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    return cmd_generate(args) if args.cmd == "generate" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
