"""
research/d2/quick_check.py — REPRODUCE.md level 2, the quick check without a GPU.

    python research/d2/quick_check.py --weights limnex-uie-e16_b5.pt
    python research/d2/quick_check.py --weights limnex-uie-e16_b5.pt --file free_e16_b5_benign_g3.json

Regenerates one of the E3 gate's free-mode files on this computer with the published package (uie's
own decode and packaged tables) and the published model file (uie.load verifies it), and compares
its sha256 with the one the GPU run recorded for that file when the gate was read
(research/d2/GATE_FINGERPRINTS.json, copied from the decode run's PROVENANCE.json). A match means
this computer, this package and this model produce exactly the URLs the gate read. Needs no corpus
and no GPU. Control-mode files need the state tables that level 2a rebuilds, so they are not offered.

The generated URLs stay in memory: nothing is written and no URL is printed.

    python research/d2/quick_check.py --record     # maintainers: GATE_FINGERPRINTS.json from
                                                   # data/e17/decode/gen/PROVENANCE.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FINGERPRINTS = ROOT / "research" / "d2" / "GATE_FINGERPRINTS.json"
PROVENANCE = ROOT / "data" / "e17" / "decode" / "gen" / "PROVENANCE.json"
DEFAULT_FILE = "free_e16_b5_benign_g0.json"


def fingerprint(urls) -> str:
    """The gate files' own bytes: json.dumps(urls) in UTF-8, no trailing newline."""
    return hashlib.sha256(json.dumps(urls).encode("utf-8")).hexdigest()


def record() -> None:
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    keep = ("mode", "label", "state", "g", "n", "seed", "sha256")
    out = {"source": "the E17 decode run's PROVENANCE.json: e16_b5 on a Kaggle T4, the run the E3 "
                     "gate read (research/E3_FINAL_REPORT.md)",
           "code_commit": prov["code_commit"],
           "environment": prov["environment"],
           "fingerprint": "sha256 of the file's bytes: json.dumps(urls), UTF-8, no trailing newline",
           "files": {f: {k: v[k] for k in keep} for f, v in sorted(prov["files"].items())}}
    FINGERPRINTS.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {FINGERPRINTS.relative_to(ROOT)}: {len(out['files'])} files")


def check(weights: str, name: str, device: str) -> int:
    import uie
    from uie.api import LABELS, _asset
    from uie.engine import generate_free
    files = json.loads(FINGERPRINTS.read_text(encoding="utf-8"))["files"]
    free = sorted(f for f, j in files.items() if j["mode"] == "free")
    if name not in free:
        raise SystemExit(f"{name!r} is not a free-mode gate file; choose one of: {', '.join(free)}")
    job = files[name]
    engine = uie.load(weights, device=device)             # refuses anything but the published model
    suffix = _asset("suffix_table.json")
    t0 = time.time()
    urls = generate_free(engine.model, engine.tokenizer, label=job["label"], n=job["n"], seed=job["seed"],
                         presence_slash=_asset("presence_slash_table.json")["table"],
                         shipped={k: suffix[k] for k in LABELS}, device=engine.device)
    got = fingerprint(urls)
    print(f"{name}: {len(urls):,} URLs regenerated in {time.time() - t0:,.0f} s on {device}")
    print(f"  this computer  {got}")
    print(f"  the gate's run {job['sha256']}")
    if got == job["sha256"]:
        print("PASS: identical to the file the E3 gate read")
        return 0
    print("FAIL: not the file the E3 gate read")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="REPRODUCE.md level 2: the quick check without a GPU")
    ap.add_argument("--weights", help="the published model file, limnex-uie-e16_b5.pt")
    ap.add_argument("--file", default=DEFAULT_FILE, help="a free-mode gate file (default: %(default)s)")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--record", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.record:
        record()
        return 0
    if not args.weights:
        ap.error("--weights is required")
    return check(args.weights, args.file, args.device)


if __name__ == "__main__":
    sys.exit(main())
