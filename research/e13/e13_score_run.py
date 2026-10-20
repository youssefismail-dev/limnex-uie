"""E13 -- run the readout's measurements in a memory-guarded pool
(research/e13/PREREG_E13.md section 6, step 6). Logs start / finish / exit per job,
never a reading, to data/e13/score/runner.log; each job writes its own log beside it.

    python research/e13/e13_score_run.py --arms e12_d1 e12_dec --v4 --n
    python research/e13/e13_score_run.py --arms e13_ctx --dx8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / "e13" / "score"
POOL, MIN_FREE, GAP = 3, 1.5 * 2 ** 30, 30
S = "research/e13/e13_score.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--v4", action="store_true")
    ap.add_argument("--n", action="store_true", help="N's measurement (e13n_read --measure)")
    ap.add_argument("--dx8", action="store_true", help="K13-2 (e13_dx8.py)")
    args = ap.parse_args()
    queue = []
    if args.v4:
        queue.append(("v4", [S, "--v4"]))
    if args.n:
        queue.append(("n_measure", ["research/e13/e13n_read.py", "--measure"]))
    if args.dx8:
        queue.append(("dx8_ctx", ["research/e13/e13_dx8.py"]))
    for arm in args.arms:
        for part in ("free", "control", "misc", "copy"):
            queue.append((f"{part}_{arm}", [S, "--part", part, "--arm", arm]))
    OUT.mkdir(parents=True, exist_ok=True)
    status = open(OUT / "runner.log", "a", encoding="utf-8")

    def note(msg):
        status.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        status.flush()

    running, exits, last = {}, {}, 0.0
    note(f"start: {len(queue)} jobs, pool {POOL}")
    while queue or running:
        free = psutil.virtual_memory().available
        if queue and len(running) < POOL and time.time() - last >= GAP and (free >= MIN_FREE or not running):
            name, cmd = queue.pop(0)
            log = open(OUT / f"{name}.log", "w", encoding="utf-8")
            running[name] = (subprocess.Popen([sys.executable, *cmd], cwd=ROOT, stdout=log,
                                              stderr=subprocess.STDOUT), log, time.time())
            last = time.time()
            note(f"started {name} (free {free / 2 ** 30:.1f} GB)")
        time.sleep(10)
        for name in list(running):
            p, log, t0 = running[name]
            if p.poll() is not None:
                log.close()
                exits[name] = p.returncode
                note(f"finished {name} exit {p.returncode} after {(time.time() - t0) / 60:.0f} min")
                del running[name]
    failed = sorted(n for n, c in exits.items() if c)
    note(f"STOP: jobs failed: {failed}" if failed else "all done")


if __name__ == "__main__":
    main()
