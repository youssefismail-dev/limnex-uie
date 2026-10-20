"""E17 -- run one arm's measurements in a memory-guarded pool (E16's runner, pointed at
E17). Logs start / finish / exit per job, never a reading, to data/e17/score/<arm>/
runner.log; each job writes its own log beside it. V4 runs first if it has not run.

    python research/e17/e17_score_run.py --arm e16_b5     # after VERIFY_DECODE
    python research/e17/e17_score_run.py --arm e17_a      # after VERIFY_GENERATE
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
POOL, MIN_FREE, GAP = 3, 1.5 * 2 ** 30, 30
S = "research/e17/e17_score.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("e16_b5", "e17_a"))
    args = ap.parse_args()
    out = ROOT / "data" / "e17" / "score" / args.arm
    queue = [] if (out.parent / "v4.json").is_file() else [("v4", [S, "--v4"])]
    queue += [(part, [S, "--arm", args.arm, "--part", part]) for part in ("free", "control", "misc", "copy")]
    out.mkdir(parents=True, exist_ok=True)
    status = open(out / "runner.log", "a", encoding="utf-8")

    def note(msg):
        status.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        status.flush()

    running, exits, last = {}, {}, 0.0
    note(f"start: {len(queue)} jobs, pool {POOL}")
    while queue or running:
        free = psutil.virtual_memory().available
        if queue and len(running) < POOL and time.time() - last >= GAP and (free >= MIN_FREE or not running):
            name, cmd = queue.pop(0)
            log = open(out / f"{name}.log", "w", encoding="utf-8")
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
