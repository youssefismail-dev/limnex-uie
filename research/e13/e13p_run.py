"""E13 P -- run every registered measurement (research/e13/PREREG_E13.md section 2.6)
in a memory-guarded pool, then the readout. Logs start / finish / exit per job,
never a reading, to data/e13p/runner.log; each job writes its own log beside it.
At the end: data/e13p/tables.md from e13p_read.py.

    python research/e13/e13p_run.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from e13p_measure import ARMS, CELLS, OUT  # noqa: E402

POOL, MIN_FREE, GAP = 3, 1.5 * 2 ** 30, 30
M = "research/e13/e13p_measure.py"


def jobs() -> list:
    out = [("near", [M, "--part", "near"])]
    out += [(f"mix_{a}_{c}", [M, "--part", "mix", "--cell", c, "--arm", a]) for c in CELLS for a in ARMS]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = open(OUT / "runner.log", "a", encoding="utf-8")

    def note(msg):
        status.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        status.flush()

    queue, running, exits, last = jobs(), {}, {}, 0.0
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
    if failed:
        note(f"STOP: jobs failed, nothing read: {failed}")
        raise SystemExit(f"jobs failed: {failed}")
    with open(OUT / "tables.md", "w", encoding="utf-8") as fh, \
            open(OUT / "tables.md.err", "w", encoding="utf-8") as err:
        code = subprocess.run([sys.executable, "research/e13/e13p_read.py"], cwd=ROOT, stdout=fh,
                              stderr=err).returncode
    note(f"read exit {code}")
    note("all done")


if __name__ == "__main__":
    main()
