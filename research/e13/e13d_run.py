"""E13-D -- run every registered measurement in a memory-guarded pool, then the
readout (research/e13/PREREG_E13D.md section 6). Logs only start / finish / exit
per job, never a reading, to data/e13d/runner.log. Each job writes its own log
beside it. At the end: data/e13d/tables.md, from e13d_read.py, which checks
validity before it reads anything.

    python research/e13/e13d_run.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e13d_common as D  # noqa: E402

POOL, MIN_FREE, GAP = 5, 1.5 * 2 ** 30, 45
M = "research/e13/e13d_measure.py"
DX7 = "--dx7" in sys.argv       # amendment 2: only DX7's fit jobs, then its reader
DX8 = "--dx8" in sys.argv       # amendment 3: only DX8's context jobs, then its reader


def jobs() -> list:
    if DX7:
        return [(f"fit_{c[0]}", [M, "--part", "fit", "--cell", c[0]]) for c in D.CELLS + (D.REFERENCE,)]
    if DX8:
        return [(f"context_{c[0]}", [M, "--part", "context", "--cell", c[0]]) for c in D.CELLS]
    out = [("validity", [M, "--part", "validity"])]
    out += [(f"decomp_{c[0]}", [M, "--part", "decomp", "--cell", c[0]]) for c in D.CELLS]
    out += [(f"quantity_{c[0]}", [M, "--part", "quantity", "--cell", c[0]]) for c in D.QUERY_CELLS]
    out += [("probe", [M, "--part", "probe"]), ("describe", [M, "--part", "describe"])]
    return out


def main() -> None:
    D.OUT.mkdir(parents=True, exist_ok=True)
    status = open(D.OUT / "runner.log", "a", encoding="utf-8")

    def note(msg):
        status.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        status.flush()

    queue, running, exits, last = jobs(), {}, {}, 0.0
    note(f"start: {len(queue)} jobs, pool {POOL}")
    while queue or running:
        free = psutil.virtual_memory().available
        if queue and len(running) < POOL and time.time() - last >= GAP and (free >= MIN_FREE or not running):
            name, cmd = queue.pop(0)
            log = open(D.OUT / f"{name}.log", "w", encoding="utf-8")
            running[name] = (subprocess.Popen([sys.executable, *cmd], cwd=D.ROOT, stdout=log,
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
    reader, tables = (("research/e13/e13d_read_dx7.py", "tables_dx7.md") if DX7 else
                      ("research/e13/e13d_read_dx8.py", "tables_dx8.md") if DX8 else
                      ("research/e13/e13d_read.py", "tables.md"))
    with open(D.OUT / tables, "w", encoding="utf-8") as fh, \
            open(D.OUT / (tables + ".err"), "w", encoding="utf-8") as err:
        code = subprocess.run([sys.executable, reader], cwd=D.ROOT, stdout=fh, stderr=err).returncode
    note(f"read exit {code}")
    note("all done")


if __name__ == "__main__":
    main()
