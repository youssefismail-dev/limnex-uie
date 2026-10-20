"""E12 -- the laptop readout, end to end (research/native/PREREG_E12.md section 5,
step 6): every scoring job in a pool of processes, longest first, a job starting
only when memory allows; then the merge, the criteria and the tables.

    python research/native/e12_verify.py --train-only   # once E12_train's output is in data/e12/train
    python research/native/e12_readout.py               # once E12_generate's is in data/e12/gen

The runner logs only start / finish / exit per job, never a reading, to
data/e12/runner.log; each job writes its own log beside it. The verifier is one of
the jobs, and e12_compare refuses to read unless it passed. At the end:
data/e12/score_report.json, compare_report.json (its console output in
compare.log) and tables.md, which E12_RESULTS.md carries verbatim.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import e12_common as K  # noqa: E402

POOL = 7
MIN_FREE = 1.5 * 2 ** 30      # start a job only with this much memory available
GAP = 45                      # seconds between starts, so each job reaches its size first
SCORE = "research/native/e12_score.py"
QC1 = "research/qc1/qc1_measure.py"
QC1_CELLS = ("free_benign", "free_phishing", "control_benign_P0Q1", "control_benign_P1Q1")


def jobs(e12: Path) -> list:
    """(name, arguments), longest first (E11's step-20,000 readout's timings)."""
    d = ["--e12", str(e12)]
    out = [("verify", ["research/native/e12_verify.py", *d])]
    for part, label in (("control", "benign"), ("free", "benign"),
                        ("control", "phishing"), ("free", "phishing")):
        out += [(f"score_{part}_{a}_{label}", [SCORE, *d, "--part", part, "--arm", a, "--label", label])
                for a in K.ARMS]
    out += [(f"qc1_mix_{c}", [QC1, "--part", "mix", "--arm", K.CANDIDATE, "--cell", c,
                              "--gen", str(e12 / "gen"), "--out", str(e12 / "qc1")]) for c in QC1_CELLS]
    out += [("score_misc", [SCORE, *d, "--part", "misc"]), ("score_copy", [SCORE, *d, "--part", "copy"])]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e12", type=Path, default=ROOT / "data" / "e12")
    ap.add_argument("--pool", type=int, default=POOL)
    args = ap.parse_args()
    e12 = args.e12
    status = open(e12 / "runner.log", "a", encoding="utf-8")

    def note(msg):
        status.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        status.flush()

    queue, running, exits, last = jobs(e12), {}, {}, 0.0
    note(f"start: {len(queue)} jobs, pool {args.pool}")
    while queue or running:
        free = psutil.virtual_memory().available
        if queue and len(running) < args.pool and time.time() - last >= GAP and \
                (free >= MIN_FREE or not running):
            name, cmd = queue.pop(0)
            log = open(e12 / f"{name}.log", "w", encoding="utf-8")
            p = subprocess.Popen([sys.executable, *cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running[name] = (p, log, time.time())
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
        note(f"STOP: jobs failed, nothing merged: {failed}")
        raise SystemExit(f"jobs failed: {failed} (see their logs in {e12})")
    for name, cmd, out in (("merge", [SCORE, "--e12", str(e12), "--merge"], "merge.log"),
                           ("compare", ["research/native/e12_compare.py", "--e12", str(e12)], "compare.log"),
                           ("tables", ["research/native/e12_tables.py", "--e12", str(e12)], "tables.md")):
        with open(e12 / out, "w", encoding="utf-8") as fh, \
                open(e12 / f"{name}.err", "w", encoding="utf-8") as err:
            code = subprocess.run([sys.executable, *cmd], cwd=ROOT, stdout=fh, stderr=err).returncode
        note(f"{name} exit {code}")
        if code:
            raise SystemExit(f"{name} failed (exit {code}); see {e12 / out}")
    note("all done")
    print(f"done: {e12 / 'tables.md'}")


if __name__ == "__main__":
    main()
