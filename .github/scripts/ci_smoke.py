"""
.github/scripts/ci_smoke.py

The published-model smoke test (see .github/workflows/smoke.yml). Run it inside the directory that
holds the two release files, the way the README's Quick start does:

    python -I ci_smoke.py

This is a public repository, so CI logs are public. The URLs this model generates are
phishing-STYLE strings for defensive research: NO generated URL is ever printed, written to the job
summary or saved to a file here. Every check is on counts, lengths, validity, membership and
sha256 digests, and the only things printed are counts, timings and digests.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

WEIGHTS = "limnex-uie-e16_b5.pt"
KNOWN = "limnex-uie-known-urls.bloom.gz"
LABELS = ("benign", "phishing")
N = 25            # URLs to draw per mode before the gate (the README's example draws 10)
SEED = 7          # the README's Python example
MIN_KEPT = 0.6    # the gate removes ~1% of draws in practice; far fewer than this means a broken model

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def digest(urls: list[str]) -> str:
    return hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest()


def main() -> int:
    import uie
    from uie.bloom import BloomFilter
    from uie.gate import Gate, normalize_for_dedup

    # The release files are not part of the repository; the README sends a stranger to download them.
    for name in (WEIGHTS, KNOWN):
        check(Path(name).is_file(), f"{name} present ({Path(name).stat().st_size:,} bytes)"
              if Path(name).is_file() else f"{name} present")
    if failures:
        return 1

    # README Quick start, step 3, Python variant, verbatim.
    t0 = time.perf_counter()
    engine = uie.load("limnex-uie-e16_b5.pt", known_urls="limnex-uie-known-urls.bloom.gz")
    print(f"load: {time.perf_counter() - t0:.1f}s (uie.load verified both files' sha256 itself)")
    check(engine.novelty_checked, "engine has the known-URL filter")

    # A second, independent handle on the filter, to check the engine's output against.
    known = BloomFilter.load(Path(KNOWN))
    check(len(known) > 14_000_000, f"filter holds {len(known):,} URLs")
    plain_gate = Gate()          # validity only: nothing known, no blocklist

    hashes: dict[str, str] = {}
    for label in LABELS:
        print(f"\n[{label}] n={N} seed={SEED}")
        t0 = time.perf_counter()
        urls = engine.generate(label, n=N, seed=SEED)          # README: always gated
        dt = time.perf_counter() - t0
        report = engine.last_report
        check(report.n_total == N and report.n_passed == len(urls),
              f"gate saw {report.n_total} drawn, kept {len(urls)}")
        check(MIN_KEPT * N <= len(urls) <= N, f"count {len(urls)} of {N} within [{math.ceil(MIN_KEPT * N)}, {N}]")
        check(len(set(urls)) == len(urls), "no duplicates")
        check(all(isinstance(u, str) and u for u in urls), "every URL is a non-empty string")
        check(all(u == u.strip() and not any(ord(c) < 0x20 or ord(c) == 0x7F or c.isspace() for c in u)
                  for u in urls), "no whitespace or control characters")
        check(all(len(u) <= 8192 for u in urls), f"max length {max(map(len, urls))} <= 8192")
        valid = sum(plain_gate.check(u).passed is True for u in urls)
        plain_gate.reset()
        check(valid == len(urls), f"{valid}/{len(urls)} pass the gate's validity checks")
        hosts = [urlsplit(u if "://" in u else "//" + u).hostname for u in urls]
        check(all(h and ("." in h or ":" in h) for h in hosts), "every URL has a host")
        inside = sum(normalize_for_dedup(u) in known for u in urls)
        check(inside == 0, f"{inside} of {len(urls)} in the known-URL filter")
        hashes[label] = digest(urls)
        print(f"  generated in {dt:.1f}s; sha256 {hashes[label]}")

        again = engine.generate(label, n=N, seed=SEED)
        check(digest(again) == hashes[label], f"same seed, same output (sha256 {digest(again)[:16]}...)")
        other = engine.generate(label, n=N, seed=SEED + 1)
        check(digest(other) != hashes[label], "a different seed gives a different batch")

    check(hashes["benign"] != hashes["phishing"], "the two modes give different batches")

    # README Quick start, step 3, command-line variant, through the installed `uie` command: the
    # same draw through `uie generate`. stdout carries the URLs and is only hashed; stderr (notice
    # and gate report) carries no URLs.
    print("\n[cli] uie generate --label phishing")
    uie_cmd = shutil.which("uie")
    check(uie_cmd is not None, "the `uie` command is on PATH")
    if uie_cmd is None:
        return 1
    cmd = [uie_cmd, "generate", "--weights", WEIGHTS, "--known-urls", KNOWN,
           "--label", "phishing", "-n", str(N), "--seed", str(SEED)]
    run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    check(run.returncode == 0, f"exit code {run.returncode}")
    if run.returncode != 0:
        print(run.stderr)
    cli_urls = run.stdout.splitlines()
    check(digest(cli_urls) == hashes["phishing"], f"CLI output == API output ({len(cli_urls)} URLs, sha256 {digest(cli_urls)[:16]}...)")

    # README: the filter is required for every generation.
    print("\n[filter is required]")
    bare = uie.load(WEIGHTS)
    try:
        bare.generate("phishing", n=1, seed=SEED)
        check(False, "generate without the filter is refused")
    except ValueError:
        check(True, "generate without the filter is refused")

    # README: engine.score(["example.com/login"]) -> log P(url | class) and their difference.
    print("\n[score] example.com/login")
    rows = engine.score(["example.com/login"])
    row = rows[0]
    numbers = {k: v for k, v in row.items() if isinstance(v, float)}
    check(len(rows) == 1 and numbers and all(math.isfinite(v) for v in numbers.values()),
          f"one row, {len(numbers)} finite scores")
    print("  fields:", ", ".join(sorted(row)))
    check(all(k in row for k in ("llr",)), "row has llr")
    llr = row["llr"]
    parts = [v for k, v in numbers.items() if k.startswith("logp") or k.startswith("log_p")]
    if len(parts) == 2:
        check(abs((max(parts) - min(parts)) - abs(llr)) < 1e-6, "llr is the difference of the two log-probabilities")
    print(f"  llr = {llr:.2f}")

    # README Quick start, step 3: `uie score --weights ... "example.com/login"` (JSON lines).
    run = subprocess.run([uie_cmd, "score", "--weights", WEIGHTS, "example.com/login"],
                         capture_output=True, text=True, encoding="utf-8")
    check(run.returncode == 0, f"uie score exit code {run.returncode}")
    if run.returncode == 0:
        cli_rows = [json.loads(line) for line in run.stdout.splitlines()]
        check(len(cli_rows) == 1 and abs(cli_rows[0]["llr"] - llr) < 1e-9, "uie score gives the API's llr")
    else:
        print(run.stderr)

    print()
    if failures:
        print(f"SMOKE FAILED: {len(failures)} check(s): " + json.dumps(failures))
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
