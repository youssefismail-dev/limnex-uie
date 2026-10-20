"""Option A's data: a second stream of new benign query rows, by E15's procedure
(research/e17/ACQUISITION_E17A.md). Acquisition only: nothing is trained with these rows
until an E17 registration is approved.

    python research/e17/e17_acquire.py benign --max-parallel 4 --attempts 4 --retry-pause 180
    python research/e17/e17_acquire.py select --final   # rerun the final selection alone
    python research/e17/e17_acquire.py crawl --crawl C --round R --blocks N   # one crawl-round (a worker)

E15's own code throughout (research/e15/e15_acquire.py: the block seeds, the encoder and
the stratum test; uie/corpus/extract.py for blocks), with this stream's rounds (9 onward),
budget, reservoir seed, order key and folder, and one exclusion more: every registrable
domain already in stream 1.

Outputs, under data/e17/qry_extra2/: shards/rNN/<crawl>.jsonl.gz (+ .done.json),
benign_P0Q1.jsonl and benign_P1Q1.jsonl (plain), manifest.json, logs/.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "e15"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e15_acquire as A  # noqa: E402
from uie.corpus.extract import LIVE_CRAWLS  # noqa: E402

OUT = ROOT / "data" / "e17" / "qry_extra2"
SHARDS = OUT / "shards"
LOGS = OUT / "logs"
STREAM1 = ROOT / "data" / "e15" / "qry_extra" / "qry_extra.jsonl"
STREAM1_SHA256 = "637accab5622f14aa432849fe9f4bc6ede7b2d18188bc3ef141d9d3c71e69e91"
FEED_INDEX = ROOT / "data" / "e15" / "qry_extra" / "feed_index.json"   # frozen at d0642d7

FIRST_ROUND = 9                    # rounds 1-8 are stream 1's
ROUND_BLOCKS = 500
CAP_BLOCKS = 4_251                 # this stream's own budget per crawl (corpus2's)
SEED_NAME = 20261013               # the reservoir
ORDER_KEY = "E17-A:"
TARGETS = {"P0Q1": 14_863, "P1Q1": 42_429}     # N_old: one more doubling


def log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGS / "acquire.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def manifest_update(**parts) -> dict:
    path = OUT / "manifest.json"
    m = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "registration": "research/e17/ACQUISITION_E17A.md"}
    m.update(parts)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=1, default=str) + "\n", encoding="utf-8")
    return m


#: Retries for one HTTP range read (ACQUISITION_E17A.md section 4): a truncated or dropped
#: response is fetched again after a growing pause. The library's own `_get` retries three
#: times at once and does not catch a truncated body (http.client.IncompleteRead), so a
#: single cut-off response ended a whole worker.
FETCH_ATTEMPTS, FETCH_PAUSE_S, FETCH_PAUSE_MAX_S = 8, 10, 300


def patient_get():
    """`uie.corpus.commoncrawl._get` with the truncated-read case retried and a pause that
    doubles (10 s, 20 s, ... at most 300 s). On success the bytes are the complete
    response, so every draw is exactly what the library's own call would have returned."""
    import http.client
    import urllib.error
    import urllib.request
    from uie.corpus import commoncrawl as cc

    def _get(url, headers=None, method="GET"):
        last = None
        for k in range(FETCH_ATTEMPTS):
            try:
                req = urllib.request.Request(url, headers=headers or {}, method=method)
                with urllib.request.urlopen(req, timeout=cc._TIMEOUT_S) as resp:
                    return resp.read() if method == "GET" else bytes(
                        resp.headers.get("Content-Length", "0"), "ascii")
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                last = exc
                if k + 1 < FETCH_ATTEMPTS:
                    time.sleep(min(FETCH_PAUSE_MAX_S, FETCH_PAUSE_S * 2 ** k))
        raise RuntimeError(f"GET {url} failed after {FETCH_ATTEMPTS} attempts: {last}")
    return _get


def cmd_crawl(args) -> None:
    from uie.corpus import commoncrawl, stratified
    from uie.corpus.extract import extract_crawl
    commoncrawl._get = stratified._get = patient_get()     # this worker only; uie/ is unchanged
    i = LIVE_CRAWLS.index(args.crawl)
    res = extract_crawl(args.crawl, args.blocks, SHARDS / f"r{args.round:02d}",
                        seed=A.block_seed(i, args.round), cache=ROOT / "data" / "cache")
    print(json.dumps(res.__dict__), flush=True)


def run_round(r: int, n_blocks: int, max_parallel: int, attempts: int, pause: int) -> dict:
    """E15's round runner, with this stream's folder and worker."""
    out = SHARDS / f"r{r:02d}"
    out.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    pending = [c for c in LIVE_CRAWLS if not (out / f"{c}.done.json").is_file()]
    for attempt in range(1, attempts + 1):
        if not pending:
            break
        log(f"round {r}: {n_blocks} blocks x {len(pending)} crawls, at most {max_parallel} "
            f"in parallel (attempt {attempt})")
        for i in range(0, len(pending), max_parallel):
            procs = {}
            for c in pending[i:i + max_parallel]:
                lf = open(LOGS / f"r{r:02d}_{c}_a{attempt}.log", "w", encoding="utf-8")
                procs[c] = (subprocess.Popen([sys.executable, str(Path(__file__)), "crawl", "--crawl", c,
                                              "--round", str(r), "--blocks", str(n_blocks)],
                                             cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT), lf)
            for c, (p, lf) in procs.items():
                p.wait()
                lf.close()
        pending = [c for c in LIVE_CRAWLS if not (out / f"{c}.done.json").is_file()]
        if pending:
            log(f"round {r}: not complete: {pending}; pausing {pause}s before retrying")
            time.sleep(pause)
    if pending:
        raise SystemExit(f"STOP: round {r}: crawls {pending} failed {attempts} attempts; "
                         f"nothing past this round is drawn")
    return {c: json.loads((out / f"{c}.done.json").read_text(encoding="utf-8")) for c in LIVE_CRAWLS}


def stream1_domains() -> set:
    from uie.splits import group_key
    if A.sha256(STREAM1) != STREAM1_SHA256:
        raise SystemExit("STOP: stream 1 is not E15's qry_extra.jsonl")
    return {group_key(r["url"]) for r in A.rows_of(STREAM1)}


def select(final: bool) -> dict:
    """E15's selection (merge.py's reservoir, then the exclusions and filters), over this
    stream's shards, with stream 1's domains excluded as well."""
    from uie.factorization import parse
    from uie.splits import group_key, group_key_host, load_split
    files = sorted(SHARDS.glob("r*/CC-MAIN-*.jsonl.gz"))
    done = [f for f in files if (f.parent / f.name.replace(".jsonl.gz", ".done.json")).is_file()]
    rng = random.Random(SEED_NAME)
    seen, pick = {}, {}
    n_rows = 0
    for path in done:
        rnd = path.parent.name
        for row in A.rows_of(path):
            p = parse(row["url"])
            if not p.registrable_label:
                continue
            domain = f"{p.registrable_label}.{p.public_suffix}"
            n_rows += 1
            c = seen.get(domain, 0) + 1
            seen[domain] = c
            if c == 1 or rng.randrange(c) == 0:     # merge.py's reservoir, k = 1
                pick[domain] = {**row, "round": rnd}
    split = load_split(A.SPLIT_BENIGN)
    feed = json.loads(FEED_INDEX.read_text(encoding="utf-8"))
    feed_hosts, feed_domains = set(feed["hosts"]), set(feed["domains"])
    first = stream1_domains()
    funnel = collections.Counter(rows_read=n_rows, domains=len(pick))
    kept = {st: [] for st in TARGETS}
    for domain, row in pick.items():
        u = row["url"]
        if split.partition_of(u) is not None:
            funnel["domain_in_existing_split"] += 1
            continue
        if group_key(u) in first:
            funnel["domain_in_stream_1"] += 1
            continue
        if group_key(u) in feed_domains or group_key_host(u) in feed_hosts:
            funnel["in_phishing_feed"] += 1
            continue
        st = A.state_of(u)
        if st is None:
            funnel["no_query"] += 1
            continue
        if len(A.encoded(u, "benign")) > A.MAX_LEN:
            funnel["over_max_len"] += 1
            continue
        kept[st].append({"url": u, "lang": row.get("lang", "und"), "ts": row.get("ts", ""),
                         "crawl": row.get("crawl", ""), "round": row["round"], "label": "benign",
                         "corpus": "qry_extra2", "stratum": st, "domain": group_key(u)})
    counts = {st: len(kept[st]) for st in TARGETS}
    funnel.update({f"query_rows_{st}": counts[st] for st in TARGETS})
    if not final:
        return {"counts": counts, "funnel": dict(funnel)}
    files_out = {}
    for st, want in TARGETS.items():
        rows = sorted(kept[st], key=lambda x: hashlib.sha256(f"{ORDER_KEY}{x['domain']}".encode()).hexdigest())
        rows = rows[:want]
        path = OUT / f"benign_{st}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for x in rows:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        files_out[st] = {"rows": len(rows), "available": counts[st], "target": want,
                         "short": len(rows) < want, "sha256": A.sha256(path), "file": path.name}
    manifest_update(benign={"crawls": LIVE_CRAWLS, "first_round": FIRST_ROUND,
                            "round_blocks": ROUND_BLOCKS, "cap": CAP_BLOCKS,
                            "seed_blocks": "20261005 + 7919*i + 1000003*(r-1)",
                            "seed_name": SEED_NAME, "order_key": ORDER_KEY,
                            "stream1_sha256": STREAM1_SHA256, "funnel": dict(funnel),
                            "files": files_out, "code_commit": A.git_commit()})
    log(f"benign: final selection {files_out}")
    return {"counts": counts, "funnel": dict(funnel), "files": files_out}


def cmd_benign(args) -> None:
    log(f"stream 2: targets {TARGETS}; rounds {FIRST_ROUND}+ of {ROUND_BLOCKS} blocks per crawl, "
        f"budget {CAP_BLOCKS}")
    stream1_domains()                                   # fail early if stream 1 is not E15's
    rounds = {}
    r = FIRST_ROUND - 1
    while True:
        r += 1
        n = min(ROUND_BLOCKS, CAP_BLOCKS - ROUND_BLOCKS * (r - FIRST_ROUND))
        if n <= 0:
            log("stream 2: the budget is spent")
            break
        t0 = time.time()
        rounds[r] = run_round(r, n, args.max_parallel, args.attempts, args.retry_pause)
        got = select(final=False)
        log(f"round {r} done in {(time.time() - t0) / 60:.1f} min: counts {got['counts']} "
            f"(targets {TARGETS}); funnel {got['funnel']}")
        manifest_update(benign_rounds=rounds)
        if all(got["counts"][st] >= TARGETS[st] for st in TARGETS):
            log(f"stream 2: both strata reached their targets after round {r}")
            break
    select(final=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benign")
    b.add_argument("--max-parallel", type=int, default=4)
    b.add_argument("--attempts", type=int, default=4)
    b.add_argument("--retry-pause", type=int, default=180)
    s = sub.add_parser("select")
    s.add_argument("--final", action="store_true")
    c = sub.add_parser("crawl")
    c.add_argument("--crawl", required=True, choices=LIVE_CRAWLS)
    c.add_argument("--round", type=int, required=True)
    c.add_argument("--blocks", type=int, required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.cmd == "benign":
        cmd_benign(args)
    elif args.cmd == "select":
        print(json.dumps(select(final=args.final), indent=1, default=str))
    else:
        cmd_crawl(args)


if __name__ == "__main__":
    main()
