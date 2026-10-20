"""E15-B -- acquire the new query-bearing rows (research/e15/PREREG_E15.md sections 9.4, 10, 11).

    python research/e15/e15_acquire.py targets       # N_old,s and L_old,s as encoded (laptop only)
    python research/e15/e15_acquire.py phishing      # the OpenPhish archive at E15's commit; select new rows
    python research/e15/e15_acquire.py benign        # rounds of 8 parallel crawls until the targets or the cap
    python research/e15/e15_acquire.py select --final  # the name rule, exclusions, filters, cap; writes the stream
    python research/e15/e15_acquire.py crawl --crawl C --round R --blocks N   # one crawl-round (a worker)

Everything follows the corpus's own code: `uie/corpus/extract.py` for blocks (suffix-stratified,
alpha 0.5, status 200 + text/html, up to 48 URLs per domain per file), `uie/corpus/merge.py`'s
name rule (one URL per registrable domain, reservoir k = 1), `uie/corpus/merge_phishing.py`'s
host rule (one URL per host, `rng.choice`), and the split files' own key functions.

Outputs, under data/e15/qry_extra/: shards/rNN/<crawl>.jsonl.gz (+ .done.json); the stream files
benign_P0Q1.jsonl, benign_P1Q1.jsonl, phishing_P0Q1.jsonl, phishing_P1Q1.jsonl (plain, never
gzipped); feed_index.json (every host and registrable domain in the fetched feed, for P7);
manifest.json; logs/.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uie.corpus.extract import LIVE_CRAWLS  # noqa: E402

OUT = ROOT / "data" / "e15" / "qry_extra"
SHARDS = OUT / "shards"
LOGS = OUT / "logs"
BUNDLE = ROOT / "data" / "bundle2"
SPLIT_BENIGN = ROOT / "data" / "corpus2" / "split_random.json.gz"
SPLIT_PHISH = ROOT / "data" / "corpus2" / "split_phishing_host.json.gz"
OLD_PHISH = ROOT / "data" / "corpus" / "phishing.jsonl"
FEEDS = ROOT / "data" / "feeds"
E12_FINAL = ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt"
TARGETS = ROOT / "data" / "e15" / "targets.json"

ROUND_BLOCKS = 500                 # section 9.4.1
CAP_BLOCKS = 4_251                 # corpus2's own budget per crawl
SEED_BLOCKS, SEED_NAME, SEED_PHISH = 20261005, 20261006, 20261009   # section 11.2.4
MAX_LEN = 384
STRATA = ("P0Q1", "P1Q1")
TRAIN_VAL = ("name_train", "name_val", "structure_train", "structure_val",
             "phishing_name_train", "phishing_name_val", "phishing_structure_train",
             "phishing_structure_val")
ATTEMPTS, RETRY_PAUSE_S = 3, 120
#: The OpenPhish commit E15 read (manifest.json "new_head", 2026-10-05). `phishing` stops here by
#: default, so a rebuild selects the same rows instead of whatever the feed holds today.
FEED_PIN = "d0642d73d6a3f7c76d2f074aba772858ba7b5c08"


def block_seed(i: int, r: int) -> int:
    """Section 11.2.4: crawl i (LIVE_CRAWLS order), round r (1, 2, ...)."""
    return SEED_BLOCKS + 7919 * i + 1_000_003 * (r - 1)


def log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGS / "acquire.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def rows_of(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def manifest_update(**parts) -> dict:
    path = OUT / "manifest.json"
    m = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "registration": "research/e15/PREREG_E15.md (sections 9.4, 10, 11)"}
    m.update(parts)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=1, default=str) + "\n", encoding="utf-8")
    return m


# --- the training tokenizer, and the query steps a row would supervise ---------------------

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        import torch
        from uie.tokenizer import URLTokenizer
        blob = torch.load(E12_FINAL, map_location="cpu", weights_only=False)
        _TOK = URLTokenizer.from_state_dict(blob["tokenizer"])
    return _TOK


def query_steps(tok, ids) -> int:
    """PREREG_E14 section 2.1's positions: from the first character after <QRY> through the
    target that ends the query (<FRG> or the end token)."""
    q_id, ends = tok.marker_id("QRY"), {tok.marker_id("FRG"), tok.eos_id}
    n = 0
    for q in (i for i, x in enumerate(ids) if x == q_id):
        e = next((i for i in range(q + 1, len(ids)) if ids[i] in ends), None)
        if e is not None:
            n += e - q
    return n


def encoded(url: str, label: str):
    return tokenizer().encode(url, label, "und", frame="domain")


def state_of(url: str):
    from uie.evaluation import FIELDS
    if not FIELDS["QRY"](url):
        return None
    return "P1Q1" if FIELDS["PTH"](url) else "P0Q1"


# --- targets: N_old,s and L_old,s, as training encodes them -----------------------------------

def cmd_targets(args) -> None:
    out = {"registration": "PREREG_E15 sections 9.4.1 (targets) and 10.4 (dose)",
           "max_len": MAX_LEN, "strata": {}}
    for label, stem in (("benign", "name_train"), ("phishing", "phishing_name_train")):
        n = collections.Counter()
        steps = collections.Counter()
        dropped = collections.Counter()
        for row in rows_of(BUNDLE / f"{stem}.jsonl.gz"):
            st = state_of(row["url"])
            if st is None:
                continue
            ids = encoded(row["url"], label)
            if len(ids) > MAX_LEN:
                dropped[st] += 1
                continue
            n[st] += 1
            steps[st] += query_steps(tokenizer(), ids)
        for st in STRATA:
            out["strata"][f"{label}_{st}"] = {"N_old": n[st], "dropped_over_max_len": dropped[st],
                                              "L_old": steps[st] / n[st] if n[st] else None}
        print(label, {st: (n[st], round(steps[st] / max(n[st], 1), 2), dropped[st]) for st in STRATA},
              flush=True)
    TARGETS.parent.mkdir(parents=True, exist_ok=True)
    TARGETS.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", TARGETS)


# --- phishing: the OpenPhish archive's later entries -----------------------------------------

def cmd_phishing(args) -> None:
    from uie.corpus.feeds import OPENPHISH, GitFeed
    from uie.corpus.merge_phishing import host_of
    from uie.splits import group_key, group_key_host, load_split
    feed = GitFeed(OPENPHISH, FEEDS)
    old_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=feed.dir, capture_output=True,
                              text=True).stdout.strip()
    log(f"phishing: fetching {OPENPHISH.repo} (local head {old_head[:12]}, pin {args.pin[:12]})")
    n_commits = feed.sync(pin=args.pin)
    new_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=feed.dir, capture_output=True,
                              text=True).stdout.strip()
    new_head_date = subprocess.run(["git", "log", "-1", "--format=%cI"], cwd=feed.dir,
                                   capture_output=True, text=True).stdout.strip()
    records = feed.collect()
    log(f"phishing: head {old_head[:12]} -> {new_head[:12]} ({new_head_date}); {n_commits:,} "
        f"commits; {len(records):,} distinct URLs in the archive")
    old_urls = {r["url"] for r in rows_of(OLD_PHISH)}
    # P7's exclusion list: every host and registrable domain anywhere in the fetched archive
    hosts, domains = set(), set()
    for r in records:
        hosts.add(group_key_host(r.url))
        domains.add(group_key(r.url))
    (OUT / "feed_index.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "feed_index.json").write_text(json.dumps({"head": new_head, "hosts": sorted(hosts),
                                                      "domains": sorted(domains)}) + "\n",
                                         encoding="utf-8")
    new = sorted((r for r in records if r.url not in old_urls), key=lambda r: (r.first_seen, r.url))
    funnel = collections.Counter(new_urls=len(new))
    split = load_split(SPLIT_PHISH)
    by_host = collections.OrderedDict()
    for r in new:
        h = host_of(r.url)
        if not h:
            funnel["no_host"] += 1
            continue
        by_host.setdefault(h, []).append(r)
    funnel["new_hosts_any"] = len(by_host)
    rng = random.Random(SEED_PHISH)
    picks = []
    for h, rs in by_host.items():
        if split.partition_of(rs[0].url) is not None:
            funnel["host_seen_before_cutoff"] += 1
            continue
        pick = rng.choice(rs)                     # merge_phishing.py's rule
        picks.append(pick)
    funnel["new_hosts_picked"] = len(picks)
    out_rows = {st: [] for st in STRATA}
    for p in picks:
        st = state_of(p.url)
        if st is None:
            funnel["no_query"] += 1
            continue
        if len(encoded(p.url, "phishing")) > MAX_LEN:
            funnel["over_max_len"] += 1
            continue
        out_rows[st].append({"url": p.url, "first_seen": p.first_seen, "source": p.source,
                             "label": "phishing", "corpus": "qry_extra", "stratum": st})
    # exact-URL disjointness from every train and val file (section 11.2.2)
    cand = {r["url"] for rs in out_rows.values() for r in rs}
    hits = set()
    for stem in TRAIN_VAL:
        for row in rows_of(BUNDLE / f"{stem}.jsonl.gz"):
            if row["url"] in cand:
                hits.add(row["url"])
    funnel["exact_url_in_train_or_val"] = len(hits)
    files = {}
    for st in STRATA:
        rows = [r for r in out_rows[st] if r["url"] not in hits]
        path = OUT / f"phishing_{st}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[st] = {"rows": len(rows), "sha256": sha256(path), "file": path.name}
        funnel[f"kept_{st}"] = len(rows)
    manifest_update(phishing={"feed": OPENPHISH.repo, "old_head": old_head, "new_head": new_head,
                              "new_head_date": new_head_date, "commits": n_commits,
                              "archive_urls": len(records), "seed_host_pick": SEED_PHISH,
                              "funnel": dict(funnel), "files": files, "code_commit": git_commit()})
    log(f"phishing: funnel {dict(funnel)}")


# --- benign: one crawl-round (a worker), and the parallel rounds -------------------------------

def cmd_crawl(args) -> None:
    from uie.corpus.extract import extract_crawl
    i = LIVE_CRAWLS.index(args.crawl)
    out = SHARDS / f"r{args.round:02d}"
    res = extract_crawl(args.crawl, args.blocks, out, seed=block_seed(i, args.round),
                        cache=ROOT / "data" / "cache")
    print(json.dumps(res.__dict__), flush=True)


def run_round(r: int, n_blocks: int, max_parallel: int = 8, attempts: int = ATTEMPTS,
              pause: int = RETRY_PAUSE_S) -> dict:
    """One round: every crawl's `n_blocks` at its own seed, at most `max_parallel` crawls
    at a time. Concurrency is operational only -- each crawl-round's draws are fixed by
    its seed -- and is recorded in the log (Common Crawl throttles many connections)."""
    out = SHARDS / f"r{r:02d}"
    out.mkdir(parents=True, exist_ok=True)
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


def cmd_benign(args) -> None:
    targets = json.loads(TARGETS.read_text(encoding="utf-8"))["strata"]
    want = {st: targets[f"benign_{st}"]["N_old"] for st in STRATA}
    if not (OUT / "feed_index.json").is_file():
        raise SystemExit("run `phishing` first: P7 needs the fetched feed's hosts and domains")
    log(f"benign: targets {want}; rounds of {ROUND_BLOCKS} blocks per crawl, cap {CAP_BLOCKS}")
    rounds = {}
    r = 0
    while True:
        r += 1
        n = min(ROUND_BLOCKS, CAP_BLOCKS - ROUND_BLOCKS * (r - 1))
        if n <= 0:
            log("benign: the cap is reached")
            break
        t0 = time.time()
        rounds[r] = run_round(r, n, args.max_parallel, args.attempts, args.retry_pause)
        got = select(final=False)
        log(f"round {r} done in {(time.time() - t0) / 60:.1f} min: counts {got['counts']} "
            f"(targets {want})")
        manifest_update(benign_rounds={k: v for k, v in rounds.items()})
        if all(got["counts"][st] >= want[st] for st in STRATA):
            log(f"benign: both strata reached their targets after round {r}")
            break
    select(final=True)


# --- selection: merge.py's name rule, then the exclusions and filters --------------------------

def select(final: bool) -> dict:
    from uie.splits import group_key, group_key_host, load_split
    from uie.factorization import parse
    files = sorted(SHARDS.glob("r*/CC-MAIN-*.jsonl.gz"))
    done = [f for f in files if (f.parent / f.name.replace(".jsonl.gz", ".done.json")).is_file()]
    rng = random.Random(SEED_NAME)
    seen, pick = {}, {}
    n_rows = 0
    for path in done:
        rnd = path.parent.name
        for row in rows_of(path):
            p = parse(row["url"])
            if not p.registrable_label:
                continue
            domain = f"{p.registrable_label}.{p.public_suffix}"
            n_rows += 1
            c = seen.get(domain, 0) + 1
            seen[domain] = c
            if c == 1 or rng.randrange(c) == 0:     # merge.py's reservoir, k = 1
                pick[domain] = {**row, "round": rnd}
    split = load_split(SPLIT_BENIGN)
    feed = json.loads((OUT / "feed_index.json").read_text(encoding="utf-8"))
    feed_hosts, feed_domains = set(feed["hosts"]), set(feed["domains"])
    funnel = collections.Counter(rows_read=n_rows, domains=len(pick))
    kept = {st: [] for st in STRATA}
    for domain, row in pick.items():
        u = row["url"]
        if split.partition_of(u) is not None:
            funnel["domain_in_existing_split"] += 1
            continue
        if group_key(u) in feed_domains or group_key_host(u) in feed_hosts:
            funnel["in_phishing_feed"] += 1
            continue
        st = state_of(u)
        if st is None:
            funnel["no_query"] += 1
            continue
        if len(encoded(u, "benign")) > MAX_LEN:
            funnel["over_max_len"] += 1
            continue
        kept[st].append({"url": u, "lang": row.get("lang", "und"), "ts": row.get("ts", ""),
                         "crawl": row.get("crawl", ""), "round": row["round"], "label": "benign",
                         "corpus": "qry_extra", "stratum": st, "domain": group_key(u)})
    counts = {st: len(kept[st]) for st in STRATA}
    funnel.update({f"query_rows_{st}": counts[st] for st in STRATA})
    if not final:
        return {"counts": counts, "funnel": dict(funnel)}
    targets = json.loads(TARGETS.read_text(encoding="utf-8"))["strata"]
    files_out = {}
    for st in STRATA:
        want = targets[f"benign_{st}"]["N_old"]
        rows = sorted(kept[st], key=lambda x: hashlib.sha256(f"E15-B:{x['domain']}".encode()).hexdigest())
        rows = rows[:want]
        path = OUT / f"benign_{st}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for x in rows:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        files_out[st] = {"rows": len(rows), "available": counts[st], "target": want,
                         "sha256": sha256(path), "file": path.name}
    manifest_update(benign={"crawls": LIVE_CRAWLS, "round_blocks": ROUND_BLOCKS, "cap": CAP_BLOCKS,
                            "seed_blocks": "20261005 + 7919*i + 1000003*(r-1)",
                            "seed_name": SEED_NAME, "funnel": dict(funnel), "files": files_out,
                            "code_commit": git_commit()})
    log(f"benign: final selection {files_out}")
    return {"counts": counts, "funnel": dict(funnel), "files": files_out}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("targets")
    p = sub.add_parser("phishing")
    p.add_argument("--pin", default=FEED_PIN, help="the feed commit to stop at (default: E15's)")
    p.add_argument("--out", default=None,
                   help="write the outputs here instead of data/e15/qry_extra (to check a rebuild "
                        "against the record without touching it)")
    b = sub.add_parser("benign")
    b.add_argument("--max-parallel", type=int, default=8)
    b.add_argument("--attempts", type=int, default=ATTEMPTS)
    b.add_argument("--retry-pause", type=int, default=RETRY_PAUSE_S)
    s = sub.add_parser("select")
    s.add_argument("--final", action="store_true")
    c = sub.add_parser("crawl")
    c.add_argument("--crawl", required=True, choices=LIVE_CRAWLS)
    c.add_argument("--round", type=int, required=True)
    c.add_argument("--blocks", type=int, required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if getattr(args, "out", None):
        global OUT, SHARDS, LOGS
        OUT = Path(args.out)
        SHARDS, LOGS = OUT / "shards", OUT / "logs"
    if args.cmd == "select":
        print(json.dumps(select(final=args.final), indent=1, default=str))
        return
    {"targets": cmd_targets, "phishing": cmd_phishing, "benign": cmd_benign,
     "crawl": cmd_crawl}[args.cmd](args)


if __name__ == "__main__":
    main()
