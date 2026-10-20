"""
uie/corpus/feeds.py

Live phishing collection — the authority on what current phishing grammar looks
like.

Why live feeds at all
---------------------
Historical corpora teach campaigns that are already dead. Phishing grammar
evolves, so a generator trained only on 2019-2024 archives learns to imitate a
museum. Live feeds define the present; historical corpora supply the volume a
generative model needs in order to train at all (thousands of live URLs will be
memorized, not learned). As live collection accumulates, the historical share is
progressively downweighted.

Why this is a git walk rather than a polling daemon
---------------------------------------------------
OpenPhish's public feed redirects to a GitHub repository that is committed
roughly hourly. That repository IS an append-only store with authoritative
timestamps, ~22 months deep and only ~14 MB. So the "start collecting and wait
months" problem largely dissolves: cloning backfills the entire history at once,
and `git pull` continues it. Backfill and live collection are the same code path.

Each URL is recorded with `first_seen` — the earliest commit in which it appears.
That timestamp is what makes the temporal generalization benchmark possible
(train on URLs first seen before T, evaluate on those first seen after), and it
cannot be recovered from a flat dump of the feed.

Honest scope
------------
The community feed is a SAMPLE of OpenPhish's detections, not their full corpus,
and it is a rolling window: a URL that appears, ages out, and reappears is
recorded at its earliest appearance. Coverage is therefore a lower bound on real
phishing, and nothing here should be read as "all phishing in this period."
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

__all__ = ["PhishRecord", "GitFeed", "OPENPHISH", "collect"]


@dataclass(frozen=True)
class PhishRecord:
    """One phishing URL, timestamped by when the feed first published it."""

    url: str
    first_seen: str          # ISO-8601 UTC, from the earliest commit containing it
    source: str              # feed identifier, e.g. "openphish"


@dataclass(frozen=True)
class FeedSpec:
    """A feed published as a git repository."""

    name: str
    repo: str
    path: str                # file within the repo holding the URL list


OPENPHISH = FeedSpec(
    name="openphish",
    repo="https://github.com/openphish/public_feed.git",
    path="feed.txt",
)


def _git(args: List[str], cwd: Path, *, stdin: Optional[bytes] = None) -> bytes:
    proc = subprocess.run(["git", *args], cwd=cwd, input=stdin,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: "
                           f"{proc.stderr.decode('utf-8', 'replace')[:400]}")
    return proc.stdout


class GitFeed:
    """A phishing feed whose history lives in a git repository.

    `sync()` is idempotent: it clones when absent and fetches when present, so
    the same call backfills on first run and collects incrementally afterwards.
    """

    def __init__(self, spec: FeedSpec, root: Path) -> None:
        self.spec = spec
        self.dir = Path(root) / spec.name

    # --- acquisition -------------------------------------------------------

    def sync(self, pin: Optional[str] = None) -> int:
        """Clone or update. Returns the commit count now available.

        `pin` reproduces a frozen corpus: the clone is checked out (detached) at that
        commit and NOT pulled, so the history walk stops exactly where the published
        corpus stopped. Without it the feed is followed to its newest commit."""
        if not (self.dir / ".git").is_dir():
            self.dir.parent.mkdir(parents=True, exist_ok=True)
            _git(["clone", "--quiet", self.spec.repo, str(self.dir)],
                 cwd=self.dir.parent)
        elif pin is None:
            _git(["pull", "--quiet", "--ff-only"], cwd=self.dir)
        if pin is not None:
            try:
                _git(["rev-parse", "--verify", "--quiet", f"{pin}^{{commit}}"], cwd=self.dir)
            except RuntimeError:
                _git(["fetch", "--quiet", "origin"], cwd=self.dir)
            _git(["checkout", "--quiet", "--detach", pin], cwd=self.dir)
        return int(_git(["rev-list", "--count", "HEAD"], cwd=self.dir).decode().strip())

    # --- history walk ------------------------------------------------------

    def _commits(self) -> List[Tuple[str, str]]:
        """(sha, ISO-8601 UTC date) oldest-first.

        Oldest-first matters: `first_seen` is assigned on first encounter, so
        the walk order determines correctness, not merely convenience.
        """
        raw = _git(["log", "--reverse", "--format=%H %cI", "HEAD"], cwd=self.dir)
        out: List[Tuple[str, str]] = []
        for line in raw.decode("utf-8", "replace").splitlines():
            sha, _, iso = line.partition(" ")
            if sha and iso:
                out.append((sha, _to_utc(iso)))
        return out

    def iter_snapshots(self) -> Iterator[Tuple[str, List[str]]]:
        """Yield (ISO date, URLs) for every commit, oldest-first.

        Blobs are read through a single `git cat-file --batch` process rather
        than one `git show` per commit: at ~12k commits the process-spawn cost
        dominates everything else.
        """
        commits = self._commits()
        if not commits:
            return
        query = "".join(f"{sha}:{self.spec.path}\n" for sha, _ in commits)
        raw = _git(["cat-file", "--batch"], cwd=self.dir, stdin=query.encode())

        pos = 0
        for _, iso in commits:
            nl = raw.find(b"\n", pos)
            if nl < 0:
                return
            header = raw[pos:nl].decode("utf-8", "replace").split()
            # "<oid> missing" when the path did not exist at that commit.
            if len(header) < 3:
                pos = nl + 1
                continue
            size = int(header[2])
            body = raw[nl + 1: nl + 1 + size]
            pos = nl + 1 + size + 1          # trailing newline after the blob
            urls = [ln.strip() for ln in body.decode("utf-8", "replace").splitlines()
                    if ln.strip() and not ln.startswith("#")]
            yield iso, urls

    # --- collection --------------------------------------------------------

    def collect(self) -> List[PhishRecord]:
        """Every distinct URL, stamped with the earliest commit containing it."""
        first: Dict[str, str] = {}
        for iso, urls in self.iter_snapshots():
            for url in urls:
                if url not in first:      # oldest-first walk => first wins
                    first[url] = iso
        return [PhishRecord(u, t, self.spec.name) for u, t in first.items()]


def _to_utc(iso: str) -> str:
    """Normalize git's offset-bearing timestamps to UTC, so records from
    different committer timezones sort correctly."""
    try:
        return (datetime.fromisoformat(iso)
                .astimezone(timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return iso


def collect(root: Path, specs: Optional[List[FeedSpec]] = None,
            pin: Optional[str] = None) -> Tuple[List[PhishRecord], dict]:
    """Sync every feed and return its records plus a provenance manifest.

    The manifest pins each feed's HEAD and a content hash, because a live feed
    cannot be reproduced by a stranger tomorrow — the release must name the exact
    snapshot it was built from. `pin` (one feed only) rebuilds that snapshot.
    """
    specs = specs or [OPENPHISH]
    if pin is not None and len(specs) != 1:
        raise ValueError("pin names one commit, so it applies to exactly one feed")
    records: List[PhishRecord] = []
    feeds_meta = []

    for spec in specs:
        feed = GitFeed(spec, root)
        n_commits = feed.sync(pin=pin)
        got = feed.collect()
        records.extend(got)
        head = _git(["rev-parse", "HEAD"], cwd=feed.dir).decode().strip()
        feeds_meta.append({
            "name": spec.name,
            "repo": spec.repo,
            "head": head,
            "commits": n_commits,
            "urls": len(got),
            "first_seen_min": min((r.first_seen for r in got), default=None),
            "first_seen_max": max((r.first_seen for r in got), default=None),
        })

    digest = hashlib.sha256()
    for rec in sorted(records, key=lambda r: (r.first_seen, r.url)):
        digest.update(f"{rec.first_seen}\t{rec.url}\n".encode("utf-8"))

    manifest = {
        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_records": len(records),
        "content_sha256": digest.hexdigest(),
        "feeds": feeds_meta,
    }
    return records, manifest


def write_jsonl(records: List[PhishRecord], manifest: dict, out_dir: Path) -> None:
    """Persist records and manifest side by side, sorted by first_seen.

    Sorted output makes the file itself deterministic, so two runs over the same
    feed HEAD produce byte-identical artifacts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phishing.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for rec in sorted(records, key=lambda r: (r.first_seen, r.url)):
            fh.write(json.dumps({"url": rec.url, "first_seen": rec.first_seen,
                                 "source": rec.source}, ensure_ascii=False) + "\n")
    (out_dir / "phishing.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    """`python -m uie.corpus.feeds [--pin COMMIT]`: collect the OpenPhish history into
    data/corpus/phishing.jsonl and its manifest. The published corpus is rebuilt with
    `--pin 48f7a982b2a4ebf679c99457fd5467359323e4c6`; its content_sha256 must then equal
    the one in the tracked data/corpus/phishing.manifest.json (REPRODUCE.md)."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/feeds"), help="where the feed is cloned")
    ap.add_argument("--out", type=Path, default=Path("data/corpus"))
    ap.add_argument("--pin", default=None, help="a feed commit to stop at (reproduces a frozen corpus)")
    args = ap.parse_args(argv)
    records, manifest = collect(args.root, pin=args.pin)
    write_jsonl(records, manifest, args.out)
    meta = manifest["feeds"][0]
    print(f"{manifest['n_records']:,} records from {meta['commits']:,} commits at {meta['head'][:12]}; "
          f"content_sha256 {manifest['content_sha256']} -> {args.out / 'phishing.jsonl'}")


if __name__ == "__main__":
    main()
