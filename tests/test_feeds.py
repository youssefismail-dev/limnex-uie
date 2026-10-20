"""
Offline tests for the live phishing collector.

Each test builds a real throwaway git repository, because the collector's whole
correctness argument rests on git semantics: `first_seen` must be the EARLIEST
commit containing a URL, and the feed is a rolling window in which URLs
disappear and reappear. Mocking git would test the mock.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from uie.corpus.feeds import FeedSpec, GitFeed, PhishRecord, collect, write_jsonl


def _run(args, cwd):
    subprocess.run(args, cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.fixture
def feed_repo(tmp_path: Path) -> Path:
    """A repo whose feed.txt changes over four commits with known dates.

    Commit 3 drops 'b' entirely and commit 4 brings it back — the rolling-window
    behaviour that makes 'earliest wins' a real requirement rather than a
    formality.
    """
    repo = tmp_path / "origin"
    repo.mkdir()
    _run(["git", "init", "-q"], repo)
    _run(["git", "config", "user.email", "t@example.com"], repo)
    _run(["git", "config", "user.name", "T"], repo)

    snapshots = [
        ("2025-01-01T00:00:00+00:00", ["https://a.test/1", "https://b.test/2"]),
        ("2025-02-01T00:00:00+00:00", ["https://b.test/2", "https://c.test/3"]),
        ("2025-03-01T00:00:00+00:00", ["https://c.test/3"]),
        ("2025-04-01T00:00:00+00:00", ["https://b.test/2", "https://d.test/4"]),
    ]
    for date, urls in snapshots:
        (repo / "feed.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
        _run(["git", "add", "feed.txt"], repo)
        subprocess.run(
            ["git", "commit", "-q", "-m", date, "--date", date],
            cwd=repo, check=True, env={**_env(), "GIT_COMMITTER_DATE": date},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return repo


def _env():
    import os
    return dict(os.environ)


def _spec(repo: Path) -> FeedSpec:
    return FeedSpec(name="testfeed", repo=str(repo), path="feed.txt")


# --- history walk ----------------------------------------------------------

def test_sync_clones_then_pulls(tmp_path, feed_repo):
    feed = GitFeed(_spec(feed_repo), tmp_path / "work")
    assert feed.sync() == 4
    assert feed.sync() == 4          # idempotent: second call pulls, not clones


def test_snapshots_are_oldest_first(tmp_path, feed_repo):
    """Order is correctness, not convenience — `first_seen` is assigned on
    first encounter, so a newest-first walk would record the wrong date."""
    feed = GitFeed(_spec(feed_repo), tmp_path / "work")
    feed.sync()
    dates = [iso for iso, _ in feed.iter_snapshots()]
    assert dates == sorted(dates)
    assert dates[0].startswith("2025-01")


def test_all_snapshots_are_read(tmp_path, feed_repo):
    feed = GitFeed(_spec(feed_repo), tmp_path / "work")
    feed.sync()
    assert len(list(feed.iter_snapshots())) == 4


# --- first_seen semantics --------------------------------------------------

def test_first_seen_is_earliest_commit(tmp_path, feed_repo):
    feed = GitFeed(_spec(feed_repo), tmp_path / "work")
    feed.sync()
    seen = {r.url: r.first_seen for r in feed.collect()}
    assert seen["https://a.test/1"].startswith("2025-01")
    assert seen["https://c.test/3"].startswith("2025-02")
    assert seen["https://d.test/4"].startswith("2025-04")


def test_reappearing_url_keeps_earliest_date(tmp_path, feed_repo):
    """'b' is present, drops out, then returns. The feed is a rolling window,
    so its later reappearance must not overwrite the original sighting."""
    feed = GitFeed(_spec(feed_repo), tmp_path / "work")
    feed.sync()
    seen = {r.url: r.first_seen for r in feed.collect()}
    assert seen["https://b.test/2"].startswith("2025-01")


def test_urls_are_deduplicated(tmp_path, feed_repo):
    feed = GitFeed(_spec(feed_repo), tmp_path / "work")
    feed.sync()
    urls = [r.url for r in feed.collect()]
    assert len(urls) == len(set(urls)) == 4


# --- parsing robustness ----------------------------------------------------

def test_blank_lines_and_comments_are_skipped(tmp_path, feed_repo):
    (feed_repo / "feed.txt").write_text(
        "# a comment\n\nhttps://e.test/5\n   \n", encoding="utf-8")
    _run(["git", "add", "feed.txt"], feed_repo)
    _run(["git", "commit", "-q", "-m", "noise"], feed_repo)

    feed = GitFeed(_spec(feed_repo), tmp_path / "work")
    feed.sync()
    _, urls = list(feed.iter_snapshots())[-1]
    assert urls == ["https://e.test/5"]


def test_commit_without_the_feed_file_is_skipped(tmp_path, feed_repo):
    """`git cat-file` reports 'missing' for such commits; the batch reader must
    resynchronise instead of misaligning every subsequent blob."""
    (feed_repo / "feed.txt").unlink()
    _run(["git", "rm", "-q", "feed.txt"], feed_repo)
    _run(["git", "commit", "-q", "-m", "drop feed"], feed_repo)
    (feed_repo / "feed.txt").write_text("https://f.test/6\n", encoding="utf-8")
    _run(["git", "add", "feed.txt"], feed_repo)
    _run(["git", "commit", "-q", "-m", "restore"], feed_repo)

    feed = GitFeed(_spec(feed_repo), tmp_path / "work")
    feed.sync()
    snaps = list(feed.iter_snapshots())
    assert snaps[-1][1] == ["https://f.test/6"]
    assert {r.url for r in feed.collect()} >= {"https://f.test/6", "https://a.test/1"}


# --- provenance and determinism --------------------------------------------

def test_manifest_pins_head_and_span(tmp_path, feed_repo):
    """A live feed cannot be reproduced by a stranger tomorrow, so the release
    must name the exact snapshot it was built from."""
    _, manifest = collect(tmp_path / "work", [_spec(feed_repo)])
    meta = manifest["feeds"][0]
    assert len(meta["head"]) == 40
    assert meta["commits"] == 4 and meta["urls"] == 4
    assert meta["first_seen_min"].startswith("2025-01")
    assert meta["first_seen_max"].startswith("2025-04")


def test_content_hash_is_stable_across_runs(tmp_path, feed_repo):
    a, man_a = collect(tmp_path / "w1", [_spec(feed_repo)])
    b, man_b = collect(tmp_path / "w2", [_spec(feed_repo)])
    assert man_a["content_sha256"] == man_b["content_sha256"]
    assert {r.url for r in a} == {r.url for r in b}


def test_timestamps_are_normalized_to_utc(tmp_path, feed_repo):
    recs, _ = collect(tmp_path / "work", [_spec(feed_repo)])
    assert all(r.first_seen.endswith("Z") for r in recs)


def test_written_jsonl_is_sorted_and_deterministic(tmp_path, feed_repo):
    recs, manifest = collect(tmp_path / "work", [_spec(feed_repo)])
    out = tmp_path / "corpus"
    write_jsonl(recs, manifest, out)
    lines = (out / "phishing.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(ln) for ln in lines]
    assert [p["first_seen"] for p in parsed] == sorted(p["first_seen"] for p in parsed)
    assert json.loads((out / "phishing.manifest.json").read_text())["n_records"] == 4


def test_record_carries_source_attribution(tmp_path, feed_repo):
    recs, _ = collect(tmp_path / "work", [_spec(feed_repo)])
    assert {r.source for r in recs} == {"testfeed"}
    assert isinstance(recs[0], PhishRecord)


def test_pin_stops_at_a_commit_and_never_pulls(tmp_path, feed_repo):
    """D2 (REPRODUCE.md): a frozen corpus is rebuilt by pinning the feed's commit; the walk must
    stop there even after the feed moves on."""
    shas = subprocess.run(["git", "rev-list", "--reverse", "HEAD"], cwd=feed_repo, check=True,
                          capture_output=True, text=True).stdout.split()
    second = shas[1]
    feed = GitFeed(_spec(feed_repo), tmp_path / "work")
    assert feed.sync(pin=second) == 2
    (feed_repo / "feed.txt").write_text("https://e.test/5\n", encoding="utf-8")
    _run(["git", "add", "feed.txt"], feed_repo)
    _run(["git", "commit", "-q", "-m", "later"], feed_repo)
    assert feed.sync(pin=second) == 2                      # the feed moved on; the pin did not
    assert {r.url for r in feed.collect()} == {"https://a.test/1", "https://b.test/2", "https://c.test/3"}
    records, manifest = collect(tmp_path / "work", specs=[_spec(feed_repo)], pin=second)
    assert manifest["feeds"][0]["head"] == second and manifest["feeds"][0]["commits"] == 2
    with pytest.raises(ValueError):
        collect(tmp_path / "work", specs=[_spec(feed_repo), _spec(feed_repo)], pin=second)
