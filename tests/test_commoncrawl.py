"""
Offline tests for the Common Crawl sampler.

Every test here runs without network. The sampler's correctness lives in two
places that are easy to get wrong and were both got wrong during development:
gzip resynchronisation (the magic bytes occur by chance inside compressed data)
and the record filter (an early version reported '.132' as a TLD by taking the
last octet of an IP-literal host).
"""
from __future__ import annotations

import gzip
import json

import pytest

from uie.corpus.commoncrawl import CrawlIndex, CrawlRecord, SampleStats


def _cdxj_line(url: str, *, status: str = "200", mime: str = "text/html",
               languages: str = "eng", timestamp: str = "20260701120000") -> str:
    """One CDXJ line: `<surt-key> <timestamp> {<json>}`.

    The timestamp is deliberately NOT repeated inside the JSON, because real
    records do not repeat it. An earlier fixture did, which meant it encoded the
    same misunderstanding as the parser and happily validated a bug that left
    every extracted URL with an empty timestamp.
    """
    host = url.split("://", 1)[-1].split("/", 1)[0]
    surt = ",".join(reversed(host.split("."))) + ")/"
    payload = {"url": url, "status": status, "mime-detected": mime,
               "languages": languages}
    return f"{surt} {timestamp} {json.dumps(payload)}"


def _block(lines: list[str]) -> bytes:
    return gzip.compress(("\n".join(lines) + "\n").encode("utf-8"))


# --- resynchronisation -----------------------------------------------------

def test_resync_decodes_a_clean_member():
    blob = _block([_cdxj_line("https://example.com/a/b")])
    assert "example.com/a/b" in CrawlIndex._resync(blob)


def test_resync_skips_leading_garbage():
    """A random offset lands mid-member; the decoder must scan forward."""
    blob = b"\x00\xff\x13garbage-preamble" + _block([_cdxj_line("https://a.com/x")])
    assert "a.com/x" in CrawlIndex._resync(blob)


def test_resync_rejects_false_gzip_magic():
    """The magic can occur by chance inside compressed bytes.

    A candidate that does not decompress into parseable CDXJ must be skipped
    rather than accepted, or the sampler silently drops real blocks.
    """
    blob = b"\x1f\x8b\x08" + b"\x00" * 40 + _block([_cdxj_line("https://b.com/y")])
    assert "b.com/y" in CrawlIndex._resync(blob)


def test_resync_returns_empty_when_no_valid_member():
    assert CrawlIndex._resync(b"no gzip member anywhere in here") == ""


def test_resync_rejects_gzip_that_is_not_cdxj():
    """Valid gzip carrying non-CDXJ payload is not a block."""
    assert CrawlIndex._resync(gzip.compress(b"just some prose\n")) == ""


# --- record filtering ------------------------------------------------------

def _parsed(lines: list[str]) -> tuple[list[CrawlRecord], SampleStats]:
    stats = SampleStats()
    recs = list(CrawlIndex._parse("\n".join(lines), "CC-MAIN-2026-30", stats))
    return recs, stats


def test_keeps_only_http_200():
    recs, _ = _parsed([_cdxj_line("https://a.com/ok"),
                       _cdxj_line("https://a.com/gone", status="404"),
                       _cdxj_line("https://a.com/moved", status="301")])
    assert [r.url for r in recs] == ["https://a.com/ok"]


def test_keeps_only_html():
    """Assets and JSON endpoints are not pages the web "named" for a reader."""
    recs, _ = _parsed([_cdxj_line("https://a.com/page"),
                       _cdxj_line("https://a.com/s.css", mime="text/css"),
                       _cdxj_line("https://a.com/d.json", mime="application/json")])
    assert [r.url for r in recs] == ["https://a.com/page"]


def test_malformed_json_is_skipped_not_fatal():
    recs, stats = _parsed(["surt) 20260701120000 {not valid json",
                           _cdxj_line("https://a.com/fine")])
    assert [r.url for r in recs] == ["https://a.com/fine"]
    assert stats.n_lines == 2


def test_stats_separate_lines_from_records():
    """n_lines counts what was read; n_records counts what survived."""
    _, stats = _parsed([_cdxj_line("https://a.com/1"),
                        _cdxj_line("https://a.com/2", status="500"),
                        _cdxj_line("https://a.com/3", mime="image/png")])
    assert (stats.n_lines, stats.n_records) == (3, 1)


def test_provenance_is_attached():
    recs, _ = _parsed([_cdxj_line("https://a.com/p", languages="fra,eng")])
    assert recs[0].crawl == "CC-MAIN-2026-30"
    assert recs[0].languages == "fra,eng"


def test_timestamp_comes_from_the_LINE_not_the_json():
    """Regression: the capture time is the second field of the CDXJ line and is
    absent from the JSON. Reading only the JSON produced an empty timestamp for
    all 6.5M extracted URLs, which left the temporal split with no benign data.
    """
    recs, _ = _parsed([_cdxj_line("https://a.com/p", timestamp="20240115093000")])
    assert recs[0].timestamp == "20240115093000"


def test_timestamp_falls_back_to_json_when_present():
    """Some records do repeat it; prefer the line but do not ignore the JSON."""
    payload = json.dumps({"url": "https://a.com/p", "status": "200",
                          "mime-detected": "text/html",
                          "timestamp": "20250101000000"})
    recs, _ = _parsed([f"com,a)/p {payload}"])
    assert recs[0].timestamp == "20250101000000"


def test_missing_languages_defaults_to_und():
    line = json.dumps({"url": "https://a.com/p", "status": "200",
                       "mime-detected": "text/html", "languages": None})
    recs, _ = _parsed([f"com,a)/p 20260701120000 {line}"])
    assert recs[0].languages == "und"


# --- TLD extraction --------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://example.com/a", "com"),
    ("https://sub.example.co.uk/a", "uk"),
    ("https://x.co.jp/", "jp"),
])
def test_tld_extraction(url, expected):
    assert CrawlRecord(url, "", "und", "c").tld == expected


@pytest.mark.parametrize("url", [
    "http://192.168.1.132/login",
    "http://10.1.1.0/",
])
def test_ip_literal_hosts_have_no_tld(url):
    """Regression: an early version reported '.132' as a TLD by taking the
    last octet, polluting the TLD distribution with numeric 'suffixes'."""
    assert CrawlRecord(url, "", "und", "c").tld is None


def test_hostless_url_has_no_tld():
    assert CrawlRecord("not a url at all", "", "und", "c").tld is None


# --- sampling bookkeeping --------------------------------------------------

def test_stats_summary_reports_blocks_not_just_records():
    """`n_blocks` is the honest denominator for TLD/language shares, because a
    block is ~3000 CONTIGUOUS SURT lines and its URLs are highly correlated."""
    stats = SampleStats(n_blocks=54, n_records=112_283, n_lines=200_000)
    text = stats.summary()
    assert "54 blocks" in text and "112,283 records" in text


def test_a_response_cut_short_is_retried_not_fatal(monkeypatch):
    # D2 step 6: an IncompleteRead (an HTTPException, not an OSError) escaped the retries and
    # ended a whole crawl's extraction
    import http.client
    import uie.corpus.commoncrawl as cc

    calls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            calls.append(1)
            if len(calls) == 1:
                raise http.client.IncompleteRead(b"partial", 10)
            return b"whole"

    monkeypatch.setattr(cc.urllib.request, "urlopen", lambda req, timeout: Resp())
    monkeypatch.setattr(cc, "_BACKOFF_S", 0.0)
    assert cc._get("https://index.example/x") == b"whole" and len(calls) == 2
