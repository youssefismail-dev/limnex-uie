"""
research/d2/audit.py — D2 step 2, the release audit (research/d2/D2_PLAN.md).

    python research/d2/audit.py

Scans (a) every tracked file of the current tree and (b) every file version ever committed (all
blobs reachable from any ref), decompressing .gz, for:

  real phishing      URLs, hosts or registrable domains of the project's phishing corpus
                     (data/corpus, data/corpus2, data/bundle2). Hosts and domains count only when
                     they never occur in the benign corpus, so shared platforms are not flagged.
  bulk URL lists     files holding many real hostnames (corpus-derived lists, generated samples)
  secrets            tokens and keys (Kaggle, GitHub, ngrok, AWS, Google, OpenAI, Anthropic,
                     Hugging Face), private keys, assigned passwords/tokens
  private details    local user paths in any style, e-mail addresses, machine names
  brands             hostnames containing a well-known brand inside Markdown documents
                     (illustrative phishing-style examples), for review

Reports locations only: file, line and category, never the matched URL or secret. Writes
research/d2/AUDIT.json and prints a summary. Changes nothing.
"""
from __future__ import annotations

import collections
import gzip
import os
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "d2" / "AUDIT.json"

PHISHING = [ROOT / "data/corpus/phishing.jsonl"] + sorted((ROOT / "data/corpus2").glob("phishing_*.jsonl.gz")) \
    + sorted((ROOT / "data/bundle2").glob("phishing_*.jsonl.gz"))
BENIGN = [ROOT / f"data/bundle2/{s}.jsonl.gz" for s in ("name_train", "name_val", "structure_train", "structure_val")]
BULK_HOSTS = 50                    # a file with this many distinct real hostnames is a list, not prose

HOST = re.compile(r"(?<![a-z0-9_.@-])((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{2,63}|xn--[a-z0-9-]{1,59}))"
                  r"(/[^\s\"'`<>()\[\]{}\\,|]*)?", re.I)
SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.I)
SECRETS = {
    "kaggle_token": re.compile(r"KGAT_[A-Za-z0-9]{16,}"),
    "kaggle_key": re.compile(r"\"key\"\s*:\s*\"[0-9a-f]{32}\""),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "ngrok_token": re.compile(r"\b[0-9][A-Za-z0-9]{20,}_[0-9A-Za-z]{15,}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "google_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "assigned_secret": re.compile(r"(?i)\b(?:api[_-]?key|secret|passw(?:or)?d|auth[_-]?token|access[_-]?token)\b"
                                  r"\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']"),
}
PRIVATE = {
    "local_path": re.compile(r"(?i)\b[a-z]:[\\/]+(?:DJ|Users)\b|(?<![\w.])/[a-z]/(?:DJ|Users)/|/home/[a-z0-9_-]+/|/Users/[A-Za-z0-9_-]+/"),
    # this machine's user, read at run time (no name is written in this file)
    "username": re.compile(re.escape(os.environ.get("USERNAME") or os.environ.get("USER") or Path.home().name), re.I),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
    "machine_name": re.compile(r"\b(?:DESKTOP|LAPTOP)-[A-Z0-9]{5,}\b"),
}
#: Addresses that are public by design. GitHub's no-reply addresses exist to be published in place of
#: a private one; the public repository's commit carries the owner's as its author (decision F).
EMAIL_OK = re.compile(r"(?i)@(?:example\.(?:com|org|net)|anthropic\.com|users\.noreply\.github\.com)$"
                      r"|^noreply@|^git@github\.com$")

#: Reviewed exceptions (D2 audit, research/d2/AUDIT.md). Each is reported, not counted as a finding.
#: Hosting platforms named in analysis: the registrable domain belongs to the platform, which
#: carries benign and phishing sites alike; naming it points at no one's site.
REVIEWED_PLATFORMS = {"appspot.com", "blogspot.com", "blogspot.ro", "fc2.com", "com.de", "pages.dev",
                      "netlify.app", "vercel.app", "workers.dev", "weebly.com", "weeblysite.com",
                      "godaddysites.com", "github.io", "serv00.net"}
#: (path, category) -> reason
REVIEWED = {
    ("E3_REPORT.md", "brand_in_md_host"):
        "model output (E3_REPORT section 3.4 qualitative samples and its parse example); matches no "
        "corpus URL or host; kept unchanged as evidence of what the model generates",
    ("tests/fixtures_e7_generate.json", "bulk_hostnames"):
        "random strings from a tiny untrained test model (tests/test_hc1_enforce.py); no corpus match",
    ("tests/fixtures_e7_generate.json", "private:email"):
        "random strings from the same tiny test model that happen to contain '@'; not addresses",
    ("tests/fixtures_e5_path.json", "private:email"):
        "random strings from a tiny test model that happen to contain '@'; not addresses",
}
BRANDS = ("paypal", "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc", "barclays", "santander", "apple",
          "icloud", "microsoft", "office365", "outlook", "amazon", "netflix", "google", "gmail", "facebook",
          "instagram", "whatsapp", "linkedin", "dhl", "fedex", "usps", "ups", "coinbase", "binance", "metamask",
          "steam", "adobe", "dropbox", "docusign", "ebay", "walmart", "americanexpress", "amex", "visa",
          "mastercard", "irs", "hmrc", "outlook", "yahoo", "aol", "att", "verizon", "spotify")


def norm_url(u: str) -> str:
    u = SCHEME.sub("", u.strip())
    host, sep, rest = u.partition("/")
    return host.lower().rstrip(".") + sep + rest


def registrable(host: str) -> str:
    from uie.factorization import PSL
    ext = PSL(host)
    return f"{ext.domain}.{ext.suffix}".lower() if ext.domain and ext.suffix else ""


def iter_urls(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)["url"]


def reference_sets():
    p_urls, p_hosts = set(), set()
    for f in PHISHING:
        for u in iter_urls(f):
            n = norm_url(u)
            p_urls.add(n.rstrip("/"))
            p_hosts.add(n.split("/", 1)[0])
    b_hosts = set()
    for f in BENIGN:
        for u in iter_urls(f):
            b_hosts.add(norm_url(u).split("/", 1)[0])
    b_domains = {registrable(h) for h in b_hosts}
    p_only_hosts = p_hosts - b_hosts
    p_only_domains = {registrable(h) for h in p_only_hosts} - b_domains - {""}
    return p_urls, p_only_hosts, p_only_domains


def text_of(name: str, data: bytes):
    if name.endswith(".gz"):
        try:
            data = gzip.decompress(data)
        except Exception:
            return None
    if b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def scan(name: str, text: str, refs):
    p_urls, p_hosts, p_domains = refs
    found = collections.defaultdict(list)
    real_hosts = set()
    for ln, line in enumerate(text.splitlines(), 1):
        for m in HOST.finditer(line):
            host, path = m.group(1).lower(), (m.group(2) or "")
            dom = registrable(host)
            if not dom:
                continue                                   # not a real public suffix (code, file names)
            real_hosts.add(host)
            full = (host + path).rstrip("/.,;:")
            if full in p_urls and path:
                found["phishing_url"].append(ln)
            elif dom in REVIEWED_PLATFORMS and host == dom:
                found["reviewed:platform_named"].append(ln)
            elif host in p_hosts:
                found["phishing_host"].append(ln)
            elif dom in p_domains and host == dom:
                found["phishing_domain"].append(ln)
            if name.endswith(".md") and any(b in host.replace("-", "") for b in BRANDS):
                found["brand_in_md_host"].append(ln)
        for key, rx in SECRETS.items():
            if rx.search(line):
                found["secret:" + key].append(ln)
        for key, rx in PRIVATE.items():
            for m in rx.finditer(line):
                if key == "email" and EMAIL_OK.search(m.group(0)):
                    continue
                if key == "email" and line[max(0, m.start() - 1):m.start()] in ("/", "@", ":"):
                    found["reviewed:url_userinfo"].append(ln)      # http://user@host, not an address
                    continue
                found["private:" + key].append(ln)
    if len(real_hosts) >= BULK_HOSTS:
        found["bulk_hostnames"].append(len(real_hosts))
    for key in list(found):
        if (name, key) in REVIEWED:
            found["reviewed:" + key] = found.pop(key)
    return found


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    refs = reference_sets()
    print(f"reference: {len(refs[0]):,} phishing URLs, {len(refs[1]):,} phishing-only hosts, "
          f"{len(refs[2]):,} phishing-only registrable domains", flush=True)
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, check=True).stdout  # noqa: E731

    # ---- (a) the current tree --------------------------------------------------------
    tree = {}
    listed = [n for n in git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode("utf-8").split("\0")
              if n and (ROOT / n).is_file()]                     # what the next commit would hold
    for name in listed:
        text = text_of(name, (ROOT / name).read_bytes())
        if text is not None:
            f = scan(name, text, refs)
            if f:
                tree[name] = {k: (v if k == "bulk_hostnames" else sorted(set(v))) for k, v in f.items()}
    # ---- (b) every blob in history ------------------------------------------------------
    objects = git("rev-list", "--all", "--objects").decode("utf-8").splitlines()
    paths = collections.defaultdict(set)
    for line in objects:
        sha, _, path = line.partition(" ")
        if path:
            paths[sha].add(path)
    shas = list(paths)
    history = collections.defaultdict(lambda: {"blobs": 0, "paths": set()})
    proc = subprocess.Popen(["git", "-C", str(ROOT), "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    n_blobs = 0
    for sha in shas:
        proc.stdin.write((sha + "\n").encode())
        proc.stdin.flush()
        header = proc.stdout.readline().decode().split()
        size = int(header[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)
        if header[1] != "blob":
            continue
        n_blobs += 1
        name = sorted(paths[sha])[0]
        text = text_of(name, data)
        if text is None:
            continue
        for k in scan(name, text, refs):
            history[k]["blobs"] += 1
            history[k]["paths"].update(paths[sha])
    proc.stdin.close()
    proc.wait()

    summary = collections.Counter()
    for f in tree.values():
        for k, v in f.items():
            summary[k] += 1
    report = {"reference": {"phishing_urls": len(refs[0]), "phishing_only_hosts": len(refs[1]),
                            "phishing_only_domains": len(refs[2])},
              "tree": {"files_tracked": len(listed), "files_with_findings": len(tree),
                       "by_category": dict(summary), "files": tree,
                       "open_findings": {k: n for k, n in summary.items() if not k.startswith("reviewed:")},
                       "reviewed_reasons": {f"{a} | {b}": r for (a, b), r in REVIEWED.items()}},
              "history": {"blobs_scanned": n_blobs,
                          "by_category": {k: {"blobs": v["blobs"], "paths": sorted(v["paths"])}
                                          for k, v in history.items()}}}
    report["tree"]["clean"] = not report["tree"]["open_findings"]
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"tree: {report['tree']['files_tracked']} files, {len(tree)} with findings; "
          + ("CLEAN (only reviewed exceptions)" if report["tree"]["clean"]
             else f"OPEN: {report['tree']['open_findings']}"))
    for k, n in sorted(summary.items()):
        print(f"  {k:<26} {n} files")
    print(f"history: {n_blobs:,} blobs scanned")
    for k, v in sorted(history.items()):
        print(f"  {k:<26} {v['blobs']} blobs, {len(v['paths'])} paths")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
