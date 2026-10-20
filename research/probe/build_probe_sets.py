"""
research/probe/build_probe_sets.py — the representation probe's two URL sets
(research/probe/PROBE_PLAN.md, "Data"). Runs no model.

    python research/probe/build_probe_sets.py

Held-out benign URLs (data/bundle2 name_val + structure_val) with Common Crawl's page-content
language. Main set: generic-TLD URLs; second set: any TLD. Languages: every language (not `und`)
with at least 500 distinct registrable domains on generic TLDs. Per language, 500 registrable
domains, one URL each; no domain appears twice in a set. Each URL is assigned now to the 20%
layer-selection part or to one of five folds of the 80% (stratified by language), so the probe
never draws a split itself. Writes data/probe/{main,all}.jsonl (git-ignored: Common Crawl URLs
are not redistributed) and research/probe/SETS.json (counts and sha256s). Refuses to overwrite.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SOURCES = [ROOT / "data/bundle2/name_val.jsonl.gz", ROOT / "data/bundle2/structure_val.jsonl.gz"]
OUT = ROOT / "data" / "probe"
RECORD = ROOT / "research" / "probe" / "SETS.json"
SEED = 20261017
PER_LANGUAGE = 500
SELECT_SHARE = 0.2
FOLDS = 5
_AUTHORITY = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.\-]*://)?([^/?#]*)")


def host_of(url: str) -> str:
    return _AUTHORITY.match(url.strip()).group(1).rsplit("@", 1)[-1].split(":")[0].rstrip(".").lower()


def tld_kind(host: str) -> str:
    """'country' if the last label is two ASCII letters or an IDN (xn--) label, else 'generic'."""
    last = host.rsplit(".", 1)[-1]
    return "country" if (len(last) == 2 and last.isascii() and last.isalpha()) or last.startswith("xn--") else "generic"


def registrable(url: str, host: str) -> str:
    from uie.factorization import parse
    try:
        p = parse(url)
        return f"{p.registrable_label}.{p.public_suffix}".lower() if p.registrable_label else host
    except Exception:
        return host


def assign_parts(rows, rng):
    """Per language: 20% to 'select', the rest dealt round-robin into folds 0-4 after a shuffle."""
    by_lang = collections.defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    for items in by_lang.values():
        rng.shuffle(items)
        n_sel = round(SELECT_SHARE * len(items))
        for i, r in enumerate(items):
            r["part"] = "select" if i < n_sel else f"fold{(i - n_sel) % FOLDS}"


def draw(pool, langs, rng):
    """pool: {lang: {domain: [urls]}} -> PER_LANGUAGE (domain, url) per language, domains unique."""
    used, rows = set(), []
    for lang in langs:
        domains = sorted(d for d in pool[lang] if d not in used)
        rng.shuffle(domains)
        chosen = domains[:PER_LANGUAGE]
        if len(chosen) < PER_LANGUAGE:
            raise SystemExit(f"STOP: {lang} has only {len(chosen)} unused domains")
        for d in chosen:
            used.add(d)
            rows.append({"url": rng.choice(sorted(pool[lang][d])), "lang": lang, "domain": d})
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if (OUT / "main.jsonl").exists() or RECORD.exists():
        raise SystemExit("STOP: the probe sets exist; they are frozen")
    generic = collections.defaultdict(lambda: collections.defaultdict(set))
    every = collections.defaultdict(lambda: collections.defaultdict(set))
    n = 0
    for src in SOURCES:
        with gzip.open(src, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                lang = r.get("lang")
                if not lang or lang == "und":
                    continue
                host = host_of(r["url"])
                dom = registrable(r["url"], host)
                every[lang][dom].add(r["url"])
                if tld_kind(host) == "generic":
                    generic[lang][dom].add(r["url"])
                n += 1
    langs = sorted(l for l in generic if len(generic[l]) >= PER_LANGUAGE)
    print(f"read {n:,} labelled held-out URLs; languages with >= {PER_LANGUAGE} generic-TLD domains: {langs}")
    rng = random.Random(SEED)
    sets = {"main": draw(generic, langs, rng), "all": draw(every, langs, rng)}
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"plan": "research/probe/PROBE_PLAN.md", "seed": SEED, "per_language": PER_LANGUAGE,
              "languages": langs, "chance": 1 / len(langs),
              "available_generic_domains": {l: len(generic[l]) for l in langs},
              "available_all_domains": {l: len(every[l]) for l in langs}, "sets": {}}
    for name, rows in sets.items():
        assign_parts(rows, rng)
        for r in rows:
            r["tld_kind"] = tld_kind(host_of(r["url"]))
        path = OUT / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        record["sets"][name] = {"file": f"data/probe/{name}.jsonl", "rows": len(rows), "sha256": sha256(path),
                                "parts": dict(collections.Counter(r["part"] for r in rows)),
                                "tld_kind": dict(collections.Counter(r["tld_kind"] for r in rows))}
    RECORD.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("languages", "chance")}, indent=1))
    print(json.dumps(record["sets"], indent=1))


if __name__ == "__main__":
    main()
