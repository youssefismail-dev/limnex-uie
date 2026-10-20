"""
research/v2/v2_p2_tables.py

V2-P2 (research/v2/PREREG_V2_P2.md section 4.3; Amendment 1, section 18): the two piece
tables, learned by source frequency, and their audit (section 4.7, V-T4).

    python research/v2/v2_p2_tables.py build     # -> data/v2/p2/tables/{rest,host}.json, build.json
    python research/v2/v2_p2_tables.py audit     # -> data/v2/p2/tables/audit.json

- **The source** is a row's registrable domain under the full Public Suffix List, private
  section included.
  - Within one private-section platform, near-duplicate sites are one source: MinHash LSH
    over character 3-grams of the site's own label, confirmed by an edit similarity above
    0.8; the clusters are connected components.
- **The count:** a pair of adjacent pieces counts once for each source whose strings contain
  it, however many times.
  - Each source weighs w_c / N_c: its class's training draw share over the class's number
    of sources.
  - The benign rest table's multi-page sources are a 15% sample, each weighing 1 / 0.15.
- **No length cap, no threshold, no banned string.** A pair found in a single source is
  never merged (Amendment 2 reads section 4.3 point 5 this way); learning stops at the
  budget, or when no pair two sources share is left.

The learning data are the kept training rows (character encoding within 384 tokens) and
stream 1. The rest table takes path, query and fragment bodies (stream 1: query bodies only);
the host table takes registrable labels and subdomain labels, one label per string.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_bpe import BPETable  # noqa: E402

DATA = ROOT / "data" / "bundle2"
STREAM = ROOT / "data" / "e15" / "qry_extra" / "qry_extra.jsonl"
RECORD = ROOT / "data" / "v2" / "p1b" / "e16_cap_record.json"
OUT = ROOT / "data" / "v2" / "p2" / "tables"
R1 = ROOT / "data" / "v2" / "r1"

TRAIN_STREAMS = (("name_train", "benign"), ("structure_train", "benign"),
                 ("phishing_name_train", "phishing"), ("phishing_structure_train", "phishing"))
CLASSES = ("benign", "phishing")
TABLES = ("rest", "host")
#: R1's registered sizes (DR2), the only limit on each table (section 4.3, point 4)
BUDGET = {"rest": 32_000, "host": 16_000}
#: Amendment 1 (section 18.1): the benign rest table's multi-page sources, a uniform sample
SAMPLE_RATE = {("rest", "benign"): 0.15}
SAMPLE_SEED = 20261070
#: the classes' training draw shares at class alpha = 1 (label counts; section 18.2)
LABEL_COUNTS = {"benign": 906_214, "phishing": 192_652}
#: near-duplicate clustering (section 18.2)
MINHASH_K, BANDS, ROWS = 120, 40, 3
MINHASH_SEED = 20261073
EDIT_SIM = 0.8
MAX_LEN = 384
#: smoke tests only: rows read per stream (None: all)
LIMIT: Optional[int] = None


# --------------------------------------------------------------------------- #
# the data: kept rows, their field strings and their sources
# --------------------------------------------------------------------------- #

def char_tokenizer():
    """`e16_cap`'s tokenizer, from its record (the rows both arms keep fit 384 tokens in it)."""
    from uie.tokenizer import URLTokenizer
    return URLTokenizer.from_state_dict(json.loads(RECORD.read_text(encoding="utf-8"))["tokenizer"])


def _open(stem: str):
    for name in (f"{stem}.jsonl.gz", f"{stem}.jsonl"):
        path = DATA / name
        if path.is_file():
            return gzip.open(path, "rt", encoding="utf-8") if name.endswith(".gz") else open(
                path, "rt", encoding="utf-8")
    raise FileNotFoundError(f"{stem} is not in {DATA}")


def field_strings(parts) -> Tuple[List[str], List[str]]:
    """(rest strings, host strings) exactly as `URLTokenizer.encode` writes those fields."""
    from uie.ngram import path_body
    body = path_body(parts.path) if parts.path else ""
    rest = [s for s in (body, parts.query if parts.has_query else "",
                        parts.fragment if parts.has_fragment else "") if s]
    host = [s for s in (parts.registrable_label,) + tuple(parts.subdomains) if s]
    return rest, host


class Sources:
    """A row's source: its registrable domain under the full suffix list (private section
    included), with the platform it sits on when that suffix is private."""

    def __init__(self) -> None:
        import tldextract
        self.ex = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
        self._cache: Dict[str, Tuple[str, Optional[str], str]] = {}

    def __call__(self, host: str) -> Tuple[str, Optional[str], str]:
        """(source key, private platform suffix or None, the site's own label)."""
        host = (host or "").lower()
        hit = self._cache.get(host)
        if hit is None:
            r = self.ex(host)
            key = r.top_domain_under_public_suffix or host
            hit = (key, r.suffix if getattr(r, "is_private", False) and r.domain else None, r.domain)
            if len(self._cache) < 4_000_000:
                self._cache[host] = hit
        return hit


def rows(tok=None) -> Iterator[Tuple[str, str, bool, object]]:
    """(class, url, is_stream_row, parts) for every kept training row, then stream 1."""
    from uie.factorization import parse
    tok = tok or char_tokenizer()
    for stem, cls in TRAIN_STREAMS:
        with _open(stem) as fh:
            for k, line in enumerate(fh):
                if LIMIT is not None and k >= LIMIT:
                    break
                url = json.loads(line)["url"]
                if len(tok.encode(url, cls)) > MAX_LEN:
                    continue                         # e16_cap drops it: so do both arms
                yield cls, url, False, parse(url)
    with open(STREAM, encoding="utf-8") as fh:
        for k, line in enumerate(fh):
            if LIMIT is not None and k >= LIMIT:
                break
            if line.strip():
                r = json.loads(line)
                yield r["label"], r["url"], True, parse(r["url"])


def collect(log=print) -> dict:
    """Every table's strings per (class, source), and the private platforms' sites."""
    src = Sources()
    data = {t: {c: defaultdict(set) for c in CLASSES} for t in TABLES}
    platforms: Dict[str, Dict[str, str]] = defaultdict(dict)     # suffix -> {source key: label}
    n = Counter()
    t0 = time.time()
    for cls, url, stream, parts in rows():
        key, platform, label = src(parts.host_raw)
        rest, host = field_strings(parts)
        if stream:                                    # stream 1: its query bodies only (section 4.1)
            rest = [parts.query] if parts.has_query and parts.query else []
        for s in rest:
            data["rest"][cls][key].add(s)
        for s in host:
            data["host"][cls][key].add(s)
        if platform:
            platforms[platform][key] = label
        n[(cls, "stream" if stream else "train")] += 1
        if sum(n.values()) % 500_000 == 0:
            log(f"    {sum(n.values()):,} rows ({time.time() - t0:.0f}s)")
    return {"data": data, "platforms": platforms, "rows": {f"{c}|{k}": v for (c, k), v in n.items()}}


# --------------------------------------------------------------------------- #
# near-duplicate sites within a platform (Lee et al., 2022)
# --------------------------------------------------------------------------- #

def edit_similarity(a: str, b: str, floor: float = EDIT_SIM) -> float:
    """1 - Levenshtein(a, b) / max(len). Banded: returns 0.0 as soon as the similarity
    cannot exceed `floor`, so only near pairs pay for the full distance."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    longest = max(la, lb)
    if longest == 0:
        return 1.0
    k = int((1.0 - floor) * longest)                 # the largest distance that can still pass
    if abs(la - lb) > k:
        return 0.0
    big = k + 1
    prev = [j if j <= k else big for j in range(lb + 1)]
    for i in range(1, la + 1):
        cur = [big] * (lb + 1)
        cur[0] = i if i <= k else big
        lo, hi = max(1, i - k), min(lb, i + k)
        best = cur[0]
        ai = a[i - 1]
        for j in range(lo, hi + 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
            cur[j] = v if v <= k else big
            if cur[j] < best:
                best = cur[j]
        if best > k:
            return 0.0
        prev = cur
    d = prev[lb]
    return 1.0 - d / longest if d <= k else 0.0


def _minhash_params():
    import numpy as np
    rng = np.random.RandomState(MINHASH_SEED)
    p = (1 << 31) - 1
    return (rng.randint(1, p, size=MINHASH_K).astype(np.int64),
            rng.randint(0, p, size=MINHASH_K).astype(np.int64), p)


def _gram_hash(g: str) -> int:
    return int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest(), "little") & 0x7FFFFFFF


def cluster_platform(labels: Dict[str, str], params) -> List[List[str]]:
    """Near-duplicate clusters (size >= 2) among one platform's sites, as lists of source keys."""
    import numpy as np
    a, b, p = params
    keys = sorted(labels)
    if len(keys) < 2:
        return []
    parent = list(range(len(keys)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        lab = labels[k]
        grams = {lab[j:j + 3] for j in range(max(1, len(lab) - 2))} if lab else {""}
        x = np.array([_gram_hash(g) for g in sorted(grams)], dtype=np.int64)
        sig = ((a[:, None] * x[None, :] + b[:, None]) % p).min(axis=1)
        for band in range(BANDS):
            buckets[(band, tuple(int(v) for v in sig[band * ROWS:(band + 1) * ROWS]))].append(i)
    for members in buckets.values():
        if len(members) < 2:
            continue
        for x_i in range(len(members)):
            for y_i in range(x_i + 1, len(members)):
                i, j = members[x_i], members[y_i]
                if find(i) == find(j):
                    continue
                if edit_similarity(labels[keys[i]], labels[keys[j]]) > EDIT_SIM:
                    parent[find(i)] = find(j)
    groups: Dict[int, List[str]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[find(i)].append(k)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: (-len(g), g[0]))


def clusters(platforms: Dict[str, Dict[str, str]], log=print) -> Dict[str, str]:
    """source key -> its cluster's representative (the first key in sorted order)."""
    params = _minhash_params()
    rep: Dict[str, str] = {}
    t0 = time.time()
    for suffix in sorted(platforms):
        for group in cluster_platform(platforms[suffix], params):
            for k in group:
                rep[k] = group[0]
    log(f"    clusters: {len(set(rep.values())):,} near-duplicate clusters over {len(rep):,} sites "
        f"({time.time() - t0:.0f}s)")
    return rep


# --------------------------------------------------------------------------- #
# the learner: source frequency (section 4.3), with presence per source
# --------------------------------------------------------------------------- #

def units(per_source: Dict[str, Dict[str, set]], rep: Dict[str, str], table: str):
    """Merge each class's sources by cluster. Returns, per class, {unit key: set of strings}."""
    out = {}
    for cls in CLASSES:
        merged: Dict[str, set] = defaultdict(set)
        for key, strings in per_source[cls].items():
            merged[rep.get(key, key)] |= strings
        out[cls] = dict(merged)
    return out


def learn_df(by_class: Dict[str, Dict[str, set]], table: str, *, budget: int,
             log=print, log_every: int = 2000) -> Tuple[BPETable, dict]:
    """Greedy BPE whose pair score is the weighted number of sources containing the pair.

    by_class: {class: {unit key: set of strings}} over the table's full data. Single-string
    units are pooled by string (their counts per class); multi-string units keep their own
    words, sampled where Amendment 1 says, each weighing 1 / rate.

    A pair in fewer than two sources is skipped, not merged, however high its weighted score;
    it returns to the heap if a later merge changes its count."""
    t0 = time.time()
    n_units = {c: len(by_class[c]) for c in CLASSES}
    draw = {c: LABEL_COUNTS[c] / sum(LABEL_COUNTS.values()) for c in CLASSES}
    unit_w = [draw[c] / max(n_units[c], 1) for c in CLASSES]
    rate = [SAMPLE_RATE.get((table, c), 1.0) for c in CLASSES]
    mult = [1.0 / r for r in rate]

    single: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    multi: List[Tuple[int, List[str]]] = []
    sampled = [0, 0]
    rng = random.Random(SAMPLE_SEED)
    for ci, cls in enumerate(CLASSES):
        for key in sorted(by_class[cls]):
            strings = by_class[cls][key]
            if len(strings) == 1:
                single[next(iter(strings))][ci] += 1
            else:
                keep = rate[ci] >= 1.0 or rng.random() < rate[ci]
                if keep:
                    multi.append((ci, sorted(strings)))
                    sampled[ci] += 1
    alphabet = sorted({ch for s in single for ch in s} | {ch for _, ss in multi for s in ss for ch in s})
    sym: List[str] = list(alphabet)
    sid: Dict[str, int] = {c: i for i, c in enumerate(sym)}
    words: List[List[int]] = []
    owner: List[int] = []                      # -1: a pooled single-string word; else the unit
    counts: List[Tuple[int, int]] = []          # pooled words: sources per class
    unit_class: List[int] = []
    unit_words: List[List[int]] = []
    for s in sorted(single):
        words.append([sid[c] for c in s])
        owner.append(-1)
        counts.append(tuple(single[s]))
    for ci, strings in multi:
        u = len(unit_words)
        unit_class.append(ci)
        unit_words.append([])
        for s in strings:
            unit_words[u].append(len(words))
            words.append([sid[c] for c in s])
            owner.append(u)
            counts.append((0, 0))
    big_unit = [len(w) > 8 for w in unit_words]
    counter: List[Optional[Dict[int, int]]] = [{} if big else None for big in big_unit]

    def key(a: int, b: int) -> int:
        return (a << 16) | b

    stats: Dict[int, List[int]] = defaultdict(lambda: [0, 0, 0, 0])   # single b/p, multi b/p
    where: Dict[int, List[int]] = defaultdict(list)

    def pairs_of(w: List[int]) -> set:
        return {key(x, y) for x, y in zip(w, w[1:])}

    for wi, w in enumerate(words):
        ps = pairs_of(w)
        for p in ps:
            where[p].append(wi)
        u = owner[wi]
        if u < 0:
            cb, cp = counts[wi]
            for p in ps:
                st = stats[p]
                st[0] += cb
                st[1] += cp
        elif counter[u] is not None:
            c = counter[u]
            for p in ps:
                c[p] = c.get(p, 0) + 1
    for u, ws in enumerate(unit_words):
        present = set().union(*(pairs_of(words[wi]) for wi in ws)) if ws else set()
        ci = unit_class[u]
        for p in present:
            stats[p][2 + ci] += 1

    def score(st: List[int]) -> float:
        return unit_w[0] * (st[0] + mult[0] * st[2]) + unit_w[1] * (st[1] + mult[1] * st[3])

    def support(st: List[int]) -> int:
        return st[0] + st[1] + st[2] + st[3]

    heap = [(-score(st), sym[p >> 16], sym[p & 0xFFFF], p) for p, st in stats.items()]
    heapq.heapify(heap)
    merges: List[Tuple[str, str]] = []
    merge_support: List[int] = []
    merge_score: List[float] = []
    done = set()
    stop = "budget"

    def other_has(u: int, wi: int, p: int) -> bool:
        for wj in unit_words[u]:
            if wj != wi:
                w = words[wj]
                for x, y in zip(w, w[1:]):
                    if key(x, y) == p:
                        return True
        return False

    literal_stop: Optional[int] = None

    while len(merges) < budget:
        best = None
        while heap:
            negs, sa, sb, p = heapq.heappop(heap)
            st = stats.get(p)
            if st is None or support(st) <= 0:
                continue
            s = score(st)
            if s != -negs:                          # stale: put it back as it is now
                heapq.heappush(heap, (-s, sa, sb, p))
                continue
            if support(st) < 2:                     # in a single source: never vocabulary (Amendment 2)
                if literal_stop is None:
                    literal_stop = len(merges)      # where "stop at the best pair" would have ended
                continue                            # a later merge that changes its count pushes it back
            best = (p, sa, sb, st)
            break
        if best is None:
            stop = "no pair shared by two sources"
            break
        p, sa, sb, st = best
        a, b = p >> 16, p & 0xFFFF
        text = sa + sb
        new = sid.get(text)
        if new is None:
            new = len(sym)
            sym.append(text)
            sid[text] = new
            if new >= (1 << 16):
                raise OverflowError("more than 65,535 symbols")
        if (sa, sb) not in done:
            done.add((sa, sb))
            merges.append((sa, sb))
            merge_support.append(support(st))
            merge_score.append(score(st))
        touched = set()
        for wi in sorted(set(where.pop(p, ()))):
            w = words[wi]
            out, i, n, hit = [], 0, len(w), False
            while i < n:
                if i < n - 1 and w[i] == a and w[i + 1] == b:
                    out.append(new)
                    i += 2
                    hit = True
                else:
                    out.append(w[i])
                    i += 1
            if not hit:
                continue
            before, after = pairs_of(w), pairs_of(out)
            lost, gained = before - after, after - before
            words[wi] = out
            for q in gained:
                where[q].append(wi)
            u = owner[wi]
            if u < 0:
                cb, cp = counts[wi]
                for q in lost:
                    sq = stats[q]
                    sq[0] -= cb
                    sq[1] -= cp
                for q in gained:
                    sq = stats[q]
                    sq[0] += cb
                    sq[1] += cp
            else:
                ci = unit_class[u]
                c = counter[u]
                for q in lost:
                    if c is not None:
                        c[q] -= 1
                        if c[q] == 0:
                            del c[q]
                            stats[q][2 + ci] -= 1
                    elif not other_has(u, wi, q):
                        stats[q][2 + ci] -= 1
                for q in gained:
                    if c is not None:
                        if c.get(q, 0) == 0:
                            stats[q][2 + ci] += 1
                        c[q] = c.get(q, 0) + 1
                    elif not other_has(u, wi, q):
                        stats[q][2 + ci] += 1
            touched |= lost | gained
        stats.pop(p, None)
        for q in touched:
            if q == p:
                continue
            sq = stats.get(q)
            if sq is None:
                continue
            if support(sq) <= 0:
                stats.pop(q, None)
                where.pop(q, None)
                continue
            heapq.heappush(heap, (-score(sq), sym[q >> 16], sym[q & 0xFFFF], q))
        if log and log_every and len(merges) % log_every == 0 and merges[-1] == (sa, sb):
            log(f"    {table}: {len(merges):>6} merges, support {merge_support[-1]:,} "
                f"({time.time() - t0:.0f}s; pairs {len(stats):,})")
    meta = {"table": table, "statistic": "source frequency (PREREG_V2_P2.md section 4.3)",
            "sources": n_units, "unit_weight": unit_w, "sample_rate": rate,
            "sampled_multi_units": sampled, "pooled_single_strings": len(single),
            "learning_words": len(words), "learning_symbols": sum(len(w) for w in words),
            "budget": budget, "merges": len(merges), "stop": stop, "literal_stop_at": literal_stop,
            "merge_support": merge_support, "merge_score": merge_score,
            "seconds": round(time.time() - t0, 1)}
    return BPETable(alphabet, merges, {k: v for k, v in meta.items()
                                       if k not in ("merge_support", "merge_score")}), meta


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inputs() -> dict:
    """V-T2: every file the tables are learned from, hashed, and the suffix list's snapshot."""
    import tldextract
    out = {}
    for stem, _ in TRAIN_STREAMS:
        path = next(DATA / n for n in (f"{stem}.jsonl.gz", f"{stem}.jsonl") if (DATA / n).is_file())
        out[path.name] = _sha256(path)
    out["stream_1"] = _sha256(STREAM)
    out["e16_cap_record (the length filter's tokenizer)"] = _sha256(RECORD)
    snap = Path(tldextract.__file__).parent / ".tld_set_snapshot"
    out["public_suffix_list_snapshot"] = _sha256(snap)
    out["tldextract"] = tldextract.__version__
    return out


def peak_rss_gb() -> Optional[float]:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2 ** 30, 2)
    except Exception:
        return None


def build() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("collecting the learning data ...", flush=True)
    got = collect()
    print(f"  rows {got['rows']} ({time.time() - t0:.0f}s; rss {peak_rss_gb()} GB)", flush=True)
    rep = clusters(got["platforms"])
    record = {"registration": "research/v2/PREREG_V2_P2.md", "inputs": inputs(), "rows": got["rows"],
              "clusters": {"sites": len(rep), "clusters": len(set(rep.values())),
                           "map_sha256": hashlib.sha256(json.dumps(sorted(rep.items())).encode()).hexdigest()},
              "tables": {}}
    (OUT / "clusters.json").write_text(json.dumps(sorted(rep.items())), encoding="utf-8")
    for table in TABLES:
        by_class = units(got["data"][table], rep, table)
        print(f"learning the {table} table ...", flush=True)
        bpe, meta = learn_df(by_class, table, budget=BUDGET[table])
        sha = bpe.save(OUT / f"{table}.json")
        (OUT / f"{table}_merges.json").write_text(json.dumps(
            {"merge_support": meta["merge_support"], "merge_score": meta["merge_score"]}), encoding="utf-8")
        meta = {k: v for k, v in meta.items() if k not in ("merge_support", "merge_score")}
        meta.update({"sha256": sha, "rss_gb_after": peak_rss_gb()})
        record["tables"][table] = meta
        print(f"  {table}: {meta['merges']:,} merges ({meta['stop']}), sha {sha[:12]}..., "
              f"{meta['seconds']:.0f}s, rss {meta['rss_gb_after']} GB", flush=True)
        del by_class
    record["seconds"] = round(time.time() - t0, 1)
    (OUT / "build.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({record['seconds']:.0f}s)")


# --------------------------------------------------------------------------- #
# the audit (section 4.7, V-T4)
# --------------------------------------------------------------------------- #

def audit() -> dict:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(ROOT / "research" / "e18"))
    import v2_r1 as R1M
    SD = R1M.SD
    from v2_bpe import covered_words
    t0 = time.time()
    build_rec = json.loads((OUT / "build.json").read_text(encoding="utf-8"))
    new = {t: BPETable.load(OUT / f"{t}.json") for t in TABLES}
    sup = {t: json.loads((OUT / f"{t}_merges.json").read_text(encoding="utf-8"))["merge_support"]
           for t in TABLES}
    old = {t: BPETable.load(R1 / f"bpe_{t}.json") for t in TABLES}
    old_limit = {"rest": 32_000, "host": 16_000}
    out: dict = {"required": {}, "reported": {}}
    # required: every merge in at least two sources (by construction; verified)
    for t in TABLES:
        out["required"][f"{t}: every merge in >= 2 sources"] = all(s >= 2 for s in sup[t])
    # whole training strings of a single source, among the pieces (len >= 4)
    rep = dict(json.loads((OUT / "clusters.json").read_text(encoding="utf-8")))
    groups: Dict[str, List[str]] = defaultdict(list)
    for k, r in rep.items():
        groups[r].append(k)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), min(g)))
    out["reported"]["near-duplicate clusters"] = {
        "clusters": len(groups), "sites": len(rep),
        "size_counts": dict(sorted(Counter(len(g) for g in ordered).items())),
        "largest": [{"size": len(g), "examples": sorted(g)[:5]} for g in ordered[:15]]}
    pieces = {t: {a + b for a, b in new[t].merges if len(a + b) >= 4} for t in TABLES}
    whole_sources = {t: defaultdict(set) for t in TABLES}
    src = Sources()
    largest = {t: {c: Counter() for c in CLASSES} for t in TABLES}
    for cls, url, stream, parts in rows():
        key = src(parts.host_raw)[0]
        key = rep.get(key, key)
        rest, host = field_strings(parts)
        if stream:
            rest = [parts.query] if parts.has_query and parts.query else []
        for t, strings in (("rest", rest), ("host", host)):
            largest[t][cls][key] += len(strings)
            for s in strings:
                if s in pieces[t]:
                    whole_sources[t][s].add(key)
    for t in TABLES:
        single = sorted(s for s in pieces[t] if len(whole_sources[t].get(s, ())) == 1)
        # how many sources shared each such piece's last merge when it was made: a piece is
        # in at least that many sources' strings, whichever single source has it whole
        first_support: Dict[str, int] = {}
        for (a, b), s in zip(new[t].merges, sup[t]):
            first_support.setdefault(a + b, s)
        shared = sorted(first_support[s] for s in single)
        out["reported"][f"{t}: pieces that are a whole string of a single source"] = {
            "count": len(single), "of": len(new[t]), "examples": single[:25],
            "sources sharing each one's merge": {
                "min": shared[0] if shared else None,
                "median": shared[len(shared) // 2] if shared else None,
                "max": shared[-1] if shared else None},
            "examples with their sources": [[s, first_support[s]] for s in single[:25]]}
        out["reported"][f"{t}: the largest sources (strings)"] = {
            c: largest[t][c].most_common(10) for c in CLASSES}
        out["reported"][f"{t}: merges, stop, smallest support"] = {
            "merges": len(new[t]), "budget": BUDGET[t], "stop": build_rec["tables"][t]["stop"],
            "smallest_support": min(sup[t]) if sup[t] else None,
            "support_at_quantiles": {q: sorted(sup[t])[int(q * (len(sup[t]) - 1))] for q in (0.0, 0.1, 0.5, 0.9)}
            if sup[t] else None}
    # heavily drawn strings (R1's sample, weight >= 50) that are a single piece. Registered
    # definition (section 4.7, v2_piece_hazard.py, whose counts section 4.7 quotes): the string
    # is itself one of the table's learned pieces. Beside it: the string encodes as one token
    # (which also counts single characters).
    sample = json.loads((R1 / "bpe_sample.json").read_text(encoding="utf-8"))["full"]
    for t in TABLES:
        heavy = sorted(k for k, v in sample[t].items() if v >= 50)
        pn = {a + b for a, b in new[t].merges}
        po = {a + b for a, b in old[t].merges[:old_limit[t]]}
        out["reported"][f"{t}: heavily drawn strings that are one piece"] = {
            "of": len(heavy), "new": sum(s in pn for s in heavy), "r1": sum(s in po for s in heavy),
            "new_examples": [s for s in heavy if s in pn][:20],
            "encoded as one token (single characters included)": {
                "new": sum(1 for s in heavy if len(new[t].encode(s)) == 1),
                "r1": sum(1 for s in heavy if len(old[t].encode(s, limit=old_limit[t])) == 1)}}
    # coverage of real held-out words by frequency band, and long words
    meta = json.loads((R1 / "meta.json").read_text(encoding="utf-8"))
    cover: dict = {}
    for label in CLASSES:
        vocab = SD.Vocab(label)
        thr = R1M._thresholds(vocab)
        for field in R1M.FIELDS_R1:
            if not meta["measurable"].get(f"{field}|{label}"):
                continue
            t = R1M.GROUP[field]
            acc = {"words": Counter(), "new": Counter(), "r1": Counter(),
                   "cwords": Counter(), "cnew": Counter(), "cr1": Counter(),
                   "long_words": 0, "long_pieces_new": 0, "long_pieces_r1": 0}
            for g in range(R1M.G):
                for value in json.loads((R1M.SAMPLES / f"F_{field}_{label}_g{g}.json").read_text(
                        encoding="utf-8"))["C"]:
                    for seg in (value.split(".") if field == "SUB" else [value]):
                        low = seg.lower()
                        if not seg or len(low) != len(seg):
                            continue
                        spans = [(m.start(), m.end()) for m in R1M.WORD.finditer(low)]
                        if not spans:
                            continue
                        words = [low[x:y] for x, y in spans]
                        seen = [int(c) for c in vocab.counts(words)]
                        bands = [R1M.band_of(c, thr) for c in seen]
                        cbands = [count_band(c) for c in seen]
                        enc_new = new[t].encode(seg)
                        enc_old = old[t].encode(seg, limit=old_limit[t])
                        if "".join(enc_new) != seg:
                            out["required"]["lossless"] = False
                        cov_new = covered_words(seg, enc_new, spans)
                        cov_old = covered_words(seg, enc_old, spans)
                        for bnd, cb, x, y, (s0, s1) in zip(bands, cbands, cov_new, cov_old, spans):
                            acc["words"][bnd] += 1
                            acc["new"][bnd] += int(x)
                            acc["r1"][bnd] += int(y)
                            for key in (cb, "1-99") if cb in ("1-9", "10-99") else (cb,):
                                acc["cwords"][key] += 1
                                acc["cnew"][key] += int(x)
                                acc["cr1"][key] += int(y)
                            if s1 - s0 > 16:
                                acc["long_words"] += 1
                                acc["long_pieces_new"] += _pieces_over(seg, enc_new, s0, s1)
                                acc["long_pieces_r1"] += _pieces_over(seg, enc_old, s0, s1)
            cover[f"{field}|{label}"] = {
                "by_count_band (section 4.7)": {
                    b: {"words": acc["cwords"][b], "held_whole_new": acc["cnew"][b] / acc["cwords"][b],
                        "held_whole_r1": acc["cr1"][b] / acc["cwords"][b]}
                    for b in COUNT_BANDS if acc["cwords"][b]},
                "1-99 band not below R1 (section 4.7's expectation)":
                    acc["cnew"]["1-99"] >= acc["cr1"]["1-99"] if acc["cwords"]["1-99"] else None,
                "by_band": {b: {"words": acc["words"][b],
                                "held_whole_new": acc["new"][b] / acc["words"][b],
                                "held_whole_r1": acc["r1"][b] / acc["words"][b]}
                            for b in R1M.BANDS if acc["words"][b]},
                "long_words_over_16": acc["long_words"],
                "pieces_per_long_word_new": acc["long_pieces_new"] / acc["long_words"] if acc["long_words"] else None,
                "pieces_per_long_word_r1": acc["long_pieces_r1"] / acc["long_words"] if acc["long_words"] else None}
        del vocab
    out["reported"]["coverage of real held-out words (R1's samples), by band"] = cover
    out["required"].setdefault("lossless", True)
    out["pass"] = all(out["required"].values())
    out["seconds"] = round(time.time() - t0, 1)
    (OUT / "audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["required"]), "->", "PASS" if out["pass"] else "FAIL")
    return out


#: section 4.7's bands: a word's count in the training vocabulary (the judge's, R1's)
COUNT_BANDS = ("never seen", "1-9", "10-99", "1-99", "100+")


def count_band(count: int) -> str:
    return "never seen" if count <= 0 else "1-9" if count < 10 else "10-99" if count < 100 else "100+"


def _pieces_over(text: str, pieces: Sequence[str], start: int, end: int) -> int:
    """How many pieces of the encoding overlap the span [start, end)."""
    pos, n = 0, 0
    for piece in pieces:
        a, b = pos, pos + len(piece)
        if a < end and b > start:
            n += 1
        pos = b
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("build", "audit"))
    args = ap.parse_args()
    build() if args.step == "build" else audit()


if __name__ == "__main__":
    main()
