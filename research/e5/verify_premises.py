"""Verify the premises behind the claimed PTH/FULL trade-off, before proposing
any architecture.

P1  What is FULL actually scored against? `mixed` is name_val + structure_val
    concatenated and then deduped to one URL per host. Those have 113,277 and
    7,061 hosts, so the pool may be overwhelmingly NAME -- in which case FULL and
    PTH do not want opposite things and the "trade-off" is an artifact.

P2  Is the path-RATE gap the same compounding-hazard mechanism as the LENGTH gap?
    Signature: the model sits BELOW BOTH components, not between them. A single
    binary decision averaged over a mixture would land between; a survival
    process with an averaged hazard lands below both.

P3  Where does the uppercase deficit live -- host or path? If it is confined to
    the path it shares a cause with the other content deficits (structure-only
    training). If it is in the host too, it is independent, since host fields are
    trained on every name row.
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.evaluation import _one_per_host
from uie.factorization import parse

B = str(ROOT / "data/bundle2")
CACHE = ROOT / "data/scratch/gen_cache.json"
cache = json.loads(CACHE.read_text(encoding="utf-8"))
SEED = 20260806


def rows(stem, limit=None):
    out = []
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line)["url"])
            if limit and len(out) >= limit:
                break
    return out


def host(u):
    p = parse(u)
    return ".".join(tuple(p.subdomains) + (p.registrable_label, p.public_suffix))


def pathbody(u):
    return "/".join(s for s in (parse(u).path or "").split("/") if s)


print("=" * 74)
print("P1  what does the FULL 'mixed' pool actually contain, after host dedup?")
for label, (nm, st) in (("benign", ("name_val", "structure_val")),
                        ("phishing", ("phishing_name_val",
                                      "phishing_structure_val"))):
    name_u, str_u = rows(nm), rows(st)
    name_hosts = {host(u) for u in name_u}
    str_hosts = {host(u) for u in str_u}
    mixed = name_u + str_u
    ded = _one_per_host(mixed, SEED)
    from_name = sum(1 for u in ded if host(u) in name_hosts
                    and host(u) not in str_hosts)
    from_str = sum(1 for u in ded if host(u) in str_hosts
                   and host(u) not in name_hosts)
    both = len(ded) - from_name - from_str
    ln = sum(len(u.split("://", 1)[-1]) for u in ded) / len(ded)
    print(f"  {label:<9} mixed {len(mixed):>8,} rows -> {len(ded):>8,} hosts   "
          f"name-only {from_name/len(ded):>6.1%}  structure-only "
          f"{from_str/len(ded):>6.1%}  both {both/len(ded):>6.1%}")
    print(f"  {'':<9} deduped mean url length {ln:>5.1f}   "
          f"(name_val {sum(len(u.split('://',1)[-1]) for u in name_u)/len(name_u):.1f}, "
          f"structure_val {sum(len(u.split('://',1)[-1]) for u in str_u)/len(str_u):.1f}, "
          f"MODEL {sum(len(u.split('://',1)[-1]) for u in cache[f'e4/{label}'])/len(cache[f'e4/{label}']):.1f})")

print()
print("=" * 74)
print("P2  path RATE: is the model between the components, or below both?")
MIX = {"benign": [("name_train", 0.355), ("structure_train", 0.645)],
       "phishing": [("phishing_name_train", 0.806),
                    ("phishing_structure_train", 0.194)]}
for label, mix in MIX.items():
    parts, blend = [], 0.0
    for stem, w in mix:
        u = rows(stem, 40000)
        r = sum(1 for x in u if pathbody(x)) / len(u)
        parts.append((stem, r))
        blend += w * r
    g = cache[f"e4/{label}"]
    m = sum(1 for x in g if pathbody(x)) / len(g)
    lo, hi = min(r for _, r in parts), max(r for _, r in parts)
    verdict = "BELOW BOTH (survival/hazard)" if m < lo else (
        "between components (single averaged decision)" if m <= hi else "above both")
    print(f"  {label:<9} " + "  ".join(f"{s.split('_')[-2] if '_' in s else s}"
                                       f" {r:.1%}" for s, r in parts)
          + f"   blend {blend:.1%}   MODEL {m:.1%}   -> {verdict}")

print()
print("=" * 74)
print("P3  uppercase: host vs path")
for label, mix in MIX.items():
    def up(urls):
        hu = sum(sum(c.isupper() for c in host(u)) for u in urls) / len(urls)
        pu = sum(sum(c.isupper() for c in pathbody(u)) for u in urls) / len(urls)
        pc = sum(len(pathbody(u)) for u in urls)
        pr = (sum(sum(c.isupper() for c in pathbody(u)) for u in urls) / pc
              if pc else 0)
        return hu, pu, pr
    print(f"  {label}")
    blend = [0.0, 0.0, 0.0]
    for stem, w in mix:
        h, p, r = up(rows(stem, 40000))
        for i, v in enumerate((h, p, r)):
            blend[i] += w * v
        print(f"    {stem:<30} host {h:>6.3f}  path {p:>6.3f}  "
              f"path-rate {r:>6.2%}")
    print(f"    {'BLEND':<30} host {blend[0]:>6.3f}  path {blend[1]:>6.3f}  "
          f"path-rate {blend[2]:>6.2%}")
    h, p, r = up(cache[f"e4/{label}"])
    print(f"    {'MODEL E4':<30} host {h:>6.3f}  path {p:>6.3f}  "
          f"path-rate {r:>6.2%}")
    print(f"    {'ratio model/blend':<30} host "
          f"{(h/blend[0] if blend[0] else 0):>6.2f}  path "
          f"{(p/blend[1] if blend[1] else 0):>6.2f}  path-rate "
          f"{(r/blend[2] if blend[2] else 0):>6.2f}")
