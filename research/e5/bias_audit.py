"""Structural bias audit: every dimension on which generated URLs diverge.

The brief is to find EVERY significant source of divergence before E4 is called
complete, not just the one currently under suspicion. So this sweeps structural
properties rather than testing a single hypothesis, and reports each against the
TRAINING BLEND, which is the distribution free generation targets (benign 35.5%
name / 64.5% structure; phishing 80.6% / 19.4%).

E3 is included so each divergence can be labelled NEW or PRE-EXISTING. A
property that was already wrong before the tokenizer fix is not a regression
from it.

Significance: each row is a mean over n samples, so the reported +- is the
standard error of the difference. Rows are sorted by |gap| / se, because a large
gap on a rare property matters less than a moderate gap on a common one.
"""
from __future__ import annotations

import collections
import gzip
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.factorization import parse

B = str(ROOT / "data/bundle2")
CACHE = ROOT / "data/scratch/gen_cache.json"
cache = json.loads(CACHE.read_text(encoding="utf-8"))

MIX = {"benign": [("name_train", 0.355), ("structure_train", 0.645)],
       "phishing": [("phishing_name_train", 0.806),
                    ("phishing_structure_train", 0.194)]}
LIMIT = 40000


def segs(u):
    return [s for s in (parse(u).path or "").split("/") if s]


def features(u):
    p = parse(u)
    s = segs(u)
    host = ".".join(tuple(p.subdomains) + (p.registrable_label, p.public_suffix))
    body = "/".join(s)
    return {
        "url length": len(u.split("://", 1)[-1]),
        "host length": len(host),
        "label length": len(p.registrable_label or ""),
        "path depth": len(s),
        "path length": len(body),
        "mean segment len": (sum(map(len, s)) / len(s)) if s else 0.0,
        "has path": float(bool(s)),
        "n subdomains": len(p.subdomains),
        "has www": float(bool(p.subdomains) and p.subdomains[0].lower() == "www"),
        "has query": float(p.has_query),
        "n query params": len(p.query.split("&")) if p.has_query and p.query else 0,
        "digits in host": sum(c.isdigit() for c in host),
        "hyphens in host": host.count("-"),
        "digits in path": sum(c.isdigit() for c in body),
        "has extension": float(bool(s) and "." in s[-1]),
        "uppercase chars": sum(c.isupper() for c in u),
    }


KEYS = list(features("http://a.com/x").keys())


def accumulate(urls):
    tot = collections.defaultdict(float)
    sq = collections.defaultdict(float)
    for u in urls:
        f = features(u)
        for k in KEYS:
            tot[k] += f[k]
            sq[k] += f[k] * f[k]
    n = len(urls)
    return ({k: tot[k] / n for k in KEYS},
            {k: max(sq[k] / n - (tot[k] / n) ** 2, 0.0) for k in KEYS}, n)


def corpus_blend(mix):
    means, var, n_eff = {k: 0.0 for k in KEYS}, {k: 0.0 for k in KEYS}, 0
    for stem, w in mix:
        urls = []
        with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                urls.append(json.loads(line)["url"])
                if len(urls) >= LIMIT:
                    break
        m, v, n = accumulate(urls)
        for k in KEYS:
            means[k] += w * m[k]
            var[k] += w * w * v[k] / n
        n_eff += n
    return means, var, n_eff


for label, mix in MIX.items():
    real, real_var, _ = corpus_blend(mix)
    print(f"\n{'='*78}\n=== {label} ===")
    print(f"  {'property':<20}{'real':>10}{'E3':>10}{'E4':>10}"
          f"{'E4 gap':>10}{'sigma':>8}  verdict")

    g4, v4, n4 = accumulate(cache[f"e4/{label}"])
    g3, v3, n3 = accumulate(cache[f"e3/{label}"])

    lines = []
    for k in KEYS:
        se = math.sqrt(real_var[k] + v4[k] / n4) or 1e-9
        gap4 = g4[k] - real[k]
        gap3 = g3[k] - real[k]
        sig = gap4 / se
        # NEW if E4 diverges materially more than E3 did.
        if abs(sig) < 3:
            verdict = "ok"
        elif abs(gap4) > abs(gap3) * 1.3:
            verdict = "DIVERGENT (worse than E3)"
        elif abs(gap4) < abs(gap3) * 0.7:
            verdict = "DIVERGENT (better than E3)"
        else:
            verdict = "DIVERGENT (pre-existing)"
        lines.append((abs(sig), k, real[k], g3[k], g4[k], gap4, sig, verdict))

    for _, k, r, a3, a4, gap, sig, verdict in sorted(lines, reverse=True):
        print(f"  {k:<20}{r:>10.2f}{a3:>10.2f}{a4:>10.2f}"
              f"{gap:>+10.2f}{sig:>8.1f}  {verdict}")

    # Depth distribution, since a matching mean can hide a wrong shape.
    print(f"\n  path depth distribution (share of urls)")
    print(f"  {'depth':<8}{'real':>10}{'E4':>10}")
    realdepth = collections.Counter()
    for stem, w in mix:
        c = collections.Counter()
        with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= LIMIT:
                    break
                c[min(len(segs(json.loads(line)["url"])), 6)] += 1
        tot = sum(c.values())
        for d, v in c.items():
            realdepth[d] += w * v / tot
    gend = collections.Counter(min(len(segs(u)), 6)
                               for u in cache[f"e4/{label}"])
    gtot = sum(gend.values())
    for d in range(7):
        tag = f"{d}" if d < 6 else "6+"
        print(f"  {tag:<8}{realdepth.get(d,0):>10.1%}"
              f"{gend.get(d,0)/gtot:>10.1%}")
