"""E12 draft sizing (E12_ROADMAP.md section 2.3) -- REAL DATA ONLY; no model output
is read, so no reading is taken.

1. The natural copy rate: the share of the gate's free-mode real queries (name_val,
   one URL per host, the gate's seed) found word for word among the class's
   training queries -- `dc1_compare.training_queries`, the definition K11d's
   copying uses. A generator sampling the real distribution copies at this rate.
2. The shared strings: which real queries recur, and how long shared and never-seen
   queries are.
3. The length budget: the share of real query URLs whose encoding with E11's
   tokenizer exceeds max_len, i.e. URLs the generator can never write -- every
   name_train row with a query, and the gate's pool.
4. The copying checks (DECISION_2026-09-30.md section 4):
   - the natural rate on the pool's fully domain-disjoint subset (pool URLs whose
     registrable domain occurs in none of the class's training URLs). Benign is
     split by registrable domain, phishing by host, so the two relationships
     differ;
   - what the natural matches are: each matched string's frequency among the
     training queries (1, 2-4, 5-19, 20+), as a share of the pool.

    python research/native/e12_sizing.py        # writes data/e12/sizing.json
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "dc1"))

B = ROOT / "data" / "bundle2"
OUT = ROOT / "data" / "e12" / "sizing.json"
CHECKPOINT = ROOT / "data" / "e11" / "train" / "e11_plan" / "final.pt"   # its tokenizer only
SEED = 20260806                                                          # the gate's seed


def rows(name: str) -> list:
    with gzip.open(B / f"{name}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


def registrable(url: str) -> str:
    from uie.factorization import parse
    p = parse(url)
    return f"{p.registrable_label}.{p.public_suffix}" if p.registrable_label else ""


def band(count: int) -> str:
    return "1" if count == 1 else "2-4" if count < 5 else "5-19" if count < 20 else "20+"


def main() -> None:
    import torch
    from uie.evaluation import FIELDS, _one_per_host
    from uie.train import load_checkpoint
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(1)
    model, tok, _ = load_checkpoint(CHECKPOINT, device=torch.device("cpu"))
    max_len = model.cfg.max_len
    del model
    out = {"checkpoint_tokenizer": str(CHECKPOINT.relative_to(ROOT)), "max_len": max_len,
           "seed": SEED, "classes": {}}
    for label in ("benign", "phishing"):
        pre = "" if label == "benign" else f"{label}_"
        known = collections.Counter()
        domains = set()
        for kind in ("name", "structure"):
            for u in rows(f"{pre}{kind}_train"):
                domains.add(registrable(u))
                q = FIELDS["QRY"](u)
                if q:
                    known[q] += 1
        train = [u for u in rows(f"{pre}name_train") if FIELDS["QRY"](u)]
        over_train = sum(len(tok.encode(u, label, "und", frame="domain")) > max_len for u in train)
        pool = [u for u in _one_per_host(rows(f"{pre}name_val"), SEED) if FIELDS["QRY"](u)]
        over_pool = sum(len(tok.encode(u, label, "und", frame="domain")) > max_len for u in pool)
        queries = [FIELDS["QRY"](u) for u in pool]
        shared = collections.Counter(q for q in queries if q in known)
        n_shared = sum(shared.values())
        ls = sorted(len(q) for q in queries if q in known)
        ln = sorted(len(q) for q in queries if q not in known)
        disjoint = [q for u, q in zip(pool, queries) if registrable(u) not in domains]
        bands = collections.Counter(band(known[q]) for q in queries if q in known)
        out["classes"][label] = {
            "pool_query_urls": len(pool),
            "natural_copy_rate": n_shared / len(pool),
            "domain_disjoint_pool": len(disjoint),
            "domain_disjoint_copy_rate": sum(q in known for q in disjoint) / max(1, len(disjoint)),
            "match_band_share_of_pool": {k: bands[k] / len(pool) for k in ("1", "2-4", "5-19", "20+")},
            "shared": n_shared, "distinct_shared": len(shared),
            "median_len_shared": ls[len(ls) // 2], "median_len_never_seen": ln[len(ln) // 2],
            "top_shared": [[q, c, known[q]] for q, c in shared.most_common(15)],
            "train_query_rows": len(train), "train_over_max_len": over_train,
            "train_over_max_len_share": over_train / len(train),
            "pool_over_max_len": over_pool, "pool_over_max_len_share": over_pool / len(pool)}
        r = out["classes"][label]
        print(f"{label:<9} natural copy {r['natural_copy_rate']:.4f} ({n_shared} of {len(pool)}); "
              f"domain-disjoint {r['domain_disjoint_copy_rate']:.4f} ({len(disjoint)} URLs); "
              f"match bands {', '.join(f'{k}: {v:.4f}' for k, v in r['match_band_share_of_pool'].items())}; "
              f"median length shared {r['median_len_shared']} / never seen {r['median_len_never_seen']}; "
              f"over {max_len}: train {r['train_over_max_len_share']:.4f}, "
              f"pool {r['pool_over_max_len_share']:.4f}", flush=True)
        del known, domains
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
