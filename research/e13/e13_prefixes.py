"""E13 C -- build the prefix artifact ContextSwap reads (research/e13/PREREG_E13.md
section 2.1). Laptop CPU; frozen with its sha256 before the training notebook is
built.

    python research/e13/e13_prefixes.py --checkpoint data/e12/train/e12_final/final.pt \
        --bundle data/bundle2 --out data/e13/prefixes.json.gz

For each phishing domain-frame training row with a non-empty query (8,272, the
registered count), e12_final writes k = 4 prefixes under the row's own suffix and
plan, with the registered raw decode (temperature 1.0, the full softmax; no D1):
the model writes everything before the suffix; the suffix, the plan and <LBL> are
fed exactly as `uie.model.generate` feeds them; the model then writes until it
emits <QRY>. A sample that ends (<EOS>, <FRG>), emits <QRY> before the host,
writes <LBL> without the supplied suffix, or reaches 384 tokens first is dropped,
never re-drawn. Rows sharing a suffix and plan are sampled together (the prefix
depends on nothing else), groups in sorted order, from one generator seeded
20261006. The sampler replays the model's own blocks with a key/value cache; the
first chunks are checked against the full forward pass (max |difference| <= 1e-3).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e13_common as E13  # noqa: E402

CHUNK = 64
CHECK_CHUNKS = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_urls(bundle: Path, stem: str) -> List[str]:
    p = Path(bundle) / f"{stem}.jsonl.gz"
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


# --- the model's own blocks, replayed with a cache (DX8's sampler) ---------------------

def prefill(model, ids: torch.Tensor):
    """The forward pass over a batch of equal-length prefixes, keeping each layer's
    keys and values. Returns the last position's logits and the cache."""
    B, t = ids.shape
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(t))
    cache = []
    for blk in model.blocks:
        q, k, v = blk.qkv(blk.ln1(x)).chunk(3, dim=-1)
        sh = (B, t, blk.n_heads, blk.d_head)
        q, k, v = (z.view(sh).transpose(1, 2) for z in (q, k, v))
        cache.append([k, v])
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, -1)
        x = x + blk.proj(att)
        x = x + blk.ff(blk.ln2(x))
    return model.head(model.ln_f(x))[:, -1], cache


def step(model, tok_ids: torch.Tensor, pos: int, cache):
    """One cached step: tok_ids (B,) at position `pos`; returns the next logits."""
    B = tok_ids.shape[0]
    x = model.tok_emb(tok_ids.view(B, 1)) + model.pos_emb(torch.tensor([pos]))
    for blk, kv in zip(model.blocks, cache):
        q, k, v = blk.qkv(blk.ln1(x)).chunk(3, dim=-1)
        sh = (B, 1, blk.n_heads, blk.d_head)
        q, k, v = (z.view(sh).transpose(1, 2) for z in (q, k, v))
        kv[0] = torch.cat([kv[0], k], dim=2)
        kv[1] = torch.cat([kv[1], v], dim=2)
        att = F.scaled_dot_product_attention(q, kv[0], kv[1]).transpose(1, 2).reshape(B, 1, -1)
        x = x + blk.proj(att)
        x = x + blk.ff(blk.ln2(x))
    return model.head(model.ln_f(x))[:, -1]


class Ids:
    def __init__(self, tok):
        self.bos, self.eos, self.pad = tok.bos_id, tok.eos_id, tok.pad_id
        self.suf, self.lbl = tok.marker_id("SUF"), tok.marker_id("LBL")
        self.qry, self.frg = tok.marker_id("QRY"), tok.marker_id("FRG")
        self.cls = tok.stoi["<C:phishing>"]
        self.frame = tok.stoi["<F:domain>"]


def sample_chunk(model, ids: Ids, forced: List[int], n: int, gen: torch.Generator,
                 max_len: int, check: bool) -> Tuple[List[Tuple[str, List[int]]], float]:
    """n prefixes for one (suffix, plan): [(status, ids)], status 'ok' or a reason."""
    rows = [[ids.bos, ids.cls, ids.frame] for _ in range(n)]
    queued: List[List[int]] = [[] for _ in range(n)]
    injected, lbl_seen = [False] * n, [False] * n
    status: List[str] = [""] * n
    done: List[List[int]] = [[] for _ in range(n)]       # an 'ok' prefix, kept when made
    with torch.no_grad():
        logits, cache = prefill(model, torch.tensor(rows))
        worst = 0.0
        pos = len(rows[0])
        while pos < max_len and not all(status):
            if check:
                full = model(torch.tensor(rows))[:, -1].float()
                worst = max(worst, float((full - logits.float()).abs().max()))
            nxt = []
            for r in range(n):
                if status[r]:                    # finished: padded, so rows stay aligned
                    rows[r].append(ids.pad)
                    nxt.append(ids.pad)
                    continue
                if queued[r]:
                    t = queued[r].pop(0)
                    if t == ids.lbl:
                        lbl_seen[r] = True
                else:
                    probs = F.softmax(logits[r].float(), dim=-1)
                    t = int(torch.multinomial(probs, 1, generator=gen))
                    if not injected[r] and t == ids.suf:
                        injected[r] = True
                        queued[r] = list(forced[1:])
                    elif t == ids.qry:
                        status[r] = "ok" if lbl_seen[r] else "early_query"
                    elif t in (ids.eos, ids.frg):
                        status[r] = "ended"
                    elif t == ids.lbl and not injected[r]:
                        status[r] = "no_suffix"
                rows[r].append(t)
                nxt.append(t)
                if status[r] == "ok":
                    done[r] = list(rows[r])
                if not status[r] and len(rows[r]) >= max_len:
                    status[r] = "max_len"
            if all(status):
                break
            logits = step(model, torch.tensor(nxt), pos, cache)
            pos += 1
    out = [(s or "max_len", done[r]) for r, s in enumerate(status)]
    return out, worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "e13" / E13.PREFIX_ARTIFACT)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only: the first N rows")
    args = ap.parse_args()
    from uie.evaluation import FIELDS
    from uie.train import load_checkpoint
    t0 = time.time()
    model, tok, cfg = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    model.eval()
    assert not model._struct and not getattr(cfg, "frame_every", False), "the plain model only"
    assert tok.presence_plan and tok.root_slash and tok.slash_plan and tok.include_frame
    I = Ids(tok)
    urls = [u for u in read_urls(args.bundle, "phishing_name_train") if FIELDS["QRY"](u)]
    if not args.limit and len(urls) != E13.ELIGIBLE_ROWS:
        raise SystemExit(f"STOP (V): {len(urls)} eligible rows, registered {E13.ELIGIBLE_ROWS}")
    if args.limit:
        urls = urls[:args.limit]
    rows, dropped = [], 0
    groups: Dict[tuple, List[int]] = collections.defaultdict(list)
    for u in urls:
        ids = tok.encode(u, "phishing", frame="domain")
        if len(ids) > cfg.max_len:
            dropped += 1
            continue
        j, k = ids.index(I.suf), ids.index(I.lbl)
        suffix, plan = tuple(ids[j + 1:k - 3]), tuple(ids[k - 3:k])
        groups[(suffix, plan)].append(len(rows))
        rows.append({"url": u, "ids": ids, "prefixes": []})
    gen = torch.Generator().manual_seed(E13.PREFIX_SEED)
    reasons = collections.Counter()
    worst, checked = 0.0, 0
    keys = sorted(groups)
    print(f"{len(rows):,} rows ({dropped} over {cfg.max_len} tokens) in {len(keys)} "
          f"(suffix, plan) groups; k = {E13.PREFIX_K}", flush=True)
    for gi, key in enumerate(keys):
        suffix, plan = key
        forced = [I.suf, *suffix, *plan, I.lbl]
        members = groups[key]
        need = E13.PREFIX_K * len(members)
        samples: List[Tuple[str, List[int]]] = []
        for c0 in range(0, need, CHUNK):
            got, w = sample_chunk(model, I, forced, min(CHUNK, need - c0), gen, cfg.max_len,
                                  check=checked < CHECK_CHUNKS)
            if checked < CHECK_CHUNKS:
                worst = max(worst, w)
                checked += 1
            samples.extend(got)
        for r, idx in enumerate(members):
            for st, pre in samples[r * E13.PREFIX_K:(r + 1) * E13.PREFIX_K]:
                reasons[st] += 1
                if st == "ok":
                    rows[idx]["prefixes"].append(pre)
        if gi % 25 == 0:
            print(f"  group {gi + 1}/{len(keys)}  ({time.time() - t0:.0f}s)", flush=True)
    if worst > 1e-3:
        raise SystemExit(f"STOP (V): the cached sampler differs from the full pass by {worst}")
    # descriptive: the written paths against the training paths (PREREG_E13 2.1)
    train_paths = {FIELDS["PTH"](u) for u in read_urls(args.bundle, "phishing_name_train")}
    train_paths.discard("")
    written = [tok.decode(p) for r in rows for p in r["prefixes"]]
    with_path = [FIELDS["PTH"](w) for w in written if FIELDS["PTH"](w)]
    per_row = collections.Counter(len(r["prefixes"]) for r in rows)
    stats = {"rows": len(rows), "dropped_over_max_len": dropped, "groups": len(keys),
             "samples": sum(reasons.values()), "reasons": dict(reasons),
             "valid_per_row": {str(k): v for k, v in sorted(per_row.items())},
             "rows_never_swappable": per_row.get(0, 0),
             "cache_check_max_abs_diff": worst, "cache_check_chunks": checked,
             "written_paths": len(with_path),
             "written_paths_exact_in_train": (sum(p in train_paths for p in with_path) / len(with_path)
                                              if with_path else None),
             "seconds": round(time.time() - t0, 1)}
    blob = {"registration": E13.REGISTRATION, "purpose": "E13 C prefix artifact (section 2.1)",
            "checkpoint_sha256": sha256(args.checkpoint), "k": E13.PREFIX_K,
            "seed": E13.PREFIX_SEED, "decode": "raw (temperature 1.0, full softmax); no D1",
            "limit": args.limit or None, "tokenizer": tok.state_dict(), "stats": stats,
            "rows": [{"url": r["url"], "ids": r["ids"], "prefixes": r["prefixes"]} for r in rows]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(args.out, "wb", mtime=0) as fh:
        fh.write(json.dumps(blob, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    print(json.dumps(stats, indent=1))
    print(f"wrote {args.out}  sha256 {sha256(args.out)}")


if __name__ == "__main__":
    main()
