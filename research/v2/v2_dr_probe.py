"""
research/v2/v2_dr_probe.py

The owner's draft-and-revise idea, probed on the frozen engine: a post-hoc investigation for
the owner's request of 2026-10-13. It decides nothing. Inference only, on the CPU, with
e16_cap (e16_b5's weights) as the drafter.

    D  the detector: can the drafter's own uncertainty over a word's letters tell its
       never-seen words (fakes) from the real rare words (training count 1-99) it should
       write more of? Words from e16_b5's free files (g0), scored where they were written.
    K  the reviser's knowledge:
       - for each slip (a fake one edit from a word seen 10+ times), does the drafter
         itself prefer the URL with the slip corrected?
       - for real rare words in held-out URLs with a far more common one-edit neighbour,
         does it prefer the neighbour? That is the damage a mode-seeking reviser would do.
    B  how well the drafter knows each frequency band: mean per-letter surprisal of words in
       held-out real URLs, by training count.
    C  the vocabulary by band: training word types and occurrences, for the capacity estimate.

Free generation conditions every URL on the domain frame, so both sources are scored with
`tokenizer.encode(url, label)` (frame "domain"), as training's own field_nll does. Words are
E18's ([a-z]{3,}); counts are E18's training vocabulary; slips use R1's rule (v2_r1.anatomy).

    python research/v2/v2_dr_probe.py               # -> data/v2/dr_probe/probe.json
"""
from __future__ import annotations

import gzip
import json
import random
import statistics as stats
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e18", ROOT / "research" / "native", ROOT / "research" / "cr1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import strong_disc as SD  # noqa: E402  (E18's vocabulary and word definition)
import v2_r1 as R1  # noqa: E402  (R1's slip rule: anatomy, edit1)

OUT = ROOT / "data" / "v2" / "dr_probe"
E16_CAP = ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt"
GEN = ROOT / "data" / "e17" / "decode" / "gen" / "free_e16_b5_{label}_g0.json"
REAL = {"benign": "name_val", "phishing": "phishing_name_val"}
LABELS = ("benign", "phishing")
N_URLS = 1_500
SEED = 20261060
FIELDS = ("LBL", "SUB", "PTH", "QRY")
BANDS = (("never", 0, 0), ("1-9", 1, 9), ("10-99", 10, 99), ("100-999", 100, 999), ("1000+", 1000, None))
RARE = (1, 99)                      # R2: real words that are rare but recur
COMMON_FACTOR = 10                  # K: a neighbour "far more common" = 10x the word's count, and >= 10
BATCH = 64
SCORES = ("entropy", "surprisal", "max_entropy", "first_entropy")


def band(c: int) -> str:
    for name, lo, hi in BANDS:
        if c >= lo and (hi is None or c <= hi):
            return name
    raise ValueError(c)


def has_rest(url: str) -> bool:
    host_end = len(url.split("/", 1)[0].split("?", 1)[0])
    return len(url) > host_end + 1


def real_urls(label: str) -> list:
    with gzip.open(ROOT / "data" / "corpus2" / f"{REAL[label]}.jsonl.gz", "rt", encoding="utf-8") as fh:
        urls = [json.loads(line)["url"] for line in fh]
    return [u for u in urls if has_rest(u)]


def gen_urls(label: str) -> list:
    urls = json.loads(Path(str(GEN).format(label=label)).read_text(encoding="utf-8"))
    return [u for u in urls if has_rest(u)]


def score(model, tok, urls: list, label: str) -> list:
    """Per URL: the total log-probability, and per field its letters as (letter, surprisal,
    entropy) at the position where the drafter wrote it."""
    import torch
    import torch.nn.functional as F
    from uie.tokenizer import _MARKERS
    marker_of = {tok.marker_id(m): m for m in _MARKERS}
    chars = {tok.stoi[c] for c in tok._chars}
    enc = [tok.encode(u, label)[:model.cfg.max_len] for u in urls]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    res = [None] * len(enc)
    model.eval()
    with torch.no_grad():
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            seqs = [enc[i] for i in idx]
            width = max(len(s) for s in seqs)
            ids = torch.tensor([s + [tok.pad_id] * (width - len(s)) for s in seqs])
            logp = F.log_softmax(model(ids[:, :-1]).float(), dim=-1)
            ent = -(logp.exp() * logp).sum(-1)
            got = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            for r, i in enumerate(idx):
                s, total, fields, cur = seqs[r], 0.0, {}, None
                for t in range(len(s) - 1):
                    target, lp = s[t + 1], float(got[r, t])
                    total += lp
                    if target in marker_of:
                        cur = marker_of[target]
                    elif target in chars and cur is not None:
                        fields.setdefault(cur, []).append((tok.itos[target], -lp, float(ent[r, t])))
                res[i] = {"logp": total, "fields": fields}
    return res


def words_of(rec: dict) -> list:
    out = []
    for f, letters in rec["fields"].items():
        if f not in FIELDS:
            continue
        text = "".join(c.lower() if len(c.lower()) == 1 else "?" for c, _, _ in letters)
        for m in SD.TOKEN.finditer(text):
            seg = letters[m.start():m.end()]
            s, h = [x[1] for x in seg], [x[2] for x in seg]
            out.append({"field": f, "word": m.group(), "start": m.start(), "end": m.end(),
                        "surprisal": sum(s) / len(s), "entropy": sum(h) / len(h),
                        "max_entropy": max(h), "first_entropy": h[0]})
    return out


def auc(pos: list, neg: list) -> float:
    """P(a positive scores above a negative), ties half (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    allv = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks over ties
    vals, inv = np.unique(allv, return_inverse=True)
    sums = np.bincount(inv, weights=ranks)
    cnts = np.bincount(inv)
    ranks = (sums / cnts)[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def flag_table(fakes: list, rares: list, reals: list) -> list:
    """Flag every word scoring above a threshold set to catch 25/50/75/90% of the fakes:
    the share of real rare words and of all real words flagged with them."""
    out = []
    for q in (0.25, 0.50, 0.75, 0.90):
        thr = float(np.quantile(fakes, 1 - q))
        out.append({"fakes_flagged": q, "threshold": thr,
                    "rare_real_flagged": float(np.mean(np.asarray(rares) >= thr)) if rares else None,
                    "all_real_flagged": float(np.mean(np.asarray(reals) >= thr)) if reals else None})
    return out


def replace_word(url: str, w: dict, new: str) -> str:
    """The URL with word `w` (a span in its field) replaced: host fields in the host, the
    rest in the rest. Spans index the field text, which is the URL text of that field."""
    host_len = len(url.split("/", 1)[0].split("?", 1)[0])
    host, rest = url[:host_len], url[host_len:]
    part = host if w["field"] in ("LBL", "SUB") else rest
    low = "".join(c.lower() if len(c.lower()) == 1 else "?" for c in part)
    hit = next((m for m in SD.TOKEN.finditer(low) if m.group() == w["word"]), None)
    if hit is None:
        return ""
    part = part[:hit.start()] + new + part[hit.end():]
    return part + rest if w["field"] in ("LBL", "SUB") else host + part


def main() -> None:
    from uie.train import load_checkpoint
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok, _ = load_checkpoint(E16_CAP)
    out = {"post_hoc": True, "decides": "nothing", "n_urls": N_URLS, "seed": SEED, "classes": {}}
    for li, label in enumerate(LABELS):
        vocab = SD.Vocab(label)

        def counts_of(words):
            return vocab.counts(list(words))

        node = {}
        # C: the vocabulary by band
        c = vocab.c
        node["vocab"] = {name: {"types": int(((c >= lo) & (c <= (hi if hi is not None else c.max()))).sum()),
                                "occurrences": int(c[(c >= lo) & (c <= (hi if hi is not None else c.max()))].sum())}
                         for name, lo, hi in BANDS[1:]}
        rng = random.Random(SEED + li)
        g_urls = rng.sample(gen_urls(label), N_URLS)
        r_urls = rng.sample(real_urls(label), N_URLS)
        g_rec, r_rec = score(model, tok, g_urls, label), score(model, tok, r_urls, label)
        g_words = [(i, w) for i, r in enumerate(g_rec) for w in words_of(r)]
        r_words = [(i, w) for i, r in enumerate(r_rec) for w in words_of(r)]
        for words in (g_words, r_words):
            cnt = counts_of([w["word"] for _, w in words])
            for (_, w), k in zip(words, cnt):
                w["count"] = int(k)
                w["band"] = band(int(k))
        print(f"  {label}: scored {len(g_urls):,} generated and {len(r_urls):,} real URLs; "
              f"{len(g_words):,} and {len(r_words):,} words ({time.time() - t0:.0f}s)", flush=True)

        # B: per-letter surprisal and entropy by band, real held-out words (and generated)
        node["bands"] = {}
        for src, words in (("real", r_words), ("generated", g_words)):
            for f in FIELDS:
                for name, _, _ in BANDS:
                    sel = [w for _, w in words if w["field"] == f and w["band"] == name]
                    if len(sel) >= 30:
                        node["bands"][f"{src} {f} {name}"] = {
                            "n": len(sel), "surprisal": stats.mean(w["surprisal"] for w in sel),
                            "entropy": stats.mean(w["entropy"] for w in sel),
                            "share": len(sel) / max(1, sum(1 for _, w in words if w["field"] == f))}

        # D: the detector, on the drafter's own words
        node["detector"] = {}
        for f in FIELDS:
            sel = [w for _, w in g_words if w["field"] == f]
            fakes = [w for w in sel if w["count"] == 0]
            rares = [w for w in sel if RARE[0] <= w["count"] <= RARE[1]]
            reals = [w for w in sel if w["count"] > 0]
            if len(fakes) < 30 or len(rares) < 30:
                continue
            node["detector"][f] = {
                "n": {"fake": len(fakes), "rare_real": len(rares), "real": len(reals)},
                "auc_fake_vs_rare_real": {s: auc([w[s] for w in fakes], [w[s] for w in rares]) for s in SCORES},
                "auc_fake_vs_real": {s: auc([w[s] for w in fakes], [w[s] for w in reals]) for s in SCORES},
                "flags_by_entropy": flag_table([w["entropy"] for w in fakes], [w["entropy"] for w in rares],
                                               [w["entropy"] for w in reals]),
                "flags_by_surprisal": flag_table([w["surprisal"] for w in fakes], [w["surprisal"] for w in rares],
                                                 [w["surprisal"] for w in reals])}

        # K: the reviser's knowledge
        kinds = {}
        slips, edits = [], []
        for i, w in g_words:
            if w["count"] != 0:
                continue
            if w["word"] not in kinds:
                kinds[w["word"]] = R1.anatomy(w["word"], counts_of)
            if kinds[w["word"]] != "near":
                continue
            variants = sorted(R1.edit1(w["word"]))
            vc = counts_of(variants)
            fix = variants[int(np.argmax(vc))]
            new = replace_word(g_urls[i], w, fix)
            if new:
                slips.append((i, w, fix, int(vc.max())))
                edits.append(new)
        common = []
        for i, w in r_words:
            if not (RARE[0] <= w["count"] <= RARE[1]) or w["field"] not in ("PTH", "QRY", "LBL"):
                continue
            variants = sorted(R1.edit1(w["word"]))
            vc = counts_of(variants)
            if int(vc.max()) >= max(10, COMMON_FACTOR * w["count"]):
                nb = variants[int(np.argmax(vc))]
                new = replace_word(r_urls[i], w, nb)
                if new:
                    common.append((i, w, nb, int(vc.max())))
                    edits.append(new)
        alt = score(model, tok, edits, label) if edits else []
        k_slip, k_rare = [], []
        for j, (i, w, fix, n) in enumerate(slips):
            k_slip.append({"field": w["field"], "slip": w["word"], "fix": fix, "fix_count": n,
                           "delta": alt[j]["logp"] - g_rec[i]["logp"]})
        for j, (i, w, nb, n) in enumerate(common):
            k_rare.append({"field": w["field"], "real": w["word"], "count": w["count"], "neighbour": nb,
                           "neighbour_count": n, "delta": alt[len(slips) + j]["logp"] - r_rec[i]["logp"]})

        def summary(rows):
            if not rows:
                return None
            d = [r["delta"] for r in rows]
            return {"n": len(rows), "prefers_alternative": float(np.mean(np.asarray(d) > 0)),
                    "median_delta_nats": float(np.median(d)),
                    "share_by_10x": float(np.mean(np.asarray(d) > np.log(10)))}
        node["knowledge"] = {
            "slips": summary(k_slip),
            "slips_by_field": {f: summary([r for r in k_slip if r["field"] == f]) for f in FIELDS},
            "real_rare_vs_common_neighbour": summary(k_rare),
            "real_rare_by_field": {f: summary([r for r in k_rare if r["field"] == f]) for f in ("PTH", "QRY", "LBL")},
            "examples_slips": k_slip[:25], "examples_real_rare": k_rare[:25]}
        out["classes"][label] = node
        print(f"  {label}: done ({time.time() - t0:.0f}s)", flush=True)
        del vocab
    out["seconds"] = round(time.time() - t0, 1)
    (OUT / "probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'probe.json'} ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
