"""E13-D -- the measurements (research/e13/PREREG_E13D.md sections 2 and 4). Laptop
only; reads E12's registered files, the e12_final checkpoint and the bundle; writes
data/e13d/*.json.

    python research/e13/e13d_measure.py --part validity
    python research/e13/e13d_measure.py --part decomp --cell free_phishing_QRY    # DX1, DX2, V2
    python research/e13/e13d_measure.py --part quantity --cell free_phishing_QRY  # DX3
    python research/e13/e13d_measure.py --part probe                              # DX5, V5
    python research/e13/e13d_measure.py --part describe                           # DX6

--dry runs any part at toy size (2 seeds, n 300, 300 probe rows) into
data/e13d/dry/, stamped "dry": true. The reader refuses dry files. Prints counts and
timings only, never a reading.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e13d_common as D  # noqa: E402

C = D.C
DRY = {"seeds": D.SEEDS[:2], "n_req": 300, "probe": 300}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample(xs, n, seed):
    return random.Random(seed).sample(list(xs), n)


# --------------------------------------------------------------------------- #
# Validity: V1 inputs, V3 the pools (V2 is in decomp, V5 in probe)
# --------------------------------------------------------------------------- #

def part_validity(args) -> dict:
    import e12_common as K
    from uie.factorization import parse
    verify = json.loads((D.E12 / "VERIFY.json").read_text(encoding="utf-8"))
    files = sorted({D.E12 / "gen" / (f"free_{a}_{c[2]}_g{g}.json" if c[1] == "free"
                                     else f"ctrl_{a}_{c[2]}_{c[3]}_g{g}.json")
                    for a in (D.PRIMARY, D.REPLICATION) for c in D.CELLS + (D.REFERENCE,)
                    for g in range(K.G)})
    bundle = sorted(C.BUNDLE / f"{p}name_{s}.jsonl.gz" for p in ("", "phishing_") for s in ("train", "val"))
    ck = D.E12 / "train" / D.PRIMARY / "final.pt"
    out = {"v1": {"e12_verify": verify.get("verdict"),
                  "hashes": {str(p.relative_to(D.ROOT)): sha256(p) for p in files + bundle + [ck]}},
           "v3": {}}
    for cell in D.CELLS + (D.REFERENCE,):
        node = {}
        for split in ("val", "train"):
            from uie.evaluation import _one_per_host
            urls = _one_per_host(D._urls(cell[2], split, cell[3]), C.POOL_SEED)
            hosts = {parse(u).host_raw for u in urls}
            node[split] = {"urls_one_per_host": len(urls), "distinct_hosts": len(hosts),
                           "field_strings": len(D.pool(cell, split))}
        out["v3"][cell[0]] = node
        print(f"  V3 {cell[0]:<28} val {node['val']['field_strings']:>7,}  train {node['train']['field_strings']:>7,}",
              flush=True)
    return out


# --------------------------------------------------------------------------- #
# DX1 + DX2 (+ V2): one harness against validation, one against the training twin
# --------------------------------------------------------------------------- #

def part_decomp(args) -> dict:
    import e12_common as K
    seeds = DRY["seeds"] if args.dry else D.SEEDS
    n_req = DRY["n_req"] if args.dry else D.N_REQ
    cell = D.BY_NAME[args.cell]
    val, train = D.pool(cell, "val"), D.pool(cell, "train")
    gen = {a: D.generated(cell, a) for a in (D.PRIMARY, D.REPLICATION)}
    n = min(n_req, len(val) // 3, len(train) // 3, *(len(g) for g in gen.values()))
    n_gate = min(1200, len(val) // 3, round(len(gen[D.PRIMARY]) / K.G))
    out = {"cell": cell[0], "n": n, "n_gate": n_gate, "seeds": list(seeds),
           "sizes": {"val": len(val), "train": len(train), **{a: len(g) for a, g in gen.items()}},
           "against_val": [], "against_train": [], "v2": []}
    for s in seeds:
        t0 = time.time()
        out["against_val"].append(C.harness(val, {
            D.PRIMARY: sample(gen[D.PRIMARY], n, s + 101), D.REPLICATION: sample(gen[D.REPLICATION], n, s + 202),
            "train": sample(train, n, s + 303)}, s, n))
        out["against_train"].append(C.harness(train, {
            D.PRIMARY: sample(gen[D.PRIMARY], n, s + 404), D.REPLICATION: sample(gen[D.REPLICATION], n, s + 505)},
            s, n))
        out["v2"].append(C.harness(val, {"x": sample(gen[D.PRIMARY], n_gate, s + 707)}, s, n_gate)["x"])
        print(f"  decomp {cell[0]} seed {s} ({time.time() - t0:.0f}s)", flush=True)
    return out


# --------------------------------------------------------------------------- #
# DX3: template coverage and the n-gram learner's curve
# --------------------------------------------------------------------------- #

def part_quantity(args) -> dict:
    seeds = DRY["seeds"] if args.dry else D.SEEDS
    n_req = DRY["n_req"] if args.dry else D.N_REQ
    cell = D.BY_NAME[args.cell]
    assert cell[4] == "QRY", "DX3 is for query cells"
    val, train = D.pool(cell, "val"), D.pool(cell, "train")
    vt = [D.template(q) for q in val]
    tt = [D.template(q) for q in train]
    cov = {}
    for f in D.FRACTIONS:
        vals = []
        for s in (seeds if f < 1 else seeds[:1]):
            have = set(D.subsample(tt, f, s + 11))
            vals.append(sum(t in have for t in vt) / len(vt))
        cov[str(f)] = sum(vals) / len(vals)
    counts = collections.Counter(tt)
    n = min(n_req, len(val) // 3)
    out = {"cell": cell[0], "n": n, "sizes": {"val": len(val), "train": len(train)},
           "coverage": cov, "templates": {"train_distinct": len(counts),
                                          "train_singleton_share": sum(c == 1 for c in counts.values()) / len(tt),
                                          "val_unseen_share": 1 - cov["1.0"]},
           "seeds": list(seeds), "draws": []}
    full = D.CharNGram().fit(train)               # f = 1 is the same data for every seed
    for s in seeds:
        t0 = time.time()
        fakes = {}
        for f in D.FRACTIONS:
            model = full if f >= 1 else D.CharNGram().fit(D.subsample(train, f, s + 13))
            fakes[str(f)] = model.sample(n, random.Random(s + 17))
            if model is not full:
                del model
        out["draws"].append(C.harness(val, fakes, s, n))
        print(f"  quantity {cell[0]} seed {s} ({time.time() - t0:.0f}s)", flush=True)
    return out


# --------------------------------------------------------------------------- #
# DX5 (+ V5): the teacher-forced obedience probe on e12_final
# --------------------------------------------------------------------------- #

def _host_end(seq, tok):
    ends = {tok.marker_id("PTH"), tok.rs_id, tok.marker_id("QRY"), tok.marker_id("FRG"), tok.eos_id}
    lbl = tok.marker_id("LBL")
    if lbl not in seq:
        return None
    return next((j for j in range(seq.index(lbl) + 1, len(seq)) if seq[j] in ends), None)


def _violation_probs(model, tok, rows, s_val):
    """The probability, renormalized over the five host ends, of an end that
    violates the supplied S, at each row's host end. rows: (seq, k)."""
    import torch
    import torch.nn.functional as F
    cols = [tok.marker_id("PTH"), tok.rs_id, tok.marker_id("QRY"), tok.marker_id("FRG"), tok.eos_id]
    out = []
    with torch.no_grad():
        for b in range(0, len(rows), 256):
            chunk = rows[b:b + 256]
            width = max(k for _, k in chunk)
            x = torch.tensor([sq[:width] + [tok.pad_id] * (width - len(sq[:width])) for sq, _ in chunk])
            logits = model(x).float()
            at = torch.tensor([k - 1 for _, k in chunk])
            p = F.softmax(logits[torch.arange(len(chunk)), at], dim=-1)[:, cols].double()
            for pr in p.tolist():
                bad = (pr[2] + pr[3] + pr[4]) if s_val else (pr[0] + pr[1])
                out.append(bad / sum(pr))
    return out


def _gen_population(label, mode, state, s_val):
    """e12_final's generated URLs whose SUPPLIED state is `state` with S = s_val:
    each URL's state rebuilt from the seeds (e12_score.state_positions), exactly as
    E12's S honouring was computed."""
    import e12_common as K
    from e12_score import state_positions
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    ps = json.loads((D.E12 / "presence_slash_table.json").read_text(encoding="utf-8"))["table"]
    key = f"{state}S{s_val}"
    out = []
    for g in range(K.G):
        if mode == "control":
            urls = json.loads((D.E12 / "gen" / f"ctrl_{D.PRIMARY}_{label}_{state}_g{g}.json").read_text(encoding="utf-8"))
            tables = K.control_tables(ps, StateSuffixTables.load(D.ROOT / "data" / "state_suffix_tables.json"),
                                      label, state)
            at = state_positions(tables, len(urls), K.seed_ctrl(label, state, g))
        else:
            urls = json.loads((D.E12 / "gen" / f"free_{D.PRIMARY}_{label}_g{g}.json").read_text(encoding="utf-8"))
            shipped = SuffixTable.load(C.BUNDLE / "suffix_table.json")
            at = state_positions(K.supplied_tables(ps, shipped.counts, label), len(urls), K.seed_free(g))
        out += [u for u, k in zip(urls, at) if k == key]
    return out


def part_probe(args) -> dict:
    import torch
    from uie.factorization import parse
    from uie.train import load_checkpoint
    torch.set_num_threads(4)
    want = DRY["probe"] if args.dry else D.PROBE_MAX
    model, tok, cfg = load_checkpoint(D.E12 / "train" / D.PRIMARY / "final.pt", device=torch.device("cpu"))
    model.eval()
    assert getattr(tok, "slash_plan", False), "the probe needs the slash-plan tokenizer"
    table = json.loads((D.E12 / "presence_slash_table.json").read_text(encoding="utf-8"))
    states = table["states"]
    lbl_id = tok.marker_id("LBL")
    plan_tokens = {i for pair in tok.plan_slots() for i in pair}
    ok_ends = {tok.marker_id("PTH"), tok.rs_id}

    def support_of(label, url, state, s_val):
        row = table["table"][label].get(parse(url).public_suffix)
        k = f"{state}S{s_val}"
        return row[states.index(k)] if row and k in states else 0

    out = {"seed": D.PROBE_SEED, "populations": {}, "generated": {}}
    # amendment 1: the model's own URLs, each fed the plan it was generated under
    for name, (label, mode, state, s_val) in D.GEN_PROBES.items():
        urls = _gen_population(label, mode, state, s_val)
        if args.dry:
            urls = random.Random(D.PROBE_SEED).sample(urls, min(want, len(urls)))
        rows, observed, support = [], [], []
        plan = tok.plan_ids({"PTH": state[1] == "1", "QRY": state[3] == "1", "SLASH": bool(s_val)})
        for u in urls:
            seq = tok.encode(u, label, "und", frame="domain")
            if lbl_id not in seq or len(seq) > cfg.max_len:
                continue
            j = seq.index(lbl_id)
            assert all(t in plan_tokens for t in seq[j - 3:j]), "the plan is not the three tokens before <LBL>"
            seq[j - 3:j] = plan
            k = _host_end(seq, tok)
            if k is None:
                continue
            rows.append((seq, k))
            observed.append(int((seq[k] not in ok_ends) if s_val else (seq[k] in ok_ends)))
            support.append(support_of(label, u, state, s_val))
        viol = _violation_probs(model, tok, rows, s_val)
        out["generated"][name] = {"label": label, "mode": mode, "state": state, "S": s_val, "n": len(viol),
                                  "viol": viol, "observed": observed, "support": support}
        print(f"  probe {name}: {len(viol)} generated rows", flush=True)
    # as registered: real validation URLs (reported beside after amendment 1)
    for name, (label, state, s_val) in D.PROBES.items():
        urls = [u for u in D._urls(label, "val", state) if bool(parse(u).path) == bool(s_val)]
        urls = random.Random(D.PROBE_SEED).sample(urls, min(want, len(urls)))
        rows, support = [], []
        for u in urls:
            seq = tok.encode(u, label, "und", frame="domain")
            k = _host_end(seq, tok)
            if k is None or len(seq) > cfg.max_len:
                continue
            rows.append((seq, k))
            support.append(support_of(label, u, state, s_val))
        viol = _violation_probs(model, tok, rows, s_val)
        out["populations"][name] = {"label": label, "state": state, "S": s_val, "n": len(viol),
                                    "viol": viol, "support": support}
        print(f"  probe {name}: {len(viol)} real rows", flush=True)
    return out


# --------------------------------------------------------------------------- #
# DX6: E12's side effects (descriptive)
# --------------------------------------------------------------------------- #

def part_describe(args) -> dict:
    import e12_common as K
    from uie.evaluation import FIELDS
    from uie.factorization import parse
    out = {"copy_by_suffix": {}, "copy_by_length": {}, "p1q1_phishing": {}}
    for label in ("benign", "phishing"):
        counts = C.train_counts(label)
        for arm in ("e12_base", "e12_final"):
            by_suf, by_len = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
            for g in range(K.G):
                for u in json.loads((D.E12 / "gen" / f"free_{arm}_{label}_g{g}.json").read_text(encoding="utf-8")):
                    q = FIELDS["QRY"](u)
                    if not q:
                        continue
                    hit = int(q in counts)
                    suf = parse(u).public_suffix
                    by_suf[suf][0] += hit
                    by_suf[suf][1] += 1
                    b = "1-10" if len(q) <= 10 else "11-30" if len(q) <= 30 else "31-80" if len(q) <= 80 else "81+"
                    by_len[b][0] += hit
                    by_len[b][1] += 1
            out["copy_by_suffix"].setdefault(label, {})[arm] = dict(by_suf)
            out["copy_by_length"].setdefault(label, {})[arm] = dict(by_len)
        del counts
    for arm in ("e12_base", "e12_final", "e12_no_r1"):
        c = collections.Counter()
        for g in range(K.G):
            for u in json.loads((D.E12 / "gen" / f"ctrl_{arm}_phishing_P1Q1_g{g}.json").read_text(encoding="utf-8")):
                c[f"P{int(bool(FIELDS['PTH'](u)))}Q{int(bool(FIELDS['QRY'](u)))}"] += 1
        out["p1q1_phishing"][arm] = dict(c)
    print("  describe done", flush=True)
    return out


# --------------------------------------------------------------------------- #
# DX7 (amendment 2): under-fitting or sampling
# --------------------------------------------------------------------------- #

def _field_targets(seq, tok, field):
    """The target positions of the field's characters and their terminators, and
    the field string they decode to (SUB fields joined by '.', as the gate does)."""
    from uie.tokenizer import _MARKERS
    bounds = {tok.marker_id(m) for m in _MARKERS} | {tok.rs_id, tok.eos_id}
    mark = tok.marker_id(field)
    targets, parts = [], []
    i = 0
    while i < len(seq):
        if seq[i] == mark:
            j = i + 1
            while j < len(seq) and seq[j] not in bounds:
                j += 1
            parts.append("".join(tok.itos[t] for t in seq[i + 1:j]))
            targets += list(range(i + 1, min(j, len(seq) - 1) + 1))
            i = j
            if field != "SUB":
                break
        else:
            i += 1
    return targets, ".".join(parts)


def _transformer_nll(model, tok, urls, label, field, max_len):
    """Per URL: (nats per symbol over the field's characters and terminators, the
    decoded field string); None for URLs too long to encode."""
    import torch
    import torch.nn.functional as F
    rows = []
    for u in urls:
        seq = tok.encode(u, label, "und", frame="domain")
        if len(seq) > max_len:
            rows.append(None)
            continue
        t, s = _field_targets(seq, tok, field)
        rows.append((seq, t, s) if t else None)
    out = [None] * len(rows)
    live = [i for i, r in enumerate(rows) if r is not None]
    with torch.no_grad():
        for b in range(0, len(live), 128):
            idx = live[b:b + 128]
            width = max(len(rows[i][0]) for i in idx)
            x = torch.tensor([rows[i][0] + [tok.pad_id] * (width - len(rows[i][0])) for i in idx])
            logp = F.log_softmax(model(x).float(), dim=-1)
            for r, i in enumerate(idx):
                seq, t, s = rows[i]
                pos = torch.tensor([j - 1 for j in t])
                tgt = torch.tensor([seq[j] for j in t])
                out[i] = (float(-logp[r, pos, tgt].sum()) / len(t), s)
    return out


def part_fit(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    torch.set_num_threads(4)
    cell = D.BY_NAME[args.cell]
    name, mode, label, state, field = cell
    f = D.field_of(field)
    model, tok, cfg = load_checkpoint(D.E12 / "train" / D.PRIMARY / "final.pt", device=torch.device("cpu"))
    model.eval()
    val = D.pool_urls(cell, "val")
    train_urls = D.pool_urls(cell, "train")
    sample_n = 300 if args.dry else 3000
    if args.dry:
        val = random.Random(D.PROBE_SEED).sample(val, min(300, len(val)))
    train_sample = random.Random(D.PROBE_SEED + 1).sample(train_urls, min(sample_n, len(train_urls)))
    ng = D.WittenBell().fit([f(u) for u in train_urls])
    rng = random.Random(D.PROBE_SEED + 2)
    v7 = []
    for _ in range(100):
        h = "".join(rng.choice(ng.alphabet) for _ in range(ng.order - 1))
        v7.append(sum(ng.prob(c, h) for c in ng.alphabet + [ng.END, "\x00"]))
    out = {"cell": name, "n_val": len(val), "n_train_sample": len(train_sample), "v7": v7}
    for split, urls in (("val", val), ("train", train_sample)):
        tr = _transformer_nll(model, tok, urls, label, field, cfg.max_len)
        rows, mismatch, skipped = [], 0, 0
        for u, r in zip(urls, tr):
            if r is None:
                skipped += 1
                continue
            nll_t, s = r
            if s != f(u):
                mismatch += 1
                continue
            rows.append((nll_t, ng.nll(f(u))))
        out[split] = {"nll_transformer": [a for a, _ in rows], "nll_ngram": [b for _, b in rows],
                      "mismatch": mismatch, "skipped": skipped}
        print(f"  fit {name} {split}: {len(rows)} strings, {mismatch} mismatched, {skipped} skipped", flush=True)
    return out


# --------------------------------------------------------------------------- #
# DX8 (amendment 3): sampling a field from a REAL context
# --------------------------------------------------------------------------- #

def _prefill(model, ids):
    """The forward pass over a batch of equal-length prefixes, keeping each layer's
    keys and values: the model's own blocks, replayed."""
    import torch.nn.functional as F
    B, t = ids.shape
    import torch
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


def _step(model, tok_ids, pos, cache):
    """One cached step: tok_ids (B,) at position pos; returns the next logits."""
    import torch
    import torch.nn.functional as F
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


def _sample_fields(model, tok, urls, label, field, max_len, seed, check=0):
    """Per URL: the field sampled raw (temperature 1.0, the full softmax, as the
    registered decode draws) after the URL's REAL prefix up to and including the
    field's opening marker; None when skipped (too long) or invalid (V8). With
    `check` > 0, the first `check` batches compare every cached step against the
    full forward pass (V10)."""
    import collections as col
    import torch
    import torch.nn.functional as F
    from uie.tokenizer import _MARKERS
    assert not model._struct, "the cached sampler assumes the plain output head"
    bounds = {tok.marker_id(m) for m in _MARKERS} | {tok.rs_id, tok.eos_id}
    mark, sub = tok.marker_id(field), tok.marker_id("SUB")
    gen = torch.Generator().manual_seed(seed)
    prefixes = []
    for u in urls:
        seq = tok.encode(u, label, "und", frame="domain")
        prefixes.append(seq[:seq.index(mark) + 1] if mark in seq and len(seq) <= max_len else None)
    groups = col.defaultdict(list)
    for i, p in enumerate(prefixes):
        if p is not None:
            groups[len(p)].append(i)
    out, v10, checked = [None] * len(urls), [], 0
    with torch.no_grad():
        for L in sorted(groups):
            for b in range(0, len(groups[L]), 64):
                batch = groups[L][b:b + 64]
                rows = [list(prefixes[i]) for i in batch]
                logits, cache = _prefill(model, torch.tensor(rows))
                done, bad = [False] * len(batch), [False] * len(batch)
                chunks = [[[]] for _ in batch]
                pos = L
                while not all(done) and pos < max_len:
                    if checked < check:
                        full = model(torch.tensor(rows))[:, -1].float()
                        v10.append(float((full - logits.float()).abs().max()))
                    picks = torch.multinomial(F.softmax(logits.float(), dim=-1), 1, generator=gen).squeeze(1)
                    for r, t in enumerate(picks.tolist()):
                        rows[r].append(t)
                        if done[r]:
                            continue
                        if field == "SUB" and t == sub:
                            chunks[r].append([])
                        elif t in bounds:
                            done[r] = True
                        elif len(tok.itos[t]) != 1:
                            bad[r] = done[r] = True
                        else:
                            chunks[r][-1].append(tok.itos[t])
                    if all(done):
                        break
                    logits = _step(model, picks, pos, cache)
                    pos += 1
                checked += 1
                for r, i in enumerate(batch):
                    s = ".".join("".join(c) for c in chunks[r]) if field == "SUB" else "".join(chunks[r][0])
                    out[i] = s if (done[r] and not bad[r] and s) else ""      # "" = invalid (V8)
    return out, v10


def part_context(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    torch.set_num_threads(4)
    cell = D.BY_NAME[args.cell]
    name, mode, label, state, field = cell
    seeds = DRY["seeds"] if args.dry else D.SEEDS
    dx1 = json.loads((D.OUT / f"decomp_{name}.json").read_text(encoding="utf-8"))
    n = min(dx1["n"], DRY["n_req"]) if args.dry else dx1["n"]
    model, tok, cfg = load_checkpoint(D.E12 / "train" / D.PRIMARY / "final.pt", device=torch.device("cpu"))
    model.eval()
    urls = D.pool_urls(cell, "val")
    val = D.pool(cell, "val")
    assert len(urls) == len(val)
    if args.dry:
        urls = urls[:max(3 * n, 400)]
    rc, v10 = _sample_fields(model, tok, urls, label, field, cfg.max_len, 20261005, check=3)
    good = [s for s in rc if s]
    gen = D.generated(cell, D.PRIMARY)
    out = {"cell": name, "n": n, "urls": len(urls), "sampled_valid": len(good),
           "skipped": sum(s is None for s in rc), "invalid_or_empty": sum(s == "" for s in rc),
           "v10_max_abs_diff": max(v10) if v10 else None, "seeds": list(seeds), "draws": []}
    for s in seeds:
        out["draws"].append(C.harness(val, {"gen": sample(gen, n, s + 101), "rc": sample(good, n, s + 909)}, s, n))
        print(f"  context {name} seed {s}", flush=True)
    out["dx1_gen"] = [d[D.PRIMARY] for d in dx1["against_val"]][:len(seeds)]
    return out


PARTS = {"validity": part_validity, "decomp": part_decomp, "quantity": part_quantity,
         "probe": part_probe, "describe": part_describe, "fit": part_fit, "context": part_context}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--cell", help="decomp / quantity: a cell name from e13d_common.CELLS")
    ap.add_argument("--dry", action="store_true", help="toy size into data/e13d/dry, never read")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    report = PARTS[args.part](args)
    report.update({"registration": "research/e13/PREREG_E13D.md", "part": args.part,
                   "dry": bool(args.dry), "seconds": round(time.time() - t0, 1)})
    tag = args.part + (f"_{args.cell}" if args.cell else "")
    out = (D.OUT / "dry" if args.dry else D.OUT) / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"wrote {out} after {report['seconds']:.0f}s")


if __name__ == "__main__":
    main()
