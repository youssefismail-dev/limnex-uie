"""QC1 -- the measurements (research/qc1/PREREG_QC1.md sections 3 and 5). Laptop;
reads the step-20,000 files and the real bundle; writes data/qc1/*.json.

    python research/qc1/qc1_measure.py --part validity
    python research/qc1/qc1_measure.py --part copy          # QC1-1, every cell and arm
    python research/qc1/qc1_measure.py --part mix --arm e11_plan --cell free_benign
    python research/qc1/qc1_measure.py --part ladder --label benign
    python research/qc1/qc1_measure.py --part near          # QC1-4
    python research/qc1/qc1_measure.py --part leak --arm e11_plan
    python research/qc1/qc1_measure.py --part budget --label benign

--dry runs any part at toy size (2 seeds, n 300) into data/qc1/dry/, stamped
"dry": true; the reader refuses dry files. Prints counts and timings only, never
a reading.

E12 reports QC1-2 on its candidate (PREREG_E12 section 2), the same measurement on
E12's files:

    python research/qc1/qc1_measure.py --part mix --arm e12_final --cell free_benign \
        --gen data/e12/gen --out data/e12/qc1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qc1_common as C  # noqa: E402

DRY = {"seeds": C.SEEDS[:2], "n_req": 300}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cells(names):
    return [c for c in C.CELLS if not names or C.cell_name(*c) in names]


def generated_queries(arm, cell, gen=C.GEN):
    mode, label, state = cell
    return [C.query(u) for u in C.generated(arm, label, state, gen)]


def real_queries(cell):
    mode, label, state = cell
    return [C.query(u) for u in C.real_pool(label, state)]


def sample(xs, n, seed):
    return random.Random(seed).sample(list(xs), n)


# --------------------------------------------------------------------------- #
# Validity (section 5): V1 inputs, V2 the harness against the gate, V3 the rates
# --------------------------------------------------------------------------- #

def part_validity(args) -> dict:
    import e11_common as K
    verify = json.loads((C.ROOT / "data" / "e11_final" / "VERIFY.json").read_text(encoding="utf-8"))
    files = sorted({C.GEN / (f"free_{a}_{l}_g{g}.json" if s is None else f"ctrl_{a}_{l}_{s}_g{g}.json")
                    for a in C.ARMS for (_, l, s) in C.CELLS for g in range(K.G)})
    bundle = sorted(C.BUNDLE / f"{p}{k}.jsonl.gz" for p in ("", "phishing_")
                    for k in ("name_val", "structure_val", "name_train", "structure_train"))
    ck = C.TRAIN / C.PRIMARY / "final.pt"
    out = {"v1": {"verify": verify.get("verdict"),
                  "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT,
                                           capture_output=True, text=True).stdout.strip(),
                  "hashes": {str(p.relative_to(C.ROOT)): sha256(p) for p in files + bundle + [ck]}}}
    rep = json.loads((C.ROOT / "data" / "e11_final" / "score_report.json").read_text(encoding="utf-8"))
    seeds = DRY["seeds"] if args.dry else C.SEEDS
    v2 = {}
    for cell in C.CELLS:
        mode, label, state = cell
        gq, rq = generated_queries(C.PRIMARY, cell), real_queries(cell)
        node = rep[mode][C.PRIMARY][label if state is None else f"{label}_{state}"]
        gate = node["fields"]["QRY"]["acceptance"]["x"]
        n = min(1200, len(rq) // 3, round(len(gq) / K.G))
        xs = [C.harness(rq, {"x": sample(gq, n, s + 707)}, s, n)["x"] for s in seeds]
        v2[C.cell_name(*cell)] = {"n_gate": n, "harness_x": xs, "gate_x": gate}
        print(f"  V2 {C.cell_name(*cell):<24} n {n}  ({len(xs)} seeds)", flush=True)
    out["v2"] = v2
    v3 = {}
    for label in ("benign", "phishing"):
        counts = C.train_counts(label)
        urls = C.real_pool(label)
        qs = [C.query(u) for u in urls]
        v3[label] = {"natural": sum(q in counts for q in qs) / len(qs)}
        if label == "phishing":
            doms = C.train_domains(label)
            dq = [C.query(u) for u in urls if C.registrable(u) not in doms]
            v3[label]["domain_disjoint"] = sum(q in counts for q in dq) / len(dq)
        del counts
    out["v3"] = v3
    return out


# --------------------------------------------------------------------------- #
# QC1-1 copying, QC1-4 near misses (string counts only)
# --------------------------------------------------------------------------- #

def part_copy(args) -> dict:
    out = {}
    for label in ("benign", "phishing"):
        counts = C.train_counts(label)
        doms = C.train_domains(label) if label == "phishing" else None
        for cell in [c for c in cells(args.cell) if c[1] == label]:
            urls = C.real_pool(label, cell[2])
            rq = [C.query(u) for u in urls]
            node = {"real": C.copy_profile(rq, counts)}
            if doms is not None:
                node["real_domain_disjoint"] = C.copy_profile(
                    [C.query(u) for u in urls if C.registrable(u) not in doms], counts)
            for arm in C.ARMS:
                gq = generated_queries(arm, cell)
                node[arm] = {"generated": C.copy_profile(gq, counts), "per_generation": len(gq) / 5}
            out[C.cell_name(*cell)] = node
            print(f"  copy {C.cell_name(*cell):<24} real {len(rq)}", flush=True)
        del counts
    return out


def part_near(args) -> dict:
    out = {}
    for label in ("benign", "phishing"):
        counts = C.train_counts(label)
        idx = C.NearIndex(q for q, c in counts.items() if c >= C.FREQUENT)
        for cell in [c for c in cells(args.cell) if c[1] == label]:
            rq = [q for q in real_queries(cell) if q not in counts]
            node = {"real": {"n": len(rq), "near": sum(idx.near(q) for q in rq)}}
            for arm in C.ARMS:
                gq = [q for q in generated_queries(arm, cell) if q not in counts]
                node[arm] = {"n": len(gq), "near": sum(idx.near(q) for q in gq)}
            out[C.cell_name(*cell)] = node
            print(f"  near {C.cell_name(*cell):<24} real never-seen {len(rq)}", flush=True)
        del counts, idx
    return out


# --------------------------------------------------------------------------- #
# QC1-2 the mix or the content (paired harness)
# --------------------------------------------------------------------------- #

def part_mix(args) -> dict:
    from uie.evaluation import MIN_N
    seeds = DRY["seeds"] if args.dry else C.SEEDS
    n_req = DRY["n_req"] if args.dry else C.N_REQ
    arm, (cell,) = args.arm[0], cells(args.cell)
    mode, label, state = cell
    counts = C.train_counts(label)
    urls = C.real_pool(label, state)
    rq = [C.query(u) for u in urls]
    gq = generated_queries(arm, cell, args.gen)
    found = [q for q in gq if q in counts]
    novel = [q for q in gq if q not in counts]
    shares = {"host": sum(q in counts for q in rq) / len(rq)}
    if label == "phishing":
        doms = C.train_domains(label)
        dq = [C.query(u) for u in urls if C.registrable(u) not in doms]
        shares["domain_disjoint"] = sum(q in counts for q in dq) / len(dq)
    n0 = min(n_req, len(rq) // 3, len(gq))
    n = min(C.feasible_n(n0, len(found), len(novel), s) for s in shares.values())
    out = {"cell": C.cell_name(*cell), "arm": arm, "n_plain": n0, "n": n, "shares": shares,
           "generated": {"found": len(found), "novel": len(novel)}, "seeds": list(seeds),
           "draws": []}
    for s in seeds:
        fakes = {"plain": sample(gq, n, s + 101),
                 "mix": C.stratified(found, novel, shares["host"], n, s + 202)}
        if "domain_disjoint" in shares:
            fakes["mix_domain_disjoint"] = C.stratified(found, novel, shares["domain_disjoint"], n, s + 303)
        out["draws"].append(C.harness(rq, fakes, s, n))
        print(f"  mix {arm} {C.cell_name(*cell)} seed {s} done", flush=True)
    for name, real_s, gen_s in (("never_seen", [q for q in rq if q not in counts], novel),
                                ("found", [q for q in rq if q in counts], found)):
        m = min(n_req, len(real_s) // 3, len(gen_s))
        if m < MIN_N:
            out[name] = {"insufficient": True, "n": m}
            continue
        out[name] = {"n": m, "draws": [C.harness(real_s, {"x": sample(gen_s, m, s + 404)}, s, m)
                                       for s in seeds]}
        print(f"  mix {arm} {C.cell_name(*cell)} stratum {name} done", flush=True)
    return out


# --------------------------------------------------------------------------- #
# QC1-5 phishing pulled toward benign
# --------------------------------------------------------------------------- #

def part_leak(args) -> dict:
    seeds = DRY["seeds"] if args.dry else C.SEEDS
    n_req = DRY["n_req"] if args.dry else C.N_REQ
    arm = args.arm[0]
    rb = [C.query(u) for u in C.real_pool("benign")]
    rp = [C.query(u) for u in C.real_pool("phishing")]
    gp = generated_queries(arm, ("free", "phishing", None))
    n = min(n_req, len(rb) // 3, len(gp), len(rp))
    out = {"arm": arm, "n": n, "draws": [C.harness(rb, {"gen": sample(gp, n, s + 505),
                                                        "real": sample(rp, n, s + 606)}, s, n)
                                         for s in seeds]}
    bc, pc = C.train_counts("benign"), C.train_counts("phishing")

    def benign_only(qs):
        return sum(q in bc and q not in pc for q in qs) / len(qs)

    out["benign_only"] = {"generated": benign_only(gp), "real": benign_only(rp)}
    return out


# --------------------------------------------------------------------------- #
# QC1-6 the length limit
# --------------------------------------------------------------------------- #

def part_budget(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    seeds = DRY["seeds"] if args.dry else C.SEEDS
    n_req = DRY["n_req"] if args.dry else C.N_REQ
    label = args.label
    model, tok, _ = load_checkpoint(C.TRAIN / C.PRIMARY / "final.pt", device=torch.device("cpu"))
    max_len = model.cfg.max_len
    del model
    urls = C.real_pool(label)
    rq_all = [C.query(u) for u in urls]
    rq_fit = [C.query(u) for u in urls if len(tok.encode(u, label, "und", frame="domain")) <= max_len]
    gq = generated_queries(C.PRIMARY, ("free", label, None))
    n = min(n_req, len(rq_fit) // 3, len(gq))
    draws = []
    for s in seeds:
        f = sample(gq, n, s + 808)
        draws.append({"all": C.harness(rq_all, {"x": f}, s, n), "fit": C.harness(rq_fit, {"x": f}, s, n)})
    return {"label": label, "max_len": max_len, "n": n, "real_all": len(rq_all),
            "real_fit": len(rq_fit), "draws": draws}


# --------------------------------------------------------------------------- #
# QC1-3 the ladder: Q3.1's, verbatim
# --------------------------------------------------------------------------- #

def collecting(fn):
    """gc.collect() after every detector call: C3's fix (research/c3/c3_ladder.py),
    memory only. Q3.1's run_ladder predates it. Under Python 3.14 each call leaves
    about 120 MB of cyclic garbage; C3 measured the values bit-identical with and
    without it. QC1's first run was stopped at 4.4 GB for lack of it."""
    import gc

    def wrapped(*a, **k):
        try:
            return fn(*a, **k)
        finally:
            gc.collect()
    return wrapped


def part_ladder(args) -> dict:
    import functools
    import q3_1_resolve as L
    from q3_content import scored
    if args.dry:
        L.SEEDS, L.D = DRY["seeds"], len(DRY["seeds"])
        L.scored = functools.partial(scored, n_req=DRY["n_req"])
    L.scored = collecting(L.scored)             # run_ladder's global; Q3.1's file untouched
    label = args.label
    rq = real_queries(("free", label, None))
    plan = generated_queries(C.PRIMARY, ("free", label, None))
    ref = generated_queries(C.REPLICATION, ("free", label, None))
    out = {"label": label, "D": L.D, "t_crit": L.TCRIT, "stages": {}}
    s1 = L.run_ladder(rq, plan, L.LADDER, f"STAGE 1  {C.PRIMARY} {label}")
    out["stages"]["stage1"] = s1
    cond = [n for n in ("clip_values_15", "first_param") if s1.get(n, {}).get("verdict") == "no change"]
    if len(cond) == 2:
        s1i = L.run_ladder(rq, plan, ["identity", "clip15_first"], f"STAGE 1b {C.PRIMARY} {label}")
        s1.update({k: v for k, v in s1i.items() if k != "identity"})
        out["stages"]["stage1_interaction"] = s1i
    hits = [n for n, r in s1.items() if r["verdict"] in ("REDUCES", "raises")]
    out["hits"] = hits
    if hits:
        out["stages"]["stage2"] = L.run_ladder(rq, ref, ["identity", "null"] + hits,
                                               f"STAGE 2  {C.REPLICATION} {label}")
    return out


# --------------------------------------------------------------------------- #
# Amendment 1 -- QC1-7 the slash (teacher-forced, class swap)
# --------------------------------------------------------------------------- #

SLASH_SEED = 20260930                     # e11_step_check.py's row seed
SLASH_ROWS = {"phishing_P0Q0": ("phishing", "P0Q0", 3000),
              "benign_P0Q1": ("benign", "P0Q1", 20000),
              "benign_P0Q0": ("benign", "P0Q0", 3000)}         # reported only


def _state_of(url: str) -> str:
    from uie.evaluation import FIELDS
    return f"P{int(bool(FIELDS['PTH'](url)))}Q{int(bool(FIELDS['QRY'](url)))}"


def _slash_rows(tok, label, state, want, max_len):
    """Seeded `name_train` rows of the state that fit, with each row's host-end index
    (the first of PTH / RS / QRY / FRG / EOS after <LBL>), e11_step_check's way."""
    pre = "" if label == "benign" else f"{label}_"
    pool = C._jsonl(f"{pre}name_train")
    ends = {tok.marker_id("PTH"), tok.rs_id, tok.marker_id("QRY"), tok.marker_id("FRG"), tok.eos_id}
    lbl = tok.marker_id("LBL")
    out = []
    for i in random.Random(SLASH_SEED).sample(range(len(pool)), len(pool)):
        u = pool[i]
        if _state_of(u) != state:
            continue
        seq = tok.encode(u, label, "und", frame="domain")
        if len(seq) > max_len or lbl not in seq:
            continue
        k = next((j for j in range(seq.index(lbl) + 1, len(seq)) if seq[j] in ends), None)
        if k is None:
            continue
        out.append((u, k))
        if len(out) == want:
            break
    return out


def _end_probs(model, tok, rows, label, batch=256):
    """The model's probabilities of the five ends at each row's host end, with the
    class token `label` (the only change under the swap)."""
    import torch
    import torch.nn.functional as F
    cols = torch.tensor([tok.marker_id("PTH"), tok.rs_id, tok.marker_id("QRY"),
                         tok.marker_id("FRG"), tok.eos_id])
    out = []
    with torch.no_grad():
        for b in range(0, len(rows), batch):
            chunk = rows[b:b + batch]
            seqs = [tok.encode(u, label, "und", frame="domain") for u, _ in chunk]
            width = max(k for _, k in chunk)
            x = torch.tensor([s[:width] + [tok.pad_id] * (width - len(s[:width])) for s in seqs])
            logits = model(x).float()
            at = torch.tensor([k - 1 for _, k in chunk])
            p = F.softmax(logits[torch.arange(len(chunk)), at], dim=-1)[:, cols].double()
            out += p.tolist()
    return out                                   # [PTH, RS, QRY, FRG, EOS] per row


def part_slash(args) -> dict:
    import torch
    from uie.train import load_checkpoint
    torch.set_num_threads(4)
    out = {"seed": SLASH_SEED, "arms": {}}
    for arm in args.arm:
        model, tok, cfg = load_checkpoint(C.TRAIN / arm / "final.pt", device=torch.device("cpu"))
        model.eval()
        node = {}
        for name, (label, state, want) in SLASH_ROWS.items():
            rows = _slash_rows(tok, label, state, 300 if args.dry else want, cfg.max_len)
            other = "benign" if label == "phishing" else "phishing"
            own, swap = _end_probs(model, tok, rows, label), _end_probs(model, tok, rows, other)
            seqs_end = [tok.encode(u, label, "und", frame="domain")[k] for u, k in rows]
            if state == "P0Q0":         # no slash = the host ends or a fragment follows
                data = [int(e in (tok.eos_id, tok.marker_id("FRG"))) for e in seqs_end]
                mass = lambda p: (p[4] + p[3]) / (p[1] + p[3] + p[4])
            else:                       # P0Q1: no slash = the query follows the host directly
                data = [int(e == tok.marker_id("QRY")) for e in seqs_end]
                mass = lambda p: p[2] / (p[1] + p[2])
            node[name] = {"n": len(rows), "data": data, "own": [mass(p) for p in own],
                          "swap": [mass(p) for p in swap]}
            print(f"  slash {arm} {name}: {len(rows)} rows", flush=True)
        out["arms"][arm] = node
        del model
    return out


# --------------------------------------------------------------------------- #
# Amendment 1 -- QC1-8 benign P1Q1 PTH and SUB: state blindness
# --------------------------------------------------------------------------- #

def part_state(args) -> dict:
    import e11_common as K
    from cr1_score import pools
    from uie.evaluation import FIELDS, _one_per_host
    seeds = DRY["seeds"] if args.dry else C.SEEDS
    n_req = DRY["n_req"] if args.dry else C.N_REQ
    rep = json.loads((C.ROOT / "data" / "e11_final" / "score_report.json").read_text(encoding="utf-8"))
    real = {st_: _one_per_host(pools(C.BUNDLE, "benign", K.STATES[st_])["name"], C.POOL_SEED)
            for st_ in ("P1Q1", "P1Q0")}
    out = {"arms": {}}
    for arm in args.arm:
        # every generated URL of the state, as the gate's PTH and SUB cells score them
        # (C.generated keeps only URLs with a query, which is right for QRY alone)
        gen = [u for g in range(K.G) for u in json.loads(
            (C.GEN / f"ctrl_{arm}_benign_P1Q1_g{g}.json").read_text(encoding="utf-8"))]
        node = {}
        for field in ("PTH", "SUB"):
            f = FIELDS[field]
            r11 = [x for x in (f(u) for u in real["P1Q1"]) if x]
            r10 = [x for x in (f(u) for u in real["P1Q0"]) if x]
            gv = [x for x in (f(u) for u in gen) if x]
            n = min(n_req, len(r11) // 3, len(r10) // 3, len(gv), len(r11))
            draws = []
            for s in seeds:
                fake = sample(gv, n, s + 909)
                draws.append({"own": C.harness(r11, {"x": fake}, s, n),
                              "cross": C.harness(r10, {"x": fake}, s, n),
                              "states": C.harness(r10, {"x": sample(r11, n, s + 910)}, s, n)})
            ng = min(1200, len(r11) // 3, round(len(gv) / K.G))
            v2 = [C.harness(r11, {"x": sample(gv, ng, s + 911)}, s, ng)["x"] for s in seeds]
            gate = rep["control"][arm]["benign_P1Q1"]["fields"][field]["acceptance"]["x"]
            node[field] = {"n": n, "real_P1Q1": len(r11), "real_P1Q0": len(r10), "generated": len(gv),
                           "draws": draws, "v2": {"n_gate": ng, "harness_x": v2, "gate_x": gate}}
            print(f"  state {arm} {field}: n {n}", flush=True)
        out["arms"][arm] = node
    return out


# --------------------------------------------------------------------------- #
# Amendment 2 -- QC1-9 domain-frame queries pulled toward the page frame
# --------------------------------------------------------------------------- #

import re  # noqa: E402

PAGE_REPEATED = re.compile(r"^(page|p|lang)=\w{1,3}$")    # pagination and language strings


def part_frame(args) -> dict:
    import e11_common as K
    from cr1_score import pools
    from uie.evaluation import MIN_N
    seeds = DRY["seeds"] if args.dry else C.SEEDS
    n_req = DRY["n_req"] if args.dry else C.N_REQ
    pages = {}
    out = {"arms": {}}
    for arm in args.arm:
        node = {}
        for cell in C.CELLS:
            mode, label, state = cell
            if (label, state) not in pages:
                rows = (C._jsonl(("" if label == "benign" else f"{label}_") + "structure_val")
                        if state is None else pools(C.BUNDLE, label, K.STATES[state])["structure"])
                pages[(label, state)] = [C.query(u) for u in rows if C.query(u)]
            pq = pages[(label, state)]
            rq, gq = real_queries(cell), generated_queries(arm, cell)
            share = {name: sum(bool(PAGE_REPEATED.match(q)) for q in qs) / len(qs) if qs else None
                     for name, qs in (("page", pq), ("real", rq), ("generated", gq))}
            n = min(n_req, len(pq) // 3, len(gq), len(rq))
            name = C.cell_name(*cell)
            if n < MIN_N:
                node[name] = {"insufficient": True, "n": n, "page_pool": len(pq), "page_repeated": share}
                continue
            node[name] = {"n": n, "page_pool": len(pq), "page_repeated": share,
                          "draws": [C.harness(pq, {"gen": sample(gq, n, s + 1001),
                                                   "real": sample(rq, n, s + 1002)}, s, n)
                                    for s in seeds]}
            print(f"  frame {arm} {name}: n {n}", flush=True)
        out["arms"][arm] = node
    return out


PARTS = {"validity": part_validity, "copy": part_copy, "near": part_near, "mix": part_mix,
         "leak": part_leak, "budget": part_budget, "ladder": part_ladder,
         "slash": part_slash, "state": part_state, "frame": part_frame}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--arm", nargs="*", default=[C.PRIMARY])
    ap.add_argument("--cell", nargs="*", help="cell names, e.g. free_benign control_benign_P0Q1")
    ap.add_argument("--label", choices=("benign", "phishing"))
    ap.add_argument("--dry", action="store_true", help="toy size into data/qc1/dry, never read")
    ap.add_argument("--gen", type=Path, default=C.GEN,
                    help="mix only: another run's generated files (E12 reports QC1-2 on e12_final)")
    ap.add_argument("--out", type=Path, default=C.OUT, help="the output directory")
    args = ap.parse_args()
    if args.gen.resolve() != C.GEN.resolve() and args.part != "mix":
        ap.error("--gen is for --part mix only")
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    report = PARTS[args.part](args)
    report.update({"registration": "research/qc1/PREREG_QC1.md", "part": args.part,
                   "dry": bool(args.dry), "seconds": round(time.time() - t0, 1)})
    if args.gen.resolve() != C.GEN.resolve():
        report["gen"] = str(args.gen)
    tag = args.part + ("_" + args.arm[0] if args.part in ("mix", "leak") else "") + \
        ("_" + args.cell[0] if args.part == "mix" else "") + \
        ("_" + args.label if args.part in ("budget", "ladder") else "")
    out = (args.out / "dry" if args.dry else args.out) / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"wrote {out} after {report['seconds']:.0f}s")


if __name__ == "__main__":
    main()
