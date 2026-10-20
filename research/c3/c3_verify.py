"""C3-QRY -- verify before anything is scored (PREREG_C3_QRY section 7).

    V1 files      arm F: the 10 registered files (benign 5 x 1,200, phishing 5 x 400),
                  sha256 and seeds as its provenance records; the checkpoint, the
                  adopted tables (sha256 0c6f2760...) and a code bundle that matched
                  its manifest, from a commit in this history; nothing else
    V2 queries    every F URL carries a query, as FIELDS reads it
    V3 identity   generation 0 of each class: the round-0 kept URLs of the leading
                  suffix groups, regenerated on this CPU, byte-identical -- until
                  at least 250 have been compared
    V4 table      per F file, the parsed suffixes agree on >= 99% of URLs with the
                  multiset table.sample(label, N, random.Random(seed)) draws
    V5 arm Q      SC1 arm T's ten Q:1 files and its report carry the pinned sha256
    V6 S0'''      re-scoring SC1 arm T's phishing_Q1 today reproduces its record

A missing F file whose provenance records FreeUnmet is K1 -- a registered
outcome -- not a verification failure.

    python research/c3/c3_verify.py --c3 data/c3 --sc1 data/sc1 --bundle data/bundle2 \
        --checkpoint data/best_e7.pt --tables data/state_suffix_tables.json
"""
import argparse
import collections
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sc1"))
sys.path.insert(0, str(HERE.parent / "cr1"))

from c3_common import (AGREEMENT_MIN, DELTA, G, IDENTITY_MIN, M, N,  # noqa: E402
                       Q_FILES_SHA256_PREFIX, STATE, T_REPORT_SHA256, TABLES_SHA256,
                       gen_seed, tag)
from uie.evaluation import FIELDS  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_ancestor(commit: str) -> bool:
    return bool(commit) and subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT, capture_output=True).returncode == 0


def identity_free(model, tok, table, urls, label, seed, m, need, generate):
    """Regenerate the round-0 batch of the leading suffix groups and compare the
    query-bearing draws, in order, with the URLs the file holds at their places:
    a group's kept URLs precede the final shuffle in group order, round-0 hits
    first. Stops once `need` URLs have been compared."""
    import torch
    n = len(urls)
    rng = random.Random(seed)
    drawn = table.sample(label, n, rng)
    perm = list(range(n))
    rng.shuffle(perm)                                  # the permutation the generator applied
    where = {orig: pos for pos, orig in enumerate(perm)}
    groups = sorted(collections.Counter(zip(["domain"] * n, drawn)).items())
    start = same = compared = 0
    for i, ((frame, suffix), count) in enumerate(groups):
        if compared >= need:
            break
        batch = generate(model, tok, label=label, n=m * count, suffix=suffix, seed=seed + i,
                         frame=frame, batch_size=32, fields=None, decoder="temperature",
                         decoder_kwargs={"temperature": 1.0},
                         field_bias={"PTH": {tok.eos_id: -DELTA[label]}},
                         device=torch.device("cpu"))
        hits = [u for u in batch if FIELDS["QRY"](u)][:count]
        for rank, u in enumerate(hits):
            same += urls[where[start + rank]] == u
            compared += 1
        start += count
    return same, compared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3", type=Path, required=True)
    ap.add_argument("--sc1", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from s0p import differences
    from sc1_score import score_states
    from sc1_verify import agreement
    from uie.controlled import StateSuffixTables
    from uie.model import generate
    from uie.train import load_checkpoint

    failures, k1 = [], []
    gens = args.c3 / "F" / "gen"
    prov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    by_file = {r["file"]: r for r in prov["files"]}
    tables_sha = sha256(args.tables)

    # V1 -- provenance-level
    if prov.get("arm") != "F":
        failures.append(f"V1: provenance arm {prov.get('arm')!r}")
    if prov["checkpoint"]["sha256"] != sha256(args.checkpoint):
        failures.append("V1: checkpoint differs from the local file")
    if not (prov["tables_sha256"] == tables_sha == TABLES_SHA256):
        failures.append("V1: tables differ from the adopted file")
    if prov.get("code_integrity", {}).get("mismatched") != []:
        failures.append("V1: the code bundle did not match its manifest on the GPU")
    if not is_ancestor(prov.get("code_commit", "")):
        failures.append(f"V1: code commit {prov.get('code_commit')!r} is not in this history")
    expected = {f"{tag(l)}_g{g}.json" for l in ("benign", "phishing") for g in range(G)}
    extra = sorted(p.name for p in gens.glob("*.json")
                   if p.name != "PROVENANCE.json" and p.name not in expected)
    if extra:
        failures.append(f"V1: unregistered files {extra}")

    tables = StateSuffixTables.load(args.tables)
    files = {}
    for label in ("benign", "phishing"):
        table = tables.for_state(label, STATE)
        for g in range(G):
            name = f"{tag(label)}_g{g}.json"
            p, rec = gens / name, by_file.get(name) or {}
            if not p.is_file():
                (k1 if rec.get("error") else failures).append(
                    f"{name}: {rec.get('error') or 'missing'}")
                continue
            urls = json.loads(p.read_text(encoding="utf-8"))
            if len(urls) != N[label]:
                failures.append(f"V1: {name} has {len(urls)} urls, registered {N[label]}")
            if rec.get("sha256") != sha256(p) or rec.get("seed") != gen_seed(label, g):
                failures.append(f"V1: {name} sha256 or seed differs from PROVENANCE")
            no_q = sum(not FIELDS["QRY"](u) for u in urls)                     # V2
            if no_q:
                failures.append(f"V2: {name} has {no_q} URLs without a query")
            a = agreement(urls, table, label, gen_seed(label, g))                # V4
            if a < AGREEMENT_MIN:
                failures.append(f"V4: {name} suffix agreement {a:.4f} < {AGREEMENT_MIN}")
            files[(label, g)] = (urls, a)
    print(f"  V1/V2/V4: {len(files)} files checked; suffix agreement min "
          f"{min((a for _, a in files.values()), default=float('nan')):.4f}", flush=True)

    # V5 -- arm Q, read only
    q_gens = args.sc1 / "T" / "gen"
    for name, prefix in Q_FILES_SHA256_PREFIX.items():
        if not sha256(q_gens / name).startswith(prefix):
            failures.append(f"V5: arm Q {name} is not the registered file")
    if sha256(args.sc1 / "T" / "cr1_report.json") != T_REPORT_SHA256:
        failures.append("V5: SC1 arm T's report is not the registered one")
    print(f"  V5: arm Q {'as registered' if not any(f.startswith('V5') for f in failures) else 'CHANGED'}",
          flush=True)

    # V6 -- the scoring environment still yields arm Q's records
    t_rep = json.loads((args.sc1 / "T" / "cr1_report.json").read_text(encoding="utf-8"))["states"]
    diff = differences(score_states(q_gens, args.bundle, ["phishing_Q1"]), t_rep)
    if diff:
        failures.append(f"V6: re-scoring arm Q's phishing_Q1 differs: {diff}")
    print(f"  V6 S0''': {'reproduced' if not diff else diff}", flush=True)

    # V3 -- byte identity on this CPU; a K1 class's F arm has failed, so it has
    # nothing to compare (and must not fail the other class's verification)
    model, tok, _ = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    model.eval()
    identity = {}
    k1_classes = {r["label"] for r in prov["files"] if r.get("error")}
    for label in ("benign", "phishing"):
        if label in k1_classes:
            identity[label] = "K1 -- not compared"
            print(f"  V3 identity {label:<9} K1 -- not compared", flush=True)
            continue
        if (label, 0) not in files:
            failures.append(f"V3: {label} has no generation-0 file")
            continue
        same, compared = identity_free(model, tok, tables.for_state(label, STATE),
                                       files[(label, 0)][0], label, gen_seed(label, 0),
                                       M[label], IDENTITY_MIN, generate)
        identity[label] = {"compared": compared, "identical": same}
        print(f"  V3 identity {label:<9} {same}/{compared}", flush=True)
        if compared < IDENTITY_MIN or same != compared:
            failures.append(f"V3: {label} identity {same}/{compared}")

    verdict = "PASS" if not failures else "FAIL"
    (args.c3 / "VERIFY.json").write_text(json.dumps(
        {"registration": "research/c3/PREREG_C3_QRY.md", "tables_sha256": tables_sha,
         "identity": identity, "k1": k1, "failures": failures, "verdict": verdict},
        indent=1), encoding="utf-8")
    print("\nK1 (registered outcome, not a fault):", k1 or "none")
    print("failures:", failures or "none")
    print("VERIFY:", verdict)


if __name__ == "__main__":
    main()
