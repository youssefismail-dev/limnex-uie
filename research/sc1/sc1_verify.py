"""SC1 -- verify the GPU generations before anything reads them (PREREG_SC1 section 6).

    V1 files      per arm: every registered (state, generation) file, none for the
                  not-measurable states, 3,000 URLs each; sha256 and seed as the
                  provenance records; mode held; the checkpoint, the tables and
                  the code commit the laptop has
    V2 states     every URL honours its state, as FIELDS reads it
    V3 identity   CR1's identity check in held mode, with the arm's own table, on
                  four states, generation 0: at least 250 URLs compared per state,
                  every one byte-identical to a CPU regeneration
    V4 table      per file, the parsed suffixes agree on >= 99% of URLs with the
                  suffixes `table.sample(label, 3000, random.Random(seed))` draws
    V5 tables     the builder, run here, reproduces the tables the GPU used, with
                  the registered row totals

    python research/sc1/sc1_verify.py --root data/sc1 --checkpoint data/best_e7.pt \
        --bundle data/bundle2 --tables data/sc1/tables.json
"""
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cr1"))

from sc1_common import (AGREEMENT_MIN, ARMS, DELTA, G, IDENTITY_MIN,  # noqa: E402
                        IDENTITY_STATES, MEASURABLE, N_MAIN, NOT_MEASURABLE, STATES,
                        gen_seed, table_counts, tag_of)
from uie.evaluation import FIELDS  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def in_state(url, state) -> bool:
    return all(bool(FIELDS[f](url)) == want for f, want in state.items())


def is_ancestor(commit: str) -> bool:
    """Was the bundle built from this repository's history?"""
    import subprocess
    if not commit:
        return False
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          cwd=ROOT, capture_output=True).returncode == 0


def agreement(urls, table: SuffixTable, label: str, seed: int) -> float:
    """V4: the share of URLs whose parsed suffix is matched, as a multiset, by the
    suffixes the registered table and seed draw -- held mode keeps every drawn
    suffix, so only a parse effect can separate the two."""
    drawn = collections.Counter(table.sample(label, len(urls), random.Random(seed)))
    parsed = collections.Counter(FIELDS["SUF"](u) for u in urls)
    return sum((drawn & parsed).values()) / len(urls)


def check_arm(root: Path, arm: str, tables: dict, tables_sha: str, ckpt_sha: str,
              failures: list, k5: list) -> dict:
    """V1, V2, V4 for one arm. Returns {(tag, g): (s, label, state, urls)}."""
    gens = root / arm / "gen"
    prov = json.loads((gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    by_file = {r["file"]: r for r in prov["files"]}
    if prov.get("arm") != arm or prov.get("mode") != "held":
        failures.append(f"{arm}: provenance says arm {prov.get('arm')!r}, mode {prov.get('mode')!r}")
    if prov["checkpoint"]["sha256"] != ckpt_sha:
        failures.append(f"{arm}: checkpoint differs from the local file")
    if prov["tables_sha256"] != tables_sha:
        failures.append(f"{arm}: tables differ from the local file")
    if prov.get("code_integrity", {}).get("mismatched") != []:
        failures.append(f"{arm}: the code bundle did not match its manifest on the GPU")
    if not is_ancestor(prov.get("code_commit", "")):
        failures.append(f"{arm}: code commit {prov.get('code_commit')!r} is not in this history")
    expected = {f"{tag_of(label, state)}_g{g}.json" for _, label, state in MEASURABLE
                for g in range(G)}
    extra = sorted(p.name for p in gens.glob("*_g*.json") if p.name not in expected)
    if extra:
        failures.append(f"{arm}: unregistered files {extra}")
    files, low = {}, []
    for s, (label, state) in enumerate(STATES):
        tag = tag_of(label, state)
        skip = (label, tuple(sorted(state.items()))) in NOT_MEASURABLE
        for g in range(G):
            p = gens / f"{tag}_g{g}.json"
            if skip:
                if p.exists():
                    failures.append(f"{arm}/{p.name}: exists for a not-measurable state")
                continue
            if not p.is_file():
                rec = by_file.get(p.name) or {}
                if rec.get("error"):          # a registered outcome (K5), not a fault
                    k5.append(f"{arm}/{p.name}: {rec['error']}")
                else:
                    failures.append(f"{arm}: missing {p.name}")
                continue
            urls = json.loads(p.read_text(encoding="utf-8"))
            rec = by_file.get(p.name)
            if len(urls) != N_MAIN:
                failures.append(f"{arm}/{p.name}: {len(urls)} urls")
            if rec is None or rec["sha256"] != sha256(p) or rec["seed"] != gen_seed(arm, s, g):
                failures.append(f"{arm}/{p.name}: sha256 or seed differs from PROVENANCE")
            off = sum(not in_state(u, state) for u in urls)
            if off:
                failures.append(f"{arm}/{p.name}: {off} URLs outside the state")
            table = SuffixTable({label: table_counts(tables, arm, label, state)})
            a = agreement(urls, table, label, gen_seed(arm, s, g))
            low.append(a)
            if a < AGREEMENT_MIN:
                failures.append(f"{arm}/{p.name}: suffix agreement {a:.4f} < {AGREEMENT_MIN}")
            files[(tag, g)] = (s, label, state, urls)
    print(f"  {arm}: files {len(files)} (expected {len(MEASURABLE) * G}); "
          f"suffix agreement min {min(low) if low else float('nan'):.4f}", flush=True)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="holds C/gen and T/gen")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=300)
    args = ap.parse_args()

    import torch
    from build_tables import build, canonical, totals_failures
    from cr1_verify import DELTA as CR1_DELTA, identity_check
    from uie.train import load_checkpoint

    failures: list = []
    assert DELTA == CR1_DELTA, "SC1's delta must be CR1's"
    tables_blob = args.tables.read_bytes()
    tables, tables_sha = json.loads(tables_blob), sha256_bytes(tables_blob)
    ckpt_sha = sha256(args.checkpoint)

    # V5 first: the tables the GPU used must be the ones the builder makes here
    rebuilt = build(args.bundle)
    if sha256_bytes(canonical(rebuilt)) != tables_sha:
        failures.append("V5: the builder does not reproduce the tables file")
    failures.extend("V5: " + b for b in totals_failures(tables))
    print(f"  V5 tables sha256 {tables_sha[:16]}: "
          f"{'reproduced' if not any(f.startswith('V5') for f in failures) else 'NOT reproduced'}")

    k5: list = []
    files = {arm: check_arm(args.root, arm, tables, tables_sha, ckpt_sha, failures, k5)
             for arm in ARMS}

    model, tok, _ = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    model.eval()
    identity = {}
    for arm in ARMS:
        for tag in IDENTITY_STATES:
            if (tag, 0) not in files[arm]:
                failures.append(f"V3 {arm}/{tag}: no generation-0 file")
                continue
            s, label, state, urls = files[arm][(tag, 0)]
            table = SuffixTable({label: table_counts(tables, arm, label, state)})
            same, compared = identity_check(model, tok, table, urls, label, state,
                                            gen_seed(arm, s, 0), "held", args.budget)
            identity[f"{arm}/{tag}"] = {"compared": compared, "identical": same}
            print(f"  V3 identity {arm}/{tag:<15} {same}/{compared}", flush=True)
            if compared < IDENTITY_MIN or same != compared:
                failures.append(f"V3 {arm}/{tag}: identity {same}/{compared}")

    verdict = "PASS" if not failures else "FAIL"
    (args.root / "VERIFY.json").write_text(json.dumps(
        {"registration": "research/sc1/PREREG_SC1.md", "tables_sha256": tables_sha,
         "checkpoint_sha256": ckpt_sha, "identity": identity, "k5": k5,
         "failures": failures, "verdict": verdict}, indent=1), encoding="utf-8")
    print("\nK5 (registered outcome, not a fault):", k5 or "none")
    print("failures:", failures or "none")
    print("VERIFY:", verdict)


if __name__ == "__main__":
    main()
