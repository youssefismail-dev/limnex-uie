"""CR1 -- verify the GPU generations BEFORE the scorer reads them.

1. files     every registered (state, generation) file exists -- none for the two
             not-measurable states -- with 3,000 URLs and the sha256 PROVENANCE
             records; the checkpoint and suffix table are the laptop's
2. states    every URL honours its state, as the gate reads it
3. identity  for four states, generation 0: regenerate the FIRST PASS's leading
             suffix groups on this CPU (the gate's own generation path) and
             compare byte for byte wherever the first-pass URL honoured -- such
             URLs stay in place in joint mode; in held mode they lead their group

    python research/cr1/cr1_verify.py --gens data/cr1/gen --checkpoint data/best_e7.pt \
        --bundle data/bundle2
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

import torch  # noqa: E402

from cr1_score import (G, N_MAIN, NOT_MEASURABLE, STATES, gen_seed, in_state,  # noqa: E402
                       tag_of)
from uie.model import generate  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402

DELTA = {"benign": 0.22, "phishing": 0.30}
IDENTITY_STATES = ("benign_P1", "phishing_Q1", "benign_P1_Q0", "phishing_P0_Q0")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_check(model, tok, table, urls, label, state, seed, mode, budget):
    """Regenerate the first pass's leading suffix groups on this CPU and
    compare where the first-pass URL honoured: joint mode keeps such a URL in
    its shuffled slot; held mode puts them first in their group's block."""
    n = len(urls)
    rng = random.Random(seed)
    drawn = table.sample(label, n, rng)
    perm = list(range(n))
    rng.shuffle(perm)                  # the permutation the generator applies
    where = {orig: pos for pos, orig in enumerate(perm)}
    groups = sorted(collections.Counter(zip(["domain"] * n, drawn)).items())
    start, same, compared, used = 0, 0, 0, 0
    for i, ((frame, suffix), count) in enumerate(groups):
        if used >= budget:
            break
        first = generate(model, tok, label=label, n=count, suffix=suffix,
                         decoder="temperature", seed=seed + i, frame=frame,
                         field_bias={"PTH": {tok.eos_id: -DELTA[label]}},
                         batch_size=32, device=torch.device("cpu"),
                         decoder_kwargs={"temperature": 1.0}, fields=state)
        ok = [k for k in range(count) if in_state(first[k], state)]
        for rank, k in enumerate(ok):
            # joint: an honouring first-pass URL keeps its shuffled slot;
            # held: the group's honouring first-pass URLs lead its block.
            pos = where[start + (k if mode == "joint" else rank)]
            same += urls[pos] == first[k]
            compared += 1
        used += count
        start += count
    return same, compared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=300)
    args = ap.parse_args()

    prov = json.loads((args.gens / "PROVENANCE.json").read_text(encoding="utf-8"))
    mode = prov["mode"]
    by_file = {r["file"]: r for r in prov["files"]}
    failures = []
    if prov["checkpoint"]["sha256"] != sha256(args.checkpoint):
        failures.append("checkpoint differs from the local file")
    if prov["suffix_table_sha256"] != sha256(args.bundle / "suffix_table.json"):
        failures.append("suffix table differs from the local file")

    files = {}
    for s, (label, state) in enumerate(STATES):
        tag = tag_of(label, state)
        skip = (label, tuple(sorted(state.items()))) in NOT_MEASURABLE
        for g in range(G):
            p = args.gens / f"{tag}_g{g}.json"
            if skip:
                if p.exists():
                    failures.append(f"{p.name}: exists for a not-measurable state")
                continue
            if not p.is_file():
                failures.append(f"missing {p.name}")
                continue
            urls = json.loads(p.read_text(encoding="utf-8"))
            rec = by_file.get(p.name)
            if len(urls) != N_MAIN:
                failures.append(f"{p.name}: {len(urls)} urls")
            if rec is None or rec["sha256"] != sha256(p) or rec["seed"] != gen_seed(s, g):
                failures.append(f"{p.name}: sha256 or seed differs from PROVENANCE")
            off = sum(not in_state(u, state) for u in urls)
            if off:
                failures.append(f"{p.name}: {off} URLs outside the state")
            files[(tag, g)] = (s, label, state, urls)
    print(f"  files checked: {len(files)} (expected {14 * G}); mode {mode}")

    model, tok, _ = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    model.eval()
    table = SuffixTable.load(args.bundle / "suffix_table.json")
    identity = {}
    for tag in IDENTITY_STATES:
        if (tag, 0) not in files:
            continue
        s, label, state, urls = files[(tag, 0)]
        seed, n = gen_seed(s, 0), N_MAIN
        same, compared = identity_check(model, tok, table, urls, label, state,
                                        seed, mode, args.budget)
        identity[tag] = {"compared": compared, "identical": same}
        print(f"  identity {tag:<15} {same}/{compared}", flush=True)
        if not compared or same / compared < 0.5:
            failures.append(f"{tag}: identity {same}/{compared}")

    verdict = "PASS" if not failures else "FAIL"
    (args.gens / "VERIFY.json").write_text(json.dumps(
        {"mode": mode, "files": len(files), "identity": identity,
         "failures": failures, "verdict": verdict}, indent=1), encoding="utf-8")
    print("\nfailures:", failures or "none")
    print("VERIFY:", verdict)


if __name__ == "__main__":
    main()
