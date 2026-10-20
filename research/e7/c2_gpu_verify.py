"""Verify GPU-made C2 generations BEFORE gate v2 reads them.

`E7_C2_generate.ipynb` ran the gate's generation loop on a Kaggle GPU. Only the
forward pass ran there: `generate()` samples on the CPU with a seeded CPU
generator, so the files should equal what this laptop would generate, except
where floating-point differences in the logits flip a draw. Four checks:

1. keys      every file the LOCAL gate will look up exists -- the key is computed
             from the local checkpoint path, exactly as run_eval computes it
2. contents  n URLs per file; sha256 equals PROVENANCE.json
3. inputs    the checkpoint and suffix table the GPU run verified are the
             local files
4. identity  regenerate a budget of suffix groups per class on this CPU through
             the gate's own call and compare byte for byte with the GPU rows

Read-only; writes nothing but its report.

    python research/e7/c2_gpu_verify.py --cache <dir> --checkpoint <best_e7.pt> --bundle <bundle2>
"""
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import uie  # noqa: E402
from uie.corpus.run_eval import _file_sha12, gen_cache_path  # noqa: E402
from uie.model import generate  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402

DELTA = {"benign": 0.22, "phishing": 0.30}
DECODER, DECODER_KWARGS = "temperature", {"temperature": 1.0}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--budget", type=int, default=400,
                    help="URLs per class to regenerate for the identity check")
    args = ap.parse_args()

    if not Path(uie.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit(f"uie imported from {uie.__file__}, not {ROOT}")
    prov = json.loads((args.cache / "PROVENANCE.json").read_text(encoding="utf-8"))
    by_name = {r["file"]: r for r in prov["files"]}
    table_path = args.bundle / "suffix_table.json"
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    table = SuffixTable.load(table_path)
    # Exactly run_eval's key: the checkpoint's file stem plus its sha256 prefix.
    tag = f"{args.checkpoint.stem}-{_file_sha12(args.checkpoint)}"
    failures = []

    # 3. inputs
    if prov["checkpoint"]["sha256"] != sha256(args.checkpoint):
        failures.append("checkpoint sha256 differs from the local file")
    if prov["suffix_table_sha256"] != sha256(table_path):
        failures.append("suffix table sha256 differs from the local file")

    report = {"cache": str(args.cache), "tag": tag, "classes": {}}
    for label in tokenizer.classes:
        d = float(DELTA[label])
        fb = {"PTH": {tokenizer.eos_id: -d}} if d else None
        for g in range(args.generations):
            gen_seed = args.seed + 1 + g * 1000
            path = gen_cache_path(args.cache, tag, label, args.n, DECODER,
                                  DECODER_KWARGS, d, gen_seed)
            # 1. keys, 2. contents
            if not path.is_file():
                failures.append(f"missing {path.name}")
                continue
            urls = json.loads(path.read_text(encoding="utf-8"))
            if len(urls) != args.n:
                failures.append(f"{path.name}: {len(urls)} urls, not {args.n}")
            rec = by_name.get(path.name)
            if rec is None or rec["sha256"] != sha256(path):
                failures.append(f"{path.name}: sha256 does not match PROVENANCE")

        # 4. identity, on generation 0: mirror generate_with_suffixes exactly.
        gen_seed = args.seed + 1
        urls = json.loads(gen_cache_path(args.cache, tag, label, args.n, DECODER,
                                         DECODER_KWARGS, d, gen_seed)
                          .read_text(encoding="utf-8"))
        rng = random.Random(gen_seed)
        drawn = table.sample(label, args.n, rng)
        perm = list(range(args.n))
        rng.shuffle(perm)                   # the same state the gate shuffles with
        where = {orig: pos for pos, orig in enumerate(perm)}
        groups = sorted(collections.Counter(zip(["domain"] * args.n, drawn)).items())
        start, same, total, used = 0, 0, 0, 0
        for i, ((frame, suffix), count) in enumerate(groups):
            if used < args.budget:
                cpu = generate(model, tokenizer, label=label, n=count, suffix=suffix,
                               decoder=DECODER, seed=gen_seed + i, frame=frame,
                               field_bias=fb, batch_size=32,
                               device=torch.device("cpu"),
                               decoder_kwargs=DECODER_KWARGS)
                same += sum(cpu[k] == urls[where[start + k]] for k in range(count))
                total += count
                used += count
            start += count
        report["classes"][label] = {"regenerated": total, "identical": same,
                                    "rate": round(same / total, 4) if total else None}
        # Independent draws would match ~0%; the same seeded stream should match
        # nearly all. 50% is a gross-mismatch guard only, not a quality bar.
        if not total or same / total < 0.5:
            failures.append(f"{label}: identity {same}/{total} -- not the gate's "
                            "stream (seeds, table, code or checkpoint differ)")
        print(f"  {label:<9} identity on {total} regenerated URLs: "
              f"{same}/{total} = {100 * same / max(total, 1):.1f}%", flush=True)

    report["failures"] = failures
    report["verdict"] = "PASS" if not failures else "FAIL"
    (args.cache / "VERIFY.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nkeys, contents, inputs:", "all OK" if not failures else failures)
    print("VERIFY:", report["verdict"])


if __name__ == "__main__":
    main()
