"""Build research/v2/V2P1B_train.ipynb and research/v2/V2P1B_generate.ipynb
(research/v2/PREREG_V2_P1B.md section 7, step 2).

    python research/v2/make_v2p1b_notebooks.py

They are P1's notebooks (make_v2p1_notebooks.py; V2P1_train ran on Kaggle), pointed at P1b:
- the `uie-v2p1b-code` bundle and P1b's modules (v2_p1b_common as `P1`, v2_p1b_train as `T`);
- the arm `p1b_mem`; the memory check at P1b's settings;
- T4 (section 3.2): cell 3 runs the batch-coupling audit on P1b's full-size start and real
  rows of six lengths, strictly before any training, and raises STOP on a failure. The run
  repeats it on the exact model it trains (v2_p1b_train); both results go to PROVENANCE, and
  the readback requires the in-run one to have passed.
Every substitution must match exactly once or more as listed, or the build stops.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "e15"))
import make_v2p1_notebooks as BASE  # noqa: E402
from make_e15_notebooks import code, md  # noqa: E402

GENERAL = [
    ("V2P1_CODE_MANIFEST.json", "V2P1B_CODE_MANIFEST.json"),
    ("uie-v2p1-code", "uie-v2p1b-code"),
    ('"v2_mem", "v2_p1_common", "v2_p1_train")', '"v2_mem", "v2_p1_common", "v2_p1_train", "v2_p1b_common", "v2_p1b_train")'),
    ("import v2_p1_common as P1", "import v2_p1b_common as P1"),
    ("import v2_p1_train as T", "import v2_p1b_train as T"),
    ("p1_mem", "p1b_mem"),
    ("V2P1_train", "V2P1B_train"),
    ("v2p1_gen", "v2p1b_gen"),
    ("PREREG_V2_P1_MEMORY.md", "PREREG_V2_P1B.md"),
    ('"V2-P1 (', '"V2-P1b ('),
]

AUDIT = r'''m1, MEM_START = P1.start(tok)
# --- T4 (PREREG_V2_P1B.md section 3.2): the batch-coupling audit, strictly before any training
import gzip
hits = sorted(p for p in DATA.iterdir() if p.name.startswith("structure_val.jsonl"))
opener = gzip.open if hits[0].name.endswith(".gz") else open
with opener(hits[0], "rt", encoding="utf-8") as fh:
    audit_urls = [json.loads(next(fh))["url"] for _ in range(400)]
AUDIT = P1.audit(m1, P1.audit_rows(tok, audit_urls), tok.pad_id)
gap = max(AUDIT["batch_vs_alone"], AUDIT["length_grouped_vs_alone"], AUDIT["training_mode_vs_alone"])
if not AUDIT["pass"]:
    raise SystemExit(f"STOP. The batch-coupling audit failed: {AUDIT}. Nothing was trained.")
print(f"T4 audit PASSED: largest gap {gap:.6f} logits (bar {AUDIT['bar']}); no stored statistic changed")
del m1'''

SPECIFIC = {
    ("train", 2): [("m1, MEM_START = P1.start(tok)\ndel m1", AUDIT),
                   ('peak = M.memory_preflight(torch.device("cuda"), math.ceil(288 / parts))',
                    'peak = M.memory_preflight(torch.device("cuda"), math.ceil(288 / parts), **P1._settings())')],
    ("train", 3): [('"start_check": {"width_384": ref, "width_512": START, "p1b_mem": MEM_START},',
                    '"start_check": {"width_384": ref, "width_512": START, "p1b_mem": MEM_START},\n'
                    '             "audit_kaggle": AUDIT,'),
                   ('"peak_gb_training": round(torch.cuda.max_memory_allocated() / 2**30, 2)}',
                    '"peak_gb_training": round(torch.cuda.max_memory_allocated() / 2**30, 2),\n'
                    '                            "audit_in_run": dict(T.LAST_AUDIT)}')],
    ("train", 4): [('("start: the memory\'s seed", s.get("memory_seed"), P1.P1_SEED),',
                    '("start: the memory\'s seed", s.get("memory_seed"), P1.P1_SEED),\n'
                    '         ("T4 audit in the run (before the first step)", (s.get("audit_in_run") or {}).get("pass"), True),')],
}


def adapt(kind: str, cells: list) -> list:
    out, ci = [], 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for old, new in GENERAL:
            src = src.replace(old, new)
        for old, new in SPECIFIC.get((kind, ci), []):
            if src.count(old) != 1:
                raise SystemExit(f"{kind} code cell {ci}: the substitution does not match once: {old[:60]!r}")
            src = src.replace(old, new)
        out.append(code(src))
        ci += 1
    leftover = [c for c in out if "p1_mem" in "".join(c["source"]) or "uie-v2p1-code" in "".join(c["source"])]
    if leftover:
        raise SystemExit(f"{kind}: a P1 name survived the adaptation")
    return out


TRAIN_MD = md(r'''
# V2-P1b — training `p1b_mem`: P1's memory engine with the LayerNorm fix

Registration: `research/v2/PREREG_V2_P1B.md`. Everything is P1's (`e16_cap`'s recipe, data,
start and batches, 20,000 steps), with the fix and the owner's decisions:
- **the memory's query normalised per token (LayerNorm)**, so training and generation
  compute the same thing (P1's BatchNorm broke generation mode);
- **T1:** no dropout on the memory's output;
- **T2:** the memory's values in their own Adam group (betas 0.9/0.999, no weight decay),
  learning rate peaking at 1e-3 on the network's schedule;
- **T4:** the batch-coupling audit runs **before any training**, and the notebook stops if
  it fails. **No tripwire:** the run goes its full course.

**Cell 3 checks the start, runs the audit and checks that the worst-case batch fits.**
**Cell 5 reads the run back:** it must print `MISMATCHES: none -- p1b_mem is as registered`.

**Attach two inputs:** **`uie-corpus2`** and **`uie-v2p1b-code`**. **Settings:** GPU T4,
Internet **On**. **Save Version → Save & Run All.** About 4–4.5 h. When done, download
`/kaggle/working/p1b_mem/final.pt`, `/kaggle/working/p1b_mem/train_result.json` and
`/kaggle/working/PROVENANCE.json` (`best.pt` is not needed).
''')

GENERATE_MD = md(r'''
# V2-P1b — `p1b_mem`'s generation: 50 files with `e16_b5`'s decode

Registration: `research/v2/PREREG_V2_P1B.md` (P1 §3.4). Run **only after the laptop has
verified `p1b_mem`'s training** (VERIFY_TRAIN and K-used). E12's 50 jobs at E12's seeds,
decoded raw + D1 + D1′ + B5, exactly as `e16_b5`.

**Attach two inputs:** **`uie-v2p1b-code`**, and the **V2P1B_train** notebook's **output**.
**Settings:** GPU T4, Internet **On**. **Save Version → Save & Run All.** About 3–3.5 h.
A rerun keeps every finished file. When done, download `/kaggle/working/v2p1b_gen`.
''')


def main() -> None:
    BASE.write("V2P1B_train.ipynb", [TRAIN_MD] + adapt("train", BASE.train_cells()))
    BASE.write("V2P1B_generate.ipynb", [GENERATE_MD] + adapt("generate", BASE.generate_cells()))


if __name__ == "__main__":
    main()
