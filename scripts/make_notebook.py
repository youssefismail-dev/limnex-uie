"""
scripts/make_notebook.py

Generate the Kaggle training notebook.

Written as a generator rather than a hand-edited `.ipynb` because notebook JSON
is unpleasant to maintain by hand and silently breaks on a stray comma. The
notebook is a build artifact; this file is the source.

It targets a "Save Version" batch run: the notebook must execute top to bottom
with no interaction, tolerate a missing dependency, and leave everything the
next phase needs in /kaggle/working.
"""
from __future__ import annotations

import json
from pathlib import Path

DATASET = "uie-corpus2"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


CELLS = [
    md(f"""
# URL Intelligence Engine — E3 training

First run on the rebuilt corpus and the two-corpus, field-scoped architecture.
Everything below changed since E2.2, and each change is traceable to a
measurement rather than a hunch.

**The corpus.** 1,132,768 benign registrable labels — 11.4x E2.2's 99,303 —
from 34,008 suffix-stratified blocks over 8 era-matched crawls. Suffix entropy
4.358 -> 6.647, distinct suffixes 251 -> 1,272, top-10 share 69.9% -> 36.8%.
Phishing regrouped by HOST rather than registrable domain, which recovered
240,815 attacker-chosen names against 94,234.

**Two corpora, because names and paths want opposite things.** Names want every
domain weighted equally; paths want many pages from one site. No single URL list
does both — E2.1 and E2.2 each measured that trade directly. So:

    name       1 url per name, every name        teaches [SUF] [LBL] [SUB]
    structure  <=40 urls, multi-page sites only  teaches [PTH] [QRY]

**Field-scoped loss** enforces it: a row supplies gradient only for the fields
its corpus is a valid teacher of. Markers and the end token are trained on ALL
rows, because URL *shape* is not a content question — scoping them would teach
"a path follows a label" only from multi-page sites and drive the generated bare
rate to zero against a real 2.9% / 47.4%.

**class_alpha = 1**, weighting by LABEL count not row count. That is the unique
exposure-neutral point: every name in both classes is seen the same number of
times (~1.65 across the run), so memorisation pressure is equal and step count
alone sets its level.

**The scheme and language tokens are gone.** Both were class markers — http
implied phishing via the web's TLS migration, and `und` was 100% of phishing
against 1.6% of benign. Language survives through the suffix, which carries
77.5% of the signal.

**Before running:** Settings -> Accelerator = **GPU T4 x2**, Internet = **On**,
add the `{DATASET}` dataset as input. Then **Save Version -> Save & Run All**.

Outputs land in `/kaggle/working/s4b_w<weight>/`, one directory per arm:
`final.pt`, `best.pt`, `train_result.json`.
"""),

    code("""
# --- locate the dataset ------------------------------------------------------
# Never hard-code the mount path. It is Kaggle's SLUG, derived from the title
# rather than from what you typed, and the depth varies: a dataset can land at
# /kaggle/input/<slug>/ or nested under /kaggle/input/datasets/<user>/<slug>/.
# A one-level glob missed the nested form and reported the dataset as absent
# while the sidebar correctly showed it attached. Search at ANY depth.
#
# Anchor on bundle.manifest.json, not on a data file. The corpora are now
# name_train / structure_train / phishing_name_train / phishing_structure_train,
# and an earlier version of this cell globbed for `train.jsonl*` — which matches
# only names STARTING with that, so every file was skipped and a correctly
# attached dataset was reported as missing. The manifest is the one file whose
# name cannot drift with the corpus layout.
#
# Kaggle also DECOMPRESSES gzip while processing an upload, so `.jsonl.gz`
# arrives as `.jsonl`; `resolve()` accepts either.
from pathlib import Path

root = Path("/kaggle/input")
candidates = sorted(root.rglob("bundle.manifest.json")) if root.exists() else []

if not candidates:
    print("tree under /kaggle/input:")
    for path in (sorted(root.rglob("*"))[:60] if root.exists() else []):
        print("   ", path)
    raise SystemExit(
        "Could not find train.jsonl.gz anywhere under /kaggle/input.\\n"
        "  If the tree above is empty, the dataset is not attached:\\n"
        "  right panel -> Add Input -> search your dataset -> click +\\n"
        "  Inputs mount when the SESSION starts, so re-run after attaching.")

DATA = candidates[0].parent
print("dataset:", DATA)
"""),

    code("""
# --- environment -------------------------------------------------------------
import subprocess, sys, importlib

sys.path.insert(0, str(DATA))

# tldextract carries the Public Suffix List. It may already be in the image; only
# install when it is genuinely missing, so the run does not depend on the network
# any more than it has to.
try:
    importlib.import_module("tldextract")
    print("tldextract: already present")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tldextract"],
                   check=True)
    print("tldextract: installed")

import torch
print("torch:", torch.__version__)
if torch.cuda.is_available():
    print("gpu  :", torch.cuda.get_device_name(0))
else:
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then rerun.")
"""),

    code("""
# --- sanity check: the bundle is what we think it is -------------------------
# A mounted-but-wrong dataset otherwise fails deep inside training, after the
# GPU quota has already been spent.
import json

print("contents:", sorted(p.name for p in DATA.iterdir()))
manifest = json.loads((DATA / "bundle.manifest.json").read_text())

print()
print("corpora:")
for stem, info in manifest["corpora"].items():
    print(f"  {stem:<28}{info['rows']:>10,} rows")

# TWO splits, keyed differently on purpose: the invented name is the LABEL for
# benign and the SUBDOMAIN for phishing, so each class is split on the unit that
# carries its invention. Keying phishing on the domain put whole platforms in
# single partitions and made two samples of REAL phishing separate at 0.9138.
print()
print("splits:")
for name, info in manifest["splits"].items():
    print(f"  {name:<32}{info['key_kind']:>8}-keyed  {info['counts']}")

print()
print("suffix table:", manifest["suffix_table"])
b = manifest["known_urls_bloom"]
print(f"novelty filter: {b['n_added']:,} urls, fp {b['expected_error_rate']:.4%}")
"""),

    code("""
# --- train -------------------------------------------------------------------
# Called IN-PROCESS on purpose. `!python -m uie.kaggle_train` would spawn a
# fresh interpreter that does NOT inherit this notebook's sys.path, so the
# package added two cells above is invisible to the child.
#
# Watch the first lines: loss starts near ln(vocab) and should fall under 3.0
# within a thousand steps. Validation loss is WEIGHTED by the same field masks
# as training, so `best.pt` is selected on the objective actually being
# optimised rather than on a full-URL loss the model no longer minimises.
from pathlib import Path
from uie.kaggle_train import run

# S4b -- LOSS WEIGHT ON PATH SEGMENT STARTS, APPLIED TO THE POSITION.
#
# THE DEFECT. S3 measured over-confidence (cross-entropy minus the model's own
# entropy) at +0.329 on the first character of a path segment against +0.032
# elsewhere in the same segments -- a 10.4x spike at ONE decision, "which word
# to start", with depth already correct at 1.769 vs a target of 1.782.
#
# WHY NOT A DECODER FIX. A temperature fitted on held-out data put the boundary
# optimum at T*=1.25 against 1.05 mid-segment, confirming the miscalibration is
# boundary-specific -- but the best cross-entropy it could buy was 0.034 nats
# against a 0.329 gap. Temperature rescales a distribution; it cannot re-rank
# one. A tenth of the defect is sharpness, the rest is SHAPE, and cross-entropy
# is a strictly proper scoring rule whose unique minimiser is the true
# conditional. That is why this is a loss change.
#
# WHAT S4 GOT WRONG, AND WHAT S4b FIXES. S4 computed the weight inside the
# content branch of the mask, so a boundary whose target was a CHARACTER got the
# full weight while a boundary whose target was EOS or a marker fell through to
# 1.0. Measured on the corpus, 17.50% of boundary positions are non-content --
# 16.64% EOS, overwhelmingly the trailing-slash form `example.com/docs/`.
#
# The loss therefore sampled the boundary decision with a bias: at w=4 the model
# saw four times the evidence for "a character follows" and unamplified evidence
# for "the URL ends here". It learned that segments START rather than what
# starts them. Measured on the S4 run:
#
#     benign EOS NLL   1.261 -> 1.423   (+0.162, worst regression anywhere)
#     benign depth     1.769 -> 2.165   against a target of 1.782
#     benign seg_len  13.353 -> 14.031  the actual goal, barely moved
#
# And between w=2 and w=4 at delta=0, segment length did not move at all
# (13.453 -> 13.437) while depth climbed a further +0.16: past w=2 the weight
# bought only more segments.
#
# S4b weights the POSITION regardless of target, leaving the content-versus-stop
# ratio at that step exactly as the data states it and raising only the step's
# importance -- which is what "the model is miscalibrated HERE" actually asks
# for. Everything else is unchanged: same arms, same steps, same delta, same
# harness. This run isolates the mask-specification fix and nothing else.
ARMS = [1.0, 2.0, 4.0]

summaries = {}
for w in ARMS:
    print("=" * 78)
    print(f"=== S4 arm: boundary_weight = {w}")
    print("=" * 78, flush=True)
    summaries[f"w{w:g}"] = run(DATA, Path("/kaggle/working") / f"s4b_w{w:g}",
              steps=20000, batch_size=256,
              d_model=384, n_layers=8, n_heads=8,   # 14.35M, as E2.2
              class_alpha=1.0,                       # exposure-neutral
              # E5. Emits <F:domain>/<F:page> and, because the model can then
              # tell the two sampling frames apart, DISABLES field-scoped
              # content masking -- path content trains on every row rather than
              # structure rows only (benign 1.65M -> 2.55M, phishing 46k ->
              # 239k). Set False to reproduce E4 bit-identically.
              frame=True,
              # E6 REJECTED. The factorised head's own mechanistic prediction
              # failed (entropy->EOS slope 5.0x -> 4.5x, not flattened) and the
              # leak it removed was not the cause. S4 builds on E5, not E6.
              factorized=False,
              boundary_weight=w)
"""),

    code("""
# --- S4b verdict -------------------------------------------------------------
# PRE-REGISTERED BEFORE THE RUN. S4 already proved the calibration half works;
# this run is judged on whether it does so WITHOUT the stop damage.
#
# Baselines are the measured S4 numbers at w=4, benign val:
#     boundary CE      3.170 -> 3.052   (kept: this part worked)
#     boundary overconf 0.330 -> 0.127  (kept)
#     EOS NLL          1.261 -> 1.423   (+0.162  THE FAILURE S4b MUST REMOVE)
#     depth            1.769 -> 2.165   (target 1.782; downstream of the above)
#
#   SUCCESS   boundary overconf <= +0.15 AND EOS NLL rise < 0.03 vs the w=1
#             control. The calibration win is kept and the stop damage is gone.
#   PARTIAL   overconf improves but EOS NLL rise is 0.03-0.10: the bias was a
#             contributor, not the whole cause.
#   REFUTED   EOS NLL still rises > 0.10. The selection bias was NOT the
#             mechanism, the damage is intrinsic to up-weighting boundaries, and
#             no variant of this approach should be tried again.
#
# REFUTED is a real outcome, not a formality: it would mean the boundary defect
# cannot be fixed by re-weighting at all, and the honest next move is to stop
# optimising path structure and characterise what the PTH discriminator actually
# keys on -- since three models spanning 24 to 32 characters of path length all
# scored the same excess.
import json
from pathlib import Path

rows = {}
for w in ARMS:
    r = json.loads((Path("/kaggle/working") / f"s4b_w{w:g}" /
                    "train_result.json").read_text())
    rows[w] = r

print(f"{'arm':<8}{'class':<10}{'bnd CE':>9}{'bnd oc':>9}{'mid oc':>9}"
      f"{'EOS NLL':>10}{'marker':>9}")
for w in ARMS:
    for label, splits in rows[w].get("boundary_nll", {}).items():
        v = splits["val"]
        sh = rows[w].get("shape_nll", {}).get(label, {}).get("val", {})
        if "boundary_ce" not in v:
            continue
        print(f"w={w:<6g}{label:<10}{v['boundary_ce']:>9.3f}"
              f"{v['boundary_overconf']:>+9.3f}{v['mid_overconf']:>+9.3f}"
              f"{sh.get('eos_nll', float('nan')):>10.3f}"
              f"{sh.get('marker_nll', float('nan')):>9.3f}")

print()
base = rows[1.0]
for w in ARMS:
    if w == 1.0:
        continue
    for label, splits in rows[w].get("boundary_nll", {}).items():
        v = splits["val"]
        b = base["boundary_nll"][label]["val"]
        if "boundary_ce" not in v:
            continue
        d_eos = (rows[w]["shape_nll"][label]["val"]["eos_nll"]
                 - base["shape_nll"][label]["val"]["eos_nll"])
        d_bnd = b["boundary_ce"] - v["boundary_ce"]
        if d_eos > 0.10:
            verdict = "REFUTED (stop damage is intrinsic)"
        elif v["boundary_overconf"] <= 0.15 and d_eos < 0.03:
            verdict = "SUCCESS"
        elif d_eos < 0.03:
            verdict = "STOP PRESERVED, calibration short of bar"
        else:
            verdict = "PARTIAL (bias was a contributor, not the cause)"
        print(f"  w={w:g} {label:<9} bnd CE {d_bnd:+.3f}  "
              f"overconf {v['boundary_overconf']:+.3f}  "
              f"EOS {d_eos:+.3f}  -> {verdict}")

print()
print("S4 comparison, benign val w=4:  EOS +0.162, depth 2.165 (target 1.782)")
print("If EOS is now flat, the local calibrated harness decides depth and seg_len.")
"""),

    code("""
# --- results -----------------------------------------------------------------
import json
from pathlib import Path

# Downstream cells inspect ONE arm. Named explicitly rather than defaulted,
# so a reader always knows which model the samples and field gaps belong to.
PRIMARY = f"s4b_w{max(ARMS):g}"
print(f"inspecting arm {PRIMARY} "
      f"(control is s4b_w1; change PRIMARY to compare)")
RESULT_DIR = Path("/kaggle/working") / PRIMARY
result = json.loads((RESULT_DIR / "train_result.json").read_text())

print(f"parameters : {result['params']/1e6:.2f}M")
print(f"wall time  : {result['seconds']/60:.1f} min")
print(f"dropped    : {result['dropped_over_max_len']:,} rows over max_len")
print()
print("final validation loss (per class, never pooled):")
for name, value in result["final"].items():
    print(f"   {name:<10} {value:.4f}")

print()
print("loss history:")
for row in result["history"][::max(1, len(result["history"]) // 10)]:
    print(f"   step {row['step']:>6}  " +
          "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "step"))
"""),

    code("""
# --- did the extra capacity generalise, or just memorise? ---------------------
# The same instrument that vetoed scaling before E2.1 now checks whether the
# rebuilt corpus fixed the label field. Read the gap against E2.1's, below:
#
#   gap flat or smaller, val NLL LOWER  -> capacity bought generalisation
#   gap WIDER, train NLL much lower     -> capacity bought memorisation; the
#                                          model is re-learning the corpus and
#                                          the next run must regularise, not grow
#
# E2 -> E2.1 for reference: benign LBL gap went +1.145 -> +0.535 while train NLL
# ROSE 1.201 -> 1.470. That rise is what justified this run.
E2_1_BENIGN = {"LBL": (1.470, 2.005), "SUB": (0.556, 1.142),
               "PTH": (1.751, 1.950), "SUF": (1.115, 1.250),
               "QRY": (1.892, 2.103), "SCH": (0.067, 0.070)}
E2_BENIGN = E2_1_BENIGN   # compared against the immediate predecessor

for label, split in result.get("field_nll", {}).items():
    print(f"--- {label} ---")
    print(f"  {'field':<6}{'train':>9}{'val':>9}{'gap':>9}{'E2.1 gap':>10}")
    for field in sorted(split["gap"], key=lambda k: -split["gap"][k]):
        before = ""
        if label == "benign" and field in E2_BENIGN:
            tr, va = E2_BENIGN[field]
            before = f"{va - tr:>+9.3f}"
        print(f"  {field:<6}{split['train'][field]:>9.3f}"
              f"{split['val'][field]:>9.3f}{split['gap'][field]:>+9.3f}{before}")
    print()
"""),

    code("""
# --- samples -----------------------------------------------------------------
# The qualitative bar. The E1 n-gram produced structurally perfect nonsense:
#     http://awr.oudlisyrroinaacogiriwwigdhmfowloenr.com/1o%sawvn?D07r%4fw5
# E2 put a skeleton around syllabic mush — `thunsangaluismundu`, `nausfargel` —
# pronounceable but composed of no real morphemes, alongside verbatim copies of
# memorised labels (`wiktionary` twice in 25 samples).
#
# What E2.1 is looking for is COMPOUNDS: labels built from real fragments the
# way `techbulletin` or `mycloudsync` are. Read the labels, not the paths.

for label, by_decoder in result["samples"].items():
    for decoder, urls in by_decoder.items():
        print(f"--- {label} / {decoder} ---")
        for url in urls[:8]:
            print("   ", url[:110])
        print()
"""),

    md("""
## Next

Download from the **Output** tab into `data/` locally:

| file | what it is |
|---|---|
| `s4b_w<w>/final.pt` | weights + architecture + tokenizer, in one file |
| `s4b_w<w>/best.pt` | lowest weighted-validation-loss checkpoint |
| `train_result.json` | loss history, config, per-field NLL, samples |

Then run the evaluation:

```
python -m uie.corpus.run_eval --checkpoint data/best.pt --bundle data/bundle2
```

**The old whole-URL AUC gate is retired** (BENCHMARK.md sections 8-9). Its bar of
0.65 sat below its own floor — two samples of REAL URLs under that split score
0.741 — so it was unreachable by any generator, including a perfect one. And it
measured a joint the model is no longer trained to produce.

What is reported now, per field, against a floor measured from real-vs-real on
the same population:

| field | scored against | gate |
|---|---|---|
| LBL, SUF, SUB | name corpus val | excess < 0.05 |
| PTH, QRY | structure corpus val | excess < 0.05 |
| FULL | mixed | **auxiliary, never gated** |

`excess = auc - floor`. A perfect generator scores 0. The 0.05 bar sits just
outside the measured spread of real-vs-real variation (0.514 to 0.601).

Real pools are deduplicated to one URL per host before scoring: the generator
produces one URL per invented host, so a comparator holding six pages of the
same site measures repetition rather than realism. On the phishing structure
corpus that difference was 0.7647 against 0.4798.
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


if __name__ == "__main__":
    # Experiment-neutral name. The notebook tracks whatever the current
    # experiment is; a per-phase filename leaves stale copies lying around and
    # makes "which one do I import?" a question with a wrong answer.
    out = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle_train.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NOTEBOOK, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({len(CELLS)} cells)")
