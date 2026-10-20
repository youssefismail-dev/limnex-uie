# Checking Limnex UIE's results yourself

**You do not need this file to use Limnex UIE.** The trained model, `e16_b5`, is published ready
to use: download `limnex-uie-e16_b5.pt` from the *Releases* page and follow the README's
*Quick start*. Nothing has to be trained or rebuilt.

This file is for readers who want to check the published numbers instead of taking them on trust:
reviewers, researchers, anyone who cites them. Each level goes one step deeper:

| level | what it shows | what it takes |
|---|---|---|
| 1 | the file you downloaded is the published model, and it runs | minutes, any computer |
| 2 | the published model scores what the model card says: rebuild the data from its public sources and run the same test | about a day, one free GPU run (or the quick check on a laptop, no GPU and no data: about 1.5–2 hours) |
| 3 | the training recipe makes this model: train a new copy from scratch | level 2's data, then about 3.5 hours on a GPU |

The training data itself is not published, only the recipe that rebuilds it: its phishing part is
real phishing URLs, and all of it comes from other people's public sources (the OpenPhish feed and
Common Crawl). Fingerprints (sha256 hashes) show that a rebuild is identical to ours.

Read the README's **Responsible Use** and **No Liability** sections first: rebuilding the data
downloads real phishing URLs from a public feed — keep them off shared machines, never open them,
and never redistribute them.

## Level 1 — check the downloaded model (minutes, any computer)

```bash
git clone https://github.com/youssefismail-dev/limnex-uie.git && cd limnex-uie
pip install -e ".[dev]"
pytest                                    # offline; about 25 minutes on a laptop
```

Download the model from the *Releases* page, then:

```bash
curl -LO https://github.com/youssefismail-dev/limnex-uie/releases/download/v0.1.0/limnex-uie-e16_b5.pt
sha256sum limnex-uie-e16_b5.pt
# must print this fingerprint (on Windows: Get-FileHash limnex-uie-e16_b5.pt, in capitals):
# 98b10b4663ab85ea4af4368b6a09ded1bd5e809d077aaf4f8f1fea013037d65f
curl -LO https://github.com/youssefismail-dev/limnex-uie/releases/download/v0.1.0/limnex-uie-known-urls.bloom.gz
uie generate --weights limnex-uie-e16_b5.pt --known-urls limnex-uie-known-urls.bloom.gz --label benign -n 5 --seed 1
uie score --weights limnex-uie-e16_b5.pt "example.com/login"
```

The fingerprint (sha256) is computed from every byte of the file: one changed byte gives a
completely different fingerprint, so a match means you have exactly the published model.
`uie.load` checks it every time and refuses any other file. It checks the known-URL filter
(`limnex-uie-known-urls.bloom.gz`, the second download, which every generation needs) the same way, by the
fingerprint of its content. The same seed always gives the same URLs, on any machine.

## Level 2 — reproduce the E3 gate with the published model (a day; one free GPU run)

### The quick check (no GPU and no data; about 1.5–2 hours on a laptop)

Before the full run, or instead of it: regenerate one of the 50 files the gate read on your own
CPU, and compare it with the fingerprint the published run recorded for that file.

```bash
python research/d2/quick_check.py --weights limnex-uie-e16_b5.pt
```

Expected: `PASS: identical to the file the E3 gate read`, for `free_e16_b5_benign_g0.json`:
8,000 URLs, sha256 `70ff0581db8db14f…` (1 h 42 min on our laptop's CPU). `--file` picks any of the
ten free-mode files. It uses only the release (the package's own decode and tables and the
published model file); the URLs stay in memory and none is printed. A match shows that your
computer, the package and the model produce exactly the URLs the gate read; the steps below show
what the gate found in them.

### 2a. Rebuild the corpus

Nothing is downloaded from this project: the corpus is rebuilt from its public sources, by recipe.

**Phishing** — the OpenPhish community feed's git history, pinned to the commit the corpus was built
from:

```bash
python -m uie.corpus.feeds --pin 48f7a982b2a4ebf679c99457fd5467359323e4c6
```

Expected: `354,143 records from 11,934 commits at 48f7a982b2a4; content_sha256 e37135db3b661fb5…`
— the same `n_records` and `content_sha256` as the tracked `data/corpus/phishing.manifest.json`
(checked: a fresh clone reproduces `data/corpus/phishing.jsonl` byte for byte).

**Benign** — Common Crawl, the 8 crawls covering the feed's window (pinned in
`uie/corpus/extract.py`, `LIVE_CRAWLS`), then the two-corpus split and the training bundle:

```bash
python -m uie.corpus.extract --cache data/corpus/shardmaps   # Common Crawl index blocks -> data/corpus2/shards
python -m uie.corpus.merge            # -> data/corpus2 (name and structure corpora)
python -m uie.corpus.build_split2     # domain / host splits -> data/corpus2
python -m uie.corpus.merge_phishing   # phishing name and structure corpora -> data/corpus2
python -m uie.corpus.pack_bundle      # the training bundle -> data/bundle2
```

Expected, honestly: **the same method and scale, not the same sample.** The published corpus was
extracted while a failed index read still drew a new random offset, which shifts every later draw
of that shard and, through the per-domain reservoir, much of the crawl's kept URLs; the original
run did not record which reads failed, so its exact sample cannot be rebuilt. Measured in the
clean-room test: one crawl (CC-MAIN-2026-12) re-extracted through 289 transient network errors
kept 2,460,121 rows against the original's 2,451,253 and shared 66% of its URLs. The sampler now
retries a failed read at the same offset, so from here on every run that loses no draw
(`n_lost_draws` 0 in its `.done.json`) is the seed's run, identical on any machine. Your bundle's
manifest therefore matches the tracked `data/corpus/bundle2.manifest.json` in its files, rules and
seeds, not in its counts and hashes, and the same holds for `suffix_table.json` and the known-URL
filter.

**The original benign data is not shared, publicly or on request.** Common Crawl URLs carry
personal data by accident: a scan of the two validation files alone found 50 URLs with an e-mail
address and 773 with token-, session-, key- or password-like parameters. Redistributing them would
republish that data, so the original corpus stays private. The original corpus was complete and
unbiased (every crawl has all 4,251 of its blocks; partial runs and incomplete shard maps are
refused); what cannot be offered is the identical sample.

The extraction is the long step: about 1.5 hours per crawl, 8 crawls, mostly waiting on the
network. It reads the index through the tracked shard maps in `data/corpus/shardmaps/` (which
public suffixes live in which index shard): they come from network probes that can miss a few
shards, so they are pinned rather than rebuilt. Control mode's per-state tables follow from the
bundle: `python -m uie.controlled --bundle data/bundle2 --out data/state_suffix_tables.json`; on
the published bundle it writes the packaged `uie/assets/state_suffix_tables.json` byte for byte
(sha256 `0c6f2760…`), which is what generation uses.

The original E0 corpus and its splits (used by the earliest experiments, not by `e16_b5`) are
rebuilt with `python -m uie.corpus.build --out data/corpus --blocks 200` and
`python -m uie.corpus.build_splits --corpus data/corpus`; their hashes are in
`data/corpus/splits.manifest.json` and `data/corpus/benign.manifest.json`.

**The added query stream** (E15: extra query-bearing URLs, benign and phishing; used only to
train, so only level 3 needs it): `research/e15/e15_acquire.py` (`targets`, `phishing`, `benign`,
`select --final`), producing `data/e15/qry_extra/qry_extra.jsonl`; the stream E16 trained on has
sha256 `637accab5622f14a…`. Its phishing rows are read at the feed commit E15 used, `d0642d73…`
(the default of `phishing --pin`; checked: the pinned step selects the recorded rows byte for byte).

### 2b. Generate the gate's 50 files (GPU, about 2.5 hours on a free Kaggle T4)

The gate reads 50 files, 180,000 URLs: free mode (8,000 URLs × 5 generations per class) and control
mode (3,000 URLs × 5 generations per state). This step needs nothing from 2a, so it can run while
the corpus rebuilds. On Kaggle, open `research/d2/reproduce_gate.ipynb` (GPU T4, internet on) and
run all: it takes this repository and the model from GitHub and runs, on any CUDA GPU just as well,

```bash
python research/d2/reproduce_gate.py generate --weights limnex-uie-e16_b5.pt
```

Expected: `50 of 50 files identical to the published run`. Each file is compared with the sha256
the published run recorded (`research/d2/GATE_FINGERPRINTS.json`) as it is written. From Kaggle,
download `repro_gen.zip` and unzip it into `data/repro/`, giving `data/repro/gen/`. Generation is
deterministic and the same on a CPU: the quick check above regenerates a free-mode file, and
`--device cpu --only <file>` regenerates any of the 50.

### 2c. Read them (laptop, about 2.5 hours)

```bash
python research/d2/reproduce_gate.py score
```

It refuses files that are not the published ones, runs the gate's instruments
(`research/native/e12_score.py`, unchanged) on `data/repro/gen/` against `data/bundle2/`, and
compares every number with the published score report (`research/d2/score_report_e16_b5.json`),
the source of every table in `research/E3_FINAL_REPORT.md` §4: 29 of 32 gated cells CLEARED, 8 of 8
slash rows, free phishing QRY x = 0.0733 (CONVICTED), control benign P1Q1 PTH 0.0539 (CONVICTED)
and SUB 0.0431 (UNDECIDED).

**On your rebuilt corpus the readings are close, not identical, and there is no pass/fail rule.**
`PASS (identical)` holds against the original corpus (checked: 3,649 of 3,649 values). Your 2a
rebuild is the same method but not the same sample (see 2a), so the gate reads the same 50 files
against different real URLs, and that moves the readings. The gate's published uncertainty does not
cover this: it measures how much the readings vary between generations of synthetic URLs, not how
much they vary with the sample of real ones. So `score` prints your gated cells beside the two
readings that were measured, the published one and our own clean-room rebuild's
(`research/d2/score_report_e16_b5_cleanroom.json`: one crawl re-extracted, the other seven the
original's, every later step rebuilt):

| | published (original corpus) | our clean-room rebuild |
|---|---|---|
| gated cells CLEARED / CONVICTED / UNDECIDED | 29 / 2 / 1 | 29 / 1 / 2 |
| free mode (10 cells) | — | all 10 verdicts the same; no x moved more than 0.006 |
| free phishing QRY, the one substantial weakness | 0.0733, CONVICTED | still CONVICTED |
| control benign P1Q1 PTH | 0.0539, CONVICTED | 0.0458, UNDECIDED |
| control benign P1Q1 SUB | 0.0431, UNDECIDED | 0.0349, CLEARED |
| control benign P0Q1 QRY | 0.0377, CLEARED | 0.0445, UNDECIDED |
| largest shift of any cell's x | — | 0.019 (a benign control cell) |
| slash rows | 8 of 8 | every rate 0.979 or above |

What to expect from your rebuild, read from that one independent rebuild: free mode as published,
the same single substantial weakness, about 29 cells cleared, and the benign control-mode cells
(the smallest real-URL pools) moving by up to about 0.02, enough to move the three cells that sit
near the 0.05 bar either way. One rebuild is one draw of that variation, not a measure of its size.

Novelty and query copying are not part of level 2: E3 measured them against the model's added
training stream (E15), which holds real phishing URLs and is not published; level 3 rebuilds it.

## Level 3 — retrain the model from scratch (GPU, about 3.5 hours on a T4)

The recipe is `E16_B5.recipe` in `uie/baseline.py` (20,000 steps, width 512, the E15 query stream
injected at rate 1); `research/e16/E16_KAGGLE.md` and the `E16_train` notebook ran it on Kaggle
from the level-2 bundle. This level is the research pipeline as it ran, documented rather than
packaged: unlike levels 1 and 2, it has not been rerun from a fresh copy of the release. GPU
training is not bit-reproducible, so a retrained model differs from the published one in its last
digits, and its generated files from the published fingerprints; the published model is the
reference.

## What is pinned, and where

| what | pinned by |
|---|---|
| model file (`limnex-uie-e16_b5.pt`) | sha256 `98b10b46…` (`uie/api.py`, from `uie/baseline.py`) |
| known-URL filter (`limnex-uie-known-urls.bloom.gz`) | content sha256 `3ef34d06…` (`uie/api.py`) |
| generation tables | sha256 in `uie/api.py` (`ASSET_SHA256`), shipped in `uie/assets/` |
| the gate's 50 files | sha256 of each (`research/d2/GATE_FINGERPRINTS.json`) |
| phishing corpus | feed commit `48f7a982…` and content hash (`data/corpus/phishing.manifest.json`) |
| benign crawls | `LIVE_CRAWLS` in `uie/corpus/extract.py`; the index shard maps in `data/corpus/shardmaps/` |
| the gate's readings | `research/d2/score_report_e16_b5.json` (numbers only) |
| training bundle | `data/corpus/bundle2.manifest.json` |
| E0 splits | `data/corpus/splits.manifest.json` |
| query stream | sha256 `637accab…` (`uie/baseline.py`, `E16_QRY_EXTRA_SHA256`); its phishing rows by feed commit `d0642d73…` (`FEED_PIN` in `research/e15/e15_acquire.py`) |
| public suffix list | `tldextract==5.3.1` (`pyproject.toml`) |
| the gate's instruments | `scikit-learn` and `scipy` (`pyproject.toml`, `[dev]`); the published report used scikit-learn 1.9.0, scipy 1.18.0, numpy 2.4.4 |
