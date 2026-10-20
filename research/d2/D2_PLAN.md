# D2 — the open-source release: plan (APPROVED by the owner, 2026-10-17; decisions locked below)

DESIGN.md §9: *Documentation, model card, responsible-use statement, reproducible benchmark,
packaging, license.* **Gate:** someone who has never heard of LIMNEX can install it, reproduce the
benchmark, and generate samples. Nothing is published by this plan: going public is a separate,
final step that only the owner takes.

## In plain words

Five pieces of work, then one test that decides whether D2 is done:

1. **Clean the code** so it runs on any computer (no hard-coded drive or user paths).
2. **Check nothing harmful or private ships** (real phishing URLs, bulk generated phishing, local
   paths, the owner's details).
3. **Write what a stranger needs:** a model card, a quick-start, a reproduction guide, a license.
4. **Give a stranger one simple way to use it:** load the model, generate URLs, score URLs.
5. **Package the model weights** so they can be downloaded.
6. **The test:** a fresh copy, in an empty folder, following only the documents. If it installs,
   passes its tests, generates, scores and reproduces the benchmark, D2 is done.

## 1. Path cleanup

**Done 2026-10-17** (`AUDIT.md`): 50 files — the counts below were the first scan; a widened scan
found Git-Bash-style drive paths in two more shell scripts.

- 125 absolute paths, all in `research/`: 45 Python scripts, 2 shell scripts, 1 Markdown file
  (`research/e5` 27 files, `qry` 13, `s4` 5, `tell` 2, `e7` 1). `uie/` and `scripts/` are clean.
- Kinds: 109 point at the repository itself → a repo-relative root computed from the file's
  location; 15 point at old scratch folders → an output-folder argument (default under the
  git-ignored `data/`); 1 points at an old Kaggle bundle → described in words.
- **Records are not rewritten:** no number, result or conclusion changes; only how a script finds
  its files.
- **Check:** zero absolute local paths in tracked files; every changed script still compiles and
  imports; the tests pass.

## 2. Release audit (before anything is public)

**Done 2026-10-17: the release files are CLEAN; no secret anywhere, in history either** (`AUDIT.md`).

| what | how | if found |
|---|---|---|
| real phishing URLs in tracked files | match every URL-like string in tracked files against the phishing corpus (URLs and hosts) | remove, or replace by a placeholder |
| bulk generated phishing | count generated phishing samples in tracked files; only small illustrative sets may stay | remove |
| private details | local user paths in every style (`E:\…`, `/e/…`), e-mail addresses, machine names | remove |
| secrets | API tokens and keys (Kaggle, GitHub, ngrok, cloud), private keys, passwords | remove; any found in history is revoked |
| real brands in phishing-style examples | bank and company names in illustrative phishing URLs in the documents | replace with fictional brands |
| **git history** | the same scans over every blob ever committed | reported; the history stays private (Decision C) |
| the new repository's commit | the author e-mail on the single starting commit | use GitHub's private no-reply address |

## 3. Documents

**Built 2026-10-17, for the owner's review:** `LICENSE` (the official Apache-2.0 text), `NOTICE`,
`MODEL_CARD.md`, `REPRODUCE.md`, and in `README.md` a Quick start, the strict Responsible Use and
No Liability sections (one text, identical in the model card, enforced by `tests/test_api.py`) and
a License section. Found and fixed on the way: `python -m uie.corpus.feeds` had no command line,
and could not stop at the published feed commit; it now has one (`--pin`), and a fresh clone
pinned at `48f7a982…` rebuilds the phishing corpus byte for byte.

- **`MODEL_CARD.md`** for `e16_b5`: what it is (character transformer from scratch, 25.5M
  parameters, no word list, no pretrained model), training data (Common Crawl crawls and
  OpenPhish's public history, both pinned), intended and out-of-scope uses, evaluation (E3 gate
  29/32 cells and 8/8 slash rows; the final human evaluation, 55/100; the representation probe;
  the typicality scorer), known limitations (the word tell, free-mode phishing queries, two P1Q1
  cells, the surface-level probe result), dual use.
- **Responsible Use and No Liability (decision A):** a strict statement in both the README and the
  model card — permitted uses (defensive research, detector hardening and evaluation, education),
  prohibited uses (any phishing, fraud, impersonation, registering or hosting generated domains,
  any unlawful or harmful use), the user's sole responsibility for their use and its legal
  compliance, provided "AS IS" with no warranty, and no liability of the author or contributors
  for any use or misuse by anyone; plus the project's conduct ("ship the tool, not the
  ammunition"; generated domains never registered or resolved).
- **`REPRODUCE.md`:** rebuild the corpus from the recipe (pinned Common Crawl crawls + seed;
  OpenPhish at commit `48f7a982b2a4`), then run the gate with the published weights; expected
  numbers and their tolerances. Full gate on a free Kaggle GPU (a ready notebook); a small CPU
  smoke version for a quick check.
- **README quick-start:** install → generate → score → reproduce, in that order.
- **`LICENSE`** (Decision A).

## 4. One simple public interface

**Built 2026-10-17, for the owner's review:** `uie/api.py` (`uie.load`, `Engine.generate`,
`Engine.score`), `uie/engine.py` (e16_b5's decode, carried from the experiment layer and proved
identical on the real model: `ENGINE_VERIFY.json`), `uie/cli.py` (`uie generate`, `uie score`),
the two generation tables in `uie/assets/` (sha256-pinned), and `pyproject.toml` as `limnex-uie`
(Apache-2.0, `uie` command, `tldextract==5.3.1`). Tests: `tests/test_api.py`.

A thin entry point over code that already exists (no new modelling):

```python
import uie
engine = uie.load("e16_b5.pt")
engine.generate(label="phishing", n=10)          # the adopted decoder (e16_b5's rules)
engine.score(["example.com/login"])              # the typicality scorer (D1)
```

plus the same as a command line (`uie generate …`, `uie score …`). Tested offline like the rest.

## 5. Weights

`e16_cap/final.pt` (97 MiB) with its sha256, as a downloadable release file (Decision D). The
loader verifies the hash.

**Built 2026-10-17, the release candidate (nothing is public).** `research/d2/make_release.py`
builds (1) the public repository: one commit whose tree is exactly this repository's audited HEAD
tree, authored by the owner with GitHub's no-reply address and no AI co-author line (decision F),
with no remote; and (2) the release files: `limnex-uie-e16_b5.pt`, `limnex-uie-known-urls.bloom.gz`
and `SHA256SUMS.txt`, plus the release-page text and the owner's launch-day steps. The release
commit's fixes, made first:

- `uie.load` refuses a known-URL filter that is not the published one, by the sha256 of its
  content (`3ef34d06…`). Phishing generation relies on the filter, and before this any file passed.
- **The quick check** (level 2 without a GPU or data, promised in step 3):
  `research/d2/quick_check.py` regenerates one gate file on a CPU and compares it with the
  fingerprint the published run recorded (`GATE_FINGERPRINTS.json`, all 50 files). On the laptop,
  `free_e16_b5_benign_g0.json` (8,000 URLs) came out identical, in 1 h 42 min.
- REPRODUCE.md said the published runs were regenerated byte for byte on a CPU; that was a sample
  (957 URLs, V2), and it now says so.
- `research/e15/e15_acquire.py phishing` stops at E15's feed commit (`--pin`, default
  `d0642d73…`; `--out` writes elsewhere): it rebuilds the recorded rows byte for byte (355 and 563).
- The real repository address, one-command downloads, version 0.1.0, the Status note removed;
  the model card describes the filter exactly (14,873,392 collected URLs on the training side, not
  only the rows the model saw).

## 6. The gate test (decides "D2 done")

A fresh clone of the release candidate into an empty folder, a fresh Python environment, and
**only the documents**:

| step | passes if |
|---|---|
| install | `pip install` succeeds |
| tests | the suite passes offline |
| generate | `uie generate` produces URLs, and the safety screen runs |
| score | `uie score` matches the published scorer within 1e-4 nats |
| reproduce | the gate on Kaggle (owner runs the notebook, as before) reproduces E3's published cells within the stated tolerance; the quick check passes on the laptop |

**Step 6 finding (2026-10-17), before the clean-room run: level 2 as written is not runnable
from the release.** It leans on files that exist only in this repository's untracked `data/`:
the 8 Common Crawl shard maps `python -m uie.corpus.extract` needs (`data/cache/shardmap_*.json`,
built by network probes, so a rebuilt map could differ); `data/state_suffix_tables.json` (control
mode and the readings); the E15 and E17 query streams that `make_e17_bundle.py` and the scorer's
copy part check; E16's own generated files and reports that `e17_compare.py` compares against; and
`data/e11/targets.json`. The E17 scripts are the research pipeline, not a reproduction path.

**The owner chose A (2026-10-17): level 2 runs from the release**, with the condition that nothing
harmful or private is published. Built:
- **Published helper files, no URLs:** the 8 shard maps (`data/corpus/shardmaps/`, suffixes only);
  the control-mode state tables as a packaged asset (`uie/assets/state_suffix_tables.json`, suffix
  counts; it also rebuilds byte for byte from the bundle with `python -m uie.controlled`); the
  published score report (`research/d2/score_report_e16_b5.json`, numbers only). Each was
  inspected string by string, then audited.
- **`research/d2/reproduce_gate.py`:** `generate` writes the 50 files with the published model
  and the packaged tables, checking each against `GATE_FINGERPRINTS.json`; `score` refuses files
  that are not the published ones, runs E12's instruments unchanged on the rebuilt bundle, and
  compares every number of the free, control, honouring, emission, slash, s_honouring and
  plan_trace sections with the published report. `reproduce_gate.ipynb` runs `generate` on
  Kaggle, taking the repository and the model from GitHub.
- **Not reproduced at level 2, said so:** novelty and query copying, which E3 measured against the
  E15 stream (real phishing URLs, not published). Level 3 is now described as the research
  pipeline as it ran, documented rather than packaged.

**Clean-room run (2026-10-17, a new empty folder: a fresh clone of the release candidate, a fresh
Python environment with torch 2.14.1, only the documents).** Constraints: the owner's internet is
limited and the laptop leaves within hours, so the full 8-crawl rebuild was replaced by one crawl.
- Level 1: install, fingerprints, generate, the phishing refusal, score: PASS.
- Phishing corpus (from a local copy of the feed, pinned): byte-identical.
- 2b on the owner's Kaggle T4, from the release zip: **50 of 50 files identical** (141 GPU minutes).
- 2c's script on the published corpus: 3,649 of 3,649 values identical (done in the private repo).
- Benign 2a, one crawl (CC-MAIN-2026-12) re-extracted: **not the same sample.** 289 transient
  network errors each drew a new offset, which shifts the shard's later draws and the reservoir:
  2,460,121 rows kept vs 2,451,253, 66% of the URLs shared. Fixed going forward (a failed read
  retries the same offset; `n_lost_draws` recorded; test). The published corpus's own sample
  cannot be rebuilt (its run recorded no failures), so REPRODUCE.md now promises the same method
  and scale, not the same sample, and 2c on a rebuilt corpus gives close, not identical, readings.
- Bugs found and fixed on the way: the packaged tables' line endings (would fail on Linux/Kaggle),
  `uie.load` on a GPU, an unretried truncated response that killed an extraction, the notebook's
  silent failure and doubled log.
- **The owner's decision (2026-10-17):** the original benign data is not shared, publicly or on
  request (a scan of the validation files found 50 URLs with e-mail addresses and 773 with
  token-like parameters). An exact benign rebuild "cannot be guaranteed", not "is impossible": the
  original run's failed reads were never recorded. No full 8-crawl rebuild (the owner's internet
  is limited).
- **Clean-room 2c (2026-10-18): the tolerance rule failed, and was withdrawn (owner: option B).**
  The clean room rebuilt the bundle offline (one crawl re-extracted, seven original) and read the
  owner's Kaggle files against it: 29 / 1 / 2 gated cells CLEARED / CONVICTED / UNDECIDED against
  the published 29 / 2 / 1. Free mode kept all 10 verdicts (largest shift 0.006) and the free
  phishing QRY weakness stayed CONVICTED; three benign control cells near the bar changed verdict
  (P1Q1 PTH and SUB, P0Q1 QRY), and six entries fell outside the rule, all benign control (largest
  shift 0.019). The rule was wrong, not the model: the gate's uncertainty counts the variation
  between generations, not the variation of the real reference sample, which a rebuild changes.
  Option A (measure that variation by resampling, overnight) was offered; the owner chose B: no
  pass/fail rule on a rebuilt corpus, `score` shows the cells beside the published and the
  clean-room readings (`score_report_e16_b5_cleanroom.json`, numbers only), and REPRODUCE.md 2c
  states both. Bugs found on the way: `score` read the rebuilt bundle's suffix table for the plan
  readings (fixed: the packaged one), and a crash lost finished parts (fixed: saved per part).
- **Legal read:** the owner found no lawyer or legal office. Claude wrote an informational
  legal-risk review (not legal advice; kept outside the repository). Its one recommended change,
  F1, is made: **generation requires the known-URL filter for both classes** (benign copies, 545 of
  the gate's 80,000 free-mode URLs were known URLs over both classes, could carry the personal data
  about 1 in 400 real benign URLs hold); the model card also says what the published filter can
  and cannot reveal (F2).
- **Launch (decision E):** the owner set it for 2026-10-20 and authorised Claude to perform it when
  he says go (create the public repository, push the release commit, publish the v0.1.0 release
  with its files, verify the public downloads).

The items found for this step (the E15 feed pin, the CPU quick check, the address, the version)
were done in step 5. **Time:** the benign rebuild in level 2a took about 1.7 hours per crawl in the
original extraction (8 crawls), longer than this plan's "about an hour", so it runs overnight at
low priority.

## Decisions — LOCKED by the owner, 2026-10-17

| | decision | locked |
|---|---|---|
| A | **License** | **Apache-2.0** for the code and the weights. It contains the explicit protections the owner asked for: §7 *Disclaimer of Warranty* (provided "AS IS") and §8 *Limitation of Liability*. In addition, a strict **Responsible Use** and **No Liability** statement in both the README and the model card (step 3). |
| B | **Name** | **Limnex URL Intelligence Engine**, short form **Limnex UIE**. Pip distribution `limnex-uie` (step 4); the import name stays `uie`. Applied to README and DESIGN now; the full pass with the documents (step 3). |
| C | **Repository** | a **new public repository, one starting commit**; this private repository stays the full record |
| D | **Weights** | a **GitHub Release** file on that repository: the model file is attached to the release page, users download it with one click or one command, and the loader checks its sha256 before loading |
| E | **Going public** | the date is set after everything is ready and audited, and after the legal read below |
| F | **Authorship** | the public repository's single commit is authored by the owner (GitHub no-reply e-mail) with **no AI co-author line**, so its contributor list shows only the owner; the README says openly that the project was built with the help of AI coding tools, including Claude Code |

**Honest note on A (recorded at the owner's request for full protection).** No license or
disclaimer can guarantee complete legal protection: Apache-2.0's clauses mainly bind people who
use the code, and laws on dual-use security tools differ by country. The strongest protection is
the project's conduct, which the release keeps: no bulk phishing data is shipped, generated
domains are never registered or resolved, and live threats are screened out. **Added step before
E:** a lawyer or the owner's university legal office reads the final LICENSE, README disclaimer
and model card. This plan is not legal advice.

## Cost

Laptop only, no training. Path cleanup and audit: about one session; documents and the interface:
about one session; the gate test: about an hour on the laptop plus one Kaggle run by the owner.
