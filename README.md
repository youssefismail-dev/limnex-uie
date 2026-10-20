# Limnex URL Intelligence Engine (Limnex UIE)

**A generative model of URLs, trained from scratch.** It reads millions of real
URLs, learns how the web names things, and invents new ones — benign and
phishing-style alike (the rare exact copies of training URLs are filtered out; see
*Quick start*). Their structure holds up against real
data: in 29 of 32 measured cells, the gate's character-level detector separates
them from held-out real URLs no better than it separates two real samples (within
its 0.05 margin). Their vocabulary does not yet: a detector that checks whether
each word occurs in real data can tell them apart. See
[Known limitations](#known-limitations-and-future-work).

```
cdn-assets-eu3.shopwave.io/static/v2/bundle.js      # illustrative, benign
blog.tenjiku-books.jp/2024/03/review                # illustrative, benign
secure-verify.examplebank-alerts.support/login      # illustrative, phishing-style
signin.examplecredit-notice.online/session          # illustrative, phishing-style
```

*These four are illustrations of the target, carried over from
[DESIGN.md](DESIGN.md); they are not model output.*

The test is whether an experienced security researcher, shown these shuffled
together with real URLs, could sort them. *(That is the project's target. The project's human
evaluation — ten people, 100 URLs, its final one — found people no better than chance; 100 URLs
cannot rule out a modest ability — see [Known limitations](#known-limitations-and-future-work).)*

**Nobody writes the rules.** The model is never told that phishing URLs contain
`login`, or to swap Latin letters for Cyrillic lookalikes. It reads data and
works out the patterns itself — a rule-based generator can only ever produce
attacks its author already knew about.

**Status:** pre-alpha, in **private development**. The baseline engine is
**`e16_b5`** (adopted 2026-10-08): `e16_cap`'s weights, a character-level
transformer trained from scratch on the project's corpus (`d_model` 512, 25.5M
parameters), decoded at temperature 1.0 with three structural rules that forbid
only what real URLs never do (D1 and D1′ for the root slash, B5 for planned
queries). **Roadmap-E3 is closed, NOT MET** (2026-10-08): `e16_b5` clears 29 of
32 gated cells and all 8 slash rows; see [Acceptance](#acceptance-status) and
[Known limitations](#known-limitations-and-future-work). The word-level
experiments that followed (V2-P1, P1b and P2) are closed under the current design
rules, and `e16_b5` remains the baseline (2026-10-15). Full record:
[research/E3_FINAL_REPORT.md](research/E3_FINAL_REPORT.md). Architecture and
roadmap: [DESIGN.md](DESIGN.md).

**This repository is an independent project** in the Limnex family. Limnex UIE is
developed separately from the Limnex phishing detector (LIMNEX), which consumes it; it never
imports from, and is never graded by, any consumer. It stays private until the
roadmap, evaluation and release preparation (D2) are complete.

## Quick start

> **Read [Responsible Use](#responsible-use) and [No Liability](#no-liability) first.** Generated
> URLs are synthetic and for defensive research only: never register, resolve, visit, host or
> send them.

1. **Install** (Python 3.11+; this installs `torch`, `numpy` and `tldextract` 5.3.1):
   ```bash
   git clone https://github.com/youssefismail-dev/limnex-uie.git
   cd limnex-uie
   pip install .
   ```
2. **Download the model** from the repository's *Releases* page: `limnex-uie-e16_b5.pt`
   (97 MiB), and the known-URL filter `limnex-uie-known-urls.bloom.gz` (34 MiB):
   ```bash
   curl -LO https://github.com/youssefismail-dev/limnex-uie/releases/download/v0.1.0/limnex-uie-e16_b5.pt
   curl -LO https://github.com/youssefismail-dev/limnex-uie/releases/download/v0.1.0/limnex-uie-known-urls.bloom.gz
   ```
   **Generation requires the filter:** the model sometimes reproduces a training URL — about 1 in
   100 raw phishing samples is a real phishing URL, and a rare benign copy can carry personal data
   that some real URLs hold (an e-mail address, a token) — and the filter removes them. `uie.load` refuses a
   model file or a filter that is not the published one (sha256 `98b10b46…` for the model,
   `3ef34d06…` for the filter's content).
3. **Generate and score:**
   ```bash
   uie generate --weights limnex-uie-e16_b5.pt --known-urls limnex-uie-known-urls.bloom.gz --label phishing -n 10
   uie score --weights limnex-uie-e16_b5.pt "example.com/login"
   ```
   or in Python:
   ```python
   import uie
   engine = uie.load("limnex-uie-e16_b5.pt", known_urls="limnex-uie-known-urls.bloom.gz")
   urls = engine.generate("phishing", n=10, seed=7)   # always gated; see engine.report_text()
   rows = engine.score(["example.com/login"])         # log P(url | class) and their difference
   ```
   **Before sharing any generated URL**, give the gate a current threat-feed snapshot
   (`--blocklist feed.txt`, or `blocked_domains=`): live threats are screened only when you supply
   one. Generation is free mode (the engine chooses the suffix and whether to write a path, a
   query and a root slash, from training statistics) and runs at about 2 URLs per second on a
   laptop CPU.
4. **More:** the model is described in [MODEL_CARD.md](MODEL_CARD.md); the published results are
   reproduced by following [REPRODUCE.md](REPRODUCE.md).

## Known limitations and future work

`e16_b5` has two documented limitations and two small open cells. Each is measured
against real held-out URLs, and each marks a boundary of the current design: a
character-level transformer trained from scratch on the project's corpus, with no
word list and no pretrained model. Each has been diagnosed, and the attempts to fix
it are recorded. **Resolving them is deferred to a later major version of the
engine; the approach has not been chosen yet.**

**1. Vocabulary: the word tell.** The engine uses fewer real words than real URLs
do ([research/E3_FINAL_REPORT.md](research/E3_FINAL_REPORT.md) §8).
- It writes 1.45 words per benign URL that never occur in real data, against 0.93
  for real URLs. In paths and queries, 84% of its words are known words, against
  94%.
- A detector that checks whether each word occurs in the real training corpus
  separates whole URLs from held-out real ones: excess over the real-vs-real floor
  +0.168 benign and +0.164 phishing, both CONVICTED. Character-level detectors do
  not (+0.027 benign, CLEARED; +0.044 phishing, UNDECIDED), and structural
  features alone give AUC 0.50 benign and 0.55 phishing. The difference is in
  vocabulary.
- Three attempts within the current design did not fix it: a learned memory
  layer, twice (V2-P1 and V2-P1b, both stopped at their registered memory
  checks), and word pieces (V2-P2), which made the tell larger
  ([research/v2/V2P2_RESULTS.md](research/v2/V2P2_RESULTS.md)).

**2. Phishing query strings in free mode: the one substantial gate failure**
(the report's §5.3).
- x = 0.0733 against the 0.05 bar, CONVICTED. It concerns the query string,
  present on about 4.7% of free-mode phishing URLs; every other free-mode phishing
  field is CLEARED.
- A decomposition of the cell (DX8) traced about 63% of the tell to the host and
  path written before the query.
- Real phishing reuses kit strings (40.3% of real phishing queries are exact
  training queries); the engine invents where real phishing repeats.
- It is not a data-volume problem: a plain character 6-gram model trained on the
  same phishing queries scores +0.036 where the transformer scores +0.124. The
  limit is in how the transformer generalises.

**3. Two small control-mode cells,** on benign URLs planned with both a path and a
query (P1Q1; the report's §§5.1–5.2).
- PTH: 0.0539, CONVICTED, 0.004 above the bar. Paths run about 10% shorter than
  real ones (19.7 against 21.7 characters); a read-only check found no structural
  violation to correct.
- SUB: 0.0431, below the bar but not proven at 95% (UNDECIDED).

**The human evaluation (2026-10-15; final).** Ten volunteers each judged 5 pairs of
same-class URLs (100 distinct URLs: 50 real held-out, 50 from `e16_b5`'s free mode;
pairs could be mixed, both real or both generated). They judged 55 of 100 URLs
correctly (95% interval 45–64%; chance 50%; p = 0.37); generated URLs were called
real 52% of the time, real ones 62%. That is no evidence that people can tell them
apart, but 100 URLs cannot show that they can't: a modest ability is not ruled out. This
is the project's final human evaluation: it was the largest feasible, and no larger panel
will be run (owner's decision, 2026-10-17).
Record: [research/eval_pages/HUMAN_PILOT_RESULTS.md](research/eval_pages/HUMAN_PILOT_RESULTS.md).

## Acceptance status

### Current: `e16_b5` (E3 final report, 2026-10-08)

Every cell is graded by real held-out URLs only. For each field (label, suffix,
subdomain, path, query) and population, a detector is trained to separate the
engine's strings from real ones; its AUC minus the AUC between two real samples is
the excess `x`, measured over 5 generations against a bar of 0.05: **CLEARED** if
`x` is proven below the bar, **CONVICTED** if proven above it, **UNDECIDED**
otherwise.

| | count |
|---|---|
| gated cells CLEARED | **29 of 32** (9 of 10 free, 20 of 22 control) |
| slash rows within tolerance | **8 of 8** |
| cells UNDECIDED | 1 (benign P1Q1 SUB, below the bar) |
| cells CONVICTED | 2 (free phishing QRY; benign P1Q1 PTH) |

Every cell and row: [research/E3_FINAL_REPORT.md](research/E3_FINAL_REPORT.md) §4.
The remaining cells are described under
[Known limitations](#known-limitations-and-future-work).

**Everything below in this section is the E5-era record (2026-09-23), kept as
history. Its numbers describe E5, not the current engine.**

### E5 under gate v2 (2026-09-23, superseded)

The canonical gate, repaired: **an independent floor per (generation, split)**,
G=5 x **D=10**, and the section-26.1 verdict computed by the gate itself
(SE = sd(generation means)/sqrt(G), t_crit 2.776). `best_e5.pt`, the adopted
`temperature` decoder, δ adopted, n_main 8000, bar 0.05. Registered in
[research/qry/PREREG_GATE_V2.md](research/qry/PREREG_GATE_V2.md); results and all
verification in [research/qry/GATE_V2_RESULTS.md](research/qry/GATE_V2_RESULTS.md).

| cell | x | SE | t | verdict |
|---|---:|---:|---:|---|
| benign LBL | +0.0055 | 0.0031 | −14.25 | ✅ CLEARED |
| benign SUF | +0.0149 | 0.0031 | −11.16 | ✅ CLEARED |
| benign SUB | +0.0067 | 0.0011 | −40.68 | ✅ CLEARED |
| benign PTH | +0.0298 | 0.0039 | −5.19 | ✅ CLEARED |
| **benign QRY** | **+0.0573** | 0.0072 | +1.01 | ⚠️ **UNDECIDED** |
| phishing LBL | +0.0204 | 0.0026 | −11.57 | ✅ CLEARED |
| phishing SUF | +0.0074 | 0.0029 | −14.87 | ✅ CLEARED |
| phishing SUB | +0.0302 | 0.0031 | −6.30 | ✅ CLEARED |
| **phishing PTH** | **+0.0443** | 0.0031 | −1.84 | ⚠️ **UNDECIDED** |
| **phishing QRY** | **+0.1333\*** | — | — | ❌ **CONVICTED** (pooled) |

\* Per generation NOT COMPUTABLE (180 query-bearing samples, below 250); the
pooled estimator convicts at D=10 and on its first 3 draws, and can never acquit.

**Roadmap-E3's acceptance gate is NOT passed:** one convicted cell, two undecided.
**One verdict changed from v1 — phishing PTH, CONVICTED to UNDECIDED.** Its v1
conviction rested on three split draws shared by every generation, a hidden
offset about eight times the SE v1 reported; with it removed, all five generation
means moved down together. The cell is **not cleared** (0.05 lies inside its
interval), the change corrects the measurement rather than the model, and the PTH
attribution branch stays closed. benign QRY moved from +0.0300 to +0.0573, as the
floor re-scoring predicted (+0.0619).

### Gate v1 readings (superseded as the acceptance reading, 2026-09-23)

Everything below comes from the v1 gate: floors seeded per split and shared by all
generations, D=3, t-values with sd/sqrt(G−1). The numbers still reproduce exactly
(`--estimator v1`) and the V3 check re-derived all 25 verdicts; they are kept as
the record, not as the current reading.

Canonical evaluation, `best_e5.pt`, seed 20260806, **n_main 8000, G=5 generations
x D=3 split seeds**, bar 0.05 (`uie/corpus/run_eval.py`; reports in
`data/e5_gate_n8000_d0/` and `data/e5_gate_n8000_dadopted/`). Full detail in
[E3_REPORT.md](E3_REPORT.md) section 30. δ is the adopted path-scoped EOS
bias (benign 0.22 / phishing 0.30).

Verdicts are the section-26.1 calibrated estimator: D splits averaged within a
generation, SE taken across the G generation means. **G=5 leaves that SE four
degrees of freedom, so the threshold is t > 2.78, not 2 sigma.**

> **Note, 2026-09-23 - SE convention.** The t-values here were computed with
> SE = sd(generation means) / sqrt(G - 1). Section 26.1 specifies sd / sqrt(G),
> the standard one-sample form, which gate v2 now encodes
> ([research/qry/PREREG_GATE_V2.md](research/qry/PREREG_GATE_V2.md)). All 25 published cells were
> re-checked under both: **no verdict differs.**

| field | benign δ=0 | benign δ=adopted | phishing δ=0 | phishing δ=adopted |
|---|---:|---:|---:|---:|
| LBL | +0.0301 ✅ | +0.0260 ✅ | +0.0187 ✅ | +0.0265 ✅ |
| SUF | +0.0029 ✅ | +0.0035 ✅ | +0.0003 ✅ | −0.0032 ✅ |
| SUB | +0.0391 ✅ | +0.0400 ✅ | **+0.0576 ❌** | +0.0651 ⚠️ |
| PTH | **+0.0767 ❌** | **+0.0728 ❌** | **+0.0787 ❌** | **+0.0834 ❌** |
| QRY | **+0.1196 ❌** | **+0.1233 ❌** | **+0.1485 ❌** | **+0.1488 ❌** |

**Roadmap-E3's acceptance gate is NOT passed.** Five field-class cells are
convicted: benign PTH, benign QRY, phishing SUB, phishing PTH, phishing QRY.
They are five failed cells, **not** five independent causal mechanisms; causal
attribution across them is open.

⚠️ phishing SUB at δ=adopted reaches t = 2.20, below the t > 2.78 threshold.
The confirmed conclusion for that cell rests on its δ=0 paired counterpart
(+0.0576, t = 3.66), not on the adopted-delta number.

#### Restated under the ADOPTED decoder (R4, 2026-09-23)

The table above is `decoder=typical`, which is **no longer the default**. The
same canonical run under the adopted `temperature` decoder is
`data/r4_temp_dadopted/` — same frozen `best_e5.pt`, same seed, same δ, same
n_main 8000 / G=5 × D=3, **verified identical floors**, so the two are paired and
only the decoder differs.

| cell | typical (published above) | temperature (ADOPTED) |
|---|---:|---:|
| benign LBL | +0.0260 ✅ | −0.0014 ✅ |
| benign SUF | +0.0035 ✅ | +0.0001 ✅ |
| benign SUB | +0.0400 ✅ | +0.0048 ✅ |
| **benign PTH** | **+0.0728 ❌** | **+0.0384 ✅ cleared** |
| benign QRY | +0.1233 ❌\* | +0.0300 ⚠️ gate D=3; **+0.0619 at D=30, UNDECIDED** — see below |
| phishing LBL | +0.0265 ✅ | +0.0144 ✅ |
| phishing SUF | −0.0032 ✅ | −0.0064 ✅ |
| **phishing SUB** | **+0.0651 ⚠️** | **+0.0332 ✅ cleared** |
| phishing PTH | **+0.0834 ❌** | **+0.0597 ❌** |
| phishing QRY | **+0.1488 ❌\*** | **+0.1431 ❌\*** |

\* pooled estimator — it can convict or abstain, never acquit.

**Two cells genuinely cleared: benign PTH and phishing SUB.** Both are canonical-
estimator comparisons at unchanged n, paired, with large t (−12.02 and −5.48).

**benign QRY is NOT cleared — UNDECIDED at +0.0619.** The gate's canonical cell reads
+0.0300, and that number is a **floor artifact**. Re-scored on the gate's exact
generations (all 15 of its draws reproduced to 4 decimals) with 30 split seeds
instead of 3, the floor falls from 0.5374 to 0.4973 and the cell reads **+0.0619,
SE 0.0085, UNDECIDED** (t = +1.40 against the bar). See
[research/qry/FLOOR_RESULTS.md](research/qry/FLOOR_RESULTS.md).

The gate's own standard error could not see this: floors are seeded per split, so
all five generations share the same three, and a floor offset common to every
generation never enters the across-generation SE. Here the hidden offset (0.040)
was four times the reported SE. It is largest at reduced n, i.e. in both QRY
cells. Whether to change the gate's floor seeding or D is an open decision.
*(Resolved 2026-09-23: gate v2 — see the current table above.)*

> **Correction, 2026-09-23.** An earlier version of this paragraph said excess
> "rises mechanically with n" and called benign QRY "not cleared" on that basis.
> That was never measured, and `uie/evaluation.py` assumes the opposite ("a
> reduced n is a weaker measurement, not a wrong one"). It is registered as **NS**
> in [research/qry/PREREG_NSWEEP.md](research/qry/PREREG_NSWEEP.md). It returned
> VOID under its registered rule; descriptively, excess does rise with n for fields
> with a tell, but **n does not explain this cell**: at the same n the sweep reads
> ~+0.107. The gate cell's floor (0.5374) sits above all 20 sweep floor draws at
> n=361 -- an unlucky floor, not a content gain, is the leading explanation. See
> [research/qry/NSWEEP_RESULTS.md](research/qry/NSWEEP_RESULTS.md).

**Remaining convictions under gate v1: phishing PTH and phishing QRY**, with
benign QRY unresolved rather than passed. *(Under gate v2: phishing QRY
convicted; phishing PTH and benign QRY undecided.)*
**QRY is the largest defect in the model**, and was measured here for the first
time. Previous reports printed it blank for want of samples, and the blank was
read as silence rather than as missing data. It fails on two independent axes:

| | benign | phishing |
|---|---:|---:|
| query rate, training corpus | 6.361% | 4.294% |
| query rate, validation corpus | 6.403% | 4.435% |
| query rate, generated (δ=0) | **2.49%** | **1.30%** |

Train and validation agree to within 0.15 points, so the emission deficit is not
a corpus mismatch: the model emits queries at roughly 40% (benign) and 30%
(phishing) of the rate present in its own training distribution. Separately, the
queries it does emit are its most distinguishable output. Whether one mechanism
causes both is the open question and is **not** assumed.

δ has **no statistically meaningful effect on any gated field** (all paired
|t| < 1.6). It is retained for the structural reason it was adopted, not for any
gate benefit.

Whole-URL AUC is reported as auxiliary and is **not** near chance: 0.726 benign /
0.899 phishing. The claim below about sorting generated from real URLs is the
project's target, **not a demonstrated result** - the human panel has never been
run. *(Since run: the final human evaluation, 2026-10-15, 55/100 — see
[Known limitations](#known-limitations-and-future-work).)*

What *is* demonstrated: host morphology, path-depth distribution, class
conditioning and phishing platform shares match held-out real data closely, with
99.8% / 97.4% novelty. What is not: lexical semantics of invented words.

The PTH residual tell is characterised and frozen in
[PTH_ATTRIBUTION.md](PTH_ATTRIBUTION.md); its dominant causal feature is
unresolved and that branch is closed.

---

## Why

Training data for URL security models is scarce, biased and stale. Phishing feeds
contain only what was *caught and reported*, which is a survivorship-biased
slice, and a URL is usually dead by the time it lands in a feed. That data is
also legally awkward to redistribute.

This project addresses the gap directly, and it is meant to be useful to anyone
working on DGA detection, domain reputation, registrar abuse, dataset
augmentation, or adversarial robustness for URL classifiers.

## Design in one page

- **The real web is the source of knowledge; curated corpora are curriculum.**
  No assembled dataset defines what "normal" looks like. Benign data comes from
  Common Crawl directly. *(Measured: curated benign corpora carry 12–17% bare
  homepages — two of them 98.6% and 100.0% — against the real web's 3.7%.)*
- **Structure is enforced only where it is unambiguous.** Path, query and
  fragment are separate components, so the model never generates `?` or `#` and
  cannot misplace them. Inside a component nothing is constrained, because
  deliberate grammar abuse is a real attack class.
- **The public suffix is a closed vocabulary**, sampled from an updateable table
  and embedded from its characters — so a new TLD works without retraining.
- **Creativity lives in inductive bias and decoding, not the loss function.**
  Novelty is free (the space is ~10¹⁵; a corpus is ~10⁶); the enemy is cliché,
  and cliché comes from the sampler. *(Measured since: in phishing the engine
  errs the other way. It invents where real phishing kits repeat; see
  [Known limitations](#known-limitations-and-future-work).)*
- **Freshness is versioned retraining, not continual learning**, because a model
  that keeps mutating cannot be benchmarked stably.

## Install

To use the engine, see [Quick start](#quick-start). For development (tests, corpus tools):

```bash
pip install -e ".[dev]"
pytest
```

Every test runs offline. The Common Crawl sampler and the live feed collector are
exercised against synthetic fixtures and throwaway git repositories.

## Reproducing the corpus

The full procedure — corpus, splits and the published gate results — is in
[REPRODUCE.md](REPRODUCE.md). In short: corpora are **not** committed. Common Crawl's terms do not clearly permit
redistributing derived datasets, so the project publishes the *recipe* instead: a
seeded sampler plus pinned crawl IDs. The phishing corpus rebuilds byte for byte; the Common
Crawl side rebuilds by the same method and scale but not as the same sample (REPRODUCE.md, 2a).

```bash
python -m uie.corpus.build --out data/corpus --blocks 200
```

That command rebuilds the original E0 benign corpus. The corpus the current
baseline trained on (`data/bundle2`, E3's two-corpus design) is built by the
modules `extract`, `merge`, `merge_phishing`, `build_split2` and `pack_bundle` in
`uie/corpus/`, each of which documents its inputs; `e16_b5` also trained on E15's
added benign query stream.

<!-- RESPONSIBLE-USE:BEGIN -->
## Responsible Use

**Read this before downloading, installing or using any part of this project.** By doing any of
those things — with the code, the model weights or anything they generate — you confirm that you
have read this section and the No Liability section below, and that you accept them.

Limnex UIE generates realistic URLs, including **phishing-style URLs**. It exists for one purpose:
**defensive security research and education** — testing, training and evaluating phishing
detectors, studying how URLs are structured, and teaching. It is not built, offered or supported
for anything else.

**Intended uses**
- Training, testing and red-teaming phishing detectors and filters that you own or are explicitly
  authorized to test.
- Measuring where detectors fail, and research on how URLs are structured.
- Teaching and coursework.

**Prohibited uses — the authors do not authorize, support or condone, and you must not:**
- use the code, the model or any output for phishing, fraud, impersonation, social engineering,
  spam, malware delivery or any other deception of real people or organizations;
- register, buy, host, resolve, visit, publish as working links, or send to anyone any URL or
  domain the engine generates;
- impersonate, or imply an affiliation with, any real brand, organization or person;
- test any system, person or organization without explicit, prior, written authorization;
- distribute bulk collections of generated phishing-style URLs;
- use it in any way that breaks a law, a regulation, the rights of others, or the terms of a
  service that applies to you.

**What the project does to limit misuse.** It ships the tool, not the ammunition: no bulk corpus of
generated phishing URLs is published. Generation runs offline and never queries DNS, and the project
never registers or resolves a generated domain. Every generated batch passes a filter for validity,
duplicates, copies of known training URLs and — when you supply a current threat-feed snapshot —
known live threats. Before sharing any generated URL, screen it against a current threat feed and
replace any domain that could belong to someone.

**Generated URLs are synthetic.** Any resemblance to a real domain, brand, organization or person
is coincidental and unintended. Brand names can appear because the model learned from real
phishing; this implies no affiliation or endorsement, and says nothing about the safety of that
brand's own sites. All trademarks belong to their owners.

**Reporting.** If you find a problem — for example a generated URL that matches a live threat —
report it privately through the repository's *Security* tab (a private advisory), not in a public
issue.

## No Liability

THE CODE, THE MODEL WEIGHTS, THE DOCUMENTATION AND EVERYTHING THEY PRODUCE ARE PROVIDED "AS IS",
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING ANY WARRANTY OF
ACCURACY, RELIABILITY, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT.

TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, LOSS OR OTHER LIABILITY — DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, CONSEQUENTIAL OR EXEMPLARY — WHETHER IN CONTRACT, TORT (INCLUDING NEGLIGENCE)
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THIS PROJECT, ANYTHING IT PRODUCES, OR THE
USE OR MISUSE OF EITHER BY YOU OR BY ANYONE ELSE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

- **You use this project entirely at your own risk.** You alone are responsible for your use, for
  anything you create with it, and for complying with every law that applies to you — including
  laws on computer misuse, fraud, privacy, data protection, trademarks and export.
- **Indemnity.** To the fullest extent permitted by law, you agree to indemnify and hold harmless
  the authors, contributors and copyright holders against any claim, loss or expense arising from
  your use or misuse of the project.
- **Not a security product.** Its outputs and scores can be wrong; it is not a phishing detector;
  nothing in this repository is legal, security or professional advice.
- **Relation to the license.** The license is the Apache License, Version 2.0 (`LICENSE`). Its
  Section 7 (Disclaimer of Warranty) and Section 8 (Limitation of Liability) apply in full; this
  section adds to them and does not limit them. The Responsible Use section states the uses the
  authors intend and those they prohibit.
- **Severability.** If any part of this section is held unenforceable, the rest remains in effect.
<!-- RESPONSIBLE-USE:END -->

## Relationship to other projects

The engine is independent. It never imports from, and is never graded by, any
consumer. SNOW Guard AI is its first reference integration — not its reason to
exist.

## Acknowledgements

Limnex UIE was built with the help of AI coding tools, including Anthropic's Claude Code.

## License

The code and the model weights are licensed under the **Apache License, Version 2.0**
([LICENSE](LICENSE)); attribution and notices are in [NOTICE](NOTICE). The training data is not
redistributed. Read [Responsible Use](#responsible-use) and [No Liability](#no-liability) above.
