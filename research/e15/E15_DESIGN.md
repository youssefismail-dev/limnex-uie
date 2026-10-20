# E15 — is new query data the right next lever? The evaluation

> **Note, 2026-10-05:** the owner did not adopt §3's recommendation (run E15-A
> first). `PREREG_E15.md` §9 (Amendment 1) records that decision; E15-B is now
> the experiment.
> - This document's evidence and verdict stand unchanged, as the executor's
>   assessment.
> - One factual correction is made there: §1.3's "16 crawls, 200 index blocks
>   each" describes E0. The training corpus is corpus2: the 8 live crawls,
>   suffix-stratified blocks (§9.2, §9.4.1).

- **Written:** 2026-10-05, for the owner's decision 3 (`research/native/DECISION_2026-10-05.md`
  §1).
- **Inputs:** the registered record only:
  - E13-D (`E13D_RESULTS.md`: DX1–DX8 and the post-hoc copy analysis);
  - E13's N (`E13_RESULTS.md` §5, `data/e13/n_measure.json`);
  - E14 (`E14_RESULTS.md`, the training records);
  - the corpus manifests (`data/corpus/*.manifest.json`, `data/bundle2/bundle.manifest.json`).
- **New measurements in this document:** none.
  - Two comparisons are new arithmetic on records that already exist: E14's per-field
    fit against `e13_ctx`'s, and the phishing feed's growth rate.
- **The registration this leads to:** `PREREG_E15.md`.

## 0. In plain words

- **The question.** Would adding a new source of real URLs with queries make the
  model write more realistic queries, without the path and subdomain damage the
  query weight caused?
- **The answer, on the evidence we have: not yet, and probably not.** Three facts
  stand against it:
  1. **The data we have is not used up.** In E14, putting more emphasis on the
     *existing* queries made the model predict *unseen* real queries much better,
     and moved two of the three failing query cells. A model that had exhausted
     its data could not do that.
  2. **Every earlier test of "is there enough data?" said yes, there is.** The
     training URLs are indistinguishable from the held-out URLs the gate tests
     against. A simple copying model trained on them passes the detector in the two
     small cells.
  3. **The real limit E14 exposed is the model's fixed budget, not the data.** When
     queries got more of the training signal, every other field got worse —
     labels, paths, subdomains, fragments — on training rows too, 14–19 times
     beyond run-to-run noise.
     - Adding query-rich rows spends that same budget on queries, so it would be
       expected to cause the same damage.
     - The only way to add data without that damage is to keep the query share of
       training fixed. Then any gain has to come from the *new content itself*,
       and nothing yet shows that the content is what's missing.
- **The one open door.** Every "is there enough data?" test so far used a stand-in
  learner (a 6-gram counter), not the transformer itself.
  - One cell, benign URLs with a path and a query (P1Q1), is the cell where data
    could still matter:
    - it is the only query cell more practice did not move;
    - its stand-in curve was still sloping;
    - a third of its held-out query patterns never appear in training.
- **So the stronger next step is to measure, not to buy data.** Retrain the model
  on *half* of the existing query content, with the same total query practice, and
  see whether the query cells get worse.
  - It needs no new source, changes nothing else, and costs one Kaggle session.
  - **If halving hurts,** doubling with genuinely new same-population data is
    justified, with a measured expected effect. `PREREG_E15.md` §4 already designs
    that experiment.
  - **If halving does not hurt,** doubling cannot help. The data direction closes
    on the transformer's own evidence, not a stand-in's.
- **Either way, E3 is not reachable in E15.** Query data cannot touch the three
  path and subdomain cells, and there is no same-population phishing source large
  enough to matter.

## 1. The four distinctions the owner asked about

### 1.1 More exposure to existing query patterns

**It works for two of the three cells, and it costs the rest.**

| | free phishing QRY | control benign P0Q1 QRY | control benign P1Q1 QRY |
|---|---|---|---|
| gate x, `e12_final` / `e12_d1p` (adopted decode) | 0.1141 / 0.1123 | 0.0621 / 0.0624 | 0.0559 / 0.0520 |
| N's A2: 4× exposure, query rows only (Q-VAL) | 0.1141 → 0.0551 | 0.0583 → 0.0276 | 0.0519 → 0.0492 |
| E14's Q: query weight 4 in the full mix (gate x) | 0.1141 → **0.0830** | 0.0621 → **0.0365 CLEARED** | 0.0559 → 0.0579 |
| held-out query fit under Q (nats per character) | −0.185 (phishing) | −0.073 (benign) | (benign, pooled) |

- **More practice on the same data improved *held-out* query fit.** It did not just
  improve the fit on the training rows.
  - So the existing data still holds query signal the baseline does not extract.
  - The query deficit in these two cells is under-learning, not missing data.
- **Benign P1Q1 QRY does not respond.**
  - Under A2, its Q-TRAIN falls more than its Q-VAL (0.0437 → 0.0347 against
    0.0519 → 0.0492). The model moves toward its training queries, not toward
    held-out ones.
  - This is the pattern data-limitation would produce. It is the reason §3 leaves
    this cell's door open.

### 1.2 Genuinely new query content and coverage

The record has measured this from four sides, and none shows a data cause.

- **No population gap between the training and validation data (DX1).** Real
  training queries pass the detector against the real validation queries:
  x_TV is −0.0019, −0.0033 and −0.0029 in the three cells. So a model that wrote
  queries exactly like its training data would pass.
- **The model does not reproduce even its training data (DX2).** Its queries are
  about as detectable against its own training strings as against validation.
  - The ratios are 0.94, 0.85 and 0.95.
  - That is a learning gap. A data gap would show the model close to its training
    data and far from validation.
- **The stand-in's data curve (DX3, an order-6 n-gram):**

  | | free phishing | benign P0Q1 | benign P1Q1 |
  |---|---|---|---|
  | training rows (domain frame) | 8,272 | 14,906 | 42,862 |
  | validation templates never seen in training | 27.8% | 19.0% | 32.7% |
  | coverage gained per doubling of data | +4.1 points | +3.2 | +4.3 |
  | stand-in's x at half → full data | 0.035 → 0.036 | 0.009 → 0.015 | **0.073 → 0.067** |

  - The stand-in is flat between half and full data in the two small cells.
  - **In P1Q1 it is still falling:** DX3's slope s = +0.0052 of improvement per
    doubling, t 1.30. That is not significant, but it is not flat.
  - DX3 itself warned: "It does not stand for the transformer's own curve."
- **The model already writes too many *new* strings, not too few** (E13-D §5):

  | | exact copies, real / model | never-seen strings, real / model |
  |---|---|---|
  | free phishing | 40.3% / 16.7% | 50.6% / 77.2% |
  | benign P0Q1 | 35.7% / 20.5% | 39.8% / 55.1% |
  | benign P1Q1 | 20.3% / 13.3% | 59.2% / 72.4% |

  - Novelty is not what it lacks.
  - Re-drawing its output to the real copy share makes the tell *bigger* (QC1-2).
    The strings it copies are the wrong ones, at the wrong frequencies.
- **Its parameter names are too common, and practice does not fix that** (N2a's M2,
  re-read in E14):
  - generated rare-name share (names seen 1–19 times): 0.0557 (benign) and 0.0956
    (phishing), against a real 0.2198 and 0.3766;
  - under Q: 0.0503 and 0.1020;
  - **[Reading]** New data would add *more* rare names to learn. It would not teach
    the model to write the rare names it already has. This is a property of how the
    model samples, not of the data.

### 1.3 Distribution shift from a new source

**The gate tests against one population, so only data from that population can
help.**
- The detector compares generated queries with **this corpus's own held-out URLs.**
- A source from another population moves the model's queries toward *that*
  population, and the tell can rise.

**What "the same population" means here** (from the manifests):

| class | the gate's population | a same-population source | volume it could add |
|---|---|---|---|
| phishing | the OpenPhish public feed, first seen 2024-10-12 → 2026-08-03 (head 48f7a98, 354,143 URLs; 80/10/10 host split) | only the same feed's later entries. PhishTank, URLhaus, PhishStats and others are different populations (different reporting, kits, brands) | the cell's 8,272 training rows came from about 15.7 query rows per feed-day. One doubling would take about **1.5 years** of new feed, and would come from a later time window |
| benign | Common Crawl: 16 crawls (8 from 2024-42 to 2026-30, 8 from 2019-04 to 2022-49), 200 index blocks each, one URL per registrable domain (`uie/corpus/build.py`) | more index blocks from **the same 16 crawls**, by the same builder, new registrable domains only | P1Q1 is about 4.7% of domain-frame rows. Doubling its 42,862 rows needs about 0.9M new registrable domains, roughly the size of today's benign sample again. Large, but feasible |

- **Benign:** genuinely same-population new data exists.
- **Phishing:** in practice it does not, at a volume that could matter.
- The far cell, free phishing QRY, is also mostly a *context* problem (DX8,
  CONTEXT-LED). Given a real host and path, its queries already read 0.046. New
  query content does not change the model's own hosts and paths.

### 1.4 The path and subdomain trade-off

**E14 measured the cost at the level of the model, not just the detector.** Below
is each field's validation fit, change against `e12_final`.
- `e13_ctx` is the noise reference: the same recipe and the same batches, changed
  only on phishing query rows.
- So its benign fields show what a retrain moves by itself.

| field | `e13_ctx` (noise reference) | `e14_q` (more query emphasis) |
|---|---|---|
| benign LBL | +0.002 | **+0.038** |
| benign PTH | +0.002 | **+0.028** |
| benign QRY | +0.014 | −0.086 |
| phishing PTH | +0.012 | **+0.055** |
| phishing SUB | +0.002 | **+0.041** |
| phishing QRY | +0.024 | −0.166 |

- The fit of every other content field fell, on training rows too, so this is not
  overfitting. The query gain was paid for from a fixed budget: 14.4M parameters
  and 20,000 steps.
- **[Reading, not a registered capacity test]** The model is at a frontier where
  fields compete for the same capacity. Any change that gives queries a larger
  share of training is expected to repeat this:
  - a loss weight;
  - oversampling;
  - **or adding query-rich rows to the pool.**
- **Consequence for the owner's idea:**
  - added at the same batch budget, new query rows would carry E14's cost;
  - added with the query share held fixed (matched exposure), they cannot carry
    it, but then any gain must come from the new content alone, which is §1.2's
    unproven link.
  - **Matched exposure is the only form of new data consistent with both E14 and
    the owner's rejection of the trade-off,** so it is the form designed in
    `PREREG_E15.md` §4.

## 2. Verdict

- **New query data is not the evidence-justified next lever today.**
  - **Free phishing QRY and benign P0Q1 QRY** are under-learning, not data-starved:
    §1.1, DX1, DX2, and DX7's small train–validation gap. A same-population phishing
    source does not exist at a useful volume (§1.3).
  - **For benign P1Q1 QRY** the case is open, not made: §1.1's train-not-validation
    pattern, DX3's still-sloping stand-in curve, 32.7% unseen templates.
- **It fails the owner's own standard for data** (`DECISION_2026-10-01.md` §2). New
  data must document "the deficiency found" and "why it explains the failure", and
  "do not add data merely because E3 failed".
  - No registered reading shows a data deficiency in any query cell.
  - The one measurement that could, the transformer's own data curve, has never been
    taken.

## 3. The stronger, evidence-based alternative: measure the transformer's own data curve first

**E15-A** (`PREREG_E15.md` §2–3): retrain `e12_final`'s exact recipe twice. Each
time, the query content of *half* the query-bearing rows is learned at double
weight, and the other half's query content is not learned at all.
- **The two halves** are complementary, split by the corpus's own split key.
- **Everything else is held fixed:**
  - the same batches;
  - the same total query practice;
  - every other field's training untouched.
- **The only thing that changes is how many distinct queries the model learns
  from.**

**Why this, and not a new-data arm straight away:**
- **It answers the owner's exact question:** genuinely new content against more
  exposure. Exposure is held fixed, so any change is the amount of distinct
  content.
- **It tells us in advance what the new-data arm can achieve.**
  - Learning curves flatten as data grows. So the damage from halving bounds,
    from above, the gain from doubling.
  - If halving costs less than one retraining's noise (0.0141), doubling cannot buy
    more.
- **It needs no data sourcing, and has no shift risk.** One Kaggle session.
- **It is on the critical path whichever way it reads:**
  - **DATA-LIMITED in a benign cell:** E15-B is licensed, with a measured ceiling.
  - **NOT DATA-LIMITED:** the data candidate is ruled out by a registered,
    transformer-level reading. That is the missing piece in R3's first unlock case
    (`DECISION_2026-09-29.md` §4, "the candidates have each been tested or ruled
    out by a registered reading"). R3 remains the owner's decision.

**What E15-A is not.** It removes data. It is never a candidate, never adopted, and
not a form of the oversampling the owner excluded as the intervention.
- It reweights existing query content only to hold exposure fixed while halving
  the content.
- Without that, the reading would mix data with exposure, the confound E14 showed
  is large.
- The owner approves it explicitly at gate 0.

## 4. Alternatives considered, and why each is not the next step

| alternative | why not now |
|---|---|
| a new-data training arm straight away | no registered data deficiency (§2). It cannot be attributed without matched exposure. It risks shift (§1.3). Sourcing about 0.9M benign domains is days of work, for a gain E15-A could show is zero |
| duplicate or oversample existing query rows | excluded by the owner; it is exposure, and E14 showed its cost |
| more training steps | R2 REJECTED ("more optimisation is not the repair") |
| more capacity | R3 LOCKED (last resort). E14's trade-off is consistent with a capacity limit, but it is not a registered reading. E15-A supplies one of the readings R3's first case needs; the decision stays the owner's |
| decoder levers for the query cells | outside the owner's override (slash and field-led failures only); D2's calibration had no effect |
| state data for benign P1Q1 (paths and subdomains too) | broader than "query data only". If E15-A reads benign P1Q1 QRY DATA-LIMITED, the same question for P1Q1's SUB and PTH is the natural next registration, not part of E15 |

## 5. Internal review

Each element of the design was checked against the evidence and the standing rules
before registration.

| check | result |
|---|---|
| every factual claim above traced to a registered record or manifest | yes. Sources are named in each section, and the two new arithmetic comparisons are stated as such |
| the design changes one thing per arm | E15-A: the amount of distinct query content. E15-B: new same-population content at matched exposure |
| exposure held fixed, so the E14 confound cannot recur | E15-A: same batches, query mass matched per class (V-mask). E15-B: identical existing-row batches plus a side stream; query mass matched; non-query mass identical |
| comparisons against frozen baselines | E15-A against E13's frozen `e12_final` N reference sets (raw decode, the same instrument and seeds). E15-B against `e12_d1p` (E14's files; the adopted decode) |
| the slash fix preserved | every E15-B arm decodes with D1 + D1′. E15-A reads only query cells on N's raw instrument, so it matches its reference |
| no gate, threshold or tolerance changed | yes. The decision margin is the existing retraining allowance (0.0141). E15-B's gates reuse DX1's and DX3's rules |
| new thresholds, declared as new | E15-B's G3, "no other field's validation fit worse by more than 0.01 nats/char": 5× the benign noise reference (0.002). E14's breaches were 0.027–0.055 |
| no capacity, steps, decoder or broad corpus change | yes. E15-B adds a separate, query-only stream; the existing corpus is untouched |
| nothing optimised against the metric | no sweep; one split, two halves; every number fixed in the dry run, before training |
| what E15 cannot do, stated | E3 is unreachable in E15 (§0) |
| predictions written down | `PREREG_E15.md` §5, including the most likely outcome, "data direction closed" |
