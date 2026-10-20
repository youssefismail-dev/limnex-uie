# QG — the free-mode query-emission gap: investigation plan and QG1 registration

- **Status:** REGISTERED 2026-09-28, before any QG code or measurement.
- **Basis:** the owner's decision of 2026-09-28 (`research/native/DECISION_2026-09-28.md`
  §1.3) and E9's reading (`research/native/E9_RESULTS.md`).
- **Evidence tags:** as in `NATIVE_REPAIR_PLAN.md`: [CONFIRMED], [HYPOTHESIS],
  [DIAGNOSTIC], [DECISION].

**Rules for the whole investigation** [DECISION, owner]:
- **Diagnostics only.** Nothing here changes the baseline (`uie/baseline.py`), a
  library default, an instrument, a threshold or a published result. An
  interesting diagnostic result is not a reason to modify the baseline.
- **Registration first.** Every experiment states its hypothesis, a measurable
  prediction, a criterion where one applies, and a decision rule, and is committed
  before it runs.
- **Decompose before fixing.** No training fix is proposed before the gap is
  decomposed. §7 of PREREG_E9 did not license query data, so more query data is
  not assumed to be the answer.
- **The owner's approval** is needed for any training or generation run (Kaggle),
  each registered separately. R3 (capacity) stays LOCKED
  (`NATIVE_REPAIR_PLAN.md` §8) unless the owner decides otherwise.

## 0. In plain words

- A query can only begin at one kind of moment: the model has finished writing the
  site's name, or has finished writing the path. At that moment it either writes
  the special "query starts here" token, or it does not.
- So a missing query can be lost in only three places:
  1. **The decision.** Shown the start of a real URL, does the model give "start a
     query now" the right probability? This is like a shop assistant who, even with
     real customers, offers an extra less often than those customers want it.
  2. **The model's own starts.** When the model writes its own names and paths, it
     may write the kind after which real URLs rarely have queries. This is the same
     assistant, handed different customers from the real shop's.
  3. **The dice.** The sampler turns probabilities into actual tokens. At
     temperature 1.0 it should lose nothing, but truncated or empty queries could
     lose a little, so it is checked.
- QG1 measures how much of the shortfall each place explains, with confidence
  intervals, before anyone touches training.

## 1. The gap [CONFIRMED, E9]

`e9_ln` / raw, free mode, 5 generations × 8,000 URLs per class (`data/e9/score_report.json`):

| class | URLs with a query | 95% CI | registered band | band centre | data rate D (rows ≤ 256 tokens) | G / D |
|---|---|---|---|---|---|---|
| benign | 4.58% (1,833 / 40,000) | 4.38–4.79% | 5.96–7.28% | 6.62% | 6.53% | 0.70 |
| phishing | 2.95% (1,178 / 40,000) | 2.78–3.12% | 4.21–5.15% | 4.68% | 4.51% | 0.65 |

- **Sources:**
  - band centres: `name_train` under the shipped suffix table (`PREREG_E8` K1b);
  - D: the same, restricted to rows that fit the model's 256-token limit
    (`FINDING_LENGTH_WEIGHTING.md` §2.3).
  - The two differ by 0.09 and 0.17 points. That difference is itself measured in
    QG1 (§5.4); it is not assumed to come from the length limit.
- **The gap is far above the known noise:**
  - sampling CI about ±0.2 points;
  - three trainings of one recipe differ by 0.20–0.31 points in free query
    emission (`E9_RESULTS.md` §10);
  - the gap is about 2.0 points (benign) and 1.6 points (phishing).
- **Paths are not the problem:** free path emission is inside its band (75.58%,
  38.20%).
- **What the gap costs the gate** (power):
  - the benign QRY cell compares only 366 generated queries per draw;
  - phishing QRY is NOT COMPUTABLE, because only 1 of 5 generations reaches the
    scorer's minimum of 250 queries.
  - At the band centre, a generation of 8,000 would carry about 530 benign and 374
    phishing queries.

## 2. Where a query comes from [CONFIRMED, code]

**Facts from the code:**
- **Presence is one token.** A query exists exactly when the `<QRY>` marker is
  emitted. The tokenizer opens `<QRY>` only for a non-empty query, which is also
  the gate's definition (`uie/tokenizer.py` `encode`; `uie/evaluation.py` `FIELDS`).
- **Two routes.** `<QRY>` follows either the host (`<LBL>`/`<SUB>`/`<PRT>`: no path
  body) or a path body (`<PTH>…`).
- **Free mode:**
  - the suffix is drawn from the shipped table and injected as
    `<SUF>…<LBL>` when the model reaches it (`uie/model.py` `generate`);
  - the class and the domain frame are supplied, and both presence controls are
    `?`;
  - the raw decode is plain multinomial sampling at temperature 1.0
    (`decode_temperature`), with no bias, truncation of the distribution or
    resampling.

**The decomposition.**
- **The query hazard of a URL.** Take a URL x encoded as in free mode. Its decision
  positions run from the `<LBL>` marker to the last position before the first
  `<QRY>`, `<EOS>` or `<FRG>`. Then

      h(x) = Σ over decision positions i of p(<QRY> | x up to i),    y(x) = 1 if x has a query

- **Four rates,** each a mean over URLs:

| | over real rows (the data) | over the model's own free URLs |
|---|---|---|
| observed queries, mean y | **D** | **G** |
| query hazard, mean h | **T** | **S** |

- **The identity.** Exactly,

      G / D = (T / D) × (S / T) × (G / S)
               decision   own starts   sampler

- **Why each factor means what it says.** At each step a query begins with
  probability p(`<QRY>`), and the walk stops at the first `<QRY>`, `<EOS>` or
  `<FRG>`. So the expected number of query starts equals the expected summed hazard
  over the steps taken (optional stopping).
  - **decision = T / D.** If the model were the data's own distribution, T would
    equal D. So T / D is the model's calibration on real prefixes. The ratio lives
    in the families of E9's R1.
  - **sampler = G / S.** If the sampler draws exactly from the model, G equals S in
    expectation. So G / S is the sampler's faithfulness, including anything the
    gate's reading of the URL string loses.
  - **own starts = S / T.** This is the hazard the model spends on its own prefixes
    against the hazard it spends on real ones. The model is held fixed, and only
    the prefixes differ.
- **The factors are split by route,** host and path, which locates the loss.

## 3. The mechanism families the owner named, and what bears on each

| family | what it would mean here | QG1 evidence | what QG1 can conclude | what would isolate it (a later stage, registered then) |
|---|---|---|---|---|
| sampling | noise, the suffix mix, the seeds | G's CI; spread between generations; everything stratified by the shipped table's suffix groups | whether the gap is noise or an artefact of the suffix mix | — |
| decoder | the sampler departs from the model (truncation at 256 tokens, empty queries, loss in the round trip) | G / S; truncation and empty-query shares | whether generation draws the model's own distribution | regenerating a subsample with raw token ids and the truncation trace |
| conditioning / prefix | supplied tokens (frame, suffix, `?` controls), or the model's own prefixes | T is measured under free mode's exact conditioning; S / T by route | whether the model's own prefixes carry less query hazard than real ones | decomposing S / T by context; a prefix-swap diagnostic (real host with a generated path, and the reverse) |
| objective / loss | what the objective still does to the decision | T / D and S / T for `e9_base` against `e9_ln`: the same start, one flag | how much of each factor E9's objective moved | — (E9 was the single-change test) |
| data distribution | training data under-represents queries relative to the target | D with and without the 256-token limit; train against validation; the page frame's rate | whether the training distribution itself sits below the target | a data-side arm, only by the owner's decision (§7 licensed none) |
| capacity / optimisation | the model cannot or does not fit the decision | T / D on training rows (tens of thousands of query events) against validation; E7 as a second old-objective training | whether the model fits the query start on its own training rows | a capacity arm (R3 is LOCKED; the owner decides) |

QG1 separates the **decision** from the **own starts** from the **sampler**. Inside
the decision factor, it cannot by itself separate data, capacity and residual
objective effects; that is QG2-D's job, and only if QG1 licenses it (§5.6).

## 4. Power [CONFIRMED arithmetic; the CIs themselves come from the bootstrap]

- **G:** 40,000 URLs per class. Half-width of the 95% CI: 0.20 points (benign),
  0.17 (phishing).
- **T / D:** the teacher-forced rows are not limited by how many queries the model
  writes. 40,000 real rows per class give about 2,600 benign and 1,800 phishing
  query events. The expected CI half-width on T / D is about ±4% (benign) and ±5%
  (phishing).
- **On a log scale the gap is** ln 0.70 = −0.36 (benign) and ln 0.65 = −0.43
  (phishing). A factor must fall below about 0.93 to be told apart from 1, which is
  about a fifth of the benign gap. So QG1 resolves each factor's share of the gap
  to roughly ±0.15–0.2.
- **The registered rule:**
  - a share whose 95% CI is wider than ±0.25 is reported as **not resolved**;
  - a cell with fewer than 100 query events, real or generated, is pooled and not
    read on its own.

## 5. QG1 — decomposition on the baseline [DIAGNOSTIC; laptop; no training, no generation]

### 5.1 Inputs

| input | exactly |
|---|---|
| baseline model | `data/e9/train/e9_ln/best.pt`, checked by `uie.baseline.BASELINE` (weights fingerprint) |
| contrast models | `data/e9/train/e9_base/best.pt` (the same start, the old objective), checked against E9's provenance; `data/best_e7.pt` for T / D only (a second old-objective training; its free files use the resting decode, so it has no raw S) |
| real rows, training | `name_train` / `phishing_name_train` (the domain frame the free target is defined on); rows longer than 256 tokens are dropped, since the model cannot read them; 40,000 rows per class, drawn with seed 20260928 |
| real rows, validation | `name_val` / `phishing_name_val`; 10,000 per class, or all if fewer; seed 20260928 |
| generated URLs | E9's free files at the raw decode, checked against `data/e9/gen/PROVENANCE.json`: `free_e9_ln_raw_*` and `free_e9_base_raw_*`. G uses all 40,000 per class; the hazard pass uses 20,000 per class (seed 20260928), with G re-counted on the same 20,000 for G / S |
| encoding | `tok.encode(url, label, frame="domain", hide_fields=("PTH", "QRY"))`: free mode's exact conditioning |

### 5.2 Per URL

Recorded for every URL:
- its suffix group;
- y and h on each route;
- whether it has a path, and the path body's length in tokens;
- its total length;
- flags: re-encoding failed, too long, a `?` with an empty query.

The route of a decision position is **host** before the `<PTH>` marker and **path**
from it on.

### 5.3 Estimands and CIs

- **Stratified by suffix group.**
  - Groups: the class's 20 largest suffixes in the shipped table, plus "other".
  - Each group's means are weighted by the shipped table's share, so D, T, S and G
    are all under the free target's suffix mix.
  - Unweighted values are reported alongside.
- **Primary:** D, T, S and G, the three factors, and their log-shares of the gap:

      share = ln(factor) / ln(G / D)          the shares sum to 1

- **By route:**
  - the three factors on each route;
  - each route's part of the gap, (D_route − G_route) / (D − G).
- **CIs:** percentile, from 1,000 bootstrap draws (seed 20260806) that resample URLs
  within each suffix group, independently for real and generated URLs.

### 5.4 Descriptive checks (registered, not decision criteria)

- **Data:**
  - D over the whole `name_train` pool under the shipped table: all rows, and rows
    of at most 256 tokens. This reproduces or corrects the 6.62 / 6.53 difference.
  - D on validation against training.
  - The page frame's query rate (`structure_train`), for context.
- **Decoder:**
  - the share of generated URLs with a `?` but an empty query;
  - the share whose re-encoding reaches 256 tokens (truncated);
  - re-encoding failures.
- **Sampling:** G per generation (g0–g4) against the binomial spread; G per suffix
  group against D.
- **Objective:** the three factors for `e9_base`, and each factor's ratio `e9_ln` /
  `e9_base`, with CIs.
- **Capacity:** T / D on training against validation rows; T / D for E7, as a second
  old-objective training.

### 5.5 Predictions, stated before any measurement [HYPOTHESIS]

- **The sampler loses nothing material.** G / S lies within [0.97, 1.03], and
  truncated plus empty queries are under 0.1 points. Reason: the raw decode is plain
  multinomial sampling (code).
- **The objective moved the decision.** T / D for `e9_base` is below T / D for
  `e9_ln`. Reason: the old weighting under-counted the longer, query-bearing rows
  (FINDING §2.3).
- **The length limit lowers D** by 0.05–0.3 points. Reason: over-long rows carry
  long queries (PREREG_E9 D7).
- **Which factor leads, decision or own starts: no confident prediction.** T / D
  for `e9_ln` is expected within [0.80, 1.05].
  - **For calibration (T / D near 1):** token weighting calibrated F1–F3 on real
    prefixes (E9 K9a).
  - **Against:** the query start competes with path continuation and with the end
    token, and its rate may differ between the frames the model is trained on.
  - **For own starts leading:** the frozen delta, which only discourages ending
    inside a path, raises query emission by 0.43 / 0.61 points (`E9_RESULTS.md`
    §4). So query starts depend on where the model's paths go.

### 5.6 Reading and decision rule

Benign decides. Phishing is reported and its agreement noted.

| reading | condition | licenses next |
|---|---|---|
| SAMPLER | G / S's 95% CI excludes 1 **and** its share is at least 0.2 in size | an instrument check first: regenerate a subsample with raw token ids and the truncation trace. No model question until it is resolved |
| DECISION-LED | not SAMPLER; the decision factor's share ≥ 0.5 **and** T / D's CI excludes 1 | QG2-D: locate the decision deficit among data distribution, residual objective and capacity; laptop diagnostics first |
| OWN-STARTS-LED | not SAMPLER; the own-starts factor's share ≥ 0.5 **and** S / T's CI excludes 1 | QG2-E: which features of the model's own prefixes carry less query hazard (by route, path presence and length, suffix; the prefix-swap diagnostic); laptop only |
| SPLIT | none of the above; the CIs of T / D and S / T both exclude 1, and each share is ≥ 0.25 | QG2-D and QG2-E, in the order of their shares |
| UNRESOLVED | anything else | reported; the owner decides |

- **No reading licenses a training run.** Any training or generation is registered
  later and needs the owner's approval.
- **A reading licenses the next question, never a fix.**

### 5.7 Validity (checked before any number is read)

| code | check | on failure |
|---|---|---|
| VQ1 | the models' fingerprints and the free files' hashes are the registered ones | stop; nothing read |
| VQ2 | the hazard code, on a model small enough to sample heavily: S and G agree within Monte Carlo error on its own raw samples, which tests the identity end to end | stop; fix the code under an amendment |
| VQ3 | on hand-built sequences with known probabilities, h, y and the routes come out exactly | same |
| VQ4 | the sample's post-stratified D is within ±0.3 points of FINDING §2.3's D | reported; the stratification is examined before any reading |

### 5.8 Not claimed by QG1

- a fix, or a mechanism beyond the factor it locates;
- anything about query content (realism): this is presence only;
- that a factor's value would hold for another training.
  - QG1 reads one training per objective.
  - E7 and `e9_base` give two old-objective samples of T / D, for spread.

## 6. Later stages (named, not registered)

Each stage is registered on QG1's reading. None may change the baseline.
- **QG2-D** (if the decision leads):
  - the page frame's query rate, and how much the frame token moves the hazard
    (frame leakage);
  - the decision on training against validation rows, by context (capacity);
  - the residual objective: clipping, and the step scale.
- **QG2-E** (if own starts lead):
  - the own-starts factor by route, path presence and length, and suffix;
  - the prefix swap.
- **A training arm,** whatever the stage, only with the owner's approval. R3
  (capacity) needs its own unlock.

## Status

    QG1  registered 2026-09-28 (this document), before any QG code or measurement
    QG1  code 144e4ea; READ 2026-09-28 (QG1_RESULTS.md): VQ1-VQ4 PASS;
         DECISION-LED in both classes (decision share 0.89 benign, 0.80 phishing).
         Licensed next: QG2-D, to be registered before it runs.
    QG2-D  approved by the owner (three probes, laptop only); registered
           2026-09-28 in PREREG_QG2D.md before any QG2-D code
    QG2-D  READ 2026-09-28 (QG2D_RESULTS.md): QUERY-STATE -- the deficit lives in
           the decision under the hidden query control; the training code is
           symmetric (audit). Observation: under-prediction grows with outcome
           rarity (a hypothesis for QG3)
    QG3    approved by the owner; registered 2026-09-28 in PREREG_QG3.md before
           any QG3 code (rarity against the hidden state, at matched frequency)
    QG3    READ 2026-09-28 (QG3_RESULTS.md): STATE-LED -- about 97-98% of the
           query's shortfall is the hidden query state, not rarity (equally rare
           uncontrolled outcomes are calibrated); e9_base and E7 agree
