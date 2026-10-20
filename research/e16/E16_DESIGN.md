# E16 — what the E15 evidence supports next (evaluation, not a registration)

- **The question, 2026-10-06** (`research/native/DECISION_2026-10-06.md` §1): should
  E16 be a larger dose of the same query-only strategy, a stronger lever, or is this a
  sufficient stopping point? "Do not start any E16 experiment or training yet; I want the
  diagnosis, proposed intervention, guards, and preregistration plan first."
- **Status:** evaluation only. Nothing is registered, coded, acquired or trained. The new
  measurements in §2.5 are post-hoc and descriptive, on files that already existed. They
  license nothing on their own.
- **Baseline:** `e15_new` (adopted 2026-10-06).

## In plain words

- **The remaining cells split into two kinds, and E15 made the split sharp.**
  - **Cells that respond to more query practice:** benign P0Q1 QRY and the far cell, free
    phishing QRY. Three different exposure levers moved both. E15 got most of E14's gain
    at about 1/16 of E14's added mass, with no cost.
  - **Cells that do not:** benign P1Q1 QRY and benign P1Q1 PTH. Three exposure levers left
    P1Q1 QRY within ±0.003. In E15 the model's fit on P1Q1 rows improved exactly as much
    as on P0Q1 rows (−0.037 nats per character), but only P0Q1's tell moved.
- **So a larger dose of the same strategy would probably clear P0Q1 and lower the far
  cell, but it would not touch P1Q1.** P1Q1 decides whether E3 can be met at all.
- **E15 also leaves two of the owner's conclusions open:**
  - **"New diverse data, not repetitive old data, avoids the trade-off."** E15 cannot
    distinguish that from a simpler explanation: its dose was about 1/16 of E14's.
    Uptake of new strings was at the chance rate, so the gain came from exposure.
  - **"Not a capacity wall."** True for the cells that moved, which are the small strata.
    For P1Q1 the evidence points the other way: every non-capacity training lever has now
    been tried on it, and none moved it.
- **Recommendation: E16 = one decisive capacity probe, with an end rule.**
  - A wider model (about 1.75× the parameters), with `e15_new`'s data, batches, steps,
    objective and decode unchanged.
  - **Its question:** can P1Q1 move at all?
  - **If it moves,** capacity is binding there and the path to E3 continues.
  - **If the bigger model fits better but P1Q1 still does not move,** every allowed
    training-side lever has been tested on P1Q1. The next document is the final E3
    report.
  - This needs the owner to unlock R3, which is locked as the last resort
    (`DECISION_2026-09-29.md` §4). §4.2 sets out the case that its unlock condition is
    now met for P1Q1.
- **Stopping now is defensible only if the goal is no longer E3.** E3 is not met: 5 cells
  and 1 slash row remain. Stopping before the P1Q1 question is answered would leave
  unknown whether E3 is reachable at all.

## 1. Where the new baseline stands

| item | `e15_new` x | SE | t | verdict | clear line at this SE | needed fall |
|---|---|---|---|---|---|---|
| free phishing QRY | 0.0774 | 0.0094 | +2.91 | CONVICTED | 0.0239 | ≈ 0.04–0.05 |
| control benign P1Q1 PTH | 0.0551 | 0.0052 | +0.98 | UNDECIDED | 0.0356 | ≈ 0.013–0.020 |
| control benign P1Q1 QRY | 0.0500 | 0.0047 | +0.00 | UNDECIDED | 0.0370 | ≈ 0.009–0.013 |
| control benign P0Q1 QRY | 0.0410 | 0.0066 | −1.36 | UNDECIDED | 0.0317 | ≈ 0.004–0.009 |
| free phishing PTH | 0.0369 | 0.0053 | −2.47 | UNDECIDED | 0.0353 | noise band |
| slash, benign control P0Q1 | 0.99859 vs 0.99551 | | | outside T | | 13 URLs |

- **The clear line** is 0.05 − 2.776 × SE. The SE comes from the spread of five
  generations, so it moves between models. The needed fall is a range for that reason.
- **Cells now CLEARED only on the margin:** control benign P1Q1 SUB (t −2.82) and control
  phishing P1Q0 PTH (t −3.06). The bar is −2.776. E3 needs every cell CLEARED in one
  model, so these can flip back on any retrained model.
- **For scale, the between-run SDs measured in E13-D (DX4):** free phishing QRY 0.0081,
  P0Q1 QRY 0.0023, P1Q1 QRY 0.0041, P1Q1 PTH 0.0036.

## 2. Diagnosis

### 2.1 Two kinds of cells: the exposure record

| cell | stratum's training rows | N's A2 (4× query practice) | E14 Q (×4 weight, everywhere) | E15-B (×2, local, new rows) |
|---|---|---|---|---|
| free phishing QRY | 8,272 | 0.114 → **0.055** | 0.114 → **0.083** | 0.112 → **0.077** (no phishing rows added) |
| benign P0Q1 QRY | 14,906 | 0.058 → **0.028** | 0.062 → **0.037** | 0.062 → **0.041** |
| benign P1Q1 QRY | 42,862 | 0.052 → 0.049 | 0.056 → 0.058 | 0.052 → 0.050 |
| benign P1Q1 PTH | (paths: abundant) | — | 0.058 → 0.068 | 0.058 → 0.055 |

- **The pattern matches E13-D's DX3.** The transformer lost to a plain 6-gram learner
  exactly where data per cell is small (phishing, P0Q1). It matched or beat it where data
  is large (P1Q1, free benign). More practice closes a learning gap on small strata. On
  P1Q1 there was no such gap left to close.
- **N's A2 makes the point twice.** In P1Q1, 4× practice lifted copying almost to the
  natural rate (0.159 → 0.211, natural 0.248), and the tell still moved only 0.003.

### 2.2 The fit and the tell have come apart (E15's main new fact)

| | held-out query fit change (nats per character) | tell change |
|---|---|---|
| E15, benign P0Q1 rows | −0.0369 | −0.0214 |
| E15, benign P1Q1 rows | −0.0374 | −0.0020 |
| E15, phishing | −0.0053 (interval spans 0) | −0.0349 |
| E14, phishing | −0.1848 | −0.0311 |

- **Equal fit gains gave very unequal tell gains, and the far cell moved without any fit
  gain.**
- So in the cells that remain, teacher-forced fit does not predict the tell. This matters
  for every lever whose main effect is fit, capacity included (§3.2).

### 2.3 What E15 did and did not establish

| claim | status |
|---|---|
| a small, local query dose gives most of E14's gain without E14's cost | **established** (G2: largest cell rise +0.0101; G3: largest field +0.0034) |
| "new diverse data avoids the trade-off that repetitive old data caused" | **not established.** E15's added mass was about 1/16 of E14's (0.98% against 15.6% of scored mass), so a dose-proportional cost predicts the same clean result. Uptake of new-stream-only strings was at the chance rate (2.28% vs 2.00%), so the registered reading attributes the gain to exposure, not content (§10.6) |
| "the query problem is not a capacity wall" | **established for the small strata, open for P1Q1.** The cells that moved are the ones where practice was scarce. P1Q1 moved under none of the three exposure levers |

### 2.4 What each remaining item is, by the record

| item | DX1/DX2 (E13-D) | DX8: context or field | exposure | other |
|---|---|---|---|---|
| free phishing QRY | LEARNING | CONTEXT-LED (63% from the model's own host and path) | responds | the context arm (E13 C) decoupled the query from its context and was closed |
| benign P0Q1 QRY | LEARNING | FIELD-LED | responds | — |
| benign P1Q1 QRY | LEARNING; not data-limited (DX3) | SPLIT (75% the field's own sampling) | immune | not state-blind (QC1-8); lexical content, not shape (QC1-3) |
| benign P1Q1 PTH | LEARNING | FIELD-LED | immune (E14 made it worse) | in scope by option B; the phishing free PTH attribution branch stays closed |
| free phishing PTH | — | — | — | noise band (t −2.47; was CLEARED at t −2.86). Read, not diagnosed (closed branch) |
| slash row | — | — | — | rare-plan query honouring: 51 of 71 "no slash" plans drop their query (§2 of the decision record) |

### 2.5 Post-hoc, descriptive (`e15_posthoc.py`, `e15_posthoc_length.py`; licenses nothing)

**A. What moved in the far cell.**
- Free phishing queries moved slightly toward real on every string measure:
  - exact copies 17.0% → 19.5% (real 40.3%);
  - exact (path, query) pairs 9.5% → 13.6% (real 30.6%);
  - top-8 share 14.0% → 16.1% (real 21.5%).
- **The context barely moved:** exact paths before a query 24.1% → 25.8%, against 57.8%
  real.
- **[INFERENCE]** The −0.035 most plausibly came through the query field itself
  (practice shared across classes), not the context. DX8's context share is still there.

**B. In every query cell, the long tail of query length is missing.**
- Medians match real (P1Q1: 19 against 19). Queries over 128 characters are about half
  as common as real:
  - free phishing 4.6% vs 8.5%;
  - P0Q1 1.0% vs 2.2%;
  - P1Q1 2.0% vs 3.6%;
  - free benign 1.7% vs 3.2%.
- E15 did not change this.
- N2a's M1 points the same way, though its readings are void by N2a's power rule. On
  real rows, `e12_final` stops between 50 and 100 characters into a value 1.10×
  (benign) and 1.41× (phishing) as often as real values do.
- **But it is not P1Q1's tell.** Matching real queries to the generated length mix
  removes 0.0036–0.0049 of P1Q1's tell, 6–8% of it, at n ≈ 1,730 over 12 seeds. For free
  benign it is 15–17%. The phishing and P0Q1 readings are too small to read (n ≈ 270–530):
  the two arms disagree even in sign.

**C. P1Q1's queries are too novel.**
- Never seen in training: 73.0% generated against 61.1% real. Exact copies: 14.9%
  against 20.3%.
- Mean characters: 29.6 against 36.2 (the long tail, B).
- QC1-2 already showed that re-drawing the output toward the natural copy share makes
  the tell bigger. So "copy more" is not the repair: the novel strings are what the
  detector sees, and N's A2 confirms raising copying does not fix it.

## 3. The options

### 3.1 A — a larger dose of the same strategy

| | expectation |
|---|---|
| benign P0Q1 QRY | **likely CLEARED.** The dose response is monotone (E15 ×2: 0.041; E14 ×4: 0.0365, CLEARED; N's A2: 0.028) |
| free phishing QRY | **falls further through transfer, about 0.06–0.07, still CONVICTED.** N's A2, with 4× practice on phishing itself, reached only 0.055 |
| benign P1Q1 QRY and PTH | **no change.** Three exposure levers, three nulls (§2.1) |
| the waived slash row | **likely to worsen.** More slashed `host?query` practice strengthens the habit that drops queries on "no slash" plans |
| costs | rise with dose. E14 showed them at 16× E15's mass; a second doubling is about 2× E15's |
| data | needs a new acquisition beyond the registered cap (4,000 of 4,251 blocks per crawl are used). Fresh domains get rarer, so yield falls |
| E3 | **cannot be reached:** P1Q1 QRY/PTH and most of the far cell remain |

**Verdict:** a useful step for one cell, and a low-information experiment. Its outcome is
largely predictable from E14, E15 and N.

### 3.2 B — stronger levers, considered

| lever | target | assessment |
|---|---|---|
| **B1 capacity (R3)** | P1Q1 QRY/PTH | **The only training-side family not yet tested on P1Q1** (audit in §4.2). Against it: P1Q1's tell has not tracked fit (§2.2), and capacity mostly buys fit. So the chance it clears P1Q1 is modest. But it is decisive either way, which no other option is. **Recommended** |
| B2 phishing practice from existing rows (query-only re-injection of old phishing query rows) | far cell | The far cell clearly responds to its own practice (N's A2: 0.055). E15's reading (gain from exposure, not content) removes the premise for requiring new rows. **But it conflicts with the owner's directive "do not duplicate, oversample, or reshape the existing QRY rows".** Even at N's level it would not clear: the line is about 0.024–0.037. The owner's call; not recommended as E16 |
| B3 path targets on the stream's P1Q1 rows | P1Q1 PTH | Paths are not scarce, so by §2.1 extra path practice should not help. P1Q1 PTH is FIELD-LED and got worse under E14's exposure. Not recommended |
| B4 query length / termination | every query cell | The long tail is short everywhere (§2.5 B), but it carries only 6–8% of P1Q1's tell. Its natural lever, weighting at field boundaries, is the closed S4/S4b branch. Not recommended |
| B5 a plan-honouring decode rule (no end of URL before `?` under a Q1 plan, by D1's principle) | the slash row | Would close the waived row by construction, without training. It is a decoder rule and needs its own registration. Not part of E16; listed so it is not forgotten |
| excluded by the owner's rules or closed | — | decoder-side sampling (temperature, top-p); more steps (R2); the query weight (N2b); D2; the context arm; the PTH attribution branch; boundary weighting (S4/S4b). Not reopened |

### 3.3 C — stopping now

- **As a generator, the engine is strong.**
  - Every benign cell except P0Q1 QRY and P1Q1 QRY/PTH is CLEARED, and those three sit
    near the 0.05 bar.
  - The far cell concerns query strings, which appear on about 4.7% of free phishing URLs.
- **As a research programme, stopping now leaves E3's feasibility unknown.** No lever has
  ever moved P1Q1, and one family has not been tried.
- **Recommended only if the owner's goal changes from E3 to "freeze what we have".**

## 4. Recommendation: E16, one capacity probe with an end rule

### 4.1 Why this, and not a larger dose

- E3 cannot be met without P1Q1. A larger dose has, on the record, about zero chance of
  moving P1Q1.
- The capacity probe is the only remaining training-side test that can. Whatever its
  result, it tells the owner whether to continue toward E3 or close it.
- It also probably helps the small strata, since a larger model learns scarce data
  faster. That is reported, not counted toward its question.

### 4.2 R3: the unlock condition, for P1Q1

`DECISION_2026-09-29.md` §4 unlocks R3 when "the architectural, objective and
representational candidates for the failing cells have each been tested or ruled out by
a registered reading". For benign P1Q1 QRY and PTH:

| candidate family | tested or ruled out by |
|---|---|
| data quantity and coverage | DX3: NOT DATA-LIMITED (E13-D). E15-B: 42,429 new P1Q1 rows, tell −0.002 |
| exposure / loss emphasis | E14 Q (+0.002; PTH +0.010); N's A2 (−0.003); E15-B |
| conditioning on the plan | E10/E11 presence plan, E12 slash plan (adopted). QC1-8: paths NOT STATE-BLIND |
| representation (subwords, class, plan at every position) | QC1-2 to QC1-5 (no MIX-LED, no BLENDING, no LEAK); E13 P (path and subdomain subwords ruled out) |
| context | DX8: P1Q1 QRY 75% field-led, PTH 86%. The context arm (E13 C) closed |
| objective | E9's length-neutral objective and R1's longer limit (adopted); boundary weighting (S4/S4b) closed |
| optimisation | R2 (40,000 steps): rejected |
| decoder | excluded by the owner's rules; D2 had no effect |

- **My reading:** case 1 is met for P1Q1.
- **Unlocking R3 is the owner's decision.** The memory of E8 (a capacity step proposed
  and withdrawn) is why this table exists: the case is laid out before the lever is
  proposed.

### 4.3 The intervention (draft)

| | `e16_cap` |
|---|---|
| model | d_model **512**, 8 layers, 8 heads, d_ff **2048**: about 25M parameters, 1.75× `e15_new`'s 14.4M. Width, not depth: the same layers and attention layout, so only the size changes |
| start | a fresh seeded start at the new width, by E12's own start procedure (`new_model_e12` at d 512), its fingerprint recorded in a laptop dry run before any training. E12's start cannot be reused: it is width 384 |
| data, batches, objective | `e15_new`'s exactly: corpus2 + the stream (query targets only, p = 1.0, the same seeds); the same sampler and batches (V-batch digest `adbddc2a…`); C pinned at 14397.885; 20,000 steps; lr 3e-4; the same clip schedule |
| memory | today's run uses 12.0 of 14.6 GB at width 384. At 512 the batch is split into 2 × 128 with gradient accumulation: the same batch and the same summed gradient |
| decode and files | raw + D1 + D1′; E12's 50 jobs at E12's seeds; every cell paired with `e15_new`'s files |
| cost | training about 3.5 h on a T4 (1.75× compute); generation about 3 h; the laptop about 3 h. One Kaggle session each, well inside the 12-hour limit |

**Considered and rejected for E16:**
- **Capacity + a second doubling, in one arm.** It would confound the P0Q1 and far-cell
  readings. It also needs a new acquisition and raises the slash-row risk.
- **A larger step (2–3×).** It risks Kaggle's session limit, and a moderate step already
  answers whether P1Q1 is capacity-sensitive.
- **Tuning the learning rate for the wider model.** That is tuning. The recipe stays.
  The risk is stated: a wider model at the same rate may under- or over-shoot.

### 4.4 Criteria (draft; `e16_cap` against `e15_new`; existing thresholds only)

| code | criterion |
|---|---|
| **K16-1** the question | control benign P1Q1 QRY **or** PTH falls by more than 0.0141 (the retraining allowance) |
| **K16-2** fit | held-out NLL falls in both classes, interval below 0 (M3 for queries, plus the training record's per-field table). This shows the probe actually added usable capacity |
| **K16-3** band profile | M2: no overshoot (old + stream reference, as §13.2) |
| G1 | no regression (K12c), against `e15_new` |
| G2 | no PTH, SUB or LBL cell rises by more than 0.0141 |
| G3 | no non-query field's validation NLL rises by more than 0.01 against `e15_new` |
| G4 | presence within T (K12a) |
| **G5** | all 8 slash rows within T. **The waiver does not carry over.** Same data, so the same "no slash" query dropping is likely. The owner may pre-decide how a repeat is treated; I recommend keeping G5 as is, because it is E3's own slash condition |
| G6–G9 | honouring, copying (own natural rate), novelty and FULL, as in E15 |
| G10 | clipping (K12e); V-batch, V-mass, V-stream, V-start, V-accum |

**Adopted iff all hold.** No retry and no tuning.

### 4.5 The end rule (registered with E16, not after)

| outcome | reading | what follows |
|---|---|---|
| K16-1 PASS | capacity is binding for P1Q1 | the route to E3 continues. Next, the small strata (dose) on the larger model |
| K16-1 FAIL, K16-2 PASS | the larger model fits better and P1Q1 does not move. Every allowed training-side family has been tested on P1Q1 | **the final E3 report.** E3 NOT MET, with the evidence. The owner then decides: freeze the engine, or open an excluded family (decoder-side sampling), knowing its risk of tuning to the metric |
| K16-2 FAIL | the probe did not add usable fit, so it does not answer the question | reported. No automatic follow-up; the owner decides |

**A partial move is reported, not counted.** P1Q1 falling by more than 2 between-run SDs
(about 0.008) but less than 0.0141 would mean "capacity-sensitive, not enough at this
size". That would make a larger step the owner's option.

### 4.6 Predictions, preliminary [HYPOTHESIS] (formalised in the registration)

| | prediction |
|---|---|
| held-out query NLL | −0.02 to −0.05 nats per character, both classes. K16-2 PASS |
| benign P1Q1 QRY | −0.000 to −0.008. **K16-1 PASS about 20%** |
| benign P1Q1 PTH | −0.000 to −0.008 |
| benign P0Q1 QRY | −0.003 to −0.010 (CLEARED possible) |
| free phishing QRY | −0.005 to −0.020, still CONVICTED |
| G5 | **likely FAIL again** on the same "no slash" row (same data) |
| overall | most likely the end rule's second row |

### 4.7 Validity, new for E16

| code | check |
|---|---|
| V-start′ | the d-512 start's fingerprint equals the dry run's, recorded before training |
| V-accum | on the laptop CPU, the accumulated gradient of 2 × 128 equals the full 256-row gradient (dropout off, float64) to 1e-10, and the update matches within float32 tolerance. A unit test |
| V-batch, V-mass, V-stream | as E15: the old rows' digest `adbddc2a…`, 94,787 ± 3 SE rows appended, stream `637accab…` |
| memory preflight | a full-size step at d 512 on the T4 before training, as every run since E10 |

## 5. If the owner prefers A (a larger dose)

The registration would contain:
- **a new acquisition:** corpus2's procedure, rounds 9 onward. The per-crawl cap is
  lifted from 4,251 to 8,251 blocks, with new seeds. Gates P1–P8 again, on the new rows
  and on old + both streams;
- **injection at twice the current rate:** two new rows per old stratum draw. That
  extends the injector by one count; the same masks and rules;
- **the same criteria as E15** against `e15_new`, with G5 unwaived;
- **an honest prediction:** P0Q1 likely CLEARED, the far cell about 0.06–0.07, P1Q1
  unchanged, and the slash row at risk.

## 6. The preregistration plan (either option)

| step | where | what | gate |
|---|---|---|---|
| 0 | laptop | this evaluation | **the owner chooses: B1 (with the R3 unlock), A, or C** |
| 1 | laptop | `PREREG_E16.md`: design, criteria, end rule, predictions, validity; for B1 also the R3 unlock, verbatim. Commit and push | **gate A: the owner approves the registration** |
| 2 | laptop | code (B1: width-512 start, gradient accumulation; A: acquisition and the ×2 injector), unit tests, the dry run (V-start′ / V-accum, or gates P1–P8), smoke tests, the bundle and notebooks. Commit and push | **gate B: the owner approves the measured numbers before any Kaggle run** |
| 3 | Kaggle | E16_train, verified on the laptop, then E16_generate | — |
| 4 | laptop | verify, score, criteria, the results document, the end rule applied as registered | **the owner's decision** |
