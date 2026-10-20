# E13 — roadmap: the lever the diagnosis licenses, what it cannot reach, and the owner's decisions

**Status: a proposal for the owner, 2026-10-01.**
- Nothing in it is registered.
- `PREREG_E13.md` is written from the owner's answers (§7), before any E13 code.
- No Kaggle work happens before the owner approves it.

**Rests on:**
- `E13D_RESULTS.md` (DX1–DX8, and section 5's descriptive copy check);
- `DECISION_2026-10-01.md` (`e12_final` is the baseline; the owner's E13 directive);
- `DECISION_2026-09-29.md` §4 (R3's conditions);
- the earlier record cited where it is used: QC1, N2a, R2, PQ2 and R4.

## In plain words

- **The diagnosis is complete** (`E13D_RESULTS.md`):
  - **The data is fine.** It is the right population, and there is enough of it.
    No data change is licensed.
  - **The model fits its data.** It assigns real strings high probability. What
    goes wrong is what it writes when it samples.
  - **Free phishing QRY is mostly the model's own context.** About 63% of its tell
    comes from the host and path the model wrote itself:
    - given a real host and path, its queries read 0.046, not 0.124;
    - the strings show it: 24.0% of the model's paths before a query are real
      training paths, against 57.8% in real data.
  - **Everywhere else, at least half of the tell is in the field itself.** It
    remains even when the context is real.
- **One lever is licensed: training on the model's own context.**
  - For phishing rows that carry a query, the model is sometimes shown a host and
    path that the baseline model wrote. It is then trained to write the real query
    after them.
  - Best case: free phishing QRY falls from about 0.11 to about 0.043. Its clear
    line is 0.037. So the cell would go from CONVICTED to UNDECIDED, not to
    CLEARED.
- **E3 cannot be reached in E13 under the current rules.**
  - **Four of the six cells stay above their clear line even with a real context.**
    No lever within the rules is licensed for that remainder.
  - **The three slash rows have no licensed lever either.** The two phishing rows
    need almost no misses: within 0.04–0.07 points of a real rate of 100%. Even
    after a real host, the model's chance of missing the slash is 0.159% (DX5's
    probe).
- **The record already holds the route for the remainder:**
  - re-run N2a on the current recipe (`DECISION_2026-09-29.md` §4);
  - its reading either licenses a query-content loss weight (N2b), or it points to
    capacity or representation. Read together with QC1's registered ruling-out of
    query subwords, the second is the "strong proof" that unlocks capacity (R3);
  - paths and subdomains need one more check, on the laptop: QC1's subword
    questions, which were only ever asked of queries.
- **Recommendation: one Kaggle round with two parts.**
  - The context arm.
  - N2a re-run on the current recipe.
  - Plus the laptop check for paths and subdomains, while the owner runs Kaggle.
  - E13 is not expected to reach E3. It is expected to shrink the largest failure,
    and to settle on registered evidence whether capacity comes next.
- **Seven decisions are the owner's (§7).**

## 1. What E13-D established

| question | answer | from |
|---|---|---|
| Is the training data the wrong population? | No: training and validation strings are indistinguishable in all six cells | DX1 (x_TV −0.0044 to +0.0054) |
| Is the model short of data? | No: the 6-gram learner's curve is flat (with the caveat in `E13D_RESULTS.md` §5: the learner largely re-writes training strings). The transformer's validation fit is within 0.06 nats per character of its training fit | DX3; DX7 |
| Does the model fit its data badly? | No: it beats a smoothed 6-gram by 0.17–0.46 nats per character in every cell | DX7 (FITS ×6) |
| Is the tell the model's own context or the field itself? | Free phishing QRY: CONTEXT-LED. Benign P0Q1 QRY and P1Q1 PTH: FIELD-LED. The other three: SPLIT | DX8 |
| Is it how many common strings the model copies? | No: re-drawing to the natural copy share makes the tell bigger | QC1-2 on E11; replicated descriptively on E12 (`E13D_RESULTS.md` §5) |
| Are the new strings near misses, or borrowed from another population? | No and no | QC1-4, QC1-5; §5 on E12 |
| Why are the slash rows missed? | MIXED in all seven populations: no single cause the registration could name | DX5 |

**What this leaves:**

| cell | E12 gate x | with a real context (DX8) | clear line (DX4) |
|---|---|---|---|
| free phishing QRY | +0.1141 CONVICTED | 0.046 | 0.037 |
| control benign P0Q1 QRY | +0.0621 UNDECIDED | 0.061 | 0.034 |
| control benign P1Q1 QRY | +0.0559 UNDECIDED | 0.048 | 0.039 |
| control benign P1Q1 PTH | +0.0576 UNDECIDED | 0.052 | 0.042 |
| control benign P1Q1 SUB | +0.0416 UNDECIDED | 0.018 | 0.029 |
| control phishing P1Q0 PTH | +0.0428 UNDECIDED | 0.030 | 0.041 |

- The real-context column is on the harness's scale, which matched the gate
  within 0.014 (V2).
- **Four cells stay above their line even with a perfect context.** Only SUB and
  phishing P1Q0 PTH sit below theirs with a real context.

## 2. The licensed lever: training on the model's own context

### 2.1 What it is

**Today the model is always trained on a real URL's context.** It learns to write a
query after real hosts and paths, then has to write one after its own. DX8 measured
the cost of that mismatch: in free phishing, most of the query tell.

**The lever, `e13_ctx`.** This is `e12_final`'s recipe with one change:
- **Which rows:** the 8,272 phishing training rows (the domain-frame name corpus)
  that carry a query. That is 4.29% of phishing's 192,652 training URLs, and the
  only population DX8 licensed.
- **What changes:** each time such a row is drawn, with probability 0.5, everything
  before its query is replaced by a host and path that `e12_final` wrote for that
  row. The model wrote it under the row's own suffix and plan, with the registered
  raw decode.
- **What is trained:**
  - only the row's real tokens, from the query marker on (the query, any
    fragment, the end);
  - the model-written part is input only, never a target. The model is not taught
    to write it.
- **The written contexts are made once, before training.** They are frozen as a
  file with its checksum, made on the laptop.
  - For a row whose written context never reaches a query under its plan, the
    row keeps its real context. Those rows are counted.
- **Everything else is `e12_final`'s:** tokenizer, objective, R1, plan, slash,
  steps, seed and settings.

### 2.2 Why this form

- **It is DX8's contrast, turned into training.**
  - DX8 compared a query written after a real context with one written after the
    model's own.
  - The lever trains the second case against the real query.
- **It is native.** The production generator is unchanged: one forward pass, raw
  decode, no resampling, masking or second prediction. Only what the model saw in
  training changes.
- **This is new for the project, so it needs the owner's approval** (§7, decision
  2). Model-written tokens enter training for the first time.
  - They enter only as context, never as targets.
  - They come from the adopted baseline, not from the model being trained. That is
    simpler and verifiable. The cost is that the trained model's own contexts will
    differ slightly from `e12_final`'s.
- **Half the time the row stays real.** So the model still learns real phishing
  hosts and paths from these rows, at half the previous rate.

### 2.3 What it can and cannot do (predictions, to be registered)

- **The ceiling is DX8's real-context level.**
  - The lever can at most make the model's own context as good as a real one: 0.046
    on the harness's scale, about 0.043 on the gate's (a projection).
  - The cell's clear line is 0.037. Reaching the ceiling would leave it UNDECIDED.
- **Prediction:** free phishing QRY between 0.05 and 0.08, UNDECIDED. The other cells
  stay within run-to-run noise.
- **The mechanism check:** DX8 run again on `e13_ctx`.
  - The gap between "own context" and "real context" should shrink by at least
    half: from 0.077 (0.1237 − 0.0463).
  - If the cell improves but this gap does not, the improvement is not the
    registered mechanism, and it is reported as such.
- **The slash rows:** the two phishing misses may shrink, since misses are about
  twice as likely after the model's own hosts (DX5). They cannot pass. Even after a
  real host the probe's miss chance is 0.159%, and the rows need about 0.04–0.07
  points.

### 2.4 Risks and their guards

| risk | why it could happen | guard (fails = not adopted) |
|---|---|---|
| the query stops depending on the path | it is trained on real queries after paths that did not produce them | FULL, the whole-URL view, must not get worse by more than 0.0141 in free phishing. The (path, query) pair-copy rate is reported (E12: 9.2%; real 30.6%) |
| other cells pay for it | shared capacity | no CLEARED cell gets worse by more than the retrain allowance (0.0141), in free or control mode |
| memorisation | real queries are trained more often in these rows | copying stays at or under the natural rate in every cell |
| presence or honouring moves | phishing query rows are retrained differently | presence and honouring stay inside E12's registered bands |
| the slash gets worse | the host end is in the model-written part for these rows | no slash row that passed in E12 fails |

## 3. What the lever cannot reach, and why nothing else is licensed

**The remainder** is what four cells keep even with a real context: the field's own
strings. The record has already tried or ruled out almost every lever for it:

| lever | status | record |
|---|---|---|
| more or different data | not licensed: no shift, not data-limited | DX1, DX3, DX7 |
| copy more common strings | not a repair: it makes the tell bigger | QC1-2 |
| query subwords | not licensed: no blending, no mix effect, names not identified | QC1 §3 |
| the class or plan at every position | not licensed: no leak, no state blindness | QC1-5, QC1-8 |
| longer queries (R1) | done in E12; kept | E12 |
| more steps (R2) | REJECTED: "more optimisation is not the repair" | R2 |
| path weighting (S3, S4, S4b) | CLOSED | NATIVE_REPAIR_PLAN §8 |
| a query-content loss weight (N2b) | not licensed by N2a, on the **old** objective. Reopening it is the owner's call | N2a; E12_ROADMAP §2.4 |
| path and subdomain subwords | **never asked** | DESIGN.md §4.2 proposed them; no reading since |
| capacity (R3) | LOCKED, last resort | DECISION_2026-09-29 §4 |
| decoder changes | excluded by the owner's rules | below |

**Why not a decoder change.** The remainder is "sampling at temperature 1", so a
decoder change looks like the natural lever. The record argues against it:
- **Truncation moves the query tell little, in either direction.**
  - On E5, against a phishing query tell of about 0.14, removing truncation
    changed it by:
    - −0.0057 in R4, not significant ("at most a few percent of the tell");
    - +0.0224 in PQ2, fragile (t 4.54).
  - Benign moved by −0.0200 in PQ2, not significant.
  - Plain temperature 1.0 has been the default since R4's adoption.
    E12_ROADMAP also records that truncation concentrates names.
- **Temperature rescaling recovered about 0.03 nats** in S3. That was for paths:
  "the distribution's SHAPE is wrong, not merely its sharpness".
- **The raw decode is the owner's baseline** (`DECISION_2026-09-28.md`).
- **A correction to the E13-D registration's rationale.**
  - DX8's prediction cited "R2's interaction, +0.0351" as temperature lowering the
    query tell.
  - R2_RESULTS records it as the lowest QRY tell measured, under R2's checkpoint
    with plain temperature-1.0 sampling, "unexplained and underpowered". It is not
    evidence that a lower temperature helps, and nothing here relies on it.

**Why not a training objective against "improbable" output.** Objectives that
penalise mass away from the data are the training-side twin of truncation. Truncation
removed little of the tell (above), so they are not proposed.

**Why not capacity now.** R3's rule opens one of two doors:
1. **Every architectural, objective and representational candidate has been
   tested or ruled out by a registered reading.** Not yet. Three remain:
   - the context lever (now licensed);
   - N2b on the current objective;
   - path and subdomain subwords (never asked).
2. **A registered reading on the current baseline that capacity binds.** The
   record names it: N2a re-measured on the current objective. Not done.
   - N2a's Amendment 2 merged the plan's two unlock readings (FREQUENCY-SPECIFIC,
     GENERAL) into CAPACITY-OR-REPRESENTATION, which "cannot tell capacity from
     representation".
   - For queries, QC1 has already ruled query subwords out by its registered
     reading. So the pair can be the strong proof, if `PREREG_E13.md` says so
     before the run (§7, decision 5).

The facts that point toward capacity are real, but none is a registered capacity
reading:
- N2a's four-times exposure left the model as detectable as before, even against
  its own training queries (benign; phishing's reference was too noisy to read);
- phishing's validation fit improved from 2.632 to 2.371 nats per character
  without any change the detector sees;
- the model is 14.4M parameters and sees about 5.1M rows in training (20,000
  steps of 256).

## 4. The slash rows

| row (E12 `e12_final`) | generated | real | tolerance | missed by |
|---|---|---|---|---|
| phishing free bare hosts | 99.67% | 100.000% | ±0.04 | 0.29 points |
| phishing control P0Q0 | 99.76% | 100.000% | ±0.07 | 0.17 points |
| benign control P0Q1 (host?q) | 99.81% | 99.551% | ±0.24 | 0.02 points |

- **Phishing:** the model leaves out the slash on 0.24–0.33% of bare phishing hosts.
  After a real host, the probe's miss chance is still 0.159% (DX5). No lever in E13
  brings that under 0.04–0.07 points.
- **Benign host?q:** a rare outcome suppressed (QC1-7). The model obeys a supplied
  "no slash" only about 70% of the time (DX5). Nothing in E13 targets it.
- **DX5 read MIXED and licensed nothing.** The registered objective lever for
  obedience (N1b, unlikelihood on plan-forbidden tokens) was conditional on E8,
  whose results are unread.
- **These rows are part of the E3 gap, and E13 does not close them.**

## 5. The recommended E13

**One Kaggle round (E13_train, then E13_generate), and one laptop check alongside.**

### E13-C — the context arm (Kaggle)

- **Arm:** `e13_ctx` (§2).
- **Compared against:** `e12_final`'s registered readings, with no retrain. The
  yardsticks are the retrain allowance (0.0141) and DX4's between-run SD (0.0081
  for free phishing QRY). A same-session retrain of `e12_final` is the owner's
  option (§7, decision 3).
- **Criteria** (fixed in the prereg):
  - K13-1: free phishing QRY falls by more than 0.0141, and is no longer CONVICTED;
  - K13-2: the mechanism check (§2.3);
  - the guards in §2.4.
- **Adopted** as the new baseline only if K13-1 and K13-2 hold and no guard fails.
  E3 is still claimed only on the unchanged finish line.

### E13-N — N2a re-measured on the current objective (Kaggle)

- **N2a's registered design** (`PREREG_N2A.md`, Amendments 1–4), with only these
  changes:
  - **recipe and reference:** `e12_final`'s, not E7's;
  - **arms:** A1 = `e12_final`'s query-row exposures (counted from its sampler);
    A2 = four times that;
  - **copying clause:** judged against the natural rate, in both directions. This
    is E12's rule (E12_ROADMAP §2.4). N2a's original clause counted any rise as
    memorisation, even below the natural rate.
- **Its readings** (N2a's registered table):

  | reading | licenses (in E14) |
  |---|---|
  | ALLOCATION or EXPOSURE | N2b, a query-content loss weight |
  | SPARSITY | nothing; the owner decides |
  | CAPACITY-OR-REPRESENTATION | N2c: capacity or representation. With QC1's registered ruling-out of query subwords, this is R3's second door, if decision 5 registers it so |

- Phishing may again be unavailable by N2a's power rule. Benign decides.
- A1 and A2 are diagnostic models. Nothing they write is data, and they are never
  candidates.

### E13-P — paths and subdomains: QC1's subword questions (laptop)

- **On `e12_final`, replicated on `e12_base`:** benign P1Q1 PTH and SUB, and
  phishing P1Q0 PTH.
  - Are the never-seen strings near misses of training strings (BLENDING)?
  - Which properties carry the tell (QC1-3's ladder)?
- **What it settles:**
  - BLENDING or IDENTIFIED: path or subdomain subwords are licensed (DESIGN.md
    §4.2), for E14;
  - neither: ruled out, which R3's first door needs.
- **Laptop only.** It reuses QC1's instruments and runs while the owner runs
  Kaggle.

### What E13 will and will not decide

| after E13 | then |
|---|---|
| K13 holds | `e13_ctx` is the baseline; free phishing QRY is UNDECIDED, not CONVICTED |
| N2a reads ALLOCATION or EXPOSURE | E14 tests N2b, a non-capacity lever for the query remainder |
| E13-P licenses subwords | E14 tests path or subdomain subwords |
| N2a reads CAPACITY-OR-REPRESENTATION, and E13-P rules path and subdomain subwords out | R3 is unlocked by its own rule: a capacity probe goes to the owner for E14 |
| N2a reads SPARSITY | the owner decides, as N2a registered |

**E3 is not expected in E13.** It is expected, at the earliest, from E14's lever.

### Cost

- **E13_train:**
  - `e13_ctx` takes about as long as one E12 arm (63–69 minutes on the T4);
  - N2a's two arms are smaller (16 and 32 minutes on E7's recipe; somewhat more on
    E12's);
  - in all, about the size of E12_train.
- **E13_generate:** one full arm, plus N2a's query sets. That is less than
  E12_generate's three arms.
- **Laptop:** the written contexts (before the notebook is built), E13-P, and the
  readout.

## 6. Alternatives

- **B. A capacity arm instead of N2a.**
  - The owner may judge R3's first door met now. The argument: four-times exposure
    did not help even on training queries, and every other recorded lever is
    exhausted.
  - It is faster: one round could test capacity directly.
  - Against it:
    - it skips the two registered checks the rule asks for;
    - if N2b or subwords would have worked, capacity was not needed, and a bigger
      model slows every later run.
  - Its size and steps would be fixed in its own registration.
- **C. The context arm only.**
  - The cheapest round.
  - The remainder's question stays open, and R3 cannot be decided after it.
- **D. Stop at E12.**
  - Record E3 as not met at this model size, with the remainder documented.

## 7. Decisions for the owner

1. **The route:** A, as recommended (§5: context arm + N2a + the laptop check), or
   B, C or D (§6).
2. **Model-written tokens as training context, never as targets** (§2.2). Without
   this, the licensed lever has no form.
3. **The comparison:** against `e12_final`'s existing readings (recommended), or a
   same-session retrain of `e12_final` as well (one more training and generation).
4. **The lever's scope:** the 8,272 phishing query rows DX8 licensed (recommended),
   or all query rows of both classes. The second widens the change beyond the
   licence.
5. **N2a re-measured, two rules fixed in advance:**
   - its copying clause is judged against the natural rate in both directions
     (E12's rule), not N2a's original wording;
   - CAPACITY-OR-REPRESENTATION, together with QC1's registered ruling-out of query
     subwords, counts as R3's strong proof. `DECISION_2026-09-29.md` §4 named the
     plan's FREQUENCY-SPECIFIC or GENERAL, which N2a's Amendment 2 merged into this
     reading.
6. **R1 stays.** `DECISION_2026-10-01.md` flagged it as a simplification candidate.
   Removing it would be a second change, and the arm without it missed more slash
   rows.
7. **E13-P on the laptop** (recommended), or leave path and subdomain subwords
   unasked.

**After the answers:**
1. `PREREG_E13.md` is registered.
2. Code and tests.
3. On the laptop: the written contexts and E13-P.
4. The two notebooks go to Downloads for the owner.
5. The laptop readout.
