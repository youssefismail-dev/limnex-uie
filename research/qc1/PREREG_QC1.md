# QC1 — what gives the queries away: pre-registration

- **Registered:** 2026-09-30, before any QC1 code or measurement.
- **Decided by:** the owner (`DECISION_2026-09-30.md` §3 and §8). QC1 comes first,
  on the laptop, from the existing step-20,000 files, on query content only.
  E12 is the final pass of the E3 programme.
- **Basis:**
  - `E12_ROADMAP.md` §§2–4;
  - `E11_STEP20K_RESULTS.md`;
  - `DECISION_2026-09-30.md` §4: the copying metric and the natural rate.
- **Frozen, unchanged:**
  - every model and generation file;
  - the gate, its thresholds and the tolerance standard;
  - every published result. E11 stays exactly as registered.
- **QC1 trains nothing and generates nothing.** Each reading licenses at most one
  lever family for E12's registration, never a fix by itself.

## 0. In plain words

- **The question.** At step 20,000, what makes `e11_plan`'s queries detectable?
  E3 needs these four cells CLEARED:

  | cell | x | verdict |
  |---|---|---|
  | free, benign QRY | 0.0350 | UNDECIDED |
  | free, phishing QRY | 0.0953 | CONVICTED |
  | control, benign P0Q1 QRY | 0.0581 | UNDECIDED |
  | control, benign P1Q1 QRY | 0.0511 | UNDECIDED |

- **Six questions, each with its rule fixed now:**
  1. Does the model copy common strings at the rate real sites do?
  2. Is the tell in the mix of common and new strings, or in the strings?
  3. Which properties of a query carry the tell?
  4. Are the new strings near misses of common ones?
  5. Are phishing queries pulled toward benign ones?
  6. How much of the tell is the length limit?
- **The answers choose E12's levers,** from a table fixed now (§4).

## 1. Material

- **Generated** (`data/e11_final/gen`, VERIFY PASS, `E11_STEP20K_RESULTS.md` §1):
  - **primary:** `e11_plan` at `final.pt`;
  - **replication:** `e11_ref` at `final.pt`. It is a separate training, with the
    controls hidden in free mode;
  - **per cell,** the five generations pooled.
- **Real:** the gate's own pools.
  - free cells: `name_val` of the class, one URL per host (`_one_per_host`, seed
    20260806);
  - control cells: CR1's state pool (`cr1_score.pools(bundle, label, state)`, the
    `name` population), one URL per host, same seed.
- **Queries** are `FIELDS["QRY"](url)`: the text after `?` and before `#`. Only
  URLs with a query enter.
- **Training queries:** the K11d set, `name_train` and `structure_train` of the
  class (`dc1_compare.training_queries`), with each string's count.

## 2. Instruments (existing; nothing new is measured by a new detector)

- **The detector.** The gate's `real_vs_fake_auc`: character 2–5-grams, logistic
  regression, 5-fold cross-validated AUC.
- **The harness.** The Q3 harness (`q3_content.scored`):
  - three disjoint real draws a, b and c of n each, and one generated draw f of n;
  - floor = AUC(a, b), and x = AUC(c, f) − floor;
  - n = min(1,800, real // 3, generated);
  - D = 12 split seeds, 20260806 + i (Q3.1's); t_crit = 2.201 (df 11).
- **Paired conditions.** Inside one reading, every condition uses the same seed's
  real draws and floor. Conditions are compared seed by seed, with a paired t
  over the 12 seeds.
- **The ladder.** `q3_1_resolve.run_ladder` and its transforms (`qtransforms.py`),
  unchanged: the same rules as Q3.1 and PQ1 on E5.
- **The rate tolerance** is the owner's standard: T = 2·√(max(p(1−p), 1/n)/n).
  - p is the natural rate;
  - n is the cell's generated queries per generation.

## 3. Readings and their rules

**QC1-1 — copying against the natural rate.** Per cell and arm.
- **Rates:**
  - r_g: the share of generated queries found word for word among the training
    queries;
  - r_n: the same share in the cell's real pool.
  - Phishing's r_n is read twice: at the host level (the gate's unit), and on the
    pool's fully domain-disjoint subset (`DECISION_2026-09-30.md` §4.2, check 3).
- **The reading:** UNDER if r_g < r_n − T; OVER if r_g > r_n + T; WITHIN otherwise.
- **The memorisation guard,** read the same way, is the share matching training
  strings seen exactly once: s_g against s_n. OVER here is a memorisation flag.
- **Reported:** the band profile (strings seen 1, 2–4, 5–19 and 20+ times),
  generated against real.

**QC1-2 — the mix, or the content.** Per cell, `e11_plan`, replicated on `e11_ref`.
- **Two conditions, paired:**
  - (i) f drawn at random from the generated queries;
  - (ii) f drawn with exactly round(r_n · n) generated queries found in training
    and the rest from those not found. This is a diagnostic re-draw of the model's
    own output, never a generator. If the generated set is too small for the
    share, n is reduced for this reading, and that is reported.
  - Phishing's re-draw uses the host-level r_n, the gate's unit. The re-draw at
    the domain-disjoint r_n is reported alongside.
- **ρ** = mean x(ii) / mean x(i).
- **The reading:**
  - **MIX-LED:** ρ ≤ 0.5, and x(i) − x(ii) has t > t_crit;
  - **CONTENT-LED:** ρ ≥ 0.8, or the difference is not significant;
  - **SPLIT:** anything else.
- **Replication:** a reading counts only if `e11_ref` reads the same for the cell.
  Otherwise it is "not replicated" and licenses nothing.
- **Reported:** x within each stratum, never-seen against never-seen and found
  against found.

**QC1-3 — which properties carry the tell.** Free cells, both classes.
- **The run:** Q3.1's ladder, verbatim. Stage 1 is on `e11_plan`; stage 2 replicates
  its hits on `e11_ref`.
- **The reading** uses Q3.1's rules:
  - IDENTIFIED if a replicated transform reaches R_T ≤ 0.50;
  - DISTRIBUTED if at least two replicate with |t| > t_crit and none reaches 0.50;
  - separately, SHAPE-SUFFICIENT if `skeleton` keeps R_T ≥ 0.85.
- **E5's readings are reported alongside** (`Q3_1_RESULTS.md`,
  `PHISHING_QRY_RESULTS.md`).

**QC1-4 — near misses.** Per cell, `e11_plan`, replicated on `e11_ref`.
- **The frequent strings** are training queries seen at least 5 times.
- **Near:** a query not found in training that is within one edit (Levenshtein)
  of a frequent string.
- **m_g and m_r** are the near shares among never-seen generated and never-seen
  real queries.
- **BLENDING:** m_g − m_r ≥ 0.05, with a two-proportion 95% CI that excludes 0, and
  the same on `e11_ref`.

**QC1-5 — phishing pulled toward benign.** Free phishing, `e11_plan`, replicated on
`e11_ref`.
- **Paired, on the benign free pool's real draws:**
  - x_gen: generated phishing queries as f;
  - x_real: the phishing pool's real queries as f.
  - n is the smaller of the sets.
- **LEAK:** x_real − x_gen > t_crit · SE (paired), and (x_real − x_gen) / x_real
  ≥ 0.2. That is, the generated phishing queries sit at least a fifth closer to
  benign than real phishing queries do.
- **Reported:** the share of generated phishing queries that are benign-only
  training strings (in benign's training set, not phishing's), against the real
  pool's.

**QC1-6 — the length limit.** Free cells, both classes, `e11_plan`.
- **Paired:** x against the full real pool, and against the pool restricted to
  URLs whose encoding fits `max_len` (256; E11's tokenizer).
- **S** = 1 − mean x(fits) / mean x(all).
- **BUDGET:** S ≥ 0.25, with the paired difference significant.

## 4. Licences for E12 (fixed now)

| reading, in either class | licenses (E12 lever family) | its mechanism |
|---|---|---|
| MIX-LED in a cell, or BLENDING in a cell, or QC1-3 IDENTIFIED on parameter names (`mask_name_chars`) | **query subwords:** subword tokens for the query field only (`DESIGN.md` §4.2) | common strings and names become units, so the model can neither under-weight nor misspell them one character at a time |
| LEAK | **the class at every position:** the class embedding added to every input position | phishing decisions stop borrowing benign's |
| BUDGET | **R1:** `max_len` 384 | the unwritable long queries become writable |
| none of these in a cell | nothing for that cell | the owner decides; capacity (R3) stays locked |

- A lever licensed in either class is available to E12 for both classes, since
  the model is shared. Its predictions are registered per cell.
- **E12's registration** takes the licensed levers and fixes its arms, criteria,
  copying treatment (`DECISION_2026-09-30.md` §4.3) and end rule. It comes back to
  the owner before any Kaggle run.

## 5. Validity, before any reading

| code | check | on failure |
|---|---|---|
| V1 | inputs: `data/e11_final/VERIFY.json` PASS; the generation and bundle files unchanged since it (hashes recorded at the start of the run); the code commit recorded | fixed under an amendment; nothing read |
| V2 | plumbing: `null` equals `identity` exactly on every ladder (Q3.1's check); at the gate's n, the harness's plain excess is within ±0.05 of the gate's x for each cell (Q3.1's kill criterion) | same |
| V3 | the natural rates reproduce `data/e12/sizing.json` exactly (benign 0.2927, phishing 0.4185 and 0.2648) | same |
| V4 | every floor within [0.42, 0.55] (Q3's bounds); a draw outside is VOID and not read | the row is VOID |

**Reading discipline:**
- The runner logs only start, finish and exit.
- Nothing is read before V1–V3 pass.
- The readout is pasted by script.

## 6. Predictions, before any measurement [HYPOTHESIS]

- **QC1-1:** UNDER in every cell and both arms. `e11_plan`'s free mode copies
  19.47% with both classes pooled, against an expected 34.5%.
- **QC1-2:**
  - benign control P0Q1 leans MIX-LED: host-route queries are the short common
    head (`lang=en`);
  - phishing is CONTENT-LED: its matched strings are kit tokens, and its unmatched
    ones are long (median 31 characters);
  - benign free and P1Q1: no confident prediction.
- **QC1-3:** DISTRIBUTED and SHAPE-SUFFICIENT, as on E5.
- **QC1-4:** BLENDING in benign, with low confidence.
- **QC1-5:** LEAK. Phishing's slash residuals already show a pull toward benign
  (`E11_STEP20K_RESULTS.md` §4).
- **QC1-6:** no BUDGET. At most 1.43% and 3.56% of real queries are unwritable.

## 7. Not in QC1

- **No fix, no training, no generation.** No change to the gate, the tolerances,
  the baseline or E11's registered readings.
- **Outside the approved scope** (`DECISION_2026-09-30.md` §5, §8, open question):
  - the three slash rows;
  - benign P1Q1 PTH and SUB;
  - phishing free PTH, whose attribution branch is closed.
- **If the owner extends E12's scope to them,** a registered amendment adds their
  diagnosis before any of their readings. The owner did (option B): see §9,
  amendment 1.

## 8. Cost and order

1. This registration.
2. The code and its tests, then a dry run on a small subset. The dry run is
   stamped and never read as a result.
3. The run: laptop, the pool runner, about 6–8 hours.
4. `QC1_RESULTS.md`, with the licences it gives E12.

## 9. Amendment 1 — the E3 finish line (owner option B; before any QC1 reading)

**Why.**
- The owner widened E12's scope to the E3 finish line, with a strict evidence
  rule (`DECISION_2026-09-30.md` §9).
- The finish-line items outside §3's query cells are diagnosed here. This is
  registered before any QC1 reading.
- **Phishing free PTH stays closed** (`PTH_ATTRIBUTION.md`): E12 reads it and
  QC1 does not diagnose it.
- **For benign P1Q1 PTH, only state-conditioning is tested.** The closed branch's
  path-body ablation ladder is not rerun.

**QC1-7 — the slash: the decision at the host end, teacher-forced.**
`e11_plan` and `e11_ref` at `final.pt`.
- **The rows.** Seeded samples of `name_train` rows (domain frame, rows that
  fit), as `e11_step_check.py` samples them:
  - (a) phishing bare hosts, path exactly `/`: 3,000 rows;
  - (b) benign host-route queries (state P0Q1): up to 20,000 rows, so that the
    data's rare no-slash case (0.45% in K11b's population) has events to
    measure.
- **The encoding.** Each row is encoded with its own state's controls (or plan)
  shown, as control mode generates.
- **The read.** At the host end (the first structural token after the host
  fields), the model's probabilities, renormalised over the ends the state allows:
  - (a) no-slash = (EOS + FRG) / (RS + EOS + FRG);
  - (b) no-slash = QRY / (RS + QRY).
- **The class swap.** Each row is read twice: under its own class token, and
  under the other class's, which is the only change.
- **Readings:**
  - **SLASH-LEAK (a):** across phishing bare hosts, the phishing-token no-slash
    mass tracks the benign-token mass. Spearman ρ ≥ 0.5, with a bootstrap 95% CI
    over hosts whose lower end is above 0.3, in both arms. Reported: the means and
    λ = mean phishing-token / mean benign-token, against the data's 0.
  - **SUPPRESSED (b):** the benign-token no-slash mass is at most half the data's
    rate on the same rows, with a bootstrap 95% CI of the ratio below 1, in both
    arms.
  - **LEAK-CONSISTENT (b):** SUPPRESSED, and the phishing-token mass on the same
    rows is lower still, so the benign value sits between the two classes.
  - **Reported:** benign bare hosts (P0Q0) the same way, against the data's rate.

**QC1-8 — benign P1Q1 PTH and SUB: does the model ignore the query state?**
- **The harness** is §2's, on the gate's field extractors (`FIELDS["PTH"]`,
  `FIELDS["SUB"]`).
- **The populations,** one per host:
  - real P1Q1: CR1's benign P1Q1 pool;
  - real P1Q0: the benign P1Q0 pool;
  - generated P1Q1: control files, pooled.
- **Three excesses:**
  - x_own: generated P1Q1 against real P1Q1 (the gate's cell);
  - x_cross: generated P1Q1 against real P1Q0;
  - x_states: real P1Q1 against real P1Q0.
- **STATE-BLIND:** x_states ≥ 0.02 (the state changes the field), and x_cross <
  x_own with a Welch t over the 12 seeds above t_crit. That is, the generated
  values sit closer to the other state's real values than to their own state's.
  It must hold in both arms.

**Licences added to §4:**

| reading | licenses | its mechanism |
|---|---|---|
| SLASH-LEAK, or SUPPRESSED and LEAK-CONSISTENT | **the class at every position** (§4's lever), with registered predictions on the slash rows | the class signal reaches the host-end decision directly, not only from the first token |
| STATE-BLIND in PTH or SUB | **the plan at every position:** the plan's own embeddings added to every input position | the path and subdomain read the state directly |

- **Both belong to one family:** conditioning read at every position. It reuses
  existing embeddings, so it adds **no parameters**.
- **It does not change the presence mechanism.** The plan tokens and the
  supplied plan stay as they are. K11a and K11b guard presence in E12.

**Validity for the new readings:**
- **V5:** the probe's rows reproduce the data. Phishing bare hosts: the data's
  no-slash share is exactly 0. Benign P0Q1: its no-slash share is within its
  binomial 95% CI of 0.449%.
- **V2 extended:** QC1-8's x_own at the gate's n is within ±0.05 of the gate's x
  (P1Q1 PTH 0.0509, SUB 0.0379).
- **V4** applies to QC1-8's floors.

**Predictions** [HYPOTHESIS]:
- **QC1-7a: SLASH-LEAK.** No real phishing bare host lacks the slash, so the
  misses are most likely benign's 2% borrowed.
- **QC1-7b: SUPPRESSED.** LEAK-CONSISTENT is uncertain.
- **QC1-8:** PTH is STATE-BLIND, since paths before queries are specific (search
  and product pages). SUB: no confident prediction.

**Cost:** QC1-7 takes minutes of CPU. QC1-8 is about 150 detector fits per arm.

## 10. Amendment 2 — frame leakage (2026-09-30, before its measurement)

**Why.**
- QC1's descriptive section (`QC1_RESULTS.md` §6) found three things:
  - the model's head is over-concentrated on strings a site repeats across its
    pages (pagination, languages);
  - the domain-frame corpus holds one URL per host;
  - the page-frame corpus holds about 30 per host and supplies 74.6% of benign's
    query training rows.
- **The hypothesis:** domain-frame query content borrows the page frame's.
- **The owner's delegation** (2026-09-30): laptop diagnostics are the executor's
  decision. Registered here, before the measurement.

**QC1-9 — domain-frame queries pulled toward the page frame.**
- **Scope:** every query cell, `e11_plan`, replicated on `e11_ref`.
- **The page pool** is the class's `structure_val` rows with a query, at the row
  level (the model trains on rows, several per site). Control cells use their
  state's rows only.
- **The harness** is §2's, on the page pool's real draws and floor. Two paired
  conditions:
  - x_gen: the cell's generated (domain-frame) queries;
  - x_real: the cell's real domain pool queries.
  - n is the smallest of the sets.
- **FRAME-LEAK** needs all of:
  - x_real ≥ 0.05, so the frames genuinely differ;
  - x_real − x_gen > t_crit · SE (paired);
  - (x_real − x_gen) / x_real ≥ 0.2;
  - the same in both arms.
- **Reported:** the shares of pagination and language strings (`page=`, `p=` and
  `lang=` followed by a short value) among page rows, real domain queries and
  generated queries.

**Licence (added to §4):** FRAME-LEAK in any cell licenses **the frame at every
position**, the conditioning family of amendment 1:
- the frame's own embedding is added to every input position;
- no new parameters.

**Predictions** [HYPOTHESIS]:
- FRAME-LEAK in benign free and P1Q1. Their inflated strings are page-repeated.
- Not in benign P0Q1, whose head is not over-concentrated.
- Phishing uncertain. Its page frame is smaller: 40.8% of its query rows, at 6
  pages per host.

## Status

    QC1  registered 2026-09-30 (this document), before any QC1 code or measurement
    QC1  code 0c66c2a (tests 10; the dry run of every part and of the reader passed);
         the run started 20:53. The two ladder jobs filled the RAM (4.4 GB: Q3.1's
         run_ladder does not collect after each detector call) and were stopped by
         PID before writing anything; fixed with C3's wrapper (7e245ab, memory only,
         values bit-identical) and rerun
    QC1  amendment 1 (section 9): the E3 finish line (owner option B) -- QC1-7 the
         slash (teacher-forced, class swap), QC1-8 benign P1Q1 PTH and SUB (state
         blindness); registered before any QC1 reading
    QC1  READ 2026-09-30 (QC1_RESULTS.md): validity PASS (V1-V5). Licensed: R1 only
         (benign BUDGET, S 0.29). Not licensed: query subwords, the class at every
         position, the plan at every position. The tell is content-led in every
         cell and lexical rather than shape; no blending, no class leak, not state
         blind; phishing slash rho 0.45-0.46 (rule 0.5). Most predictions wrong.
         Descriptive: the head is over-concentrated; 74.6% of benign query training
         rows come from the page frame (~30 pages per host) -- frame leakage is a
         hypothesis, unregistered. E12's options are the owner's (section 7 there)
    QC1  amendment 2 (section 10, 72d85ef): frame leakage, registered before its
         measurement; READ: NO FRAME-LEAK in every cell and both arms (every gap
         negative: generated domain queries sit further from the page frame than
         real ones). The frame at every position is not licensed
