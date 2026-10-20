# E12 — the final refinement pass: pre-registration

- **Registered:** 2026-09-30, before any E12 code, training or generation.
- **Decided by:**
  - **the owner:**
    - the base, the checkpoint, QC1 first and the finish-line scope
      (`DECISION_2026-09-30.md` §§1–3, 9);
    - E12 as the final pass, with an end rule (§8);
    - the delegation of every laptop decision to the executor.

    The owner approves this plan before any Kaggle run.
  - **the executor:** the recipe below, from QC1's evidence (`QC1_RESULTS.md`,
    amendments 1–2).
- **Frozen, unchanged:**
  - the gate v2, its thresholds and the tolerance standard;
  - every instrument and every published result;
  - E11 as registered (`e11_plan` at `final.pt` is the official baseline);
  - the library's defaults: every change is behind a flag that is off by default.
- **Not used, by the owner's rules:**
  - capacity (R3), more steps (R2) and decoder changes;
  - rejection, masking, a second pass or hybrids;
  - any change to how P and Q are decided.

## 0. In plain words

- **E12 is the last experiment of the E3 programme.** It trains `e11_plan`'s
  recipe with the changes the evidence supports, and reads everything E3 needs.
- **What it changes, and why:**
  1. **The slash joins the plan.** QC1 found the model's own slash decision off
     in both classes. Phishing bare hosts miss it about 0.3% of the time, where
     real ones never do. Benign `host?q` without a slash is written a fifth to a
     ninth as often as it occurs. No native fix was licensed.
     - Decisions the model is **told** are calibrated. QG2-D found "query told:
       0.98 / 1.01", and E11's supplied plan made paths and queries exact.
     - So the slash becomes one more planned bit, supplied from the data like P
       and Q. The model still writes every character.
  2. **A longer length limit (R1, 384 tokens).** QC1 licensed it. Real benign
     queries too long for the model account for 0.29 of the benign free-mode
     tell; 384 recovers about half of them.
  3. **The frame at every position, if and only if QC1-9 licenses it.** QC1-9
     (amendment 2) tests whether the model's domain-frame queries borrow the
     page-frame corpus's repetitive queries (about 30 pages per site; 74.6% of
     benign's query training rows).
     - If its registered rule reads FRAME-LEAK, replicated, the frame token's
       meaning is added at every position of the sequence, with no new
       parameters.
     - If not, this change is not made.
- **What it cannot promise.** Phishing's query content (0.0953, CONVICTED) has no
  lever the evidence licenses. E3 needs it CLEARED. The honest prediction is
  therefore that E3 is **not** met. The final report would then close E3 with
  the evidence.

## 1. Decisions, with their reasons

**D1 — The base: `e11_plan`'s recipe, exactly.**
- The root slash, the clipping repair (cap 5.0 after step 1,000), the presence
  plan with field dropout 0, the supplied plan in free mode, the fixed loss
  constant C = 14,397.885, 20,000 steps, and the raw decode.
- `e12_base` trains it again in this session. Two reasons:
  - it is the same-session control every change is measured against;
  - it is a replicate of E11's `e11_plan`, so the run-to-run spread of every
    cell is measured, not assumed.

**D2 — The slash plan.**
- **The tokenizer flag** `slash_plan` (needs `presence_plan` and `root_slash`)
  makes the plan three tokens after the suffix: `<P:v> <Q:v> <S:v>`.
  - v = 1 when a `/` follows the host (a path body or the bare root slash);
  - v = 0 when none does (`host`, `host?q`, `host#f`).
  - P and Q are unchanged, and P = 1 implies S = 1.
- **Free mode** supplies it as E11's D6 supplies P and Q:
  - The presence table gains S. Per class and suffix, it counts the six states
    that occur: P0Q0S0, P0Q0S1, P0Q1S0, P0Q1S1, P1Q0S1, P1Q1S1. It is built from
    every `name_train` row with the gate's field definitions.
  - Each generation draws its state counts from a multinomial over the six states.
  - Each state is generated with control mode's native call, the suffix drawn from
    w_k(s) ∝ shipped(s) · n(s, k) / n(s).
  - The URLs are then shuffled.
- **Control mode.** P and Q are the request, as before. S is supplied from the
  same table:
  - each state's suffix draw (SC1's state tables) is split by S at that suffix's
    own rate within the state;
  - P1 states always carry S = 1.
- **Why this is native.** The model writes every character. The plan is supplied
  from the data's own frequencies, exactly as the owner approved for P and Q
  (`DECISION_2026-09-29.md` §7.2).
- **Why it should work.** Told decisions are calibrated (QG2-D), and E11's supplied
  plan met the presence tolerance (K11a).
- **The risk, stated.**
  - Phishing's slash rows allow at most about two misses in 4,700 (T ±0.04 free,
    ±0.07 control). So the model must honour `<S:1>` at least 99.93–99.96% of
    the time.
  - Training shows it no counterexample, but nothing guarantees that rate.

**D3 — R1: `max_len` 384** (licensed by QC1-6, benign BUDGET).
- **What changes.** Rows up to 384 tokens are trained instead of dropped, and
  generation may run to 384.
- **The new positions.** Positions 256–383 are new learned embedding rows: 49,152
  parameters, 0.34% of the model. Only rows longer than 256 tokens use them.
  - They are drawn from N(0, 0.02), the model's own initialisation, by a separate
    seeded generator.
  - This is not a width or depth change (not R3). It is the sequence-length data
    decision the repair plan named (`REPAIR_PLAN.md` R1).
- **C is unchanged.**
- **The expected size** [MEASURED, real data only]. At 384, the real free pools'
  unwritable share falls from 1.43% to 0.68% (benign) and 3.56% to 1.87%
  (phishing), about half.

**D4 — The frame at every position (conditional on QC1-9).**
- **If QC1-9 reads FRAME-LEAK in any cell, replicated** (the licence of QC1
  amendment 2):
  - `ModelConfig.frame_every` adds the frame token's own embedding
    (`tok_emb(ids[:, frame_index])`, `frame_index` = 2) to the input at every
    position;
  - no new parameters;
  - every other part of the model is unchanged.
- **If not, D4 is not used,** and the arms are the second set in D6.
- **The arm set is decided by that pre-registered rule,** not by the executor
  after the fact.

**D5 — The start.**
- Every arm starts from E11's shared start, `5aa980f7…`, rebuilt on Kaggle exactly
  as E11 built it: E10's 107-token seeded start plus `<RS>`'s row from seed
  20260929.
- New rows are drawn by separate seeded generators:
  - `<S:0>` and `<S:1>` from seed 20261001;
  - positions 256–383 from seed 20261002.
- Every other row is copied by token and position.
- The part equal to E11's start must hash to E11's value, and the new rows must be
  identical across the arms that have them (V-id).

**D6 — The arms** (one session, one start):

If QC1-9 licenses D4:

| arm | recipe |
|---|---|
| `e12_base` | `e11_plan`'s recipe (D1) |
| `e12_final` | + the slash plan (D2) + R1 (D3) + the frame at every position (D4): **the E3 candidate** |
| `e12_no_frame` | `e12_final` without D4 (leave-one-out) |
| `e12_no_r1` | `e12_final` without D3 (leave-one-out) |

If it does not:

| arm | recipe |
|---|---|
| `e12_base` | `e11_plan`'s recipe (D1) |
| `e12_final` | + the slash plan (D2) + R1 (D3): **the E3 candidate** |
| `e12_no_r1` | `e12_final` without D3 (leave-one-out) |

- **No arm ablates the slash plan.** Its target, the slash rows, is read in every
  arm, and `e12_base` is the arm without it.

**D7 — Generation.**
- E11's 50 jobs per arm: free mode 5 × 8,000 per class at `seed_free(g)`, and
  control 8 states × 5 at `seed_ctrl`. 150 files for the three arms.
  [Corrected before any E12 data: this line said "E11's 200 jobs per arm". 200
  was E11's total over its four arms; see Status.]
- The raw decode.
- Slash-plan arms supply S as D2 says.

**D8 — Readout.**
- `final.pt`, step 20,000, for every arm (owner decision 2).
- E11's instruments unchanged: the gate v2 free cells, CR1's control cells,
  emission, slash, honouring and novelty.
- QC1-1's copy instrument, per class and mode.

## 2. Criteria (fixed now)

| code | criterion |
|---|---|
| **E3** (primary; `e12_final`) | the finish line (`DECISION_2026-09-29.md` §3): every gated cell CLEARED (free: the 10 cells; control: every computable cell), and K12b |
| **K12a** presence | K11a's rule, every arm: free-mode PTH and QRY within T, both classes |
| **K12b** slash | K11b's rule, every arm: the 8 slash rows within T |
| **K12c** no regression | K11c's rule, each arm against `e12_base`: no gated cell's verdict worsens with x up by more than 0.0141 |
| **K12d** guardrails | novelty: K11d's rule against `e12_base`. Memorisation: QC1-1's instrument per class and mode, where neither the copy rate nor the rate of copying strings seen once in training may be OVER its matched natural rate (r_g > r_n + T). UNDER is reported, not a failure (`DECISION_2026-09-30.md` §4) |
| **K12e** training | K11e's clipping rule, every arm: after step 1,000, at most 1% of steps clipped in every length band |

**Outcomes:**
- **A candidate** is an arm passing K12a–K12e. Adopting one is the owner's
  decision.
- **E3 is claimed** only if `e12_final` is a candidate and meets the finish line.
- **The end rule** (`DECISION_2026-09-30.md` §8). E12 is final. If E3 is not met,
  the final E3 report records every criterion's state and the evidence. There is
  no E13 tuning inside this programme.

**Reported, not criteria:**
- each lever's effect, leave-one-out against `e12_final`;
- the run-to-run spread (`e12_base` against E11's `e11_plan`, every cell);
- honouring per state, and S honouring;
- FULL;
- QC1-1 and QC1-2 on `e12_final`.

## 3. Predictions, before any run [HYPOTHESIS]

- **K12b:**
  - every slash-plan arm passes. The phishing rows are the risk (D2).
  - `e12_base` fails the same three rows E11 did.
- **K12a:** every arm passes, as E11's plan arm did.
- **R1** (leave-one-out): benign free QRY is lower with it, by 0.003–0.010.
  Phishing: no change beyond noise (NO BUDGET).
- **The frame at every position,** if licensed (leave-one-out): benign free and
  P1Q1 QRY are lower with it. The page-repeated strings' share moves toward the
  real rate.
- **Phishing free QRY stays CONVICTED in every arm.** No lever targets it.
- **The run-to-run spread** of free cells stays under the retrain allowance
  (0.0141), as E9 measured (at most 0.0108).
- **E3 finish line for `e12_final`: NOT met.** Phishing free QRY is the expected
  failure, and some benign query and P1Q1 cells may stay UNDECIDED.

## 4. Validity

| code | check | on failure |
|---|---|---|
| V1 | hashes and completeness: checkpoints (`final.pt` as training recorded), the code bundle, the tables (SC1's, the shipped table, the presence-slash table), every file with its registered n and seed | fixed under an amendment; nothing read |
| V2 | byte identity on the laptop, at least 250 URLs per arm and mode, including the six-state supplied plan and the S-split control mode | same |
| V3 | flags off are unchanged (tests; the pre-E12 fingerprints); `uie/` unchanged since the code commit | same |
| V4 | the scorer reproduces HC1 C2's E7 cells exactly | same |
| V-id | D5's start: E11's part equal to E11's hash, new rows identical across arms; each arm's settings as registered | same |

## 5. Cost and order

1. This registration.
2. Code and tests: flags off by default; a local smoke of every arm on CPU. Then
   the bundle and the notebooks.
3. **The owner's approval of the Kaggle plan.**
4. `E12_train` (the owner): about 1.1 h per arm.
5. `E12_generate` (the owner): 50 files per arm, about 1.6 h per arm, from
   `final.pt`. [Corrected before any E12 data: this said "200 files per arm".]
6. The laptop: validity, criteria, the readout (about 3.5 h), then
   `E12_RESULTS.md`, and the E3 claim or the final E3 report.

## Status

    E12  registered 2026-09-30 (this document), before any E12 code
    E12  D4 resolved by its registered rule: QC1-9 read NO FRAME-LEAK in every cell
         and both arms (QC1_RESULTS.md 6b) -> the frame at every position is not
         used; the arms are D6's second set: e12_base, e12_final (slash plan +
         R1), e12_no_r1 (slash plan)
    E12  code (this commit's parent chain): uie flags off by default (8a2ed85);
         e12_common, e12_targets (the presence-slash table 5a0eeac9..., collapses
         to E11's presence table, slash rates = E11's targets exactly), bundle and
         notebook builders; the notebooks check E11's start and the 384-token
         memory before any training; local smoke of all three arms passed
         (flags, start parts, byte-identical regeneration); tests/test_e12.py 20
    E12  APPROVED by the owner 2026-09-30 ("approved"): the Kaggle plan as presented --
         bundle uie-e12-code sha 8cd70366... (commit f4a1ee5), E12_train (start check
         and memory preflight before any training, ~3.3 h), then E12_generate from
         final.pt (~5 h); verifier f1fba18
    E12  clarification, before any E12 data exists (no training has run): K12d's
         phishing memorisation reading uses the HOST-LEVEL natural rate, the gate's
         own unit (one URL per host) and DECISION_2026-09-30 section 4.2's
         gate-consistent reference; the domain-disjoint reading is reported beside it.
         The cells are QC1-1's (free benign, free phishing, control benign P0Q1 and
         P1Q1)
    E12  readout tools, before any E12 data exists: e12_score (E11's instruments
         unchanged, plus S honouring from the seeds and QC1-1's copy instrument),
         e12_compare (K12a-e, the finish line, the E3 claim), e12_tables,
         e12_readout (the laptop pipeline), e12_paste; tests/test_e12_compare.py
         16, test_e12.py 22. Dry pipeline test on stand-ins built from E11's
         published files (never in data/e12): every step runs; E12's copy part
         reproduces QC1-1's registered numbers exactly on the same files
    E12  clarifications, before any E12 data exists:
         - text: D7 and section 5 said "200 jobs / files per arm"; the registered
           set is E11's 50 per arm, 150 files (E11's 200 were its four arms'
           total). The code (e12_common.jobs), the notebooks and the approved plan
           always had 150; nothing that runs changes
         - section 3's predictions are scored mechanically: "no change beyond
           noise" and "stays under the retrain allowance" mean |x difference| <=
           0.0141, the registered allowance (E9's measured spread 0.0108 is
           inside it); "lower by 0.003-0.010" means e12_final - e12_no_r1 in
           [-0.010, -0.003]; "fails the same three rows" means the same set of
           K12b rows as e11_plan's K11b at step 20,000
         - QC1-2 on e12_final (reported, section 2) is QC1's instrument unchanged,
           pointed at data/e12/gen (qc1_measure --gen, mix only; its defaults
           read E11's files as before); one arm, so it is not replicated and
           licenses nothing
         - reported beside R1's leave-one-out: the slash plan's effect,
           e12_no_r1 against e12_base (no arm ablates it, D6)
    E12  trained 2026-10-01 (results (19)): VERIFY_TRAIN PASS -- every final.pt
         as training recorded; code f4a1ee5, unchanged since; every start
         contains E10's (26926388...) and E11's (5aa980f7...); the slash arms'
         new token rows identical; settings as registered. 62.9 / 69.4 / 63.6
         minutes (e12_base / e12_final / e12_no_r1).
         K12e PASS for all three arms: 0.00% of steps clipped in every band after
         step 1,000 (7-8 non-finite steps in the shortest band; E11: 7).
         Rows dropped over max_len: e12_base 10,316 (E11: 10,316); e12_final
         3,851 (R1); e12_no_r1 10,415 (the S token lengthens every row by one).
         e12_base's final validation mean 1.5203 (E11's e11_plan: 1.5186).
    E12  noted after training, BEFORE any generation (the training log's quick
         check: n = 200 per state, suffix com, the final weights; not a
         registered reading):
         - told S = 0, the slash arms mostly write the slash anyway where the
           data almost never omits it: benign P0Q1S0 honoured 13.0% (e12_final)
           and 34.5% (e12_no_r1); benign P0Q0S0 96.5% / 93.5%
         - why: in benign name_train, host?q without the slash is 67 rows of
           14,906 (0.45%); a bare host without it, 4,856 of 236,092 (2.06%).
           Phishing has neither, so the supplied plan never asks phishing for
           S = 0, and its low S = 0 figures do not enter any reading
         - expected consequence, stated now: benign control P0Q1's slash row
           (real 99.55%, T 0.24 points) may fail again in the slash arms. If S = 0
           is ignored as often as the quick check shows, the arm writes the slash
           at about 99.8-99.9%. The bare-host rows are not affected the same way
         - phishing P0Q0S1 honoured 99.5% (one miss in 200, whether slash or
           field), too coarse to resolve the 99.96% the phishing bare-host rows
           need (D2's stated risk); the readout resolves it
         - section 3's predictions stay as registered and are scored as
           registered. Nothing changes: E12_generate runs as registered
    E12  READ 2026-10-01 (E12_RESULTS.md): VERIFY PASS (byte identity in every arm
         and mode; scorer exact). E3 NOT MET for e12_final: 6 cells not cleared
         (free phishing QRY +0.1141 CONVICTED; control benign P0Q1 QRY, P1Q1
         SUB/PTH/QRY and phishing P1Q0 PTH UNDECIDED) and 3 slash rows missed.
         No candidate: every arm fails K12b; K12a, K12c, K12d and K12e pass in every
         arm. The slash plan moves E11's three failing slash rows toward the real
         rates but not into tolerance; R1 bought nothing measurable. Predictions:
         6 held, 3 did not, 1 not applicable. By the end rule this is the final E3
         report of this programme; E13 is the owner's decision (open)
