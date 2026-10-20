# E17 — option A on the wider baseline, and B5 (the query-plan decode rule)

- **Registered:** 2026-10-07, before any E17 code, training or generation. The gates
  (§3) are run after this document is committed. **Awaiting the owner's gate A**
  (the design) and gate B (the gates' measured numbers).
- **Rests on:** `research/native/DECISION_2026-10-07.md` (the adoption of `e16_cap`, the
  owner's choice of E17, verbatim), `research/e16/E16_RESULTS.md`,
  `research/e17/ACQUISITION_E17A.md` (stream 2), `research/e15/PREREG_E15.md` (the gates,
  the injector).
- **Baseline:** `e16_cap` (official since 2026-10-07), with its 50 files.

## 0. In plain words

**Two levers, each judged on its own, then combined:**
1. **B5, a decode rule. No training.**
   - When a supplied plan says the URL has a query, the URL may not end, or start a
     fragment, before its query begins.
   - This targets the slash row, which fails because the model drops the query on the
     rare "no slash" plans.
   - It is judged on `e16_cap`'s own weights.
2. **Option A, more data.**
   - Stream 2 (57,292 new benign query rows) is injected alongside stream 1 into
     `e16_cap`'s exact recipe. Each old row of a stratum now brings two new rows.
   - **The question:** does the wider model turn more query practice into a lower far
     cell (free phishing QRY, 0.0756)? In E15, benign query practice moved it −0.035 on
     the smaller model.

**The order:**
- B5's generation runs first, needing no training, and is read on the laptop while
  `e17_a` trains.
- `e17_a` is then generated with whichever decode B5's reading leaves standing. So A is
  always judged against a reference with the same decode.

## 1. The owner's decision, verbatim

> Your recommendation for E17 is perfect. Let's attack the remaining issues
> simultaneously:
> E17 will be Option A (the doubled data stream we acquired) injected into this new
> wider baseline. Proceed with running the P1-P8 quality gates on the downloaded stream.
> Alongside this, we will implement Option B5 (the decoding rule) to structurally force
> the slash plan and bypass the training-induced query dropping.
> Proceed to draft the registration and execute the P1-P8 gates for E17.

(The rest of the message, the adoption of `e16_cap`, is recorded in
`DECISION_2026-10-07.md` §1.)

## 2. B5: the query-plan rule

**Where the failure is.**
- On `e16_cap`, the benign control P0Q1 slash row reads 0.99987 against the real 0.99551
  (T 0.00244).
- D1 places the slash exactly. The miss comes from the 71 "no slash" (`host?query`)
  plans: the model ends the URL instead of starting the query, so those URLs leave the
  host+query pool.
- The quick check shows the query kept on 2.5% of those plans (`e15_new`: 5.5%;
  `e12_d1p`: 46% in its files). It gets worse as models learn queries better.

**The rule, `generate(..., query_rule=True)`, off by default:**
- **Applies:** only to supplied plans with Q = 1 (every plan in both modes is supplied,
  E12 D2).
- **When:** at every sampled step from the label marker until `<QRY>` has been sampled.
- **What:** the tokens that would end the URL or start a fragment before the query
  (`<EOS>`, `<FRG>`) get probability 0. Their mass goes to `<QRY>`.
  - So the probability of ending the current field at that step is unchanged; the URL
    then starts its query instead of ending.
  - Every other token keeps its probability exactly. This is the same exact
    redistribution as D1 (`slash_redistribute`), applied after D1, so both stay exact.
- **Not touched:** plans with Q = 0; every step after `<QRY>`; every weight.
- **Decode of every B5 arm:** raw + D1 + D1′ + B5.

**Arm `e16_b5`:** `e16_cap`'s weights with this decode, E12's 50 jobs at E12's seeds.
Generation only.

**B5's criteria** (`e16_b5` against `e16_cap`: the same weights, so the decoder rules,
`PREREG_E13.md` §3.2 with R-a′):

| code | criterion |
|---|---|
| **K17-B1** | all 8 slash rows within T (K12b) |
| **K17-B2** | ~~Q honouring 100% on every Q = 1 plan, both modes (by construction; checked from the files).~~ **Amended (§10, adopted 2026-10-07): on every Q = 1 plan, both modes, no URL ends or starts a fragment before its query begins (checked from the files).** No control state's P, Q or S honouring falls (amendment 1) |
| R-a′ | no gated cell REGRESSES (`DECISION_2026-10-05.md` §2.3) |
| R-b | presence: K12a within T, and no free PTH or QRY rate moves materially except QRY toward its target |
| R-c, R-d | obedience and copying, as `PREREG_E13.md` §3.2 |
| R-e, R-f | validity; a single pass (the rule rescales end-token logits only) |

**Adopted iff all hold.** B5 then becomes the decode of `e17_a`'s generation.

## 3. Option A: stream 2 on `e16_cap`

### 3.1 The stream and its gates

**Stream 2:** `data/e17/qry_extra2/`.
- benign P0Q1: 14,863 rows, `23c64419…`; benign P1Q1: 42,429 rows, `a153e75e…`;
- by E15's procedure (rounds 9–16), with every stream-1 domain excluded;
- held unused until this registration's gate B.

**Gates P1–P8, as `PREREG_E15.md` §9.5, with one change: the reference is the old corpus
plus stream 1,** since `e16_cap` trained on both.

| gate | on stream 2 |
|---|---|
| P1 shift | DX1's rule with stream 2 as the twin (E13-D's harness, seeds and n rule): NO SHIFT |
| P2 coverage | the gain in validation-template coverage from adding stream 2 to old + stream 1 is at least g · log2(1 + r) − 0.01. Here g is DX3's per-doubling gain and r = N_new2 / (N_old + N_new1) = 0.5, so the bar is g · 0.585 − 0.01 |
| P3 leakage | no `split_random` domain; no exact URL in any train or validation stream; one URL per domain; **no stream-1 domain** |
| P4 population | suffix and crawl TV against the old training rows ≤ the train-vs-validation reference |
| P5 diversity | distinct templates at equal size ≥ 0.9 × the old twin's; top-10 share ≤ the old twin's + 0.05 |
| P6 novelty | never-seen share (against old + stream-1 queries) ≥ the validation pool's own − 0.05 |
| P7 contamination | no host or domain in the phishing feed (frozen at d0642d7) |
| P8 the numbers | counts; the rate; appended rows and added query positions per batch; the hashes; the seeds |

- **A stratum is admitted iff every gate passes.** A stratum that fails is not
  injected, and A runs on what is admitted.
- **If neither stratum is admitted, A does not run.** E17 is then B5 alone.

### 3.2 The arm `e17_a`

| | `e16_cap` | `e17_a` |
|---|---|---|
| model, start | d_model 512 …; E12's start at width 512 (`caca0107…`) | the same: the same start |
| data | corpus2 + stream 1 | corpus2 + **stream 1 + stream 2** |
| injection | one new row per old stratum draw (rate 1) | **two per old draw (rate 2)**: one draw each from a pool of streams 1 and 2, per stratum. Query-only targets, weight 1.0 |
| batches, objective, steps, lr, accumulation | `e16_cap`'s | the same. The old rows' batch digest is `adbddc2a…` |
| decode | raw + D1 + D1′ | raw + D1 + D1′, **+ B5 if B5 is adopted** |

**The injector, generalised.**
- The rate per stratum is r = N_new / N_old, no longer capped at 1. Each old draw appends
  ⌊r⌋ new rows, plus one more with probability r − ⌊r⌋ (the existing coin).
- At r ≤ 1 this is today's injector bit for bit. At r = 2 there is no coin.
- Every new row is still drawn at the old rows' per-row rate.

### 3.3 Criteria (`e17_a` against its reference: `e16_b5` if B5 is adopted, else `e16_cap`)

A retrained model, so the retraining allowance (0.0141) applies.

| code | criterion |
|---|---|
| **K17-1** the question | free phishing QRY **or** control benign P0Q1 QRY **or** control benign P1Q1 QRY falls by more than 0.0141, and none of the four query cells (those three, plus free benign QRY) rises by more than 0.0141 |
| **K17-2** fit | benign held-out query NLL (M3) falls against `e16_cap`, interval below 0, pooled. Per stratum and phishing reported |
| **K17-3** band profile | M2: no overshoot (old + both streams as the reference) |
| G1–G10 | as `PREREG_E16.md` §5, against the reference. The copying, novelty and band references are old + both streams for `e17_a` (`PREREG_E15.md` §13.2's rule). **G5 strict** |

**Adopted iff all hold.** Replacing the baseline is the owner's decision.

## 4. The report

- **B5:** the slash rows; Q, P and S honouring; and how many URLs the rule changed (the
  moved mass and the rows whose query it started).
- **A, as E15's four-way report:**
  - uptake of stream-2-only strings, against `e16_b5`/`e16_cap` as the chance reference;
  - the realised dose;
  - coverage and diversity;
  - the trade-off (G2, G3, the per-field table).
- **The fit against the tell,** for the remaining E3 cells.
- **E3,** read for the record on `e17_a` with its decode.

## 5. Validity

| code | check |
|---|---|
| V1 | hashes and completeness: the bundle, the streams, `e16_cap`'s weights, `e17_a`'s `final.pt`, every file with its registered n and seed |
| V2 | byte identity on the laptop CPU, at least 250 URLs per mode, with each arm's own decode (B5 included) |
| V3 | options off unchanged: the full suite. `query_rule=False` is generation bit for bit; r = 1 is today's injector bit for bit |
| V4 | the scorer reproduces `e12_final`'s published cells exactly |
| V-start | `e17_a`'s start equals `e16_cap`'s recorded start (`caca0107…`), rebuilt on Kaggle. The width-384 check as in E16's amendment 2 |
| V-batch | the old rows' digest equals `adbddc2a…` |
| V-mass | per stratum, appended rows = 2 × old rows drawn, exactly; the query positions within 3 SE of P8 |
| V-stream | the training record's hash of the pooled stream, and its strata and counts, equal the registered ones |
| V-B5 | on the laptop CPU, unit tests: off is bit-identical; a Q = 1 plan never ends before `<QRY>`; Q = 0 plans are untouched; the redistribution leaves every non-end token's probability exactly as it was |

## 6. Predictions, preliminary [HYPOTHESIS] (formalised at gate B)

| item | prediction |
|---|---|
| gates | P1 NO SHIFT (the same crawls and procedure as stream 1); P3–P7 pass. **P2 is the risk:** coverage grows about logarithmically with data, so stream 2 adds less than stream 1's +2.8/+3.8 points. P1Q1's bar is about 1.5 points, so P1Q1 may fail P2 (about 40%) |
| B5 | **K17-B1 PASS (about 85%):** with the queries kept, the P0Q1 row returns to about its real 99.55%. Cells move at noise level, because the rule changes only the few Q = 1 rows that would have ended early. Free phishing QRY may get slightly worse: forced queries where the model wanted to stop. R-a′ PASS (about 75%) |
| A | far cell −0.005 to −0.025 (transfer, with diminishing returns over E15's −0.035). **K17-1 PASS about 30%, through the far cell.** P1Q1 QRY and P0Q1 QRY −0.000 to −0.008. Fit −0.01 to −0.03. Costs within noise at twice E15's dose. G5 passes only with B5 |

## 7. Execution steps and gates

| step | where | what | gate |
|---|---|---|---|
| 1 | laptop | this registration, committed; then the gates P1–P8 on stream 2 (`e17_gates.py`) and P8's numbers, appended as amendment 1 | **gates A and B: the owner approves the design and the numbers** |
| 2 | laptop | code: B5 (`uie/model.py`), the generalised injector (`uie/train.py`), the pooled stream, tests (V-B5, V3), the full suite, the dry run, the bundle, the notebooks `E17_decode`, `E17_train`, `E17_generate`, `E17_KAGGLE.md`, the reading scripts | — |
| 3 | Kaggle | `E17_decode` (`e16_b5`'s 50 files, about 2.3 h), then `E17_train` (`e17_a`, about 3.5 h) | — |
| 4 | laptop | B5's reading (verify, score, the decoder criteria), while `e17_a` trains; then VERIFY_TRAIN and K17-2 | — |
| 5 | Kaggle | `E17_generate` (`e17_a`, with the decode B5's reading leaves, about 2.3 h) | — |
| 6 | laptop | verify, score, the criteria, `E17_RESULTS.md` | **the owner's decision** |

## 8. Unchanged and closed

- **Unchanged:**
  - gate v2, every threshold and tolerance, and E3;
  - G5 for every arm;
  - `e16_cap`'s files (not regenerated);
  - the closed branches: the query weight (N2b), the context arm, D2, the PTH
    attribution branch, boundary weighting, more steps (R2).
- **B5 is a decoder rule** that the owner authorised in this decision, in the scope of
  §2 only.

## 9. Amendment 1 (2026-10-07): the measured numbers, before any E17 code or training (for gate B)

The gates ran on the laptop (`research/e17/e17_gates.py`, committed at 14cdc37 before
it ran; record `data/e17/gates.json`).

### 9.1 In plain words

- **Both strata pass every gate, so both are admitted.**
- **The closest margin is P4 on P1Q1:** suffix mix TV 0.0903 against the train-vs-
  validation reference of 0.0926.
- **P2, the gate I predicted might fail P1Q1, passes:** +1.72 points of template coverage
  against a bar of 1.51.
- **The dose:** two new rows per old stratum draw. That is 9.48 appended rows per batch
  and 282 added query positions per batch: query mass +15.8% over `e16_cap`, scored mass
  +1.96% over `e12_final`'s.

### 9.2 The gates

| gate | benign P0Q1 | benign P1Q1 |
|---|---|---|
| P1 shift | NO SHIFT (x_NV +0.0112, n 600) | NO SHIFT (x_NV +0.0052, n 1,800) |
| P2 coverage | 83.74% → 85.02%: **+1.28 points** (bar +0.89) | 71.12% → 72.85%: **+1.72 points** (bar +1.51) |
| P3 leakage | `split_random` 0; exact train/validation URLs 0; stream-1 domains 0; one per domain | the same |
| P4 population | suffix TV 0.107 (ref 0.157); crawl TV 0.012 (ref 0.029) | suffix TV 0.0903 (ref 0.0926); crawl TV 0.011 (ref 0.013) |
| P5 diversity | distinct-template ratio 1.050; top-10 0.408 (old 0.418) | 0.963; top-10 0.163 (old 0.163) |
| P6 novelty | never-seen 54.9% (validation 53.2%; bar 48.2%) | 72.8% (validation 73.6%; bar 68.6%) |
| P7 contamination | 0 | 0 |
| **admitted** | **yes** | **yes** |

New-template share (stream-2 templates absent from old + stream 1): 19.3% (P0Q1) and
27.4% (P1Q1).

### 9.3 P8: the frozen numbers

| stratum | N_old | N_new1 | N_new2 | rate | old rows per batch | appended per batch | L_old | L_new1 | L_new2 | added query positions per batch |
|---|---|---|---|---|---|---|---|---|---|---|
| benign_P0Q1 | 14,863 | 14,863 | 14,863 | 2.0 | 1.230 | 2.460 | 24.10 | 23.93 | 24.08 | 59.05 |
| benign_P1Q1 | 42,429 | 42,429 | 42,429 | 2.0 | 3.511 | 7.022 | 32.27 | 31.81 | 31.74 | 223.12 |

- **Appended rows per batch:** 9.482 (`e16_cap`: 4.741).
- **Added query positions per batch:** 282.2, against `e12_final`'s 749.8: query mass
  +37.6%. Against `e16_cap` (which adds 141.1): **+15.8%**. Scored mass +1.96% of C.
- **Expected over 20,000 steps:** appended rows 2 × the old rows drawn (about 189,600),
  query positions about 5.64M.
- **Files:**
  - stream 1: `benign_P0Q1` `4ef1b466…`, `benign_P1Q1` `ce66a7db…`;
  - stream 2: `benign_P0Q1` `23c64419…`, `benign_P1Q1` `a153e75e…`.
  - The pooled training stream's hash is fixed when it is built (step 2) and checked by
    V-stream.
- **Seeds:** the injector's draw 20261007 and coin 20261008 (no coin at rate 2).

### 9.4 For gate B

- Approving these numbers admits both strata at rate 2, as §3.2 registers.
- Step 2 (code, dry run, bundle, notebooks) follows. `E17_decode` (B5, no training) can
  run first on Kaggle.

## 10. Amendment 2 (proposed and ADOPTED 2026-10-07, in force): K17-B2's first clause

Found at step 2, while writing the reading scripts, **before any E17 generation or
training**. It was proposed at f048da5 and adopted by the owner (§10.5) before any E17
output was downloaded or read. `e17_compare.py` judges by the amended clause
(`AMENDMENT_2 = True`) and still writes the registered reading beside it.

### 10.1 In plain words

- K17-B2's first clause asks for "Q honouring 100%" on every plan that says "this URL
  has a query".
- B5 can only stop a URL from **ending** before its query starts. Two things are out of
  its reach:
  - a URL whose "?" is followed by nothing (`host/?`): the model itself writes these
    after choosing the "?", and B5 does not act after the "?" (§2: "every step after
    `<QRY>`" is not touched);
  - a URL cut off at the length limit before its query: no end token is ever sampled.
- The instruments count both as "no query". So the clause as worded fails on URLs B5
  never touches. Calling it "by construction" was my error.

### 10.2 The evidence

On `e16_cap`'s adopted files (no B5; already read in E16), for every Q = 1 plan in both
modes, with each plan rebuilt from the seeds:

| | URLs |
|---|---|
| Q = 1 plans | 44,514 |
| with a query (the instruments) | 44,018 |
| ended before the query began (**what B5 removes**) | 458 |
| "?" then nothing (the model's own) | 36 |
| cut off at max_len before the query | 2 |

- About 38 URLs would fail "100%" whatever B5 does.
- **A laptop check of B5 on `e16_cap`.** This is not a reading. It uses the dry run's
  plans, n = 192 each. Each cell shows a begun query / a non-empty query:

  | plan | without B5 | with B5 |
  |---|---|---|
  | benign P0Q1S0 ("no slash") | 8 / 8 | **192 / 192** |
  | benign P0Q1S1 | 192 / 192 | 192 / 192 |
  | benign P1Q1S1 | 190 / 190 | 192 / 192 |
  | phishing P0Q1S0, a plan the files never draw (every real phishing host+query URL has the slash) | 20 / 20 | 192 / 191 |
  | phishing P0Q1S1 | 192 / 191 | 192 / 191 (the same empty URL as without B5) |
  | phishing P1Q1S1 | 186 / 185 | **192 / 191** (the same empty URL as without B5) |

  - On every plan the files draw, B5 created no empty query: each empty query with B5
    is the URL the model writes without it.
  - The one empty forced query is `…godaddysites.com/?`, on the plan that never occurs.
    There the model wrote a "/" as a character inside the host, which D1 (a rule on the
    end markers) does not police.
  - Every one of the 73 real "no slash" Q = 1 plans in `e16_cap`'s files is benign, and
    none has a slash written.

### 10.3 The wording (adopted, in force)

K17-B2, first clause: **on every Q = 1 plan, both modes, no URL ends or starts a fragment
before its query begins.**
- Checked from the files: every Q = 1 URL that ended within max_len has a "?" before
  any "#". URLs cut off at max_len are counted apart.
- Reported beside it, not judged: Q honouring by the instruments, the empty queries and
  the cut-offs, for `e16_b5` and `e16_cap`.

Unchanged:
- the second clause (no control state's P, Q or S honouring falls);
- every other criterion;
- B5 itself: no code change, and the bundle and notebooks stand.

### 10.4 The alternative

- **Keep the clause as registered.** B5 then fails K17-B2 on URLs it does not touch. B5
  is not adopted, and `e17_a` is generated without it (`E17_generate_d1p`).
- **Widening B5** (no end right after "?") is not proposed. It goes beyond the rule the
  owner authorised (§2), and it still cannot reach the cut-offs.

### 10.5 The owner's decision, verbatim (2026-10-07)

> You are completely right about the edge cases—I choose Option 1.
>
> Please adopt Amendment 2 for K17-B2 exactly as proposed, so the rule accurately
> reflects B5's actual scope: preventing premature URL endings before the query begins,
> rather than requiring non-empty post-query content.
>
> Please record the amendment as formally adopted and in force before any E17 results
> are read.

- Adopted exactly as proposed: §10.3's wording, nothing else.
- B5, the bundle, the notebooks and every other criterion are unchanged.

## Status

    E17  registered 2026-10-07 (this document), before any E17 code, training or
         generation, and before the gates are run. Awaiting the owner's gates A and B.
    E17  Amendment 1, 2026-10-07 (section 9): gates P1-P8 on stream 2 -- both benign
         strata ADMITTED, every gate PASS (closest: P4 P1Q1 suffix TV 0.0903 vs 0.0926;
         P2 P1Q1 +1.72 vs +1.51). Rate 2: 9.48 appended rows and 282 added query
         positions per batch (+15.8% query mass over e16_cap). Before any E17 code or
         training. Awaiting the owner's gates A and B.
    E17  Gates A and B APPROVED by the owner, 2026-10-07, after Amendment 1, verbatim:
         "okay , pls continue". Both strata admitted at rate 2; B5 as section 2.
         Step 2 (code, dry run, bundle, notebooks) begins after this line.
    E17  Step 2 DONE, 2026-10-07. Code commit 04d5c00 (B5 = generate(query_rule);
         the injector's rate above 1; both off by default); full suite 1,036 PASS.
         Dry runs PASS (b5: 1/48 -> 48/48 "no slash" plans keep their query on
         e16_cap, Q0 plans identical; inject: rate 2, the old rows' digest equals the
         replay over 400 batches; train: real width-512 steps). Pooled stream:
         114,584 rows, sha256 128b7d69f029e05554cfc9cfce75545c94a19c4f40f013f398a5d580706b4855.
         Bundle uie-e17-code.zip, built from 04d5c00:
         sha256 2905df34c8a8e05f1acf16985bc3e4884ce301bd4ff6496328a37dea0eec3830.
         The four notebooks ran end to end on the laptop CPU, with Kaggle's paths
         and tiny jobs; the GPU, memory and start checks were stubbed there (no GPU;
         torch 2.13) and run for real on Kaggle. Reading scripts: e17_verify,
         e17_score (+ e17_score_run), e17_fit, e17_compare.
    E17  Amendment 2 PROPOSED, 2026-10-07, before any E17 generation or training
         (section 10): K17-B2's first clause, as worded, cannot pass whatever B5 does
         (on e16_cap: 36 empty queries and 2 cut-offs that B5 does not touch).
         NOT in force. Awaiting the owner; e16_b5 is not read until the owner decides.
    E17  Amendment 2 ADOPTED by the owner, 2026-10-07, exactly as proposed, before any
         E17 output was downloaded or read (verbatim in section 10.5: "I choose
         Option 1 ... Please record the amendment as formally adopted and in force
         before any E17 results are read"). In force: K17-B2's first clause is
         section 10.3's -- no Q = 1 URL ends or starts a fragment before its query
         begins. e17_compare.py: AMENDMENT_2 = True.
    E17  E17_decode READ, 2026-10-08 (results (31).zip): VERIFY_DECODE PASS (V2 957/957
         identical with B5 on), V4 PASS. e16_b5 vs e16_cap: K17-B1 PASS (all 8 slash
         rows within T; benign control P0Q1 0.99987 -> 0.99525 vs the real 0.99551),
         K17-B2 PASS (as amended: 0 of 44,514 Q = 1 URLs end before their query; 1
         cut off; 37 empty, reported), R-a' PASS (no regression; one verdict change
         recorded, not counted: control benign P1Q1 PTH UNDECIDED -> CONVICTED,
         +0.0031 [-0.0033, +0.0095]), R-b, R-c, R-d, R-e, R-f PASS -> B5 ADOPTED.
         e17_a is generated with raw + D1 + D1' + B5 (E17_generate_b5). E3 on e16_b5:
         NOT MET -- 3 cells (free phishing QRY 0.0733, control benign P1Q1 PTH
         0.0539, SUB 0.0431), 0 slash rows.
    E17  E17_train VERIFIED, 2026-10-08 (results (32).zip): VERIFY_TRAIN PASS --
         V-start (caca0107...), V-batch, V-mass (appended exactly 2 x the old draws;
         5,642,436 query positions vs P8 5,644,000, z -0.10), V-stream, the recipe,
         V3/V-B5 tests; no clipping after step 1,000; final val mean 1.4870
         (e16_cap 1.4890).
    E17  K17-2 PASS, 2026-10-08 (data/e17/k17_2.json; checkpoints only, before
         e17_a's files exist): benign held-out query M3 e16_cap 1.5113 -> e17_a
         1.4815, diff -0.0299 [-0.0330, -0.0269]; P0Q1 -0.0301, P1Q1 -0.0298.
         Reported: phishing -0.0171 [-0.0281, -0.0062].
    E17  READ, 2026-10-08 (research/e17/E17_RESULTS.md; results (33).zip):
         VERIFY_GENERATE PASS (957/957, decoded with B5). e17_a vs e16_b5: K17-1 FAIL
         (free phishing QRY 0.0733 -> 0.0802, +0.0069; control benign P0Q1 QRY
         -0.0105; P1Q1 QRY 0.0000); K17-2, K17-3, G1-G10 PASS (G5 strict PASS).
         e17_a NOT ADOPTED. E3: e16_b5 3 cells + 0 slash rows; e17_a 4 + 0.
         Whether e16_b5 becomes the baseline, and E18, are the owner's decisions.
