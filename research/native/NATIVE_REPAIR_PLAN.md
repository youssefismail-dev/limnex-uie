# Native E7 repair — design review and plan (revision 2)

**Status: PLAN.**
- **Direction and order:** approved by the owner, 2026-09-26.
- **This revision:** the deeper review the owner asked for before implementation.
- **Each experiment** is registered in its own document before its code.
- **History:** revision 1 is de54131; §9 lists every change and its reason.

**Status update, 2026-09-28** [DECISION] (`DECISION_2026-09-28.md`). The lines this
supersedes in §0 and §8 are kept as written and marked.
- **Baseline:** `e9_ln` with the raw decode (`uie/baseline.py`) replaces E7 as the
  active base.
- **N1 (E8):** permanently unread and obsolete.
- **N2a:** done (`N2A_RESULTS.md`). **N2b, N2c:** not licensed.
- **N3:** `PREREG_E9` §6 sends the adopted baseline here. Not yet scheduled: the
  high-n free reading has not been run.
- **Active question:** the query-emission gap (`research/qgap/QG_PLAN.md`).
- **Root slash:** the approved representation fix was tested only inside E8, which
  is now unread, so it has no reading. The baseline's tokenizer has no root-slash
  token.

**The question.** Given everything measured so far, what is the smallest
scientifically defensible native E7 repair program that can get through E3 in both
Free and Controlled modes?

**Evidence tags used throughout:**

| tag | means |
|---|---|
| **[CONFIRMED]** | measured, or established by code or arithmetic, in a cited record |
| **[HYPOTHESIS]** | supported by evidence, not established |
| **[DIAGNOSTIC]** | a measurement taken to inform a decision; never a mechanism, never shipped |
| **[DECISION]** | an owner or engineering decision, not an empirical claim |

---

## 0. Owner decisions and the native principle [DECISION]

- **E7 is the active research and engineering base.** E5, R4, the E3 report,
  gate v2 and every published number stay frozen historical records.
  *(Superseded 2026-09-28: the base is `e9_ln` with the raw decode; see the status
  update above.)*
- **The root-slash fix is approved.** It is a representation (tokenization) fix,
  not a scoring change.
- **DC1:** adopted for FULL FREE (`uie/free.py`, 296e3c3); rejected for FULL
  CONTROL (1.94× cost).
- **The native principle.** The production solution must be native generation and
  native conditioning. Production resampling or rejection, masking, decode-time
  second predictions and hybrids may exist **only** as diagnostic or reference
  instruments. None of them can be the mechanism by which a criterion is met.
- **No E3 realism claim** is made for a cell whose real sample is too small to
  estimate (§2). Such cells are reported as limitations.

---

## 1. Current state [CONFIRMED unless tagged]

### 1.1 FULL FREE — E7 + DC1

Gate v2 functions, n_main 40,000 / 25,000, G = 5, D = 10.

| cell | benign | phishing |
|---|---|---|
| LBL / SUF / SUB | CLEARED (+0.0041 / +0.0153 / +0.0085) | CLEARED (+0.0171 / +0.0076 / +0.0314) |
| PTH | +0.0335 CLEARED; its interval narrowed, its mean did not demonstrably move | **+0.0446 UNDECIDED** |
| QRY | **+0.0718 CONVICTED** | **+0.1177 CONVICTED** |
| FULL (aux.) | +0.2189 | +0.4012 |

- **At the canonical instrument (n_main 8,000), E7 + DC1 has not been read.** E7
  alone has (HC1 C2):
  - benign PTH +0.0369 UNDECIDED;
  - benign QRY +0.0541 UNDECIDED at n = 289;
  - phishing PTH +0.0604 UNDECIDED;
  - phishing QRY pooled UNDECIDED at +0.0924. The pooled estimator can convict or
    abstain, never clear.

### 1.2 FULL CONTROL — E7 + SC1 arm T (held resampling), no DC1

CR1's instrument (pools restricted to the state, n_main 3,000, G = 5).

- **CONVICTED:** benign P1 Q1 SUB +0.0715, PTH +0.0984, QRY +0.0843; benign P0 Q1
  QRY +0.0889.
- **UNDECIDED:** phishing P1 Q0 LBL +0.0427, SUB +0.0486, PTH +0.0511.
- **CLEARED:** 15 of the 22 computable cells.
- **These readings depend on held resampling.** The native target excludes it.
- **Native honouring** (C1, E7 with no resampling, both controls honoured,
  n = 1,000 each; C1 used the class-marginal suffix table):

| | P1 Q1 | P1 Q0 | P0 Q1 | P0 Q0 |
|---|---|---|---|---|
| benign | 96.4% | 98.9% | 94.1% | 98.3% |
| phishing | 71.8% | 95.4% | 79.4% | 97.4% |

### 1.3 Two corrections to the owner's earlier summary

- **DC1 does not confirm "control leakage into earlier fields"** [HYPOTHESIS, not
  a finding].
  - On likelihood, DC1's validation-confirmed blend pulled LBL toward free mode in
    every benign state (w 0.30–0.59).
  - At the gate, no LBL change was significant. The one significant cell, benign
    P1 Q1 SUB, had w = 1.08.
  - SUB is generated after LBL (`SUF → LBL → SUB → PTH → QRY`), so its gain may be
    downstream of LBL's blend. Nothing separated the two.
- **The missing phishing QRY setting does not explain why no QRY cell cleared.**
  Benign QRY had its settings and stayed CONVICTED.

### 1.4 What E3 gates — and what it does not

E3's gate (DESIGN §9, rebuilt E3) is **every field's excess below 0.05**, CLEARED
under §26.1. Novelty is reported alongside and whole-URL AUC is auxiliary. The
owner's objective extends the gate to controlled mode per measurable joint state,
on CR1's instrument [DECISION].

| category | E3-gated | status |
|---|---|---|
| QRY realism | yes | failing in every measurable query cell: 2 free, 2 control |
| PTH realism | yes | free: 1 UNDECIDED. Control: 1 CONVICTED, 1 UNDECIDED |
| LBL / SUB realism | yes | free: CLEARED. Control: 1 CONVICTED, 2 UNDECIDED |
| SUF | yes | CLEARED everywhere measured (table-driven since SC1) |
| FULL root-slash artifact | **no** (auxiliary) | present in every mode |
| control honouring | **no**: the objective's own requirement, **kept separate from realism** | natively below 0.98 in 6 of 8 joint states |
| novelty / copying | **no** (reported alongside, a guardrail) | held in both modes |

Meeting the honouring requirement is **never** evidence of realism. A controlled URL
can carry its query and still be a detectable query.

---

## 2. Evaluability: what cannot be evaluated, and how much

### 2.1 The rule [CONFIRMED]

The rule is `uie/evaluation.discriminate_with_floor` (MIN_N 250):
- per field, the gate keeps one real URL per host with that field non-empty;
- it sets n = min(1,200, pool / 3, fakes), since three disjoint real samples are
  drawn;
- the cell is insufficient when n < 250.

**So a cell needs at least 750 distinct real hosts carrying the field.**

### 2.2 Measured, on the validation split [CONFIRMED]

Counts are distinct real hosts with the field present, with the implied maximum n
in brackets. **✗** marks n < 250.

| class · state (share of the class's real URLs) | LBL | SUF | SUB | PTH | QRY |
|---|---|---|---|---|---|
| benign, free mode (113,277 URLs) | 113,277 | 113,277 | 47,161 | 82,032 | 7,253 (2,417) |
| benign P1 Q1 (4.81%) | 5,451 | 5,451 | 2,924 (974) | 5,451 | 5,451 |
| benign P1 Q0 (67.61%) | 76,581 | 76,581 | 33,561 | 76,581 | absent |
| benign P0 Q1 (1.59%) | 1,802 (600) | 1,802 (600) | **669 (223) ✗** | absent | 1,802 (600) |
| benign P0 Q0 (25.99%) | 29,443 | 29,443 | 10,007 | absent | absent |
| phishing, free mode (24,081 URLs) | 24,081 | 24,081 | 19,004 | 9,388 | 1,068 (**356**) |
| phishing P1 Q1 (2.78%) | **669 (223) ✗** | **669 ✗** | **466 (155) ✗** | **669 ✗** | **669 ✗** |
| phishing P1 Q0 (36.21%) | 8,719 | 8,719 | 6,832 | 8,719 | absent |
| phishing P0 Q1 (1.66%) | **399 (133) ✗** | **399 ✗** | **323 (107) ✗** | absent | **399 ✗** |
| phishing P0 Q0 (59.36%) | 14,294 | 14,294 | 11,383 | absent | absent |

- **Free mode:** no cell is unevaluable for lack of real URLs. Phishing QRY is
  measurable, but its n is capped at 356, against 1,200 for every other cell.
  CLEARED there therefore needs a lower excess, because the interval is wider.
- **Control mode:**
  - of the 32 state × field cells where the field exists, **10 cannot be
    evaluated**: all 5 of phishing P1 Q1, all 4 of phishing P0 Q1, and benign P0 Q1
    SUB;
  - a further 8 are absent by definition (the control removes the field).
- **Real URLs in the unevaluable strata:**
  - **1,068 phishing URLs, 4.44% of the class**, whose state-level realism cannot
    be tested on any field;
  - **669 benign URLs, 0.59% of the class**, whose SUB field only cannot be tested.
- **Generated URLs in those strata (E7 + DC1, free mode, 325,000 URLs):**
  - phishing: 1.95% (P1 Q1 1.21% + P0 Q1 0.74%);
  - benign: 0.36% (P0 Q1 with a subdomain).

  In free mode these URLs **are** scored, inside the pooled per-field cells. They
  are unevaluable only for state-level claims.

### 2.3 Estimated — rough order of magnitude, **not measured**

**If a generator matched the real joint-state mix** (E7 does not: it under-emits
both paths and queries, §3 K1):
- about **4–5% of its phishing output** would sit in states whose realism cannot
  be tested with this validation split;
- about **0.6% of its benign output** would have an untestable SUB field.

*Derivation:* the measured real shares above, applied to output. *Assumption:*
that the generator's state mix equals the real one. Under E7's current mix the
figures are the measured 1.95% and 0.36%.

**A controlled request for phishing P1 Q1 or P0 Q1:** 100% of its output is
untestable for realism. Honouring is still measurable.

**Real data needed to make them testable** (arithmetic on the measured counts,
assuming new real URLs fall into these strata at today's rates — itself
unverified):

| stratum | now | needed | roughly |
|---|---|---|---|
| phishing P1 Q1: LBL, SUF, PTH, QRY | 669 hosts | 750 | about 1.1× today's hosts |
| phishing P1 Q1: SUB | 466 hosts; 69.7% carry a subdomain | 750 | about 1.6× today's P1 Q1 hosts (≈ 1,080) |
| phishing P0 Q1: LBL, SUF, QRY | 399 | 750 | ≈ 1.9× |
| phishing P0 Q1: SUB | 323; 81% carry a subdomain | 750 | ≈ 2.3× |
| benign P0 Q1: SUB | 669; 37.1% carry a subdomain | 750 | ≈ 1.12× today's P0 Q1 hosts (≈ 2,020) |

**Separate, and generator-side** [CONFIRMED]: at n_main 8,000, E7's phishing free
query rate (1.74%, HC1 C4) yields about 139 query URLs per generation, which is
below 250. The phishing QRY cell then falls to the pooled estimator, which can
never clear. This limit comes from the generated side, not the real side. §3 K1
addresses it.

---

## 3. What prevents E3

### Known [CONFIRMED]

**K1. Emission deficit.** In free mode, E7 writes both queries and paths too
rarely:

| | benign real | benign E7 + DC1 | phishing real | phishing E7 + DC1 |
|---|---|---|---|---|
| queries | 6.40% | 3.88% | 4.44% | 1.95% |
| paths | 72.4% | 59.2% | 39.0% | 30.0% |

- **Achievable targets.** Under free mode's own suffix source (the shipped
  table), a model with the training data's per-suffix rates would write queries in
  6.62% / 4.68% of URLs and paths in 73.37% / 39.51%. Computed from `name_train`.
- **Consequence for E3.** Phishing QRY cannot be read per generation at the
  canonical size (§2.3).

**K2. Native honouring fails because the URL ends before the required field.**
- 1,162 of the 1,176 failures in C1's failing cells.
- Forbidden fields appear in 0.8–1.2% of URLs.

**K3. Root slash.**
- The tokenizer drops the empty path (E3 §7.1), so E7 writes `host/` 0% of the
  time, against 97.9% (benign) and 99.9% (phishing) of training bare hosts, and
  99.6% / 100% of `host/?q`.
- By CR1's arithmetic, this is the whole FULL reading in the path-off states.
- **Measured in free mode** [DIAGNOSTIC, §5.3]: rendering the slash alone takes
  FULL from +0.2168 to +0.0720 (benign) and from +0.4018 to +0.0520 (phishing).
- The normalisation was E4's fix for a masking asymmetry that E7's `field_mask`
  no longer has: with the frame named, every field trains on every row.

**K4. The 256-token length budget** deletes query rows tail-selectively. It is a
minority contributor: about 31% of the p99 value-length gap and 18% at p90 (Q6).

**K5. Query shape is compressed** (C3-C), and rare parameter names collapse (R2's
band profile). Q7 found the under-fit present on training data **for E5**; it has
not been re-measured on E7, which N2a-0 does.

**K6. SUF is solved** by state tables (SC1).

### Hypotheses [HYPOTHESIS]

**H1. The query content tell is under-fitting of rare and long query structure.**
Supported by K4 (partly), K5, Q6's "fails to reproduce a tail it saw", R2's weak
movement under more exposure, and DC1's flatter, later-stopping query calibration
cutting the benign tell by a quarter. *What limits the fit is U1.*

**H2. The emission and honouring failures come from how the end decision is
trained.**
- E7 decides presence implicitly, at the end of the URL, as the rare branch of
  continue-or-end.
- Control dropout (p = 0.5, independent) shows each control on half the rows and
  both controls on a quarter.
- *Caveat:* why phishing Q:1 is so much worse is recorded as not established (C1).

**H3. The controls shape fields they do not govern.** Supported by C3 (the
single-control `<Q:1>` lowered path presence among query URLs by 13.4 / 10.9
points) and by DC1's LBL blend weights. *Mechanism unknown* (CR1).

### Unknown

- **U1.** What limits learning the query tail: exposure or allocation, capacity,
  character-level tokenization, or sparsity.
- **U2.** Why phishing Q:1 honouring is weaker.
- **U3.** The name-tell mechanism in control states.
- **U4.** The phishing PTH residual. PTH attribution is closed.
- **U5.** About 80% of benign's query shape tell (C3-B, UNRESOLVED).

---

## 4. Lever map

| failure | lever axis | proposed change | status of the link |
|---|---|---|---|
| FULL root slash (K3) | representation | root-slash token (§5.2) | [CONFIRMED] mechanism; effect predicted by the [DIAGNOSTIC] in §5.2 |
| emission (K1), phishing QRY measurability | conditioning design | presence plan (§5.1) | [HYPOTHESIS] H2, tested by N1 |
| native honouring (K2) | conditioning design; objective in reserve | presence plan; unlikelihood (N1b) | [HYPOTHESIS] H2, tested by N1 |
| control-state name and path tells (U3) | conditioning design | presence plan | [HYPOTHESIS] H3, read by N1 |
| query content (H1, U1) | set by N2a: data (R1), objective (allocation), capacity or tokenization | N2b / N2c | U1 unknown |
| phishing PTH (U4) | none targeted | read at every step | branch closed |

---

## 5. N1's two changes

### 5.1 Presence plan (plan-first conditioning)

**Training.**
- The presence tokens `<P:v> <Q:v>` carry the **true** values, derived by the
  tokenizer's unchanged presence definition.
- They sit **right after the suffix and before the label**:
  `… <SUF> suffix <P:v> <Q:v> <LBL> label …`
- They are **training targets**.
- There is no dropout and no `?`.

**Generation.**
- **FULL FREE:** the suffix comes from the shipped table, as today. The model then
  samples the plan: each slot is a categorical over its two legal tokens, and the
  probability mass the model puts outside them is reported. The URL follows.
- **FULL CONTROL:** the suffix comes from SC1's state table, the plan is supplied,
  and the URL follows.

One conditioning mechanism, two modes, no resampling.

**Why after the suffix** (revision 2): free mode keeps its current suffix source,
so there is no confound, and the plan can depend on the suffix as it does in real
data. For example, `co.kr` carries a query 20% of the time (C3-C).

**What it is expected to do** [HYPOTHESIS, tested by N1]:
- presence becomes one explicit categorical per URL, whose maximum-likelihood fit
  is its empirical frequency given class, frame and suffix;
- every row teaches the URL under its true plan.

**Not claimed:** query content.

### 5.2 Root-slash token

**The change.**
- One new token marks "root slash, no path body", in `<PTH>`'s position. It covers
  `host/` and `host/?q`.
- `<PTH>` keeps meaning a **non-empty** path. E4's fix and every presence
  definition are unchanged. The gate's parser reads `host/` as no path, so no gated
  cell can change by construction; only retraining noise moves them.
- The vocabulary grows by one token **only** when the flag is on, so E7's and E5's
  tokenizers stay byte-identical.

**Independent evaluation**, three ways:
1. **[DIAGNOSTIC] the slash reference.** E7's raw free-mode files are rescored for
   FULL with the root slash rendered on every URL that has no path body, using the
   gate's own `field_report` and canonical seeds. It must first reproduce DC1's
   unrendered base FULL exactly. The rendered reading is the registered
   **prediction** of what a perfect slash does to FULL. *(Result: §5.3.)*
2. **RS against the same-recipe baseline:** `e8_rs` against `e8_base` (§6, N1) —
   the slash's native effect under E7's conditioning.
3. **RS on top of the plan:** `e8_pfrs` against `e8_pf` — its effect on the
   candidate.

### 5.3 Slash-reference result [DIAGNOSTIC]

Run with `research/native/slash_reference.py`; data in
`data/native/slash_reference_{benign,phishing}.json`. Inputs are E7's raw
free-mode files: DC1's base arm, G = 5, D = 10, canonical seeds.

| FULL (auxiliary) | as generated | root slash rendered | URLs changed |
|---|---|---|---|
| benign | +0.2168 | **+0.0720** | 81,612 of 200,000 |
| phishing | +0.4018 | **+0.0520** | 88,092 of 125,000 |

- **Validity.** Both unrendered readings equal DC1's recorded base FULL exactly, so
  the FULL-only replication is faithful.
- **What it shows.** The missing root slash accounts for about two-thirds of
  benign's whole-URL tell and seven-eighths of phishing's.
- **Correction.** CR1's single-feature arithmetic (+0.048 / +0.244) was a lower
  bound: a trained discriminator gains more from the slash than the slash's
  standalone AUC. Why it gains more is not established.
- **Use.** This is E8's registered prediction for the slash arms. It is not a
  mechanism, and nothing renders slashes in generation.

---

## 6. The program

### N0 — instruments and decisions

**[DECISION] Approved with the plan, 2026-09-26:**
- **D1. Control-mode E3** is CR1's instrument on native output (no resampling).
  Honouring is measured on the same URLs and reported separately.
- **D2. Honouring bar:** 0.98 per measurable joint state (C1's bar).
- **D3. Free-mode E3:** the canonical gate v2 decides (n_main 8,000, G = 5,
  D = 10, canonical generation seeds). The DC1-scale reading is reported alongside
  **for the final candidate only**. A canonical CLEARED with a high-n CONVICTED is
  reported as such.
- **D4. Phishing P1 Q1 and P0 Q1:** no realism claim (§2); honouring only.
- **D5. Naming:** E8 is the native experiment. Its checkpoints are `e8_base`,
  `e8_rs`, `e8_pf` and `e8_pfrs`.

### N1 — E8: a 2 × 2 ablation with a same-recipe baseline

Registered in full in `PREREG_E8.md`.

**Arms.** Four training runs, each on E7's recipe exactly: 20k steps, batch 256,
lr 3e-4, warmup 500, d_model 384, 8 layers, max_len 256, `bundle2`, **the same
seed** 20260803.

| arm | conditioning | root slash |
|---|---|---|
| `e8_base` | E7's (prefix controls, dropout 0.5) | no |
| `e8_rs` | E7's | yes |
| `e8_pf` | presence plan | no |
| `e8_pfrs` | presence plan | yes — the candidate |

**References, no training:**
- E7 itself, raw: HC1 C2's canonical free files are reused at the same seeds.
- E7 + DC1: the adopted free baseline, generated at the canonical size.

**Validity [V-base] — read first.** `e8_base` must reproduce E7 within noise:
- no free gated cell's verdict changes with |Δx| > 0.0141;
- honouring within ±1 point per joint state, on the same protocol;
- emission within ±0.5 points.

Otherwise the training pipeline changed, and no arm is read.

**Primary readings and their criteria**, per arm, against `e8_base`:

| code | reading | criterion |
|---|---|---|
| **K1a** | native honouring, per joint state (all 8) | ≥ 0.98 in every measurable state. **Separate from realism.** |
| **K1b** | free emission against the achievable targets | queries within ±10% (relative) of 6.62% / 4.68%; paths within ±3 points of 73.37% / 39.51% |
| **K1c** | root-slash rate (RS arms) | within 2 points of the training rate among bare hosts |
| **K1d** | realism regression, free (canonical) and control (CR1), every gated cell | verdict worsens **and** x rises by more than 0.0141 (E7 C2's rule) |
| **K1e** | novelty and query copying | worse by ≥ 0.5 points, CI excluding 0 (SC1 K3) |

- **K1a and K1b apply to the plan arms, K1c to the slash arms, K1d and K1e to all.**
- **Attribution.**
  - An effect is credited to the plan only through `e8_pf` − `e8_base`, and to the
    slash only through `e8_rs` − `e8_base` or `e8_pfrs` − `e8_pf`.
  - The plan × slash interaction is reported.
  - A difference smaller than the gap between `e8_base` and E7 is **not**
    attributed.
- **Candidate rule, fixed in advance:**
  - `e8_pfrs` is the candidate if it passes K1a–K1e.
  - If only K1a fails, N1b is licensed.
  - If the plan fails K1b, K1d or K1e, it is not taken forward. `e8_rs` can then
    stand alone if it passes K1c–K1e.
  - The owner decides adoption.
- **Not claimed by N1:** any QRY content cell.

**Wording.** N1 **tests** whether the plan and the token address the documented
emission, honouring and root-slash mechanisms. It "fixes" something only where its
criteria show it.

### N1b — conditional: unlikelihood on plan-forbidden tokens

**Registered only if K1a is the sole failure.**
- **Training-time only.** It reuses HC1's rule set as a training constraint, never
  as a generation-time mask.
- **It penalises three things:**
  - the end token while a planned field is missing;
  - a forbidden field's marker;
  - delimiters that would create or swallow a field.
- **Kill:**
  - K1d or K1e fires;
  - or honouring is still below 0.98. The native honouring route at this size is
    then exhausted, and the owner decides.

### N2a — query content diagnostic [DIAGNOSTIC]

**Never shipped; never generates data for use.**

**N2a-0 — E7 reference, read-only, runs while N1 trains.** Measure on E7, with the
existing instruments:
- **M1:** the termination hazard ratio inside query values, by position bucket, on
  train **and** validation query rows (Q7, teacher-forced);
- **M2:** the parameter-name band profile of generated queries against the
  faithful-sampler null (Q8/R2);
- **M3:** teacher-forced QRY NLL on train and validation rows, by name-frequency
  band;
- **M4:** the query copy rate.

It also computes each metric's **minimum detectable change** at the planned sample
sizes. A metric whose E7 gap is under twice that is void for the decision. This is
registered before any arm runs.

**Arms (Kaggle).** E7's recipe trained on **query-bearing rows only**:
- **A1:** exactly E7's number of query-row exposures. E7 made about two passes
  over its pool (5.12M exposures, Q8/R2); the exact count is fixed in N2a's
  registration from the sampler;
- **A2:** four times that.

**Sub-deficits,** read separately:
- **LENGTH:** M1 at value positions 25–100;
- **FREQUENCY:** M2's rare bands (1–4, 5–19).

**Closure** = 1 − gap(arm) / gap(E7). "Closes" means closure ≥ 0.5 **and** the
arm's 95% CI excludes E7's value. "Little" means closure < 0.25.

**Decision table (registered):**

| reading | condition | licenses |
|---|---|---|
| ALLOCATION | A1 closes both sub-deficits on validation | N2b: allocation inside the mixture |
| EXPOSURE | not ALLOCATION; A2 closes both on validation | N2b: exposure (query-content loss weight) |
| SPARSITY | A2 closes on train (M1 train; M3 train NLL) but little on validation, and copying rises ≥ 0.5 points | no training fix inside the frozen corpus; the owner decides |
| LENGTH-SPECIFIC | on train, A2 closes FREQUENCY but LENGTH is little | N2c-T (representation of long values); R1 as the known partial |
| FREQUENCY-SPECIFIC | on train, A2 closes LENGTH but FREQUENCY is little | N2c-C (capacity: **R3's registered unlock condition**) |
| GENERAL | on train, both little at A2 | N2c-C |
| MIXED | anything else | no content repair licensed; report |

- **Copy guard.** In any arm whose copy rate exceeds E7's by ≥ 0.5 points (CI
  excluding 0), validation closures are void, except as evidence for SPARSITY.
- **Limit, stated in advance.** N2a separates length-specific, frequency-specific,
  general and exposure. It cannot by itself prove tokenization or capacity; that is
  N2c's job, and only if licensed.
- **Not R2 again.** R2 asked whether more total optimisation moves the tail. N2a
  asks whether this architecture fits the tail when it gets the exposure.

### N2b — content repair inside the mixture

Registered after N2a, on N1's winner. The lever is the one N2a licenses:
- **a query-content loss weight;** and/or
- **R1**, a longer `max_len` sized by `r1_sizing.py`. Its registered condition, "a
  training run is already happening", is met.

Each lever is its own arm. **Guards** (the S3/S4 lesson):
- held-out, unweighted QRY NLL must **fall**;
- the band profile must not overshoot the null;
- copying must not rise;
- no other field regresses. A trade-off is itself evidence of capacity in the
  mixture.

**Kill:** no QRY cell moves by more than 0.0141, or any guard fails.

### N2c — tokenization (N2c-T) or capacity (N2c-C)

Only on N2a's reading. Registered then.

### N3 — the E3 reading

Of the adopted candidate:
- free mode on the canonical gate, plus the high-n reading;
- control mode with CR1's instrument on native output, per measurable state;
- honouring, **reported separately**;
- novelty, copying and FULL;
- the evaluability limits of §2.

**E3 is claimed only where every gated cell is CLEARED.**

**Cells with no targeted lever:**
- **Phishing PTH** (U4): read at every step. If it is the last cell standing, the
  owner decides whether a registration is justified.
- **Control-state LBL, SUB and PTH:** N1's plan is the only lever with evidence
  behind it. If they survive it, a registered diagnostic comes before any repair.

---

## 7. Sequencing and cost

**Order, as approved:** N1 → (N1b) → N2a → N2b → (N2c) → N3.

**Overlap, where it saves time:**
- N2a-0 (read-only) and N2a's registration and code are done while N1 trains and
  generates on Kaggle.
- N2a's two short runs go in the Kaggle session after N1's.
- N1b is decided on N1's reading.

**Cost of N1:**
- training, 4 arms × about 1.1 h on a T4 (PREREG_E7);
- generation, about 7–8 h. Free canonical is 80k URLs per arm; native control is
  6 measurable states × 5 × 3,000 plus the 2 unmeasurable states × 5 × 1,000 for
  honouring; plus E7 control and E7 + DC1 free;
- scoring, on the laptop.

So N1 is **two Kaggle sessions** (train, then generate).

---

## 8. Closed doors — kept closed

| item | status | the only thing that reopens it |
|---|---|---|
| E5 line | frozen; E7 is the base *(since 2026-09-28: `e9_ln` + raw)* | never (owner) |
| R2 (more steps) | REJECTED | not repeated; N2a asks a different question |
| R3 capacity | LOCKED | N2a's FREQUENCY-SPECIFIC or GENERAL reading |
| S3 / S4 / S4b weighting (PTH) | CLOSED | not reopened; QRY weighting only via N2a, with N2b's guards |
| PTH attribution | CLOSED | owner, after N2 |
| delta branch | CLOSED | not reopened; the frozen delta stays in the resting decode *(retired for the baseline 2026-09-28; earlier results keep it)* |
| resampling, rejection, masking, blend, second predictions | diagnostic or reference only | never as the production mechanism |
| routed / specialized generators | REJECTED | never; N2a's model never generates data |
| single-control states | out of scope | owner |
| gate v2, thresholds, instruments | unchanged | never, inside this program |

---

## 9. Changes in revision 2, and why

1. **Wording.** N1 *tests whether it addresses* documented mechanisms. "Removes" is
   used only where criteria show it.
2. **N1 is a 2 × 2 ablation with a same-recipe baseline (`e8_base`) and a validity
   check against E7.** An improvement can then be credited to the plan or the
   slash, rather than to a new checkpoint.
3. **Plan tokens moved after the suffix.** Free mode keeps its suffix source, and
   the plan conditions on the suffix, as real presence does.
4. **Emission targets corrected** to what free mode's suffix source makes
   achievable (6.62% / 4.68% queries; 73.37% / 39.51% paths), not the plain
   domain-frame rates.
5. **Path emission added** as a known deficit (59.2% against 72.4% benign; 30.0%
   against 39.0% phishing).
6. **The root slash is evaluated three ways,** with a diagnostic prediction first
   (§5.2).
7. **Honouring is separated from realism** in every criterion and reading.
8. **N2a's readings are quantitative:** two sub-deficits, train against validation,
   a copy guard, a power check, and a registered decision table with its limits
   stated.
9. **The evaluability rule is measured and the estimates are labelled** (§2). One
   generator-side limit is identified (phishing QRY at the canonical size).
10. **Every statement carries its evidence tag.**
11. **The slash reference was measured** (§5.3). It replaces revision 1's
    single-feature figures (+0.048 / +0.244) with the measured effect: FULL
    +0.2168 → +0.0720 and +0.4018 → +0.0520.
12. **Every number here comes from a committed script:** `evaluability.py`,
    `emission_targets.py` and `slash_reference.py`, in `research/native/`.
