# E7 + presence control — branch decision and repair plan

**Status: PLAN ONLY, for the owner's review.** Nothing here is registered or
implemented. No code, model, suffix table, gate, threshold, verdict rule or
published result has been changed. Written 2026-09-24, after CR1 (15c0cc9).

The labels used throughout are the ones `research/qry/REPAIR_PLAN.md` used:

    FACT        a measurement or a line of code, with its source
    INFERENCE   a conclusion drawn from facts, labelled as such
    HYPOTHESIS  consistent with the evidence, not established

---

## 0. The question, stated precisely

"Fixing E5" cannot mean editing E5: `best_e5.pt` is frozen. The **E5 line**
means new trained checkpoints on the E5 recipe, aimed at E5's open gate cells.
The **E7 + control line** means repairing the *controlled generation path* on
the frozen `best_e7.pt` first, and retraining only if a model-level cause is
established. Either way E5 stays the production and reference model until the
owner decides otherwise.

## 1. Decision: **E7 + presence control**

In one sentence: E7 + control is the only branch whose leading failure has an
established, non-model mechanism with a principled fix that needs no training.
That fix also tests the branch's main unexplained failure. The E5 line's
remaining failures are all model-level, and its registered repair plan is
exhausted.

| criterion | E5 line | E7 + presence control |
|---|---|---|
| **Actionable failure** | QRY content: a shape defect, distributed, "no single-property repair can remove this tell" (Q3.1, PQ1). Phishing PTH: the attribution branch is closed by the owner. | SUF: the suffix is drawn without the control (`uie/suffixes.py:151`, FACT). Bounded mathematically and measured (section 2). |
| **Understood enough to target** | No. `REPAIR_PLAN.md`: "no repair in this plan is unlocked for the QRY defect … Any further QRY repair requires a NEW pre-registration". | Yes for SUF in phishing. The same experiment discriminates between the two explanations of the unexplained LBL/SUB tells (section 3). |
| **Path without blind tuning** | The only unlocked-in-principle lever is capacity (R3), "the intervention most likely to appear to work for the wrong reason". More optimisation (R2) failed acceptance. | The fix is fixed by principle: the requested state's suffix distribution, in the frame the generator samples. There is nothing to tune. |
| **Preserves completed work** | Leaves E7, HC1, HC2 and CR1 unused. | Uses all of them, and every E5-era instrument too (gate v2, R4's decoder, the adopted delta, the frame, the QRY map). E5 stays the frozen reference. |
| **Cost and complexity** | A training run per hypothesis, with retraining noise of ±0.0141 on every field (w1 against E5, `PREREG_R2.md`) and full re-baselines. | No training. About 4 GPU-hours (Kaggle) and 8 CPU-hours of scoring. Research scripts only; `uie/` untouched. |
| **Risk of new artifacts** | Global: every field is retrained. | Local: only the suffix mix of *controlled* generation changes; the unconditional path stays bit-identical. |
| **Failure classes** | Model (QRY content, PTH content) and measurement capacity (phishing QRY: n_fake 180 < MIN_N 250; "could not currently be VERIFIED as clearing it", PQ). | Pipeline/control layer (SUF), measurement artifact (FULL), measurement limits (4 cells), model (QRY content, *shared with E5*), unknown (LBL/SUB, PTH). |
| **Gate and results kept frozen** | Yes. | Yes: scored by CR1's scorer, unchanged and S0-verified to be gate v2. |
| **Pre-registrable with clear kill** | Only weakly: the remaining hypotheses (capacity, exposure) are unquantified. | Yes: sharp predictions (a bound and measured TVs), a paired design, explicit kill criteria. |

### What this decision does *not* buy — stated so it is not oversold

- **It does not touch the defect blocking roadmap-E3's gate.** QRY content is a
  model-level shape defect in *both* branches (E5: benign UNDECIDED +0.0573,
  phishing CONVICTED pooled +0.1333; E7 unconditional: benign UNDECIDED
  +0.0541, phishing pooled UNDECIDED +0.0924; CR1: convicted in every state
  where it can be measured).
  The E7 branch offers the better *instrument* for it — `<Q:1>` supplies query
  URLs at 1.04–1.29 draws each — but no repair.
- **It does not promote E7.** E7's unconditional phishing PTH rose +0.0161
  over E5 (+0.0443 → +0.0604), above retraining noise with its verdict
  unchanged: HC1 C2's "cell to watch". That stays open.
- **The SUF outcome is partly by construction.** SUF in this project is
  table-driven by design (DESIGN sections 4.3 and 4.5): the suffix is supplied,
  not generated. A correct table *should* clear SUF. The informative outcomes of
  the first experiment are therefore the name fields and non-regression, not
  SUF alone (section 9).

### A correction to my own earlier reasoning

HC2's design review (`PREREG_HC2.md` section 2.2) expected joint resampling to
"tilt the mix by the model's own P(field | suffix): if E7 learned that
dependence … it reproduces the real shift." That premise was incomplete. The
tilt is P(field | suffix, **control**), and an effective control makes that
close to 1 for every suffix. Joint resampling therefore tilts only as much as
the control *fails*.

CR1 measured exactly this. Phishing `<P:1>` (97.9% first pass) moved the TLD mix
by 0.017, which is noise. Phishing `<Q:1>` (77.9%) moved it by 0.119, about
half the real 0.253. HC2's registered results (R1–R3) stand. The same account
explains why R3 found joint better only where the control fails most.

---

## 2. Current failure map (E7 + control)

**What is broken** — CR1, 59 applicable cells: 23 CLEARED, 9 UNDECIDED, 23
CONVICTED, 4 not computable. No state is cleared on every field.

| failure | where (CR1 x) | class | status |
|---|---|---|---|
| SUF tell | convicted in phishing P:1 (+0.0695), P:1 Q:0 (+0.0754), Q:1 (+0.1084); benign P:0 (+0.0641), P:0 Q:0 (+0.0660), P:1 Q:1 (+0.0735). Undecided in benign Q:1, P:0 Q:1, phishing P:0 Q:0 | pipeline / control layer | **mechanism established for phishing**; by code only for benign |
| LBL and SUB tells | phishing P:1 (+0.061 / +0.073), P:1 Q:0 (+0.069 / +0.080), Q:1 (+0.159 / +0.154, n 356 / 263); SUB also in benign Q:1 and P:1 Q:1 | unknown: pipeline or model | **unexplained** |
| QRY tell | every state where it can be measured (+0.098 to +0.156; pooled +0.105) | model, shared with E5 | established as a shape defect (Q3, Q3.1, PQ1) |
| PTH tell | benign Q:1 (+0.110), P:1 Q:1 (+0.117); phishing P:1 (+0.068), P:1 Q:0 (+0.073); UNDECIDED at +0.043 to +0.045 in four states | unknown; the PTH attribution branch is closed | descriptive only |
| FULL ≈ 0.49 in path-off states | five states | measurement artifact (auxiliary) | explained: the root slash (CR1 section 4) |
| not computable / not measurable | 4 cells / 2 states | measurement limit (real pools) | established |

**Known (FACT)**
- The suffix is drawn from the class-marginal table by label alone, before the
  URL (`uie/suffixes.py:151`).
- Rejection moves the kept mix by at most (1 − a)/a in TV: 0.021 for phishing
  `<P:1>` (a = 0.979).
- On top-10 TLDs, flipping the phishing path control moves the generated mix by
  0.017 (noise 0.018), against 0.154 in real data. Generated `<P:1>` phishing
  URLs sit 0.028 from the class mix and 0.103 from real path-bearing URLs.
- **New, training data only (this plan):** the shipped table counts *every* row
  of both training corpora. The page-frame share of those rows varies by state:
  2.2–49.3% for phishing and 7.5–76.8% for benign (overall 19.4% and 64.5%). For
  phishing P:1 and P:1 Q:0, a state-filtered table built the shipped way would
  sit 0.072 and 0.076 (top-10 TV; 0.116 and 0.118 over all suffixes) from the
  domain-frame table. That is the same order as the real state shift being
  fixed (0.094). See the appendix.
- Presence enforcement is exact (HC2 R1, R2). Masking was ruled out (HC1 H1).

**Suspected (HYPOTHESIS)**: section 3.

**Ruled out**

| candidate | evidence |
|---|---|
| presence not honoured | HC2 R1: 1,200/1,200 in all 8 cells; CR1 VERIFY: 210,000/210,000 |
| runaway or forced URLs | HC2: long-URL share at the model's own rate (≤ 0.17%); masking's structural shift gone |
| GPU/CPU sampling mismatch | byte identity: C2 (403 + 833 URLs), CR1 (4 states, 1,192 URLs) |
| a scorer that is not the gate | S0: 73 values reproduced exactly |
| FULL as a realism signal in path-off states | saturated by one character (CR1 section 4) |
| "joint resampling reproduces the real TLD shift" | false for effective controls: bounded by (1 − a)/a (section 1) |
| E7 unconditional regression against E5 | HC1 C2: none attributable at the registered bar (phishing PTH +0.0161 to watch) |

---

## 3. Root-cause hypotheses, ranked by evidential support

The ranking orders evidence; it does not certify any of them.

**H1 — control-blind suffix draw (SUF). Strongest.**
- *For:* code; the (1 − a)/a bound; the noise-controlled measurement. The SUF
  verdicts co-vary with how far the real state's TLD mix departs from the class
  mix: phishing Q:1 0.242 and P:1 0.094 convict; phishing Q:0 0.011 clears.
- *Against / open:* benign's top-10 shifts are small (path 0.025, query 0.050),
  yet benign P:0 SUF convicts at +0.064. The benign cause may lie in the tail
  the top-10 cannot see, or in H2. Magnitude for benign: **Unknown.**

**H2 — frame-mixed table.**
- *For:* training-data composition (section 2). The page-frame share, and with
  it the table, changes with the state.
- *Against:* the unconditional SUF clears with this same mixed table (benign
  +0.0165, phishing +0.0054), so at class level the offset (top-10 TV 0.033) is
  tolerated. Whether it matters per state is **Unknown**. Section 4's arm C
  isolates it from H1.

**H3 — names are downstream of the suffix (H_DOWN).**
- *For:* architecture — `[SUF]` precedes `[LBL]` "so the label is invented
  knowing the TLD" (`uie/suffixes.py`). Pattern — phishing's name fields clear in
  the mixture Q:0 but convict in its minority part P:1 Q:0, and LBL/SUB/SUF
  convict together, which is what a marginal-shaped suffix mix would produce.
- *Against:* nothing measured. In phishing Q:1, LBL (+0.159) exceeds SUF
  (+0.108); that neither supports nor excludes it.

**H4 — the control does not reach the names (model level).**
- *For:* the same pattern is equally consistent with a model whose names ignore
  the control. C1 showed only that presence is learned (96–99%), nothing about
  names.
- *Against:* nothing measured. **H3 and H4 make opposite predictions under the
  first experiment**, which is why it is chosen.

**H5 — QRY content shape defect (model, shared with E5). Established.**
Q3, Q3.1, PQ1: a shape defect, distributed; parameter-name identity carries
16–28%. The registered repairs are exhausted. Not addressed by the first
experiment.

**H6 — PTH content.** Unknown in the controlled states. E3_REPORT section 13
records model-level path defects (uppercase, phishing length). The attribution
branch stays closed, so PTH is reported descriptively only.

**H7 — train/held-out TLD drift within a state.** Unmeasured. It would cap any
table-based fix. Q5 found train ≈ val for query metrics. The benign split is
random, the phishing split host-grouped. Measured in the design review, before
registration (section 8, step 1).

**H8 — real-pool limits. Established.** 2 states not measurable, 4 cells not
computable, phishing Q:1 at n = 356. Generation cannot fix them.

---

## 4. First repair experiment: **SC1 — the state-conditioned suffix draw**

### 4.1 The intervention — one change, implemented correctly

In controlled generation, draw each URL's suffix from the **domain-frame
training distribution of the requested state**, p_name_train(s | class, state).
Keep that suffix while presence is enforced (**held-suffix resampling**, HC2's
validated mode). The generator then samples exactly

    p_train(s | class, state) · p_E7(url | s, control, state)

— the requested state's TLD mix, and the model's own URL given that TLD.

**Why these two details are not extra fixes.**
- **The name corpus:** it is the frame the generator samples (`GENERATOR_FRAME
  = "domain"`) and the population SUF is scored against — in both frames,
  `FIELD_POPULATION_BY_FRAME`. A state filter on the shipped all-rows table
  would change the frame mix state by state (section 2): not a correct
  implementation of the hypothesis.
- **Held mode:** a state-conditioned table already carries p(s | state).
  Joint resampling would tilt it again by P(state | s, control) and
  double-count. Held mode is exact, and HC2 R2 proved its implementation equal
  to HC1's exact arm.

### 4.2 Arms

| arm | suffix table | resampling | role |
|---|---|---|---|
| **CR1** (existing, read-only) | shipped, all training rows, class marginal | joint | historical baseline; the regression reference |
| **C** (new) | name_train, **class marginal** | held | matched control |
| **T** (new) | name_train, **the requested state** | held | intervention |

**T − C isolates state conditioning** (H1) and discriminates H3 from H4. C − CR1
is descriptive: the frame source and the mode together (H2). T − CR1 is the
safety comparison against the pipeline in use.

Arm C costs about 2 GPU-hours and 3.6 CPU-hours. It is what makes an
improvement attributable. Without it, the frame-source correction (0.072 in
phishing P:1) and the state conditioning (0.094) cannot be told apart.

### 4.3 Files and components

**No change to `uie/`.** The existing API already does held-mode resampling from
any `SuffixTable`: `generate_with_presence(..., table=<state table>,
hold_suffix=True)`. New, all in `research/sc1/`:

- `PREREG_SC1.md`.
- `build_tables.py`: per-state and class-marginal tables from
  `name_train` / `phishing_name_train` only, written to
  `data/sc1/tables.json` with sha256. Unit test: with no state filter and all
  `*_train` files it must reproduce the shipped `suffix_table.json` byte for
  byte, proving it is the same construction.
- `SC1_generate.ipynb` and `make_code_bundle.py`, adapted from CR1's.
- `sc1_verify.py`: files, hashes, seeds, the table hash, every URL in its state,
  held-mode byte identity against a CPU regeneration, and the manipulation check.
- `sc1_compare.py`: paired Δ, Holm, harm flags, novelty, cost. Unit-tested on
  synthetic draws.
- Scoring: `research/cr1/cr1_score.py`, **unchanged**, run on each arm's
  generation directory.

### 4.4 Frozen

- `best_e7.pt` (sha256 `916d0385…`) and `best_e5.pt`.
- R4's decoder (`temperature` 1.0), the adopted delta (0.22 / 0.30), frame
  `domain`.
- `data/bundle2`, including the shipped `suffix_table.json`.
- All of `uie/`: model, presence, suffixes, evaluation, run_eval.
- Gate v2: bar 0.05, MIN_N 250, G = 5, D = 10, t_crit, the section-26.1 and
  pooled rules.
- `cr1_score.py`.
- Every E5, R4, gate-v2, E7, HC1, HC2 and CR1 result and data file.
- The unconditional path and its table: E5's and E7's gate readings carry over
  untouched.

### 4.5 Configuration

- Checkpoint `best_e7.pt`.
- Tables from `name_train` / `phishing_name_train` only; held-out files are
  never read to build a table.
- The 14 measurable states of CR1, in CR1's order.
- G = 5 generations × 3,000 compliant URLs per state per arm; max_rounds 100.
- Scoring: CR1's instrument — SEED 20260806, split seeds v2, D = 10, n = min(3,000,
  1,200) reduced by the pools.

### 4.6 Considered and not chosen

- **Re-weighting CR1's URLs by suffix instead of generating.** Given its suffix,
  a CR1 URL is an exact draw of the target, so suffix-subsampling reproduces T
  for free. But target suffixes that CR1 rarely drew cannot be filled, and
  truncating the target's tail creates its own SUF tell. Not clean.
- **A state filter on the shipped table.** It mixes frames by state, as above.
- **Joint mode with a state table.** It double-counts, as above.
- **Free-mode rejection first** (E7 with `?` controls, keep URLs in the state).
  This tests the model's own conditional; it does not repair anything. It is
  also costly: phishing Q:1 needs about 57 draws per URL at the 1.74% query
  rate. It is kept as the follow-up that separates H3 from H4 (section 7).
- **Smoothing sparse-state tables.** A free parameter that could only be chosen
  by looking at results. Raw counts, like the shipped table; coverage reported
  instead.

---

## 5. Pre-registration design (draft)

**Hypotheses**
- **H_SUF (primary).** Controlled SUF tells arise because the suffix is drawn
  without the requested state. Drawing it from the state's domain-frame
  training distribution (T) removes them, relative to the same draw without the
  state (C).
- **H_DOWN vs H_INDEP (key secondary).** Phishing's LBL/SUB tells in the
  path/query states follow the suffix mix (H_DOWN), or they do not (H_INDEP).
- **Safety.** T introduces no new tell against the pipeline in use (CR1).

**Seeds** (state s in CR1's order, generation g; blocks of 10⁹ as in CR1). CR1
occupies offsets 0 to about 1.0 × 10⁸ within each block (100 rounds × 1,000,003
plus the suffix groups).

| use | seed |
|---|---|
| arm C | 20261001 + (100·s + g)·10⁹ + 3·10⁸ |
| arm T | 20261001 + (100·s + g)·10⁹ + 6·10⁸ |
| T replication (harm flags only) | 20261001 + (100·s + g)·10⁹ + 8·10⁸ |
| split seeds | CR1's (SEED 20260806, v2) |

Because the split seeds are CR1's, the real samples and floors of draw (g, d)
are identical across arms wherever n matches.

**Sample sizes.** Per arm and state: 15,000 compliant URLs (5 × 3,000); n per draw
1,200 or the pool's limit, exactly as CR1. Power, from CR1's SUF SEs (0.0020 to
0.0058): the expected Δ (−0.05 to −0.10) gives |t| ≥ 6, even if pairing removed
no noise at all (SE × √2). A Δ of −0.01 would be marginal, and not a useful fix
anyway.

**Statistic.** Per (state, field), paired by generation:

    δ_g = mean_d excess_T(g, d) − mean_d excess_C(g, d)
    Δ = mean δ_g,   SE = sd(δ_g)/√5,   t = Δ/SE,   df = 4,   95% CI = Δ ± 2.776·SE

This is section 26.1's one-sample form applied to paired differences, the form
HC2 R3 used. **Matched n:** SUF and LBL are always matched, because every URL
carries them. PTH and QRY are matched in the states that force them. SUB can
differ, since benign SUB is fake-limited at about 38% presence. A cell is tested
only if every draw's n agrees across arms within 5%; otherwise it is reported
descriptively (NS found, descriptively, that tells grow with n).

**Multiplicity**, each a Holm–Bonferroni family at α = 0.05:
- **primary:** SUF in phishing P:1, P:1 Q:0, Q:1 (3 tests), where the mechanism
  is measured;
- **secondary:** SUF in benign P:0, P:0 Q:0, P:1 Q:1 (3), where the mechanism is
  shown by code only;
- **H_DOWN:** LBL and SUB in the three phishing states (6).

Safety is per cell with replication, deliberately uncorrected: a false alarm
costs one replication, a missed harm costs a bad adoption.

**Success.**
- H_SUF is **supported in a state** if the Holm-adjusted Δ < 0 **and** T's SUF
  verdict is CLEARED.
- **Supported** overall: all 3 primary states. **Partial**: 1–2. **Refuted**:
  none, with the manipulation check passing.
- Registered expectation, not a criterion: where the design-review premise
  holds, T's SUF x lies within 0.02 of E7's unconditional SUF (+0.0165 benign,
  +0.0054 phishing).

**H_DOWN reading.**
- Per cell: H_DOWN if the Holm-adjusted Δ < 0. H_INDEP if the 95% CI of Δ lies
  inside ±0.02 (HC1 H3's equivalence form). Otherwise inconclusive.
- Overall: **H_DOWN** if ≥ 4 of 6 cells support it and none supports H_INDEP.
  **H_INDEP** if ≥ 4 of 6 support H_INDEP. Otherwise **mixed**, stated as such.

**Kill criteria.**

| code | trigger | consequence |
|---|---|---|
| **K1** — hypothesis | H_SUF refuted in all three primary states while T's TLD mix matches its table | the suffix line closes |
| **K2** — harm | any (state, field) with Δ(T − CR1) ≥ +0.02 and 95% CI excluding 0; a CR1-CLEARED verdict turning CONVICTED; a pooled row turning TELL | replicate that state once (T replication seeds); not adopted if the replication flags the same cell by the same rule |
| **K3** — memorisation | T's novelty rate below CR1's in the same state by ≥ 0.5 pp, with the 95% CI of the difference excluding 0 (E7 baseline 99.75% / 98.75%) | not adopted |
| **K4** — runaways | T's share of URLs at 240+ characters above CR1's by ≥ 0.2 pp, CI excluding 0 | not adopted |
| **K5** — cost | any request exhausting 100 rounds (`PresenceUnmet`) | reported; not adopted for that state; max_rounds is not raised after the fact |

**Validity (technical — a failure is fixed and amended with a date, never read
as a result).**
- VERIFY passes: files, hashes, seeds, table sha256, 100% in-state, held-mode
  byte identity on 4 states per arm.
- **S0′:** `cr1_score.py` re-scoring 2 CR1 states reproduces `cr1_report.json`
  exactly.
- **Manipulation check:** each arm's parsed-suffix mix (top-10 + other) is
  within the noise maximum of its own table. If not, the implementation is
  wrong.

**Adoption gate (the pipeline change only).**
- H_SUF supported;
- no K2 flag surviving replication;
- K3 and K4 not triggered, K5 clear;
- validity passed.

Then the owner decides. Adoption would mean controlled generation uses
state-conditioned domain-frame tables with held resampling. The builder then
moves into `uie/` under its own tested commit. E5 stays production, and E7's
promotion stays a separate decision.

---

## 6. Evaluation

**Gate metrics.** Per arm × state × field (LBL, SUF, SUB, PTH, QRY; FULL
auxiliary): x, SE, t, 95% CI, n, generation means, the section-26.1 verdict and
the pooled rows. This is CR1's full table, from CR1's scorer, so every number is
comparable to CR1.

**Field and state metrics.**
- Paired Δ tables: T − C for the hypotheses; T − CR1 for safety; C − CR1
  descriptive.
- TLD mix per state and arm against real (CR1's `cr1_suf_mix.py` method).
- Per-state table coverage: distinct suffixes, singleton share, Good–Turing
  unseen mass, against the real pool's distinct count.

**Novelty and memorisation.**
- The gate's novelty rate: bloom filter of training URLs, per arm and state.
- Memorised count and distinct-URL rate.
- Registrable-domain copy rate: share of generated label + suffix pairs seen in
  training. This is the sharpest check on a table that concentrates mass.

**Termination and cost.**
- Presence: 100% by construction; verified.
- First-pass rate per state; draws per kept URL (harmonic under held mode, so the
  first pass is the marginal); rounds; `PresenceUnmet`.
- 240+ character share; mean length.

**Regression against history.**
- The unconditional path is unchanged: no `uie/` edit, the full suite, and E7's
  96-configuration generate fixture. So E5's and E7's gate v2 readings carry
  over without re-running.
- CR1 is re-scored for S0′ only.
- FULL is reported, but no claim is read from it in path-off states.

---

## 7. Decision tree after SC1

| outcome | reading | next |
|---|---|---|
| **H_SUF supported, H_DOWN** | the controlled name tells were the suffix mix | adopt the pipeline change (owner); T becomes the controlled baseline; go to **C3-QRY** |
| **H_SUF supported, H_INDEP** | SUF fixed; names do not follow the suffix | adopt the SUF fix; run **FR1** (free-mode rejection, phishing P:1 and P:1 Q:0) to separate H4's control-layer and model-level forms |
| **H_SUF supported, mixed** | partial | adopt the SUF fix if the gate passes; FR1 on the cells that stayed |
| **Partial** | a reduction, but not CLEARED somewhere | read-only diagnosis: manipulation, premise (H7), tail coverage. At most one registered follow-up; nothing tuned |
| **Refuted (K1)** | the SUF tell is not the TLD mix | close the suffix line; read-only re-characterisation of what SUF detects (e.g. parsed against drawn suffix) |
| **Harm (K2–K4)** | the fix creates a tell | not adopted; C − CR1 shows whether the frame source is responsible |

**When to abandon the branch.**
- (i) K1 fires and the re-characterisation finds no control-layer cause.
- (ii) FR1 shows that the free model's *own* conditional names also convict.
- (iii) Held resampling cannot enforce the product-relevant states within
  100 rounds.

In (i) or (ii), every remaining controlled failure is model-level. The branch
then has no pipeline lever left, and the next step in either branch is a
training run. At that point **E5-line vs E7-line is reopened as a new decision
on training evidence**, not continued here.

---

## 8. Implementation sequence

| step | work | output / commit |
|---|---|---|
| 0 | owner review of this plan; the decisions in the section below | — |
| 1 | design review, read-only (about 1 h): (a) frame composition per state — done, appendix; (b) table coverage per state; (c) **premise H7**: train against held-out TLD mix per state, top-10 + other with noise floors, generated data untouched. A state failing (c) has its prediction withdrawn *before* registration | numbers go into the pre-registration |
| 2 | `PREREG_SC1.md` | commit, **before any SC1 code** |
| 3 | code in `research/sc1/` and its tests; suite green; `uie/` untouched | commit |
| 4 | CPU simulation of the notebook (the lesson from E7 and C2) | — |
| 5 | Kaggle: arms C and T, 140 files, about 4 h GPU (one session, or one per arm) | provenance |
| 6 | VERIFY; S0′ | must PASS before any scoring |
| 7 | score C, then T, with `cr1_score.py` (about 7.5 h, below-normal priority, memory guard) | `data/sc1/{C,T}/cr1_report.json` |
| 8 | `sc1_compare.py`; diagnostics | — |
| 9 | replication, only if a K2 flag fired | — |
| 10 | `SC1_RESULTS.md`; pre-registration status; push; owner decision | commit |

**Owner decisions needed before step 1:**
1. the branch;
2. SC1 with its matched control arm C;
3. the domain-frame (`name_train`) table source.

---

## 9. Risks

**Scientific.**
- SUF can clear for the design reason and still say nothing about the model;
  the name fields and non-regression carry the science.
- Sparse states (phishing Q:1: 8,272 training rows, 1,068 real hosts, n 356)
  give weak evidence either way.
- Train/held-out drift (H7) would cap the fix; it is measured first.
- H_DOWN could look supported because `.com`-heavy labels are simply easier.
  The novelty and copy-rate checks guard against that reading.

**Engineering.**
- Held-mode cost: phishing `<Q:1>` needed up to 24 rounds under the marginal
  table in HC2.
- The environment lessons: tldextract before import, `model.to(device)`.
- Seed blocks are registered as disjoint.
- The table file must be byte-identical on Kaggle and the laptop (sha256 in the
  provenance).

**Evaluation.**
- Unmatched n breaks pairing, hence the 5% rule.
- Scorer drift, hence S0′.
- Pooled rows are convict-only and compared descriptively.
- About 60 safety cells: false alarms are absorbed by replication, not by
  loosening the rule.

**Ways SC1 could hide the problem rather than solve it.**
1. Building a table from held-out data would be leakage: training files only,
   hashed, and tested.
2. Choosing any option after seeing results: every option is fixed now, by
   principle.
3. Lost diversity: a sparse table drops unseen TLDs, which a discriminator at
   n = 1,200 may not punish. Coverage is reported against the real pool.
4. Memorisation rising with concentrated tables: K3 and the copy rate.
5. Reading "not computable" as safe: it is not, and never will be.
6. Reading FULL in path-off states: it stays saturated.
7. Calling the controlled path "safe" after SC1: QRY and PTH still convict.

---

## 10. Roadmap after SC1 — each step only if its trigger holds

1. **SC1**, now.
2. **FR1**, only if the name tells survive SC1. Free-mode rejection in phishing
   P:1 and P:1 Q:0 (about 3.4 draws per URL; Q:1 is excluded at about 57). The
   free model's names clear → a control-layer problem. They convict → a
   model-level problem.
3. **C3-QRY**, re-registered from `PREREG_E7.md` section 4, where it never ran
   because C1 failed. Is `<Q:1>` query content the same as free-mode query
   content, with SC1's table removing the suffix confound C3 itself named? If
   consistent, `<Q:1>` measures QRY at the real-pool cap in every generation.
   That removes PQ's "could not be verified" problem for any later repair.
4. **A QRY-content repair**: a training run under a new E-number, **only with
   new evidence**. The registered options are exhausted, and nothing here
   pre-commits one. The E7 branch's contribution is that one can be *verified*.
5. **Owner decisions:** promote SC1's builder into `uie/`; later, E7 against E5
   as production, which first needs HC1 C2's phishing PTH watch item resolved.

**Not proposed, and why.**
- Masking: HC1 H1 failed.
- PTH attribution: the branch is closed.
- Any bar, MIN_N, D or verdict change.
- Loss weighting: H6 refuted.
- Corpus work: Q5.
- Capacity: R3 locked.
- Tuned smoothing.
- Any change to the unconditional path or the shipped table. If C − CR1 shows
  the frame mixing matters, it is noted for the unconditional path as a
  separate question and not proposed.

---

## Appendix — training-data composition by state (read-only; no held-out data)

Top-10 TLDs by the class's `name_train`; TV over top-10 + other unless stated.
† not measurable in CR1 (too few held-out URLs); shown for the table's composition only.

| class | state | name rows | page-frame rows | page-frame share | TV name vs page | TV all rows vs name | same, all suffixes |
|---|---|---:|---:|---:|---:|---:|---:|
| phishing | all | 192,652 | 46,491 | 19.4% | 0.170 | 0.033 | 0.053 |
| phishing | P:1 | 75,027 | 43,376 | 36.6% | 0.196 | **0.072** | **0.116** |
| phishing | P:0 | 117,625 | 3,115 | 2.6% | 0.193 | 0.005 | 0.007 |
| phishing | Q:1 | 8,272 | 5,710 | 40.8% | 0.128 | 0.052 | 0.120 |
| phishing | Q:0 | 184,380 | 40,781 | 18.1% | 0.188 | 0.034 | 0.052 |
| phishing | P:1 Q:1 † | 5,286 | 5,131 | 49.3% | 0.120 | 0.059 | 0.143 |
| phishing | P:1 Q:0 | 69,741 | 38,245 | 35.4% | 0.215 | **0.076** | **0.118** |
| phishing | P:0 Q:1 † | 2,986 | 579 | 16.2% | 0.133 | 0.022 | 0.047 |
| phishing | P:0 Q:0 | 114,639 | 2,536 | 2.2% | 0.180 | 0.004 | 0.006 |
| benign | all | 906,214 | 1,648,460 | 64.5% | 0.051 | 0.033 | 0.071 |
| benign | P:1 | 655,216 | 1,601,453 | 71.0% | 0.050 | 0.036 | 0.066 |
| benign | P:0 | 250,998 | 47,007 | 15.8% | 0.066 | 0.010 | 0.031 |
| benign | Q:1 | 57,768 | 170,055 | 74.6% | 0.021 | 0.015 | 0.083 |
| benign | Q:0 | 848,446 | 1,478,405 | 63.5% | 0.054 | 0.034 | 0.073 |
| benign | P:1 Q:1 | 42,862 | 142,187 | 76.8% | 0.023 | 0.018 | 0.093 |
| benign | P:1 Q:0 | 612,354 | 1,459,266 | 70.4% | 0.053 | 0.037 | 0.068 |
| benign | P:0 Q:1 | 14,906 | 27,868 | 65.2% | 0.035 | 0.023 | 0.123 |
| benign | P:0 Q:0 | 236,092 | 19,139 | 7.5% | 0.051 | 0.004 | 0.013 |

## Evidence index

`research/qry/REPAIR_PLAN.md` (exhausted), `PHISHING_QRY_RESULTS.md` (PQ1–3),
`FLOOR_RESULTS.md` (FL1–2), `NSWEEP_RESULTS.md` (NS, VOID), `R4_RESULTS.md`,
`GATE_V2_RESULTS.md`, `PREREG_R2.md` (retraining noise); `ARCHITECTURE_DECISION.md` (sections 1, 9 and 11);
`E3_REPORT.md` (sections 13, 27.2 and 30); `research/e7/` (PREREG_E7, C1);
`research/hc1/HC1_RESULTS.md` (C2, C4); `research/hc2/` (PREREG section 2.2,
R1–R3); `research/cr1/CR1_RESULTS.md` (sections 3 and 4); `uie/suffixes.py`,
`uie/presence.py`, `uie/evaluation.py`.
