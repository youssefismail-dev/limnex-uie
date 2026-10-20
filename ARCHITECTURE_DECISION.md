# Architecture decision: routed / specialized generators

**Status: RECOMMENDATION, awaiting the owner's approval.** Nothing in this document
is implemented. No code, gate, threshold, roadmap, checkpoint or published result
has been changed by it.

**Date:** 2026-09-23. **Reviewed against:** `DESIGN.md` (sections 3, 4.4, 4.5, 7,
9-D1, 10), `E3_REPORT.md`, the README's acceptance status, and `research/qry/`
(Q1-Q8, R2, R4, PQ1-PQ3, NS, FL).

---

## 1. Executive summary

**Decision: REJECT PROPOSAL.**

The proposal has two defining elements: a **learned router** that interprets what
the user wants, and **separate specialized generators** per class. The evidence
supports neither.

- **The router is unnecessary, and a learned one is harmful.** The request space is
  small and typed: class, frame, suffix, language, count, seed, plus any structural
  flags. The existing call `generate(label=..., frame=..., suffix=...)` already
  routes deterministically, with zero routing error. A learned router would add an
  error source whose characteristic failure — assigning the wrong class — is
  **label noise in a dataset whose labels are its ground truth**, and it would break
  the seeded, reproducible recipe the project is built on.
- **The defects blocking the gate are not class interference.** PQ1 found that
  phishing query content fails for the *same* reason benign does: shape, not
  vocabulary, with the same dominant component at nearly the same share. Splitting
  would move the same defect into two models, and would cut the phishing model to
  its own 238,214 rows — **1/11 of the data the shared trunk trains on** — in the
  class that already memorises under over-exposure (E2.2: +1.227 train/val gap).

The **goal** behind the proposal — honouring requested structure such as "PTH
required" or "no QRY" — is legitimate, and DESIGN section 4.5 already allows it.
But **routing cannot deliver it**: a router chooses *which* model runs; it cannot
make that model produce a query. Structural control has to live inside the
generator in either architecture, and the current one can gain it the same way E5
gained frames — as a conditioning token. That is a separate proposal (section 9),
not adopted here.

**On LIMENX:** the defensible fifth member is the generator's **likelihood**, not its
**samples** — a class-conditional log-likelihood ratio from the single shared model,
which DESIGN section 3 already anticipates. Training LIMENX on generated URLs is
scientifically weak today, because generated URLs are still highly distinguishable
from real ones as whole URLs. The split proposal would specifically weaken the
likelihood-ratio path.

**Next step:** the pre-registered repair of the evaluation gate (Option C). Every
future architectural comparison — including any version of this one — will be
judged by that gate, and FL showed its standard error can currently be **four times
too small**.

---

## 2. Current architecture

**One class-conditional autoregressive transformer**, character level, 14.33M
parameters (d_model 384, 8 layers, 8 heads), checkpoint `best_e5.pt`.

- **Input:** control tokens — class `<C:*>`, language, frame `<F:*>` — followed by
  the factorized URL with field markers (`SCH USR SUF LBL SUB PRT PTH QRY FRG`) and
  EOS.
- **Training data:** four corpora — benign and phishing, each in a domain frame
  (one URL per name) and a page frame (many URLs per site) — with class sampling at
  `class_alpha = 1.0`.
- **Controls actually built:** class, language, frame, a suffix **pinned** from a
  per-class table (never sampled by the model), decode-time field biases (e.g. the
  adopted delta on PTH EOS), and the decoder (`temperature`, adopted by R4).
- **Designed but not built:** `length` and `path_depth` (DESIGN section 4.5).
- **Interfaces:** the generation API (primary) and the typicality scorer
  `log P(url | class)` (secondary; deliverable D1, blocked until roadmap-E3's gate
  passes).
- **Evaluation:** the canonical gate — per field, a discriminator's AUC in excess
  of a real-vs-real floor, bar 0.05, G=5 generations x D=3 split seeds.

**What "routing" means today:** the caller supplies the class as a token, and the
same weights serve both classes, steered by it. DESIGN section 4.4 chose this
deliberately:

> *Single*, not one model per class: a shared trunk means benign data improves
> phishing generation and vice versa, and the class becomes a control rather than a
> separate system.

**Current acceptance state (adopted decoder):** convicted — phishing PTH, phishing
QRY. Benign QRY — **UNDECIDED, not cleared** (+0.0619 at D=30). All other cells
clear. The gate is not passed.

---

## 3. Proposed architecture — as the owner described it, unmodified

- A **main AI / router** whose job is to understand what the user wants to
  generate.
- It routes phishing requests to a **phishing URL generator** and benign requests
  to a **benign URL generator**.
- It passes the requested **structural conditions** — "PTH required", "QRY
  required", "PTH and QRY disabled", and whatever other controls the generators
  expose.
- Generative responsibility is therefore split across **specialized models**
  instead of one undifferentiated model.

"Router" is evaluated under both readings: **(a)** a learned model that interprets
free-form requests, and **(b)** a deterministic dispatcher over structured
parameters. The wording "understands what the user wants" points to (a).

---

## 4. Evidence from existing UIE research

Each finding is stated with what it implies for the proposal.

| # | Finding | Source | Bearing on the proposal |
|---|---|---|---|
| 1 | Phishing QRY content fails for the **same** reason as benign: `skeleton` keeps the tell (R_T 1.104 / 0.865 / 0.940 phishing vs 1.114 / 1.061 / 1.029 benign); parameter-name identity carries 16-27% vs 17-28% | PQ1 | The main defect is **shared across classes**, so it is not caused by the classes interfering. Splitting does not target it. |
| 2 | The corpus is tail-rich; the model under-produces the tail (name TTR 0.36 vs 0.60; value p90 15 vs 25); tail termination is under-fitted; rare names sit below a learnability threshold | Q5, Q3.1, Q7, Q8 | The mechanism is **exposure to rare structure**. A phishing-only model sees less of it. |
| 3 | Training rows: benign **2,545,571**, phishing **238,214** (10.7x) | E5/R2 training logs | A phishing-only generator trains on 1/11 of what the shared trunk sees, and loses everything it learns about URL grammar from benign data — the grammar phishing imitates. |
| 4 | Phishing label field: **+1.227** train/val gap under over-exposure; `class_alpha = 1.0` exists to prevent it | E2.2 | A standalone phishing model must cycle its small corpus many times, which is the measured failure mode. |
| 5 | Benign val loss bottoms near **step 11,000** and rises; phishing keeps improving (to step 35,000 in R2). `best.pt` is chosen on the *mean* | E3_REPORT, R2 | **The one class-asymmetric finding.** The classes want different training lengths. It motivates per-class checkpoint selection — which needs no architectural split. Its effect on the gate is unmeasured. |
| 6 | Two sampling frames were handled by **naming** them (the frame token), not by splitting models; E5 adopted | E5, `tokenizer.py` | Direct precedent: heterogeneity solved by conditioning inside one model. |
| 7 | E6 (factorised head) was built on a hypothesised mechanism that turned out not to exist, and was rejected | E3_REPORT section 27.3 | Architectural changes must follow a **demonstrated** mechanism. No class-interference mechanism has been demonstrated. |
| 8 | Retraining noise **+-0.0141** at identical configuration | w1 vs E5 | Any retrain perturbs every cell. Separate models *would* isolate repairs — a genuine advantage of the proposal. |
| 9 | The gate's SE is blind to a shared floor offset: 0.040 hidden, **4x** the reported SE | FL | Any A-vs-B architecture comparison run now would be judged by an over-confident instrument. |
| 10 | Phishing's own prefixes elicit ~1/3 the query probability of real ones (attachment ratio 0.32); **cause unexplained** | Q2b, PQ4 deferred | Not evidence for splitting; the registered test (PQ4) has not been run. |
| 11 | Whole-URL excess: benign **+0.2227**, phishing **+0.3992** | R4 gate, AUXILIARY | Generated URLs are highly distinguishable as whole URLs, which bears directly on LIMENX augmentation (section 7). |

**The tempting misreading.** Both convicted cells are phishing, which invites "phishing
needs its own model". The evidence does not support it: benign QRY is not cleared
either (+0.0619), finding 1 shows the QRY mechanism is shared, and phishing's weak
fields are exactly where phishing has the least data. That points to scarcity,
which a split makes worse.

---

## 5. Technical comparison

**A** = current architecture. **B** = proposed routed / specialized architecture.
No scores; each row states what is known and what is not.

| Dimension | A: single class-conditional model | B: router + specialized generators |
|---|---|---|
| **Generation quality** | Gate not passed: 2 convicted, 1 undecided. Causes largely located (shared tail and shape defect, max_len deletion, tail hazard under-fitting); decoder truncation already fixed by R4. | Unknown. No identified mechanism by which splitting fixes the shared defect. The phishing model plausibly gets **worse** (findings 2-4). Improvement cannot be claimed without an experiment. |
| **Controllability** | Class, frame, language, pinned suffix, decode-time biases. No field-presence control. | Class via routing. **Field presence still requires conditioning inside each generator** — routing adds none. |
| **PTH/QRY control** | Decode-time biases can suppress or encourage a field but cannot guarantee a coherent URL: forcing a marker mid-sequence follows a prefix generated without knowing a query was coming. A conditioning token at the start would fix that (section 9). | Same problem, duplicated. "QRY required" handed to a generator with no QRY-conditioning is the same forced marker. |
| **Benign/phishing separation** | Soft: shared weights steered by the class token. Cross-class leakage is **unmeasured**. | Hard by construction: no leakage possible. A genuine advantage, but against an unmeasured problem. |
| **Data efficiency** | All 2.78M rows train one trunk; phishing borrows benign grammar. | Phishing model on 238k rows alone; URL grammar learned twice. |
| **Model capacity** | 14.33M shared. Capacity is **not** shown to be binding (R3 locked; R2's kill criterion did not fire). | 2 x 14.33M if sizes match — more total capacity, directed at a constraint not shown to exist, in a class that memorises when over-exposed. |
| **Training complexity** | One run, one schedule; checkpoint chosen on the mean (finding 5). | Two runs, two schedules, per-class early stopping; double the Kaggle GPU budget. A learned router adds its own training or prompting. |
| **Evaluation complexity** | One canonical gate run covers both classes. | Gate per generator (the gate is already per class, so this is modest), **plus** routing accuracy, **plus** cross-model calibration for any likelihood signal. |
| **Reproducibility** | One checkpoint + seed + decoder fully specifies an artifact (DESIGN section 7). | Two checkpoints + router version. A learned router is **not** seed-reproducible unless made deterministic, which removes its reason to exist. |
| **Failure isolation** | Poor: any retrain moves every cell by up to +-0.0141 (finding 8). | Good: a phishing repair cannot regress benign cells. **B's clearest advantage.** |
| **Maintainability** | One model on the freshness schedule (DESIGN section 8). | Two models to retrain and keep tokenizer-compatible. |
| **Open-source usability** | One ~57 MB checkpoint, one call, no external services. | Two checkpoints; a learned router means extra weights or an API dependency, conflicting with an offline, reproducible recipe. |
| **Inference complexity** | One forward pass per token; the class is a token. | Dispatch plus generator; a learned router adds latency and a failure point. |
| **Risk of specialization under-fitting** | None from specialization. | Phishing model faces under-fit if trained briefly, **memorisation** if trained long (finding 4). |
| **Risk of duplicated knowledge** | None: grammar learned once. | High: host, path and query grammar learned twice, and the phishing copy from 1/11 the data. |
| **Risk of router errors** | None: routing is a caller-supplied argument. | Learned router: misrouting becomes silent label noise; ambiguous requests have no correct route. Deterministic router: none — but that is A's existing API. |

---

## 6. Risks and failure modes the proposal would introduce

**From a learned router**

1. **Silent label noise.** In a dataset maker the class label is the product.
   Every misrouted request writes a phishing-style URL under a benign label or the
   reverse, and nothing downstream can detect it. One percent misrouting is one
   percent label noise in whatever LIMENX or anyone else trains on.
2. **Requests with no correct answer.** "A URL for a bank's login page" is benign
   or phishing depending on intent the router cannot see.
3. **Router distribution shift.** Its behaviour depends on how users phrase
   requests, which drifts, and on the router's version, which the provenance record
   must then carry.
4. **Dual use.** A natural-language front end lowers the barrier to targeted
   phishing generation. DESIGN section 10 holds the line "ship the tool, not the
   ammunition", and section 11.3 defers brand conditioning to inference
   deliberately. An intent-understanding router works against both.

**From separate generators**

5. **Phishing memorisation**, and with it falling novelty (finding 4).
6. **Duplicated, weaker grammar** in the phishing model (findings 2 and 3).
7. **Calibration mismatch** between the two models, which damages the
   likelihood-ratio signal (section 7).
8. **Version drift:** two checkpoints must stay compatible with one tokenizer and
   one suffix table across retraining cycles.

**What B would genuinely improve** — stated fairly:

- **Failure isolation** (finding 8).
- **Per-class training length and hyperparameters** (finding 5).

Both are reachable **within A** — per-class checkpoint selection, or per-class
fine-tuning from the shared trunk — without giving up shared data. Neither is
measured, and each would need its own pre-registration.

---

## 7. LIMENX fifth-model integration analysis

Although the proposal is rejected, the integration question stands on its own, and
the choice of architecture affects it.

### 7.1 Candidate mechanisms

| Mechanism | What it would be | Verdict | Why |
|---|---|---|---|
| Synthetic URLs to train LIMENX | Augment LIMENX training data with generated URLs | **Weak today** | Generated URLs are highly distinguishable as whole URLs (+0.22 benign, +0.40 phishing, finding 11). A detector trained on them can learn "synthetic vs real" instead of "phishing vs benign" — shortcut learning. The labels are "phishing-style", not verified phishing. And it changes the existing four members; it does not add a fifth. |
| Adversarial / challenging samples | Train on hard generated cases | **Weak for training** | Same shortcut objection. Hard for the generator's own reasons is not the same as hard for real attackers' reasons. |
| Evaluation harness | Stress-test LIMENX with generated URLs, graded against real data | **Legitimate, but not a member** | Useful for robustness probing ("real grades it"). It adds no signal to the ensemble. |
| Learned representation | LIMENX trains a probe on UIE's hidden states | **Plausible, unproven** | Couples LIMENX to UIE's internal layer layout, which would need a stable interface. The model was trained with class as an **input**, not to separate classes. |
| Separate detector on generated data | A classifier trained only on UIE output | **Weakest** | Inherits every objection to augmentation, with no real data to anchor it. |
| **Class-conditional likelihood ratio** | For a URL, compute log P(url \| phishing) - log P(url \| benign) from the one shared model | **Defensible — the recommended candidate** | See 7.2. |

### 7.2 Why the likelihood ratio, and not raw typicality

DESIGN section 3 names "its typicality score" as the fifth-member candidate, and
section 7 defines typicality as `log P(url | class)`. **The raw likelihood alone is
a poor detector.** A density model's likelihood tracks length and character
complexity: short, simple URLs score as "typical" under *both* classes. That is the
known failure of likelihood-based anomaly scores.

The **ratio** is the principled form. Both terms come from the **same weights**, so
everything the two classes share — length, character set, generic URL grammar —
largely cancels, and what remains is the evidence the model has learned for one
class over the other. If the model's densities were exact, this ratio would be the
optimal discriminant between the two classes. It uses no sampling, no decoder, and
no synthetic data inside the detector.

**This is where B does specific damage.** With two separately trained models — 10.7x
data asymmetry, different training lengths — the two log-likelihoods carry
different calibration, and the ratio mixes class evidence with model mismatch. This
is a reasoned argument, not a measurement, but it points one way: the shared model
is what makes the ratio meaningful.

### 7.3 The mechanism, concretely

```
UIE (unchanged, never sees LIMENX)
  ships a versioned, deterministic scoring function:
    score(url) -> ( log P(url | benign), log P(url | phishing) )
    teacher-forced; no decoder, no sampling
        |
        v   one-way
LIMENX (owns everything detector-related)
  member 5 = LLR(url) = log P(url|phishing) - log P(url|benign)
  calibrated on LIMENX's own calibration split
  fused through LIMENX's existing calibrated ensemble
  kept ONLY if it wins on the frozen benchmark_v2
    under a pre-registered kill criterion (DESIGN section 3)
```

### 7.4 Preconditions and open risks

1. **Leakage.** DESIGN section 9 already excludes PhiUSIIL and quarantines
   PhreshPhish, the source of `benchmark_v2`. The live phishing-feed collection
   still needs checking: every `benchmark_v2` URL must be tested against UIE's
   training-set filter (`data/bundle2/known_urls.bloom.gz`) before any number
   counts.
2. **Reference-population shift.** UIE's benign distribution is Common Crawl.
   LIMENX's benchmark benign sets come from curated corpora, which E0 measured as
   structurally different — bare homepages are 3.7% of the real web against 12-17%
   in curated corpora, and two ran near 100%. The ratio may score curated benign
   URLs as atypical for reasons unrelated to maliciousness. Must be measured, not
   assumed.
3. **Diversity is the only reason a fifth member helps.** Its errors must be less
   correlated with the existing four than theirs are with each other. Unmeasured.
4. **Gating.** D1 is blocked until roadmap-E3's gate passes (DESIGN section 9), and
   this document does not change that. **Observation for a later decision, not a
   change:** the scorer's value is measured by detection on real data, not by
   sample realism — teacher-forced likelihoods do not pass through the decoder, and
   R4 showed how much the decoder alone moves sample quality. Whether the E3 gate is
   the right precondition for D1 is a fair question.

---

## 8. Dataset-maker vs detector: keeping the roles separate

| Role | What it produces | How it is judged | Owner |
|---|---|---|---|
| **Generation** | Samples from P(url \| class, controls) | The realism gate, novelty, validity | UIE |
| **Synthetic augmentation** | A *use* of samples | Detection on **real** held-out data only; the realism gate is necessary but not sufficient | The consumer |
| **Representation learning** | A *use* of internals | Probe or detector performance on real data | The consumer, via a stable interface |
| **Typicality / likelihood** | A byproduct of the same trained density | Detection on real data; vulnerable to likelihood pathologies, which the ratio mitigates | UIE exposes it; the consumer judges it |
| **Detection** | Verdicts on real URLs | The consumer's frozen benchmark and kill criterion | LIMENX |

**UIE does not become a detector.** It exposes a density; LIMENX decides whether a
function of that density earns a seat in its ensemble. UIE never sees LIMENX's
labels, models or scores, so the one-way rule holds.

**Stated plainly:**
- *"Can the generator's samples make LIMENX better?"* — **scientifically weak today.**
  The measured distinguishability makes shortcut learning the likely outcome.
- *"Can its likelihood ratio add an independent signal?"* — **defensible, but
  unproven.** It may add little beyond four strong members, and the benchmark is the
  only thing that can say.

---

## 9. Decision

### REJECT PROPOSAL

**Why not ACCEPT WITH MODIFICATIONS.** The modifications needed would replace both
defining elements — the learned router with the existing deterministic call, and
the separate generators with the existing shared model. Calling that "acceptance"
would misrepresent it. The goal was sound; the mechanism is not.

**Why rejected**, in one line per element:

- **Learned router:** unnecessary for a small typed request space, and its
  failure mode is silent label noise in the dataset.
- **Specialized generators:** the blocking defect is shared across classes
  (finding 1), the phishing class is data-scarce and memorises (findings 3 and 4),
  and the design's shared-trunk rationale (section 4.4) is supported, not
  contradicted, by everything measured since.

### Kept separate: the reviewer's suggestions, NOT the owner's proposal, NOT adopted

These are my suggestions, prompted by the proposal's goal. This decision does not
adopt them. Each would need its own pre-registration, and each must be judged by
the repaired gate.

1. **Field-presence conditioning tokens inside the current model** — for example
   `<Q:1>` / `<Q:0>` and `<P:1>` / `<P:0>`, derived from the lossless
   factorization. They fit section 4.5's rule ("free or structurally derivable").
   They would make "QRY required" or "no PTH" exact and coherent from the first
   token. They could also let the gate sample query-bearing URLs directly, which
   bears on the phishing QRY measurement-capacity problem (`MIN_N`). Costs: a
   retrain, with its noise; control dropout so unconditional generation still
   works; and a check that conditioned query content matches what the model
   produces unconditioned.
2. **Per-class checkpoint selection or per-class fine-tuning** from the shared
   trunk — the one class-asymmetric finding (finding 5), reachable without a split.
   It gains failure isolation and per-class training length. Hypothesis only.

---

## 10. Exact next steps

1. **The owner decides** on this recommendation.
2. **If approved: pre-register the gate repair, Option C (A and B together).**
   - **Why both.** A alone makes the SE honest but leaves the floor noise large, so
     verdicts become honest but low-powered. B alone shrinks the noise but leaves it
     invisible to the SE. C does both.
   - **Reproducibility.** Keep the current seeding selectable, the way
     `--decoder typical` still reproduces pre-R4 baselines. Record the estimator
     version in every report. Verify the legacy path reproduces published numbers
     bit-identically before anything else.
   - **Re-baseline.** One canonical gate run under the adopted decoder with the
     repaired estimator — about 6 hours, on R4's 10,204 s + 11,922 s.
   - **What does not need re-litigating:** R4's adoption. Its comparison was paired
     on identical floors, so the shared offset cancelled exactly.
   - **Fix the rules first.** How the acceptance table is re-read under the new
     estimator is decided before the run, not after.
3. **Only after the repaired baseline exists:** the optional suggestions in section
   9, each pre-registered separately, if the owner wants them.
4. **LIMENX:** no work now. The integration stays gated by DESIGN section 3 and D1.

---

## 11. What must NOT be changed yet

- **Code:** nothing, including `uie/`, the research scripts and the tokenizer.
- **Evaluation gate:** no change to seeding, D, `MIN_N`, the bar, the floor or the
  estimator until the repair is pre-registered and approved.
- **Thresholds:** the 0.05 bar, the adopted delta (benign 0.22 / phishing 0.30),
  the decoder (`temperature`).
- **Model:** no retraining, no new control tokens, no checkpoint change.
  `best_e5.pt` stays the frozen baseline.
- **Architecture:** DESIGN section 4.4 (single model) and section 4.5 (control
  surface) stand as written.
- **Roadmap:** D1 stays blocked until roadmap-E3's gate passes; R3 stays locked;
  the PTH attribution branch stays closed.
- **Published results:** no edits to existing results, pre-registrations or
  experiment records; nothing deleted or rewritten.
- **LIMENX:** untouched. No integration work, no benchmark access, no changes to
  its ensemble.
- **Status of benign QRY:** reported as **UNDECIDED / not cleared**, never as
  convicted.
