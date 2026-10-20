# QRY repair plan — PLAN ONLY, NOT IMPLEMENTED

**Status: nothing here has been implemented.** No generator, model, training
code, tokenizer, decoder, sampling policy, loss weighting, checkpoint, gate or
corpus has been modified. Produced only after the root-cause investigation
reached a causal map with four confirmed contributors, two plausible-unmeasured
candidates and five refuted hypotheses.

**Scope: BENIGN.** Phishing query content has never been measured and nothing
here applies to it.

**Supersedes** `REPAIR_PLAN_DRAFT.md`, which was written when only two
contributors were known and correctly declined to recommend anything.

---

## The distinction this document keeps

    FACT        a measurement with a stated interval
    INFERENCE   a conclusion drawn from measurements, labelled as such
    HYPOTHESIS  consistent with the evidence, not established
    REPAIR      a proposal, none of which is implemented

---

## What is established (FACT)

| | measurement |
|---|---|
| corpus is tail-rich | name TTR 0.5853 ±0.01 vs generated 0.3626 at equal n (Q5) |
| train ≈ val | every query metric within ~1 sd (Q5) |
| deletion, not truncation | `encode_corpus` drops over-length rows (Q6, code) |
| deletion is tail-selective | 41.3% / 44.6% of 100+ char rows dropped vs 0.16% / 0.25% baseline (Q6) |
| model stops too eagerly | hazard ratio 1.07–1.13 to pos 50, **1.315 [1.169,1.494]** at 50–100 (Q7 H1) |
| it under-fits TRAIN too | train 1.346 vs val 1.315 (Q7 H2) |
| rare names not reproduced | bands <20: observed 0.66–4.09% vs null 6.91–20.40%, outside 95% CI (Q8 H3) |
| mask does not down-weight QRY | trained/own = 1.0 in both corpora (Q8 H6) |
| decoder truncates rare outcomes | `<QRY>` retained at 3.11% of legal positions; monotone across 4 decoders (Q1) |

## What is inferred (INFERENCE)

- The hazard excess **compounds**: ~8% to position 50 and ~32% beyond gives
  ~54% fewer values reaching 100 characters. *Derived from measured hazards and
  measured survival counts.*
- The deep-tail escalation being query-specific while positions 4–50 are general
  (path field shows the same) is **consistent with** queries receiving ~1/12 the
  field-character share of paths. *Consistent with, not demonstrated.*

## What is hypothesis (HYPOTHESIS)

- That the exposure threshold (H3) **causes** the under-fit hazard (H1).
- That capacity is binding (H7).
- That ~2 epochs is insufficient (H8).
- None is separable read-only. **All three require a training run to test.**

---

# Repairs

## R1 — raise the length budget so the tail stops being deleted

**1. Problem.** `max_len = 256` deletes 41–45% of rows whose longest query value
exceeds 100 characters, against a 0.16–0.25% baseline. CONFIRMED (Q6).

**2. Evidence.** Drop rate rises monotonically across five buckets by a factor
of 258; value p99 falls 108 → 87 and max 1729 → 218 between corpus and loss;
8.6% of the parameter-name vocabulary is removed. Replicates in both corpora.

**3. Proposed intervention.** Raise `ModelConfig.max_len`. The value is a data
decision, not an architectural one, and the docstring already says so.

**4. Expected mechanism.** Fewer tail rows deleted → the loss sees the p99 it is
currently denied → the hazard has the evidence to fit.

**5. Risk.** Positional embedding grows linearly and attention cost grows
quadratically; both raise training cost. Longer sequences change the length
distribution of batches, which interacts with `bucket_window`. **Requires
retraining**, so every other field is exposed to retraining noise (~0.0141,
measured w1-vs-E5).

**6. Interaction.** **PREREQUISITE STATUS WITHDRAWN.** This repair was
registered as a prerequisite on the argument that measuring a hazard repair over
a censored tail is meaningless. The sizing undercuts that argument: deletion
removes rows whose longest value exceeds 100 characters, which chiefly affects
the **100+ bucket that Q7 already reported UNMEASURED**, not the 50-100 bucket
where the 1.315 escalation actually lives. Both empirical and model hazard are
computed on the same surviving rows, so the ratio is internally consistent
either way.

**R1 is therefore OPTIONAL, not prerequisite.** If a training run happens for R2
anyway, `max_len` 384 is a defensible cheap-ish addition at 2.25x — but it must
not be bundled into R2's first test, or attribution is lost.

**7. Minimal experiment.** Read-only first: recompute Q6's drop table at
candidate budgets. **DONE 2026-09-22 (`r1_sizing.py`, `data/qry/r1_sizing.json`),
and the result DEMOTES this repair.**

| max_len | qry drop% | 100+ drop% | p99 seen by loss | names kept | attn cost |
|---:|---:|---:|---:|---:|---:|
| **256 (current)** | 1.35 / 1.94 | 41.3 / 44.6 | **87 / 88** | 91.3% / 92.3% | 1.00x |
| 384 | 0.76 / 0.99 | 25.4 / 26.4 | 93 / 93 | 94.5% / 95.2% | 2.25x |
| 512 | 0.50 / 0.65 | 17.2 / 17.9 | 98 / 98 | 95.8% / 96.7% | 4.00x |
| 768 | 0.29 / 0.39 | 12.4 / 12.9 | 101 / 100 | 97.0% / 98.5% | 9.00x |

(name_train / structure_train; corpus p99 is 108 / 112.)

**The budget saturates below the corpus tail.** Even at 768 — nine times the
attention cost — the loss sees p99 101 against a corpus 108, and 12% of
100+ char rows are still deleted. Translating into the total value-length p99
gap of 68 units (corpus 108 -> production 40):

    max_len 384   recovers  6 of 21 deletion units  =  9% of TOTAL gap  at 2.25x
    max_len 512   recovers 11 of 21                 = 16% of TOTAL gap  at 4.00x
    max_len 768   recovers 14 of 21                 = 21% of TOTAL gap  at 9.00x

**Quadrupling attention cost to recover 16% of the deficit is a poor trade.**
Q6's 41.3% drop rate is arresting; the 0.12% of all rows it represents is not.

**8. Acceptance.** Value p99 seen by the loss rises from 87 toward 108; drop rate
in the 100+ bucket falls below ~10%; **no regression in LBL/SUF/SUB/PTH/FULL
beyond E3_REPORT §30.7 paired noise (all gated |t| < 1.6)**.

**9. Kill.** Any currently-clearing cell moves to FAILS; PTH excess rises beyond
paired noise; training cost rises beyond the compute budget; novelty falls.

**10. Rollback.** New checkpoint under a new name; `best_e5.pt` stays the frozen
baseline and the canonical gate is run against both before adoption.

---

## R2 — give the tail enough optimisation to be learnable

**1. Problem.** The model under-fits the tail hazard **on its own training
data** (1.346 train vs 1.315 val), and rare names below ~165 corpus occurrences
are reproduced at a fifth to a tenth of their due rate. CONFIRMED (Q7 H1,
Q8 H3).

**2. Evidence.** Hazard ratios with CIs excluding 1.0 in five of six buckets on
both splits; frequency-band profile outside the faithful-sampler null in both
directions. Checkpoint config: 5.12M exposures over a ~2.55M-row pool ≈ 2
epochs.

**3. Proposed intervention.** More optimisation — steps, or a schedule that
gives the tail more effective passes. **The specific form is deliberately left
open**, because the discriminating measurement has not been made.

**4. Expected mechanism.** Under-fitting on training data is, by definition, a
failure to extract available signal. More passes over the same data is the
intervention that directly addresses that, and unlike capacity it is cheap to
test.

**5. Risk.** **Memorisation.** This project has measured that exact failure —
E2.2 found phishing's label field at +1.227 train/val gap under over-exposure,
and `class_alpha = 1.0` exists precisely to keep exposure neutral. More steps
pushes toward the failure mode the sampler was designed to avoid. Novelty and
the train/val gap must be watched as first-class outcomes, not afterthoughts.

**6. Interaction.** **AFTER R1.** More passes over a censored tail teaches the
censored distribution harder.

**7. Minimal experiment.** One training run at increased steps, R1 applied,
everything else frozen. Measure the hazard ratio — **the same Q7 instrument** —
before and after.

**8. Acceptance.** Hazard ratio at 50–100 falls toward 1.0 with a CI that
excludes the current 1.315; **train/val hazard gap does not widen**; novelty does
not fall; no gated-field regression beyond paired noise.

**9. Kill.** Train/val hazard gap widens (memorisation); novelty falls; any
clearing cell moves to FAILS; hazard does not move (→ capacity or exposure is
binding, not optimisation, and R3 becomes the live question).

**10. Rollback.** As R1.

---

## R3 — capacity, only if R2 fails

**1. Problem.** Possible representational insufficiency for rare/long query
structure. **PLAUSIBLE, UNMEASURED (H7).**

**2. Evidence.** Only indirect: H2's signature (fits neither split) is consistent
with capacity. It is equally consistent with R2's cause. **No measurement
discriminates them read-only.**

**3. Proposed intervention.** Not specified. **Deliberately.**

**4. Expected mechanism.** N/A until R2 has run.

**5. Risk.** Cost, and the strong possibility of solving the wrong problem.

**6. Interaction.** **STRICTLY AFTER R2, and only if R2's kill criterion fires.**

**7–10.** To be registered only if reached.

> **This repair should NOT be attempted now.** "Bigger model" is the intervention
> most likely to appear to work for the wrong reason, and the project has ruled
> capacity out as binding once before (`field_mask` docstring).

---

## R4 — decoder truncation

**1. Problem.** `decode_typical(mass=0.95)` retains the query marker at 3.11% of
legal positions and concentrates parameter names further (top-1 13.06%
truncated vs 7.86% untruncated). CONFIRMED (Q1, Q4, Q5).

**2. Evidence.** Monotone across four decoders spanning a 5.4x range in emission;
~46% of the name-TTR gap and ~24% of the value p99 gap.

**3. Proposed intervention.** Change truncation aggressiveness, or the decoder,
**at inference only**.

**4. Expected mechanism.** Truncation removes low-probability-but-typical
outcomes; the tail is exactly that.

**5. Risk.** Untruncated sampling was never gated. Removing truncation may
degrade fields that currently clear, and `decode_typical` was chosen on evidence
(E3) — reverting it is not free.

**6. Interaction.** **INDEPENDENT of R1–R3** — inference-time only, no
retraining. But **do not stack it with R1/R2 in one test**: it moves the same
downstream symptom, and a combined run cannot attribute the result.

**7. Minimal experiment.** Generate at the frozen checkpoint under candidate
decoder settings; run the Q3.1 ladder at D=12 and the descriptive statistics.
**No training.**

**8. Acceptance.** Name TTR at equal n rises materially toward 0.5853; QRY excess
does not rise; **no regression in LBL/SUF/SUB/PTH/FULL**.

**9. Kill.** Any gated field regresses; QRY excess rises; novelty or distinct-URL
rate falls; FULL rises.

**10. Rollback.** Revert the decoder parameter. No artifact changes at all —
this is the cheapest repair to undo.

---

## Repair order

REVISED 2026-09-22 after the R1 sizing measurement.

    FIRST          R4   decoder; inference-only, instantly reversible, NO
                        retraining. Largest benefit per unit of risk.
    SECOND         R2   optimisation; the 46% term, but needs a training run
    CONDITIONAL    R3   capacity, ONLY if R2's kill criterion fires
    OPTIONAL       R1   length budget; DEMOTED from prerequisite on measured
                        cost/benefit. Consider max_len 384 only if a training
                        run is already happening, and never in R2's first test
    DO NOT ATTEMPT      any loss-weighting change -- H6 REFUTED the premise
                        (trained/own = 1.0); S4/S4b stays closed
    DO NOT ATTEMPT      corpus work -- Q5 showed the corpus is tail-rich

### OUTCOMES, 2026-09-23 -- this plan is now EXHAUSTED for QRY

    R4   ADOPTED     research/qry/R4_RESULTS.md. All three registered
                     acceptance clauses met; no kill criterion fired; zero of
                     ten cells regressed and five improved significantly.
                     temperature (1.0) is now the project default.
    R2   REJECTED    research/qry/R2_RESULTS.md. PRIMARY criterion 1 FAILED
                     (band 5-19 +2.23pp against a >2.9pp bar, p 0.059), and
                     H8's second condition failed too (100-999 over-production
                     rose instead of falling). Criterion 2 met; memorisation
                     kill NOT triggered.
    R3   STILL LOCKED
                     R3 is gated "STRICTLY AFTER R2, and only if R2's kill
                     criterion fires". NONE of R2's kill criteria fired:
                     memorisation did not appear, and the band profile DID
                     move (weakly). R2 failed on ACCEPTANCE, which is not the
                     same event. **R3 is not unlocked and must not be started
                     on the strength of R2's failure.**
    R1   STILL OPTIONAL
                     Gated on "only if a training run is already happening".
                     None is. Unchanged.

**Consequence: no repair in this plan is unlocked for the QRY defect.** R4 is
spent, R2 is rejected, R3 and R1 remain gated by conditions that did not occur,
and the two remaining directions are explicitly forbidden on refuted premises.
Any further QRY repair requires a NEW pre-registration, not a resumption of this
one.

The plan anticipated this: *"None of these repairs is expected to make the QRY
gate cell pass."* That expectation held.

**The one measurement this plan names and never made** is the last row of the
decision table: **phishing query content -- UNMEASURED, never measured.** The
entire QRY causal map (Q1-Q8) was built on the BENIGN corpus. Phishing QRY is now
one of only two cells still convicted under the adopted decoder, and nothing is
known about whether the benign causes apply to it.

**Why R1 was demoted.** Registering it as prerequisite was reasonable on Q6's
drop rates alone. The sizing showed the recovery saturates below the corpus tail
and costs quadratically, and that the censoring it removes sits mostly in a
bucket already reported UNMEASURED. Demoting it is the evidence acting on the
plan, which is what the read-only-first rule existed to allow.

**On stacking.** R1+R2 are sequential because R2 is uninterpretable without R1.
R4 targets the same downstream symptom by a different mechanism and must be
tested **independently first**, or its effect cannot be separated from R1/R2.
No combined intervention is proposed.

**Expectation to state plainly.** `skeleton` retains the entire content tell with
all lexical content destroyed (Q3.1). **None of these repairs is expected to make
the QRY gate cell pass.** They address measured causes of a measured deficit;
promising gate passage from this evidence would be overselling it.

---

## Decision table

| Cause | Status | Evidence | Estimated contribution | Proposed repair | Priority | Validation |
|---|---|---|---|---|---|---|
| max_len row deletion | CONFIRMED | Q6: 41–45% of 100+ char rows deleted, 258x baseline; p99 108→87 | ~31% of p99 gap, but only ~16% recoverable at 4x attention cost (r1_sizing) | R1 raise budget | **optional — DEMOTED** | sizing DONE; canonical gate if pursued |
| Model under-fits tail hazard | CONFIRMED | Q7 H1: 1.315 [1.169,1.494] at 50–100; 1.346 on train | ~46% of p99 gap | R2 optimisation | **after R1** | Q7 hazard, train vs val |
| Rare structure below learnability threshold | CONFIRMED | Q8 H3: bands <20 at 1/5–1/10 of null, outside 95% CI | **not separable from the hazard term — not summed** | R2 (same repair) | after R1 | Q8 band profile vs null |
| Decoder truncation | CONFIRMED | Q1/Q4/Q5: monotone over 4 decoders; 3.11% retention | ~24% of p99 gap; ~46% of name-TTR gap | R4 decoder | independent | Q3.1 ladder D=12 |
| General long-field under-sustain | CONFIRMED (not QRY-specific) | Q7 H5: path 1.096–1.203 at pos 4–50 | not quantified for QRY alone | covered by R2 | after R1 | Q7 hazard, both fields |
| Capacity | PLAUSIBLE, UNMEASURED | H2 signature; not discriminating | **unquantifiable read-only** | R3 | only if R2 fails | requires training run |
| Insufficient optimisation (~2 epochs) | PLAUSIBLE, UNMEASURED | checkpoint config; under-fits train | **unquantifiable read-only** | R2 | after R1 | requires training run |
| Tail-poor corpus | REFUTED | Q5 | — | none | — | — |
| Pre-loss truncation of values | REFUTED | Q6: deletion, not truncation | — | none | — | — |
| Generalisation failure | REFUTED | Q7 H2 | — | none | — | — |
| Absolute-position / context | NOT SUPPORTED | Q7 H4: direction reverses | — | none | — | — |
| Field-mask under-weighting | REFUTED | Q8 H6: trained/own = 1.0 | — | **none — do not attempt** | — | — |
| Phishing query content | **UNMEASURED** | never measured | — | none | — | — |

Contributions are percentages **of the value-length p99 gap** where a measurement
supports one, and are marked unquantifiable otherwise. The hazard and exposure
terms are **not summed**, because they may be one mechanism measured twice.

---

## Recommended first experiment, on approval

**R1's read-only half is DONE** and demoted the repair. The recommendation moves
to **R4's minimal experiment: the canonical gate under an alternative decoder
setting, at the frozen checkpoint.**

Part of R4's benefit is **already measured** and should not be re-measured:
Q4 found untruncated sampling reduces the content tell by only 15.1% (t = -1.64,
not significant), and Q5 found name TTR rising 0.3626 -> 0.4648, about 46% of
the TTR gap. What is **not** measured is the RISK side — whether removing
truncation regresses the fields that currently clear (benign LBL/SUF/SUB,
phishing LBL/SUF) — and that is what the experiment must establish.

**Cost: a full canonical run at n_main 8000, G=5 x D=3, both classes, which
took ~6.3h on this machine.** It changes nothing: generation at the frozen
checkpoint plus measurement, instantly abandonable, no weight or config written.

**It should not start without explicit approval**, because six hours of machine
time is the user's to spend, and because R4's upside on the CONTENT tell is
already known to be small and non-significant — its real value is deciding
whether the cheap, reversible repair is safe, not whether it is large.
