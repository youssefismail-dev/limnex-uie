# Repair plan — DRAFT, SUPERSEDED

> **SUPERSEDED by REPAIR_PLAN.md.** This draft was written when only two
> contributors were known. Q7/Q8 since confirmed two more (the under-fit tail
> hazard and a learnability threshold) and REFUTED the field-mask premise this
> draft speculated about. Kept as the historical record; **do not act on it.**

**Status:** draft for review. **Nothing in this file has been implemented.** No
generator, decoder, checkpoint, sampling policy, gate, training data or weights
has been touched. Produced because Q3.1 returned **DISTRIBUTED**; it would not
have been produced under UNRESOLVED.

**Read section 6 first if you only read one part.** This plan explains why the
obvious targeted fixes are *not* recommended, and it does not claim the defect
is solved.

---

## 1. Suspected mechanism

**The model under-produces the long tail of rare query content**, and Q5 shows
the tail is present in its training data. Measured twice, independently, in
benign:

| observation | real | generated | measurement |
|---|---|---|---|
| parameter-name diversity at equal n=725 | TTR **0.6023** | 0.3626 (typical) / 0.4648 (untruncated) | Q3.1 §5 |
| top-20 name share of all parameters | **26.68%** | 47.49% / 42.62% | Q3.1 §5 |
| value length p90 (median identical at 4) | **25** | 15 | Q3 §A |

Both are the same failure shape: the centre of the distribution is right, the
tail is missing. The model writes the *common* case correctly and does not
produce the rare case often enough.

**Q5 FINDING 3 SPLITS THESE INTO TWO DEFECTS, NOT ONE**, and this is the single
most consequential result for the plan:

| | training | model untruncated | production | model | decoder |
|---|---:|---:|---:|---|---|
| name TTR | 0.5853 | 0.4648 | 0.3626 | **54%** | **46%** |
| value len p90 | 27.0 | 15.0 | 15.0 | **100%** | **0%** |

Name concentration splits like Q1's emission finding (52/48). The value-length
tail is **entirely the model** — truncated and untruncated sampling both stop at
p90 = 15.

> **Therefore no single-site intervention can address both.** A decoder change
> would partially fix name diversity and do nothing whatsoever for value length.

**Explicitly not established:** *why* the model fails to learn a tail its corpus
contains; that closing either deficit would close the gate.

---

## 2. What would have to change

Three candidate intervention sites, ordered by how directly they target the
*measured* defect. **None is recommended for implementation yet** — see §6.

### Site A — decoder (smallest, most reversible)

Production uses `decode_typical(mass=0.95)`. Q1 measured it retaining `<QRY>` at
3.11% of legal positions, and Q3.1 §5 shows it concentrating names further
(top-1 13.06% truncated vs 7.86% untruncated).

Change: the decoder's truncation aggressiveness, or the decoder itself, at
inference only. **No retraining, no weight change, instantly reversible.**

Targets the *amplifier*, not the cause: untruncated sampling still sits at
TTR 0.4648 against real's 0.6023, so this recovers part of one contributor.

### Site B — training objective (largest, least reversible)

Tail under-production is a distributional property of the learned model. Any
genuine fix lives in the loss or the data weighting — e.g. how rare tokens are
weighted in the field-scoped loss.

**This site is NOT specified further here on purpose.** The project's own record
(S3, S4, S4b) is that loss-reweighting interventions aimed at a distributional
symptom produced an apparent 61% calibration gain that turned out to be entropy
inflation from a mask bug. A loss change must not be proposed from a symptom
measurement alone.

### Site C — corpus / field coverage — **DONE, and it came back clean**

Q5 measured it (Q5_RESULTS.md). At a common subsample the training corpus is
**tail-RICH** and matches validation on every metric (TTR 0.5853 vs 0.5971,
top-20 29.68% vs 28.62%, p90 27.0 vs 25.0). The structure corpus is richer still
(TTR 0.6449, p99 116.5).

> **The corpus is NOT the problem. The model is failing to learn a tail that is
> present in its training data.** The "faithful reproduction" world is excluded
> and a generator-side intervention is warranted in principle.

No corpus work is indicated.

---

## 3. Why this targets the defect and not a proxy

The measured defect is that generated query content is distinguishable from real
at +0.1405 excess, and that the distinguishing information is shape plus
parameter-name identity, with a long-tail deficit as the measured mechanism.

- It does **not** optimise the excess metric directly. No transform, threshold or
  discriminator is altered.
- It targets a property measured **independently of the gate** — TTR at equal n,
  top-k concentration, and the p90/median split are descriptive statistics
  computed without a discriminator, so they cannot be gamed by the thing being
  fixed.
- The success criterion in §5 is defined on **those descriptive statistics
  first**, with the gate as a secondary confirmation. Reversing that order would
  be optimising against the metric — which this project has explicitly forbidden.

---

## 4. Expected measurable outcome

For Site A or C, at the frozen checkpoint:

| quantity | now | target | how measured |
|---|---|---|---|
| name TTR @ n=725 | 0.3626 | toward 0.6023 | Q3.1 §5, equal-n subsample |
| top-20 name share | 47.49% | toward 26.68% | same |
| value length p90 | 15 | toward 25 | Q3 descriptive |
| QRY excess (benign) | +0.1405 | **any reduction is informative; passing the 0.05 bar is NOT expected** | canonical ladder, D=12 |

> **The gate is not expected to pass.** `skeleton` retains the full tell with all
> lexical content destroyed, so repairing names cannot remove the shape carrier.
> A plan that promised gate passage from this evidence would be overselling it.

---

## 5. Pre-registered validation (to be registered separately before any run)

1. **Paired against the frozen baseline**, same generation and split seeds.
2. Descriptive statistics **first**, gate excess second.
3. D = 12 split seeds, df 11, **t_crit 2.201** — the existing rule, unchanged.
4. Full canonical `run_eval` at n_main 8000, G=5 × D=3, **both classes**, to
   catch regressions the QRY ladder cannot see.

### Success criteria (all required)

- name TTR @ n=725 moves >= 0.10 toward real, **and**
- QRY excess does not rise, **and**
- **no regression** in LBL / SUF / SUB / PTH / FULL beyond the paired noise
  established in E3_REPORT §30.7 (all gated |t| < 1.6).

### Kill criteria (any one stops the work)

- any currently-clearing cell (benign LBL/SUF/SUB, phishing LBL/SUF) moves to
  FAILS;
- PTH excess rises in either class beyond paired noise;
- emission rate falls in either class;
- QRY excess rises;
- distinct-URL rate or novelty falls (mode collapse in the other direction);
- the descriptive statistic moves but the gate excess does not — that would
  indicate the statistic is not the carrier after all.

### Rollback

Site A is decode-time: revert the parameter, no artifact changes. Site C is
read-only. Site B would require a new checkpoint under a new name with
`best_e5.pt` retained as the frozen baseline; the canonical gate is re-run
against **both** before anything is adopted.

---

## 6. Recommendation

**Site C is done and came back clean — the corpus is tail-rich — so the
"do nothing to the generator" outcome is excluded. But implementation is still
NOT recommended, for a different reason than before.**

Q5 Finding 3 shows the two deficits have **different mechanisms**: name
concentration is 54% model / 46% decoder, value-length tail is 100% model / 0%
decoder. A decoder change (Site A) addresses part of one and none of the other.
A training change (Site B) cannot be specified because **the cause of the
learning failure is unknown** — the corpus contains the tail and the model does
not reproduce it, and nothing has yet tested why.

Untested candidates for that cause, none of which is a fix:

- **capacity** — 14.33M parameters against a heavy-tailed value distribution;
- **tokenisation** — character-level over values reaching ~100+ characters;
- **context/length budget** — whether long values are truncated before the loss
  ever sees them;
- **loss weighting** — whether field-scoped masking under-weights query content,
  which is where S3/S4 went wrong before and must not be assumed.

**The next experiment should diagnose the learning failure, not repair it.** The
cheapest discriminating check is whether long values survive into training at
all — a tokenisation/length-budget question answerable read-only against
`max_len` and the encoder, with no training run.

Recommending a training intervention now would repeat the S4 pattern: a
plausible mechanism, a symptom measurement, and no verification that the
mechanism is the one operating. That produced an apparent 61% calibration gain
which turned out to be entropy inflation from a mask bug.

**This plan remains a draft and is not recommended for implementation.**