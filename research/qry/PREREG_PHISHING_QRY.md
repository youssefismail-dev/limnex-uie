# Phishing QRY characterisation — pre-registration

**Registered:** 2026-09-23, before any experiment in this file was run.
**Frozen inputs:** `data/best_e5.pt`, `data/bundle2`, `uie/corpus/run_eval.py`
at `8ed0e32`. No generator, checkpoint, decoder, gate code or bar is modified by
this work. **This phase selects no intervention.**

**Production decoder is now `temperature` (R4, adopted 2026-09-23).** Every
"production" arm below means `temperature`, not `typical`. This differs from the
benign phase, which ran when `typical` was the default.

---

## 1. Why this phase exists

Under the adopted decoder, **two cells remain convicted: phishing PTH and
phishing QRY.** Phishing QRY is the larger, at **+0.1431** (pooled, n=356) — and
it is the one row the repair plan's own decision table marks:

    | Phishing query content | UNMEASURED | never measured | — | none | — | — |

`REPAIR_PLAN.md` is now exhausted for benign QRY: R4 adopted, R2 rejected, R3
locked (its gate — R2's *kill* criterion — did not fire), R1 gated on a training
run that is not happening, and loss-weighting and corpus work both forbidden on
refuted premises. No repair is unlocked. This phase is therefore **diagnosis,
read-only first**, exactly as Q1–Q8 were for benign.

---

## 2. What IS already established for phishing — do not re-run

This is more than the decision-table row suggests, and overstating the gap would
waste the phase. **Emission (axis A) is well characterised. Content (axis B) is
not measured at all.**

**Q1 — emission decomposition, both classes, n=4000/cell, delta=0:**

| | corpus (val) | model marginal | production | model share | decoder share |
|---|---:|---:|---:|---:|---:|
| benign | 6.403% | 4.25% | 2.30% | 52% | 48% |
| **phishing** | **4.435%** | **1.95%** | **1.20%** | **76%** | **24%** |

Both kill criteria cleared. `greedy` emitted exactly **0.00%** in phishing.

**Q2 / Q2b — teacher-forced, 1,200 URLs per side:**

    typical retention at legal positions      4.621%   (benign 3.109%)
    attachment ratio, gen vs real prefixes    0.32 mean / 0.25 median
                                              (benign 1.01 / 0.97 -- NO drift)
    eligible positions per URL                real 6.67   gen 4.46
                                              (benign   22.38      14.39)
    pathless rows                             real 62.58%  gen 68.67%
                                              (benign   27.25%      42.42%)

**Mechanism 3, `CHARACTERISATION.md`:** *phishing's own prefixes elicit ~1/3 the
query probability of real prefixes.* Graded **strong as a measurement,
unexplained as a cause.** Benign has no such drift; its deficit is structural
(too few path opportunities), phishing's is probabilistic.

## 3. What is NOT established

**Everything about phishing query CONTENT.** Q3, Q3.1, Q4, Q5, Q6, Q7 and Q8 all
state "benign only" in their own limitations sections. So for phishing there is
no measurement of: which shape property carries the tell, whether the tell is
shape or vocabulary, whether selection explains it, corpus tail richness, the
`max_len` budget, the termination hazard, or the exposure/band profile.

`CHARACTERISATION.md` section 5 names two of these as open explicitly:
*"Why phishing prefixes elicit low query probability"* (#2) and *"Whether Q4
replicates in phishing"* (#3).

---

## 4. Why phishing is expected to differ — the prior, stated before measuring

Registering the expectation matters: if phishing turns out to behave identically
to benign, that is informative only if the opposite was genuinely plausible
beforehand. Five recorded differences say it is.

| | benign | phishing |
|---|---|---|
| emission gap, model share | 52% | **76%** |
| attachment ratio (gen vs real prefixes) | 1.01 — no drift | **0.32** |
| mechanism | structural: too few paths | **probabilistic: prefix collapse** |
| pathless rows, real | 27.25% | **62.58%** |
| training rows | 2,545,571 | **238,214** (10.7x fewer) |

**And the lead this phase is built on.** `E3_REPORT.md` records E5's per-field
train/val NLL gaps at step 20,000:

    field        LBL      PTH      QRY      SUB      SUF
    benign     +0.065   +0.024   -0.107   +0.001   -0.049
    phishing   +0.008   +0.197   +0.162   +0.032   -0.026

**Benign QRY validation is EASIER than train (-0.107). Phishing QRY validation is
HARDER than train (+0.162).** A 0.27 swing between classes on the same field.

This matters because **Q7 H2 tested "generalisation failure" for benign and
REFUTED it.** That refutation was never extended to phishing, and the NLL gap
points the other way. The same is true of phishing PTH (+0.197), the other
convicted cell.

---

## 5. Hypotheses

**PH1 — SHARED SHAPE DEFECT.** Phishing QRY content fails for the benign reason:
the tell is SHAPE (counts, lengths, character-class layout), not vocabulary, and
survives lexical destruction. Benign's mechanism 4a.

**PH2 — LEXICAL DEFECT.** Phishing QRY content fails on vocabulary/identity —
the opposite of benign. Lexical destruction removes the tell.

**PH3 — GENERALISATION FAILURE.** Phishing QRY is a memorisation/generalisation
failure, not an under-fitting one: the model fits training queries and fails to
transfer. **This hypothesis does not exist for benign — it was refuted there.**

PH1 and PH2 are mutually exclusive on PQ1. PH3 is independent of both and is
tested separately by PQ3: a shape defect and a generalisation defect can coexist.

**PH4 — SELECTION.** The emitted queries are a survivorship-biased subsample and
their content is distinguishable because of the selection. Benign's H4, which
Q4 found explains only 15.1% (t = -1.64, ns).

**PH5 — SEPARABLE.** Emission and content have different causes in phishing, as
Q4 concluded for benign.

---

## 6. Experiments

All at the **frozen** `best_e5.pt`. Read-only: no training, no weight or config
write, no intervention.

### PQ1 — is the phishing content tell SHAPE or VOCABULARY? (PRIMARY)

The decisive benign result (4a) was that **`skeleton` retains the entire tell
with every character destroyed** — R_T 1.03, replicated across two arms. It is
an isolating null, and it is the single most transferable test available.

Method: `research/qry/qtransforms.py` unchanged, applied to phishing
query-bearing rows on both sides, floor recomputed under every transform per the
standing rule. Production arm = `temperature`.

Primary transform set, fixed now: `skeleton`, `mask_name_chars`, `clip_values_20`,
`fixlen_names`, `alnum_flat`, `normalize_seps`. `t_mask_names` is **excluded** —
it is documented CONFOUNDED (it destroys intra-name punctuation).

**Pre-registered predictions.**

- **PH1 (shape):** `skeleton` retains the tell, R_T >= 0.85, CI excluding 0.5.
- **PH2 (vocabulary):** `skeleton` destroys it, R_T <= 0.4.
- Between 0.4 and 0.85: **reported as intermediate**, and neither hypothesis is
  claimed. No post-hoc transform is added to break the tie.

**Kill criteria.**
- If the untransformed phishing excess does not reproduce the canonical
  **+0.1431 +- 0.05**, the harness disagrees with the gate and the experiment is
  **void** — fix the harness, do not interpret.
- If any transform moves the real-vs-real floor by more than 0.05, that transform
  has changed something other than the property under test and is dropped from
  the analysis, not reinterpreted.

### PQ2 — does the Q4 selection test replicate? (`CHARACTERISATION.md` open #3)

Content scored separately on `temperature`-emitted and `typical`-emitted queries,
**at equal n**, same real comparator, floor recomputed.

**Prediction, from Q1's decomposition.** Phishing's decoder share is 24% against
benign's 48%, so if selection drives content the phishing effect should be
**smaller** than benign's already-null -0.0201. Registered: **|A - T| < 0.04**,
supporting PH5. A phishing effect LARGER than benign's would contradict the
emission decomposition and must be reported as such rather than absorbed.

### PQ3 — generalisation: hazard and NLL on train vs val (tests PH3)

`research/qry/q7_hazard.py --field QRY --corpus phishing`, on **`phishing_name_val`
and `phishing_name_train`**, at matched row counts.

Benign's answer was a train/val hazard gap of ~zero (1.346 train vs 1.315 val) —
the signature of under-fitting, and the basis for refuting generalisation failure.

**Predictions.**
- **PH3 supported:** the train/val hazard gap OPENS materially — train tracks
  empirical (ratio -> 1.0) while val does not, mirroring the +0.162 NLL gap.
- **PH3 refuted:** the gap stays ~zero as in benign, and phishing QRY is an
  under-fitting problem after all, despite the NLL asymmetry.

**Tooling blocker, and it is a silent one.** `q7_hazard.py:179` hard-codes the
CLASS LABEL, not merely the split name:

    enc = [tok.encode(u, "benign", "und", frame=frame) for u in chunk]

Pointed at `phishing_name_val` as it stands, it would score phishing URLs under
**benign** class conditioning and print a clean, plausible, wrong hazard table
with no error. The frame would be wrong too (`CORPUS_FRAME` is keyed on the
corpus argument). **This must be fixed and tested BEFORE any phishing number is
read**, not adjusted after seeing results, and the fix must be verified by
reproducing the published benign numbers bit-identically on the benign path.

### PQ4 — why phishing prefixes elicit low query probability (open #2)

Mechanism 3 is measured and unexplained. `CHARACTERISATION.md` names three
untested candidates and this experiment tests exactly those three, with no
additions:

1. **host subpopulation** — the model generates hosts resembling the query-less
   subpopulation. Test: partition real phishing hosts by whether they carry
   queries; measure which partition generated hosts resemble.
2. **suffix composition** — generated suffix marginal vs real, and query rate
   conditional on suffix.
3. **path shape** — eligible-position count is already measured at 4.46 vs 6.67;
   decompose into path depth and segment length.

**This is axis A (emission), not the convicted cell.** It is included because
`CHARACTERISATION.md` names it as the main unexplained mechanism in phishing, and
it is **explicitly lowest priority** — deferred if PQ1–PQ3 consume the budget.

---

## 7. Feasibility and power — the limitations, registered in advance

**Generation cost.** At delta=0 the phishing query yield is 1.95% (temperature)
and 1.20% (typical). For 400 query rows per arm that is ~20,500 and ~33,300 URLs.
Benign's Q4 generated 32,000 in 6,782 s on an idle machine, so phishing needs
roughly **2.7x that, ~4-5 h**, and more under contention. PQ1 and PQ2 share one
generation; it must not be run twice.

**The real comparator pool is small.** `phishing_name_val` holds 24,081 URLs of
which ~4.435% carry queries — **about 1,068 rows**, against benign's 7,253.

> **CONSEQUENCE, registered now.** At n=400 each split consumes ~37% of the
> available real rows, against ~6.6% in benign. Split seeds are therefore
> **strongly correlated**, and the split-seed spread **understates** the true
> uncertainty. It is reported as a **LOWER BOUND** and is never used to acquit —
> the same asymmetry already applied to the pooled gate estimator. Where power
> allows, n is reduced to 300 to lower overlap, and n is reported on every line.

**The phishing QRY gate cell is currently unfalsifiable in the clearing
direction.** At the adopted decoder it emits ~2.31%, giving n_fake 180 at
n_main 8000 — below the MIN_N of 250 the canonical estimator requires. It
survives only on the pooled estimator, **which can convict or abstain but never
acquit.** Reaching canonical measurability needs n_main ~11,100, and R4's
phishing half already took 11,922 s at n_main 8000.

> So even a successful repair of this cell **could not currently be verified as
> clearing it.** This is a measurement-capacity problem, it is logically prior to
> any repair, and it is recorded here rather than discovered later. No gate run
> is proposed in this phase.

---

## 8. Standing discipline — carried unchanged from the benign phase

- Report n and uncertainty on every rate. AUC is **not** comparable across sample
  sizes; equalise n or do not compare.
- Recompute the real-vs-real floor under every transform.
- Null control before any difference is called real.
- Never pool axis A and axis B into one number.
- Do not convert "unmeasured" into "pass"; never use a pooled measurement to
  acquit.
- Log failed hypotheses and void instruments in this file rather than deleting
  them.
- **Do not select an intervention during this phase.**
- Do not invent a new experiment because a metric failed.
- Tooling changes are made and tested **before** the numbers they produce are
  read.

## 9. Explicitly out of scope

- R3 / capacity. Still locked; R2's kill criterion did not fire.
- Any retraining, any loss-weighting change, any corpus work.
- The closed PTH attribution branch.
- Phishing **PTH**, the other convicted cell. It shares the +0.197 NLL
  asymmetry and PQ3 may well bear on it, but it is a separate cell and folding it
  in now would repeat the attribution mistake this discipline exists to prevent.
- Any change to the 0.05 bar, the delta, the decoder, or the gate.

## 10. Status

    PQ1  RUN 2026-09-23   PH1 SUPPORTED -- the tell is SHAPE, not vocabulary
    PQ2  RUN 2026-09-23   criterion MET (PH5) but the sign is REVERSED vs benign
    PQ3  RUN 2026-09-23   PH3 REFUTED -- no generalisation gap in termination
    PQ4  NOT RUN          deferred as registered; open question #2 stays open

Results: `PHISHING_QRY_RESULTS.md`. The section-4 prior -- that phishing would
differ from benign -- was WRONG on content, and that is the phase's main finding.
