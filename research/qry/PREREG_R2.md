# R2 — pre-registration: is the tail deficit exposure-limited or capacity-limited?

**Registered:** 2026-09-22, before any training run.
**This is a DIAGNOSTIC, not a repair.** Q7/Q8 left H7 (capacity) and H8
(insufficient optimisation) both PLAUSIBLE and UNMEASURED, and read-only
diagnosis cannot separate them. R2's purpose is to separate them. Framing it as
a repair would lose that information if it fails.

**Runs on Kaggle GPU. Produces a NEW checkpoint under a NEW name.
`best_e5.pt` remains the frozen baseline and is not touched.**

---

## What is being varied, and what is held

| | baseline (`best_e5.pt`) | R2 run |
|---|---|---|
| **steps** | **20,000** | **40,000** |
| batch_size | 256 | 256 |
| lr / warmup / weight decay | 3e-4 / 500 / 0.01 | unchanged |
| d_model / layers / heads / d_ff | 384 / 8 / 8 / 1536 | unchanged |
| **max_len** | **256** | **256 — UNCHANGED** |
| class_alpha | 1.0 | unchanged |
| boundary_weight | 1.0 | **1.0 — S4/S4b stays closed** |
| frame token | on | unchanged |
| decoder at eval | typical | **typical — unchanged** |

**Exactly one variable moves.** `max_len` stays 256 because R1 was demoted on
measured cost/benefit; bundling it here would destroy attribution. The decoder
stays `typical` because R4 varies that, separately and in parallel.

Exposures: 20,000 x 256 = 5.12M -> **40,000 x 256 = 10.24M**, i.e. ~2 epochs
over the ~2.55M-row benign pool becomes ~4.

---

## The discriminating prediction, fixed in advance

Q8 measured a learnability threshold against a faithful-sampler null. If the
deficit is **exposure-limited**, doubling exposures should move that threshold.

| band | corpus % | observed % (baseline) | null % [95% CI] |
|---|---:|---:|---|
| 1 (hapax) | 6.93 | 0.66 | 6.91 [5.14, 8.71] |
| 2–4 | 13.33 | 2.51 | 13.32 [10.95, 15.70] |
| **5–19** | 20.30 | **4.09** | 20.40 [17.41, 23.22] |
| 20–99 | 23.97 | 20.05 | 23.88 [20.98, 27.05] |
| 100–999 | 27.41 | 41.42 | 27.48 [24.27, 30.87] |

- **H8 supported** — band 5–19 rises materially toward its null (20.40%), and
  the over-production at 100–999 falls toward 27.48%.
- **H7 supported instead** — the band profile does not move outside its own
  measurement spread despite twice the exposure. Capacity, not exposure, is
  binding, and R3 becomes the live question.
- **Ambiguous** — movement smaller than the null CI width. Reported as such; a
  4x run would then be the next step, not a conclusion.

**Second instrument, same logic.** Q7's hazard ratio at 50–100 is currently
**1.315 [1.169, 1.494]** on val and **1.346** on train. If exposure-limited,
both should fall toward 1.0.

---

## Acceptance criteria (all required)

1. Band 5–19 observed share rises by more than the null CI half-width (>2.9pp).
2. Q7 hazard ratio at 50–100 falls with a CI excluding the current 1.315.
3. **No regression in currently-clearing gate cells beyond retraining noise**
   (see the confound below).
4. Novelty does not fall.

## Kill criteria (any one rejects the run)

- **MEMORISATION.** Train hazard fits while val does not — i.e. the train/val
  hazard gap OPENS. It is currently ~zero (1.346 vs 1.315), which is the
  signature of under-fitting. This project measured exactly this failure before
  (E2.2, phishing label +1.227 train/val gap under over-exposure) and
  `class_alpha = 1.0` exists to prevent it. **More steps pushes toward the
  failure mode the sampler was designed to avoid.**
- Novelty or distinct-URL rate falls.
- Any currently-clearing cell moves to FAILS **by more than retraining noise**.
- The band profile does not move -> H8 refuted, H7 becomes live, and **no
  further step increase is attempted** on the same reasoning.

## THE CONFOUND THAT LIMITS THIS RUN

R2 produces a new checkpoint, so every field is exposed to **retraining noise,
measured at ±0.0141** (w1-vs-E5). Current clearing margins:

    benign SUF  +0.0029   margin 0.047   assessable
    benign LBL  +0.0301   margin 0.020   marginal
    benign SUB  +0.0391   margin 0.011   <- SMALLER THAN THE NOISE

> **A benign-SUB regression after R2 is UNINTERPRETABLE from a single run.**
> Retraining alone could produce it.

Two honest options, to be chosen before the run, not after:

- **(a) Accept the limit.** Report SUB as not assessable, and treat only
  movements larger than 0.0141 as attributable. Cheapest, and honest.
- **(b) Pay for a control.** A second run at 20,000 steps with the R2 seed,
  isolating retraining noise from the step change. Doubles GPU cost and makes
  the safety claim real.

**Recommendation: (a) for the first run**, because R2's primary purpose is the
H7/H8 discrimination, which the band profile and hazard answer without needing
the gate. Escalate to (b) only if R2 looks adoptable.

---

## Kaggle setup

Dataset unchanged — the existing corpus bundle. `max_len` is not a CLI flag and
stays at the `ModelConfig` default of 256, which is what we want.

```bash
python -m uie.kaggle_train \
    --data     /kaggle/input/<existing-corpus-dataset> \
    --out      /kaggle/working \
    --steps    40000 \
    --batch-size 256 \
    --lr       3e-4 \
    --d-model  384 \
    --n-layers 8 \
    --n-heads  8 \
    --class-alpha 1.0 \
    --boundary-weight 1.0 \
    --eval-every 1000 \
    --frame
```

`--d-model 384 --n-layers 8 --frame` are **required**: the script's defaults are
256 / 6 / no-frame, which would silently train a different architecture and make
the comparison meaningless.

`--boundary-weight 1.0` is stated explicitly rather than left to default, so the
artifact records that S4 was off.

**Expected cost:** twice the baseline run. If the 20,000-step run did not fit
comfortably in one Kaggle session, 40,000 will need checkpoint-resume — confirm
the baseline wall-clock before launching.

**Output:** a new checkpoint. **Name it distinctly** (e.g. `best_r2_40k.pt`).
Do **not** overwrite `best_e5.pt`.

## Evaluation after the run (all on the new checkpoint, nothing adopted)

1. `research/qry/q8_exposure.py` — band profile vs null. **Primary.**
2. `research/qry/q7_hazard.py --field QRY --split name_val` and `--split
   name_train` — hazard, and the train/val gap that is the memorisation kill.
3. `uie/corpus/run_eval.py` at n_main 8000, G=5 x D=3 — safety, read against
   the ±0.0141 confound above.

## Rollback

`best_e5.pt` is untouched throughout. Rollback is: do not adopt. The canonical
gate is run against **both** checkpoints before anything is adopted, and
adoption is a separate decision that this registration does not pre-authorise.
