# R2 — results: is the tail deficit exposure-limited or capacity-limited?

**Run:** Kaggle GPU, 40,000 steps, completed 2026-09-23. Wall 7,461 s.
**Checkpoint:** `data/best_r2_40k.pt` (from the run's `best.pt`).
`best_e5.pt` untouched throughout. **Nothing is adopted.**

Registered in `PREREG_R2.md`. R2 is a **DIAGNOSTIC**, not a repair: Q7/Q8 left
H7 (capacity) and H8 (insufficient optimisation) both standing, and read-only
diagnosis cannot separate them.

---

## 1. The run did what was registered

Read from the checkpoint itself, not from the notebook's echo:

    model_config   IDENTICAL to best_e5.pt on every field
                   (d_model 384, n_layers 8, n_heads 8, d_ff 1536,
                    max_len 256, vocab_size 101, dropout 0.1)
    tokenizer      sha256 IDENTICAL          -> frame token confirmed present
    tensor shapes  IDENTICAL (101 tensors)
    params         14,333,568 = the 14.33M E5 reports
    train_config   IDENTICAL on every field INCLUDING seed 20260803
                   except  steps  20,000 -> 40,000

Two keys appeared to differ; both are E5 predating the flag, and both defaults
are semantic no-ops: `boundary_weight` (default 1.0, `train.py:95`) and
`struct_ids` (default empty tuple, `model.py:87`). **Exactly one variable moved.**

**Not stated in the pre-registration, and it should have been:** the LR schedule
is length-dependent (`train.py:376-380`, cosine over `cfg.steps`). R2 is
therefore not "E5 plus 20k more steps" — it is a 40k run with its own cosine,
and the two trajectories diverge from step 501. This is the only way to run a
step-scaling experiment, but it means exposure and schedule are confounded by
construction.

---

## 2. PRIMARY READOUT — band profile. **Criterion 1 FAILED.**

Registered test: *band 5-19 observed share rises by more than the null CI
half-width (>2.9pp)*.

Parameter-name frequency bands, generated samples scored against the
faithful-sampler null (`q8_exposure.py`, Q4-protocol generation):

| band | null % | E5 obs % | R2 obs % | change | z | p | direction |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 (hapax) | 6.91 | 0.66 | 1.39 | +0.73 | 1.37 | 0.171 | toward null |
| 2-4 | 13.32 | 2.51 | 3.08 | +0.57 | 0.65 | 0.513 | toward null |
| **5-19** | **20.40** | **4.09** | **6.32** | **+2.23** | **1.89** | **0.059** | toward null |
| 20-99 | 23.88 | 20.05 | 18.64 | -1.41 | -0.67 | 0.505 | AWAY |
| 100-999 | 27.48 | 41.42 | 43.61 | +2.18 | 0.83 | 0.409 | AWAY |
| 1000-inf | 8.01 | 17.81 | 15.10 | -2.71 | -1.36 | 0.173 | toward null |
| OOV | — | 13.46 | 11.86 | -1.60 | | | |

Temperature arm, independently: band 5-19 **+2.51pp** (z 1.87, p 0.061).

**Criterion 1 is not met on either arm** (+2.23 / +2.51 against a 2.9pp bar),
and neither reaches p < 0.05.

**The second half of the registered H8 condition also failed.** H8 required the
over-production at 100-999 to *fall* toward 27.48. It **rose** (+2.18pp typical,
+0.14pp temperature).

Per the pre-registration's own decision rule this is the **AMBIGUOUS** branch:
*"movement smaller than the null CI width. Reported as such; a 4x run would then
be the next step, not a conclusion."*

### Why the 4x run is nonetheless the wrong next step

    band 5-19 remaining gap after R2   14.08 pp
    one doubling of exposure bought     2.23 pp
    doublings needed at that rate        6.3
    => compute multiplier                 80x   (steps ~3,180,000)

That assumes returns stay linear, which is optimistic. And it would be spent on
an axis the loss curve says is already exhausted — see section 4.

**H8 is not refuted.** Six sub-measurements (3 rare bands x 2 arms) all moved
toward the null, which is unlikely to be pure noise. **H8 is also not supported
by the registered test**, and it is shown to be *practically irrelevant*:
exposure cannot close this gap at any affordable scale. **H7 is not confirmed
either** — the profile did move.

---

## 3. Criterion 2 — hazard. **MET.** Memorisation kill — **NOT triggered.**

`q7_hazard.py`, matched to the baseline's row counts (4,931 val / 4,927 train;
286 rows in the critical bucket in both):

| bucket | E5 val | R2 val | E5 trn | R2 trn | E5 v-t | R2 v-t |
|---|---|---|---:|---:|---:|---:|
| 0-4 | 1.128 [1.05,1.22] | 1.083 [1.01,1.17] | 1.038 | 1.026 | +0.091 | +0.057 |
| 4-10 | 1.067 [1.04,1.09] | 1.032 [1.01,1.06] | 1.061 | 1.025 | +0.006 | +0.007 |
| 10-25 | 1.085 [1.06,1.11] | 1.046 [1.02,1.07] | 1.086 | 1.039 | -0.001 | +0.007 |
| 25-50 | 1.106 [1.05,1.17] | 1.032 [0.97,1.10] | 1.068 | 0.991 | +0.038 | +0.041 |
| **50-100** | **1.315 [1.17,1.49]** | **1.137 [1.02,1.27]** | 1.346 | 1.202 | -0.030 | -0.066 |

Criterion 2 as registered — *CI excludes the baseline point estimate 1.315* —
is **met** (upper bound 1.274).

**Qualification, stated because it matters:** a stricter reading (non-overlapping
marginal CIs) would fail, since E5's [1.169, 1.494] and R2's [1.019, 1.274]
overlap. Marginal-CI overlap is the wrong test for what is a *paired* design —
same rows, same buckets — and a paired bootstrap would have more power. The
artifacts store summary statistics rather than per-row pairs, so that test is
not available without re-instrumenting `q7_hazard.py`.

**Memorisation kill criterion: NOT triggered.** Val-minus-train at 50-100 is
-0.030 (E5) and -0.066 (R2): train fits *worse* than val in both. The gap did
not open. More steps did not buy fit through memorisation.

An n discrepancy was caught and corrected: the baseline ran at
`--max-queries 5000`, the first R2 run at the default 6000. A larger n tightens
the CI and so biases criterion 2 in R2's favour. Re-run at matched n; the answer
was unchanged (1.133 -> 1.137).

---

## 4. The free evidence contradicts H8's premise on the benign corpus

From the run's own history:

    benign val NLL    min 1.6709 @ step 11,000  ->  1.7159 @ 40,000
                      (+0.0450 above its own optimum)
    phishing val NLL  still improving, min 1.4814 @ 35,000

**Benign validation loss bottoms at ~step 11,000 — about 1.1 epochs — and rises
monotonically for the remaining 29,000 steps.** H8's premise is that ~2 epochs is
too few. On the benign corpus, which is where QRY lives, the model was already
past its generalisation optimum at ~1.1 epochs.

`best.pt` is selected on `metrics["mean"]` (`train.py:463`), which is half
phishing. The mean bottomed at step 35,000, so **`best_r2_40k.pt` sits 0.044
above this run's own benign optimum.** The selection criterion is 50% a corpus
that has nothing to do with the defect under diagnosis.

This does not by itself settle the band profile — aggregate NLL is head-dominated
and a model can worsen on it while emitting more tail types, which is roughly
what happened. But it removes any case for spending more steps.

### This was already known, and the pre-registration should have said so

**`E3_REPORT.md` already contains this finding for E5**, at section "Validation
loss": min **1.6693 at step 11,000**, rising to 1.6866 at 20,000 — and its
falsified-hypothesis table (line ~984) already lists:

    | E5 is underfit at 20k steps | benign val loss minimum at step 11,000,
      rising after |

R2's own history reproduces it almost exactly (1.6709 at step 11,000), which is
expected given the shared seed and identical early data order.

**So H8's premise was already refuted in the aggregate-NLL sense before
`PREREG_R2.md` registered H8 as "PLAUSIBLE, UNMEASURED."** That registration did
not cite the prior evidence, and I did not check for it before running. A Kaggle
run was spent partly re-deriving a recorded result.

**R2 was not purely redundant.** E3 refuted "underfit" on *aggregate* NLL; R2
asked the narrower question of whether more exposure moves the *tail band
profile*, which aggregate NLL does not answer — and the answer turned out to be
"weakly, and not enough." That part is new. But the framing was avoidably stale,
and the lesson is procedural: **check the existing report's falsified-hypothesis
tables before registering a hypothesis as unmeasured.**

---

## 5. Content discriminability — no gain under the production decoder

`q4_selection.py` re-run on the R2 checkpoint, scored paired by split seed:

| arm | E5 | R2 | diff | SE | t | |
|---|---:|---:|---:|---:|---:|---|
| **typical (production)** | +0.1325 | +0.1318 | **-0.0006** | 0.0263 | -0.02 | not significant |
| temperature | +0.1124 | +0.0351 | -0.0773 | 0.0303 | -2.55 | not significant (t_crit df=2 = 4.303) |

**Under the decoder production actually uses, the QRY content tell is unchanged.**

The temperature arm dropped substantially. An internal control argues the drop is
not an artifact of the n change (476 -> 415 affected both arms; typical moved
-0.0006, temperature -0.0773) — but at 3 split seeds it is underpowered and does
not reach significance. It is an observation, not a result.

Query emission rose slightly: typical 2.38% -> 2.62%, temperature 4.15% -> 4.28%
(real 6.40%).

---

## 6. Verdict

| | registered outcome |
|---|---|
| 1 band 5-19 rises >2.9pp | **FAILED** (+2.23 / +2.51, p ~ 0.06) |
| 2 hazard 50-100 CI excludes 1.315 | **MET** |
| 3 no gate-cell regression | **NOT RUN** — see below |
| 4 novelty does not fall | not run (comes with 3) |
| KILL memorisation gap opens | **NOT triggered** |
| KILL band profile does not move | not triggered (it moved, weakly) |

**RECOMMENDATION: DO NOT ADOPT.** Grounds:

1. The primary registered criterion failed.
2. Under the production decoder the defect is unchanged (-0.0006, t = -0.02).
3. Benign generalisation degraded; the checkpoint is 0.044 past its own optimum
   on the corpus that matters, and was selected on a half-phishing criterion.
4. Exposure is not a viable lever: ~80x compute to close the band gap.

**The 7-hour safety gate is not worth running.** `PREREG_R2.md` says to escalate
to a paid control *"only if R2 looks adoptable."* It does not. Adoption is
already excluded on criterion 1 and on the production-decoder result, so the gate
would measure safety for a checkpoint nobody will ship — and its most sensitive
cell (benign SUB, margin 0.011) is below the ±0.0141 retraining-noise floor and
was pre-registered as not assessable from a single run anyway.

**What R2 bought as a diagnostic — which was its purpose:** the H7/H8 question is
answered well enough to close. Neither hypothesis is cleanly confirmed, but the
practical conclusion is firm and does not depend on which is true: **more
optimisation is not the repair.** Under-training is not the binding constraint,
because the binding constraint appears at ~1.1 epochs and gets worse afterwards.

## Open, not chased

The R2 x temperature interaction (+0.0351, the lowest QRY content tell measured
in this project) is unexplained and underpowered. It is recorded here and
deliberately **not** turned into a follow-up experiment: the standing rule is
*do not invent a new experiment because a metric failed.* Powering it would cost
only additional split seeds, if it is ever wanted.
