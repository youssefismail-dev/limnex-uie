# Gate v2 — results: verification, and the new baseline

**Registered:** `PREREG_GATE_V2.md` (2f6630b), before any code. **Implemented:**
b6d12e7; memory amendment f4b186c. **Run:** 2026-09-23. **Checkpoint:**
`best_e5.pt` (frozen), `temperature` 1.0, delta 0.22 / 0.30, n_main 8000, G=5.

## 1. Verification — every registered check passed before any v2 number was read

| check | what it required | result |
|---|---|---|
| **V1** legacy path | `--estimator v1` report equals R4's as a whole document | **byte-identical** — 13,054 bytes, sha256 `587f8659d26d7a37…` |
| **V2** code | unit tests: seeds, SE form, verdicts, NOT COMPUTABLE, pooled rule, cache key | **18/18 pass** (`tests/test_gate_v2.py`) |
| **V3** verdict code | in-gate `s261_verdict` (SE = sd/sqrt(G)) on the stored draws of the published runs reproduces the published table | **25/25 cells**: every x to 4 decimals, every verdict (`v3_verdict_check.py`) |
| memory amendment | amended code, `--estimator v1 --classes benign`, equals R4's benign block | **identical** |
| pooled seeds | the pooled rule's first 3 draws are exactly v1's | **identical** (0.1174, 0.1543, 0.1577) |

V3 also confirms the count behind the SE-convention notes: the three published
runs hold 30 cells, 5 of them pooled, so **25** carry a section-26.1 verdict.

Memory, with `gc.collect()` after each scoring call: peak **1.09-1.17 GB** in all
three runs, against 4.5 GB for the same benign scoring before the amendment.

## 2. The v2 baseline — the acceptance reading from now on

G = 5 generations x **D = 10** split seeds, an **independent floor per
(generation, split)**, SE = sd(generation means)/sqrt(5), t_crit 2.776 (df 4).
Reports: `data/gate_v2_baseline_benign/`, `data/gate_v2_baseline_phishing/`.

| cell | x | SE | t vs bar | 95% CI | verdict |
|---|---:|---:|---:|---|---|
| benign LBL | +0.0055 | 0.0031 | −14.25 | [−0.0031, +0.0141] | **CLEARED** |
| benign SUF | +0.0149 | 0.0031 | −11.16 | [+0.0063, +0.0235] | **CLEARED** |
| benign SUB | +0.0067 | 0.0011 | −40.68 | [+0.0036, +0.0098] | **CLEARED** |
| benign PTH | +0.0298 | 0.0039 | −5.19 | [+0.0190, +0.0406] | **CLEARED** |
| benign QRY | +0.0573 | 0.0072 | +1.01 | [+0.0373, +0.0773] | **UNDECIDED** |
| phishing LBL | +0.0204 | 0.0026 | −11.57 | [+0.0132, +0.0276] | **CLEARED** |
| phishing SUF | +0.0074 | 0.0029 | −14.87 | [−0.0007, +0.0155] | **CLEARED** |
| phishing SUB | +0.0302 | 0.0031 | −6.30 | [+0.0216, +0.0388] | **CLEARED** |
| phishing PTH | +0.0443 | 0.0031 | −1.84 | [+0.0357, +0.0529] | **UNDECIDED** |
| phishing QRY | per generation: NOT COMPUTABLE (0/5; n_fake 180 < 250) | | | pooled +0.1333, min +0.0945 over 10 draws | **TELL** (pooled; D=3: TELL) |

**Seven cleared, two undecided, one convicted (pooled). Roadmap-E3's gate is NOT
passed**: an UNDECIDED cell has never counted as passing, and the pooled rule can
convict but never acquit.

## 3. What changed from v1 — one verdict

The generations are the **same** in both runs (the same cache files), so every
difference below comes from the splits: which real rows, which floor and which
fake subsample each draw uses.

| cell | v1 x | v1 SE | v1 verdict | v2 x | v2 SE | v2 verdict | dAUC | dfloor | dx |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| benign LBL | −0.0014 | 0.0020 | CLEARED | +0.0055 | 0.0031 | CLEARED | +0.0046 | −0.0022 | +0.0068 |
| benign SUF | +0.0001 | 0.0043 | CLEARED | +0.0149 | 0.0031 | CLEARED | +0.0106 | −0.0042 | +0.0148 |
| benign SUB | +0.0048 | 0.0023 | CLEARED | +0.0067 | 0.0011 | CLEARED | +0.0046 | +0.0027 | +0.0020 |
| benign PTH | +0.0384 | 0.0026 | CLEARED | +0.0298 | 0.0039 | CLEARED | −0.0067 | +0.0018 | −0.0085 |
| benign QRY | +0.0300 | 0.0086 | UNDECIDED | +0.0573 | 0.0072 | UNDECIDED | −0.0080 | **−0.0352** | +0.0272 |
| phishing LBL | +0.0144 | 0.0048 | CLEARED | +0.0204 | 0.0026 | CLEARED | +0.0061 | +0.0001 | +0.0061 |
| phishing SUF | −0.0064 | 0.0040 | CLEARED | +0.0074 | 0.0029 | CLEARED | +0.0046 | −0.0092 | +0.0138 |
| phishing SUB | +0.0332 | 0.0044 | CLEARED | +0.0302 | 0.0031 | CLEARED | +0.0024 | +0.0054 | −0.0030 |
| **phishing PTH** | **+0.0597** | 0.0018 | **CONVICTED** | **+0.0443** | 0.0031 | **UNDECIDED** | **−0.0101** | **+0.0053** | **−0.0154** |
| phishing QRY (pooled) | +0.1431 | — | TELL | +0.1333 | — | TELL | −0.0088 | +0.0010 | −0.0098 |

v1 SEs are the in-gate function on v1's draws (sd/sqrt(G)); dx = dAUC − dfloor.

### 3.1 phishing PTH: CONVICTED -> UNDECIDED

Two thirds of the fall is AUC (−0.0101), one third a higher floor (+0.0053). The
per-generation means show where it came from:

    v1 (3 shared splits):        0.0595  0.0592  0.0561  0.0665  0.0575   all above 0.05
    v2 (10 independent each):    0.0488  0.0439  0.0491  0.0472  0.0325   all below 0.05

The same five generations moved **together**. That is the signature of a split
effect shared by every generation: in v1 each generation was scored against the
same three real samples and floors, so their luck entered every generation mean
alike and never entered the SE.

The size is consistent with that. Writing a draw as generation effect + split
effect + residual, v1's generation means give the residual (sd ≈ 0.007) and v2's
per-draw spread gives split + residual (sd ≈ 0.026), so the split effect alone is
sd ≈ 0.025. v1's mean of three shared splits then carries a hidden offset of
sd ≈ 0.025/sqrt(3) ≈ **0.014 — eight times the SE v1 reported (0.0018)**. The
observed shift, 0.0154, is about one such sd. (Estimate under the additive
model, with independent terms.)

**What this does and does not say.** The v1 conviction rested on an offset the
instrument could not see; under the repaired instrument the cell is UNDECIDED,
with 0.05 inside its interval [0.0357, 0.0529] — **not cleared**. It is a change
in the measurement, not in the model, and it is not causal evidence: the PTH
attribution branch (`PTH_ATTRIBUTION.md`) stays closed.

### 3.2 benign QRY: the floor artifact, removed

+0.0300 -> **+0.0573**, almost entirely the floor (0.5374 -> 0.5022, −0.0352), as
FL diagnosed. FL2's independent D=30 re-scoring read **+0.0619**; the gap, 0.0046,
is inside the registered 0.02, so no investigation was triggered. The verdict is
unchanged — UNDECIDED — and the value is now honest.

### 3.3 Everything else

No other verdict changed. Values moved by up to 0.015 (benign SUF), all through
the same split mechanism, all far from the bar. The pooled phishing QRY cell
convicts at D=10 and on its first 3 draws alike. Gate pass/novelty and emission
rates are identical (same generations). Precision/recall moved slightly (benign
0.921/0.897 -> 0.910/0.891; phishing 0.928/0.900 -> 0.927/0.911) because v2 seeds
that sample per class; auxiliary, never gated.

## 4. Registered expectations against outcomes (section 7)

| expectation | outcome |
|---|---|
| SEs rise, most for reduced-n cells (QRY) | **Not borne out.** 3 rose, 6 fell; benign QRY's fell (0.0086 -> 0.0072). |
| benign QRY lands near +0.0619; gap > 0.02 investigated | **Met:** +0.0573, gap 0.0046. |
| verdict changes decomposed into AUC and floor | **Done:** one change, phishing PTH (section 3.1). |

**Why the SE expectation was wrong.** v2 changes two things at once, and they push
the SE in opposite directions. With split variance s² and residual variance e²,

    v1:  SE^2 = (g^2 + e^2 / 3) / G      split effect shared -> absent from the SE
    v2:  SE^2 = (g^2 + (s^2 + e^2) / 10) / G

v2's SE is the smaller one whenever s² < (7/3) e². Independent floors add s² to
the SE; averaging 10 splits instead of 3 divides everything by more. My
expectation assumed the first effect dominates everywhere; it dominates only
where split variance is large relative to the residual (phishing PTH: 0.0018 ->
0.0031). The repair was never about the SE's size. It is about the SE now
**including** the split variance, so no offset is hidden behind it.

## 5. Status

The README's adopted-decoder acceptance table is replaced by section 2; the v1
tables stay, marked as v1. E7's safety test (C2) is read against these two
reports.
