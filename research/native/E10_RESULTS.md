# E10 — results: the explicit presence decision

- **Registration:** `PREREG_E10.md` (e850603), before any code or run.
- **Code:** 1c00650 (training and generation, run by the owner on Kaggle).
  - laptop tools: 99fc6d5;
  - readout additions before any reading: 92e6425 and 29a04e8.
- **Read:** 2026-09-29, after VERIFY passed.
- **Numbers:** Appendix A is `e10_tables.py`'s output, rendered from the reports in
  `data/e10/` and pasted by script. Every number in the text appears there or has a
  declared source.
- **Companions:**
  - `PC1_RESULTS.md`: why K10a failed;
  - `PREREG_E11.md`: what comes next, with predictions from this readout.

## In plain words

- **Told what to write, the plan model writes it.** In control mode, with the plan
  supplied, it honours every request 95.0–99.7% of the time. The baseline recipe
  manages 85.4–98.1%.
- **URLs written to a supplied plan are also more realistic:**
  - every control cell the base arm convicts becomes UNDECIDED with the plan:
    - benign path and query: subdomain 0.062 → 0.046, path 0.094 → 0.055, query
      0.065 → 0.054;
    - benign query without a path: 0.077 → 0.064;
  - the phishing path-only subdomain cell clears;
  - in free mode the benign query cell clears (0.028);
  - the phishing query cell becomes measurable for the first time, at 0.093
    (UNDECIDED).
- **Left to choose its own plan in free mode, it chooses wrongly** (K10a, read
  before generation):
  - too many benign paths, 77.0% against 73.4%. PC1 traced 83% of that to gradient
    clipping;
  - too few queries.
  - So free-mode emission misses its registered bands (**K10b fails**).
- **No gate cell regresses** (K10c passes).
- **One guardrail fails** (K10d). In control mode, 22.87% of the plan arm's queries
  are exact copies of training queries, against 20.53% for the baseline recipe.
  Novelty is unchanged.
- **The registered candidate rule:** K10a failed, so the explicit-decision
  hypothesis is refuted as stated. The next question is the hidden state's
  exposure. E11's `e11_joint` arm asks it.
- **What it means.** The plan is the better way to condition. It should not choose
  free mode's presence itself.
  - E11's `e11_plan` supplies the plan from the data's distribution instead,
    approved by the owner (`DECISION_2026-09-29.md` §7).
  - From this readout it is predicted to land within 0.05 points of all four real
    presence rates (`PREREG_E11.md` §3.1).

## 1. Validity: PASS

| check | result |
|---|---|
| V1 | the 100 registered files are present with their registered n, seed and hash; checkpoints as training recorded; one code commit (1c00650) in this history |
| V2 | byte identity, regenerated on the laptop: 1,036 of 1,036 free URLs and 269 of 269 control URLs per arm |
| V3 | `uie/` and the common modules unchanged since 1c00650 |
| V4 | the scorer reproduces E7's published free cells exactly: 0.00e+00 in both classes |
| V-id | one shared start, equal to `e9_ln`'s (`26926388…`); settings as registered; E9's data counts |

## 2. Training

| arm | final validation (benign / phishing / mean) | best step | minutes |
|---|---|---|---|
| `e10_base` | 1.6678 / 1.5263 / 1.5971 | 18,000 | 66.6 |
| `e10_plan` | 1.6071 / 1.4539 / 1.5305 | 19,000 | 66.4 |

The plan arm's validation loss is not comparable: it scores the plan tokens and
has no hidden controls.

## 3. K10a: FAIL (read before generation)

| class | plan P(path) against target | plan P(query) against target |
|---|---|---|
| benign | 77.10% against 73.3%: +3.8, outside ±3 | 0.927 of 6.53%, inside ±10% |
| phishing | 39.83% against 39.4%, inside | 0.893 of 4.51%, outside |

- **The cause is PC1.** Clipping discounts short rows (CLIP-LED, 0.83 of the benign
  path gap).
- **The plan's query shortfall** sits in "a query after a path", and its cause is
  open.
- **The probe artefact** (`PREREG_PC1.md` amendment 1) moves these numbers by under
  0.0001.

## 4. K10b, free-mode emission: FAIL

`e10_plan` writes its own plan in free mode. Each rate is from 40,000 URLs per
class.

| class | PTH (band) | QRY (band) |
|---|---|---|
| benign | 77.03% (70.37–76.37): **outside** | 5.93% (5.96–7.28): **outside** |
| phishing | 40.23% (36.51–42.51): inside | 3.97% (4.21–5.15): **outside** |

- **Emission follows the plan.** Benign PTH 77.03% against the plan's 77.10%.
- **Queries are lost at the plan, not after it.** The plan's own Q was 6.05% and
  4.03%.

**Reported against the detectability tolerance** (`DECISION_2026-09-29.md` §2; not
an E10 criterion):

| arm | benign path | benign query | phishing path | phishing query |
|---|---|---|---|---|
| `e10_base` | +2.51 | −1.55 | −1.64 | −1.79 |
| `e10_plan` | +3.66 | −0.69 | **+0.72, the only one inside** | −0.71 |

The tolerances are ±0.99, ±0.56, ±1.09 and ±0.47.

## 5. K10c, the gate: PASS

**Free mode** (gate v2, G = 5, D = 10): no free cell regresses.

| cell | `e10_base` | `e10_plan` |
|---|---|---|
| benign QRY | 0.0286 UNDECIDED | **0.0280 CLEARED** |
| phishing QRY | not computable | **0.0928 UNDECIDED**, computable for the first time: the plan arm writes enough phishing queries to measure |

FULL (auxiliary) is essentially unchanged: 0.143 against 0.146 benign, 0.380
against 0.385 phishing. The missing root slash dominates it.

**Control mode** (CR1's instrument, native):

| cell | `e10_base` | `e10_plan` |
|---|---|---|
| benign P1Q1 SUB | 0.0618 CONVICTED | **0.0460 UNDECIDED** |
| benign P1Q1 PTH | 0.0944 CONVICTED | **0.0549 UNDECIDED** |
| benign P1Q1 QRY | 0.0649 CONVICTED | **0.0542 UNDECIDED** |
| benign P0Q1 QRY | 0.0773 CONVICTED | **0.0642 UNDECIDED** |
| benign P1Q0 PTH | 0.0148 CLEARED | 0.0074 CLEARED |
| phishing P1Q0 SUB | 0.0441 UNDECIDED | **0.0196 CLEARED** |
| phishing P1Q0 PTH | 0.0382 CLEARED | 0.0408 UNDECIDED; x rises 0.0026 < 0.0141, not a regression by the rule |

- **Every control cell CONVICTED in the base arm is UNDECIDED with the plan.**
- The three that `e9_ln` convicted: P1Q1 PTH and QRY, and P0Q1 QRY, all of them
  query-state cells. The base run also convicts P1Q1 SUB, which `e9_ln` had as
  UNDECIDED.
- The full tables are in Appendix A.

- **The bare state** (benign P0Q0) clears in both arms. The plan's LBL rises
  0.0142 → 0.0237 and stays CLEARED.
- **K10c passes:** no gated cell, free or control, has its verdict worsen with x
  rising by more than 0.0141.

## 6. Honouring (reported)

| state | `e10_base` | `e10_plan` |
|---|---|---|
| benign P1Q1 | 93.43% | 98.69% |
| benign P1Q0 | 98.13% | 99.72% |
| benign P0Q1 | 94.47% | 94.99% |
| benign P0Q0 | 96.98% | 99.10% |
| phishing P1Q1 | 85.44% | 97.32% |
| phishing P1Q0 | 96.12% | 99.54% |
| phishing P0Q1 | 88.70% | 95.12% |
| phishing P0Q0 | 98.09% | 99.59% |

- The plan meets the objective's 0.98 bar in five of eight states; the base, in
  two.
- A query without a path remains the hardest state for both (94.47% / 94.99% benign).

## 7. K10d, guardrails: FAIL

| mode | reading | change [95% CI] |
|---|---|---|
| free | novelty | −0.00 points [−0.07, +0.06] |
| free | query copying | −0.74 points [−2.64, +1.15] |
| control | novelty | +0.02 points [−0.03, +0.07] |
| control | query copying | **+2.34 points [+1.76, +2.92]: worse** |

The plan arm, told to write a query, reproduces a training query more often:
22.87% against 20.53%.

## 8. Reported: `e10_base` against `e9_ln`

Same recipe, same start (`26926388…`); only GPU nondeterminism differs.
- **Free gate cells:** the largest difference is 0.0069, inside the 0.0141
  allowance.
- **Emission** (points): benign PTH +0.30, QRY **+0.48**; phishing PTH −0.33, QRY
  −0.05.
- **Honouring:** within 1.4 points (phishing P1Q1 −1.32).
- **QG1's decomposition** (PREREG_E10 §2, reported; Appendix B):
  - both runs read DECISION-LED;
  - the benign query decision differs between the two runs: 0.793 [0.765, 0.821]
    for `e10_base` against 0.735 [0.709, 0.762] for `e9_ln`;
  - phishing agrees: 0.691 against 0.708.
- **What it means for the tolerance's edge flag** (`DECISION_2026-09-29.md` §2):
  - a single training's benign query rate moves by about half a point from
    nondeterminism alone;
  - that is close to the ±0.56 tolerance;
  - the seed-to-seed spread S is at least this.

## 9. The registered predictions (PREREG_E10 §3), scored

| prediction | result | scored |
|---|---|---|
| K10a passes | fails: the plan over-predicts benign paths (clipping, PC1) and under-predicts phishing queries | **wrong** |
| K10b passes (about 6.4% benign queries) | 5.93% / 3.97%; benign paths 77.03% | **wrong** |
| K10c, K10d: Unknown | K10c passes; K10d fails on control copying | — |
| `e10_base` reproduces `e9_ln` within 0.0141 on free cells | 0.0069 | right |

## 10. What changed my view

- **The plan is good for content, not only for presence.** Before this reading I
  expected presence fixes to leave the convicted content cells untouched: they were
  E12's question.
  - With a supplied plan, the three convicted benign P1Q1 cells drop to UNDECIDED.
  - Path content moves most, from 0.094 to 0.055.
  - A plausible reason [HYPOTHESIS]: the plan is always present and always trained,
    so the model writes a path knowing for certain whether a query follows. Under
    50% control dropout that information is missing half the time.
  - For E12, this means content work should build on the plan arm, not the hidden
    one.
- **The plan's weakness is its own free choice, which E11 no longer uses.** E11's
  supplied plan keeps everything this readout found good and removes the part that
  failed.
- **The guardrail failure needs watching.** E11's K11d compares `e11_plan` with
  `e11_ref` on the same rule.

## 11. What is and is not claimed

**Claimed** [CONFIRMED by the registered readout]:
- the plan's own free choice misses its bands;
- the plan honours requests far better;
- supplied plans improve the benign P1Q1 content cells;
- control-mode copying rises.

**Not claimed:**
- why the plan improves content: a hypothesis;
- anything for another training: one training per arm.

## Appendix A — the rendered tables

`python research/native/e10_tables.py`, verbatim.

### Validity

Verdict: **PASS**; failures: none

Shared start: {'e10_base': '26926388bb575e2c9f70f7a611f762b3e53062f58f6638c70d13f0e46f213ad1', 'e10_plan': '26926388bb575e2c9f70f7a611f762b3e53062f58f6638c70d13f0e46f213ad1'}; equals e9_ln's: True

| arm / mode | URLs regenerated | identical |
|---|---|---|
| e10_base/free | 1036 | 1036 |
| e10_base/control | 269 | 269 |
| e10_plan/free | 1036 | 1036 |
| e10_plan/control | 269 | 269 |

V4 benign: max |x − published| = 0.00e+00

V4 phishing: max |x − published| = 0.00e+00

### Training

| arm | final val (benign / phishing / mean) | best step | minutes |
|---|---|---|---|
| e10_base | 1.6678 / 1.5263 / 1.5971 | 18000 | 66.6 |
| e10_plan | 1.6071 / 1.4539 / 1.5305 | 19000 | 66.4 |

### K10a — the plan's own presence rates (shipped-weighted)

| class | plan P(path) | target | plan P(query) | target | ratio |
|---|---|---|---|---|---|
| benign | 77.10% | 73.30% | 6.05% | 6.53% | 0.927 |
| phishing | 39.83% | 39.40% | 4.03% | 4.51% | 0.893 |

K10a: **FAIL** ['benign plan P 0.7710 is +0.0380 from 0.733', 'phishing plan Q 0.0403 is 0.893 of 0.0451']

### K10b — free-mode emission

| arm | benign PTH | benign QRY | phishing PTH | phishing QRY |
|---|---|---|---|---|
| e10_base | 75.88% | 5.07% | 37.88% | 2.90% |
| e10_plan | 77.03% | 5.93% | 40.23% | 3.97% |
| bands | 70.37–76.37% | 5.96–7.28% | 36.51–42.51% | 4.21–5.15% |

K10b: **FAIL** ['benign QRY: 0.0593 outside (0.0596, 0.0728)', 'benign PTH: 0.77025 outside (0.7037, 0.7637)', 'phishing QRY: 0.039725 outside (0.0421, 0.0515)']

### Reported: free-mode emission against the detectability tolerance (DECISION_2026-09-29 §2; not an E10 criterion)

| arm | class | field | generated | real | difference (points) | T (points) | within |
|---|---|---|---|---|---|---|---|
| e10_base | benign | PTH | 75.88% | 73.37% | +2.51 | ±0.99 | no |
| e10_base | benign | QRY | 5.07% | 6.62% | -1.55 | ±0.56 | no |
| e10_base | phishing | PTH | 37.88% | 39.51% | -1.64 | ±1.09 | no |
| e10_base | phishing | QRY | 2.90% | 4.68% | -1.79 | ±0.47 | no |
| e10_plan | benign | PTH | 77.03% | 73.37% | +3.66 | ±0.99 | no |
| e10_plan | benign | QRY | 5.93% | 6.62% | -0.69 | ±0.56 | no |
| e10_plan | phishing | PTH | 40.23% | 39.51% | +0.72 | ±1.09 | yes |
| e10_plan | phishing | QRY | 3.97% | 4.68% | -0.71 | ±0.47 | no |

### Gate v2, FULL FREE (x, verdict)

| class | field | E7 (canonical) | e10_base | e10_plan |
|---|---|---|---|---|
| benign | LBL | +0.0017 CLEA | +0.0104 CLEA | +0.0061 CLEA |
| benign | SUF | +0.0165 CLEA | +0.0182 CLEA | +0.0130 CLEA |
| benign | SUB | +0.0048 CLEA | +0.0023 CLEA | +0.0034 CLEA |
| benign | PTH | +0.0369 UNDE | +0.0087 CLEA | +0.0088 CLEA |
| benign | QRY | +0.0541 UNDE | +0.0286 UNDE | +0.0280 CLEA |
| benign | FULL | +0.2191 AUXI | +0.1464 AUXI | +0.1433 AUXI |
| phishing | LBL | +0.0238 CLEA | +0.0118 CLEA | +0.0135 CLEA |
| phishing | SUF | +0.0054 CLEA | +0.0109 CLEA | +0.0069 CLEA |
| phishing | SUB | +0.0334 CLEA | +0.0173 CLEA | +0.0143 CLEA |
| phishing | PTH | +0.0604 UNDE | +0.0375 CLEA | +0.0354 CLEA |
| phishing | QRY | NOT COMPUTABLE | NOT COMPUTABLE | +0.0928 UNDE |
| phishing | FULL | +0.4014 AUXI | +0.3845 AUXI | +0.3798 AUXI |

### FULL CONTROL, measurable states (x, verdict)

| state | field | e10_base | e10_plan |
|---|---|---|---|
| benign_P0Q0 | LBL | +0.0142 CLEA | +0.0237 CLEA |
| benign_P0Q0 | SUF | +0.0032 CLEA | +0.0007 CLEA |
| benign_P0Q0 | SUB | +0.0148 CLEA | +0.0072 CLEA |
| benign_P0Q0 | PTH | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q0 | FULL | +0.4910 AUXI | +0.4910 AUXI |
| benign_P0Q1 | LBL | +0.0184 CLEA | +0.0244 CLEA |
| benign_P0Q1 | SUF | +0.0079 CLEA | +0.0082 CLEA |
| benign_P0Q1 | SUB | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q1 | PTH | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P0Q1 | QRY | +0.0773 CONV | +0.0642 UNDE |
| benign_P0Q1 | FULL | +0.4857 AUXI | +0.4880 AUXI |
| benign_P1Q0 | LBL | +0.0037 CLEA | +0.0032 CLEA |
| benign_P1Q0 | SUF | +0.0025 CLEA | -0.0002 CLEA |
| benign_P1Q0 | SUB | +0.0026 CLEA | +0.0014 CLEA |
| benign_P1Q0 | PTH | +0.0148 CLEA | +0.0074 CLEA |
| benign_P1Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE |
| benign_P1Q0 | FULL | +0.0085 AUXI | +0.0107 AUXI |
| benign_P1Q1 | LBL | +0.0187 CLEA | +0.0219 CLEA |
| benign_P1Q1 | SUF | -0.0010 CLEA | -0.0016 CLEA |
| benign_P1Q1 | SUB | +0.0618 CONV | +0.0460 UNDE |
| benign_P1Q1 | PTH | +0.0944 CONV | +0.0549 UNDE |
| benign_P1Q1 | QRY | +0.0649 CONV | +0.0542 UNDE |
| benign_P1Q1 | FULL | +0.0878 AUXI | +0.0634 AUXI |
| phishing_P0Q0 | LBL | +0.0382 CLEA | +0.0304 CLEA |
| phishing_P0Q0 | SUF | -0.0032 CLEA | -0.0023 CLEA |
| phishing_P0Q0 | SUB | +0.0343 CLEA | +0.0243 CLEA |
| phishing_P0Q0 | PTH | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P0Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P0Q0 | FULL | +0.4990 AUXI | +0.4994 AUXI |
| phishing_P1Q0 | LBL | +0.0348 CLEA | +0.0240 CLEA |
| phishing_P1Q0 | SUF | -0.0004 CLEA | +0.0028 CLEA |
| phishing_P1Q0 | SUB | +0.0441 UNDE | +0.0196 CLEA |
| phishing_P1Q0 | PTH | +0.0382 CLEA | +0.0408 UNDE |
| phishing_P1Q0 | QRY | NOT COMPUTABLE | NOT COMPUTABLE |
| phishing_P1Q0 | FULL | +0.0509 AUXI | +0.0376 AUXI |

K10c: **PASS**; regressions: none

### Honouring (control, native)

| state | e10_base | e10_plan |
|---|---|---|
| benign P1Q1 | 93.43% | 98.69% |
| benign P1Q0 | 98.13% | 99.72% |
| benign P0Q1 | 94.47% | 94.99% |
| benign P0Q0 | 96.98% | 99.10% |
| phishing P1Q1 | 85.44% | 97.32% |
| phishing P1Q0 | 96.12% | 99.54% |
| phishing P0Q1 | 88.70% | 95.12% |
| phishing P0Q0 | 98.09% | 99.59% |

### Guardrails (K10d: e10_plan against e10_base)

| arm | mode | novel | copied queries |
|---|---|---|---|
| e10_base | free | 99.56% | 21.04% of 3,184 |
| e10_base | control | 99.69% | 20.53% of 38,547 |
| e10_plan | free | 99.56% | 20.30% of 3,961 |
| e10_plan | control | 99.68% | 22.87% of 38,889 |

K10d: **FAIL** ['control_copying']
- free_novelty: change -0.00 pts [-0.07, +0.06]
- free_copying: change -0.74 pts [-2.64, +1.15]
- control_novelty: change +0.02 pts [-0.03, +0.07]
- control_copying: change +2.34 pts [+1.76, +2.92]

### The plan's off-slot mass (mean per URL; generation records)

| arm | free | control |
|---|---|---|
| e10_base | 0.0 | 0.0 |
| e10_plan | 0.06994200715236439 | 0.0 |

### Reported: e10_base against e9_ln (same recipe, another run)

- largest free-cell x difference: 0.0069 (within 0.0141: True)
- emission difference: benign PTH +0.30 pts, benign QRY +0.48 pts, phishing PTH -0.33 pts, phishing QRY -0.05 pts
- honouring difference: benign_P1Q1 -0.24 pts, benign_P1Q0 +0.01 pts, benign_P0Q1 -0.49 pts, benign_P0Q0 +0.09 pts, phishing_P1Q1 -1.32 pts, phishing_P1Q0 -0.38 pts, phishing_P0Q1 -1.06 pts, phishing_P0Q0 +0.15 pts

### Outcome

- Candidate rule: **K10a fails: the plan decision is itself miscalibrated; the explicit-decision hypothesis is refuted as stated. Next question: the hidden state's exposure (the owner's)**

## Appendix B — QG1's decomposition for `e10_base` (reported)

`python research/qgap/qg1_read.py --with-e10`, the E10 table, verbatim.

### E10, reported: `e10_base` (the baseline recipe, another run, same start) against `e9_ln`

| class | factor | e9_ln | e10_base |
|---|---|---|---|
| benign | decision | 0.735 [0.709, 0.762] | 0.793 [0.765, 0.821] |
| benign | own_starts | 0.979 [0.940, 1.022] | 1.022 [0.979, 1.066] |
| benign | sampler | 0.981 [0.924, 1.041] | 0.973 [0.923, 1.024] |
| benign | G/D | 0.706 [0.656, 0.762] | 0.789 [0.738, 0.844] |
| benign | decision, host route | 0.539 [0.499, 0.580] | 0.625 [0.579, 0.673] |
| benign | decision, path route | 0.806 [0.774, 0.839] | 0.854 [0.820, 0.889] |
| benign | reading | DECISION-LED | DECISION-LED |
| phishing | decision | 0.708 [0.678, 0.742] | 0.691 [0.661, 0.723] |
| phishing | own_starts | 0.964 [0.923, 1.004] | 0.948 [0.909, 0.987] |
| phishing | sampler | 0.951 [0.876, 1.025] | 1.005 [0.935, 1.082] |
| phishing | G/D | 0.649 [0.592, 0.710] | 0.658 [0.601, 0.724] |
| phishing | decision, host route | 0.592 [0.549, 0.640] | 0.630 [0.584, 0.680] |
| phishing | decision, path route | 0.780 [0.738, 0.827] | 0.728 [0.689, 0.771] |
| phishing | reading | DECISION-LED | DECISION-LED |
