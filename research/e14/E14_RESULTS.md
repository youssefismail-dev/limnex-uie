# E14 — results

- **Registration:** `research/e14/PREREG_E14.md` (66c56d0). Approved by the owner unchanged.
- **Read:** 2026-10-05, after every validity check passed.
- **Readout:** Appendix A, pasted verbatim from `research/e14/e14_tables.py`.
- **Criteria:** `data/e14/compare_report.json` (`research/e14/e14_compare.py`).

## In plain words

- **Nothing is adopted under the registered rules, and E3 is not met.** The
  registration said E3 could not be met in E14 (§0). That holds.
- **Query exposure (Q) did what N predicted, and paid for it elsewhere.**
  - **The gains:**
    - free phishing QRY fell from 0.1141 to 0.0830, more than twice the 0.0141
      the rule needs. It is still CONVICTED;
    - benign P0Q1 QRY fell from 0.0621 to 0.0365 and is now **CLEARED**;
    - held-out query loss fell in both classes;
    - copying rose toward the real rate and stayed under it in every cell.
  - **The costs:**
    - the path and subdomain cells all got worse, each by less than the
      retraining allowance;
    - without the decoder rule, the model also obeyed the slash plan less well.
  - **Q is rejected on G3.** One slash row that `e12_final` passed now misses:
    phishing `host/?q`, 98.17% against a real 100% (±1.54).
- **D1′ closed the slash condition completely, and was rejected by noise.**
  - **All 8 slash rows pass,** on both `e12_d1p` and `e14_qd`. No arm in E11–E13
    passed more than 7. On `e12_d1p`, S honouring is 100% in 10 of 13
    populations and one URL short in the other three.
  - **It is rejected on R-a's "verdict worsens" clause.** Benign P1Q1 PTH went
    from UNDECIDED to CONVICTED while its tell did not move (0.0576 → 0.0575,
    Holm p 1.00).
    - What moved was the spread between the five generations, which tipped the
      t-statistic across the bar.
    - The registration named this exact risk in advance (§3.2), and it is the
      clause E13 flagged after D2.
- **The combination arm is rejected the same way.** Free benign QRY went from
  CLEARED to UNDECIDED at Δ +0.0008, Holm p 1.00, against `e14_q`.
  - It could not have been adopted anyway, because Q was not.

## 1. Validity: PASS

| check | result |
|---|---|
| VERIFY_TRAIN | `e14_q` is `e12_final`'s recipe except the weight. Same start; the start check passed on Kaggle's torch 2.11; same rows, same dropped rows, same C, same per-band step counts (the same batches); weight 4.0 |
| VERIFY_DECODE | byte identity 669/669 free and 413/413 control. V7: 84,634 of 84,634 P0-plan URLs identical to E13's `e12_d1` (torch 2.11 against 2.10) |
| VERIFY_GENERATE | byte identity 1,082 of 1,082 over `e14_q` and `e14_qd` |
| V4 | the scorer reproduces `e12_final`'s published cells exactly (difference 0.0) |

## 2. Q — query exposure (`e14_q`; not adopted)

| criterion | result |
|---|---|
| K14-1 target | **PASS**: free phishing QRY −0.0311; benign P0Q1 QRY −0.0256 (now CLEARED); benign P1Q1 QRY +0.0020 (no move) |
| K14-2 held-out fit | **PASS**: query NLL per character −0.0729 benign and −0.1848 phishing on validation, both intervals below 0 |
| K14-3 band profile | **PASS**: rare-band share 0.050 / 0.102 against nulls of 0.220 / 0.377; no overshoot |
| G1 no regression | PASS: no cell crossed the retraining rule |
| G2 presence | PASS |
| **G3 slash** | **FAIL**: phishing `free_host_query`, which `e12_final` passed (98.786%), misses at 98.165% (T ±1.544) |
| G4 honouring | PASS |
| G5 copying | PASS: every cell rose toward its natural rate, none over it (Appendix A) |
| G6 novelty | PASS |
| G7 FULL | PASS: +0.0031 benign, +0.0063 phishing |
| G8 clipping | PASS |

**Q's costs, measured but not judged.** Each move is under the 0.0141 retraining
allowance, so G1 passes. They point the same way, and they fall on exactly the
cells E3 still needs:

| cell | `e12_final` | `e14_q` | change |
|---|---|---|---|
| free phishing PTH | 0.0376 CLEARED | 0.0457 UNDECIDED | +0.0081 |
| control benign P1Q1 SUB | 0.0416 UNDECIDED | 0.0536 UNDECIDED | +0.0120 |
| control benign P1Q1 PTH | 0.0576 UNDECIDED | 0.0680 CONVICTED | +0.0104 |
| control phishing P1Q0 PTH | 0.0428 UNDECIDED | 0.0481 UNDECIDED | +0.0053 |

- **The raw slash plan also weakened:**
  - phishing bare hosts missing the slash rose from 76 to 98;
  - benign `S0` honouring fell, from 70.4% to 63.4% (P0Q1, n 71) and from 97.5%
    to 93.4% (P0Q0, n 316).
- **[HYPOTHESIS]** This is the trade-off `NATIVE_REPAIR_PLAN.md` N2b named: "A
  trade-off is itself evidence of capacity in the mixture."
  - Every move is within one retraining's noise, so no single cell proves it.
  - It is the direction of all four together that suggests it.
  - It is not established here, and it does not unlock R3. That needs
    `DECISION_2026-09-29.md` §4's strong proof.
- **Fit against memorisation.** The train/validation gap in query NLL roughly
  doubled: +0.041 → +0.085 benign, +0.027 → +0.105 phishing. That is A2's
  signature.
  - Held-out loss fell anyway, copying stayed under the natural rate everywhere,
    and novelty held.
  - The fit gain is real; the widening gap is reported, not judged.

## 3. D1′ — the slash rule on plans with a path (`e12_d1p`; not adopted)

| criterion | result |
|---|---|
| **K14-4 target** | **PASS: all 8 slash rows within T.** The phishing rows read 100.000% |
| **R-a cells** | **FAIL**: control benign P1Q1 PTH, UNDECIDED → CONVICTED at Δ −0.0001, Holm p 1.00 |
| R-b presence | PASS |
| R-c obedience | PASS: S honouring 100.000% in 10 of 13 populations; the other three are one URL short each (99.993–99.997%) |
| R-d copying and novelty | PASS |
| R-e validity, R-f cost | PASS |

**Why the verdict flipped.** A cell is CONVICTED when (x − 0.05) / SE exceeds
2.776. The SE comes from the five generations.

| arm | x | SE | (x − 0.05) / SE | verdict |
|---|---|---|---|---|
| `e12_final` | 0.0576 | 0.0030 | 2.53 | UNDECIDED |
| `e12_d1p` | 0.0575 | 0.0027 | 2.78 | CONVICTED |

- The tell did not change. The spread between generations did, because D1′
  re-drew 18,280 of the 95,366 URLs with a path, a fresh sample of the same
  distribution (§2.2).
- §3.2 registered this risk by name: the UNDECIDED P1-state cells, P1Q1 PTH
  first, "already tipped once to CONVICTED at Δ ≈ 0". Under the rule as written,
  D1′ is rejected.

## 4. The combination (`e14_qd`; not adopted)

- **K14-4 PASS:** 8 of 8 slash rows. D1′ repairs Q's raw slash losses
  completely.
- **R-a FAIL** against `e14_q`: free benign QRY, CLEARED → UNDECIDED.
  - x 0.0309 → 0.0317, SE 0.0061 → 0.0067, t 3.13 → 2.73. Δ +0.0008, Holm p 1.00.
- It needed Q adopted in any case.

## 5. E3 (read on every arm; nothing counts, since nothing is adopted)

| arm | cells not CLEARED | slash rows missed |
|---|---|---|
| `e14_q` | 6 | 3 |
| `e12_d1p` | 6 | **0** |
| `e14_qd` | 7 | **0** |

Every arm still has these cells uncleared: free phishing QRY, control benign
P1Q1 SUB, PTH and QRY, and control phishing P1Q0 PTH. The full lists are in
Appendix A.

## 6. Predictions (§4), scored

| prediction | held? |
|---|---|
| Q: free phishing QRY to 0.07–0.09, not CLEARED | **yes** (0.0830, CONVICTED) |
| Q: benign P0Q1 QRY falls, stays UNDECIDED | **no**: it CLEARED |
| Q: benign P1Q1 QRY moves less than 0.0141 | yes (+0.0020) |
| Q: K14-2 holds | yes |
| Q: K14-3 holds | yes |
| Q: copying rises toward the natural rate, not over it | yes |
| Q: the guard most at risk is G1 | **no**: G1 passed. The crowding was real but under the allowance, and the guard that failed was G3, which was not predicted |
| D1′: K14-4 holds (8 of 8) | yes |
| D1′: no R fails | **no**: R-a, by the boundary flip §3.2 warned of |
| D1′: V7 holds | yes |
| Combination: K14-4 holds, no R fails | K14-4 yes; **no R fails: no** (R-a, boundary) |
| E3: NOT MET on every arm | yes |
| E3: the six listed cells stay uncleared | 5 of 6; benign P0Q1 QRY cleared under Q |

**10 held, 5 did not.**

## 7. What this leaves for the owner

None of these is licensed by E14's registered rules. Each needs the owner's
decision, and any change of rule is a new registration made after seeing this
data. It must be labelled as such.

1. **The "verdict worsens" clause has now rejected three interventions at Δ ≈ 0:**
   D2 in E13, and D1′ and the combination here.
   - Each time, re-drawn strings changed the spread between generations, not the
     tell.
   - DC1's original rule counted only CLEARED → CONVICTED, and it would have
     passed all three.
   - Whether the owner's "material regression" should mean a verdict flip
     without a measurable change is the owner's call. Changing it now would
     knowingly pass D1′, so it can only be done openly.
2. **D1′ is the slash fix.** No arm in E11–E14 has passed all 8 rows except the
   two that carry it. The best before was E13's D1, with 7 of 8.
3. **Q is a real query lever with a real cost.** It is the first lever to clear
   a failing query cell and to move free phishing QRY by more than the allowance.
   - It makes every remaining path and subdomain cell worse, and those are the
     cells E3 still needs.
   - Under the registration it is closed: one weight, no retry (§2.1).
4. **The next step the owner named, the path and subdomain cells, is now the
   binding one.** Three of the five cells uncleared on every arm are paths or
   subdomains. The other two are the query cells Q could not reach without that
   cost.

## Appendix A — the registered readout

`python research/e14/e14_tables.py`, verbatim.

### Validity

VERIFY_TRAIN, VERIFY_DECODE, VERIFY_GENERATE: PASS, PASS, PASS; V4: max |x - published| 0.0

V5: band steps on the w = 1 mass {'all 0-30': 3454, 'all 30-60': 9973, 'all 60-120': 5637, 'all 120-inf': 936}; query weight 4.0, query share of the scored positions drawn 0.0521
V7: P0-plan URLs identical to e12_d1's 84,634 of 84,634; P1-plan URLs re-drawn by D1' 18,280 of 95,366 (descriptive)
VERIFY_DECODE byte identity: e12_d1p/free 669/669; e12_d1p/control 413/413
VERIFY_GENERATE byte identity: e14_q/free 669/669; e14_q/control 413/413; e14_qd/free 669/669; e14_qd/control 413/413

### The criteria

| lever | criteria | adopted |
|---|---|---|
| Q (e14_q) | K14-1 PASS K14-2 PASS K14-3 PASS G1 PASS G2 PASS G3 **FAIL** G4 PASS G5 PASS G6 PASS G7 PASS G8 PASS | **no** |
| D1' (e12_d1p) | K14-4 PASS R-a **FAIL** R-b PASS R-c PASS R-d PASS R-e PASS R-f PASS | **no** |
| combination (e14_qd) | K14-4 PASS R-a **FAIL** R-b PASS R-c PASS R-d PASS R-e PASS R-f PASS | **no** |

Adopted: {'Q': False, "D1'": False, 'combination': False}

### K14-1 -- the target query cells (e14_q against e12_final)

| cell | e12_final | e14_q | fall | moves (> 0.0141) |
|---|---|---|---|---|
| free phishing QRY | +0.1141 CONVICTED | +0.0830 CONVICTED | +0.0311 | yes |
| control benign_P0Q1 QRY | +0.0621 UNDECIDED | +0.0365 CLEARED | +0.0256 | yes |
| control benign_P1Q1 QRY | +0.0559 UNDECIDED | +0.0579 CONVICTED | -0.0020 | no |

### K14-2 -- held-out query fit (N2a's M3, nats per query character)

| class | split | rows | e12_final | e14_q | difference [95% CI] | falls |
|---|---|---|---|---|---|---|
| benign | validation | 7,204 | 1.5998 | 1.5269 | -0.0729 [-0.0786, -0.0676] | yes |
| benign | train | 19,827 | 1.5588 | 1.4422 | -0.1166 [-0.1201, -0.1133] | yes |
| benign | gap (val - train) | | +0.0410 | +0.0847 | | |
| phishing | validation | 1,048 | 2.5377 | 2.3528 | -0.1848 [-0.2194, -0.1543] | yes |
| phishing | train | 8,130 | 2.5110 | 2.2479 | -0.2631 [-0.2774, -0.2512] | yes |
| phishing | gap (val - train) | | +0.0267 | +0.1049 | | |

### K14-3 -- the band profile (N2a's M2: rare-band share of generated names)

| class | e12_final | e14_q [95% CI] | null | overshoot |
|---|---|---|---|---|
| benign | 0.0557 | 0.0503 [0.0434, 0.0569] | 0.2198 | no |
| phishing | 0.0956 | 0.1020 [0.0898, 0.1145] | 0.3766 | no |

N2a's M1 (value-termination hazard, positions 25-100, 1.0 ideal) and M4 (query copying):

| class | model | M1 train | M1 validation | M4 |
|---|---|---|---|---|
| benign | e12_final | 0.952 [0.928, 0.977] | 1.002 [0.955, 1.053] | 0.1691 |
| benign | e14_q | 0.966 [0.943, 0.991] | 1.024 [0.973, 1.077] | 0.2298 |
| phishing | e12_final | 1.095 [1.056, 1.134] | 1.112 [0.996, 1.258] | 0.1678 |
| phishing | e14_q | 1.048 [1.014, 1.081] | 1.092 [0.982, 1.233] | 0.2276 |

### The slash rows (K12b; T as registered)

| row | real | e12_final | e14_q | e12_d1p | e14_qd |
|---|---|---|---|---|---|
| benign free_bare | 98.004% (±0.626) | 97.731% | 97.843% | 98.039% | 98.051% |
| benign control P0Q0 | 97.944% (±0.518) | 97.704% | 97.825% | 97.885% | 97.885% |
| benign free_host_query | 99.576% (±1.502) | 99.371% | 99.061% | 99.842% | 99.843% |
| benign control P0Q1 | 99.551% (±0.244) | 99.812% **miss** | 99.771% | 99.771% | 99.745% |
| phishing free_bare | 100.000% (±0.042) | 99.675% **miss** | 99.581% **miss** | 100.000% | 100.000% |
| phishing control P0Q0 | 100.000% (±0.067) | 99.759% **miss** | 99.652% **miss** | 100.000% | 100.000% |
| phishing free_host_query | 100.000% (±1.544) | 98.786% | 98.165% **miss** | 100.000% | 100.000% |
| phishing control P0Q1 | 100.000% (±0.200) | 99.960% | 99.919% | 100.000% | 100.000% |

### The slash misses, by supplied state (phishing free bare hosts; S from the seeds)

| arm | bare hosts | without the slash | by supplied state |
|---|---|---|---|
| e12_final | 23,376 | 76 | {'P0Q0S1': 56, 'P0Q1S1': 1, 'P1Q0S1': 17, 'P1Q1S1': 2} |
| e14_q | 23,362 | 98 | {'P0Q0S1': 74, 'P0Q1S1': 3, 'P1Q0S1': 19, 'P1Q1S1': 2} |
| e12_d1p | 23,373 | 0 | {} |
| e14_qd | 23,356 | 0 | {} |

### Gated cells that are not CLEARED in some arm, and FULL

| cell | e12_final | e14_q | e12_d1p | e14_qd |
|---|---|---|---|---|
| free benign QRY | +0.0325 CLEARED | +0.0309 CLEARED | +0.0307 CLEARED | +0.0317 UNDECIDED |
| free benign FULL | +0.0168 AUXILIARY | +0.0199 AUXILIARY | +0.0180 AUXILIARY | +0.0190 AUXILIARY |
| free phishing PTH | +0.0376 CLEARED | +0.0457 UNDECIDED | +0.0394 CLEARED | +0.0463 UNDECIDED |
| free phishing QRY | +0.1141 CONVICTED | +0.0830 CONVICTED | +0.1123 CONVICTED | +0.0783 CONVICTED |
| free phishing FULL | +0.0219 AUXILIARY | +0.0282 AUXILIARY | +0.0225 AUXILIARY | +0.0258 AUXILIARY |
| control benign_P1Q1 SUB | +0.0416 UNDECIDED | +0.0536 UNDECIDED | +0.0434 UNDECIDED | +0.0542 UNDECIDED |
| control benign_P1Q1 PTH | +0.0576 UNDECIDED | +0.0680 CONVICTED | +0.0575 CONVICTED | +0.0671 CONVICTED |
| control benign_P1Q1 QRY | +0.0559 UNDECIDED | +0.0579 CONVICTED | +0.0520 UNDECIDED | +0.0586 UNDECIDED |
| control benign_P1Q1 FULL | +0.0699 AUXILIARY | +0.0688 AUXILIARY | +0.0690 AUXILIARY | +0.0692 AUXILIARY |
| control benign_P1Q0 FULL | +0.0103 AUXILIARY | +0.0162 AUXILIARY | +0.0108 AUXILIARY | +0.0170 AUXILIARY |
| control benign_P0Q1 QRY | +0.0621 UNDECIDED | +0.0365 CLEARED | +0.0624 UNDECIDED | +0.0382 CLEARED |
| control benign_P0Q1 FULL | +0.0813 AUXILIARY | +0.0720 AUXILIARY | +0.0812 AUXILIARY | +0.0698 AUXILIARY |
| control benign_P0Q0 FULL | +0.0123 AUXILIARY | +0.0141 AUXILIARY | +0.0129 AUXILIARY | +0.0131 AUXILIARY |
| control phishing_P1Q0 PTH | +0.0428 UNDECIDED | +0.0481 UNDECIDED | +0.0433 UNDECIDED | +0.0505 UNDECIDED |
| control phishing_P1Q0 FULL | +0.0447 AUXILIARY | +0.0516 AUXILIARY | +0.0434 AUXILIARY | +0.0538 AUXILIARY |
| control phishing_P0Q0 FULL | +0.0213 AUXILIARY | +0.0327 AUXILIARY | +0.0279 AUXILIARY | +0.0377 AUXILIARY |

### Copying against the natural rate (QC1-1's cells)

| cell | natural | e12_final | e14_q | e12_d1p | e14_qd |
|---|---|---|---|---|---|
| free_benign | 0.2927 | 0.1691 | 0.2298 | 0.1686 | 0.2305 |
| control_benign_P0Q1 | 0.4290 | 0.2767 | 0.3570 | 0.2762 | 0.3562 |
| control_benign_P1Q1 | 0.2477 | 0.1606 | 0.1977 | 0.1618 | 0.1967 |
| free_phishing | 0.4185 | 0.1678 | 0.2276 | 0.1715 | 0.2273 |

### Honouring (control states, P and Q) and S honouring

| population | e12_final | e14_q | e12_d1p | e14_qd |
|---|---|---|---|---|
| PQ benign_P1Q1 | 98.25% | 98.01% | 98.26% | 98.07% |
| PQ benign_P1Q0 | 99.68% | 99.60% | 99.73% | 99.71% |
| PQ benign_P0Q1 | 99.16% | 99.14% | 99.19% | 99.25% |
| PQ benign_P0Q0 | 99.58% | 99.60% | 99.62% | 99.63% |
| PQ phishing_P1Q1 | 96.02% | 96.08% | 96.24% | 96.30% |
| PQ phishing_P1Q0 | 99.46% | 99.42% | 99.48% | 99.50% |
| PQ phishing_P0Q1 | 99.22% | 98.94% | 99.52% | 99.42% |
| PQ phishing_P0Q0 | 99.65% | 99.65% | 99.73% | 99.68% |
| S control_benign_P1Q1 S1 | 99.913% of 15,000 | 99.940% of 15,000 | 100.000% of 15,000 | 100.000% of 15,000 |
| S control_benign_P1Q0 S1 | 99.920% of 15,000 | 99.873% of 15,000 | 100.000% of 15,000 | 99.993% of 15,000 |
| S control_benign_P0Q1 S0 | 70.423% of 71 | 63.380% of 71 | 100.000% of 71 | 98.592% of 71 |
| S control_benign_P0Q1 S1 | 99.806% of 14,929 | 99.745% of 14,929 | 99.993% of 14,929 | 100.000% of 14,929 |
| S control_benign_P0Q0 S0 | 97.468% of 316 | 93.354% of 316 | 100.000% of 316 | 100.000% of 316 |
| S control_benign_P0Q0 S1 | 99.748% of 14,684 | 99.775% of 14,684 | 100.000% of 14,684 | 100.000% of 14,684 |
| S free_benign S0 | 94.872% of 195 | 92.308% of 195 | 100.000% of 195 | 100.000% of 195 |
| S free_benign S1 | 99.887% of 39,805 | 99.897% of 39,805 | 99.997% of 39,805 | 100.000% of 39,805 |
| S control_phishing_P1Q1 S1 | 99.740% of 5,000 | 99.820% of 5,000 | 100.000% of 5,000 | 99.980% of 5,000 |
| S control_phishing_P1Q0 S1 | 99.880% of 15,000 | 99.833% of 15,000 | 100.000% of 15,000 | 99.987% of 15,000 |
| S control_phishing_P0Q1 S1 | 99.620% of 5,000 | 99.640% of 5,000 | 100.000% of 5,000 | 100.000% of 5,000 |
| S control_phishing_P0Q0 S1 | 99.693% of 15,000 | 99.607% of 15,000 | 99.993% of 15,000 | 100.000% of 15,000 |
| S free_phishing S1 | 99.790% of 40,000 | 99.725% of 40,000 | 100.000% of 40,000 | 100.000% of 40,000 |

### E3 (section 3.4)

| arm | levers | E3 | cells not CLEARED | slash rows missed |
|---|---|---|---|---|
| e14_q | Q | **NOT MET** | free phishing PTH +0.0457 UNDECIDED; free phishing QRY +0.0830 CONVICTED; control benign_P1Q1 SUB +0.0536 UNDECIDED; control benign_P1Q1 PTH +0.0680 CONVICTED; control benign_P1Q1 QRY +0.0579 CONVICTED; control phishing_P1Q0 PTH +0.0481 UNDECIDED | phishing free_bare, phishing control P0Q0, phishing free_host_query |
| e12_d1p | D1' | **NOT MET** | free phishing QRY +0.1123 CONVICTED; control benign_P1Q1 SUB +0.0434 UNDECIDED; control benign_P1Q1 PTH +0.0575 CONVICTED; control benign_P1Q1 QRY +0.0520 UNDECIDED; control benign_P0Q1 QRY +0.0624 UNDECIDED; control phishing_P1Q0 PTH +0.0433 UNDECIDED | — |
| e14_qd | Q, D1', combination | **NOT MET** | free benign QRY +0.0317 UNDECIDED; free phishing PTH +0.0463 UNDECIDED; free phishing QRY +0.0783 CONVICTED; control benign_P1Q1 SUB +0.0542 UNDECIDED; control benign_P1Q1 PTH +0.0671 CONVICTED; control benign_P1Q1 QRY +0.0586 UNDECIDED; control phishing_P1Q0 PTH +0.0505 UNDECIDED | — |
