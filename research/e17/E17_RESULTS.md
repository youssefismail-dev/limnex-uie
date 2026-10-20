# E17 — results: B5 (the query-plan rule) and option A (stream 2) on `e16_cap`

- **Registration:** `research/e17/PREREG_E17.md` (amendments 1–2). Bundle 04d5c00 /
  2905df34.
- **Read:** 2026-10-08, after every validity check passed. B5 was read first and decided
  `e17_a`'s decode.
- **Criteria:** `data/e17/compare_b5.json` and `data/e17/compare_a.json`
  (`research/e17/e17_compare.py`).

## In plain words

- **B5 works and is ADOPTED under its criteria.**
  - The last failing slash row is fixed: benign control P0Q1 went from 0.99987 to
    **0.99525**, against the real 0.99551 (T 0.00244). **All 8 slash rows now pass.**
  - Not one of 44,514 query plans ended before its query. Without B5, 458 did.
  - No cell regresses under R-a′. The benign query-plan cells each rose a little
    (+0.002 to +0.009). None of these rises is significant, but they all point the same
    way.
  - One verdict flipped at noise level: control benign P1Q1 PTH went from 0.0508
    UNDECIDED to 0.0539 CONVICTED (+0.0031, interval [−0.0033, +0.0095]). The rule
    records it but does not count it.
- **Option A (stream 2) is NOT ADOPTED: K17-1 fails.**
  - The far cell got worse, not better. Free phishing QRY went 0.0733 → **0.0802**
    (+0.0069), and is still CONVICTED. The prediction was −0.005 to −0.025.
  - Control benign P0Q1 QRY fell **−0.0105** (0.0377 → 0.0272, CLEARED). That is less
    than the bar of 0.0141.
  - P1Q1 QRY did not move (0.0424 → 0.0424). Its verdict went from CLEARED to UNDECIDED,
    because its uncertainty changed between the runs.
  - Every other criterion passes, including G5 (thanks to B5) and the fit, K17-2.
- **More data improved the fit but not the tells.**
  - Benign held-out query fit improved by 0.030 nats per character, the top of the
    predicted range. Phishing improved by 0.017.
  - Yet P1Q1 QRY did not move, and the far cell rose.
  - The new queries were barely taken up: stream-2-only strings make up 1.4% of the
    generated P0Q1 queries, against 1.3% for models that never saw stream 2.
- **E3, closest yet on `e16_b5`: 3 cells and 0 slash rows** (`e16_cap`: 3 and 1).
  - The three cells are free phishing QRY (0.0733 CONVICTED), benign P1Q1 PTH (0.0539
    CONVICTED, the noise flip) and benign P1Q1 SUB (0.0431 UNDECIDED).
  - On `e17_a` it is 4 cells and 0 slash rows: P1Q1 QRY joins the list.

## 1. Validity: PASS

| check | result |
|---|---|
| VERIFY_DECODE | 50 files; byte identity 957/957 on the laptop CPU, with B5 on; the decode exactly as registered |
| VERIFY_TRAIN | `e16_cap`'s recipe, with only the stream changed. **V-start:** Kaggle's width-384 rebuild equals `e12_final`'s record, and the width-512 start equals `e16_cap`'s (`caca0107…`). **V-batch:** `adbddc2a…`. **V-mass:** appended rows exactly 2 × the old draws (49,200 and 140,374); 5,642,436 query positions against P8's 5,644,000 (z −0.10). **V-stream:** `128b7d69…`, rate 2. Tests pass. Clipping 0% after step 1,000. Memory 8.6 of 14.6 GB |
| VERIFY_GENERATE | 50 files; byte identity 957/957; decoded with B5, as B5's reading requires |
| V4 | the scorer reproduces `e12_final`'s published cells exactly (0.0) |

## 2. B5: `e16_b5` against `e16_cap` (the same weights)

| criterion | result |
|---|---|
| **K17-B1** | **PASS**: all 8 slash rows within T. Benign control P0Q1: 0.99525 (real 0.99551, diff −0.0003, T 0.0024) |
| **K17-B2** | **PASS** (as amended): 0 of 44,514 Q = 1 URLs end before their query. Reported: 1 cut off at max_len and 37 empty queries (`e16_cap`: 458, 2 and 36). No control state's P, Q or S honouring falls |
| R-a′ | PASS: no regression. One verdict change, recorded and not counted: P1Q1 PTH UNDECIDED → CONVICTED, +0.0031 [−0.0033, +0.0095] |
| R-b | PASS: K12a within T; free QRY rose by +0.0005 (benign, toward its target) and +0.0007 (phishing), both immaterial |
| R-c, R-d, R-e, R-f | PASS |

**Adopted under its criteria:** `e17_a` was generated with raw + D1 + D1′ + B5.

| cell | `e16_cap` | `e16_b5` | Δ |
|---|---|---|---|
| control benign P0Q1 QRY | 0.0288 CLEARED | 0.0377 CLEARED | +0.0089 |
| control benign P1Q1 SUB | 0.0399 UNDECIDED | 0.0431 UNDECIDED | +0.0032 |
| control benign P1Q1 PTH | 0.0508 UNDECIDED | 0.0539 CONVICTED | +0.0031 |
| control benign P1Q1 QRY | 0.0400 CLEARED | 0.0424 CLEARED | +0.0024 |
| free phishing QRY | 0.0756 CONVICTED | 0.0733 CONVICTED | −0.0023 |

How much the rule did:
- B5 acted on every query-plan row. The end mass it moved adds up to about 497 rows
  (63 free, 435 control), close to the 458 early endings measured on `e16_cap`.
- The small rises in the benign query-plan cells point the same way. A small real cost
  of forcing the queries cannot be ruled out, but none of the rises is significant.

## 3. Option A: `e17_a` against `e16_b5`

| criterion | result |
|---|---|
| **K17-1** | **FAIL**. Free phishing QRY +0.0069 (0.0733 → 0.0802); control benign P0Q1 QRY −0.0105; control benign P1Q1 QRY 0.0000. None falls by more than 0.0141. Free benign QRY +0.0075, under the 0.0141 limit |
| **K17-2** | **PASS**: benign held-out query NLL −0.0299 [−0.0330, −0.0269]; P0Q1 rows −0.0301, P1Q1 rows −0.0298. Reported: phishing −0.0171 [−0.0281, −0.0062] |
| **K17-3** | PASS: rare-name share 0.046 against a null of 0.198 (benign); 0.100 against 0.378 (phishing) |
| G1 | PASS (P1Q1 QRY went CLEARED → UNDECIDED at an unchanged x) |
| G2 | PASS: the largest PTH, SUB or LBL rise is +0.0055 (free phishing PTH, still CLEARED) |
| G3 | PASS: no non-query field's validation NLL rose by more than 0.0041 (phishing LBL) |
| G4 | PASS |
| **G5** | **PASS** (strict): all 8 slash rows |
| G6–G9 | PASS: copying under the natural rate everywhere (against old + both streams); novelty within the rule; FULL benign +0.0006, phishing +0.0034 |
| G10 | PASS |

## 4. What it says (reported)

- **The fit/tell split is back, and wider than in E16.**
  - P1Q1 rows' query fit improved by 0.030, and P1Q1 QRY's tell moved 0.000.
  - In E16, width gave −0.051 fit and −0.010 tell. So more data teaches the model to
    predict real held-out queries better, but its own queries do not get harder to
    tell apart from real ones.
- **The far cell went the wrong way.**
  - Phishing query fit improved (−0.017), yet the cell rose by +0.0069.
  - The data was all benign. Within recent models the cell ranges 0.073–0.080
    (`e15_new` 0.0774, `e16_cap` 0.0756, `e16_b5` 0.0733, `e17_a` 0.0802), so this rise
    may be partly run-to-run variation.
  - Either way, benign query practice did not transfer to it on the wider model. In
    E15, it had moved the cell −0.035 on the narrower one.
  - This fits DX8's reading: the cell is CONTEXT-LED, so its tell comes mostly from the
    model's own host and path before the query.
- **Uptake was at chance level.** The share of generated queries that are stream-2-only
  strings:

  | cell | `e17_a` | `e16_b5` | `e16_cap` |
  |---|---|---|---|
  | P0Q1 | 1.40% | 1.29% | 1.30% |
  | P1Q1 | 0.49% | 0.43% | 0.47% |
  | free benign | 0.72% | 0.72% | 0.72% |

  The new data was not memorised, and it was barely reflected in the output.
- **Coverage and diversity (reported):**
  - Free benign template coverage rose: 45.4% → 47.1%.
  - P1Q1 was unchanged (50.6%).
  - P0Q1 became less diverse: 325 → 293 distinct templates per 1,000. Never-seen
    queries went 63.5% → 59.7%.
- **Fit against the tell, for the reference's remaining E3 cells:**

  | cell | Δx | query fit (M3) | the field's own fit |
  |---|---|---|---|
  | free phishing QRY | +0.0069 | −0.017 | QRY −0.018 |
  | P1Q1 PTH | +0.0003 | −0.030 | PTH −0.0003 |
  | P1Q1 SUB | −0.0011 | −0.030 | SUB +0.0014 |

  Better fit to queries was not accompanied by any movement in these three cells. The
  fit of P1Q1's path and subdomain fields did not change at all.

## 5. Predictions (`PREREG_E17.md` §6) against the outcome

| prediction | outcome |
|---|---|
| gates: P1 NO SHIFT, P3–P7 pass; P1Q1 may fail P2 (about 40%) | all pass, P2 included ✓ |
| B5: K17-B1 PASS (about 85%) | PASS ✓ |
| B5: cells move at noise level; free phishing QRY may get slightly worse | noise level ✓; free phishing QRY slightly better (−0.0023) |
| B5: R-a′ PASS (about 75%) | PASS ✓ |
| A: far cell −0.005 to −0.025 | **+0.0069: outside the range, the wrong way** |
| A: K17-1 PASS about 30%, through the far cell | **FAIL** |
| A: P1Q1 QRY and P0Q1 QRY −0.000 to −0.008 | P1Q1 0.0000 ✓; **P0Q1 −0.0105, better than the range** |
| A: fit −0.01 to −0.03 | −0.030 ✓ (the top) |
| A: costs within noise at twice E15's dose | ✓ (G1–G10 pass) |
| G5 passes only with B5 | passed, with B5 ✓ |

## 6. Cost

| run | time |
|---|---|
| E17_decode | 2.53 h |
| E17_train | 3.37 h (202 min); memory 8.6 of 14.6 GB |
| E17_generate_b5 | 2.73 h, plus about 50 min waiting for a Kaggle machine |

## 7. What is decided, and what is the owner's

- **Under the registered rules:**
  - B5 is adopted as a decode rule.
  - `e17_a` is not adopted (K17-1).
  - Replacing the baseline is the owner's decision (§3.3).
- **For the owner:**
  1. **Whether to make `e16_b5` the official configuration.** That is `e16_cap`'s
     weights, decoded with raw + D1 + D1′ + B5.
     - It is the closest to E3 so far: 3 cells and 0 slash rows.
     - The cost is small rises (+0.002 to +0.009) in the benign query-plan cells. These
       include the noise-level PTH flip, recorded and not counted.
     - Binding it in `uie/baseline.py` is the owner's separate step.
  2. **What E18 is, if any.** Every allowed training-side family has now been tested on
     the remaining cells: exposure (E13–E15), capacity (E16) and more data (E17).
     - The two P1Q1 cells, at their current SEs:
       - PTH 0.0539 (SE 0.0012) is above the 0.05 bar itself. It clears at about 0.0467,
         a fall of 0.007.
       - SUB 0.0431 (SE 0.0038) clears at about 0.0395, a fall of 0.004.
     - The far cell has not responded to data or width.
     - No new experiment is proposed here. The options and their risks are for the
       owner to open.
