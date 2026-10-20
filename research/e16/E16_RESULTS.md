# E16 — results: the capacity probe

- **Registration:** `research/e16/PREREG_E16.md` (amendments 1–2). Bundle eeeb760 /
  d7379975.
- **Read:** 2026-10-07, after every validity check passed.
- **Criteria:** `data/e16/compare_report.json` (`research/e16/e16_compare.py`), against
  `e15_new`.

## In plain words

- **Capacity moved P1Q1 for the first time.** Three exposure levers never moved it.
  - Benign P1Q1 QRY fell from 0.0500 to **0.0400**, and is now **CLEARED**, though only
    just (t −2.86 against the bar of −2.776).
  - That fall is 0.0100: under K16-1's 0.0141, but over the end rule's partial-response
    line of 0.0082 (2.4 between-run SDs).
  - Benign P1Q1 PTH fell 0.0043, to 0.0508, and is still UNDECIDED.
- **The end rule reads row 2: capacity-sensitive, but not enough at 1.77×.**
  - There is no final E3 report.
  - E17 is the owner's choice, as registered: option A (stream 2, already acquired) on
    `e16_cap`, or a further capacity step (R3 approved again).
- **Not adopted under the registered rules.**
  - K16-1 fails (the fall is under 0.0141).
  - G5 fails on the same slash row as `e15_new`, now further out: 0.99987 against
    0.99551 (T 0.00244). The quick check shows "no slash" P0Q1 plans keeping their
    query 2.5% of the time (`e15_new`: 5.5%).
  - Everything else passes. Every content field fits better, and no cell rises by more
    than 0.0053.
- **E3: the closest yet.** Three cells and one slash row remain (`e15_new`: five and one):
  - free phishing QRY, 0.0756 CONVICTED. Capacity barely touched it (−0.0018), although
    phishing query fit improved by 0.10 nats per character;
  - benign P1Q1 PTH, 0.0508 UNDECIDED;
  - benign P1Q1 SUB, 0.0399 UNDECIDED: a noise-level flip from CLEARED (+0.0026);
  - the benign control P0Q1 slash row.
- **Also newly CLEARED:** benign P0Q1 QRY (0.0410 → 0.0288, t −4.93) and free phishing
  PTH (0.0369 → 0.0255).

## 1. Validity: PASS

| check | result |
|---|---|
| VERIFY_TRAIN | `e15_new`'s recipe at width 512 in 2 parts, verified against the record the notebook made before training. V-start′: Kaggle's width-384 rebuild equalled `e12_final`'s recorded start; training started from the recorded width-512 start (`caca0107…`). V-batch: `adbddc2a…`. V-mass′: the injection equals `e15_new`'s exactly. V-stream `637accab…`. V-accum tests pass. Clipping 0% after step 1,000. Memory 8.1 of 14.6 GB |
| VERIFY_GENERATE | 50 files; byte identity 957/957 on the laptop CPU |
| V4 | the scorer reproduces `e12_final`'s published cells exactly (0.0) |

## 2. Criteria (`e16_cap` against `e15_new`)

| criterion | result |
|---|---|
| **K16-1** | **FAIL**: P1Q1 QRY −0.0100 (now CLEARED); P1Q1 PTH −0.0043. Neither falls by more than 0.0141 |
| **K16-2** | **PASS**: benign held-out query NLL −0.0512 [−0.0550, −0.0478]; on P1Q1 rows −0.0512 [−0.0556, −0.0470]. Reported: phishing −0.1014 [−0.1221, −0.0841]; benign training sample −0.0620 |
| **K16-3** | PASS: rare-name share 0.049 against a null of 0.208 (benign); 0.091 against 0.378 (phishing) |
| G1 | PASS (P1Q1 SUB went CLEARED → UNDECIDED at +0.0026, under the retrain rule) |
| G2 | PASS: largest PTH, SUB or LBL rise +0.0053 |
| G3 | PASS: every non-query field's validation NLL fell (benign LBL −0.044, PTH −0.032; phishing PTH −0.077, SUB −0.047) |
| G4 | PASS |
| **G5** | **FAIL**: benign control P0Q1, 0.99987 against 0.99551 (diff +0.0044, T 0.0024). The other 7 rows pass |
| G6–G9 | PASS: copying UNDER the natural rate in every cell; novelty within the rule; FULL benign +0.0001, phishing −0.0083 |
| G10 | PASS |

**End rule: row 2.** K16-1 FAIL; K16-2 PASS; P1Q1 QRY's fall (0.0100) is at least 2 ×
its between-run SD (0.0082).

## 3. The cells that changed

| cell | `e15_new` | `e16_cap` | Δ |
|---|---|---|---|
| control benign P1Q1 QRY | 0.0500 UNDECIDED | **0.0400 CLEARED** | −0.0100 |
| control benign P1Q1 PTH | 0.0551 UNDECIDED | 0.0508 UNDECIDED | −0.0043 |
| control benign P0Q1 QRY | 0.0410 UNDECIDED | **0.0288 CLEARED** | −0.0122 |
| free phishing PTH | 0.0369 UNDECIDED | **0.0255 CLEARED** | −0.0114 |
| free phishing QRY | 0.0774 CONVICTED | 0.0756 CONVICTED | −0.0018 |
| control benign P1Q1 SUB | 0.0373 CLEARED | 0.0399 UNDECIDED | +0.0026 |
| free benign QRY | 0.0256 CLEARED | 0.0165 CLEARED | −0.0091 |
| control phishing P1Q0 PTH | 0.0405 CLEARED | 0.0294 CLEARED | −0.0111 |

Every other cell stays CLEARED.

## 4. What it says (reported)

- **Width moved what data and exposure could not.** For P1Q1 QRY, three exposure levers
  gave −0.003, +0.002 and −0.002; width gave −0.010.
  - The fit/tell split E15 found is partly closed: fit −0.051 on P1Q1 rows, tell −0.010.
  - Under E15's exposure, the same fit gain had moved the tell by only −0.002.
- **The far cell is the exception.**
  - Phishing query fit improved by 0.10 nats per character, more than any lever before,
    yet the tell moved by only 0.002.
  - That fits DX8's reading: the cell is CONTEXT-LED. Most of its tell comes from the
    model's own host and path before the query, which neither width nor fit addresses.
- **The slash row worsens with every model that learns queries better.** The model
  increasingly drops the query on the rare "no slash" plan.
  - This is a plan-honouring failure, not a slash failure: D1 places the slash exactly.
  - The candidate fix listed in `E16_DESIGN.md` §3.2 (B5, a decode rule that honours a Q1
    plan) has not been registered.
- **Descriptive (coverage report):** P1Q1 template coverage 48.8% → 50.6%. Never-seen
  queries are unchanged (81%); the long tail is still short (queries over 128 characters:
  1.8% against a real 3.6%).

## 5. Predictions (`PREREG_E16.md` §9) against the outcome

| prediction | outcome |
|---|---|
| K16-2 PASS (about 90%) | ✓ |
| P1Q1 QRY 0.038–0.052 | 0.0400 ✓ |
| P1Q1 PTH 0.043–0.057 | 0.0508 ✓ |
| P0Q1 QRY 0.031–0.042 | **0.0288: better than the range** |
| free phishing QRY 0.052–0.080 | 0.0756 ✓ |
| G5 FAIL about 55% | FAIL ✓ |
| end rule: row 3 about 50%, row 2 about 25%, row 1 about 20% | **row 2** |
| adoption about 10% | not adopted ✓ |

## 6. What is decided, and what is the owner's

- **Under the registered rules:** not adopted, and the baseline stays `e15_new`.
- **The end rule (row 2):** no final report. E17 is the owner's choice:
  1. **option A on `e16_cap`**: stream 2 (data/e17/qry_extra2, 57,292 rows, already
     acquired and hashed), after its population gates. This tests whether the wider
     model gets more out of more data;
  2. **a further capacity step,** which needs R3 approved again.
- **Separately, the owner's decision:** whether `e16_cap` should replace `e15_new` despite
  K16-1 and G5. It is the best model on E3's finish line so far: 3 cells and 1 slash row.

## 7. The owner's decision (2026-10-07, after this reading)

- **K16-1 and G5 waived, for this adoption only. `e16_cap` ADOPTED** as the official
  baseline, superseding `e15_new`. Recorded verbatim in
  `research/native/DECISION_2026-10-07.md`.
- **E17 is option A (stream 2) on `e16_cap`, plus B5,** a decode rule that honours the
  query plan.
