# E15-B — results

- **Registration:** `research/e15/PREREG_E15.md` (§§9–13; gate B 74e505c). Bundle
  e48a078 / 637aca12.
- **Read:** 2026-10-06, after every validity check passed.
- **Criteria:** `data/e15/compare_report.json` (`research/e15/e15_compare.py`), against
  the frozen `e12_d1p` (E14's files). K15-2 and G3 are against `e12_final`, as
  registered.

## In plain words

- **Not adopted, on one guard: G5, the slash rows.** Every other criterion passes,
  including the one the bet was about.
- **K15-1 passes. The new data moved the target cell.**
  - Benign control P0Q1 QRY fell from 0.0624 to **0.0410** (−0.0214, past the 0.0141
    bar). It is still UNDECIDED, but it moved.
  - Benign control P1Q1 QRY moved −0.0020: no real change, as with every lever so far.
  - I predicted K15-1 would fail (about a 20% chance of passing, through P0Q1). It
    passed, through P0Q1.
- **No path, subdomain or field cost.**
  - No PTH, SUB or LBL cell rose by more than 0.0141 (the largest rise is +0.0101).
  - No non-query field's validation fit worsened by more than 0.0034 (the limit is
    0.01).
  - Two cells became CLEARED: benign P1Q1 SUB and phishing P1Q0 PTH. Benign P1Q1 PTH
    went from CONVICTED to UNDECIDED.
- **The surprise: free phishing QRY fell from 0.1123 to 0.0774** (−0.035), with no
  phishing rows injected. It is still CONVICTED. This is the largest move this cell
  has made in any experiment (E14's 4× exposure: −0.031).
- **Why G5 fails:**
  - One slash row, benign control `host?query` (P0Q1), has too many slashes: 99.86%
    against the real 99.55% (±0.24). `e12_d1p` was at 99.77%, just inside.
  - The decoder rule itself is exact: every "slash" plan got its slash, and every
    "no slash" plan went without.
  - The miss comes from the 71 plans (of 15,000) that ask for `host?query` with no
    slash. `e15_new` kept the query on 20 of them; `e12_d1p` on 33. The 51 that
    dropped their query leave the host+query pool, which is then almost all slashed.
  - That is 13 URLs, but probably not chance (z ≈ 2.3). It fits the new data:
    real `host?query` URLs nearly always carry the slash.
- **The gain is not attributable to new content** (§10.6's rule).
  - The model writes new-stream-only queries at about the chance rate: P0Q1 2.28%
    against 2.00% for `e12_d1p`, which never saw the stream; P1Q1 0.70% against
    0.55%.
  - What rose was copying of *old* training queries (P0Q1 27.6% → 31.4%), still well
    under the natural rate.
  - So the reading is exposure (and better learning of common queries), not new
    strings. Coverage of the validation templates rose 1–1.6 points.

## 1. Validity: PASS

| check | result |
|---|---|
| VERIFY_TRAIN | e12_final's recipe plus the stream; same start (c7e71edf), rows, dropped rows and C. V-batch: the old rows' digest equals the replay's (adbddc2a). V-mass: 94,787 rows appended (expected 94,820, z −0.11); 2,817,474 query positions (expected 2,822,000, z −0.32); query mass +18.8%. V-stream 637accab. V-mask: tests pass |
| VERIFY_GENERATE | 50 files; byte identity 957/957 (free 669, control 288) |
| V4 | the scorer reproduces `e12_final`'s published cells exactly (difference 0.0) |

## 2. Criteria (`e15_new` against `e12_d1p`)

| criterion | result |
|---|---|
| **K15-1** licensed query cells | **PASS**: P0Q1 0.0624 → 0.0410 (−0.0214); P1Q1 0.0520 → 0.0500 (−0.0020). Neither rises |
| **K15-2** held-out query fit | **PASS**: benign −0.0373 nats/char [−0.0401, −0.0344]; P0Q1 −0.0369, P1Q1 −0.0374. Phishing −0.0053 (interval spans 0). Train–validation gap 0.041 → 0.039 |
| **K15-3** band profile | **PASS**: rare-name share 0.050 [0.044, 0.058] against a null of 0.208 (benign); 0.098 against 0.378 (phishing) |
| G1 no regression | PASS: no cell's verdict worsened with x up by more than 0.0141 |
| G2 paths and subdomains | PASS: largest rise +0.0101 (control benign P0Q0 SUB, still CLEARED) |
| G3 non-query fit | PASS: largest rise +0.0034 (benign LBL, phishing LBL) |
| G4 presence | PASS: free PTH and QRY within T, both classes |
| **G5 slash** | **FAIL**: benign control P0Q1, 0.99859 against 0.99551 (diff +0.0031, T 0.0024). The other 7 rows pass |
| G6 honouring | PASS: no P, Q or S honouring falls by amendment 1's rule (S0 on benign P0Q1: 1.000 → 0.986, n = 71, not significant) |
| G7 copying | PASS: every cell UNDER its own natural rate (§13.2); no distance worsening |
| G8 novelty | PASS: free 0.9948 → 0.9957, control 0.9964 → 0.9969 |
| G9 FULL | PASS: benign −0.0033, phishing +0.0004 |
| G10 training | PASS: clipping within K12e; VERIFY_TRAIN passed |

**Adopted iff all hold: NOT ADOPTED (G5).** No retry and no tuning (§9.8).

**E3 (for the record): NOT MET.** `e15_new`: 5 cells not cleared + 1 slash row.
`e12_d1p`: 6 cells + 0 slash rows.

## 3. Every cell (x, verdict): `e12_d1p` → `e15_new`

| cell | `e12_d1p` | `e15_new` | Δ |
|---|---|---|---|
| free benign LBL | 0.0075 CLEARED | 0.0062 CLEARED | −0.0013 |
| free benign PTH | 0.0028 CLEARED | 0.0049 CLEARED | +0.0021 |
| free benign QRY | 0.0307 CLEARED | 0.0256 CLEARED | −0.0051 |
| free benign SUB | 0.0043 CLEARED | 0.0037 CLEARED | −0.0006 |
| free benign SUF | 0.0142 CLEARED | 0.0166 CLEARED | +0.0024 |
| free phishing LBL | 0.0084 CLEARED | 0.0106 CLEARED | +0.0022 |
| free phishing PTH | 0.0394 CLEARED | 0.0369 **UNDECIDED** | −0.0025 |
| free phishing QRY | 0.1123 CONVICTED | **0.0774** CONVICTED | **−0.0349** |
| free phishing SUB | 0.0168 CLEARED | 0.0193 CLEARED | +0.0025 |
| free phishing SUF | 0.0094 CLEARED | 0.0117 CLEARED | +0.0023 |
| control benign P0Q0 LBL | 0.0132 CLEARED | 0.0091 CLEARED | −0.0041 |
| control benign P0Q0 SUB | 0.0087 CLEARED | 0.0188 CLEARED | +0.0101 |
| control benign P0Q0 SUF | 0.0027 CLEARED | 0.0006 CLEARED | −0.0021 |
| control benign P0Q1 LBL | 0.0163 CLEARED | 0.0182 CLEARED | +0.0019 |
| **control benign P0Q1 QRY** | 0.0624 UNDECIDED | **0.0410** UNDECIDED | **−0.0214** |
| control benign P0Q1 SUF | 0.0102 CLEARED | 0.0089 CLEARED | −0.0013 |
| control benign P1Q0 LBL | 0.0027 CLEARED | −0.0017 CLEARED | −0.0044 |
| control benign P1Q0 PTH | 0.0091 CLEARED | 0.0113 CLEARED | +0.0022 |
| control benign P1Q0 SUB | 0.0025 CLEARED | 0.0024 CLEARED | −0.0001 |
| control benign P1Q0 SUF | 0.0005 CLEARED | 0.0009 CLEARED | +0.0004 |
| control benign P1Q1 LBL | 0.0270 CLEARED | 0.0249 CLEARED | −0.0021 |
| control benign P1Q1 PTH | 0.0575 CONVICTED | 0.0551 **UNDECIDED** | −0.0024 |
| control benign P1Q1 QRY | 0.0520 UNDECIDED | 0.0500 UNDECIDED | −0.0020 |
| control benign P1Q1 SUB | 0.0434 UNDECIDED | 0.0373 **CLEARED** | −0.0061 |
| control benign P1Q1 SUF | 0.0040 CLEARED | 0.0060 CLEARED | +0.0020 |
| control phishing P0Q0 LBL | 0.0199 CLEARED | 0.0198 CLEARED | −0.0001 |
| control phishing P0Q0 SUB | 0.0277 CLEARED | 0.0300 CLEARED | +0.0023 |
| control phishing P0Q0 SUF | −0.0079 CLEARED | −0.0036 CLEARED | +0.0043 |
| control phishing P1Q0 LBL | 0.0252 CLEARED | 0.0247 CLEARED | −0.0005 |
| control phishing P1Q0 PTH | 0.0433 UNDECIDED | 0.0405 **CLEARED** | −0.0028 |
| control phishing P1Q0 SUB | 0.0354 CLEARED | 0.0250 CLEARED | −0.0104 |
| control phishing P1Q0 SUF | −0.0003 CLEARED | −0.0026 CLEARED | −0.0023 |

## 4. The four-way report (§10.6)

| question | evidence | reading |
|---|---|---|
| **Did new content help?** | Uptake (new-stream-only queries written), with `e12_d1p` as the chance reference: P0Q1 2.28% vs 2.00%; P1Q1 0.70% vs 0.55%; free benign 0.95% vs 1.33%. Old-training copies: P0Q1 27.6% → 31.4%, P1Q1 16.2% → 17.9% | uptake is at about the chance rate, so by §10.6's rule **the gain is not attributed to new content** |
| **Was it more exposure?** | +18.8% query mass (+99% in the two strata), verified. E14's +300% everywhere moved P0Q1 −0.026 and free phishing −0.031; E15-B moved them −0.021 and −0.035 | not separable within this arm (registered). The size of the effect at about 1/16 of E14's added mass is reported, not explained |
| **Did coverage improve?** | Validation-template coverage: P0Q1 68.8% → 70.4%, P1Q1 47.6% → 48.8%, free benign 45.2% → 46.2%. Distinct templates per 1,000: 374 → 342, 570 → 562, 642 → 616. Never-seen: 72.4% → 66.3% (P0Q1) | slightly better coverage, slightly less variety: the model writes more of the common real query shapes |
| **Was there a trade-off?** | G2, G3: none (largest cell rise +0.0101; largest field +0.0034). G5: one slash row over by 0.0007 beyond T | **no path or subdomain cost at this dose.** One slash-plan obedience cost, through dropped queries on "no slash" plans |

## 5. Predictions (§13.3) against the outcome

| prediction | outcome |
|---|---|
| P0Q1 QRY 0.054 (0.047–0.061) | **0.0410: outside, better** |
| P1Q1 QRY 0.049 (0.044–0.056) | 0.0500 ✓ |
| free benign QRY 0.029 (0.025–0.034) | 0.0256 ✓ |
| free phishing QRY 0.112 (0.102–0.122), untouched | **0.0774: far outside** |
| P1Q1 SUB 0.044 (0.038–0.050) | 0.0373: just outside, better |
| P1Q1 PTH 0.058 (0.052–0.064) | 0.0551 ✓ |
| phishing P1Q0 PTH 0.044 (0.038–0.050) | 0.0405 ✓ |
| free phishing PTH 0.040, "a flip possible" | 0.0369, flipped to UNDECIDED ✓ |
| K15-1 most likely FAILS | **wrong: PASS** |
| K15-2 passes; benign QRY val −0.02 to −0.05 | ✓ (−0.0373) |
| other fields within ±0.006 | ✓ (max +0.0034) |
| G5 passes, 8 of 8 | **wrong: 7 of 8** |
| uptake 3–8%, well above zero | **wrong: about the chance rate** |
| coverage +1–3 points | ✓ (+1.3, +1.6) |
| no detectable trade-off | ✓ for paths and subdomains |
| overall NOT ADOPTED | ✓, but on G5, not K15-1 |

## 6. What is decided, and what is the owner's

- Under the registered rules, `e15_new` is **NOT ADOPTED**. There is no retry and no
  tuning (§9.8). The baseline stays `e12_d1p`.
- Whether G5's miss should be waived, and what follows, is the owner's decision. The
  facts for it are in §§1–4: the miss is one slash row, by 13 URLs among 71
  "no slash" plans, in the opposite direction to every earlier slash failure (too
  many slashes, not too few).

## 7. The owner's decision (2026-10-06, after this reading)

- **G5 waived, for this adoption only. `e15_new` ADOPTED** as the official baseline,
  replacing `e12_d1p`. Recorded verbatim in `research/native/DECISION_2026-10-06.md`.
- **Unchanged:** G5's rule and T for later experiments, and E3. On `e15_new`, E3 still
  counts the benign control P0Q1 slash row as missed.
- **The next step is evaluated, not registered:** `research/e16/E16_DESIGN.md`.
