# V2-P2 — results: NO EFFECT WITH COST, both arms (word pieces made the words worse)

Registration: `research/v2/PREREG_V2_P2.md` (registered 217dc23; Amendments 1–3; gate B
28fafe5). Code run: 22867b4. Bundle `uie-v2p2-code.zip` 8d18d953…. Read once, 2026-10-14
23:23 (`v2_p2_compare.py` → `data/v2/p2/compare.json`). `e16_b5` stays the baseline; nothing
is adopted.

| arm | reading (tiered, binding) | strict |
|---|---|---|
| `p2_a` (rest pieces) | **NO EFFECT WITH COST (G1, G2, G3, G6, G7, G9, G-D4, C1)** | NO EFFECT WITH COST (… + W1) |
| `p2_b` (rest + host pieces) | **NO EFFECT WITH COST (G1, G2, G3, G6, G7, G9, G-D4, G-D4-H, C1, G-VAL, H-TYPO)** | the same |

## In plain words

- **The test was clean.** Every validity check passed: training, generation, the scorer and
  the word judges. This is a trustworthy result about the idea, not a broken run.
- **The answer is no, and clearly so.** Word pieces did not make the engine's words more
  real. They made them much **less** real. The judges' tell on paths and queries roughly
  doubled in both arms (benign 0.17 → 0.34 in A, 0.32 in B). Host pieces made hosts worse
  too (phishing 0.10 → 0.20 in B).
- **The cost reached structure as well.** The fields that switched to pieces fit the data
  worse, and the E3 gate's path and query cells went from CLEARED to CONVICTED. Word order,
  which was never a tell before, became one.
- **My forecast was wrong.** I gave A about 10% for SUCCESS and 40% for PARTIAL; the
  measured direction is the opposite of what I predicted, with t-values near 40.
- **The likeliest reason (post hoc, decides nothing):** the arms kept `e16_cap`'s 20,000
  steps but had to learn 32,000–48,000 new pieces from scratch. They were still improving
  when the learning-rate schedule ran out; `e16_cap` had stopped improving by step 8,000.
  This record cannot tell "pieces are worse here" from "pieces need much more training".

## 1. Validity: PASS at every step

| check | p2_a | p2_b |
|---|---|---|
| V-gen-equiv (before Kaggle) | PASS 1,355 / 1,355 | (shared) |
| Kaggle start check: the arm start peels off to `caca0107…` | PASS | PASS |
| memory check | 2 parts (9.3 of 14.6 GB) | 2 parts |
| training | 3.64 h, peak 6.66 GB, clipping after step 1,000: 0% | 3.48 h, peak 6.44 GB, 0% |
| VERIFY_TRAIN (full, with the name gap, 23:05 and 23:15) | PASS | PASS |
| VERIFY_GENERATE: V-gen | 1,355 / 1,355 identical | 1,355 / 1,355 identical |
| V-rules, from the tokens (D1/D1′, B5, pieces, plan) | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| V4 (the scorer reproduces `e12_final`) | max difference 0 | max difference 0 |
| V-D1b (`e16_b5` and the copier reproduce R2) | PASS | PASS |
| V-floor (all 22 judges within 0.03 of 0.5) | none flagged | none flagged |

- Each arm produced 180,000 URLs (50 files). A had 2 rows cut at max length and B had 4.
- Reported, not deciding: the URL strings show 1 row (A) and 25 rows (B) whose path does not
  match its plan, against `e16_b5`'s 5. B's 25 come from a `?` written letter by letter inside
  a host (the host table holds no `?`, `/` or `#`). The token records show the plan was
  honoured.

## 2. The criteria (§8): K1 and K2 fail in every class

No view is VOID: `e16_b5` is CONVICTED on every deciding view, and no floor is flagged.
Values are excess AUC (0 = indistinguishable from real); the clear line is 0.05.

| view | class | `e16_b5` | `p2_a` | `p2_b` | copier control (R) |
|---|---|---|---|---|---|
| **D1b-R** (A's pass mark) | benign | +0.1720 | **+0.3354** | +0.3210 | +0.0098 CLEARED |
| | phishing | +0.0857 | **+0.2582** | +0.2478 | +0.1524 |
| D1b-H | benign | +0.0581 | +0.0823 | +0.1317 | |
| | phishing | +0.1019 | +0.1036 | +0.2017 | |
| **D1b-W** (B's pass mark) | benign | +0.1500 | +0.2845 | **+0.2887** | |
| | phishing | +0.1209 | +0.1827 | **+0.2541** | |

- Every arm value is CONVICTED. Against `e16_b5`, paired by generation, A's D1b-R is up by
  +0.163 (t 40.9) benign and +0.173 (t 32.3) phishing; B's D1b-W is up by +0.139 (t 65.5)
  and +0.133 (t 57.6). K2 needs a 30% cut; both arms moved the other way.
- **The blind-spot judge (D1b-x)** reads the same in every cell, so the result does not
  depend on the judge's vocabulary.
- **B's host gain** (B against A on D1b-H) is +0.049 benign and +0.098 phishing: host pieces
  made hosts **worse**. Not recorded (it needs a fall).
- **A's specificity** (A against `e16_b5` on D1b-H): it holds in phishing (+0.002, t 0.5) and
  fails in benign (+0.024, t 8.7). A changed only the rest fields, yet benign hosts got
  slightly worse.
- **Other views:**
  - D1-R and D1-W agree with D1b.
  - D0-W went from CLEARED/UNDECIDED to CONVICTED.
  - **D4-R (word order)** went from +0.020 CLEARED (benign) and +0.046 UNDECIDED (phishing)
    to +0.13 to +0.17 CONVICTED in both arms.

## 3. The guards (§9)

| guard | p2_a | p2_b | what failed |
|---|---|---|---|
| G1 | FAIL | FAIL | free PTH and QRY cells, both classes, went from CLEARED to UNDECIDED or CONVICTED (e.g. benign QRY +0.016 → +0.140 in A); B also fails phishing SUB |
| G2 | FAIL | FAIL | PTH cells rise by about +0.05 to +0.11. In B, LBL and SUB rise too (benign LBL +0.004 → +0.039) |
| G3 | FAIL | FAIL | validation NLL per character against `e16_cap`. A: PTH +0.24/+0.18, LBL +0.12 benign. B: PTH +0.25/+0.19, LBL +0.51/+0.21, SUB +0.14/+0.32 (benign/phishing) |
| G6 | FAIL | FAIL | small honouring losses in a few control states (for example A's benign P1Q1, 0.9969 → 0.9949) |
| G7 | FAIL | FAIL | query distance (K12d), both modes. Query *copying* is below the natural rate (A free benign 0.157 vs 0.314) |
| G9 | FAIL | FAIL | free FULL up by +0.023/+0.026 (A) and +0.041/+0.054 (B) |
| G-D4 | FAIL | FAIL | word order on R up by about +0.11 to +0.13 |
| G-D4-H | (reported: pass) | FAIL | host word order, phishing +0.040 |
| C1 | FAIL | FAIL | phishing distinct-word rate |
| G-VAL | pass | FAIL | phishing validity failures 0.68% against a limit of 0.64% (`e16_b5` 0.49%) |
| H-TYPO | (reported: FAIL) | FAIL | more lookalike host words: benign 7.2% → 9.0% in B (real 5.1%) |
| W1 | strict FAIL, tiered pass | pass | |
| G4, G5, G8, G10, G-COPY, G-PAIR, C2, G-DUP | pass | pass | |
| H-RD, H-HOST, H-LIVE, H-LBL1, H-SUB1, H-LBL, H-DUP, H-VAL | (reported: pass) | pass | the anti-memorisation design held: no host copying |

The anti-memorisation design worked as intended. No copy guard failed and no host-copy guard
failed. The engine invented; it just invented worse words.

**E3's finish line, for the record:** NOT MET in both arms. A leaves 9 cells not cleared and
B leaves 15, against `e16_b5`'s 3. No slash row is missed.

## 4. Reported, never deciding

- **Tokens per URL** (as a ratio of the character encoding):
  - A: benign 0.67, phishing 0.86 (its hosts stay letters);
  - B: benign 0.50, phishing 0.54. The prediction for the rest fields alone was 0.30–0.50.
- **The piece rule's removed mass per step:**
  - A: PTH and QRY 0; LBL 0.0009.
  - B: LBL 0.150, SUB 0.126, PTH 0.045.

  B's model put 13–15% of its host-field probability on pieces that the rule had to remove.
  It had not learned which table belongs to which field.
- **The page-row name gap (§5) is uninformative under the name scope.** By design (§4.5),
  page rows do not train name content, so names on page rows are untrained in both arms:
  - A's benign LBL is 12.50 nats/char on trained sites and 12.64 on new sites, against
    `e16_cap`'s 0.60 and 3.23;
  - B's is 4.57 and 4.65.

  The "gap" therefore shrinks, A's from +2.62 to +0.14, for a trivial reason. It says
  nothing about memorisation. I missed this when I registered the prediction ("falls by at
  least half"). Generation uses only the domain frame, where names are trained.
- **Examples of words (benign / phishing):**
  - A: "vodeview", "trungvirtual", "haiclone" / "dekorwoodanceurs", "reservisation",
    "sanorangh".
  - B: "telemindigateur", "pilkonkursphone", "verdiysuridos" / "unverification",
    "dsyzmwqjpnje", "tuvhuknca".

  Many are real fragments glued into non-words.
- **Hosts:**
  - A: "dunesakeephotography.com", "marceladvocatesameadromes.wixsite.com".
  - B: "servercchioeta.ca", "hotite--business-block-33chess.weebly.com".
- **Validation loss by step** (per token, so the level is not comparable across arms; the
  shape is):

  | step | `e16_cap` benign | `p2_a` benign | `p2_b` benign |
  |---|---|---|---|
  | 4,000 | 1.620 | 3.718 | 4.353 |
  | 8,000 | 1.551 | 3.486 | 4.124 |
  | 12,000 | 1.587 | 3.381 | 4.012 |
  | 16,000 | 1.606 | 3.337 | 3.962 |
  | 20,000 | 1.607 | 3.326 | 3.948 |

  `e16_cap` stopped improving by step 8,000. Both arms were still falling at step 12,000–14,000
  and flattened only as the schedule decayed.

## 5. Why (post hoc; decides nothing)

1. **A worse model of the piece fields.** G3 shows it directly: the fields that switched to
   pieces fit worse per character, and only those (SUF is unchanged at ±0.001). The judges and
   the gate see the same weakness from the outside.
2. **The budget.** The arms kept `e16_cap`'s step budget and start, but 32,000 (A) or 48,000
   (B) piece embeddings started from nothing. Each row also takes 0.5–0.86 as many positions,
   so the network computes less per URL. The curves were still falling when the schedule
   ended. The evidence I cited for pieces (Mielke & Eisner 2019) compares models trained to
   convergence, not a fixed fine-tuning budget from a character model.
3. **Gluing.** Word order turned into a tell (D4-R), and the examples are runs of real
   fragments that form no word. With pieces, one sampling step commits to a longer fragment,
   so a weak model's errors become whole wrong chunks rather than single letters.
4. **Host pieces (B) were not learned well enough to help.** That is seen in the 13–15%
   removed mass in host fields and in host tells twice A's.

What this record cannot decide: whether pieces would help with enough training, a
from-scratch run, or a different budget. Testing that needs a new registration and the
owner's decision (§11).

## 6. Predictions [HYPOTHESIS] against the outcome

| prediction (§13) | outcome |
|---|---|
| A benign D1b-R falls 25–60%; CLEARED 10–15% | **rose 95%** (0.172 → 0.335) |
| A phishing D1b-R falls 10–40% | **rose 201%** |
| B phishing D1b-H falls 15–50% | **rose 98%** |
| B's host gain over A recorded | the opposite: B worse than A on hosts |
| the name gap falls by at least half | it fell, trivially (§4) |
| tokens per field 0.30–0.50 (rest); hosts in B 0.35–0.55 | B's whole URL 0.50–0.54; not measured per field |
| host copy guards fail in B about 25% | none failed |
| G1/G2 fail about 25% | both failed in both arms |
| A: SUCCESS 10%, PARTIAL 40%, NO EFFECT 25%, WITH COST 20% | NO EFFECT WITH COST |
| B: SUCCESS 5–8%, PARTIAL 32%, NO EFFECT 25%, WITH COST 30% | NO EFFECT WITH COST |

My forecasts were badly wrong in direction, not just in size.

## 7. Deviations from the registration

1. **VERIFY_TRAIN ran first with `--skip-gap`** (a dry-run flag) so that the owner could start
   generation before leaving.
   - The skipped part is the page-row name gap, which is reported, not deciding.
   - Both arms passed every deciding check in that run.
   - The full check ran later (23:05 and 23:15) and passed. Its reports replace the fast
     ones, which are kept as `VERIFY_TRAIN_skipgap.json`.
   - Generation therefore started before the full check had run. Since it passed, this had
     no consequence.
2. **A low-RAM stop at 21:19.** I let `p2_a`'s misc step and the words samples step start
   together; both load the corpus. The run_job alarm stopped both at 0.2–0.4 GB free.
   - Both were rerun from the start, one job at a time.
   - The half-written samples were moved aside, unused, to
     `data/v2/p2/words_partial_killed_2119`.
   - Every step is seeded, so the results do not depend on this.
3. **Both arms were scored in parallel** on the laptop. This does not affect results.
4. **The training notebooks' encoding step** printed nothing for 35–50 minutes before step
   200. This was not a fault, but the run sheet did not say so.

## 8. What follows (the owner decides; nothing runs before that)

By §11, nothing is retried or tuned without a new registration.
- **Close the word-piece route.** Together with P1 (memory unused) and P1b (memory recalled
  training names), this is the third V2 lever on the word tell that did not move it.
  `e16_b5` stays the baseline.
- **Or test the budget question** before closing it: a longer or from-scratch piece run.
  That needs a new registration and GPU time.

My recommendation and its reasons go to the owner separately.
