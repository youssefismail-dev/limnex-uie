# V2-R2 — can a vocabulary memory beat the word tell without copying? Results (Part A)

- **Registration:** `research/v2/PREREG_R2.md` (a75db25). **Code:** `research/v2/v2_r2.py`
  (cccf492), committed before any run; full suite 1,086 passed.
- **Record:** `data/v2/r2/`: `vocab.json`, `oracle.json`, `A_D1b.json`, `A_D4.json`,
  `A_copy.json`, `A_result.json`, `readA.log`, `wrapcheck.json`.
- **Post-hoc diagnosis:** `research/v2/v2_r2_diag.py` (4135f92), committed before it ran,
  output in `data/v2/r2/diag/`. **It is not a registered reading and decides nothing by
  itself.**
- **Read once, 2026-10-09.**
- **Part B did not run under R2's rule** (§3). It was registered separately as R3
  (`PREREG_R3.md`); its results are in `R3_RESULTS.md`.
- **Unchanged:** `e16_b5` frozen; no training; no GPU.

## In plain words

1. **The honest judge works as intended.** It still convicts `e16_b5`, by about as much as
   before: benign paths +0.172, phishing paths +0.086, benign hosts +0.058, phishing hosts
   +0.102.
2. **A pure copier passes the word test outright.** Pasting real training paths into
   `e16_b5`'s URLs clears benign paths (+0.010). This is the hard number behind the rule
   that copying guards can never be dropped.
3. **A word memory used blindly is only part of the answer.**
   - Filling every invented word with a real rare word closes 59% of the benign path tell.
   - Keeping invention at the data's own rate, it closes 39%.
   - In names it backfires (+0.058 → +0.202), because real names are coined, not dictionary
     words.
   - It does nothing for phishing.
4. **What the tell really is** (post-hoc diagnosis):
   - In every class and field, the engine is short of real words that are rare but recur. It
     produces 55–75% of the real rate.
   - Benign fills the gap with invented words, which are also longer than real new words.
   - Phishing fills it with common words: it barely over-invents at all.
   - The structure of URLs carries almost none of the tell.

## 1. The honest judge

| | benign | phishing |
|---|---|---|
| training rows, half A / half B | 1,280,452 / 1,274,222 | 109,351 / 129,792 |
| word types, Vocab_A / Vocab_B | 933,950 / 936,301 | 102,811 / 122,672 |
| r_f: half B's words new to Vocab_A (LBL, SUB, PTH, QRY, FRG) | 0.668, 0.028, 0.112, 0.130, 0.109 | 0.420, 0.419, 0.224, 0.470, 0.151 |
| u_f: `e16_b5`'s words new to Vocab_A | 0.758, 0.030, 0.244, 0.178, 0.333 | 0.402, 0.406, 0.199, 0.465, 0.270 |
| free-word rate e_f = min(1, r/u) | 0.881, 0.920, 0.460, 0.732, 0.326 | 1.0, 1.0, 1.0, 1.0, 0.559 |

- **V-split PASS:** every training row is in exactly one half.
- **By the data's own measure, benign over-invents mostly in paths** (2.2× the data's rate)
  **and queries** (1.4×). Its names are close (1.13×).
- **Phishing does not over-invent** in any field except fragments.

## 2. Part A: ceilings on existing files (the judge D1b; excess, mean of 5 generations)

| | benign path, query, fragment | phishing path, query, fragment | benign host | phishing host | benign whole URL | phishing whole URL |
|---|---|---|---|---|---|---|
| floor (AUC) | 0.4968 | 0.4936 | 0.5026 | 0.5049 | 0.5036 | 0.4985 |
| **base** (`e16_b5`) | **+0.1720** | **+0.0857** | **+0.0581** | **+0.1019** | **+0.1500** | **+0.1209** |
| O1 slips fixed | +0.1259 (closes 27%) | +0.0822 (4%) | +0.0707 (−22%) | +0.1008 (1%) | +0.1189 (21%) | +0.1157 (4%) |
| O2 memory, all fakes | +0.0713 (**59%**) | +0.1115 (−30%) | **+0.2021 (−248%)** | +0.2030 (−99%) | +0.2115 (−41%) | +0.2196 (−82%) |
| O2e memory + data-rate invention | +0.1057 (**39%**) | +0.0905 (−6%) | +0.0500 (14%, UNDECIDED) | +0.0968 (5%) | +0.0965 (36%) | +0.1207 (0%) |
| **O4 copier** | **+0.0098 (94%, CLEARED)** | +0.1524 (−78%) | — | — | — | — |

- Every verdict is CONVICTED except O4 benign (CLEARED) and O2e benign host (UNDECIDED).
- **Why the phishing copier gets worse:** random training paths come from a different mix of
  pages than real held-out phishing URLs (whole sites, page after page), so it is caught on
  the mix, not on words.
- **Word order (D4, path part):**
  - benign: base +0.0195, O2e +0.0159, O4 +0.0044, all CLEARED;
  - phishing: base +0.0463 and O2e +0.0461 (both UNDECIDED), O4 +0.1582 (CONVICTED).
  - Substitution does not create an order tell.
- **Copying (paths, queries and fragments found exactly in training):**

| | real (natural) | base | O2e | O4 |
|---|---|---|---|---|
| benign | 26.1% | 17.5% | 17.6% | 100% |
| phishing | 59.9% | 46.9% | 46.9% | 100% |

## 3. The rule for Part B: not met

Part B was to run if O2e closed at least half of the base excess on the path part or the host,
in either class. It closed at most 39% (benign path part), so **Part B did not run under R2.**

The decision to test the mechanism anyway is R3: a separate registration, made for the reason
in §4, not a change to this rule.

## 4. Post-hoc diagnosis (not registered; decides nothing by itself)

**Words by how often the judge's vocabulary has seen them** (shares of all words):

| | never | 1–9 | 10–99 | 100–999 | 1,000+ |
|---|---|---|---|---|---|
| benign paths, real | 10.3% | 11.7% | 18.7% | 25.5% | 33.9% |
| benign paths, `e16_b5` | **24.0%** | **5.5%** | **11.7%** | 24.5% | 34.3% |
| benign paths, O2e | 17.9% | 8.0% | 15.2% | 24.7% | 34.3% |
| benign paths, O2 | 12.6% | 10.1% | 18.1% | 24.8% | 34.3% |
| phishing paths, real | 23.2% | 11.4% | 15.2% | 16.2% | 34.0% |
| phishing paths, `e16_b5` | 26.4% | **7.5%** | **11.0%** | **18.4%** | **36.8%** |
| benign hosts, real | 39.1% | 7.3% | 6.4% | 6.5% | 40.7% |
| benign hosts, `e16_b5` | **45.0%** | **3.3%** | 4.5% | 6.2% | 41.0% |
| phishing hosts, real | 24.3% | 9.9% | 11.7% | 11.2% | 42.8% |
| phishing hosts, `e16_b5` | **29.0%** | **5.8%** | 10.2% | 11.7% | 43.4% |

**The judge refitted on one feature group at a time** (AUC, real against base):

| | structural features only | vocabulary features only |
|---|---|---|
| benign paths | 0.526 | **0.657** |
| phishing paths | 0.538 | 0.562 |
| benign hosts | 0.508 | 0.551 |
| phishing hosts | 0.542 | 0.585 |

**The strongest single features** (benign paths, AUC real against base):
- the share of unseen words (0.623), the number of unseen words (0.621);
- the longest unseen word (0.610) and the mean unseen-word length (0.607). The inventions are
  *longer* than real new words;
- the frequency of the URL's rarest word (0.427, that is, 0.573 the other way).

What this shows:
- **The shortfall is the same everywhere:** too few real words in the 1–99 bands. Benign
  paths have 17.2% against 30.4%; phishing paths 18.5% against 26.6%.
- **What fills the gap differs:**
  - benign: invented words, which are longer than real new words;
  - phishing paths: common words. The 100+ bands hold 55.2% against 50.2%, while unseen words
    are only +3 points.
- **The tell is in the words, not the URL's shape.**
- **Why O2e falls short:**
  - O2e's dice favoured one-off words: judging from the band shares, about half of its
    replacements are words the judge has never seen either.
  - Even O2, whose word bands are close to real, keeps +0.071. So *which* rare word goes in
    *which* slot matters, and dice cannot know that.

## 5. Findings

- **F1 — The judge is honest and still convicts.**
  - Splitting the vocabulary leaves the tell about where it was (benign paths +0.172 against
    D1's +0.192).
  - It cannot be passed by copying training *words*.
  - It *can* be passed by copying training *paths* (F2).
- **F2 — A copier passes the word test (benign paths +0.010, CLEARED, 100% copied).** A
  word-level judge alone would reward copying. Copying guards are part of the definition of
  success, not an add-on.
- **F3 — Fixing slips is worth about a quarter** of the benign path tell (27%), and nothing
  in phishing or hosts. This is the ceiling for what word pieces can give through spelling.
- **F4 — A blind word memory helps paths, hurts names, ignores phishing.**
  - Paths: 39–59% closed.
  - Names: much worse, because names are coined.
  - Phishing: no effect, because its tell is not invention.
- **F5 — The root problem is the long tail of real words** (post-hoc).
  - The engine reproduces the commonest words and misses real words that are rare but recur.
  - Benign substitutes inventions, which are also too long. Phishing substitutes common words.

## 6. Limitations

- **L1 — The oracles fill words without context** (dice). They under-state what a memory
  that fits words to context could do, which is why R3 exists.
- **L2 — Shared platforms skew the phishing split.** A whole platform domain falls into one
  half (half A has 109k phishing rows against 129k).
- **L3 — The judge is a little lenient toward `e16_b5`.** `e16_b5` was trained on both halves,
  so words it memorised from half B look known to the judge.
- **L4 — §4 is post-hoc.** Its numbers are descriptive.

## 7. Predictions against outcome (`PREREG_R2.md` §6)

| prediction | outcome |
|---|---|
| base still CONVICTED, close to D1 (benign path part about +0.18) | ✓ +0.172 |
| O1 closes 15–35% | ✓ 27% |
| O2 closes 70–95%; may overshoot | ✗ 59%; no overshoot in paths (it hurt hosts instead) |
| O2e closes 55–85% (benign paths); 40–75% (phishing hosts) | ✗ 39%; ✗ 5% |
| O4 near the floor while copying 100% | ✓ benign (+0.010); ✗ phishing (the page mix differs) |
| Part B runs (~85%) | ✗ the rule was not met (it was then registered separately as R3) |
