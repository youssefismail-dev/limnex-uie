# V2-R1 — the lexical diagnosis: results

- **Registration:** `research/v2/PREREG_R1.md` (d160c6b). **Code:** `research/v2/v2_r1.py`,
  `research/v2/v2_bpe.py` (cbcc213), committed before any run; full suite 1,077 passed.
- **Record:** `data/v2/r1/`: `r1_result.json`, `read.log`, `verify.json`, `meta.json`, the
  detector records (`VD1`, `D1F`, `D4`, `D1S`), `words.json`, `cover.json`, `base.json`, and
  the piece tables `bpe_rest.json` (sha `3d83a6a8…`) and `bpe_host.json` (`42f89af7…`).
- **Read once, 2026-10-09,** after VERIFY passed.
- **Cost:** about 1.5 hours of laptop computing (about 1 hour of wall time, with independent
  parts in parallel). No GPU, no training, nothing generated.
- **Unchanged:** `e16_b5` frozen; `uie/` untouched; E3 closed; P1 not begun.

## In plain words

1. **Where the word problem is.**
   - Benign URLs: mostly in the **path** (+0.186), then the query (+0.102) and the name
     (+0.070). Subdomains are clean (CLEARED).
   - Phishing URLs: the **subdomains** (+0.118), the path (+0.093) and the name (+0.078).
2. **What kind of words go wrong: rare ones, not common ones.**
   - The engine uses common words at the right rate. In benign paths, 40.6% of its words are
     among the 1,000 commonest, against 40.8% in real paths.
   - It loses the long tail. Real benign paths use words outside the 32,000 commonest 19.0%
     of the time; the engine does so 9.1% of the time, and fills the gap with made-up words
     (23.1% of its path words, against 7.6% in real paths).
   - This holds in 5 of the 6 cells that show any shortfall (benign subdomains show none).
     Phishing paths are the exception.
3. **What the fakes are.**
   - In benign paths, the engine overproduces all three kinds: one-letter slips of common
     words, two real words glued together, and outright inventions.
   - Elsewhere, inventions dominate.
4. **Pieces hold common words, not rare ones.**
   - 8,000 pieces hold 73% of the commonest real path words whole, but only 0.3% of the
     words beyond the 32,000 commonest. Even 32,000 pieces hold just 42% of all real benign
     path words.
   - So pieces reach the minority of the benign problem (the slips). They fit phishing paths
     much better.
5. **Word order is not a giveaway today.** A word-sequence detector cannot convict `e16_b5`:
   CLEARED in benign, UNDECIDED in phishing. The engine strings words in realistic orders;
   the problem is which words.
6. **For P1 this means:** the registered rules say GO for both arms. But the evidence says
   pieces alone will most likely *not* reach the gold standard on benign URLs. What is
   missing looks more like a **memory for rare real words** than a better spelling of
   common ones.

---

## 1. Validity: all PASS

| check | result |
|---|---|
| V-D1′ | E18's D1 rerun on E18's cached samples: identical to E18's record (0.5008540694444444 and the `e16_b5` task) |
| V-pool | E18's real selection rebuilt: equal to E18's cached A, B and C in all 10 class × generation cells |
| V-floor | every per-field D1 floor (0.4959–0.5012) and D4 floor (0.4996, 0.5028) is within 0.03 of 0.5 |
| V-D4 | D4 convicts `e9_ln` in both classes (+0.183 benign, +0.378 phishing) |
| V-BPE | 0 losslessness failures across every encoding; the tables used are the ones recorded |
| V-N | N_f set by the rule (§2.1). Phishing QRY at N = 200 is not measurable |
| V-inputs | 38 input files hashed |

## 2. Evidence

### 2.1 R1-1: the word tell, field by field (D1, excess over the floor, mean of 5 generations)

| field | class | N | `e16_b5` | SE | t | verdict | `e15_new` | `e15_new` − `e16_b5` (t) |
|---|---|---|---|---|---|---|---|---|
| LBL | benign | 6,000 | +0.0699 | 0.0042 | +4.79 | CONVICTED | +0.0779 | +0.0080 (1.96) |
| LBL | phishing | 6,000 | +0.0778 | 0.0034 | +8.20 | CONVICTED | +0.0913 | +0.0136 (5.26) |
| SUB | benign | 3,000 | +0.0135 | 0.0055 | −6.66 | **CLEARED** | +0.0170 | +0.0035 (1.14) |
| SUB | phishing | 4,700 | **+0.1177** | 0.0055 | +12.37 | CONVICTED | +0.1352 | +0.0175 (7.28) |
| PTH | benign | 5,200 | **+0.1856** | 0.0039 | +35.04 | CONVICTED | +0.2077 | +0.0221 (7.54) |
| PTH | phishing | 2,300 | +0.0927 | 0.0092 | +4.65 | CONVICTED | +0.1062 | +0.0135 (5.33) |
| QRY | benign | 400 | +0.1019 | 0.0167 | +3.11 | CONVICTED | +0.0919 | −0.0101 (−0.77) |
| QRY | phishing | 200 | — | — | — | not measurable | — | — |

### 2.2 R1-5: word order (D4), with its controls

| source | benign | phishing |
|---|---|---|
| floor (AUC) | 0.4996 | 0.5028 |
| **`e16_b5`** | **+0.0305 (t −4.06), CLEARED** | **+0.0429 (t −0.88), UNDECIDED** |
| `e15_new` | +0.0369, UNDECIDED | +0.0583, UNDECIDED |
| `e9_ln` (positive control) | +0.1825, CONVICTED | +0.3783, CONVICTED |
| chimeras | +0.1465, CONVICTED | +0.0911, CONVICTED |
| shuffled real URLs (D4) | +0.2735, CONVICTED | +0.3728, CONVICTED |
| shuffled real URLs (D1, the check on the control) | +0.0741, CONVICTED | +0.2526, CONVICTED |
| share of URLs the shuffle changed | 67.2% | 78.7% |

### 2.3 R1-2: words by training-frequency band (token shares, real → `e16_b5`)

| field | class | never seen | beyond the 32k commonest | the 1k commonest | S_common | reading |
|---|---|---|---|---|---|---|
| LBL | benign | 63.3% → 73.7% | 17.3% → 8.6% | 5.3% → 5.6% | 0.01 | RARE-WORD SHORTFALL |
| SUB | benign | 2.4% → 3.0% | 2.5% → 1.1% | 92.0% → 93.8% | — | NO SHORTFALL |
| PTH | benign | **7.6% → 23.1%** | **19.0% → 9.1%** | 40.8% → 40.6% | 0.11 | RARE-WORD SHORTFALL |
| QRY | benign | 11.7% → 16.4% | 13.5% → 6.3% | **48.5% → 55.5%** | −1.33 | RARE-WORD SHORTFALL |
| LBL | phishing | 22.7% → 30.8% | 4.1% → 1.5% | 60.3% → 61.9% | 0.27 | RARE-WORD SHORTFALL |
| SUB | phishing | 25.1% → 33.5% | 5.1% → 2.4% | 49.1% → 51.9% | 0.19 | RARE-WORD SHORTFALL |
| PTH | phishing | 10.9% → 18.3% | 2.4% → 1.7% | 69.9% → 70.8% | 0.59 | **COMMON-WORD SHORTFALL** |

- **Pooled by table:** host benign RARE (S −0.03); path benign RARE (0.09); host phishing
  RARE (0.23); path phishing COMMON (0.59).
- **The class vocabularies differ in depth.** In benign, rank 1k / 8k / 32k means a training
  count of 1,039 / 152 / 43. In phishing it means 66 / 8 / 2, so phishing's rare bands are
  thin.

### 2.4 R1-3: what the never-seen words are (near miss / join / other)

| field | class | real | `e16_b5` | the engine's excess per URL: near, join, other |
|---|---|---|---|---|
| LBL | benign | 0.11 / 0.34 / 0.54 | 0.13 / 0.23 / 0.64 | +0.028, **−0.056**, +0.151 |
| SUB | benign | 0.20 / 0.31 / 0.48 | 0.28 / 0.20 / 0.52 | +0.003, −0.002, +0.004 |
| PTH | benign | 0.20 / 0.28 / 0.52 | 0.32 / 0.24 / 0.44 | **+0.194, +0.115, +0.212** |
| QRY | benign | 0.24 / 0.14 / 0.61 | 0.33 / 0.25 / 0.43 | +0.049, +0.052, −0.054 |
| LBL | phishing | 0.08 / 0.07 / 0.85 | 0.09 / 0.06 / 0.85 | +0.010, +0.003, +0.074 |
| SUB | phishing | 0.11 / 0.06 / 0.83 | 0.14 / 0.05 / 0.81 | +0.042, +0.005, +0.138 |
| PTH | phishing | 0.18 / 0.06 / 0.76 | 0.19 / 0.06 / 0.75 | +0.033, +0.009, +0.109 |

- In benign paths, never-seen words per URL: real 0.241, `e16_b5` 0.763 (3.2×).

### 2.5 R1-4: real words held whole by one piece

| table, class | 2k | 4k | 8k | 16k | 32k | characters per piece at 8k |
|---|---|---|---|---|---|---|
| path, benign | 21.6% | 27.0% | 32.0% | 37.0% | 42.4% | 3.19 |
| path, phishing | 36.7% | 46.2% | 54.0% | 60.8% | 66.1% | 2.93 |
| host, benign | 30.1% | 31.7% | 33.8% | 36.1% | 37.9% | 2.91 |
| host, phishing | 40.9% | 47.2% | 51.4% | 55.6% | 58.5% | 3.08 |

**By frequency band** (benign paths, 8k → 32k): the 1k commonest 73.1% → 85.0%; 1k–8k
7.4% → 31.3%; 8k–32k 1.5% → 4.5%; beyond 32k 0.3% → 0.9%; never seen 0%. Phishing paths and
both hosts follow the same pattern: 92–99% of the commonest words are held whole, and almost
no rare ones.

### 2.6 R1-6: reference values for P1's scorecard (mean over the 5 files)

**Exact copies of training values: generated against the natural rate (real held-out)**

| field | benign: real / `e16_b5` / `e15_new` | phishing: real / `e16_b5` / `e15_new` |
|---|---|---|
| LBL | 8.6% / 5.8% / 5.3% | 66.3% / 57.9% / 57.0% |
| SUB | 92.9% / 94.7% / 93.9% | 23.7% / 19.2% / 18.4% |
| PTH | 18.2% / **19.9%** / 19.9% | 63.0% / 49.1% / 46.8% |
| QRY | 23.0% / 22.3% / 20.6% | 41.5% / 24.1% / 19.8% |
| registrable domain | 0.0% / **1.4%** / 0.8% | 63.7% / 56.8% / 55.8% |

- **Strings seen once in training** are copied less than real data copies them, everywhere
  (for example benign paths 1.5% against 2.3%; phishing queries 0.5% against 4.9%). No sign
  of rare-string memorisation.
- **Near misses of frequent values** (share of never-seen values): benign QRY 5.1% against a
  real 3.3%; PTH 1.5% against 0.9%; SUB 17.2% against 13.8%; phishing QRY 2.8% against 1.4%.
- **Repeated values within a file:** benign paths 8.9% against a real 5.4%. Benign labels
  0.1% against a real 2.5%.

**The sample gate (`uie/gate.py`), share of URLs**

| | benign: real / `e16_b5` | phishing: real / `e16_b5` |
|---|---|---|
| memorised (known-URL filter) | 0.02% / 0.14% | 5.52% / 1.23% |
| duplicate in batch | 0.01% / — | 0.10% / 0.07% |
| validity failures (all reasons) | — / 0.11% | — / 0.60% (mostly labels over 63 characters, 0.36%) |
| "active threat" (300-URL snapshot) | — / 0.00% | **38.25% / 38.35%** |

**Distinct words per 1,000 at an equal n (20,000; phishing path 19,145)**

| | real | `e16_b5` | `e15_new` | real against real |
|---|---|---|---|---|
| benign host | 559.5 | **716.0** | 721.3 | 556.9 / 555.2 |
| benign path | 567.3 | 552.4 | 559.6 | 564.3 / 570.0 |
| phishing host | 404.1 | 454.4 | 452.6 | 403.6 / 408.1 |
| phishing path | 290.9 | 352.2 | 352.8 | 291.3 / 292.0 |

---

## 3. Findings

**F1 — The word tell sits in different fields per class.**
- Benign: the path carries the largest tell of any field (+0.186), then the query (+0.102,
  low power) and the label (+0.070). Subdomains are CLEARED (§2.1).
- Phishing: subdomains lead (+0.118), then the path (+0.093) and the label (+0.078).

**F2 — The shortfall is in rare real words, not common ones (5 of the 6 cells that show
one; §2.3).**
- The engine reproduces the commonest words at the real rate.
- It writes far fewer rare-but-real words, and invents in their place.
- **The prediction that failed:** I predicted a common-word shortfall for benign paths
  (~50%). It is rare-word (S 0.11).
- **Benign queries also over-use the commonest words** (55.5% against 48.5%): a mild
  cliché alongside the inventions.
- **Only phishing paths** show a common-word shortfall (S 0.59).

**F3 — The fakes are of every kind in benign paths, mostly inventions elsewhere (§2.4).**
- In benign paths, the excess is split across slips of common words (+0.19 per URL), joins
  (+0.12) and inventions (+0.21). So about 37% of the benign path excess is slips.
- In benign labels, the engine joins real words *less* than real labels do (−0.056 per URL)
  and invents more (+0.151). Real names are often two real words; the engine coins instead.
- In phishing, inventions dominate in every field.

**F4 — Pieces hold common words, not rare ones (§2.5).**
- The commonest words are held whole 73–99% of the time, depending on the table.
- Words beyond the 32,000 commonest are almost never held (0.3–5%), even with 32,000
  merges.
- Coverage keeps rising with table size and does not level off, because the real
  vocabulary is long-tailed.

**F5 — So pieces address the minority of the benign problem.**
- In benign paths, they reach the slips (about 37% of the excess) and some joins.
- They cannot reach the rare-word replacement, which is the larger part.
- They fit phishing paths best: a common-word shortfall, with 54–66% of words held.

**F6 — Word order is not a tell today (§2.2).**
- D4 clears `e16_b5` in benign and is undecided in phishing.
- Meanwhile it catches `e9_ln`, chimeras and shuffled URLs strongly. Within the limit in L1,
  the engine's word sequences look real.

**F7 — Copying and diversity, the baselines P1 is judged against (§2.6).**
- **Copying:** `e16_b5` copies at or below the natural rate in most fields. Benign paths sit
  slightly above it (19.9% against 18.2%), with more repeated paths (8.9% against 5.4%), and
  benign queries sit at it (22.3% against 23.0%). It regenerates a training registrable
  domain in 1.4% of benign URLs; the real natural rate is 0%, by construction.
- **Diversity of host words:** host words are far more diverse than real (benign 716
  against 560 distinct words per 1,000): the inventions.
- **Diversity of path words:** benign path words are slightly *less* diverse than real (552
  against 567) — the common words repeated.

**F8 — Width moved each field's tell by a tenth or less (§2.1).**
- Going from 384 to 512 lowered the other convicted tells by 0.008–0.022. Benign queries did
  not move (−0.010, t −0.77).
- Benign paths moved −0.022, against a gap of about 0.15 to the bar.

## 4. Limitations

- **L1 — The shuffle control was not a clean test of order.** It also moved words such as
  `com` inside hosts, making a structural change that D1 itself catches (+0.074 benign,
  +0.253 phishing). D4's sensitivity to order is shown, but not isolated. P1 needs a cleaner
  control: a shuffle within the path and query only.
- **L2 — Coverage is a proxy.** A word split into two common pieces may still be written
  reliably, so coverage understates what pieces could do.
- **L3 — Sample-based tables.** The tables were learned on 600,000 rows (21% of training
  rows); full-corpus tables would hold somewhat more.
- **L4 — The word definition** (`[a-z]{3,}`) ignores digits, words of 1–2 letters and
  non-Latin scripts.
- **L5 — The bands are thin in phishing** (§2.3): its vocabulary is small.
- **L6 — Power.** Benign queries were read at N = 400 (low power). Phishing queries are not
  measurable.
- **L7 — The live-threat reference is not usable as registered.** The 300-URL snapshot's
  registrable domains are mostly shared hosting platforms, so 38% of *real* held-out
  phishing URLs match as well. P1 needs a different safety measure (§5.3).
- **L8 — Per-field reading.** Each field is read alone; cross-field effects are in R0's
  D1-R and D1-H and in E18's whole-URL D1.
- **L9 — Letter-by-letter engines only.** R1 reads only engines that write letter by
  letter; what pieces actually change is P1's question.

## 5. Implications for P1

### 5.1 What the registered rules decide (`PREREG_R1.md` §5)

| rule | outcome |
|---|---|
| DR1, fields | arm A: PTH, QRY, FRG. Arm B adds LBL and SUB. **Arm B kept** |
| DR2, piece counts | path table **32,000** merges (+16.4M parameters); host table **16,000** (+8.2M) |
| DR3, go or no-go | **GO for both tables**, but each only because of phishing: the path table because phishing paths read COMMON-WORD SHORTFALL, the host table because phishing's 32k coverage is 58.5%. **On benign alone, both tables would be NO-GO** (rare-word shortfall, coverage below 50%) |
| DR4, D4 | valid, and order-sensitive (with L1's caveat): it joins P1's guards |

### 5.2 What the evidence says beyond the rules

1. **The gold standard looks out of reach for P1 as designed.**
   - Benign D1-R must fall by about 81% (R0). The largest part of that tell (benign paths) is
     the replacement of rare real words, and pieces hold almost none of those words.
   - **My revised odds** (judgement, not registered): benign D1-R CLEARED ~5%; phishing
     D1-R ~25%; arm A SUCCESS ~3–5%; arm B (whole-URL D1) ~1–2%.
   - What P1 would most likely show is a **partial gain**: fewer slips in benign paths, a
     real improvement in phishing paths, and little change in benign names and the rare
     tail.
2. **The size rule makes the engine much bigger.**
   - DR2's sizes add +16.4M parameters for arm A (+64%) and +24.6M for arm B (+96%), all in
     lookup rows.
   - A 32,000-way output layer also adds about 65% compute per token.
   - It sits awkwardly with the capacity lock (R3) and with reading the result. The gain
     from 8k to 32k is about 10 points of coverage in benign.
3. **What R1 points to instead: a missing memory for rare real words.** The engine knows the
   common vocabulary; it does not reproduce the long tail, so it invents. Levers that
   address the tail directly:
   - a large lexicon with spelling (parameter-heavy);
   - retrieval or copying from real words (a memory outside the weights);
   - a pretrained prior (kept as a future option, by your decision);
   - more capacity, which the yardstick shows moves this tell by a tenth per step.
   - **None is registered or designed yet.**

### 5.3 For your review: decisions before P1 is registered

1. **Run P1 as designed or redirect it.** P1 as designed tests pieces, accepting a likely
   partial result. A redirected P1 would target the rare-word memory and needs a design
   review first.
2. **If pieces:** accept DR2's sizes (32k / 16k), or cap them (for example 8k / 8k, about 10
   and 2 points less coverage in benign).
3. **A cheap step that would sharpen this choice (not run; it needs your approval).** Two
   laptop "ceiling" readings on existing files, about 1–2 hours:
   - **slips fixed:** every near-miss fake replaced by the common word it misspells, then
     D1-PTH and D1-R re-read. This is the most pieces could gain through slips;
   - **rare words supplied:** every invented word replaced by a real rare word of similar
     length, then re-read. This is the most a rare-word memory could gain.
   - Together they would show which lever can reach the bar, before any GPU time.
4. **Fixes P1 needs either way:**
   - a cleaner word-order control (L1);
   - a safety measure that does not match on shared hosting platforms (L7);
   - path copy guards that start from e16_b5's near-natural copy rate (F7).
5. **Next, after your review:** the revised V2 finish line, set against E3's conditions, and
   the final P1 registration.

## 6. Predictions against outcome

| prediction (`PREREG_R1.md` §6) | outcome |
|---|---|
| LBL CONVICTED in both classes (~85%) | ✓ |
| SUB CONVICTED in phishing (~60%); benign weak or not measurable | ✓ phishing; benign measurable and CLEARED |
| PTH CONVICTED in both classes (~90%) | ✓ |
| QRY benign CONVICTED (~60%); phishing not measurable (~80%) | ✓ ✓ |
| benign path: COMMON-WORD SHORTFALL (~50%) | ✗ RARE-WORD (S 0.11) |
| phishing host: RARE-WORD SHORTFALL (~55%) | ✓ (S 0.23) |
| the excess mostly inventions (~60%); near misses 15–25% of fakes | mixed: inventions dominate in phishing; benign paths split 37% slips / 22% joins / 41% inventions |
| path table 8k coverage 60–75%, 32k 75–85% (benign) | ✗ 32.0%, 42.4% |
| host table 8k 40–55%, 32k 55–70% | ✗ benign (33.8%, 37.9%); ✓ phishing (51.4%, 58.5%) |
| D4 valid (~90%); convicts the shuffled URLs (~60%); convicts `e16_b5` (~90%) | ✓ ✓ ✗ (benign CLEARED, phishing UNDECIDED) |
| copies below the natural rate in every field (~80%) | ✗ benign SUB and PTH slightly above; QRY at it |
| MEMORIZED under 1%; validity failures under 1% | benign ✓; phishing memorised 1.23% ✗; validity ✓ |
| both tables GO (~70%); arm B kept (~85%); 8k for paths (~55%); more for hosts (~50%) | ✓ ✓ ✗ (32k) ✓ (16k) |

My largest misses were the coverage and the common/rare split. Both point the same way: the
real vocabulary's tail is far longer than I assumed.

## 7. Records

| item | location |
|---|---|
| registration | `research/v2/PREREG_R1.md` |
| code and tests | `research/v2/v2_r1.py`, `research/v2/v2_bpe.py`, `tests/test_v2_r1.py`, `tests/test_v2_bpe.py` |
| the full readout | `data/v2/r1/read.log`, `data/v2/r1/r1_result.json` |
| validity | `data/v2/r1/verify.json` |
| the piece tables | `data/v2/r1/bpe_rest.json`, `data/v2/r1/bpe_host.json`, `data/v2/r1/bpe.json` |
