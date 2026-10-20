# C3-QRY — results: does the query control change the queries?

**Registered:** `PREREG_C3_QRY.md` (b4576d2), before any C3 code or generation;
approved by the owner as written. **Code:** 49d0972. Committed before any C3
number was read, none changing a sample, a seed, a threshold or a registered
statistic: 52c926e, 80d6560, 1653c24, 6b489a8 (section 6 below).
**Generation (arm F):** Kaggle Tesla T4, 2026-09-25: 10 files in 110 min, 462,800
draws, at most 5 rounds, no `FreeUnmet`. **Validity:** V1–V6 PASS, 20:08–22:15;
V3 regenerated 387 benign and 274 phishing URLs byte-identically on the laptop
CPU. **Analysis:** laptop CPU, below-normal priority, memory guard: scoring
22:18–22:37 (`cr1_score.py` unchanged, peak 0.71 GB); C3-A 22:37; C3-B
22:37–23:16 (peak 0.49 GB); C3-C 23:17.

**Frozen, and unchanged:** E5, E7, R4's decoder, the delta, `data/bundle2`,
`uie/` (the adopted `uie/controlled.py` and its tables included), gate v2 and
its thresholds, `cr1_score.py`, Q3.1's ladder code, and every published result.
SC1's data was read, never written. Outputs are in `data/c3/`.

## Verdicts

| test | registered rule (section) | result |
|---|---|---|
| **C3-A benign** (primary) | CONSISTENT if the 95% CI of Δ(Q − F) lies in [−0.02, +0.02]; DIFFERENT if \|Δ\| ≥ 0.02 and Holm p < 0.05 (5) | **DIFFERENT**: Δ +0.0201 [+0.0124, +0.0277], Holm p 0.0038. The controlled queries are the more detectable |
| **C3-A phishing** (primary) | the same | **INCONCLUSIVE**: Δ +0.0183 [−0.0083, +0.0450], Holm p 0.128 |
| C3-B, per ladder | SHAPE / VOCABULARY / INTERMEDIATE (5) | benign **SHAPE** in both modes; phishing **INTERMEDIATE** in both modes |
| C3-B, per class | IDENTIFIED / DISTRIBUTED / UNRESOLVED (5) | benign **UNRESOLVED**; phishing **DISTRIBUTED** |
| K1 feasibility | any F request exhausts its 100 rounds (8) | none (at most 5 rounds) |
| K2 instrument | V1–V6 (7) | **PASS** |
| K3 ladder validity | a VOID row; a void ladder (8) | all four ladders valid; `structure_only` VOID (degenerate) in 3 of 4 |
| **Success** | C3-A CONSISTENT or DIFFERENT in both classes, and no ladder void (8) | **NOT MET — PARTIAL**: phishing's C3-A is INCONCLUSIVE |

## 1. C3-A — the control changes benign queries; phishing is unanswered

| class | x_Q (gate verdict) | x_F (gate verdict) | Delta | 95% CI | p | Holm p | pairing | **reading** |
|---|---|---|---|---|---|---|---|---|
| benign | +0.0970 (CONVICTED) | +0.0770 (CONVICTED) | +0.0201 | [+0.0124, +0.0277] | 0.00191 | 0.00383 | paired | **DIFFERENT** |
| phishing | +0.1433 (CONVICTED) | +0.1249 (CONVICTED) | +0.0183 | [-0.0083, +0.0450] | 0.128 | 0.128 | paired | **INCONCLUSIVE** |

Per generation, δ_g = the mean excess of Q minus that of F (g = 0–4):
- benign: +0.0100, +0.0199, +0.0256, +0.0247, +0.0201. All five are positive.
- phishing: +0.0007, +0.0446, −0.0002, +0.0383, +0.0083.

**Benign: DIFFERENT, at the margin.**
- The difference is real: the CI excludes 0 and every generation agrees in sign.
- Its size is about the margin, not clearly beyond it. Δ = 0.020052 clears the
  registered 0.02 by 0.00005, and the CI runs from 0.012 to 0.028.
- The reading is DIFFERENT, as registered. Recomputed independently from the 50
  paired draws; the pairing held exactly, with identical floors in both arms
  (0.502016).

**Phishing: INCONCLUSIVE.**
- The estimate points the same way, but its SE (0.0096) is about twice what
  section 5's power statement assumed from SC1 (0.0033–0.0057).
- Two generations show about +0.04 and three about 0, so the CI covers both no
  effect and twice the margin.
- Section 8: a registered outcome that licenses nothing.

**Neither mode is realistic on its own.** Both arms' QRY is CONVICTED (benign
x_F +0.0770, x_Q +0.0970; phishing +0.1249, +0.1433). The control adds to a tell
that free generation already has.

## 2. The control changes more than the query (descriptive)

Section 5's descriptive contrasts, same scoring. Δ = x_Q − x_F; † marks an
unpaired contrast (n differs):

| class | LBL | SUF | SUB | PTH | FULL |
|---|---|---|---|---|---|
| benign | +0.0013† [-0.0137, +0.0163] | -0.0006† [-0.0238, +0.0226] | +0.0015† [-0.0148, +0.0178] | +0.0494† [+0.0263, +0.0725] | +0.0300 [+0.0244, +0.0356] |
| phishing | +0.0605 [+0.0333, +0.0877] | +0.0088 [-0.0278, +0.0455] | +0.0536† [+0.0336, +0.0736] | n/c | +0.0328 [+0.0197, +0.0459] |

Each arm's own gate reading:

| class | arm | LBL | SUF | SUB | PTH | FULL |
|---|---|---|---|---|---|---|
| benign | Q | +0.0148 CLEARED | -0.0009 CLEARED | +0.0567 UNDECIDED | +0.0964 CONVICTED | +0.2190 AUXILIARY |
| benign | F | +0.0135 CLEARED | -0.0003 CLEARED | +0.0552 UNDECIDED | +0.0470 UNDECIDED | +0.1890 AUXILIARY |
| phishing | Q | +0.1174 CONVICTED | +0.0138 CLEARED | +0.1455 CONVICTED | -- NOT COMPUTABLE | +0.3285 AUXILIARY |
| phishing | F | +0.0569 UNDECIDED | +0.0050 CLEARED | +0.0919 CONVICTED | -- NOT COMPUTABLE | +0.2956 AUXILIARY |

- **SUF is about 0 in both classes**, as the design intends: the suffix mix is
  equal.
- **Benign.**
  - The names do not move: LBL and SUB are within ±0.018.
  - The path and the whole URL are worse under the control: PTH +0.0494
    (unpaired) and FULL +0.0300.
  - Q's PTH is CONVICTED (+0.0964); F's is UNDECIDED (+0.0470).
- **Phishing.**
  - The names are worse under the control: LBL +0.0605, where Q is CONVICTED
    (+0.1174) and F UNDECIDED (+0.0569); and SUB +0.0536 (unpaired).
  - FULL is worse too: +0.0328.

**Path presence among query-bearing URLs** (C3-C, two-proportion 95% CIs):

| class | real | Q | F | Q - F | Q - real | F - real |
|---|---|---|---|---|---|---|
| benign | 0.752 | 0.618 | 0.752 | -0.1337 [-0.1471, -0.1203] | -0.1333 [-0.1459, -0.1207] | +0.0004 [-0.0143, +0.0152] |
| phishing | 0.626 | 0.501 | 0.610 | -0.1094 [-0.1322, -0.0866] | -0.1258 [-0.1559, -0.0957] | -0.0164 [-0.0524, +0.0196] |

- **Free generation reproduces the real rate:**
  - benign: F 0.752, real 0.752 (+0.0004 [−0.0143, +0.0152]);
  - phishing: F 0.610, real 0.626 (−0.0164 [−0.0524, +0.0196]).
- **Under `<Q:1>`, E7 writes a path 13.4 points less often in benign and 10.9
  points less in phishing.** This is E7's own conditional under `<Q:1>`, given a
  query. Arm Q's resampling removes only URLs without one.

So `<Q:1>` shifts decisions made before the query is written: whether to write a
path, and, in phishing, the names.
- **Why: Unknown. More evidence required.**
- **Is the benign QRY difference carried by the path shift?** Unknown, and C3 did
  not test it. Comparing Q and F within path-bearing URLs only would test it.
  That is a new analysis and needs its own registration.
- These are descriptive: no test was registered for them.

## 3. C3-B — where the tell lives

**Benign: SHAPE in both modes; the class reading is UNRESOLVED.**
- `skeleton` keeps 97.6% (Q) and 95.5% (F) of the tell without reducing it. The
  tell is carried by shape, not vocabulary, as E5's was (Q3.1: 1.029).
- Only the parameter count reduces in both modes (`first_param` R_T 0.867 in Q,
  0.799 in F). Q3.1's rule therefore reads UNRESOLVED: no property reaches 50%,
  and fewer than two reduce in both modes.
- The nearest second, name identity (`mask_name_chars`), reduces in Q (0.869) and
  misses in F at t −2.02, against −2.201.

**Phishing: INTERMEDIATE in both modes; the class reading is DISTRIBUTED.**
- `skeleton` removes 29% (Q, R_T 0.707) and 37% (F, 0.631) of the tell, a
  significant reduction. E5's phishing tell, by contrast, `skeleton` left intact
  (PQ1: 0.94).
- Four properties reduce in both modes, none alone to 50%:
  - digit vs alpha (`alnum_flat` 0.568 / 0.501);
  - name identity (0.609 / 0.638);
  - value composition (0.882 / 0.822);
  - the long-value tail (0.921 / 0.912).

**Across the two modes, the ladders are close but not identical:**
- **Phishing:** F's ladder reduces on every rung Q's does (`structure_only` aside,
  VOID in F), and on three more: `clip_values_10`, `first_param` and
  `fixlen_values` (R_T 0.877–0.919).
- **Benign:** both reduce on the parameter count. The value-tail rungs reduce F's
  tell by about 4% and not Q's; name identity reduces Q's (0.869) and not F's
  (0.918).
- The pre-specified `clip15_first` interaction ran in no ladder. In each,
  `clip_values_15` or `first_param` already changed the tell on its own (Q3.1's
  rule).

**Does E5's map transfer?** E5's tell was shape-carried and distributed (Q3.1
benign, PQ1 phishing). E7's matches it in part only:
- **Benign:** the shape part transfers (SHAPE in both modes). The distributed part
  is not established (UNRESOLVED: only the parameter count reduces in both
  modes).
- **Phishing:** the distributed part transfers (DISTRIBUTED in both modes). The
  shape part does not (INTERMEDIATE): `skeleton` removes 29–37% of the tell, so
  part of it is carried by which characters appear, not only by structure and
  character class.

Benign (R_T = excess under the transform / identity's; t against t_crit 2.201):

| rung | component | Q excess | Q R_T | Q t | Q verdict | F excess | F R_T | F t | F verdict | E5 R_T | E5 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| identity | -- | +0.1127 | 1.0 | -- | baseline | +0.0997 | 1.0 | -- | baseline | 1.0 | baseline |
| null | -- | +0.1127 | 1.0 | -- | no change | +0.0997 | 1.0 | -- | no change | 1.0 | no change |
| clip_values_20 | value tail > 20 | +0.1117 | 0.991 | -0.92 | no change | +0.0958 | 0.961 | -4.22 | REDUCES | 0.98 | no change |
| clip_values_15 | value tail > 15 (generated p90) | +0.1131 | 1.003 | 0.3 | no change | +0.0963 | 0.965 | -2.48 | REDUCES | 0.968 | REDUCES |
| clip_values_10 | value tail > 10 | +0.1135 | 1.007 | 0.98 | no change | +0.0959 | 0.961 | -2.36 | REDUCES | 0.987 | no change |
| first_param | parameter count -> 1 | +0.0977 | 0.867 | -5.7 | REDUCES | +0.0797 | 0.799 | -6.63 | REDUCES | 0.89 | REDUCES |
| fixlen_names | name LENGTH (vs mask_name_chars) | +0.1154 | 1.024 | 0.98 | no change | +0.0964 | 0.967 | -1.55 | no change | 1.009 | no change |
| mask_pct | percent-encoding PRESENCE | +0.1136 | 1.007 | 1.21 | no change | +0.0998 | 1.001 | 0.11 | no change | 1.028 | raises |
| normalize_seps | separator structure | +0.1131 | 1.004 | 1.16 | no change | +0.1007 | 1.009 | 2.6 | raises | 1.012 | no change |
| mask_punct | intra-field punctuation | +0.1149 | 1.02 | 2.53 | raises | +0.1017 | 1.02 | 3.61 | raises | 1.024 | raises |
| alnum_flat | digit-vs-alpha distinction (vs skeleton) | +0.1055 | 0.936 | -1.58 | no change | +0.0910 | 0.912 | -1.91 | no change | 0.899 | no change |
| mask_name_chars | param-name identity ONLY | +0.0980 | 0.869 | -3.35 | REDUCES | +0.0916 | 0.918 | -2.02 | no change | 0.826 | REDUCES |
| mask_values | value composition | +0.1261 | 1.119 | 4.3 | raises | +0.1089 | 1.092 | 2.25 | raises | 1.026 | no change |
| fixlen_values | value LENGTH (vs mask_values) | +0.1155 | 1.024 | 2.21 | raises | +0.1007 | 1.01 | 0.46 | no change | 0.986 | no change |
| sort_params | ordering | +0.1139 | 1.01 | 1.06 | no change | +0.1003 | 1.006 | 0.58 | no change | 0.999 | no change |
| skeleton | structure + char class | +0.1100 | 0.976 | -0.43 | no change | +0.0953 | 0.955 | -0.71 | no change | 1.029 | no change |
| structure_only | count + separators + lengths | +0.0912 | 0.809 | -4.73 | VOID degenerate | +0.0872 | 0.875 | -2.4 | VOID degenerate | 0.742 | REDUCES |

Phishing:

| rung | component | Q excess | Q R_T | Q t | Q verdict | F excess | F R_T | F t | F verdict | E5 R_T | E5 verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| identity | -- | +0.1495 | 1.0 | -- | baseline | +0.1371 | 1.0 | -- | baseline | 1.0 | baseline |
| null | -- | +0.1495 | 1.0 | -- | no change | +0.1371 | 1.0 | -- | no change | 1.0 | no change |
| clip_values_20 | value tail > 20 | +0.1377 | 0.921 | -5.2 | REDUCES | +0.1256 | 0.916 | -3.73 | REDUCES | 0.92 | REDUCES |
| clip_values_15 | value tail > 15 (generated p90) | +0.1382 | 0.924 | -3.47 | REDUCES | +0.1258 | 0.917 | -2.72 | REDUCES | 0.938 | REDUCES |
| clip_values_10 | value tail > 10 | +0.1403 | 0.939 | -2.17 | no change | +0.1251 | 0.912 | -2.79 | REDUCES | 0.896 | REDUCES |
| first_param | parameter count -> 1 | +0.1410 | 0.943 | -2.18 | no change | +0.1203 | 0.877 | -3.12 | REDUCES | 0.937 | REDUCES |
| fixlen_names | name LENGTH (vs mask_name_chars) | +0.1429 | 0.956 | -1.29 | no change | +0.1284 | 0.936 | -1.59 | no change | 1.061 | no change |
| mask_pct | percent-encoding PRESENCE | +0.1516 | 1.014 | 1.79 | no change | +0.1384 | 1.009 | 1.1 | no change | 1.017 | raises |
| normalize_seps | separator structure | +0.1499 | 1.003 | 0.17 | no change | +0.1380 | 1.006 | 0.39 | no change | 0.998 | no change |
| mask_punct | intra-field punctuation | +0.1486 | 0.994 | -0.63 | no change | +0.1370 | 0.999 | -0.09 | no change | 0.994 | no change |
| alnum_flat | digit-vs-alpha distinction (vs skeleton) | +0.0849 | 0.568 | -4.23 | REDUCES | +0.0686 | 0.501 | -4.58 | REDUCES | 0.788 | REDUCES |
| mask_name_chars | param-name identity ONLY | +0.0911 | 0.609 | -7.08 | REDUCES | +0.0875 | 0.638 | -4.65 | REDUCES | 0.844 | REDUCES |
| mask_values | value composition | +0.1319 | 0.882 | -2.24 | REDUCES | +0.1127 | 0.822 | -3.5 | REDUCES | 0.888 | REDUCES |
| fixlen_values | value LENGTH (vs mask_values) | +0.1442 | 0.965 | -1.09 | no change | +0.1260 | 0.919 | -2.37 | REDUCES | 0.926 | REDUCES |
| sort_params | ordering | +0.1491 | 0.998 | -0.25 | no change | +0.1365 | 0.995 | -0.42 | no change | 0.999 | no change |
| skeleton | structure + char class | +0.1057 | 0.707 | -4.11 | REDUCES | +0.0866 | 0.631 | -4.22 | REDUCES | 0.94 | no change |
| structure_only | count + separators + lengths | +0.1022 | 0.684 | -3.31 | REDUCES | +0.0823 | 0.6 | -3.13 | VOID degenerate | 0.713 | VOID degenerate |

## 4. C3-C — emission and query shape

| class | F first-pass query rate | round-0 draws | E7, shipped table (HC1 C4) | real |
|---|---|---|---|---|
| benign | 0.0453 | 240,000 | 0.0362 | 0.0640 |
| phishing | 0.0233 | 160,000 | 0.0174 | 0.0444 |

Top-10 suffixes by draws, F's first-pass query rate:

- benign: `com` 0.0312, `de` 0.0319, `cn` 0.0675, `net` 0.0363, `org` 0.0330, `co.kr` 0.2009, `it` 0.0366, `fr` 0.0402, `nl` 0.0256, `ru` 0.0377
- phishing: `com` 0.0230, `dev` 0.0151, `app` 0.0096, `top` 0.0103, `cc` 0.0208, `net` 0.0335, `xyz` 0.0071, `info` 0.0295, `ru` 0.0296, `live` 0.0379

| class | arm | queries | params / query | name length | value p50 | value p90 | %-encoded | numeric values | bare flags |
|---|---|---|---|---|---|---|---|---|---|
| benign | real | 7,253 | 1.786 | 6.73 | 4.0 | 25.0 | 0.0989 | 0.446 | 0.0388 |
| benign | Q | 15,000 | 1.415 | 5.56 | 4.0 | 15.0 | 0.0480 | 0.537 | 0.0461 |
| benign | F | 6,000 | 1.415 | 5.53 | 4.0 | 15.0 | 0.0508 | 0.539 | 0.0430 |
| phishing | real | 1,068 | 1.517 | 9.02 | 9.0 | 50.9 | 0.0552 | 0.278 | 0.1840 |
| phishing | Q | 15,000 | 1.237 | 5.94 | 8.0 | 35.0 | 0.0334 | 0.288 | 0.1776 |
| phishing | F | 2,000 | 1.212 | 5.75 | 8.0 | 34.0 | 0.0345 | 0.328 | 0.1790 |

- **E7's free query rate under the Q:1 table:** 4.53% benign, 2.33% phishing.
  - Higher than under the shipped table (3.62%, 1.74%; HC1 C4). The Q:1 table is
    drawn from query-bearing URLs, and the per-suffix rates vary widely (`co.kr`
    20.1%, `xyz` 0.7%).
  - Lower than real (6.40%, 4.44%).
- **On the registered shape statistics, Q and F are close.** For example, benign
  has 1.415 parameters per query in both, and value p90 15 in both. The clearest
  difference is phishing's numeric-value share, 0.288 against 0.328.
- **Both are far from real:**
  - fewer parameters (benign 1.415 against 1.786);
  - shorter names (5.5 against 6.7);
  - shorter long values (p90 15 against 25);
  - about half the percent-encoding (0.05 against 0.10).
- **So the shape tell is the model's, not the control's.**

## 5. What this licenses (section 8), and nothing more

| finding | licenses |
|---|---|
| benign **DIFFERENT** | the control changes benign queries. Work on E7's free-mode queries must be measured in free mode, or the control's effect addressed first. `<Q:1>` output is **not** a valid stand-in for benign free-mode query content |
| phishing **INCONCLUSIVE** | nothing. The phishing question is unanswered |
| C3-B's readings | which components carry E7's tell in each mode, and that E5's map transfers only in part. Benign: shape-carried, like E5's; parameter count reduces it in both modes. Phishing: distributed, like E5's, over character class, name identity, value composition and the long-value tail; not purely shape-carried. A repair aimed at them needs its own registration |

**Not claimed** (section 10):
- C3 measures E7, not E5.
- It changes no gate, bar, verdict or published number, and adopts nothing.
- Phishing QRY is capped at n = 356 by its real pool.
- Section 2's findings are descriptive.

What follows, if anything, is the owner's decision (section 9, step 6).

## 6. Committed before any number was read

| commit | change | why |
|---|---|---|
| 52c926e | the notebook writes its provenance after every file; a rerun keeps a file only with its own record | a rerun would have lost C3-C's draw counts |
| 80d6560 | K1 recorded as registered in the laptop scripts; the verifier's V3 skips a K1 class | the scripts would have crashed on K1 |
| 1653c24 | C3-B: `gc.collect()` after every discriminator call, as the gate does | Q3.1's ladder grew ~120 MB per call under Python 3.14 and was stopped by the memory guard in the stand-in dry run; values bit-identical |
| 6b489a8 | C3-A contrasts FULL, as section 5 registers; the table renderer | FULL had been left out of the descriptive set |

## 7. Validity (section 7)

| check | result |
|---|---|
| V1 files | pass |
| V2 queries | pass |
| V3 identity | pass -- benign 387/387, phishing 274/274 |
| V4 table | pass |
| V5 arm Q | pass |
| V6 S0''' | pass |
| K1 | none |

## Reproduce

    python research/c3/c3_verify.py --c3 data/c3 --sc1 data/sc1 --bundle data/bundle2 \
        --checkpoint data/best_e7.pt --tables data/state_suffix_tables.json
    python research/c3/c3_score.py --gens data/c3/F/gen --bundle data/bundle2 --out data/c3/F
    python research/c3/c3_compare.py --c3 data/c3 --sc1 data/sc1 --bundle data/bundle2
    python research/c3/c3_ladder.py --c3 data/c3 --sc1 data/sc1
    python research/c3/c3_describe.py --c3 data/c3 --sc1 data/sc1 --bundle data/bundle2
    python research/c3/c3_tables.py --c3 data/c3 --sc1 data/sc1

Arm F was generated by `research/c3/C3_generate.ipynb` on Kaggle, from
`uie-c3-code.zip` (35 files, built at 52c926e) and E7's checkpoint. Arm Q is SC1
arm T's `Q1` files (`data/sc1/T/gen`).
