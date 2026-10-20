# CR1 — results: the absolute realism of controlled output

**Registered:** `PREREG_CR1.md` (abb1cba), before any CR1 generation or scoring;
code b79854f; S0 001f382. **Generation:** Kaggle Tesla T4, 2026-09-24, 70 files
× 3,000 compliant URLs (14 states × G = 5) in 107 min — E7 (`data/best_e7.pt`,
sha256 `916d0385…`) with presence enforced by **joint resampling**
(`uie.presence`, HC2). **Scoring:** laptop CPU, 2026-09-24 12:11–15:45 (3.56 h
of scoring), below-normal priority, peak 0.74 GB, memory guard never triggered.
E5, R4, gate v2, its thresholds and every published result — E7's, HC1's and
HC2's included — are unchanged. Outputs: `data/cr1/`.

## The answer

The question: for each control state, how realistic are the compliant
controlled URLs against real URLs **in that same state** — the absolute excess
x on every applicable field — and is each one safe under the gate's rule?

**No state is safe on every field.** The 14 measurable states have 59
applicable (state, field) cells:

| verdict | cells |
|---|---:|
| **CLEARED** — safe | 23 |
| UNDECIDED — not shown safe | 9 (1 of them pooled) |
| **CONVICTED** — a tell | 23 (1 of them a pooled TELL) |
| not computable — too few real URLs | 4 |

"Safe" is the gate's CLEARED: the whole 95% interval below 0.05. A point
estimate below 0.05 is not enough, and seven cells show why: x < 0.05 but
UNDECIDED, e.g. benign P:1 PTH, x = +0.0428, 95% CI [+0.0334, +0.0522].

The example that motivated the question, phishing with a path and no query
(`<P:1> <Q:0>`): LBL +0.0686, SUF +0.0754, SUB +0.0804, PTH +0.0731, **all four
CONVICTED**.

| state | LBL | SUF | SUB | PTH | QRY | FULL (aux) |
|---|---|---|---|---|---|---:|
| benign P:1 | +0.0084 CLEARED | +0.0100 CLEARED | +0.0091 CLEARED | +0.0428 UNDECIDED | pooled +0.1047 **TELL** | +0.0360 |
| benign P:0 | +0.0063 CLEARED | +0.0641 **CONVICTED** | +0.0300 CLEARED | n/a | pooled +0.0903 UNDECIDED | +0.4869 |
| benign Q:1 | +0.0227 CLEARED | +0.0563 UNDECIDED | +0.0627 **CONVICTED** | +0.1103 **CONVICTED** | +0.1041 **CONVICTED** | +0.2322 |
| benign Q:0 | +0.0043 CLEARED | +0.0222 CLEARED | +0.0042 CLEARED | +0.0427 UNDECIDED | n/a | +0.2178 |
| phishing P:1 | +0.0608 **CONVICTED** | +0.0695 **CONVICTED** | +0.0725 **CONVICTED** | +0.0679 **CONVICTED** | not computable | +0.1012 |
| phishing P:0 | +0.0201 CLEARED | +0.0350 CLEARED | +0.0397 UNDECIDED | n/a | not computable | +0.5008 |
| phishing Q:1 | +0.1592 **CONVICTED** | +0.1084 **CONVICTED** | +0.1541 **CONVICTED** | not computable | +0.1558 **CONVICTED** | +0.3338 |
| phishing Q:0 | +0.0183 CLEARED | +0.0097 CLEARED | +0.0304 CLEARED | +0.0454 UNDECIDED | n/a | +0.4073 |
| benign P:1 Q:1 | +0.0327 CLEARED | +0.0735 **CONVICTED** | +0.0732 **CONVICTED** | +0.1174 **CONVICTED** | +0.1044 **CONVICTED** | +0.1422 |
| benign P:1 Q:0 | +0.0002 CLEARED | +0.0177 CLEARED | +0.0032 CLEARED | +0.0446 UNDECIDED | n/a | +0.0479 |
| benign P:0 Q:1 | +0.0129 CLEARED | +0.0370 UNDECIDED | not computable | n/a | +0.0975 **CONVICTED** | +0.4883 |
| benign P:0 Q:0 | +0.0109 CLEARED | +0.0660 **CONVICTED** | +0.0225 CLEARED | n/a | n/a | +0.4928 |
| phishing P:1 Q:1 | not measurable (registered) | | | | | |
| phishing P:1 Q:0 | +0.0686 **CONVICTED** | +0.0754 **CONVICTED** | +0.0804 **CONVICTED** | +0.0731 **CONVICTED** | n/a | +0.1045 |
| phishing P:0 Q:1 | not measurable (registered) | | | | | |
| phishing P:0 Q:0 | +0.0311 CLEARED | +0.0365 UNDECIDED | +0.0330 CLEARED | n/a | n/a | +0.4991 |

x is the excess AUC over the real-against-real floor, the mean of five
generation means (G = 5 × D = 10, gate v2's estimator), judged by gate v2's
section-26.1 test against the 0.05 bar. **n/a**: the state excludes the field.
**pooled**: the gate's pooled fallback, which can convict and never clear.
**not computable**: too few real URLs, even pooled (section 5). **FULL** is
auxiliary and never gates; in the five path-off states it is saturated by one
character (section 4).

## 1. Verification — passed before any CR1 number was read

| check | requirement | result |
|---|---|---|
| **S0** | the scorer, given gate v2's own inputs, reproduces the gate v2 report for E7 phishing exactly | **PASS** — 73 values (001f382) |
| files | the 70 registered files, 3,000 URLs each, none for the two excluded states; hashes and seeds as the provenance records; the laptop's checkpoint and suffix table | **PASS** |
| states | every URL in its state, as the gate reads it | **PASS** — 210,000 / 210,000 (the scorer asserts it again) |
| identity | a CPU regeneration of the first pass equals the Kaggle GPU files, four states | **302/302, 267/267, 316/316, 307/307** |

## 2. What the table shows — descriptive

- **Where a control forbids a field, the name fields mostly clear.** LBL, SUF
  and SUB all clear in benign Q:0, phishing Q:0 and benign P:1 Q:0; the one
  open field in each is PTH, UNDECIDED at +0.043 to +0.045. The exceptions are
  benign P:0 and benign P:0 Q:0, where SUF convicts (+0.064, +0.066), and
  phishing P:0 and P:0 Q:0, which each leave one name field UNDECIDED (SUB
  +0.040, SUF +0.037).
- **Where a control asks for a field, most fields convict.** Phishing P:1,
  P:1 Q:0 and Q:1 convict on every computable field (+0.061 to +0.159). Benign
  Q:1 and P:1 Q:1 convict on SUB and on the structure they carry (PTH +0.110
  and +0.117, QRY +0.104 in both), P:1 Q:1 on SUF as well. Benign P:1 is the
  exception: its names clear and PTH is UNDECIDED at +0.043.
- **The query is not realistic anywhere it can be measured.** QRY convicts in
  all four states that require it (+0.098 to +0.156) and, pooled, in benign P:1,
  where it is left free (+0.105). Benign P:0's pooled QRY is +0.090 — UNDECIDED
  over the rule's 10 draws, TELL on the first 3.
- **The minority state convicts; the mixture does not.** Phishing's name fields
  clear in Q:0 (+0.010 to +0.030), a mix of URLs with and without a path, but
  convict in its path-on part, P:1 Q:0 (+0.069 to +0.080). Most real phishing
  hosts have no path (P:1 is 9,388 of 24,081); most benign hosts have one (P:0
  is 31,245 of 113,277), and there it is benign P:0 whose SUF convicts. Names
  that do not follow the control would show exactly this: the class mix is
  mostly the majority state's, so it is wrong mainly in the minority state. For
  SUF that is measured in section 3; for LBL and SUB it is **Unknown. More
  evidence required.**

## 3. Why SUF convicts: the suffix is drawn without the control

**Code.** `generate_with_suffixes` draws each URL's suffix from the
class-marginal table by label alone — `table.sample(label, n, rng)`
(`uie/suffixes.py:151`) — before the URL is generated. The control reaches the
model, never the table. Under joint resampling the TLD mix can therefore move
only through rejections.

**Mathematics.** Rejection keeps a draw with probability h(suffix); the kept mix
is p(s)·h(s)/a, a being the acceptance rate, and its total variation from the
table is at most (1 − a)/a. Phishing `<P:1>` accepted 97.9% of first draws, so
its TLD mix can sit at most **0.021** from the class-average table, whatever the
model does.

**Measurement** (`cr1_suf_mix.py`): TLDs collapsed to the class's top ten real
TLDs plus "other", which keeps the sampling noise small; noise floor = TV
between two disjoint samples of one pool, 40 repeats.

| control flipped | real URLs: TV on vs off | generated: TV on vs off | generated noise floor, mean (max) | first-pass acceptance, on |
|---|---:|---:|---:|---:|
| phishing path | 0.154 | **0.017** | 0.018 (0.032) | 97.9% |
| phishing query | 0.253 | 0.119 | 0.018 (0.029) | 77.9% |
| benign path | 0.025 | 0.009 | 0.014 (0.023) | 99.4% |
| benign query | 0.050 | 0.011 | 0.014 (0.023) | 96.4% |

Every real shift is above its own noise floor's maximum (0.039, 0.105, 0.017,
0.037).

- **Phishing's TLD mix does not follow the path control** — 0.017 is within
  noise, as the bound requires — while real phishing URLs with a path have a
  clearly different mix. Generated `<P:1>` phishing URLs carry the class-average
  mix: TV 0.028 to it, 0.103 to real phishing URLs with a path. That is the SUF
  tell in phishing P:1 and P:1 Q:0.
- The query control rejects 22% of phishing draws and moves the mix about half
  as far as real data (0.119 of 0.253). HC2's design review left this open:
  joint resampling reproduces the real shift only as far as E7's own
  P(field | suffix) carries it. For the phishing query that is about half; for
  the phishing path, none.
- **Benign:** the top ten TLDs cover 39.2% of real benign hosts (phishing:
  76.4%), so this coarse measurement cannot see where benign's TLD differences
  lie. The code argument holds for benign too; how much of benign's SUF tells
  (P:0 +0.064, P:0 Q:0 +0.066, P:1 Q:1 +0.074) it explains is **Unknown. More
  evidence required.**
- Whether phishing's LBL and SUB tells follow from the TLD mix — labels and
  subdomains depend on the suffix, in real and generated URLs alike — or stand
  apart from it is **Unknown. More evidence required.**

## 4. FULL (auxiliary) is saturated wherever the path is off

FULL reads +0.487 to +0.501 in all five path-off states, which is the ceiling of
the measure (AUC 1 against a floor near 0.5). One character accounts for it.

- The gate reads `host/` as having **no path**: its path "/" has an empty body.
  FULL keeps the slash; `strip_scheme` deliberately leaves trailing slashes in.
- Real path-off URLs are written `host/` 98.0–100% of the time. Generated ones
  never are (0.000), because the tokenizer normalises an empty path away by
  design (E3_REPORT section 7.3).
- A single binary feature with rates p_real and p_gen gives, on its own,
  AUC 0.5 + (p_real − p_gen)/2. Here that is +0.490 to +0.500, the whole range
  FULL has. **In these states FULL cannot measure anything else.**

| state | real: root slash (one per host) | generated: root slash | slash alone: (p_real - p_gen) / 2 | FULL x |
|---|---:|---:|---:|---:|
| benign P:1 | 0.000 | 0.000 | +0.000 | +0.0360 |
| benign P:0 | 0.982 | 0.000 | +0.491 | +0.4869 |
| benign Q:1 | 0.229 | 0.000 | +0.114 | +0.2322 |
| benign Q:0 | 0.270 | 0.000 | +0.135 | +0.2178 |
| phishing P:1 | 0.000 | 0.000 | +0.000 | +0.1012 |
| phishing P:0 | 1.000 | 0.000 | +0.500 | +0.5008 |
| phishing Q:1 | 0.362 | 0.001 | +0.180 | +0.3338 |
| phishing Q:0 | 0.620 | 0.000 | +0.310 | +0.4073 |
| benign P:1 Q:1 | 0.000 | 0.000 | +0.000 | +0.1422 |
| benign P:1 Q:0 | 0.000 | 0.000 | +0.000 | +0.0479 |
| benign P:0 Q:1 | 0.998 | 0.000 | +0.499 | +0.4883 |
| benign P:0 Q:0 | 0.980 | 0.000 | +0.490 | +0.4928 |
| phishing P:1 Q:0 | 0.000 | 0.000 | +0.000 | +0.1045 |
| phishing P:0 Q:0 | 1.000 | 0.000 | +0.500 | +0.4991 |

In the states where the path is free (Q:1, Q:0) the slash alone supplies
+0.114 to +0.310 and FULL reads more (+0.218 to +0.407), so FULL also sees other
differences there. How much each part contributes cannot be separated this way:
AUC contributions do not add. Where a path is required, neither side can carry
a root slash, so FULL is an ordinary whole-URL reading: benign P:1 +0.036,
benign P:1 Q:0 +0.048, phishing P:1 +0.101, phishing P:1 Q:0 +0.105, benign
P:1 Q:1 +0.142. FULL never gates, and nothing here changes a verdict. This
count does not revisit E3's inconclusive intervention on the unconditional
FULL regression (E3_REPORT section 13.4), which asked a different question.

## 5. Blanks, and the pooled rows

- **n/a**: the state excludes the field (PTH under P:0, QRY under Q:0). Both
  sides are empty, and the gate's function prints NOT COMPUTABLE for them.
- **not computable**: the field is allowed, but the real pool cannot supply
  three disjoint samples of 250 (the gate's MIN_N), even pooled. All four cases
  follow from the counts registered in the pre-registration's section 1:
  - phishing P:1 QRY and phishing Q:1 PTH: the 669 real phishing URLs with both
    a query and a path;
  - phishing P:0 QRY: the 399 with a query and no path;
  - benign P:0 Q:1 SUB: 669 of that state's 1,802 real URLs have a subdomain.
    That it equals phishing's 669 is a coincidence, confirmed by counting both
    directly.
- **pooled**: too few generated URLs carry the field in each generation, so the
  gate's pooled fallback runs over all 15,000. It can convict, never clear. Its
  verdict is the D = 10 rule; the first-3-draw verdict is reported alongside, as
  gate v2 does.
- **not measurable**: phishing P:1 Q:1 and P:0 Q:1, registered before
  generation; neither generated nor scored.

| state | field | per generation (real / generated values) | pooled x | min | max | n | D=10 | D=3 |
|---|---|---|---:|---:|---:|---:|---|---|
| benign P:1 | QRY | 5451 / 159 | +0.1047 | +0.0759 | +0.1468 | 756 | **TELL** | TELL |
| benign P:0 | QRY | 1802 / 69 | +0.0903 | +0.0122 | +0.1653 | 332 | UNDECIDED | TELL |
| phishing P:1 | QRY | 669 / 109 | still insufficient (real 669, pooled generated 598) | | | | NOT COMPUTABLE | -- |
| phishing P:0 | QRY | 399 / 35 | still insufficient (real 399, pooled generated 155) | | | | NOT COMPUTABLE | -- |
| phishing Q:1 | PTH | 669 / 1568 | still insufficient (real 669, pooled generated 7784) | | | | NOT COMPUTABLE | -- |
| benign P:0 Q:1 | SUB | 669 / 977 | still insufficient (real 669, pooled generated 4868) | | | | NOT COMPUTABLE | -- |

## 6. Reduced n

Most cells run at the gate's n = 1,200 per draw. The exceptions:
- phishing Q:1 at 356 (SUB 263);
- benign P:0 Q:1 at 600;
- phishing Q:0 PTH at 890;
- SUB in seven benign states, at 948 to 1,196.

NS found, descriptively and short of its registered standard, that an existing
tell reads *lower* at smaller n (`research/qry/NSWEEP_RESULTS.md`, section 2).
So a conviction at reduced n is, if anything, conservative. A clearance at
reduced n is weaker evidence than one at 1,200. The clearances affected are:
- benign P:0 Q:1 LBL at n = 600 (+0.0129, far from the bar);
- five benign SUB cells at 948 to 1,185.

## 7. Every cell

| state | field | x | SE | t vs bar | 95% CI | n | generation means | verdict |
|---|---|---:|---:|---:|---|---:|---|---|
| benign P:1 | LBL | +0.0084 | 0.0014 | -28.85 | [+0.0045, +0.0123] | 1200 | 0.0059 0.0115 0.0041 0.0093 0.0110 | CLEARED |
| benign P:1 | SUF | +0.0100 | 0.0045 | -8.89 | [-0.0025, +0.0225] | 1200 | 0.0065 0.0210 -0.0044 0.0091 0.0180 | CLEARED |
| benign P:1 | SUB | +0.0091 | 0.0031 | -13.31 | [+0.0005, +0.0177] | 1185 | 0.0091 0.0124 0.0098 -0.0022 0.0162 | CLEARED |
| benign P:1 | PTH | +0.0428 | 0.0034 | -2.13 | [+0.0334, +0.0522] | 1200 | 0.0495 0.0374 0.0462 0.0325 0.0487 | UNDECIDED |
| benign P:1 | QRY | -- | -- | -- | -- | real 5451, generated 159 per generation | -- | NOT COMPUTABLE |
| benign P:1 | FULL | +0.0360 | 0.0023 | -6.09 | [+0.0296, +0.0424] | 1200 | 0.0319 0.0345 0.0426 0.0310 0.0401 | AUXILIARY |
| benign P:0 | LBL | +0.0063 | 0.0049 | -8.91 | [-0.0073, +0.0199] | 1200 | 0.0127 -0.0082 -0.0028 0.0159 0.0139 | CLEARED |
| benign P:0 | SUF | +0.0641 | 0.0020 | +6.98 | [+0.0585, +0.0697] | 1200 | 0.0640 0.0580 0.0632 0.0646 0.0707 | **CONVICTED** |
| benign P:0 | SUB | +0.0300 | 0.0047 | -4.23 | [+0.0170, +0.0430] | 948 | 0.0355 0.0227 0.0386 0.0382 0.0151 | CLEARED |
| benign P:0 | QRY | -- | -- | -- | -- | real 1802, generated 69 per generation | -- | NOT COMPUTABLE |
| benign P:0 | FULL | +0.4869 | 0.0022 | +198.03 | [+0.4808, +0.4930] | 1200 | 0.4881 0.4793 0.4925 0.4853 0.4891 | AUXILIARY |
| benign Q:1 | LBL | +0.0227 | 0.0025 | -11.10 | [+0.0158, +0.0296] | 1200 | 0.0251 0.0306 0.0218 0.0196 0.0162 | CLEARED |
| benign Q:1 | SUF | +0.0563 | 0.0043 | +1.47 | [+0.0444, +0.0682] | 1200 | 0.0456 0.0515 0.0518 0.0645 0.0681 | UNDECIDED |
| benign Q:1 | SUB | +0.0627 | 0.0013 | +10.05 | [+0.0591, +0.0663] | 1196 | 0.0629 0.0668 0.0618 0.0632 0.0589 | **CONVICTED** |
| benign Q:1 | PTH | +0.1103 | 0.0042 | +14.38 | [+0.0986, +0.1220] | 1200 | 0.1224 0.1178 0.1059 0.0999 0.1057 | **CONVICTED** |
| benign Q:1 | QRY | +0.1041 | 0.0048 | +11.17 | [+0.0908, +0.1174] | 1200 | 0.0924 0.1194 0.0950 0.1059 0.1076 | **CONVICTED** |
| benign Q:1 | FULL | +0.2322 | 0.0027 | +66.83 | [+0.2247, +0.2397] | 1200 | 0.2250 0.2271 0.2379 0.2384 0.2326 | AUXILIARY |
| benign Q:0 | LBL | +0.0043 | 0.0029 | -15.71 | [-0.0038, +0.0124] | 1200 | 0.0122 0.0054 0.0067 0.0028 -0.0055 | CLEARED |
| benign Q:0 | SUF | +0.0222 | 0.0058 | -4.82 | [+0.0061, +0.0383] | 1200 | 0.0055 0.0289 0.0304 0.0348 0.0116 | CLEARED |
| benign Q:0 | SUB | +0.0042 | 0.0011 | -40.54 | [+0.0011, +0.0073] | 1113 | 0.0068 0.0048 0.0055 0.0036 0.0002 | CLEARED |
| benign Q:0 | PTH | +0.0427 | 0.0040 | -1.80 | [+0.0316, +0.0538] | 1200 | 0.0383 0.0550 0.0490 0.0329 0.0384 | UNDECIDED |
| benign Q:0 | FULL | +0.2178 | 0.0034 | +49.92 | [+0.2084, +0.2272] | 1200 | 0.2245 0.2227 0.2069 0.2220 0.2131 | AUXILIARY |
| phishing P:1 | LBL | +0.0608 | 0.0030 | +3.58 | [+0.0525, +0.0691] | 1200 | 0.0550 0.0586 0.0698 0.0657 0.0547 | **CONVICTED** |
| phishing P:1 | SUF | +0.0695 | 0.0039 | +4.99 | [+0.0587, +0.0803] | 1200 | 0.0641 0.0792 0.0577 0.0760 0.0703 | **CONVICTED** |
| phishing P:1 | SUB | +0.0725 | 0.0059 | +3.84 | [+0.0561, +0.0889] | 1200 | 0.0880 0.0574 0.0835 0.0628 0.0709 | **CONVICTED** |
| phishing P:1 | PTH | +0.0679 | 0.0035 | +5.17 | [+0.0582, +0.0776] | 1200 | 0.0688 0.0774 0.0568 0.0718 0.0648 | **CONVICTED** |
| phishing P:1 | QRY | -- | -- | -- | -- | real 669, generated 109 per generation | -- | NOT COMPUTABLE |
| phishing P:1 | FULL | +0.1012 | 0.0027 | +18.87 | [+0.0937, +0.1087] | 1200 | 0.1010 0.0989 0.0926 0.1081 0.1057 | AUXILIARY |
| phishing P:0 | LBL | +0.0201 | 0.0058 | -5.13 | [+0.0040, +0.0362] | 1200 | 0.0249 0.0302 0.0327 0.0094 0.0034 | CLEARED |
| phishing P:0 | SUF | +0.0350 | 0.0041 | -3.69 | [+0.0236, +0.0464] | 1200 | 0.0269 0.0493 0.0386 0.0310 0.0290 | CLEARED |
| phishing P:0 | SUB | +0.0397 | 0.0042 | -2.45 | [+0.0280, +0.0514] | 1200 | 0.0483 0.0471 0.0387 0.0248 0.0398 | UNDECIDED |
| phishing P:0 | QRY | -- | -- | -- | -- | real 399, generated 35 per generation | -- | NOT COMPUTABLE |
| phishing P:0 | FULL | +0.5008 | 0.0021 | +217.25 | [+0.4950, +0.5066] | 1200 | 0.4981 0.4999 0.4988 0.4982 0.5090 | AUXILIARY |
| phishing Q:1 | LBL | +0.1592 | 0.0022 | +48.91 | [+0.1531, +0.1653] | 356 | 0.1667 0.1586 0.1542 0.1610 0.1553 | **CONVICTED** |
| phishing Q:1 | SUF | +0.1084 | 0.0045 | +13.11 | [+0.0959, +0.1209] | 356 | 0.1177 0.0976 0.1175 0.0982 0.1110 | **CONVICTED** |
| phishing Q:1 | SUB | +0.1541 | 0.0045 | +23.23 | [+0.1416, +0.1666] | 263 | 0.1513 0.1501 0.1438 0.1548 0.1706 | **CONVICTED** |
| phishing Q:1 | PTH | -- | -- | -- | -- | real 669, generated 1568 per generation | -- | NOT COMPUTABLE |
| phishing Q:1 | QRY | +0.1558 | 0.0045 | +23.50 | [+0.1433, +0.1683] | 356 | 0.1598 0.1590 0.1550 0.1660 0.1392 | **CONVICTED** |
| phishing Q:1 | FULL | +0.3338 | 0.0046 | +61.26 | [+0.3210, +0.3466] | 385 | 0.3232 0.3388 0.3251 0.3485 0.3331 | AUXILIARY |
| phishing Q:0 | LBL | +0.0183 | 0.0029 | -10.87 | [+0.0102, +0.0264] | 1200 | 0.0116 0.0288 0.0172 0.0192 0.0149 | CLEARED |
| phishing Q:0 | SUF | +0.0097 | 0.0032 | -12.65 | [+0.0008, +0.0186] | 1200 | 0.0133 0.0034 0.0196 0.0097 0.0024 | CLEARED |
| phishing Q:0 | SUB | +0.0304 | 0.0031 | -6.32 | [+0.0218, +0.0390] | 1200 | 0.0238 0.0259 0.0416 0.0313 0.0296 | CLEARED |
| phishing Q:0 | PTH | +0.0454 | 0.0029 | -1.60 | [+0.0373, +0.0535] | 890 | 0.0505 0.0489 0.0360 0.0411 0.0504 | UNDECIDED |
| phishing Q:0 | FULL | +0.4073 | 0.0011 | +328.52 | [+0.4042, +0.4104] | 1200 | 0.4061 0.4073 0.4059 0.4115 0.4058 | AUXILIARY |
| benign P:1 Q:1 | LBL | +0.0327 | 0.0029 | -5.94 | [+0.0246, +0.0408] | 1200 | 0.0311 0.0441 0.0281 0.0309 0.0293 | CLEARED |
| benign P:1 Q:1 | SUF | +0.0735 | 0.0020 | +11.58 | [+0.0679, +0.0791] | 1200 | 0.0688 0.0731 0.0786 0.0695 0.0777 | **CONVICTED** |
| benign P:1 Q:1 | SUB | +0.0732 | 0.0045 | +5.12 | [+0.0607, +0.0857] | 974 | 0.0887 0.0609 0.0707 0.0755 0.0702 | **CONVICTED** |
| benign P:1 Q:1 | PTH | +0.1174 | 0.0057 | +11.88 | [+0.1016, +0.1332] | 1200 | 0.1096 0.1209 0.1382 0.1096 0.1086 | **CONVICTED** |
| benign P:1 Q:1 | QRY | +0.1044 | 0.0026 | +20.76 | [+0.0972, +0.1116] | 1200 | 0.1052 0.0949 0.1050 0.1056 0.1111 | **CONVICTED** |
| benign P:1 Q:1 | FULL | +0.1422 | 0.0032 | +28.93 | [+0.1333, +0.1511] | 1200 | 0.1346 0.1396 0.1492 0.1372 0.1502 | AUXILIARY |
| benign P:1 Q:0 | LBL | +0.0002 | 0.0030 | -16.34 | [-0.0081, +0.0085] | 1200 | -0.0054 -0.0053 0.0024 0.0110 -0.0016 | CLEARED |
| benign P:1 Q:0 | SUF | +0.0177 | 0.0026 | -12.59 | [+0.0105, +0.0249] | 1200 | 0.0199 0.0124 0.0266 0.0136 0.0160 | CLEARED |
| benign P:1 Q:0 | SUB | +0.0032 | 0.0018 | -25.69 | [-0.0018, +0.0082] | 1178 | 0.0088 -0.0005 0.0058 0.0026 -0.0007 | CLEARED |
| benign P:1 Q:0 | PTH | +0.0446 | 0.0038 | -1.45 | [+0.0340, +0.0552] | 1200 | 0.0403 0.0499 0.0519 0.0318 0.0490 | UNDECIDED |
| benign P:1 Q:0 | FULL | +0.0479 | 0.0068 | -0.31 | [+0.0290, +0.0668] | 1200 | 0.0361 0.0698 0.0492 0.0315 0.0530 | AUXILIARY |
| benign P:0 Q:1 | LBL | +0.0129 | 0.0060 | -6.22 | [-0.0038, +0.0296] | 600 | 0.0221 -0.0011 0.0053 0.0312 0.0069 | CLEARED |
| benign P:0 Q:1 | SUF | +0.0370 | 0.0058 | -2.23 | [+0.0209, +0.0531] | 600 | 0.0600 0.0343 0.0309 0.0279 0.0319 | UNDECIDED |
| benign P:0 Q:1 | SUB | -- | -- | -- | -- | real 669, generated 977 per generation | -- | NOT COMPUTABLE |
| benign P:0 Q:1 | QRY | +0.0975 | 0.0010 | +47.48 | [+0.0947, +0.1003] | 600 | 0.0980 0.0990 0.1000 0.0959 0.0945 | **CONVICTED** |
| benign P:0 Q:1 | FULL | +0.4883 | 0.0040 | +108.26 | [+0.4772, +0.4994] | 721 | 0.4941 0.4809 0.4895 0.4994 0.4776 | AUXILIARY |
| benign P:0 Q:0 | LBL | +0.0109 | 0.0028 | -14.16 | [+0.0031, +0.0187] | 1200 | 0.0152 0.0086 0.0189 0.0033 0.0084 | CLEARED |
| benign P:0 Q:0 | SUF | +0.0660 | 0.0028 | +5.61 | [+0.0582, +0.0738] | 1200 | 0.0679 0.0581 0.0706 0.0606 0.0728 | **CONVICTED** |
| benign P:0 Q:0 | SUB | +0.0225 | 0.0036 | -7.65 | [+0.0125, +0.0325] | 961 | 0.0104 0.0248 0.0325 0.0242 0.0206 | CLEARED |
| benign P:0 Q:0 | FULL | +0.4928 | 0.0029 | +154.64 | [+0.4847, +0.5009] | 1200 | 0.4973 0.5007 0.4916 0.4901 0.4843 | AUXILIARY |
| phishing P:1 Q:0 | LBL | +0.0686 | 0.0030 | +6.16 | [+0.0603, +0.0769] | 1200 | 0.0676 0.0739 0.0595 0.0765 0.0655 | **CONVICTED** |
| phishing P:1 Q:0 | SUF | +0.0754 | 0.0045 | +5.70 | [+0.0629, +0.0879] | 1200 | 0.0634 0.0863 0.0845 0.0747 0.0682 | **CONVICTED** |
| phishing P:1 Q:0 | SUB | +0.0804 | 0.0036 | +8.43 | [+0.0704, +0.0904] | 1200 | 0.0802 0.0895 0.0690 0.0769 0.0864 | **CONVICTED** |
| phishing P:1 Q:0 | PTH | +0.0731 | 0.0021 | +10.86 | [+0.0673, +0.0789] | 1200 | 0.0723 0.0760 0.0678 0.0698 0.0796 | **CONVICTED** |
| phishing P:1 Q:0 | FULL | +0.1045 | 0.0015 | +37.08 | [+0.1003, +0.1087] | 1200 | 0.1064 0.1084 0.1015 0.1006 0.1055 | AUXILIARY |
| phishing P:0 Q:0 | LBL | +0.0311 | 0.0022 | -8.77 | [+0.0250, +0.0372] | 1200 | 0.0359 0.0346 0.0267 0.0253 0.0331 | CLEARED |
| phishing P:0 Q:0 | SUF | +0.0365 | 0.0053 | -2.54 | [+0.0218, +0.0512] | 1200 | 0.0292 0.0422 0.0549 0.0281 0.0282 | UNDECIDED |
| phishing P:0 Q:0 | SUB | +0.0330 | 0.0051 | -3.35 | [+0.0188, +0.0472] | 1200 | 0.0474 0.0422 0.0270 0.0201 0.0285 | CLEARED |
| phishing P:0 Q:0 | FULL | +0.4991 | 0.0027 | +166.80 | [+0.4916, +0.5066] | 1200 | 0.5009 0.4935 0.5049 0.4920 0.5043 | AUXILIARY |

95% CI = x ± t_crit · SE, t_crit = 2.776 (df 4).

## What this means

- **Presence control is exact (HC2); realism under it is not.** Judged against
  real URLs in its own state, no controlled state clears every field.
- **Closest to safe:** benign Q:0, phishing Q:0 and benign P:1 Q:0 — the names
  cleared, PTH UNDECIDED at +0.043 to +0.045 — and benign P:1, whose names clear
  and whose PTH is UNDECIDED at +0.043, but whose free query convicts (pooled).
- **Furthest:** every state that asks phishing for a path or a query, and every
  benign state with a query.
- **One cause is visible in the code:** the suffix is drawn before the control
  applies and without it. For phishing's path control the TLD mix cannot move
  more than 0.021 and does not move at all within noise. The other tells have no
  established mechanism. **Unknown. More evidence required.**
- **CR1 is descriptive** (pre-registration section 4). It changes no
  threshold, verdict or published number, and adopts nothing. E5 stays
  production. Whether and how to use controlled generation is the owner's
  decision; acting on section 3 would need its own registration.

## Reproduce

    python research/cr1/cr1_verify.py --gens data/cr1/gen --checkpoint data/best_e7.pt --bundle data/bundle2
    python research/cr1/cr1_score.py --gens data/cr1/gen --bundle data/bundle2 --out data/cr1
    python research/cr1/cr1_slash.py --gens data/cr1/gen --bundle data/bundle2 \
        --report data/cr1/cr1_report.json --out data/cr1/slash.json
    python research/cr1/cr1_suf_mix.py --gens data/cr1/gen --bundle data/bundle2 --out data/cr1/suf_mix.json
    python research/cr1/cr1_tables.py data/cr1/cr1_report.json data/cr1/slash.json

The generations come from `research/cr1/CR1_generate.ipynb` on Kaggle (steps in
`CR1_KAGGLE.md`). Their provenance, with every file's seed, hash and resampling
statistics, is in `data/cr1/gen/PROVENANCE.json`.
