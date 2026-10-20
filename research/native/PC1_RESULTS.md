# PC1 results — why the plan over-predicts benign paths

**The run:**

| step | commit or outcome |
|---|---|
| registered | `PREREG_PC1.md`, 82f28a1, before any code |
| instrument | 948a3ad, tests 8 |
| amendment 1 (V6) | 216b989, before the reading |
| amendment 2 (R4) | after the reading, mechanical |
| read | 2026-09-29, laptop, `e10_plan`'s `best.pt` |

**Not read:** E10_generate's files, `e10_base`, E8.

**Every number below is from the appendix,** `pc1_read.py`'s output, pasted verbatim.

## 0. In plain words

- **The answer is mostly gradient clipping.**
  - At the end of training, every batch of the shortest rows (under 19 scored tokens
    per row) had its gradient clipped. On average its rows counted 0.66 as much as
    everyone else's.
  - Those rows are mostly bare hosts: 88% of benign one-URL-per-site rows without a
    path are short, against 12% of rows with one.
  - So the model learned from bare hosts at about two-thirds weight, and its plan
    says "path" too often.
- **Weighting the data the way training effectively weighted it** gives 76.45%.
  - The plan says 77.10% and the data 73.33%.
  - Clipping accounts for 83% of the overshoot.
  - What is left, 0.65 points, is inside the benign path tolerance (±0.99).
- **It is not the frame, and not rare suffixes.**
  - The overshoot sits in the common suffixes.
  - Once clipping is accounted for, neither frame pulls the other.
- **The plan slot's 9% off-slot mass is harmless.**
  - It is the suffix's own continuation, for example `com` → `com.br`, at the data's
    rate (0.9086 against 0.9100).
  - The generator's renormalisation handles it correctly.
- **The query shortfall is a separate problem.**
  - It sits in "a query after a path".
  - Clipping pushes the other way, toward more queries, so it cannot explain it.
  - Its cause is open.
- **Side finding (amendment 1).** K10a's probe read three suffixes' plans through a
  public-suffix artefact. The effect is under 0.0001, and K10a's reading stands.

## 1. Validity

**PASS**:

| check | result |
|---|---|
| V1 | K10a reproduced exactly (difference 0) |
| V2 | rows that fit 2,545,305 / 238,199, with 10,313 dropped: exactly training's |
| V3 | the targets 0.7333 / 0.0653 and 0.3940 / 0.0451 |
| V4 | 0 |
| V5 | 33.6% of short batches clipped, against the record's 32.7%; none in 30–120 |
| V6 | 0 where the probe parses to the suffix itself (amendment 1) |

**Amendment 2 was made after the reading.**
- R4's rule did not cover six suffixes whose only domain-frame row is longer than
  256 tokens:
  - benign: 0.0002% of the shipped weight;
  - phishing: 0.0004%.
- They are left out, with their weight reported (`PREREG_PC1.md`). The reading
  cannot depend on them.

## 2. R1: clipping — CLIP-LED (benign)

**The clip factors on `best.pt`** (appendix, M2):
- **The shortest batches** (under 19 scored tokens per row): 38 of 38 clipped.
  - Their gradient norms run 1.12–2.16, median 1.52, and their mean weight ω is
    0.656.
- **19–23.5:** 13 of 38 clipped, ω 0.970.
- **Above 23.5:** never clipped; every norm is under 0.9.
- **What it means.** At the end of training, a clip at 1.0 is not catching spikes.
  It is a steady discount on the shortest rows, and the norm falls smoothly with
  batch length.

**Which rows are short** (under 30 scored tokens):

| class, frame | no-path rows | path rows |
|---|---|---|
| benign, domain | 88.0% | 12.0% |
| phishing, domain | 48.8% | 19.3% |

Phishing hosts are longer, so the discount separates path from no-path less there.

**The reading** (domain frame, shipped weights):

| class | plan P | data P | clip-weighted data P | gap | clipping's part | share | residual |
|---|---|---|---|---|---|---|---|
| benign | 77.10% | 73.33% | 76.45% | +3.77 | +3.12 | **0.83** | +0.65 |
| phishing (reported) | 39.83% | 39.40% | 40.85% | +0.43 | +1.45 | — | −1.02 |

- **Benign: CLIP-LED** (share ≥ 2/3).
- **Phishing,** reported only.
  - Clip weighting predicts +1.45 points. The plan sits at +0.43, between the plain
    and the clip-weighted rate.
  - Something else pulls phishing paths down by about a point, for example on `io`,
    `app` and `dev`. PC1 does not explain it.
- **Checkpoint jitter is small.** `final.pt` against `best.pt`: plan P 77.20% against
  77.10% (benign) and 39.98% against 39.83% (phishing).

## 3. R2–R4: frame, where, off-slot mass

- **R2: NOT-NEEDED** (benign).
  - The residual after clipping, +0.65 points, is under 1.0.
  - For the record, on each frame's own training rows the residual is +0.05 in
    log-odds for the domain frame and +0.11 for the page frame. Neither frame pulls
    the other.
- **R3: COMMON-LED** (benign).
  - The 24 common suffixes (58.0% of the weight) carry 0.54 of the gap. The middle
    band carries 0.37, and the 1,134 rare suffixes 0.09.
  - At the top suffixes the clip-weighted rate tracks the plan closely:

    | suffix | data | clip-weighted | plan |
    |---|---|---|---|
    | `com` | 69.7% | 72.4% | 73.0% |
    | `de` | 74.4% | 76.4% | 77.6% |
    | `ru` | 77.4% | 81.9% | 83.4% |
    | `cn` | 63.7% | 69.7% | 70.4% |

- **R4: LEGAL-EXPLAINED** (both classes).
  - The model's legal mass is 0.9086 against the data's 0.9100 (benign) and 0.9527
    against 0.9556 (phishing). The mean difference is 0.006 / 0.005, against the
    registered 0.02.
  - For example `com`: 0.630 against 0.642.

## 4. R5: the query — Q-PATH (both classes)

| class | plan Q | data Q | shortfall | after a path | without a path | path mix | clipping's push |
|---|---|---|---|---|---|---|---|
| benign | 6.05% | 6.53% | −0.48 | −0.49 (1.02 of it) | −0.01 | +0.02 | +0.29 |
| phishing | 4.03% | 4.51% | −0.49 | −0.35 (0.71) | −0.14 (0.29) | +0.00 | +0.21 |

- **The shortfall is in the query-after-path decision.**
- **Clipping pushes queries up,** because query rows are long and never clipped.
  - Against the rate training effectively aimed at, the plan is further short:
    0.886 (benign) and 0.852 (phishing), instead of 0.926 and 0.892 of the plain
    rate.
  - So the plan has a real query deficit that PC1 locates but does not explain.

## 5. Predictions, scored

| prediction (PREREG_PC1 §6) | result | scored |
|---|---|---|
| V1–V6 pass | pass, after amendment 1's cause was found | right, with the amendment |
| R1 CLIP-PARTIAL | CLIP-LED (0.83) | **wrong category, right mechanism.** I assumed a mean short-batch weight near 0.87; the shortest bin measures 0.66 |
| R2 no prediction | NOT-NEEDED | — |
| R3 COMMON-LED | COMMON-LED | right |
| R4 LEGAL-EXPLAINED | LEGAL-EXPLAINED (after amendment 2) | right |
| R5 Q-PATH or Q-SPLIT, not Q-MIX; clipping's push > 0 | Q-PATH; +0.29 / +0.21 | right |
| phishing's clip shift smaller than benign's | +1.45 against +3.12 | right |
| `final.pt` within 0.5 points of `best.pt` | 0.10 / 0.15 | right |

## 6. What the reading licenses (PREREG_PC1 §7)

**CLIP-LED.** An objective-level repair of the clipping interaction becomes a
candidate for every E11 arm, hidden or plan.
- **The repair:** a clip level that no longer binds by length after warm-up.
  - At the end of training the shortest batches reach 2.16, so a level near 3 would
    not bind late.
  - E11's registration fixes the value.
  - **The check:** at most 1% of steps clipped in any length band after warm-up.

**Projections for E11** [HYPOTHESIS]. They assume the plan tracks the weighted rate
one for one, which it does for benign within 0.65 points:

| class | plan path after the repair | against the real rate | plan query after the repair | against the real rate |
|---|---|---|---|---|
| benign | about 73.98% | +0.61, inside ±0.99 | about 5.75% | −0.87, outside ±0.56 |
| phishing | about 38.39% | −1.12, at the edge of ±1.09 | about 3.81% | −0.87, outside ±0.47 |

- **What that means.** The repair should fix benign paths. It does not fix queries;
  it lowers them a little. Queries need their own lever.
- **Plausible but not claimed:**
  - `e9_ln`'s own benign path overshoot (+2.21) shares the cause: its short batches
    were clipped on 24.8% of steps.
  - So does its under-prediction of URLs that end right after the host (QG2-D:
    0.908).
  - E11 can test both.

## 7. Not claimed

- anything about `e10_base`, `e9_ln` or E10_generate's files;
- a fix;
- the cause of the plan's query deficit, or of phishing's −1.02-point residual;
- anything for another training: one checkpoint was read.

## Appendix A — the rendered tables

`python research/native/pc1_read.py`, verbatim.

### Validity

Verdict: **PASS**; failures: none

- V1 benign: max |K10a here − score_plan.json| = 0.0e+00
- V1 phishing: max |K10a here − score_plan.json| = 0.0e+00
- V2: rows that fit {'benign': 2545305, 'phishing': 238199}, dropped 10,313 (training recorded {'benign': 2545305, 'phishing': 238199}, 10,313)
- V3 benign: shipped-weighted domain P 0.7333 (target 0.733), Q 0.0653 (target 0.0653)
- V3 phishing: shipped-weighted domain P 0.3940 (target 0.394), Q 0.0451 (target 0.0451)
- V4: max |len − 3 − Σ field mask| over the sampled rows = 0
- V5: clipped share under 30 scored tokens per row 33.6% of 152 batches (record 32.7%); 30-60: 0.0% of 48; 60-120: 0.0% of 24; 120-inf: — of 0 → **PASS**
- V6 (amendment 1): max |frame-generalised − plan_by_suffix| where the probe parses to the suffix itself = benign 0.0e+00, phishing 0.0e+00
  - benign `bg` (probe parses as `x.bg`, shipped 0.30%): plan P probe 0.8195, training prefix 0.8427
  - benign `sapporo.jp` (probe parses as `x.sapporo.jp`, shipped 0.00%): plan P probe 0.8188, training prefix 0.8219
  - benign `se` (probe parses as `x.se`, shipped 1.41%): plan P probe 0.7609, training prefix 0.7550
  - phishing `bg` (probe parses as `x.bg`, shipped 0.02%): plan P probe 0.2902, training prefix 0.3528
  - phishing `se` (probe parses as `x.se`, shipped 0.04%): plan P probe 0.2420, training prefix 0.3192
- K10a's quantities with the training prefix, benign: P 0.771018, Q 0.060477, legal mass 0.908647 (reading unchanged)
- K10a's quantities with the training prefix, phishing: P 0.398347, Q 0.040274, legal mass 0.952702 (reading unchanged)

### Clip factors on best.pt (M2)

| scored tokens per row | batches | clipped | mean ω | gradient norm: min / median / max |
|---|---|---|---|---|
| 0.0–19.0 | 38 | 38 | 0.6561 | 1.12 / 1.52 / 2.16 |
| 19.0–23.5 | 38 | 13 | 0.9698 | 0.83 / 0.98 / 1.16 |
| 23.5–27.2 | 38 | 0 | 1.0000 | 0.72 / 0.79 / 0.88 |
| 27.2–30.0 | 38 | 0 | 1.0000 | 0.68 / 0.73 / 0.77 |
| 30.0–60.0 | 48 | 0 | 1.0000 | 0.53 / 0.63 / 0.71 |
| 60.0–120.0 | 24 | 0 | 1.0000 | 0.42 / 0.48 / 0.51 |
| 120.0–∞ | 0 | 0 | 1.0000 | — |

Non-finite norms: 0

### Rows under 30 scored tokens (rows that fit)

| class, frame | no-path rows | of them short | path rows | of them short | with userinfo |
|---|---|---|---|---|---|
| benign domain | 250,914 | 88.0% | 653,545 | 12.0% | 1 |
| benign page | 46,610 | 48.6% | 1,594,236 | 5.2% | 0 |
| phishing domain | 117,575 | 48.8% | 74,686 | 19.3% | 4 |
| phishing page | 3,093 | 53.7% | 42,845 | 13.8% | 0 |

### R1 — clipping (domain frame, shipped weights; points)

| class | plan P | data P | clip-weighted data P | gap Δ | clip part C | C / Δ | residual E | reading |
|---|---|---|---|---|---|---|---|---|
| benign | 77.10% | 73.33% | 76.45% | +3.77 | +3.12 | 0.83 | +0.65 | CLIP-LED |
| phishing | 39.83% | 39.40% | 40.85% | +0.43 | +1.45 | 3.36 | -1.02 | CLIP-LED (reported) |

### R2 — each frame on its own training rows (log-odds)

| class | frame | rows | model P | data P | clip-weighted | ε plain | ε residual |
|---|---|---|---|---|---|---|---|
| benign | domain | 904,459 | 76.26% | 72.26% | 75.42% | +0.210 | +0.046 |
| benign | page | 1,640,846 | 97.61% | 97.16% | 97.33% | +0.178 | +0.113 |
| phishing | domain | 192,261 | 39.78% | 38.85% | 40.34% | +0.039 | -0.024 |
| phishing | page | 45,938 | 94.16% | 93.27% | 93.68% | +0.152 | +0.084 |

R2: benign **NOT-NEEDED**; phishing **OTHER** (reported)

### R3 — where the gap sits (domain-frame rows that fit per suffix; shipped weights)

| class | band | suffixes | shipped weight | contribution to Δ | share of Δ |
|---|---|---|---|---|---|
| benign | common | 24 | 58.0% | +2.03 | 0.54 |
| benign | middle | 74 | 30.3% | +1.39 | 0.37 |
| benign | rare | 1134 | 11.7% | +0.35 | 0.09 |
| phishing | common | 4 | 62.3% | -0.48 | -1.11 |
| phishing | middle | 20 | 23.2% | +0.81 | 1.89 |
| phishing | rare | 659 | 14.5% | +0.10 | 0.23 |

R3: benign **COMMON-LED**; phishing **SPREAD** (reported)

### R4 — the plan slot's off-slot mass (domain frame, shipped weights)

| class | model legal mass | data 1 − continuation | mean \|difference\| | left out (amendment 2) | reading |
|---|---|---|---|---|---|
| benign | 0.9086 | 0.9100 | 0.0059 | edu.ec, gob.bo, gov.kh, puglia.it, vt.us (0.0002% weight) | **LEGAL-EXPLAINED** |
| phishing | 0.9527 | 0.9556 | 0.0050 | aws (0.0004% weight) | **LEGAL-EXPLAINED** |

### R5 — the plan's query shortfall (domain frame, shipped weights; points)

| class | plan Q | data Q | Δ_Q | path mix M | no-path K0 | after-path K1 | clip C_Q | reading |
|---|---|---|---|---|---|---|---|---|
| benign | 6.05% | 6.53% | -0.48 | +0.02 (-0.04) | -0.01 (0.02) | -0.49 (1.02) | +0.29 | **Q-PATH** |
| phishing | 4.03% | 4.51% | -0.49 | +0.00 (-0.00) | -0.14 (0.29) | -0.35 (0.71) | +0.21 | **Q-PATH** |

### Reported: final.pt against best.pt (domain frame, shipped weights)

| class | plan P best / final | plan Q best / final |
|---|---|---|
| benign | 77.10% / 77.20% | 6.048% / 6.126% |
| phishing | 39.83% / 39.98% | 4.027% / 4.066% |

### Reported: top 20 shipped suffixes, benign

| suffix | shipped | domain rows | data P | clip-w. P | plan P | page rows | page data P | page plan P | legal mass | 1 − cont |
|---|---|---|---|---|---|---|---|---|---|---|
| com | 12.07% | 118,411 | 69.7% | 72.4% | 73.0% | 188,883 | 97.3% | 97.8% | 0.630 | 0.642 |
| de | 4.67% | 49,899 | 74.4% | 76.4% | 77.6% | 68,920 | 97.2% | 97.6% | 0.934 | 0.928 |
| org | 2.94% | 27,046 | 72.5% | 75.3% | 76.9% | 47,797 | 96.8% | 97.7% | 0.782 | 0.791 |
| nl | 2.85% | 29,351 | 74.2% | 76.9% | 76.7% | 43,341 | 97.4% | 97.9% | 0.999 | 1.000 |
| net | 2.80% | 26,082 | 67.4% | 71.4% | 70.5% | 45,254 | 97.2% | 97.6% | 0.940 | 0.939 |
| ru | 2.69% | 18,135 | 77.4% | 81.9% | 83.4% | 50,346 | 97.6% | 98.0% | 0.996 | 0.998 |
| fr | 2.51% | 24,543 | 74.9% | 77.4% | 78.0% | 39,569 | 96.9% | 97.4% | 0.999 | 1.000 |
| co.uk | 2.46% | 24,711 | 72.2% | 73.3% | 73.2% | 38,021 | 97.3% | 97.5% | 0.999 | 1.000 |
| pl | 2.26% | 17,063 | 78.8% | 81.9% | 83.0% | 40,497 | 97.4% | 97.8% | 0.986 | 0.984 |
| it | 2.08% | 18,821 | 75.0% | 77.1% | 76.8% | 34,150 | 97.5% | 97.6% | 0.999 | 1.000 |
| com.au | 1.79% | 16,011 | 77.5% | 78.3% | 79.8% | 29,557 | 97.6% | 97.8% | 0.999 | 1.000 |
| ca | 1.71% | 17,302 | 76.0% | 78.8% | 80.2% | 26,431 | 97.6% | 97.7% | 0.905 | 0.893 |
| com.br | 1.64% | 13,482 | 76.3% | 77.5% | 79.5% | 28,331 | 97.9% | 97.8% | 0.999 | 1.000 |
| cn | 1.60% | 14,610 | 63.7% | 69.7% | 70.4% | 26,359 | 95.8% | 97.0% | 0.996 | 0.995 |
| ch | 1.57% | 17,938 | 76.2% | 78.9% | 79.3% | 21,993 | 96.6% | 97.2% | 0.974 | 0.969 |
| cz | 1.52% | 12,569 | 75.7% | 78.8% | 79.5% | 26,078 | 97.3% | 97.4% | 0.999 | 1.000 |
| eu | 1.51% | 13,859 | 71.6% | 75.2% | 74.7% | 24,621 | 96.8% | 97.3% | 0.977 | 0.979 |
| be | 1.49% | 14,787 | 75.9% | 78.6% | 79.0% | 23,267 | 97.1% | 97.7% | 0.950 | 0.936 |
| dk | 1.46% | 14,336 | 69.4% | 73.6% | 72.9% | 22,882 | 96.7% | 97.0% | 0.998 | 1.000 |
| se | 1.41% | 13,604 | 69.7% | 73.8% | 75.5% | 22,450 | 97.1% | 97.2% | 0.996 | 0.992 |

### Reported: top 20 shipped suffixes, phishing

| suffix | shipped | domain rows | data P | clip-w. P | plan P | page rows | page data P | page plan P | legal mass | 1 − cont |
|---|---|---|---|---|---|---|---|---|---|---|
| com | 27.86% | 49,813 | 31.9% | 33.1% | 32.2% | 16,248 | 91.9% | 93.2% | 0.911 | 0.917 |
| io | 14.22% | 30,356 | 43.8% | 43.9% | 42.4% | 3,649 | 94.2% | 94.2% | 0.998 | 0.999 |
| app | 10.68% | 19,854 | 33.1% | 33.2% | 31.1% | 5,651 | 92.0% | 92.9% | 0.998 | 1.000 |
| dev | 9.51% | 18,948 | 30.2% | 30.4% | 28.5% | 3,741 | 95.9% | 96.6% | 0.999 | 1.000 |
| top | 4.05% | 9,216 | 63.6% | 69.0% | 69.8% | 477 | 87.8% | 89.4% | 0.999 | 1.000 |
| cc | 2.91% | 5,944 | 46.6% | 53.2% | 53.8% | 1,014 | 93.4% | 96.4% | 0.999 | 1.000 |
| org | 1.66% | 3,096 | 51.0% | 52.4% | 54.5% | 876 | 96.1% | 95.6% | 0.955 | 0.971 |
| xyz | 1.66% | 3,813 | 14.8% | 17.8% | 16.4% | 139 | 84.2% | 88.8% | 0.997 | 1.000 |
| net | 1.54% | 2,758 | 39.1% | 41.2% | 40.9% | 908 | 89.6% | 92.3% | 0.952 | 0.956 |
| vip | 1.19% | 2,654 | 31.1% | 37.4% | 33.3% | 185 | 66.5% | 85.4% | 0.997 | 1.000 |
| shop | 1.06% | 2,380 | 38.3% | 42.7% | 42.1% | 139 | 90.6% | 96.7% | 0.998 | 0.997 |
| me | 1.06% | 1,876 | 48.6% | 49.1% | 50.5% | 641 | 97.0% | 96.2% | 0.957 | 0.957 |
| com.au | 0.98% | 1,920 | 96.5% | 96.5% | 94.8% | 410 | 98.8% | 98.2% | 1.000 | 1.000 |
| cn | 0.87% | 1,652 | 45.4% | 50.0% | 46.0% | 412 | 94.2% | 94.8% | 0.998 | 1.000 |
| icu | 0.80% | 1,853 | 49.9% | 55.2% | 54.3% | 64 | 95.3% | 93.1% | 0.999 | 1.000 |
| cfd | 0.76% | 1,757 | 82.3% | 84.3% | 87.6% | 67 | 95.5% | 97.6% | 0.999 | 1.000 |
| click | 0.72% | 1,528 | 41.9% | 43.9% | 41.8% | 189 | 97.9% | 95.0% | 0.999 | 1.000 |
| site | 0.70% | 1,526 | 19.7% | 19.9% | 18.9% | 141 | 87.9% | 86.6% | 0.995 | 1.000 |
| info | 0.60% | 1,009 | 52.5% | 55.3% | 53.4% | 413 | 95.2% | 94.5% | 0.996 | 0.996 |
| ru | 0.57% | 1,119 | 62.6% | 67.6% | 63.2% | 237 | 88.2% | 96.0% | 0.949 | 0.929 |

### Readings

| class | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| benign | CLIP-LED | NOT-NEEDED | COMMON-LED | LEGAL-EXPLAINED | Q-PATH |
| phishing (R1–R3 reported) | CLIP-LED | OTHER | SPREAD | LEGAL-EXPLAINED | Q-PATH |
