# SC1 — results: the state-conditioned suffix draw

**Registered:** `PREREG_SC1.md` (2d1db7e), before any SC1 code or generation;
approved by the owner as written. **Code:** 43f8e1e; one crash fix (9ac15cb)
and the table renderer (9d44db3), both committed before any SC1 number was read.
**Generation:** Kaggle Tesla T4, 2026-09-24: 140 files in 217 min, no
`PresenceUnmet`. **Validity:** V1–V6 all PASS. **Scoring:** laptop CPU,
2026-09-25 00:02–06:18, `cr1_score.py` unchanged, below-normal priority, peak
0.76 GB; comparison 06:21.

**Frozen, and unchanged:** E5, E7, R4's decoder, the delta, `data/bundle2` and its
shipped table, `uie/`, gate v2 and every published result, CR1's included.
Outputs are in `data/sc1/`.

## Verdicts

| test | registered rule | result |
|---|---|---|
| **H_SUF** (primary) | Holm-significant Δ(T − C) < 0 **and** T's SUF CLEARED, in phishing P:1, P:1 Q:0 and Q:1 | **SUPPORTED** — 3 of 3 |
| secondary SUF | the same rule in benign P:0, P:0 Q:0 and P:1 Q:1 | **SUPPORTED** — 3 of 3 |
| **H_DOWN against H_INDEP** | H_DOWN if at least 4 of 6 cells show a Holm-significant drop and none is equivalent | **H_DOWN** — 6 of 6 cells, 0 equivalent |
| K1 (hypothesis) | H_SUF refuted with V4 passing | not triggered |
| K2 (harm) | Δ(T − CR1) ≥ +0.02 with CI excluding 0; CLEARED → CONVICTED; pooled UNDECIDED → TELL | **none** |
| K3 (memorisation) | novelty falls ≥ 0.5 points, CI excluding 0 | none (largest fall 0.38 points; section 4) |
| K4 (runaways) | 240+ share rises ≥ 0.2 points, CI excluding 0 | none |
| K5 (cost) | any `PresenceUnmet` | none |
| validity | V1–V6 | **PASS** |
| **Adoption gate (7.5)** | all of the above | **MET — the decision is the owner's** |

## 1. The SUF tells were the suffix draw

## Primary — SUF, T − C (phishing)

| state | CR1 SUF | C SUF | T SUF | Δ(T − C) [95% CI] | p (Holm) | pairing | supported |
|---|---|---|---|---|---:|---|---|
| phishing P:1 | +0.0695 **CONVICTED** | +0.0673 **CONVICTED** | -0.0028 CLEARED | -0.0700 [-0.0814, -0.0587] | 0.00014 | paired | yes |
| phishing P:1 Q:0 | +0.0754 **CONVICTED** | +0.0690 **CONVICTED** | +0.0030 CLEARED | -0.0660 [-0.0789, -0.0532] | 0.00014 | paired | yes |
| phishing Q:1 | +0.1084 **CONVICTED** | +0.1836 **CONVICTED** | +0.0138 CLEARED | -0.1698 [-0.1876, -0.1521] | 3.6e-05 | paired | yes |

## Secondary — SUF, T − C (benign)

| state | CR1 SUF | C SUF | T SUF | Δ(T − C) [95% CI] | p (Holm) | pairing | supported |
|---|---|---|---|---|---:|---|---|
| benign P:0 | +0.0641 **CONVICTED** | +0.0368 CLEARED | +0.0028 CLEARED | -0.0339 [-0.0436, -0.0243] | 0.0012 | paired | yes |
| benign P:0 Q:0 | +0.0660 **CONVICTED** | +0.0343 CLEARED | -0.0033 CLEARED | -0.0377 [-0.0508, -0.0245] | 0.0014 | paired | yes |
| benign P:1 Q:1 | +0.0735 **CONVICTED** | +0.0839 **CONVICTED** | +0.0066 CLEARED | -0.0773 [-0.0910, -0.0636] | 0.00029 | paired | yes |

Every contrast is paired, so both arms were scored on the same real samples and
floors. **The design review predicted these contrasts before any generation:**

| state | predicted Δ(T − C) | measured Δ(T − C) |
|---|---:|---:|
| phishing P:1 | −0.068 | −0.070 |
| phishing P:1 Q:0 | −0.070 | −0.066 |
| phishing Q:1 | −0.161 | −0.170 |
| benign P:0 | −0.028 | −0.034 |
| benign P:0 Q:0 | −0.034 | −0.038 |
| benign P:1 Q:1 | −0.067 | −0.077 |

Prediction P1 held in all 28 arm-states: each arm's SUF lies within ±0.02 of its
own table's ceiling. **T clears SUF in all 14 states.** Arm C behaved as P3
predicted: in phishing Q:1 it read worse than CR1 (+0.075, predicted +0.06),
because held mode removes CR1's joint tilt there.

## 2. H_DOWN: phishing names follow the suffix — but only partly

## H_DOWN against H_INDEP — LBL and SUB, T − C

| state / field | CR1 | C | T | Δ(T − C) [95% CI] | p (Holm) | H_DOWN | H_INDEP |
|---|---|---|---|---|---:|---|---|
| phishing P:1 LBL | +0.0608 **CONVICTED** | +0.0521 UNDECIDED | +0.0324 CLEARED | -0.0196 [-0.0343, -0.0050] | 0.02 | yes | no |
| phishing P:1 SUB | +0.0725 **CONVICTED** | +0.0748 **CONVICTED** | +0.0517 UNDECIDED | -0.0231 [-0.0342, -0.0120] | 0.013 | yes | no |
| phishing P:1 Q:0 LBL | +0.0686 **CONVICTED** | +0.0656 **CONVICTED** | +0.0427 UNDECIDED | -0.0229 [-0.0287, -0.0171] | 0.0019 | yes | no |
| phishing P:1 Q:0 SUB | +0.0804 **CONVICTED** | +0.0717 **CONVICTED** | +0.0486 UNDECIDED | -0.0231 [-0.0311, -0.0151] | 0.0054 | yes | no |
| phishing Q:1 LBL | +0.1592 **CONVICTED** | +0.2015 **CONVICTED** | +0.1174 **CONVICTED** | -0.0841 [-0.0918, -0.0763] | 4.4e-05 | yes | no |
| phishing Q:1 SUB | +0.1541 **CONVICTED** | +0.1685 **CONVICTED** | +0.1455 **CONVICTED** | -0.0231 [-0.0343, -0.0118] | 0.013 | yes | no |

- **H_DOWN holds in all six cells.** Given the right TLD mix, the model's labels
  and subdomains become measurably more realistic. The name tells were, in part,
  the suffix mix.
- **Only part, though.** One cell clears: phishing P:1 LBL.
  - Three move from CONVICTED to UNDECIDED.
  - **Phishing Q:1's names stay convicted:** LBL +0.117, SUB +0.146.

  What carries the remaining tell is **Unknown. More evidence required.**
- The three SUB reductions all read −0.0231. That was checked against the raw
  scorer reports. They are −0.02313, −0.02310 and −0.02305, from different
  per-generation differences and standard errors, at n 1,200, 1,200 and 263. It is
  a coincidence of point estimates, not a computing error.

## 3. The whole picture: arm T against CR1

| arm | CLEARED | UNDECIDED | CONVICTED | not computable | CLEARED on every applicable field |
|---|---:|---:|---:|---:|---|
| CR1 (in use) | 23 | 9 (1 pooled) | 23 (1 pooled TELL) | 4 | none |
| C (control) | 27 | 10 (2 pooled) | 18 | 4 | benign P:0 Q:0, phishing P:0 Q:0 |
| **T** | **36** | 9 (2 pooled) | **10** | 4 | benign P:1 Q:0, benign P:0 Q:0, phishing P:0 Q:0 |

**Three states now clear every applicable field, under CR1's rule.**
- **Benign P:0 Q:0 and phishing P:0 Q:0: through SUF, against the pipeline in
  use.** Their SUF fell from +0.066 and +0.037 to −0.003 and −0.002. The
  domain-frame table alone (arm C) already got both over the bar; the state
  table takes them to about zero.
- **Benign P:1 Q:0: not SC1's doing.** Its PTH clears by 0.0005: x +0.0412,
  upper bound 0.0495. T − CR1 on that cell is −0.003 and not significant; CR1
  read +0.0446, UNDECIDED. That flip is noise at the bar.

## Arm T: every state and field

| state | LBL | SUF | SUB | PTH | QRY | FULL (aux) |
|---|---|---|---|---|---|---:|
| benign P:1 | +0.0069 CLEARED | -0.0032 CLEARED | +0.0136 CLEARED | +0.0426 CLEARED | pooled +0.0707 UNDECIDED | +0.0357 |
| benign P:0 | +0.0026 CLEARED | +0.0028 CLEARED | +0.0172 CLEARED | n/a | pooled +0.0861 UNDECIDED | +0.4868 |
| benign Q:1 | +0.0148 CLEARED | -0.0009 CLEARED | +0.0567 UNDECIDED | +0.0964 **CONVICTED** | +0.0970 **CONVICTED** | +0.2190 |
| benign Q:0 | +0.0037 CLEARED | +0.0051 CLEARED | +0.0050 CLEARED | +0.0488 UNDECIDED | n/a | +0.2219 |
| phishing P:1 | +0.0324 CLEARED | -0.0028 CLEARED | +0.0517 UNDECIDED | +0.0487 UNDECIDED | not computable | +0.0694 |
| phishing P:0 | +0.0165 CLEARED | -0.0091 CLEARED | +0.0270 CLEARED | n/a | not computable | +0.5013 |
| phishing Q:1 | +0.1174 **CONVICTED** | +0.0138 CLEARED | +0.1455 **CONVICTED** | not computable | +0.1433 **CONVICTED** | +0.3285 |
| phishing Q:0 | +0.0169 CLEARED | +0.0022 CLEARED | +0.0369 CLEARED | +0.0575 **CONVICTED** | n/a | +0.4052 |
| benign P:1 Q:1 | +0.0227 CLEARED | +0.0066 CLEARED | +0.0715 **CONVICTED** | +0.0984 **CONVICTED** | +0.0843 **CONVICTED** | +0.1153 |
| benign P:1 Q:0 | +0.0018 CLEARED | -0.0020 CLEARED | +0.0068 CLEARED | +0.0412 CLEARED | n/a | +0.0391 |
| benign P:0 Q:1 | +0.0114 CLEARED | +0.0118 CLEARED | not computable | n/a | +0.0889 **CONVICTED** | +0.4884 |
| benign P:0 Q:0 | +0.0024 CLEARED | -0.0033 CLEARED | +0.0202 CLEARED | n/a | n/a | +0.4929 |
| phishing P:1 Q:1 | not measurable (registered) | | | | | |
| phishing P:1 Q:0 | +0.0427 UNDECIDED | +0.0030 CLEARED | +0.0486 UNDECIDED | +0.0511 UNDECIDED | n/a | +0.0607 |
| phishing P:0 Q:1 | not measurable (registered) | | | | | |
| phishing P:0 Q:0 | +0.0305 CLEARED | -0.0024 CLEARED | +0.0247 CLEARED | n/a | n/a | +0.4996 |

**What SC1 did not touch.** Everything left is outside the suffix:
- **QRY content**, convicted in all four query states it can measure (+0.084 to
  +0.143). This is the model-level shape defect shared with E5 (Q3, PQ1).
- **PTH** in benign Q:1 and P:1 Q:1 (+0.096, +0.098) and phishing Q:0;
  UNDECIDED at +0.049 to +0.051 in phishing P:1 and P:1 Q:0.
- **Phishing Q:1's names.**
- **SUB** in benign P:1 Q:1.

## 4. Safety: no registered harm, and two things to watch

## Safety: T − CR1, every computable cell

| state | LBL | SUF | SUB | PTH | QRY |
|---|---|---|---|---|---|
| benign P:1 | -0.0015 | -0.0132 | +0.0045† | -0.0003 | -- |
| benign P:0 | -0.0037 | -0.0613 | -0.0128† | n/a | -- |
| benign Q:1 | -0.0079 | -0.0572 | -0.0060† | -0.0140 | -0.0070 |
| benign Q:0 | -0.0007 | -0.0171 | +0.0008† | +0.0060 | n/a |
| phishing P:1 | -0.0283 | -0.0722 | -0.0208 | -0.0192 | -- |
| phishing P:0 | -0.0037 | -0.0441 | -0.0128 | n/a | -- |
| phishing Q:1 | -0.0418 | -0.0946 | -0.0086 | -- | -0.0125 |
| phishing Q:0 | -0.0014 | -0.0075 | +0.0065 | +0.0121† | n/a |
| benign P:1 Q:1 | -0.0100 | -0.0669 | -0.0017 | -0.0189 | -0.0200 |
| benign P:1 Q:0 | +0.0016 | -0.0197 | +0.0036† | -0.0034 | n/a |
| benign P:0 Q:1 | -0.0014 | -0.0252 | -- | n/a | -0.0086 |
| benign P:0 Q:0 | -0.0085 | -0.0693 | -0.0023† | n/a | n/a |
| phishing P:1 Q:0 | -0.0258 | -0.0725 | -0.0318 | -0.0220 | n/a |
| phishing P:0 Q:0 | -0.0006 | -0.0389 | -0.0084 | n/a | n/a |

(Δ = x_T − x_CR1; † unpaired, n limited by the generated count; -- not computable
in an arm.) Of the 53 computable cells, 46 moved down and 7 up; none met K2.

**Watch item 1 — PTH in the two Q:0 states.** These are the only rises whose
interval excludes 0:

| cell | T − CR1 | T − C | verdict CR1 → T |
|---|---|---|---|
| benign Q:0 PTH | +0.0060 [+0.0020, +0.0101], paired | −0.0004 [−0.0065, +0.0056] | UNDECIDED → UNDECIDED |
| phishing Q:0 PTH | +0.0121 [+0.0011, +0.0231], unpaired | +0.0158 [−0.0022, +0.0338] | UNDECIDED → **CONVICTED** |

- Both sit below the registered +0.02 margin, and K2 flags CLEARED → CONVICTED,
  not UNDECIDED → CONVICTED. So **neither is a kill**.
- Phishing Q:0 PTH is the only verdict in arm T worse than CR1's.
- The benign rise does not come from the state conditioning (T − C is 0); it
  comes with the domain-frame table and held mode that both new arms share. The
  phishing one cannot be attributed at this precision. **Unknown.**

**Watch item 2 — exact copies in the phishing path states.**
- Novelty falls by 0.37 points in phishing P:1 (99.38% → 99.01%) and by 0.38
  points in P:1 Q:0. Both are significant.
- Memorised URLs rise from 93 to 148 and from 93 to 150 of 15,000.
- Both falls are below K3's 0.5 points, so **not a kill**, but a real rise in
  whole-URL copies.
- Phishing Q:0 moved the other way (260 → 213).
- The registrable-domain copy rate did not rise (phishing P:1 0.497 → 0.473).

## Novelty, runaways, cost (all 15,000 URLs per state and arm)

| state | novelty CR1 / C / T | T − CR1 [95% CI] | 240+ chars CR1 / C / T | first pass CR1 / C / T | draws per URL CR1 / C / T | max rounds C / T |
|---|---|---|---|---|---|---|
| benign P:1 | 0.9999 / 0.9997 / 0.9999 | +0.0000 [-0.0003, +0.0003] | 5 / 10 / 4 | 0.994 / 0.993 / 0.994 | 1.006 / 1.007 / 1.006 | 2 / 1 |
| benign P:0 | 0.9956 / 0.9945 / 0.9947 | -0.0009 [-0.0024, +0.0007] | 0 / 0 / 0 | 0.992 / 0.994 / 0.994 | 1.008 / 1.007 / 1.006 | 2 / 1 |
| benign Q:1 | 0.9998 / 1.0000 / 0.9999 | +0.0001 [-0.0001, +0.0004] | 1 / 2 / 9 | 0.964 / 0.965 / 0.965 | 1.037 / 1.037 / 1.037 | 2 / 2 |
| benign Q:0 | 0.9975 / 0.9983 / 0.9980 | +0.0005 [-0.0005, +0.0016] | 3 / 1 / 2 | 0.990 / 0.989 / 0.990 | 1.010 / 1.011 / 1.010 | 2 / 2 |
| phishing P:1 | 0.9938 / 0.9928 / 0.9901 | -0.0037 [-0.0057, -0.0016] | 3 / 4 / 7 | 0.979 / 0.978 / 0.983 | 1.021 / 1.025 / 1.017 | 10 / 2 |
| phishing P:0 | 0.9826 / 0.9846 / 0.9822 | -0.0004 [-0.0034, +0.0026] | 1 / 4 / 0 | 0.986 / 0.987 / 0.990 | 1.014 / 1.014 / 1.010 | 3 / 2 |
| phishing Q:1 | 0.9999 / 1.0000 / 0.9998 | -0.0001 [-0.0004, +0.0001] | 28 / 34 / 39 | 0.779 / 0.757 / 0.882 | 1.286 / 1.817 / 1.166 | 51 / 34 |
| phishing Q:0 | 0.9827 / 0.9847 / 0.9858 | +0.0031 [+0.0003, +0.0060] | 2 / 3 / 1 | 0.990 / 0.991 / 0.991 | 1.011 / 1.009 / 1.009 | 2 / 2 |
| benign P:1 Q:1 | 0.9999 / 0.9999 / 0.9999 | -0.0001 [-0.0003, +0.0002] | 16 / 9 / 7 | 0.956 / 0.957 / 0.962 | 1.046 / 1.045 / 1.039 | 4 / 3 |
| benign P:1 Q:0 | 0.9999 / 0.9999 / 0.9998 | -0.0001 [-0.0004, +0.0002] | 9 / 9 / 6 | 0.986 / 0.984 / 0.987 | 1.014 / 1.016 / 1.014 | 2 / 3 |
| benign P:0 Q:1 | 0.9999 / 0.9999 / 0.9997 | -0.0002 [-0.0005, +0.0001] | 7 / 4 / 3 | 0.940 / 0.941 / 0.939 | 1.064 / 1.064 / 1.064 | 3 / 3 |
| benign P:0 Q:0 | 0.9943 / 0.9947 / 0.9945 | +0.0002 [-0.0015, +0.0019] | 0 / 0 / 0 | 0.980 / 0.981 / 0.978 | 1.020 / 1.020 / 1.022 | 2 / 2 |
| phishing P:1 Q:0 | 0.9938 / 0.9933 / 0.9900 | -0.0038 [-0.0058, -0.0018] | 5 / 11 / 0 | 0.960 / 0.959 / 0.968 | 1.041 / 1.046 / 1.034 | 9 / 5 |
| phishing P:0 Q:0 | 0.9833 / 0.9850 / 0.9824 | -0.0009 [-0.0039, +0.0020] | 2 / 1 / 0 | 0.978 / 0.980 / 0.985 | 1.023 / 1.022 / 1.015 | 4 / 2 |

## 5. Cost: the state table made the dearest control cheaper

Phishing Q:1, the one state where resampling is expensive:

| arm | first-pass acceptance | draws per URL |
|---|---:|---:|
| CR1 (joint) | 77.9% | 1.29 |
| C (held, class table) | 75.7% | 1.82 |
| T (held, state table) | **88.2%** | **1.17** |

So the state table drew suffixes on which E7 writes queries more readily.
Everywhere else, cost is unchanged within 0.01.

## 6. FULL (auxiliary)

**Path-on states** measure the whole URL, and there FULL fell:

| state | CR1 | T |
|---|---:|---:|
| phishing P:1 | 0.101 | 0.069 |
| phishing P:1 Q:0 | 0.105 | 0.061 |
| benign P:1 Q:1 | 0.142 | 0.115 |

In path-off states it stays saturated near 0.49 by the root slash (CR1 section
4). FULL never gates; no claim is read from it.

## What this means

- **CR1's SUF tells are closed, without retraining.** Drawing the suffix from
  the requested state's domain-frame training distribution removed them in every
  state. The size of each change was predicted before any generation. It was a
  pipeline defect, and it had two measured parts, the state and the frame
  (design review 2.5).
- **Phishing names were partly the suffix mix too (H_DOWN).** The phishing P:1
  label clears. Phishing Q:1's names do not, and **what carries them is
  Unknown**.
- **Under CR1's rule, three states now clear every field**, and two of them
  because of the change. Against CR1: CLEARED cells 23 → 36, CONVICTED 23 → 10;
  of 53 computable cells, 46 fell.
- **The adoption gate is met**, with two watch items that are not kills: PTH
  in the Q:0 states (+0.006 benign, +0.012 phishing, which becomes CONVICTED),
  and whole-URL copies in the phishing path states (+0.37 points).
- **Adoption is the owner's decision** (section 7.5). It would mean controlled
  generation uses the state-conditioned `name_train` tables with held-suffix
  resampling, with the builder moving into `uie/` under its own tested commit.
  E5 stays production; E7's promotion is separate.
- **What the evidence points to next** (the plan's roadmap; each needs its own
  registration, and none was started):
  - **FR1**: the plan's trigger, "if the name tells survive SC1", is met for
    phishing Q:1 (convicted) and for the phishing P:1 states (undecided).
    Phishing Q:1 is costly to test that way (about 57 draws per URL).
  - **C3-QRY**: query content is now the largest remaining tell, and it is
    shared with E5.

## Reproduce

    python research/sc1/build_tables.py --bundle data/bundle2 --out data/sc1/tables.json
    python research/sc1/sc1_verify.py --root data/sc1 --checkpoint data/best_e7.pt \
        --bundle data/bundle2 --tables data/sc1/tables.json
    python research/sc1/s0p.py --cr1 data/cr1 --bundle data/bundle2 --out data/sc1
    python research/cr1/cr1_score.py --gens data/sc1/C/gen --bundle data/bundle2 --out data/sc1/C
    python research/cr1/cr1_score.py --gens data/sc1/T/gen --bundle data/bundle2 --out data/sc1/T
    python research/sc1/sc1_compare.py --root data/sc1 --cr1 data/cr1 --bundle data/bundle2 \
        --design data/sc1/design_review/design_review.json
    python research/sc1/sc1_tables.py --root data/sc1 --cr1 data/cr1

The generations come from `research/sc1/SC1_generate.ipynb` on Kaggle (steps in
`SC1_KAGGLE.md`). Their provenance, with every file's seed, hash and resampling
statistics, is in `data/sc1/{C,T}/gen/PROVENANCE.json`.

## Appendix — arm T's cells that are not CLEARED (exact list)

Extracted by code from `data/sc1/T/cr1_report.json`; x and 95% CI as the
section-26.1 test reports them.

**CONVICTED (10)**

| state | field | x | 95% CI |
|---|---|---:|---|
| benign Q:1 | PTH | +0.0964 | [+0.0820, +0.1108] |
| benign Q:1 | QRY | +0.0970 | [+0.0845, +0.1095] |
| phishing Q:1 | LBL | +0.1174 | [+0.1021, +0.1327] |
| phishing Q:1 | SUB | +0.1455 | [+0.1349, +0.1561] |
| phishing Q:1 | QRY | +0.1433 | [+0.1255, +0.1611] |
| phishing Q:0 | PTH | +0.0575 | [+0.0500, +0.0650] |
| benign P:1 Q:1 | SUB | +0.0715 | [+0.0665, +0.0765] |
| benign P:1 Q:1 | PTH | +0.0984 | [+0.0915, +0.1053] |
| benign P:1 Q:1 | QRY | +0.0843 | [+0.0724, +0.0962] |
| benign P:0 Q:1 | QRY | +0.0889 | [+0.0706, +0.1072] |

**UNDECIDED under the section-26.1 test (7)**

| state | field | x | 95% CI |
|---|---|---:|---|
| benign Q:1 | SUB | +0.0567 | [+0.0459, +0.0675] |
| benign Q:0 | PTH | +0.0488 | [+0.0407, +0.0569] |
| phishing P:1 | SUB | +0.0517 | [+0.0350, +0.0684] |
| phishing P:1 | PTH | +0.0487 | [+0.0315, +0.0659] |
| phishing P:1 Q:0 | LBL | +0.0427 | [+0.0335, +0.0519] |
| phishing P:1 Q:0 | SUB | +0.0486 | [+0.0389, +0.0583] |
| phishing P:1 Q:0 | PTH | +0.0511 | [+0.0447, +0.0575] |

**UNDECIDED under the pooled rule (2):** these fields are too rare in their
state for a per-generation estimate.
- benign P:1 QRY: pooled +0.0707, UNDECIDED over 10 draws, TELL on the first 3.
- benign P:0 QRY: pooled +0.0861, UNDECIDED over 10 draws, TELL on the first 3.

That is why section 3 counts UNDECIDED as 9: the 7 plus these 2.
