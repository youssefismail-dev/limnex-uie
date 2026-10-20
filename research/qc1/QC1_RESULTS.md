# QC1 — results: what gives the queries away

- **Registration:** `PREREG_QC1.md` (3e0a1ed), before any QC1 code. Amendment 1,
  the E3 finish line (owner option B, 89c631a), came before any QC1 reading.
- **Code:** 0c66c2a, 7e245ab (the ladder's memory fix: C3's wrapper, values
  bit-identical) and 037d367 (amendment 1).
- **Read:** 2026-09-30, after validity passed.
- **Material:** the step-20,000 files, `e11_plan` (primary) and `e11_ref`
  (replication). Laptop only; nothing trained, nothing generated.
- **Numbers:** two appendices, pasted by script. Every number in the text appears
  there or has a declared source.
  - **Appendix A** is `qc1_read.py`, the registered readout;
  - **Appendix B** is `qc1_describe.py`: descriptive, not a registered reading,
    and it licenses nothing.

## In plain words

- **Of the four lever families, QC1 licenses one:** R1, a longer length limit. It
  comes from benign, where real queries too long for the model to write account
  for 0.29 of the free-mode tell.
- **Most of my predictions were wrong.** That is the useful part.
  - **The tell is in the strings themselves, not in how many common strings the
    model copies.**
    - Re-drawing the model's output to the real copy share makes the tell bigger,
      not smaller.
    - The strings the model does copy are told apart from the strings real sites
      copy at 0.15–0.17.
    - It reuses the wrong common strings, at the wrong frequencies.
  - **It is no longer shape.** On E5, replacing every letter and digit kept the
    whole tell. On E11 at step 20,000, which characters appear carries about half
    of benign's tell and most of phishing's.
  - **It is not near misses** (NO BLENDING in every cell).
  - **It is not phishing borrowing benign's queries.** The pull is real but small,
    about a tenth.
- **The finish-line items license nothing either:**
  - **phishing's missing slash** tracks the benign-conditioned model at ρ 0.45–0.46,
    just short of the registered 0.5;
  - **benign's rare "host?q" without a slash** is suppressed to 0.11–0.22 of its
    real rate, but the class explains it in only one arm;
  - **benign P1Q1 paths are not state-blind.** They clearly follow the query state.
- **What remains is a picture, not a licence:**
  - the model concentrates on a few common strings (strongly for phishing,
    moderately for benign);
  - 74.6% of its benign query training rows come from the page-frame corpus, which
    holds about 30 pages per site, where pagination and language parameters
    repeat.

  Whether that corpus's repetition leaks into the domain frame was a new
  hypothesis (§6). Amendment 2 tested it (QC1-9), and it is **refuted** (§6b).

## 1. Validity: PASS

| check | result |
|---|---|
| V1 | `data/e11_final/VERIFY.json` PASS; the 49 input files unchanged since the run; commit recorded |
| V2 | QC1's harness reproduces the gate at the gate's n within ±0.05 in all six cells; the largest difference is +0.0139 (free phishing) |
| V3 | the natural rates reproduce `data/e12/sizing.json` exactly |
| V4 | no draw voided |
| V5 | the probe's rows reproduce the data: phishing bare hosts 0 real no-slash in 3,000; benign host-route 67 of 14,822, inside the binomial CI of the 0.449% target |

## 2. The registered readings

| reading | result | as predicted? |
|---|---|---|
| QC1-1 copying against the natural rate | UNDER in every cell and both arms at the host level. Phishing against its domain-disjoint rate: UNDER (`e11_plan`), WITHIN (`e11_ref`). Strings seen once in training are copied less than real, not more: no memorisation | yes |
| QC1-2 the mix, or the content | CONTENT-LED in every cell, replicated. Re-drawing to the natural share raises x (ρ 1.08–1.60). Within strata, copied against copied reads 0.1467–0.1708; never-seen against never-seen 0.0414–0.0728 | benign P0Q1 was predicted to lean MIX-LED: **wrong** |
| QC1-3 the ladder | benign: neither IDENTIFIED nor DISTRIBUTED. One replicated property: flattening letters and digits (R_T 0.542, then 0.451). Phishing: IDENTIFIED on lexical content: flattening letters and digits 0.189 / 0.168, `skeleton` 0.394 / 0.398; parameter names 0.519 / 0.441 (above 0.50 in stage 1). SHAPE-SUFFICIENT in neither class | DISTRIBUTED and shape-sufficient, as E5: **wrong** |
| QC1-4 near misses | NO BLENDING in every cell. Generated exceeds real by 0.32–2.04 points, under the 5-point margin | BLENDING in benign: **wrong** |
| QC1-5 phishing pulled toward benign | NO LEAK. The gap is certain (t +42 to +50) but small: relative 0.08–0.10 | LEAK: **wrong** |
| QC1-6 the length limit | benign BUDGET (S 0.29, t +2.42); phishing NO BUDGET (S 0.18) | no BUDGET: **wrong** for benign |
| QC1-7 the slash | phishing bare hosts: NO SLASH-LEAK (ρ 0.45 / 0.46, λ 0.24 / 0.27). Benign host-route: SUPPRESSED in both arms (ratio 0.11 / 0.22); LEAK-CONSISTENT in `e11_ref` only | SLASH-LEAK: **wrong**; SUPPRESSED: right |
| QC1-8 benign P1Q1 state blindness | PTH NOT STATE-BLIND in both arms: x against its own state 0.0563 / 0.0881, against P1Q0 0.3036 / 0.2679. SUB: not replicated | PTH STATE-BLIND: **wrong** |

A correction to `E12_ROADMAP.md` §2.3. The length limit could account for "at most
about 0.007" of benign's tell: that bound was mine and it is wrong. QC1-6
measured 0.0555 against 0.0392, a difference of 0.0163.

## 3. The licences (registered table, §4 and amendment 1)

| lever family | licensed | by |
|---|---|---|
| query subwords | **no** | no MIX-LED, no BLENDING, names not IDENTIFIED |
| the class at every position | **no** | no LEAK, no SLASH-LEAK; LEAK-CONSISTENT not replicated |
| the plan at every position | **no** | no replicated STATE-BLIND |
| R1, `max_len` 384 | **yes** | benign BUDGET |

## 4. What gives the queries away

- **It is the strings, not the rate.** The model copies too few real strings, and
  raising the copy share makes it more detectable. So the strings it does copy are
  wrong, and the fix is not "copy more".
- **Lexical content, not shape.** Flattening letters and digits removes about half
  of benign's tell and four-fifths of phishing's. Structure-only and shape-only
  transforms remove far less than on E5.
  - [INFERENCE] The structural repairs since E5 fixed the shape; what is left is
    which names and values the model writes.
- **Not the mechanisms the table anticipated:** no near misses, no class leak for
  queries, no state blindness.
- **One real, small contributor:** benign's long queries that the 256-token limit
  deletes (QC1-6).

## 5. The finish-line items

- **Phishing's missing slash, 0.29–0.32% on real phishing bare hosts.**
  - Swapped to the benign class, the same hosts get 1.06–1.31%. So the phishing
    class cuts benign's tendency to about a quarter (λ 0.24–0.27), not to zero.
  - The tracking is moderate (ρ 0.45–0.46). The registered rule needed 0.5, so no
    lever is licensed.
- **Benign's rare "host?q" without a slash** is written at 0.049–0.100% against
  real 0.452%.
  - That is a rare outcome suppressed. It is not explained by the class in both
    arms.
- **Benign P1Q1 paths follow the query state** (x 0.30 against P1Q0 paths). Their
  remaining tell (0.05) is not state blindness.

## 6. Descriptive: the head, and the corpora (Appendix B; not a registered reading)

- **At equal sample size, the model concentrates on a few common strings:**
  - strongly for phishing: 41 distinct strings against 148, and the top 8 cover
    85% against 52%;
  - moderately for benign free mode and P1Q1: the top 8 cover 20% against 13%,
    and 27% against 16%;
  - not for benign P0Q1: 535 against 493, and 10% against 13%.
- **Total variation between the copied-string distributions sits at its noise
  floor** (0.78 / 0.78 for benign free). It shows nothing at these sizes. An
  earlier figure of mine (0.72) was computed without equal n and is withdrawn;
  it is Q3.1's lesson about comparing counts across sample sizes.
- **The strings the model inflates** are pagination and languages: benign free
  `page=4` 1.8% against 0.6% of copies, `page=5` 1.0% against 0.2%, `lang=en` 8.3%
  against 4.2%. For phishing, the kit strings `m=1` (28.9% against 16.6%) and
  `naps` (14.5% against 5.8%).
- **The corpora:**
  - the domain-frame corpus holds one URL per host;
  - the page-frame corpus holds 29.89 per host on average, up to 40, and supplies
    74.6% of benign's query training rows (40.8% of phishing's).
- **[HYPOTHESIS, not registered] Frame leakage.** Domain-frame queries borrow the
  page frame's query content, where one site repeats pagination and language
  parameters across its pages. That would produce the inflation above.
  - It is testable in minutes with QC1-5's harness, using the page-frame pool.
  - Testing it would be a new registered reading, so it is the owner's decision
    (§7).

## 6b. Amendment 2 — frame leakage: refuted (QC1-9)

- **Registered** (`PREREG_QC1.md` §10, 72d85ef) before its measurement, under the
  owner's delegation of laptop decisions.
- **NO FRAME-LEAK in every cell and both arms.** The generated domain-frame queries
  sit **further** from the page frame than real domain queries do. Every gap is
  negative: −0.0100 to −0.0329 (Appendix A).
- **The page-repeated strings are not borrowed.** Pagination and language strings
  are:
  - 6.8% of page-frame queries;
  - 6.5% of real domain queries;
  - 8.3% of `e11_plan`'s (7.3% of `e11_ref`'s) in free benign.

  The model's excess is its own, not the page frame's.
- **The hypothesis in §6 is refuted.** The frame at every position is not licensed.
- **By PREREG_E12's pre-registered rule** (39dc856, committed before this reading),
  E12's arms are the set without D4.

## 7. What this means for E12 (the owner decides)

> **Superseded by the owner's delegation (2026-09-30).** The executor decides
> laptop matters. QC1-9 was run (§6b), and E12 is registered in
> `research/native/PREREG_E12.md` with the arm set its rule selects. The options
> below are kept as the record of the choice.

- **Under the registered rules, E12 may train one lever: R1.**
  - It could plausibly clear benign free QRY, which is UNDECIDED at 0.0350 and is
    the cell QC1-6 measured.
  - Nothing licensed targets the other six cells or the three slash rows.
  - **So E12 with R1 alone is not expected to pass E3.**
- **The options:**
  1. **Test frame leakage first:** QC1 amendment 2, registered before it runs,
     minutes on the laptop. If it holds, it would license a lever aimed at the
     measured concentration: the frame at every position, or a sampler that stops
     one site's repeated pages outweighing others. E12 then trains R1 plus that
     lever, once, as the final round.
  2. **E12 with R1 only,** the final round under the end rule. E3 is very unlikely;
     the final report documents the residuals.
  3. **Close the E3 programme now** with the final report, since the one licensed
     lever is small.
- **Not an option under the owner's rules:** capacity (R3 stays locked; this
  result is not its registered proof), more steps, and decoder changes.

## 8. What is and is not claimed

**Claimed** [CONFIRMED by the registered readout]:
- the query tell is content-led in every cell;
- it is carried by lexical content far more than on E5;
- it is not near misses, class leakage or the copy rate;
- the length limit accounts for about 0.29 of benign's free-mode tell;
- the phishing slash residual is not licensed as class leakage by the registered
  rule;
- benign P1Q1 paths follow the query state;
- one lever family is licensed: R1.

**Not claimed:**
- frame leakage: a hypothesis;
- any cause of the head's concentration;
- that any lever would pass E3.

## Appendix A — the registered readout

`python research/qc1/qc1_read.py`, verbatim.

### Validity

Verdict: **PASS**; failures: none

| cell | the gate's x | the harness at the gate's n (mean of 12) | difference |
|---|---|---|---|
| free_benign | +0.0350 | +0.0307 | -0.0043 |
| free_phishing | +0.0953 | +0.1092 | +0.0139 |
| control_benign_P0Q1 | +0.0581 | +0.0504 | -0.0077 |
| control_benign_P1Q1 | +0.0511 | +0.0446 | -0.0065 |

### QC1-1 — copying against the natural rate

| cell | arm | generated | natural (T) | reading | strings seen once: generated / natural | reading |
|---|---|---|---|---|---|---|
| free_benign | e11_plan | 18.84% | 29.27% (±3.97) | UNDER | 2.59% / 4.59% | UNDER |
| free_benign | e11_ref | 19.46% | 29.27% (±4.45) | UNDER | 3.25% / 4.59% | WITHIN |
| free_phishing | e11_plan | 20.34% | 41.85% (±5.10) | UNDER | 0.43% / 5.06% | UNDER |
| free_phishing (domain-disjoint) | e11_plan | 20.34% | 26.48% (±4.57) | UNDER | 0.43% / 4.03% | UNDER |
| free_phishing | e11_ref | 21.94% | 41.85% (±6.01) | UNDER | 0.37% / 5.06% | UNDER |
| free_phishing (domain-disjoint) | e11_ref | 21.94% | 26.48% (±5.37) | WITHIN | 0.37% / 4.03% | UNDER |
| control_benign_P0Q1 | e11_plan | 28.49% | 42.90% (±1.82) | UNDER | 5.87% / 8.05% | UNDER |
| control_benign_P0Q1 | e11_ref | 29.93% | 42.90% (±1.82) | UNDER | 6.59% / 8.05% | UNDER |
| control_benign_P1Q1 | e11_plan | 16.50% | 24.77% (±1.59) | UNDER | 1.83% / 3.45% | UNDER |
| control_benign_P1Q1 | e11_ref | 15.98% | 24.77% (±1.61) | UNDER | 1.80% / 3.45% | UNDER |

The band profile (share of queries matching training strings seen 1 / 2–4 / 5–19 / 20+ times):

- free_benign: real 4.6% / 4.6% / 6.9% / 13.1%; e11_plan 2.6% / 3.2% / 3.2% / 9.9%; e11_ref 3.3% / 3.0% / 3.2% / 10.0%
- free_phishing: real 5.1% / 5.3% / 5.1% / 26.3%; e11_plan 0.4% / 0.5% / 0.9% / 18.5%; e11_ref 0.4% / 0.5% / 1.3% / 19.8%
- control_benign_P0Q1: real 8.0% / 7.7% / 10.7% / 16.4%; e11_plan 5.9% / 6.9% / 7.1% / 8.7%; e11_ref 6.6% / 7.8% / 6.8% / 8.8%
- control_benign_P1Q1: real 3.4% / 3.6% / 5.7% / 12.0%; e11_plan 1.8% / 1.9% / 2.6% / 10.1%; e11_ref 1.8% / 2.1% / 2.5% / 9.6%

### QC1-2 — the mix, or the content

| cell | arm | n | x plain | x at the natural share | rho | t | reading | replicated |
|---|---|---|---|---|---|---|---|---|
| free_benign | e11_plan | 1692 | +0.0614 | +0.0808 | 1.32 | -5.22 | CONTENT-LED | yes |
| free_benign | e11_ref | 1392 | +0.0508 | +0.0719 | 1.41 | -7.31 | CONTENT-LED | yes |
| free_phishing | e11_plan | 356 | +0.1009 | +0.1354 | 1.34 | -4.41 | CONTENT-LED | yes |
| free_phishing | e11_ref | 356 | +0.1010 | +0.1430 | 1.42 | -6.04 | CONTENT-LED | yes |
| control_benign_P0Q1 | e11_plan | 600 | +0.0488 | +0.0780 | 1.60 | -3.10 | CONTENT-LED | yes |
| control_benign_P0Q1 | e11_ref | 600 | +0.0680 | +0.0737 | 1.08 | -0.93 | CONTENT-LED | yes |
| control_benign_P1Q1 | e11_plan | 1800 | +0.0599 | +0.0854 | 1.43 | -5.47 | CONTENT-LED | yes |
| control_benign_P1Q1 | e11_ref | 1800 | +0.0642 | +0.0852 | 1.33 | -7.20 | CONTENT-LED | yes |

Strata (x within never-seen and within found strings), and phishing at the domain-disjoint share:

- free_benign e11_plan: never-seen +0.0676; found +0.1708
- free_benign e11_ref: never-seen +0.0529; found +0.1467
- free_phishing e11_plan: never-seen insufficient; found insufficient; x at the domain-disjoint share +0.0938
- free_phishing e11_ref: never-seen insufficient; found insufficient; x at the domain-disjoint share +0.1159
- control_benign_P0Q1 e11_plan: never-seen +0.0429; found +0.1524
- control_benign_P0Q1 e11_ref: never-seen +0.0414; found +0.1555
- control_benign_P1Q1 e11_plan: never-seen +0.0634; found +0.1570
- control_benign_P1Q1 e11_ref: never-seen +0.0728; found +0.1512

### QC1-3 — the ladder (Q3.1's, verbatim)

**benign:** NEITHER; replicated: ['alnum_flat']; skeleton R_T 0.658 (not shape-sufficient)

| transform | stage 1 R_T | t | verdict | stage 2 R_T | t | verdict |
|---|---|---|---|---|---|---|
| identity | 1.0 | None | baseline | 1.0 | None | baseline |
| null | 1.0 | None | no change | 1.0 | None | no change |
| clip_values_20 | 0.996 | -0.15 | no change | — | — | — |
| clip_values_15 | 1.032 | 1.12 | no change | — | — | — |
| clip_values_10 | 1.043 | 1.22 | no change | — | — | — |
| first_param | 0.966 | -0.64 | no change | — | — | — |
| fixlen_names | 0.956 | -0.79 | no change | — | — | — |
| mask_pct | 1.008 | 0.37 | no change | — | — | — |
| normalize_seps | 1.017 | 1.8 | no change | — | — | — |
| mask_punct | 1.023 | 1.95 | no change | — | — | — |
| alnum_flat | 0.542 | -3.38 | REDUCES | 0.451 | -4.61 | REDUCES |
| mask_name_chars | 0.769 | -2.09 | no change | — | — | — |
| mask_values | 1.122 | 1.39 | no change | — | — | — |
| fixlen_values | 1.031 | 0.62 | no change | — | — | — |
| sort_params | 1.041 | 2.08 | no change | — | — | — |
| skeleton | 0.658 | -2.09 | no change | — | — | — |
| structure_only | 0.284 | -5.63 | VOID degenerate | — | — | — |
| clip15_first | 0.97 | -0.48 | no change | — | — | — |

**phishing:** IDENTIFIED; replicated: ['alnum_flat', 'fixlen_names', 'mask_name_chars', 'mask_values', 'skeleton', 'structure_only']; skeleton R_T 0.394 (not shape-sufficient)

| transform | stage 1 R_T | t | verdict | stage 2 R_T | t | verdict |
|---|---|---|---|---|---|---|
| identity | 1.0 | None | baseline | 1.0 | None | baseline |
| null | 1.0 | None | no change | 1.0 | None | no change |
| clip_values_20 | 0.915 | -2.08 | no change | — | — | — |
| clip_values_15 | 0.933 | -1.35 | no change | — | — | — |
| clip_values_10 | 0.903 | -1.96 | no change | — | — | — |
| first_param | 0.945 | -0.92 | no change | — | — | — |
| fixlen_names | 0.901 | -2.27 | REDUCES | 0.843 | -2.96 | REDUCES |
| mask_pct | 1.028 | 2.08 | no change | — | — | — |
| normalize_seps | 0.984 | -0.85 | no change | — | — | — |
| mask_punct | 0.991 | -0.53 | no change | — | — | — |
| alnum_flat | 0.189 | -4.78 | REDUCES | 0.168 | -6.18 | REDUCES |
| mask_name_chars | 0.519 | -3.26 | REDUCES | 0.441 | -4.77 | REDUCES |
| mask_values | 0.762 | -5.03 | REDUCES | 0.727 | -3.12 | REDUCES |
| fixlen_values | 0.917 | -1.28 | no change | — | — | — |
| sort_params | 1.009 | 0.46 | no change | — | — | — |
| skeleton | 0.394 | -3.54 | REDUCES | 0.398 | -5.05 | REDUCES |
| structure_only | 0.093 | -4.94 | REDUCES | -0.009 | -5.39 | REDUCES |
| clip15_first | 0.861 | -1.72 | no change | — | — | — |

### QC1-4 — near misses (one edit from a training string seen 5+ times)

| cell | arm | generated never-seen, near | real never-seen, near | difference [95% CI] | reading | replicated |
|---|---|---|---|---|---|---|
| free_benign | e11_plan | 4.74% of 2,132 | 3.84% of 5,130 | +0.90 [-0.15, +1.94] | NO BLENDING | no |
| free_benign | e11_ref | 4.63% of 1,685 | 3.84% of 5,130 | +0.79 [-0.34, +1.92] | NO BLENDING | no |
| free_phishing | e11_plan | 3.02% of 1,488 | 1.29% of 621 | +1.74 [+0.49, +2.98] | NO BLENDING | no |
| free_phishing | e11_ref | 3.32% of 1,053 | 1.29% of 621 | +2.04 [+0.64, +3.44] | NO BLENDING | no |
| control_benign_P0Q1 | e11_plan | 7.60% of 10,625 | 7.29% of 1,029 | +0.32 [-1.35, +1.98] | NO BLENDING | no |
| control_benign_P0Q1 | e11_ref | 7.73% of 10,349 | 7.29% of 1,029 | +0.44 [-1.23, +2.11] | NO BLENDING | no |
| control_benign_P1Q1 | e11_plan | 3.81% of 12,348 | 2.97% of 4,101 | +0.84 [+0.22, +1.46] | NO BLENDING | no |
| control_benign_P1Q1 | e11_ref | 3.33% of 12,022 | 2.97% of 4,101 | +0.35 [-0.26, +0.96] | NO BLENDING | no |

### QC1-5 — phishing pulled toward benign

| arm | n | x, real phishing against benign | x, generated phishing against benign | gap (t) | relative | reading |
|---|---|---|---|---|---|---|
| e11_plan | 1068 | +0.4477 | +0.4046 | +0.0432 (+49.68) | 0.10 | NO LEAK |
| e11_ref | 1068 | +0.4477 | +0.4102 | +0.0375 (+42.41) | 0.08 | NO LEAK |

Benign-only training strings among phishing queries: e11_plan generated 1.34%, real 0.56%; e11_ref generated 1.41%, real 0.56%

### QC1-6 — the length limit

| class | real pool / fits | n | x all | x fits | S | t | reading |
|---|---|---|---|---|---|---|---|
| benign | 7,253 / 7,149 | 1800 | +0.0555 | +0.0392 | 0.29 | +2.42 | BUDGET |
| phishing | 1,068 / 1,030 | 343 | +0.1021 | +0.0833 | 0.18 | +1.72 | NO BUDGET |

### QC1-7 — the slash at the host end (teacher-forced; own class and swapped)

| arm | rows | n | data no-slash | model, own class | model, other class | reading |
|---|---|---|---|---|---|---|
| e11_plan | phishing bare hosts | 3,000 | 0.000% | 0.316% | 1.313% | NO SLASH-LEAK (rho 0.45 [0.42, 0.47], lambda 0.24) |
| e11_plan | benign host-route queries | 14,822 | 0.452% | 0.049% | 0.055% | SUPPRESSED (ratio 0.11 [0.08, 0.16]) |
| e11_plan | benign bare hosts (reported) | 3,000 | 2.067% | 1.729% | 0.597% | — |
| e11_ref | phishing bare hosts | 3,000 | 0.000% | 0.289% | 1.062% | NO SLASH-LEAK (rho 0.46 [0.43, 0.49], lambda 0.27) |
| e11_ref | benign host-route queries | 14,822 | 0.452% | 0.100% | 0.031% | SUPPRESSED, LEAK-CONSISTENT (ratio 0.22 [0.18, 0.29]) |
| e11_ref | benign bare hosts (reported) | 3,000 | 2.067% | 1.929% | 0.396% | — |

### QC1-8 — benign P1Q1 paths and subdomains: state blindness

| field | arm | n | x own state | x other state (P1Q0) | x between real states | Welch t | reading |
|---|---|---|---|---|---|---|---|
| PTH | e11_plan | 1800 | +0.0563 | +0.3036 | +0.3342 | -38.54 | NOT STATE-BLIND |
| PTH | e11_ref | 1800 | +0.0881 | +0.2679 | +0.3342 | -23.54 | NOT STATE-BLIND |
| SUB | e11_plan | 974 | +0.0392 | +0.0563 | +0.0993 | -5.12 | NOT STATE-BLIND |
| SUB | e11_ref | 974 | +0.0603 | +0.0374 | +0.0993 | +6.48 | STATE-BLIND |

### QC1-9 — domain-frame queries pulled toward the page frame

| cell | arm | n | x, real domain against page | x, generated against page | gap (t) | relative | reading | page-repeated strings: page / real / generated |
|---|---|---|---|---|---|---|---|---|
| free_benign | e11_plan | 1800 | +0.1628 | +0.1899 | -0.0271 (-7.43) | -0.17 | NO FRAME-LEAK | 6.8% / 6.5% / 8.3% |
| free_benign | e11_ref | 1800 | +0.1628 | +0.1833 | -0.0204 (-4.96) | -0.13 | NO FRAME-LEAK | 6.8% / 6.5% / 7.3% |
| free_phishing | e11_plan | 278 | +0.2889 | +0.3000 | -0.0111 (-1.86) | -0.04 | NO FRAME-LEAK | 0.2% / 0.3% / 0.5% |
| free_phishing | e11_ref | 278 | +0.2889 | +0.3100 | -0.0211 (-2.67) | -0.07 | NO FRAME-LEAK | 0.2% / 0.3% / 0.4% |
| control_benign_P0Q1 | e11_plan | 1340 | +0.2812 | +0.2962 | -0.0151 (-7.25) | -0.05 | NO FRAME-LEAK | 6.8% / 8.0% / 8.8% |
| control_benign_P0Q1 | e11_ref | 1340 | +0.2812 | +0.2912 | -0.0100 (-2.92) | -0.04 | NO FRAME-LEAK | 6.8% / 8.0% / 8.9% |
| control_benign_P1Q1 | e11_plan | 1800 | +0.1604 | +0.1933 | -0.0329 (-10.31) | -0.21 | NO FRAME-LEAK | 6.7% / 6.1% / 7.6% |
| control_benign_P1Q1 | e11_ref | 1800 | +0.1604 | +0.1894 | -0.0290 (-9.61) | -0.18 | NO FRAME-LEAK | 6.7% / 6.1% / 7.5% |

### Licences for E12 (section 4)

| lever family | licensed | by |
|---|---|---|
| query subwords | **no** | {'MIX-LED': [], 'BLENDING': [], 'IDENTIFIED on parameter names': False} |
| the class at every position | **no** | {'LEAK': False, 'SLASH-LEAK': False, 'SUPPRESSED and LEAK-CONSISTENT': False} |
| the plan at every position | **no** | {'STATE-BLIND': []} |
| the frame at every position | **no** | {'FRAME-LEAK': []} |
| R1 (max_len 384) | **yes** | {'BUDGET': ['benign']} |

## Appendix B — descriptive

`python research/qc1/qc1_describe.py`, verbatim.

**DESCRIPTIVE: not a registered reading; it licenses nothing.**

### The head: queries found word for word in training, generated (`e11_plan`) against real, at equal n (30 seeded subsamples)

| cell | n (copies per side) | distinct strings, generated / real | top 8 cover, generated / real | total variation, generated vs real / real vs real (at n/2) |
|---|---|---|---|---|
| free_benign | 495 | 324 / 374 | 20% / 13% | 0.78 / 0.78 |
| control_benign_P0Q1 | 773 | 535 / 493 | 10% / 13% | 0.77 / 0.70 |
| control_benign_P1Q1 | 1,350 | 637 / 812 | 27% / 16% | 0.69 / 0.65 |
| free_phishing | 380 | 41 / 148 | 85% / 52% | 0.48 / 0.46 |

The favourites, each as a share of the copied strings (generated | real):

- **free_benign**, generated's top: `lang=en` 8.3% | 4.2%, `page=2` 3.4% | 2.3%, `page=1` 2.4% | 1.2%, `page=4` 1.8% | 0.6%, `lang=de` 1.2% | 0.6%, `page=3` 1.2% | 1.1%, `page=5` 1.0% | 0.2%, `lang=es` 1.0% | 0.3%
  - real's top: `lang=en` 4.2% | 8.3%, `page=2` 2.3% | 3.4%, `page=1` 1.2% | 2.4%, `page=3` 1.1% | 1.2%, `p=1` 0.8% | 1.0%, `ReturnUrl=%2F` 0.8% | 0.2%, `et_blog` 0.8% | 0.8%, `SA` 0.7% | 0.4%
- **control_benign_P0Q1**, generated's top: `lang=en` 2.1% | 2.6%, `p=1` 1.2% | 1.7%, `cat=1` 1.1% | 1.6%, `C=S;O=A` 1.0% | 1.2%, `C=N;O=D` 1.0% | 0.9%, `C=M;O=A` 1.0% | 0.6%, `C=D;O=A` 0.9% | 1.0%, `C=N;O=A` 0.9% | 0.9%
  - real's top: `lang=en` 2.6% | 2.1%, `p=1` 1.7% | 1.2%, `SA` 1.7% | 0.3%, `cat=1` 1.6% | 1.1%, `mailpoet_page=subscriptions` 1.6% | 0.0%, `MA` 1.6% | 0.6%, `C=S;O=A` 1.2% | 1.0%, `MD` 1.2% | 0.3%
- **control_benign_P1Q1**, generated's top: `lang=en` 8.1% | 5.1%, `page=2` 5.9% | 3.2%, `page=3` 3.6% | 1.4%, `page=1` 3.2% | 1.7%, `page=4` 1.6% | 0.9%, `lang=fr` 1.5% | 0.7%, `lang=de` 1.4% | 0.7%, `et_blog` 1.4% | 1.2%
  - real's top: `lang=en` 5.1% | 8.1%, `page=2` 3.2% | 5.9%, `page=1` 1.7% | 3.2%, `page=3` 1.4% | 3.6%, `et_blog` 1.2% | 1.4%, `ReturnUrl=%2F` 1.2% | 0.2%, `id=1` 1.0% | 0.5%, `amp=1` 1.0% | 0.8%
- **free_phishing**, generated's top: `m=1` 28.9% | 16.6%, `naps` 14.5% | 5.8%, `email=3mail@b.c` 10.3% | 6.0%, `i=1` 9.7% | 5.4%, `lca` 8.7% | 7.8%, `eta=3mail@a.b.c0` 5.3% | 3.4%, `8c5781c2-7b1e-4035-b1d7-990e` 4.5% | 3.6%, `email=3mail@a.b.c0` 3.2% | 0.7%
  - real's top: `m=1` 16.6% | 28.9%, `lca` 7.8% | 8.7%, `email=3mail@b.c` 6.0% | 10.3%, `naps` 5.8% | 14.5%, `i=1` 5.4% | 9.7%, `8c5781c2-7b1e-4035-b1d7-990e` 3.6% | 4.5%, `eta=3mail@a.b.c0` 3.4% | 5.3%, `_lp=1&_token=uuid_3cqn1r92k8` 2.9% | 0.0%

### The training corpora

| corpus | rows | hosts | rows per host | largest host | rows with a query |
|---|---|---|---|---|---|
| name_train | 906,214 | 906,214 | 1.00 | 1 | 57,768 |
| structure_train | 1,648,460 | 55,147 | 29.89 | 40 | 170,055 |
| benign page-frame share of query rows | | | | | 74.6% |
| phishing_name_train | 192,652 | 192,652 | 1.00 | 1 | 8,272 |
| phishing_structure_train | 46,491 | 7,802 | 5.96 | 40 | 5,710 |
| phishing_page-frame share of query rows | | | | | 40.8% |
