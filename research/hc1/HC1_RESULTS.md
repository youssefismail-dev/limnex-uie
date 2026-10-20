# HC1 — results: hard generation-time presence rules on E7

**Registered:** `PREREG_HC1.md` (69862d8), before any HC1 code and before the
preserved C2 generations were opened. **Implemented:** 8d2f91d (off by default,
bit-identical against 96 configurations from the pre-HC1 code; suite 637).
**Amended:** 62b7753 (H3 only — memory cleanup, and one descriptive estimator
corrected; no verdict rule changed). **Run:** 2026-09-24, laptop CPU at
below-normal priority, one job at a time. **Checkpoint:** `data/best_e7.pt`,
sha256 `916d03857d7d3db4…`. E7's own result (C1 FAIL) is unchanged.

## Verdicts

| test | registered rule | result |
|---|---|---|
| **H1** — the rules hold | 100% honour in all 8 cells, no URL truncated | **FAIL** — every failure is a truncation in the phishing field-on cells: the registered "limit of masking", not a defect of the rules |
| **H2** — the mask is the model's own probability | \|z\| < 3 in all 8 cells | **PASS** — largest \|z\| 1.73 |
| **H3** — masking against the exact conditional | 95% CI of Δ inside ±0.02 | **EQUIVALENT** in all four comparisons (QRY and FULL, both classes) |
| **C2** — the unconditional mode must not regress | no worse verdict with x rising > 0.0141 | **NO ATTRIBUTABLE REGRESSION** |

**Adoption (section 5) is not met:** it requires H1 to pass. See section 7.

---

## 1. What ran

| step | output | time | peak memory |
|---|---|---|---|
| C2 generations — Kaggle GPU, `E7_C2_generate.ipynb` | `data/gen_cache_e7_gpu/` (10 × 8,000) | 30 min (GPU) | — |
| C2 verification — `research/e7/c2_gpu_verify.py` | `VERIFY.json` there | 4 min | — |
| C2 gate v2, benign / phishing | `data/hc1_c2_e7_{benign,phishing}/` | 24 + 13 min | 1.15 / 1.08 GB |
| H1 + H2 — `research/hc1/hc1_run.py` | `data/hc1/h1/` (every URL + trace) | 45 min | 0.38 GB |
| H3 — `research/hc1/hc1_h3.py` | `data/hc1/h3/` | 28 min (rerun) | 0.63 GB (rerun) |

**The GPU generations verified in full**, and more tightly than expected: every
file name matches the local gate's own cache key, every count and hash matches
the provenance file, the checkpoint and suffix table are the laptop's, and
**all 403 benign and 833 phishing URLs regenerated on the laptop CPU are
byte-identical to the GPU's**. `generate()` samples on the CPU with a seeded CPU
generator and only the forward pass ran on the GPU; in these samples the
floating-point differences never flipped a draw.

**H3's first run was stopped by the harness's own memory guard** (0.35 → 6.1
GB): it called the discriminator without the `gc.collect()` the gate received
on 2026-09-23. Fixed (value-neutral) and rerun in full; the aborted run is kept
in `data/hc1/h3_run1_aborted/`. The rerun reproduced the aborted run exactly
where they overlap: both exact-arm files byte-identical, and all 20 benign
paired deltas.

## 2. H1 — the rules hold: FAIL, by truncation only

| cell | honoured | truncated (w < 0.5 / w ≥ 0.5) | masked steps / URL |
|---|---|---|---:|
| benign P:1 | 1200/1200 | 0 | 15.6 |
| benign P:0 | 1200/1200 | 0 | 14.2 |
| benign Q:1 | 1200/1200 | 0 | 27.2 |
| benign Q:0 | 1200/1200 | 0 | 28.1 |
| phishing P:1 | 1200/1200 | **2** (2 / 0) | 24.0 |
| phishing P:0 | 1200/1200 | 0 | 22.9 |
| phishing Q:1 | **1190**/1200 | **13** (11 / 2) | 39.0 |
| phishing Q:0 | 1200/1200 | 0 | 26.5 |

- **Every one of the 10 URLs that failed to honour was truncated** — none
  finished without its field. Six cells are perfect, and H2 passes, so the
  rules and their accounting are sound.
- **The truncations are masking's own.** 13 of the 15 had w < 0.5 — the model
  wanted to stop, often at almost every step (w as small as 5·10⁻²⁰ over 247
  masked steps), and the mask kept it writing to `max_len`:

      kldyynixydftdsbhdyyfvsitrf4yrs507309filesfdyyyyspyr1htjliyaf4rfixxaibx…

  Unconstrained, 0.10% of C1's phishing `<Q:1>` URLs reach 240 characters;
  masked, 1.08% do — about eleven times as many.
- **Registered reading:** "for truncation, a limit of masking that resampling
  does not have." Masking guarantees presence only for URLs that finish, and in
  phishing `<Q:1>` — where C1 honoured 76.7% — it cannot always finish them.

## 3. H2 — the mask is the model's own probability: PASS

| cell | mean w (H1) | C1 honouring | z |
|---|---:|---:|---:|
| benign P:1 | 0.9942 | 0.9942 | −0.03 |
| benign P:0 | 0.9939 | 0.9920 | +1.32 |
| benign Q:1 | 0.9623 | 0.9640 | −0.48 |
| benign Q:0 | 0.9898 | 0.9898 | +0.01 |
| phishing P:1 | 0.9741 | 0.9752 | −0.26 |
| phishing P:0 | 0.9894 | 0.9882 | +0.60 |
| phishing Q:1 | 0.7869 | 0.7668 | +1.73 |
| phishing Q:0 | 0.9893 | 0.9900 | −0.40 |

The identity E_q[w] = P_p(rules hold) holds on the real model in all eight
cells: the probability the mask removes is exactly the probability the
unconstrained model spends on breaking the rules. That also means the masking
**accounts** for its distortion correctly — the distortion itself is H3's.

## 4. H3 — what masking costs against the exact conditional

### 4.1 The exact arm, and three estimates that agree

| | benign `<Q:1>` | phishing `<Q:1>` |
|---|---:|---:|
| draws for 1,200 accepted (shortfall) | 1,252 (0) | 1,983 (0) |
| **first pass** — one unconstrained draw per slot | **95.8%** (1150/1200) | **78.5%** (942/1200) |
| C1 — unconstrained, n = 4,000 | 96.4% | 76.7% |
| H2 — mean w of the masked sample | 96.2% | 78.7% |
| accepted / draws — the harmonic mean (amendment) | 95.8% | 60.5% |

Three independent estimates of the honouring probability agree in both classes.
The accepted/draws rate falls to 60.5% for phishing for exactly the reason the
amendment gives: with the suffix held, suffixes that rarely take a query
dominate the draw count.

### 4.2 Distortion, from the weights

TV(masked, exact) = **0.016** [0.013, 0.019] benign and **0.164** [0.148, 0.182]
phishing. The registered prediction, 1 − c1 (0.036 / 0.233), overshoots: it
assumes forced rows keep no weight, and they keep some. Rows with w < 0.5:
0.5% benign, 19.3% phishing.

### 4.3 Realism — the registered comparison: EQUIVALENT in all four

| class | field | x masked | x exact | Δ | 95% CI | n per draw | verdict |
|---|---|---:|---:|---:|---|---:|---|
| benign | QRY | +0.1017 | +0.1102 | −0.0085 | [−0.0146, −0.0025] | 1200 | **equivalent** |
| benign | FULL | +0.4637 | +0.4645 | −0.0008 | [−0.0016, −0.0000] | 1200 | **equivalent** |
| phishing | QRY | +0.1623 | +0.1656 | −0.0033 | [−0.0193, +0.0127] | 356 | **equivalent** |
| phishing | FULL | +0.4699 | +0.4710 | −0.0011 | [−0.0019, −0.0002] | 1200 | **equivalent** |

Every interval lies inside ±0.02. Δ is slightly negative in all four — the masked
arm marginally *less* distinguishable from real — which the stated limit (one
generation per arm; the interval omits generation noise) does not let us read
as an effect. phishing QRY is scored at n = 356, the most its real pool
supports. Parsed-suffix TV between the arms: 0.000 benign, 0.004 phishing —
delimiter effects on the host are negligible.

### 4.4 Where the distortion shows: structure (descriptive)

Masking re-routes a URL the model wanted to end at the host into a path first.
Excluding the 13 runaways:

| `<Q:1>` URLs | benign masked / exact | phishing masked / exact |
|---|---|---|
| carrying a path | 63.7% / 62.8% | **56.2% / 50.4%** |
| median path length, when present | 12 / 12 | 12 / 11 |
| mean path length, all URLs | 10.6 / 10.0 | 10.5 / 7.8 |

In phishing, 5.8 points more URLs carry a path (about 2.9 standard errors); in
benign the shift is noise. It is the up-weighting section 3 predicted, and the
gate's discriminator on QRY and FULL does not see it at ±0.02.

## 5. C2 — the unconditional mode: NO ATTRIBUTABLE REGRESSION

| cell | E5 (gate v2) | E7 (gate v2) | Δx | C2 |
|---|---|---|---:|---|
| benign LBL | CLEARED +0.0055 | CLEARED +0.0017 | −0.0038 | no regression |
| benign SUF | CLEARED +0.0149 | CLEARED +0.0165 | +0.0016 | no regression |
| benign SUB | CLEARED +0.0067 | CLEARED +0.0048 | −0.0019 | no regression |
| **benign PTH** | CLEARED +0.0298 | **UNDECIDED** +0.0369 | +0.0071 | **not attributable** |
| benign QRY | UNDECIDED +0.0573 | UNDECIDED +0.0541 | −0.0032 | no regression |
| phishing LBL | CLEARED +0.0204 | CLEARED +0.0238 | +0.0034 | no regression |
| phishing SUF | CLEARED +0.0074 | CLEARED +0.0054 | −0.0020 | no regression |
| phishing SUB | CLEARED +0.0302 | CLEARED +0.0334 | +0.0032 | no regression |
| phishing PTH | UNDECIDED +0.0443 | UNDECIDED +0.0604 | +0.0161 | no regression |
| **phishing QRY** | CONVICTED (pooled) +0.1333 | **UNDECIDED** (pooled) +0.0924 | −0.0409 | no regression |

- **One verdict worsened, and it is not attributable:** benign PTH, cleared to
  undecided, with x rising 0.0071 — half the 0.0141 that retraining alone
  produces at identical configuration.
- **phishing PTH's x rose by 0.0161**, above that noise, **but its verdict did
  not change** (undecided both); by the registered rule that is no regression.
  It is the cell to watch.
- **phishing QRY is no longer convicted:** its pooled excess fell from +0.1333 to
  +0.0924 and the pooled rule now abstains. The pooled rule can never acquit, so
  this is "not convicted", not "cleared".

## 6. C4 — descriptive

| | benign E5 → E7 (real) | phishing E5 → E7 (real) |
|---|---|---|
| SUB emitted | 38.17% → 37.74% (41.63%) | 75.66% → 75.34% (78.92%) |
| PTH emitted | 59.01% → 59.08% (72.42%) | 30.04% → 29.51% (38.98%) |
| QRY emitted | 4.51% → **3.62%** (6.40%) | 2.31% → **1.74%** (4.44%) |
| FULL excess (auxiliary) | +0.2134 → +0.2191 | +0.4011 → +0.4014 |
| validation NLL | 1.6866 → 1.6688 | 1.5249 → 1.5298 |

E7's unconditional mode writes **fewer** queries than E5's — further from the
real rate in both classes — while its query **content** is no worse (QRY above).
E7's own unconditional query rate, 3.62%, is the right baseline for C1's 96.4%
under `<Q:1>`; C1's write-up could only compare with E5's 4.5%.

## 7. What this means

- **The rules are right, and the mask is exactly the model's own probability.**
  H2 holds in all eight cells; six of the eight deliver 100% presence with no
  truncation.
- **Masking cannot always finish a URL the model wants to end.** In the phishing
  field-on cells it produced 15 runaway URLs in 2,400 — 13 under `<Q:1>`, where
  C1 honoured only 76.7% — and 10 of them therefore lack the field. That is H1's
  failure, and by section 5 **masking as registered is not recommended for
  adoption.**
- **Where masking finishes, its URLs are as realistic as exact draws** (H3,
  equivalent in all four comparisons) — but it shifts their structure as
  predicted: in phishing `<Q:1>`, 5.8 points more URLs carry a path.
- **E7's unconditional mode is safe against E5** (C2), with one non-attributable
  worsening and phishing QRY no longer convicted; it writes fewer queries than
  E5 does.

**Options the evidence supports — the owner's decision; none was started:**

1. **Resampling as the enforcement.** Exact by construction — the model's own
   conditional — and it never forces a URL to continue: its only truncations
   are the model's natural ones (about 0.1% of phishing `<Q:1>` URLs in C1).
   Measured here at 1.04× the draws for benign `<Q:1>` and 1.65× for phishing
   `<Q:1>`, with URLs as realistic as masking's and without the structural
   shift.
2. **Masking with a length guard** — redraw any row that reaches `max_len`.
   Closes H1's failure mode; keeps the structural shift.

Either needs its own short registration — at the least, H1 on all eight cells.
E5 remains the production checkpoint until the owner decides otherwise.

## Reproduce

    python research/e7/c2_gpu_verify.py --cache data/gen_cache_e7_gpu \
        --checkpoint data/best_e7.pt --bundle data/bundle2
    python -m uie.corpus.run_eval --checkpoint data/best_e7.pt --bundle data/bundle2 \
        --n 8000 --generations 5 --disc-seeds 10 --delta-benign 0.22 --delta-phishing 0.30 \
        --decoder temperature --temperature 1.0 --estimator v2 \
        --gen-cache data/gen_cache_e7_gpu --classes benign --out data/hc1_c2_e7_benign
    # (and --classes phishing --out data/hc1_c2_e7_phishing)
    python research/hc1/c2_compare.py
    python research/hc1/hc1_run.py --checkpoint data/best_e7.pt --bundle data/bundle2 \
        --c1 data/e7_eval/c1/c1_report.json --out data/hc1/h1
    python research/hc1/hc1_h3.py --checkpoint data/best_e7.pt --bundle data/bundle2 \
        --h1 data/hc1/h1 --c1 data/e7_eval/c1/c1_report.json --out data/hc1/h3
