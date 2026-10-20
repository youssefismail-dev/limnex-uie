# E8 (N1) — native presence plan and root-slash token: pre-registration

**Registered:** 2026-09-26, before any E8 code, training or generation.
- **Plan:** `NATIVE_REPAIR_PLAN.md` revision 2. Its direction and order were
  approved by the owner, 2026-09-26.
- **Frozen:**
  - `data/best_e7.pt` (sha256 `916d0385…`);
  - gate v2 and its thresholds;
  - `research/cr1/cr1_score.py`;
  - `data/bundle2`;
  - SC1's state tables (`0c6f2760…`);
  - DC1's settings and `uie/free.py`;
  - every published result.
- **Evidence tags** as in the plan: [CONFIRMED], [HYPOTHESIS], [DIAGNOSTIC],
  [DECISION].

## 1. Question

Does **plan-first presence conditioning** address the documented emission and
native-honouring failures (plan §3 K1, K2)? Does a **root-slash token** address
the documented root-slash artifact (K3)? Can each effect be attributed to its own
change, without a regression in realism?

**Not asked:** query content. N1 claims nothing about any QRY content cell.

## 2. Arms

Every arm uses E7's recipe exactly, as `uie.kaggle_train.run` was called for E7:

    steps 20,000 · batch 256 · lr 3e-4 · d_model 384 · 8 layers · 8 heads ·
    class_alpha 1.0 · frame on · boundary_weight 1.0 · not factorised ·
    eval_every 1,000 · seed 20260803 · max_len 256 · the same bundle2 streams

| arm | presence conditioning | root slash | training difference from E7 |
|---|---|---|---|
| `e8_base` | E7's: prefix `<P:*> <Q:*>`, dropout 0.5 | no | none: E7's recipe on the E8 code |
| `e8_rs` | E7's | **yes** | the tokenizer carries `<RS>` |
| `e8_pf` | **plan** | no | plan tokens, no dropout |
| `e8_pfrs` | **plan** | **yes** | both — **the candidate** |

**Checkpoint selection:** `best.pt`, on the mean per-class validation loss of the
arm's own free mode.
- **E7-conditioned arms:** controls shown as `?`, exactly as E7.
- **Plan arms:** the true plan tokens scored as targets. The plan is a function of
  the URL, so this is −log P(URL | class, frame, suffix), the free-mode likelihood.
- Selection is within an arm. Arms are compared only through the instruments of §5.

**References, no training:**
- **E7 itself, raw.** Its canonical free-mode files from HC1 C2
  (`data/gen_cache_e7_gpu/`, verified byte-identical on the CPU) are reused at the
  same seeds.
- **E7 + DC1, free mode** (the adopted baseline).

## 3. The two changes, exactly

### 3.1 Root-slash token — tokenizer flag `root_slash`

- **Defaults off.** It is stored in the tokenizer state only when on, so every
  existing state is byte-identical.
- **Vocabulary.** When on, one token `<RS>` is appended after the presence
  controls. Only tokenizers with the flag gain it.
- **Encode.** When the path body is empty but the raw path is not (`host/`,
  `host/?q`), `<RS>` is written in `<PTH>`'s place, with no characters. A
  non-empty body is `<PTH>` as today.
- **Decode.** `<RS>` renders the root slash.
- **The gate's parser, presence definitions and controls are unchanged.**
  `host/` has no path body, so `<RS>` never means P = 1.

### 3.2 Presence plan — tokenizer flag `presence_plan`

- **Requires `include_fields`. Defaults off,** stored only when on.
- **Encode.** No presence controls in the prefix. Immediately after the SUF field
  (the suffix's characters) comes `<P:v> <Q:v>`, with v the **true** value by the
  tokenizer's unchanged presence definition. `<LBL>` follows.
- **Training.**
  - Control dropout must be 0; any other value is refused.
  - The plan tokens are targets at weight 1.0, as shape.
  - `<RS>` is also shape at weight 1.0.
- **Generation, FULL FREE.**
  - The suffix is drawn from the shipped table and injected, as today.
  - The model then fills each plan slot as a categorical over its two legal tokens:
    its distribution renormalised over `{<P:0>, <P:1>}`, then `{<Q:0>, <Q:1>}`.
  - The mass the model puts outside the legal tokens is recorded per URL and
    reported. A slot has a grammar; this is not a presence mask.
- **Generation, FULL CONTROL.** The suffix comes from SC1's state table and the
  plan tokens are supplied, just as the suffix is.
- **Refused in plan mode:**
  - a partial request (one field set);
  - `enforce` (masking);
  - `blend` (a second prediction);
  - generation without a supplied suffix.

## 4. Generation (Kaggle T4; the laptop verifies)

**Decoding for every arm and reference:** R4's temperature 1.0 with the frozen path
delta (benign 0.22, phishing 0.30) — the resting decode.

**FULL FREE (canonical):**
- per class, G = 5 generations of n_main 8,000;
- seeds are the gate's canonical generation seeds, 20260807 + 1000·g;
- the shipped suffix table;
- each arm's own free mode: `?` for `e8_base` and `e8_rs`, a sampled plan for
  `e8_pf` and `e8_pfrs`;
- E7 + DC1 via `uie.free.generate_free` at the same seeds.

**FULL CONTROL, native, with no resampling of any kind:**
- all 8 joint states × G = 5;
- 3,000 URLs per generation in the 6 measurable states, and 1,000 in phishing
  P1 Q1 and P0 Q1 (honouring only; plan §2);
- the suffix is drawn from SC1's state table, once per URL;
- seeds are 20261001 + (100·s + g)·10⁹ + 9·10⁸, with s = 8…15 as in CR1's
  joint-state block. The 9·10⁸ offset is unused by CR1 (0), SC1 (3·10⁸, 6·10⁸,
  8·10⁸), C3 (4.5·10⁸) and DC1 (1.5·10⁸);
- the same seeds for every arm and for E7, so all contrasts are paired.

**Volume:**
- 180,000 URLs per new arm;
- E7 control: 100,000;
- E7 + DC1 free: 80,000.

**Provenance, per file:** sha256, seconds, forward rows, and for plan arms the
off-grammar plan mass.

**Two notebooks, two sessions:** training (≈ 4.5 h), then generation (≈ 7–8 h).

## 5. Readings

**Definitions** — all with the gate's own `uie.evaluation.FIELDS`:
- **Presence of a URL:** PTH is present iff the path body is non-empty; QRY is
  present iff the query is non-empty.
- **Honouring:** a controlled URL honours its request iff both realised presences
  equal the request.
- **Emission:** the share of free-mode URLs with PTH present, and with QRY present,
  over 5 × 8,000 URLs per class.
- **Root-slash rate:** among free-mode URLs with no path body, the share whose
  scheme-stripped string has the root slash (`host/`, `host/?…`). Split by
  query absent or present.

**Validity [V]** — any failure is fixed under an amendment, and nothing is read:
- **V1.** Hashes: checkpoints, code bundle, suffix and state tables, DC1 settings.
- **V2.** Byte identity on the laptop CPU for at least 250 URLs per arm per mode
  (DC1's method).
- **V3.** With both flags off, the E8 code reproduces E7's tokenizer state and
  every existing generation fixture byte for byte (tests).
- **V4.** The free scorer (`cr1_score.score`, unrestricted pools, n_main 8,000)
  reproduces HC1 C2's recorded E7 canonical cells from E7's files.
- **V-base.** `e8_base` reproduces E7 within noise:
  - no free gated cell's verdict changes with |Δx| > 0.0141;
  - honouring within ±1 point per joint state;
  - emission within ±0.5 points.

  If it fails, the pipeline changed. No arm is read, and it is investigated under
  an amendment.

**Primary** (§6: K1a–K1c) and **realism** (K1d), per arm:
- **free mode:** gate v2 at the canonical configuration, scored by `cr1_score.score`
  (G = 5, D = 10, canonical seeds);
- **control mode:** CR1's instrument per measurable joint state, on the native URLs
  as delivered — violating URLs included, because that is what a user receives;
- **guardrails (K1e):**
  - whole-URL novelty: `normalize_for_dedup` against `known_urls.bloom.gz`;
  - query copying: query strings verbatim in `name_train` or `structure_train`.

**Reported, not criteria:**
- FULL (auxiliary) per arm, with the slash arms set against the root-slash
  reference (plan §5.3);
- the off-grammar plan mass;
- per-state honouring failure modes;
- each arm against E7 + DC1;
- the plan × slash interaction.

## 6. Criteria (fixed now)

Baseline for every contrast: `e8_base`.

| code | applies to | criterion |
|---|---|---|
| **K1a** honouring | plan arms | ≥ 0.98 in each of the 8 joint states. **Reported separately from realism; never evidence about it.** |
| **K1b** emission | plan arms | QRY emission within ±10% (relative) of the achievable targets 6.62% / 4.68%, i.e. [5.96, 7.28]% / [4.21, 5.15]%. PTH emission within ±3 points of 73.37% / 39.51%, i.e. [70.37, 76.37]% / [36.51, 42.51]% |
| **K1c** root slash | slash arms | within 2 points of the training rates: bare hosts 97.9% / 99.9%; query without path 99.6% / 100% |
| **K1d** realism | all arms | no gated cell, free (10) or control (22 computable), has its verdict worsen **and** its x rise by more than 0.0141 (E7 C2's rule) |
| **K1e** guardrails | all arms | neither whole-URL novelty nor query copying worsens by ≥ 0.5 points with a 95% CI excluding 0 |

- **Targets.** The achievable emission targets are computed from `name_train`
  under the shipped table (plan §3 K1). They are free mode's own attainable rates,
  not the plain domain-frame rates.
- **Attribution:**
  - **plan** = `e8_pf` − `e8_base`;
  - **slash** = `e8_rs` − `e8_base` and `e8_pfrs` − `e8_pf`;
  - **interaction** = the difference between the two slash contrasts.
  - No effect smaller than |`e8_base` − E7| on the same metric is attributed to a
    change.
- **Candidate** (the owner decides adoption):
  - `e8_pfrs`, if it passes K1a–K1e.
  - **Only K1a fails:** N1b (unlikelihood) is licensed, and nothing else.
  - **The plan fails K1b, K1d or K1e:** the plan is not taken forward. `e8_rs`
    stands alone if it passes K1c–K1e.
  - **K1c fails:** the slash token is not taken forward.
- **Not claimed:**
  - QRY content;
  - E3 passage;
  - any realism for phishing P1 Q1 and P0 Q1 (unmeasurable; honouring only).

## 7. Predictions, stated before any run [HYPOTHESIS]

- **Plan arms:** emission within K1b's bands; honouring higher than `e8_base` in
  every state, most in phishing P1 Q1 and P0 Q1.
- **Slash arms:**
  - root-slash rate near the training rate;
  - FULL falls toward the root-slash reference (plan §5.3, measured on E7): from
    +0.2168 / +0.4018 as generated to +0.0720 / +0.0520 with the slash rendered.
    E8's own FULL is compared with its non-slash counterpart, not with these
    numbers directly;
  - no gated cell moves beyond retraining noise.
- **`e8_base`:** E7 within noise.

## 8. Order

1. This registration.
2. Code and tests. Flags off must reproduce E7 byte for byte.
3. The training notebook: the owner runs it.
4. The generation notebook: the owner runs it.
5. Verification, scoring and comparison on the laptop.
6. `E8_RESULTS.md`.

The owner decides adoption.

## Status

    E8 (N1)  registered 2026-09-26
    E8 (N1)  run 2026-09-26/27; V1-V4 PASS; V-base FAILED -> no arm read
             (Amendment 1; E8_RESULTS.md). Awaiting the owner's decision.
    E8 (N1)  OWNER DECISION 2026-09-28 (DECISION_2026-09-28.md): permanently
             UNREAD and OBSOLETE -- E9 addressed the objective-induced emission
             bias E8 was measuring. No K1 criterion is ever evaluated; the
             artifacts (data/e8/) and this record are kept for provenance only.

**Amendment 1, 2026-09-27 — V-base failed; the investigation. It records facts
and changes no criterion or threshold.**

**What failed:**
- Honouring differs from E7's in 3 of 8 states: benign P0Q1 +1.95, phishing P1Q1
  +5.24, phishing P0Q1 +3.66 points, against ±1.
- Benign path emission differs by −1.17 points, against ±0.5.
- These are real model differences, 3–8 standard errors of the difference.
- As section 5 requires, **no arm is read.**

**Investigation** [CONFIRMED]:
- **Identical to E7's:** `e8_base`'s tokenizer state, model configuration,
  training settings, data (kept and dropped rows), training code (network, loss,
  batching; the E8 additions are empty for E7's tokenizer) and environment (torch
  2.10.0+cu128).
- **What differs:** all 101 weight tensors. The model is built before
  `train()` seeds, so its initial weights are not fixed by the seed, and GPU
  arithmetic is not deterministic.
- **The runs diverge from the first checkpoint:** validation loss at step 1,000 is
  2.18807 (E7) against 2.18216 (e8_base).

**Reading of the failure:**
- The premise "if it fails, the pipeline changed" does not hold here. The failure
  is spread between two independent trainings of one recipe.
- The ±1 and ±0.5 tolerances were not calibrated on measured spread. This pair is
  the first measurement.

**Decision: the owner's.** The options are in `E8_RESULTS.md` §5: leave E8
unread; amend V-base's role (post hoc); or measure the spread with more seeds
first. Until then no K1 criterion is evaluated or reported.
