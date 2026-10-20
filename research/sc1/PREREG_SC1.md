# SC1 — the state-conditioned suffix draw: pre-registration

**Registered:** 2026-09-24, before any SC1 code and before any SC1 generation.

**Approved by the owner (2026-09-24),** from `research/control/REPAIR_PLAN.md`
(4fd9511):
1. the branch: E7 + presence control;
2. SC1, including the matched control arm C;
3. the table source: the domain-frame training corpus (`name_train`).

**Then, after review, this registration as written:** the arms, the statistical
methods, validity checks V1–V6 and kill criteria K1–K5.

**Frozen:**
- `data/best_e7.pt` (sha256 `916d0385…`) and `data/best_e5.pt`;
- R4's decoder (`temperature` 1.0), the adopted delta (benign 0.22, phishing
  0.30), frame `domain`;
- `data/bundle2`, including the shipped `suffix_table.json` (sha256
  `aaa6fb85…`);
- every file under `uie/` — nothing in the library is edited;
- gate v2: the 0.05 bar, MIN_N 250, G = 5, D = 10, t_crit, the section-26.1
  and pooled rules;
- `research/cr1/cr1_score.py`;
- every published result and its data, CR1's included. Nothing under
  `data/cr1/` is written.

---

## 1. Question

CR1 scored E7's controlled output (joint resampling, `uie.presence`) against real
URLs in the same state. SUF convicted in six states and was undecided in three.

The suffix is drawn from a class-marginal table by label alone, before the URL
(`uie/suffixes.py:151`). Resampling can move that mix by at most (1 − a)/a in
TV, so an effective control leaves it at the class average. The shipped table
also counts both training corpora, whose frame proportions change by state.

**SC1 asks two things.**
1. Does drawing the suffix from the requested state's domain-frame training
   distribution remove the SUF tells?
2. Do phishing's LBL and SUB tells in the path and query states follow the
   suffix mix?

The owner does not adopt anything by this registration; section 7.5 states what
adoption would require.

## 2. Design review — measured before registration

Read-only throughout: no model loaded, no URL generated. The script is
`research/sc1/design_review.py`, committed with this registration; outputs are
in `data/sc1/design_review/`.

### 2.1 The construction is the shipped one

The shipped `suffix_table.json` was rebuilt from every `*_train` row by counting
`parse(url).public_suffix` per row and skipping empty suffixes. It is
**identical**: both classes, every count (benign 2,554,633 rows, 1,232
suffixes; phishing 239,143 rows, 683 suffixes). SC1's tables therefore differ
from the shipped one *only in which rows are counted*.

Presence computed from a single parse equals `FIELDS` on 20,000 rows, and every
row's label matches its file.

### 2.2 Premise H7: each state's training mix transfers to its held-out pool

TLD mix over the class's top-10 `name_train` TLDs plus "other". Training rows
of the state are compared with the held-out hosts of the state (the pool the
scorer uses, one per host). The null is two multinomial samples of the same
sizes from the pooled mix, 1,000 repeats.

| state | train rows | held-out hosts | TV | null p99 | p |
|---|---:|---:|---:|---:|---:|
| benign P:1 | 655,215 | 82,032 | 0.0038 | 0.0061 | 0.32 |
| benign P:0 | 250,997 | 31,245 | 0.0034 | 0.0096 | 0.92 |
| benign Q:1 | 57,768 | 7,253 | 0.0088 | 0.0198 | 0.73 |
| benign Q:0 | 848,444 | 106,024 | 0.0036 | 0.0053 | 0.21 |
| phishing P:1 | 75,027 | 9,388 | 0.0075 | 0.0201 | 0.92 |
| phishing P:0 | 117,625 | 14,693 | 0.0090 | 0.0168 | 0.52 |
| phishing Q:1 | 8,272 | 1,068 | 0.0278 | 0.0598 | 0.60 |
| phishing Q:0 | 184,380 | 23,013 | 0.0067 | 0.0138 | 0.67 |
| benign P:1 Q:1 | 42,862 | 5,451 | 0.0108 | 0.0228 | 0.65 |
| benign P:1 Q:0 | 612,353 | 76,581 | 0.0037 | 0.0061 | 0.42 |
| benign P:0 Q:1 | 14,906 | 1,802 | 0.0265 | 0.0371 | 0.21 |
| benign P:0 Q:0 | 236,091 | 29,443 | 0.0047 | 0.0103 | 0.72 |
| phishing P:1 Q:0 | 69,741 | 8,719 | 0.0082 | 0.0204 | 0.89 |
| phishing P:0 Q:0 | 114,639 | 14,294 | 0.0093 | 0.0162 | 0.52 |

**The premise holds in all 14 states** (every p ≥ 0.2). A table estimated from
a state's training rows matches that state's held-out TLD mix within sampling
noise.

### 2.3 Coverage: what a state table cannot draw

| state | T rows | T distinct | Good–Turing unseen | held-out hosts whose suffix T lacks | … C lacks |
|---|---:|---:|---:|---:|---:|
| benign P:1 | 655,215 | 1,190 | 0.0002 | 0.0002 | 0.0001 |
| benign P:0 | 250,997 | 978 | 0.0005 | 0.0005 | 0.0001 |
| benign Q:1 | 57,768 | 932 | 0.0035 | 0.0040 | 0.0010 |
| benign Q:0 | 848,444 | 1,187 | 0.0002 | 0.0001 | 0.0001 |
| phishing P:1 | 75,027 | 582 | 0.0024 | 0.0019 | 0.0014 |
| phishing P:0 | 117,625 | 455 | 0.0010 | 0.0008 | 0.0003 |
| phishing Q:1 | 8,272 | 274 | 0.0125 | **0.0122** | 0.0037 |
| phishing Q:0 | 184,380 | 662 | 0.0009 | 0.0006 | 0.0006 |
| benign P:1 Q:1 | 42,862 | 893 | 0.0052 | 0.0046 | 0.0009 |
| benign P:1 Q:0 | 612,353 | 1,141 | 0.0002 | 0.0002 | 0.0001 |
| benign P:0 Q:1 | 14,906 | 585 | 0.0139 | **0.0111** | 0.0011 |
| benign P:0 Q:0 | 236,091 | 962 | 0.0006 | 0.0005 | 0.0001 |
| phishing P:1 Q:0 | 69,741 | 551 | 0.0025 | 0.0019 | 0.0010 |
| phishing P:0 Q:0 | 114,639 | 451 | 0.0010 | 0.0008 | 0.0003 |

The class-marginal `name_train` table (arm C) holds every suffix the shipped
table holds (1,232 benign, 683 phishing). State tables lose tail coverage only in
the two sparsest states, about 1.1–1.2% of held-out hosts. That cost is
reported, not smoothed away.

### 2.4 Parse stability: the rendered URL keeps its drawn suffix

HC2's held-mode outputs (`data/hc2/held/`, suffix drawn from the shipped table
and held, n = 1,200 per cell) were compared on their parsed-suffix mix against
the table. The null is a 1,200-draw multinomial from the table, 1,000 repeats.

| cell | TV | null p99 | p | parsed suffix not in the table |
|---|---:|---:|---:|---:|
| benign P:1 | 0.0394 | 0.0469 | 0.046 | 0.00% |
| benign Q:1 | 0.0202 | 0.0461 | 0.77 | 0.25% |
| phishing P:1 | 0.0354 | 0.0534 | 0.30 | 0.17% |
| phishing Q:1 | 0.0286 | 0.0525 | 0.57 | 0.33% |

All four are within the null's 99th percentile, and at most 0.33% of URLs parse
to a suffix the table does not hold. **Held mode delivers its table.** This
fixes the manipulation check (section 6, V4).

### 2.5 The SUF ceiling of each table, and what it shows about CR1

**Method.** Training URLs drawn i.i.d. from a table's own rows (5 × 3,000 with
replacement; seed 20260925 + 1000·s + 10·table + g) are scored as the generated
side by the gate's SUF instrument, exactly as `cr1_score` scores a generation:
`discriminate_with_floor` on the state's held-out pool, split seeds v2, D = 10,
n = min(3,000, 1,200), the section-26.1 verdict. The result is the SUF a
generator would score if its parsed suffixes followed that table exactly:

- **T**: `name_train`, in the state;
- **C**: `name_train`, class marginal;
- **S**: all `*_train` rows, class marginal — the shipped table, i.e. CR1's
  table without the joint tilt.

No model is involved.

| state | CR1 SUF (measured, joint) | S ceiling | C ceiling | T ceiling | CR1 tilt bound |
|---|---|---|---|---|---:|
| benign P:1 | +0.0100 ± 0.0045 CLEARED | +0.0094 ± 0.0010 CLEARED | +0.0021 ± 0.0026 CLEARED | +0.0033 ± 0.0029 CLEARED | 0.006 |
| benign P:0 | +0.0641 ± 0.0020 **CONVICTED** | +0.0687 ± 0.0033 **CONVICTED** | +0.0343 ± 0.0056 CLEARED | +0.0062 ± 0.0031 CLEARED | 0.008 |
| benign Q:1 | +0.0563 ± 0.0043 UNDECIDED | +0.0524 ± 0.0015 UNDECIDED | +0.0579 ± 0.0017 **CONVICTED** | +0.0045 ± 0.0007 CLEARED | 0.037 |
| benign Q:0 | +0.0222 ± 0.0058 CLEARED | +0.0230 ± 0.0088 CLEARED | +0.0018 ± 0.0070 CLEARED | +0.0049 ± 0.0063 CLEARED | 0.010 |
| phishing P:1 | +0.0695 ± 0.0039 **CONVICTED** | +0.0734 ± 0.0034 **CONVICTED** | +0.0635 ± 0.0038 **CONVICTED** | −0.0046 ± 0.0028 CLEARED | 0.021 |
| phishing P:0 | +0.0350 ± 0.0041 CLEARED | +0.0391 ± 0.0041 UNDECIDED | +0.0291 ± 0.0032 CLEARED | −0.0102 ± 0.0032 CLEARED | 0.014 |
| phishing Q:1 | +0.1084 ± 0.0045 **CONVICTED** | +0.1539 ± 0.0069 **CONVICTED** | +0.1701 ± 0.0040 **CONVICTED** | +0.0092 ± 0.0051 CLEARED | 0.283 |
| phishing Q:0 | +0.0097 ± 0.0032 CLEARED | +0.0121 ± 0.0023 CLEARED | +0.0048 ± 0.0041 CLEARED | −0.0005 ± 0.0050 CLEARED | 0.010 |
| benign P:1 Q:1 | +0.0735 ± 0.0020 **CONVICTED** | +0.0781 ± 0.0053 **CONVICTED** | +0.0790 ± 0.0027 **CONVICTED** | +0.0125 ± 0.0065 CLEARED | 0.047 |
| benign P:1 Q:0 | +0.0177 ± 0.0026 CLEARED | +0.0161 ± 0.0044 CLEARED | +0.0117 ± 0.0033 CLEARED | +0.0043 ± 0.0040 CLEARED | 0.014 |
| benign P:0 Q:1 | +0.0370 ± 0.0058 UNDECIDED | +0.0413 ± 0.0089 UNDECIDED | +0.0340 ± 0.0057 CLEARED | +0.0093 ± 0.0056 CLEARED | 0.064 |
| benign P:0 Q:0 | +0.0660 ± 0.0028 **CONVICTED** | +0.0692 ± 0.0052 **CONVICTED** | +0.0350 ± 0.0018 CLEARED | +0.0013 ± 0.0036 CLEARED | 0.020 |
| phishing P:1 Q:0 | +0.0754 ± 0.0045 **CONVICTED** | +0.0828 ± 0.0026 **CONVICTED** | +0.0695 ± 0.0039 **CONVICTED** | −0.0006 ± 0.0036 CLEARED | 0.041 |
| phishing P:0 Q:0 | +0.0365 ± 0.0053 UNDECIDED | +0.0386 ± 0.0076 UNDECIDED | +0.0339 ± 0.0067 UNDECIDED | −0.0044 ± 0.0058 CLEARED | 0.023 |

(x ± SE, section-26.1 verdict. Tilt bound = (1 − a)/a from CR1's first pass.)

Three results, each read before any SC1 generation.

1. **CR1's SUF is the table's.** In the 13 states where CR1's joint tilt is
   bounded at 0.064 or less, CR1's measured SUF equals the shipped table's
   ceiling within noise. The largest gap is 0.0074 (phishing P:1 Q:0), and the
   ceiling involves no model. The model's rendering adds nothing detectable to
   SUF, consistent with 2.4. The exception is phishing Q:1: joint resampling
   rejects 22% there, and its tilt pulled CR1 down to +0.108 from the table's
   +0.154.
2. **Both hypotheses carry weight, in different states.** From S to C isolates
   the frame source (H2); from C to T isolates the state (H1).
   - **Benign P:0 and P:0 Q:0:** frame −0.034 and −0.034, state −0.028 and
     −0.034. About half each.
   - **Phishing P:1, P:1 Q:0, Q:1 and benign Q:1, P:1 Q:1:** frame at most
     ±0.016, state −0.053 to −0.161. The state.
3. **T clears SUF in every state**, from −0.010 to +0.013; the largest upper
   bound, benign P:1 Q:1, is 0.031. CLEARED is attainable in all three primary
   states.

The ceilings are **predictions** for arms C and T, not results. They assume the
generator's parsed suffixes follow its table: 2.4 supports that, and V4 checks
it on every file.

### 2.6 Which comparisons pair exactly

`discriminate_with_floor` draws the three real samples, `rng.sample(pool, 3n)`,
*before* the generated sample. Two arms scored with the same split seed therefore
see identical real samples and identical floors **if and only if their n is
equal**, and then the floor cancels exactly in a difference. n is per
generation: min(1,200, ⌊real field values / 3⌋, generated field values).

In CR1, the cells limited by the generated count were:
- benign SUB in six states (P:1, P:0, Q:1, Q:0, P:1 Q:0, P:0 Q:0);
- phishing Q:0 PTH.

**Every cell in a registered test family below was capped by 1,200 or by the
real pool, never by the generated count**, so it pairs exactly unless an arm's
field rate changes drastically. Section 5.2 handles the cells that do not pair.

### 2.7 Cost

CR1 took 107 min on a Kaggle T4 for 70 files. Its first-pass rates run from
77.9% (phishing Q:1) to 99.4% (benign P:1). Held mode costs the same where
rejection is rare. Where it is not, it costs the harmonic mean of 1/h(suffix):
HC2 measured 1.65 draws per kept URL for phishing `<Q:1>` held, against 1.27
joint. Estimate for SC1's two arms: 3.5–4 GPU-hours, and about 7.5 CPU-hours of
scoring.

### 2.8 What the review fixes, and what it could not tune

- The table construction was fixed by the owner's decisions and section 2.1.
  **No option was left open for the review to tune**: no smoothing, no minimum
  count, no source choice.
- The review fixes **the predictions** (section 4, from 2.5), **the
  manipulation check** (V4, from 2.4) and **the pairing rule** (5.2, from 2.6).
- No held-out data enters any table. The held-out pools were read only to test
  the premise (2.2) and coverage (2.3), and to measure the ceilings (2.5).

---

## 3. Intervention, arms, generation

### 3.1 The tables

- **Source:** `data/bundle2/name_train.jsonl.gz` (benign) and
  `phishing_name_train.jsonl.gz` (phishing). Nothing else is read to build a
  table.
- **A row's suffix** is `parse(row["url"]).public_suffix`; rows with an empty
  suffix are skipped; its class is the row's `label`. This is the shipped
  construction (2.1).
- **A row's state** is the gate's: `FIELDS["PTH"]` and `FIELDS["QRY"]` on its
  URL string.
- **T:** one table per measurable state, counting the class's rows in that
  state. A single-control state conditions on its one field only.
- **C:** one table per class, counting all the class's rows.
- **Raw counts:** no smoothing, no minimum count, no pruning.
- **Written to** `data/sc1/tables.json` as `{"C": {label: {suffix: count}}, "T":
  {tag: {label: {suffix: count}}}}`. Its sha256 enters this registration's
  status and every provenance file. Its row totals must equal section 2.3's.

### 3.2 Arms

| arm | table | resampling | states |
|---|---|---|---|
| **CR1** (existing, read-only) | shipped, all `*_train` rows, class marginal | joint | 14 |
| **C** (matched control) | `name_train`, class marginal | held | 14 |
| **T** (intervention) | `name_train`, the requested state | held | 14 |

- **T − C isolates the state conditioning**, the one variable between them.
- **C − CR1** is descriptive: the frame source and the mode together.
- **T − CR1** is the safety comparison against the pipeline in use.

Held mode is exact for p_table(s) · p_E7(url | s, control, state) and was proved
equal to HC1's exact arm (HC2 R2). Joint mode would tilt a state table again by
P(state | s, control) and count the state twice.

### 3.3 Generation

CR1's call, with three arguments changed:

    generate_with_presence(model, tokenizer, label=label, n=3000,
        table=SuffixTable({label: <arm table>}), fields=state,
        seed=<arm seed>, hold_suffix=True, max_rounds=100, stats=stats,
        decoder="temperature", decoder_kwargs={"temperature": 1.0},
        field_bias={"PTH": {tokenizer.eos_id: -DELTA[label]}}, device=cuda)

Frame `domain` and batch size 32 are the defaults, as in CR1. The run happens on
a Kaggle GPU from a hashed code bundle, and the notebook asserts the checkpoint
and table hashes. A `PresenceUnmet` is recorded as that file's error and never
retried with other settings.

### 3.4 Seeds

State s is CR1's index (0–15, CR1's order); generation g is 0–4. In CR1's
blocks of 10⁹, CR1 used offsets 0 to about 1.0 × 10⁸: 100 rounds × 1,000,003,
plus the suffix groups. SC1 uses disjoint offsets in the same blocks.

| use | seed |
|---|---|
| arm C | 20261001 + (100·s + g)·10⁹ + 3·10⁸ |
| arm T | 20261001 + (100·s + g)·10⁹ + 6·10⁸ |
| T replication (section 7.4) | 20261001 + (100·s + g)·10⁹ + 8·10⁸ |
| scoring | CR1's: SEED 20260806, split seeds v2 |

### 3.5 Sample sizes

- Per arm and measurable state: G = 5 generations × 3,000 compliant URLs.
- Scored at n = min(3,000, 1,200) per draw, reduced by the pools, D = 10.
  Identical to CR1.
- Power: CR1's SUF SEs are 0.0020–0.0058, so SE × √2 ≤ 0.0082 even if pairing
  removed no noise at all. Against that, the ceilings predict Δ(T − C) of
  −0.068, −0.070 and −0.161 in the primary states (|t| ≥ 8) and −0.028 to
  −0.067 in the secondary ones (|t| ≥ 3.4).

---

## 4. Hypotheses and registered predictions

**H_SUF (primary).** In controlled generation, SUF tells arise because the
suffix is drawn without the requested state. Drawing it from the state's
domain-frame training distribution (T) removes them, relative to the same draw
without the state (C).

**H_DOWN against H_INDEP (key secondary).** Phishing's LBL and SUB tells in the
path and query states follow the suffix mix (H_DOWN), or do not (H_INDEP). The
tokenizer emits `[SUF]` before `[LBL]`, so the label is generated knowing the
suffix. Whether that dependence carries the tells is the open question.

**Safety.** T introduces no new tell against the pipeline in use (CR1).

**Registered predictions** (from 2.5; these are not criteria):

- **P1 — each arm realises its table.** In every state, each arm's SUF x lies
  within ±0.02 of its own table's ceiling: T CLEARED everywhere; C as the C
  column of 2.5. A miss means the generator does not deliver its table in that
  state. It is reported and read against V4.
- **P2 — the contrast.** Δ_SUF(T − C) is close to ceiling_T − ceiling_C:
  - primary: −0.068 (phishing P:1), −0.070 (P:1 Q:0), −0.161 (Q:1);
  - secondary: −0.028 (benign P:0), −0.034 (P:0 Q:0), −0.067 (P:1 Q:1).
- **P3 — the control arm against CR1.** C − CR1 on SUF follows the frame source
  where the tilt is negligible: about −0.030 in benign P:0 and −0.031 in benign
  P:0 Q:0. It is **positive in phishing Q:1**: about +0.06 (0.170 against
  0.108), because held mode removes CR1's joint tilt there. That is a property
  of the control arm, not a harm; safety is T − CR1.
- **No prediction on LBL, SUB, PTH or QRY.** That is what SC1 is run to
  measure.

---

## 5. Scoring and statistics

### 5.1 Scoring

- Each arm is scored by `research/cr1/cr1_score.py`, **unchanged**:
  `--gens data/sc1/{C,T}/gen --out data/sc1/{C,T}`. Its report still names
  CR1's registration (hard-coded); here it is read as the instrument's output.
- The paired statistics, flags and diagnostics come from
  `research/sc1/sc1_compare.py` and are written to
  `data/sc1/sc1_report.json`.

### 5.2 The contrast between two arms, per (state, field)

**Paired, when n is equal in all five generations** (recomputed from the files
and the pool exactly as `discriminate_with_floor` computes it):

    δ_g = mean over d of excess_A(g, d) − mean over d of excess_B(g, d)
    Δ = mean δ_g;  SE = sd(δ_g)/√5 (sample sd);  t = Δ/SE, df 4
    95% CI = Δ ± 2.776·SE;  p two-sided from t(4)

`excess(g, d)` is the scorer's `excess_draws`, generation-major.

**Unpaired otherwise:** Δ = x_A − x_B (section-26.1 x), SE = √(SE_A² + SE_B²),
the same t, df 4 (conservative) and CI, and the cell is marked unpaired.

**Pooled rows** are compared descriptively, plus K2's verdict flag (7.3).

### 5.3 Multiplicity

Holm–Bonferroni at α = 0.05 within each family, on the two-sided p. A supported
reduction also needs Δ < 0.

| family | tests |
|---|---|
| **Primary** | SUF, T − C, in phishing P:1, P:1 Q:0, Q:1 (3) |
| **Secondary SUF** | SUF, T − C, in benign P:0, P:0 Q:0, P:1 Q:1 (3) |
| **H_DOWN** | LBL and SUB, T − C, in phishing P:1, P:1 Q:0, Q:1 (6) |

Safety is **per cell, uncorrected, with replication** (7.4): a false alarm
costs one replication, a missed harm costs a bad adoption.

---

## 6. Validity — technical conditions, checked before any contrast is read

Any failure is fixed under a dated amendment and the affected files are
regenerated. It is never read as a result.

- **V1 files.** 140 files (14 states × 5 × 2 arms), none for phishing P:1 Q:1 or
  P:0 Q:1, 3,000 URLs each. Their sha256 match the provenance, as do the
  checkpoint sha256 and the table sha256 (3.1), and the code bundle matches its
  manifest.
- **V2 states.** Every URL honours its state as `FIELDS` reads it.
- **V3 identity.** `cr1_verify.identity_check` in held mode, with the arm's
  table, on benign P:1, phishing Q:1, benign P:1 Q:0 and phishing P:0 Q:0,
  generation 0, both arms. At least 250 URLs compared per state, all
  byte-identical to a CPU regeneration.
- **V4 manipulation.** In every file, the multiset of parsed suffixes agrees on
  **≥ 99% of URLs** with the multiset `table.sample(label, 3000,
  random.Random(seed))` draws from the registered table and seed. Measured
  parse loss: ≤ 0.33% (2.4). This verifies the table *and* the seed. A mix
  statistic could not, since benign P:1's T and C tables are nearly identical.
- **V5 tables.** The committed builder, run on the laptop, reproduces the Kaggle
  copy byte for byte, with section 2.3's row totals.
- **V6 S0′.** `cr1_score.score` and `pools`, called on CR1's generations for
  benign P:0 and phishing P:1, reproduce `data/cr1/cr1_report.json` for those
  states exactly: every field's draws and acceptance block, and the pooled
  rows. This proves the laptop's environment still yields CR1's floors, which
  T − CR1 pairs on.

---

## 7. Criteria

### 7.1 Primary — H_SUF

- **Supported in a state:** Δ_SUF(T − C) < 0 with Holm-adjusted p < 0.05,
  **and** T's SUF section-26.1 verdict CLEARED. CLEARED is attainable in every
  state (2.5, point 3).
- **H_SUF supported:** all three primary states. **Partial:** one or two.
  **Refuted:** none, with V4 passing.

The secondary SUF family is read by the same rule, per state.

### 7.2 H_DOWN against H_INDEP

- **Per cell:** H_DOWN if Δ(T − C) < 0 with Holm-adjusted p < 0.05. H_INDEP if
  the 95% CI of Δ lies inside ±0.02 (HC1 H3's equivalence form). Otherwise
  inconclusive.
- **Overall:** **H_DOWN** if at least 4 of the 6 cells support it and none
  supports H_INDEP. **H_INDEP** if at least 4 of 6 support H_INDEP. Otherwise
  **mixed**, reported as such.

### 7.3 Kill

- **K1 — the hypothesis.** H_SUF refuted with V4 passing: the SUF tell is not
  the suffix mix. The suffix line closes; only read-only re-characterisation
  follows.
- **K2 — harm.** Any of these in T against CR1:
  - a cell (LBL, SUF, SUB, PTH, QRY, section-26.1 computable in both) with
    Δ ≥ +0.02 and the 95% CI excluding 0;
  - a CR1-CLEARED verdict turning CONVICTED;
  - a pooled row turning TELL where CR1's was UNDECIDED.

  Each flag is replicated once (7.4) before it counts.
- **K3 — memorisation.** The gate's novelty rate
  (`Gate(known_membership=known_urls bloom).filter`, all 15,000 URLs of the
  state) falls from CR1 to T by ≥ 0.5 points, with the 95% CI of the difference
  excluding 0. E7's baseline: 99.75% benign, 98.75% phishing.
- **K4 — runaways.** The share of URLs of 240+ characters rises from CR1 to T by
  ≥ 0.2 points, with the CI excluding 0.
- **K5 — cost.** Any T request exhausting its 100 rounds (`PresenceUnmet`).
  Reported; that state is not adoptable; max_rounds is never raised afterwards.

### 7.4 Replication of a K2 flag

The flagged state's T arm is generated again with the replication seeds (3.4) —
same G, n and call — and scored the same way. **The flag counts only if the same
cell flags again by the same rule.** No other state is regenerated.

### 7.5 Adoption gate (the pipeline change only)

All of:
- H_SUF supported;
- no T state with SUF CONVICTED;
- no K2 flag surviving replication;
- K3 and K4 not triggered;
- no K5;
- V1–V6 passed.

Then the owner decides. Adoption would mean controlled generation uses the
state-conditioned domain-frame tables with held resampling, the builder moving
into `uie/` under its own tested commit. E5 stays production; E7's promotion is
a separate decision.

---

## 8. Reported, descriptively

- The full CR1-style table for arms C and T (x, SE, t, CI, n, generation means,
  verdicts, pooled rows).
- Δ tables for T − C, T − CR1 and C − CR1, marked paired or unpaired.
- **H2, the frame source:** C − CR1 on SUF, where the joint tilt is negligible
  (P states).
- The TLD mix against real, per state and arm (`cr1_suf_mix.py`'s method).
- Coverage and parse loss per file.
- Novelty, memorised count, distinct-URL rate, and the registrable-domain copy
  rate (label + suffix seen in training).
- First-pass rate, draws per kept URL (harmonic under held mode, so the first
  pass is the marginal), rounds, 240+ share, mean length.
- FULL (auxiliary; saturated in path-off states, CR1 section 4 — no claim is
  read from it there).

## 9. Execution

1. This registration, committed together with `design_review.py`.
2. Code in `research/sc1/`, with tests:
   - `build_tables.py` — its test must rebuild the shipped table exactly;
   - `SC1_generate.ipynb` and `make_code_bundle.py`;
   - `sc1_verify.py` — V1–V5;
   - `s0p.py` — V6;
   - `sc1_compare.py` — tested on synthetic draws.

   `uie/` untouched, suite green. One commit.
3. A CPU simulation of the notebook.
4. Kaggle, arms C and T (the owner runs it). Provenance per file.
5. V1–V5, then V6. Scoring starts only after all pass.
6. `cr1_score.py` on C, then T: laptop, below-normal priority, memory guard.
7. `sc1_compare.py`; replication only if K2 flags.
8. `SC1_RESULTS.md`; this status; push. The owner decides.

Outputs go to `data/sc1/`. Nothing under `data/cr1/`, `data/hc1/`, `data/hc2/`
or `data/e7_eval/` is written.

## 10. Not claimed

- **SUF passing certifies the table, not the model.** SUF is supplied by design
  (DESIGN sections 4.3 and 4.5), and 2.5 predicts it. What SC1 learns about the
  model is H_DOWN and the safety cells.
- **SC1 does not touch QRY or PTH content**, which convict in CR1 for reasons
  outside the suffix.
- Not computable is not safe, and FULL in path-off states measures nothing.
- **No state becomes "safe" by SC1.** CR1's rule (CLEARED on every applicable
  field) is the only one that says so.

## Status

    SC1  registered 2026-09-24 (2d1db7e), owner-approved; design review by
         design_review.py -> data/sc1/design_review/design_review.json
         code 43f8e1e (suite 673 passing); tables data/sc1/tables.json
           sha256 62aa7d9bafd5bb799ec5d9321e9c2aca0dfc2bc7590ba76b8e8877b07e44d453
           (every total and distinct-suffix count = section 2.3)
         Kaggle bundle data/kaggle/uie-sc1-code.zip built from 43f8e1e (33 files);
           notebook simulated in Kaggle's layout: PASS
         generation: Kaggle T4, 2026-09-24, 140 files in 217 min; no
           PresenceUnmet (K5 clear)
         validity: V6 S0' PASS (2 CR1 states reproduced exactly);
           V1-V5 PASS 2026-09-25 -- 140/140 files, every URL in state,
           identity 8/8 byte-identical (273-356 URLs each), suffix agreement
           >= 0.994 in every file, tables rebuilt identical
         sc1_compare.py dry-run on CR1 data (arms identical): every contrast 0;
           a K2 crash on status-only entries fixed before any SC1 number (9ac15cb)
         scored 2026-09-25 00:02-06:18 (C then T), compared 06:21 -> SC1_RESULTS.md:
           H_SUF SUPPORTED (3/3; secondary 3/3); H_DOWN (6/6); K1-K5 none;
           ADOPTION GATE MET -- the decision is the owner's
           watch (not kills): PTH in the Q:0 states; phishing path-state novelty
         ADOPTED by the owner 2026-09-25: uie/controlled.py (3b3ae24) --
           StateSuffixTables + generate_controlled; adopted tables
           data/state_suffix_tables.json sha256 0c6f2760...13cc9 (= SC1's arm-T
           tables in all 14 measured states); suite 681 passing; the library call
           regenerated arm T's benign_P0_g4.json byte for byte on CPU
