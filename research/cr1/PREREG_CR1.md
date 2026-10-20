# CR1 — the absolute realism of controlled output: pre-registration

**Registered:** 2026-09-24, before any CR1 generation or scoring.
**Requested and scoped by the owner (2026-09-24):** for every control state,
the compliant controlled URLs scored against real URLs **in the same state**,
on every applicable field, as absolute excess — and whether each is safe under
the gate's rule. Both families of states; generation on a Kaggle GPU, scoring on
the laptop. **Descriptive:** no adoption decision follows from it by itself.
**Frozen:** `data/best_e7.pt`, E5, R4, gate v2 and its thresholds, and every
published result, including E7's and HC1's; HC2's run is not touched.

## 1. States

| family | per class | states |
|---|---|---|
| single control (the other `?`) | P:1, P:0, Q:1, Q:0 | 8 |
| combined | P:1 Q:1, P:1 Q:0, P:0 Q:1, P:0 Q:0 | 8 |

**Not measurable — registered before anything was generated.** The gate draws
three disjoint real samples of n per measurement (one URL per host) and needs
n ≥ 250. Held-out real phishing URLs with a query and a path number 669
(n ≤ 223), and with a query and no path 399 (n ≤ 133). These two states are
neither generated nor scored. benign P:0 Q:1 (1,802 real URLs) is measured at
reduced n; where a single field cannot reach 250, the gate's own function says
so and it is reported as not computable.

## 2. Generation — Kaggle GPU, `research/cr1/CR1_generate.ipynb`

E7 with presence enforced by resampling (`uie.presence.generate_with_presence`,
**mode joint** — HC2's adopted default: its R3 kept joint, and on
phishing `<Q:1>` joint's TLD mix beat held by Δ −0.040 on SUF), through the canonical generation
path (marginal suffix table, frame `domain`, `temperature` 1.0, adopted delta).
Per measurable state: **G = 5 generations × 3,000 compliant URLs**, so every
field the state allows reaches the gate's 1,200 rows per draw or close to it —
SUB, the rarest (34–81% of real URLs by state), reaches about 1,000 in the
sparsest benign states and is reported with its n. A field the state leaves
free and rare (QRY under P:1, say) falls to the gate's pooled fallback, as it
does in the canonical gate. Seed of state s (the order of section
1), generation g: 20261001 + (100·s + g) × 10⁹ — disjoint blocks; a request's
redraw rounds and suffix groups never leave their block.

**Before scoring**, `research/cr1/cr1_verify.py` must PASS: every registered
file present and none for the two excluded states, 3,000 URLs each, hashes and
seeds as the provenance records, the laptop's checkpoint and suffix table,
**every URL in its state**, and byte identity against a CPU regeneration of the
first pass for four states (50% gross-mismatch floor).

## 3. Scoring — gate v2's own instrument

`research/cr1/cr1_score.py` runs the canonical gate's scoring loop, calling its
own functions — `field_report`, `split_seed` (v2), `_aggregate`,
`s261_verdict`, the pooled fallback — with one change: **the real pools (name
and structure) are restricted to the state.** Fields: LBL, SUF, SUB, PTH, QRY
where the state allows them; FULL is reported and, as in the gate, auxiliary.
D = 10 splits, n = min(3,000, 1,200) per draw, reduced where the real pool is
smaller, MIN_N 250, split seeds the gate's (seed 20260806).

**S0 — the scorer is the gate (check before any CR1 number).** Given the
canonical gate's own inputs — unrestricted pools and the ten generations HC1's
C2 scored — the scorer must reproduce the gate v2 report for E7 phishing
(`data/hc1_c2_e7_phishing/eval_report.json`) exactly: every field's x, SE,
t and verdict, and the pooled QRY row.

## 4. Reading

Per state and field: x, SE, t, n and the section-26.1 verdict against the 0.05
bar. **"Safe" means CLEARED**, as for E5; UNDECIDED is not passing; CONVICTED
is a tell. A field the state excludes (PTH under P:0, QRY under Q:0) is not
applicable. Pooled rows can convict, never acquit. **No threshold, bar or rule
is changed; no published number is re-read.** A state's verdicts describe that
state; adoption of controlled generation remains the owner's decision.

## Status

    CR1  registered (abb1cba); code (b79854f)
         S0 PASS 2026-09-24: the scorer reproduced the gate v2 E7 phishing
           report exactly -- 73 values (s0_compare.py)
         generation: Kaggle T4, 70 files in 107 min; VERIFY PASS (210,000
           URLs in state; 4 states byte-identical to a CPU regeneration)
         scored 2026-09-24 -> CR1_RESULTS.md: no state safe on every field
           (of 59 cells: 23 CLEARED, 9 UNDECIDED, 23 CONVICTED, 4 not computable)
