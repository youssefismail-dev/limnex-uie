# Q3.1 — pre-registration: which shape property carries the query tell?

**Registered:** 2026-09-22, before any measurement in this file was run.
**Frozen:** `data/best_e5.pt`, `data/bundle2`, the 0.05 bar, the canonical gate,
the decoder and the sampling policy. Nothing in `uie/` is modified.
**Continues** Q3 (research/qry/Q3_RESULTS.md); does not replace it. The original
Q3 ladder, its numbers and its withdrawn transform are preserved as written.

## The state this starts from

> The query-content tell is **shape-driven rather than vocabulary-driven**, but
> the dominant shape component is **unresolved**.

Established by Q3: `skeleton` (every letter to `a`, every digit to `0`,
punctuation and lengths intact) retains the entire excess in both arms
(R_T 1.11 t=+0.41; R_T 1.06 t=+0.23), so shape is **sufficient**. Every
individual component came back null at 3 split seeds, and the one that looked
significant (`mask_names`, t=−10.80) was **confounded** with intra-name
punctuation and withdrawn; its clean replacement gave the same point estimate at
t=−1.91 / −1.49.

Q3's own stated limit: at 3 seeds an SE carries 2 df and is unstable, so the
ladder could not rank components. **That is the limitation this addendum
addresses, and the reason is resolution, not a new methodology.**

## What changes, and what explicitly does not

| | Q3 | Q3.1 |
|---|---|---|
| split seeds | 3 (df 2) | **12 (df 11)** |
| decision rule | two-sided t, p<0.05 | **unchanged** |
| t threshold | 4.303 | **2.201** — the SAME rule at the new df |
| paired-by-seed design | yes | unchanged |
| floor recomputed per transform | yes | unchanged |
| floor bounds / degeneracy | [0.42, 0.55], 0.50 | unchanged |
| arms | typical, temperature | unchanged |
| scope | benign only | **benign only** |

> The threshold moves because **degrees of freedom rise**, not because a more
> favourable cutoff was chosen. `scipy.stats.t.ppf(.975, 2) = 4.303` is exactly
> the 4.30 Q3 used, so the rule is identical and only the data changed.

**Not changed:** the registered Q3 hypothesis, metrics, scope or thresholds; the
generator, decoder, checkpoint, sampling policy or canonical gate; training data
or weights. **No fix is implemented in this phase.**

## Design

**Stage 1** — the full ladder on the `typical` arm (production; what the gate
measures) at D=12.
**Stage 2** — every transform reaching |t| > 2.201 in stage 1 is re-run on the
`temperature` arm at D=12, with identity and null as baselines. **A finding is
promoted only if it replicates in the second arm.** Registering the two stages
in advance is what keeps this from being an adaptive fishing expedition.

### Candidates, and the descriptive measurement motivating each

| transform | component | why it is a candidate |
|---|---|---|
| `clip_values_20/15/10` | long-value tail | real p90 **25** vs generated **15** at an identical median of 4 |
| `first_param` | parameter count | 1.786 vs 1.592 |
| `fixlen_names` | name length | 6.73 vs 5.31 |
| `mask_pct` | percent-encoding **presence** | 9.89% vs 3.15% |
| `normalize_seps` | separator structure | bare flags 3.88% vs 1.98% |
| `mask_punct` | intra-field punctuation | value hyphen 8.95% vs 3.77% |
| `alnum_flat` | digit-vs-alpha distinction | numeric values 42.8% vs 55.6% |
| `mask_values`, `fixlen_values` | value identity / value length | re-run at higher power |
| `mask_name_chars` | name identity | re-run; Q3 left it at t=−1.91 |
| `sort_params` | ordering | re-run |
| `skeleton`, `structure_only` | sufficiency | re-run |
| `clip15_first` | **pre-specified interaction** | run only per the rule below |

`count_only` is **excluded**: Q3 found it VOID degenerate in both arms
(distinct 0.013 / 0.012). Re-running it at higher power would not fix a
degenerate discriminator.

### Reading the pairs

Two comparisons are only interpretable as differences, and are registered as
such rather than read individually:

- `mask_values` (kills value identity, **keeps** length) vs `fixlen_values`
  (kills both) → isolates **value length**.
- `skeleton` (keeps the digit/letter distinction) vs `alnum_flat` (destroys it)
  → isolates **digit-vs-alpha composition**.
- `mask_name_chars` (kills name identity, keeps length) vs `fixlen_names`
  (kills length) → isolates **name length**.

### Necessity, sufficiency, counterfactual

- **Necessity** — a destructive transform must significantly *reduce* excess.
- **Sufficiency** — an isolating projection retaining the property must keep a
  substantial share *without* tripping the degeneracy bound.
- **Counterfactual** — done **destructively, never by transplant**. `clip_values_15`
  caps both sides at the *generated* p90, removing the real side's excess tail
  and leaving generated largely untouched. Substitution is forbidden here: four
  substitutive path transforms produced floors of 0.907, 0.294 and 0.13 and had
  to be discarded.

**No property is promoted to "cause" on one test.** A causal claim needs
necessity *and* (sufficiency or a dose-response), plus replication in arm 2.

### Interaction rule

`clip15_first` is run **only if** `clip_values_15` and `first_param` are each
individually non-significant. It is the single pre-specified combination; no
combinatorial grid is run under any outcome.

### Dose-response

`clip_values_20/15/10` form an ordered series. A real tail effect should grow
monotonically as the cap tightens. A coincidence has no reason to be monotone,
so monotonicity is treated as corroborating evidence and its absence as
disconfirming.

## Kill criteria (unchanged from Q3 / the path work)

- floor outside **[0.42, 0.55]** → row reported **VOID**, not interpreted.
- distinct fraction < **0.50** → **VOID degenerate**.
- `null` must remain a byte-exact no-op, or the harness is misplumbed.
- `identity` must reproduce the Q3 baseline (+0.1294 typical, +0.0795
  temperature) within its own seed spread, or the run is void.

## Decision rule, fixed in advance

**IDENTIFIED** — one property (or the one pre-specified pair) that:
1. reduces excess with |t| > 2.201 at df 11, **and**
2. replicates with |t| > 2.201 in the second arm, **and**
3. accounts for **>= 50%** of identity excess (R_T <= 0.50), **and**
4. where a dose-response series exists, is monotone in it.

**DISTRIBUTED** — no single property meets IDENTIFIED, but **two or more**
properties each reduce with |t| > 2.201 **in both arms**, and no one of them
reaches 50% alone.

**UNRESOLVED** — neither. Reported as such; a repair plan is **not** produced.

> Largest point estimate is **not** a decision criterion. Q3 already produced one
> spuriously significant transform from a fluke-tight variance, and that is
> exactly what this rule exists to prevent.

## Status

    Stage 1  registered, not yet run
    Stage 2  registered, conditional on stage 1
