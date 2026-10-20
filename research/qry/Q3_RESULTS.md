# Q3 results — the query tell is SHAPE, not vocabulary

> **FOLLOW-UP: see Q3_1_RESULTS.md.** This document is preserved **unchanged** as
> the original Q3 record. Q3.1 re-ran the ladder at D=12 (df 11, t_crit 2.201 —
> the same rule at higher resolution) and reached **DISTRIBUTED**: parameter-name
> identity (17-28%), parameter count (11-33%) and the long-value tail (3-6%) each
> replicate in both arms, none reaching 50%. `skeleton` sufficiency replicates.
>
> **One metric below is WITHDRAWN:** the `name type-token ratio` row compares
> 12,952 real parameters against 758 generated, and TTR falls with sample size.
> At equal n the direction REVERSES — real 0.6023 vs generated 0.3626. See
> Q3_1_RESULTS.md section 5. Every other number here stands.

**Run:** 2026-09-22. `data/qry/q3_benign.json`, `data/qry/q3b_namefix.json`,
logs alongside. Frozen `best_e5.pt`; no generator, decoder, checkpoint or gate
code touched.

**Scope, per registration:** query-bearing rows only, both sides. Real = 7,253
benign query strings (host-deduped `name_val`). Generated = the two Q4 arms,
`typical` 476 and `temperature` 507. n = 476 / 507, three split seeds, floor
recomputed under every transform.

---

## A — component distributions (no discriminator, cannot be gamed)

| metric | real | typical | temperature |
|---|---:|---:|---:|
| param count, mean | 1.786 | 1.592 | 1.430 |
| param count >= 3 | 17.22% | 13.03% | 8.48% |
| name length, mean | 6.728 | 5.313 | 5.199 |
| **value length, mean** | **11.2** | **6.46** | **6.80** |
| value length, median | 4.0 | 4 | 3 |
| **value length, p90** | **25** | **15** | **15** |
| value purely numeric | 42.75% | 55.59% | 54.53% |
| value purely alpha | 25.10% | 23.69% | 22.59% |
| **percent-encoded queries** | **9.89%** | **3.15%** | **5.52%** |
| value entropy, mean | 1.8647 | 1.6605 | 1.6546 |
| ~~name type-token ratio~~ **WITHDRAWN** | ~~0.3086~~ | ~~0.3562~~ | ~~0.4648~~ |
| **repeated-name queries** | **0.57%** | **3.57%** | **0.59%** |

**The median value length is identical (4 vs 4) while p90 is 25 against 15.** The
model is not writing short values wrongly — it is missing the long tail. The same
shape appears in parameter count, name length and percent-encoding (3x under).

**Vocabulary is not the problem.** The commonest names coincide:
real `id, page, p, page_id, lang, m, cat, c, utm_source, bo_table`;
generated `id, page, lang, p, route, product_id, path, page_id, view, cat`.

**One decoder-specific artifact:** `typical` repeats a parameter name in 3.57% of
queries against real's 0.57% — a 6x excess — while `temperature` sits at 0.59%,
matching real. Truncation re-selects the same name; this is not a model defect.

---

## B — ablation ladder, paired by split seed

Differences are `excess(transform) − excess(identity)` per seed, so they pair.
**2 df: |t| > 4.30 required.** Baseline: typical +0.1294, temperature +0.0795.

| transform | component | typical R_T | t | temperature R_T | t |
|---|---|---:|---:|---:|---:|
| `null` | control | 1.00 | — | 1.00 | — |
| `mask_names` | name identity **+ intra-name punctuation** | 0.66 | −10.80 | 0.66 | −19.01 |
| `mask_name_chars` | name identity **only** | 0.72 | −1.91 | 0.69 | −1.49 |
| `mask_values` | value composition | 1.17 | +0.76 | 1.30 | +1.62 |
| `fixlen_values` | value length | 1.02 | +0.14 | 1.01 | +0.20 |
| `mask_digits` | digits | 1.09 | +0.67 | 1.05 | +0.53 |
| `mask_alpha` | letters | 0.95 | −0.24 | 0.74 | −1.05 |
| `mask_encoding` | encoding identity | 1.03 | +2.92 | 0.93 | −4.48 |
| `sort_params` | ordering | 1.02 | +0.75 | 0.98 | −0.44 |
| **`skeleton`** | **structure + char class (isolating)** | **1.11** | **+0.41** | **1.06** | **+0.23** |
| `structure_only` | count + separators + lengths (isolating) | 0.84 | −0.64 | 1.11 | +0.67 |
| `count_only` | parameter count | 0.39 | — | 0.48 | — |

`count_only` came back **VOID degenerate** in both arms (distinct fraction 0.013 /
0.012) — collapsing every query to `a=a&a=a` leaves almost no unique strings. Its
R_T is not interpretable and is not counted. The pre-registered kill criterion
fired exactly as designed, on the transform built to be coarsest.

### The finding

**`skeleton` retains the full tell in both arms.** Mapping every letter to `a`
and every digit to `0`, keeping punctuation and lengths, changes excess by
+0.0147 (t = +0.41) and +0.0048 (t = +0.23). You can destroy **all** lexical
content and lose nothing.

> **Shape is SUFFICIENT.** Parameter count, separator layout, name and value
> length profiles and character-class arrangement reproduce the entire
> discriminative signal without a single real character surviving.

**No other component carries a detectable share.** Ordering, percent-encoding
identity, value length, digit identity, letter identity and value composition are
all null in both arms.

### A correction, and a limit on what this can resolve

`mask_names` was **confounded** and its t = −10.80 should not be quoted. It
rewrites a name as `"a" * len`, which also deletes intra-name punctuation —
`utm_source` becomes `aaaaaaaaaa`, not `aaa_aaaaaa`. Three of the ten commonest
real names carry an underscore. The conservation tests missed it because they
asserted name length and `set(k) <= {"a"}` without asserting that character
classes inside the name survived; `tests/test_qry_transforms.py` now pins both,
and `mask_name_chars` removes identity and nothing else.

Clean result: the point estimate barely moves (−0.0436 -> −0.0363 typical,
−0.0267 -> −0.0243 temperature), so the punctuation confound was only ~17% of
it — but the clean reduction is **not significant** (t = −1.91, −1.49).

The gap is not a stronger effect, it is a smaller variance estimate:
`mask_names` happened to draw three tightly-clustered seeds. **With 3 seeds an SE
has 2 df and is itself unstable**, so a t built on it swings wildly. The honest
statement:

> Parameter-name identity plausibly contributes ~30% of the tell — the point
> estimate is consistent across two independent arms and two transforms — but it
> is **NOT established** at the registered threshold. Three split seeds cannot
> resolve an effect of this size.

---

## C — emission and content kept separate

Nothing in this experiment touched a URL without a query, and nothing here is
evidence about emission rate. The two axes remain separate, per Q4, which showed
the content defect is not a selection artifact of the decoder (+0.1124
untruncated against +0.1325 truncated, paired t = −1.64).

The two arms agree on every qualitative conclusion, which is the replication that
matters: `typical` and `temperature` were harvested from independent generations
at different rates and give the same ladder.

---

## D — what this answers, and what it does not

**Answers** the second half of the phase question. Generated queries are
distinguishable because their **shape** is wrong — too few parameters, names and
values too short, a missing long tail (p90 15 vs 25), too numeric, under-encoded
— and shape alone reproduces the entire tell.

**Does not answer:**

1. **Which shape property dominates.** `skeleton` proves shape is sufficient;
   it does not rank count vs name length vs value length vs char class.
   `structure_only` (0.84 / 1.11) and `fixlen_values` (1.02 / 1.01) disagree
   between arms and are individually null.
2. **Whether name identity really contributes ~30%.** Underpowered, see above.
3. **Whether phishing behaves the same.** Q3 ran on benign only. Phishing's
   emission mechanism already differs (Q2b attachment ratio 0.32 vs benign 1.01),
   so content may differ too.
4. **Whether the shape deficit shares a cause with the path-length deficit.**
   Both are "missing long tail" failures and E3_REPORT records a path-length
   deficit, but no measurement links them.

---

## Status

    Q1  COMPLETE  emission: 3 terms, model-dominant, decoder amplifies
    Q2  COMPLETE  the query DECISION is healthy; prefixes are not
    Q2b COMPLETE  benign drift structural (1.01); phishing probabilistic (0.32)
    Q3  COMPLETE  content tell is SHAPE; lexical identity not required
    Q4  COMPLETE  content is NOT a selection artifact; axes separable
