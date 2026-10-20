# PTH ATTRIBUTION — FROZEN RECORD

**Status: CLOSED.** Decision C: the attribution branch is closed and the residual
tell documented. No generator, loss, sampling-policy or acceptance-gate change was
made on the basis of this study.

## Instrument
`discriminate_with_floor` (char_wb 2-5 gram TF-IDF -> logistic regression, 5-fold CV),
unmodified. Ablation acts on the discriminator's INPUT: deterministic transforms
applied identically to real and generated path bodies, with the floor (real-vs-real)
RECOMPUTED under every transform. Cached generations, E5 + adopted delta
(benign 0.22 / phishing 0.30), G=5 generation seeds x D=3 split seeds, n=1800 requested.

Validity criteria, pre-registered: floor within [0.42, 0.55]; distinct fraction >= 0.50.

## Baseline residual tell (unchanged, reproduced exactly)

| class | excess | per-generation |
|---|---|---|
| benign | **+0.0813** | 0.0855, 0.0773, 0.1009, 0.0704, 0.0723 |
| phishing | **+0.0641** | — |

## Full ablation ladder

### benign

| transform | family | direction | excess | R_T | AUC-floor | distinct | validity |
|---|---|---|---:|---:|---:|---:|---|
| `identity` | -- | baseline | +0.0813 | 1.00 | 0.492 | 0.89 | VALID |
| `null` | -- | null control | +0.0813 | 1.00 | 0.492 | 0.89 | VALID |
| `scramble` | F1 | destructive | +0.0716 | 0.88 | 0.492 | 0.89 | VALID |
| `lower` | F3 | destructive | +0.0813 | 1.00 | 0.492 | 0.89 | VALID |
| `digitmask` | F3 | destructive | +0.0887 | 1.09 | 0.492 | 0.88 | VALID |
| `alphamask` | F1/F3 | destructive | +0.0851 | 1.05 | 0.510 | 0.70 | VALID |
| `flatten` | F4 | destructive | +0.0762 | 0.94 | 0.490 | 0.88 | VALID |
| `punctmask` | F4 | destructive | +0.0805 | 0.99 | 0.493 | 0.89 | VALID |
| `fixlen` | F5 | destructive | +0.0603 | 0.74 | 0.500 | 0.86 | VALID |
| `noext` | F6 | destructive | +0.0773 | 0.95 | 0.494 | 0.89 | VALID |
| `edges` | F6 | isolating | +0.0528 | 0.65 | 0.493 | 0.84 | VALID |
| `skeleton` | F3/F4/F5 | isolating | +0.1013 | 1.25 | 0.500 | 0.69 | VALID |
| `shape_only` | F4/F5 | isolating (neg control) | +0.0668 | 0.82 | 0.494 | 0.37 | INVALID: distinct 0.37<0.5 |
| `types` | F2 | CONFOUNDED | +0.0375 | 0.46 | 0.437 | 0.94 | VALID |
| `novel` | F2 | destructive | +0.0768 | 0.94 | 0.424 | 0.99 | VALID |
| `nmatch` | F2 | power control | +0.0538 | 0.66 | 0.496 | 0.91 | VALID |

### phishing

| transform | family | direction | excess | R_T | AUC-floor | distinct | validity |
|---|---|---|---:|---:|---:|---:|---|
| `identity` | -- | baseline | +0.0641 | 1.00 | 0.487 | 0.66 | VALID |
| `null` | -- | null control | +0.0641 | 1.00 | 0.487 | 0.66 | VALID |
| `scramble` | F1 | destructive | +0.0478 | 0.75 | 0.504 | 0.66 | VALID |
| `lower` | F3 | destructive | +0.0641 | 1.00 | 0.487 | 0.65 | VALID |
| `digitmask` | F3 | destructive | +0.0694 | 1.08 | 0.484 | 0.66 | VALID |
| `alphamask` | F1/F3 | destructive | +0.0872 | 1.36 | 0.477 | 0.47 | INVALID: distinct 0.47<0.5 |
| `flatten` | F4 | destructive | +0.0534 | 0.83 | 0.491 | 0.62 | VALID |
| `punctmask` | F4 | destructive | +0.0643 | 1.00 | 0.488 | 0.66 | VALID |
| `fixlen` | F5 | destructive | +0.0397 | 0.62 | 0.499 | 0.59 | VALID |
| `noext` | F6 | destructive | +0.0594 | 0.93 | 0.486 | 0.65 | VALID |
| `edges` | F6 | isolating | +0.0519 | 0.81 | 0.487 | 0.60 | VALID |
| `skeleton` | F3/F4/F5 | isolating | +0.0940 | 1.47 | 0.474 | 0.46 | INVALID: distinct 0.46<0.5 |
| `shape_only` | F4/F5 | isolating (neg control) | +0.0444 | 0.69 | 0.477 | 0.24 | INVALID: distinct 0.24<0.5 |
| `types` | F2 | CONFOUNDED | +0.0223 | 0.35 | 0.406 | 0.90 | INVALID: floor 0.406<0.42 |
| `novel` | F2 | destructive | -0.0001 | -0.00 | 0.424 | 0.96 | VALID |
| `nmatch` | F2 | power control | +0.0338 | 0.53 | 0.497 | 0.73 | VALID |

## F2: the instrument limitation (why repetition is untestable here)

Five designs were attempted. Four manufactured or destroyed signal; the fifth,
the only one that conserves every per-string property, breaks the baseline.

| attempt | what it did | outcome |
|---|---|---|
| `match_ttr` | substituted repeated real segments for other real words | excess +0.324 — **F1 contamination**, manufactured a lexical signal |
| `splice_lex` | gave generated paths real words of identical length | excess +0.370 — substitution again manufactured signal |
| `freq` | segment -> its own frequency count | floor 0.658 inflated, distinct 0.13 — degenerate |
| `rank` | segment -> its frequency rank within its side | floor 0.294, **below chance** — not a baseline |
| `types` | emitted distinct segment TYPES as documents | **CONFOUNDED**: collapses depth to 1, deletes all delimiters, body length becomes segment length, edges become the segment's. A four-family ablation, not an F2 one. Its R_T 0.46/0.35 is **withdrawn**. |
| `novel` | dropped whole rows reusing a segment; survivors kept **verbatim** | conserves depth/delimiters/lengths/edges/words; TTR -> 1.000 both sides. **But destroys the floor.** |

### The `novel` result and its power control

**benign**

| condition | n | AUC | floor | excess | repetition |
|---|---:|---:|---:|---:|---|
| `identity` | 1096 | 0.5734 | 0.4921 | +0.0813 | intact |
| `nmatch` | 597 | 0.5501 | 0.4963 | +0.0538 | intact (power control) |
| `novel` | 597 | 0.5011 | 0.4243 | +0.0768 | REMOVED |

Retention/depth diagnostics under `novel`: depth real 1.768 -> 1.546, fake 1.769 -> 1.504; differential gap shift -0.043. TTR after: real 1.000 / fake 1.000.

**phishing**

| condition | n | AUC | floor | excess | repetition |
|---|---:|---:|---:|---:|---|
| `identity` | 534 | 0.5511 | 0.4871 | +0.0641 | intact |
| `nmatch` | 274 | 0.5308 | 0.4970 | +0.0338 | intact (power control) |
| `novel` | 274 | 0.4238 | 0.4238 | -0.0001 | REMOVED |

Retention/depth diagnostics under `novel`: depth real 1.521 -> 1.422, fake 1.479 -> 1.417; differential gap shift +0.037. TTR after: real 1.000 / fake 1.000.

**Reading.** Under `novel`, AUC and floor collapse *together* — in phishing to the
identical value 0.4238, making excess zero by construction rather than by measurement.
A floor of 0.424 means real-vs-real classifies *worse than chance*, so it is not a
baseline. The `nmatch` control rules out sample size: at the same n, random
subsampling leaves the floor healthy (0.4963 / 0.4970). **The novelty filter itself
breaks the instrument.**

*Hypothesis, untested and marked as such:* document frequency is TF-IDF's operating
principle, so removing repetition removes the structure the model runs on. If true,
every intervention on repetition also disables the instrument — which would explain
all five failures.

## Causal status — FROZEN

| family | status |
|---|---|
| **F1 lexical identity** | **REFUTED as a dominant necessary factor.** `scramble` conserves char multiset, segment lengths and depth exactly; valid floors (0.492 / 0.504); retains R_T 0.88 / 0.75 |
| **F2 repetition** | **UNTESTABLE with this instrument.** Necessity not established; sufficiency not obtainable; counterfactual not obtainable. The corrected destructive transform breaks the real-vs-real floor, so its apparent collapse cannot be read causally |
| **F3 composition** | **Not necessary.** `lower` R_T 1.00; `digitmask` 1.09 |
| **F4 structure** | **Not necessary alone** (`flatten` 0.94 / 0.83; `punctmask` 0.99 / 1.00); **sufficient in benign** under `skeleton` (R_T 1.25, valid) |
| **F5 length** | **Contributory but not cleanly isolated** — `fixlen` R_T 0.74 / 0.62, valid floors, but truncation/padding also destroys content |
| **F6 edge patterns** | **Minor.** `noext` 0.95 / 0.93; isolating `edges` retains 0.65 / 0.81 |

## Methodological findings carried forward

1. **Substitutive transforms manufacture signal.** Only transforms that DESTROY
   information symmetrically on both sides are valid on this instrument.
2. **`real_vs_fake_auc` accepts a `seed` it never uses.** The pipeline is fully
   deterministic, so the 'discriminator seeds' only ever re-split data.
3. **The 15 draws are effectively 3.** Real partitions overlap 95-100% across
   generation seeds, because the generation seed reaches the real side only through n.
4. **Floors must be bounded on BOTH sides.** A sub-chance floor is not a baseline;
   `FLOOR_LOW = 0.42` was added after `rank` returned 0.294.

## Resting point

**E5 + adopted delta** (benign 0.22 / phishing 0.30) remains the current model.
The PTH excess (+0.0813 / +0.0641) stands as a **measured residual tell whose
dominant causal feature is unresolved**. Word quality is ruled out; composition and
delimiters are ruled out individually; structure is sufficient but not necessary;
repetition is untestable. The next generator change should not target any of the
properties enumerated here on the strength of this study.

