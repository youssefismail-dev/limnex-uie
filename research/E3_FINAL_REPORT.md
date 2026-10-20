# The E3 programme: final report

- **Programme:** the URL Intelligence Engine, 2026-08-03 to 2026-10-08. 253 commits,
  56 test files, 1,036 tests.
- **Adopted engine:** `e16_b5` (`research/native/DECISION_2026-10-08.md`).
- **Verdict:** **E3 CLOSED, NOT MET.** 29 of 32 gated cells CLEARED; all 8 slash rows
  within tolerance; 3 cells outstanding.
- **Closed by the owner on 2026-10-08**, after the last two candidate routes were
  checked and found empty.

---

## 1. Executive summary

### What E3 asked for

E3 is a deliberately unforgiving finish line, set by the owner on 2026-09-29. It is
claimed only when **both** hold:

1. **every gated cell is CLEARED** — for each field of each population, a detector
   trained to separate the engine's URLs from real held-out URLs must do so no better
   than real data separates from itself, *proven* at the 5% bar with 95% confidence;
2. **the root slash is written at the real rate** on URLs with no path body, in both
   classes, in free and control mode — all 8 rows, within a tolerance of
   `T = 2√(p(1−p)/8000)`.

Nothing here is graded by the engine or by a model. Every cell is graded by **real
held-out URLs**, through a measurement whose floor is established on real-vs-real
comparisons. That constraint was never relaxed, and it is why the result below can be
trusted in either direction.

### What was achieved

| | E11 `e11_plan` | E12 `e12_final` | **`e16_b5` (adopted)** |
|---|---|---|---|
| gated cells NOT cleared | 7 | 6 | **3** |
| slash rows missed | 3 | 3 | **0** |

- **The structural failures are gone.** The slash condition — the owner's second E3
  condition, and the single largest term in the whole-URL detector's edge — is **fully
  solved**. All 8 rows pass, several of them exactly. One earlier engine (`e12_d1p`, E14's
  D1′) also reached 8 of 8, but it was not adopted, and the row degraded again on both
  engines that followed as they learned queries better. B5 closes it *structurally*, so it
  cannot drift back.
- **29 of 32 gated cells are CLEARED**, including every cell in control mode's
  P0Q0, P0Q1 and P1Q0 states for both classes, and 9 of 10 free-mode cells.
- **Plan obedience is near-exact.** Told what to write, the engine honours the request
  99.55–99.81% of the time, and no URL under a query plan now ends before its query.
- **The engine is controllable.** It writes to a supplied plan, places the root slash
  correctly, and honours path/query presence — the properties that make it usable as a
  generator, independent of the gate.

### What is outstanding, stated precisely

Three cells remain, and they are not of one kind. The distinction matters:

| cell | x | bar | status | honest description |
|---|---|---|---|---|
| control benign P1Q1 **SUB** | **0.0431** | 0.05 | UNDECIDED | **Below the bar.** Not a defect — simply not *proven* below it. Needs −0.004 to prove. |
| control benign P1Q1 **PTH** | **0.0539** | 0.05 | CONVICTED | **A minor, fully diagnosed weakness.** Above the bar by 0.004. Needs −0.007 to clear. |
| free phishing **QRY** | **0.0733** | 0.05 | CONVICTED | **The single substantial weakness.** Needs −0.041 — more than any lever in the programme has ever moved it. |

So the "just short" framing is true for two of the three, and genuinely so: SUB sits
below the bar already, and PTH misses it by 0.004. The third is a different matter and
is treated as such in §5.3.

### Why it was closed

Not from fatigue, but because the search space was exhausted in a traceable way. Every
family the programme's own rules permit was tested against these cells:

| family | tested in | outcome on the remaining cells |
|---|---|---|
| exposure (more query practice) | E13-N, E14, E15 | moved the small cells; P1Q1 never responded |
| capacity (a wider model) | E16 | moved P1Q1 QRY for the first time; not enough |
| more data (a second query stream) | E17 | improved fit; moved no tell |
| decode rules (plan violations) | E13, E14, E17 | solved the slash condition completely |
| decode calibration (knobs) | E13 D2 | no measurable effect; closed |
| structural decode rule for PTH | E18 check | **no violation exists to forbid** |

The last row is the one that ended it. On 2026-10-08 a read-only check asked whether the
engine's paths do anything real paths never do — the only kind of thing a decode rule can
honestly forbid. The answer was no, twice over (§5.2). With that, no permitted route to
the remaining cells was left open.

---

## 2. How the engine is graded

Understanding the three outstanding numbers requires understanding what they are.

**The measurement.** For a given field (label, suffix, subdomain, path, query) and a
given population, a detector is trained to tell the engine's strings from real held-out
strings, and its AUC is recorded. But an AUC of 0.5 is the wrong target: two samples of
*real* data do not score 0.5 against each other, because the split is domain-disjoint and
the crawls span two years. Measured on real-vs-real, ordinary variation alone runs 0.51
to 0.79. A bar set at an absolute AUC can therefore sit *below its own floor* and be
unreachable by a perfect generator — which is exactly what happened to an earlier
pre-registered 0.65.

So three disjoint real samples are drawn. Two establish the **floor**; the third is what
the engine is scored against, at identical size. The number that means anything is:

> **x = AUC(real vs generated) − AUC(real vs real)**

x ≈ 0 means indistinguishable, given how variable real data is. x > 0 is a real tell.

**The verdict.** x is measured across 5 independent generations; the bar is 0.05. With 4
degrees of freedom the critical value is 2.776, so:

- **CLEARED** — x is *proven* below 0.05: (x − 0.05)/SE < −2.776
- **CONVICTED** — x is *proven* above 0.05: (x − 0.05)/SE > +2.776
- **UNDECIDED** — neither is proven

This is why SUB and PTH are different despite looking close. SUB is *below* the bar and
unproven (t = −1.79). PTH is *above* it and proven (t = +3.19). They are not two
instances of the same near-miss.

**Not gated.** FULL (the whole-URL detector) is auxiliary by registration and reported
only. Two phishing control states (P0Q1, P1Q1) are registered as not measurable — their
real pools are too small — and carry honouring checks only.

---

## 3. The journey to `e16_b5`

### 3.1 The foundations (E5–E9): making the objective honest

The early engines failed in ways that turned out to be *measurement and objective* bugs,
not model weakness.

- **E8 could not be read at all**, and this was the programme's first real lesson. Before
  any arm could be judged, the registration required a retrained baseline to reproduce
  its predecessor within fixed tolerances. It did not — honouring differed by up to 5.2
  points against a ±1-point tolerance. The registered consequence was absolute: **no arm
  was read, nothing was adopted.** The rule was obeyed rather than argued with, and that
  set the standard for everything after.
- **E9 found the cause of three separate biases in one place.** The training score gave
  every URL the same total weight regardless of length, so each character of a short URL
  counted up to ten times as much as a character of a long one. The engine had learned
  short URLs' habits: end early, stop query values early, add few parameters. With every
  character weighted equally, all three habits matched real data (benign ratios 1.164 →
  1.000, 1.058 → 1.018, 0.892 → 0.997). Free-mode paths were fixed with no patch at all:
  path presence rose 58.7% → 75.6% benign, 29.6% → 38.2% phishing.

**Lesson:** several apparent generation failures were one objective defect. Fixing the
objective was worth more than any amount of tuning.

### 3.2 The plan (E10–E12): making the engine controllable

- **E10** supplied the engine with a *plan* (does this URL have a path? a query?) and
  found that, told what to write, it writes it — 95.0–99.7% honouring against the
  baseline's 85.4–98.1%. URLs written to a supplied plan were also markedly more
  realistic. But left to choose its own plan, it chose wrongly.
- **E11** made presence a trained target. `e11_plan` became the first engine to hit the
  real path and query rates in free mode within tolerance (benign path 73.38% vs 73.37%
  real; query 6.57% vs 6.62%). Reading at a common step 20,000 rather than at mixed
  steps cut CONVICTED cells from four to one.
- **E12** added the slash plan and a longer context. It reached **6 cells and 3 slash
  rows**, with 9 of 10 free cells CLEARED. The slash plan moved all three failing slash
  rows toward real, but none into tolerance.

**Lesson:** supplying and training the plan converted an uncontrollable generator into a
controllable one. This is what makes the engine useful in practice, gate aside.

### 3.3 The diagnostics (E13-D, DC1, QC1, CR1, N2a): finding out *why*

Rather than guess at fixes, the programme stopped and measured the cause of each
remaining cell. This is the part that makes the final report conclusive rather than
merely tired.

- **Every remaining cell reads LEARNING**, replicated on a second model. The data is not
  the problem — real training URLs are indistinguishable from the real validation URLs
  the gate tests against (x between −0.0044 and +0.0054, i.e. no population shift). The
  engine simply has not learned it: its output is as distinguishable from its own
  training data as from held-out data.
- **There is enough data.** A plain character 6-gram model, trained on the same 8,272
  phishing training queries, writes queries the detector barely tells apart from real
  ones: **+0.036, against the transformer's +0.124 at the same sample size.** This is one
  of the programme's most important and least comfortable findings — the limitation is
  the transformer's behaviour, not data scarcity.
- **DX8 split each cell's tell into "the field itself" and "the context the model wrote
  before it".** This is what later made the phishing cell explicable (§5.3).

**Lesson:** the diagnostics repeatedly refuted the intuitive explanation. "Not enough
phishing query data" was the natural hypothesis, and it was measured and found false.

### 3.4 The levers (E13–E17): what was tried, what moved

| experiment | lever | result |
|---|---|---|
| **E13** | D1, a slash decode rule | Every host planned without a path obeyed its slash: phishing's 57 misses → 0; benign `host?q` 70% → 100% obedience. 7 of 8 rows. Not adopted (the rule required all 8). |
| **E13** | D2, field calibration (temperature + end bias, fitted on real held-out data) | **No measurable effect** (Δ +0.0018, −0.0018). Both targets tipped at the boundary. **Closed.** |
| **E13** | C, a context arm | Moved free phishing QRY 0.1141 → 0.1004, just under the 0.0141 needed. Closed. |
| **E14** | Q, 4× query exposure | Free phishing QRY 0.1141 → 0.0830; benign P0Q1 QRY → CLEARED. **But** every path and subdomain cell worsened, and a slash row broke. Rejected. |
| **E14** | D1′, extending the slash rule to path plans | **All 8 slash rows passed** — the first time in the programme. Rejected on a noise-level verdict flip at Δ ≈ 0, and the row later degraded again on the engines that followed. |
| **E15** | a new, diverse benign query stream, injected at a small dose | Benign P0Q1 QRY 0.0624 → 0.0410. **And a surprise: free phishing QRY fell 0.1123 → 0.0774 (−0.035) with no phishing rows injected at all** — the largest move that cell ever made. Adopted (G5 waived). |
| **E16** | capacity: a wider model (d_model 512, 25.5M parameters) | Moved benign P1Q1 QRY for the first time after three exposure levers had failed (0.0500 → 0.0400, CLEARED). Reached **3 cells, 1 slash row.** Adopted (K16-1, G5 waived). |
| **E17** | B5, a query-plan decode rule | **Solved the last slash row by construction**: 0.99987 → 0.99525 against a real 0.99551. Not one of 44,514 query plans now ends before its query (458 did before). **Adopted.** |
| **E17** | option A: a second query stream, doubled dose | Held-out query fit improved by 0.030 nats/char — the top of the predicted range. **No tell moved.** The far cell got *worse*. Not adopted. |
| **E18** | a structural decode rule for PTH | **No violation exists to forbid** (§5.2). Closed. |

### 3.5 Why `e16_b5` is the right engine to stop on

`e16_b5` is `e16_cap`'s weights — the widest, best-fitting model trained — decoded with
`raw + D1 + D1′ + B5`.

- **It is the best engine the programme produced**, on the finish line's own terms:
  3 cells and 0 slash rows, against 6 and 3 for E12.
- **Its decode rules cost nothing and cannot be tuned.** D1, D1′ and B5 forbid only
  things real URLs *never* do — a plan that says "slash" and gets none, a plan that says
  "query" and ends first. Because the true probability of those events is zero, the
  rules move the engine toward real behaviour by construction. They have no fitted
  parameters, so they cannot be tuned to the metric. This is the sharp line the
  programme held: **a rule may forbid an impossibility; a knob fitted to close a gap is
  tuning, and was rejected every time it was proposed.**
- **It is the most controllable.** Near-exact plan obedience plus a guaranteed query
  under a query plan makes its output predictable for a downstream consumer.
- **Its cost was modest:** 3.37 h on a single T4 for training, 8.6 GB peak.

---

## 4. The final results (complete view)

Engine: **`e16_b5`**. Every number from `data/e17/score_report_e16_b5.json`.
`x` is the excess over the real-vs-real floor; `t` is (x − 0.05)/SE against ±2.776.

### 4.1 Free mode (8,000 URLs × 5 generations per class)

| class | field | x | SE | t | verdict |
|---|---|---|---|---|---|
| benign | LBL | +0.0044 | 0.0009 | −49.47 | **CLEARED** |
| benign | SUF | +0.0132 | 0.0033 | −11.13 | **CLEARED** |
| benign | SUB | +0.0048 | 0.0013 | −35.25 | **CLEARED** |
| benign | PTH | +0.0076 | 0.0053 | −7.95 | **CLEARED** |
| benign | QRY | +0.0155 | 0.0058 | −5.97 | **CLEARED** |
| benign | *FULL* | +0.0150 | 0.0028 | −12.44 | *auxiliary* |
| phishing | LBL | +0.0081 | 0.0021 | −19.56 | **CLEARED** |
| phishing | SUF | +0.0094 | 0.0037 | −10.89 | **CLEARED** |
| phishing | SUB | +0.0107 | 0.0037 | −10.49 | **CLEARED** |
| phishing | PTH | +0.0266 | 0.0037 | −6.37 | **CLEARED** |
| phishing | **QRY** | **+0.0733** | 0.0065 | **+3.58** | **CONVICTED** |
| phishing | *FULL* | +0.0146 | 0.0021 | −16.66 | *auxiliary* |

**9 of 10 free gated cells CLEARED.**

### 4.2 Control mode (plan supplied; 3,000 URLs × 5 generations per state)

| state | LBL | SUF | SUB | PTH | QRY |
|---|---|---|---|---|---|
| **benign P0Q0** | +0.0050 ✓ | +0.0051 ✓ | +0.0123 ✓ | — | — |
| **benign P0Q1** | +0.0107 ✓ | +0.0105 ✓ | — | — | +0.0377 ✓ |
| **benign P1Q0** | +0.0036 ✓ | +0.0021 ✓ | +0.0052 ✓ | +0.0109 ✓ | — |
| **benign P1Q1** | +0.0243 ✓ | +0.0046 ✓ | **+0.0431 ?** | **+0.0539 ✗** | +0.0424 ✓ |
| **phishing P0Q0** | +0.0129 ✓ | −0.0042 ✓ | +0.0194 ✓ | — | — |
| **phishing P1Q0** | +0.0161 ✓ | −0.0019 ✓ | +0.0207 ✓ | +0.0294 ✓ | — |

✓ CLEARED · ? UNDECIDED · ✗ CONVICTED · — not computable (the field is absent in that
state, or its real pool is too small)

**20 of 22 control gated cells CLEARED.** Phishing P0Q1 and P1Q1 are registered as not
measurable and carry honouring checks only.

| detail | value |
|---|---|
| benign P1Q1 SUB | x +0.0431, SE 0.0038, t −1.79 → UNDECIDED |
| benign P1Q1 PTH | x +0.0539, SE 0.0012, t +3.19 → CONVICTED |
| benign P1Q1 QRY | x +0.0424, SE 0.0025, t −3.04 → CLEARED (by a margin of 0.0007) |

### 4.3 The slash condition — all 8 rows PASS

| row | engine | real | tolerance T | within? |
|---|---|---|---|---|
| benign, free, bare hosts | 98.092% | 98.004% | ±0.626 | ✓ |
| benign, control P0Q0 | 97.931% | 97.944% | ±0.518 | ✓ |
| benign, free, host+query | 99.372% | 99.576% | ±1.502 | ✓ |
| benign, control P0Q1 | 99.525% | 99.551% | ±0.244 | ✓ |
| phishing, free, bare hosts | 99.991% | 100% | ±0.042 | ✓ |
| phishing, control P0Q0 | 100.000% | 100% | ±0.067 | ✓ |
| phishing, free, host+query | 99.692% | 100% | ±1.544 | ✓ |
| phishing, control P0Q1 | 100.000% | 100% | ±0.200 | ✓ |

**The owner's second E3 condition is fully met.** The benign control P0Q1 row — which
failed on E11, E12, `e15_new` and `e16_cap`, and was waived twice to permit those
adoptions — now sits **about a tenth of the way into its tolerance** (0.026 points out of
0.244).

This row also tells the programme's most instructive small story. It did not fail because
the engine placed slashes badly — D1 placed them exactly. It failed because on the rare
`host?query` plan the engine *dropped the query* and ended the URL, removing it from the
population the row measures. And it got **worse with every engine that learned queries
better**: 99.77% (`e12_d1p`) → 99.86% (`e15_new`) → 99.987% (`e16_cap`) against a real
99.551%. Improving the model made this number worse. Only a structural rule could fix it,
and B5 did.

### 4.4 Presence and obedience

| measure | engine | real |
|---|---|---|
| benign path rate (free) | 73.51% | 73.37% |
| benign query rate (free) | 6.64% | 6.62% |
| phishing path rate (free) | 39.97% | 39.51% |
| phishing query rate (free) | 4.73% | 4.68% |
| plan honouring, benign P0Q0 / P0Q1 / P1Q0 / P1Q1 | 99.553% / 99.573% / 99.807% / 99.687% | — |
| query plans ending before their query | **0 of 44,514** | — |

### 4.5 Scoreboard

| | count |
|---|---|
| gated cells CLEARED | **29 of 32** |
| slash rows within tolerance | **8 of 8** |
| cells UNDECIDED | 1 (below the bar, unproven) |
| cells CONVICTED | 2 |

---

## 5. The final three cells

### 5.1 benign P1Q1 SUB — **not a weakness**

**x = 0.0431, SE 0.0038, t = −1.79 → UNDECIDED.**

This cell **is below the 0.05 bar.** Its point estimate has been below the bar in every
recent engine. It is not CLEARED only because the verdict demands *proof* at 95%
confidence, and with 5 generations at this standard error the measurement cannot deliver
that: it would need x ≤ 0.0395, a further −0.004.

In other words, the shortfall here is in the **precision of the measurement**, not in the
engine. The cell drifted between 0.037 and 0.043 across E15–E17 with nothing ever aimed
at it, which is the behaviour of a quantity sitting near the bar and moving with run-to-
run noise. It was CLEARED on `e15_new`.

**Treated as essentially fine.** The owner closed it with no patch, which is the correct
call: any intervention aimed at a cell already below the bar would be fitting to noise.

### 5.2 benign P1Q1 PTH — **a minor, well-understood weakness**

**x = 0.0539, SE 0.0012, t = +3.19 → CONVICTED.**

Unlike SUB, this one is **above** the bar, and provably so. It needs −0.007 to clear. It
is small and stable, having sat between 0.051 and 0.055 across the last four engines.

**It was checked properly before being closed.** On 2026-10-08, a read-only check
(`research/e18/PTH_CHECK_RESULT.md`) asked the only question a decode rule can act on:
*do these paths do anything that real paths never do?* 1,600 fresh URLs were generated at
seeds deliberately outside the gate's range, and compared against 185,049 real paths, with
the decision rule fixed before any number was seen.

**The answer was no, by two independent routes:**

| | generated | real train | real val |
|---|---|---|---|
| any structural violation (16 properties) | 0.313% | 0.231% | 0.146% |
| paths with a 4-gram unseen in 185,049 real paths | 29.99% | — | 28.65% (control) |

- On all twelve properties where real paths score zero — malformed percent-encoding,
  non-ASCII, whitespace, control characters, backslashes, double slashes, characters
  outside the real 81-character set, over-length, over-depth — **the engine also scores
  zero**, and on several it is *cleaner* than real data.
- The feature-independent scan is the stronger result: the engine invents novel character
  sequences at the same rate genuinely new real data does (30.0% vs 28.7%, a difference
  of ~1.2 standard errors). There is no hidden structural anomaly.
- Everything found is worth about **0.0008** of a 0.007 gap.

**What remains is a matter of degree:** the engine's paths run about 10% shorter (19.69
vs 21.74 characters) and slightly shallower (1.72 vs 1.87 segments). That is precisely
what a field temperature and end-bias would target — and that is **D2, which was fitted
on real held-out data for this exact cell in E13 and moved it −0.0018.**

A knob set by anything other than real data would be tuning the generator to the metric.
So the cell is closed honestly rather than closed by force.

> One genuine plan violation did turn up — 0.19% of URLs dropped the path under a plan
> that asked for one — and it is of exactly the B5 family. But it cannot touch this cell:
> the scorer builds its comparison only from URLs that *have* the field
> (`discriminate_with_floor`), so a path-less URL was never in the PTH comparison.
> Forcing a path there adds paths rather than removing a tell.

### 5.3 free phishing QRY — **the single substantial weakness**

**x = 0.0733, SE 0.0065, t = +3.58 → CONVICTED. It needs −0.041 to clear.**

This is the one cell that is genuinely far from the line, and it is the honest headline
limitation of the engine: *a detector can tell the engine's phishing query strings from
real phishing query strings.* Its history across the programme:

| engine | x | what changed |
|---|---|---|
| E11 `e11_plan` | 0.0953 | — |
| E12 `e12_final` | 0.1141 | slash plan, longer context |
| E13 C (context arm) | 0.1004 | −0.0137, closed |
| E14 Q (4× query exposure) | 0.0830 | −0.031, rejected on cost |
| E15 `e15_new` | 0.0774 | **−0.035 from *benign* data alone** |
| E16 `e16_cap` | 0.0756 | capacity: −0.0018 |
| **`e16_b5`** | **0.0733** | B5: −0.0023 |
| E17 `e17_a` | 0.0802 | twice the data: **+0.0069** |

**Why it remains unsolved — four findings that together explain it:**

**(a) Most of the tell is not in the query at all.** DX8 decomposed the cell and found
**about 63% of it comes from the host and path the engine writes *before* the query.**
Given a *real* host and path, the same engine's phishing queries score **0.046** — close
to the bar. The cell is labelled CONTEXT-LED. This is the central fact: a query-shaped
intervention works on roughly a third of the problem. Even a perfect query would leave
the cell near 0.046, still above its 0.032 clear line.

**(b) The engine is too creative; real phishing is repetitive.** This is the most
counter-intuitive finding of the programme, and it inverts the natural hypothesis. Real
phishing URLs come from kits that reuse the same strings over and over:

| | real | engine (E12) |
|---|---|---|
| phishing queries that are exact training queries | **40.3%** | 16.7% |
| queries never seen in training | 50.6% | **77.2%** |
| paths before a query that are exact training paths | **57.8%** | 24.0% |

The engine invents where reality repeats. A detector finds this easily. Crucially, this
means that **adding more *diverse* phishing data would push the engine further from real
behaviour, not closer** — which is the main reason the LLM-synthetic-data proposal was
assessed as likely to backfire.

**(c) Better fit does not move the tell — diminishing returns, measured.** The cell has
been decoupled from model quality for three consecutive experiments:

| experiment | improvement in held-out phishing query fit | movement in the cell |
|---|---|---|
| E16 (capacity) | **−0.10 nats/char** — the largest fit gain in the programme | −0.0018 |
| E17 (double data) | −0.017 nats/char | **+0.0069** (worse) |

E17 is the decisive one: query fit improved by the top of its predicted range, uptake of
the new strings was at chance level, and the cell moved the *wrong way*. The engine is
fitting the data and still failing to deceive the discriminator.

**(d) It is not a data-volume problem.** A plain character 6-gram model, trained on the
same 8,272 phishing training queries, scores **+0.036** where the transformer scores
+0.124 at the same sample size. The data is sufficient for a simple model to nearly pass.
The limitation is in how the transformer generalises — it learns a smoother, more
inventive distribution than the lumpy, repetitive truth.

**What this means in practice.** The weakness is narrow and well-bounded. It concerns
query strings on phishing URLs, which occur on about 4.7% of free-mode phishing output.
Every other phishing field — label, suffix, subdomain, path — is CLEARED, and the
whole-URL auxiliary reading is +0.0146. ~~The engine is not broadly detectable as fake;
it has one identified signature in one field of one class.~~

> **Corrected 2026-10-08 (§8).** That sentence holds only for character-level detectors.
> A later test with a word-level detector found a second, broader signature: the engine
> uses fewer real words than real URLs do, in both classes. See §8.

---

## 6. Validity

Every result in this report rests on checks that were run before anything was read.

| check | result |
|---|---|
| **V1** completeness | every file, checkpoint, table and seed matches its record |
| **V2** reproducibility | 957 URLs per run regenerated on a separate CPU and compared **byte for byte**; identical, with the decode rules active |
| **V3** no silent change | options off are bit-identical to the previous engine; full suite 1,036 tests |
| **V4** instrument integrity | the scorer reproduces the published reference cells **exactly** (difference 0.00) |
| **V-start** | training began from a start fingerprint rebuilt and matched on the GPU host (`caca0107…`) |
| **V-batch / V-mass / V-stream** | batch digests, injected row counts and stream hashes all exact |

**Methodological commitments held throughout:**

- Every experiment was **preregistered** — criteria, predictions and kill conditions
  written and committed *before* the data existed.
- **Predictions were scored against outcomes**, including the ones that were wrong (E16
  predicted a 10% adoption chance and was not adopted; E17 predicted the far cell would
  improve by 0.005–0.025 and it worsened by 0.0069).
- **No criterion was moved to fit a result.** Where a rule proved unachievable as worded,
  it was amended *before* the data was read and recorded verbatim — as with E17's
  Amendment 2, which was found and adopted before any E17 output was downloaded.
- **Waivers are recorded as waivers**, not absorbed into the result. `e15_new` and
  `e16_cap` were adopted over failing criteria by explicit owner decision, documented
  with the exact miss.

---

## 7. Final verdict

> **E3: CLOSED, NOT MET.**
>
> `e16_b5` clears **29 of 32** gated cells and **all 8** slash rows. Of the three
> outstanding cells, one (SUB, 0.0431) sits below the bar and is unproven rather than
> defective; one (PTH, 0.0539) is a minor weakness, above the bar by 0.004, with no
> honest route left to it; and one (free phishing QRY, 0.0733) is a substantial,
> diagnosed limitation requiring −0.041.

The programme did not meet its finish line. It did something that is arguably worth more:
it established **why** the line was not met, with each remaining gap attributed to a
measured cause rather than left as an unexplained failure. The structural half of E3 —
the owner's own second condition, and the largest single term in whole-URL detectability
— was solved completely.

**What is not claimed.** This report does not claim the remaining cells are unreachable.
It claims that every family the programme's rules permit was tried and measured, and that
the routes which remain would require either reopening a closed branch or fitting the
generator to its own metric. Those are owner decisions, and they stay open.

**What the engine is.** A controllable URL generator that writes to a supplied plan with
near-exact obedience, places the root slash at real rates in all eight populations, and
produces output the gate's character-level detector cannot separate from real data in 29
of 32 measured cells. It has two known signatures: phishing query strings (§5.3), and a
lower share of real words than real URLs, in both classes (§8).

---

## 8. Addendum (2026-10-08): the word-level finding

Added after the report was written. **Nothing above is changed except the two sentences
marked as corrected.** E3 stays closed and `e16_b5` stays the baseline.

**The test.** `research/e18/STRONG_DISC_RESULT.md`, registered before it ran
(`STRONG_DISC_PREREG.md`). Four whole-URL detectors were compared against real held-out
URLs, each with a real-vs-real floor and two positive controls.

| detector | benign excess | phishing excess |
|---|---|---|
| the gate's own model (character n-grams), 5× the gate's data | +0.027 CLEARED | +0.044 UNDECIDED |
| character CNN | +0.001 CLEARED | −0.000 CLEARED |
| character transformer | +0.009 CLEARED | +0.002 CLEARED |
| **word-level features** (are the words found in the real training corpus?) | **+0.168 CONVICTED** | **+0.164 CONVICTED** |

**What the word-level detector measures.**
- `e16_b5` writes words that do not occur in real data more often than real URLs do:
  1.45 unseen words per benign URL against 0.93.
- In paths and queries, 84% of its words are known words, against 94% for real URLs.
- Structural features alone (lengths, counts, depth) give no separation (AUC 0.50 benign,
  0.55 phishing). The whole difference is in vocabulary.

**What it means for this report.**
- The realism claims in §§1, 4 and 7 hold **for character-level detectors**, which is
  what the gate measures. They do not hold against a detector with word knowledge.
- It gives a common cause behind several findings above: the engine's "too inventive"
  behaviour (§5.3 b), and plausibly part of the phishing-query cell.
- Two caveats from the same test:
  - the character CNN and transformer, trained from scratch at this sample size, were
    weaker than the gate's own model, so their clean readings add little;
  - with more data, the gate's own model moves phishing whole URLs from +0.015 to +0.044
    (UNDECIDED), so the gate's smaller sample understated that reading.

**What follows.** A V2 design that addresses the vocabulary gap is drafted separately
under `research/v2/`. It does not change E3 or any result here.

---

## Appendix — records

| item | location |
|---|---|
| adopted engine | `research/native/DECISION_2026-10-08.md` |
| E3's definition | `research/native/DECISION_2026-09-29.md` §3 |
| final cell data | `data/e17/score_report_e16_b5.json` |
| B5's criteria | `data/e17/compare_b5.json`, `research/e17/E17_RESULTS.md` |
| the PTH check | `research/e18/PTH_CHECK_RESULT.md`, `data/e18/pth_check.json` |
| experiment results | `research/*/[A-Z]*_RESULTS.md` (E8–E17, plus CR1, DC1, QC1, SC1, C3, HC1, HC2, N2A, PC1, QG1–3) |
| owner decisions | `research/native/DECISION_2026-09-28.md` … `DECISION_2026-10-08.md` |
| validity records | `data/e17/VERIFY_{DECODE,TRAIN,GENERATE}.json` |

*Report written 2026-10-08. Every figure is drawn from the committed records above.*
