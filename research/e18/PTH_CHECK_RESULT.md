# The benign P1Q1 PTH check — result: NO enforceable structural violation

- **Requested 2026-10-08** (`research/native/DECISION_2026-10-08.md` §4):
  a read-only check, and "if you find nothing, we close the book on it and leave it as
  is".
- **Tool:** `research/e18/pth_check.py`. **Record:** `data/e18/pth_check.json`,
  `data/e18/fresh_benign_P1Q1.json`.
- **Nothing was registered, trained or changed.** This measures only.

## The question

`e16_b5` leaves benign P1Q1 PTH at 0.0539 (SE 0.0012); it clears at 0.0467, so it needs
−0.007. D1 and B5 earned their place by forbidding things real URLs **never** do, which
needs no tuning and cannot be fitted to the gate. So: do these paths do anything that
real benign P1Q1 paths never do?

## How it was kept honest

- **Fresh generations, never the gate's 50 files:** 1,600 control benign P1Q1 URLs from
  the adopted baseline at g = 100, 101, far outside the gate's g 0–4. 1,597 have a path.
- **"What real paths never do" is read from the TRAINING pool** (185,049 P1Q1 paths),
  which the model already saw, so a rule justified by it is not fitted to the gate. The
  validation pool (24,746) only confirms.
- **The decision rule was fixed before the numbers were seen.** A feature counts only if
  real ≈ 0, generated is clearly above 0, and the rate is **about 1% or more** — because
  a tell present in fraction *f* of generated and 0 of real contributes about *f* to the
  excess, and 0.007 is needed.

## 1. The battery: nothing qualifies

Every well-formedness and degeneracy property, generated against real:

| feature | generated | real train | real val |
|---|---|---|---|
| double slash, empty segment, `.`/`..` segment | 0.0000% | ≤0.0011% | 0.0000% |
| malformed percent-encoding, percent not UTF-8 | 0.0000% | ≤0.0016% | 0.0000% |
| whitespace, control char, non-ASCII, backslash | 0.0000% | 0.0000% | 0.0000% |
| character unseen in real paths (81-char set) | 0.0000% | 0.0000% | 0.0000% |
| longer / deeper / longer-segment than any real | 0.0000% | 0.0000% | 0.0000% |
| `@` sign | 0.0626% (1) | 0.0411% | 0.0323% |
| repeated block (≥4 chars, ×3) | 0.0626% (1) | 0.0286% | 0.0081% |
| same char ×6 | 0.1879% (3) | 0.1643% | 0.1051% |
| **ANY of the above** | **0.3131%** (5) | **0.2307%** | **0.1455%** |

- **The model commits no violation that real paths never commit.** On the twelve
  properties where real is 0, generated is also 0. On several it is *cleaner* than real
  data.
- The three non-zero properties all occur in real paths too, at comparable rates, and
  amount to 1, 1 and 3 URLs.
- The whole excess over real is 0.08 percentage points, worth about **0.0008** of the
  tell. The gap is 0.007.

## 2. A scan that does not depend on my choice of features

The battery only tests properties I thought to name. So, data-driven: character 4-grams
present in a path but in **none** of the 185,049 real training paths. Real held-out
validation paths are the control, because genuinely new real text also contains
sequences the training pool never had.

| | paths with a 4-gram unseen in training |
|---|---|
| generated | **29.99%** (479 / 1,597) |
| real validation (control) | **28.65%** (7,090 / 24,746) |

The model invents unseen character sequences at the same rate real held-out data does —
a difference of 1.3 points, about 1.2 standard errors. There is no hidden structural
anomaly that the battery missed.

## 3. The one real plan violation — and why it cannot help this cell

3 of 1,600 fresh URLs (0.19%) dropped the path under a P1 plan, writing `host/?query`
instead. The gate's own figure over 15,000 URLs is a P/Q honouring rate of 0.99687.
That is a genuine violation of the B5 family, and a rule could forbid it.

**It still cannot move PTH.** `uie/evaluation.py::discriminate_with_floor` builds both
sides as `[x for x in (extract(u) for u in ...) if x]`, so a URL with no path is **not in
the PTH comparison at all**. Forcing a path onto those 0.3% would *add* paths to the
compared set, not remove a tell. It is an honouring matter, and honouring already passes.

## 4. What is actually left: a matter of degree

| | generated | real train | real val |
|---|---|---|---|
| mean path length | 19.69 | 21.74 | 20.78 |
| mean depth (segments) | 1.72 | 1.87 | 1.82 |

The model's paths are about 10% shorter and slightly shallower than real ones. That is a
distributional difference, not a broken rule, and it is exactly what a field temperature
and end-bias would target — **E13's D2, fitted on real held-out paths for this very cell,
which moved it −0.0018 and is closed**. Setting such a knob by anything other than real
data would be tuning the generator to the metric.

## Verdict

**No enforceable structural violation exists.** There is no honest inference-time fix for
benign P1Q1 PTH, and per the owner's instruction the book is closed on it. The cell
stands at 0.0539 CONVICTED.

**Limitation, stated plainly:** this rules out *structural* fixes. It does not prove the
cell is unimprovable — a training-side change could still move it. It says only that the
decode-rule route, which worked for D1 and B5, has nothing to grip here.
