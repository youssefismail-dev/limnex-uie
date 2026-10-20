# HC2 — presence by resampling on E7: pre-registration and design review

**Registered:** 2026-09-24, before any HC2 code.
**Approved by the owner (2026-09-24):** Option 1 of `research/hc1/HC1_RESULTS.md`
— enforce presence by resampling. **Frozen:** `data/best_e7.pt` (sha256
`916d0385…`); E5, R4, gate v2, thresholds and every published result; E7's and
HC1's results and data.

## 1. Why

HC1's masking held the rules exactly (H2) and as realistically as exact draws
(H3), but could not always finish a URL (H1 FAIL: 15 runaways in the phishing
field-on cells). Resampling never forces a token: a URL that fails is simply
redrawn, so every output is a draw from the model's own conditional.

## 2. Design review

**2.1 Exactness.** Rejection sampling: drawing from p and keeping only draws in
a set A yields exactly p(x | A). Cost: 1/P(A) draws per kept URL.

**2.2 What to hold fixed on a redraw — the one real choice.**

- **Held suffix** (HC1 H3's exact arm): redraw the URL with the slot's suffix.
  Exact for p(x | suffix, A) — right when a *caller* fixes the suffix — but it
  keeps the table's class-marginal TLD mix. Cost: the harmonic mean of
  1/P(A | suffix), up to 1.65× measured.
- **Joint** (redraw suffix and URL, the suffix again from the table): exact for
  the whole generator's conditional p(suffix, x | A). Cost 1/P(A): about 1.30×
  for phishing `<Q:1>`.

The choice is decided by a fact about real data, measured on the held-out pools
before any HC2 code: **presence shifts the TLD mix.**

| real held-out URLs (name population) | TV(TLD mix given the field, class marginal) |
|---|---:|
| phishing, with a query | **0.350** (`.com` 26.2% → 39.6%; `.app` 10.3% → 4.9%) |
| benign, with a query | 0.159 |
| phishing, with a path | 0.145 |
| benign, with a path | 0.040 |

Held-suffix resampling would therefore give phishing query-bearing URLs a TLD
mix 0.33 in TV from real ones, by design. The joint version tilts the mix by
the model's own P(field | suffix): if E7 learned that dependence from its
training data, it reproduces the real shift. **Default: joint.** Held suffix
stays available for a caller-fixed suffix. R3 tests the choice.

**2.3 Termination.** Redraws continue in rounds; after 100 rounds without
success a request **fails loudly** (an error, never a silent violating URL).
**2.4 Truncation** is not an acceptance condition: accepted URLs reach
`max_len` only at the model's own rate, reported. **2.5 Presence** is the
gate's own definition, `uie.evaluation.FIELDS` on the rendered URL. **2.6
Isolation:** a new module, `uie/presence.py`; `uie/model.py`, `uie/suffixes.py`
and the gate are not edited. With no control set, it returns
`generate_with_suffixes` unchanged — the gate's own path.

## 3. Tests

Canonical generation path throughout (marginal suffix table, frame `domain`,
`temperature` 1.0, adopted delta); HC1's seeds — cell i (C1's order) seed
20260925 + 10,000·i, n = 1,200 — so every cell is paired with HC1's masked arm.
Redraw round k uses seed + 1,000,003·k.

- **R1 — presence holds (PRIMARY).** Joint resampling, all 8 cells: every URL
  honours its control, and no request exhausts its rounds.
- **R2 — implementation (PRIMARY).** Held-suffix resampling on the two
  `<Q:1>` cells reproduces HC1 H3's exact arms **exactly** (as multisets:
  `data/hc1/h3/<class>_Q1_exact.json`).
- **R3 — joint against held (SECONDARY, decides the default).** On the four
  field-on cells, both modes scored with the gate's discriminator on the **SUF**
  field against the real pool **restricted to URLs with that field**; D = 10
  split seeds shared (20260925 + 900 + d); Δ = x_joint − x_held.
  **Joint stays the default unless Δ ≥ +0.02 with the CI excluding 0 in any
  cell**; then held suffix is recommended.
- **Descriptive.** Draws per kept URL (against 1/first-pass); rounds; the share
  of kept URLs at 240+ characters (the truncation proxy HC1 used), against the
  first pass; path and query shares against HC1's masked arm; the 8 joint
  settings at n = 1,000.

## 4. Adoption — not pre-authorised

E7 + joint resampling can be recommended as the controllable generator if R1
and R2 pass. HC1's C2 carries over unchanged — the unconditional mode is
untouched. Replacing E5 remains the owner's decision.

## 5. Execution

This registration; `uie/presence.py` with tests; one chain on the laptop at
below-normal priority — R1, then R2 and the held P:1 cells, then R3, then the
joint settings; results. Outputs in `data/hc2/`; nothing under `data/hc1/` or
`data/e7_eval/` is written.

## Status

    HC2  registered (ce82350); implemented (0a3e5fe)
         R1 PASS; R2 PASS; R3 joint stays default -> HC2_RESULTS.md
         adoption condition met (section 4); the decision is the owner's
