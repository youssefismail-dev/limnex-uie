# NS — is the gate's excess biased at reduced n? — pre-registration

**Registered:** 2026-09-23, before any number in this file was computed.
**Frozen:** `data/best_e5.pt`, `data/bundle2`, and `discriminate_with_floor`
unmodified (`uie/evaluation.py` at `8dcf0b8`). No gate, bar, `MIN_N`, delta,
decoder or checkpoint change. **Read-only: no generation, no training.**

---

## 1. Why this exists

I wrote into `R4_RESULTS.md` section 2 and the README's "Restated under the
ADOPTED decoder" that the gate's excess **"rises mechanically with n"**, and called
it "already documented in this project". **It was not.** What is documented is the
sample-size dependence of TTR and top-k concentration — different statistics.

The gate's own code states the opposite (`uie/evaluation.py:552-555`):

> A reduced n is a weaker measurement, not a wrong one, and it is flagged so it
> reads as such.

And `E3_REPORT.md` section 5.1 chose `MIN_N = 250` from the standard error alone
(~0.037 at n=250). That reasons about **variance**. It never considered **bias**.

**Neither position has ever been measured.** It is load-bearing:

1. The README's only unresolved cell, **benign QRY**, rests on it.
2. The gate measures low-emission fields at **reduced n** — benign QRY canonically
   at n~361, phishing QRY below `MIN_N`. Until this is known, **no QRY verdict,
   before or after any repair, can be trusted to acquit.**
3. It sets the cost of making phishing QRY measurable.

## 2. The mechanism, and where the two positions differ

`excess = AUC(real vs fake) - floor(real vs real)`, each a 5-fold CV of char_wb
2-5 gram TF-IDF + logistic regression **fit on the sample it is given.** n can act
on each term separately:

- **AUC.** A genuine difference is learned better from more data — a learning
  curve. If the curve is still rising between n=250 and n=1200, E[AUC] rises with
  n and excess is **biased low at small n.**
- **floor.** There is no signal to learn at any n, so E[floor] ~ 0.5 throughout;
  only its variance shrinks. Both positions agree. A floor that drifts with n
  would be a separate instrument defect.
- **Under the null** (no tell), E[excess] ~ 0 at every n under both positions.

**The positions differ ONLY for fields that carry a tell** — which are exactly the
fields the gate exists to catch.

| | name | prediction |
|---|---|---|
| **H_WNW** | the gate's assumption, "weaker, not wrong" | E[excess] flat in n over 250-1200; reduced n adds variance only |
| **H_LC** | my claim, learning-curve bias | E[excess] rises with n for fields with a tell, scaling with the tell; ~0 for null fields |

## 3. Prior evidence already in the record — disclosed before measuring

**(a) Leans AGAINST H_LC.** Benign QRY under `typical`: Q4 at n=476 (delta 0)
reads **+0.1325**; the gate's pooled estimate at n=1124 (delta 0.22) reads
**+0.1233**. No rise from 476 to 1124. Confounded by delta and estimator, so it is
suggestive, not decisive.

**(b) An anomaly H_LC cannot explain.** Benign QRY under `temperature`: Q4 at
n=476 (delta 0) reads **+0.1124** with floors 0.490 / 0.497 / 0.516; the gate's
canonical cell at n~361 (delta 0.22) reads **+0.0300** — AUC 0.5675 against a
**floor of 0.5374**. A 0.08 gap across a 30% change in n would need an
implausibly steep learning curve. **The elevated floor is the likelier lead.**
This study records floor(n) separately and will say whether a floor that high is
typical at n~361 — but it is **not designed to resolve the anomaly**, which stays
OPEN if unresolved.

**Given (a), H_WNW is the better-supported prior.** Saying so in advance is what
makes a result in either direction informative.

## 4. Design

**Instrument:** `discriminate_with_floor`, unmodified, called as the gate calls it
— field populations from `FIELD_POPULATION` (every field scores against `name`;
FULL against `mixed` = name + structure), pools built with `run_eval`'s own
`_find` / `_rows`, `dedup_by_host=True`.

**Fakes:**
- **Main grid:** `data/gen_cache/` — E5, `typical`, adopted delta as
  `field_bias`, 5 seeds x 1,800 per class (`research/tell/regen.py`), pooled to
  **9,000 per class.** The decoder is the pre-adoption one, which does not matter
  here: what is under test is the **instrument**, not the model.
- **QRY (secondary):** the Q4 query-bearing samples, delta 0, both decoders.

**Fields and roles, fixed from the published gate (typical, adopted delta):**

| role | benign | phishing |
|---|---|---|
| **NULL CONTROL** | SUF +0.0035 | SUF -0.0032 |
| **WEAK TELL** | LBL +0.0260, SUB +0.0400 | LBL +0.0265 |
| **BAR-SCALE TELL (primary)** | PTH +0.0728 | PTH +0.0834, SUB +0.0651 |
| **STRONG TELL (reference)** | FULL +0.2269 | FULL +0.3995 |

**n grid (main):** 250, 400, 600, 800, 1000, 1200. **Every cell is feasible** —
counted before registering (rows only, no excess computed). The tightest ceiling
is phishing PTH: 2,668 fakes against 3,129 real.

**QRY grid (secondary, range limited by the data):**
benign 150, 200, 250, 300, 361, 420, 476; phishing 150, 200, 250, 300, 356.
Phishing tops out at 356 because its real pool holds 1,068 query rows and the
instrument needs 3n. **At the top point the fake side uses every available row**,
so only the real draw and CV vary there.

**Replicates:** R = 20 per (class, field, n), seeds `20260923 + r`. **AUC, floor and
excess are all recorded per draw.**

**Cost:** ~1,700 instrument calls, ~20-30 min on 4 single-threaded workers.

## 5. Analysis, fixed now

Per (class, field): OLS of excess on ln(n) over every draw, and
**Delta = b x ln(1200/250)**, with a 95% CI from the OLS standard error. The same
fit is run for AUC and for floor separately, to decompose any effect.

Replicates share finite pools, so the OLS standard error is **slightly
understated** — reported as such, never used to acquit.

## 6. Decision rule, fixed now

**V — validity.** SUF, the null control, in **both** classes: |Delta| < 0.01 and the
CI covers 0. If either fails, the instrument drifts **without** a tell; the study is
**VOID** for attributing any bias to learning, and the drift itself becomes the
finding.

**Primary, on the three BAR-SCALE fields:**

- **H_WNW holds — "weaker, not wrong":** Delta < 0.01 in all three **and** every CI
  upper bound < 0.02.
- **H_LC holds — "biased low at small n":** Delta >= 0.02 with CI lower bound > 0
  in **at least two of three**, **and** the dose-response ordering holds:
  Delta(null) ~ 0 < Delta(weak) < Delta(bar-scale).
- **Anything else: INTERMEDIATE** — reported as such, neither claimed. No field is
  added afterwards to break the tie.

**Secondary, descriptive, no decision:** QRY slopes within their own ranges;
whether FULL saturates; floor(n) behaviour, including where **0.5374** falls among
the benign QRY floors at n=361.

## 7. Consequences, fixed now

- **H_WNW:** my claim is withdrawn from the README and `R4_RESULTS.md`. Benign
  QRY's canonical +0.0300 is then an unbiased, if noisy, estimate — **and the
  floor anomaly in 3(b) becomes the open explanation** for the gap to Q4. Making
  phishing QRY canonically measurable needs only n_main ~11,100.
- **H_LC:** `E3_REPORT.md` section 5.1's `MIN_N` rationale is amended (it covered
  variance, not bias); **every reduced-n verdict is re-read** against the measured
  Delta; the "not cleared" reading of benign QRY gains a measured basis; QRY
  verification needs a matched-n design.
- **INTERMEDIATE:** both readings stay hypotheses, and benign QRY stays *Unknown*.

## 8. Out of scope

Any change to the gate, bar, `MIN_N`, delta, decoder or checkpoint. Any repair.
Any new generation.

**The closed PTH attribution branch is not reopened.** PTH appears here only as a
field with a known tell, to calibrate the **instrument**; nothing here re-attributes
the PTH tell or its cause.

## Status

    NS  RUN 2026-09-23 -- registered verdict VOID; see NSWEEP_RESULTS.md
        (the null-control tolerance was tighter than the design could
        resolve; descriptively, tells rise with n via the AUC term)
