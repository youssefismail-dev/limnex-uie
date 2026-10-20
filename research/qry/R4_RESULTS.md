# R4 — results: does removing decoder truncation regress anything?

**Run:** local CPU, canonical gate at the **frozen** `best_e5.pt`,
`--decoder temperature --temperature 1.0`, n_main 8000, G=5 x D=3,
delta benign 0.22 / phishing 0.30. Completed 2026-09-23.
Benign 10,204 s + phishing 11,922 s. Output `data/r4_temp_dadopted/`.

Registered as R4 in `REPAIR_PLAN.md` section "R4 — decoder truncation".
Inference-only: **no weights, no config, nothing written to the model.**

> R4's registered purpose is the **RISK** side, not the benefit side. Q4 had
> already measured the content upside at 15.1% (t = -1.64, not significant) and
> Q5 the name-TTR upside at ~46% of the gap. What was unmeasured was whether
> removing truncation **regresses fields that currently clear**.

---

## 1. Result — no regression anywhere, five cells significantly better

Paired §26.1 test: per-generation means over D=3, SE = sd/sqrt(G-1),
t_crit(df=4) = 2.776. **The floors are identical across both runs**, confirming
the same real comparator and the same splits, so the pairing is legitimate and
the only difference is the decoder.

> **Note, 2026-09-23 - SE convention.** The t-values here were computed with
> SE = sd(generation means) / sqrt(G - 1). Section 26.1 specifies sd / sqrt(G),
> the standard one-sample form, which gate v2 now encodes
> (`PREREG_GATE_V2.md`). Every paired t in this document stays on the same side of its critical value (2.776 at df=4, 4.303 at df=2), and all 25 published gate cells were
> re-checked under both: **no verdict differs.**

| cls | field | baseline (typical) | R4 (temperature) | diff | SE | t | |
|---|---|---:|---:|---:|---:|---:|---|
| benign | LBL | +0.0260 | -0.0014 | -0.0274 | 0.0034 | -7.98 | **significant** |
| benign | SUF | +0.0035 | +0.0001 | -0.0034 | 0.0063 | -0.54 | |
| benign | SUB | +0.0400 | +0.0048 | -0.0352 | 0.0035 | -10.00 | **significant** |
| benign | PTH | +0.0728 | +0.0384 | -0.0344 | 0.0029 | -12.02 | **significant** |
| benign | FULL | +0.2269 | +0.2227 | -0.0042 | 0.0028 | -1.50 | |
| phishing | LBL | +0.0265 | +0.0144 | -0.0122 | 0.0076 | -1.61 | |
| phishing | SUF | -0.0032 | -0.0064 | -0.0032 | 0.0070 | -0.46 | |
| phishing | SUB | +0.0651 | +0.0332 | -0.0320 | 0.0058 | -5.48 | **significant** |
| phishing | PTH | +0.0834 | +0.0597 | -0.0237 | 0.0035 | -6.82 | **significant** |
| phishing | FULL | +0.3995 | +0.3992 | -0.0003 | 0.0025 | -0.12 | |

**Every diff is negative.** Zero cells regressed at any significance level.

Verdict changes, all favourable:

    phishing PTH   TELL      -> UNDECIDED     a conviction lifted
    phishing SUF   UNDECIDED -> ok
    benign  LBL/SUF/SUB      ok -> ok         all three held

Novelty rose rather than fell: benign 99.7% -> 99.8%, phishing 97.8% -> 98.8%.
Gate pass: benign 99.7% -> 99.7%, phishing 97.1% -> 98.2%.

**There is no retraining-noise confound here.** R4 uses the *same* checkpoint;
only the decoder differs. The ±0.0141 figure (which is nondeterminism at
identical config, measured w1-vs-E5) does not apply.

### Correction to `REPAIR_PLAN.md`

That document names five cells as currently clearing — "benign LBL/SUF/SUB,
phishing LBL/SUF". Against `data/e5_gate_n8000_dadopted` that is wrong: only
**three** read `ok` (benign LBL, SUF, SUB); phishing LBL and SUF were both
UNDECIDED. The safety conclusion is unaffected and in fact stronger — all three
`ok` cells held, and phishing SUF improved *into* `ok`.

---

## 2. QRY did NOT improve — and the number that suggests it is a trap

The summary lines invite a wrong reading:

    baseline benign QRY   insufficient (n_fake 211), pooled TELL +0.1233 (n=1124)
    R4       benign QRY   +0.0300 UNDECIDED, 15 draws (n ~ 361)

That looks like a collapse. **It is not a comparison.** Decoder, estimator
(pooled vs canonical) and n all differ between the two lines, so the difference
cannot be attributed to any one of them.

> **Correction, 2026-09-23.** This paragraph originally said excess "rises
> mechanically with n" and called that "the same sample-size confound already
> documented in this project for TTR and top-k concentration." **It was not
> documented** — the TTR and top-k results concern different statistics — and
> `uie/evaluation.py` assumes the opposite ("a reduced n is a weaker measurement,
> not a wrong one"). Whether reduced n biases the gate's excess is **unmeasured**,
> registered as NS in `PREREG_NSWEEP.md`. The phishing comparison below does not
> depend on it: same estimator, same n.

**The clean cell is phishing QRY**, where both runs were starved, both fell back
to the pooled estimator, and both landed at **the same n = 356**:

    baseline draws  [0.1199, 0.1626, 0.1640]
    R4 draws        [0.1174, 0.1543, 0.1577]
    paired diff     -0.0057   SE 0.0021   t -2.74   t_crit(df=2) 4.303
                    NOT significant; 3.8% of the baseline tell

This agrees with Q4's controlled measurement at equal n (A - T = -0.0201,
t = -1.64, ns). Two independent designs, same answer: **removing truncation
moves QRY content by at most a few percent of the tell.**

R4 satisfies its registered acceptance clause "QRY excess does not rise" — but
by not moving, not by fixing.

---

## 3. What R4 did buy: emission, which is the separate Q1 defect

| | real | baseline | R4 | ratio to real |
|---|---:|---:|---:|---:|
| benign QRY | 6.40% | 2.81% | **4.51%** | 0.439 -> 0.704 |
| phishing QRY | 4.44% | 1.70% | **2.31%** | 0.384 -> 0.521 |
| benign PTH | 72.42% | 60.98% | 59.01% | 0.842 -> 0.815 |
| benign SUB | 41.63% | 38.30% | 38.17% | 0.920 -> 0.917 |

Queries appear far more often, and are no less distinguishable once they appear.
Emission and content are separable defects, and R4 moves only the first.

---

## 4. Verdict

**R4 is SAFE, confirmed.** The registered risk question — does removing
truncation regress fields that currently clear — is answered: it does not.
Every measured cell improved, five significantly, and two verdicts got better.

**R4 is NOT a fix for the QRY content defect**, which was already the
expectation going in and is now confirmed at the gate scale.

## 5. ADOPTED 2026-09-23

`temperature` (temperature 1.0) is now the project default, on the user's
approval. `decode_temperature` defaults to 1.0, so `decoder="temperature"` with
no kwargs is exactly what R4 gated.

Changed:

    uie/model.py              generate(...)                 decoder default
    uie/suffixes.py           generate_with_suffixes(...)   decoder default
    uie/corpus/run_eval.py    run(...) and --decoder         default

Every research script already passed `decoder=` explicitly; only test call sites
relied on the default, and the suite passes unchanged (557). Report artifacts
record the decoder actually used, and `--decoder typical` reproduces every
pre-adoption baseline. **The new reference run under the adopted decoder is
`data/r4_temp_dadopted/`.**

### This SUPERSEDES a prior rejection, and that needs stating plainly

`E3_REPORT.md` section 13.4 lists as REFUTED: *"`typical` decoding truncates the
tail -> pure `temperature=1.0` sampling is no better, often worse"*, repeated in
the two later falsified-hypothesis tables (lines ~979, ~1117), attributed to E4.

**I could not locate the measurement behind that line.** It is not in
`research/`; it predates the current layout. So this adoption overturns a
recorded conclusion whose underlying data I have not inspected. The grounds for
overturning it anyway:

- R4 is a **paired canonical gate at n_main 8000, G=5 x D=3, with verified
  identical floors**, on the **current** checkpoint. The E4-era line compares an
  older checkpoint by an unrecorded method.
- R4's effect sizes on PTH are large and tightly estimated: benign -0.0344
  (t -12.02), phishing -0.0237 (t -6.82).
- The direction is consistent across all ten cells with no counterexample.

**A different E3 rejection is NOT superseded and still stands.** Section 6.2
rejected the decoder as an explanation for the *path-emission* gap (all three
decoders within 1.4 points, all far from target). R4 **confirms** that: benign
PTH emission moved 60.98% -> 59.01%, slightly worse. Emission and
discriminability are different quantities, and R4 only overturns the second.

### What adoption does not claim

R4 tested exactly one alternative. Q1 characterised four decoders; `temperature`
is the one that was gated, not the one shown best among all candidates. No
search over decoder settings was run, and none is proposed here.

`typical` was never shown to be *wrong* — it is beaten by a decoder that was
simply never taken to the gate.
