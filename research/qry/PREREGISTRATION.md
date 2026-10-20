# QRY characterisation — pre-registration

**Registered:** 2026-09-22, before any experiment in this file was run.
**Frozen inputs:** `data/best_e5.pt`, `data/bundle2`, `uie/corpus/run_eval.py`
at `6f0d43c`. No generator, checkpoint, decoder or gate code is modified by this
work. The 0.05 bar is untouched.

## What is already established (E3_REPORT section 30)

QRY is the model's largest field defect, and it is defective on **two axes**:

    A  emission   2.49% benign / 1.30% phishing generated
                  against 6.361% / 4.294% in TRAINING and
                  6.403% / 4.435% in validation
    B  content    excess +0.1196 to +0.1488 across four configurations,
                  measured on query-bearing rows only

Train and validation agree to within 0.15 points, so A is not a corpus mismatch.

**The objective of this phase is NOT to fix either axis.** It is to determine
whether A and B share a mechanism or are separable.

## The decision structure that makes A measurable

`URLTokenizer.encode` opens a field only when its body is non-empty, and appends
the marker immediately before the body. Markers run
`SCH USR SUF LBL SUB PRT PTH QRY FRG`. So while the path body is being generated,
**every character step is a four-way choice**: another path character, `<QRY>`,
`<FRG>`, or `<eos>`. "Does this URL carry a query" is therefore not one decision
at one position — it is a decision renewed at every path step, and any
truncating decoder filters the query-opening token at every one of them.

The production decoder is `decode_typical` (`mass=0.95`), which keeps tokens
whose surprisal lies NEAREST the distribution's entropy and truncates the rest.
A token is cut for being *atypical*, not for being improbable. Where one outcome
dominates, entropy is low and a genuinely rare alternative sits far from it.

## Hypotheses

**H1 — DECODER.** The emission deficit is produced by typical sampling
truncating the query-opening marker. The model's own distribution assigns
approximately the corpus rate.

**H2 — MODEL.** The model itself under-assigns probability to opening a query.
The decoder is faithful.

**H3 — BOTH.** Model under-assigns *and* the decoder compounds it.

H1, H2 and H3 are mutually exclusive on the primary measurement below and
jointly exhaustive.

**H4 — SHARED MECHANISM (axis A and axis B).** If the emission deficit is a
decoder artifact (H1), then queries that ARE emitted are a biased sample — the
ones that survived truncation — and their content would be distinguishable
*because of the selection*, not because the model models queries badly. Under H4
the two axes have ONE cause.

**H5 — SEPARABLE.** Emission and content have different causes.

## Experiment Q1 — decoder ablation on emission rate (primary, decisive for A)

Pure ancestral sampling (`decode_temperature`, `temperature=1.0`) draws exactly
from the model's distribution with no truncation. **Its QRY emission rate IS the
model's implied marginal rate.** Nothing else needs to be computed.

Conditions, all at `delta=0` on the frozen checkpoint, both classes:

| decoder | role |
|---|---|
| `typical` (mass 0.95) | production control — must reproduce ~2.49% / ~1.30% |
| `temperature` (t=1.0) | **treatment** — the model's true marginal |
| `top_p` (p=0.9) | second truncating decoder, to show the effect is not specific to typicality |
| `greedy` | degenerate reference; expected to emit ~0 queries |

n = 4000 per cell, one seed per cell, seeds fixed in advance and shared across
decoders so the suffix draw is paired.

### Pre-registered predictions

- **If H1:** `temperature` emission lands in **[5.4%, 7.4%] benign** and
  **[3.4%, 5.4%] phishing** — i.e. covering the corpus rate — while `typical`
  stays near 2.5% / 1.3%.
- **If H2:** `temperature` emission stays below **3.5% benign / 2.0% phishing**,
  within noise of `typical`.
- **If H3:** `temperature` rises materially above `typical` but its interval
  excludes the corpus rate.

At n=4000 the standard error on a 6% rate is 0.38 points, so these intervals are
separable. Rates are reported with Wilson intervals.

### Kill criteria

- If `typical` does not reproduce 2.49% / 1.30% within +-0.6 points, the harness
  disagrees with the canonical run and **the experiment is void** — fix the
  harness, do not interpret the result.
- If `greedy` emits a non-trivial query rate, the decision-structure model above
  is wrong and Q1 is reinterpreted before anything is concluded.

## Experiment Q2 — teacher-forced probe at the true transition (mechanism detail)

Exact, no sampling. For each real validation URL, teacher-force the model and at
the position where the URL actually leaves the path body, record the
distribution over {continue, `<QRY>`, `<FRG>`, `<eos>`}, plus **whether the
typical filter at that position would have retained `<QRY>`**.

This distinguishes "model assigns low probability" from "decoder discards a
probability the model did assign", at the level of individual positions, and
gives the retention rate directly. It runs on real prefixes, which are the
target distribution; generated prefixes are a separate condition, because if the
model's own prefixes differ systematically the conditional may differ too.

## Experiment Q3 — content, conditioned on existence (axis B, separate)

**Run only on rows that carry a query, both sides.** A URL without a query is not
evidence about query content, and a URL with one is not evidence about emission.
The two axes are never pooled into one number.

Decomposition, chosen because each component is a documented property of query
syntax rather than an invented feature family:

    parameter count            key/value structure and separators
    parameter-name identity    percent-encoding presence and correctness
    parameter-value length     character-class composition
    ordering and repetition    value entropy

Any discriminator-based measurement recomputes its real-vs-real floor **under
the same transformation**, per the standing rule that a transform which moves
the floor has changed something other than the property under test.

## Experiment Q4 — the selection test (decisive between H4 and H5)

If Q1 supports H1, then the queries `typical` emits are a **survivorship-biased
subsample** of the queries `temperature` emits. Q4 measures axis B separately on
both sets, against the same real comparator and a recomputed floor:

- If the `temperature` queries are markedly LESS distinguishable than the
  `typical` queries, the content defect is substantially a selection artifact of
  the decoder -> **H4**.
- If both are distinguishable to a similar degree, content is a genuine modelling
  failure independent of emission -> **H5**.

This is the experiment that answers the phase's actual question, and it is only
interpretable after Q1.

## Standing discipline

- Report n and uncertainty on every rate.
- Null control: `typical` vs `typical` at a different seed, to establish the
  rate's own draw-to-draw spread before any difference is called real.
- Log failed hypotheses and void instruments in this file rather than deleting
  them.
- Do not convert "unmeasured" into "pass"; do not use a pooled measurement to
  acquit.
- Do not select an intervention during this phase.

## Status

    Q1  registered, not yet run
    Q2  registered, not yet run
    Q3  registered, not yet run
    Q4  registered, blocked on Q1
