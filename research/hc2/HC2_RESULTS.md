# HC2 — results: presence by resampling

**Registered:** `PREREG_HC2.md` (ce82350), before any HC2 code. **Implemented:**
0a3e5fe — `uie/presence.py`, a new module; the gate, `uie/model.py` and
`uie/suffixes.py` untouched; suite 651. **Run:** 2026-09-24 05:29–07:40, laptop
CPU at below-normal priority, peak 0.43 GB. **Checkpoint:** `data/best_e7.pt`
(sha256 `916d0385…`). E7's and HC1's results unchanged. Outputs: `data/hc2/`.

## Verdicts

| test | registered rule | result |
|---|---|---|
| **R1** — presence holds | joint resampling, 8 cells: every URL honours; no request exhausts | **PASS** — 1,200/1,200 in all 8; at most 4 rounds |
| **R2** — implementation | held mode reproduces HC1 H3's exact arms exactly | **PASS** — both classes identical |
| **R3** — joint against held | joint stays default unless Δ ≥ +0.02 with the CI excluding 0 | **joint stays the default** — and is significantly *better* where the TLD shift is large |

**Adoption condition (section 4) is met:** R1 and R2 pass, and HC1's C2 carries
over — the unconditional mode is untouched. Whether to adopt E7 with joint
resampling as the controllable generator is the owner's decision; CR1, running,
adds its absolute realism.

## R1 — presence holds, at the predicted cost, without masking's side effects

| cell | honoured | draws (per kept URL) | 1 / first pass | rounds | 240+ chars: HC2 / C1 | path share: HC2 / HC1 masked |
|---|---|---:|---:|---:|---|---|
| benign P:1 | 1200/1200 | 1208 (1.007) | 1.007 | 1 | 0.00% / 0.10% | 100% / 100% |
| benign P:0 | 1200/1200 | 1205 (1.004) | 1.004 | 1 | 0.00% / 0.00% | 0% / 0% |
| benign Q:1 | 1200/1200 | 1251 (1.042) | 1.044 | 2 | 0.08% / 0.13% | 62.7% / 63.7% |
| benign Q:0 | 1200/1200 | 1221 (1.018) | 1.018 | 1 | 0.08% / 0.00% | 58.1% / 59.9% |
| phishing P:1 | 1200/1200 | 1220 (1.017) | 1.017 | 1 | 0.00% / 0.03% | 100% / 100% |
| phishing P:0 | 1200/1200 | 1216 (1.013) | 1.014 | 1 | 0.00% / 0.00% | 0% / 0% |
| phishing Q:1 | 1200/1200 | 1521 (**1.268**) | 1.274 | 4 | 0.17% / 0.10% | **49.7% / 56.0%** |
| phishing Q:0 | 1200/1200 | 1215 (1.012) | 1.013 | 1 | 0.00% / 0.00% | 29.8% / 30.6% |

- **The cost is exactly what the design predicted:** draws per kept URL equal
  1 / first-pass rate in every cell (phishing `<Q:1>`: 1.268 against 1.274).
- **No runaways:** the long-URL share stays at the model's own rate (at most
  0.17%, against masking's 1.08%).
- **No structural shift:** phishing `<Q:1>` URLs carry a path 49.7% of the time
  — HC1's exact arm 50.4%, masking 56.0%. Resampling removes the shift masking
  introduced.
- **Cross-check:** for the two `<Q:1>` cells the first pass accepted 1,150 and
  942 URLs — identical to HC1 H3's first pass, as the shared seeds require.

## R2 — the library is the exact arm

Held-suffix resampling through `uie.presence` reproduced HC1 H3's exact arms
**exactly** for both classes (as multisets). It also shows the cost of holding
the suffix: phishing `<Q:1>` needed 1,983 draws and up to **24 rounds**,
against joint's 1,521 draws and 4 rounds.

## R3 — joint against held, on the TLD mix

| cell | SUF excess: joint | held | Δ | 95% CI | TLD-mix TV to real: joint / held |
|---|---:|---:|---:|---|---|
| benign P:1 | +0.0228 | +0.0219 | +0.0009 | [−0.0002, +0.0020] | 0.193 / 0.191 |
| benign Q:1 | +0.0502 | +0.0549 | −0.0047 | [−0.0072, −0.0022] | 0.218 / 0.227 |
| phishing P:1 | +0.0945 | +0.0915 | +0.0029 | [−0.0008, +0.0067] | 0.217 / 0.218 |
| **phishing Q:1** | **+0.1214** | **+0.1617** | **−0.0403** | [−0.0504, −0.0301] | **0.276 / 0.336** |

Scored against real URLs **with** the field (n = 1,200; phishing `<Q:1>` 356,
all its real pool supports). Joint is never worse beyond 0.003, and where real
data showed the largest TLD shift — phishing with a query, TV 0.35 — it is
significantly better, as the design review predicted. **Joint stays the
default.** (The absolute SUF excesses above 0.05 here are CR1's subject.)

## Descriptive — the 8 combined settings (n = 1,000, joint)

| setting | benign: draws (first pass) | phishing: draws (first pass) |
|---|---|---|
| P:1 Q:1 | 1,046 (95.7%) | 1,428 (70.5%) |
| P:1 Q:0 | 1,014 (98.6%) | 1,043 (95.8%) |
| P:0 Q:1 | 1,061 (94.2%) | 1,250 (79.5%) |
| P:0 Q:0 | 1,026 (97.6%) | 1,010 (99.0%) |

Every setting honoured 1,000/1,000; at most 5 rounds. The costliest request —
phishing with a path and a query — needs 1.43 draws per URL.

## What this means

Resampling does what masking could not: every URL honours its control, none is
forced, the cost is the model's own compliance rate (1.00–1.43×), and joint
mode keeps presence and TLD in the proportion the model learned — which, for
phishing queries, is measurably closer to real data than holding the suffix.
**Resampling is the enforcement to use.** Its absolute realism — how each
controlled state compares with real URLs in that state — is CR1.

## Reproduce

    python research/hc2/hc2_run.py --checkpoint data/best_e7.pt --bundle data/bundle2 \
        --c1 data/e7_eval/c1 --hc1 data/hc1 --out data/hc2
