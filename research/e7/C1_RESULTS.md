# E7 — C1 results: controllability FAILS

**Registered:** `PREREG_E7.md` section 4, C1 (11ac503), before E7 was trained.
**Run:** 2026-09-23 22:57 → 2026-09-24 02:01 (3.06 h), laptop CPU at below-normal
priority, peak 348 MB. **Checkpoint:** `data/best_e7.pt`, sha256
`916d03857d7d3db4…`. **Harness:** `research/e7/c1_honoring.py` (9a82d85) — the
canonical gate's own generation path (`generate_with_suffixes`: marginal suffix
table, frame `domain`, `temperature` 1.0, adopted delta 0.22 / 0.30 on PTH EOS),
one control set and the other `?`, presence measured with the gate's own
extractors (`uie.evaluation.FIELDS`). Seeds 20260924 + 10,000·i, one per cell.
**Outputs:** `data/e7_eval/c1/` — `c1_report.json` and every generated URL.

## 1. Verdict: FAIL — 3 of 8 cells below the registered 0.98

| class | control | honoured | 95% CI | bar 0.98 |
|---|---|---:|---|---|
| benign | PTH:1 | 99.4% (3977/4000) | [99.1, 99.6] | ✅ |
| benign | PTH:0 | 99.2% (3968/4000) | [98.9, 99.4] | ✅ |
| benign | **QRY:1** | **96.4%** (3856/4000) | [95.8, 96.9] | ❌ |
| benign | QRY:0 | 99.0% (3959/4000) | [98.6, 99.2] | ✅ |
| phishing | **PTH:1** | **97.5%** (3901/4000) | [97.0, 98.0] | ❌ |
| phishing | PTH:0 | 98.8% (3953/4000) | [98.4, 99.1] | ✅ |
| phishing | **QRY:1** | **76.7%** (3067/4000) | [75.3, 78.0] | ❌ |
| phishing | QRY:0 | 99.0% (3960/4000) | [98.6, 99.3] | ✅ |

The rule is on point estimates in all 8 cells; the intervals are reported, not
used. benign QRY:1 and phishing QRY:1 fail with their whole interval below the
bar; phishing PTH:1's interval reaches it (upper bound 98.0%).

## 2. The joint settings — descriptive, n = 1,000 each

| class | setting | PTH honoured | QRY honoured | both |
|---|---|---:|---:|---:|
| benign | P:1 Q:1 | 99.2% | 97.0% | 96.4% |
| benign | P:1 Q:0 | 99.5% | 99.4% | 98.9% |
| benign | P:0 Q:1 | 95.3% | 98.7% | 94.1% |
| benign | P:0 Q:0 | 99.2% | 99.1% | 98.3% |
| phishing | P:1 Q:1 | 98.0% | 72.8% | 71.8% |
| phishing | P:1 Q:0 | 96.9% | 98.5% | 95.4% |
| phishing | P:0 Q:1 | 92.4% | 84.2% | 79.4% |
| phishing | P:0 Q:0 | 98.5% | 98.8% | 97.4% |

Asking for a query **without** a path is the hardest combination for the path
control in both classes (95.3% / 92.4%).

## 3. How the cells fail — descriptive, from the saved URLs

| cell | failures | how |
|---|---:|---|
| benign PTH:1 | 23 | all ended at the host |
| benign QRY:1 | 144 | 67 ended at the host, 68 after the path, 9 opened an empty query |
| phishing PTH:1 | 99 | 98 ended at the host, 1 skipped to a query |
| phishing QRY:1 | 933 | 716 ended after the path, 213 at the host, 4 opened empty |
| the four "field off" cells | 32 – 47 each | the field appeared anyway |

**Where a field was required, the URL almost always ended before reaching it**
— 1,162 of the 1,176 failures across the three failing cells (135 + 98 + 929
of 144 + 99 + 933); 13 opened an empty query and 1 skipped the path. That is the
residual the pre-registration anticipated: "residual failures would come from
generation ending first". The control is learned — under `<Q:1>`, 96.4% of
benign URLs carry a query, against 6.4% of real benign validation URLs and 4.5%
of E5's unconditional generations (E7's own unconditional rate is C4, not
measured here) — but sampling still draws the end token with whatever
probability the model leaves on it.

**Where a field was forbidden, it appeared in 0.8–1.2% of URLs.** From the
string alone it cannot be told whether the model emitted the field's marker or
wrote a delimiter (`/`, `?`) inside another field; both produce the same URL.

Why phishing QRY:1 is so much weaker than benign QRY:1 is **not established**
here.

## 4. Consequence — as registered

- **E7 is not adopted as a control** (section 4, C1: "FAIL in any cell").
- **C2, C3 and C4 are not run under E7.** The C2 generations produced on Kaggle
  (`Downloads\results (4).zip`) were never opened: under the amendment of
  2026-09-23 they are used only if C1 passes.
- **The follow-up the pre-registration names — hard decode-time constraints —
  is a separate experiment:** HC1, `research/hc1/PREREG_HC1.md`. It trains
  nothing, leaves this result unchanged, and registers its own use of the
  preserved generations.

## 5. What this does not say

- Nothing about E7's **unconditional** mode: that is C2 and C4, not run here.
- The quick check printed by the training notebook (n = 200, no delta) is not
  this test; it pointed the same way (phishing QRY:1 78.0%).

> **Note, 2026-09-24.** E7's own unconditional query rate, which section 3
> could not cite, has since been measured in HC1's C4
> (`research/hc1/HC1_RESULTS.md`): 3.62% benign, 1.74% phishing. This result
> is unchanged.

## Reproduce

    python research/e7/c1_honoring.py --checkpoint data/best_e7.pt \
        --bundle data/bundle2 --out data/e7_eval/c1
