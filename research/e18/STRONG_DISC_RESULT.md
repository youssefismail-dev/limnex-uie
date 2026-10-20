# The stronger-discriminator test — result: DEEPER TELL (vocabulary)

- **Registration:** `research/e18/STRONG_DISC_PREREG.md` (83821c8), code 6e1b928, both
  committed before any detector ran. Read once, after all four finished (2026-10-08).
- **Record:** `data/e18/strong_disc/` (`D0`–`D3.json`, `result.json`).
- **Informational only.** E3 stays closed; nothing is tuned, reopened or adopted.

## In plain words

- **The engine's URLs survive every detector that reads characters,** but not one that
  knows words.
- A detector that checks whether the URL's words are real words — looked up in 2.8M real
  training URLs — separates `e16_b5` from real URLs clearly in both classes: **excess
  +0.168 benign, +0.164 phishing, CONVICTED.**
- **What it sees:** the engine invents words.
  - A generated benign URL holds 1.45 words never seen in real data, against 0.93 for a
    real held-out URL.
  - In paths and queries, 84% of generated words are known words, against 94% of real
    ones.
  - Every character n-gram in an invented word looks plausible, so character models can't
    tell. Only knowing the actual words can.
- **There is no structural tell.** The same detector restricted to lengths, counts,
  depth and character mix scores 0.50 (benign) and 0.55 (phishing) on its own.

This is the programme's "too inventive" finding seen at whole-URL level, and it is the
most important real-world qualifier on the engine's realism.

## 1. Every reading

Excess over the real-vs-real floor; mean of 5 generations; t against the 0.05 bar.

| detector | class | `e16_b5` | `e9_ln` (control P1) | chimeras (control P2) |
|---|---|---|---|---|
| D0 gate's model | benign | +0.0269 CLEARED | +0.2085 CONVICTED | +0.0878 CONVICTED |
| D0 gate's model | phishing | +0.0437 UNDECIDED | +0.4147 CONVICTED | +0.0849 CONVICTED |
| **D1 vocabulary + global** | benign | **+0.1675 CONVICTED** | +0.3348 CONVICTED | +0.0629 CONVICTED |
| **D1 vocabulary + global** | phishing | **+0.1637 CONVICTED** | +0.4523 CONVICTED | +0.1189 CONVICTED |
| D2 character CNN | benign | +0.0010 CLEARED | +0.1560 CONVICTED | +0.0024 CLEARED |
| D2 character CNN | phishing | −0.0001 CLEARED | +0.3806 CONVICTED | +0.0008 CLEARED |
| D3 character transformer | benign | +0.0085 CLEARED | +0.2044 CONVICTED | +0.0008 CLEARED |
| D3 character transformer | phishing | +0.0022 CLEARED | +0.3879 CONVICTED | +0.0016 CLEARED |

Every floor is within 0.005 of 0.5.

## 2. Validity, as registered

- **V-P1 PASS for all four detectors:** each convicts `e9_ln` in both classes.
- **V-floor PASS** for all four.
- **V-P2 — the deep models turned out weaker, not stronger.**
  - D2 and D3 miss the chimeras that even D0 catches (+0.085 to +0.088).
  - With 6,000 URLs a side, a small CNN or transformer trained from scratch catches only
    gross tells such as `e9_ln`'s. That fits the prediction that they would be
    data-limited.
  - So their CLEARED verdicts on `e16_b5` add little evidence.
  - D1 beats D0 on phishing chimeras (t > 2.776), but not on benign ones.

## 3. The reading

**Outcome: DEEPER TELL**, through D1 in both classes:
- convicted (benign t 40.1, phishing t 31.0);
- above D0 by +0.141 (benign) and +0.120 (phishing), paired t 41.1 and 20.1.

**Characterization (read-only, generation 0):**

| | benign | phishing |
|---|---|---|
| D1, structural features only | AUC **0.502** | AUC **0.552** |
| D1, vocabulary features only | AUC **0.650** | AUC **0.634** |
| never-seen words per URL, real → generated | 0.93 → 1.45 | 0.76 → 1.06 |
| longest never-seen word, real → generated | 8.9 → 10.3 chars | 6.0 → 7.4 chars |
| known words in path and query, real → generated | 94% → 84% | 96% → 93% |
| known words in the host, real → generated | 55% → 48% | 78% → 71% |

- The whole tell is in the words.
- The engine writes plausible-looking pseudo-words where real URLs reuse real ones.
  Example: `hirina-nadlya-izmir-suzuki-sovremennaya-priroditeley-khatiyunktiyurut`.
- This is distinct from copying, which is measured against the same training corpus and
  stays under the natural rate.

**Also worth noting:**
- D0 itself, with 5× the gate's data, moves phishing from the gate's whole-URL +0.015 to
  **+0.044 UNDECIDED**.
- So the gate's small sample (1,200 per side) also flattered the phishing whole-URL
  reading somewhat.

## 4. Predictions against outcome

| prediction | outcome |
|---|---|
| P1: every detector convicts `e9_ln` | ✓ |
| P2: D3, and probably D1, beat D0 on chimeras | ✗ for D3; D1 only on phishing. D0 caught chimeras better than expected |
| D0 near or slightly above +0.015 | +0.027 / +0.044 ✓ |
| **D1 the most likely to find a tell, through vocabulary** | **✓** |
| D2 roughly matches D0 | ✗: D2 was weaker |
| D3 data-limited | ✓ |
| a tell more likely in phishing than benign | ✗: equal |
| overall: HOLDS 35 / MIXED 35 / DEEPER TELL 30 | DEEPER TELL |

## 5. What this licenses, and what it does not

- **The realism claim must be scoped.** The engine's URLs are indistinguishable from real
  ones *to detectors that read characters*. They are **not** indistinguishable to a
  detector with word knowledge: any dictionary-, corpus- or language-model-based check.
  Real-world detectors increasingly have such knowledge, so this is a genuine limitation.
- **It gives a cause, not just a symptom.** It explains the "too inventive" findings in
  the E3 report, including the phishing-query weakness, and points to their common
  source.
- **Nothing is tuned against it, and E3 stays closed.** If the owner ever opens a new
  programme, "reuse real vocabulary at the real rate" is the measured target. A
  vocabulary-aware detector should then join the gate as a standing instrument, so the
  improvement is judged honestly rather than assumed.
