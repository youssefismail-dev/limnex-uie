# The typicality scorer — card (`uie/typicality.py`)

**What it gives:** for any URL, `log P(url | benign)` and `log P(url | phishing)` under the
adopted engine, and their difference `llr` (positive = more phishing-like). Each is the
probability of the whole URL, as the engine encodes it for training.

```python
from uie.typicality import TypicalityScorer
s = TypicalityScorer.load("data/e16/train/e16_cap/final.pt")
s.score(["https://www.wikipedia.org/wiki/Cat"])
# [{'llr': -17.08, 'log_p_phishing': -77.31, 'log_p_benign': -60.23, 'tokens': 34, 'truncated': False, 'idn': False}]
```

| | |
|---|---|
| weights | `e16_cap` `final.pt`, sha256 `98b10b46…` — the adopted `e16_b5` (its decode rules affect generation only) |
| trained on | `data/bundle2` training streams (Common Crawl benign, 8 crawls 2024–2026; OpenPhish phishing 2024-10-12 to 2026-08-03) plus E15's benign query stream |
| input | any URL; scheme read and dropped, host lowercased, the rest unchanged; scored in the `domain` frame |
| long URLs | the first 384 tokens scored, flagged `truncated` |
| IDN hosts | scored as given (punycode or Unicode), flagged `idn` |
| speed | about 36 scorings per second on the laptop CPU (one scoring = one URL under one class) |
| dependencies | `uie` and `torch` only; a consumer imports `uie.typicality` and nothing else |

**What it is not.** A density, not a verdict, and not calibrated: use it as a feature or rank it.

**Known limits** (README, "Known limitations"):
- **Rare real words look unusual.** The engine knows fewer rare real words than real data
  contains, so a legitimate URL with uncommon words can score as atypical for *both* classes;
  `llr` is less affected than either log-probability alone, but not immune.
- **It reflects its training data.** Phishing patterns seen by OpenPhish up to August 2026 and
  the benign web of the same crawls. URLs it trained on are scored as memorised, not judged;
  a consumer measuring it must exclude them (`research/d1/D1_PLAN.md`, the leakage audit).

**Checks (2026-10-15):** 12 unit tests (`tests/test_typicality.py`); an outside script importing
only `uie.typicality` loads the real weights and scores URLs (D1a's gate, `DESIGN.md` §9). On
1,000 of UIE's own held-out names, phishing outscores benign with AUC 0.997 — an
in-distribution sign check, not a measure of detection on anyone else's data.
