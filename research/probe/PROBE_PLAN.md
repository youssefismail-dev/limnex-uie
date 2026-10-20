# The representation probe — plan (APPROVED by the owner, 2026-10-17)

DESIGN.md §5 (c) and §6: *can a linear classifier read a concept off the model's internal states?*
If it can, the model learned that concept with no supervision. The owner approved running it
(2026-10-17); this plan fixes how, before anything is measured.

## In plain words

- UIE was only ever trained to predict the next character of a URL. Nobody told it what
  language a website is written in.
- For every held-out benign URL, Common Crawl recorded **the language of the page's text**
  (`lang`, detected from the page, not from the URL).
- So: feed held-out URLs through UIE, read its internal state, and train the simplest possible
  reader (a straight-line classifier) to guess the page's language from that state alone. If the
  reader succeeds **better than it can from an untrained copy of the same model**, UIE learned
  something about the web that nobody gave it.
- The honest catch: a country domain (`.de`, `.jp`) gives the language away for free. So the
  main test uses only websites on **generic domains** (`.com`, `.net`, `.org`, new gTLDs), where
  the language must come from the names and words themselves.

## What can and cannot be probed

| concept (DESIGN.md §5) | label source | probed? |
|---|---|---|
| language | Common Crawl's page-content language (`lang`) | **yes — the main test** |
| TLD category | the URL's own suffix | only as a validity check (it is written in the URL) |
| platform / CMS, region | no label in the corpus | **no** — any label would be a rule written by us, which the probe would then just detect |

## Data

- **Held-out benign URLs only:** `data/bundle2/name_val` + `structure_val` (323,085 URLs; never
  trained on; the adopted `e16_cap/final.pt`).
- **Main set (generic TLDs):** every language with at least 500 distinct registrable domains on
  generic TLDs (`und` excluded); **500 domains per language, one URL per domain** (no site twice,
  so the reader cannot memorise sites), seed 20261017. Expected about 10 languages; chance = 1/K.
- **Second set (all TLDs):** the same languages, 500 domains each from any TLD, same rules.
- Generic vs country TLD: the last label is a country code if it is two ASCII letters or an IDN
  label (`xn--`); otherwise generic.

## What is compared (same URLs, same reader)

| representation | what it shows |
|---|---|
| **UIE** (`e16_cap/final.pt`): the state after the embedding and after each of the 8 blocks (9 layers), averaged over every token after the fixed three-token prefix (`<bos>`, class `benign`, frame `domain`) — i.e. over the URL's own tokens, encoded exactly as the typicality scorer encodes | the learned model |
| **Untrained control:** the same architecture with random weights (seed 20261017) | what the architecture and raw characters give with no learning |
| **Surface control:** character 1–4-gram counts of the same URL (hashed, 2^18) | what the URL's surface text alone gives |
| chance | 1/K |

- **Reader:** multinomial logistic regression, L2, C = 1.0, features standardised (n-grams:
  sublinear term frequency); identical for every representation.
- **Layer choice, fair to both models:** 20% of each set (stratified) picks the best layer for UIE
  and, separately, for the untrained control (5-fold cross-validation inside that 20%); the other
  80% is read once, by 5-fold cross-validation (out-of-fold predictions). The 20/80 assignment and
  the folds are drawn when the sets are built and stored with them.
- **Metric:** accuracy on the balanced set (= balanced accuracy); 95% intervals by bootstrap over
  URLs (2,000 resamples; one URL per domain, so URLs are independent).

## The reading (fixed now), on the main set (generic TLDs)

| reading | rule |
|---|---|
| **REPRESENTS LANGUAGE** | UIE above chance (interval above 1/K) **and** above the untrained control (paired difference, interval above 0) |
| **NOT SHOWN** | otherwise |

**Reported beside it, deciding nothing:** UIE vs the surface control — *beyond surface* (interval
above 0), *equal to surface*, or *below surface* — which is the world-model-vs-Markov-chain
question of DESIGN.md §6; the all-TLD set; the per-layer curve; per-language accuracy and
confusions.

**Validity (must pass, or nothing is read):**
1. The extraction is the model's real forward pass: the hidden-state pass reproduces the
   typicality scorer's log P(url | benign) on 100 URLs within 1e-4 nats.
2. Positive control: on the all-TLD set, UIE's layer chosen there for language separates country
   from generic TLDs with **balanced** accuracy ≥ 0.95 (5-fold, the 80% part; a fact written in
   the URL, so failing it means the reading pipeline is broken; balanced, so an uneven
   country/generic mix cannot pass it by itself).

## Predictions [HYPOTHESIS]

- REPRESENTS LANGUAGE: about 85% (UIE around 50–70% on generic TLDs, chance about 10%).
- Against the surface control: *equal* about 50%, *below* 30%, *beyond* 20%.
- The best layer is a middle one (3–6), not the last.

## Cost and files

- Laptop, inference only (no training), low priority, RAM checked first: about 25–30 minutes
  (≈10,000 URLs through two models), then a few minutes of fitting readers.
- `research/probe/build_probe_sets.py` (the two sets, written and hashed first; no model run),
  `research/probe/run_probe.py`, `tests/test_probe.py`; results in `research/probe/RESULTS.md`.
- Nothing in `uie/` changes: hidden states are read with forward hooks on `URLTransformer.blocks`.
