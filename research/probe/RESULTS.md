# The representation probe — results: REPRESENTS LANGUAGE (below surface)

Plan: `PROBE_PLAN.md` (approved and registered, ab18b2a). Read once, 2026-10-17
(`run_probe.py` → `results.json`, `run_probe.log`). Laptop, inference only, 15 minutes.

## In plain words

- UIE was never told what language a website is written in. A straight-line reader of its
  internal state guesses the **page's** language correctly **60.5%** of the time on generic-domain
  websites (8 languages, chance 12.5%).
- That is **13 points more** than the same reader gets from an untrained copy of UIE (47.4%): the
  difference is what training put there. **Reading: REPRESENTS LANGUAGE.**
- But a reader given only character snippets of the URL does slightly better (**62.7%**, 2.2
  points more, significant). So the language UIE knows is about what the URL's text already shows;
  this probe finds **no evidence of knowledge beyond surface statistics**.

## Validity

| check | result |
|---|---|
| V1: the probe reads the model's real forward pass | log P from the hooked pass = the scorer's, max \|diff\| 7.6e-6 nats (limit 1e-4) — PASS |
| V2: positive control (country vs generic TLD, all-TLD set) | balanced accuracy 0.996 (bar 0.95) — PASS |

## The main set: generic TLDs (no country code to give the answer away)

3,200 URLs read (8 languages × 400; 800 more chose the layers). Chance 0.125.

| reader on | layer | accuracy (95% interval) |
|---|---|---|
| **UIE** (`e16_cap/final.pt`) | 5 | **0.605** (0.589–0.622) |
| untrained copy (same architecture, random weights) | 2 | 0.474 |
| character 1–4-grams of the URL (surface) | — | 0.627 |

| difference | value (95% interval) | reading |
|---|---|---|
| UIE − untrained | **+0.131** (+0.111 to +0.152) | above 0 → learned |
| UIE − surface | **−0.022** (−0.037 to −0.006) | **below surface** |

**Reading: REPRESENTS LANGUAGE; against the surface control: below surface.**

## Reported beside it (decides nothing)

- **All-TLD set** (68% country domains): UIE 0.766 (0.751–0.780), untrained 0.637, surface 0.776;
  UIE − untrained +0.129; UIE − surface −0.010 (−0.023 to +0.003): *equal to surface*.
- **Layer curve (main set, 80% part):** UIE 0.462 (embedding) → 0.543, 0.568, 0.592, 0.602,
  **0.605 (layer 5)**, 0.599, 0.600, 0.606; the untrained copy stays flat at 0.45–0.49. The
  language signal is built up over the first five blocks, then held.
- **Per language (main set, UIE):** Spanish 0.69, French 0.69, German 0.67, Chinese 0.64,
  Indonesian 0.63, Russian 0.61, Japanese 0.57, **English 0.37**. English is the default language
  of `.com`, so it is the one most often confused with the others. The largest confusions: Chinese
  read as Japanese (48 of 400), Spanish as English (48), English as French (47).

## What it means

- **Supported:** training put language information into UIE's internal states that the
  architecture alone does not give — learned with no supervision, as DESIGN.md §5 predicted
  ("very likely emergent").
- **Not supported:** that UIE's knowledge of language goes beyond the URL's surface text. A bag of
  character snippets carries as much or slightly more. On this probe, UIE's representation
  behaves like a learned summary of the surface, not like a model of the web beyond it.
- **Untested:** platform and region (no labels in the corpus).
- The untrained copy is high (47%) because averaging random character embeddings is already a
  crude bag of characters; that is why the surface control, not chance, is the bar for the
  world-model question.

## Predictions [HYPOTHESIS] against the outcome

| prediction | outcome |
|---|---|
| REPRESENTS LANGUAGE, about 85% (UIE 50–70% on generic TLDs) | REPRESENTS LANGUAGE, 0.605 ✓ |
| against surface: equal 50%, below 30%, beyond 20% | below (main set); equal (all-TLD set) |
| best layer in the middle (3–6) | layer 5 ✓ |
