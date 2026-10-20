# Human pilot, 2026-10-15: can people tell `e16_b5`'s URLs from real ones?

**Result: no evidence that they can.** Ten people judged 55 of 100 URLs correctly (chance is
50%). The test is small, so it cannot show that people *can't* tell: a modest ability is not
ruled out.

## Design

- **Participants:** ten volunteers (students and a professor), one after another on one laptop.
  Self-reported: 3 look at URLs as part of work or study, 3 often, 2 sometimes, 2 rarely; 2 had
  seen this project's URLs before.
- **What each person saw:** 5 pairs of URLs, two at a time, both of the same class (ordinary
  website or phishing-style, shown on screen). The question: *which is real?* Answers: only A,
  only B, both real, both generated. Each person's 5 pairs were 3 real-vs-generated, 1 both
  real and 1 both generated, in a shuffled order; people were told any combination could occur.
- **The URLs:** 100 different URLs, 10 per person, none repeated and no host used twice.
  - real: the gate's whole-URL held-out population (the `mixed` validation pool of
    `data/bundle2`, one URL per host), never used in training: 25 benign, 25 phishing;
  - generated: `e16_b5`'s free-mode files (`data/e17/decode/gen`), 5 from each of the 5
    generation files per class: 25 benign, 25 phishing.
  - Drawn by seed 20261015 (`build_eval_sets.py`); nothing hand-picked. In the 30 mixed pairs
    the real URL sat on side A exactly 15 times, so a side preference earns nothing.
- **Blind:** the page held URLs only; the answer key was a separate file, read only for grading.
- **Decision rule, stated before the study:** with 100 judgements, 61 or more correct is better
  than chance (two-sided 5%).

## Results (`grade_laptop.py`)

Validity: all 10 participants, every pair answered exactly once, each by its own participant,
and the export matches the set's hash.

Each answer judges both URLs of its pair, so 50 answers give one judgement per URL:

| | result |
|---|---|
| URLs judged correctly | **55 of 100 (55%)**, 95% interval 45–64%; binomial p = 0.37 |
| the same, allowing for each person's own habits (labels shuffled within each person) | p = 0.43 |
| real URLs called real | 31 of 50 (62%) |
| **generated URLs called real** | **26 of 50 (52%)** |
| benign | 29 of 50 (58%); called real: real 15/25, generated 11/25 |
| phishing | 26 of 50 (52%); called real: real 16/25, generated 15/25 |
| exact answers (all four options; chance about 25%) | 15 of 50 (30%) |

Per person, URLs judged correctly out of 10: 6, 7, 7, 2, 4, 7, 4, 6, 7, 5, the spread chance alone
produces at 10 judgements each. The two people who had seen the project's URLs scored 6 and 7.
No one left a comment.

## What it supports, and what it does not

- **Supports:** in this test, people could not reliably pick out `e16_b5`'s URLs; generated URLs
  were called real about half the time. Phishing-style URLs were a coin flip.
- **Does not support:** "people cannot tell them apart". The interval reaches 64%, so a modest
  ability, of the size the word-level detector shows, is not excluded at 100 URLs. Nothing per
  person, and nothing about security experts as a group.

## Deviations

- The design was settled in conversation with the owner on 2026-10-15, not in a registration
  document; the decision rule above was stated before any answer existed. The page moved from
  single URLs to pairs before the study ran.
- `build_eval_sets.py` drew the set before it was committed; the URLs are reproduced exactly
  from the committed script and seed (the set hash below is unchanged by later edits).

## Records (git-ignored: they contain real phishing URLs)

| file | sha256 |
|---|---|
| `data/eval_pages/answers/laptop_page_as_used.html` (the page the participants used) | `03b55d2c7bed5a9a…` |
| `data/eval_pages/answers/uie_human100_laptop-20261015_all.json` (the answers) | `b3afbda2c7614509…` |
| `data/eval_pages/answers/laptop_key_as_used.json` (the key) | `6222eb5c73053a66…` |
| the set itself (`set_sha256` in the page and the answers) | `ed8707a0c61dd157…` |

A second, one-person page (`human_eval_10_remote.html`, 5 pairs from 10 further URLs) is built
for one more participant; it is not part of this result.
