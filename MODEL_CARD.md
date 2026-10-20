# Model card — Limnex UIE, engine `e16_b5`

**Limnex URL Intelligence Engine** (Limnex UIE) is a generative model of URLs, trained from scratch.
Given a class — *benign* or *phishing* — it writes new URLs character by character; the rare exact
copies of training URLs are removed by the known-URL filter, which the public interface requires
for every generation, benign and phishing. It can also score how typical a URL
is of each class. It exists for **defensive
security research and education** only: read [Responsible Use](#responsible-use) and
[No Liability](#no-liability) below before using it.

## Model details

| | |
|---|---|
| **engine** | `e16_b5`, adopted 2026-10-08 (`research/native/DECISION_2026-10-08.md`) |
| **what it is** | a character-level, class-conditional, decoder-only transformer, trained from scratch: no word list, no pretrained model, no templates |
| **size** | 25.5 million parameters: width 512, 8 layers, 8 attention heads, feed-forward 2,048, context 384 tokens, a 110-symbol alphabet |
| **how a URL is written** | each URL is generated under a plan (with or without a path, a query and a root slash) drawn from training statistics, at temperature 1.0, with three structural rules that forbid only what real URLs never do: D1 and D1′ (the root slash) and B5 (planned queries) |
| **weights** | `limnex-uie-e16_b5.pt`, 101,922,403 bytes, sha256 `98b10b4663ab85ea4af4368b6a09ded1bd5e809d077aaf4f8f1fea013037d65f` (the checkpoint `e16_cap/final.pt`) |
| **known-URL filter** | `limnex-uie-known-urls.bloom.gz`, 35,188,207 bytes, content sha256 `3ef34d06ff0e4735b1810fa3592d03ee182e748e05b08927c9eff53bde377eae`: a Bloom filter of the 14,873,392 collected URLs on the training side of the split — every extracted Common Crawl URL of a training site and every training phishing URL, not only the rows the model saw (expected false-positive rate 0.0115%); generation requires it and uses it to remove copies of real URLs (a phishing copy is a real phishing URL; a benign copy can carry personal data). It answers only "was this exact URL collected?" for a URL one already has; it cannot list any URL |
| **training** | 20,000 steps, batch 256 with 2 accumulation steps, learning rate 3e-4, about 3.5 hours on a free Kaggle T4 GPU; the full recipe is `E16_B5.recipe` in `uie/baseline.py` |
| **interfaces** | `uie.load` → `Engine.generate` (always passed through the safety gate) and `Engine.score` (log P(url \| class)); the `uie` command line — see the README's *Quick start* |
| **license** | Apache License 2.0, for the code and the weights (`LICENSE`, `NOTICE`) |

## Intended use

- Generating synthetic benign and phishing-style URLs to **train, test and stress phishing
  detectors** that you own or are authorized to test.
- Research on how URLs are structured, and teaching.
- Scoring URLs (`score`) as a research signal: how typical a URL is of each class.

## Out of scope

- **Anything offensive or deceptive** — see [Responsible Use](#responsible-use).
- **Production phishing detection.** The score is a likelihood, not a validated detector, and the
  engine was not evaluated as one.
- **Unicode URLs.** Non-ASCII characters are outside the 110-symbol alphabet and become an unknown
  token; internationalised hosts are modelled only in their punycode (`xn--`) form.
- **Claims that people cannot tell its URLs from real ones** (see *Human evaluation*).

## Training data

| | source | training rows |
|---|---|---|
| benign | Common Crawl: only the 8 crawls covering the phishing feed's window (2024-10 to 2026-08), so time cannot give the class away; extracted toward 1,000,000 registrable names and split into two corpora — by name, one URL per domain (906,214 rows), and by structure, up to 40 URLs from domains with at least 32 pages (1,648,460 rows) | 2,554,674 |
| added benign queries | a stream of 57,292 benign query rows from E15 (sha256 `637accab…`) | 57,292 |
| phishing | the OpenPhish community feed, its public git history at commit `48f7a982b2a4` (11,934 commits, about 22 months); by name (192,652) and by structure (46,491) | 239,143 |

- Validation and test splits are disjoint from training by registrable domain (benign) and by host
  (phishing); every evaluation below uses held-out real URLs.
- **The data is not redistributed.** Common Crawl's terms do not clearly permit redistributing
  derived datasets, so the repository publishes the recipe that rebuilds it: the phishing side
  byte for byte, the Common Crawl side by the same method and scale but not as the same sample
  ([REPRODUCE.md](REPRODUCE.md), 2a). The exact benign sample cannot be guaranteed again: the
  original extraction met network failures it did not record, and a failed read then drew a
  different block (since fixed; a clean-room rebuild of one crawl shared 66% of its URLs).
- **The original benign data is not shared, publicly or on request.** Common Crawl URLs carry
  personal data by accident (e-mail addresses, session and reset tokens: 823 such URLs in the two
  validation files alone). So a stranger's final scores are close to the published ones, not
  identical. Our own clean-room rebuild read 29 / 1 / 2 gated cells CLEARED / CONVICTED /
  UNDECIDED against the published 29 / 2 / 1: free mode and the one substantial weakness unchanged,
  three benign control cells near the 0.05 bar moved, by up to 0.019 ([REPRODUCE.md](REPRODUCE.md),
  2c).
- The phishing feed is a community *sample* of one provider's detections: a lower bound on real
  phishing, not all of it. The benign sample is a sample of Common Crawl, not of the whole web.

## Evaluation

All numbers are from `research/E3_FINAL_REPORT.md` unless stated otherwise.

**The E3 gate (the project's finish line).** For each URL field (label, suffix, subdomain, path,
query), a detector is trained to tell the engine's URLs from real held-out URLs, and its AUC is
compared with the AUC that real data reaches against itself: x = AUC(real vs generated) −
AUC(real vs real). A cell is CLEARED when x is proven below 0.05 (5 independent generations, 95%
confidence).

| measure | result |
|---|---|
| gated cells CLEARED | **29 of 32** (9 of 10 in free mode) |
| root slash at the real rate | **8 of 8** rows |
| plan obedience | 99.55–99.81% |
| free-mode whole-URL detector (auxiliary, not gated) | x = +0.0150 (benign), +0.0146 (phishing) |
| outstanding | free phishing QRY x = 0.0733 (CONVICTED); control benign P1Q1 PTH 0.0539 (CONVICTED, 0.004 above the bar); control benign P1Q1 SUB 0.0431 (below the bar, not proven) |

E3 was closed **NOT MET** on 2026-10-08 because of these three cells.

**Vocabulary (the word tell).** A detector that checks whether each word occurs in real training
data separates whole generated URLs from held-out real ones: excess over the real-vs-real floor
+0.168 (benign) and +0.164 (phishing), both CONVICTED. Character-level detectors do not (+0.027
benign, CLEARED; +0.044 phishing, UNDECIDED). The engine writes 1.45 words per benign URL that never
occur in real data, against 0.93 for real URLs.

**Copying.** The weights' samples copy training strings less often than real held-out data repeats
them, in every gated cell (`research/e16/E16_RESULTS.md`, gates G6–G9, under E16's decode). Whole
URLs are still copied sometimes: of 300 phishing samples, 3 were exact copies of training URLs
(`research/d2/ENGINE_VERIFY.json`). The public interface therefore generates phishing URLs only
with the known-URL filter, which removes such copies.

**Human evaluation (final).** Ten people each judged five pairs of same-class URLs (100 URLs: 50
real held-out, 50 generated): 55 of 100 correct (95% interval 45–64%; chance 50%; p = 0.37). That
is no evidence that people can tell the URLs apart, but 100 URLs cannot rule out a modest ability.
This is the project's final human evaluation: no larger panel will be run
(`research/eval_pages/HUMAN_PILOT_RESULTS.md`).

**Representation probe.** A straight-line reader of the model's internal state recovers the
*page's* language (a label the model never saw) on generic-domain websites: 60.5% against 12.5%
chance and 47.4% for an untrained copy of the model, so training put it there. A reader of plain
character snippets of the same URLs does slightly better (62.7%): the probe found no evidence that
the model knows more than the URL's text shows (`research/probe/RESULTS.md`).

**Reproducibility.** All 50 files the gate read were regenerated byte for byte from the release
alone (on a Kaggle GPU), and files of both modes on a CPU;
the public interface's decoder returns exactly what the research pipeline returns
(`research/d2/ENGINE_VERIFY.json`); the phishing corpus rebuilds byte for byte from the pinned feed
commit.

## Limitations

- **Vocabulary:** it invents more non-words than real URLs contain (above).
- **Phishing query strings in free mode** are its single substantial weakness: real phishing
  reuses kit strings, and the engine invents where real phishing repeats.
- **Two small control-mode cells** (benign URLs planned with both a path and a query).
- **Surface-level knowledge:** the probe found it knows a page's language no better than the URL's
  own text does; platform and region could not be probed (no labels).
- **Coverage:** URLs from 2024–2026, one public phishing feed, a Common Crawl sample; about 37% of
  the held-out benign pages are in English. Behaviour on later URLs is unmeasured.
- **Unicode** is not modelled (above); **generated URLs may coincide** with real domains — the
  gate screens copies of training URLs and, given a snapshot, live threats, but it cannot know the
  whole web.
- These boundaries are recorded in the README's *Known limitations*; resolving them is deferred to
  a later major version.

## Ethical considerations and dual use

The engine produces realistic phishing-style URLs, which is what makes it useful for defence and
also what makes it open to misuse. The project limits this by what it ships and how it behaves —
no bulk corpus of generated phishing URLs, offline generation that never queries DNS, a gate on
every generated batch, and the statement below. Brand names can appear in generated URLs because
the model learned from real phishing; that implies no affiliation and nothing about those brands'
own sites.

<!-- the Responsible Use and No Liability text below is identical to the README's (tests/test_api.py) -->

<!-- RESPONSIBLE-USE:BEGIN -->
## Responsible Use

**Read this before downloading, installing or using any part of this project.** By doing any of
those things — with the code, the model weights or anything they generate — you confirm that you
have read this section and the No Liability section below, and that you accept them.

Limnex UIE generates realistic URLs, including **phishing-style URLs**. It exists for one purpose:
**defensive security research and education** — testing, training and evaluating phishing
detectors, studying how URLs are structured, and teaching. It is not built, offered or supported
for anything else.

**Intended uses**
- Training, testing and red-teaming phishing detectors and filters that you own or are explicitly
  authorized to test.
- Measuring where detectors fail, and research on how URLs are structured.
- Teaching and coursework.

**Prohibited uses — the authors do not authorize, support or condone, and you must not:**
- use the code, the model or any output for phishing, fraud, impersonation, social engineering,
  spam, malware delivery or any other deception of real people or organizations;
- register, buy, host, resolve, visit, publish as working links, or send to anyone any URL or
  domain the engine generates;
- impersonate, or imply an affiliation with, any real brand, organization or person;
- test any system, person or organization without explicit, prior, written authorization;
- distribute bulk collections of generated phishing-style URLs;
- use it in any way that breaks a law, a regulation, the rights of others, or the terms of a
  service that applies to you.

**What the project does to limit misuse.** It ships the tool, not the ammunition: no bulk corpus of
generated phishing URLs is published. Generation runs offline and never queries DNS, and the project
never registers or resolves a generated domain. Every generated batch passes a filter for validity,
duplicates, copies of known training URLs and — when you supply a current threat-feed snapshot —
known live threats. Before sharing any generated URL, screen it against a current threat feed and
replace any domain that could belong to someone.

**Generated URLs are synthetic.** Any resemblance to a real domain, brand, organization or person
is coincidental and unintended. Brand names can appear because the model learned from real
phishing; this implies no affiliation or endorsement, and says nothing about the safety of that
brand's own sites. All trademarks belong to their owners.

**Reporting.** If you find a problem — for example a generated URL that matches a live threat —
report it privately through the repository's *Security* tab (a private advisory), not in a public
issue.

## No Liability

THE CODE, THE MODEL WEIGHTS, THE DOCUMENTATION AND EVERYTHING THEY PRODUCE ARE PROVIDED "AS IS",
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING ANY WARRANTY OF
ACCURACY, RELIABILITY, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT.

TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, LOSS OR OTHER LIABILITY — DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, CONSEQUENTIAL OR EXEMPLARY — WHETHER IN CONTRACT, TORT (INCLUDING NEGLIGENCE)
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THIS PROJECT, ANYTHING IT PRODUCES, OR THE
USE OR MISUSE OF EITHER BY YOU OR BY ANYONE ELSE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

- **You use this project entirely at your own risk.** You alone are responsible for your use, for
  anything you create with it, and for complying with every law that applies to you — including
  laws on computer misuse, fraud, privacy, data protection, trademarks and export.
- **Indemnity.** To the fullest extent permitted by law, you agree to indemnify and hold harmless
  the authors, contributors and copyright holders against any claim, loss or expense arising from
  your use or misuse of the project.
- **Not a security product.** Its outputs and scores can be wrong; it is not a phishing detector;
  nothing in this repository is legal, security or professional advice.
- **Relation to the license.** The license is the Apache License, Version 2.0 (`LICENSE`). Its
  Section 7 (Disclaimer of Warranty) and Section 8 (Limitation of Liability) apply in full; this
  section adds to them and does not limit them. The Responsible Use section states the uses the
  authors intend and those they prohibit.
- **Severability.** If any part of this section is held unenforceable, the rest remains in effect.
<!-- RESPONSIBLE-USE:END -->
