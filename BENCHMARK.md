# UIE Benchmark — E0

**Frozen 2026-08-03.** This document states what the corpus contains, how to
reproduce it exactly, and — at least as importantly — what it does *not*
establish.

It exists before any model does. A generator cannot be evaluated against a
yardstick invented after seeing its output, and "the samples look realistic" is
not a measurement.

---

## 1. Reproducing this corpus

The URLs are **not** distributed. Common Crawl's terms do not clearly permit
redistributing derived datasets, so the release ships the **recipe**: a seeded
sampler plus pinned sources. Running it reproduces the corpus byte-for-byte.

```bash
python -m uie.corpus.build        --out data/corpus --blocks 200   # benign
python -m uie.corpus.feeds                                          # phishing
python -m uie.corpus.build_splits --corpus data/corpus              # splits
python -m uie.corpus.benchmark_stats                                # this document's numbers
```

| Source | Pinned to |
|---|---|
| Benign | 16 Common Crawl indexes, seed `20260803`, 200 blocks/crawl |
| Phishing | `openphish/public_feed` @ `48f7a982b2a4`, 11,934 commits |
| Phishing content hash | `e37135db3b661fb5…` |
| Split (random) | `a758774cb3771799…` |
| Split (temporal) | `93ea36bd18b6b47d…` |

`load_split` refuses any split file whose contents no longer match its recorded
hash. A split that can be silently regenerated is not a split, and results
measured against an edited one are not comparable to earlier ones.

---

## 2. Composition

| | URLs | domains | distinct suffixes |
|---|---|---|---|
| **benign** | 6,496,447 | 99,471 | 250 |
| **phishing** | 354,143 | 94,234 | **719** |

Benign is Common Crawl only. No curated corpus contributes to the learned benign
distribution: measured against the real web's 2.9% bare homepages, curated benign
sets run 12–17%, and two of them 98.6% and 100.0%. Since Common Crawl supplies
unlimited benign URLs at better quality, adding curated data could only degrade
it.

Phishing is the OpenPhish live feed, backfilled from its git history — 22 months,
each URL stamped with the earliest commit containing it. Historical archives
(GramBeddings, Kaggle, Mendeley) are available as supporting volume but are **not
in this frozen corpus**; the live feed proved 99.99% disjoint from them (34
overlapping URLs in 313,023), so they are two different populations rather than
one with more of it.

**Era-matching.** Benign is drawn from crawls contemporaneous with the phishing
window rather than from the newest index. Pairing 2026 benign with 2021 phishing
would teach the model that current-era styling is benign — a temporal confound of
exactly the kind §5 documents.

---

## 3. Measured statistics

Reported **per class and per stratum, never pooled.** A pooled number lets a
model be excellent on English `.com` and useless on Japanese `.jp` while still
looking fine — the failure an Internet World Model must not be permitted to hide.

### By class

| | n | bare | query | path depth | subdomains | length | label entropy |
|---|---|---|---|---|---|---|---|
| benign | 6,496,447 | 2.9% | 17.3% | 2.28 | 0.61 | 80.7 | 2.79 |
| phishing | 354,143 | **47.4%** | 6.3% | **0.94** | 0.85 | 49.7 | 2.62 |

### By era (benign)

| | n | bare | query | path depth | length |
|---|---|---|---|---|---|
| historical (2019–2022) | 3,060,087 | 3.2% | 20.6% | 2.33 | 83.6 |
| live (2024–2026) | 3,436,360 | 2.6% | 14.4% | 2.24 | 78.2 |

### By suffix kind

| | n | bare | path depth | subdomains | entropy |
|---|---|---|---|---|---|
| benign / legacy gTLD | 3,552,341 | 2.7% | 2.29 | 0.61 | 2.87 |
| benign / ccTLD | 2,725,039 | 3.1% | 2.28 | 0.62 | 2.68 |
| benign / new gTLD | 219,066 | 2.8% | 2.10 | 0.37 | 2.66 |
| phishing / legacy gTLD | 113,091 | 48.5% | 1.05 | 0.77 | 2.88 |
| phishing / ccTLD | 112,760 | 41.0% | 1.04 | 0.88 | 2.50 |
| phishing / new gTLD | 128,292 | **52.1%** | 0.76 | 0.88 | 2.48 |

### Suffix and language distribution

```
benign     com 44.2%  org 5.9%  ru 4.2%  de 3.9%  net 3.4%  nl 1.9%  it 1.7%
phishing   com 27.7%  io 14.5%  app 10.0%  dev 9.4%  top 3.6%  cc 2.6%  net 2.6%
```

**159 languages** in the benign side (from the Common Crawl index, not inferred):
`eng 48.8%, rus 5.3%, deu 4.8%, zho 4.6%, jpn 4.0%, fra 4.0%, spa 3.9%, por 2.1%`.

Two inversions worth noting. Phishing spans **719 distinct suffixes against
benign's 250** despite having 18× fewer URLs — attackers shop across the TLD space
far more widely than the web occupies it. And phishing concentrates on
developer/free-hosting suffixes (`io`, `app`, `dev` together 33.9%) that barely
register on the benign side.

---

## 4. Splits

Both are **frozen artifacts**, stored as domain → partition and verified by hash
on load. Grouping is by registrable domain, never by URL: benign runs ~65 URLs
per domain, so a URL-level split would put the same site on both sides and a
model could score well purely by having memorised it.

| split | partitions |
|---|---|
| random, domain-disjoint | train 154,935 · val 19,366 · test 19,369 |
| temporal, cutoff 2026-01-01 | train 151,367 · eval 42,303 |

**Straddling domains stay in train.** 1,459 domains appear on both sides of the
temporal cutoff. Promoting their later URLs to eval would measure "a new page on
a site the model already knows" — a far easier question than generalization to
new sites. They remain in train and their post-cutoff URLs are dropped. The count
is published because it is a real cost.

---

## 5. Known biases — read this before using the corpus

### 5.1 The path-depth shortcut *(the significant one)*

**Path depth alone separates the classes at AUC 0.791.** The rule
*"depth == 0 → phishing"* achieves **47.4% recall at 2.9% FPR** — from a single
integer, with no model involved.

This is almost certainly not a fact about phishing. It is a fact about **how the
two sides were observed**:

- a phishing feed reports the **entry point** a victim clicks — naturally shallow;
- Common Crawl indexes pages discovered by **following links** — naturally deep.

Supporting evidence that it is methodological: the direction of this correlation
*flips between corpora*. PhreshPhish shows phishing shallower (0.78 vs 2.22);
GramBeddings shows phishing **deeper** (2.87 vs 1.41). A property of phishing
would not reverse sign depending on who collected the data.

**Consequences.** For a generative model the effect is mild — it reproduces the
marginal, so roughly half of sampled phishing-style URLs will be bare domains.
The damage lands **downstream**: a detector trained on generated data would learn
"shallow ⇒ phishing" and fail on real traffic, having inherited an observation
bias as though it were signal.

**Not corrected here, deliberately.** Matching the depth distributions
artificially would replace a *known, measured* bias with a hidden one. Instead:
it is documented, depth-stratified evaluation is mandatory, and the shortcut AUC
is republished with every corpus version so drift is visible.

### 5.2 Feed bias in the phishing side

The suffix concentration on `io`/`app`/`dev` (33.9% combined) is unlikely to
reflect phishing in general. It reflects what OpenPhish's community feed detects
and publishes, which is a *sample* of their detections and a rolling window. Feed
coverage is a lower bound on real phishing, and nothing here should be read as
"all phishing in this period."

### 5.3 Block correlation in the benign sample

Common Crawl's index is SURT-sorted, so a sampled block is ~3,000 **contiguous**
lines — roughly 1,950 URLs across only ~33 registrable domains. The effective
sample size for **suffix and language shares** is therefore the number of blocks
drawn, not the number of URLs. Path-depth statistics are far better behaved (they
vary within a host) and were stable across two samples with entirely different
TLD mixes.

### 5.4 Timestamp resolution

Benign rows in this build are dated from their crawl ID (`CC-MAIN-2026-30` → ISO
week 30 of 2026), giving ~2-week resolution — finer than any cutoff applied here.
Per-URL capture timestamps are extracted correctly as of `3cc260d`; a
re-extraction would sharpen this but changes no conclusion.

### 5.5 Class collisions

35 domains of 193,670 (0.02%) appear in both classes. All are bare public-suffix
hosts (`appspot.com`, `blogspot.com`, `fc2.com`, `com.de`) where "the domain" is
not a unit of ownership. They are assigned by first sighting. Sub-domains of
those platforms separate correctly.

---

## 6. E1 — the baseline E2 must beat

An order-5 character n-gram with Witten-Bell smoothing, over **the same
factorization E2 will use**. It differs only in how the components are learned:
counting rather than gradient descent. So any win E2 posts is attributable to
learning, not to a better representation.

Fitted on 150,000 URLs per class from the `train` partition (a memory bound, not
a statistical one — n-gram counts saturate quickly on short strings). Evaluated
on held-out `test`.

### The bar

| metric | benign | phishing |
|---|---|---|
| **held-out perplexity** (per char) | **22.050** | **18.834** |
| — legacy gTLD / ccTLD / new gTLD | 20.37 / 24.12 / 21.31 | 19.85 / 18.17 / 18.62 |
| **gate pass rate** | 99.7% | 98.8% |
| **novelty rate** | 100.0% | 100.0% |
| suffix total-variation vs real | 0.155 | 0.307 |
| suffix coverage | 96.5% | 76.5% |
| Δ mean path depth | +0.41 | +0.46 |
| Δ mean URL length | −0.7 | +0.3 |
| Δ bare-homepage rate | −0.000 | −0.422 |

**Discrimination via log-likelihood ratio: balanced accuracy 0.885** (benign
0.913, phishing 0.858) — from a purely *generative* classifier that was never
trained to discriminate. E2's version of the same signal has to beat this.

### What the baseline actually produces

```
http://awr.oudlisyrroinaacogiriwwigdhmfowloenr.com/1o%sawvn?D07r%4fw5Co%52fr
https://utrcecwhtmiedbennia.com.au/60ailtn/-alke//ctcas7otir%/cy0ne3wie
https://cdtoun-u9-tnii-cblakf4sohces.top/euta.
http://f2zreukelbyoem.app/jx.
```

**This is the most important result in E1, and it is not a number.**

The n-gram is *strong* on everything measured automatically: structure is valid
99% of the time, suffixes are real, URL length is within 0.7 characters of the
truth, and the benign bare-homepage rate is exact to three decimals. Several
distributional metrics are nearly satisfied.

And no human would believe `awr.oudlisyrroinaacogiriwwigdhmfowloenr.com` could
exist. The model has learned *the shape of a URL* and nothing about **what web
names mean** — it produces character soup inside a perfect skeleton.

Two consequences:

1. **It validates making the comparative human panel the primary metric** (§3 of
   `DESIGN.md`). A model can nearly satisfy the automatic distributional checks
   while being obviously wrong to a person. Had the benchmark rested on those
   checks alone, this baseline would look close to solved.
2. **Novelty at 100% confirms novelty is a weak metric on its own.** Every sample
   is new, because gibberish is trivially new. Novelty is only meaningful
   reported *beside* quality, which is why the gate reports them together.

The failures point at where the capacity is missing: `label_too_long`
rejections (56 benign, 172 phishing) and a label-entropy deficit of −0.48 show
the n-gram generating long, repetitive labels. Real registrable labels are short,
varied and pronounceable. That gap — meaning inside the skeleton — is precisely
what E2 exists to close.

---

## 7. What this benchmark does not establish

- **It does not define phishing.** Phishing is intent plus served content, not a
  property of a string. Every label here means "reported by a feed as phishing"
  or "crawled from the open web" — nothing stronger.
- **It is not the web.** It is a sample of Common Crawl, which is itself a sample
  of the web, biased toward what is linked and crawlable.
- **It does not measure a generator.** That requires held-out real URLs, a
  discriminator, a novelty check and a human panel — E1 supplies the baselines,
  E3 the grading.
- **Passing it is necessary, not sufficient.** A model matching every statistic
  here has not proven it is good; a model failing one has proven it is bad,
  cheaply and diagnosably.

---

## 8. Corrections (recorded after E2.2)

Two defects in this document's own numbers were found while diagnosing E2.2.
Both are recorded rather than quietly repaired, because a benchmark whose
errata are invisible is worth less than one with none.

### 8.1 The E1 baseline was a unigram, not an order-5 n-gram

`CharNgram._prob` interpolated from high order **down** to the empty context, so
the final iteration — whose Witten-Bell weight is ~1.0, having seen every
character in the corpus — overwrote everything the longer contexts contributed.
Every prediction collapsed to the unigram distribution, at every order.

Confirmed directly: `P("l" | "secure")` on a corpus where `l` follows `...cure`
in one of five equally frequent words returns **0.0173** (the unigram frequency
of `l`) instead of **0.200**, identically for orders 2, 3, 5 and 7.

Nothing caught it. The tests asserted probabilities were finite, normalised, and
higher for corpus-like strings — all true of a unigram. Only an ORDER SWEEP
exposes it, and none existed; `test_higher_order_uses_longer_context` now does.

Consequences: the perplexities in §6 (**22.050** benign / **18.834** phishing)
were produced by a unigram model, so E2's headline "71.5% improvement over E1"
was measured against a far weaker baseline than this document claimed. The
qualitative E1 sample — `awr.oudlisyrroinaacogiriwwigdhmfowloenr.com` — is
character salad with correct letter frequencies, which is exactly what a unigram
produces, and should have prompted the question at the time. **E1 must be re-run
before any E1-vs-E2 comparison is quoted again.**

### 8.2 The Real-vs-Fake bar of 0.65 was below the metric's floor

The bar was pre-registered without a control. Measured after E2.2, on 1,200
samples per side, three draws:

| comparison | benign | phishing |
|---|---|---|
| real vs real, same partition | 0.478 / 0.521 / 0.504 | 0.476 / 0.497 / 0.487 |
| **real TRAIN vs real VAL** | **0.745 / 0.734 / 0.748** | **0.833 / 0.815 / 0.835** |

The same-partition control sits at 0.50, so the discriminator is unbiased. But
the split is DOMAIN-DISJOINT by construction, so real training URLs and real
validation URLs differ in site, TLD mix and language — and a char n-gram
separates them at **0.741 benign / 0.828 phishing**.

A generator trained on `train` cannot be less distinguishable from `val` than a
real sample of the web drawn the same way. **0.65 was unreachable by any
generator, including a perfect one, and by reality itself.**

The meaningful quantity is therefore EXCESS OVER FLOOR:

| | AUC | floor | excess |
|---|---|---|---|
| E2 benign | 0.9033 | 0.741 | 0.162 |
| E2.1 benign | 0.8815 | 0.741 | 0.140 |
| E2.2 benign | 0.8796 | 0.741 | 0.139 |
| E2.2 phishing | 0.8441 | 0.828 | **0.016** |

This is not the bar being relaxed to fit a result. The control is independent of
the model, returns the same numbers whether or not anything was ever trained,
and does not flatter the outcome — benign still plainly fails. What it changes
is that **phishing generation is already at the achievable floor**, a result
that sat unread through three iterations because the bar was set beneath it.

The floor is itself a symptom of corpus size: with 99,303 labels split into
disjoint halves, train and val occupy different regions of name-space. A larger
corpus lowers the floor and sharpens the benchmark as well as the model.


---

## 9. The evaluation that replaced section 6 (recorded after E2.2)

Section 8.2 showed the pre-registered Real-vs-Fake bar of 0.65 sat below its own
floor. This section records what replaced it and the evidence for it.

### 9.1 Calibration: what the discriminator does on REAL data

Seven conditions, three independent draws each, identical discriminator
throughout. Six are real-vs-real; the last is the anchor.

| condition | AUC |
|---|---|
| same crawl, random URL split | 0.5144 +/-0.0256 |
| domain-disjoint, same crawl | 0.5206 +/-0.0061 |
| natural repetition vs 1 url/domain | 0.5400 +/-0.0063 |
| crawls 4 months apart | 0.5717 +/-0.0428 |
| crawls 21 months apart | 0.6005 +/-0.0557 |
| **uniform vs stratified sampling** | **0.7911 +/-0.0057** |
| real vs GENERATED (benign, E2.2) | 0.8715 +/-0.0019 |

The instrument is unbiased (0.5144 on the null) and does NOT penalise the things
a generator legitimately does differently - domain-disjointness costs 0.02, not
repeating domains costs 0.04. But two sets of entirely real URLs differing only
in HOW THEY WERE SAMPLED separate at 0.7911. The metric measures "same
distribution or not", and is sharp at it.

### 9.2 Sensitivity: a corruption ladder

Real URLs degraded by controlled amounts; ground truth is the severity itself.

| corruption | old whole-URL AUC | new LBL excess |
|---|---|---|
| unmodified (control) | 0.4953 | -0.0299 |
| swap 2 characters | 0.5327 - **passes 0.65** | 0.0768 |
| replace 25% of characters | 0.5820 - **passes 0.65** | 0.1705 |
| replace 50% | 0.7208 | 0.3562 |
| shuffle characters | 0.7684 | 0.3944 |
| random CV syllables (E2's failure mode) | 0.9409 | 0.4864 |

Both are calibrated on the control and monotonic in severity. The old metric
**rates a label with a quarter of its characters randomised as acceptable**.

Repeating the ladder on PATHS explains why: there, a two-character swap already
gives 0.7871. **The whole-URL metric's sensitivity to a field is proportional to
that field's share of the string** - a label is ~10 of ~70 characters, a path is
half the URL. It was blind precisely where it mattered, because the registrable
label is the shortest content field and the one that dominated every diagnosis
this project made.

### 9.3 What is measured now

Each field scored against the population it was TRAINED on, with a floor from
real-vs-real on that same population:

    LBL, SUF, SUB  ->  name corpus val
    PTH, QRY       ->  structure corpus val
    FULL           ->  auxiliary, reported never gated

**excess = auc - floor.** A perfect generator scores 0 and could never have
scored 0.65 on the old metric. The 0.05 bar sits just outside the measured
spread of real-vs-real variation (0.514-0.601).

Whole-URL AUC is retained as auxiliary because field-scoped metrics are blind to
COMPOSITION errors - a perfect label beside a perfect path in an implausible
pairing passes both - and composition is the known weakness of field-scoped
training.

### 9.4 What this does not fix

Neither metric supports cross-field comparison: path excess at a nominally equal
corruption is 0.333 against the label's 0.077, because paths are longer strings.
The new metric gives a calibrated zero PER FIELD, so the same field is comparable
across models. That is the only comparison actually made.
