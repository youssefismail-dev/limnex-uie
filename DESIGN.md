# Limnex URL Intelligence Engine — Design & Roadmap

**Name:** Limnex URL Intelligence Engine (Limnex UIE) — the owner's decision, 2026-10-17
(`research/d2/D2_PLAN.md`, decision B).
**Status:** architecture frozen 2026-08-03. Implementation proceeds phase by phase
per §9, and may refine this document when implementation uncovers better ideas.
**Form:** an independent open-source research project with its own repository.

---

## 1. What this is

**A program that reads millions of real URLs, learns how the web names things, and
then invents new URLs that have never existed but look completely real.**

It generates two kinds.

**Benign:**
```
cdn-assets-eu3.shopwave.io/static/v2/bundle.js
blog.tenjiku-books.jp/2024/03/review
support.northvale-clinic.org/patients/forms
```

**Phishing-style:**
```
secure-verify.examplebank-alerts.support/login
myaccount-billing.exampIebank-help.co/update
signin.examplecredit-notice.online/session
```

None of those exist. The model invented them. The test is whether an experienced
security researcher, shown these shuffled together with real URLs, could sort them.

**Nobody writes the rules.** We do not tell it that phishing URLs contain `login`,
or to swap Latin letters for Cyrillic lookalikes. It reads data and works out the
patterns itself. This is the defining constraint — a rule-based generator can only
ever produce attacks its author already knew about.

**The real web is the source of knowledge; curated corpora are curriculum.** No
assembled dataset is permitted to define what "normal" or "phishing" looks like.
Wherever the web is directly observable — which is the entire benign side — it is
observed directly. Curated corpora are a fallback only where the web cannot be
observed *with labels*, which is the phishing side, and their influence shrinks as
live collection accumulates (§9, E0).

One honesty note about that principle, because its strongest phrasing is not
implementable: a generative model is *defined* by maximizing likelihood over its
training data, so "do not optimize toward reproducing a distribution" cannot be
literally achieved — training is exactly that. The question is only **which**
distribution, and the answer is the web's rather than a convenience sample's.

**Generation is the objective.** The model also ends up able to score how typical
any URL is, which is genuinely useful — but it is a by-product, not the reason the
project exists.

**Benign and phishing are equally important.** Benign is not a supporting cast.
It is arguably the harder and more valuable half: phishing feeds are narrow and
curated, while benign URLs are the long tail of the entire web.

## 2. What this is not

- **Not a detector.** It never decides whether a URL is dangerous.
- **Not rules, templates, or mutations** of URLs it has seen.
- **Not a corpus dump.** It is a model you query, not a file.
- **Not part of SNOW Guard AI.**

---

## 3. Relationship to SNOW Guard AI

```
      URL Intelligence Engine   (own repo, own benchmark, own release)
                  │
                  │   generated URLs  +  typicality scores
                  ▼
            SNOW Guard AI    (one consumer among possible others)
```

**The dependency is strictly one-way.** UIE never imports from, is never graded
by, and never assumes the existence of SNOW Guard. It must be useful to someone
working on DGA detection, domain reputation, or registrar abuse who has never
heard of SNOW Guard AI.

**SNOW Guard is not blocked on it.** SNOW Guard's own path is V2 → V3. If UIE
never delivers, that roadmap is unaffected. SNOW Guard holds a *gated integration
point*, not a dependency: when UIE reaches E4, test its typicality score as a
fifth ensemble member against the frozen `benchmark_v2`, with a pre-registered
kill criterion — ship only if it wins.

**Rule-based attack generators stay with SNOW Guard**
(`phishguard-ml/docs/ADVERSARIAL_VECTORS.md`). They grade a detector, so they
belong on the detection side. Later they serve as *baselines* UIE must beat.

---

## 4. Design

### 4.1 Raw URLs

UIE generates complete raw URLs, scheme included. It does **not** generate in
SNOW Guard's normalized space, even though that would save capacity — baking one
consumer's preprocessing into a general library makes every other consumer inherit
a normalization they never asked for. Consumers normalize on their own way in.

### 4.2 Structure is enforced where it is unambiguous, and nowhere else

A flat character model must rediscover URL grammar from data, spending capacity on
constraints already known exactly. But enforcing grammar *all the way down* would
make the model incapable of representing malformed URLs — and **deliberate grammar
abuse is a real attack class** (double slashes, backslashes, encoded delimiters,
`@` tricks, malformed queries). Both extremes lose something real.

**The balance point: enforce structure at the top level, keep freedom inside each
component.**

| Component | Generated how | Rationale |
|---|---|---|
| **Public suffix** (`.com`, `.co.uk`) | sampled externally (§4.3) | Closed vocabulary; validity for free. |
| **Registrable label** (`shopwave`) | character-level | Where invention happens: coinages, mimicry, hyphenation. Must handle unseen strings. |
| **Subdomains** | character-level, variable count | Where spoofing structure lives. |
| **Path** | subword (BPE), delimiters free inside | Heavily templated; characters are wasteful. |
| **Query** | subword (BPE), delimiters free inside | Same. |
| **Fragment** | subword (BPE) | Same. |

**Path, query and fragment are separate components.** The `?` and `#` separators
are supplied by the grammar, so the model never generates them and can never
misplace them. Inside a component, generation is unconstrained — `/`, `&`, `=`,
`%`, `\` are all producible, so grammar abuse remains representable.

**We deliberately do not constrain the tokenizer.** A tokenizer forbidden from
merging across delimiters throws away exactly the compression BPE exists to
provide, on templated paths where it is worth the most. Merged tokens like `?id=`
usually *help* — a valid pattern becomes one atomic unit instead of three
independent chances to go wrong. Structure is learned at the sequence level, not
the token level; large language models emit valid JSON and code with tokenizers
that merge punctuation freely. The structural guarantee comes from §4.2's
factorization, not from crippling the tokenizer.

### 4.3 Public suffix — future-proof by construction

The Public Suffix List changes continuously. A fixed categorical output layer
would mean every PSL update changes the output shape and forces a retrain. So the
suffix is split in two, and neither half requires retraining:

**Sampling** does not involve the neural model at all. The suffix is drawn from an
**external frequency table**, updateable from the PSL and registration data
without touching a single weight.

**Representation** embeds the suffix **from its characters**, not from a lookup
table. A suffix the model never saw during training still yields a meaningful
embedding from its spelling, so the model generalizes — `.finance` lands near
`.credit` and `.loans` because they look alike and behave alike, without anyone
specifying that relationship.

Together: a new TLD works on day one. Add it to the table; its character-derived
embedding conditions generation sensibly. No retraining.

### 4.4 The model

**A single conditional autoregressive transformer.**

*Single*, not one model per class: a shared trunk means benign data improves
phishing generation and vice versa, and the class becomes a control rather than a
separate system.

*Autoregressive*, over the alternatives:

| Family | Verdict |
|---|---|
| **Autoregressive transformer** | **Chosen.** Strong samples on short strings, stable training, trivial conditioning, and an explicit likelihood — which the typicality capability needs. |
| GAN | Historical choice for domain generation (DeepDGA lineage), but discrete-text GANs train unstably and give no likelihood. |
| VAE / latent-variable | Offers a traversable space, but carries **posterior collapse**, which is *worse* for short strings: latents earn their keep when there is global structure the decoder cannot infer locally, and a 30-character URL has little. Published VAE-for-text results come from sentence- and document-length data. **Staged (§11.1), not default.** |
| Discrete diffusion | Immature for short structured strings. |

### 4.5 Control surface

Control is not a convenience feature. **Conditioning is what turns a density model
into a world model at generation time** — it is how you reach a rare stratum that
the raw distribution would seldom produce (§5).

Control tokens are restricted to attributes that are **free or structurally
derivable**, requiring no hand-written semantics:

- `class` — benign | phishing-style *(primary)*
- `suffix` — the public suffix (§4.3)
- `script` / `language` — detectable from characters
- `length`, `path_depth` — heavy-tail and structure control

**Brand conditioning is deliberately excluded from the core training design.**
Two reasons, the second decisive:

1. Training with a brand token requires brand *labels* on training URLs, and
   nothing in the data provides them. You would have to detect the targeted brand
   with a brand list and a matching heuristic — hand-written rules re-entering
   through the label pipeline. Whatever the rule cannot see, the model can never
   be conditioned on: **the rule becomes a ceiling.**
2. E3's coverage probe asks whether the model *spontaneously* discovers brand
   mimicry. **Training with brand tokens destroys that experiment** — supervised
   mimicry is not discovered mimicry.

Brand targeting remains available later as an **optional inference-time
capability**: autoregressive models support constrained generation, so a brand
string can be pinned into the label and the model generates around it — targeting
with zero rule-derived training labels.

### 4.6 Decoding is pluggable

Decoding is a **runtime choice, not an architectural one** — strategies swap
without retraining a single weight. The architecture therefore defines a decoder
interface; it does not name a winner.

**What *is* architectural is the reason decoding matters.** Two facts:

**Novelty is free.** Registrable labels of ~10 characters over `[a-z0-9-]` span
roughly 37¹⁰ ≈ 5 × 10¹⁵ strings. A corpus of ~2 × 10⁶ URLs covers an invisible
fraction. A well-fitted model sampling honestly produces unseen URLs essentially
every time.

**The real enemy is cliché, and it comes from the decoder, not the objective.**
Maximum-likelihood training spreads probability generously; standard practice then
samples at low temperature or top-k, which *sharpens toward the peak* — and the
peak of URL-space is `secure-login-verify`. The boring output is produced by the
sampler.

This is why the loss function is left alone. **You do not expand a distribution
with an objective — you expand it with inductive bias.** "Novel and plausible"
cannot be optimized directly, because plausibility is *defined* by the data; any
term pushing away from the data pushes toward less plausible. What genuinely
produces new ideas is §4.2's factored representation: a model learning suffix ×
label × path convention × language as separable concepts can generate combinations
that **never co-occurred in training**. That is invention without leaving the
plausible manifold.

Two objectives that sound right and are wrong: adversarial or reverse-KL training
is *mode-seeking* and makes samples less diverse; explicit novelty rewards mostly
teach the model to emit improbable garbage.

**Initial default:** typical sampling — it draws tokens whose information content
sits near the distribution's expected information content, avoiding the
high-probability cliché region and the low-probability garbage region at once,
where temperature can only trade one against the other. **This is a default, not a
commitment.** Temperature, top-p, typical and Mirostat are benchmarked against one
model in E3 (§9), which is nearly free — one training run, N evaluations.

### 4.7 Validity, novelty, and safety gate

Every sample passes a deterministic filter. **This is the only place rules belong
— as filters, never as generators.**

1. **Validity** — parseable, IDNA-encodable, suffix in the PSL, length limits.
2. **Novelty** — not in the training corpus and not a near-duplicate.
   **Memorization is the default failure mode of generative models on short
   strings**; it is measured, never assumed.
3. **Intra-batch deduplication** — mode collapse shows here first.
4. **Active-threat filter** — see below.

**Active-threat filter.** The novelty check only covers the *training* corpus,
which is historical. A threat registered last week is not in it. And collision is
not as unlikely as the 10¹⁵ space suggests: the model samples *from the phishing
distribution*, concentrating precisely where real phishing lives, and phishing
kits are highly templated. A model that learns the pattern will regenerate strings
that exist.

So generated **registrable domains** are checked against an **offline snapshot**
of current threat feeds (PhishTank / OpenPhish and equivalents). Set membership,
deterministic, no network at generation time.

**Live DNS or passive-DNS lookup is explicitly rejected.** It breaks the offline
constraint the project rests on, and it leaks: querying DNS for a generated domain
tells an attacker's authoritative nameserver that someone is probing it. §10
commits to never resolving generated domains, and an offline snapshot already
provides the protection.

**Collision rate is reported as a research metric, not only a safety check.** How
often the model independently generates a domain on a current blocklist is
*evidence it is generating in the right region of the space* — a validation signal
available no other way. Report it; filter it from releases.

Gate pass-rates are reported per run. A high rejection rate is a finding about the
model, not a nuisance to tune away.

---

## 5. The Internet World Model question

The goal is not merely a probability distribution over URL strings. It is a model
of **how different parts of the web name things** — CMS conventions, CDN patterns,
SaaS structures, banking and government portal styles, e-commerce layouts,
regional and language conventions — *without anyone defining those categories*.

**Is that already achieved by §4, or does it need explicit mechanism?**

Three claims must be separated, because they behave differently:

| Claim | Status |
|---|---|
| The model **represents** these concepts internally | **Very likely emergent.** Predicting URLs well *requires* distinguishing WordPress-shaped from REST-API-shaped from government-portal-shaped, because that is what makes the next character predictable. |
| We can **verify** it does | **Not free — needs a probe.** Cheap, and nobody has done it. *(Done 2026-10-17 for language: learned, but no more than the URL's surface text carries — `research/probe/RESULTS.md`.)* |
| We can **control** generation by them | **Not emergent.** Unconditional sampling yields the marginal distribution; nothing lets you *ask for* a stratum. |

**Where distribution-modelling and world-modelling genuinely diverge: frequency.**
Density estimation is dominated by what is common. A model can achieve excellent
likelihood while being useless on rare-but-systematic phenomena — if Japanese
e-commerce is 0.1% of the data, getting it entirely wrong costs almost nothing in
aggregate loss. A world model should know how Japanese e-commerce sites name
things. **Same architecture and same objective; different evaluation and different
data policy.**

This produces three concrete commitments:

**(a) Evaluation is stratified, not only aggregate** (§6). This generalizes the
per-class rule: per-class becomes **per-stratum**.

**(b) A real tension in the data policy, resolved deliberately.** A world model
benefits from upweighting rare strata so they are actually learned — but
rebalancing distorts the density, and the typicality scorer (§7) needs the density
*correct*. You cannot have both from one model at one weighting. **Resolution:
train on the natural distribution, evaluate stratified, and reach rare strata at
generation time through conditioning (§4.5).**

**(c) A representation probe, not a new mechanism** (§6). Test whether a linear
classifier can read platform, TLD category, language or region off the model's
internal states. If it can, the concepts are there — learned with no supervision.
That is a publishable finding in its own right: *an autoregressive URL model learns
web-platform structure without being told platforms exist.*

**Unsupervised structural clustering as a conditioning signal is deliberately
staged, not adopted.** It is attractive — cluster URLs by structure, condition on
cluster ID, let the model learn styles nobody named. But it is unproven and has
real failure modes: clustering on hand-designed features reintroduces rules, while
raw character features tend to cluster on superficial similarity (path length, not
"CMS"); the best features would come from the model's own representation, which
costs a second full training run; `k` is arbitrary and smuggles in an assumption
about how many styles the web has; and conditioning on a variable that turns out to
be noise fragments learning. **Measure first (the probe), then add mechanism only
if measurement justifies it.**

---

## 6. How we know it works

Self-contained by design: every metric is computable by a stranger, from public
corpora, with no reference to SNOW Guard.

| Metric | What it measures | Target |
|---|---|---|
| **Real-vs-Fake AUC** | Can a discriminator separate generated URLs from **held-out real** ones? | → 0.50 |
| **Human panel (comparative)** | Can a researcher sort a shuffled mix of generated and real URLs? | → chance. **Final evaluation, 2026-10-15:** ten people, 100 URLs, 55 correct (95% CI 45–64%); the largest feasible — no larger panel (owner, 2026-10-17) |
| **Novelty / memorization rate** | Fraction surviving §4.7 | reported beside every quality number |
| **Generative precision & recall** | Precision = realistic; **recall = covers the real variety** rather than collapsing | both high |
| **Compositional held-out test** | Withhold attribute *combinations*, then request them | plausible output |
| **Representation probe** | Can a linear classifier recover platform / language / region from internal states? (§5) | above chance. **Run 2026-10-17 (language only; no platform/region labels):** REPRESENTS LANGUAGE — 0.605 vs chance 0.125 and 0.474 untrained, but below a character n-gram reader (0.627) (`research/probe/RESULTS.md`) |
| **Temporal generalization** | Train to date *T*, evaluate on URLs first seen after *T* (§8) | measured, then tracked |
| **Blocklist collision rate** | How often it independently regenerates live threats (§4.7) | reported |
| **Held-out NLL** | Density quality | diagnostic and sanity floor |

**Stratified reporting is mandatory.** Every metric is reported **per class and
per stratum** — TLD category, language/script, path-depth band, region — never
pooled. A model excellent on English `.com` and useless on Japanese `.jp` looks
fine in aggregate. Stratified reporting is what makes §5's world-model claim
falsifiable rather than rhetorical, and what keeps benign genuinely first-class.

**Two methodological requirements:**

- **Novelty must be checked against a corpus far larger than the training set** (a
  Common Crawl URL index). Checked only against our own ~2M URLs, we would certify
  as "never existed" things that are common on the web.
- **Real-vs-Fake must use held-out real URLs.** Scored against training data it
  rewards memorization — the exact failure it exists to catch.

**The success bar is comparative, not absolute.** *"I haven't seen this URL but it
could exist"* is too weak: humans judge URLs poorly in isolation, and a plain
n-gram would pass it on most samples — so it fails to separate a creative world
model from a Markov chain, the exact distinction the project rests on. The valid
bar is the shuffled sorting test above. The informal phrasing is retained only as a
sanity check.

**NLL is a diagnostic, not the headline.** Likelihood and sample quality are
largely independent — a model can have excellent NLL and produce boring samples.
But NLL cannot be gamed by cherry-picking samples, so it stays as a floor.

---

## 7. Interfaces

**Primary — the generation API.**
`generate(class=…, n=…, suffix=…, script=…, decoder=…)`. An on-demand conditional
generator, not a static file. Corpus artifacts are a *serialized output* — 
versioned, hash-verified, carrying provenance (model version, decoder and its
parameters, seed) and gate results.

**Secondary — the typicality scorer.**
`log P(url | class)` for an arbitrary URL. Free, since it is the training
objective exposed. Useful to anyone, with or without a detector.

**Implemented (D2, 2026-10-17):** `uie.load(weights)` → `Engine.generate(label, n,
seed, blocked_domains)` — free mode, `e16_b5`'s decode (`uie/engine.py`), always
passed through the §4.7 gate — and `Engine.score(urls)`; plus the `uie` command line
(README, *Quick start*). Choosing the suffix, script or decoder is not exposed in this
first public interface; control mode stays in the research layer.

---

## 8. Freshness — versioned retraining, not continual learning

The web changes: new TLDs, new hosting platforms, new phishing kits. A model
frozen on 2026 data decays. But **continual or incremental learning is explicitly
rejected**, and not for convenience:

- **It destroys reproducibility.** A model that keeps updating cannot be
  benchmarked stably — every published number refers to a different model. That
  contradicts E0's frozen splits and published benchmark, which is the entire
  basis for this being a research project rather than a private notebook.
- **Catastrophic forgetting** is a research problem in its own right, and not one
  worth adopting as a side quest.

**Instead: versioned periodic retraining.** Each release pins a **corpus snapshot
date** and is immutable, exactly as SNOW Guard freezes model versions. Freshness
comes from releasing v2, not from mutating v1.

**And this turns a maintenance concern into a research contribution.** Training to
date *T* and evaluating on URLs first seen after *T* measures **how fast the model
goes stale** — and that number is a finding *about the internet*, not merely about
the model. It belongs in the metric set (§6) and, as far as we know, does not
exist in the literature.

---

## 9. Roadmap

Each phase has a gate; nothing starts before the previous gate is green. The
discipline is carried from SNOW Guard deliberately: pre-registered kill criteria,
frozen splits, honest negative results.

### E0 — Corpus and benchmark

**Benign — Common Crawl is the sole source of the learned distribution.**
Curated corpora are excluded from benign training entirely. This is measured, not
assumed: against the real web's **3.7%** bare homepages, the curated benign
corpora run **12–17%** (and two ran 98.6% / 100.0%, carrying zero path
information). Since CC supplies unlimited benign URLs at better quality, adding
curated data could only degrade the distribution.

*Sampling method matters and the naive approach is wrong.* CDXJ is sorted by SURT
— *reversed* domain — so the TLD is the **leading sort key** and each shard is a
contiguous TLD range. Reading shard prefixes yields a badly skewed sample
(measured: `ee`=16.9%, `mk`=12.7%). Correct method: seek to a random byte offset,
resync forward to the next independently-gzipped block, and validate the resync by
parsing. Blocks hold a near-constant line count, so uniform block sampling is
uniform line sampling — and it avoids the 104 MB `cluster.idx` download.

*Era-matching is required.* Phishing corpora date from ~2019–2024; the current CC
index is 2026. Pairing them directly teaches the model that **2026-era styling is
benign and 2021-era styling is phishing** — a temporal confound of exactly the kind
found in the curated corpora. CC publishes monthly crawls back to 2013, so benign
data is drawn from crawls contemporaneous with each phishing source.

**Phishing — live feeds are the authority; historical corpora supply volume.**
Live sources (URLhaus, OpenPhish) define what current phishing grammar looks like,
and a collector starts on **day one of E0** because volume accumulates only over
time — a live feed is a snapshot of what is active now, measured in thousands, and
a generative model cannot be trained on thousands of strings without memorizing.
Historical corpora (GramBeddings + Kaggle + Mendeley, union **546,086**, pairwise
overlap 1.2–5.7%) provide the volume needed to train at all. As the live corpus
fills, the historical share is progressively downweighted.

Two exclusions, both permanent:
- **PhiUSIIL** — excluded entirely, benign *and* phishing. Its benign side is
  100.0% bare homepages and its phishing side 72.8%; neither can teach path
  structure.
- **PhreshPhish** — quarantined. It is the sole source of SNOW Guard's
  `benchmark_v2`, so training on it would leak benchmark information *through the
  generator* into any consumer's training set, silently invalidating the
  integration gate of §3.

**Curated corpora keep a non-training role.** They are retained as offline test
fixtures (network-free, stable), as a parser/extractor regression corpus at
volume, as the reference that makes the distribution-shift finding above
reproducible by others, and as a **negative control** — a distribution the model
should demonstrably *not* match. They are **not** validation data: validating
against a distribution that is not the target tunes toward the wrong thing, so
model validation uses held-out Common Crawl. They live in a physically separate
directory, with a test asserting the training path never reads from it.

**Reproducibility: publish the recipe, not the data.** Common Crawl's terms do not
clearly permit redistributing derived datasets. Ship the extraction script, the
exact crawl IDs, and the filters — a stranger reproduces the corpus byte-for-byte.
This is stronger than shipping a CSV, and it converges with §10's release
principle. Live-feed snapshots are hashed and dated (§8).

**Strata come free from the index:** CC's `languages` field and the TLD, with no
extra pipeline.

**Measured baseline** (`CC-MAIN-2026-30`, 112,283 URLs, uniform random blocks):
bare 3.7% · query 17.8% · depth≥2 76.9% · mean depth 2.39. *Caveat carried
forward:* a block is ~3000 **contiguous** SURT lines, so the effective sample size
for TLD and language *shares* is the number of blocks drawn, not the number of
URLs. Path-depth statistics are far better behaved, and were stable across two
samples with entirely different TLD mixes.

Freeze splits, including a **temporal** split for §8. Define strata (§6). Define
the evaluation protocol. **Publish the benchmark before any model exists.**
*Gate:* a stranger can reproduce the benchmark from the repo alone.

### E1 — Deliberately dumb baselines
N-gram / Markov models over the factorized representation. Character n-grams are
**notoriously strong on short strings**; a neural model that cannot beat a 5-gram
has not earned its complexity.
*Gate:* baseline numbers published for every E0 metric, per stratum. These become
the bar.

### E2 — The generative model and generation API
Conditional autoregressive transformer over §4.2–4.5, pluggable decoder per §4.6.
No brand conditioning. Ships the generation API.
*Gate:* beats E1 on Real-vs-Fake AUC **and** held-out NLL, with novelty reported
alongside — a model that wins by memorizing has not won.

### E3 — Evaluation, probes, and the honest report
Full grading per §6, per stratum. **Decoder ablation** (temperature / top-p /
typical / Mirostat against one trained model). **Representation probe** (§5).
Comparison against rule-based generators as baselines. Coverage probe: does the
model spontaneously produce known attack classes, including brand mimicry it was
never conditioned on?
*Gate:* the report exists and states its failures. A negative result is a
publishable result and does not block release.

### E2.1 — Sampling the corpus by name, not by URL
E3 failed its pre-registered bar: Real-vs-Fake AUC **0.868 benign / 0.828
phishing** against a bar of **< 0.65**. E2.1 is the retraining iteration that
followed, and the bar is unchanged — the criterion is not renegotiated because
the model missed it.

Four hypotheses were measured and rejected before the cause was found: TLD skew
(neutralising the suffix moved AUC by −0.02), length calibration (length and AUC
move in *opposite* directions across the decoder sweep), decoder tuning
(exhausted at 0.804), and checkpoint selection (best 6.250 vs final 6.292).

Scoring URL components in isolation located it. The registrable label is a
stronger tell than the entire path — **AUC 0.836 for the label alone vs 0.731
for the path**, against 0.839 for the whole URL. Generated labels over-produce
open consonant-vowel syllables and end in a vowel 37.0% of the time against
29.4% for real ones: the model was inventing phonotactically plausible
pseudo-words where real labels are compounds of real morphemes.

The per-field train/val breakdown said why, and contradicted the intuitive fix:

| field | class | train | val | gap |
|---|---|---|---|---|
| **LBL** | benign | 1.201 | 2.346 | **+1.144** |
| SUB | benign | 0.478 | 1.185 | +0.707 |
| PTH | benign | 1.727 | 1.987 | +0.259 |
| SUF | benign | 1.248 | 1.319 | +0.071 |

The label field is the most **overfit** field in the model, by four times the
path's margin, and validation is 99.79% label-disjoint from training — so that
memorisation is worth nothing at evaluation time. The model learned *which*
labels exist rather than *how* labels are built, and had no structure to fall
back on when asked to invent one. Scaling capacity, the intuitive response to
bad samples, increases memorisation capacity and would widen exactly this gap.

The cause was the sampler. E2 drew 1.2M benign URLs uniformly from 5.28M, which
preserves the corpus's head-heavy shape: **the top 1,000 labels carried 38.4% of
the entire label-field gradient**, and the draw reached only 55,004 of the
79,473 training labels. Capping URLs per name fixes it inside the same budget —
32 per registrable label yields all 79,473 labels in 1.02M URLs, *fewer* than E2
uploaded, with the head's share falling to 3.1%. Capping by label string does
not remove morphemes, which recur across thousands of distinct labels; it stops
one string being seen 200,000 times.

The two classes cap on different fields because the invented name lives in
different places. Benign names the site in the registrable label. Phishing sits
on shared hosting, where the label is the platform and the attacker's name is
the subdomain — `vercel` carries 26,773 URLs across 17,494 *distinct* subdomain
prefixes, so a label cap would delete the diversity we want. Phishing caps by
full host instead.

*Gate:* the E3 suite is re-run in full and unchanged. Capacity is decided by the
per-field gap this run reports, not in advance: if the label gap closes while
train loss rises, the model has become capacity-bound and E2.2 scales it with
evidence attached.

#### E2.1 result — mechanism confirmed, criterion still missed

**The bar was missed. Real-vs-Fake AUC 0.882 benign / 0.886 phishing against
< 0.65.** The bar is not renegotiated.

The mechanism did what the diagnosis predicted. The benign label's train/val gap
fell from +1.145 to **+0.535** while *train* NLL rose 1.201 → 1.470 — the model
can no longer memorise labels — and val NLL improved 2.346 → **2.005**. Generated
label length moved from 9.61 (real 10.72) to 10.84 (real 10.62), and the samples
changed in kind, from `thunsangaluismundu` and `nausfargel` to `all4venue.gr`,
`checkhotelkonkolani.com` and `whatsapwcb.top` — compounds of real morphemes
rather than invented syllables.

Component-isolated discrimination, **paired** (one real reference set per class
reused across both models, three draws, ±1 s.d.):

| class | component | E2 | E2.1 | delta |
|---|---|---|---|---|
| benign | FULL | 0.9033 ±0.0023 | 0.8815 ±0.0037 | **−0.0218 ±0.0017** |
| benign | LABEL | 0.9033 ±0.0030 | 0.8095 ±0.0068 | **−0.0938 ±0.0058** |
| benign | PATH | 0.7931 ±0.0048 | 0.8039 ±0.0064 | +0.0108 ±0.0032 |
| phishing | FULL | 0.8561 ±0.0020 | 0.8855 ±0.0031 | **+0.0294 ±0.0027** |
| phishing | LABEL | 0.8358 ±0.0031 | 0.8630 ±0.0016 | +0.0272 ±0.0022 |
| phishing | PATH | 0.6934 ±0.0107 | 0.7212 ±0.0127 | +0.0278 ±0.0025 |

Benign improved by 13 standard deviations: the label tell fell 0.094 and the
path tell rose 0.011, a trade of roughly 9:1 in the intervention's favour.

**Pairing was not optional.** An unpaired comparison — E2 scored against E2's
validation draw, E2.1 against E2.1's — reported the *opposite sign* for benign.
At n=1200 the real-set draw is worth more than the effect, so only paired deltas
are reportable, and absolute AUC levels are not comparable across runs.

Phishing regressed on **every** component, and the cause was a dosage error
rather than the idea. Capping phishing by host cut it 303k → 259k rows, which at
unchanged steps and 50/50 class balancing raised it from ~8.4 to ~9.9 epochs; its
path train NLL fell 1.625 → 1.360 while val rose. The cap also bought nothing:
phishing's naming unit was **already flat** at 1.5 URLs/host, so there was no
skew to remove. The intervention was applied where the problem did not exist.

**The structural finding is worth more than the result.** After the fix, benign
reads LABEL 0.810, PATH 0.804, FULL 0.882 — no single component is the tell any
more. Driving the label to a perfect 0.500 would still leave the path separating
at 0.804. Reaching 0.65 requires every field to be near-indistinguishable
simultaneously; there is no one thing left to find.

### E2.2 — Capacity, on evidence

Three changes, each traceable to an E2.1 measurement:

1. **Keep the benign label cap.** It earned −0.094 on the label tell for +0.011
   on the path, and is the only reason the model composes morphemes at all.
2. **Revert the phishing cap.** It cost 15% of the scarce class and bought
   nothing measurable. Back to the full 303,176, which also restores E2's ~8.4
   phishing epochs.
3. **Scale capacity.** This is the trigger the E2.1 gate named in advance: train
   NLL rose on every benign field after the cap (LBL +0.269, QRY +0.316, PTH
   +0.024) while val improved. That is underfitting, and it only became visible
   once memorisation was removed — which is why scaling was the wrong move
   *before* E2.1 and is the right one after it.

*Gate:* E3 re-run unchanged, and the E2 vs E2.2 comparison must be **paired** —
E2.1 showed an unpaired comparison can invert the sign of the result.

**Known defect carried into E2.2, deliberately.** The benign cap distorts path
depth:

| benign train | n | mean depth | bare | ≥3 levels |
|---|---|---|---|---|
| uncapped | 5,276,149 | 2.31 | 2.94% | 33.7% |
| capped (E2.1) | 1,017,551 | **1.85** | **7.64%** | 21.1% |
| val (uncapped) | 60,000 | 2.33 | 2.76% | 33.9% |

E0 made Common Crawl the sole benign source precisely because curated corpora
carry 12–17% bare homepages against the real web's ~3.7%; the cap re-introduces
that artifact at 7.64%, smaller in size but identical in kind. The mechanism is
arithmetic: 23,531 labels hold exactly one URL and a one-URL site is almost
always a homepage, so capping lifts them from 0.45% of the corpus to 2.3% while
demoting large deep-pathed sites.

It is carried rather than fixed because E2.2 already changes two things, and a
third sampling intervention would confound the capacity measurement this run
exists to make. The artifact costs +0.011 AUC; capacity has ~20x that headroom.

#### E2.2 result — capacity buys structure on compositional fields and lookup on discrete ones

**The bar was missed again, and the two changes cancelled.** Paired against
E2.1 on a byte-identical validation set (the RNG decoupling working as
designed), with benign training rows byte-identical too, so capacity was the
only benign variable:

| class | component | E2.1 | E2.2 | delta |
|---|---|---|---|---|
| benign | LABEL | 0.8095 ±0.0068 | 0.8258 ±0.0038 | **+0.0163 ±0.0034** |
| benign | PATH | 0.8039 ±0.0064 | 0.7954 ±0.0093 | **−0.0085 ±0.0036** |
| benign | FULL | 0.8815 ±0.0037 | 0.8796 ±0.0051 | −0.0019 ±0.0026 |
| phishing | FULL | 0.8855 ±0.0031 | 0.8441 ±0.0027 | **−0.0414 ±0.0058** |

The per-field NLL predicted this exactly, and three independent instruments
agree:

| benign field | train | val | gap | vs E2.1 gap |
|---|---|---|---|---|
| LBL | 1.117 | 2.095 | +0.978 | **+0.443** (train −0.353, val **+0.090**) |
| PTH | 1.669 | 1.885 | +0.216 | +0.017 (train −0.082, val −0.065) |
| QRY | 1.741 | 1.974 | +0.233 | +0.022 (train −0.151, val −0.129) |

Tripling capacity **re-memorised the label field** — train NLL fell 0.353 while
val got *worse* — and **genuinely improved path and query**, whose val NLL fell.
The label field is ~79,000 arbitrary discrete strings, where capacity buys a
lookup table; paths and queries are compositional, where capacity buys
structure. One model with one capacity setting cannot serve both.

**Cap and capacity are coupled, not independent.** Cap 32 was tuned against
4.84M parameters; 14.35M has room to memorise 79,474 labels seen 32 times each.

Reverting the phishing cap worked as diagnosed: its path gap closed 0.925 →
0.696 with val NLL down 0.128, and every phishing component improved.

#### Where this leaves the pre-registered bar

Benign FULL across three iterations: **0.9033 → 0.8815 → 0.8796.** Two
well-executed, correctly-diagnosed interventions bought 0.024 total, with
increments shrinking. The bar is 0.65 — **0.23 away** — and no remaining
sampling or capacity move is worth more than ~0.01.

The binding constraint is the corpus, not the model. **It holds 99,303 benign
registrable labels.** That is the entire vocabulary from which the model must
learn how the Internet names things, and against the space of nameable domains
it is a tiny sample: a small model over-smooths it into phonotactic mush, a
large one memorises it, and neither generalises. No capacity or sampling change
manufactures label diversity the corpus does not contain.

### E2.3 — Depth-preserving cap
Restore the population path-depth distribution without giving up label
flattening: pass one accumulates per-label depth histograms and the depth
reweighting the uniform cap induces, pass two samples each label's quota with
weights that return the aggregate histogram to the uncapped population's.
*Gate:* bare rate within 1pp of 2.94% and mean depth within 0.1 of 2.31, with
the benign label component AUC no worse than E2.1's 0.8095 — the label win must
survive the depth correction, paired.


### E3 - Rebuilt corpus, two-corpus architecture, replaced evaluation

E2.2 established that the corpus, not the model, was the binding constraint:
99,303 benign labels is too few to learn how the Internet names things, so a
small model over-smoothed them into pseudo-words and a large one memorised them.
E3 rebuilds the corpus and changes what the model is asked to learn.

#### The corpus

| | E2.2 | E3 |
|---|---|---|
| benign labels | 99,303 | **1,132,768** |
| distinct suffixes | 251 | **1,272** |
| suffix entropy | 4.358 | **6.647** |
| top-10 suffix share | 69.9% | **36.8%** |
| phishing names | 94,234 | **240,815** |
| crawls | 16 (2019-2026) | 8, era-matched to the feed |

Two sampling changes did this. Blocks are drawn **suffix-stratified** at
alpha=0.5 rather than uniformly: the CDXJ index is SURT-sorted, so a shard IS a
suffix range, and uniform draws hand `.com` ~42% of the budget before a URL is
read. And phishing is grouped by **host**, not registrable domain - 73% of
phishing sits on shared hosting where the attacker's invention is the subdomain
and the registrable domain belongs to the platform.

The 8 historical crawls were dropped. They existed to era-match archived
phishing corpora that were never collected, leaving 47.1% of benign in a window
containing no phishing at all - and with it a clean "2019-era web implies
benign" signal (historical crawls are 76.6% https, live 92.5%, phishing 64.4%).

#### Two corpora, and field-scoped loss

Names and paths want opposite sampling. Names want every domain weighted
equally; paths want many pages from one site. E2.1 and E2.2 each measured that
trade - capping for names broke path depth (bare 2.90% to 7.64%), and capacity
helped paths while re-memorising labels. No single URL list satisfies both.

    name       1 url per name, every name          teaches [SUF] [LBL] [SUB]
    structure  <=40 urls, multi-page sites only    teaches [PTH] [QRY]

`field_mask` enforces it. Markers and the end token are trained on **all** rows:
URL SHAPE is not a content question, and scoping it would teach "a path follows
a label" only from multi-page sites, driving the generated bare rate to zero
against a real 2.9% benign / 47.4% phishing.

**Known limitation, carried deliberately.** The model learns P(name-fields) from
every domain and P(path-fields | name) from multi-page sites only, then composes
them at generation. That composition matches no real corpus, so the model can
pair a small-site name with a large-site path. It is a real shortcoming - smaller
than the marginal gains it buys, and smaller than the DISTORTED correlation
E2.1's cap produced, but real.

#### Class sampling: alpha = 1

`p(class)` proportional to `n_labels ** alpha`, so exposures per label go as
`(N_p/N_b)**(alpha-1)`. **alpha = 1 is the unique exposure-neutral point** -
every name in both classes is seen equally often (~1.65 times across the run),
so memorisation pressure is equal and step count alone sets its level. The unit
is LABELS, not rows: rows-per-name differ by class (benign 1.00, phishing 1.47),
so weighting by rows still over-exposes phishing.

#### Two fields removed

**Scheme.** http/https tracks when a URL was observed, not how it was named. A
model given it learns `http implies phishing` from the web's TLS migration.

**Language.** `und` was 100% of phishing rows against 1.6% of benign, so any
non-`und` token guaranteed the class. Inference was tested as the symmetric
alternative and rejected on measurement: language is 77.5% recoverable from the
TLD alone and the entire rest of the URL adds 3.3 points, so an inferred token
would launder the suffix through a second channel at 19% label noise. Language
conditioning moves to the supplied suffix.

#### Evaluation replaced

See BENCHMARK.md sections 8-9. The whole-URL AUC gate is retired: its bar sat
below its own floor, and it measured a joint the model is not trained to
produce. Fields are now scored against the population each was trained on, with
a floor measured from real-vs-real, and the reported number is EXCESS OVER FLOOR.

*Gate:* every field's excess below 0.05, novelty reported alongside, whole-URL
AUC reported as auxiliary. Capacity, decoder and any further corpus change are
decided by the per-field table, not by a single scalar.

**Status: NOT PASSED.** Canonical evaluation of the resting configuration
(`best_e5.pt`, with and without the adopted delta) is in **E3_REPORT.md section
30** and summarised in the README. It was re-measured at `n_main` 8000, G=5 x
D=3, after the n=1200 evaluation was found to rest on a single draw of an
instrument whose own split-seed spread was nearly twice the bar.

Five field-class cells are convicted, consistently across both decode
configurations: **benign PTH, benign QRY, phishing SUB, phishing PTH, phishing
QRY.** Clearing: benign LBL/SUF/SUB, phishing LBL/SUF. These are five failed
cells, **not** five independent causal mechanisms; attribution across them is
open. Verdicts use the section-26.1 calibrated estimator, whose SE has four
degrees of freedom at G=5, so the threshold is t > 2.78 rather than 2 sigma.

**No cell is UNMEASURED.** QRY, blank in every earlier report for want of
samples, is now measured by a pooled estimator and is the model's **largest
defect** (+0.1196 to +0.1488 across four configurations). It fails on two
independent axes: emission rate (2.49% benign / 1.30% phishing generated, against
6.36% / 4.29% in training) and conditional content. Train and validation query
rates agree to within 0.15 points, so the deficit is not a corpus mismatch. The
two axes are **not** assumed to share a cause; characterising that is the current
work, and no generator intervention is yet justified.

The PTH residual tell is characterised and FROZEN in
[PTH_ATTRIBUTION.md](PTH_ATTRIBUTION.md): F1 lexical identity refuted, F2
repetition untestable on this instrument, F3 not necessary, F4 sufficient in
benign but not necessary, F5 contributory but not isolated, F6 minor. Its
dominant causal feature is unresolved and the branch is closed — the next
generator change must not target those properties on the strength of that
study.

### Standing conclusions at the E3 checkpoint

- **PTH attribution work is CLOSED** (PTH_ATTRIBUTION.md, frozen). Re-measuring at
  full capacity raised PTH's excess rather than lowering it, but that does not
  reopen attribution.
- **The delta intervention branch is CLOSED.** Delta is neutral on every gated
  field (paired |t| < 1.6). It is retained for the structural reason it was
  adopted, not for gate benefit, and the earlier "delta worsens PTH" claim is
  retracted in E3_REPORT section 30.7.
- **QRY is the highest-priority characterisation target.**
- **No generator intervention is yet justified.**
- **No threshold change is justified.** The 0.05 bar is untouched.
- **No claim of human indistinguishability is justified.** The human panel has
  never been run *(since run: the final ten-person evaluation, 2026-10-15, 55/100 —
  no evidence people can tell, not proof they can't)*; whole-URL AUC is 0.726 benign / 0.899 phishing.

### Naming: `E<n>` are MODEL EXPERIMENTS, `D<n>` are DELIVERABLES

These were previously both called `E4`/`E5`, which made "we are at E5" mean two
unrelated things: the frame-token model, and the open-source release. The
experiment numbering is load-bearing — it names four checkpoints on disk
(`best_e4.pt`, `best_e5.pt`, `best_e6.pt`) and 64 KB of `E3_REPORT.md` — so the
DELIVERABLES were renamed instead.

    E4, E5, E6   model experiments INSIDE roadmap-E3 (see E3_REPORT.md)
    D1, D2       deliverables that follow roadmap-E3's gate

### D1 - The typicality scorer — ✅ DONE (2026-10-17)
Ship §7's secondary interface, proven against a real consumer integration.
*Gate:* an external consumer can score URLs without importing anything internal.
**Met.** `uie.typicality` (aaa6dca: `TypicalityScorer`, `log_prob`, `llr`, `score`) is the
public interface; `research/d1/SCORER_CARD.md` documents it. A real consumer, LIMNEX, scored
360,204 URLs through a standalone bundle that imports only `uie.typicality`, and built it into
a saved serving bundle as a fifth member (LIMNEX v3.8, not served). What the consumer concluded is its own record (LIMNEX
`research/uie_member/`, `research/fusion_opt/`, `research/v38_readiness/`) and grades nothing
here.
~~Blocked until roadmap-E3's gate passes~~ — **Unblocked by the owner, 2026-10-15:**
roadmap-E3 is finished (closed, NOT MET), and everything that depended on it is open.
D1 serves `e16_b5`, whose known limits are recorded in the README
(`research/native/DECISION_2026-10-15.md`).

### D2 — Open-source release
Documentation, model card, responsible-use statement, reproducible benchmark,
packaging, license.
*Gate:* someone who has never heard of LIMNEX can install it, reproduce the
benchmark, and generate samples.
**Known blocker:** 115 absolute local paths across `research/` must be
parameterised first. `uie/` is clean.

---

## 10. Risks

**Benign corpus breadth — the highest-risk input in the project.** Phishing feeds
are narrow; benign URLs are the long tail of the entire web. Train on a narrow
benign sample and the model produces a cramped view of "normal." This is E0's
central problem, not a detail.

**Memorization masquerading as realism.** The most realistic-looking URLs are the
ones it copied. Novelty is therefore reported *beside* every quality number.

**Mode collapse onto corpus clichés.** The model rediscovers `secure-login-*` and
stops. Caught by intra-batch duplication and generative recall.

**Synthetic data may not help downstream.** The question of whether generated URLs
improve a detector is genuinely open, and synthetic augmentation disappoints as
often as it delivers. This is why SNOW Guard's integration is gated with a
pre-registered kill criterion, and why a negative result is publishable here.

**Premature generalization.** The classic failure of spinning out a library:
building for imagined users and helping nobody, including the consumer you had.
Mitigation — **independent, but never consumer-less**: E4's gate requires a real
integration. Independence means *not graded by* the consumer, not *ignoring* it.

**Dual use.** This produces realistic phishing-style URLs. The purpose is
defensive and the precedent for publishing is solid (DeepDGA and successors are
public, as is most adversarial-ML tooling). The line to hold:

- **Ship the tool, not the ammunition.** Code, model and benchmark are
  publishable. A large pre-generated corpus of realistic phishing URLs is directly
  usable and is not a release artifact.
- **Generated domains are never registered and never resolved** (§4.7).
- Anything published passes the active-threat filter first.
- The responsible-use statement and model card are E5 gates, not afterthoughts.

---

## 11. Staged and open

### 11.1 Latent traversal — staged
Control tokens let you request categories defined in advance; a latent space would
let you *explore structure you did not specify* — interpolate between two URLs,
find axes of variation the model organized on its own. That is "emergence from a
learned representation" literally rather than figuratively.

**Decision: staged, not abandoned.** Build the autoregressive model first. The
work is shared — the expensive part is the decoder, and a latent-variable version
reuses it — so this is the first half of that model either way, not a detour.
Revisit only if §5's probe shows structure that conditioning cannot reach.

### 11.2 Unsupervised structural clustering — staged
See §5. Measure first; add mechanism only if the probe justifies it.

### 11.3 Brand conditioning — deferred to inference
See §4.5. Never in core training.

### 11.4 Still open
- Benign corpus sources and breadth — E0's central problem
- ~~Repository location, license~~ — decided 2026-10-17: a fresh public repository with one
  starting commit; Apache-2.0 (`research/d2/D2_PLAN.md`)
- ~~Project name~~ — decided 2026-10-17: Limnex URL Intelligence Engine (Limnex UIE)
