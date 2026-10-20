# Corpus Bias & Quality Audit

**Date:** 2026-08-04 · **Stage:** before the 1M-label production extraction
**Verdict:** **DO NOT LAUNCH** until §3's three required corrections are made.

Every check below asks one question: *could the model learn this as a class
marker or a fact about the web, when it is really an artifact of how the data
was collected?* An artifact learned as a fact teaches the generator something
false, and no amount of downstream tuning removes it.

Populations audited:

| population | n | how collected |
|---|---|---|
| benign E0 | 6,496,447 | uniform random blocks, 16 era-matched crawls |
| phishing | 354,143 | OpenPhish live feed + archived corpora |
| benign pilot | 370,337 | **proposed** stratified sampler, α=0.5, 200 blocks, one crawl |

---

## 1. Measured profile

| metric | benign E0 | phishing | benign pilot |
|---|---|---|---|
| unique domains | 99,470 | 94,234 | 5,881 |
| URLs per domain | 65.31 | 3.76 | 62.97 |
| suffix entropy (bits) | 4.359 | 4.537 | **6.006** |
| distinct suffixes | 251 | 719 | 87 |
| top-10 suffix share | 0.699 | 0.746 | **0.341** |
| https | 0.850 | 0.644 | **0.932** |
| www | 0.402 | 0.096 | 0.398 |
| has query | 0.173 | 0.063 | 0.130 |
| mean path depth | 2.300 | **0.942** | 2.240 |
| bare homepages | 0.029 | **0.474** | 0.022 |
| mean URL length | 80.7 | 49.7 | 74.2 |
| mean subdomains | 0.607 | 0.885 | 0.549 |
| language entropy | 3.541 | **0.000** | 4.190 |
| `und` language share | 0.016 | **1.000** | 0.021 |
| exact duplicate rate | 0.0081 | 0.0171 | 0.0077 |
| distinct path shapes | 2,662 | 604 | 378 |
| top-10 path-shape share | 0.747 | 0.911 | 0.758 |

---

## 2. Checks that PASSED

**Cross-class domain contamination — PASS.** 35 of 94,234 phishing domains also
appear in benign (**0.04%**), and all inspected cases are shared-hosting
platforms legitimately carrying both (`appspot.com`, `blogspot.com`, `fc2.com`).
No cleanup needed; a domain hosting both is a fact about the web.

**Train/validation contamination — PASS, safe by construction.** `SplitAssignment`
stores a global `domain -> partition` map and `partition_of()` resolves by
registrable domain, so disjointness is a property of the data structure rather
than of the code that uses it. Verified separately: benign val is 99.79%
label-disjoint from train.
*Constraint to preserve:* the name corpus and structure corpus must be
partitioned through the **same loaded `SplitAssignment`**. Splitting them
independently would let a domain land in name-train and structure-val. The
existing `pack_kaggle` pattern (load once, apply to every row) already does this
and must not be changed.

**Duplicate and near-duplicate rate — PASS.** Exact duplicates are 0.81% benign,
1.71% phishing, 0.77% pilot. Structural duplication is reported as distinct path
*shapes* (digits and alphabetic segments folded): phishing concentrates far more
(top-10 shapes = 91.1% vs benign 74.7%), which is a genuine property of templated
kits rather than a sampling defect.

**Sampling yield under stratification — PASS.** 29.4 labels/block against the
uniform sampler's 31.0, and 63.0 URLs/domain against 65.3. Stratification changes
*which* domains are collected, not how many or how deeply — which is what it was
designed to do.

**Popularity/crawl bias — ACKNOWLEDGED, not fixable.** Common Crawl samples what
is linked and crawlable, so unlinked and newly-registered domains are
under-represented. This is inherent to the source and already stated in
BENCHMARK.md §7. The per-domain quota removes the *page-count* skew; it cannot
remove the *inclusion* skew.

**Domain age — NOT MEASURABLE.** Requires WHOIS, which is a live network lookup
and is rejected by DESIGN.md §4.7. Recorded as an unmeasured risk rather than a
passed check.

---

## 3. Required corrections (all three block the extraction)

### 3.1 The language control token is a near-perfect class marker — CRITICAL

| | `und` share |
|---|---|
| benign | 0.0156 |
| phishing | **1.0000** |

The phishing feed carries no language label, so every phishing row is `und`
(language entropy **0.000**). Benign rows carry a real language 98.4% of the time.

**Any non-`und` language token guarantees the class is benign.** The two control
signals `<C:*>` and `<L:*>` are therefore entangled: the model can read class off
the language token, `<C:phishing>` + `<L:eng>` is entirely out of distribution and
would produce unreliable output, and the language conditioning that §4.5 promises
does not exist for half the model.

This is not a discriminative leak — a conditional generator is *given* the class —
but it is worse in a subtler way: it makes one advertised control surface a
function of another, so the API lies about what it can do.

**Correction: language dropout.** During training, replace the language token
with `und` for a random fraction (start at p = 0.5) of *benign* rows. `und` then
appears in both classes at comparable rates and stops identifying the class,
while non-`und` still conditions when present. This is standard conditioning
dropout, not a new invention. Expected effect: P(phishing | `und`) falls from
0.777 to ≈0.10.

### 3.2 Stratification makes the suffix a STRONGER class marker — CRITICAL, self-inflicted

Total variation distance between the two classes' suffix distributions:

| | TV distance |
|---|---|
| benign E0 vs phishing | 0.579 |
| **benign pilot vs phishing** | **0.771** |

Flattening benign's suffixes moves it *further* from phishing, which concentrates
on cheap new gTLDs (`io`, `app`, `dev`, `top`, `cc`, `xyz`, `vip`) while
stratified benign spreads across ccTLDs. Jaccard overlap of the two top-15 lists
is **0.154**.

Two consequences. The model gains a shortcut — it can infer class from suffix
instead of learning naming differences. And joint coverage degrades: benign
`.top` and phishing `.cz` are both nearly untrained, so those combinations
generate unreliably.

**This is acceptable only because §4.5 supplies the suffix at generation** rather
than sampling it from the model. The training marginal is overridden by an
external table, so a distorted marginal costs nothing — *provided the override
actually happens.*

**Correction: it currently does not.** `uie/corpus/run_e3.py` calls `generate()`
without a `suffix` argument, so E3 samples the model's own distorted suffix
marginal and scores it against real data. That converts a deliberate design
choice into a measured tell. Before stratified data is used:

1. E3 must supply suffixes from a natural frequency table, and
2. that table must be **class-conditional** — phishing's real suffix
   distribution is not benign's, and one shared table would be a new artifact.

Without both, stratification must not ship.

### 3.3 The pilot is not era-representative — CRITICAL for the production run

| | https | http |
|---|---|---|
| benign E0 (16 crawls) | 0.850 | 0.150 |
| benign pilot (CC-MAIN-2026-30 only) | **0.932** | 0.068 |
| phishing | 0.644 | 0.356 |

The pilot drew from a single recent crawl, so its https share is 8pp above the
era-matched corpus. Year tokens in paths show the same: the pilot peaks at 2026
(19.1%) and 2025 (13.7%), while E0 peaks at 2019 (9.9%) and 2018 (8.6%).

Left uncorrected this would pair *recent* benign against *era-matched* phishing,
teaching the model that http implies phishing — the exact artifact SNOW Guard
found and fixed in V3.5 by stripping the scheme, here reintroduced at the corpus
level.

**Correction:** build shard maps for **all 16 era-matched crawls** (~285s each,
≈1.3h total) and allocate the block budget across them exactly as E0 did. The
pilot validated the *mechanism*, not the era mix.

---

## 4. Findings that require a decision, not a fix

### 4.1 The phishing feed's bare-homepage artifact

| | bare homepages | mean depth |
|---|---|---|
| benign | 2.9% | 2.30 |
| phishing | **47.4%** | **0.94** |

Nearly half of all phishing URLs are bare homepages. This is what a **feed**
reports — the landing page of a campaign — not a property of phishing itself.
E0 rejected curated *benign* corpora for exactly this defect (PhiUSIIL 100%,
LegitPhish 98.6% bare) and then inherited it on the phishing side.

The model will learn "phishing ⇒ shallow, bare, short" and it will be right about
this dataset and wrong about the web. It cannot be fixed by sampling — it is what
the source contains. It must be stated prominently in the model card, and any
claim that generated phishing URLs are realistic must be read against it.

### 4.2 Expanding benign to 1M leaves an 11:1 class asymmetry

Benign goes to 1,000,000 labels; phishing stays at 94,234. With 50/50 class
balancing in the sampler, each phishing label would be seen ~11x more often than
each benign one. E2.2 already measured phishing's label field as the most
memorised in the model (+1.227 train/val gap).

There is no collection fix — historical phishing feeds cannot be manufactured.
Options, in order of my preference:

1. **Drop 50/50 balancing to something proportional-ish** (e.g. α-tempered class
   weights), so phishing epochs stay near E2's ~8.4 rather than rising to ~90.
2. **Train the phishing conditional for fewer steps** with a separate schedule.
3. **Accept and measure** — report phishing's memorisation gap every run and
   treat the phishing generator as the weaker deliverable.

I recommend (1), and it needs a decision before the training config is written,
not after.

---

## 5. Structural class differences: real, artifact, or both

Recorded because they are large enough to dominate any discriminator, and because
"is this real?" has a different answer for each.

| property | benign | phishing | reading |
|---|---|---|---|
| mean depth | 2.30 | 0.94 | **mostly artifact** (feed reports landing pages) |
| bare rate | 2.9% | 47.4% | **artifact**, see §4.1 |
| `www.` prefix | 40.2% | 9.6% | **mixed** — phishing genuinely skips `www`, but feed capture contributes |
| distinct suffixes | 251 | 719 | **real** — phishing exploits cheap gTLDs |
| has query | 17.3% | 6.3% | **mixed** |
| top-10 path shapes | 74.7% | 91.1% | **real** — kits are templated |
| mean subdomains | 0.61 | 0.89 | **real** — shared hosting puts the name in the subdomain |
| mean length | 80.7 | 49.7 | follows from depth |

E0 already documented that path depth alone separates the classes at AUC 0.791
and that its sign flips between corpora. Nothing here contradicts that; it
quantifies it per-field.

---

## 6. Verdict

**The pipeline is NOT safe to run at production scale today.** Three corrections
are required first:

1. Language dropout on benign rows (§3.1)
2. Class-conditional suffix supply in E3, before stratified data is used (§3.2)
3. Shard maps and block allocation across all 16 era-matched crawls (§3.3)

One decision is required (§4.2, class balancing under 11:1 asymmetry), and one
limitation must be documented rather than fixed (§4.1, the phishing feed's
bare-homepage artifact).

The stratified sampler itself is sound: yield, depth fidelity and reproducibility
all check out, and the suffix flattening it was built for works as designed
(entropy 4.359 → 6.006, top-10 share 0.699 → 0.341). Its two problems are that it
was validated on one crawl instead of sixteen, and that it depends on a
generation-time suffix override that is not yet implemented.
