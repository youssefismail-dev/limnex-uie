# E3 — Engineering Report

**Status:** **roadmap-E3's acceptance gate is NOT PASSED.** Baseline = E5 +
path-scoped delta (best_e5.pt, delta 0.22 benign / 0.30 phishing, decode-time).
E6 rejected. Evaluation harness rebuilt and calibrated; two earlier rejections
overturned as artifacts.

> **READ SECTION 30 FIRST for the current state.** The canonical evaluation was
> re-measured at `n_main` 8000, G=5 x D=3, on 2026-09-22. It supersedes every
> earlier acceptance verdict in this document, including section 29's. Five
> field-class cells are convicted (benign PTH, benign QRY, phishing SUB,
> phishing PTH, phishing QRY) and **QRY - blank in every earlier section - is
> the model's largest defect.** Field numbers quoted in sections 18, 19 and 29
> are single-draw measurements at n<=1200 and are preserved as the dated record,
> **not** as current status.

**Date:** 2026-08-06 for sections 1-29; **2026-09-22 for section 30**
(sections 1-10 = E3; 11-15 = E4 and after)
Sections 1-10 are preserved as written, including two claims that E4 later
falsified - see 11.1.
**Checkpoint under report:** `data/best.pt` (14,370,816 parameters); section 30
scores `data/best_e5.pt`
**Corpus:** `data/bundle2`
**Test suite:** 450 passing

This document is the permanent record of where E3 ended. It is written to be
read on its own: what was built, what it achieved, what was wrong with it, how
the defect was found, why the fix requires retraining, and what happens next.

---

## 1. Executive summary

E3 achieved the project's central generative milestone and simultaneously
exposed an architectural defect in the loss masking that had been present since
field-scoped training was introduced.

**The milestone.** Benign registrable labels — 16.77 bits of entropy, 112,443
distinct values in 113,277 held-out rows — are now **statistically
indistinguishable from real ones**. A discriminator scores generated labels at
AUC 0.5232 against a real-vs-real floor of 0.5046, an excess of **+0.0186**,
which is 1.1σ and inside noise. This is the hard field, and it is done.

**The defect.** The model appeared to under-generate URL paths by 33 percentage
points on the phishing side. Investigation showed it was emitting the `<PTH>`
marker on 100% of rows — a perfect match to training — and then closing it
**empty**, producing `host/`. Root cause: `field_mask` weighted the two branches
of a single decision differently, so the name corpus taught "empty path" at full
weight while its counter-evidence was masked to zero.

**The fix.** A structure field now opens only when its body is non-empty. This
makes "does this URL have a path" a pure marker decision, trained identically on
all rows. It changes tokenization, so the model must be retrained.

---

## 2. What E3 was

### 2.1 Model

| | |
|---|---|
| Parameters | 14,370,816 |
| `d_model` / `n_layers` / `n_heads` / `d_ff` | 384 / 8 / 8 / 1536 |
| `max_len` / `dropout` / vocab | 256 / 0.1 / 99 |
| Steps / batch / lr | 20,000 / 256 / 3e-4 |
| `weight_decay` / `warmup` / `grad_clip` | 0.01 / 500 / 1.0 |
| `class_alpha` | 1.0 (exposure-neutral) |
| Seed | 20260803 |

Vocabulary of 99 = 4 specials + 9 field markers + 2 class tokens + 84
characters. **No scheme token and no language token** — both were removed in
E2.3 on the grounds that the scheme encodes *when* a URL was observed rather
than how it was named, and that 96% of the language token's information was
already carried by the TLD.

### 2.2 Corpus (`data/bundle2`)

| Corpus | Train | Val | Distinct val hosts |
|---|---:|---:|---:|
| `name` (benign) | 906,214 | 113,277 | 113,277 |
| `structure` (benign) | 1,648,460 | 209,808 | 7,061 |
| `phishing_name` | 192,652 | 24,081 | 24,081 |
| `phishing_structure` | 46,491 | 5,828 | **941** |

Two splits with **different key kinds**, which the run log confirms were both
loaded: `split_random.json.gz` is domain-keyed, `split_phishing_host.json.gz` is
host-keyed. The distinction exists because grouping phishing by domain rather
than host had previously dropped 78% of rows while making the survivors *look*
better balanced.

Suffix table: 1,232 benign suffixes, 683 phishing, built from all four training
corpora so its marginal matches the rows the model saw. Novelty Bloom filter:
14,873,392 training URLs at 0.0115% false-positive rate. Encoding dropped 9,933
rows (0.36%).

---

## 3. Evaluation results

Every field is scored against the population it was **trained** on, and every
AUC is reported beside a floor measured from real-vs-real on that same
population. The number that means anything is **excess over floor**. Error bars
are Hanley-McNeil, computed per row from the actual sample size.

### 3.1 Benign

| Field | Pop | AUC | Floor | Excess | ± | n | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| LBL | name | 0.5232 | 0.5046 | **+0.0186** | 0.0167 | 1200 | ok (1.1σ) |
| SUF | name | 0.5143 | 0.5111 | +0.0032 | 0.0167 | 1200 | ok (0.2σ) |
| SUB | name | 0.5438 | 0.5176 | +0.0262 | 0.0252 | 526 | ok (1.0σ) |
| PTH | structure | 0.6185 | 0.4870 | **+0.1315** | 0.0194 | 884 | **TELL (6.8σ)** |
| QRY | structure | — | — | — | — | 66 | insufficient |
| FULL | mixed | 0.5955 | 0.4704 | +0.1250 | 0.0167 | 1200 | auxiliary (7.5σ) |

### 3.2 Phishing

| Field | Pop | AUC | Floor | Excess | ± | n | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| LBL | name | 0.5292 | 0.5298 | **−0.0006** | 0.0167 | 1200 | ok (0.0σ) |
| SUF | name | 0.5214 | 0.5319 | −0.0106 | 0.0167 | 1200 | ok (−0.6σ) |
| SUB | name | 0.5435 | 0.4768 | **+0.0667** | 0.0186 | 969 | marginal (3.6σ vs 0; 0.9σ over bar) |
| PTH | structure | — | — | — | — | 205 | insufficient |
| QRY | structure | — | — | — | — | 21 | insufficient |
| FULL | mixed | 0.6263 | 0.4992 | +0.1271 | 0.0167 | 1200 | auxiliary (7.6σ) |

### 3.3 Gate, novelty, diversity

| | Benign | Phishing |
|---|---|---|
| Gate pass | 100.0% | 98.2% |
| Novel vs 14.87M training URLs | 100.0% | 98.4% |
| Rejections | none | 19 memorized, 3 label-too-long |
| Precision / recall (k=5) | 0.932 / 0.855 | 0.936 / 0.846 |
| Distinct rate | 1.000 | 1.000 |
| Distinct 4-gram ratio | 0.511 | 0.467 |

19 memorized out of 1,200 is 1.6%. The Bloom false-positive rate predicts ~0.14,
so these are real reproductions of training URLs, not filter noise.

### 3.4 Qualitative samples

```
www.barakla.fr/fr/carrier-savin-group-renais-savoir
vriendenheeksecuriteit.nl/teamdevent-beheusen/
meritusek.pl/dekoracje-i-dekoracje/kultury/dragon-bez-weskiej-dla-zelenia/
www.nicholastidium.com/?attachment_id=16629
clinicwebmedia.ca/tag/medical-dinosis/

priyasrivani012.github.io/Netflix-Clone
amazon-clone-free-verified.vercel.app/
s.team-np.com/
glow899.serv00.net/m/services//case/
jeffarmer-biz-lentar-russia.pages.dev/
```

Three things are worth recording. The model produces **language-appropriate**
content without a language token — `.fr` takes an `/fr/` prefix, `.nl` shows
Dutch compounding, `.pl` shows Polish morphology and case endings. This
validates the E2.3 decision to remove that token. It reproduces real **CMS
artifacts** (`?attachment_id=`, `/tag/`). And on the phishing side it found the
free-hosting infrastructure (`github.io` with a capitalised repo name,
`vercel.app`, `serv00.net` complete with a doubled slash) and produced
`s.team-np.com`, which is the genuine Steam typosquat pattern.

---

## 4. Milestones achieved

1. **Benign label field solved.** 16.77 bits, essentially every value unique,
   excess +0.0186 at 1.1σ. This was the dominant tell through E2.x and is now at
   the floor.
2. **Suffix control validated.** Excess +0.0032 / −0.0106. Suffixes are supplied
   from an external table rather than sampled, so a TLD registered after
   training can still be used; these numbers confirm the mechanism is neutral.
3. **Language token removal vindicated.** Language-appropriate output emerges
   from the TLD alone.
4. **Novelty is real.** 100% / 98.4% novel against 14.87M training URLs.
5. **Evaluation is now correctly floored.** Replaced a pre-registered absolute
   bar of 0.65 that sat *below its own floor* (real-vs-real measured 0.741–0.828)
   and was therefore unreachable by any generator, including a perfect one.

---

## 5. Issues investigated during E3

### 5.1 Evaluation harness bugs (found before reading any results)

**Cross-class pools.** `run_eval` built held-out pools from the benign val files
and reused that dict for both classes, so phishing generations were scored
against benign real URLs. That measures the gap between the two classes — real,
large, and not the model's doing. Fixed to build pools per class. Two regression
tests added.

**All-or-nothing sample size.** `discriminate_with_floor` needs `3n` disjoint
host samples. `phishing_structure_val` holds 941 distinct hosts against a
requested n=1200, so the phishing path and query fields printed "insufficient" —
blanking exactly the fields under investigation. Now shrinks `n` to what the
pool supports, with `MIN_N = 250` as the floor (at n=250 the standard error on
excess is ~0.037, comparable to the 0.05 bar, so smaller stops being a
measurement). Reduced rows are flagged.

**Fixed error bar.** A hardcoded `±0.04` was printed for every reduced-n row
regardless of actual n, overstating uncertainty on the larger ones by ~2×. Now
computed per row from Hanley-McNeil. This reclassified phishing `SUB` from
"TELL" to "marginal": +0.0667 is 3.6σ above zero but only 0.9σ above the bar.

### 5.2 A correction to the headline reading

`parse()` uses the **ICANN-only** PSL section, so `github.io`, `vercel.app`,
`pages.dev` and `netlify.app` are not public suffixes:

```
amazon-clone-free-verified.vercel.app  ->  LBL='vercel'  SUB='amazon-clone-free-verified'
priyasrivani012.github.io              ->  LBL='github'  SUB='priyasrivani012'
```

The attacker's invented name lands in **`SUB`**, not `LBL`.

| | phishing LBL | benign LBL |
|---|---|---|
| Distinct | 10,488 | 112,443 of 113,277 |
| Entropy | 9.60 bits | 16.77 bits |
| Top-10 share | **39.8%** | 0.0% |
| Top-10 values | webflow, vercel, pages, github, blogspot, weebly, weeblysite, godaddysites, r2, netlify | *(no mass)* |

**Consequence.** ~40% of the phishing label field is a ten-way platform lookup,
so phishing `LBL` at −0.0006 is a **much weaker claim** than benign `LBL` at
+0.0186. It was initially reported as though the two were equivalent; they are
not. Phishing creativity lives in `SUB` (12.90 bits, 78.9% coverage) — which is
the field carrying a tell at +0.0667. That tell deserves more weight than
"marginal" suggests.

### 5.3 Generation batching inefficiency (open, TD-EVAL-BATCH)

`generate_with_suffixes` issues one `generate()` call per *distinct* suffix so
the forced-suffix injection stays rectangular. For 1,200 draws that is 211 calls
for benign and 115 for phishing — mean batch 5.7 and 10.4 against a
`batch_size=32` mostly unused. There is no KV cache either, so each of up to 256
steps recomputes the whole prefix. Evaluation generation takes ~420s (benign) and
~199s (phishing) on CPU.

Reading `_generate_chunk`, the grouping constraint appears unnecessary: the loop
already appends **exactly one token per row per step**, whether from `queued[i]`
or from sampling, so rows stay equal-length even with per-row forced suffixes of
different lengths. If that holds, 326 calls collapse into ~38 full batches. **Not
yet verified** — the same docstring records a bug that once produced AUC 1.0000,
so this earns caution before being acted on.

---

## 6. Root-cause analysis: the path shortfall

### 6.1 The symptom

Generated URLs contained a non-empty path far less often than training data did:

| | Model | Training blend | Gap |
|---|---:|---:|---:|
| benign | 73.7% | 88.3% | −14.6 pts |
| phishing | 17.1% | 49.5% | **−32.4 pts** |

The training blend is the correct target because generation cannot know which
corpus a host came from. Benign is 35.5% name / 64.5% structure; phishing is
80.6% / 19.4%.

### 6.2 Hypotheses eliminated, in order

**Decoder.** Rejected. All three decoders agree within 1.4 points:

```
class     typical  temperature  top_p   target
benign      75.8%        75.0%  74.0%    88.3%
phishing    19.2%        17.8%  17.8%    49.5%
```

**Forced suffix injection (H1).** Rejected. Free-suffix generation is *worse*,
not better — benign 69.4% vs 74.6% forced; phishing 15.8% vs 19.0%. The
injection was slightly helping.

**Skewed suffix table.** Rejected. The table is built from all four training
corpora, so its marginal matches the training rows.

**Loss-mask omission.** Rejected at this stage. `field_mask` gives markers and
EOS weight 1.0 on rows from both corpora, exactly as designed.

**Platform-hosting rule (H3).** Rejected. The per-suffix breakdown had suggested
one — `.io` on target (+2.4%) while `.com`/`.app`/`.dev` ran 31–37 points under —
and the hypothesis was that the model had internalised "platform-hosted ⇒
terminate after the subdomain." Under real host prefixes, `github.io`,
`vercel.app`, `pages.dev` and `workers.dev` all behave identically at ~100%.
There is no platform-specific rule; the per-suffix spread was a symptom measured
through free generation.

### 6.3 Two probes that had to be discarded

Both would have produced confident wrong answers, and are recorded so the same
mistakes are not repeated.

**Probe 1 — raw `P(PTH)` at the decision point.** Reported −44.7% for the
no-subdomain stratum. Invalid: `<SUB>` is a legal continuation at that position,
so the readout was "will there be a path *immediately*", with mass siphoned off
by trajectories that emit a subdomain first.

**Probe 2 — renormalised `P(PTH | name section ends here)`.** Returned **99.5%
in every single stratum** — benign, phishing, every suffix, every platform. That
uniformity is what exposed it: at that position the model may also continue the
label with more characters, so ~26% of the mass sits outside the choice being
renormalised over. The probabilities do not sum over a clean binary choice.

**Lesson.** No single-position readout can answer "will this URL have a path."
The event spans many steps, so it must be measured by letting the model run.

### 6.4 The measurement that worked, and the contradiction it created

Handing the model a **real host prefix** and letting it continue freely:

| | Real has-path | Model continues with a path |
|---|---:|---:|
| benign `name_val` | 70.2% | **100.0%** |
| benign `structure_val` | 97.2% | **100.0%** |
| phishing `name_val` | 36.3% | **100.0%** |
| phishing `structure_val` | 92.5% | 99.8% |

Given a correct host the model opens a path essentially always — including on
hosts that genuinely have none. An A/B against the model's *own* generated hosts,
recut and re-continued, gave **100.0%** as well. Yet free generation gave 15.8%.

Three measurements of the same model disagreed, which meant the reconstruction
was measuring something other than what generation does.

### 6.5 The actual finding

Instrumenting the real generation loop and recording **raw token ids** — no
decode/encode round trip anywhere:

```
                       benign   phishing
emitted <PTH>          100.0%      99.8%
ended by EOS           100.0%     100.0%
ended by max_len         0.0%       0.0%
mean length              42.7       32.2 tokens
```

**The model emits the path marker on essentially every row.** It then closes it
**empty**, which decodes to `host/`. Every "bare" sample ends in a trailing
slash: `goldencareers.site/`, `s.team-np.com/`, `www.verifysummit.com/`.

Every earlier measurement used `has_path` = "has non-empty segments", so an
empty-bodied `<PTH>` scored identically to no path at all. The reconstruction
probes all agreed at 100% because they were measuring the **marker**, which was
never broken.

Splitting shape from content:

| Source | P(open PTH) | P(body \| PTH) | P(non-empty) |
|---|---:|---:|---:|
| `name_train` | 99.5% | 72.5% | 72.1% |
| `structure_train` | 100.0% | 97.2% | 97.1% |
| **benign BLEND** | **99.8%** | **88.4%** | **88.2%** |
| benign MODEL | 100.0% | 74.5% | 74.5% |
| *gap* | *+0.2%* | *−13.9%* | *−13.7%* |
| `phishing_name_train` | 100.0% | 38.7% | 38.7% |
| `phishing_structure_train` | 100.0% | 93.3% | 93.3% |
| **phishing BLEND** | **100.0%** | **49.3%** | **49.3%** |
| phishing MODEL | 100.0% | 16.0% | 16.0% |
| *gap* | *+0.0%* | *−33.3%* | *−33.3%* |

Marker gap **+0.0%**. Content gap **−33.3%**. The defect is entirely in the
content decision.

### 6.6 The masking asymmetry

`encode` opened `<PTH>` whenever `parts.path` was truthy — and `"/"` is truthy.
So `example.com/` emitted `<PTH>` followed immediately by EOS.

Having emitted `<PTH>`, the model chooses between *"emit a character"*
(non-empty) and *"emit EOS"* (empty). `field_mask` classifies markers and EOS as
**shape** (weight 1.0 on every row) and structure-field characters as **content**
(weight 0 outside the structure corpus). Verified directly on the mask:

```
EMPTY path, name corpus:      <PTH> w=1.0   EOS w=1.0    <- taught
NON-EMPTY path, name corpus:  <PTH> w=1.0   'l' w=0.0    <- masked
NON-EMPTY path, structure:    <PTH> w=1.0   'l' w=1.0    <- taught
```

**The two branches of one decision sat under different loss weights.** The name
corpus supplied the evidence *for* an empty body at full weight while its
counter-evidence was masked to zero.

**Why phishing is 2.4× worse.** The unopposed evidence is proportional to
(share of name rows) × (empty-path rate within them):

| Corpus | Rows | Rows with empty `<PTH>` | Share |
|---|---:|---:|---:|
| `name_train` | 906,214 | 246,075 | 27.2% |
| `structure_train` | 1,648,460 | 46,305 | 2.8% |
| `phishing_name_train` | 192,652 | **117,625** | **61.1%** |
| `phishing_structure_train` | 46,491 | 3,115 | 6.7% |

Phishing's name corpus is 80.6% of its rows at a 61.1% empty rate; benign's is
35.5% at 27.2%. Weighting the evidence gives ~26% for phishing and ~84% for
benign against observed 16.0% and 74.5% — correct direction, correct ordering,
correct rough magnitude. *(Approximate. A contextual model is not a frequency
counter, so the exact numbers should not be over-read.)*

The irony is documented: `field_mask`'s own docstring argues markers and EOS must
be trained on all rows *precisely so* the bare rate does not collapse to zero. It
prevented that error and introduced its mirror image.

---

## 7. The fix

### 7.1 What changed

In `uie/tokenizer.py`, a structure field now opens only when its body is
non-empty:

```python
body = path_body(parts.path) if parts.path else ""
if body:
    field("PTH", body)
if parts.has_query and parts.query:
    field("QRY", parts.query)
if parts.has_fragment and parts.fragment:
    field("FRG", parts.fragment)
```

`QRY` and `FRG` had the identical defect (`http://a.com/?` and `http://a.com/#`
both opened empty markers) and were fixed together. `PRT` was already correct —
an empty port is falsy.

### 7.2 Why this fixes the architecture

After the change, "does this URL have a path" is a choice between `<PTH>` and
`EOS` — **two shape tokens, both weight 1.0 on every row**. The asymmetry cannot
arise, because both branches are now the same *kind* of token. Content remains
pure characters, correctly scoped to the structure corpus.

It is also semantically right: `example.com/` and `example.com` address one
resource, as do `a.com/x?` and `a.com/x`.

### 7.3 What it costs

The encoding is now deliberately lossy in exactly one place: empty structure
fields normalise away, so `a.com/` round-trips to `a.com`. This **reverses an
earlier explicit decision** (`test_present_but_empty_query_still_has_its_marker`
asserted the opposite) and relaxes the round-trip contract from "lossless" to
"lossless except empty structure fields."

Verified as safe downstream: `normalize_for_dedup` already strips trailing
slashes, so novelty and memorization detection are unaffected. Empty `QRY`/`FRG`
bodies are negligible in the corpus (200 / 307 / 0 / 0 and 1 / 0 / 0 / 0 rows).

### 7.4 An alternative that was considered and not taken

`field_mask` could instead treat the **first** token after a structure marker as
shape (unmasked on all rows), leaving later characters scoped. That is also
symmetric and preserves the exact round trip. It was rejected because it leaks
one character of content per field from the name corpus, whose paths are known to
be unrepresentative (depth 1.29 vs a population 2.30) — it would bias the
first-character distribution of every path. The tokenizer gate leaks nothing.

### 7.5 Tests

363 passing. New coverage:

- `test_an_empty_structure_field_does_not_open_a_marker` — replaces the reversed
  assertion, carrying the full rationale
- `test_a_non_empty_structure_field_still_opens_its_marker` — the fix must not
  cost a real body
- `test_both_branches_of_the_path_decision_carry_equal_weight` — asserts the
  property on `field_mask` itself, so a regression is caught at the mask rather
  than 45 minutes of training later

---

## 8. Why retraining is required

The change alters **tokenization**, so every cached encoding is stale and the
E3 checkpoint's learned distribution is defined over a sequence grammar that no
longer exists. Specifically:

1. **The token sequences change** for 27.2% of benign name rows, 2.8% of benign
   structure rows, 61.1% of phishing name rows and 6.7% of phishing structure
   rows. Those rows now end at EOS with no `<PTH>` at all.
2. **The loss weights change** for exactly the decision under repair.
3. **`best.pt` cannot be adapted.** Its `<PTH>` conditional was fit against a
   grammar where the marker was near-obligatory; under the new grammar the same
   weights would emit `<PTH>` on ~100% of rows and then be forced to produce
   content it was never trained to produce.

**What does *not* change:** the corpus itself. No re-collection, no re-split, no
re-sampling. The bundle's JSONL rows, both splits, the suffix table and the Bloom
filter are all unaffected — only the encoding step downstream of them. Estimated
retrain cost: **~45 minutes on Kaggle**, identical hyperparameters.

---

## 9. Open questions and assumptions

**Open — the benign `PTH` tell (+0.1315, 6.8σ).** Plausibly explained by the
same defect: if only 74.5% of paths receive content against a real 88.4%, the
ones that do are a skewed slice. **This is a hypothesis, not a finding.** The
retrain tests it directly. If the tell survives, it is a separate problem.

**Open — the phishing `SUB` tell (+0.0667, 3.6σ).** This is where attacker
naming actually lives, so it matters more than its "marginal" label suggests. It
is upstream of `<PTH>` and there is no reason to expect the retrain to move it.

**Open — will the model emit `<PTH>` then EOS after the fix?** Under the new
grammar that sequence never appears in training, so it is off-distribution. If
the model produces it anyway, `host/` reappears. Must be checked post-retrain.

**Open — composition limitation (architectural, not a bug).** Names are learned
from every domain and paths from multi-page sites only, so sampling a whole URL
composes two conditionals estimated on different populations. No real corpus has
that composition. This is what `FULL` at +0.125 measures. Closing it requires a
design change to how generation composes fields, not a training fix.

**Open — phishing `PTH`/`QRY` are unmeasurable.** 941 held-out multi-page kit
hosts cannot support the comparison. Partly a corpus limit, partly a symptom of
the defect (only 205 of 1,200 generations had a path to score). The retrain
should raise `n_fake` substantially; the pool ceiling remains.

**Open — TD-EVAL-BATCH** (§5.3), unverified.

**Open — the human panel has never been run.** Every quality claim in this
document rests on discriminator AUC and the metrics above.

**Assumption — the training blend is the right target for free generation.**
Generation cannot know which corpus a host came from, so blend proportions are
used throughout. This is a restatement of the composition limitation, not an
independent assumption.

**Assumption — the evidence-weighting account in §6.6 is approximate.** It
predicts ~26%/~84% against observed 16.0%/74.5%. It explains direction,
ordering and rough magnitude; it is not a quantitative model.

---

## 10. Exact next steps

### Before retraining

1. **Rebuild the bundle's encodings.** Corpus files are unchanged; re-run the
   packing step so encodings and the manifest reflect the new tokenizer.
2. **Re-verify the four corpora and both splits load** with the correct
   `key_kind` values (domain-keyed and host-keyed).
3. **Confirm the empty-`<PTH>` row count drops to zero** under the new encoder.

### Retraining

4. **Same hyperparameters**, so the comparison is clean: 20,000 steps, batch 256,
   lr 3e-4, `class_alpha=1.0`, seed 20260803. ~45 minutes on Kaggle.

### After retraining

5. **Run `pathprobe6`-equivalent first**, before anything else. The single
   decisive number is `P(body | PTH)` against the blend: benign 88.4%, phishing
   49.3%. If those close, the fix worked.
6. **Check for `<PTH>`-then-EOS** off-distribution behaviour (§9).
7. **Run `python -m uie.corpus.run_eval --checkpoint data/best.pt --bundle
   data/bundle2`.** Watch in this order:
   - benign `PTH` excess — does +0.1315 fall? (tests the §9 hypothesis)
   - phishing `PTH` `n_fake` — does it rise from 205 far enough to score?
   - benign `LBL` excess — **must stay at the floor**; this is a no-regression
     gate on the milestone already achieved
   - phishing `SUB` excess — expected unchanged at ~+0.0667
8. **Do not re-tune the bar or the metric to fit the outcome.** The bar is 0.05
   excess, cleared by more than 1σ.

### Then

9. Decide whether to address the composition limitation, which is now likely the
   largest remaining source of `FULL` separation.
10. Run a human panel.
11. Commit. The repository currently has ~30 uncommitted files; naming was
    deliberately deferred.

---
---

# PART II — E4 and the frame discovery

Sections 1–10 record E3 as written. E4 falsified two of their claims, and the
investigation that followed changed the diagnosis substantially. Everything
below is dated 2026-08-06, after the E4 retrain.

## 11. E4 results

E4 = E3 plus the empty-body gate (§7). Same hyperparameters, same corpus, same
seed. Checkpoint `data/best_e4.pt`.

### 11.1 Two claims from Part I that E4 FALSIFIED

**§9 predicted the benign `PTH` tell was plausibly the same defect.** It was not.
The tell moved **+0.1315 → +0.1323** — 0.0008, i.e. not at all.

**§7 implied the fix would close the path gap.** It closed roughly half:

| | training blend | E3 | E4 | change |
|---|---:|---:|---:|---:|
| benign P(non-empty path) | 88.2% | 74.5% | 84.4% | +9.8 pts |
| phishing P(non-empty path) | 49.3% | 16.0% | 33.0% | +17.0 pts |

The empty-body defect itself is **fully fixed**: `P(body | PTH)` is 100.0% and
`<PTH>`-then-EOS occurred 0/800 in both classes, so the §9 off-distribution risk
did not materialise.

### 11.2 E4 field results

| Field | E3 | E4 | |
|---|---:|---:|---|
| benign LBL | +0.0186 | +0.0320 | slightly worse (1.9σ), still under bar |
| benign SUB | +0.0262 | +0.0171 | better |
| benign PTH | +0.1315 | +0.1323 | unchanged |
| benign FULL | +0.1250 | +0.2228 | worse |
| phishing LBL | −0.0006 | −0.0132 | still at floor |
| phishing SUB | +0.0667 | +0.0721 | ~same |
| phishing PTH | unmeasurable | +0.2108 | now measurable |
| phishing FULL | +0.1271 | +0.4174 | AUC 0.9166 |

Gate: benign 99.8% pass / 99.8% novel; phishing 97.2% / 97.4%.

---

## 12. The frame discovery

### 12.1 The two corpora are sampling FRAMES, not distribution halves

| Frame | Corpus | How it draws |
|---|---|---|
| **domain** | `name` | one URL per registrable name, over every name |
| **page** | `structure` | up to 40 URLs from one multi-page site |

They disagree systematically about paths (benign): domain frame 72.3% carry a
path at conditional length 28.3; page frame 97.2% at 39.7.

**The generator emits ONE URL PER INVENTED HOST and never revisits a host.** It
therefore samples the *domain* frame and cannot produce page-frame output at all.

### 12.2 What that invalidated

`FIELD_POPULATION` routed `PTH`/`QRY` to `structure` — the page frame. Measured
on E4 benign `PTH`:

| Scored against | full AUC | length-only AUC | real len | gen len |
|---|---:|---:|---:|---:|
| `structure` (page frame) | 0.6440 | **0.6013** | 38.8 | 28.5 |
| `name` (domain frame) | 0.6290 | **0.5151** | 29.3 | 28.5 |

Against the frame the generator actually samples, **length contributes nothing**
(0.5151 = chance). The model's conditional path length (27.9) matches the domain
frame (28.3) almost exactly. The length signal was the frame, not the model.

`FULL` was never a counterweight: its `mixed` pool is **93.9% name-frame hosts**
(benign) and **96.1%** (phishing) after host dedup, because `structure_val` has
7,061 hosts against `name_val`'s 113,277. Deduped mean length 41.7 against the
model's 42.0. Both metrics were always asking for domain-frame output.

**There was no PTH/FULL trade-off.** There was one misrouted table entry.

### 12.3 Three "defects" that dissolved

| Claim | Reality |
|---|---|
| phishing path-rate gap −16.3 pts | measured against the *blend*; against the domain frame it is **37.2% vs 38.7%** — a match |
| `FULL` regression | 93.9% name-frame pool; the model matches it |
| phishing `SUB` tell +0.0721 | direct measurement gives **AUC 0.5175** ≈ chance; the excess came from a floor at 0.4859, which is 0.74σ *below* chance |

---

## 13. Validated remaining defects

Everything below survived measurement and is **frame-independent**.

### 13.1 Uppercase deficit — CONFIRMED

Length-normalised, so not a length artifact. Entirely in the path; host uppercase
is 0.000 in every corpus and in the model.

| | blend | model | ratio |
|---|---:|---:|---:|
| benign uppercase / path char | 4.27% | 1.99% | **0.47** |
| phishing uppercase / path char | 8.71% | 4.44% | **0.51** |
| benign digits / path char | 9.70% | 6.45% | 0.67 |
| phishing digits / path char | 13.70% | 13.81% | 1.01 |
| hyphens (both classes) | — | — | 0.97 / 1.04 |

Specifically **capitals**, not rare characters generally. Drives the residual
0.6290 benign `PTH` tell now that length is excluded.

### 13.2 Phishing path length — CONFIRMED

11.6 characters against 16.8 in the *correct* frame (−31%). Real, not a frame
artifact.

### 13.3 The terminating-EOS asymmetry — the mechanism

E4 fixed *"does a path open"*. The identical split still governs *"where does it
end"*:

```
name row, path 'about':   [PTH] a b o u t EOS
                          [PTH] w=1.0   (shape, every row)
                          a..t  w=0.0   (content, structure rows only)
                          EOS   w=1.0   (shape, every row)
```

Path **length** is taught by every row; path **content** only by structure rows.

### 13.4 Hypotheses REFUTED by measurement

| Hypothesis | Result |
|---|---|
| `typical` decoding truncates the tail | pure `temperature=1.0` sampling is **no better, often worse** — **SUPERSEDED 2026-09-23 by R4**, see `research/qry/R4_RESULTS.md`: at gate scale, paired with identical floors, temperature beat typical on all ten cells and regressed none. `temperature` is now the adopted default. Section 6.2's separate rejection of the decoder as the cause of the *path-emission* gap still stands and is confirmed by R4. |
| PTH tell = segment vocabulary | generated segments are **56.4% in-vocab against a 48.5% real floor** — above it |
| PTH tell = wrong scoring population alone | scoring against `name` gives a similar total; the *driver* changes, not the size |
| trailing slash causes the FULL regression | **not conclusive** — subtractive and injection directions disagree, and injecting to 100% scores *better* than matching the real rate. Raw metric retained |

---

## 14. Ranked issues

| # | Issue | Root cause | Impact | Cost |
|---|---|---|---|---|
| 1 | `PTH`/`QRY` scored in the wrong frame | evaluation | 0.09 AUC spurious | **fixed, no retrain** |
| 2 | Uppercase 0.47–0.51× | content starved: structure rows only | residual 0.6290 benign tell | retrain |
| 3 | Phishing path length −31% in-frame | terminating-EOS asymmetry + 46k structure rows | phishing PTH 0.7038 | retrain |
| 4 | `FULL` trailing slash | inconclusive | phishing only | undecided |
| 5 | ~~path-rate gap~~ | frame artifact | dissolved | — |
| 6 | ~~phishing SUB~~ | low floor estimate | dissolved | — |

---

## 15. E5 — the FRAME control token

**Status: IMPLEMENTED BUT NOT TRAINED. Everything in 15.2 is a PREDICTION.**

### 15.1 Part 1 — scoring alignment (DONE, validated)

`FIELD_POPULATION_BY_FRAME` maps each field to a population *per frame*, and
`GENERATOR_FRAME = "domain"` selects the one the generator samples. The page
frame is **kept**, reported by `run_eval` as a diagnostic that never gates —
"can the model produce multi-page-site paths" stays a live question, it just
cannot be asked of an unconditioned generator.

### 15.2 Part 2 — `<F:domain>` / `<F:page>` (implemented, untrained)

Derivation: field-scoped masking exists for exactly one reason — the model cannot
tell which frame a row came from, so name-corpus paths would contaminate its idea
of a path. **Name the frame and that reason disappears.** The model conditions on
it and learns `P(path | frame)` directly.

Consequences:

- Path content trains on **2.55M rows instead of 1.65M** (benign) and **239k
  instead of 46k** (phishing) — directly attacking issues 2 and 3, both of which
  are content-starvation signatures.
- Shape and content come under **one teacher**, eliminating the bug class that
  produced both the empty-path defect and the short-path bias.
- Both `PTH` metrics become fair: generate `frame=page` to score against
  structure, `frame=domain` against name. No trade-off.

**Risks.** The token encodes a corpus-construction artifact. Phishing
`frame=page` has only 46k rows. Generation now requires choosing a frame, so a
wrong choice biases output silently. Removing the masking is a genuine bet: if
the frame token underfits, name-frame paths contaminate the page frame — the
exact failure the masking was built to prevent.

### 15.3 Pre-registered validation

1. **PRIMARY GATE** — `len|path` given `frame=page` matches `structure_val`
   (39.7 benign / 29.1 phishing); given `frame=domain` matches `name_val`
   (28.3 / 15.9).
2. Uppercase ratio moves toward 1.0 from 0.47 / 0.51. **If it does not, the
   content-starvation account of issue 2 is wrong** and needs another
   explanation. This is what makes the proposal falsifiable rather than a story.
3. Phishing `len|path` in-frame closes from 11.6 toward 15.9.
4. **NO-REGRESSION: benign `LBL` must stay at the floor.** The banked milestone.
5. Contamination check: `frame=page` depth must not collapse toward 1.29.

### 15.4 Implementation state

| Component | State |
|---|---|
| `FIELD_POPULATION_BY_FRAME`, `GENERATOR_FRAME` | done |
| `field_report(frame=...)`, unknown frame rejected | done |
| `run_eval` cross-frame diagnostic | done |
| `URLTokenizer(include_frame=)`, `<F:*>` vocab, `encode(frame=)` | done |
| `state_dict` round-trip; pre-E5 checkpoints default `False` | done |
| `field_mask` drops scoping when the frame is named | done |
| `CORPUS_FRAME` maps name->domain, structure->page | done |
| `encode_corpus` derives the frame from the corpus | done |
| `generate(frame=)` supplies `<F:*>`; unknown frame rejected | done |
| `generate_with_suffixes(frame=)` accepts a name or a mixture | done |
| `kaggle_train --frame` switch | done |

`include_frame=False` by default, so E4 behaviour is bit-identical until
deliberately enabled — asserted by
`test_frame_token_does_not_change_behaviour_when_disabled`.

Test suite: **376 passing** (`tests/test_frame_wiring.py` adds 8, including an
end-to-end train of a framed model so vocab, encoding, mask and loss must agree).

**E5 is launched with `--frame`.** Everything else stays identical to E4: 20,000
steps, batch 256, lr 3e-4, `class_alpha=1.0`, seed 20260803, same corpus bundle.
The vocabulary grows by exactly 2 tokens.

---

# PART III — E5 as measured

E5 = E4 plus `<F:domain>`/`<F:page>` conditioning, with field-scoped content
masking disabled. Identical hyperparameters, corpus and seed. Checkpoint
`data/best_e5.pt`, 14,372,352 parameters, vocab 101.

Numbers are recorded as measured. No reinterpretation.

## 16. Training

| | |
|---|---|
| Wall time | 3,941.7 s |
| Encoded | benign 2,545,571 / phishing 238,214 |
| Dropped over `max_len` | 10,032 (0.36%) — E4 dropped 9,933; +99 because every sequence gained one token |

Validation loss (weighted by the field masks actually optimised):

| step | benign | phishing |
|---:|---:|---:|
| 1,000 | 2.1527 | 2.1947 |
| 5,000 | 1.7300 | 1.6812 |
| 9,000 | 1.6757 | 1.5921 |
| **11,000** | **1.6693** | 1.5642 |
| 13,000 | 1.6835 | 1.5467 |
| 15,000 | 1.6831 | 1.5340 |
| 20,000 | 1.6866 | 1.5249 |

Benign validation loss reached its minimum at step 11,000 and rose thereafter.
Phishing fell by 0.009 over the final 5,000 steps.

Per-field NLL train/val gaps at step 20,000:

| | LBL | PTH | QRY | SUB | SUF |
|---|---:|---:|---:|---:|---:|
| benign | +0.065 | +0.024 | −0.107 | +0.001 | −0.049 |
| phishing | +0.008 | **+0.197** | +0.162 | +0.032 | −0.026 |

**E5 loss values are not comparable to E4's.** With masking disabled the loss
averages over path-content tokens on name rows that E4 excluded entirely.

## 17. Pre-registered validation (15.3)

### P2 — uppercase per path character — NOT CONFIRMED

| class / frame | real | E5 | ratio | E4 ratio |
|---|---:|---:|---:|---:|
| benign / domain | 3.89% | 2.17% | 0.56 | 0.47 |
| benign / page | 4.48% | 1.41% | 0.31 | — |
| phishing / domain | 8.12% | 4.64% | 0.57 | 0.51 |
| phishing / page | 11.18% | 7.09% | 0.63 | — |

Phishing path content increased from 46,491 to 239,143 training rows (5.1×) and
the ratio moved +0.06. Benign page frame is below E4's domain-frame ratio.

### P1 — `len|path` must match the frame generated in — FAILED

| class / frame | real | E5 | gap | E4 gap |
|---|---:|---:|---:|---:|
| benign / domain | 28.3 | 21.8 | −6.5 | −0.4 |
| benign / page | 39.7 | 31.8 | −7.9 | — |
| phishing / domain | 15.9 | 12.4 | −3.6 | −4.5 |
| phishing / page | 29.1 | 16.5 | −12.6 | — |

### P5 — contamination — PASSED

| class | metric | real dom | E5 dom | real page | E5 page |
|---|---|---:|---:|---:|---:|
| benign | depth\|path | 1.78 | 1.72 | 2.14 | 2.18 |
| benign | P(path) | 72.3% | 61.8% | 97.2% | 98.1% |
| benign | digit rate | 8.6% | 5.7% | 10.3% | 6.7% |
| phishing | depth\|path | 1.52 | 1.43 | 2.09 | 1.73 |
| phishing | P(path) | 38.7% | 29.2% | 93.3% | 83.6% |
| phishing | digit rate | 11.7% | 10.9% | 22.1% | 17.8% |

The two frames separate in every metric and in the correct direction.

## 18. Field evaluation (domain frame, the frame generated)

### Benign

| Field | Pop | AUC | Floor | Excess | ± | n | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| LBL | name | 0.5438 | 0.5046 | +0.0392 | 0.0167 | 1200 | ok |
| SUF | name | 0.5143 | 0.5111 | +0.0032 | 0.0167 | 1200 | ok |
| SUB | name | 0.5454 | 0.5135 | +0.0319 | 0.0271 | 456 | ok |
| PTH | name | 0.5757 | 0.4962 | +0.0795 | 0.0210 | 759 | TELL |
| QRY | name | — | — | — | — | 43 | insufficient |
| FULL | mixed | 0.7269 | 0.4704 | +0.2565 | 0.0167 | 1200 | auxiliary |
| *PTH (page, diagnostic)* | structure | 0.6907 | 0.5052 | +0.1855 | 0.0210 | 759 | diagnostic |

### Phishing

| Field | Pop | AUC | Floor | Excess | ± | n | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| LBL | name | 0.5094 | 0.5298 | −0.0204 | 0.0167 | 1200 | ok |
| SUF | name | 0.5214 | 0.5319 | −0.0106 | 0.0167 | 1200 | ok |
| SUB | name | 0.5819 | 0.4721 | +0.1099 | 0.0187 | 958 | TELL |
| PTH | name | 0.5011 | 0.4593 | +0.0417 | 0.0298 | 376 | ok |
| QRY | name | — | — | — | — | 9 | insufficient |
| FULL | mixed | 0.8907 | 0.4992 | +0.3916 | 0.0167 | 1200 | auxiliary |
| *PTH (page, diagnostic)* | structure | 0.6886 | 0.5083 | +0.1802 | 0.0346 | 279 | diagnostic |

Gate: benign 99.8% pass / 99.8% novel; phishing 97.4% / 97.6%.
Precision/recall: benign 0.923 / 0.888; phishing 0.932 / 0.884.

## 19. E4 → E5, like-for-like (domain frame both sides)

| Field | E4 | E5 | Δ |
|---|---:|---:|---:|
| benign LBL | +0.0320 | +0.0392 | +0.0072 |
| benign SUF | +0.0086 | +0.0032 | −0.0054 |
| benign SUB | +0.0171 | +0.0319 | +0.0148 |
| benign PTH | +0.1378 | +0.0795 | **−0.0583** |
| benign FULL | +0.2228 | +0.2565 | +0.0337 |
| phishing LBL | −0.0132 | −0.0204 | −0.0072 |
| phishing SUF | −0.0106 | −0.0106 | 0.0000 |
| phishing SUB | +0.0721 | +0.1099 | +0.0378 |
| phishing PTH | +0.3301 | +0.0417 | **−0.2884** |
| phishing FULL | +0.4174 | +0.3916 | −0.0258 |

TELL count fell from 3 to 2.

### 19.1 What improved

- Phishing `PTH`: +0.3301 → +0.0417, below the 0.05 bar.
- Benign `PTH`: +0.1378 → +0.0795.
- Phishing `FULL`: −0.0258.
- Recall: benign 0.864 → 0.888; phishing 0.857 → 0.884.
- Frames separate (P5), with no contamination.

### 19.2 What stayed the same

- `SUF` both classes (supplied, not generated).
- Phishing `LBL` at the floor.
- Benign `LBL` below the 0.05 bar.
- Gate and novelty within 0.5 points.

### 19.3 What failed

- P1: `len|path` short in all four class/frame combinations.
- P2: uppercase moved +0.06 to +0.09 against a 5.1× data increase for phishing.
- Benign `PTH` remains a TELL at +0.0795.
- `QRY` unmeasurable in both classes (`n_fake` 43 and 9).

### 19.4 Regressions

- **Benign `LBL`** +0.0320 → +0.0392. Trend across three runs: +0.0186, +0.0320,
  +0.0392.
- **Phishing `SUB`** +0.0721 → +0.1099; absolute AUC 0.5581 → 0.5819.
- Benign `FULL` +0.2228 → +0.2565.
- Benign `SUB` +0.0171 → +0.0319.

**Significance of these deltas is not established in this section.** Each single
AUC carries SE ≈ 0.0167 at n=1200, so a difference of two AUCs carries
SE ≈ 0.0236 and the observed deltas are 0.9σ–1.6σ. Section 20 tests them.

## 20. Hypothesis ledger after E5

### Falsified by measurement

| Hypothesis | Evidence |
|---|---|
| `typical` decoding truncates the tail (E4) | `temperature=1.0` and `top_p=0.98` no better, often worse |
| PTH tell is the empty-path defect (§9) | tell moved +0.1315 → +0.1323 |
| PTH tell is segment vocabulary | generated 56.4% in-vocab vs 48.5% real floor |
| Platform-hosting rule | all platforms ~100% under real prefixes |
| Uppercase deficit is content starvation (P2) | 5.1× phishing path data, +0.06 ratio |
| E5 is underfit at 20k steps | benign val loss minimum at step 11,000, rising after |

### Open

| Question | Status |
|---|---|
| Why `len\|path` is short in every frame | **Unknown. More evidence required.** |
| Why uppercase runs at ~0.56× | **Unknown. More evidence required.** |
| Whether benign `LBL` drift is real or noise | untested — §21 |
| Whether phishing `SUB` regression is real or noise | untested — §21 |
| Why phishing `PTH` train/val gap is +0.197 | untested |
| `FULL` trailing-slash attribution | inconclusive since E4 |

---

# PART IV — E6 REJECTED, and the diagnosis corrected

## 21. E6: factorised output head — REJECTED

E6 = E5 plus a factorised output head, `P(token) = P(structural) · P(which | group)`,
with `P(structural) = sigmoid(gate(h))`. Cost 385 parameters. Checkpoint
`data/best_e6.pt` (14,372,737 params, vocab 101, 10 structural / 91 content).

### 21.1 Why it was built

E5 measurement: at path positions the model's EOS mass tracked its own
predictive entropy — 0.111% at 0.0–0.5 nats, 3.022% at 2.5–3.0 nats (27×).
The proposed mechanism was **softmax competition**: EOS shares one normalisation
with 84 content characters, so character uncertainty spreads mass into EOS. The
factorised head removes that channel by construction.

### 21.2 Primary prediction — FAILED

Pre-registered: the entropy→EOS slope flattens.

| entropy (nats) | E5 EOS | E6 EOS | ratio |
|---|---:|---:|---:|
| 0.0–0.5 | 0.144% | 0.174% | 1.21 |
| 1.0–1.5 | 2.716% | 2.568% | 0.95 |
| 2.5–3.0 | 2.964% | 3.093% | 1.04 |

High/low entropy ratio: benign **5.0× → 4.5×**, phishing **4.0× → 3.4×**. Not
flattened. The coupling is not a normalisation artifact; it runs through the
hidden state, which the factorisation does not touch.

### 21.3 Field results — two regressions, no improvements

7 seeds, paired, same seeds both sides:

| field | E5 | E6 | Δ | σ | verdict |
|---|---:|---:|---:|---:|---|
| benign/LBL | +0.0250±0.0150 | +0.0308±0.0119 | +0.0058 | 0.8 | no change |
| benign/SUF | +0.0172±0.0172 | +0.0172±0.0172 | 0.0000 | 0.0 | no change |
| benign/SUB | +0.0280±0.0056 | +0.0450±0.0101 | +0.0171 | **3.9** | **WORSE** |
| benign/PTH | +0.0805±0.0293 | +0.0711±0.0330 | −0.0095 | −0.6 | no change |
| phishing/LBL | +0.0156±0.0203 | +0.0308±0.0184 | +0.0152 | 1.5 | no change |
| phishing/SUF | +0.0109±0.0238 | +0.0109±0.0238 | 0.0000 | 0.0 | no change |
| phishing/SUB | +0.0611±0.0301 | +0.0416±0.0214 | −0.0195 | −1.4 | no change |
| phishing/PTH | −0.0110±0.0357 | +0.0661±0.0315 | +0.0771 | **4.3** | **WORSE** |

**phishing `PTH` was E5's headline result — below the floor at −0.0110,
indistinguishable from real. E6 pushes it to +0.0661, over the 0.05 bar.**
`PTH` did not improve on either class, which is what the change existed to do.

### 21.4 Disposition

**REJECTED. `data/best_e5.pt` remains the baseline.** The `--factorized` flag
stays in the codebase defaulting OFF, with this null result recorded so the idea
is not re-derived.

Why E6 is worse is **unknown**. A single scalar gate deciding structural-vs-content
before the within-group distribution may be less expressive than a joint softmax,
but that is a hypothesis; E6's validation loss against E5's (benign 1.6866 /
phishing 1.5249) would test it and was not captured.

## 22. The diagnosis, corrected

### 22.1 The "leak" was an artifact of my probe design

Earlier probes measured EOS mass ONLY at positions where the path definitively
continues. The empirical hazard there is 0 by construction, so any model mass
looked like excess. Measuring the empirical hazard alongside the model's, over
ALL positions including terminal ones:

| context | n | empirical | E5 model | ratio | E6 model | ratio |
|---|---:|---:|---:|---:|---:|---:|
| benign boundary | 2,734 | 41.697% | 46.033% | 1.10 | 45.674% | 1.10 |
| benign non-boundary | 58,959 | **1.727%** | 1.993% | 1.15 | 1.954% | 1.13 |
| phishing boundary | 999 | 43.644% | 46.184% | 1.06 | 46.215% | 1.06 |
| phishing non-boundary | 16,743 | 4.217% | 4.454% | 1.06 | 4.394% | 1.04 |

**Real paths end mid-segment 1.727% of the time — not zero.** And the model is
boundary-aware: it predicts a 23.1× boundary/non-boundary ratio against an
empirical 24.1×.

The over-prediction is **uniform and small: 1.06–1.15×**.

### 22.2 What that accounts for

Expected mean path length from the measured hazards:

```
model      1 / (0.0443·0.46033 + 0.9557·0.01993) = 25.4 chars
empirical  1 / (0.0443·0.41697 + 0.9557·0.01727) = 28.6 chars   (real: 28.3)
```

Perfect calibration buys **25.4 → 28.6**, about **half** the observed ~6.5-char
deficit (generated 21.8–23.8). The remainder is self-conditioning drift, the same
mechanism established for uppercase.

### 22.3 An error in the earlier logit-bias rejection

A uniform 1.15× over-prediction in odds requires δ = ln(1/1.15) ≈ **0.15**.
Per context:

| | over-prediction | implied δ |
|---|---:|---:|
| benign non-boundary | 1.15× | 0.146 |
| benign boundary | 1.19× | 0.176 |
| phishing non-boundary | 1.06× | 0.057 |
| phishing boundary | 1.11× | 0.103 |

**The earlier sweep tested δ ∈ {0.3, 0.6, 0.9, 1.2} — 2× to 8× too strong.**
The depth overshoot (1.8 → 2.1) and digit distortion used to reject the method
are the expected consequence of an over-strong correction. **That rejection was
unfounded; the correct range was never tested.**

## 23. Hypothesis ledger — path length

### Falsified

| Hypothesis | Evidence |
|---|---|
| `typical` decoding truncates the tail | pure sampling no better |
| Positional length prior (H1a) | ratios 0.69–1.51, no trend with host length |
| Softmax entropy leakage (H1b as *mechanism*) | E6 removed the channel; coupling survived (5.0×→4.5×) |
| Mid-path EOS is a *leak* | empirical non-boundary hazard is 1.727%, not 0 |
| Global logit bias | inflates host 17.2→18.4, label 11.3→12.6 |

### Confirmed

| Finding | Evidence |
|---|---|
| Model is boundary-aware | 23.1× predicted vs 24.1× empirical |
| Over-prediction uniform, 1.06–1.15× | boundary/non-boundary table |
| Accounts for ~half the deficit | 25.4 → 28.6 vs 28.3 real |
| Field-scoped bias leaves host fields untouched | `lbl_len` 11.4 flat, `host_len` 17.1–17.3 flat over δ 0–1.2 |

### Open

- The other ~half of the deficit: self-conditioning drift. **Unknown mechanism.**
- Why E6 regressed two fields. **Unknown.**
- Uppercase: percent-encoding (0.21×) and rate-dependent mode collapse
  (benign 3.03%→0.13% vs phishing 12.71%→12.50%). Deferred by decision.

---

# PART V — the calibrated scale, two overturned verdicts, and the located defect

## 26. The evaluation harness was wrong four times

Every comparison before this section varied ONE source of randomness and reported
its spread as the total. Four errors of the same kind, each producing a confident
wrong conclusion:

| # | Error | What it produced |
|---|---|---|
| 1 | Analytic Hanley-McNeil SE omitted pool-resampling and CV-fold variance | ±0.0167 reported where the true spread is 0.018–0.050 |
| 2 | Single-seed discriminator comparison | two phantom regressions (benign LBL, phishing SUB), both 0.5σ over 7 seeds |
| 3 | Single-GENERATION comparison | a 3.6σ benign `SUB` regression from a δ that cannot causally touch subdomains |
| 4 | Theoretical 2σ threshold on a small-sample statistic | 2.9σ on a KNOWN NULL |

Error 4 is the one that exposed the rest. With G=5 the statistic has 4 degrees of
freedom, so |t| > 2 has p ≈ 0.116, not 0.05 — and across 4 fields that is a ~40%
family-wise false-positive rate.

### 26.1 The replacement

`research/e5/harness.py`. Per arm: **G = 5 generation seeds × D = 3 discriminator
seeds**, generations cached on disk keyed by `(checkpoint, label, n, bias, seed)`
and reused across every experiment.

```
per-generation estimate = mean over D            (averages out fit noise)
arm estimate            = mean over G
SE                      = sd(per-generation means) / sqrt(G)
paired difference       = per-index differences; SE = sd(diffs)/sqrt(G)
```

Both variance components are reported, because they are comparable in size and
omitting either is what went wrong. `n = 1800` is forced, not chosen: `SUB` is
UNMEASURABLE at n=600 — ~42% field coverage yields ~250 values against a 3n
requirement with a 250 floor.

### 26.2 The null self-test

**`--null` compares an arm against ITSELF under different generation seeds.** It
must use *different* seeds; identical seeds produce identical output and pass
trivially while proving nothing.

The first run FAILED at 2.9σ on `PTH`. A harness that reports a regression
between two draws of the same distribution is measuring its own noise, and no
experiment run on it counts. **This test should have existed before the first
comparison, not after the fourth.**

### 26.3 Empirically measured thresholds

Rather than assume a threshold, the null distribution was measured: a pool of
M=16 generations at n=1800, excess cached per `(generation, field, disc_seed)`,
then R=200 null replicates formed by random disjoint 5+5 partitions. Steps 1–2
cost the compute; step 3 is free, so R can be large. Replicates share generations
and are therefore correlated — acceptable for a percentile, and stated so it is
not over-read.

| field | median \|σ\| | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| LBL | 0.81 | 2.61 | 3.08 | 6.03 | **8.84** |
| SUF | 0.75 | 2.39 | 2.74 | 4.30 | 4.83 |
| SUB | 0.82 | 2.30 | 2.73 | 5.04 | 6.89 |
| PTH | 0.73 | 2.08 | 2.53 | 5.37 | 5.44 |

**Under a known null, |σ| reached 8.84.** Because all four fields are scanned,
the family must be controlled at ~p98.75, which lands near p99:

```
family-wise thresholds:  LBL 6.03   SUF 4.30   SUB 5.04   PTH 5.37
```

A verdict counts only if |σ| exceeds the FAMILY-WISE threshold.

## 27. Two rejections overturned; δ adopted

### 27.1 Both were measurement artifacts

Re-measured on the calibrated harness, same fields, same checkpoints:

| | old (uncalibrated) | new (calibrated) |
|---|---|---|
| δ benign `SUB` | +0.0106, **3.6σ** | +0.0008, **0.5σ** |
| δ benign `PTH` | +0.0566, **3.3σ** | −0.0046, **−0.8σ** |
| E6 benign `SUB` | +0.0171, **3.9σ** | +0.0023, **1.2σ** |

All three were reported as regressions in earlier sections. They were noise.

### 27.2 δ path-scoped EOS bias — ADOPTED

Subtract δ from the EOS logit **only while inside a path body**. Decode-time
only: no retrain, no checkpoint change, revertible by setting δ=0.

| | benign δ=0.22 | phishing δ=0.30 |
|---|---|---|
| `len\|path` | 21.39 → **23.69** (+2.30, 5.5σ) | 12.43 → **13.30** (+0.87, 2.0σ) |
| depth (target) | 1.68 → **1.77** (1.78) | 1.39 → **1.46** (1.52) |
| `P(path)` | flat 61.2 → 61.5% | flat 31.5 → 31.7% |
| field harm | none, worst 0.8σ | none, worst 1.0σ |
| deficit closed | 33% | 25% |

Full field results, G=5 × D=3, paired:

```
benign      LBL +0.3s   SUF  0.0s   SUB +0.5s   PTH -0.8s
phishing    LBL +0.7s   SUF -1.0s   SUB +1.0s   PTH +0.8s
```

**Caveat that keeps this honest:** δ improves structural realism that the field
discriminators do not reward — `PTH` excess is unmoved. In the domain frame,
length-only AUC is 0.5151 (chance), so length was never what that discriminator
detected. δ is an efficient mitigation, not a cure.

The derived δ was correct for benign (own-trajectory hazard inflation 1.15×) and
too low for phishing (0.06 against an empirical optimum ≥0.30), which S1 explains:
phishing shows no own-trajectory inflation, so a δ derived from teacher-forced
calibration understates what free generation needs there.

### 27.3 E6 factorised head — REJECTED, on corrected grounds

**Not harmful. Simply no benefit.** The regression charge is withdrawn (1.2σ).
E6 fails because:

1. its primary mechanistic prediction failed — entropy→EOS slope 5.0× → 4.5×,
   not flattened
2. the mechanism it removed does not exist — mid-path EOS is not a softmax leak
3. it costs 385 parameters for nothing measurable

The phishing re-judgement was deliberately **not run**: no result could accept
E6, so it would only restate the autopsy.

## 28. S3 corrected — the defect is located

### 28.1 The first S3 measurement was invalid

It compared the model's **conditional** entropy at segment starts (2.836 benign /
2.182 phishing) against the **marginal** entropy of first characters pooled over
all contexts (3.292 / 3.505), reading the ratios 0.86 / 0.62 as over-confidence.

`H(X) ≥ H(X|C)` always, **so a perfect model also scores below 1.0.** The
confound pointed in the direction of the desired conclusion.

### 28.2 The valid measurement

```
gap = E_real[-log p_model(x)] - E[H(p_model)]
    = cross-entropy - MODEL entropy
```

**Correction — this quantity is NOT the KL divergence.** Every table and heading
in this section labelled it `gap = KL` until §28.4 was run; those labels were
wrong and have been changed in place, with the error recorded here rather than
erased. The two quantities differ by an entropy term:

```
KL(p_real || p_model) = CE - H(p_real)
gap                   = CE - H(p_model)
                      = KL + [H(p_real) - H(p_model)]
```

They coincide only where the model's entropy already matches the truth's — which
is precisely the thing under test, so the identity cannot be assumed. Two
consequences follow, and the second is the one that matters:

- `gap >= 0` is **not** guaranteed. An over-smoothed model drives it negative,
  which §28.4 observes directly (−0.283 at T=1.6). A divergence cannot do that,
  and that impossibility is what exposed the error.
- **KL was never measured and cannot be**, here or anywhere else in this report.
  `H(p_real)` is the entropy of the true conditional distribution over the next
  character; a corpus supplies one *sample* from it per context, never the
  distribution. Only *differences* in cross-entropy are estimable.

What `gap` does measure is **over-confidence**: how far the model's surprise on
real data exceeds the surprise its own distribution predicts. Confidence beyond
accuracy, in nats. It is zero for a perfect model, positive when over-confident,
negative when under-confident — a valid and well-posed diagnostic, just not a
divergence. **The S3 finding stands unchanged under the corrected reading,**
because it was always a comparison of the same statistic at two position classes,
and no step of it depended on the KL label.

Levels are irrelevant; only whether the boundary gap exceeds the mid-segment gap.

| | n | NLL | H_model | gap (over-confidence) |
|---|---:|---:|---:|---:|
| benign boundary | 3,752 | 3.168 | 2.839 | **+0.329** |
| benign mid-segment | 57,941 | 1.410 | 1.379 | +0.032 |
| phishing boundary | 1,705 | 2.514 | 2.224 | **+0.290** |
| phishing mid-segment | 16,037 | 1.399 | 1.319 | +0.080 |

**Ratio 10.39× benign, 3.63× phishing.**

By characters since the last separator (benign):

```
since:  0 (start)    1      2      3      4      5      6      7      8
gap:      +0.329  +0.026 +0.047 +0.075 +0.037 +0.049 +0.045 +0.035 +0.023
```

**A spike, not a gradient.** Position 0 is +0.329; every other position sits in a
flat band of +0.023 to +0.075. The flat band is the internal control: uniform
miscalibration would put it near +0.329 too.

### 28.3 What is now established

The model is miscalibrated at exactly one decision — **which word to start** —
and is well calibrated everywhere else.

| Finding | Evidence |
|---|---|
| Depth correct; deficit is intra-segment | 1.79 vs 1.78 |
| Known words at exact length; invented words 20–25% short | 1.00 / 0.96 vs 0.80 / 0.75 |
| Model CAN sustain long segments | NLL falls 2.203 → 1.458 within the same segments |
| `P("/")` calibrated on real prefixes | 1.02 benign, 0.89 phishing |
| Own-trajectory hazard inflation | benign 1.14–1.21×; phishing none |
| **Boundary over-confidence spike** | **×10.4 benign, ×3.6 phishing, position 0 only** |

Ruled out by measurement: capacity, attention span / horizon, softmax structure,
termination hazard, decoder truncation, compounding entropy collapse, rarity as a
consistent driver.

**Attribution branch CLOSED — see PTH_ATTRIBUTION.md (frozen).**

**S4 target: reduce the boundary over-confidence. Any S4 design must predict a
specific reduction in it, so it can fail cleanly the way E6's entropy-flattening
prediction did.**

### 28.4 Temperature scaling — how much of the spike is *sharpness*?

Over-confidence has two possible causes, and they demand different fixes. The
distribution may be the right SHAPE but too SHARP — recoverable by a single
scalar, no retraining. Or it may be the wrong shape, in which case no rescaling
reaches it and only training can.

Temperature separates them, because `p_T = softmax(logits / T)` is exactly the
one-parameter sharpness family. **The objective is cross-entropy, not the §28.2
gap:** since `KL = CE − H(p_real)` and `H(p_real)` does not depend on `T`,
minimising CE minimises KL *exactly*, without ever needing to know `H(p_real)`.
Fitted on held-out validation, teacher-forced (`research/e5/s4_temperature.py`).

> An earlier version of this sweep minimised the §28.2 gap as if it were KL. It
> returned negative "divergences" (−0.59, −0.91) — impossible — which is what
> forced the correction now recorded in §28.2. Both runs are kept in the ledger;
> only the CE-based one is interpreted.

```
benign      T     bnd CE   bnd overconf     mid CE   mid overconf
          1.00    3.1684        +0.3290     1.4103        +0.0317
          1.05    3.1543        +0.2480     1.4101        -0.0256
          1.15    3.1382        +0.1084     1.4173        -0.1326
          1.25   *3.1341        -0.0070     1.4325        -0.2319
          1.40    3.1429        -0.1455     1.4666        -0.3703
          1.60    3.1718        -0.2830     1.5273        -0.5409

phishing  1.00    2.5136        +0.2900     1.3987        +0.0800
          1.10    2.4943        +0.1174    *1.3955        -0.0148
          1.20   *2.4904        -0.0248     1.4007        -0.1044
          1.40    2.5116        -0.2447     1.4298        -0.2771
```

| | boundary T\* | mid-segment T\* | boundary CE gain |
|---|---:|---:|---:|
| benign | **1.25** | 1.05 | 0.0343 nats |
| phishing | **1.20** | 1.10 | 0.0232 nats |

**Two findings, pointing opposite ways.**

**(a) The miscalibration is boundary-SPECIFIC.** Boundary positions want
`T* = 1.25 / 1.20`; mid-segment positions inside the same path bodies want
`1.05 / 1.10`. A globally over-sharp model would want one temperature
everywhere. This independently confirms S3 through a different statistic, and it
is what makes a *targeted* intervention well-posed rather than a blunt
sharpness knob.

**(b) But sharpness is only a small part of it.** The best available CE gain is
**0.034 nats** (benign) and **0.023** (phishing) — while the over-confidence gap
being corrected is +0.329 and +0.290. Temperature has one degree of freedom; it
can move the distribution's scale, not its ranking. **Roughly a tenth of the
boundary defect is sharpness. The rest is shape, and no decode-time rescaling
can reach it.**

Note also that the T\* which zeroes the over-confidence gap (≈1.25 benign) is
*not* the same as the T\* minimising CE — they nearly coincide here by accident.
The gap is a diagnostic; CE is the objective.

**What this does not settle, and why it was not pursued.** A small CE gain does
not imply a small effect on *generation*: flattening at the segment start widens
the typical set at exactly the decision that picks a word. A decode-time arm was
built and then **abandoned before it produced a result**, on cost grounds — each
arm needs 5 × 1800 generations, and `generate_with_suffixes` groups samples by
suffix, so 1,800 URLs run as 236 sequential decodes at a median batch size of 2.
Measuring a ~0.03-nat calibration change at that price was the wrong trade
against testing the mechanism that can actually reach a shape error.

One measurement from that arm is worth keeping, because it constrains S4. The
structural statistic's null was calibrated on the 16 baseline generations of
§26.3, and **|σ| reached 11.47 on mean segment length** with p90 = 2.11 — so the
`|σ| > 2` threshold that had been pre-registered for it sits at roughly p90.
§26.3's error recurred in a statistic written the same day, caught only because
the null was measured before the arms were compared.

**S4 (§29) therefore goes to the loss, not the decoder.** Cross-entropy is a
strictly proper scoring rule — its unique minimiser over distributions is the
true conditional — so gradient pressure moves the whole distribution rather than
rescaling it. That is the only mechanism available that can reach a shape error.

---

## 29. Canonical acceptance of the resting configuration

Every delta number before this section came from a research harness written for
the investigation. This is the first time `uie/corpus/run_eval.py` -- the
official gate -- scored the configuration actually shipped.

Wiring only: `run_eval` gained an optional per-class `--delta`, default 0.0. The
delta=0 run reproduces the previous report EXACTLY (benign LBL +0.0392, SUF
+0.0032, SUB +0.0319, PTH +0.0795; phishing LBL -0.0204, SUF -0.0106, SUB
+0.1099, PTH +0.0417), which is the proof the wiring is non-invasive. The bar,
`discriminate_with_floor`, the decoder, `field_mask`, `class_alpha` and the
checkpoint are untouched.

`best_e5.pt`, seed 20260806, n=1200, bar 0.05:

| field | benign d=0 | benign d=adopted | phishing d=0 | phishing d=adopted |
|---|---:|---:|---:|---:|
| LBL | +0.0392 | +0.0229 | -0.0204 | -0.0111 |
| SUF | +0.0032 | +0.0032 | -0.0106 | -0.0106 |
| SUB | +0.0319 | +0.0472 | **+0.1099** | **+0.0992** |
| PTH | **+0.0795** | **+0.1018** | +0.0417 | **+0.0905** |
| QRY | unmeasured | unmeasured | unmeasured | unmeasured |
| FULL | +0.2565 | +0.2580 | +0.3916 | +0.3885 |

**ROADMAP-E3'S GATE IS NOT PASSED, under either configuration.** delta=0 fails
two cells; delta=adopted fails three.

### 29.1 The instruments disagree about delta

> **SUPERSEDED by 30.7.** Measured at n_main=8000 with paired seeds, delta is
> neutral on every gated field (all |t| < 1.6). The disagreement below was
> single-draw noise.


delta was adopted in 27.2 on evidence of "no field harm, worst 1.0 sigma". Here
PTH excess RISES in both classes and phishing PTH flips PASS -> FAIL.

This does not establish that delta is harmful, and it is not read that way. The
canonical run is ONE draw at n=1200; 27.2's harness was G=5 x D=3 at n=1800, and
the discriminator-split variance measured there spanned 0.0419 / 0.1311 / 0.0836
on IDENTICAL data -- a range wider than the disagreement. The defensible
statement is narrower: **delta's PTH-neutrality is not established by the
official instrument**, and the two disagree. delta does improve LBL in both
classes and phishing SUB slightly.

### 29.2 QRY is unmeasured, and that is a generator fact

> **SUPERSEDED by 30.6.** QRY is now measured via the pooled estimator and is a
> TELL in all four configurations (+0.1196 to +0.1488) — the model's largest
> field defect. The coverage point below was right and understated.


37 benign and 19 phishing query strings per 1200 generated, against a floor of
1200. delta suppresses EOS inside the path BODY and cannot manufacture a query.
Reported as unmeasured rather than converted to a pass: an absent field is a
coverage failure wearing a measurement failure's clothes.

### 29.3 Largest failure is the least investigated

> **WITHDRAWN — see 30.11.** The +0.1099 reading was the 100th percentile of 15
> draws; phishing SUB's calibrated mean is +0.0576. The real misallocation was
> away from QRY, which no prior section measured.


Phishing SUB (+0.0992 to +0.1099) is the largest single gate failure and has
never been investigated. The entire S1-S4 and attribution effort went to PTH,
which is the smaller failure. That is a misallocation the record should state
plainly.

---

# PART VI — the canonical evaluation at full measurement capacity

## 30. E3 re-measured at n_main = 8000 — five convicted cells, and QRY

> **Note, 2026-09-23 - superseded as the acceptance reading.** This section's
> verdicts come from the v1 gate: floors seeded per split and shared by every
> generation, D=3. The repaired gate v2 (independent floor per generation and
> split, D=10, verdict computed in the gate) is the acceptance reading from
> 2026-09-23: [research/qry/GATE_V2_RESULTS.md](research/qry/GATE_V2_RESULTS.md).
> Every number below still reproduces exactly with `--estimator v1`.

Section 29 scored the shipped configuration on the official gate for the first
time, and reached "NOT PASSED" on two cells at delta=0 and three at
delta=adopted, with QRY blank in all four. Every one of those verdicts rested on
**one generation and one split**, on an instrument whose own split-seed spread on
identical data had been measured at 0.0419 / 0.1311 / 0.0836 — a range of 0.089,
nearly twice the 0.05 bar it was being read against.

This section replaces those verdicts. Nothing about the gate changed: the 0.05
bar, `discriminate_with_floor`, `field_mask`, `class_alpha`, the decoder and
`best_e5.pt` are all untouched, and `git diff` across this work touches exactly
one file, `uie/corpus/run_eval.py`, with additions only.

### 30.1 Why n_main had to rise, and why that is not a bar change

Four cells could not be judged at n=1200. Every one was starved on the **fake**
side, not by the corpus. Real pools after host-dedup, against MIN_N = 250:

| field | benign pool | cap = pool/3 | phishing pool | cap = pool/3 |
|---|---:|---:|---:|---:|
| LBL / SUF | 113,277 | 37,759 | 24,081 | 8,027 |
| SUB | 47,161 | 15,720 | 19,004 | 6,334 |
| PTH | 82,032 | 27,344 | 9,388 | 3,129 |
| QRY | 7,253 | 2,417 | **1,068** | **356** |

Only phishing QRY is bound by the real side, at 356. Everything else was limited
by how few generated rows carried the field: benign SUB was scored at n=450 and
phishing PTH at n=371, while LBL, SUF and FULL enjoyed the full n=1200 — **the
same 0.05 bar applied to cells measured at a third of each other's capacity.**

`field_report` still requests `min(n_main, 1200)`, unchanged. Raising `n_main`
therefore **cannot** push any field above the already-published standard of
n=1200; it only lets the starved cells reach it. That is the whole intervention.

### 30.2 Three additions to the evaluator, and what each may not do

**Auxiliary cap.** Improved precision/recall compares every generated point
against every real one, so it is O(n^2) in time and memory; at a raised `n_main`
it aborts the run before a field is scored. The gate pass rate, novelty rate,
precision/recall and diversity are held at `N_MAIN` = 1200, which also keeps them
comparable with every published run — distinct-4gram ratio in particular falls
with set size by construction. At `n_main` = `N_MAIN` the slice is the whole list,
so the canonical 1,200-sample run remains bit-identical.

**Pooled fallback.** Reaching MIN_N for a rarely-emitted field in a *single*
generation needs `n_main` around 23,600, which at the measured 4.72 s per unit of
`n_main` across both configurations is roughly two days of wall clock for one
cell. The G generations already exist and are already independent, so they are
pooled. **It is a different estimator and is reported as one.** Its spread comes
from D split seeds on one pooled set, so it cannot see generation variance —
which 26.1 established is the dominant term. Its error bar is therefore known to
be understated, and that cuts one way only:

    TELL       available — an excess far above the bar survives an error bar
               that is too small, a fortiori
    UNDECIDED  the default
    "ok"       NEVER — "ok" is a claim about a spread this estimator does
               not measure

A pooled row can convict and can abstain. **It cannot acquit.**

**Emission rates.** The gate is field-conditional: `discriminate_with_floor`
extracts the field and drops empties on *both* sides, so it asks whether emitted
queries look real and can never ask whether enough are emitted. That blind spot
is recorded in 30.6 and is not a small one.

### 30.3 The estimator, and a degrees-of-freedom caveat that matters

The printed verdict column is deliberately conservative — `ok` only if every one
of 15 draws clears, `TELL` only if every one fails — so a field whose spread
straddles the bar even once reads UNDECIDED. That is correct for a printed table
and wrong as a verdict, because the 15 draws are not 15 independent replicates:
three split seeds share one generation.

**The primary statistical verdict is the 26.1 calibrated estimator**: average the
D splits within a generation, then take the standard error across the G
generation means, treating the generation as the unit of replication.

> **G = 5 gives that SE four degrees of freedom.** The appropriate two-sided
> threshold is **t > 2.78**, not a naive 2-sigma rule. Every `FAILS` below clears
> 2.78 except phishing SUB at delta=adopted (t = 2.20), which is called out
> explicitly in 30.5 rather than counted.

> **Note, 2026-09-23 - SE convention.** The t-values here were computed with
> SE = sd(generation means) / sqrt(G - 1). Section 26.1 specifies sd / sqrt(G),
> the standard one-sample form, which gate v2 now encodes
> (`research/qry/PREREG_GATE_V2.md`). All 25 published cells were
> re-checked under both: **no verdict differs.**

### 30.4 Results — delta = 0

`best_e5.pt`, seed 20260806, n_main 8000, G=5, D=3, bar 0.05. Reports in
`data/e5_gate_n8000_d0/`.

**Benign**

| field | excess | SE | t | 26.1 verdict | sd | min | max | n |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| LBL | +0.0301 | 0.0041 | −4.87 | clears | 0.0118 | −0.0009 | +0.0460 | 1200 |
| SUF | +0.0029 | 0.0024 | −19.95 | clears | 0.0201 | −0.0216 | +0.0383 | 1200 |
| SUB | +0.0391 | 0.0008 | −13.92 | clears | 0.0047 | +0.0280 | +0.0469 | 1200 |
| PTH | **+0.0767** | 0.0029 | **+9.12** | **FAILS** | 0.0202 | +0.0494 | +0.1100 | 1200 |
| QRY | **+0.1196** | 0.0121 | — | **pooled TELL** | 0.0172 | +0.0955 | +0.1342 | 995 |
| FULL | +0.2375 | 0.0033 | +57.55 | auxiliary | 0.0173 | +0.2099 | +0.2671 | 1200 |

**Phishing**

| field | excess | SE | t | 26.1 verdict | sd | min | max | n |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| LBL | +0.0187 | 0.0045 | −7.03 | clears | 0.0205 | −0.0127 | +0.0591 | 1200 |
| SUF | +0.0003 | 0.0038 | −13.26 | clears | 0.0312 | −0.0427 | +0.0483 | 1200 |
| SUB | **+0.0576** | 0.0021 | **+3.66** | **FAILS** | 0.0198 | +0.0244 | +0.0887 | 1200 |
| PTH | **+0.0787** | 0.0062 | **+4.63** | **FAILS** | 0.0181 | +0.0385 | +0.1075 | 1200 |
| QRY | **+0.1485** | 0.0173 | — | **pooled TELL** | 0.0244 | +0.1140 | +0.1680 | 356 |
| FULL | +0.3991 | 0.0011 | +303.72 | auxiliary | 0.0076 | +0.3877 | +0.4112 | 1200 |

Note that phishing LBL prints UNDECIDED — its spread is wide enough that one draw
of fifteen crossed the bar — while its calibrated mean of +0.0187 clears at
−7.03. This is exactly the disagreement 30.3 anticipates, and the calibrated
verdict governs.

### 30.5 Results — delta = adopted (benign 0.22, phishing 0.30)

Same generation and split seeds as 30.4, so the two configurations pair
draw-for-draw. Reports in `data/e5_gate_n8000_dadopted/`.

**Benign**

| field | excess | SE | t | 26.1 verdict | sd | min | max | n |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| LBL | +0.0260 | 0.0026 | −9.38 | clears | 0.0103 | +0.0110 | +0.0457 | 1200 |
| SUF | +0.0035 | 0.0027 | −17.26 | clears | 0.0210 | −0.0216 | +0.0382 | 1200 |
| SUB | +0.0400 | 0.0017 | −6.04 | clears | 0.0053 | +0.0304 | +0.0481 | 1200 |
| PTH | **+0.0728** | 0.0032 | **+7.16** | **FAILS** | 0.0221 | +0.0301 | +0.1039 | 1200 |
| QRY | **+0.1233** | 0.0051 | — | **pooled TELL** | 0.0073 | +0.1159 | +0.1332 | 1124 |
| FULL | +0.2269 | 0.0011 | +159.12 | auxiliary | 0.0168 | +0.2045 | +0.2582 | 1200 |

**Phishing**

| field | excess | SE | t | 26.1 verdict | sd | min | max | n |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| LBL | +0.0265 | 0.0035 | −6.75 | clears | 0.0207 | −0.0108 | +0.0599 | 1200 |
| SUF | −0.0032 | 0.0036 | −14.81 | clears | 0.0322 | −0.0427 | +0.0600 | 1200 |
| SUB | +0.0651 | 0.0069 | *+2.20* | *see below* | 0.0258 | +0.0145 | +0.1113 | 1200 |
| PTH | **+0.0834** | 0.0021 | **+16.20** | **FAILS** (TELL even on the all-draw rule) | 0.0131 | +0.0596 | +0.1034 | 1200 |
| QRY | **+0.1488** | 0.0145 | — | **pooled TELL** | 0.0205 | +0.1199 | +0.1640 | 356 |
| FULL | +0.3995 | 0.0019 | +179.57 | auxiliary | 0.0075 | +0.3787 | +0.4091 | 1200 |

> **Phishing SUB at delta=adopted is NOT counted on its own evidence.** Its excess
> of +0.0651 exceeds the bar, but t = 2.20 is below the t > 2.78 threshold that
> four degrees of freedom demand. The confirmed conclusion for phishing SUB rests
> on its **delta=0 paired counterpart, +0.0576 at t = 3.66**, which does clear.
> Stating it any other way would overstate the adopted-delta result.

Phishing PTH at delta=adopted is the only cell in the study that convicts under
the conservative all-draw rule as well: its minimum over 15 draws is +0.0596.

### 30.6 QRY — measured for the first time, and the worst field in the model

QRY has been blank in every previous report. It is not blank because it is
healthy; it was blank because fewer than 250 generated rows carried the field.

| configuration | excess | pooled n | floor reached |
|---|---:|---:|---|
| benign delta=0 | **+0.1196** | 995 | yes (MIN_N 250) |
| benign delta=0.22 | **+0.1233** | 1124 | yes |
| phishing delta=0 | **+0.1485** | 356 | yes — **at the real-side cap** |
| phishing delta=0.30 | **+0.1488** | 356 | yes — **at the real-side cap** |

**QRY is a robust TELL across both classes and both decode configurations.** It is
roughly twice the next-worst field in either class. The phishing cells sit exactly
at 356, the maximum the held-out corpus will ever permit (1,068 query-bearing rows
after host-dedup, divided by three disjoint samples), so phishing QRY is now
measured at the highest capacity this corpus can support and its per-draw SE of
~0.031 against a 0.05 bar is a permanent property of the data, not of this run.

**QRY emission — the chain TRAIN -> VALIDATION -> GENERATED**

| | benign | phishing |
|---|---:|---:|
| training corpus | 6.361% | 4.294% |
| validation corpus | 6.403% | 4.435% |
| generated, delta=0 | **2.49%** | **1.30%** |
| generated, delta=adopted | **2.81%** | **1.70%** |

Train and validation agree to within 0.05 and 0.15 percentage points. **The
emission deficit therefore cannot be explained by a train/validation corpus
mismatch** — the model emits queries at roughly 40% and 30% of the rate present in
its own training distribution. This was checked precisely because it was the most
likely way the finding could have been an evaluation artifact. It is not one.

**QRY is defective on two independent axes:**

1. **Emission rate** — too few queries are produced at all.
2. **Conditional content** — the queries that *are* produced are the most
   distinguishable thing the model emits.

These are recorded as two defects. **It is not assumed that they share a cause.**
A model could emit too few queries that were individually perfect, or the right
number that looked wrong; this model does neither, and whether one mechanism
produces both is exactly the open question.

### 30.7 delta — no effect on the gated fields, and a retraction

Paired across identical generation and split seeds, per-generation means, G = 5:

**Benign**

| field | delta=0 | delta=adopted | diff | SE | t |
|---|---:|---:|---:|---:|---:|
| LBL | +0.0301 | +0.0260 | −0.0042 | 0.0048 | −0.88 |
| SUF | +0.0029 | +0.0035 | +0.0005 | 0.0005 | +1.15 |
| SUB | +0.0391 | +0.0400 | +0.0009 | 0.0015 | +0.60 |
| PTH | +0.0767 | +0.0728 | −0.0039 | 0.0048 | −0.81 |
| FULL | +0.2375 | +0.2269 | −0.0106 | 0.0026 | −4.01 (auxiliary) |

**Phishing**

| field | delta=0 | delta=adopted | diff | SE | t |
|---|---:|---:|---:|---:|---:|
| LBL | +0.0187 | +0.0265 | +0.0079 | 0.0051 | +1.56 |
| SUF | +0.0003 | −0.0032 | −0.0035 | 0.0030 | −1.14 |
| SUB | +0.0576 | +0.0651 | +0.0075 | 0.0077 | +0.98 |
| PTH | +0.0787 | +0.0834 | +0.0047 | 0.0053 | +0.89 |
| FULL | +0.3991 | +0.3995 | +0.0005 | 0.0011 | +0.44 |

Every gated |t| < 1.6. **delta has no statistically meaningful effect on any
gated field in either class.**

> **RETRACTION.** The n=1200 analysis reported that delta caused "a small
> consistent worsening of PTH, +0.0106 in benign at t = 2.2." At n=8000 the
> benign difference is **−0.0039 (t = −0.81)** and the phishing difference
> **+0.0047 (t = +0.89)** — opposite signs, neither significant. An effect that
> reverses direction when measured better was noise. The earlier claim is
> withdrawn, not refined. Section 29.1's narrower statement — that delta's
> PTH-neutrality was *not established* by the official instrument — is now
> superseded: it is established, and delta is neutral.

### 30.8 A mechanistic observation, offered as a hypothesis only

Field emission ratios (generated / real) fall monotonically with field depth, in
both classes independently:

| | SUB | PTH | QRY |
|---|---:|---:|---:|
| benign, delta=0 | 0.915 | 0.838 | 0.388 |
| phishing, delta=0 | 0.980 | 0.789 | 0.293 |

And delta, which is scoped to the **path** EOS, moved the field *downstream* of
its target far more than the target itself:

| | PTH ratio | QRY ratio |
|---|---|---|
| benign | 0.838 -> 0.842 | 0.388 -> **0.439** |
| phishing | 0.789 -> **0.771** | 0.293 -> **0.384** |

delta suppresses EOS *within* the path body, so it lengthens paths that already
exist rather than causing paths to exist — which is why its effect on PTH
*presence* is nil, and negative in phishing. The extra length appears to carry the
sequence further right, where more queries are reached.

> This is recorded as **evidence consistent with a premature-termination,
> rightward-compounding hypothesis. It is NOT proof.** No causal test has been
> run. In particular it does not establish that EOS is the cause, and a prior
> prediction made from this shape — that delta would raise PTH's ratio and leave
> QRY's flat — was **wrong on both legs**, which is itself a caution against
> treating the picture as settled.

### 30.9 Frozen status

**E3 = NOT PASSED.**

Confirmed defective field-class cells, consistent across both decode
configurations:

    benign    PTH     FAILS      t = +9.12 / +7.16
    benign    QRY     TELL       pooled, all four configurations
    phishing  SUB     FAILS      t = +3.66 at delta=0 (adopted t=2.20 not counted)
    phishing  PTH     FAILS      t = +4.63 / +16.20
    phishing  QRY     TELL       pooled, at the corpus cap

Clearing: benign LBL, benign SUF, benign SUB, phishing LBL, phishing SUF.

> These are **five failed field-class cells, not five independent causal
> mechanisms.** Causal attribution across them remains open and nothing here
> licenses treating them as separate defects with separate fixes.

**No cell is UNMEASURED.** That is what this run bought: every field in both
classes now carries a verdict, including the two that had never been measured at
all.

Standing conclusions:

- **PTH attribution work is CLOSED** (PTH_ATTRIBUTION.md, frozen). Re-measurement
  raised PTH's excess — phishing PTH rose from +0.0595 at n=371 to +0.0787 at
  n=1200 — but this does not reopen attribution.
- **The delta intervention branch is CLOSED.** delta is neutral on the gate;
  it is retained for the structural reason it was adopted, not for gate benefit.
- **QRY is now the highest-priority characterisation target.**
- **No generator intervention has yet been justified.**
- **No threshold change is justified.**
- **No claim of human indistinguishability is justified.** The human panel has
  never been run, and whole-URL AUC remains 0.726 benign / 0.899 phishing.

### 30.10 Provenance

- `n_main` = 8000, G = 5 generations, D = 3 split seeds, 15 draws per field.
- Both configurations share generation and split seeds, so every delta comparison
  in 30.7 is paired.
- Both runs exited 0; wall clock 23,780 s and 25,165 s, run concurrently.
- Full test suite passed before and after (`PYTEST_EXIT=0`, 450 tests).
- Evaluator changes committed as `6f0d43c`; `git diff` against the pre-run
  baseline touches only `uie/corpus/run_eval.py`, additions only.
- `uie/model.py`, `uie/train.py`, `uie/evaluation.py`, `uie/gate.py`,
  `uie/suffixes.py` and `data/best_e5.pt` are byte-identical to the baseline.

### 30.11 What section 29 got wrong

Section 29 is preserved as written. Three of its statements are superseded here
and should not be cited going forward:

- **29.1** held that delta's PTH-neutrality was not established by the official
  instrument. It is now established (30.7); delta is neutral.
- **29.2** called QRY "unmeasured … a coverage failure wearing a measurement
  failure's clothes." The coverage half was right and understated; the
  measurement half is now resolved, and QRY is the model's largest tell (30.6).
- **29.3** called phishing SUB "the largest single gate failure and the least
  investigated," and named the PTH focus a misallocation. That rested on a
  single draw of +0.1099, which the 15-draw distribution later placed at the
  **100th percentile** — the literal maximum. Phishing SUB's calibrated mean is
  +0.0576. The misallocation claim is withdrawn; the genuine misallocation was
  toward PTH and away from **QRY**, which no prior section measured at all.
