# D1 — the typicality scorer, tested in LIMNEX: plan (DRAFT, for the owner's review)

Nothing here runs before the owner approves. `e16_b5` and LIMNEX's served v3.7 stay unchanged
unless a gate below says otherwise and the owner adopts.

## 0. In plain words

- UIE's model can say how *normal* a URL looks, separately for benign and for phishing:
  `log P(url | benign)` and `log P(url | phishing)`. Their difference is a phishing score that
  comes from a model trained on different data (the web itself and the OpenPhish feed) than
  anything LIMNEX has.
- **D1 has two parts.** First, UIE gets a clean public scoring function (UIE's own deliverable).
  Then LIMNEX, as an outside user, tests that score as a fifth ensemble member against the
  frozen `benchmark_v2` — and keeps it **only if it wins**, as LIMNEX's own roadmap requires
  (`docs/Project_Context.md`, the UIE integration point).
- **Honest odds: about 1 in 3** that it wins. The score shares the engine's word-knowledge gap
  (rare real words look unusual to it), and LIMNEX's four members are already strong.

## 1. D1a — UIE's public scorer (UIE repo)

**What:** a new module `uie/typicality.py`; no existing `uie/` file changes.

```python
scorer = TypicalityScorer.load("data/e16/train/e16_cap/final.pt")   # e16_b5's weights
scorer.log_prob(urls, "benign")       # nats per URL, one number each
scorer.llr(urls)                      # log P(url|phishing) - log P(url|benign): higher = more phishing-like
```

**Fixed choices** (made now, before any score is seen):

| choice | value | why |
|---|---|---|
| weights | `e16_cap`'s `final.pt` (`98b10b46…`) = `e16_b5` | the adopted engine; the decode rules only affect generation |
| input | scheme stripped and normalised exactly as the gate does (`FIELDS["FULL"]`) | what the model was trained on |
| frame | `domain`, the frame generation uses | it is the distribution the engine samples from |
| plan tokens | read from the URL itself (does it have a path, a query), as in training | `log P` is then the probability of the whole encoded URL |
| long URLs | the first `max_len` tokens are scored and the URL is flagged; the same tokens for both classes, so the difference stays fair | the model never saw longer rows |
| internationalised (IDN) hosts | scored as punycode, flagged | LIMNEX decides how members treat IDNs (§2) |

**Tests:** equal to `URLTransformer.sequence_log_prob` on the same tokens; batch-size
invariant; deterministic; normalisation; long and IDN URLs flagged; and **the independence
check** — the scorer runs from an outside script that imports only `uie.typicality`.

**D1a's gate (`DESIGN.md` §9):** an external consumer can score URLs without importing anything
internal. Plus a short model card: what the score means, and its known limit (rare real words
look atypical).

**Cost:** about a day on the laptop. No training.

## 2. D1b — the LIMNEX test (registered in LIMNEX's repo, by LIMNEX's rules)

**The member:** `uie = scorer.llr(url)`. LIMNEX's fusion is rank-based, so no calibration is
needed.

**Scores needed:**
- LIMNEX's validation fold — **255,690 URLs** (`cache/cnn/train_urls.csv`, `is_val`), in the
  exact order of `cache/v3.7/member_val.npz`, which the fusion is fitted on;
- `benchmark_v2` — **104,514 URLs**, the frozen external test.
- That is about 720,000 scorings (two classes each). **Laptop CPU: about 4 hours** at low
  priority with the RAM alarm (inference only), **or Kaggle GPU: about 15 minutes** in one
  ready notebook. Scores are cached once and never recomputed.

**Leakage audit, before any score is read.** UIE trained on Common Crawl benign URLs and on
OpenPhish phishing (2024-10 to 2026-08); `benchmark_v2` is PhreshPhish. A URL UIE trained on
would look "typical" because it was memorised, not because the score works. So, measured on
both the validation fold and the benchmark: exact-URL, host and registrable-domain overlap with
UIE's training rows (`data/bundle2` train streams + E15's stream). From that:
- a **UIE-disjoint slice** (no host overlap with UIE's training) is defined on both;
- the fusion weights are fitted on the UIE-disjoint part of the validation fold, so a memorised
  URL cannot buy the member weight;
- the gate must hold on the benchmark's UIE-disjoint slice as well (below).

**Fusion: identical methodology, five members.** `fit_rank_pauc` (pAUC@2%) + λ = 0.5 shrinkage
(now towards 1/5) + isotonic + the 5%-FPR threshold, as `build_serving_bundle_v36.py`. IDN
handling: the UIE member abstains on IDN hosts like the deep members (`idn_abstain_members`), so
the IDN gate is not touched.

**Validity check first:** the same code, run with the four v3.7 members only, must reproduce
v3.7's published weights and `benchmark_v2` numbers exactly. Otherwise nothing is read.

**The gate (kill criterion, fixed now):**

| condition | rule | source |
|---|---|---|
| no regression | `gate_v36.py` against v3.7: ROC ≥ −0.005, recall@5%FPR ≥ −0.01, P1 flip 0, homograph recall, real PoCs, legit-IDN FPR | LIMNEX's existing gate |
| **it wins** | recall@1%FPR or recall@5%FPR on `benchmark_v2` improves, with a paired bootstrap 95% CI above 0 (2,000 resamples, seeded) | the integration point: "improve external recall at low FPR" |
| not memorisation | the same improvement holds on the benchmark's UIE-disjoint slice | the leakage audit |

**All three pass → WIN:** a v3.8 candidate, with its cost reported (about 100 MB more memory,
two forward passes per URL, the explanation engine showing a fifth member). Adoption is then the
owner's call. **Anything fails → KILLED:** recorded, nothing ships, v3.7 stays, as P3 killed
stacking.

**Reported, never deciding:** the member's own ROC and recalls on `benchmark_v2`; the weight
it gets; results on the domain-disjoint slice; where it helps (which URLs it moves).

## 3. Predictions [HYPOTHESIS]

| reading | prediction |
|---|---|
| UIE member alone on `benchmark_v2` | ROC 0.75–0.85, weaker than every LIMNEX member |
| overlap with UIE's training | small on the benchmark's exact URLs; noticeable on phishing hosts and benign registrable domains |
| the gate | WIN about 30%; KILLED about 70% |

Why it could still win: it is the only member that was not trained on LIMNEX's corpus, so its
mistakes differ from theirs — the thing ensembles reward.

## 4. Steps and approvals

1. **Owner approves this plan.**
2. D1a: the scorer, tests, model card → owner reviews → commit (UIE repo).
3. D1b registration written in LIMNEX's repo (the gate above, verbatim) → owner approves →
   commit.
4. Leakage audit and scoring (laptop or Kaggle, the owner picks) → validity check → the gate,
   read once → results note in both repos.
5. If WIN: v3.8 is the owner's decision. If KILLED: closed, recorded.

**Total:** about 3–4 working days, mostly the laptop; no GPU unless the owner prefers Kaggle
for the scoring.

## 5. Decisions for the owner

1. Approve the plan (or change the gate).
2. Scoring on the laptop (about 4 hours, low priority) or Kaggle (one notebook, about 15 min).
3. Confirm the D1b registration lives in LIMNEX's repo (UIE stays independent: it is never
   graded by LIMNEX; LIMNEX tests UIE as a consumer).
