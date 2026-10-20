# HC1 — hard generation-time presence constraints on E7: pre-registration

**Registered:** 2026-09-24, before any HC1 code exists and before the preserved
C2 generations are opened.
**Approved by the owner (2026-09-24):** the follow-up E7's pre-registration names
— enforcing presence at generation time — with EOS masking as the mechanism;
autonomous execution; the preserved Kaggle C2 artifact used only after this
registration.
**Frozen:** `data/best_e7.pt` (sha256 `916d0385…`). E7's own result — **C1
FAIL** (`research/e7/C1_RESULTS.md`) — stands unchanged. E5, R4, gate v2 and
every published conclusion are untouched.

---

## 1. Why

E7 C1 failed — registered rule: >= 0.98 in all 8 cells, n = 4,000 each (`research/e7/C1_RESULTS.md`):

| class | control | honoured (C1) | 95% CI | n |
|---|---|---:|---|---:|
| benign | PTH:1 | 0.9942 | [0.991, 0.996] | 4000 |
| benign | PTH:0 | 0.9920 | [0.989, 0.994] | 4000 |
| benign | **QRY:1** | **0.9640** | [0.958, 0.969] | 4000 |
| benign | QRY:0 | 0.9898 | [0.986, 0.992] | 4000 |
| phishing | **PTH:1** | **0.9752** | [0.970, 0.980] | 4000 |
| phishing | PTH:0 | 0.9882 | [0.984, 0.991] | 4000 |
| phishing | **QRY:1** | **0.7668** | [0.753, 0.780] | 4000 |
| phishing | QRY:0 | 0.9900 | [0.986, 0.993] | 4000 |

These are also H2's reference values (section 4), fixed here before any HC1 URL
exists.

The controls are learned — under `<Q:1>`, 96.4% of benign URLs carry a query,
against 6.4% of real benign URLs and 4.5% of E5's unconditional generations —
but sampling can still end a URL before the required field: 1,162 of the 1,176
failures in the three failing cells ended before reaching it. E7's
pre-registration names the follow-up: enforce presence at generation time.

## 2. What HC1 is — a decoding rule; nothing is trained

**Named HC1, not E8.** `E<n>` names a trained checkpoint (DESIGN, "Naming");
HC1 trains nothing. It changes only how `best_e7.pt` samples when a presence
control is set.

**Presence is what the gate reads from the URL string** (`uie.evaluation.FIELDS`
on the rendered URL). Two facts fix it:

- **The grammar.** Fields appear in the order `… PRT, PTH, QRY, FRG`, then EOS,
  and a marker is emitted only for a non-empty field.
- **The parser** (`uie.factorization.parse`) starts the fragment at the first
  `#`, the query at the first `?` before it, and the path at the first `/`
  before both; everything earlier is the host.

So a marker alone does not decide presence: a delimiter **character** written
inside another field can create a part with no marker, or swallow one. Found in
the smoke test of this harness, before registration, on an untrained model; the
rules cover both routes. "Host fields" are those rendered before the path:
SCH, USR, SUB, LBL, SUF, PRT.

| control | rules, at every sampled step |
|---|---|
| **P:1** | until the path is present: forbid EOS and the QRY and FRG markers; forbid `?` and `#` in host fields (they would swallow the path); inside PTH with no content yet, forbid EOS, every marker, and `?`/`#` as the first character |
| **P:0** | never the PTH marker; never `/` in a host field |
| **Q:1** | until the query is present: forbid EOS and the FRG marker; forbid `#` in host fields and the path (it would start the fragment first); inside QRY with no content yet, forbid EOS, every marker and `#` as the first character |
| **Q:0** | never the QRY marker; never `?` in a host field or the path |
| `?` / absent | no rule; the unconditional mode is unchanged byte for byte |

Joint settings compose the rules. Tokens the caller supplies (the suffix) are
never sampled, so no rule applies to them. A delimiter absent from the
vocabulary cannot be sampled and needs no rule (E7's has `/` and `?`, not `#`).

**Mechanism — masking.** Forbidden logits are set to −∞ after the adopted delta
and before the `temperature` decoder samples: each token is drawn from the
model's distribution renormalised over the allowed tokens.
`generate(..., fields=..., enforce="mask")`; the default `enforce=None` is
today's generator, **bit-identical** — proven against a fixture of 96
configurations captured from the pre-HC1 code.

## 3. What masking costs, stated in advance

Masking is not the model's conditional distribution. Let m_t be the probability
mass the rules remove at step t. A URL x that never touches a forbidden token has

    q(x) = p(x) / w(x),     w(x) = Π_t (1 − m_t)

so masking **up-weights exactly the URLs the model wanted to end** — the forced
rows. The exact conditional, p(x | rules hold), is what resampling until the
rules hold would produce. Two consequences, both used below:

- **E_q[w] = P_p(rules hold).** The mean weight of a masked sample equals the
  probability that unconstrained sampling obeys the rules — which C1 measured.
- **TV(q, p*) = ½ E_q | 1 − w / E_q[w] |**, estimable from the masked sample
  alone. Expected near 1 − c1: masking re-routes the whole failing mass.

## 4. Tests

### H1 — the rules hold (PRIMARY)

8 cells (2 classes × P:1, P:0, Q:1, Q:0, the other control `?`), n = 1,200
each, through the canonical generation path (`generate_with_suffixes`: marginal
suffix table, frame `domain`, `temperature` 1.0, adopted delta),
`enforce="mask"`. Seed for cell i (C1's order): 20260925 + 10,000·i.

**PASS:** every URL honours its control by the gate's own definition
(`uie.evaluation.FIELDS`), in all 8 cells, **and** no URL reaches `max_len`
(256) unfinished. Any exception is a defect of the rules or their code, reported
as such — or, for truncation, a limit of masking that resampling does not have.

### H2 — the mask is the model's own probability (PRIMARY — implementation)

Per cell, from H1's URLs: the mean of w against C1's honouring rate for the same
cell (same checkpoint, decoder and delta; independent seeds):

    z = (mean w − c1) / sqrt(se_w² + se_c1²)

**PASS:** |z| < 3 in all 8 cells. The identity is exact up to one one-sided
gap: an unconstrained URL can honour its control while touching a forbidden
token (an empty field, or a delimiter that happens to leave the string intact —
C1 shows at most 9 empty queries in 4,000), so E_q[w] <= c1 by at most that
rate. Verified exactly in the unit tests, on a stub where both sides equal 1/4.
A failure means the mask or its accounting does not match the model, and H1's
100% would then say nothing about fidelity.

### H3 — what masking costs against the exact conditional (SECONDARY)

On the two query-on cells, which motivated HC1.

- **The exact arm is generated by rejection, suffix by suffix.** The same seed,
  suffix draws and groups as H1's masked arm; each group is first generated
  without enforcement (seed + i, exactly as the masked arm's group), and every
  URL that does not honour `<Q:1>` is redrawn **with the same suffix** and seed
  seed + i + 1,000,003·k on retry k, until it does (at most 100 retries per
  group; any shortfall is reported). The accepted URLs are exact draws from
  p(x | suffix, rules hold), and the two arms carry **the same suffix mix by
  construction**. (C1's own honouring URLs were considered and rejected as the
  exact arm: rejection tilts their suffix mix by up to 1 − c1 in total
  variation — 0.233 for phishing — which would confound the comparison.)
- **Its acceptance rate is a third, independent estimate** of the probability
  that unconstrained sampling honours the control, beside C1's rate and H2's
  mean w. Reported against both.
- **Distortion, from the weights:** TV(q, p*) with a bootstrap CI (2,000
  resamples of H1's 1,200 URLs), against the prediction 1 − c1; the share of
  rows with w < 0.5.
- **Realism, by the gate's discriminator:** each arm against the real pool on
  the QRY field and on FULL (the field's gate population, one URL per host,
  n = 1,200 per draw), D = 10 split seeds shared by the arms
  (20260925 + 500 + d), so each seed yields a paired difference.
  Δ = x_mask − x_exact, per class and field:
  - **equivalent** if the 95% CI of Δ lies inside (−0.02, +0.02);
  - **distorts** if Δ >= 0.02 and the CI excludes 0;
  - otherwise **unresolved at this n**.
  **Stated limit:** one generation per arm, so the CI omits generation noise
  and is narrower than a full estimate — which favours a decisive reading in
  either direction; it is reported with that caveat.
- Descriptive: mean lengths of LBL, SUB, PTH and QRY in both arms. The algebra
  in section 3 predicts that masking lengthens what comes before the query.

**Consequence:** if masking distorts in any class, **resampling** is the
recommended enforcement — exact, and cheap at these acceptance rates.

### C2 — the unconditional mode must not regress (SAFETY — carried over)

E7's registered C2, verbatim: gate v2 in its canonical configuration (G=5,
D=10, n_main 8000, `temperature` 1.0, adopted delta) on E7 in unconditional
mode, against gate v2 on E5 (`data/gate_v2_baseline_{benign,phishing}/`).
**KILL** if any gated cell's section-26.1 verdict worsens — CLEARED to
UNDECIDED or CONVICTED, or UNDECIDED to CONVICTED — **and** its x rises by more
than **0.0141**; a worse verdict with a rise of 0.0141 or less is **not
attributable**. HC1's rules never touch the unconditional mode, so this is a
property of the checkpoint, needed before E7 could serve as the production
generator.

**Generations:** the preserved Kaggle C2 artifact (`results (4).zip`, 10 files
made by `E7_C2_generate.ipynb`, E7 amendment 2026-09-23), extracted to
`data/gen_cache_e7_gpu/`. **Before the gate reads one file**,
`research/e7/c2_gpu_verify.py` must PASS: every key the local gate computes,
every count and hash, the checkpoint and suffix table, and byte identity of at
least 400 regenerated URLs per class on the laptop CPU (50% gross-mismatch
floor). If it fails, the files are not used and C2 is generated on the laptop.

### C4 — descriptive

Unconditional emission rates (PTH, QRY, SUB), E7 against E5; unconditional
validation NLL against E5's (benign 1.6688 vs 1.6866; phishing 1.5298 vs
1.5249).

## 5. Adoption — not pre-authorised

E7 + HC1 can be recommended as the controllable generator if H1 and H2 pass and
C2 shows no attributable regression, with masking or resampling as H3 decides.
Replacing E5 as the production checkpoint remains the owner's decision.

## 6. Execution, and what is preserved

In order: this registration; the implementation with its tests; C2
(verification, then scoring); H1/H2; H3 (its rejection arm, then scoring);
results. Laptop CPU at below-normal priority, one job at a time. Preserved: every generated URL with its per-row
trace (w, masked steps, truncation), the per-cell and per-test reports, the gate
reports and logs, the verification report, the Kaggle provenance file, seeds and
configurations in the reports, and the checkpoint's hash. The E7 code package
the Kaggle runs used stays at `data/kaggle/uie-e7-code.zip` (sha256 `81f16666…`).

**Amendment, 2026-09-24 — H3 only; no verdict rule changes.** Written after
C2, H1 and H2 completed and before H3's rerun.

1. **Memory (operational).** H3's first run was stopped by the harness's own
   memory guard (resident memory 0.35 → 6.1 GB). H3 calls the discriminator
   directly, without the `gc.collect()` the gate received on 2026-09-23 for the
   same retention. It is added after every call — it frees only unreachable
   objects, so no value can change — and **H3 is rerun in full**, so its report
   comes from one run. The aborted run's files are kept in
   `data/hc1/h3_run1_aborted/`; its benign numbers must reproduce exactly.
2. **A descriptive estimator, corrected.** Section 4 called the exact arm's
   acceptance rate "a third, independent estimate" of the honouring
   probability. It is not: with the suffix held, a slot with honouring rate
   P_s needs 1/P_s draws on average, so accepted/draws is the **harmonic** mean
   of the per-suffix rates — below the marginal rate wherever P_s varies (the
   aborted run: phishing 60.5% against C1's 76.7%). The marginal rate is the
   **first pass** — one unconstrained draw per slot, in the table's suffix mix —
   now recorded and compared with C1 and H2; the accepted/draws rate is
   reported with this reading. Neither quantity enters a verdict.

## 7. Out of scope

Retraining; constraints on E5; hard constraints in the canonical gate; other
fields; LIMENX; any change to E5, R4, gate v2 or a published conclusion.

## Status

    HC1  registered (69862d8); implemented (8d2f91d); H3 amendment (62b7753)
         C2 NO ATTRIBUTABLE REGRESSION; H1 FAIL (truncation only: 15 runaways in
         the phishing field-on cells); H2 PASS; H3 EQUIVALENT (4/4)
         -> HC1_RESULTS.md. Masking as registered NOT recommended for adoption
         (section 5 needs H1); the next step is the owner's decision
