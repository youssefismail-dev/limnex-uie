# QG2-D — where the query-start decision loses its probability: registration

- **Registered:** 2026-09-28, before any QG2-D code or measurement.
- **Licensed by:** QG1's reading, DECISION-LED (`QG1_RESULTS.md`, 1f75f32).
- **Approved by the owner:** the three probes below, laptop only.
- **Rules:** those of `QG_PLAN.md`.
  - These are diagnostics only. Nothing changes the adopted baseline
    (`uie/baseline.py`) or any default, instrument or published result.
  - No reading licenses a fix. A training arm needs the owner's approval.
- **Model:** the baseline `e9_ln`, checked by its weights fingerprint (VQ1 as in
  QG1).

## 0. In plain words

QG1 found that the model, shown a real URL beginning, gives "start a query now"
about a quarter too little probability, and about half the right probability when
a query should begin straight after the site's name. QG2-D asks where that
probability went, and why:
1. **Slices.** Is the model wrong the same way everywhere, or only in free mode's
   slice? This is checked for URLs from multi-page sites, for the same URLs told a
   different frame, and when one of the presence controls is visible.
2. **Instead.** At the moment a real host ends, what does the model predict in
   place of a query: a path, the end of the URL, or more host?
3. **Audit.** Does the training code treat hidden controls or the two frames
   asymmetrically?

## 1. What was known at registration [CONFIRMED, from reading the code during design]

- **Frames** come from the corpus (`uie/train.py` `CORPUS_FRAME`): `name_*` is
  the domain frame, `structure_*` the page frame. Both train with their own frame
  token.
- **Rows are drawn uniformly within a class across both corpora** (`batches`).
  - Benign: 906,214 domain against 1,648,460 page rows, so 65% page.
  - Phishing: 192,652 against 46,491, so 19% page (`qg1_data.json`).
- **Presence controls.** Each is hidden independently with probability 0.5, by a
  draw that does not read the row (`FieldDropout`). So both are hidden on about 25%
  of rows, which is free mode's own state.
- **Loss weights.** Field markers, `<QRY>` included, and the end token are trained
  with weight 1 on every row. Control positions have weight 0 (`field_mask`).
- **The page frame's query rates** are higher overall: 10.32% / 12.28%, plain
  (QG1 §3). Their split by route is not known.

## 2. Instrument

- **The token profile.** At every decision position (QG1's definition, `<LBL>` to
  the first `<QRY>`, `<EOS>` or `<FRG>`), the model's probability is split into
  categories:
  - `<QRY>`, `<PTH>`, `<EOS>`, `<FRG>`, `<PRT>`, `<SUB>`;
  - **content**, any character;
  - **other**, everything else.

  The category of the true next token is recorded too.
- **Regions.** Positions before `<PTH>` form the host region; from `<PTH>` on, the
  path region.
- **Calibration of category c in a region:** Σ p(c) / Σ 1[next = c]. For `<QRY>`
  this is QG1's decision factor on that route.
- **Host-end snapshot.**
  - Take the last host-region position: the one whose true next token is `<PTH>`,
    `<QRY>`, `<EOS>` or `<FRG>`. Record the model's category probabilities there
    and the true outcome.
  - The model's structural split there is its probability over those four tokens,
    renormalised. It is compared with the data's outcome frequencies.
- **Path-end snapshot:** the same at the last path-region position, over `<QRY>`,
  `<EOS>` and `<FRG>`.
- **CIs:** percentile, from 1,000 row-bootstrap draws (seed 20260806).
  - Slices on the same rows are resampled together (paired). Page rows are
    resampled independently.
  - QG2-D uses plain sums, not post-stratification. Its contrasts are within the
    same rows, and S0 below is recomputed as their reference.

## 3. Probe 1 — slices

| slice | rows | frame token | P control | Q control |
|---|---|---|---|---|
| **S0** free mode (reference) | QG1's 40,000 domain training rows per class | domain | `?` | `?` |
| **S1** multi-page sites | 40,000 `structure_train` rows per class (seed 20260928, ≤ 256 tokens) | page | `?` | `?` |
| **S2** frame swapped | S0's rows | **page** | `?` | `?` |
| **S3** path state told | S0's rows | domain | **true value** | `?` |
| **S4** query state told | S0's rows | domain | `?` | **true value** |

**What each slice tests:**
- **S1** tests whether the page frame is calibrated. If the model pools the two
  frames, one frame is over-predicted and the other under-predicted, in opposite
  directions.
- **S2** tests how far the frame token alone moves the decision on the same URLs.
- **S3** tests whether knowing whether a path follows removes the host-route
  deficit.
- **S4** tests, when the model is told a query exists, where it puts it: the route
  split of its query probability, against the data's route split among rows with
  a query.

## 4. Probe 2 — instead of a query

In S0, for each class:
- the per-category calibration in the host region and in the path region: which
  categories carry more probability than the data asks for;
- the host-end and path-end structural splits, model against data.

S1 and S3 are reported alongside.

## 5. Probe 3 — the audit (code and data; registered checklist)

| item | question | how |
|---|---|---|
| A1 | Every training row is encoded under its corpus's frame | code, and a count on the bundle's files |
| A2 | Hiding is independent of presence, of the other control, of frame and of class; the hidden token is free mode's `?` | code; and `FieldDropout` run on 100,000 rows: hidden rates by presence, by frame and by class, with binomial CIs |
| A3 | A training row's control positions are where generation puts them | an exact comparison of the token prefix: `encode(…, hide both)` against `generate`'s prefix, per class |
| A4 | `<QRY>` is a weight-1 target wherever it occurs, in both frames | `field_mask` on sampled rows of both corpora |
| A5 | The presence controls are derived as the fields are opened, on the host route too | `encode` on `host/?q`, `host?q`, `host/p?q` and `host#f`: controls against markers |
| A6 | What free mode's slice is in training: rows by frame × class × control state | arithmetic from the counts and p = 0.5 |

## 6. Predictions, before any measurement [HYPOTHESIS]

- **A2 finds no dependence on presence.** This is known at registration from the
  code; the run confirms it on rows. A1 and A3–A5 find no asymmetry.
- **At the host end in S0,** the model over-predicts `<PTH>` and under-predicts
  `<QRY>`. Reason: free path emission slightly overshoots, 75.58% against a data
  rate of 73.3%.
- **S3 closes at least half of the host-route deficit.** When told whether a path
  follows, the model splits the query and the end of the URL nearly as the data
  does. Confidence is moderate.
- **Pooling across frames: no confident prediction.** It predicts opposite-signed
  deficits in S0 and S1 on the host route, and is more plausible for benign (65%
  page rows) than phishing (19%). The two classes' similar host-route deficits
  (0.539 and 0.592) weigh against it.

## 7. Reading and decision rule

- Host route, benign deciding; phishing reported with its agreement.
- r_X = the host-route `<QRY>` calibration in slice X.
- Closure of S3 = (r_S3 − r_S0) / (1 − r_S0).
- Rows are read in order; the first that holds applies.

| reading | condition | licenses next (registered then; the baseline is unchanged) |
|---|---|---|
| AUDIT-FOUND | an audit item finds an asymmetry that predicts the observed direction | a test of that asymmetry itself |
| FRAME-POOLING | r_S1's 95% CI lies entirely above 1, r_S0's entirely below 1, and the data's host-end query share is lower in the page frame than in the domain frame | a frame-handling question (sampling or conditioning); any training arm needs the owner's approval |
| PATH-COMPETITION | S3's closure ≥ 0.5 with the paired CI of r_S3 − r_S0 excluding 0, **and** at the host end in S0 the model's `<PTH>` share exceeds the data's (CI excluding equality) | the path-versus-query competition at the host end |
| QUERY-STATE | S3's closure < 0.25 **and** r_S1's CI lies entirely below 1 | the hidden-query state's learning (conditioning), with capacity as the alternative (R3 stays LOCKED; the owner decides) |
| UNRESOLVED | anything else | reported; the owner decides |

**The path route** (QG1: 0.806 / 0.780) is reported on the same slices, with its
end snapshot, as a description. No rule reads it.

**Validity** (checked before any number is read):
- **VQ1:** the model's fingerprint.
- **VP1:** S0's host-route and path-route decision ratios reproduce QG1's within
  ±0.05. QG1's were post-stratified, so a small difference is expected.
- **VP2:** at every recorded position, the category probabilities sum to 1 within
  1e-4.
- **VP3:** in S3 and S4, the true controls are encoded as registered. Checked by
  decoding their positions.

## 8. Not claimed

- a fix, or that a mechanism found here is the whole cause;
- anything for other trainings (one training, `e9_ln`);
- anything about query content.

## Status

    QG2-D  registered 2026-09-28 (this document), before any QG2-D code or measurement
    QG2-D  code bfd502d; READ 2026-09-28 (QG2D_RESULTS.md): validity PASS; audit
           no finding; reading QUERY-STATE in both classes. Next stage (QG3) needs
           the owner's approval before registration.
