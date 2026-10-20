# E7 — explicit field-presence control: pre-registration

**Registered:** 2026-09-23, before any E7 training. Developed on branch
`e7-field-presence` in a separate worktree, so the gate-v2 baseline — which runs
from `master` — cannot pick up unverified code.

**Why "E7":** E6 is taken — the factorised output head, REJECTED (E3_REPORT
sections 21 and 27.3), whose checkpoint `best_e6.pt` is still on disk. The
numbering names checkpoints, so it is load-bearing (DESIGN, "Naming").

**Origin:** the reviewer suggestion in `ARCHITECTURE_DECISION.md` section 9 —
structural control inside the single shared model — approved by the owner for
implementation.

---

## 1. What E7 changes — one variable

- **Six control tokens**, `<P:0> <P:1> <P:?>` and `<Q:0> <Q:1> <Q:?>`, appended
  after the existing controls:
  `[BOS, <C:class>, <F:frame>, (<L:lang>), <P:*>, <Q:*>]`.
- **Presence is the tokenizer's own definition**, which is also the gate's:
  PTH is present iff the path body is non-empty; QRY iff the query is non-empty.
  It is derived from each URL — no labels, no new data. The frozen `bundle2`
  streams are unchanged. This is DESIGN section 4.5's rule: controls must be
  "free or structurally derivable".
- **Control dropout.** In training, each control is independently replaced by
  `?` with **p = 0.5**, re-drawn every time a row is batched. One model then learns
  both the conditional P(url | class, frame, P, Q) and the marginal
  P(url | class, frame) — the unconditional mode.
- **Loss: unchanged.** `field_mask` already gives weight 0 to every target before
  the first field marker, so the controls are inputs only; the model is never
  trained to predict them.
- **Generation:** `generate(..., fields={"PTH": True|False|None, "QRY": ...})`.
  A missing key or `None` means `?`. A checkpoint without the tokens (E5)
  **refuses** a `fields` request rather than ignoring it. The gate code is
  unchanged and scores E7 in unconditional mode automatically.

## 2. Held fixed — the E5 recipe

steps 20,000 · batch 256 · lr 3e-4 · warmup 500 · d_model 384 · 8 layers ·
8 heads · d_ff 1536 · max_len 256 · frame on · class_alpha 1.0 ·
boundary_weight 1.0 · seed 20260803 · not factorised · the same `bundle2` streams.

**Checkpoint selection:** `best.pt` on the mean per-class **unconditional**
validation loss (controls shown as `?`). That is the same prediction task, on the
same targets, as E5's validation loss, so the two are directly comparable. The
conditioned validation loss is reported beside it.

## 3. Known costs, stated in advance

- **+2 tokens per row**, so a few more rows exceed `max_len` 256. When E5 added
  one token (the frame), 99 more rows were dropped; expect about +200 of 2.78M.
  They are the longest URLs — the tail. The actual count is reported from the
  training log.
- **A new checkpoint** carries retraining noise: +-0.0141 at identical
  configuration (measured under the v1 estimator).
- **Vocabulary 101 -> 107.** E5 is unaffected: the flag defaults off, and
  `from_state_dict` defaults it off for older checkpoints.

## 4. Hypotheses and registered tests

### C1 — Controllability (PRIMARY)

For each class and each control value — P:1, P:0, Q:1, Q:0, with the other
control `?` — generate n = 4,000 under the adopted decoder (temperature 1.0) and
delta, and measure the presence rate of the field.

- **PASS:** honoring **>= 0.98 in all 8 cells** (2 classes x 4 values).
- The four joint settings (P:1/Q:1, P:1/Q:0, P:0/Q:1, P:0/Q:0) are reported per
  class, as descriptive results.
- **FAIL** in any cell: E7 is not adopted as a control. Hard decode-time
  constraints are then a follow-up, not part of E7.

*Why 0.98:* in training, every `<Q:1>` row carries a query and every `<Q:0>` row
does not, so the controls are deterministic. A well-trained model should honour
them almost always; residual failures would come from generation ending first.

### C2 — The unconditional mode must not regress (SAFETY)

Gate v2 in its canonical configuration (G=5, D=10, n_main 8000, temperature 1.0,
adopted delta) on E7 in unconditional mode, against gate v2 on E5
(`data/gate_v2_baseline*`).

- **KILL** if any gated cell's section-26.1 verdict worsens — CLEARED to
  UNDECIDED or CONVICTED, or UNDECIDED to CONVICTED — **and** its x rises by more
  than **0.0141**, the measured retraining noise.
- A verdict that worsens with a rise of 0.0141 or less is reported as **not
  attributable**: retraining alone could produce it.
- Unconditional validation NLL is reported against E5's; descriptive, not a gate.

### C3 — Is conditioning consistent? (SECONDARY, within one checkpoint)

On the same E7 checkpoint — so no retraining noise — compare QRY content excess
(`discriminate_with_floor` against the real query-bearing pool, n = 400 per arm,
D = 12 split seeds, as in Q3.1) between:

- **(a)** `<Q:1>`-conditioned generation, and
- **(b)** the query-bearing rows of unconditional (`?`) generation, at equal n.

Per class:

- **|a - b| < 0.02 with the CI covering 0:** conditioning samples the
  query-bearing conditional consistently. Then `<Q:1>` generation is a valid and
  cheap way to measure the QRY cell at full n, which bears on phishing QRY's
  `MIN_N` problem. Using it in the canonical gate would be a separate decision.
- **A significant gap of 0.02 or more:** conditioning changes the queries
  themselves; reported with its direction.
- **Registered confound:** both arms draw suffixes from the class's marginal
  table, but in (b) the model's per-suffix query propensity tilts which suffixes
  end up carrying queries. The suffix mixes of the two arms are reported (total
  variation distance) so this can be read.
- **No directional prediction on content.** PQ1 and Q3 found the query tell is
  shape, which presence conditioning does not obviously touch.

### C4 — Descriptive

Emission rates (PTH, QRY, SUB) in unconditional mode, E7 against E5.

## 5. Adoption

**Not pre-authorised.** E7 can replace E5 as the production checkpoint only if C1
passes and C2 shows no attributable regression — and even then the decision is
the owner's.

## 6. Kaggle execution

- **Code travels separately.** The `uie` package ships as a small Kaggle dataset,
  `uie-e7-code`, with a manifest carrying the git commit, attached alongside the
  existing `uie-corpus2` data bundle. The data is not re-uploaded.
- **The notebook** (`research/e7/E7_train.ipynb`) puts the code dataset first on
  `sys.path` and verifies the tokenizer supports fields **before** training. The
  data bundle carries its own older `uie`, which must not shadow the new code.
- **Outputs:** downloads are renamed `best_e7.pt` / `final_e7.pt`; `best_e5.pt` is
  never overwritten.
- **Expected time:** about 1.1 h on a T4 or P100 (R2's 40,000 steps took
  7,461 s).

**Amendment, 2026-09-23 — harness only.** Kaggle attempt 1 stopped in cell 2
before any training: the notebook imported `uie` before installing
`tldextract`. Fixed by installing first, **pinned to 5.3.1**, the version the
local evaluation parses with. Measured: 5.3.1 and 5.3.2 (what an unpinned
install resolves to today) ship **byte-identical** Public Suffix List snapshots
(SHA-256 `b69315c085d53972…`, 6,871 ICANN rules each), so the pin changes no
host split. The versions E5's and R2's Kaggle runs used were not recorded
(Unknown); E7 now records tldextract, torch, Python and the code commit in
`e7_result.json`. No change to the design, the data or the code dataset.

**Amendment, 2026-09-23 — where C2's generations are made (operational).**
Registered before any C2 generation exists. To save ~4 h of laptop CPU while C1
runs, C2's 10 generations are produced on a Kaggle GPU by
`E7_C2_generate.ipynb`, which runs the gate's own loop (`generate_with_suffixes`,
seed 20260806, n_main 8000, G=5, `temperature` 1.0, adopted delta, unconditional
mode) and names each file with the gate's cache key for `data/best_e7.pt`. Only
the forward pass runs on the GPU: `generate()` samples on the CPU with a seeded
CPU generator, so the files should equal the laptop's generations except where
floating-point differences flip a draw. **Before the gate reads them**,
`c2_gpu_verify.py` checks every key, count, hash and input, and regenerates at
least 400 URLs per class on the laptop CPU for a byte-identity rate, reported
with a 50% gross-mismatch floor. Scoring stays local, on the instrument that
scored E5. The generations are used **only if C1 passes**; otherwise they are
discarded unread.

## 7. Out of scope

Any gate change, including using `<Q:1>` generation in the canonical gate; hard
decode-time constraints; other structural controls (SUB, FRG, length,
path_depth); LIMENX; D1.

## Status

    E7  registered (11ac503); implemented, E5 path verified bit-identical
        (974788f); Kaggle package (7405606)
        attempt 1 failed before training (harness, see section 6); notebook
        fixed (d8b3b5d); trained on Kaggle, 70 min
        checkpoint data/best_e7.pt, sha256 916d03857d7d3db4...
        C1 FAIL, 2026-09-24: 3 of 8 cells below 0.98 -> C1_RESULTS.md
        NOT ADOPTED as a control. C2 / C3 / C4 not run under E7; the Kaggle C2
        generations stay unread for the separately registered HC1
        merged into master as a recorded negative result
