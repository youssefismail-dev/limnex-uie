# E11 roadmap — presence and the root slash together (DRAFT, not a registration)

> **Superseded (2026-09-29).** The owner decided §6's four questions
> (`DECISION_2026-09-29.md` §7), and E11 is registered in `PREREG_E11.md`. This draft
> is kept as the planning record.

- **Status:** a planning draft, 2026-09-29, written after PC1's reading
  (`PC1_RESULTS.md`) and before E10_generate's readout (`PREREG_E10.md`).
- **E11 is registered only after:**
  - E10_generate's readout;
  - the owner's decisions in §6.
- **Built on:** the owner's decisions of 2026-09-29 (`DECISION_2026-09-29.md`).
- **Nothing here is claimed as evidence.** Every result cited is tagged with its
  source.

## 0. In plain words

- **What E11 fixes: URL structure.**
  - how often a URL has a path or a query;
  - the "/" after a bare host, which our tokenizer cannot write today.
- **What it does not fix: content.** What queries and paths look like is E12's
  question. It is where E3's convicted control cells are.
- **What PC1 changed.** The plan's benign path overshoot is mostly a training
  artefact.
  - Gradient clipping discounts the shortest training rows, which are mostly bare
    hosts.
  - An objective repair should remove it, and would apply to every arm.
  - The query shortfall is a different problem, and the repair would not fix it.
- **The hard part is now queries.**
  - On current evidence, neither way of letting the model decide presence reaches
    the query tolerance:
    - the hidden controls: 0.69 of the rate;
    - the plan: about 0.87 short after the repair.
  - A plan supplied from the data reaches it by construction, if honouring holds.
    Whether that counts as native is the owner's call (§6).
- **How it is judged.** Presence rates must meet the detectability tolerance, and
  nothing E3's gate measures may get worse.

## 1. Fixed by the owner's decisions (2026-09-29)

| decision | consequence for E11 |
|---|---|
| detectability tolerance (§2 of the decision) | every presence criterion is \|q − p\| ≤ T, with T at one gate generation (8,000 URLs) |
| E3 needs the root slash | `<RS>` is in every arm; its rate is a criterion, in free mode and in control mode's path-off states |
| R3 LOCKED, last resort; R2 REJECTED | no arm changes model size or training length |
| content is E12's | E11's content cells are no-regression guards, not targets |

**Targets at the tolerance.** Free mode, shipped-weighted, domain frame, all
`name_train` rows, as K9b:

| outcome | benign: real ± T | phishing: real ± T |
|---|---|---|
| path | 73.37% ± 0.99 | 39.51% ± 1.09 |
| query | 6.62% ± 0.56 | 4.68% ± 0.47 |
| root slash on bare hosts | 97.9%, T at n ≈ 2,000 (about ±0.64) | 99.9%, T at n ≈ 4,700 (about ±0.09) |
| root slash before a host-route query | 99.6%, T at its n | 100%, T at its n |

The registration recomputes the slash rows on its exact populations.

**Where things stand:**

| model | benign path | phishing path | query (benign / phishing) | slash |
|---|---|---|---|---|
| `e9_ln` / raw, generated [CONFIRMED, `E9_RESULTS.md`] | 75.58%, +2.21, outside | 38.20%, −1.31, outside | 4.58% / 2.95%, outside | 0% |
| `e10_plan`'s own plan [CONFIRMED, K10a] | 77.10%, +3.73, outside | 39.83%, +0.32, inside | 6.05% / 4.03%, outside | — |

- The plan row is its own probability, not generated emission. That comes with
  E10_generate.
- Its gaps are against the same all-row targets. On rows that fit, as K10a and PC1
  read them, benign is +3.77.

## 2. What we know going in

**The query-gap investigation** [CONFIRMED, `research/qgap/`]:
- **QG1.** The query shortfall is the model's query-start decision, worst right
  after the host (0.539 / 0.592 of the data's rate).
- **QG2-D.** Told whether a query exists, the model writes one at the data's rate
  (0.98 / 1.01).
- **QG3.** 97–98% of the hidden-control deficit is the hidden state, not rarity.

**PC1** [CONFIRMED, `PC1_RESULTS.md`]:
- **Clipping discounts the shortest rows.** At the end of training, gradient
  clipping at 1.0 cut every batch of the shortest rows (under 19 scored tokens per
  row): mean weight 0.66. Nothing over 23.5 was cut.
  - Of benign one-URL-per-site rows, 88% of those without a path are short, against
    12% of those with one.
- **CLIP-LED.** Weighted as training effectively weighted it, the data says
  76.45%; the plan says 77.10%, the plain data 73.33%. Clipping explains 83% of
  the gap.
  - Not the frame, not rare suffixes (COMMON-LED). The off-slot mass is the
    suffix's own continuation (LEGAL-EXPLAINED).
- **The plan's query shortfall sits after a path** (Q-PATH). Clipping pushes
  queries up, not down, so the deficit is real and its cause is open.
- **Phishing's plan sits 1.02 points below its clip-weighted rate.** Unexplained.
- **Projections** [HYPOTHESIS, `PC1_RESULTS.md` §6]. With the clipping repaired, the
  plan's paths land at:
  - benign about 73.98% (+0.61, inside);
  - phishing about 38.39% (−1.12, at the edge).

  Its queries fall a little, to about 5.75% / 3.81%, both outside.

**E10_generate** [pending]:
- K10b, the emission;
- K10c, no regression of content;
- K10d, novelty and copying;
- honouring, which option C depends on;
- phishing QRY's measurability;
- `e10_base` against `e9_ln`, the run-to-run spread S for the tolerance's edge
  flag.

## 3. The components

### 3.1 The root slash (every arm)

- **The token.** `<RS>` already exists: the tokenizer flag `root_slash`, written
  for E8, with E8's tests. E8's results stay unread.
  - It is written where `<PTH>` would be when the path is the bare `/`.
  - It is shape: always trained, and it ends the path field.
  - `<PTH>` keeps meaning a non-empty path, so every presence definition, control
    and gate reading is unchanged.
- **What changes in the sequence.** At a host end the model now chooses among
  `<PTH>`, `<RS>`, `<QRY>`, `<FRG>` and the end of the URL. A host-route query
  becomes `<RS> <QRY>` in 99.6–100% of real URLs.
  - This is where the query-start deficit is worst (QG1).
  - With hidden controls, the new option also competes with `<PTH>`, so it can move
    path presence. E11 measures it.
- **The start.**
  - `<RS>` makes the vocabulary 108 tokens, so the seeded start changes.
  - Proposed engineering: build the 108-token start from the 107-token seeded
    start. Every existing row and weight is copied; only the new row is drawn,
    from a separate seeded stream.
  - V-id checks that every shared weight equals E10's start (`26926388…`,
    reproduced on Kaggle's torch 2.10).
  - E11's arms are then comparable with E10's, with one new row as the only start
    difference.

### 3.2 The objective repair (licensed by PC1's CLIP-LED; the owner approves)

- **The defect** [CONFIRMED].
  - E9's objective gives every scored token the same weight, but the gradient is
    clipped before E9's rescaling.
  - At the end of training the shortest batches have norms 1.1–2.2 and are all cut
    at 1.0. Every other batch stays under 0.9.
  - So the clip is not catching spikes. It is a steady discount on short rows.
- **The repair.**
  - Keep a clip for warm-up protection, but at a level that does not bind by length
    after warm-up.
  - PC1 puts the late maximum at 2.16, so a level near 3.0 fits. The registration
    fixes it.
  - This is a training setting, not a gate threshold.
- **The registered check:** in the new run's clipping record, at most 1% of steps
  clipped in every length band after warm-up.
- **The expected effect.**
  - Benign path presence falls by about 2–3 points in both mechanisms.
  - Queries fall a little (PC1: +0.29 / +0.21 of today's rate is clipping's push).
  - Other short-row decisions move too, such as ending a URL right after the host
    (QG2-D 0.908). The no-regression guard covers content.
- **The alternative,** if a higher level proves unstable: batches built to a fixed
  token count instead of 256 rows. It is a larger change, held in reserve.

### 3.3 How presence is decided

| option | what it is | for | against / open |
|---|---|---|---|
| **A. hidden + joint dropout** | the controls are hidden together or shown together, never one alone | Generation never uses a partial state (free: both hidden; control: both shown). Joint dropout doubles the exposure of both states in use (25% → 50% of rows each), giving up only the partial states. It is the registered next question after K10a's failure (the hidden state's exposure) | Whether exposure is what the hidden state lacks is unknown. QG3 located the deficit, not its cause. `e9_ln`'s queries are at 0.69 of the rate, a long way to go |
| **B. the plan** (model-filled) | E10's `e10_plan` | an explicit decision; after the repair its paths are projected inside the tolerance (benign) or at the edge (phishing) | its query decision after a path is short (0.886 / 0.852 of the rate training aimed at), cause open; E10_generate says whether it helps or hurts content and honouring |
| **C. supplied plan** | free mode draws the plan from the data's per-suffix table (domain frame), as it draws the suffix; the model writes the URL to it | presence matches the data by construction, times honouring. It is the only option projected to reach the query tolerance. It needs no extra training: it is B's checkpoint, generated differently | It is not resampling, rejection, masking, a second prediction or a hybrid: it is the native conditioning control mode already uses. But the model no longer decides presence in free mode, and whether that is "native" enough is the **owner's ruling**. It also needs honouring near the data's rates: phishing's query states were at 87–90% under E9, and E10_generate reports the plan's |

### 3.4 Draft arm set (the owner chooses in §6)

**Recommended: four arms, one session, one shared start.**

| arm | recipe | what it isolates |
|---|---|---|
| `e11_rs` | the baseline recipe + `<RS>` | the slash alone, against `e10_base` (shared start) |
| `e11_ref` | `e11_rs` + the repair | the repair alone, against `e11_rs` |
| `e11_joint` | `e11_ref` + joint dropout | hidden-state exposure, against `e11_ref` |
| `e11_plan` | the plan + `<RS>` + the repair | the plan against the hidden controls. Generated both as B (its own plan) and, if the owner rules it native, as C (the data's plan) |

- **Why four, not three.** The repair changes the objective every later arm
  inherits, and the slash changes the host-end decision. Both move path presence.
  Without `e11_rs` their effects cannot be told apart.
- **The three-arm variant** drops `e11_rs` and saves about 1.2 h, at the price of
  that attribution.

**How E10_generate's results change this:**

| E10_generate shows | E11 |
|---|---|
| the plan hurts content (K10c fails) or honouring | `e11_plan` stays as the mechanism check only; the default becomes A |
| the plan helps content or honouring | `e11_plan` is the lead arm |
| URLs written to a supplied plan are realistic (control cells no worse than `e10_base`'s) and honouring is near the data's | C is a real candidate, if the owner rules it native |
| `e10_base` against `e9_ln`: the run-to-run spread | S for the tolerance's edge flag |

### 3.5 Draft criteria (with the tolerance; fixed in the registration)

| code | criterion |
|---|---|
| K11a presence | free mode: PTH and QRY within T of the real rate, both classes, per arm (and per generation mode for `e11_plan`) |
| K11b root slash | the slash rate within T on bare hosts and before host-route queries: free mode, both classes, and control's path-off states |
| K11c no regression | against each arm's reference (§3.4): no gated cell's verdict worsens with x rising > 0.0141 (the registered rule), free and control |
| K11d guardrails | novelty and query copying, as K10d |
| K11e the repair | at most 1% of steps clipped per length band after warm-up; the path shift in the predicted direction |
| K11f honouring | reported against the objective's 0.98 (K1a), separately from realism |
| reported | FULL (auxiliary), expected to fall sharply with the slash; QG1's decomposition for the winner |

## 4. Honest expectations

- **Likely:**
  - the slash rate meets its tolerance, since it is a common outcome and QG3 found
    uncontrolled outcomes calibrated;
  - FULL falls sharply;
  - with the repair, the plan's benign paths come inside the tolerance (projected
    +0.61).
- **Uncertain:**
  - **The hidden controls' benign paths after the repair.** At a real host end,
    `e9_ln`'s +3.12 points of extra path is made of two parts (QG2-D):
    - the end of the URL under-predicted (−2.44), which is short-row territory,
      where the repair should act;
    - the host-route query under-predicted (−0.68), the hidden-state deficit, which
      it will not touch.
  - phishing paths after the repair (projected at the edge);
  - whether joint dropout lifts the hidden-state queries meaningfully.
- **Unlikely without C:** queries inside the tolerance in both classes.
  - If the owner rules C out, E11 will probably meet paths and the slash but not
    queries.
  - The query decision then needs its own registered question. For the plan, the
    after-path deficit.

## 5. After E11 (outline only)

- **E12, content.**
  - Re-measure N2a's registered query-content diagnostic on E11's winner.
  - N2a read SPARSITY on the old objective, and E9 showed that objective distorted
    query learning. The clipping repair changes it again.
  - Then the licensed lever:
    - R1, the 256-token limit, which drops 41–45% of rows with long query values;
    - or N2b, the query-content weight.
  - Target: the convicted control cells.
- **E13:** combine the winners, then N3, the E3 reading, with the slash condition.
- **Capacity (R3)** only under the decision's §4.

## 6. The owner's decisions before E11's registration

1. **The repair** (§3.2). Approve changing the training's clip level after warm-up,
   with the registered check.
2. **Option C** (§3.3). Does a plan supplied from the data count as native for free
   mode? It decides whether E11 can aim at the query tolerance now.
3. **Four arms or three** (§3.4).
4. **The shared start** (§3.1). Approve the 108-token start built from the 107-token
   one.

## 7. Not in E11

- content repairs;
- gate changes;
- decode-time fixes, resampling, rejection or masking;
- capacity or longer training.
