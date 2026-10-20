# V2-P1b — P1 rerun with the query LayerNorm: registration

- **Registered:** 2026-10-13, with the owner's decisions on T1–T4 (§1.2), before any P1b
  training. The draft (7b2af0f) proposed them; this document fixes them.
- **It rests on `PREREG_V2_P1_MEMORY.md`.** Everything there carries over unless this
  document changes it:
  - the arm's network, data, recipe, start and decode;
  - the 50 jobs and every instrument;
  - **every criterion:** K1, K2, K-used, all guards, Amendment 1's tiered copy rule and the
    binding reading;
  - the stop conditions and the validity checks.
- **No registered criterion changes here.**
- **Frozen:** `e16_b5`, E3, and P1's own record (`V2P1_RESULTS.md`).
- **The arm:** `p1b_mem`.
- **The owner's authorisation (§1.2):** preparation, testing and bundling. The owner runs
  Kaggle.
- **Sources checked for §2:** the memory layer's reference implementation,
  `facebookresearch/XLM`.
  - `xlm/model/memory/memory.py`: `mem_values_optimizer` defaults to `adam,lr=0.001`; the
    input, query and value dropouts default to 0; `mem_query_batchnorm` defaults to False.
  - `xlm/optim.py`: `adam` is Adam with betas (0.9, 0.999), eps 1e-8 and weight decay 0, at
    a constant rate.

## 0. In plain words

- **What broke P1:** the memory's query BatchNorm. In training it normalised each batch with
  that batch's own statistics; in generation it used a running average. Our batches are
  grouped by length, so the engine learned to depend on which URLs shared its batch, and
  generation cannot reproduce that.
- **The fix:** normalise each query by itself (LayerNorm). Training and generation then
  compute exactly the same thing.
- **The audit (§3):** a direct test of the property that broke. It found the BatchNorm leak
  at full size (26 logits) and nothing else.
- **Decided with the fix (§1.2):**
  - T1: no dropout on the memory's output.
  - T2: the memory's values get their own Adam settings: learning rate 1e-3, no weight decay.
  - T4: the audit runs on Kaggle strictly before the first training step, and the notebook
    aborts if it fails.
  - **No tripwire (T3):** the run goes its full 20,000 steps.

## 1. The owner's messages, verbatim

### 1.1 The rerun (2026-10-13)

> We are proceeding with Option 1 (Fix and rerun), but under strict conditions to ensure we
> don't waste another run.
> Formal Preregistration: Write and log a short registration for this fix before any new
> training begins.
> Comprehensive Audit: Double-check the entire architecture and pipeline. I want absolute
> certainty that there are no other batch-dependent leaks (e.g., in other normalization
> layers, loss calculations, or attention mechanisms) that could break generation mode.
> Maximizing the Memory's Potential (Crucial): You noted that even under its batch-dependent
> training conditions, the memory only improved the fit slightly (0.027 nats). Since we are
> doing a rerun, I want to give this architecture its absolute best chance to succeed. Are
> there any other mathematically sound, low-cost architectural or hyperparameter tweaks we
> should bundle with this fix?
> Directive:
> Do not start training yet. Draft the short registration document including the LayerNorm
> fix, confirm your audit for any other leaks, and propose any additional tweaks to
> maximize the memory layer's effectiveness. Present this to me for a final GO before
> executing on Kaggle."

### 1.2 The decisions (2026-10-13)

> P1b decisions and authorization — implementation and bundling only.
> My decisions on T1-T4 are:
> T1 — YES: Disable dropout on the memory output, consistent with the reference
> implementation.
> T2 — YES: Use a separate optimizer for memory values with Adam, learning rate 0.001, and no
> weight decay. Implement the required optional hook inside uie/, with the feature disabled
> by default. Prove that the default-off path is bit-identical to the existing behavior and
> does not alter existing checkpoints.
> T3 — NO: We have tried early tripwires before and found that their problems outweigh their
> benefits. They often prematurely kill models that just have a slow start. Let the training
> run its full course so we can see the complete picture. Do not add the proposed
> early-stopping tripwire.
> T4 — YES: Include the full-scale batch-coupling audit directly inside the Kaggle notebook
> so it runs strictly before the first training step. If the audit fails on Kaggle, the
> notebook should abort automatically before wasting GPU hours.
> Current authorization is limited to preparation, testing, and bundling.
> Next Steps:
> Finalize and commit the P1b preregistration with these exact decisions. Do not silently
> change any registered criterion.
> Wait for your background test suite to fully pass and report the actual result; do not
> describe a still-running suite as passed.
> Build the Kaggle bundle (uie-v2p1b-code.zip) and the notebook.
> Report the final commit hashes, changed files, and estimated compute cost.
> Provide me with the bundle and run sheet so I can execute it on Kaggle.

## 2. What changes from P1

### 2.1 The fix

- **The query normalisation becomes a LayerNorm over each head's 512-dimensional query, per
  token** (`mem_query_norm = "layer"`). It replaces P1's BatchNorm over 2,048 dimensions.
- It keeps no running statistics, so training and generation compute the same function.
- Its affine parameters start at 1 and 0.
- **Everything else in the memory is P1's:** placement after block 4, 262,144 slots, 4 heads,
  top-32, keys and values and their initialisation, float32, padding skipped, the seed
  20261050.
- **Parameters:** 136,318,976 in the memory (P1's minus 3,072), 161,792,000 in all.

### 2.2 The decided tweaks

| # | decision | exactly |
|---|---|---|
| T1 | **YES** | `mem_branch_dropout = False`: the memory's output joins the residual stream without dropout. Every other dropout (0.1) is unchanged |
| T2 | **YES** | See below |
| T3 | **NO** (the owner, §1.2) | No tripwire. The run trains its full 20,000 steps. P1's stop conditions (§6 there) are unchanged and get nothing new |
| T4 | **YES** | See §3.2 |

**T2, exactly:**
- **The values (the 262,144 × 512 table) form their own parameter group** in the training
  optimiser: Adam with betas (0.9, 0.999), eps 1e-8 and **no weight decay**, the reference
  implementation's settings.
- **Their learning rate peaks at 1e-3.** It follows the network's own warmup (500 steps) and
  cosine decay to 0, scaled by 1e-3 / 3e-4.
  - The reference keeps 1e-3 constant. The schedule is kept so that the registered `final.pt`
    is taken with the values settled, not mid-motion, as for every other weight.
- **The implementation.** The training optimiser is AdamW. With weight decay 0, AdamW's
  update is exactly Adam's, which a test proves step for step. The group has its own rate,
  betas, weight decay and per-parameter state, so the values are optimised exactly as by a
  separate Adam optimiser. The single optimiser keeps mixed-precision scaling and clipping
  in one place.
- **Every other weight keeps `e16_cap`'s settings:** AdamW, 3e-4, weight decay 0.01, betas
  (0.9, 0.95).
- **Clipping is unchanged:** one global norm over every weight's gradient, the values'
  included. Cap 1.0, then 5.0 after step 1,000.
- **The hook in `uie/` (the one change there):**
  - `uie.train.train` builds its optimiser from `model.optimizer_groups(cfg)` when the model
    defines it and returns groups. Otherwise it is built exactly as before.
  - Each step, a group's rate is the schedule × its own `lr_scale` if it has one; otherwise
    it is the schedule, as before.
  - No `uie` model defines the hook. P1b's model defines it only when `mem_values_lr` is set.
  - Off by default, then, and proven bit-identical (V-hook, §4).

## 3. The audit

### 3.1 Done on the laptop (2026-10-13; `v2_p1b_audit.py`, `data/v2/p1b/audit.json`)

**The test** (`v2_mem.batch_coupling_check`):
- each row's logits are computed alone, in one batch with rows of other lengths, in
  length-grouped batches, and in training mode with dropout off;
- any stored statistic that changes during the training-mode pass is recorded.

Real validation rows of 19–202 tokens:

| model | batched vs alone | training mode vs alone | stored statistics changed |
|---|---|---|---|
| `e16_cap` | 0.000015 | 0.000015 | none |
| **P1 (`final.pt`, BatchNorm)** | 0.000013 | **26.1** | **running mean and variance** |
| P1b's start (LayerNorm) | 0.0045 | 0.0045 | none |
| P1b's start, memory switched off | 0.0000016 | 0.0000016 | none |

**What it shows:**
- **The BatchNorm was the only batch-coupled component.** Elsewhere:
  - every normalisation is a per-token LayerNorm;
  - attention is causal and padding comes last, so no real token attends to padding;
  - the loss, its normalisation, gradient accumulation and length bucketing exist only in
    training, where they shape updates and not the function computed;
  - fp16 autocast in training against fp32 in generation is precision, not coupling. It has
    been so in every engine since E5, and the memory runs in float32 in both.
- **The fix's 0.0045 is not a leak.** It is identical in training and evaluation, nothing is
  stored, and it vanishes with the memory off. Counted directly, **1 of 63,616 slot choices**
  differs between scoring alone and in a batch: a near-tie at the edge of the top-32,
  decided by rounding.

### 3.2 On Kaggle, strictly before the first training step (T4)

- **In `V2P1B_train`'s cell 3,** before any training call: `batch_coupling_check` on P1b's
  full-size start, with T1's setting, on real validation rows of six lengths (about 20 to
  200 tokens) from the attached corpus.
- **It passes iff:** every gap is under 0.05 logits and no stored statistic changes.
  P1's BatchNorm gives 26.
- **On a failure the cell raises STOP:** nothing is trained.
- **Inside the run,** V2's start builder repeats the check on the exact model object it hands
  to training, on synthetic rows of six lengths, and raises before the first step if it
  fails.

## 4. Validity (added to P1's §7)

| code | check | on failure |
|---|---|---|
| V-hook | **The default-off path is bit-identical.** A 4-step CPU training of a toy model, with 1 and 2 parts, must reproduce the pre-hook code's own weights, final validation losses, result fields and saved checkpoint settings (a test). AdamW with weight decay 0 equals Adam step for step (a test). The group's rate is schedule × scale (a test). See the note below | stop; nothing is bundled |
| V-iso | `uie/` equals the commit that adds the hook | INVALID |
| V-audit | §3.2 on Kaggle, recorded in PROVENANCE; repeated on the laptop at VERIFY_TRAIN on the trained `final.pt` | STOP before training / INVALID |
| V-recipe | as P1, plus: `mem_query_norm` "layer", `mem_branch_dropout` False, `mem_values_lr` 1e-3 in the checkpoint's settings; 161,792,000 parameters | INVALID |

**V-hook's reference values (disclosed exactly):**
- First recorded at 18de524 at the laptop's default of 6 threads: `c2d4c450…` and
  `e09cfb9c…`.
- The full suite then failed the test. `test_e9` sets torch to 1 thread and never restores
  it, and CPU matrix sums round differently with the thread count.
- **The old and the new `uie/train.py` were then run side by side:** identical weights and
  losses at 1 thread (`72eb5989…`, `e881490b…`) and at 6 threads (`c2d4c450…`, `e09cfb9c…`)
  (`data/v2/p1b/fingerprint_old_vs_new.json`).
- The test now pins 1 thread and restores the count afterwards, as E9's own byte-identity
  test pins it. Its reference values are the old code's at 1 thread.

Everything else in P1's §7 applies unchanged to `p1b_mem`: V-start, V-batch, V-mass,
V-stream, K12e, V-gen at its 98% bar, V-rules, V4, V-D1b and V-floor.

## 5. Predictions [HYPOTHESIS]

| reading | prediction |
|---|---|
| the audit on Kaggle | PASS (about 98%) |
| K-used | PASS (about 70%). Evaluation-mode fit at least `e16_cap`'s, paths −0.02 to −0.06 nats per character |
| K2 (the tell falls 30% or more) | benign about 25%; phishing about 10% |
| SUCCESS | about 5%. The bar is far, and P1's in-conditions gain was small |

## 6. Cost

- **Kaggle T4:** training about 4–4.5 h (P1's loop: 3.9 h; T2 adds one more parameter group,
  not compute); generation about 3–3.5 h, only after the laptop's checks pass.
  **About 7–8 GPU-hours.**
- **Laptop (CPU, low priority, with the low-RAM alarm):**
  - the code, tests and bundle, before any run;
  - VERIFY_TRAIN and K-used, about 25 min;
  - VERIFY_GENERATE, about 1–1.5 h;
  - scoring, about 3–4 h unattended.
- **Calendar:** about 2 days.

## 7. Steps

1. **This registration,** committed.
2. **Code:** the `uie/` hook with V-hook's tests; P1b's common module, runner and start
   builder (with the in-run audit); the notebooks; the bundle `uie-v2p1b-code.zip`; the run
   sheet `V2P1B_KAGGLE.md`. Then the full suite, its actual result reported. Committed and
   pushed before any Kaggle run.
3. **The owner runs `V2P1B_train` on Kaggle.** Then the laptop runs VERIFY_TRAIN and K-used.
   **Only if both pass,** the owner runs `V2P1B_generate`. Then VERIFY_GENERATE, scoring and
   the reading, as P1 registered.
4. **STOP:** the owner decides.

## Status

- **2026-10-13:** drafted (7b2af0f).
- **2026-10-13:** registered with the owner's decisions (§1.2): T1 yes, T2 yes, T3 no, T4
  yes. Before any P1b code beyond the fix and audit (18de524), and before any training.
- **2026-10-13: code, tests and the bundle (§7 step 2). Ready for the owner's Kaggle run.**
  - **Commits:**
    - `339b05b`: V-hook's reference values, disclosed;
    - `422cb23`: T2's hook in `uie/train.py` and all of P1b's code. This is the V-iso
      reference;
    - `abd3e02`: pins V-iso to `422cb23`.
  - **Full suite:** 1,122 passed (pytest's own exit code 0), at low priority with the
    low-RAM alarm, which never fired.
    - An earlier run failed the new V-hook test on a thread count left by `test_e9`. Proven
      not the hook: the old and new code are bit-identical at 1 and 6 threads (§4's note).
    - The test now pins its thread count.
  - **The bundle:** `data/kaggle/uie-v2p1b-code.zip`, built from `abd3e02`, sha256
    `0845cf807c95779018d493bad7d3bc14916557a13bd9e552b827a0f4eb37a7c0`. It is 2.2 MB and
    holds 45 code files, the three tables, stream 1 (57,292 rows) and `e16_cap`'s record;
    `uie/` equals `422cb23`.
  - **The pre-flight on the laptop CPU,** from the extracted bundle's own code (no
    training):
    - the manifest verifies;
    - T4's audit at full size passes (real rows of 19–202 tokens: largest gap 0.0000018;
      synthetic rows: 0.0035, one near-tie; nothing stored changed);
    - 161,792,000 parameters;
    - the values group: weight decay 0, betas (0.9, 0.999), scale 3.33 (peak 1e-3).
  - **The run sheet:** `V2P1B_KAGGLE.md`. No Kaggle run has happened.
- **2026-10-13: READ: STOPPED at K-used** (registered label MEMORY UNUSED;
  `V2P1B_RESULTS.md`). VERIFY_TRAIN PASS. K-used benign +0.0140 [+0.0086, +0.0195] nats per
  character (FAIL); phishing −0.0186; usage 89.6% (PASS). Generation did not run; nothing is
  adopted. A post-hoc diagnosis (8f2729b, decides nothing) locates the benign loss in domain
  names on page rows: the memory recalls training site names. Amendment 1 (Kaggle-side
  checks) was logged and withdrawn (6a6509f) before any output existed. The owner decides
  next.
