# C3-QRY — does the query control change the queries? Pre-registration

**Registered:** 2026-09-25, before any C3 code and before any C3 generation.
**Approved by the owner (2026-09-25)** as written, including the batch-doubling
rule of section 3.

**Requested by the owner (2026-09-25),** after SC1's adoption (3b3ae24): QRY is
the largest remaining tell and the primary blocker of roadmap-E3's gate, so its
behaviour is characterised before any repair is attempted.

**Frozen:**
- `data/best_e7.pt` (sha256 `916d0385…`) and `data/best_e5.pt`;
- R4's decoder (`temperature` 1.0), the adopted delta (0.22 / 0.30), frame
  `domain`;
- `data/bundle2`;
- the adopted controlled path (`uie/controlled.py`, 3b3ae24) and its tables
  (`data/state_suffix_tables.json`, sha256 `0c6f2760…13cc9`);
- all of `uie/`: nothing in the library is edited;
- gate v2 and its thresholds;
- `research/cr1/cr1_score.py`;
- Q3.1's ladder code (`research/qry/q3_content.py`, `qtransforms.py`,
  `q3_1_resolve.py`);
- every published result — E3's, R4's, gate v2's, E7's, HC1's, HC2's, CR1's and
  SC1's. SC1's data is **read, never written**.

---

## 1. The question

E7 writes queries two ways:
- **controlled**, because `<Q:1>` asked for one — the adopted path;
- **free**, because the model chose to (`<Q:?>`) — what the canonical gate
  measures.

**C3's primary question.** Given the same host context, **is the query the
control produces the same query the model writes of its own accord?**

- **Why it matters.** It decides whether `<Q:1>` generation is a valid instrument
  for query content:
  - the phishing QRY cell cannot currently be verified at the gate's n (PQ:
    n_fake 180 < MIN_N 250);
  - `<Q:1>` supplies query-bearing URLs at 1.04–1.17 draws each.
- **Why it can be expected.** E7 was trained with control dropout, so
  p(url | `<Q:?>`, query present) should equal p(url | `<Q:1>`) if the model
  learned the control consistently. E7's registration named this test (C3,
  `PREREG_E7.md` section 4). It never ran, because C1 failed.

**Secondary question.** Where does E7's query tell live, in each mode? E5's was
**shape, not vocabulary, and distributed** (Q3.1; PQ1). If E7's is the same, the
QRY causal map built on E5 (Q1–Q8) transfers to E7.

**C3 selects no repair.** It measures what any later, separately registered
repair would be judged by.

## 2. Arms

| arm | tokens | suffix | kept | source |
|---|---|---|---|---|
| **Q** (controlled) | `<P:?> <Q:1>` | the Q:1 state table, held | every URL (presence enforced by resampling) | **SC1 arm T, `benign_Q1` and `phishing_Q1`, existing** (5 × 3,000 each; V1–V6 passed) |
| **F** (free, matched) | `<P:?> <Q:?>` | **the same Q:1 state table**, held | only URLs that carry a query | **new**, section 3 |
| F0 (free, as the gate) | `<P:?> <Q:?>` | the shipped table | the query rows of free generation | reference numbers only (section 6) |

- **Q against F isolates the control token.** Class, checkpoint, decoder, delta,
  frame and the suffix mix are equal, and in both arms the model chooses the
  path.
- F0 differs from F in the suffix mix as well, so it is a reference, not a
  comparison.

Both arms' kept URLs follow the same logic:
- Q keeps draws from p(url | s, `<Q:1>`, query present);
- F keeps draws from p(url | s, `<Q:?>`, query present);
- in both, s ~ the Q:1 table.

## 3. The F arm — generation

**Matched free generation**, per class and generation:
- the URL is drawn from `<Q:?>` output, so no presence control is set and every
  control token is `?`;
- the suffix is drawn from `StateSuffixTables.for_state(label, {"QRY": True})` —
  the adopted table, the Q arm's — and held;
- a URL is kept only if it carries a query, as the gate reads it: `FIELDS["QRY"]`
  non-empty.

    rng = random.Random(seed); drawn = table.sample(label, N, rng)
    for group i = (frame, suffix, count), in sorted order:
        round r = 0, 1, ... : draw M x 2^min(r, 10) x (count - kept) URLs with
            generate(..., suffix, fields=None, seed = seed + i + 1,000,003 * r)
            keep, in draw order, the ones with a query, up to count
        more than 100 rounds -> the request FAILS (K1), never returns short
    out = the kept URLs, group by group; rng.shuffle(out)

Drawing independent URLs and keeping the first `count` that carry a query is an
exact sample from p(url | s, `<Q:?>`, query present). **M and the doubling set
only the cost, not the sample.** M is 40 for benign and 80 for phishing, fixed
now. The batch doubles each round, capped at 1,024×, so even a suffix on which
E7 almost never writes a query finishes in a few rounds. K1 then fires only
where the rate is practically zero, which is itself a finding.

The rest of the call is SC1's:

    decoder="temperature", decoder_kwargs={"temperature": 1.0},
    field_bias={"PTH": {eos: -DELTA[label]}}, frame "domain", batch size 32

**Sizes.** Benign: 5 generations × **1,200**; phishing: 5 × **400** query-bearing
URLs. These are exactly what the QRY field is scored at: n = min(1,200, real
query pool / 3) — 1,200 benign; 356 phishing, where the real pool of 1,068
query-bearing hosts is the cap. Both arms therefore meet the scorer at the same
n, and every QRY comparison pairs exactly.

**Seeds.** 20261001 + (100·s + g)·10⁹ + **4.5·10⁸**, where s = 2 for benign Q:1
and s = 6 for phishing Q:1 (CR1's order), and g = 0–4.

| block user | offsets used |
|---|---|
| CR1 | 0 – about 1.0·10⁸ |
| SC1 arm C | 3·10⁸ – about 4.0·10⁸ |
| **C3 arm F** | **4.5·10⁸ – about 5.5·10⁸** |
| SC1 arm T | 6·10⁸ – about 7.0·10⁸ |
| SC1 replication | 8·10⁸ – about 9.0·10⁸ |

All ranges are disjoint.

**Cost.**
- About 400,000 draws, at E7's free query rates of 3.62% and 1.74% (HC1 C4).
  That is **3–4 h on a Kaggle T4** at CR1's measured rate of about 2,000 draws a
  minute.
- The rate under the Q:1 table is not known in advance; it is reported (C3-C).

## 4. Hypotheses

- **H_C (consistency, primary).** Given the same host context, `<Q:1>` does not
  change the query the model writes: Δ_QRY(Q − F) is within ±0.02.
  *Alternative H_X:* the control changes the queries.
- **H_S (shape, secondary).** In both modes, E7's query tell is carried by shape
  rather than vocabulary, as E5's was.
- **No directional prediction on content**, as in E7's registration.

## 5. Measurements and tests

### C3-A — consistency (PRIMARY)

**Scoring.**
- F is scored by `cr1_score.score` and `cr1_score.pools` (unchanged): the real
  pools restricted to the Q:1 state, G = 5 × D = 10, SEED 20260806, split seeds
  v2, n requested = min(n_main, 1,200) = 1,200.
- Q's records are SC1 arm T's, read from its report (sha256 `7a9e6485…`).
- Same pools, seeds and n, so every QRY draw pairs on identical real samples and
  floors.

**Statistic.** SC1's paired form, per class:

    δ_g = mean over d of excess_Q(g, d) − mean over d of excess_F(g, d)
    Δ = mean δ_g;  SE = sd(δ_g)/√5;  df 4;  95% CI = Δ ± 2.776·SE

**Reading, per class:**

| reading | rule |
|---|---|
| **CONSISTENT** | the 95% CI of Δ lies inside [−0.02, +0.02] (HC1 H3's equivalence form) |
| **DIFFERENT** | \|Δ\| ≥ 0.02 and Holm-adjusted p < 0.05 over the two classes; the direction is reported |
| **INCONCLUSIVE** | anything else |

**Power.** SC1's paired QRY contrasts between independently generated arms at
G = 5 had SEs of 0.0033–0.0057, i.e. 95% half-widths of **0.009–0.016**. So
CONSISTENT is reachable in both classes if the true difference is small. An
INCONCLUSIVE is a registered outcome, not a failure.

**Descriptive** (same scoring): Δ(Q − F) on LBL, SUF, SUB, PTH and FULL. They are
paired where n is equal and unpaired otherwise; SUF is expected near 0, since
the suffix mix is equal by design.

### C3-B — where the tell lives (SECONDARY)

Q3.1's ladder, **unchanged**:
- **Code:** `research/qry/q3_content.scored` (the floor is recomputed under
  every transform), `research/qry/qtransforms.py`.
- **Transforms:** Q3.1's 17, including `identity` and `null`, plus its
  pre-specified `clip15_first` interaction under Q3.1's rule.
- **Seeds:** D = 12, 20260806 + i; df 11, t_crit 2.201.
- **n:** 1,800 benign; 356 phishing, the real-pool cap.
- **Real queries:** Q3.1's `load_real`.
- **Fake queries:** every query string of the arm.

It runs four ladders: {benign, phishing} × {Q, F}.

**Reading per ladder** (PQ1's thresholds; Q3.1's test):

| reading | rule |
|---|---|
| **SHAPE** | `skeleton` R_T ≥ 0.85 and `skeleton` does not REDUCE (t ≥ −2.201) |
| **VOCABULARY** | R_T ≤ 0.40 and it REDUCES |
| **INTERMEDIATE** | otherwise; no transform is added afterwards to break the tie |

**Reading per class — Q3.1's decision rule.** The "second arm" a finding must
replicate in is the class's other mode:
- **IDENTIFIED:** one property reduces with |t| > 2.201 in both modes, R_T ≤ 0.50,
  and is monotone in its dose series where one exists.
- **DISTRIBUTED:** two or more properties reduce in both modes, none alone
  reaching 50%.
- **UNRESOLVED:** neither.

E5's published ladders (Q3.1 benign; PQ1 phishing) are shown alongside,
descriptively.

### C3-C — emission and structure (DESCRIPTIVE)

- F's first-pass acceptance: E7's free query rate under the Q:1 table, per
  class and per top-10 suffix. Set against E7's rate under the shipped table
  (3.62% / 1.74%, HC1 C4) and the real rate (6.40% / 4.44%).
- Path presence among query-bearing URLs: real, Q and F, with two-proportion
  95% CIs. (Masking once added paths to query URLs, HC1; resampling removed it,
  HC2.)
- Query shape, the statistics that motivated Q3.1's transforms: parameters per
  query, name length, value length p50 and p90, the percent-encoding share, the
  numeric-value share and the bare-flag share. Real against Q against F.

## 6. Reference numbers, not compared

E7's canonical free-generation gate (gate v2, HC1 C2): benign QRY UNDECIDED
+0.0541; phishing pooled UNDECIDED +0.0924. These come from different suffix
mixes and n, so no test is run against them.

## 7. Validity — technical, checked before any number is read

A failure is fixed under a dated amendment and never read as a result.

- **V1 files.**
  - F: 10 files (benign 5 × 1,200, phishing 5 × 400), with sha256 and seeds as
    the provenance records.
  - The checkpoint, `data/state_suffix_tables.json` (sha256 `0c6f2760…`) and the
    code bundle's manifest all match.
  - There are no other files.
- **V2 queries.** Every F URL carries a query as `FIELDS` reads it.
- **V3 identity.** The first-round kept URLs of the leading suffix groups
  (generation 0, at least 250 URLs per class) regenerate byte-identically on
  the laptop CPU.
- **V4 table.** In every F file, the parsed suffixes agree on at least 99% of
  URLs with the multiset that `table.sample(label, N, random.Random(seed))`
  draws (SC1's V4).
- **V5 the Q arm.** SC1 arm T's ten Q:1 files still carry the sha256 its
  provenance records: benign `f1f21d66…`, `36901a2d…`, `b5136cbc…`, `21a5e122…`,
  `578c15dc…`; phishing `df87ea0b…`, `dc74a373…`, `e11ce93f…`, `5329dce5…`,
  `451e9bef…`. Its report is still sha256 `7a9e6485…`.
- **V6 S0‴.** Re-scoring SC1 arm T's `phishing_Q1` on the day of C3's scoring
  reproduces its record exactly. The environment must still yield the floors the
  pairing relies on.

## 8. Criteria

**Success (the question is answered).**
- C3-A reads CONSISTENT or DIFFERENT in both classes, and
- none of the four C3-B ladders is void.

**Partial.** Anything short of that; the unanswered parts are reported as such
and license nothing.

**Kill:**

| code | trigger | consequence |
|---|---|---|
| **K1 — feasibility** | any F request exhausts its 100 rounds | that class's F arm has failed; its C3-A and C3-B are not read; M, the round limit and the table are never changed after the fact |
| **K2 — instrument** | any of V1–V6 fails | fix under an amendment; never read |
| **K3 — ladder validity** (Q3.1 / PQ) | a floor outside [0.42, 0.55], or a distinct fraction < 0.50 | that row is VOID |
| | `null` is not a byte-exact no-op, or `identity`'s excess differs from the same arm's C3-A x by more than 0.05 | the whole ladder is void |

**What each outcome licenses. Nothing here is a repair.**

| outcome | what it licenses |
|---|---|
| **CONSISTENT** (both classes) | `<Q:1>` output is a valid instrument for E7's query content, so a later repair can be measured there at the real-pool cap. Using it inside the canonical gate would be its own registration |
| **DIFFERENT** (in a class) | the control changes that class's queries: work on free-mode queries must be measured in free mode, or the control's effect addressed first |
| **C3-B's readings** | which components carry E7's tell in each mode, and whether E5's map transfers. A repair aimed at them needs its own registration |

## 9. Execution

1. This registration, committed.
2. Code in `research/c3/`, with tests; `uie/` untouched:
   - the matched-free generator;
   - `C3_generate.ipynb` and the code bundle (uie/, the generator, the tables
     file);
   - the verifier (V1–V6);
   - F's scoring via `cr1_score`'s functions;
   - the C3-A comparison;
   - the C3-B driver on Q3.1's code;
   - C3-C's statistics.

   The generator's tests pin: exactness, every control token `?`, the
   determinism of the seeds, and the refusal at 100 rounds.
3. A CPU simulation of the notebook in Kaggle's layout.
4. Kaggle: arm F, 10 files, about 3–4 h (the owner runs it).
5. V1–V6; then scoring F; C3-A; the four C3-B ladders; C3-C. Laptop,
   below-normal priority, memory guard.
6. `C3_RESULTS.md`; this status; push. The owner decides what, if anything,
   follows.

Outputs go to `data/c3/`. Nothing under `data/sc1/`, `data/cr1/` or `data/qry/`
is written.

## 10. Not claimed

- **C3 measures E7, not E5.** E5's query tell is cited from Q3.1 and PQ1, not
  re-measured.
- C3 changes no gate, bar, verdict or published number, and adopts nothing.
- Phishing's query field is capped at n = 356 by its real pool, in C3 as in
  every instrument. That limit is the data's.
- A CONSISTENT reading would say the control does not change the queries. It
  would **not** say the queries are realistic: E7's query tell is known (CR1,
  SC1), and C3-B describes it rather than removes it.

## Status

    C3-QRY  registered 2026-09-25, owner-approved (b4576d2)
    code    49d0972  arm F's generator, the notebook and its code bundle, the
                     verifier (V1-V6), F's scoring, C3-A, the C3-B driver, C3-C;
                     15 tests
            52c926e  the notebook writes PROVENANCE after every file, and a
                     rerun keeps an earlier outcome only with its own record.
                     Before, a rerun read finished files back with EMPTY draw
                     counts -- the counts C3-C reads. Operational: nothing
                     sampled changes, and the bundle's code is identical
            80d6560  K1 as section 8 registers it, in the laptop scripts: the
                     class is not scored and its C3-A and C3-B are not read;
                     Holm's family stays the registered two classes (a class
                     not read enters with p = 1, so the other is never tested
                     less strictly); C3-C reports the finished files' rate and
                     the K1 lines, not the failed arm's URLs; V3 does not
                     compare a K1 class. 2 tests; suite 698
            1653c24  C3-B: gc.collect() after every discriminator call, as the
                     gate has done since 2026-09-23 (memory only). Q3.1's
                     run_ladder predates that fix; under Python 3.14 each call
                     left ~120 MB of cyclic garbage. Values bit-identical with
                     and without it; Q3.1's files untouched. 1 test; suite 699
    smoke   real E7 on the laptop CPU, generation 0 at small n: every URL
            carries a query; suffix agreement 1.0; identity 16/16 benign, 8/8
            phishing; round-0 free query rate 4.75% benign, 2.88% phishing
    cost    simulated -- not an amendment. The 10 files' exact suffix groups
            (their seeds fix them); E7's per-suffix free query rates from its
            unconditional C2 generations (data/gen_cache_e7_gpu, 5 x 8,000 per
            class: the F condition, only the suffix mix differs) through an
            empirical-Bayes beta-binomial; 400 replicates at SC1's measured 34
            draws/s: 4.1 h median, 4.8 h p99, 5.6 h max; at most 10 of the 100
            rounds; K1 in 0 of 400. Section 3's "3-4 h" was low; nothing
            registered depends on it. The residual risk is a suffix whose rate
            is far below anything C2 shows, which a unimodal prior cannot
            represent
    checks  notebook, Kaggle layout, tiny model: PASS -- a normal run and its
            reruns, a session killed during a file, K1
            verifier on that notebook output: V5 as registered; V6 S0'''
            reproduced -- today's scoring environment still yields SC1 arm
            T's phishing_Q1 record; V3 identity 6/6 in both classes; every
            other entry is the simulation's own (6-URL files, 2 of 10 files,
            tiny-model hosts). On a simulated K1 output, V3 reads "K1 -- not
            compared" and the K1 lines are listed
            laptop chain on STAND-IN data (SC1 arm C's published Q1 files at
            the registered sizes; nothing under data/ written). K1 case
            (benign g4, phishing all five): every script exits 0 with the
            registered records. Normal case: scoring (28.7 min), C3-A (QRY
            paired in both classes), C3-C and C3-B exit 0. C3-B's first
            attempt was stopped by the memory guard -> 1653c24; rerun 39.3
            min, peak RSS 473 MB, all four ladders valid, 12 draws per rung.
            Only exit codes, structure and validity were read: the arm-Q
            rungs are a deterministic function of SC1's published files and
            are recomputed identically in the real run
    bundle  data/kaggle/uie-c3-code.zip, built at 52c926e: 35 files, sha256
            7c44455783236276...; C3_generate.ipynb sha256 b3d1f9ffbe1f1d5e...
    Kaggle  run by the owner 2026-09-25: best_e7 sha256 verified, 35/35 files,
            commit 52c926ee555d, tables 0c6f2760 verified; Tesla T4, torch
            2.10.0+cu128. 10 files in 110 min, 462,800 draws, at most 5
            rounds, no FreeUnmet
    6b489a8  C3-A contrasts FULL, as section 5 registers (it had been left
            out of the descriptive set); the table renderer. Committed
            before anything was scored
    V1-V6   PASS, 2026-09-25 20:08-22:15; V3 identity 387/387 benign,
            274/274 phishing
    results C3_RESULTS.md. C3-A: benign DIFFERENT, Delta +0.0201 [+0.0124,
            +0.0277], Holm p 0.0038 -- it clears the 0.02 margin by 0.00005;
            phishing INCONCLUSIVE, +0.0183 [-0.0083, +0.0450], p 0.128.
            C3-B: all four ladders valid; benign SHAPE in both modes,
            UNRESOLVED; phishing INTERMEDIATE in both modes, DISTRIBUTED.
            Success criterion NOT MET: PARTIAL (section 8). The owner
            decides what, if anything, follows
