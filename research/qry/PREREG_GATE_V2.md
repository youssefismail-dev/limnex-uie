# Gate v2 — pre-registration: independent floors, D=10, the section-26.1 verdict in the gate

**Registered:** 2026-09-23, before any code for it was written.
**Approved by the owner:** Option C (A + B); SE = sd / sqrt(G); the section-26.1
t-test as the verdict the gate prints; the envelope kept as a printed
non-acceptance metric; the pooled rule unchanged, with its D=3 result reported;
the legacy path kept selectable and verified.
**Frozen:** `data/best_e5.pt`, the R4 adoption (`temperature`), the 0.05 bar, the
adopted delta (benign 0.22 / phishing 0.30), `discriminate_with_floor`, `MIN_N`.

---

## 1. Why

FL (`FLOOR_RESULTS.md`) showed the gate's floors are seeded per **split**
(`seed + dsi`), so all generations share the same D floors. The section-26.1 SE is
taken **across generation means**, so a floor offset common to every generation
never enters it. Measured on benign QRY: the hidden offset was **0.040, four times
the reported SE**, and the D=30 value fell outside the gate's own 95% interval.

## 2. What changes — estimator v2, exhaustively

1. **Independent floor per (generation, split).** Split seed =
   `seed + 100_000 + 1_000 * g + d`, asserted for G < 100 and D < 1000, a block
   disjoint from the v1 seeds (`seed + d`) and the generation seeds
   (`seed + 1 + 1000 * g`). The seed drives the whole split — real triple, floor,
   fake subsample — exactly as in v1; only its value changes.
2. **D = 10** for the canonical run (a run parameter, `--disc-seeds 10`; G stays 5).
3. **The section-26.1 verdict is computed and printed by the gate** — the rule the
   README already uses, until now computed outside the gate:
   - per-generation mean over that generation's usable draws
   - x = mean of the G generation means
   - **SE = sd(generation means) / sqrt(G)**, with sd the sample standard
     deviation (n - 1 denominator)
   - t_crit = the 0.975 quantile of Student t with G - 1 degrees of freedom
     (2.7764 at G = 5)
   - **CONVICTED** if (x - 0.05) / SE > t_crit, **CLEARED** if
     (0.05 - x) / SE > t_crit, **UNDECIDED** otherwise
   - requires at least one usable draw in **every** generation; otherwise reported
     as NOT COMPUTABLE, with the count of usable generations
   - FULL stays **AUXILIARY** and never gates.
4. **The envelope tag** (`ok` if max < bar, `TELL` if min > bar) is still computed
   and printed, labelled **"envelope — not the acceptance verdict"**. It widens
   mechanically with D (FL2: width 0.108 at D=3, 0.182 at D=10).
5. **Pooled fallback — rule unchanged:** TELL if min over D draws > bar, otherwise
   UNDECIDED, never ok. Its seeds stay `seed + d` (there is no generation index to
   seed by), so its **first 3 draws are exactly v1's**. v2 also prints the verdict
   on those 3 draws, so a flip caused purely by D is visible.
6. **Auxiliary precision/recall sample:** v1 draws the real embedding sample from
   one RNG created before the class loop, so a phishing-only run draws differently
   from a combined run. v2 seeds it per class
   (`seed + 200_000 + class_index`) so per-class runs equal a combined run.
   **Affects precision/recall only; never gated.**
7. **Report JSON (v2 only):** an `estimator` block (version, floor seeding, G, D,
   SE formula, t_crit) and a per-field `acceptance` block; pooled fields get a
   `d3_verdict`.

## 3. What does NOT change

The bar; `discriminate_with_floor`; `MIN_N`; per-field n (`min(n_main, 1200)`);
the field populations and frame; host dedup; delta; the decoder; the checkpoint;
G = 5; the generation seeds; the pooled rule; FULL's auxiliary status.

**The gate's pass condition is not changed.** The README has never counted an
UNDECIDED cell as passing, and this repair keeps that.

**Not fixed by v2, stated so it is not over-read:** pooled cells remain
convict-only (phishing QRY today), and the small-n learning-curve bias NS saw
descriptively is out of scope.

## 4. The legacy path

`--estimator v1` reproduces the pre-v2 gate exactly: seeds `seed + d`, the shared
auxiliary RNG, the envelope as the printed verdict, and an identical report JSON
with no new keys. **The default becomes v2**, as with the decoder adoption; a
published command reproduces with `--estimator v1` (plus `--decoder typical` where
it predates R4).

## 5. Operational additions — value-neutral

- **`--gen-cache DIR`:** memoises each generation, keyed by checkpoint hash, class,
  n_main, decoder and its kwargs, delta and generation seed. Generation is
  deterministic (FL2 reproduced all 15 of R4's draws). The cache is pre-seeded
  with FL2's saved benign generations — R4's exact benign generations — which the
  full-report check in section 6 then verifies field by field.
- **`--classes`:** runs a subset of classes, to keep peak memory near 4 GB. v1
  bit-exact reproduction needs the full class set, because of the shared RNG in
  item 6.

## 6. Verification — kill criteria, before any v2 number is read

- **V1, the legacy path.** `--estimator v1`, G=5, D=3, n_main 8000,
  `temperature` 1.0, delta 0.22 / 0.30, `best_e5.pt`, both classes in one process.
  The resulting `eval_report.json` must **equal R4's
  (`data/r4_temp_dadopted/eval_report.json`) exactly, as a whole document.** Any
  difference: **STOP** — v2 is not run.
- **V2, the code.** Unit tests for the seed functions, the SE and verdict
  computation, t_crit, the NOT COMPUTABLE case and the cache key.
- **V3, the verdict code.** Applied to R4's stored draws with sd / sqrt(G), the
  in-gate verdict function must reproduce the table computed by hand on
  2026-09-23 (25 cells, zero flips against sd / sqrt(G-1)).

## 7. Expectations — for interpretation, not kill criteria

- SEs rise, most for reduced-n cells (QRY), because floor noise now enters them.
- Benign QRY lands near FL2's **+0.0619**; a gap larger than 0.02 is investigated
  before it is reported.
- Any cell whose verdict changes from v1 to v2 is reported with its change split
  into AUC and floor contributions. No prediction is made about which cells
  change: this repairs an instrument, and its readings are the new baseline.

## 8. How the acceptance table is re-read — fixed now

- The v2 baseline's section-26.1 verdicts **replace** the README's
  adopted-decoder table. The v1 tables stay, marked as computed under the old
  estimator and SE convention.
- Pooled cells are reported with both D=10 and D=3 verdicts; they can never
  acquit.

## 9. Cost and safety

About 4.5 h. The V1 run generates phishing (~2.8 h; benign comes from the cache)
and scores at D=3 in one process (~2 GB, as R4 did). v2 then scores from the cache,
one class per process (~4 GB peak). Below-normal priority, memory watched
throughout.

## 10. Dated notes on the SE convention (approved)

Added with this registration. They document; they change no result:

- **README and E3_REPORT section 30:** the published t-values used
  sd / sqrt(G - 1), while section 26.1 specifies sd / sqrt(G). All 25 published
  cells were re-checked on 2026-09-23: **no verdict differs.**
- **R4_RESULTS, PREREG_FLOOR, FLOOR_RESULTS:** the same convention was used;
  their verdicts are unchanged under sd / sqrt(G). FL2's benign QRY moves from
  t = +1.40 to +1.57 — still UNDECIDED.

## Amendment, 2026-09-23 — memory (operational, value-neutral)

Measured while V1 ran, before any v2 number existed. Each discriminator call left
memory behind that was never returned. Ten calls on the benign pool, resident
memory above the loaded baseline:

    default allocator            +681 MB   (rising every call)
    PYTHONMALLOC=malloc          +903 MB   (worse -- not allocator fragmentation)
    gc.collect() after each      +174 MB   (flat after the first call)

The retained memory is uncollected cyclic garbage: each call builds a few very
large sklearn objects, and the generational collector reaches them too rarely.
V1 itself stood at 4.5 GB after its benign half; at D=10 the benign half makes
~300 calls, projecting ~12 GB in one process.

**Change:** the gate calls `gc.collect()` after each `field_report` and each pooled
call. Garbage collection frees only unreachable objects, so no computed value can
change.

**Registered check:** V1 ran on the pre-amendment code. Once it passes, the
amended code is run with `--estimator v1 --classes benign` from the generation
cache, and its benign block must equal R4's exactly. Benign is the first class,
so its auxiliary RNG state matches a combined run. Any difference: STOP.

## Status

    GATE-V2  registered (2f6630b); implemented (b6d12e7); memory amendment (f4b186c)
             V1 PASSED 2026-09-23: the legacy path's eval_report.json is
               byte-identical to R4's (13,054 bytes, sha256 587f8659d26d7a37...)
             V2 PASSED: 18 unit tests (tests/test_gate_v2.py)
             V3 PASSED: 25/25 published cells, x and verdict (v3_verdict_check.py)
             amendment re-check PASSED: benign block identical to R4's
             v2 BASELINE COMPLETE -> GATE_V2_RESULTS.md; README table replaced
