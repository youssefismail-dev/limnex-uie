# FL — was benign QRY's gate reading a floor artifact? — pre-registration

**Registered:** 2026-09-23, before any number in this file was computed.
**Frozen:** `data/best_e5.pt`, `data/bundle2`, `discriminate_with_floor` unmodified,
and the R4 gate configuration exactly as recorded in
`data/r4_temp_dadopted/eval_report.json`. **Nothing here changes the gate.**

---

## 1. Why

Under the adopted decoder the gate's benign QRY cell reads **+0.0300** (AUC
0.5675, floor **0.5374**, 15 draws), the only unresolved cell in the README.
NS (`NSWEEP_RESULTS.md`) found that at the same n=361 the instrument reads
~+0.107 on comparable samples, and that the gate's floor sits above all 20 sweep
floors at n=361 (sweep mean 0.4936, sd 0.0258 per draw).

The gate seeds each floor with `seed + dsi` — the **split** index only — so a
cell's 15 draws share **3** independent floors, which the G=5 generations cannot
average away. **Leading hypothesis H_UF:** the cell's three floor draws were
unluckily high, and that — not content and not n — produced the low reading.

## 2. FL1 — the gate's own three floors (read-only, minutes)

The floor depends only on the real pool, n and the seed: the real triple is
drawn before the fakes. So the gate's floors can be recomputed without the
gate's fakes, at the gate's seeds **20260806, 20260807, 20260808**, with the
fake list serving only to fix n.

Per-generation n is not stored (the report gives n=360), so the floor is
computed on a band: **n in {330, 340, 350, 355, 360, 365, 370, 380, 390}**.
Readout: F(n) = mean of the three seeds' floors.

**Decision, fixed now:**
- **H_UF supported** if |F(360) - 0.5374| <= 0.01 **and** F(360) - 0.4936 >= 0.02:
  the gate's floors are reproduced, and they are high.
- **NOT REPRODUCED** if |F(360) - 0.5374| > 0.02: the gate's floor did not come
  from where the code says it did. That is an instrument/pipeline discrepancy,
  escalated before anything else.
- Otherwise **INCONCLUSIVE** (per-generation n is unknown; FL2 is exact).

## 3. FL2 — re-score the cell with 30 split seeds (the exact gate generations)

**Regenerate** R4's five benign generations exactly: `generate_with_suffixes`
with `label=benign`, `n=8000`, the bundle's `suffix_table.json`,
`decoder=temperature`, `decoder_kwargs={"temperature": 1.0}`,
`field_bias={"PTH": {eos_id: -0.22}}`, `seed = 20260806 + 1 + g*1000` for
g = 0..4, CPU, default threads. Saved to disk so scoring is re-runnable.

**Reproduction kill criterion:** re-scoring QRY at the gate's own three split
seeds must reproduce **all 15** of R4's stored `excess_draws` to 4 decimals. If
any differs, the regeneration is not the gate's cell: **VOID, do not interpret.**

**Extended scoring:** D = **30** split seeds, `20260806 .. 20260835` (the first
three are the gate's), per generation, at each generation's own n. 150 draws.

**Estimator, unchanged from the README's section-26.1 treatment:** per-generation
mean over D, x = mean of the 5 generation means, SE = sd(generation means) /
sqrt(G-1), t_crit(df=4) = 2.776.

> **Note, 2026-09-23 - SE convention.** "Unchanged from the README's section-26.1
> treatment" is true of the README's *published* convention, sd / sqrt(G - 1), but
> not of section 26.1 itself, which specifies sd / sqrt(G). Gate v2 encodes the
> latter (`PREREG_GATE_V2.md`). FL2's verdict is unchanged under either.

**Verdict rule — the README's, fixed now:**
- **CONVICTED** if (x - 0.05) / SE > 2.776
- **CLEARED** if (0.05 - x) / SE > 2.776
- **UNDECIDED** otherwise

**Predictions.** Under H_UF: the D=30 floor mean lands near 0.494 (+-0.01) and x
near ~0.07 — **not CLEARED**. If the gate's reading is sound, x stays near 0.03.

## 4. Consequences, fixed now

- **Not CLEARED:** the README reports benign QRY as not cleared under the adopted
  decoder, with the D=30 numbers, and records that the +0.0300 reading was a
  floor artifact of D=3. The gate's per-split floor seeding is documented as a
  structural weakness at reduced n. **Any change to the gate's D or seeding is a
  separate decision for the user**, not made here.
- **CLEARED:** benign QRY is cleared at the gate's n. Reported with the caveat
  that NS found — descriptively, not at the registered standard — that small-n
  cells may read low.

## 5. Out of scope

Changing the gate's D, seeding, bar, MIN_N, delta, decoder or checkpoint. Any
training. Phishing QRY. The closed PTH branch.

## 6. Cost

FL1: 27 instrument calls, ~3 min. FL2: regenerating 40,000 URLs, ~2 h on CPU at
below-normal priority, single process (no worker pool, so no memory blowup),
then 150 scoring calls, ~15 min.

## Status

    FL1  RUN 2026-09-23 -- H_UF SUPPORTED (gate floors reproduced, 0.541)
    FL2  RUN 2026-09-23 -- 15/15 reproduced; D=30 x +0.0619 UNDECIDED,
         NOT cleared. See FLOOR_RESULTS.md
