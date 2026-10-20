"""
uie/corpus/run_eval.py

The evaluation suite. Replaces `run_e3`.

Why the old one was retired
---------------------------
`run_e3` reported one headline number: a discriminator's AUC on WHOLE URLs,
against a pre-registered bar of 0.65. Three things were measured afterwards that
together make that number unusable as a gate.

**The bar sat below its own floor.** Two samples of REAL URLs, nothing
generated, score well above 0.5 whenever they differ in any way:

    same crawl, random split          0.5144
    domain-disjoint, same crawl       0.5206
    real repeats domains vs 1/domain  0.5400
    crawls 4 months apart             0.5717
    crawls 21 months apart            0.6005
    uniform sampling vs stratified    0.7911     <- both sides REAL
    real train vs real val (E2 split) 0.741 / 0.828

A generator cannot be less distinguishable from held-out data than a real
sample drawn the same way. 0.65 was unreachable by anything, including a
perfect generator, and two full iterations were read against it.

**It measured a joint the model is not trained to produce.** Under field-scoped
training the model learns P(name fields) from EVERY domain and P(path fields |
name) from multi-page sites only. Sampling a whole URL composes two conditionals
estimated on different populations. No real corpus has that composition, so a
whole-URL discriminator scores the composition rather than the fields.

**One scalar hid real progress.** E2.1 cut the label tell by 0.094 and E2.2
improved path and query while regressing the label; the whole-URL number moved
-0.022 and -0.002 respectively. Both were correct, well-diagnosed changes that
the headline reported as failure or noise.

What this measures instead
--------------------------
Every field is scored against the population it was TRAINED on, and every AUC is
reported beside a floor measured from real-vs-real on that same population. The
number that matters is EXCESS OVER FLOOR:

    excess ~ 0   indistinguishable, given how variable real data is
    excess > 0   a real tell, over and above natural variation

This is not a lower bar. It is the same bar with its zero point measured instead
of assumed. A generator that was genuinely perfect would score excess 0 here and
could never score 0.65 on the old metric.

Whole-URL AUC is still computed and reported, as AUXILIARY. Field-scoped metrics
are blind to composition errors — a perfect label beside a perfect path in an
implausible pairing passes both — and composition is exactly the known weakness
of field-scoped training. It informs; it does not gate.
"""
from __future__ import annotations

import csv
import gc
import gzip
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from uie.bloom import BloomFilter
from uie.evaluation import (FIELD_POPULATION, FIELDS, GENERATOR_FRAME,
                            discriminate_with_floor, diversity, embed,
                            field_report, human_panel, precision_recall,
                            probe_representation)
from uie.gate import Gate
from uie.metrics import BY_SUFFIX_KIND, compare, describe
from uie.factorization import parse
from uie.model import DECODERS, generate
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.train import load_checkpoint

N_ABLATION = 400
N_MAIN = 1_200
N_PANEL = 60

#: Excess over floor at which a field counts as carrying a real tell. Set from
#: the measured spread of real-vs-real conditions (0.514 to 0.601 across crawl,
#: split and repetition differences), so it sits just outside ordinary variation
#: rather than at an arbitrary round number.
EXCESS_BAR = 0.05

#: Gate estimator versions (research/qry/PREREG_GATE_V2.md). v1 is the gate as it
#: stood before 2026-09-23, kept selectable so every published number reproduces
#: exactly. v2 gives every (generation, split) its own floor, uses D=10 in the
#: canonical run, and computes the section-26.1 verdict IN the gate.
ESTIMATORS = ("v1", "v2")


def split_seed(seed: int, g: int, d: int, estimator: str) -> int:
    """Seed for split `d` of generation `g`.

    v1 seeds by split only (`seed + d`), so every generation shares the same D
    floors -- and the section-26.1 SE, taken ACROSS generation means, cannot see
    a floor offset they all have in common. FL measured that hidden offset on
    benign QRY at four times the reported SE (research/qry/FLOOR_RESULTS.md).
    v2 gives every (generation, split) its own floor, in a seed block disjoint
    from the v1 seeds (`seed + d`) and the generation seeds (`seed + 1 + 1000 g`).
    """
    if estimator == "v1":
        return seed + d
    if estimator != "v2":
        raise ValueError(f"unknown estimator {estimator!r}; known: {ESTIMATORS}")
    if not (0 <= g < 100 and 0 <= d < 1000):
        raise ValueError(f"v2 split seeds need 0 <= g < 100 and 0 <= d < 1000; "
                         f"got g={g}, d={d}")
    return seed + 100_000 + 1_000 * g + d


def t_crit(df: int) -> float:
    """Two-sided 95% critical value of Student t: 2.7764 at df = 4."""
    from scipy.stats import t
    return float(t.ppf(0.975, df))


def s261_verdict(by_generation: Sequence[Sequence[float]],
                 bar: float = EXCESS_BAR) -> dict:
    """The section-26.1 acceptance verdict, computed in the gate (v2).

    Per-generation mean over that generation's usable draws; x = the mean of the
    G generation means; SE = sd(generation means) / sqrt(G) with the SAMPLE sd,
    which is the form section 26.1 specifies. The t distribution with G - 1
    degrees of freedom already pays for estimating that sd from G values. The
    published t-values divided by sqrt(G - 1) instead; re-checked 2026-09-23, no
    published verdict differs.

    Needs a usable draw in EVERY generation. A missing generation mean cannot be
    invented, and dropping one would change G, and the threshold with it.
    """
    G = len(by_generation)
    usable = sum(1 for g in by_generation if len(g) > 0)
    if G < 2 or usable < G:
        return {"verdict": "NOT COMPUTABLE", "generations": G,
                "generations_usable": usable}
    gm = [statistics.mean(g) for g in by_generation]
    x = statistics.mean(gm)
    se = statistics.stdev(gm) / math.sqrt(G)
    tc = t_crit(G - 1)
    if se == 0.0:
        t = math.inf if x > bar else (-math.inf if x < bar else 0.0)
    else:
        t = (x - bar) / se
    verdict = "CONVICTED" if t > tc else ("CLEARED" if -t > tc else "UNDECIDED")
    return {"verdict": verdict, "x": round(x, 4), "se": round(se, 4),
            "t_vs_bar": round(t, 2) if math.isfinite(t) else t,
            "t_crit": round(tc, 4), "df": G - 1, "generations": G,
            "generations_usable": usable,
            "generation_means": [round(m, 4) for m in gm]}


def pooled_verdict(excess: Sequence[float], bar: float = EXCESS_BAR) -> str:
    """The pooled fallback's rule, UNCHANGED by v2: convict or abstain, never
    acquit. Its minimum falls as draws are added, so v2 also reports it on the
    first 3 draws -- exactly v1's -- to keep a flip caused only by D visible."""
    return "TELL" if excess and min(excess) > bar else "UNDECIDED"


def _file_sha12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def gen_cache_path(cache_dir: Path, ckpt_tag: str, label: str, n_main: int,
                   decoder: str, decoder_kwargs: Optional[dict], delta: float,
                   gen_seed: int) -> Path:
    """One file per generation, keyed by everything that determines it.

    Generation is deterministic given these (FL2 reproduced all 15 of R4's
    draws from regenerated URLs), so caching changes no value. It lets the
    estimator be re-run on identical generations, and a killed run resume.
    """
    kw = "-".join(f"{k}={v}" for k, v in sorted((decoder_kwargs or {}).items()))
    return cache_dir / (f"{ckpt_tag}_{label}_n{n_main}_{decoder}"
                        f"{'_' + kw if kw else ''}_d{delta}_s{gen_seed}.json")


def _rows(path: Path) -> List[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


def _find(bundle: Path, stem: str) -> Optional[Path]:
    for name in (f"{stem}.jsonl.gz", f"{stem}.jsonl"):
        if (bundle / name).is_file():
            return bundle / name
    return None


def _aggregate(draws: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Collapse G x D per-field rows into a mean and the OBSERVED spread.

    The spread is reported, never summarised away: the whole reason for multiple
    draws is that this instrument's single-draw noise is nearly twice the 0.05
    bar it is judged against.

    A field counts as `insufficient` only when EVERY draw was -- one usable draw
    is still evidence -- and `draws` records how many survived, so a field
    measured on 3 of 15 cannot be mistaken for one measured on 15.
    """
    out: Dict[str, dict] = {}
    for name, rows in draws.items():
        usable = [r for r in rows if not r.get("insufficient")]
        if not usable:
            first = dict(rows[0])
            first["draws"] = 0
            first["draws_attempted"] = len(rows)
            out[name] = first
            continue
        ex = [r["excess"] for r in usable]
        agg = dict(usable[0])
        agg.update({
            "auc": statistics.mean([r["auc"] for r in usable]),
            "floor": statistics.mean([r["floor"] for r in usable]),
            "excess": statistics.mean(ex),
            "excess_sd": statistics.pstdev(ex) if len(ex) > 1 else 0.0,
            "excess_min": min(ex),
            "excess_max": max(ex),
            "n": int(statistics.mean([r["n"] for r in usable])),
            "draws": len(usable),
            "draws_attempted": len(rows),
            # Raw per-draw excesses are kept, not just summarised. The
            # first entry is the G=1/D=1 draw, so a multi-draw report
            # can always be checked against the single-draw one.
            "excess_draws": [round(x, 4) for x in ex],
        })
        out[name] = agg
    return out


def main(checkpoint: Path, bundle: Path, out_dir: Path, *,
         seed: int = 20260806, n_main: int = N_MAIN,
         delta: Optional[Dict[str, float]] = None,
         generations: int = 1, disc_seeds: int = 1,
         # ADOPTED 2026-09-23, was "typical" -- research/qry/R4_RESULTS.md.
         # The reference run under this decoder is data/r4_temp_dadopted.
         decoder: str = "temperature",
         decoder_kwargs: Optional[dict] = None,
         # Gate v2 (research/qry/PREREG_GATE_V2.md). `--estimator v1`
         # reproduces every report published before 2026-09-23 exactly.
         estimator: str = "v2",
         gen_cache: Optional[Path] = None,
         classes: Optional[Sequence[str]] = None) -> dict:
    """`delta` applies the ADOPTED path-scoped EOS bias, per class.

    Decode-time only, and OFF by default (`{}` / 0.0 per class) so an
    unparameterised run stays bit-identical to every earlier report. It exists
    because the adopted resting configuration is E5 *plus* delta (benign 0.22,
    phishing 0.30, E3_REPORT 27.2) and no canonical evaluation had ever scored
    that configuration -- every delta number to date came from an ad-hoc
    research harness rather than from this evaluator.

    Nothing else changes: the 0.05 excess bar, `discriminate_with_floor`, the
    decoder, the checkpoint and the field populations are untouched. This is a
    wiring step so the official gate can be applied to what we actually ship.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; known: {ESTIMATORS}")
    rng = random.Random(seed)
    ckpt_tag = (f"{Path(checkpoint).stem}-{_file_sha12(Path(checkpoint))}"
                if gen_cache else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_checkpoint(checkpoint, device=device)
    model.to(device).eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Held-out pools, one per (class, training population). Per CLASS because a
    # phishing generation scored against benign held-out URLs measures the gap
    # between the two classes — which is real, large and not the model's doing —
    # instead of measuring generation quality. `mixed` is only for the auxiliary
    # whole-URL number and is not a target the model is trained on.
    by_class: Dict[str, Dict[str, List[str]]] = {}
    for label in tokenizer.classes:
        prefix = "" if label == "benign" else f"{label}_"
        pools: Dict[str, List[str]] = {}
        for pop, stem in (("name", "name_val"), ("structure", "structure_val")):
            path = _find(bundle, f"{prefix}{stem}")
            if path:
                pools[pop] = _rows(path)
        missing = set(FIELD_POPULATION.values()) - {"mixed"} - set(pools)
        if missing:
            raise FileNotFoundError(
                f"missing held-out pools {sorted(missing)} for class {label!r} "
                f"in {bundle}. Expected {prefix}name_val.jsonl[.gz] and "
                f"{prefix}structure_val.jsonl[.gz] — each field is scored "
                f"against the population it was TRAINED on, so neither a single "
                f"pooled val set nor another class's pool can be substituted.")
        pools["mixed"] = pools["name"] + pools["structure"]
        by_class[label] = pools

    print(f"model {sum(p.numel() for p in model.parameters())/1e6:.2f}M on {device}")
    for label, pools in by_class.items():
        for pop, rows in pools.items():
            print(f"  held-out {label}/{pop:<10} {len(rows):>9,} urls")

    bloom_path = next((bundle / n for n in
                       ("known_urls.bloom.gz", "known_urls.bloom")
                       if (bundle / n).is_file()), None)
    membership = BloomFilter.load(bloom_path) if bloom_path else None
    if membership:
        print(f"  novelty vs {membership.n_added:,} training urls "
              f"(fp {membership.expected_error_rate:.4%})")

    # The suffix is SUPPLIED, never sampled from the model. Benign block
    # sampling is stratified at alpha=0.5, which flattens the suffix marginal on
    # purpose to maximise data per conditional; letting the model choose its own
    # suffix would ship that distortion into the output and be scored as a
    # generation flaw. Per class, because benign and phishing genuinely name on
    # different TLDs (total variation 0.579).
    table_path = bundle / "suffix_table.json"
    if table_path.is_file():
        suffixes = SuffixTable.load(table_path)
    else:
        train_files = [bundle / f"{s}.jsonl.gz" for s in
                       ("name_train", "structure_train",
                        "phishing_name_train", "phishing_structure_train")]
        suffixes = SuffixTable.from_files([f for f in train_files if f.is_file()])
        suffixes.save(table_path)
    for cls in suffixes.classes():
        print(f"  suffix table  {suffixes.summary(cls)}")

    # `delta` is recorded so the artifact says which CONFIGURATION it scored.
    # The previous report named only a checkpoint, which made an E5 run and an
    # E5+delta run indistinguishable on disk.
    report: Dict[str, object] = {"checkpoint": str(checkpoint), "seed": seed,
                                 "excess_bar": EXCESS_BAR,
                                 "delta": dict(delta or {}),
                                 "generations": generations,
                                 "disc_seeds": disc_seeds,
                                 "decoder": decoder,
                                 "decoder_kwargs": dict(decoder_kwargs or {}),
                                 "suffix_table": str(table_path)}
    if classes:
        unknown = sorted(set(classes) - set(tokenizer.classes))
        if unknown:
            raise ValueError(f"unknown classes {unknown}; "
                             f"known: {list(tokenizer.classes)}")
        if estimator == "v1" and set(classes) != set(tokenizer.classes):
            print("  NOTE: v1 draws the precision/recall sample from one RNG "
                  "shared across classes, so a subset run will not reproduce "
                  "a combined v1 report's auxiliary metrics exactly.")
    if estimator == "v2":
        # Recorded only under v2, so a v1 report stays byte-for-byte the
        # document it always was.
        report["estimator"] = {
            "version": "v2",
            "floor_seeding": "independent per (generation, split): "
                             "seed + 100000 + 1000 g + d",
            "generations": generations, "disc_seeds": disc_seeds,
            "se": "sd(generation means) / sqrt(G), sample sd",
            "t_crit": round(t_crit(generations - 1), 4)
                      if generations >= 2 else None,
            "acceptance": "section-26.1 t-test; the envelope is printed but "
                          "is not the verdict",
            "pooled": "rule unchanged (TELL if min > bar, never ok); seeds "
                      "seed + d; d3_verdict = the same rule on the first 3",
            "registration": "research/qry/PREREG_GATE_V2.md"}
        print(f"  estimator v2: independent floor per (generation, split), "
              f"G={generations} x D={disc_seeds}, SE = sd/sqrt(G)")

    for class_index, label in enumerate(tokenizer.classes):
        if classes and label not in classes:
            continue
        if estimator == "v2":
            # v1 draws the precision/recall sample from ONE RNG created before
            # this loop, so a phishing-only run would draw differently from a
            # combined run. Per class, runs can be split to bound memory.
            # Auxiliary metric only; never gated.
            rng = random.Random(seed + 200_000 + class_index)
        populations = by_class[label]
        print(f"\n=== {label} ===", flush=True)
        started = time.time()
        d = float((delta or {}).get(label, 0.0))
        # Scoped to the path field only, exactly as adopted: a global EOS bias
        # was measured to move host composition it has no business touching.
        fb = {"PTH": {tokenizer.eos_id: -d}} if d else None
        if d:
            print(f"  delta {d} on PTH EOS (adopted decode-time bias)")
        # MULTI-DRAW. One draw cannot classify a field against a 0.05 bar when
        # the instrument's own split-seed spread on IDENTICAL data runs
        # 0.0419 / 0.1311 / 0.0836 -- a range of 0.089, nearly twice the bar.
        # 26.1 established G x D for the research harness; the official gate
        # never received the same treatment, so every gate verdict in this
        # project's history has rested on a single draw. G=1, D=1 reproduces
        # that exact behaviour, so the default path is unchanged.
        draws: Dict[str, List[dict]] = {}
        gate_report = None
        fake = None
        every: List[str] = []               # every generation, for the pooled floor
        for g in range(generations):
            # g=0 keeps `seed + 1`, which is what makes G=1 bit-identical.
            gen_seed = seed + 1 + g * 1000
            # `decoder` is a PARAMETER so the canonical gate can score an
            # alternative sampling rule on the SAME instrument. Default
            # "typical" with no kwargs is bit-identical to every published run.
            # Judging R4 through a separate harness would have broken
            # comparability with the baseline artifacts, which is the entire
            # reason for using the canonical gate to judge it.
            cache_file = (gen_cache_path(Path(gen_cache), ckpt_tag, label,
                                         n_main, decoder, decoder_kwargs, d,
                                         gen_seed) if gen_cache else None)
            if cache_file is not None and cache_file.is_file():
                fake_g = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                fake_g = generate_with_suffixes(model, tokenizer, label=label,
                                                n=n_main, table=suffixes,
                                                decoder=decoder, seed=gen_seed,
                                                device=device, field_bias=fb,
                                                decoder_kwargs=decoder_kwargs
                                                or None)
                if cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(fake_g), encoding="utf-8")
            if fake is None:                    # first draw feeds the novelty
                fake = fake_g                   # gate, embeddings and diagnostic
                # The AUXILIARY block is held at N_MAIN however large `n_main`
                # grows. Two reasons, and neither is a shortcut. (1) Improved
                # precision/recall compares every generated point against every
                # real one, so it is O(n^2) in both time and memory -- at
                # n_main = 36,000 that is a 36k x 36k distance block, which does
                # not fit and would abort the run before a single field was
                # scored. (2) COMPARABILITY: the gate pass rate, novelty rate,
                # precision/recall and diversity numbers already published were
                # all measured on 1,200 samples, and a figure computed on 30x
                # the data is not the same statistic -- distinct-4gram ratio in
                # particular falls monotonically with set size by construction.
                # `n_main` is raised to buy the FIELD discriminator more
                # field-bearing rows; nothing else in this function asked for
                # it. At n_main = N_MAIN this slice is the whole list, so the
                # canonical 1,200-sample run stays bit-identical.
                aux = fake_g[:N_MAIN]
                gate = Gate(known_membership=membership)
                kept, gate_report = gate.filter(aux)
            every.extend(fake_g)
            for dsi in range(disc_seeds):
                r = field_report(populations, fake_g, n=min(n_main, 1200),
                                 seed=split_seed(seed, g, dsi, estimator))
                # MEMORY, not maths. Each call builds a few very large, cyclic
                # sklearn objects, and the generational collector reaches them
                # too rarely: measured on the benign pool, resident memory rose
                # ~50 MB per call without this and plateaued with it. Collecting
                # frees only unreachable objects, so no value can change.
                gc.collect()
                for name, row in r.items():
                    draws.setdefault(name, []).append(row)
            print(f"    draw {g + 1}/{generations} x {disc_seeds} split "
                  f"[{time.time() - started:.0f}s]", flush=True)

        fields = _aggregate(draws)
        if estimator == "v2":
            # draws[name] holds G x D rows in generation-major order.
            for name, rows in draws.items():
                by_gen = [[r["excess"] for r in
                           rows[g * disc_seeds:(g + 1) * disc_seeds]
                           if not r.get("insufficient")]
                          for g in range(generations)]
                acc = s261_verdict(by_gen)
                if name == "FULL":
                    acc["verdict"] = "AUXILIARY"
                fields[name]["acceptance"] = acc
        print(f"  {'field':<6}{'pop':<11}{'AUC':>8}{'floor':>8}"
              f"{'excess':>9}{'sd':>8}{'min':>9}{'max':>9}{'dr':>4}   "
              + ("verdict" if estimator == "v1"
                 else "envelope -- NOT the verdict"))
        for name, r in fields.items():
            if r.get("insufficient"):
                print(f"  {name:<6}{r['population']:<11}"
                      f"{'--':>8}{'--':>8}{'--':>9}{'--':>8}{'--':>9}{'--':>9}"
                      f"{r.get('draws', 0):>4}   insufficient (n_fake "
                      f"{r.get('n_fake', 0)})")
                continue
            # Judged on the MEAN against the bar, with the observed spread
            # beside it. A field whose spread STRADDLES the bar is reported
            # UNDECIDED rather than forced to a side the evidence cannot carry.
            tag = "AUXILIARY" if name == "FULL" else (
                "ok" if r["excess_max"] < EXCESS_BAR else
                ("TELL" if r["excess_min"] > EXCESS_BAR else "UNDECIDED"))
            print(f"  {name:<6}{r['population']:<11}{r['auc']:>8.4f}"
                  f"{r['floor']:>8.4f}{r['excess']:>+9.4f}{r['excess_sd']:>8.4f}"
                  f"{r['excess_min']:>+9.4f}{r['excess_max']:>+9.4f}"
                  f"{r['draws']:>4}   {tag}", flush=True)

        if estimator == "v2":
            tc_txt = (f"t_crit {t_crit(generations - 1):.3f} "
                      f"(df {generations - 1})" if generations >= 2
                      else "needs G >= 2")
            print(f"  -- ACCEPTANCE: section-26.1 t-test, SE = sd(generation "
                  f"means)/sqrt(G), {tc_txt} --")
            print(f"  {'field':<6}{'x':>9}{'SE':>8}{'t':>8}   verdict")
            for name, r in fields.items():
                a = r.get("acceptance", {})
                if "x" not in a:
                    print(f"  {name:<6}{'--':>9}{'--':>8}{'--':>8}   "
                          f"NOT COMPUTABLE ({a.get('generations_usable', 0)}"
                          f"/{a.get('generations', generations)} generations "
                          f"usable)")
                    continue
                t_txt = (f"{a['t_vs_bar']:>+8.2f}"
                         if isinstance(a["t_vs_bar"], float)
                         and math.isfinite(a["t_vs_bar"]) else f"{'inf':>8}")
                print(f"  {name:<6}{a['x']:>+9.4f}{a['se']:>8.4f}{t_txt}   "
                      f"{a['verdict']}", flush=True)

        # POOLED FALLBACK for fields the generator emits too rarely to measure.
        #
        # A field is blanked when fewer than MIN_N = 250 generated rows carry
        # it. Query is emitted on 1.06% of phishing rows and 2.76% of benign
        # ones, so reaching 250 query-bearing rows IN A SINGLE GENERATION needs
        # n_main around 23,600 -- and at the measured 4.72 s per unit of n_main
        # across both configurations that is roughly two days of wall clock for
        # one cell. The G generations already exist, are already independent and
        # are already paid for; pooling them clears the same floor at 1/G the
        # generation cost.
        #
        # IT IS A DIFFERENT ESTIMATOR AND IS NOT REPORTED AS THE SAME ONE. Its
        # spread comes from D split seeds on ONE pooled set, so it cannot see
        # generation variance -- and 26.1 established that generation variance
        # is the dominant term, not the split. Its error bar is therefore known
        # to be understated, which cuts one way only:
        #
        #   TELL       still available -- an excess far above the bar survives
        #              an error bar that is too small, a fortiori.
        #   UNDECIDED  the default.
        #   "ok"       NEVER. "ok" is the claim that the spread stays below the
        #              bar, and this estimator does not measure that spread.
        #
        # So a pooled row can convict and can abstain. It cannot acquit.
        starved = sorted(k for k, v in fields.items() if v.get("insufficient"))
        pooled: Dict[str, dict] = {}
        if starved and len(every) > n_main:
            pdraws: Dict[str, List[dict]] = {}
            for dsi in range(disc_seeds):
                for name in starved:
                    pop = FIELD_POPULATION[name]
                    if not populations.get(pop):
                        continue
                    row = discriminate_with_floor(populations[pop], every,
                                                  name, n=min(n_main, 1200),
                                                  seed=seed + dsi)
                    gc.collect()                 # memory only; see above
                    row["population"] = pop
                    row["frame"] = GENERATOR_FRAME
                    pdraws.setdefault(name, []).append(row)
            pooled = _aggregate(pdraws)
            if estimator == "v2":
                for name, rows in pdraws.items():
                    if name in pooled:
                        pooled[name]["d3_verdict"] = pooled_verdict(
                            [r["excess"] for r in rows[:3]
                             if not r.get("insufficient")])
            for v in pooled.values():
                v["pooled_over_generations"] = generations
                v["n_fake_pooled"] = len(every)
            print(f"  -- pooled over {generations} generations "
                  f"({len(every):,} rows); split-seed spread ONLY, "
                  f"cannot acquit --")
            for name, r in pooled.items():
                if r.get("insufficient"):
                    print(f"  {name:<6}{r['population']:<11}"
                          f"{'--':>8}{'--':>8}{'--':>9}{'--':>8}"
                          f"{'--':>9}{'--':>9}{r.get('draws', 0):>4}   "
                          f"still insufficient (n_fake {r.get('n_fake', 0)})")
                    continue
                tag = "TELL" if r["excess_min"] > EXCESS_BAR else "UNDECIDED"
                print(f"  {name:<6}{r['population']:<11}{r['auc']:>8.4f}"
                      f"{r['floor']:>8.4f}{r['excess']:>+9.4f}"
                      f"{r['excess_sd']:>8.4f}{r['excess_min']:>+9.4f}"
                      f"{r['excess_max']:>+9.4f}{r['draws']:>4}   "
                      f"{tag} (pooled, n={r['n']})"
                      + (f"   [first 3 draws: {r.get('d3_verdict')}]"
                         if estimator == "v2" else ""), flush=True)

        # Cross-frame DIAGNOSTIC. The generator emits one URL per host, so it
        # samples the domain frame; scoring it against the page frame measures
        # the frame gap, not quality. Reported so the coverage is not lost —
        # "can the model produce multi-page-site paths" stays a live question —
        # but it never gates, because an unconditioned generator cannot pass it.
        # It becomes a real target once `<F:page>` conditioning exists.
        other = "page" if GENERATOR_FRAME == "domain" else "domain"
        cross = field_report(populations, fake, n=min(n_main, 1200), seed=seed,
                             frame=other)
        diag = {k: v for k, v in cross.items()
                if not v.get("insufficient")
                and v["population"] != fields.get(k, {}).get("population")}
        if diag:
            print(f"  -- diagnostic, {other} frame (NOT a gate) --")
            for name, r in diag.items():
                print(f"  {name:<6}{r['population']:<11}{r['auc']:>8.4f}"
                      f"{r['floor']:>8.4f}{r['excess']:>+9.4f}{r['se']:>8.4f}"
                      f"{r['n']:>7}   diagnostic")

        emb_real = embed(model, tokenizer,
                         rng.sample(populations["mixed"],
                                    min(len(populations["mixed"]), N_MAIN)),
                         label, device)
        emb_fake = embed(model, tokenizer, aux, label, device)
        pr = precision_recall(emb_real, emb_fake, k=5)

        # FIELD EMISSION RATES. The gate cannot see these.
        # `discriminate_with_floor` extracts a field and drops the empties on
        # BOTH sides, so it only ever compares rows that HAVE the field -- it
        # asks whether the queries the model writes look real, never whether
        # it writes them often enough. Measured here because the two answers
        # can disagree sharply: phishing query emission runs 1.06% generated
        # against 4.4% real, a 4.2x deficit that no value of `n_main` and no
        # excess bar would ever surface.
        rates = {}
        for fname, pop in FIELD_POPULATION.items():
            if fname == "FULL" or not populations.get(pop):
                continue
            ex = FIELDS[fname]
            gen_rate = sum(1 for u in every if ex(u)) / max(len(every), 1)
            real_rate = (sum(1 for u in populations[pop] if ex(u)) /
                         max(len(populations[pop]), 1))
            rates[fname] = {"generated": round(gen_rate, 5),
                            "real": round(real_rate, 5),
                            "ratio": (round(gen_rate / real_rate, 3)
                                      if real_rate else None),
                            "n_generated": len(every)}
        print("  -- field emission rates (NOT gated; the gate is "
              "field-conditional) --")
        for fname, r in rates.items():
            print(f"  {fname:<6}generated {100*r['generated']:>6.2f}%   "
                  f"real {100*r['real']:>6.2f}%   ratio {r['ratio']}")

        report[label] = {
            "fields": fields,
            "pooled_fields": pooled,
            "emission_rates": rates,
            "frame": GENERATOR_FRAME,
            "cross_frame_diagnostic": diag,
            "gate": {"pass_rate": round(gate_report.pass_rate, 4),
                     "novelty_rate": round(gate_report.novelty_rate, 4),
                     "by_reason": dict(gate_report.by_reason)},
            "precision_recall": pr,
            "diversity": diversity(aux),
            "examples": aux[:8],
            "n_main": n_main,
            "n_aux": len(aux),
        }
        print(f"  gate {100*gate_report.pass_rate:.1f}% pass, "
              f"{100*gate_report.novelty_rate:.1f}% novel   "
              f"precision {pr['precision']:.3f} recall {pr['recall']:.3f}")

    (out_dir / "eval_report.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir/'eval_report.json'}")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("data/best.pt"))
    ap.add_argument("--bundle", type=Path, default=Path("data/corpus2"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus2"))
    ap.add_argument("--n", type=int, default=N_MAIN)
    ap.add_argument("--delta-benign", type=float, default=0.0,
                    help="adopted path-scoped EOS bias for benign (27.2: 0.22)")
    ap.add_argument("--delta-phishing", type=float, default=0.0,
                    help="adopted path-scoped EOS bias for phishing (27.2: 0.30)")
    ap.add_argument("--generations", type=int, default=1,
                    help="independent generations; 26.1 used 5")
    ap.add_argument("--disc-seeds", type=int, default=1,
                    help="split seeds per generation; 26.1 used 3")
    # ADOPTED 2026-09-23 (R4). Pass --decoder typical to reproduce the
    # pre-adoption baselines in data/e5_gate_n8000_dadopted and earlier;
    # every report artifact records the decoder it actually ran under.
    ap.add_argument("--decoder", default="temperature",
                    help="sampling rule; ADOPTED default is temperature "
                         "(R4). Use --decoder typical for pre-2026-09-23 "
                         "baselines")
    ap.add_argument("--temperature", type=float, default=None,
                    help="passed to the decoder when it accepts one")
    # Gate v2 is the default (research/qry/PREREG_GATE_V2.md). Pass
    # --estimator v1 to reproduce any report published before 2026-09-23.
    ap.add_argument("--estimator", choices=ESTIMATORS, default="v2",
                    help="v2 (default): independent floor per (generation, "
                         "split) and the section-26.1 verdict in the gate; "
                         "v1: the pre-2026-09-23 gate, exactly")
    ap.add_argument("--gen-cache", type=Path, default=None,
                    help="directory memoising generations; value-neutral")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="score only these classes (bounds memory)")
    args = ap.parse_args()
    main(args.checkpoint, args.bundle, args.out, n_main=args.n,
         delta={"benign": args.delta_benign,
                "phishing": args.delta_phishing},
         generations=args.generations, disc_seeds=args.disc_seeds,
         decoder=args.decoder,
         decoder_kwargs=({"temperature": args.temperature}
                         if args.temperature is not None else None),
         estimator=args.estimator, gen_cache=args.gen_cache,
         classes=args.classes)
