"""NS -- is the gate's excess estimator biased at reduced n?

Pre-registered in research/qry/PREREG_NSWEEP.md (commit eee2194) BEFORE any
number here was computed. Read-only: frozen artifacts, the gate's instrument
`discriminate_with_floor` called unmodified and exactly as run_eval calls it,
no generation, no training.

The two positions agree under the null and differ only where there is a tell:

    H_WNW  "weaker, not wrong" -- E[excess] is flat in n; reduced n adds variance
    H_LC   learning-curve bias -- E[excess] rises with n for fields with a tell

AUC and floor are recorded separately on every draw, because excess is their
difference and n can act on each term independently.
"""
from __future__ import annotations

import os

# One BLAS/OpenMP thread per worker, set before numpy is imported anywhere. Left
# unset, every worker fans out across all cores -- the oversubscription that lagged
# this laptop on 2026-09-22 (50 threads on 12 cores).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                             # noqa: E402
import json                                                 # noqa: E402
import math                                                 # noqa: E402
import statistics as st                                     # noqa: E402
import sys                                                  # noqa: E402
import time                                                 # noqa: E402
from multiprocessing import Pool                            # noqa: E402
from pathlib import Path                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---- registered design (PREREG_NSWEEP.md section 4) -- do not edit post hoc ----
R = 20
SEEDS = tuple(20260923 + i for i in range(R))
N_GRID = (250, 400, 600, 800, 1000, 1200)
MAIN_FIELDS = ("SUF", "LBL", "SUB", "PTH", "FULL")
QRY_GRID = {"benign": (150, 200, 250, 300, 361, 420, 476),
            "phishing": (150, 200, 250, 300, 356)}
ROLE = {("benign", "SUF"): "null", ("phishing", "SUF"): "null",
        ("benign", "LBL"): "weak", ("benign", "SUB"): "weak",
        ("phishing", "LBL"): "weak",
        ("benign", "PTH"): "bar", ("phishing", "PTH"): "bar",
        ("phishing", "SUB"): "bar",
        ("benign", "FULL"): "strong", ("phishing", "FULL"): "strong"}
LN_RATIO = math.log(1200 / 250)          # Delta = slope x ln(1200/250)
GATE_QRY_FLOOR = 0.5374                  # PREREG section 3(b), benign QRY, n~361
GATE_QRY_DRAWS = 15                      # G=5 x D=3 draws behind that mean
# -------------------------------------------------------------------------------

B = Path("data/bundle2")
OUT = Path("data/qry")
_W: dict = {}


def _below_normal() -> None:
    """Keep the laptop responsive while the user is on it."""
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def _free_mb() -> float:
    """Free physical RAM. The watchdog below stops the run before the laptop
    swaps: on 2026-09-23 each long-lived worker grew to ~3 GB resident, taking
    free RAM to 352 MB and lagging the machine. Growth was cumulative across
    tasks, which is why workers are also recycled (`--max-tasks-per-child`)."""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 20
    except Exception:
        return float("inf")


def _load() -> dict:
    # Pools are built with run_eval's OWN readers, so the real side is exactly
    # the gate's: `name` per class, `mixed` = name + structure for FULL.
    from uie.corpus.run_eval import _find, _rows
    data = {}
    for cls in ("benign", "phishing"):
        pre = "" if cls == "benign" else f"{cls}_"
        name = _rows(_find(B, f"{pre}name_val"))
        struct = _rows(_find(B, f"{pre}structure_val"))
        cache: list = []
        for p in sorted(Path("data/gen_cache").glob(
                f"best_e5_{cls}_n1800_*.json")):
            cache += json.loads(p.read_text(encoding="utf-8"))
        q4 = {arm: json.loads(Path(f"data/qry/q4_{cls}_{arm}_queries.json")
                              .read_text(encoding="utf-8"))
              for arm in ("typical", "temperature")}
        data[cls] = {"name": name, "mixed": name + struct,
                     "cache": cache, "q4": q4}
    return data


def _init() -> None:
    _below_normal()
    _W["data"] = _load()


def _task(t: tuple) -> dict:
    from uie.evaluation import FIELD_POPULATION, discriminate_with_floor
    source, cls, field, arm, n, seed = t
    d = _W["data"][cls]
    if source == "main":
        real, fake = d[FIELD_POPULATION[field]], d["cache"]
    else:
        real, fake = d["name"], d["q4"][arm]
    t0 = time.time()
    r = discriminate_with_floor(real, fake, field, n=n, seed=seed)
    return {"source": source, "cls": cls, "field": field, "arm": arm,
            "n_req": n, "n": r.get("n"), "reduced_n": r.get("reduced_n"),
            "insufficient": bool(r.get("insufficient")),
            "auc": r.get("auc"), "floor": r.get("floor"),
            "excess": r.get("excess"), "seed": seed,
            "seconds": round(time.time() - t0, 2)}


def tasks() -> list:
    out = []
    for cls in ("benign", "phishing"):
        for field in MAIN_FIELDS:
            for n in N_GRID:
                for s in SEEDS:
                    out.append(("main", cls, field, None, n, s))
        for arm in ("typical", "temperature"):
            for n in QRY_GRID[cls]:
                for s in SEEDS:
                    out.append(("qry", cls, "QRY", arm, n, s))
    return out


def _key(row) -> tuple:
    if isinstance(row, dict):
        return (row["source"], row["cls"], row["field"], row["arm"],
                row["n_req"], row["seed"])
    return tuple(row)


# ------------------------------------------------------------------ analysis --

def ols(xs: list, ys: list) -> dict:
    from scipy import stats
    m = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(rss / (m - 2) / sxx)
    tc = float(stats.t.ppf(0.975, m - 2))
    return {"slope": b, "se": se, "t_crit": tc, "n_draws": m}


def fit(rows: list, metric: str, span: float) -> dict:
    xs = [math.log(r["n"]) for r in rows]
    ys = [r[metric] for r in rows]
    f = ols(xs, ys)
    return {"delta": round(f["slope"] * span, 4),
            "lo": round((f["slope"] - f["t_crit"] * f["se"]) * span, 4),
            "hi": round((f["slope"] + f["t_crit"] * f["se"]) * span, 4),
            "n_draws": f["n_draws"]}


def per_n(rows: list) -> dict:
    out = {}
    for n in sorted({r["n"] for r in rows}):
        g = [r for r in rows if r["n"] == n]
        ex = [r["excess"] for r in g]
        out[n] = {"draws": len(g),
                  "excess": round(st.mean(ex), 4),
                  "excess_se": round(st.stdev(ex) / math.sqrt(len(ex)), 4)
                  if len(ex) > 1 else None,
                  "auc": round(st.mean(r["auc"] for r in g), 4),
                  "floor": round(st.mean(r["floor"] for r in g), 4)}
    return out


def verdict(fits: dict) -> dict:
    """The registered decision rule, PREREG_NSWEEP.md section 6, verbatim."""
    ex = {k: v["excess"] for k, v in fits.items()}
    null = [ex[k] for k in ex if ROLE[k] == "null"]
    weak = [ex[k] for k in ex if ROLE[k] == "weak"]
    bar = [ex[k] for k in ex if ROLE[k] == "bar"]
    valid = all(abs(f["delta"]) < 0.01 and f["lo"] <= 0 <= f["hi"]
                for f in null)
    wnw = all(f["delta"] < 0.01 and f["hi"] < 0.02 for f in bar)
    lc_hits = sum(1 for f in bar if f["delta"] >= 0.02 and f["lo"] > 0)
    mean = lambda fs: st.mean(f["delta"] for f in fs)          # noqa: E731
    ordering = mean(null) < mean(weak) < mean(bar)
    if not valid:
        v = "VOID -- the null control drifts with n"
    elif wnw:
        v = "H_WNW -- weaker, not wrong"
    elif lc_hits >= 2 and ordering:
        v = "H_LC -- biased low at small n"
    else:
        v = "INTERMEDIATE -- neither claimed"
    # Not anticipated by the registered rule; reported, never used to decide.
    unanticipated = [k for k in ex if ROLE[k] == "bar"
                     and ex[k]["delta"] < -0.01]
    return {"verdict": v, "validity_ok": valid, "wnw_all_bar": wnw,
            "lc_hits_of_3": lc_hits, "dose_response_ordering": ordering,
            "mean_delta": {"null": round(mean(null), 4),
                           "weak": round(mean(weak), 4),
                           "bar": round(mean(bar), 4)},
            "unanticipated_negative_bar": [f"{c} {f}" for c, f in unanticipated]}


def analyse(draw_path: Path) -> dict:
    rows = [json.loads(l) for l in draw_path.read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    bad = [r for r in rows if r["insufficient"] or r["reduced_n"]]
    rows = [r for r in rows if not r["insufficient"]]
    out: dict = {"draws": len(rows), "insufficient_or_reduced": len(bad),
                 "main": {}, "qry": {}}

    fits = {}
    for cls in ("benign", "phishing"):
        for field in MAIN_FIELDS:
            g = [r for r in rows if r["source"] == "main"
                 and r["cls"] == cls and r["field"] == field]
            fits[(cls, field)] = {m: fit(g, m, LN_RATIO)
                                  for m in ("excess", "auc", "floor")}
            out["main"][f"{cls} {field}"] = {
                "role": ROLE[(cls, field)], **fits[(cls, field)],
                "per_n": per_n(g)}
    out["decision"] = verdict(fits)

    for cls in ("benign", "phishing"):
        for arm in ("typical", "temperature"):
            g = [r for r in rows if r["source"] == "qry"
                 and r["cls"] == cls and r["arm"] == arm]
            span = math.log(max(r["n"] for r in g) / min(r["n"] for r in g))
            out["qry"][f"{cls} {arm}"] = {
                "span_ln": round(span, 4),
                **{m: fit(g, m, span) for m in ("excess", "auc", "floor")},
                "per_n": per_n(g)}

    # The floor anomaly (PREREG 3b). The floor depends only on pool, n and seed
    # -- the real triple is drawn before the fakes -- so both arms share it.
    fl = [r["floor"] for r in rows if r["source"] == "qry"
          and r["cls"] == "benign" and r["arm"] == "typical" and r["n"] == 361]
    if len(fl) > 1:
        mu, sd = st.mean(fl), st.stdev(fl)
        out["floor_anomaly"] = {
            "gate_floor_mean": GATE_QRY_FLOOR, "sweep_draws_at_361": len(fl),
            "sweep_floor_mean": round(mu, 4), "sweep_floor_sd": round(sd, 4),
            "sweep_floor_min": round(min(fl), 4),
            "sweep_floor_max": round(max(fl), 4),
            "z_of_gate_mean_as_15_draw_mean":
                round((GATE_QRY_FLOOR - mu) / (sd / math.sqrt(GATE_QRY_DRAWS)), 2)}
    return out


def report(s: dict) -> None:
    print(f"\ndraws {s['draws']}   insufficient/reduced {s['insufficient_or_reduced']}")
    print("\n=== MAIN GRID  Delta = E[excess](1200) - E[excess](250), OLS on ln n ===")
    print(f"  {'cell':<15}{'role':<8}{'Delta excess [95% CI]':>28}"
          f"{'Delta AUC':>11}{'Delta floor':>13}")
    for k, v in s["main"].items():
        e, a, f = v["excess"], v["auc"], v["floor"]
        print(f"  {k:<15}{v['role']:<8}"
              f"{e['delta']:>+10.4f} [{e['lo']:+.4f},{e['hi']:+.4f}]"
              f"{a['delta']:>+11.4f}{f['delta']:>+13.4f}")
    d = s["decision"]
    print(f"\n  validity (null control flat)   {d['validity_ok']}")
    print(f"  WNW on all three bar-scale     {d['wnw_all_bar']}")
    print(f"  LC hits of 3                   {d['lc_hits_of_3']}")
    print(f"  ordering null < weak < bar     {d['dose_response_ordering']}"
          f"   {d['mean_delta']}")
    if d["unanticipated_negative_bar"]:
        print(f"  NOT ANTICIPATED by the rule: negative bar-scale Delta in "
              f"{d['unanticipated_negative_bar']}")
    print(f"\n  REGISTERED VERDICT:  {d['verdict']}")

    print("\n=== QRY (secondary, descriptive) -- Delta over each arm's own range ===")
    for k, v in s["qry"].items():
        e = v["excess"]
        pn = " ".join(f"{n}:{x['excess']:+.3f}" for n, x in v["per_n"].items())
        print(f"  {k:<22} Delta {e['delta']:+.4f} [{e['lo']:+.4f},{e['hi']:+.4f}]"
              f"  floor Delta {v['floor']['delta']:+.4f}")
        print(f"      per n: {pn}")
    if "floor_anomaly" in s:
        fa = s["floor_anomaly"]
        print(f"\n  floor anomaly: gate mean {fa['gate_floor_mean']} vs sweep "
              f"{fa['sweep_floor_mean']} (sd {fa['sweep_floor_sd']}, range "
              f"{fa['sweep_floor_min']}-{fa['sweep_floor_max']}, "
              f"{fa['sweep_draws_at_361']} draws); z as a 15-draw mean "
              f"{fa['z_of_gate_mean_as_15_draw_mean']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: run only the first N tasks, to a "
                         "separate file that the analysis never reads")
    ap.add_argument("--analyse-only", action="store_true")
    # Operational only -- neither changes any computed value. Each draw depends
    # on (task, seed) alone, verified by 4 draws reproducing to the last digit
    # across separate runs on different workers.
    ap.add_argument("--max-tasks-per-child", type=int, default=25,
                    help="recycle each worker after this many tasks")
    ap.add_argument("--min-free-mb", type=int, default=2500,
                    help="stop cleanly if free RAM falls below this")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    draw_path = OUT / ("nsweep_smoke.jsonl" if args.limit else "nsweep_draws.jsonl")

    if not args.analyse_only:
        todo = tasks()
        if args.limit:
            # a spread of cheap and expensive tasks, not the first N in order
            todo = todo[::max(1, len(todo) // args.limit)][:args.limit]
            draw_path.unlink(missing_ok=True)
        done = set()
        if draw_path.is_file():
            done = {_key(json.loads(l)) for l in draw_path.read_text(
                encoding="utf-8").splitlines() if l.strip()}
        todo = [t for t in todo if _key(t) not in done]
        print(f"tasks {len(todo)} to run ({len(done)} already on disk), "
              f"{args.workers} workers", flush=True)
        _below_normal()
        t0 = time.time()
        low_memory = False
        with Pool(args.workers, initializer=_init,
                  maxtasksperchild=args.max_tasks_per_child or None) as pool, \
                draw_path.open("a", encoding="utf-8") as fh:
            for i, row in enumerate(pool.imap_unordered(_task, todo, 1), 1):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                free = _free_mb()
                if free < args.min_free_mb:
                    low_memory = True
                    print(f"  STOPPING: free RAM {free:.0f} MB < "
                          f"{args.min_free_mb} MB after {i} tasks. Everything "
                          f"written so far is kept; rerun to resume.", flush=True)
                    break
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i:>5}/{len(todo)}  {el:>6.0f}s  "
                          f"eta {el / i * (len(todo) - i):>5.0f}s  "
                          f"free RAM {free:>6.0f} MB", flush=True)
        if low_memory:
            sys.exit(3)
        if args.limit:
            print(f"smoke done -> {draw_path} (never analysed)")
            return

    s = analyse(draw_path)
    (OUT / "nsweep_summary.json").write_text(
        json.dumps(s, indent=1, default=str) + "\n", encoding="utf-8")
    report(s)
    print(f"\nwrote {OUT / 'nsweep_summary.json'}")


if __name__ == "__main__":
    main()
