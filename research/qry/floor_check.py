"""FL -- was benign QRY's gate reading a floor artifact?

Pre-registered in research/qry/PREREG_FLOOR.md (commit 3f544cc) before any
number here was computed. The gate itself is not changed.

    fl1  recompute the gate's three floors at its own split seeds (minutes)
    fl2  regenerate R4's five benign generations EXACTLY, verify by reproducing
         all 15 stored excess draws, then score with 30 split seeds (~2 h)

Single process on purpose: the NS sweep showed long-lived worker pools growing to
~3 GB each on this laptop.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---- registered (PREREG_FLOOR.md) -- do not edit post hoc ---------------------
SEED = 20260806                     # the gate's seed (R4 eval_report)
CKPT = Path("data/best_e5.pt")
N_MAIN = 8000
G = 5
DELTA = 0.22                        # benign, adopted
TEMPERATURE = 1.0
D_EXT = 30                          # split seeds SEED .. SEED+29
N_BAND = (330, 340, 350, 355, 360, 365, 370, 380, 390)
GATE_FLOOR = 0.5374                 # R4 benign QRY floor, mean of 15 draws
SWEEP_FLOOR = 0.4936                # NS, n=361, mean of 20 draws
BAR, TCRIT = 0.05, 2.776            # section 26.1, df = G-1 = 4
# ---------------------------------------------------------------------------------

B = Path("data/bundle2")
OUT = Path("data/qry")


def _below_normal() -> None:
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def _real_pool() -> list:
    from uie.corpus.run_eval import _find, _rows
    return _rows(_find(B, "name_val"))       # the gate's benign `name` pool


def fl1() -> dict:
    """The floor depends only on pool, n and seed -- the real triple is drawn
    before the fakes -- so the fake list here only fixes n."""
    from uie.evaluation import discriminate_with_floor
    real = _real_pool()
    fakes = json.loads(Path("data/qry/q4_benign_temperature_queries.json")
                       .read_text(encoding="utf-8"))
    band = {}
    for n in N_BAND:
        fl = []
        for dsi in range(3):
            r = discriminate_with_floor(real, fakes[:n], "QRY", n=1200,
                                        seed=SEED + dsi)
            assert r["n"] == n, (r["n"], n)
            fl.append(r["floor"])
        band[n] = {"floors": fl, "mean": round(st.mean(fl), 4)}
        print(f"  n={n}  floors {[round(x, 4) for x in fl]}  "
              f"mean {band[n]['mean']:.4f}", flush=True)
    f360 = band[360]["mean"]
    if abs(f360 - GATE_FLOOR) <= 0.01 and f360 - SWEEP_FLOOR >= 0.02:
        verdict = "H_UF SUPPORTED -- the gate's floors are reproduced, and high"
    elif abs(f360 - GATE_FLOOR) > 0.02:
        verdict = "NOT REPRODUCED -- escalate: pipeline discrepancy"
    else:
        verdict = "INCONCLUSIVE -- FL2 is exact"
    out = {"band": band, "F360": f360, "gate_floor": GATE_FLOOR,
           "sweep_floor": SWEEP_FLOOR, "verdict": verdict}
    print(f"\n  F(360) = {f360:.4f}   gate {GATE_FLOOR}   sweep {SWEEP_FLOOR}"
          f"\n  REGISTERED VERDICT: {verdict}", flush=True)
    return out


def _generate() -> list:
    import torch
    from uie.suffixes import SuffixTable, generate_with_suffixes
    from uie.train import load_checkpoint
    dev = torch.device("cpu")
    model, tok, _ = load_checkpoint(CKPT, device=dev)
    model.to(dev).eval()
    table = SuffixTable.load(B / "suffix_table.json")
    fb = {"PTH": {tok.eos_id: -DELTA}}      # exactly run_eval's construction
    gens = []
    for g in range(G):
        p = OUT / f"fl2_gen_g{g}.json"
        if p.is_file():
            gens.append(json.loads(p.read_text(encoding="utf-8")))
            print(f"  gen {g}: cached", flush=True)
            continue
        t0 = time.time()
        urls = generate_with_suffixes(model, tok, label="benign", n=N_MAIN,
                                      table=table, decoder="temperature",
                                      seed=SEED + 1 + g * 1000, device=dev,
                                      field_bias=fb,
                                      decoder_kwargs={"temperature": TEMPERATURE})
        p.write_text(json.dumps(urls), encoding="utf-8")
        gens.append(urls)
        print(f"  gen {g}: {len(urls)} urls  [{time.time() - t0:.0f}s]",
              flush=True)
    return gens


def _estimate(rows: list, d: int) -> dict:
    gm = [st.mean(r["excess"] for r in rows if r["g"] == g and r["dsi"] < d)
          for g in range(G)]
    x = st.mean(gm)
    se = st.stdev(gm) / math.sqrt(G - 1)
    if (x - BAR) / se > TCRIT:
        v = "CONVICTED"
    elif (BAR - x) / se > TCRIT:
        v = "CLEARED"
    else:
        v = "UNDECIDED"
    sub = [r for r in rows if r["dsi"] < d]
    return {"D": d, "x": round(x, 4), "se": round(se, 4),
            "t_vs_bar": round((x - BAR) / se, 2), "verdict": v,
            "gen_means": [round(m, 4) for m in gm],
            "floor_mean": round(st.mean(r["floor"] for r in sub), 4),
            "auc_mean": round(st.mean(r["auc"] for r in sub), 4)}


def fl2() -> dict:
    from uie.evaluation import discriminate_with_floor
    gens = _generate()
    real = _real_pool()
    stored = json.load(open("data/r4_temp_dadopted/eval_report.json"))[
        "benign"]["fields"]["QRY"]["excess_draws"]

    draws_path = OUT / "fl2_draws.jsonl"
    done = {}
    if draws_path.is_file():
        for l in draws_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done[(r["g"], r["dsi"])] = r

    def score(g: int, dsi: int, fh) -> dict:
        if (g, dsi) in done:
            return done[(g, dsi)]
        r = discriminate_with_floor(real, gens[g], "QRY", n=1200, seed=SEED + dsi)
        row = {"g": g, "dsi": dsi, "n": r["n"], "auc": r["auc"],
               "floor": r["floor"], "excess": r["excess"]}
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        done[(g, dsi)] = row
        return row

    with draws_path.open("a", encoding="utf-8") as fh:
        # ---- reproduction kill criterion: all 15 stored draws, 4 decimals ----
        mism = []
        for g in range(G):
            for dsi in range(3):
                row = score(g, dsi, fh)
                want = stored[g * 3 + dsi]
                if round(row["excess"], 4) != round(want, 4):
                    mism.append((g, dsi, row["excess"], want))
        ns = [done[(g, 0)]["n"] for g in range(G)]
        print(f"  reproduction: {15 - len(mism)}/15 draws match R4; "
              f"per-generation n {ns}", flush=True)
        if mism:
            out = {"verdict": "VOID -- regeneration is not the gate's cell",
                   "mismatches": mism, "per_generation_n": ns}
            print(f"  REGISTERED VERDICT: {out['verdict']}  {mism}", flush=True)
            return out
        # ---- extended scoring ----
        t0 = time.time()
        for g in range(G):
            for dsi in range(3, D_EXT):
                score(g, dsi, fh)
            print(f"  gen {g}: {D_EXT} splits scored  [{time.time() - t0:.0f}s]",
                  flush=True)

    rows = list(done.values())
    d3, d30 = _estimate(rows, 3), _estimate(rows, D_EXT)
    out = {"reproduced": "15/15", "per_generation_n": ns,
           "gate_D3": d3, "extended_D30": d30,
           "verdict": d30["verdict"]}
    print(f"\n  D=3  (gate)  x {d3['x']:+.4f}  SE {d3['se']:.4f}  floor "
          f"{d3['floor_mean']:.4f}  AUC {d3['auc_mean']:.4f}  -> {d3['verdict']}")
    print(f"  D=30         x {d30['x']:+.4f}  SE {d30['se']:.4f}  floor "
          f"{d30['floor_mean']:.4f}  AUC {d30['auc_mean']:.4f}  -> {d30['verdict']}")
    print(f"  generation means (D=30): {d30['gen_means']}")
    print(f"\n  REGISTERED VERDICT (D=30, section-26.1 rule): {d30['verdict']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fl1", "fl2"])
    a = ap.parse_args()
    _below_normal()
    OUT.mkdir(parents=True, exist_ok=True)
    res = fl1() if a.step == "fl1" else fl2()
    p = OUT / f"{a.step}.json"
    p.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
