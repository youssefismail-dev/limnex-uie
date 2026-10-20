"""E13 K13-2 -- DX8 re-measured on e13_ctx (research/e13/PREREG_E13.md section 3.1):
free phishing QRY, sampled after a REAL context by e13_ctx, against e13_ctx's own
generated queries, paired seed for seed. DX8's procedure exactly
(research/e13/e13d_measure.py part_context: the cached sampler, seed 20261005, one
sample per validation URL, the raw decode; QC1's harness, D = 12, DX1's n), with
e13_ctx's checkpoint and files in place of e12_final's.

    python research/e13/e13_dx8.py            # writes data/e13/dx8_ctx.json

The registered reading: K13-2 holds if the paired mean x_GV - x_RC is at most half
of e12_final's 0.0774 (E13D_RESULTS.md section 4).
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e13d_common as D  # noqa: E402
from e13d_measure import _sample_fields, sample  # noqa: E402

E13D = ROOT / "data" / "e13"
CELL = "free_phishing_QRY"


def main() -> None:
    import torch
    from uie.train import load_checkpoint
    torch.set_num_threads(4)
    t0 = time.time()
    cell = D.BY_NAME[CELL]
    name, mode, label, state, field = cell
    dx1 = json.loads((D.OUT / f"decomp_{name}.json").read_text(encoding="utf-8"))
    n = dx1["n"]
    model, tok, cfg = load_checkpoint(E13D / "train" / "e13_ctx" / "final.pt",
                                      device=torch.device("cpu"))
    model.eval()
    urls, val = D.pool_urls(cell, "val"), D.pool(cell, "val")
    assert len(urls) == len(val)
    rc, v10 = _sample_fields(model, tok, urls, label, field, cfg.max_len, 20261005, check=3)
    good = [s for s in rc if s]
    gen = D.generated(cell, "e13_ctx", gen=E13D / "ctx" / "gen")
    draws = []
    for s in D.SEEDS:
        draws.append(D.C.harness(val, {"gen": sample(gen, n, s + 101), "rc": sample(good, n, s + 909)},
                                 s, n))
        print(f"  seed {s} done", flush=True)
    ok = [d for d in draws if 0.42 <= d["floor"] <= 0.55]
    diffs = [d["gen"] - d["rc"] for d in ok]
    out = {"registration": "research/e13/PREREG_E13.md (section 3.1, K13-2)", "cell": name,
           "n": n, "urls": len(urls), "sampled_valid": len(good),
           "v10_max_abs_diff": max(v10) if v10 else None, "draws": draws, "void": len(draws) - len(ok),
           "x_gv": st.mean(d["gen"] for d in ok), "x_rc": st.mean(d["rc"] for d in ok),
           "diff": st.mean(diffs), "threshold": 0.5 * 0.0774,
           "seconds": round(time.time() - t0, 1)}
    out["pass"] = out["diff"] <= out["threshold"]
    (E13D / "dx8_ctx.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"x_GV {out['x_gv']:+.4f}  x_RC {out['x_rc']:+.4f}  gap {out['diff']:+.4f} "
          f"(K13-2 needs <= {out['threshold']:.4f}) -> {'PASS' if out['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
