"""Root-slash reference (DIAGNOSTIC, read-only): FULL (auxiliary) for E7's raw
free-mode generations (DC1's base arm), as generated and with the root slash
rendered on every URL that has no path body. Scored with the gate's own
field_report, restricted to FULL by passing only the 'mixed' population, with
the canonical seeds. The unrendered run must reproduce DC1's base FULL
(+0.2168 benign, +0.4018 phishing) or the replication is invalid.

Not a mechanism: nothing here is proposed for generation.

    python research/native/slash_reference.py benign      # and phishing

Result (2026-09-26): FULL benign +0.2168 -> +0.0720, phishing +0.4018 -> +0.0520;
both unrendered readings equal DC1's base arm exactly.
"""
import gc
import json
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research" / "cr1"))
from cr1_score import G, D, SEED, pools  # noqa: E402
from uie.corpus.run_eval import _aggregate, s261_verdict, split_seed  # noqa: E402
from uie.evaluation import _parse, _path_body, field_report  # noqa: E402

label = sys.argv[1]


def render(u: str) -> str:
    p = _parse(u)
    if _path_body(p.path or ""):
        return u
    cut = min([i for i in (u.find("?"), u.find("#")) if i >= 0] or [len(u)])
    return u if u[:cut].endswith("/") else u[:cut] + "/" + u[cut:]


mixed = pools(R / "data/bundle2", label)["mixed"]
gens = [json.loads((R / f"data/dc1/gen/free_base_{label}_g{g}.json").read_text(encoding="utf-8"))
        for g in range(G)]
out = {}
for variant in ("as_generated", "root_slash_rendered"):
    t0 = time.time()
    draws = []
    for g, fake in enumerate(gens):
        fake_g = fake if variant == "as_generated" else [render(u) for u in fake]
        for d in range(D):
            r = field_report({"mixed": mixed}, fake_g, n=1200, seed=split_seed(SEED, g, d, "v2"))
            gc.collect()
            draws.append(r["FULL"])
    agg = _aggregate({"FULL": draws})["FULL"]
    by_gen = [[x["excess"] for x in draws[g * D:(g + 1) * D] if not x.get("insufficient")] for g in range(G)]
    acc = s261_verdict(by_gen)
    changed = sum(render(u) != u for f in gens for u in f) if variant != "as_generated" else 0
    out[variant] = {"x": acc.get("x"), "se": acc.get("se"), "urls_changed": changed,
                    "urls": sum(len(f) for f in gens), "seconds": round(time.time() - t0)}
    print(label, variant, json.dumps(out[variant]), flush=True)
(R / "data/native").mkdir(parents=True, exist_ok=True)
(R / f"data/native/slash_reference_{label}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
