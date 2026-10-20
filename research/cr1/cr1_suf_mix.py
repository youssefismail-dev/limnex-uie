"""CR1, descriptive: does the TLD mix follow the presence control?

The suffix is drawn from the class-marginal table BEFORE the URL, by label only
(`uie/suffixes.py`, `table.sample(label, n, rng)`); the control never reaches
it. Under joint resampling the TLD mix can move only through rejections. This
measures how far it moves, against how far real URLs move between the same two
states, on the class's top-10 real TLDs plus "other" (a coarse space, so the
sampling-noise floor stays small), with that floor measured.

    python research/cr1/cr1_suf_mix.py --gens data/cr1/gen --bundle data/bundle2 --out data/cr1/suf_mix.json
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uie.evaluation import FIELDS, _one_per_host  # noqa: E402
from cr1_score import G, SEED, pools, tag_of  # noqa: E402

TOP, REPS, M_GEN = 10, 40, 7500


def tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    def gen(tag):
        return [u for g in range(G) for u in
                json.loads((args.gens / f"{tag}_g{g}.json").read_text(encoding="utf-8"))]

    out = {}
    for label in ("benign", "phishing"):
        marginal = _one_per_host(pools(args.bundle, label)["name"], SEED)
        top = [k for k, _ in Counter(FIELDS["SUF"](u) for u in marginal).most_common(TOP)]
        cover = sum(FIELDS["SUF"](u) in top for u in marginal) / len(marginal)

        def dist(urls):
            c = Counter(s if s in top else "other" for s in (FIELDS["SUF"](u) for u in urls))
            n = sum(c.values())
            return {k: v / n for k, v in c.items()}

        def noise(xs, m):
            """TV between two disjoint random samples of m from one pool: mean, max."""
            r, v = random.Random(SEED), []
            for _ in range(REPS):
                s = r.sample(xs, 2 * m)
                v.append(tv(dist(s[:m]), dist(s[m:])))
            return sum(v) / len(v), max(v)

        print(f"{label}: top-{TOP} TLDs cover {cover:.1%} of real hosts")
        for f in ("PTH", "QRY"):
            on, off = tag_of(label, {f: True}), tag_of(label, {f: False})
            r_on, r_off = (_one_per_host(pools(args.bundle, label, {f: w})["name"], SEED)
                           for w in (True, False))
            g_on, g_off = gen(on), gen(off)
            m = min(len(r_on), len(r_off)) // 2
            rn, gn = noise(r_on + r_off, m), noise(g_on + g_off, M_GEN)
            rs, gs = tv(dist(r_on), dist(r_off)), tv(dist(g_on), dist(g_off))
            row = {"real_shift": rs, "real_noise_mean": rn[0], "real_noise_max": rn[1], "m_real": m,
                   "gen_shift": gs, "gen_noise_mean": gn[0], "gen_noise_max": gn[1], "m_gen": M_GEN,
                   "top": top, "top_cover": cover}
            for tag, real, fake in ((on, r_on, g_on), (off, r_off, g_off)):
                row[tag] = {"gen_vs_real_state": tv(dist(fake), dist(real)),
                            "gen_vs_real_marginal": tv(dist(fake), dist(marginal)),
                            "real_state_vs_marginal": tv(dist(real), dist(marginal))}
            out[f"{label}_{f}"] = row
            print(f"  {f}: real shift {rs:.3f} (noise {rn[0]:.3f}, max {rn[1]:.3f})  "
                  f"generated shift {gs:.3f} (noise {gn[0]:.3f}, max {gn[1]:.3f})")
            for tag in (on, off):
                t = row[tag]
                print(f"     {tag:<13} generated vs real in state {t['gen_vs_real_state']:.3f}  "
                      f"vs real class marginal {t['gen_vs_real_marginal']:.3f}  "
                      f"(real state vs marginal {t['real_state_vs_marginal']:.3f})")
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
