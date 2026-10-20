"""
research/eval_pages/grade_laptop.py

Grades the 10-person pair study (human_eval_100_laptop.html) against its key.

    python research/eval_pages/grade_laptop.py data/eval_pages/answers/<export>.json

Each answer ("only_A_real", "only_B_real", "both_real", "both_generated") says, for both URLs
of its pair, whether the rater judged it real. So 50 answers give 100 per-URL judgements, one
per distinct URL. The headline question: are generated URLs judged real as often as real ones?

- per-URL accuracy (balanced: 50 real, 50 generated; chance 50%), with the rule stated before
  the study: 61 or more of 100 is better than chance (two-sided 5%);
- a permutation p-value that shuffles real/generated labels within each rater, so a rater who
  calls everything "real" cannot create or hide an effect;
- per class, per pair type, exact answers (chance 25%), and the raters' comments.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEY = ROOT / "data" / "eval_pages" / "keys" / "laptop_key.json"
JUDGED = {"only_A_real": (True, False), "only_B_real": (False, True),
          "both_real": (True, True), "both_generated": (False, False)}


def binom_two_sided(k: int, n: int) -> float:
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] + 1e-15))


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main(path: str) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ex = json.loads(Path(path).read_text(encoding="utf-8"))
    key = json.loads(KEY.read_text(encoding="utf-8"))
    problems = []
    if ex.get("set_sha256") != key["set_sha256"]:
        problems.append("the export is from a different set of URLs")
    if ex.get("placeholder"):
        problems.append("the export comes from the placeholder template")
    recs = ex.get("records", [])
    parts = [r["participant"] for r in recs]
    if sorted(parts) != list(range(1, 11)):
        problems.append(f"participants present: {sorted(parts)}")
    ids_seen = Counter(a["id"] for r in recs for a in r["answers"])
    if set(ids_seen) != set(key["items"]) or any(c != 1 for c in ids_seen.values()):
        problems.append("not every pair answered exactly once")
    for r in recs:
        for a in r["answers"]:
            if key["items"][a["id"]]["a"]["block"] != r["participant"]:
                problems.append(f"participant {r['participant']} answered a pair from another block")
            if a["answer"] not in JUDGED:
                problems.append(f"unknown answer {a['answer']}")
    print("VALIDITY:", "PASS" if not problems else "FAIL " + "; ".join(problems))

    # per-URL judgements
    rows = []          # (participant, class, is_real, judged_real)
    exact = []         # (participant, type, class, correct)
    comments = []
    for r in recs:
        for a in r["answers"]:
            k = key["items"][a["id"]]
            ja, jb = JUDGED[a["answer"]]
            ra, rb = k["a"]["source"] == "real", k["b"]["source"] == "real"
            cls = k["a"]["class"]
            rows += [(r["participant"], cls, ra, ja), (r["participant"], cls, rb, jb)]
            exact.append((r["participant"], k["type"], cls, (ja, jb) == (ra, rb)))
            if a.get("reason"):
                comments.append((k["type"], cls, a["answer"], (ra, rb), a["reason"]))

    def acc(rs):
        return sum(1 for _, _, real, judged in rs if real == judged), len(rs)

    k_all, n_all = acc(rows)
    print(f"\nPER-URL ACCURACY: {k_all}/{n_all} = {k_all / n_all:.0%}  "
          f"(95% CI {wilson(k_all, n_all)[0]:.0%}-{wilson(k_all, n_all)[1]:.0%}; "
          f"binomial p = {binom_two_sided(k_all, n_all):.3f}; better than chance needs 61+)")
    real_judged = sum(1 for *_, real, j in rows if real and j)
    gen_judged = sum(1 for *_, real, j in rows if not real and j)
    n_real = sum(1 for *_, real, _ in rows if real)
    n_gen = n_all - n_real
    print(f"  judged real: real URLs {real_judged}/{n_real} = {real_judged / n_real:.0%}; "
          f"generated URLs {gen_judged}/{n_gen} = {gen_judged / n_gen:.0%}")

    # permutation within rater
    rng = random.Random(20261015)
    by_p = defaultdict(list)
    for p, _, real, j in rows:
        by_p[p].append((real, j))
    obs = k_all
    hits = 0
    R = 20000
    for _ in range(R):
        tot = 0
        for lst in by_p.values():
            labels = [x[0] for x in lst]
            rng.shuffle(labels)
            tot += sum(1 for lab, (_, j) in zip(labels, lst) if lab == j)
        hits += abs(tot - n_all / 2) >= abs(obs - n_all / 2) - 1e-9
    print(f"  permutation p (labels shuffled within each rater, {R:,} draws) = {hits / R:.3f}")

    for cls in ("benign", "phishing"):
        k, n = acc([x for x in rows if x[1] == cls])
        rj = sum(1 for _, c, real, j in rows if c == cls and real and j)
        gj = sum(1 for _, c, real, j in rows if c == cls and not real and j)
        print(f"  {cls:<9} {k}/{n} = {k / n:.0%} (CI {wilson(k, n)[0]:.0%}-{wilson(k, n)[1]:.0%}); "
              f"judged real: real {rj}/25, generated {gj}/25")

    print("\nEXACT ANSWERS (all four options; chance about 25%):")
    k = sum(1 for *_, c in exact if c)
    print(f"  overall {k}/{len(exact)} = {k / len(exact):.0%}")
    for t in ("real-vs-generated", "both-real", "both-generated"):
        sub = [x for x in exact if x[1] == t]
        print(f"  {t:<18} {sum(1 for *_, c in sub if c)}/{len(sub)}")
    ans = Counter(a["answer"] for r in recs for a in r["answers"])
    print("  answers given:", dict(ans))
    mixed = [x for x in exact if x[1] == "real-vs-generated"]
    picked_side = defaultdict(int)
    for r in recs:
        for a in r["answers"]:
            kk = key["items"][a["id"]]
            if kk["type"] != "real-vs-generated" or a["answer"] not in ("only_A_real", "only_B_real"):
                continue
            real_side = "A" if kk["a"]["source"] == "real" else "B"
            picked_side["picked the real one" if a["answer"].endswith(real_side + "_real") else "picked the generated one"] += 1
    print(f"  mixed pairs answered with one side: {dict(picked_side)} (of {len(mixed)} mixed pairs)")

    print("\nPER RATER (per-URL accuracy out of 10):")
    bg = {r["participant"]: (r["background"], r["seen_project_urls_before"]) for r in recs}
    for p in sorted(by_p):
        print(f"  {p:>2}: {sum(1 for real, j in by_p[p] if real == j)}/10  background={bg[p][0]}, seen before={bg[p][1]}")

    print(f"\nCOMMENTS ({len(comments)}):")
    for t, cls, a, truth, txt in comments:
        print(f"  [{cls}, {t}, answered {a}] {txt}")


if __name__ == "__main__":
    main(sys.argv[1])
