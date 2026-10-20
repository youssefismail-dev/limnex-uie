"""E11 amendment 3 -- the reading at step 20,000 beside the best.pt reading, and
amendment 3's predictions scored (PREREG_E11.md section 3.3). Reads the reports
e11_compare.py and e11_score.py wrote for both readings; computes nothing new.

    python research/native/e11_side_by_side.py > data/e11_final/side_by_side.md

Refuses to run unless both readings' VERIFY.json say PASS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dc1"))

import e11_common as K  # noqa: E402
from e8_compare import GATED, rate  # noqa: E402

ARMS = tuple(K.ARMS)
BEST, FINAL = ROOT / "data" / "e11", ROOT / "data" / "e11_final"
CRIT = ("K11a", "K11b", "K11c", "K11d", "K11e")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def reading(d: Path) -> tuple:
    v = load(d / "VERIFY.json")
    if v.get("verdict") != "PASS":
        raise SystemExit(f"STOP: {d / 'VERIFY.json'} is not PASS; nothing is read")
    return load(d / "score_report.json"), load(d / "compare_report.json")


def acc(rep, mode, arm, key, field):
    a = (((rep.get(mode, {}).get(arm, {}).get(key) or {}).get("fields") or {})
         .get(field) or {}).get("acceptance", {})
    x = a.get("x")
    return (None if x is None or x != x else float(x)), a.get("verdict")


def pct(r, d=2) -> str:
    return "—" if r is None else f"{100 * r:.{d}f}%"


def cellx(x, v) -> str:
    return (v or "—") if x is None else f"{x:+.4f} {(v or '')[:4]}"


def pts(a, b) -> str:
    return "—" if a is None or b is None else f"{100 * (b - a):+.2f}"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=Path, default=FINAL,
                    help="the step-20,000 reading (default data/e11_final)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # a redirected Windows console is cp1252
    (rb, cb), (rf, cf) = reading(BEST), reading(args.final)
    tprov = load(BEST / "train" / "PROVENANCE.json")["arms"]
    step = {a: min(load(BEST / "train" / a / "train_result.json")["history"],
                   key=lambda h: h["mean"])["step"] for a in ARMS}

    print("### The criteria at both checkpoints\n\n| arm | best.pt step | "
          + " | ".join(CRIT) + " | candidate |\n|---|---|" + "---|" * (len(CRIT) + 1))
    for arm in ARMS:
        row = []
        for k in CRIT:
            b, f = cb["arms"][arm][k], cf["arms"][arm][k]
            row.append("—" if b is None else
                       f"{'PASS' if b['pass'] else 'FAIL'} → {'PASS' if f['pass'] else '**FAIL**'}")
        print(f"| {arm} | {step[arm]:,} | " + " | ".join(row) + " | "
              f"{'yes' if cb['arms'][arm]['candidate'] else 'no'} → "
              f"**{'yes' if cf['arms'][arm]['candidate'] else 'no'}** |")
    print("\nEach cell reads best.pt → final.pt (step 20,000). K11e is a training record, "
          "the same for both.")

    for code, title in (("K11a", "free-mode presence"), ("K11b", "the root slash")):
        print(f"\n### {code} — {title}, best.pt → final.pt\n\n| arm | row | best.pt | final.pt "
              f"| change (points) | real | T | within |\n|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            for name, xf in cf["arms"][arm][code]["rows"].items():
                xb = cb["arms"][arm][code]["rows"][name]
                print(f"| {arm} | {name} | {pct(xb['q'])} | {pct(xf['q'])} | {pts(xb['q'], xf['q'])} "
                      f"| {pct(xf['p'], 3)} | ±{100 * xf['T']:.2f} | "
                      f"{'yes' if xb['pass'] else 'no'} → {'yes' if xf['pass'] else '**no**'} |")

    print("\n### Gate v2, FULL FREE, best.pt → final.pt (x, verdict)\n\n| class | field | "
          + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS))
    for label in K.LABELS:
        for f in GATED + ("FULL",):
            print(f"| {label} | {f} | " + " | ".join(
                f"{cellx(*acc(rb, 'free', a, label, f))} → {cellx(*acc(rf, 'free', a, label, f))}"
                for a in ARMS) + " |")

    print("\n### FULL CONTROL, measurable states, best.pt → final.pt (x, verdict)\n\n| state | field | "
          + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS))
    for key in sorted(rf["control"][ARMS[0]]):
        for f in GATED + ("FULL",):
            if all(acc(rf, "control", a, key, f)[0] is None for a in ARMS):
                continue
            print(f"| {key} | {f} | " + " | ".join(
                f"{cellx(*acc(rb, 'control', a, key, f))} → {cellx(*acc(rf, 'control', a, key, f))}"
                for a in ARMS) + " |")

    print("\n### Honouring (control, native), best.pt → final.pt\n\n| state | "
          + " | ".join(ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for label in K.LABELS:
        for st in K.STATES:
            print(f"| {label} {st} | " + " | ".join(
                f"{pct(rate(rb, 'honouring', a, f'{label}_{st}'))} → "
                f"{pct(rate(rf, 'honouring', a, f'{label}_{st}'))}" for a in ARMS) + " |")

    print("\n### Copying and novelty, best.pt → final.pt\n\n| arm | mode | novel | copied queries |\n|---|---|---|---|")
    for arm in ARMS:
        for mode in ("free", "control"):
            b, f = rb["novelty"][arm][mode], rf["novelty"][arm][mode]
            print(f"| {arm} | {mode} | {pct(b['novel'] / b['n'])} → {pct(f['novel'] / f['n'])} | "
                  f"{pct(b['copied'] / b['q'])} → {pct(f['copied'] / f['q'])} |")
    print("\nK11d readings against each arm's reference (points, 95% CI), best.pt → final.pt:\n")
    for arm in ARMS:
        db, df = cb["arms"][arm]["K11d"], cf["arms"][arm]["K11d"]
        print(f"- {arm} against {df['reference']}: " + "; ".join(
            f"{k} {100 * db['readings'][k]['worsening']:+.2f} → {100 * x['worsening']:+.2f} "
            f"[{100 * x['lo']:+.2f}, {100 * x['hi']:+.2f}]" for k, x in df["readings"].items()))

    pb, pf = rb.get("plan", {}), rf.get("plan", {})
    if pf:
        print("\n### e11_plan's own plan (checkpoint, shipped-weighted), best.pt → final.pt\n\n"
              "| class | plan P | plan Q |\n|---|---|---|")
        for label in K.LABELS:
            print(f"| {label} | {pct(pb[label]['P'])} → {pct(pf[label]['P'])} | "
                  f"{pct(pb[label]['Q'], 3)} → {pct(pf[label]['Q'], 3)} |")

    print("\n### Each change at the saved checkpoints and at step 20,000\n\n"
          "| change | arm (step) against reference (step) | benign PTH | benign QRY | phishing PTH "
          "| phishing QRY | benign free QRY x | benign P0Q1 QRY x | benign P1Q1 QRY x |\n"
          "|---|---|---|---|---|---|---|---|---|")
    names = {"e11_rs": "the slash", "e11_ref": "the repair", "e11_joint": "joint dropout",
             "e11_plan": "the plan"}
    for arm in ARMS:
        for tag, c, stp in (("saved", cb, lambda a: step.get(a)), ("20,000", cf, lambda a: 20000)):
            at = c["arms"][arm]["attribution"]
            ref = at["reference"]
            refstep = 18000 if ref == "e10_base" else stp(ref)
            em = at["emission"]
            dx = at["cell_x"]
            print(f"| {names[arm]} | {arm} ({stp(arm):,}) against {ref} ({refstep:,}) | "
                  + " | ".join(f"{100 * em[k]:+.2f}" for k in
                               ("benign PTH", "benign QRY", "phishing PTH", "phishing QRY")) + " | "
                  + " | ".join("—" if dx.get(k) is None else f"{dx[k]:+.4f}" for k in
                               ("benign QRY", "benign P0Q1 QRY", "benign P1Q1 QRY")) + " |")
    print("\nEmission in points; cells as the change in x. `e10_base` is E10's registered "
          "reading (best.pt, step 18,000), the one comparison that still crosses steps.")

    print("\n### Amendment 3's predictions, scored\n")
    out = []
    k11a = cf["arms"]["e11_plan"]["K11a"]["pass"]
    out.append(("`e11_plan` passes K11a again", "passes" if k11a else "fails",
                "right" if k11a else "**wrong**"))
    full = []
    for arm in ARMS:
        for label in K.LABELS:
            full.append((f"{arm} free {label}", acc(rb, "free", arm, label, "FULL")[0],
                         acc(rf, "free", arm, label, "FULL")[0]))
        for key in rf["control"][arm]:
            full.append((f"{arm} {key}", acc(rb, "control", arm, key, "FULL")[0],
                         acc(rf, "control", arm, key, "FULL")[0]))
    moves = [(n, f - b) for n, b, f in full if b is not None and f is not None]
    worst = max(moves, key=lambda m: abs(m[1]))
    over = [f"{n} {d:+.4f}" for n, d in moves if abs(d) > 0.02]
    out.append(("FULL moves by at most 0.02 in every arm",
                f"largest move {worst[1]:+.4f} ({worst[0]}); over 0.02: {', '.join(over) or 'none'}",
                "right" if not over else "**wrong**"))
    sl = [(f"{arm} {name}", xf["q"] - cb["arms"][arm]["K11b"]["rows"][name]["q"])
          for arm in ARMS for name, xf in cf["arms"][arm]["K11b"]["rows"].items()
          if xf["q"] is not None and cb["arms"][arm]["K11b"]["rows"][name]["q"] is not None]
    worst = max(sl, key=lambda m: abs(m[1]))
    over = [f"{n} {100 * d:+.2f}" for n, d in sl if abs(d) > 0.003]
    out.append(("the slash rows move by at most 0.3 points",
                f"largest move {100 * worst[1]:+.2f} ({worst[0]}); over 0.3: {', '.join(over) or 'none'}",
                "right" if not over else "**wrong**"))
    ph = {arm: [r for r in ("phishing free_bare", "phishing control P0Q0")
                if not cf["arms"][arm]["K11b"]["rows"][r]["pass"]] for arm in ARMS}
    held = all(len(v) == 2 for v in ph.values())
    out.append(("K11b still fails on the phishing bare-host rows",
                "; ".join(f"{a}: {len(v)} of 2 fail" for a, v in ph.items()),
                "right" if held else "**wrong**" if not any(ph.values()) else "partly"))
    p = {a: rate(rf, "emission", a, "benign", "PTH") for a in ("e11_rs", "e11_ref", "e11_joint")}
    out.append(("benign free PTH about 76% (`e11_rs`), 74.5% (`e11_ref`), 74.5–75% (`e11_joint`)",
                ", ".join(f"{a} {pct(v)}" for a, v in p.items()),
                "reported (see the difference from each stated value)"))
    d = p["e11_joint"] - p["e11_ref"]
    out.append(("joint dropout moves benign free PTH by under 1 point", f"{100 * d:+.2f} points",
                "right" if abs(d) < 0.01 else "**wrong**"))
    xr = acc(rf, "free", "e11_ref", "benign", "QRY")[0]
    q = []
    for arm in ("e11_rs", "e11_joint", "e11_plan"):
        xb, xf = acc(rb, "free", arm, "benign", "QRY")[0], acc(rf, "free", arm, "benign", "QRY")[0]
        q.append((arm, xb, xf, xf is not None and xb is not None and xf < xb,
                  xf is not None and abs(xf - xr) <= 0.01))
    out.append(("the three 14,000-step arms improve on benign free QRY, to within 0.01 of "
                "`e11_ref`'s (`e11_ref` at step 20,000: " + f"{xr:+.4f})",
                "; ".join(f"{a} {xb:+.4f} → {xf:+.4f} ({'improves' if imp else 'does not improve'}, "
                          f"{'within' if near else 'not within'} 0.01)" for a, xb, xf, imp, near in q),
                "right" if all(i and n for *_, i, n in q) else
                "**wrong**" if not any(i for *_, i, _ in q) else "partly"))
    cq = []
    for st in ("P1Q1", "P0Q1"):
        xb, vb = acc(rb, "control", "e11_plan", f"benign_{st}", "QRY")
        xf, vf = acc(rf, "control", "e11_plan", f"benign_{st}", "QRY")
        cq.append((st, xb, vb, xf, vf))
    out.append(("`e11_plan`'s control query cells improve (whether they reach UNDECIDED: "
                "no prediction)",
                "; ".join(f"{st} QRY {xb:+.4f} {vb[:4]} → {xf:+.4f} {vf[:4]}" for st, xb, vb, xf, vf in cq),
                "right" if all(xf < xb for _, xb, _, xf, _ in cq) else
                "**wrong**" if all(xf >= xb for _, xb, _, xf, _ in cq) else "partly"))
    regs = []
    for arm in ARMS:
        for r in cb["arms"][arm]["K11c"]["regressions"]:
            key = " ".join(map(str, r["cell"]))
            regs.append((arm, key, r["arm"][0] - r["base"][0],
                         cf["arms"][arm]["attribution"]["cell_x"].get(key)))
    out.append(("K11c: the regressions found at step 14,000 shrink",
                "; ".join(f"{a} {k} {b:+.4f} → " + ("—" if f is None else f"{f:+.4f}")
                          for a, k, b, f in regs),
                "right" if all(f is not None and f < b for *_, b, f in regs) else "partly"
                if any(f is not None and f < b for *_, b, f in regs) else "**wrong**"))
    cps = []
    for arm in ARMS:
        for k in ("free_copying", "control_copying"):
            b = cb["arms"][arm]["K11d"]["readings"][k]
            if b["worsened"]:
                cps.append((arm, k, b["worsening"], cf["arms"][arm]["K11d"]["readings"][k]["worsening"]))
    out.append(("K11d: the copying rises found at step 14,000 shrink",
                "; ".join(f"{a} {k} {100 * b:+.2f} → {100 * f:+.2f}" for a, k, b, f in cps),
                "right" if all(f < b for *_, b, f in cps) else "partly"
                if any(f < b for *_, b, f in cps) else "**wrong**"))
    print("| prediction (`PREREG_E11` §3.3) | result | scored |\n|---|---|---|")
    for pred, res, sc in out:
        print(f"| {pred} | {res} | {sc} |")
    print("\nCheckpoint files: best.pt as E11_RESULTS.md; final.pt "
          + ", ".join(f"{a} {tprov[a]['final_sha256'][:8]}…" for a in ARMS) + ".")


if __name__ == "__main__":
    main()
