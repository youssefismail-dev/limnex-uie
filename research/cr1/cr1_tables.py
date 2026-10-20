"""CR1 -- render the results tables from data/cr1/cr1_report.json.

The verdict, detail, pooled and root-slash tables in CR1_RESULTS.md are this
script's output, pasted unedited (section 3's numbers come from cr1_suf_mix.py
and the generation provenance).

    python research/cr1/cr1_tables.py data/cr1/cr1_report.json [data/cr1/slash.json]
"""
import json
import sys

FIELDS = ("LBL", "SUF", "SUB", "PTH", "QRY")


def pretty(tag):
    """benign_P1_Q0 -> benign P:1 Q:0"""
    label, *codes = tag.split("_")
    return " ".join([label] + [f"{c[0]}:{c[1]}" for c in codes])


def verdict(v):
    return f"**{v}**" if v in ("CONVICTED", "TELL") else v


def excluded(state, field):
    """A field the state rules out: PTH under P:0, QRY under Q:0."""
    return state.get(field) is False


def cell(s, f):
    """Compact 'x verdict' for the headline matrix."""
    if excluded(s["state"], f):
        return "n/a"
    acc = s["fields"][f]["acceptance"]
    if acc["verdict"] != "NOT COMPUTABLE":
        return f"{acc['x']:+.4f} {verdict(acc['verdict'])}"
    p = s["pooled_fields"].get(f)
    if p and not p.get("insufficient"):
        return f"pooled {p['excess']:+.4f} {verdict(p['verdict'])}"
    return "not computable"


def headline(states):
    print("| state | " + " | ".join(FIELDS) + " | FULL (aux) |")
    print("|---|" + "---|" * len(FIELDS) + "---:|")
    for tag, s in states.items():
        if "fields" not in s:
            print(f"| {pretty(tag)} | not measurable (registered) |" + " |" * len(FIELDS))
            continue
        full = s["fields"]["FULL"]["acceptance"]
        print(f"| {pretty(tag)} | " + " | ".join(cell(s, f) for f in FIELDS)
              + f" | {full['x']:+.4f} |")


def detail(states):
    print("| state | field | x | SE | t vs bar | 95% CI | n | generation means | verdict |")
    print("|---|---|---:|---:|---:|---|---:|---|---|")
    for tag, s in states.items():
        if "fields" not in s:
            continue
        for f in FIELDS + ("FULL",):
            if f != "FULL" and excluded(s["state"], f):
                continue
            row, acc = s["fields"][f], s["fields"][f]["acceptance"]
            if acc["verdict"] == "NOT COMPUTABLE":
                print(f"| {pretty(tag)} | {f} | -- | -- | -- | -- | real {row.get('n_real')}, "
                      f"generated {row.get('n_fake')} per generation | -- | NOT COMPUTABLE |")
                continue
            lo, hi = acc["x"] - acc["t_crit"] * acc["se"], acc["x"] + acc["t_crit"] * acc["se"]
            gm = " ".join(f"{m:.4f}" for m in acc["generation_means"])
            print(f"| {pretty(tag)} | {f} | {acc['x']:+.4f} | {acc['se']:.4f} | {acc['t_vs_bar']:+.2f} "
                  f"| [{lo:+.4f}, {hi:+.4f}] | {row['n']} | {gm} | {verdict(acc['verdict'])} |")


def pooled(states):
    print("| state | field | per generation (real / generated values) | pooled x | min | max | n | D=10 | D=3 |")
    print("|---|---|---|---:|---:|---:|---:|---|---|")
    for tag, s in states.items():
        if "fields" not in s:
            continue
        for f in FIELDS:
            if excluded(s["state"], f):
                continue
            row = s["fields"][f]
            if row["acceptance"]["verdict"] != "NOT COMPUTABLE":
                continue
            p = s["pooled_fields"].get(f, {})
            per = f"{row.get('n_real')} / {row.get('n_fake')}"
            if p.get("insufficient") or not p:
                print(f"| {pretty(tag)} | {f} | {per} | still insufficient (real {p.get('n_real')}, "
                      f"pooled generated {p.get('n_fake')}) | | | | NOT COMPUTABLE | -- |")
                continue
            print(f"| {pretty(tag)} | {f} | {per} | {p['excess']:+.4f} | {p['excess_min']:+.4f} "
                  f"| {p['excess_max']:+.4f} | {p['n']} | {verdict(p['verdict'])} | {p['d3_verdict']} |")


def slash(states, path):
    sl = json.load(open(path, encoding="utf-8"))
    print("| state | real: root slash (one per host) | generated: root slash | slash alone: (p_real - p_gen) / 2 | FULL x |")
    print("|---|---:|---:|---:|---:|")
    for tag, s in states.items():
        if tag not in sl:
            continue
        r = sl[tag]
        x = s["fields"]["FULL"]["acceptance"]["x"]
        print(f"| {pretty(tag)} | {r['root_real_1ph']:.3f} | {r['root_gen']:.3f} | {r['bound']:+.3f} | {x:+.4f} |")


def main():
    rep = json.load(open(sys.argv[1], encoding="utf-8"))
    states = rep["states"]
    print(f"G={rep['G']} D={rep['D']} n_main={rep['n_main']} seed={rep['seed']}\n")
    print("## headline\n"); headline(states)
    print("\n## detail\n"); detail(states)
    print("\n## pooled\n"); pooled(states)
    if len(sys.argv) > 2:
        print("\n## FULL root slash\n"); slash(states, sys.argv[2])
    secs = sum(s.get("seconds", 0) for s in states.values())
    print(f"\nscoring time: {secs / 3600:.2f} h over "
          f"{sum(1 for s in states.values() if 'fields' in s)} states")


if __name__ == "__main__":
    main()
