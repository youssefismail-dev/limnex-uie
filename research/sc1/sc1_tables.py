"""SC1 -- render the results tables from data/sc1/sc1_report.json and the three
scorer reports. The tables in SC1_RESULTS.md are this script's output, pasted
unedited; nothing is typed by hand.

    python research/sc1/sc1_tables.py --root data/sc1 --cr1 data/cr1
"""
import argparse
import json
from pathlib import Path

FIELDS = ("LBL", "SUF", "SUB", "PTH", "QRY")


def pretty(tag):
    label, *codes = tag.split("_")
    return " ".join([label] + [f"{c[0]}:{c[1]}" for c in codes])


def verdict(v):
    return f"**{v}**" if v in ("CONVICTED", "TELL") else v


def xv(field_row):
    a = (field_row or {}).get("acceptance", {})
    if a.get("x") is None:
        return a.get("verdict", "--")
    return f"{a['x']:+.4f} {verdict(a['verdict'])}"


def ci(r):
    return f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cr1", type=Path, required=True)
    args = ap.parse_args()
    rep = load(args.root / "sc1_report.json")
    arms = {"CR1": load(args.cr1 / "cr1_report.json")["states"],
            "C": load(args.root / "C" / "cr1_report.json")["states"],
            "T": load(args.root / "T" / "cr1_report.json")["states"]}

    def suf_family(name, fam, reading):
        print(f"\n## {name}\n")
        print("| state | CR1 SUF | C SUF | T SUF | Δ(T − C) [95% CI] | p (Holm) | pairing | supported |")
        print("|---|---|---|---|---|---:|---|---|")
        for key, r in fam.items():
            tag = key.split("/")[0]
            cells = [xv(arms[a][tag]["fields"]["SUF"]) for a in ("CR1", "C", "T")]
            if r is None:
                print(f"| {pretty(tag)} | " + " | ".join(cells) + " | not computable | | | |")
                continue
            print(f"| {pretty(tag)} | " + " | ".join(cells) +
                  f" | {ci(r)} | {r['p_holm']:.2g} | {'paired' if r['paired'] else 'UNPAIRED'} "
                  f"| {'yes' if reading['per_state'][tag] else 'no'} |")
        print(f"\nReading: **{reading['overall']}**")

    suf_family("Primary — SUF, T − C (phishing)", rep["primary"], rep["h_suf"])
    suf_family("Secondary — SUF, T − C (benign)", rep["secondary"], rep["h_suf_secondary"])

    print("\n## H_DOWN against H_INDEP — LBL and SUB, T − C\n")
    print("| state / field | CR1 | C | T | Δ(T − C) [95% CI] | p (Holm) | H_DOWN | H_INDEP |")
    print("|---|---|---|---|---|---:|---|---|")
    for key, r in rep["h_down_cells"].items():
        tag, f = key.split("/")
        cells = [xv(arms[a][tag]["fields"][f]) for a in ("CR1", "C", "T")]
        per = rep["h_down"]["per_cell"].get(key, {})
        if r is None:
            print(f"| {pretty(tag)} {f} | " + " | ".join(cells) + " | not computable | | | |")
            continue
        print(f"| {pretty(tag)} {f} | " + " | ".join(cells) + f" | {ci(r)} | {r['p_holm']:.2g} "
              f"| {'yes' if per.get('H_DOWN') else 'no'} | {'yes' if per.get('H_INDEP') else 'no'} |")
    hd = rep["h_down"]
    print(f"\nReading: **{hd['overall']}** ({hd['n_down']} of 6 cells H_DOWN, {hd['n_indep']} H_INDEP)")

    for arm in ("T", "C"):
        print(f"\n## Arm {arm}: every state and field\n")
        print("| state | " + " | ".join(FIELDS) + " | FULL (aux) |")
        print("|---|" + "---|" * len(FIELDS) + "---:|")
        for tag, s in arms[arm].items():
            if "fields" not in s:
                print(f"| {pretty(tag)} | not measurable (registered) |" + " |" * len(FIELDS))
                continue
            row = []
            for f in FIELDS:
                if s["state"].get(f) is False:
                    row.append("n/a")
                elif s["fields"][f]["acceptance"].get("x") is not None:
                    row.append(xv(s["fields"][f]))
                else:
                    p = (s.get("pooled_fields") or {}).get(f)
                    row.append(f"pooled {p['excess']:+.4f} {verdict(p['verdict'])}"
                               if p and not p.get("insufficient") else "not computable")
            full = s["fields"]["FULL"]["acceptance"]
            print(f"| {pretty(tag)} | " + " | ".join(row) + f" | {full['x']:+.4f} |")

    print("\n## Safety: T − CR1, every computable cell\n")
    print("| state | " + " | ".join(FIELDS) + " |")
    print("|---|" + "---|" * len(FIELDS))
    tcr1 = rep["contrasts"]["T-CR1"]
    for tag in [t for t, s in arms["T"].items() if "fields" in s]:
        row = []
        for f in FIELDS:
            r = tcr1.get(f"{tag}/{f}")
            if f"{tag}/{f}" not in tcr1:
                row.append("n/a")
            elif r is None:
                row.append("--")
            else:
                flag = " ⚑" if (r["delta"] >= 0.02 and r["lo"] > 0) else ""
                row.append(f"{r['delta']:+.4f}{'' if r['paired'] else '†'}{flag}")
        print(f"| {pretty(tag)} | " + " | ".join(row) + " |")
    print("\n(Δ = x_T − x_CR1; † unpaired; ⚑ K2 margin rule; -- not computable in an arm)")
    print(f"\nK2 flags: {rep['k2_flags'] or 'none'}; replication: {rep['k2_replication'] or '--'}")
    print(f"K3: {rep['k3'] or 'none'}; K4: {rep['k4'] or 'none'}; K5: {rep['k5'] or 'none'}")

    print("\n## Novelty, runaways, cost (all 15,000 URLs per state and arm)\n")
    print("| state | novelty CR1 / C / T | T − CR1 [95% CI] | 240+ chars CR1 / C / T | first pass CR1 / C / T | draws per URL CR1 / C / T | max rounds C / T |")
    print("|---|---|---|---|---|---|---|")
    for tag, d in rep["descriptive"].items():
        nov = d["novelty_T_minus_CR1"]
        print(f"| {pretty(tag)} | {d['CR1']['novelty']:.4f} / {d['C']['novelty']:.4f} / {d['T']['novelty']:.4f} "
              f"| {nov['delta']:+.4f} [{nov['lo']:+.4f}, {nov['hi']:+.4f}] "
              f"| {d['CR1']['long']} / {d['C']['long']} / {d['T']['long']} "
              f"| {d['CR1']['first_pass']:.3f} / {d['C']['first_pass']:.3f} / {d['T']['first_pass']:.3f} "
              f"| {d['CR1']['draws_per_url']:.3f} / {d['C']['draws_per_url']:.3f} / {d['T']['draws_per_url']:.3f} "
              f"| {d['C']['rounds']} / {d['T']['rounds']} |")

    print("\n## Registered predictions (section 4; not criteria)\n")
    print("| state | T SUF x | T ceiling | within ±0.02 | C SUF x | C ceiling | within | Δ(T − C) observed / predicted | C − CR1 observed / predicted |")
    print("|---|---:|---:|---|---:|---:|---|---|---|")
    for tag, p in rep["predictions"].items():
        pt, pc, p2, p3 = p["P1_T"], p["P1_C"], p["P2_delta_T_minus_C"], p["P3_C_minus_CR1"]
        obs2 = f"{p2['observed']:+.4f}" if p2["observed"] is not None else "--"
        obs3 = f"{p3['observed']:+.4f}" if p3["observed"] is not None else "--"
        print(f"| {pretty(tag)} | {pt['x']:+.4f} | {pt['ceiling']:+.4f} | {'yes' if pt['within'] else '**no**'} "
              f"| {pc['x']:+.4f} | {pc['ceiling']:+.4f} | {'yes' if pc['within'] else '**no**'} "
              f"| {obs2} / {p2['predicted']:+.4f} | {obs3} / {p3['predicted']:+.4f} |")

    print("\n## Adoption gate (section 7.5)\n")
    print(f"**{rep['adoption']['gate']}**" + (f" — {rep['adoption']['reasons']}" if rep["adoption"]["reasons"] else ""))
    print(f"\nValidity: VERIFY {rep['validity']['verify']}, S0' {rep['validity']['s0p']}")


if __name__ == "__main__":
    main()
