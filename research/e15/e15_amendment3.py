"""E15-B Amendment 3's tables (research/e15/PREREG_E15.md section 9.5 P8, section 10.4,
section 11): every measured number, rendered from the records so none is transcribed by
hand. Prints markdown.

    python research/e15/e15_amendment3.py > data/e15/amendment3_tables.md

Reads data/e15/targets.json, gates.json, qry_extra/manifest.json and
batch_digest_e12_final.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E15D = ROOT / "data" / "e15"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def yn(v):
    return "—" if v is None else ("PASS" if v else "**FAIL**")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    g, m = load(E15D / "gates.json"), load(E15D / "qry_extra" / "manifest.json")
    rp = load(E15D / "batch_digest_e12_final.json") if (E15D / "batch_digest_e12_final.json").is_file() else None

    print("### Acquisition\n")
    ph = m["phishing"]
    print(f"- **OpenPhish archive:** {ph['feed']}, head {ph['old_head'][:12]} → {ph['new_head'][:12]} "
          f"({ph['new_head_date']}); {ph['commits']:,} commits; {ph['archive_urls']:,} distinct URLs.")
    f = ph["funnel"]
    print(f"- **Phishing funnel:** {f['new_urls']:,} new URLs → {f['new_hosts_any']:,} hosts → "
          f"{f.get('host_seen_before_cutoff', 0):,} seen before the cutoff → {f['new_hosts_picked']:,} new hosts "
          f"(one URL each) → {f.get('no_query', 0):,} without a query, {f.get('over_max_len', 0):,} over 384 tokens, "
          f"{f.get('exact_url_in_train_or_val', 0)} exact-URL overlaps → **{f['kept_P0Q1']:,} P0Q1 + "
          f"{f['kept_P1Q1']:,} P1Q1**.")
    b = m["benign"]
    rounds = m.get("benign_rounds", {})
    print(f"- **Benign:** {len(rounds)} rounds of up to {b['round_blocks']} blocks per crawl over the 8 "
          f"crawls, in parallel (cap {b['cap']}). Seeds: blocks {b['seed_blocks']}; name reservoir "
          f"{b['seed_name']}.")
    print("\n| round | crawl | blocks | records | kept URLs | domains | minutes | HTTP errors | resync failures |\n"
          "|---|---|---|---|---|---|---|---|---|")
    tot = {"blocks": 0, "records": 0, "kept": 0, "err": 0, "rs": 0}
    for r in sorted(rounds, key=int):
        for c, v in rounds[r].items():
            print(f"| {r} | {c} | {v['blocks']:,} | {v['records']:,} | {v['kept']:,} | {v['domains']:,} | "
                  f"{v['seconds'] / 60:.1f} | {v.get('n_http_errors', '—')} | {v.get('n_failed_resync', '—')} |")
            tot["blocks"] += v["blocks"]
            tot["records"] += v["records"]
            tot["kept"] += v["kept"]
            tot["err"] += v.get("n_http_errors", 0)
            tot["rs"] += v.get("n_failed_resync", 0)
    print(f"| all | | **{tot['blocks']:,}** | {tot['records']:,} | {tot['kept']:,} | | | {tot['err']} | {tot['rs']} |")
    bf = b["funnel"]
    print(f"\n- **Benign funnel:** {bf['rows_read']:,} URLs read → {bf['domains']:,} registrable domains "
          f"(one URL each, reservoir) → {bf.get('domain_in_existing_split', 0):,} already in the corpus split → "
          f"{bf.get('in_phishing_feed', 0):,} in the phishing feed → {bf.get('no_query', 0):,} without a query → "
          f"{bf.get('over_max_len', 0):,} over 384 tokens → **{bf['query_rows_P0Q1']:,} P0Q1 + "
          f"{bf['query_rows_P1Q1']:,} P1Q1** available.")
    print("\n| stream file | rows | available | target (N_old) | sha256 |\n|---|---|---|---|---|")
    for st, v in b["files"].items():
        print(f"| benign_{st}.jsonl | {v['rows']:,} | {v['available']:,} | {v['target']:,} | `{v['sha256']}` |")
    for st, v in ph["files"].items():
        print(f"| phishing_{st}.jsonl | {v['rows']:,} | — | — | `{v['sha256']}` |")

    print("\n### The gates\n")
    print("| stratum | N_new / N_old | P3 leakage | P4 mix | P7 contamination | admitted |\n|---|---|---|---|---|---|")
    for k, s in g["strata"].items():
        p4 = s["P4"]
        mix = (f"suffix TV {p4['suffix_tv_new_train']:.3f} vs {p4['suffix_tv_ref_train_val']:.3f}"
               + (f"; crawl TV {p4['crawl_tv_new_train']:.3f} vs {p4['crawl_tv_ref_train_val']:.3f}"
                  if "crawl_tv_new_train" in p4 else "") + f" {yn(p4['pass'])}")
        print(f"| {k} | {s['N_new']:,} / {s['N_old']:,} | {yn(s['P3']['pass'])} (key overlap "
              f"{s['P3']['split_key_overlap']}, exact URL {s['P3']['exact_url_in_train_or_val']}) | {mix} | "
              f"{yn(s['P7']['pass'])} | **{'yes' if s.get('admitted') else 'no'}** |")
    print("\n| gate cell | P1 population (DX1) | P2 coverage gain | P5 diversity | P6 new strings |\n|---|---|---|---|---|")
    for c, n in g["cells"].items():
        p1, p2, p5, p6 = n["P1"], n["P2"], n["P5"], n["P6"]
        print(f"| {c} | {p1['reading']}: x_NV {p1['x_tv']:+.4f} (t {p1['t']:.2f}, n {p1['n']}); old twin "
              f"{p1['x_tv_old_twin']:+.4f}; x_GV {p1['x_gv']:+.4f} {yn(p1['pass'])} | "
              f"{100 * p2['gain']:+.2f} pts (needs ≥ {100 * p2['threshold']:+.2f}; r {p2['r']:.3f}) "
              f"{yn(p2['pass']) if p2['binding'] else '(reported)'} | "
              f"ratio {p5['distinct_ratio_median']:.3f} (≥ 0.9); top-10 {p5['top10_new']:.3f} vs "
              f"{p5['top10_old']:.3f} (+0.05) {yn(p5['pass'])} | never seen {100 * p6['never_seen_new']:.1f}% vs "
              f"val {100 * p6['never_seen_val']:.1f}% (−5) {yn(p6['pass'])}; new templates "
              f"{100 * p6['new_template_share']:.1f}% |")

    p8 = g["P8"]
    print("\n### P8: the frozen numbers (admitted strata only)\n")
    print("| stratum | admitted | N_new | N_old | p (inject) | old rows per batch | appended per batch | "
          "L_old | L_new | added query positions per batch | stratum query mass |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in p8["strata"].items():
        ratio = s["stratum_query_mass_ratio"]
        print(f"| {k} | {'yes' if s['admitted'] else 'no'} | {s['N_new']:,} | {s['N_old']:,} | {s['p_inject']:.4f} | "
              f"{s['old_rows_per_batch']:.3f} | {s['appended_per_batch']:.3f} | {s['L_old']:.2f} | "
              f"{s['L_new']:.2f} | {s['added_query_positions_per_batch']:.1f} | "
              f"{'×' + format(ratio, '.3f') if s['admitted'] and ratio else '—'} |")
    print(f"\n- **Appended rows per batch:** {p8['appended_rows_per_batch']:.3f} "
          f"({100 * p8['appended_rows_per_batch'] / p8['batch']:.2f}% of {p8['batch']}).")
    print(f"- **Added query positions per batch:** {p8['added_query_positions_per_batch']:.1f}, against "
          f"`e12_final`'s {p8['query_positions_per_batch_e12_final']:.0f}: query mass "
          f"**+{100 * p8['query_mass_increase']:.1f}%**; scored mass +{100 * p8['scored_mass_increase']:.2f}% "
          f"(C {p8['C']:.3f}); **{p8['dose_vs_e14']:.3f} of E14's added mass** "
          f"({p8['e14_added_per_batch']:.0f} per batch).")
    print(f"- **Seeds:** {json.dumps(p8['seeds'])}.")
    if rp:
        print(f"- **V-batch reference** (`e15_replay.py`): the old rows' batch digest over {rp['steps']:,} steps "
              f"(seed {rp['seed']}, batch {rp['batch_size']}, window {rp['bucket_window']}): "
              f"`{rp['digest']}`; counts {rp['counts']}, dropped {rp['dropped']:,}.")
    print(f"- **Code:** the acquisition and gates at `{m['benign'].get('code_commit', '?')[:12]}`.")


if __name__ == "__main__":
    main()
