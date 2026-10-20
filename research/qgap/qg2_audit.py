"""QG2-D probe 3 -- the audit (research/qgap/PREREG_QG2D.md section 5, A1-A6): does
training treat the frames or the hidden controls asymmetrically? Code and data only;
nothing is trained or changed.

    python research/qgap/qg2_audit.py        # writes data/qgap/qg2_audit.json
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import qg_common as Q  # noqa: E402
from qg1_measure import vq1_model  # noqa: E402
from uie.kaggle_train import STREAMS  # noqa: E402
from uie.model import generate  # noqa: E402
from uie.train import CORPUS_FRAME, FieldDropout, TrainConfig, encode_corpus, field_mask  # noqa: E402

PER_STREAM = 25_000
Z = 1.959964


def wilson(k: int, n: int):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, c - h, c + h


def sample(stem: str, n: int):
    pool = list(Q.rows(stem))
    return random.Random(Q.SEED).sample(pool, min(n, len(pool)))


def main() -> None:
    model, tok, cfg = vq1_model("e9_ln")
    out = {"registration": "research/qgap/PREREG_QG2D.md (probe 3)"}
    frame_id = {f: tok.stoi[f"<F:{f}>"] for f in ("domain", "page")}
    frame_pos = 2                                        # <BOS> <C:*> <F:*>

    # A1 -- every row under its corpus's frame (code: the stream table and CORPUS_FRAME)
    rows = []                                            # (ids, mask, label, corpus)
    mism = 0
    for stem, label, corpus in STREAMS:
        urls = sample(stem, PER_STREAM)
        enc, _ = encode_corpus(urls, [label] * len(urls), tok, cfg.max_len,
                               corpora=[corpus] * len(urls))
        for ids, mask in enc.get(label, []):
            want = frame_id[CORPUS_FRAME[corpus]]
            mism += int(ids[frame_pos] != want)
            rows.append((ids, mask, label, corpus))
    out["A1"] = {"streams": [list(s) for s in STREAMS], "corpus_frame": dict(CORPUS_FRAME),
                 "rows_checked": len(rows), "frame_mismatches": mism, "pass": mism == 0}

    # A2 -- hiding independent of presence, the other control, frame and class
    drop = FieldDropout(tok, TrainConfig().field_dropout or 0.5,
                        seed=TrainConfig().seed + 7919)
    pos = tok.field_control_positions()
    qm, pm = tok.marker_id("QRY"), tok.marker_id("PTH")
    groups: dict = {}

    def tally(key, hidden):
        k, n = groups.get(key, (0, 0))
        groups[key] = (k + int(hidden), n + 1)

    both = [0, 0]
    for ids, mask, label, corpus in rows:
        new, _ = drop((ids, mask))
        hid = {f: new[pos[f]] == tok.field_control_id(f, None) for f in pos}
        present = {"PTH": pm in ids, "QRY": qm in ids}
        for f in pos:
            tally((f, "all"), hid[f])
            tally((f, f"present={present[f]}"), hid[f])
            tally((f, f"frame={CORPUS_FRAME[corpus]}"), hid[f])
            tally((f, f"class={label}"), hid[f])
        tally(("QRY", f"PTH hidden={hid['PTH']}"), hid["QRY"])
        both[0] += int(hid["PTH"] and hid["QRY"])
        both[1] += 1
    rates = {f"{f} | {g}": dict(zip(("rate", "lo", "hi", "n"), (*wilson(k, n), n)))
             for (f, g), (k, n) in sorted(groups.items())}
    off = [key for key, r in rates.items() if not r["lo"] <= 0.5 <= r["hi"]]
    b = wilson(*both)
    out["A2"] = {"p": 0.5, "rates": rates, "both_hidden": {"rate": b[0], "lo": b[1], "hi": b[2]},
                 "hidden_token_is_free_modes": all(
                     drop.hidden[f] == tok.field_control_id(f, None) for f in pos),
                 "groups_whose_CI_excludes_0.5": off,
                 "pass": not off and b[1] <= 0.25 <= b[2]}

    # A3 -- a training row's controls sit where generation puts them
    a3 = {}
    for label in Q.LABELS:
        ids_out = []
        generate(model, tok, label=label, n=1, suffix="com", seed=1, decoder="temperature",
                 decoder_kwargs={"temperature": 1.0}, ids_out=ids_out)
        suf = tok.marker_id("SUF")
        g = ids_out[0][:ids_out[0].index(suf)]
        e = tok.encode("x.com", label, frame="domain", hide_fields=("PTH", "QRY"))
        e = e[:e.index(suf)]
        ctrl_at = {f: e[p] == tok.field_control_id(f, None) for f, p in pos.items()}
        a3[label] = {"generate_prefix": g, "encode_prefix": e, "identical": g == e,
                     "control_positions_hold": all(ctrl_at.values())}
    out["A3"] = {**a3, "pass": all(v["identical"] and v["control_positions_hold"]
                                   for v in a3.values())}

    # A4 -- <QRY> is a weight-1 target wherever it occurs, in both frames
    viol, seen = {"name": 0, "structure": 0}, {"name": 0, "structure": 0}
    for ids, mask, label, corpus in rows:
        if qm in ids:
            j = ids.index(qm)
            seen[corpus] += 1
            viol[corpus] += int(field_mask(ids, tok, corpus)[j - 1] != 1.0)
    out["A4"] = {"rows_with_a_query": seen, "weight_not_1": viol,
                 "pass": not any(viol.values())}

    # A5 -- presence derived as the fields are opened, on the host route too
    cases = {"host.com/?q=1": (False, True), "host.com?q=1": (False, True),
             "host.com/p?q=1": (True, True), "host.com#f": (False, False)}
    a5 = {}
    for url, (p_want, q_want) in cases.items():
        ids = tok.encode(url, "benign", frame="domain")
        got = (ids[pos["PTH"]] == tok.field_control_id("PTH", True),
               ids[pos["QRY"]] == tok.field_control_id("QRY", True))
        a5[url] = {"controls": got, "markers": (pm in ids, qm in ids),
                   "ok": got == (p_want, q_want) == (pm in ids, qm in ids)}
    out["A5"] = {**a5, "pass": all(v["ok"] for v in a5.values())}

    # A6 -- free mode's slice in training: class by names, rows uniform within class
    data = json.loads((Q.DATA / "qg1_data.json").read_text(encoding="utf-8"))
    names = {lab: data[lab]["train"]["rows"] for lab in Q.LABELS}
    tot = sum(names.values())
    a6 = {}
    for lab in Q.LABELS:
        dom, page = data[lab]["train"]["rows_fit"], data[lab]["page"]["rows_fit"]
        p_cls = names[lab] / tot
        a6[lab] = {"p_class": p_cls, "domain_rows": dom, "page_rows": page,
                   "domain_share": dom / (dom + page),
                   "free_mode_slice_share_of_all_exposures": p_cls * dom / (dom + page) * b[0]}
    out["A6"] = a6

    out["findings"] = [k for k in ("A1", "A2", "A3", "A4", "A5") if not out[k]["pass"]]
    Q.DATA.mkdir(parents=True, exist_ok=True)
    (Q.DATA / "qg2_audit.json").write_text(json.dumps(out, indent=1, default=str),
                                           encoding="utf-8")
    for k in ("A1", "A2", "A3", "A4", "A5"):
        print(f"  {k}: {'pass' if out[k]['pass'] else 'FINDING'}")
    for lab, v in a6.items():
        print(f"  A6 {lab}: p(class) {v['p_class']:.3f}, domain share {v['domain_share']:.3f}, "
              f"free-mode slice {100 * v['free_mode_slice_share_of_all_exposures']:.1f}% of exposures")
    print("wrote", Q.DATA / "qg2_audit.json")


if __name__ == "__main__":
    main()
