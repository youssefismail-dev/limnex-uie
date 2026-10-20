"""
research/v2/v2_r2_diag.py

POST-HOC diagnosis after R2's Part A (not a registered reading; it decides nothing by itself).
Part A showed a word memory closes only part of the honest judge's tell. This asks what the rest
of the tell IS, on Part A's existing samples:

  1. single-feature AUCs: each of the judge's own features (D1 with Vocab_B), plus four
     descriptive ones, scored alone -- real C against each source;
  2. the judge's frequency bands: every word placed by its count in Vocab_B;
  3. feature groups: the judge refitted on its structural features only, and on its vocabulary
     features only.

    python research/v2/v2_r2_diag.py
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_r2 as R2  # noqa: E402

SD, R1, G, LABELS = R2.SD, R2.R1, R2.G, R2.LABELS
OUT = R2.OUT / "diag"
NAMES = ["len", "len_host", "len_path", "len_query", "len_frag", "n_labels", "max_label", "last_label",
         "n_segs", "mean_seg", "max_seg", "trailing_slash", "has_query", "n_params", "digit_share",
         "upper_share", "nonalnum_share", "n_hyphen", "n_underscore", "n_dot", "has_pct", "has_at",
         "colon_host", "entropy", "max_digit_run", "n_words", "mean_word_len", "max_word_len",
         "known_share", "mean_logcount", "min_logcount", "n_unseen", "max_unseen_len",
         "host_known_share", "rest_known_share"]
STRUCT = list(range(28))
VOCAB = list(range(28, 35))
BANDS = ((0, 0), (1, 9), (10, 99), (100, 999), (1000, 10 ** 12))


def extra(s: str, vocab) -> list:
    words = R2.WORD.findall(s.lower())
    c = vocab.counts(words) if words else np.zeros(0)
    known = c[c > 0]
    unseen = [w for w, k in zip(words, c) if k == 0]
    return [float(np.mean(np.log1p(known))) if len(known) else 0.0,          # frequency of KNOWN words
            float(np.mean([len(w) for w in unseen])) if unseen else 0.0,    # length of new words
            float(np.mean([len(w) for w in words])) if words else 0.0,
            len(unseen) / len(words) if words else 0.0]


EXTRA = ["mean_logcount_known", "mean_unseen_len", "mean_word_len_all", "unseen_share"]


def auc(y, x) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, x)) if len(set(x)) > 1 else 0.5


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    for label in LABELS:
        vocab = R2.DictVocab(R2.judge_vocab(label))
        for part, sources in R2.PART_A.items():
            if part == "whole":
                continue
            key = f"{part}|{label}"
            feats, bands, groups = {}, {}, {}
            for g in range(G):
                node = R1._read(R2.SAMPLES / f"A_{part}_{label}_g{g}.json")
                X = {k: np.array([SD.d1_features(s, vocab) + extra(s, vocab) for s in node[k]], dtype=float)
                     for k in ("C",) + sources}
                for k in sources:
                    y = np.array([0] * len(X["C"]) + [1] * len(X[k]))
                    both = np.vstack([X["C"], X[k]])
                    for j, name in enumerate(NAMES + EXTRA):
                        feats.setdefault(k, {}).setdefault(name, []).append(auc(y, both[:, j]))
                    if k in ("base", "O2e"):
                        for gname, cols in (("structural", STRUCT), ("vocabulary", VOCAB)):
                            fp = _hgb(cols)
                            texts = list(range(len(both)))
                            groups.setdefault(k, {}).setdefault(gname, []).append(
                                SD.oof_auc(lambda xtr, ytr, xte, seed, _b=both, _fp=fp: _fp(_b[xtr], ytr, _b[xte], seed),
                                           texts, y, SD.seed_of(label, g) * 10 + 1))
                for k in ("C",) + sources:
                    cnt = Counter()
                    for s in node[k]:
                        for c in vocab.counts(R2.WORD.findall(s.lower())):
                            cnt[next(i for i, (lo, hi) in enumerate(BANDS) if lo <= c <= hi)] += 1
                    tot = sum(cnt.values()) or 1
                    bands.setdefault(k, []).append([cnt[i] / tot for i in range(len(BANDS))])
            out[key] = {
                "single_feature_auc": {k: {n: st.mean(v) for n, v in d.items()} for k, d in feats.items()},
                "bands": {k: [st.mean(col) for col in zip(*v)] for k, v in bands.items()},
                "groups": {k: {n: st.mean(v) for n, v in d.items()} for k, d in groups.items()}}
            print(f"  {key}: done", flush=True)
    R1._write(OUT / "diag.json", out, indent=1)
    _print(out)


def _hgb(cols):
    def fp(xtr, ytr, xte, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
        m.fit(np.asarray(xtr)[:, cols], ytr)
        return m.predict_proba(np.asarray(xte)[:, cols])[:, 1]
    return fp


def _print(out: dict) -> None:
    band_names = ["unseen", "1-9", "10-99", "100-999", "1000+"]
    for key, node in out.items():
        print(f"\n== {key}")
        print("  bands (word shares by Vocab_B count):")
        for k, v in node["bands"].items():
            print(f"    {k:<5} " + "  ".join(f"{n} {x:.3f}" for n, x in zip(band_names, v)))
        print("  feature groups (judge refitted; AUC real vs source):")
        for k, v in node["groups"].items():
            print(f"    {k:<5} " + "  ".join(f"{n} {x:.4f}" for n, x in v.items()))
        print("  strongest single features, real vs base (|AUC - 0.5|):")
        base = node["single_feature_auc"]["base"]
        top = sorted(base.items(), key=lambda kv: -abs(kv[1] - 0.5))[:10]
        for n, a in top:
            row = "  ".join(f"{k} {node['single_feature_auc'][k][n]:.3f}" for k in node["single_feature_auc"]
                            if k != "base")
            print(f"    {n:<20} base {a:.3f}   {row}")


if __name__ == "__main__":
    main()
