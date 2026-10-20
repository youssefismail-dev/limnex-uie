"""QC1's registered rules and helpers (research/qc1/PREREG_QC1.md).

The rules are pinned on hand-built numbers at and around each threshold; the
near-miss index against a brute-force search; the stratified re-draw's exact
share; the harness's real draws disjoint; and the reader's refusal of dry runs.
"""
import collections
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "qc1"))
import qc1_common as C  # noqa: E402


def test_tcrit_is_the_t_quantile_at_df_11():
    from scipy import stats
    assert C.TCRIT == pytest.approx(float(stats.t.ppf(0.975, C.D - 1)), abs=1e-12)
    assert C.SEEDS == tuple(20260806 + i for i in range(12))


def test_tolerance_is_the_owners_standard_with_the_edge_rule():
    sys.path.insert(0, str(ROOT / "research" / "native"))
    from e11_compare import tolerance
    for p, n in ((0.2927, 526), (0.4185, 374), (0.0, 4710), (1.0, 130), (0.05, 3000)):
        assert C.tolerance(p, n) == pytest.approx(tolerance(p, n), abs=1e-15)


def test_rate_reading_edges():
    t = C.tolerance(0.30, 500)
    assert C.rate_reading(0.30 - t - 1e-9, 0.30, 500)[0] == "UNDER"
    assert C.rate_reading(0.30 - t + 1e-9, 0.30, 500)[0] == "WITHIN"
    assert C.rate_reading(0.30 + t + 1e-9, 0.30, 500)[0] == "OVER"


def test_copy_profile_counts_and_bands():
    counts = collections.Counter({"a=1": 1, "b=2": 3, "c=3": 7, "d=4": 25})
    p = C.copy_profile(["a=1", "b=2", "c=3", "d=4", "new", "new2", "a=1", "zz"], counts)
    assert p["n"] == 8 and p["rate"] == pytest.approx(5 / 8)
    assert p["singleton"] == pytest.approx(2 / 8)
    assert p["bands"] == {"1": 2 / 8, "2-4": 1 / 8, "5-19": 1 / 8, "20+": 1 / 8}


def test_mix_reading_cases():
    plain = [0.10, 0.11, 0.09, 0.10, 0.12, 0.10, 0.11, 0.09, 0.10, 0.10, 0.11, 0.09]
    half = [x * 0.4 for x in plain]
    assert C.mix_reading(plain, half)["reading"] == "MIX-LED"
    same = [x + (0.001 if i % 2 else -0.001) for i, x in enumerate(plain)]
    assert C.mix_reading(plain, same)["reading"] == "CONTENT-LED"
    mid = [x * 0.65 for x in plain]
    assert C.mix_reading(plain, mid)["reading"] == "SPLIT"


def test_leak_budget_and_blending_rules():
    real = [0.30 + 0.005 * (i % 3) for i in range(12)]
    assert C.leak_reading(real, [x * 0.6 for x in real])["reading"] == "LEAK"
    assert C.leak_reading(real, [x * 0.9 for x in real])["reading"] == "NO LEAK"
    allx = [0.10 + 0.002 * (i % 4) for i in range(12)]
    assert C.budget_reading(allx, [x * 0.6 for x in allx])["reading"] == "BUDGET"
    assert C.budget_reading(allx, [x * 0.9 for x in allx])["reading"] == "NO BUDGET"
    assert C.blending_reading(300, 1000, 200, 1000)["reading"] == "BLENDING"
    assert C.blending_reading(240, 1000, 200, 1000)["reading"] == "NO BLENDING"     # 4 points
    # +10 points, but the 95% CI [-8.3, +28.3] spans 0
    assert C.blending_reading(3, 20, 1, 20)["reading"] == "NO BLENDING"


def _brute_lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def test_levenshtein_le1_and_the_near_index_match_brute_force():
    rng = random.Random(7)
    alphabet = "ab=&1"
    words = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6))) for _ in range(400)]
    frequent = words[:60]
    idx = C.NearIndex(frequent)
    for q in words[60:]:
        for s in frequent[:10]:
            assert C.levenshtein_le1(q, s) == (_brute_lev(q, s) <= 1)
        assert idx.near(q) == any(_brute_lev(q, s) <= 1 for s in frequent)


def test_stratified_draw_has_the_exact_share():
    found = [f"f{i}" for i in range(300)]
    novel = [f"n{i}" for i in range(900)]
    out = C.stratified(found, novel, 0.2927, 500, seed=1)
    assert len(out) == 500 and len(set(out)) == 500
    assert sum(x.startswith("f") for x in out) == round(0.2927 * 500)
    assert C.feasible_n(1800, 300, 900, 0.2927) == max(
        n for n in range(1801) if round(0.2927 * n) <= 300 and n - round(0.2927 * n) <= 900)


def test_real_split_draws_are_disjoint_and_seeded():
    real = [f"q{i}" for i in range(1000)]
    a, b, c = C.real_split(real, 300, seed=20260806)
    assert len({*a, *b, *c}) == 900
    assert (a, b, c) == C.real_split(real, 300, seed=20260806)


def test_slash_leak_reading():
    rng = random.Random(3)
    swap = [rng.uniform(0.001, 0.05) for _ in range(400)]
    tracking = [0.25 * s + rng.gauss(0, 0.0005) for s in swap]
    r = C.slash_leak_reading(tracking, swap)
    assert r["reading"] == "SLASH-LEAK" and r["lambda"] == pytest.approx(0.25, abs=0.03)
    floor = [0.004 + rng.gauss(0, 0.0005) for _ in swap]          # a flat floor, no tracking
    assert C.slash_leak_reading(floor, swap)["reading"] == "NO SLASH-LEAK"


def test_suppressed_reading():
    rng = random.Random(4)
    data = [int(rng.random() < 0.01) for _ in range(20000)]
    own = [0.002] * len(data)                                       # a fifth of the data's rate
    r = C.suppressed_reading(own, data, [0.001] * len(data))
    assert r["reading"] == "SUPPRESSED, LEAK-CONSISTENT" and r["ratio"] < 0.5
    assert C.suppressed_reading(own, data, [0.003] * len(data))["reading"] == "SUPPRESSED"
    assert C.suppressed_reading([0.01] * len(data), data, own)["reading"] == "NOT SUPPRESSED"


def test_state_blind_reading_and_wilson():
    own = [0.060 + 0.002 * (i % 3) for i in range(12)]
    cross = [0.030 + 0.002 * (i % 3) for i in range(12)]
    states = [0.050] * 12
    assert C.state_blind_reading(own, cross, states)["reading"] == "STATE-BLIND"
    assert C.state_blind_reading(own, cross, [0.010] * 12)["reading"] == "NOT STATE-BLIND"
    assert C.state_blind_reading(cross, own, states)["reading"] == "NOT STATE-BLIND"
    lo, hi = C.wilson(67, 14906)
    assert lo < 67 / 14906 < hi and hi - lo < 0.0025


def test_frame_leak_reading():
    real = [0.20 + 0.004 * (i % 3) for i in range(12)]
    assert C.frame_leak_reading(real, [x * 0.6 for x in real])["reading"] == "FRAME-LEAK"
    assert C.frame_leak_reading(real, [x * 0.9 for x in real])["reading"] == "NO FRAME-LEAK"
    alike = [0.03 + 0.001 * (i % 3) for i in range(12)]
    assert C.frame_leak_reading(alike, [x * 0.5 for x in alike])["reading"].startswith("UNINFORMATIVE")


def test_the_reader_refuses_dry_runs(tmp_path, monkeypatch):
    import qc1_read
    monkeypatch.setattr(C, "OUT", tmp_path)
    (tmp_path / "copy.json").write_text(json.dumps({"dry": True}), encoding="utf-8")
    with pytest.raises(SystemExit, match="dry run"):
        qc1_read.load("copy.json")
