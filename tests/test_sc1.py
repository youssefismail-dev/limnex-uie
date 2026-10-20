"""SC1 -- the state-conditioned suffix draw (research/sc1/PREREG_SC1.md).

Pins what the registration rests on: the constants SC1 restates from CR1, the
disjointness of the seed blocks, the table construction (and, with the bundle
present, that its counting rule rebuilds the shipped table exactly), the V4
suffix-agreement check, every statistical rule the comparison applies, and the
notebook against its generator.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "cr1"))
sys.path.insert(0, str(ROOT / "research" / "sc1"))

import build_tables as B  # noqa: E402
import sc1_common as C  # noqa: E402
import sc1_compare as X  # noqa: E402
from s0p import differences  # noqa: E402
from sc1_verify import agreement  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402

BUNDLE = ROOT / "data" / "bundle2"
TABLES = ROOT / "data" / "sc1" / "tables.json"
NOTEBOOK = ROOT / "research" / "sc1" / "SC1_generate.ipynb"


# --------------------------------------------------------------------------- #
# constants and seeds
# --------------------------------------------------------------------------- #

def test_restated_constants_are_cr1s():
    import cr1_score
    import cr1_verify
    assert C.STATES == cr1_score.STATES
    assert C.NOT_MEASURABLE == cr1_score.NOT_MEASURABLE
    assert (C.G, C.N_MAIN) == (cr1_score.G, cr1_score.N_MAIN)
    assert C.DELTA == cr1_verify.DELTA
    assert C.IDENTITY_STATES == cr1_verify.IDENTITY_STATES
    assert all(C.tag_of(l, s) == cr1_score.tag_of(l, s) for l, s in C.STATES)
    assert len(C.MEASURABLE) == 14
    assert set(C.REGISTERED_T_ROWS) == {C.tag_of(l, s) for _, l, s in C.MEASURABLE}


def test_seed_blocks_are_disjoint():
    # within one block of 1e9, a request uses seed + group + 1,000,003 * round:
    # at most 100 rounds and fewer groups than suffixes in any table (1,232)
    reach = 100 * 1_000_003 + 1_300
    starts = [0, C.OFFSET["C"], C.OFFSET["T"], C.OFFSET["T_rep"]]      # CR1 first
    assert all(a + reach < b for a, b in zip(starts, starts[1:]))
    assert starts[-1] + reach < 1_000_000_000
    for arm in C.OFFSET:
        for s, _, _ in C.MEASURABLE:
            for g in range(C.G):
                assert C.gen_seed(arm, s, g) == \
                    20261001 + (100 * s + g) * 1_000_000_000 + C.OFFSET[arm]
    assert C.gen_seed("C", 0, 0) == 20261001 + 300_000_000


def test_table_counts_selects_the_arms_table():
    tables = {"C": {"benign": {"com": 1}}, "T": {"benign_P1": {"benign": {"de": 2}}}}
    assert C.table_counts(tables, "C", "benign", {"PTH": True}) == {"com": 1}
    assert C.table_counts(tables, "T", "benign", {"PTH": True}) == {"de": 2}
    assert C.table_counts(tables, "T_rep", "benign", {"PTH": True}) == {"de": 2}


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

ROWS = [{"url": "a.shop.de/x", "label": "benign"}, {"url": "b.shop.de"},
        {"url": "c.shop.com/x?q=1", "label": "benign"}, {"url": "d.shop.com?q=1"},
        {"url": "e.shop.org/y"}, {"url": "localhost"}]


def test_count_suffixes_is_suffixtables_rule():
    assert B.count_suffixes(ROWS) == SuffixTable.from_urls(ROWS).counts
    with_path = B.count_suffixes(ROWS, keep=lambda u: "/" in u)
    assert with_path == {"benign": {"de": 1, "com": 1, "org": 1}}


def _write(path: Path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_build_on_a_tiny_bundle(tmp_path):
    _write(tmp_path / "name_train.jsonl.gz", ROWS)
    _write(tmp_path / "phishing_name_train.jsonl.gz", [
        {"url": "x.login.io/a", "label": "phishing"}, {"url": "y.login.io", "label": "phishing"},
        {"url": "z.login.app?q=1", "label": "phishing"}])
    t = B.build(tmp_path)
    assert t["C"] == {"benign": {"de": 2, "com": 2, "org": 1},
                      "phishing": {"io": 2, "app": 1}}
    T = {tag: next(iter(v.values())) for tag, v in t["T"].items()}
    assert T["benign_P1"] == {"de": 1, "com": 1, "org": 1}
    assert T["benign_P0"] == {"de": 1, "com": 1}
    assert T["benign_Q1"] == {"com": 2}
    assert T["benign_P1_Q1"] == {"com": 1} and T["benign_P0_Q1"] == {"com": 1}
    assert T["benign_P1_Q0"] == {"de": 1, "org": 1} and T["benign_P0_Q0"] == {"de": 1}
    assert T["phishing_P1"] == {"io": 1} and T["phishing_P0"] == {"io": 1, "app": 1}
    assert T["phishing_Q1"] == {"app": 1} and T["phishing_Q0"] == {"io": 2}
    assert "phishing_P1_Q1" not in t["T"] and "phishing_P0_Q1" not in t["T"]
    _write(tmp_path / "name_train.jsonl.gz", [{"url": "a.b.com", "label": "phishing"}])
    with pytest.raises(ValueError):
        B.build(tmp_path)


def test_canonical_bytes_and_registered_totals():
    a = {"T": {"x": {"benign": {"b": 1, "a": 2}}}, "C": {"benign": {"z": 1}}}
    b = {"C": {"benign": {"z": 1}}, "T": {"x": {"benign": {"a": 2, "b": 1}}}}
    assert B.canonical(a) == B.canonical(b) and B.canonical(a).endswith(b"\n")
    exact = {"C": {l: {"x": n} for l, n in C.REGISTERED_C_ROWS.items()},
             "T": {t: {t.split("_")[0]: {"x": n}} for t, n in C.REGISTERED_T_ROWS.items()}}
    assert B.totals_failures(exact) == []
    exact["T"]["phishing_Q1"]["phishing"]["x"] += 1
    assert len(B.totals_failures(exact)) == 1


@pytest.mark.skipif(not (BUNDLE / "suffix_table.json").is_file(), reason="needs data/bundle2")
def test_counting_rule_rebuilds_the_shipped_table():
    """PREREG_SC1 section 9: the builder's rule must rebuild the shipped table."""
    stems = ("name_train", "structure_train", "phishing_name_train", "phishing_structure_train")
    rows = (r for s in stems for r in B.read_rows(BUNDLE / f"{s}.jsonl.gz"))
    shipped = json.loads((BUNDLE / "suffix_table.json").read_text(encoding="utf-8"))
    assert B.count_suffixes(rows) == shipped


@pytest.mark.skipif(not TABLES.is_file(), reason="needs data/sc1/tables.json")
def test_built_tables_have_the_registered_totals():
    assert B.totals_failures(json.loads(TABLES.read_text(encoding="utf-8"))) == []


# --------------------------------------------------------------------------- #
# V4 and V6
# --------------------------------------------------------------------------- #

def test_agreement_is_multiset_and_shuffle_proof():
    table = SuffixTable({"benign": {"com": 5, "de": 3, "org": 2}})
    drawn = table.sample("benign", 200, random.Random(7))
    urls = [f"h{i}.example.{s}" for i, s in enumerate(drawn)]
    random.Random(1).shuffle(urls)
    assert agreement(urls, table, "benign", 7) == 1.0
    urls[:4] = [f"x{i}.example.net" for i in range(4)]
    assert agreement(urls, table, "benign", 7) == pytest.approx(0.98)


def test_s0p_differences_are_nan_aware():
    rec = {"s": {"fields": {"SUF": {"excess": float("nan"), "excess_draws": [0.1]}},
                 "pooled_fields": {}}}
    same = json.loads(json.dumps(rec))
    assert differences(rec, same) == []
    same["s"]["fields"]["SUF"]["excess_draws"] = [0.2]
    assert differences(rec, same) == [("s", "fields", "SUF")]


# --------------------------------------------------------------------------- #
# the comparison's statistics
# --------------------------------------------------------------------------- #

def _draws(means, noise=None):
    noise = noise or [0.0] * 50
    return [means[i // 10] + noise[i] for i in range(50)]


def test_paired_contrast_cancels_shared_split_noise():
    rng = random.Random(3)
    shared = [rng.gauss(0, 0.03) for _ in range(50)]      # same real samples, same floor
    shift = [0.01, 0.02, 0.03, 0.02, 0.02]
    a = _draws([0.05 + s for s in shift], shared)
    b = _draws([0.05] * 5, shared)
    r = X.paired(a, b)
    se = math.sqrt(sum((s - 0.02) ** 2 for s in shift) / 4) / math.sqrt(5)
    assert r["paired"] and r["df"] == 4
    assert r["delta"] == pytest.approx(0.02) and r["se"] == pytest.approx(se)
    assert r["lo"] == pytest.approx(0.02 - 2.776445 * se, abs=1e-6)
    assert r["p"] == pytest.approx(2 * X.tdist.sf(0.02 / se, 4))


def test_unpaired_contrast():
    r = X.unpaired({"x": 0.07, "se": 0.004}, {"x": 0.01, "se": 0.003})
    assert not r["paired"] and r["delta"] == pytest.approx(0.06)
    assert r["se"] == pytest.approx(0.005) and r["t"] == pytest.approx(12.0)


def test_gen_means_needs_five_by_ten():
    assert X.gen_means(_draws([1, 2, 3, 4, 5])) == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        X.gen_means([0.0] * 49)


def test_holm():
    assert X.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == \
        pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_pairing_follows_n_exactly():
    assert X.pairable(1200, [3000] * 5, [2990] * 5)            # both capped
    assert not X.pairable(1200, [1150] * 5, [1160] * 5)       # fake-limited, unequal
    assert X.pairable(356, [2000] * 5, [1000] * 5)            # real-limited
    assert X.cap_of(1068) == 356 and X.cap_of(90000) == 1200


def test_contrast_picks_paired_unpaired_or_none():
    row = {"acceptance": {"x": 0.05, "se": 0.003}, "excess_draws": _draws([0.05] * 5)}
    row2 = {"acceptance": {"x": 0.02, "se": 0.003}, "excess_draws": _draws([0.02] * 5)}
    assert X.contrast(row, {"acceptance": {"verdict": "NOT COMPUTABLE"}}, True) is None
    assert X.contrast(row, row2, True)["paired"]
    assert not X.contrast(row, row2, False)["paired"]


def test_h_suf_reading():
    ok = {"delta": -0.07, "p_holm": 0.001, "t_verdict": "CLEARED"}
    und = {**ok, "t_verdict": "UNDECIDED"}
    assert X.read_h_suf({"a": ok, "b": ok, "c": ok}, True)["overall"] == "SUPPORTED"
    assert X.read_h_suf({"a": ok, "b": und, "c": ok}, True)["overall"] == "PARTIAL"
    assert X.read_h_suf({"a": und, "b": und, "c": None}, True)["overall"] == "REFUTED"
    assert X.read_h_suf({"a": und}, False)["overall"].startswith("NOT READ")


def test_h_down_reading():
    down = {"delta": -0.05, "p_holm": 0.001, "lo": -0.07, "hi": -0.03}
    indep = {"delta": 0.001, "p_holm": 0.9, "lo": -0.01, "hi": 0.012}
    both = {"delta": -0.01, "p_holm": 0.01, "lo": -0.015, "hi": -0.005}
    none = {"delta": 0.01, "p_holm": 0.5, "lo": -0.03, "hi": 0.05}
    assert X.read_h_down({i: down for i in range(4)} | {4: none, 5: none})["overall"] == "H_DOWN"
    assert X.read_h_down({i: indep for i in range(4)} | {4: down, 5: down})["overall"] == "H_INDEP"
    r = X.read_h_down({i: down for i in range(4)} | {4: both, 5: none})
    assert r["overall"] == "MIXED" and r["n_indep"] == 1         # a 'both' cell counts twice


def test_k2_flags():
    cr1 = {"s": {"fields": {"PTH": {"acceptance": {"verdict": "CLEARED"}}},
                 "pooled_fields": {"QRY": {"verdict": "UNDECIDED"}}}}
    t = {"s": {"fields": {"PTH": {"acceptance": {"verdict": "CONVICTED"}}},
               "pooled_fields": {"QRY": {"verdict": "TELL"}}}}
    contrasts = {"s/LBL": {"delta": 0.03, "lo": 0.001},
                 "s/SUF": {"delta": 0.019, "lo": 0.01},       # under the margin
                 "s/SUB": {"delta": 0.05, "lo": -0.001},      # CI touches 0
                 "s/QRY": None}
    rules = {(f["cell"], f["rule"]) for f in X.k2_flags(t, cr1, contrasts)}
    assert rules == {("s/LBL", "margin"), ("s/PTH", "CLEARED->CONVICTED"),
                     ("s/QRY", "pooled UNDECIDED->TELL")}
    # the scorer's report lists the registered not-measurable states without fields
    skip = {"phishing_P1_Q1": {"status": "NOT MEASURABLE (registered)"}}
    assert X.k2_flags({**t, **skip}, {**cr1, **skip}, contrasts) == X.k2_flags(t, cr1, contrasts)


def test_novelty_runaway_and_proportions():
    d = X.prop_diff(9800, 10000, 9900, 10000)
    assert d["delta"] == pytest.approx(-0.01) and d["hi"] < 0
    assert X.novelty_harm(d)
    assert not X.novelty_harm({"delta": -0.004, "hi": -0.001})
    assert X.runaway_harm({"delta": 0.003, "lo": 0.0005})
    assert not X.runaway_harm({"delta": 0.003, "lo": -0.0001})


def test_adoption_gate():
    assert X.adoption("SUPPORTED", [], [], [], [], [], True)["gate"] == "MET"
    assert X.adoption("SUPPORTED", [], None, [], [], [], True)["gate"].startswith("PENDING")
    r = X.adoption("PARTIAL", [], [], [], [], [], True)
    assert r["gate"] == "NOT MET" and "H_SUF PARTIAL" in r["reasons"]
    assert X.adoption("SUPPORTED", [], [], [], [], [], False)["gate"] == "NOT MET"


# --------------------------------------------------------------------------- #
# the notebook
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not (NOTEBOOK.is_file() and TABLES.is_file()),
                    reason="needs the notebook and data/sc1/tables.json")
def test_notebook_is_its_generator_and_runs_the_registered_call():
    # research/c3 has a make_notebook.py too: load SC1's by path, under its own
    # name, so the result never depends on sys.path order
    import importlib.util
    spec = importlib.util.spec_from_file_location("sc1_make_notebook",
                                                  ROOT / "research" / "sc1" / "make_notebook.py")
    make_notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_notebook)
    sha = hashlib.sha256(TABLES.read_bytes()).hexdigest()
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["cells"] == make_notebook.cells(sha)       # no hand edits
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<cell>", "exec")
    for needle in ("hold_suffix=True", "max_rounds=MAX_ROUNDS", 'decoder="temperature"',
                   '{"temperature": 1.0}', '{"PTH": {tokenizer.eos_id: -DELTA[label]}}',
                   "seed=gen_seed(arm, s, g)", "table_counts(tables, arm, label, state)",
                   make_notebook.E7_SHA256, sha):
        assert needle in src, needle
