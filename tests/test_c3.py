"""C3-QRY (research/c3/PREREG_C3_QRY.md).

Pins arm F's generator -- an exact rejection sample with every control token `?`,
the registered batch doubling, a loud failure instead of a short result -- its
constants against their sources, the verifier's identity logic, and every
reading rule of C3-A and C3-B, before any C3 number exists.
"""
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("research/c3", "research/sc1", "research/cr1"):
    sys.path.insert(0, str(ROOT / sub))

import c3_common as K  # noqa: E402
import free_matched as FM  # noqa: E402
from c3_compare import read_qry  # noqa: E402
from c3_describe import query_shape  # noqa: E402
from c3_ladder import component_reading, ladder_validity, shape_reading  # noqa: E402
from c3_verify import identity_free  # noqa: E402
from sc1_verify import agreement  # noqa: E402
from test_hc2_presence import SuffixStub, tok_with_org  # noqa: E402
from uie.evaluation import FIELDS  # noqa: E402
from uie.model import generate  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402

TABLE = SuffixTable({"benign": {"com": 2, "org": 3}})
KW = dict(decoder="temperature", decoder_kwargs={"temperature": 1.0})


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

def test_constants_match_their_sources():
    import cr1_score
    import cr1_verify
    import sc1_common
    assert cr1_score.STATES[K.STATE_INDEX["benign"]] == ("benign", {"QRY": True})
    assert cr1_score.STATES[K.STATE_INDEX["phishing"]] == ("phishing", {"QRY": True})
    assert K.DELTA == sc1_common.DELTA == cr1_verify.DELTA
    assert K.ROUND_STRIDE == 1_000_003 and K.MAX_ROUNDS == 100 and K.DOUBLING_CAP == 10
    assert K.N == {"benign": 1200, "phishing": 400} and K.M == {"benign": 40, "phishing": 80}
    assert K.N["phishing"] >= 1068 // 3              # arm F meets phishing's real-pool cap
    assert sorted(K.Q_FILES_SHA256_PREFIX) == sorted(
        f"{K.tag(l)}_g{g}.json" for l in ("benign", "phishing") for g in range(K.G))


def test_arm_f_seed_block_is_disjoint():
    reach = 99 * K.ROUND_STRIDE + 1_300              # last round, more groups than any table has
    used = [(0, reach), (300_000_000, 300_000_000 + reach),       # CR1, SC1 C
            (600_000_000, 600_000_000 + reach), (800_000_000, 800_000_000 + reach)]  # T, T_rep
    lo, hi = K.OFFSET_F, K.OFFSET_F + reach
    assert all(hi < a or lo > b for a, b in used) and hi < 1_000_000_000
    assert K.gen_seed("benign", 0) == 20261001 + 200 * 1_000_000_000 + K.OFFSET_F
    assert K.gen_seed("phishing", 4) == 20261001 + 604 * 1_000_000_000 + K.OFFSET_F


@pytest.mark.skipif(not (ROOT / "data" / "state_suffix_tables.json").is_file(),
                    reason="needs data/state_suffix_tables.json")
def test_tables_hash_is_the_adopted_file():
    blob = (ROOT / "data" / "state_suffix_tables.json").read_bytes()
    assert hashlib.sha256(blob).hexdigest() == K.TABLES_SHA256


# --------------------------------------------------------------------------- #
# arm F's generator
# --------------------------------------------------------------------------- #

def _free(model, tok, n=16, seed=3, m=4, **extra):
    return FM.generate_free_matched(model, tok, label="benign", n=n, table=TABLE, seed=seed,
                                    m=m, **KW, **extra)


def test_every_url_carries_a_query_and_the_sample_is_deterministic():
    tok = tok_with_org()
    model = SuffixStub(tok, q_org=0.0, q_com=-1.0)      # queries are the minority
    urls = _free(model, tok)
    assert len(urls) == 16 and all(FIELDS["QRY"](u) for u in urls)
    assert urls == _free(model, tok)
    assert urls != _free(model, tok, seed=4)
    assert agreement(urls, TABLE, "benign", 3) == 1.0    # every suffix held


def test_equals_an_independent_reference_loop():
    """Suffixes drawn, grouped and sorted; per group, the first query-bearing
    draws in order, batches m * 2**min(r, cap) * deficit; then shuffled."""
    tok = tok_with_org()
    model = SuffixStub(tok, q_org=0.0, q_com=-1.0)
    rng = random.Random(9)
    drawn = TABLE.sample("benign", 12, rng)
    out = []
    for i, ((fr, s), count) in enumerate(sorted(collections.Counter(
            zip(["domain"] * 12, drawn)).items())):
        kept, r = [], 0
        while len(kept) < count:
            batch = generate(model, tok, label="benign", n=3 * 2 ** min(r, 10) * (count - len(kept)),
                             suffix=s, seed=9 + i + 1_000_003 * r, frame=fr, batch_size=32,
                             fields=None, **KW)
            kept += [u for u in batch if FIELDS["QRY"](u)][:count - len(kept)]
            r += 1
        out += kept
    rng.shuffle(out)
    assert _free(model, tok, n=12, seed=9, m=3) == out


def test_every_control_token_is_free_and_batches_double(monkeypatch):
    tok = tok_with_org()
    model = SuffixStub(tok, q_org=-2.5, q_com=-2.5)       # rare: forces several rounds
    calls = []

    def spy(*a, **kw):
        calls.append(kw)
        return generate(*a, **kw)

    monkeypatch.setattr(FM, "generate", spy)
    stats = {}
    _free(model, tok, n=4, m=2, stats=stats)
    assert calls and all(c["fields"] is None for c in calls)
    rounds = collections.defaultdict(list)
    for c in calls:
        rounds[c["suffix"]].append(c)
    for seq in rounds.values():                            # per group: m * 2**r * deficit
        for r, c in enumerate(seq):
            assert c["n"] % (2 * 2 ** min(r, 10)) == 0
    assert stats["rounds"] >= 2 and stats["round0_draws"] <= stats["draws"]


def test_an_unreachable_request_fails_loudly():
    tok = tok_with_org()
    model = SuffixStub(tok, q_org=-50.0, q_com=-50.0)      # never writes a query
    with pytest.raises(FM.FreeUnmet):
        _free(model, tok, n=2, m=1, max_rounds=3)


def test_a_control_or_masking_is_refused():
    tok = tok_with_org()
    model = SuffixStub(tok)
    for bad in ({"fields": {"QRY": True}}, {"enforce": "mask"}):
        with pytest.raises(ValueError):
            _free(model, tok, **bad)


def test_identity_logic_finds_every_round0_url():
    tok = tok_with_org()
    model = SuffixStub(tok, q_org=0.0, q_com=-1.0)
    kw = dict(decoder="temperature", decoder_kwargs={"temperature": 1.0},
              field_bias={"PTH": {tok.eos_id: -K.DELTA["benign"]}})
    urls = FM.generate_free_matched(model, tok, label="benign", n=20, table=TABLE, seed=21,
                                    m=6, **kw)
    same, compared = identity_free(model, tok, TABLE, urls, "benign", 21, 6, 20, generate)
    assert compared > 0 and same == compared
    tampered = [u + "z" for u in urls]                     # every URL changed
    same2, compared2 = identity_free(model, tok, TABLE, tampered, "benign", 21, 6, 20, generate)
    assert compared2 == compared and same2 == 0


# --------------------------------------------------------------------------- #
# readings
# --------------------------------------------------------------------------- #

def test_c3a_readings():
    rows = {"benign": {"delta": 0.001, "lo": -0.012, "hi": 0.014, "p": 0.8, "paired": True},
            "phishing": {"delta": 0.05, "lo": 0.03, "hi": 0.07, "p": 0.001, "paired": True}}
    r = read_qry(rows)
    assert r["benign"]["reading"] == "CONSISTENT" and r["phishing"]["reading"] == "DIFFERENT"
    assert r["phishing"]["p_holm"] == pytest.approx(0.002)
    r2 = read_qry({"benign": {"delta": 0.015, "lo": -0.01, "hi": 0.04, "p": 0.2, "paired": True},
                   "phishing": None})
    assert r2["benign"]["reading"] == "INCONCLUSIVE" and r2["phishing"]["reading"] == "NOT COMPUTABLE"
    r3 = read_qry({"benign": {"delta": 0.021, "lo": 0.001, "hi": 0.041, "p": 0.04, "paired": True},
                   "phishing": {"delta": 0.03, "lo": 0.02, "hi": 0.04, "p": 0.03, "paired": True}})
    assert r3["benign"]["reading"] == "INCONCLUSIVE"      # Holm 0.06 >= 0.05


def test_c3a_contrasts_the_registered_fields():
    # section 5: QRY primary; LBL, SUF, SUB, PTH and FULL descriptive
    from c3_compare import FIELDS6
    assert FIELDS6 == ("LBL", "SUF", "SUB", "PTH", "QRY", "FULL")
    assert set(FIELDS6) == set(FIELDS)                  # every field the gate reads


def test_k1_is_not_read_and_keeps_the_registered_family():
    ok = {"delta": 0.03, "lo": 0.02, "hi": 0.04, "p": 0.03, "paired": True}
    r = read_qry({"benign": ok, "phishing": None}, k1=["phishing"])
    assert r["phishing"]["reading"] == "K1 -- not read (PREREG section 8)"
    # a family of one would give Holm 0.03 -> DIFFERENT; the registered two, 0.06
    assert r["benign"]["p_holm"] == pytest.approx(0.06)
    assert r["benign"]["reading"] == "INCONCLUSIVE"


def test_k1_classes_come_from_the_provenance_errors(tmp_path, monkeypatch):
    # c3_score puts research/c3 first on sys.path when imported; keep that
    # side effect inside this test
    monkeypatch.setattr(sys, "path", list(sys.path))
    from c3_score import k1_labels
    recs = [{"file": "benign_Q1_g0.json", "label": "benign", "error": None},
            {"file": "phishing_Q1_g1.json", "label": "phishing", "error": None},
            {"file": "phishing_Q1_g2.json", "label": "phishing",
             "error": "FreeUnmet: suffix 'io': 1 of 3 phishing URLs without a query after 100 rounds"}]
    (tmp_path / "PROVENANCE.json").write_text(json.dumps({"files": recs}), encoding="utf-8")
    assert k1_labels(tmp_path) == {"phishing": [
        "phishing_Q1_g2.json: FreeUnmet: suffix 'io': 1 of 3 phishing URLs without a query "
        "after 100 rounds"]}


def _row(verdict="no change", R=1.0, draws=None):
    return {"verdict": verdict, "R_T": R, "excess": 0.1, "excess_draws": draws or [0.1]}


def test_shape_reading():
    assert shape_reading({"skeleton": _row("no change", 0.95)}) == "SHAPE"
    assert shape_reading({"skeleton": _row("REDUCES", 0.30)}) == "VOCABULARY"
    assert shape_reading({"skeleton": _row("REDUCES", 0.90)}) == "INTERMEDIATE"
    assert shape_reading({"skeleton": _row("no change", 0.60)}) == "INTERMEDIATE"
    assert shape_reading({"skeleton": _row("VOID floor", 0.9)}) == "VOID"


def test_ladder_validity():
    good = {"identity": _row(draws=[0.1, 0.2]), "null": _row(draws=[0.1, 0.2])}
    assert ladder_validity(good, 0.12) == []
    assert ladder_validity({**good, "null": _row(draws=[0.1, 0.3])}, 0.12)
    assert ladder_validity(good, 0.2)                     # identity 0.1 vs C3-A x 0.2


def test_the_ladder_collects_after_every_call_and_changes_nothing(monkeypatch):
    import gc
    from c3_ladder import collecting
    calls = []
    monkeypatch.setattr(gc, "collect", lambda *a: calls.append(1) or 0)
    f = collecting(lambda a, b=0: (a, b))
    assert f(1, b=2) == (1, 2) and f(3) == (3, 0) and len(calls) == 2
    g = collecting(lambda: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        g()
    assert len(calls) == 3                          # collected even when the call fails


def test_component_readings():
    base = {t: _row() for t in ("clip_values_20", "clip_values_15", "clip_values_10",
                                "first_param", "mask_name_chars", "mask_values")}
    one = dict(base, mask_name_chars=_row("REDUCES", 0.40))
    assert component_reading({"Q": one, "F": one})["reading"] == "IDENTIFIED"
    two = dict(base, mask_name_chars=_row("REDUCES", 0.80), mask_values=_row("REDUCES", 0.85))
    r = component_reading({"Q": two, "F": two})
    assert r["reading"] == "DISTRIBUTED" and sorted(r["reduce_in_both"]) == \
        ["name identity", "value composition"]
    only_q = dict(base, first_param=_row("REDUCES", 0.3))
    assert component_reading({"Q": only_q, "F": base})["reading"] == "UNRESOLVED"
    tail_bad = dict(base, clip_values_20=_row("REDUCES", 0.30), clip_values_15=_row("REDUCES", 0.45),
                    clip_values_10=_row("REDUCES", 0.40))  # not monotone: 0.30 < 0.45
    assert component_reading({"Q": tail_bad, "F": tail_bad})["reading"] == "UNRESOLVED"
    tail_ok = dict(base, clip_values_20=_row("REDUCES", 0.48), clip_values_15=_row("REDUCES", 0.40),
                   clip_values_10=_row("REDUCES", 0.30))
    assert component_reading({"Q": tail_ok, "F": tail_ok})["identified"] == ["long-value tail"]


def test_query_shape_arithmetic():
    s = query_shape(["a=1&b=xy", "flag", "q=%20x-y"])
    assert s["params_per_query"] == pytest.approx(4 / 3)
    assert s["name_length"] == pytest.approx((1 + 1 + 4 + 1) / 4)
    assert s["value_length_p50"] == 2 and s["value_length_p90"] == pytest.approx(5.2)
    assert s["percent_encoded_queries"] == pytest.approx(1 / 3)
    assert s["numeric_values"] == pytest.approx(1 / 3)
    assert s["bare_flags"] == pytest.approx(1 / 4)
    assert s["hyphenated_values"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# the notebook
# --------------------------------------------------------------------------- #

NB = ROOT / "research" / "c3" / "C3_generate.ipynb"


@pytest.mark.skipif(not NB.is_file(), reason="needs the notebook")
def test_notebook_is_its_generator_and_runs_the_registered_call():
    # research/sc1 has a make_notebook.py too: load C3's by path, under its own name
    import importlib.util
    spec = importlib.util.spec_from_file_location("c3_make_notebook",
                                                  ROOT / "research" / "c3" / "make_notebook.py")
    make_notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_notebook)
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["cells"] == make_notebook.cells()
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<cell>", "exec")
    for needle in ("generate_free_matched(", "table=tables.for_state(label, STATE)",
                   "seed=gen_seed(label, g)", "m=M[label]", "max_rounds=MAX_ROUNDS",
                   "doubling_cap=DOUBLING_CAP", 'decoder="temperature"',
                   '{"temperature": 1.0}', '{"PTH": {tokenizer.eos_id: -DELTA[label]}}',
                   make_notebook.E7_SHA256):
        assert needle in src, needle
