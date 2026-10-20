"""SC1's adoption: controlled generation with state-conditioned suffix tables
(uie/controlled.py; research/sc1/SC1_RESULTS.md).

Pins that the library IS what SC1 measured: the same counting rule, the same
tables (against SC1's own file when the data is present), and the same call --
suffix from the requested state's table, held while presence is enforced -- byte
for byte; and that the paths SC1 did not adopt are untouched.
"""
import gzip
import json
from pathlib import Path

import pytest

from test_e7_fields import e7_tok, tiny_model
from test_hc2_presence import SuffixStub, tok_with_org
from uie.controlled import StateSuffixTables, generate_controlled, state_code
from uie.presence import generate_with_presence
from uie.suffixes import SuffixTable, generate_with_suffixes

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "bundle2"
SC1_TABLES = ROOT / "data" / "sc1" / "tables.json"

ROWS = [{"url": "a.shop.de/x", "label": "benign"}, {"url": "b.shop.de"},
        {"url": "c.shop.com/x?q=1", "label": "benign"}, {"url": "d.shop.com?q=1"},
        {"url": "e.shop.org/y"}, {"url": "localhost"},
        {"url": "x.login.io/a", "label": "phishing"}, {"url": "y.login.io", "label": "phishing"},
        {"url": "z.login.app?q=1", "label": "phishing"}]


def test_state_code():
    assert state_code({"PTH": True}) == "P1"
    assert state_code({"QRY": False}) == "Q0"
    assert state_code({"PTH": False, "QRY": True}) == "P0_Q1"
    assert state_code({"QRY": True, "PTH": True}) == "P1_Q1"          # order is fixed
    assert state_code({"PTH": True, "QRY": None}) == "P1"             # None = free
    for bad in (None, {}, {"PTH": None}):
        with pytest.raises(ValueError):
            state_code(bad)
    with pytest.raises(ValueError):
        state_code({"FRG": True})


def test_counting_is_suffixtables_rule_in_every_state():
    t = StateSuffixTables.from_rows(ROWS)
    b = t.counts["benign"]
    assert b["P1"] == {"de": 1, "com": 1, "org": 1} and b["P0"] == {"de": 1, "com": 1}
    assert b["Q1"] == {"com": 2} and b["Q0"] == {"de": 2, "org": 1}
    assert b["P1_Q1"] == {"com": 1} and b["P0_Q1"] == {"com": 1}
    assert b["P1_Q0"] == {"de": 1, "org": 1} and b["P0_Q0"] == {"de": 1}
    p = t.counts["phishing"]
    assert p["P1"] == {"io": 1} and p["P0"] == {"io": 1, "app": 1} and p["P0_Q1"] == {"app": 1}
    # every path state together is the class marginal, which is SuffixTable's count
    marginal = SuffixTable.from_urls(ROWS).counts
    for label in ("benign", "phishing"):
        both = {}
        for code in ("P1", "P0"):
            for s, c in t.counts[label].get(code, {}).items():
                both[s] = both.get(s, 0) + c
        assert both == marginal[label]


def _write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_build_reads_only_the_domain_frame_and_refuses_mislabels(tmp_path):
    _write(tmp_path / "name_train.jsonl.gz", ROWS[:6])
    _write(tmp_path / "phishing_name_train.jsonl.gz", ROWS[6:])
    _write(tmp_path / "structure_train.jsonl.gz", [{"url": "never.counted.nl/x"}])
    t = StateSuffixTables.build(tmp_path)
    assert t.counts == StateSuffixTables.from_rows(ROWS).counts
    _write(tmp_path / "name_train.jsonl.gz", [{"url": "a.b.com", "label": "phishing"}])
    with pytest.raises(ValueError):
        StateSuffixTables.build(tmp_path)


def test_serialisation_is_byte_exact(tmp_path):
    t = StateSuffixTables.from_rows(ROWS)
    t.save(tmp_path / "t.json")
    back = StateSuffixTables.load(tmp_path / "t.json")
    assert back.counts == t.counts and back.to_bytes() == t.to_bytes()
    assert (tmp_path / "t.json").read_bytes() == t.to_bytes() and t.to_bytes().endswith(b"\n")
    shuffled = StateSuffixTables({"phishing": t.counts["phishing"], "benign": t.counts["benign"]})
    assert shuffled.sha256() == t.sha256()


def test_for_state_selects_the_requested_table():
    t = StateSuffixTables.from_rows(ROWS)
    assert t.for_state("benign", {"QRY": True}).counts == {"benign": {"com": 2}}
    assert t.for_state("benign", {"PTH": True, "QRY": False}).counts == \
        {"benign": {"de": 1, "org": 1}}
    with pytest.raises(KeyError):
        t.for_state("phishing", {"PTH": True, "QRY": True})        # no such rows here
    with pytest.raises(ValueError):
        StateSuffixTables({"benign": {"Q1": {}}}).for_state("benign", {"QRY": True})


def test_generate_controlled_is_sc1_arm_ts_call_byte_for_byte():
    """The notebook generated arm T with SuffixTable({label: <state counts>}) and
    hold_suffix=True; the library must return exactly those URLs."""
    tok = tok_with_org()
    model = SuffixStub(tok)
    tables = StateSuffixTables({"benign": {"Q1": {"com": 1, "org": 3},
                                           "Q0": {"com": 3, "org": 1}}})
    for fields in ({"QRY": True}, {"QRY": False}):
        kw = dict(label="benign", n=24, seed=11, max_rounds=100)
        got = generate_controlled(model, tok, state_tables=tables, fields=fields, **kw)
        want = generate_with_presence(model, tok, table=tables.for_state("benign", fields),
                                      fields=fields, hold_suffix=True, **kw)
        assert got == want and len(got) == 24
        assert got == generate_controlled(model, tok, state_tables=tables, fields=fields, **kw)


def test_no_control_is_refused_and_the_gate_path_is_untouched():
    tok = e7_tok()
    model = tiny_model(tok)
    tables = StateSuffixTables.from_rows(ROWS)
    for fields in (None, {}, {"PTH": None, "QRY": None}):
        with pytest.raises(ValueError):
            generate_controlled(model, tok, label="benign", n=4, state_tables=tables,
                                fields=fields)
    shipped = SuffixTable({"benign": {"com": 1, "org": 1}})
    kw = dict(label="benign", n=8, table=shipped, seed=5)
    assert generate_with_presence(model, tok, **kw) == generate_with_suffixes(model, tok, **kw)


@pytest.mark.skipif(not (BUNDLE / "name_train.jsonl.gz").is_file() or not SC1_TABLES.is_file(),
                    reason="needs data/bundle2 and data/sc1/tables.json")
def test_the_adopted_tables_are_sc1s_arm_t_tables():
    """Built from the bundle, the library's tables must equal the ones SC1's arm T
    was generated from (sha256 62aa7d9b... registered in PREREG_SC1's status) in
    all 14 measured states, and add only the two phishing states SC1 could not
    measure."""
    lib = StateSuffixTables.build(BUNDLE)
    sc1 = json.loads(SC1_TABLES.read_text(encoding="utf-8"))["T"]
    for tag, by_label in sc1.items():
        label, code = tag.split("_", 1)
        assert lib.counts[label][code] == by_label[label], tag
    measured = {tuple(tag.split("_", 1)) for tag in sc1}
    extra = {(l, c) for l, by in lib.counts.items() for c in by} - measured
    assert extra == {("phishing", "P1_Q1"), ("phishing", "P0_Q1")}
