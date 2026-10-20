"""REPRODUCE.md level 2 from the release alone (D2 step 6): the pinned shard maps, the published
fingerprints and score report, the reproduction script's job list and comparison, the notebook."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D2 = ROOT / "research" / "d2"
if str(D2) not in sys.path:
    sys.path.insert(0, str(D2))

import reproduce_gate as R  # noqa: E402


def test_every_live_crawl_has_a_complete_pinned_shard_map():
    from uie.corpus.extract import LIVE_CRAWLS
    from uie.corpus.stratified import ShardMap
    for crawl in LIVE_CRAWLS:
        m = ShardMap.load(ROOT / "data" / "corpus" / "shardmaps" / f"shardmap_{crawl}.json")  # refuses partial
        assert m.crawl == crawl and m.n_shards == 300
        assert all(s.replace(".", "").replace("-", "").isalnum() for v in m.shards.values() for s in v)


def test_the_job_list_is_the_published_runs_fifty_files():
    import e17_common as E17
    fp = R.fingerprints()
    jobs = E17.jobs(E17.DECODE_ARM)
    assert len(fp) == 50 and sorted(j["file"] for j in jobs) == sorted(fp)
    for j in jobs:
        assert {k: fp[j["file"]][k] for k in ("mode", "label", "state", "g", "n", "seed")} == \
            {k: j[k] for k in ("mode", "label", "state", "g", "n", "seed")}


def test_the_published_report_holds_every_compared_section_and_no_url():
    text = R.PUBLISHED_REPORT.read_text(encoding="utf-8")
    rep = json.loads(text)
    for sec in R.COMPARED:
        assert R.ARM in rep[sec], sec
    assert "http" not in text and ".com" not in text


def test_the_comparison_finds_any_changed_number_and_nothing_else():
    rep = json.loads(R.PUBLISHED_REPORT.read_text(encoding="utf-8"))
    sec = rep["emission"][R.ARM]
    assert R.same(sec, json.loads(json.dumps(sec))) == []
    changed = json.loads(json.dumps(sec))
    changed["benign"]["PTH"]["rate"] += 1e-6
    diffs = R.same(sec, changed)
    assert len(diffs) == 1 and "benign/PTH/rate" in diffs[0]
    nan = float("nan")                                      # unmeasurable cells: NaN on both sides
    assert R.same({"auc": nan}, {"auc": nan}) == [] and len(R.same({"auc": nan}, {"auc": 0.5})) == 1


def test_the_notebook_only_runs_the_script():
    nb = json.loads((D2 / "reproduce_gate.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        compile(src, "cell", "exec")
    assert any('"generate"' in src and "reproduce_gate.py" in src for src in code)


def test_a_rebuilt_corpus_is_shown_beside_both_measured_readings():
    """No pass/fail rule on a rebuilt corpus (the gate's uncertainty leaves out the reference
    sample): the gated cells are shown beside the published and the clean-room readings."""
    pub = json.loads(R.PUBLISHED_REPORT.read_text(encoding="utf-8"))
    clean = json.loads(R.CLEANROOM_REPORT.read_text(encoding="utf-8"))
    p, c = R.gated_cells(pub), R.gated_cells(clean)
    assert len(p) == len(c) == 32
    assert R.tally(p) == "29/2/1" and R.tally(c) == "29/1/2"          # E3's tally; the clean room's
    assert all(c[k][0] == p[k][0] for k in p if k[0] == "free")       # free mode kept every verdict
    lines = R.describe(pub, clean, clean)
    assert "published 29/2/1   our clean room 29/1/2   yours 29/1/2" in lines[0]
    moved = [l for l in lines if l.startswith("  ")]
    assert len(moved) == 3 and any("benign_P1Q1/PTH" in l for l in moved)
    assert R.describe(pub, clean, pub)[1].startswith("free mode: 10 of 10")
