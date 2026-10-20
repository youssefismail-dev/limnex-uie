"""research/v2/v2_bpe.py: the deterministic BPE learner R1 uses to count coverage
(research/v2/PREREG_R1.md section 3, R1-4; V-BPE in section 4)."""
import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bpe():
    spec = importlib.util.spec_from_file_location("v2_bpe", ROOT / "research" / "v2" / "v2_bpe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v2_bpe"] = mod
    spec.loader.exec_module(mod)
    return mod


#: Sennrich et al.'s toy corpus. Worked by hand: (e,s) and (s,t) tie at 9 and code-point order
#: takes (e,s) first; then (l,o) beats (o,w) at 7; then (e,w) beats (n,e) and (w,est) at 6.
TOY = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
TOY_MERGES = [("e", "s"), ("es", "t"), ("l", "o"), ("lo", "w"), ("e", "w"), ("ew", "est"),
              ("n", "ewest"), ("d", "est"), ("i", "dest"), ("w", "idest"), ("e", "r"),
              ("low", "er")]


def test_the_worked_example_including_ties(bpe):
    table = bpe.learn(TOY, 100, log=None)
    assert table.merges == TOY_MERGES          # every word ends as one piece: 12 merges, no more
    assert table.alphabet == sorted(set("lowernsti d".replace(" ", "")))


def test_weights_decide_before_ties(bpe):
    assert bpe.learn({"ab": 3, "ac": 2}, 1, log=None).merges == [("a", "b")]
    assert bpe.learn({"ab": 2, "ac": 2}, 1, log=None).merges == [("a", "b")]   # tie: code points
    assert bpe.learn({"ab": 2, "ac": 3}, 1, log=None).merges == [("a", "c")]


def test_encoding_follows_rank_and_limit(bpe):
    table = bpe.learn(TOY, 100, log=None)
    assert table.encode("newest") == ["newest"]
    assert table.encode("lowest") == ["low", "est"]
    assert table.encode("lowest", limit=2) == ["l", "o", "w", "est"]
    assert table.encode("") == []
    assert table.encode("x") == ["x"]


def test_learning_is_deterministic(bpe):
    rng = random.Random(7)
    corpus = {"".join(rng.choice("abcde/-=?") for _ in range(rng.randint(1, 12))): rng.randint(1, 5)
              for _ in range(400)}
    one, two = bpe.learn(corpus, 300, log=None), bpe.learn(dict(corpus), 300, log=None)
    assert one.merges == two.merges and one.sha256() == two.sha256()
    assert len(set(one.merges)) == len(one.merges)              # no merge is listed twice


def test_lossless_including_unseen_characters(bpe):
    rng = random.Random(11)
    corpus = {"".join(rng.choice("abc.x/") for _ in range(rng.randint(2, 15))): 1 for _ in range(300)}
    table = bpe.learn(corpus, 200, log=None)
    for _ in range(500):
        s = "".join(rng.choice("abc.x/ZÉ%") for _ in range(rng.randint(0, 30)))
        for limit in (0, 10, None):
            assert "".join(table.encode(s, limit=limit)) == s


def test_encode_at_equals_encode_with_each_limit(bpe):
    rng = random.Random(5)
    corpus = {"".join(rng.choice("abcd/.=") for _ in range(rng.randint(2, 14))): rng.randint(1, 4)
              for _ in range(500)}
    table = bpe.learn(corpus, 250, log=None)
    limits = (0, 1, 7, 50, 120, 250, 999)
    for _ in range(300):
        s = "".join(rng.choice("abcd/.=x") for _ in range(rng.randint(0, 25)))
        at = table.encode_at(s, limits)
        for k in limits:
            assert at[k] == table.encode(s, limit=k), (s, k)


def test_save_and_load_round_trip(bpe, tmp_path):
    table = bpe.learn(TOY, 100, log=None)
    sha = table.save(tmp_path / "t.json")
    back = bpe.BPETable.load(tmp_path / "t.json")
    assert back.merges == table.merges and back.alphabet == table.alphabet and back.sha256() == sha


def test_covered_words(bpe):
    assert bpe.covered_words("index.php", ["index", ".", "php"], [(0, 5), (6, 9)]) == [True, True]
    assert bpe.covered_words("index.php", ["ind", "ex", ".php"], [(0, 5), (6, 9)]) == [False, True]
    assert bpe.covered_words("a/login", ["a/log", "in"], [(2, 7)]) == [False]
    with pytest.raises(ValueError):
        bpe.covered_words("abc", ["ab"], [(0, 3)])
