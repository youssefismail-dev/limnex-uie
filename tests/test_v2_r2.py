"""V2-R2 (research/v2/PREREG_R2.md): the split, the oracles, the honesty rule and the lexicon
wrapper, checked on toy inputs before anything runs. No data is loaded."""
import importlib.util
import inspect
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def r2():
    sys.path.insert(0, str(ROOT / "research" / "v2"))
    spec = importlib.util.spec_from_file_location("v2_r2", ROOT / "research" / "v2" / "v2_r2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v2_r2"] = mod
    spec.loader.exec_module(mod)
    return mod


TOY = {"login": 50, "logo": 20, "shop": 30, "wave": 12, "secure": 40, "verify": 15,
       "account": 25, "home": 60, "com": 99, "lantic": 1, "zebra": 2}


@pytest.fixture()
def mem(r2, monkeypatch):
    monkeypatch.setattr(r2, "RARE_RANK", 6)           # the six commonest are not "rare" here
    return r2.Memory(dict(TOY))


def test_half_is_deterministic_and_balanced(r2):
    assert r2.half_of("Example.COM") == r2.half_of("example.com")
    halves = Counter(r2.half_of(f"site{i}.com") for i in range(4000))
    assert set(halves) == {"A", "B"} and abs(halves["A"] - 2000) < 150


def test_recase(r2):
    assert r2.recase("LOGIM", "login") == "LOGIN"
    assert r2.recase("Logim", "login") == "Login"
    assert r2.recase("logim", "login") == "login"
    assert r2.recase("lOgIm", "login") == "login"


def test_field_spans(r2):
    assert r2._rest_fields("/a/b?x=1#top") == [(0, 4, "PTH"), (4, 8, "QRY"), (8, 12, "FRG")]
    assert r2._rest_fields("/a#b?c") == [(0, 2, "PTH"), (2, 6, "FRG")]
    assert r2._rest_fields("?q") == [(0, 0, "PTH"), (0, 2, "QRY")]
    spans = r2._host_fields("login.secure.brandx.com")
    assert [f for _, _, f in spans] == ["SUB", "SUB", "LBL", "SUF"]


def test_memory(mem):
    assert mem.nearest_frequent("logim") == "login"       # one edit, seen >= 10 times
    assert mem.nearest_frequent("lantix") is None          # its neighbour is seen once
    assert mem.is_prefix("log") and mem.is_prefix("logo") and not mem.is_prefix("logz")
    rng = random.Random(1)
    draws = {mem.draw_rare(rng) for _ in range(200)}
    assert draws <= {"logo", "verify", "wave", "zebra", "lantic"} and "home" not in draws


def test_oracles(r2, mem):
    rng, stats = random.Random(0), Counter()
    s = "h.h/logim/shop/qqqzzz?next=homx"
    o1 = r2.transform_url(s, "rest", "O1", mem, {}, rng, stats)
    assert o1 == "h.h/login/shop/qqqzzz?next=home"         # slips fixed, the rest untouched
    o2 = r2.transform_url(s, "rest", "O2", mem, {}, random.Random(0), Counter())
    words = r2.WORD.findall(o2)
    assert all(w in TOY for w in words) and "shop" in words  # every fake replaced, known kept
    kept = r2.transform_url(s, "rest", "O2e", mem, {"PTH": 1.0, "QRY": 1.0}, random.Random(0), Counter())
    assert kept == s                                          # keep 1: invention kept as is
    host = r2.transform_url("secur.paypol.com", "host", "O1", mem, {}, random.Random(0), Counter())
    assert host.endswith(".com") and host.startswith("secure.")   # the suffix is never touched


def test_honesty_rule(r2):
    """V-honest: the code that builds oracles and the wrapper never opens Vocab_B."""
    src = "".join(inspect.getsource(f) for f in (r2.part_oracle, r2.LexiconWrapper, r2.part_gen,
                                                   r2.Memory, r2.transform, r2.keep_rates))
    assert "judge_vocab" not in src and '"B")' not in src
    assert "memory_vocab" in inspect.getsource(r2.part_oracle)


class _Dummy:
    class cfg:
        max_len = 384

    def parameters(self):
        import torch
        yield torch.zeros(1)

    def eval(self):
        return self

    def __call__(self, ids):
        import torch
        return torch.zeros(ids.shape[0], ids.shape[1], 300)


@pytest.fixture()
def wrap(r2, mem):
    from uie.tokenizer import URLTokenizer
    tok = URLTokenizer(include_frame=True, include_fields=True, root_slash=True, presence_plan=True,
                       slash_plan=True).fit(["http://abcdefghijklmnopqrstuvwxyz.com/ABCDEFGHIJKLMNOPQRSTUVWXYZ/x?a=1#f"])
    return r2.LexiconWrapper(_Dummy(), tok, mem, {"PTH": 0.0, "QRY": 0.0, "LBL": 1.0}), tok


def _row(tok, path_text):
    ids = [tok.bos_id, tok.stoi["<C:benign>"], tok.stoi["<F:domain>"], tok.marker_id("SUF")]
    ids += [tok.stoi[c] for c in "com"] + tok.plan_ids({"PTH": True, "QRY": False, "SLASH": True})
    ids += [tok.marker_id("LBL")] + [tok.stoi[c] for c in "abc"] + [tok.marker_id("PTH")]
    return ids + [tok.stoi[c] for c in path_text]


def test_wrapper_constrains_words(wrap):
    w, tok = wrap
    letters = {c: tok.stoi[c] for c in "abcdefghijklmnopqrstuvwxyz"}
    bad = set(w.disallowed(_row(tok, "lo")).tolist())
    assert letters["g"] not in bad and letters["x"] in bad and tok.stoi["G"] not in bad
    assert tok.eos_id not in bad                              # under 3 letters it may end
    bad = set(w.disallowed(_row(tok, "logi")).tolist())
    assert letters["n"] not in bad and letters["o"] in bad and tok.eos_id in bad   # must finish "login"
    bad = w.disallowed(_row(tok, "logo"))
    assert bad is None or tok.eos_id not in set(bad.tolist())      # a complete word may end
    assert w.disallowed(_row(tok, "")) is None                    # a word may start with anything
    assert w.disallowed(_row(tok, "login/")) is None


def test_wrapper_free_fields_and_violations(r2, wrap, mem):
    w, tok = wrap
    free = r2.LexiconWrapper(_Dummy(), tok, mem, {"PTH": 1.0})
    assert free.disallowed(_row(tok, "logi")) is None            # keep 1: the word is free
    row = _row(tok, "logz")                                       # a letter the rule forbids
    w.check_last(row)
    assert w.stats["violations"] == 1
    w.check_last(_row(tok, "login/"))
    assert w.stats["violations"] == 1                             # a complete word ended: fine


def test_dict_vocab(r2):
    v = r2.DictVocab({"home": 3})
    assert list(v.counts(["home", "zzz"])) == [3, 0]
