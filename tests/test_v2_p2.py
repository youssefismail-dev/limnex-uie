"""V2-P2 (research/v2/PREREG_V2_P2.md): the tables' learner, the piece tokenizer, the training
seams and the piece rule. Toy scale only: nothing here trains a full-size model."""
from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "research" / "v2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import v2_p2_tables as T  # noqa: E402
from v2_bpe import BPETable  # noqa: E402
from v2_p2_tokenizer import HybridTokenizer, dropout_encode  # noqa: E402

RECORD = ROOT / "data" / "v2" / "p1b" / "e16_cap_record.json"
CORPUS = ROOT / "data" / "corpus2"
needs_record = pytest.mark.skipif(not RECORD.is_file(), reason="e16_cap's record is not on this machine")
needs_corpus = pytest.mark.skipif(not (CORPUS / "structure_val.jsonl.gz").is_file(), reason="corpus2 is absent")


def _record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _real_urls(n: int = 300) -> list:
    out = []
    for stem, label in (("name_val", "benign"), ("structure_val", "benign"),
                        ("phishing_name_val", "phishing"), ("phishing_structure_val", "phishing")):
        with gzip.open(CORPUS / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= n:
                    break
                out.append((json.loads(line)["url"], label))
    return out


def _toy_tables(urls) -> dict:
    """Small tables learned by the P2 learner from the URLs' own fields."""
    from uie.factorization import parse
    rest = {c: {} for c in T.CLASSES}
    host = {c: {} for c in T.CLASSES}
    for k, (u, label) in enumerate(urls):
        parts = parse(u)
        r, h = T.field_strings(parts)
        rest[label].setdefault(f"s{k % 97}", set()).update(r or [])
        host[label].setdefault(f"s{k % 89}", set()).update(h or [])
    for d in (rest, host):
        for c in T.CLASSES:
            for key in [k for k, v in d[c].items() if not v]:
                del d[c][key]
    out = {}
    for name, d in (("rest", rest), ("host", host)):
        bt, _ = T.learn_df(d, name, budget=300, log=None)
        out[name] = {"alphabet": bt.alphabet, "merges": [list(m) for m in bt.merges]}
    return out


# --------------------------------------------------------------------------- #
# the learner
# --------------------------------------------------------------------------- #

def _brute(by_class, table, budget):
    """Reference: every pair's score recomputed from scratch at each step."""
    n_units = {c: len(by_class[c]) for c in T.CLASSES}
    draw = {c: T.LABEL_COUNTS[c] / sum(T.LABEL_COUNTS.values()) for c in T.CLASSES}
    unit_w = [draw[c] / max(n_units[c], 1) for c in T.CLASSES]
    rate = [T.SAMPLE_RATE.get((table, c), 1.0) for c in T.CLASSES]
    rng = random.Random(T.SAMPLE_SEED)
    units = []
    for ci, cls in enumerate(T.CLASSES):
        for key in sorted(by_class[cls]):
            ss = by_class[cls][key]
            if len(ss) == 1:
                units.append([ci, True, [list(next(iter(ss)))]])
            elif rate[ci] >= 1.0 or rng.random() < rate[ci]:
                units.append([ci, False, [list(s) for s in sorted(ss)]])
    merges = []
    while len(merges) < budget:
        cnt = {}
        for ci, single, ws in units:
            pres = set()
            for w in ws:
                pres |= set(zip(w, w[1:]))
            for p in pres:
                cnt.setdefault(p, [0, 0, 0, 0])[(0 if single else 2) + ci] += 1
        shared = [p for p in cnt if sum(cnt[p]) >= 2]      # a pair in one source is never merged
        if not shared:
            break

        def sc(c):
            return unit_w[0] * (c[0] + (1.0 / rate[0]) * c[2]) + unit_w[1] * (c[1] + (1.0 / rate[1]) * c[3])
        best = min(shared, key=lambda p: (-sc(cnt[p]), p[0], p[1]))
        a, b = best
        merges.append((a, b))
        for u in units:
            new_ws = []
            for w in u[2]:
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        out.append(a + b)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                new_ws.append(out)
            u[2] = new_ws
    return merges


@pytest.mark.parametrize("trial", range(4))
@pytest.mark.parametrize("table", ["rest", "host"])
def test_learner_equals_brute_force(trial, table):
    rng = random.Random(100 + trial)
    by_class = {c: {} for c in T.CLASSES}
    for c in T.CLASSES:
        for k in range(rng.randint(20, 50)):
            n = rng.choice([1, 1, 1, 2, 3, 12])
            by_class[c][f"{c}{k}.com"] = {"".join(rng.choice("abcd-/") for _ in range(rng.randint(2, 8)))
                                          for _ in range(n)}
    bt, meta = T.learn_df(by_class, table, budget=120, log=None)
    assert list(bt.merges) == _brute(by_class, table, 120)
    assert all(s >= 2 for s in meta["merge_support"])
    assert meta["literal_stop_at"] is not None      # a single-source pair outranked a shared one


def test_one_source_cannot_make_a_piece():
    """A string repeated across one source's many strings counts once: it never merges, even
    when its class weight ranks it first; the pairs several sources share still merge."""
    by_class = {"benign": {"big.com": {f"zqxjk-{i}" for i in range(40)}},
                "phishing": {f"p{i}.com": {"login"} for i in range(5)}}
    bt, meta = T.learn_df(by_class, "host", budget=50, log=None)
    made = {a + b for a, b in bt.merges}
    assert not any("zq" in m or "xj" in m for m in made)
    assert "login" in made
    assert meta["literal_stop_at"] == 0             # the literal reading would have learned nothing
    assert meta["stop"] == "no pair shared by two sources"


def test_edit_similarity_matches_levenshtein():
    def lev(a, b):
        prev = list(range(len(b) + 1))
        for i in range(1, len(a) + 1):
            cur = [i] + [0] * len(b)
            for j in range(1, len(b) + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            prev = cur
        return prev[-1]
    rng = random.Random(5)
    for _ in range(400):
        a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 12)))
        b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 12)))
        exact = 1 - lev(a, b) / max(len(a), len(b), 1)
        got = T.edit_similarity(a, b)
        if exact > T.EDIT_SIM:
            assert got == pytest.approx(exact)
        else:
            assert got <= T.EDIT_SIM + 1e-12


def test_near_duplicate_clusters_campaigns_not_words():
    params = T._minhash_params()
    labels = {f"case10009zwpol0lm10oi2208fvupvlm4kq{x}.github.io": f"case10009zwpol0lm10oi2208fvupvlm4kq{x}"
              for x in ("", "ao", "a", "b")}
    labels.update({"brandx-secure-login.github.io": "brandx-secure-login",
                   "brandx-account-verify.github.io": "brandx-account-verify"})
    groups = T.cluster_platform(labels, params)
    assert len(groups) == 1 and len(groups[0]) == 4
    assert all("brandx" not in k for k in groups[0])


# --------------------------------------------------------------------------- #
# the tokenizer
# --------------------------------------------------------------------------- #

def test_dropout_at_zero_is_canonical():
    rng = random.Random(3)
    strings = ["".join(rng.choice("ab/-.") for _ in range(rng.randint(1, 30))) for _ in range(300)]
    words = {s: 1 for s in strings}
    from v2_bpe import learn
    table = learn(words, 60, log=None)
    for s in strings:
        assert dropout_encode(table, s, random.Random(1), 0.0) == table.encode(s)


@needs_record
@needs_corpus
@pytest.mark.parametrize("arm", ["a", "b"])
def test_tokenizer_lossless_canonical_and_dropout(arm):
    urls = _real_urls(150)
    tables = _toy_tables(urls)
    tok = HybridTokenizer(base_state=_record()["tokenizer"], rest=tables["rest"],
                          host=tables["host"] if arm == "b" else None, dropout=0.1)
    base = tok.base
    assert tok.itos[:tok.n_base] == base.itos
    for k, (u, label) in enumerate(urls):
        c = base.encode(u, label)
        for ids in (tok.encode(u, label), tok.encode(u, label, rng=random.Random(k), p=0.1)):
            assert tok.decode(ids) == base.decode(c)
            assert tok.chars_of(ids) == c
        assert tok.encode(u, label, rng=random.Random(k), p=0.0) == tok.encode(u, label)
        assert (tok.encode(u, label, rng=random.Random(9 + k), p=0.1)
                == tok.encode(u, label, rng=random.Random(9 + k), p=0.1))          # V-drop
    again = HybridTokenizer.from_state_dict(json.loads(json.dumps(tok.state_dict())))
    assert all(again.encode(u, l) == tok.encode(u, l) for u, l in urls[:100])
    # pieces only where their table writes
    for u, label in urls:
        ids = tok.encode(u, label)
        field = None
        for t in ids:
            if t in tok._marker_of:
                field = tok._marker_of[t]
            elif t >= tok.n_base:
                assert tok.field_tables().get(field) == tok.piece_table[t]


# --------------------------------------------------------------------------- #
# the training seams
# --------------------------------------------------------------------------- #

def _toy_data(tmp: Path, n: int = 300) -> Path:
    data = tmp / "data"
    data.mkdir()
    import uie.kaggle_train as KT
    for stem, label, corpus in KT.STREAMS + KT.VAL_STREAMS:
        src = CORPUS / f"{stem.replace('_train', '_val')}.jsonl.gz"
        rows = []
        with gzip.open(src, "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= n:
                    break
                r = json.loads(line)
                rows.append(json.dumps({"url": r["url"], "label": label, "lang": r.get("lang", "und")}))
        (data / f"{stem}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return data


def _toy_stream(tmp: Path, n: int = 40) -> Path:
    """Stream-1 rows short enough for the toy's max_len (256; the real runs use 384)."""
    src = ROOT / "data" / "e15" / "qry_extra" / "qry_extra.jsonl"
    lines = [x for x in src.read_text(encoding="utf-8").splitlines()
             if x.strip() and json.loads(x)["stratum"].endswith("P1Q1") and len(json.loads(x)["url"]) <= 180][:n]
    path = tmp / "stream.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


TOY = dict(steps=4, batch_size=8, lr=3e-4, d_model=32, n_layers=2, n_heads=2, eval_every=2,
           frame=True, fields=True, presence_plan=True, root_slash=True, slash_plan=True,
           seed_init=True, e12_start=True, max_len=256, loss_norm="fixed", accum_steps=2)


def _run_toy(tmp: Path, name: str, data: Path, stream: Path, p2: dict = None):
    import uie.kaggle_train as KT
    import v2_p2_train as P2T
    out = tmp / name
    stub = KT.generate
    KT.generate = lambda *a, **k: ["a.com"] * k.get("n", 1)     # sampling is not under test
    try:
        if p2 is None:
            KT.run(data, out, qry_extra=str(stream), **TOY)
        else:
            P2T.run(data, out, qry_extra=str(stream), **p2, **TOY)
    finally:
        KT.generate = stub
    blob = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
    return blob, json.loads((out / "train_result.json").read_text(encoding="utf-8"))


needs_stream = pytest.mark.skipif(not (ROOT / "data" / "e15" / "qry_extra" / "qry_extra.jsonl").is_file(),
                                  reason="stream 1 is absent")


@needs_corpus
@needs_stream
def test_v_equiv_character_mode_is_bit_identical(tmp_path):
    """V-equiv: the P2 seams with no tables, no dropout and no scope train exactly the
    unpatched run (weights, losses, batch digest, injection counts)."""
    torch.set_num_threads(1)
    data, stream = _toy_data(tmp_path), _toy_stream(tmp_path)
    a, ra = _run_toy(tmp_path, "plain", data, stream)
    b, rb = _run_toy(tmp_path, "seams", data, stream, dict(rest=None, host=None, dropout=0.0, scope=False))
    assert a["model"].keys() == b["model"].keys()
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert ra["final"] == rb["final"] and ra["history"] == rb["history"]
    assert ra["field_nll"] == rb["field_nll"]                           # per character = per token
    assert ra["qry_extra"] == rb["qry_extra"]                           # digest, rows, positions
    assert ra["dropped_over_max_len"] == rb["dropped_over_max_len"]


@needs_corpus
@needs_stream
@needs_record
def test_piece_run_keeps_the_batches_and_scopes_names(tmp_path):
    """With pieces, dropout and the name scope: the same rows in the same batches (digest),
    the same injection (rows and character positions), and a P2 record."""
    torch.set_num_threads(1)
    data, stream = _toy_data(tmp_path), _toy_stream(tmp_path)
    tables = _toy_tables(_real_urls(150))
    _, ra = _run_toy(tmp_path, "plain", data, stream)
    blob, rb = _run_toy(tmp_path, "pieces", data, stream,
                        dict(rest=tables["rest"], host=tables["host"], dropout=0.1, scope=True))
    assert rb["qry_extra"]["old_batch_digest"] == ra["qry_extra"]["old_batch_digest"]
    for s, v in ra["qry_extra"]["strata"].items():
        for f in ("old_rows_seen", "appended_rows", "appended_query_positions"):
            assert rb["qry_extra"]["strata"][s][f] == v[f]
    assert rb["dropped_over_max_len"] == ra["dropped_over_max_len"]
    assert rb["p2"]["pieces"]["rest"] > 0 and rb["p2"]["name_scope"] is True
    tok = HybridTokenizer.from_state_dict(blob["tokenizer"])
    assert tok.vocab_size == blob["model_config"]["vocab_size"]


@needs_record
@needs_corpus
def test_name_scope_mask(tmp_path):
    """V-scope (section 12): on a page row, name-field content (SUF, LBL, SUB, USR) is never a
    target; markers, plan and end tokens are; every other weight is field_mask's (e16_cap's
    mask, the frame on). A name row's mask is field_mask's exactly."""
    import uie.train as UT
    import v2_p2_train as P2T
    tables = _toy_tables(_real_urls(100))
    tok = HybridTokenizer(base_state=_record()["tokenizer"], rest=tables["rest"], host=tables["host"])
    plan = {t for slot in tok.plan_slots() for t in slot}
    P2T._STATE.update({"scope": True})
    zeroed = 0
    try:
        for u, label in _real_urls(60):
            ids = tok.encode(u, label, frame="page")
            m = P2T.row_mask(ids, tok, "structure")
            full = UT.field_mask(ids, tok, "structure")              # e16_cap's: every field taught
            current = None
            for t, w, wf in zip(ids[1:], m, full):
                if t in tok._marker_of:
                    current = tok._marker_of[t]
                    assert w == wf == 1.0
                elif t in (tok.eos_id, tok.rs_id) or t in plan:
                    assert w == wf == 1.0
                    if t == tok.rs_id:
                        current = None                       # field_mask: nothing follows <RS>'s field
                elif current in ("SUF", "LBL", "SUB", "USR") and tok.token_text(t) is not None:
                    assert w == 0.0
                    zeroed += wf == 1.0
                else:
                    assert w == wf
            name_ids = tok.encode(u, label)
            assert P2T.row_mask(name_ids, tok, "name") == UT.field_mask(name_ids, tok, "name")
    finally:
        P2T._STATE.clear()
    assert zeroed > 0                                   # the scope did remove name content


@needs_record
def test_start_peels_off_to_e16_caps_start():
    """V-start': the piece rows are appended; every other weight is this torch's e16_cap
    start exactly. (Only Kaggle's torch rebuilds the recorded caca0107...; the laptop's
    differs, as E16's amendment 2 found, so the notebook's start check compares that.)"""
    import v2_p2_common as C
    from uie.free import weights_sha256
    from uie.kaggle_train import new_model_e12
    from uie.model import ModelConfig
    tables = {"rest": {"alphabet": list("ab"), "merges": [["a", "b"]]},
              "host": {"alphabet": list("cd"), "merges": [["c", "d"]]}}
    tok = HybridTokenizer(base_state=_record()["tokenizer"], **tables)
    model, prints = C.start(tok)
    r = C.E16.recipe_cap(Path(C.STREAM_FILE))
    base, _ = new_model_e12(ModelConfig(vocab_size=tok.n_base, d_model=r["d_model"], n_layers=r["n_layers"],
                                        n_heads=r["n_heads"], d_ff=4 * r["d_model"], struct_ids=(),
                                        max_len=r["max_len"], frame_every=r.get("frame_every", False)),
                            tok.base, C.E16.SEED)
    assert prints["base_init_sha256"] == weights_sha256(base) == weights_sha256(C.peel(model, tok.n_base))
    assert prints["params"] == C.params(tok) == C.E16_CAP_PARAMS + 2 * 512
    sb, sm = base.state_dict(), model.state_dict()
    assert sb.keys() == sm.keys()
    for k in sb:
        if k in ("tok_emb.weight", "head.weight"):
            assert sm[k].shape[0] == tok.vocab_size == tok.n_base + 2
            assert torch.equal(sm[k][:tok.n_base], sb[k])
        else:
            assert torch.equal(sm[k], sb[k])
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


# --------------------------------------------------------------------------- #
# generation: the piece rule
# --------------------------------------------------------------------------- #

@needs_record
def test_piece_rule_masks_pieces_outside_their_fields():
    import v2_p2_common as C
    from uie.model import ModelConfig, URLTransformer
    tables = {"rest": {"alphabet": list("ab"), "merges": [["a", "b"]]},
              "host": {"alphabet": list("cd"), "merges": [["c", "d"]]}}
    tok = HybridTokenizer(base_state=_record()["tokenizer"], **tables)
    torch.manual_seed(0)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1, n_heads=2,
                                       d_ff=32, max_len=64))
    rule = C.PieceRule(model, tok).eval()
    rest_id, host_id = tok.piece_ids[("rest", "ab")], tok.piece_ids[("host", "cd")]
    pre = [tok.bos_id, tok.stoi["<C:benign>"], tok.stoi["<F:domain>"], tok.marker_id("SUF"), tok.stoi["c"]]
    cases = {"SUF": pre, "LBL": pre + [tok.marker_id("LBL")], "PTH": pre + [tok.marker_id("LBL"), tok.marker_id("PTH")]}
    with torch.no_grad():
        for field, ids in cases.items():
            x = torch.tensor([ids])
            z = rule(x)[0, -1]
            assert torch.isfinite(z[rest_id]) == (field == "PTH")
            assert torch.isfinite(z[host_id]) == (field == "LBL")
            full = model(x)[0, -1]                      # the last position, the head over all
            keep = torch.isfinite(z)
            assert torch.allclose(z[keep], full[keep], atol=1e-5)
    rec = rule.record()
    assert set(rec["fields"]) == {"SUF", "LBL", "PTH"} and rec["violations"] == 0
    # a piece written outside its field is counted when it comes back as input
    with torch.no_grad():
        rule(torch.tensor([pre + [rest_id]]))           # a rest piece inside SUF
    assert rule.record()["violations"] == 1


@needs_record
@pytest.mark.parametrize("mode", ["free", "control"])
def test_generation_calls_equal_e17s_for_characters(mode):
    """P2's free_urls / control_urls (E17's body plus the cut record) give E17's own calls'
    URLs exactly for a character model; asking for the cut record changes nothing."""
    import v2_p2_common as C              # first: it puts research/e17 on the path
    import e17_common as E17
    from uie.controlled import StateSuffixTables
    from uie.model import ModelConfig, URLTransformer
    from uie.suffixes import SuffixTable
    tok = HybridTokenizer(base_state=_record()["tokenizer"])
    torch.manual_seed(3)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1, n_heads=2,
                                       d_ff=32, max_len=48)).eval()
    shipped = SuffixTable.load(ROOT / "data" / "bundle2" / "suffix_table.json")
    st = StateSuffixTables.load(ROOT / "data" / "state_suffix_tables.json")
    ps = json.loads((ROOT / "data" / "e12" / "presence_slash_table.json").read_text(encoding="utf-8"))["table"]
    rec = {}
    if mode == "free":
        want = E17.free_urls(model, tok.base, "phishing", 0, shipped, ps, b5=True, n=24)
        got = C.free_urls(model, tok, "phishing", 0, shipped, ps, n=24)
        got_rec = C.free_urls(model, tok, "phishing", 0, shipped, ps, n=24, record=rec)
    else:
        want = E17.control_urls(model, tok.base, "benign", "P1Q1", 0, st, ps, b5=True, n=24)
        got = C.control_urls(model, tok, "benign", "P1Q1", 0, st, ps, n=24)
        got_rec = C.control_urls(model, tok, "benign", "P1Q1", 0, st, ps, n=24, record=rec)
    assert got == want == got_rec
    cut, rules = rec["cut_at_max_len"], rec["token_rules"]
    assert set(cut) <= set(got) and len(cut) > 0                 # max_len 48: some rows are cut
    assert rules["rows"] == 24 and rules["no_plan"] == rules["plan_mismatch"] == 0
    assert rec["slash_moved_mean"] is not None and rec["slash_rows_moved"] >= 0
    assert rules["d1_violations"] == rules["b5_violations"] == 0  # the decode enforces both
    assert rules["d1_checked"] > 0
    assert (rules["d1_checked"] + rules["d1_cut"] + rules["cut_before_host"]
            + rules["ended_before_host"]) == 24


@needs_record
def test_token_rules_count_violations():
    """token_rules on hand-made rows: a host ended against S, and a Q = 1 row that ended
    before its query, are counted; a row cut before the host ends is counted apart."""
    import v2_p2_common as C
    tok = HybridTokenizer(base_state=_record()["tokenizer"])
    head = [tok.bos_id, tok.stoi["<C:benign>"], tok.stoi["<F:domain>"], tok.marker_id("SUF"), tok.stoi["c"]]

    def row(p, q, s, tail):
        return head + tok.plan_ids({"PTH": p, "QRY": q, "SLASH": s}) + [tok.marker_id("LBL"), tok.stoi["a"]] + tail
    rows = [row(False, False, True, [tok.eos_id]),                     # S = 1 ended bare: D1 violation
            row(False, True, False, [tok.eos_id]),                     # Q = 1 ended: B5 violation (S0 ok)
            row(True, True, True, [tok.marker_id("PTH"), tok.stoi["b"], tok.marker_id("QRY"), tok.eos_id]),
            row(False, False, False, [tok.stoi["b"]]),                 # cut inside the host
            [tok.marker_id("QRY")] + row(False, True, False, [tok.eos_id])]   # B5 closed before <LBL>
    r = C.token_rules(tok, rows)
    assert (r["d1_checked"], r["d1_violations"], r["d1_cut"]) == (4, 1, 1)
    assert (r["b5_checked"], r["b5_violations"], r["b5_cut"], r["b5_closed_before_host"]) == (2, 1, 0, 1)
    # with the supplied plans given, a row's own plan tokens are cross-checked
    plans = ["P0Q0S1", "P0Q1S0", "P1Q1S1", "P0Q0S0", "P0Q1S1"]
    r2 = C.token_rules(tok, rows, plans)
    assert r2["plan_mismatch"] == 1 and r2["d1_violations"] == 2   # the last row's S0 tokens vs S1


def test_piece_rule_is_a_no_op_for_a_character_tokenizer():
    import v2_p2_common as C
    from uie.model import ModelConfig, URLTransformer
    if not RECORD.is_file():
        pytest.skip("no record")
    tok = HybridTokenizer(base_state=_record()["tokenizer"])
    torch.manual_seed(1)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1, n_heads=2,
                                       d_ff=32, max_len=64)).eval()
    ids = torch.tensor([[tok.bos_id, tok.stoi["<C:benign>"], tok.marker_id("SUF")]])
    with torch.no_grad():
        assert torch.equal(C.PieceRule(model, tok)(ids), model(ids))


# --------------------------------------------------------------------------- #
# the chunked vocabulary head (memory, Amendment 3)
# --------------------------------------------------------------------------- #

def _pair(vocab: int = 3000, seed: int = 4):
    """A URLTransformer and a PieceTransformer with the same weights, dropout off."""
    import v2_p2_train as P2T
    from uie.model import ModelConfig, URLTransformer
    cfg = ModelConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=2, d_ff=64, max_len=64)
    torch.manual_seed(seed)
    a = URLTransformer(cfg).eval()
    b = P2T.PieceTransformer(cfg).eval()
    b.load_state_dict(a.state_dict())
    return a, b


@pytest.mark.parametrize("weighted", [True, False])
def test_chunked_loss_equals_the_full_loss(monkeypatch, weighted):
    """The same loss and the same gradients as URLTransformer.loss, in many chunks."""
    import v2_p2_train as P2T
    monkeypatch.setattr(P2T, "CHUNK_LOGITS", 3000 * 7)          # 7 positions per chunk
    a, b = _pair()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(4, 3000, (5, 40), generator=g)
    ids[1, 25:] = 0                                              # padding
    ids[3, 12:] = 0
    w = (torch.rand((5, 39), generator=g) > 0.3).float() if weighted else None
    la, lb = a.loss(ids, 0, w), b.loss(ids, 0, w)
    assert torch.allclose(la, lb, atol=1e-6, rtol=1e-6)
    la.backward()
    lb.backward()
    ga, gb = dict(a.named_parameters()), dict(b.named_parameters())
    for k in ga:
        assert torch.allclose(ga[k].grad, gb[k].grad, atol=1e-6, rtol=1e-5), k
    with torch.no_grad():
        assert torch.allclose(a.loss(ids, 0, w), b.loss(ids, 0, w), atol=1e-6, rtol=1e-6)


def test_hidden_states_restore_the_head():
    import v2_p2_train as P2T
    a, b = _pair()
    ids = torch.randint(4, 3000, (2, 10))
    head = b.head
    with torch.no_grad():
        h = P2T.hidden_states(b, ids)
        assert h.shape == (2, 10, 32)
        assert torch.allclose(b.head(h), a(ids), atol=1e-5)
    assert b.head is head and b.head.weight is b.tok_emb.weight


@needs_record
def test_fit_per_character_chunked_equals_full(monkeypatch):
    """field_nll_chars on a piece model: the chunked head gives the full head's numbers."""
    import v2_p2_train as P2T
    from uie.model import ModelConfig, URLTransformer
    urls = _real_urls(40)
    tables = _toy_tables(urls)
    tok = HybridTokenizer(base_state=_record()["tokenizer"], **tables)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=2, d_ff=64, max_len=384)
    torch.manual_seed(2)
    model = URLTransformer(cfg).eval()
    monkeypatch.setattr(P2T, "CHUNK_LOGITS", tok.vocab_size * 50)
    got = P2T.field_nll_chars(model, tok, [u for u, l in urls if l == "benign"], "benign",
                              torch.device("cpu"), min_tokens=1)
    monkeypatch.setattr(P2T, "CHUNK_LOGITS", 1 << 40)          # one chunk: the full head
    want = P2T.field_nll_chars(model, tok, [u for u, l in urls if l == "benign"], "benign",
                               torch.device("cpu"), min_tokens=1)
    assert got.keys() == want.keys() and all(abs(got[k] - want[k]) < 1e-5 for k in got)


@needs_record
@needs_corpus
def test_shape_nll_chunked_equals_uies():
    """The after-training structure NLL from the chunked head equals uie.evaluation.shape_nll."""
    import v2_p2_train as P2T
    from uie.evaluation import shape_nll
    from uie.model import ModelConfig, URLTransformer
    urls = _real_urls(30)
    tables = _toy_tables(urls)
    tok = HybridTokenizer(base_state=_record()["tokenizer"], **tables)
    torch.manual_seed(5)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=2,
                                       d_ff=64, max_len=384)).eval()
    bench = [u for u, l in urls if l == "phishing"]
    a = shape_nll(model, tok, bench, "phishing", torch.device("cpu"))
    b = P2T.shape_nll_chunked(model, tok, bench, "phishing", torch.device("cpu"))
    assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-5 for k in a)
    assert "skipped" in P2T.boundary_nll_skipped(model, tok, bench, "phishing", torch.device("cpu"))


@needs_record
@needs_corpus
def test_after_training_generate_calls_run_on_a_piece_model():
    """uie.kaggle_train._run's own after-training calls (the samples with three decoders, and
    the quick honouring check with a supplied plan and suffix), on a piece model and tokenizer:
    the toy runs stub `generate`, so they are checked here."""
    import v2_p2_train as P2T
    from uie.model import ModelConfig, generate
    tables = _toy_tables(_real_urls(40))
    tok = HybridTokenizer(base_state=_record()["tokenizer"], **tables)
    torch.manual_seed(6)
    model = P2T.PieceTransformer(ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layers=1, n_heads=2,
                                             d_ff=32, max_len=64))
    for decoder in ("typical", "top_p", "temperature"):
        urls = generate(model, tok, label="benign", n=5, decoder=decoder, seed=20260803, device=torch.device("cpu"))
        assert len(urls) == 5 and all(isinstance(u, str) for u in urls)
    plan = {"PTH": True, "QRY": False, "SLASH": True}
    urls = generate(model, tok, label="phishing", n=5, decoder="temperature", seed=20260923,
                    device=torch.device("cpu"), suffix="com", fields=plan)
    assert len(urls) == 5


@needs_record
def test_the_runs_start_is_the_notebooks_start():
    """The start training builds (the patched new_model_e12, at the run's own config and seed)
    is the start the notebook checks before training (v2_p2_common.start), at full size: the
    readback's "the whole start" row can then only fail if Kaggle's torch changes mid-session."""
    import v2_p2_common as C
    import v2_p2_train as P2T
    from uie.free import weights_sha256
    from uie.model import ModelConfig
    from uie.train import TrainConfig
    tables = {"rest": {"alphabet": list("ab"), "merges": [["a", "b"]]},
              "host": {"alphabet": list("cd"), "merges": [["c", "d"]]}}
    tok = HybridTokenizer(base_state=_record()["tokenizer"], **tables)
    r = C.E16.recipe_cap(Path(C.STREAM_FILE))
    assert TrainConfig().seed == C.E16.SEED
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=r["d_model"], n_layers=r["n_layers"], n_heads=r["n_heads"],
                      d_ff=4 * r["d_model"], struct_ids=(), max_len=r["max_len"], frame_every=r.get("frame_every", False))
    P2T.patch(rest=tables["rest"], host=tables["host"], dropout=0.1, scope=True)
    try:
        run_model, run_prints = P2T._start(cfg, tok, TrainConfig().seed)
    finally:
        P2T.restore()
    _, want = C.start(tok)
    assert weights_sha256(run_model) == want["init_weights_sha256"]
    assert {k: run_prints[k] for k in ("base_init_sha256", "rest_rows_sha256", "host_rows_sha256")} == \
        {k: want[k] for k in ("base_init_sha256", "rest_rows_sha256", "host_rows_sha256")}
    assert isinstance(run_model, P2T.PieceTransformer)


def test_memory_preflight_runs_on_the_cpu():
    """The worst-case check's code path at full width (on the CPU it measures nothing)."""
    import v2_p2_common as C
    assert C.memory_preflight(torch.device("cpu"), rows=1, vocab_size=500) == 0.0
