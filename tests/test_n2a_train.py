"""N2a's arms (research/native/n2a_train.py; PREREG_N2A Amendments 2 and 3).

Pins, before any arm runs:
- the exposure constants ARE Amendment 3's arithmetic on E7's recorded counts;
- the schedule: A2 is exactly four times A1, warm-up and evaluation keep E7's
  shares (500 and 1,000 of 20,000);
- the sampler, given the registered class weights, delivers each class's
  registered draws per row;
- "query-bearing" is exactly the tokenizer's test for opening <QRY>, which is
  also the gate's QRY field;
- end to end on a tiny model: only query rows train, validation is hidden, every
  training setting except the three registered ones is E7's, E7's tokenizer is
  kept, generation comes from final.pt with the resting decode at Amendment 2's
  seeds, and a checkpoint that is not E7's is refused.
"""
import gzip
import hashlib
import importlib.util
import json
import random
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from test_e7_fields import FIT
from uie.model import ModelConfig, URLTransformer
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.tokenizer import URLTokenizer
from uie.train import TrainConfig, URLDataset, batches, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "n2a_train_under_test", ROOT / "research" / "native" / "n2a_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


N = _load()

# E7's recorded counts: name-corpus label counts (class_alpha weights by these)
# and kept rows per class (e8_base's encoding, identical to E7's record).
E7_LABEL_COUNTS = {"benign": 906_214, "phishing": 192_652}
E7_KEPT = {"benign": 2_545_305, "phishing": 238_199}
E7_DRAWS = 20_000 * 256


# ---- the registered arithmetic ------------------------------------------------------

def test_exposure_constants_are_amendment_3s_arithmetic_on_e7s_counts():
    total = sum(E7_LABEL_COUNTS.values())
    for c in ("benign", "phishing"):
        per_row = E7_DRAWS * E7_LABEL_COUNTS[c] / total / E7_KEPT[c]
        assert round(per_row, 5) == N.EXPOSURE[c]


def test_a2_is_exactly_four_times_a1_and_shares_are_e7s():
    kept = {"benign": 180_003, "phishing": 51_257}
    a1, a2 = N.schedule(kept, "A1"), N.schedule(kept, "A2")
    assert a1["steps"] == N.a1_steps(kept) == round(
        (1.65888 * 180_003 + 3.76842 * 51_257) / 256)
    assert a2["steps"] == 4 * a1["steps"]
    for s in (a1, a2):
        assert s["warmup"] == round(0.025 * s["steps"])
        assert s["eval_every"] == round(0.05 * s["steps"])
    # the shares reproduce E7's own schedule at E7's length
    assert N.WARMUP_SHARE * 20_000 == 500 and N.EVAL_SHARE * 20_000 == 1_000


def test_generation_seeds_are_amendment_2s_blocks():
    assert N.gen_seed("benign", 0) == 20261001 + 1800 * 10**9
    assert N.gen_seed("phishing", 4) == 20261001 + 1904 * 10**9
    seeds = {N.gen_seed(label, g) for label in ("benign", "phishing") for g in range(N.G)}
    assert len(seeds) == 2 * N.G
    # distinct from every E8 seed (free and every control block)
    import e8_common as K
    e8 = {K.seed_free(g) for g in range(K.G)} | {
        K.seed_ctrl(label, st, g) for label in K.LABELS for st in K.STATES for g in range(K.G)}
    assert not seeds & e8


def test_the_sampler_delivers_the_registered_exposure_per_class():
    """With label_counts = class_weights(kept) and class_alpha 1.0, each class is
    drawn in proportion to exposure x rows, and rows cycle per epoch within a
    class, so each ROW gets EXPOSURE[c] draws per sum(weights) draws."""
    kept = {"benign": 3_000, "phishing": 400}
    tag = {"benign": 1, "phishing": 2}                    # the class, recoverable from ids
    rows = {c: [([i + 3, tag[c]], [1.0]) for i in range(k)] for c, k in kept.items()}
    ds = URLDataset(rows, label_counts=N.class_weights(kept))
    cfg = TrainConfig(batch_size=64, bucket_window=4)
    budget = sum(N.class_weights(kept).values())          # draws in one A1 budget
    draws = round(budget * 20)
    stream = batches(ds, cfg, pad_id=0, rng=random.Random(1))
    seen = {"benign": 0, "phishing": 0}
    got = 0
    while got < draws:
        ids, _ = next(stream)
        for r in ids.tolist():
            seen["benign" if r[1] == 1 else "phishing"] += 1
        got += ids.shape[0]
    total = sum(seen.values())
    for c in kept:
        p = N.class_weights(kept)[c] / budget
        sd = (total * p * (1 - p)) ** 0.5
        assert abs(seen[c] - total * p) < 4 * sd
        # draws per row, scaled back to one A1 budget
        assert seen[c] / kept[c] * budget / total == pytest.approx(N.EXPOSURE[c], rel=0.03)


# ---- the row filter -----------------------------------------------------------------

EXTRA = ["a.com/?q=1", "a.com?q=1", "a.com/?", "a.com/p", "a.com/p?x=y&z=2", "a.com",
         "b.org/?q", "c.net/#f", "d.io/p?#f"]


def e7_type_tok():
    return URLTokenizer(include_frame=True, include_fields=True).fit(FIT + EXTRA)


def test_query_bearing_is_exactly_the_tokenizers_qry():
    tok = e7_type_tok()
    for url in FIT + EXTRA:
        opens = "<QRY>" in [tok.itos[i] for i in tok.encode(url, "benign")]
        assert N.has_query(url) == opens, url


# ---- end to end on a tiny model -----------------------------------------------------

STREAM_ROWS = {
    "name_train": ["a.com/?q=1", "b.com", "c.org/x", "d.com/?id=7&k=2", "e.net/y?z=1"],
    "structure_train": ["a.com/p?q=2", "a.com/p", "a.com/r?s=t"],
    "phishing_name_train": ["f.com/?login=1", "g.io/a", "h.com/b?c=d"],
    "phishing_structure_train": ["f.com/c?d=e", "f.com/c"],
    "name_val": ["i.com/?q=3", "j.com"],
    "structure_val": ["i.com/p?q=4"],
    "phishing_name_val": ["k.com/?x=1", "l.io"],
    "phishing_structure_val": ["k.com/z?w=2"],
}


def _bundle(tmp: Path) -> Path:
    data = tmp / "data"
    data.mkdir()
    for stem, urls in STREAM_ROWS.items():
        label = "phishing" if stem.startswith("phishing") else "benign"
        with gzip.open(data / f"{stem}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for u in urls:
                fh.write(json.dumps({"url": u, "label": label, "lang": "und"}) + "\n")
    return data


def _e7_like(tmp: Path):
    tok = URLTokenizer(include_frame=True, include_fields=True).fit(
        FIT + [u for us in STREAM_ROWS.values() for u in us])
    torch.manual_seed(0)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, max_len=64, d_model=16,
                                       n_layers=1, n_heads=2, d_ff=32))
    cfg = TrainConfig(steps=20_000, batch_size=4, lr=3e-4, eval_every=1_000, eval_batches=1,
                      class_alpha=1.0, field_dropout=0.5)
    path = tmp / "e7_like.pt"
    save_checkpoint(model, tok, cfg, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), tok, cfg


def test_a_checkpoint_that_is_not_e7s_is_refused(tmp_path):
    path, _, _, _ = _e7_like(tmp_path)
    with pytest.raises(SystemExit):
        N.load_e7(path)                                   # E7's real hash expected


def test_counts_other_than_the_registered_ones_stop_the_run(tmp_path):
    data = _bundle(tmp_path)
    e7, sha, _, _ = _e7_like(tmp_path)
    table = tmp_path / "table.json"
    SuffixTable({"benign": {"com": 1}, "phishing": {"com": 1}}).save(table)
    with pytest.raises(SystemExit, match="kept query rows"):
        N.run(data, e7, "A1", tmp_path / "A1", table, device=torch.device("cpu"),
              n_gen=2, g=1, steps=1, expected_sha256=sha)       # default: the registered counts
    assert not (tmp_path / "A1" / "final.pt").exists()


def test_a_presence_plan_tokenizer_is_refused(tmp_path):
    tok = URLTokenizer(include_frame=True, include_fields=True,
                       presence_plan=True).fit(FIT)
    model = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, max_len=64, d_model=16,
                                       n_layers=1, n_heads=2, d_ff=32))
    path = tmp_path / "plan.pt"
    save_checkpoint(model, tok, TrainConfig(), path)
    with pytest.raises(SystemExit):
        N.load_e7(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_encode_keeps_query_rows_only_and_hides_validation(tmp_path):
    data = _bundle(tmp_path)
    tok = URLTokenizer(include_frame=True, include_fields=True).fit(
        FIT + [u for us in STREAM_ROWS.values() for u in us])
    tr, va, kept, rows = N.encode(data, tok, 64)
    want_tr = {c: sum(1 for s, us in STREAM_ROWS.items() if s.endswith("_train")
                      and (s.startswith("phishing")) == (c == "phishing")
                      for u in us if N.has_query(u)) for c in ("benign", "phishing")}
    assert kept == want_tr == rows["train_query_rows"]
    for enc in (tr, va):
        for c in enc:
            for ids, _ in enc[c]:
                assert "<QRY>" in [tok.itos[i] for i in ids]
    for c in va:
        for ids, _ in va[c]:
            names = [tok.itos[i] for i in ids]
            assert "<P:?>" in names and "<Q:?>" in names
    for c in tr:
        for ids, _ in tr[c]:
            names = [tok.itos[i] for i in ids]
            assert "<Q:1>" in names and "<Q:?>" not in names


def test_run_end_to_end_keeps_e7s_recipe_and_generates_from_final(tmp_path):
    data = _bundle(tmp_path)
    e7, sha, tok, e7_cfg = _e7_like(tmp_path)
    table = tmp_path / "table.json"
    SuffixTable({"benign": {"com": 3, "org": 1}, "phishing": {"com": 1, "io": 2}}).save(table)
    out = tmp_path / "A1"
    rec = N.run(data, e7, "A1", out, table, device=torch.device("cpu"), n_gen=6, g=2,
                steps=3, expected_sha256=sha, expect_kept=None)

    # every training setting is E7's except the three registered ones
    e7_train = asdict(e7_cfg)
    for k, v in rec["train_config"].items():
        if k in N.CHANGED:
            assert rec["changed_from_e7"][k] == [e7_train[k], v]
        else:
            assert v == e7_train[k], k
    assert rec["train_config"]["field_dropout"] == 0.5
    assert rec["schedule"] == {"steps": 3, "warmup": 1, "eval_every": 1}

    # E7's tokenizer, unchanged, in both checkpoints
    for name in ("best.pt", "final.pt"):
        _, t, _ = load_checkpoint(out / name)
        assert t.state_dict() == tok.state_dict()

    # generation: from final.pt, resting decode, Amendment 2's seeds
    final, ftok, _ = load_checkpoint(out / "final.pt")
    final.eval()
    shipped = SuffixTable.load(table)
    import e8_common as K
    for label in ("benign", "phishing"):
        for g in range(2):
            name = f"gen_A1_{label}_g{g}.json"
            got = json.loads((out / name).read_text(encoding="utf-8"))
            want = generate_with_suffixes(final, ftok, label=label, n=6, table=shipped,
                                          seed=N.gen_seed(label, g), **K.resting(ftok, label))
            assert got == want
            meta = rec["generation"]["files"][name]
            assert meta["seed"] == N.gen_seed(label, g) and meta["n"] == 6
            assert meta["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    saved = json.loads((out / "N2A_RECORD.json").read_text(encoding="utf-8"))
    assert saved["final_sha256"] == hashlib.sha256((out / "final.pt").read_bytes()).hexdigest()
    assert saved["finite"] is True and saved["status"] == "DIAGNOSTIC"
