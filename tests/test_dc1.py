"""DC1 -- calibrated per-field decoding for E7 (research/dc1/PREREG_DC1.md).

Pins: every DC1 option off, empty or neutral leaves generation bit-identical
(the 96-case E7 fixture); DC1's own tracker enters a supplied label while the
legacy one behind the frozen path delta is untouched; settings act only inside
their field; the blend moves content characters only; and the fit scores
exactly what the sampler samples -- recomputed log-probabilities match the
sampler's, token for token, with every option on.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "dc1"))

import dc1_common as K  # noqa: E402
from dc1_fit import Layout, fit_field, sequence_logp  # noqa: E402
from test_e7_fields import e7_tok, tiny_model  # noqa: E402
from test_hc1_enforce import FIXTURE as E7_FIXTURE  # noqa: E402
from test_hc1_enforce import stub  # noqa: E402
from uie.model import blend_content, generate  # noqa: E402
from uie.tokenizer import URLTokenizer  # noqa: E402

SETTINGS = {"LBL": {"T": 1.3, "delta": 0.4, "w": 0.5},
            "SUB": {"T": 0.8, "delta": -0.3, "w": 1.5},
            "PTH": {"T": 1.2, "delta": -0.5, "w": 0.2},
            "QRY": {"T": 1.4, "delta": 0.7, "w": 0.0}}


def dc1_kwargs(settings, blend_ok):
    out = {"field_temperature": {f: s["T"] for f, s in settings.items()},
           "field_stop_bias": {f: s["delta"] for f, s in settings.items()}}
    if blend_ok:
        out["blend"] = {f: s["w"] for f, s in settings.items()}
    return out


# ---- 1. off, empty or neutral means off ---------------------------------------------

def test_neutral_dc1_options_are_bit_identical_to_the_fixture():
    tok = e7_tok()
    model = tiny_model(tok)
    neutral = [dict(field_temperature={}, field_stop_bias={}, blend={}),
               dict(field_temperature={f: 1.0 for f in K.FIELDS_FIT},
                    field_stop_bias={f: 0.0 for f in K.FIELDS_FIT},
                    blend={f: 1.0 for f in K.FIELDS_FIT}),
               dict(logp=[], ids_out=[])]          # observing must not change the draw
    for case in E7_FIXTURE["cases"]:
        fb = {"PTH": {tok.eos_id: -0.3}} if case["field_bias"] else None
        kw = dict(label=case["label"], n=len(case["urls"]), seed=case["seed"],
                  fields=case["fields"], decoder=case["decoder"], field_bias=fb,
                  suffix=case["suffix"])
        for extra in neutral:
            assert generate(model, tok, **kw, **extra) == case["urls"], (case, extra)


def test_blend_is_a_no_op_when_the_controls_are_already_free():
    tok = e7_tok()
    model = tiny_model(tok)
    base = generate(model, tok, label="benign", n=16, seed=9, suffix="com")
    assert generate(model, tok, label="benign", n=16, seed=9, suffix="com",
                    blend={f: 0.3 for f in K.FIELDS_FIT}) == base


# ---- 2. scope ---------------------------------------------------------------------

def test_label_settings_reach_a_supplied_label():
    tok = e7_tok()
    model = tiny_model(tok)
    ids = []
    generate(model, tok, label="benign", n=12, seed=3, suffix="com",
             field_stop_bias={"LBL": -40.0}, ids_out=ids)
    lbl = tok.marker_id("LBL")
    markers = set(tok.struct_ids()) - {tok.eos_id}
    reached = 0
    for seq in ids:
        if lbl not in seq:
            continue                                # never asked for a suffix
        reached += 1
        after = seq[seq.index(lbl) + 1:]
        assert not (set(after) & markers) and tok.eos_id not in after, seq
        assert len(seq) == 64                       # the label ran to max_len
    assert reached >= 6


def test_settings_for_a_field_never_entered_change_nothing():
    tok = e7_tok()
    model = stub(tok, once=("PTH", "QRY", "FRG"), EOS=1.0, a=2.0, PTH=1.5)
    base = generate(model, tok, label="benign", n=24, seed=5, suffix="com")
    assert generate(model, tok, label="benign", n=24, seed=5, suffix="com",
                    field_temperature={"QRY": 0.3}, field_stop_bias={"QRY": 2.0}) == base
    assert generate(model, tok, label="benign", n=24, seed=5, suffix="com",
                    field_temperature={"PTH": 0.3}, field_stop_bias={"PTH": 2.0}) != base


def test_blend_moves_content_characters_only():
    z_c, z_u = torch.randn(5, 10), torch.randn(5, 10)
    content = torch.tensor([1, 3, 4])
    out = blend_content(z_c, z_u, 0.25, content)
    other = [0, 2, 5, 6, 7, 8, 9]
    assert torch.equal(out[:, other], z_c[:, other])
    assert torch.allclose(out[:, content],
                          z_u[:, content] + 0.25 * (z_c[:, content] - z_u[:, content]))


def test_bad_settings_are_refused():
    tok = e7_tok()
    model = tiny_model(tok)
    for bad in (dict(field_temperature={"QRY": 0.0}), dict(field_temperature={"QRY": -1}),
                dict(field_stop_bias={"QRY": float("nan")}), dict(blend={"QRY": float("inf")})):
        with pytest.raises(ValueError):
            generate(model, tok, label="benign", n=1, seed=1, **bad)
    old = URLTokenizer(include_frame=True).fit(["https://a.example.com/x"])
    with pytest.raises(ValueError):
        generate(tiny_model(old), old, label="benign", n=1, seed=1, blend={"QRY": 0.5})


# ---- 3. the fit scores exactly what the sampler samples -----------------------------

@pytest.mark.parametrize("fields", [None, {"PTH": True, "QRY": True},
                                    {"PTH": False, "QRY": True}])
def test_the_fit_scores_exactly_what_the_sampler_samples(fields):
    tok = e7_tok()
    model = tiny_model(tok)
    layout = Layout(tok)
    blend_ok = fields is not None
    logp, ids = [], []
    generate(model, tok, label="benign", n=12, seed=11, suffix="com", fields=fields,
             decoder="temperature", logp=logp, ids_out=ids,
             field_bias={"PTH": {tok.eos_id: -K.FROZEN_DELTA["benign"]}},
             **dc1_kwargs(SETTINGS, blend_ok))
    assert len(logp) == len(ids) == 12
    # matched settings agree to ~2e-6; a wrong blend weight (0.5 -> 0.6) moves the
    # tiny model's sum by ~7e-4, so the tolerance must sit between the two
    for lp, seq in zip(logp, ids):
        assert sequence_logp(model, layout, seq, "benign", SETTINGS, blend_ok) == \
            pytest.approx(lp, abs=1e-4), seq


def test_steps_follow_both_trackers_on_a_real_url():
    tok = e7_tok()
    layout = Layout(tok)
    ids = tok.encode("https://www.example.com/a/b?q=1&r=2", "benign", frame="domain")
    steps = layout.steps(ids)
    suf, lbl = tok.marker_id("SUF"), tok.marker_id("LBL")
    j_suf = ids.index(suf)
    forced = set(range(j_suf + 1, ids.index(lbl, j_suf) + 1))
    assert all(t + 1 not in forced for t, _, _, _ in steps)       # suffix and <LBL> unscored
    assert any(f == "LBL" for _, f, _, _ in steps)
    assert all(leg != "LBL" for _, f, leg, _ in steps)              # legacy never enters it
    assert all(leg == "PTH" for _, f, leg, _ in steps if f == "PTH")
    assert all(leg == "QRY" for _, f, leg, _ in steps if f == "QRY")


# ---- 4. the fit recovers a known miscalibration -------------------------------------

def _synthetic(layout, n, seed, T=1.0, d=0.0, w=None):
    g = torch.Generator().manual_seed(seed)
    zc = torch.randn(n, layout.V, generator=g) * 2
    zu = torch.randn(n, layout.V, generator=g) * 2 if w is not None else None
    z = zc + d * layout.stop_mask
    if w is not None:
        z = z + (w - 1.0) * (zc - zu) * layout.content_mask
    y = torch.multinomial(torch.softmax(z / T, dim=-1), 1, generator=g).squeeze(1)
    return {"zc": zc, "zu": zu, "y": y, "url": torch.arange(n),
            "lpth": torch.zeros(n, dtype=torch.bool)}


def test_fit_recovers_a_known_temperature_and_stop_bias():
    layout = Layout(e7_tok())
    r = fit_field(_synthetic(layout, 20000, 1, T=1.3, d=0.6), layout, 0.0, False)
    assert r["T"] == pytest.approx(1.3, abs=0.05)
    assert r["delta"] == pytest.approx(0.6, abs=0.15)
    assert r["ce_fit"] < r["ce_default"]


def test_fit_recovers_a_known_blend_and_leaves_calibrated_alone():
    layout = Layout(e7_tok())
    r = fit_field(_synthetic(layout, 20000, 2, w=0.4), layout, 0.0, True)
    assert r["w"] == pytest.approx(0.4, abs=0.08)
    assert r["T"] == pytest.approx(1.0, abs=0.05) and abs(r["delta"]) < 0.15
    calibrated = fit_field(_synthetic(layout, 20000, 3), layout, 0.0, False)
    assert calibrated["T"] == pytest.approx(1.0, abs=0.05) and abs(calibrated["delta"]) < 0.15


# ---- 5. settings -> generate() -------------------------------------------------------

def test_generation_kwargs_apply_only_confirmed_fields():
    s = {"classes": {"benign": {
        "free": {"LBL": {"T": 1.2, "delta": 0.1, "w": None, "applied": True},
                 "QRY": {"T": 1.5, "delta": 0.3, "w": None, "applied": False}},
        "P1Q1": {"PTH": {"T": 0.9, "delta": -0.2, "w": 0.6, "applied": True}}}}}
    assert K.generation_kwargs(s, "benign", "free") == {
        "field_temperature": {"LBL": 1.2}, "field_stop_bias": {"LBL": 0.1}}
    assert K.generation_kwargs(s, "benign", "P1Q1") == {
        "field_temperature": {"PTH": 0.9}, "field_stop_bias": {"PTH": -0.2},
        "blend": {"PTH": 0.6}}


# ---- 6. the laptop scripts: job list, placement, readings ----------------------------

def test_the_verifier_rebuilds_the_notebooks_job_list():
    from dc1_verify import expected_jobs
    on = {"applied": True, "T": 1.1, "delta": -0.1, "w": 0.5}
    s = {"classes": {lab: {m: {"QRY": dict(on)} for m in K.MODES} for lab in ("benign", "phishing")}}
    jobs = expected_jobs(s)
    assert len(jobs) == 56
    # 40 seeds: the two free arms share theirs (they pair), and the six cost
    # references reuse DC1's generation-0 seeds (identical draws, cost only)
    assert len({j["file"] for j in jobs}) == 56 and len({j["seed"] for j in jobs}) == 40
    by = {(j["mode"], j["arm"], j["label"], j["state"], j["g"]): j["seed"] for j in jobs}
    assert all(by[("free", "base", l, None, g)] == by[("free", "dc1", l, None, g)]
               for l in ("benign", "phishing") for g in range(K.G))
    assert all(by[("control", "base", l, s, 0)] == by[("control", "dc1", l, s, 0)]
               for l, sts in K.MEASURABLE.items() for s in sts)
    free_only = {"classes": {lab: {m: {"QRY": dict(on, applied=(m == "free"))} for m in K.MODES}
                             for lab in ("benign", "phishing")}}
    assert {j["mode"] for j in expected_jobs(free_only)} == {"free"}
    off = {"classes": {lab: {m: {"QRY": dict(on, applied=False)} for m in K.MODES}
                       for lab in ("benign", "phishing")}}
    assert expected_jobs(off) == []                                     # K1: nothing runs


def test_seed_blocks_are_disjoint_from_every_earlier_experiment():
    seeds = [K.seed_free(l, g) for l in ("benign", "phishing") for g in range(K.G)]
    seeds += [K.seed_ctrl(l, s, g) for l, sts in K.MEASURABLE.items() for s in sts for g in range(K.G)]
    for x in seeds:
        off = (x - 20261001) % 1_000_000_000
        assert off in (0, 150_000_000)
        s = ((x - 20261001) // 1_000_000_000) // 100
        assert (off == 0 and s in (16, 17)) or (off == 150_000_000 and 8 <= s <= 15)
    assert len(set(seeds)) == len(seeds)


def test_placement_matches_the_free_generator():
    from dc1_verify import _placement
    from uie.suffixes import SuffixTable, generate_with_suffixes
    tok = e7_tok()
    model = tiny_model(tok)
    table = SuffixTable({"benign": {"com": 5, "org": 3, "net": 2}})
    urls = generate_with_suffixes(model, tok, label="benign", n=10, table=table, seed=77)
    groups, where = _placement(table, "benign", 10, 77)
    start = 0
    for i, ((frame, suffix), count) in enumerate(groups):
        got = generate(model, tok, label="benign", n=count, suffix=suffix, seed=77 + i, frame=frame)
        assert [urls[where[start + k]] for k in range(count)] == got
        start += count


def test_readings_follow_the_registration():
    from dc1_compare import prop_worsening, read_mode
    t = [("benign", None, "PTH"), ("benign", None, "QRY")]
    cells = {t[0]: {"delta": -0.03, "p": 0.001, "v_base": "UNDECIDED", "v_dc1": "CLEARED"},
             t[1]: {"delta": -0.01, "p": 0.2, "v_base": "UNDECIDED", "v_dc1": "UNDECIDED"},
             ("benign", None, "LBL"): {"delta": +0.02, "p": 0.001, "v_base": "CLEARED", "v_dc1": "CLEARED"},
             ("benign", None, "SUF"): {"delta": +0.001, "p": 0.9, "v_base": "CLEARED", "v_dc1": "CONVICTED"}}
    m = read_mode(cells, t)
    assert m["improved"] == [t[0]] and m["fixed"] == [t[0]]
    assert {r["cell"] for r in m["regressions"]} == {str(("benign", None, "LBL")),
                                                     str(("benign", None, "SUF"))}
    same = prop_worsening(9000, 10000, 9000, 10000, False)
    assert same["worsening"] == 0 and same["ok"]                     # no change is never harm
    worse = prop_worsening(9000, 10000, 8800, 10000, False)          # novelty fell 2 points
    assert worse["worsening"] == pytest.approx(0.02) and not worse["ok"]
    small = prop_worsening(900000, 1000000, 897000, 1000000, False)  # 0.3 points, significant
    assert small["lo"] > 0 and small["ok"]                          # below the 0.5-point margin


NB_DC1 = ROOT / "research" / "dc1" / "DC1_generate.ipynb"


@pytest.mark.skipif(not NB_DC1.is_file(), reason="needs the notebook")
def test_the_notebook_is_its_generator_and_makes_the_registered_calls():
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("dc1_make_notebook",
                                                  ROOT / "research" / "dc1" / "make_notebook.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    nb = json.loads(NB_DC1.read_text(encoding="utf-8"))
    assert nb["cells"] == mk.cells()
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<cell>", "exec")
    for needle in ("generate_with_suffixes(counted", "generate_controlled(counted",
                   "generation_kwargs(settings, lab, st or \"free\") if job[\"arm\"] == \"dc1\" else {}",
                   '{"PTH": {tokenizer.eos_id: -FROZEN_DELTA[lab]}}', '{"temperature": 1.0}',
                   "seed_free(lab, g)", "seed_ctrl(lab, st, g)", mk.E7_SHA256):
        assert needle in src, needle
