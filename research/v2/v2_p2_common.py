"""
research/v2/v2_p2_common.py

V2-P2 (research/v2/PREREG_V2_P2.md) -- the constants and calls shared by P2's Kaggle
notebooks and the laptop tools.

    p2_a   e16_cap's network, with the path, query and fragment written in rest pieces.
    p2_b   the same, with names and subdomains written in host pieces too.

Both arms:
- `e16_cap`'s recipe, data, batches and start (plus the piece rows), with C re-derived;
- the name scope and BPE-dropout 0.1 in training;
- decoded as `e16_b5` (raw + D1 + D1' + B5), plus the piece rule.

Their 50 files are E12's jobs at E12's seeds, paired with `e16_b5`'s.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, HERE, ROOT / "research" / "native", ROOT / "research" / "e13",
           ROOT / "research" / "e15", ROOT / "research" / "e16", ROOT / "research" / "e17"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from v2_p1_common import (ACCUM_FALLBACK, ACCUM_STEPS, COUNTS, DROPPED, E15, E16, E16_CAP_FINAL_VAL,  # noqa: E402,F401
                          E16_CAP_RECORD, E16_CAP_SHA256, E16_START, E17, INJECTION, LOSS_CONSTANT,
                          STREAM_FILE, STREAM_ROWS, STREAM_SHA256, V_BATCH_DIGEST, decode_kwargs,
                          e16_cap_record)
from e8_common import N_FREE  # noqa: E402
from v2_p2_tokenizer import rules_sha256, table_sha256  # noqa: E402,F401

REGISTRATION = "research/v2/PREREG_V2_P2.md"
#: V-iso (section 12): uie/ byte-identical to 422cb23, its state since V2-P1b
ISO_COMMIT = "422cb230658c1149aff7c10d51f0fa0f9d795ef0"
E16_CAP_PARAMS = E16.PARAMS                     # 25,473,024
BASELINE = E17.DECODE_ARM                       # e16_b5
ARMS: Dict[str, Tuple[str, ...]] = {"p2_a": ("rest",), "p2_b": ("rest", "host")}
DROPOUT = 0.1
NAME_SCOPE = True
CODE_MANIFEST = "V2P2_CODE_MANIFEST.json"
TABLE_FILES = {"rest": "p2_rest.json", "host": "p2_host.json"}
#: Amendment 3 (gate B): the tables' sha256 and each arm's C, pinned after the build
TABLE_SHA256: Dict[str, Optional[str]] = {
    "rest": "a48312b8f097e259e35b999d37e8ba58d04a287bf8663cb42410f25e13dd1ce8",
    "host": "a4ed9ca41e3e8dded2846ead68f57c08e4699cebb1b3dcbc2e77ddec80b8b8af"}
LOSS_CONSTANT_P2: Dict[str, Optional[float]] = {"p2_a": 6208.307292719264, "p2_b": 5067.853260707934}


# --------------------------------------------------------------------------- #
# the tables and the recipe
# --------------------------------------------------------------------------- #

def load_tables(arm: str, directory: Path, check: bool = True) -> Dict[str, Optional[dict]]:
    """The arm's tables from `directory` (the bundle's names, or the build's), each checked
    against its pinned hash."""
    out: Dict[str, Optional[dict]] = {"rest": None, "host": None}
    for name in ARMS[arm]:
        path = Path(directory) / TABLE_FILES[name]
        if not path.is_file():
            path = Path(directory) / f"{name}.json"            # the build's own name
        state = json.loads(path.read_text(encoding="utf-8"))
        state = {"alphabet": state["alphabet"], "merges": [list(m) for m in state["merges"]]}
        if check and TABLE_SHA256[name] is not None and table_sha256(state) != TABLE_SHA256[name]:
            raise ValueError(f"the {name} table is not the registered one")
        out[name] = state
    return out


def recipe(stream: Path, arm: str, accum_steps: int = ACCUM_STEPS) -> dict:
    """`uie.kaggle_train.run`'s arguments: e16_cap's recipe, with the arm's C (section 5)."""
    r = E16.recipe_cap(stream, accum_steps)
    c = LOSS_CONSTANT_P2[arm]
    if c is None:
        raise ValueError(f"{arm}'s C is not pinned yet (Amendment 3)")
    return {**r, "loss_constant": float(c)}


def train_settings(arm: str, directory: Path) -> dict:
    """`v2_p2_train.run`'s P2 arguments for the arm."""
    t = load_tables(arm, directory)
    return {"rest": t["rest"], "host": t["host"], "dropout": DROPOUT, "scope": NAME_SCOPE}


def tokenizer(base_state: dict, arm: str, directory: Path):
    from v2_p2_tokenizer import HybridTokenizer
    t = load_tables(arm, directory)
    return HybridTokenizer(base_state=base_state, rest=t["rest"], host=t["host"], dropout=DROPOUT)


def params(tok) -> int:
    """e16_cap's parameters plus one embedding row (d_model wide, tied to the head) per piece."""
    return E16_CAP_PARAMS + (tok.vocab_size - tok.n_base) * E16.D_MODEL


def peel(model, n_base: int):
    """The model with its piece rows removed: a `URLTransformer` over the first `n_base`
    vocabulary rows and every other weight unchanged (section 5: it must be e16_cap's start)."""
    from dataclasses import replace
    from uie.model import URLTransformer
    base = URLTransformer(replace(model.cfg, vocab_size=n_base))
    state = {k: (v[:n_base].clone() if k in ("tok_emb.weight", "head.weight") else v.clone())
             for k, v in model.state_dict().items()}
    base.load_state_dict(state)
    return base


def start(tok) -> Tuple[object, dict]:
    """Section 3's start, as training builds it: E12's procedure at width 512 with e16_cap's
    tokenizer (e16_cap's start), then the piece rows. Returns the model and its fingerprints."""
    from uie.free import weights_sha256
    from uie.kaggle_train import new_model_e12
    from uie.model import ModelConfig
    from v2_p2_train import with_pieces
    r = E16.recipe_cap(Path(STREAM_FILE))
    base = tok.base
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=r["d_model"], n_layers=r["n_layers"],
                      n_heads=r["n_heads"], d_ff=4 * r["d_model"], struct_ids=(),
                      max_len=r["max_len"], frame_every=r.get("frame_every", False))
    from dataclasses import replace
    base_model, prints = new_model_e12(replace(cfg, vocab_size=base.vocab_size), base, E16.SEED)
    model, piece_prints = with_pieces(base_model, cfg, tok)
    return model, {**prints, **piece_prints, "init_weights_sha256": weights_sha256(model),
                   "params": sum(p.numel() for p in model.parameters())}


#: the worst-case check's rows per batch: 256 plus a margin for the appended stream rows
#: (about 4.7 per batch at rate 1), as V2-P1's check
PREFLIGHT_ROWS = 288


def memory_preflight(device: torch.device, rows: int, vocab_size: int) -> float:
    """Section 5's worst-case check, BEFORE any training: one micro-batch of `rows` rows at
    max_len (384), every target scored, forward and backward under training's autocast with
    the piece model's chunked head, then a clip and an AdamW step (whose state the piece
    rows enlarge). Returns the peak memory in GB (0.0 on a CPU); raises if it does not fit."""
    from uie.model import ModelConfig
    from v2_p2_train import PieceTransformer
    r = E16.recipe_cap(Path(STREAM_FILE))
    cfg = ModelConfig(vocab_size=vocab_size, d_model=r["d_model"], n_layers=r["n_layers"],
                      n_heads=r["n_heads"], d_ff=4 * r["d_model"], struct_ids=(),
                      max_len=r["max_len"], frame_every=r.get("frame_every", False))
    cuda = device.type == "cuda"
    model = PieceTransformer(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-12, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    x = torch.randint(4, vocab_size, (rows, cfg.max_len), device=device)
    w = torch.ones((rows, cfg.max_len - 1), device=device)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    with torch.amp.autocast("cuda", enabled=cuda):
        loss = model.loss(x, 0, w)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if cuda else 0.0
    del model, opt, x, w, loss
    if cuda:
        torch.cuda.empty_cache()
    return peak


# --------------------------------------------------------------------------- #
# generation: the piece rule (section 4.6) and e16_b5's calls
# --------------------------------------------------------------------------- #

class PieceRule(torch.nn.Module):
    """At generation, a rest piece only inside PTH, QRY and FRG, and a host piece only inside
    LBL and SUB (section 4.6). Each row's field is the one its last field marker opened (none
    after <RS> or before any marker). The probability mass removed is recorded per field
    (under the logits as given, before any decode rule).

    For a piece model it returns the last position's logits only, shape (rows, 1, vocab):
    generation reads nothing else, and the head over every position would cost rows x
    positions x vocab floats at each step. For a character model it is the model, exactly.

    V-rules: every token the decoder sampled is checked against the field it was written in
    when it comes back as input; `violations` counts pieces outside their fields. (A row cut
    at max_len never feeds back its last token; that token's field is the same step's mask.)"""

    def __init__(self, model, tok) -> None:
        super().__init__()
        from uie.tokenizer import _MARKERS
        self.model = model
        self.cfg = model.cfg
        self.tok = tok
        names = list(_MARKERS) + ["<none>"]
        self.names = names
        v = tok.vocab_size
        ids = [tok.marker_id(m) for m in _MARKERS]
        rs = tok.stoi.get("<RS>")
        self.n_fields = len(names)
        lookup = torch.full((v,), -1, dtype=torch.long)
        for k, i in enumerate(ids):
            lookup[i] = k
        if rs is not None:
            lookup[rs] = len(_MARKERS)               # after <RS>: no field
        self.register_buffer("lookup", lookup, persistent=False)
        forbid = torch.zeros((self.n_fields + 1, v), dtype=torch.bool)
        pieces = getattr(tok, "piece_table", {})
        allowed = tok.field_tables() if hasattr(tok, "field_tables") else {}
        for k, name in enumerate(names + ["<start>"]):
            for pid, table in pieces.items():
                if allowed.get(name) != table:
                    forbid[k, pid] = True
        self.register_buffer("forbid", forbid, persistent=False)
        self.active = bool(pieces)
        self.removed = torch.zeros(self.n_fields + 1, dtype=torch.float64)
        self.steps = torch.zeros(self.n_fields + 1, dtype=torch.long)
        self.checked = 0
        self.violations = 0

    def field_of(self, ids: torch.Tensor) -> torch.Tensor:
        """Each row's current field index: its last field marker's (the <none> slot after
        <RS>), or the <start> slot before any marker."""
        hit = self.lookup.to(ids.device)[ids]                       # (B, T): field index or -1
        pos = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        last = torch.where(hit >= 0, pos, torch.full_like(pos, -1)).max(dim=1).values
        return torch.where(last >= 0, hit.gather(1, last.clamp(min=0).unsqueeze(1)).squeeze(1),
                           torch.full_like(last, self.n_fields))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return self.model(ids)
        from v2_p2_train import hidden_states
        forbid = self.forbid.to(ids.device)
        if ids.shape[1] > 1:                                        # the token sampled last step
            wrote = forbid[self.field_of(ids[:, :-1]), ids[:, -1]]
            self.violations += int(wrote.sum())
            self.checked += int(ids.shape[0])
        logits = self.model.head(hidden_states(self.model, ids)[:, -1:])   # (B, 1, V)
        field = self.field_of(ids)
        bad = forbid[field]                                         # (B, V)
        z = logits[:, -1].float()
        mass = (torch.softmax(z, dim=-1) * bad).sum(dim=-1).double().cpu()
        self.removed.index_add_(0, field.cpu(), mass)
        self.steps.index_add_(0, field.cpu(), torch.ones_like(field.cpu()))
        return logits.masked_fill(bad.unsqueeze(1), float("-inf"))

    def snapshot(self) -> tuple:
        return self.removed.clone(), self.steps.clone(), self.checked, self.violations

    def record(self, since: Optional[tuple] = None) -> Dict[str, object]:
        """The totals, or (with a `snapshot()`) what happened since it: one file's share."""
        removed, steps, checked, violations = self.snapshot()
        if since is not None:
            removed, steps = removed - since[0], steps - since[1]
            checked, violations = checked - since[2], violations - since[3]
        names = self.names + ["<start>"]
        return {"fields": {names[k]: {"steps": int(steps[k]), "mass_removed": float(removed[k])}
                           for k in range(self.n_fields + 1) if int(steps[k])},
                "tokens_checked": checked, "violations": violations}


def load_checkpoint(path: Path, *, device: Optional[torch.device] = None):
    """A P2 checkpoint: the model (on `device`), its piece tokenizer and its config."""
    from uie.model import ModelConfig, URLTransformer
    from v2_p2_tokenizer import HybridTokenizer
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    model = URLTransformer(cfg)
    model.load_state_dict(blob["model"])
    tok = HybridTokenizer.from_state_dict(blob["tokenizer"])
    if device is not None:
        model.to(device)
    return model, tok, cfg


def row_plans(tables, n: int, seed: int) -> List[str]:
    """Each generated row's supplied state, in `generate`'s order: `e13_common.draw_and_generate`
    draws the states' counts from the seed and generates them state by state, so its `ids_out`
    holds each state's rows as one block, in the tables' order."""
    import collections
    import random
    states = list(tables)
    rng = random.Random(seed)
    counts = collections.Counter(rng.choices(states, weights=[sum(tables[k].values()) for k in states], k=n))
    return [k for k in states for _ in range(counts[k])]


def token_rules(tok, rows: List[List[int]], plans: Optional[List[str]] = None) -> Dict[str, int]:
    """Section 12's V-rules, exactly, on the generated rows' tokens (`generate`'s `ids_out`: each
    row with its end token if it ended), each row against its supplied plan (`plans`, states
    such as "P0Q1S1"; without them, the plan tokens a row carries before <LBL>). A row's own
    plan tokens, where it has them, must agree with the supplied plan (`plan_mismatch`):
    - D1 and D1': the first host-end token after <LBL> (<PTH>, <RS>, <QRY>, <FRG>, <EOS>) is one
      the plan allows -- S = 1: <PTH> or <RS>; S = 0: <QRY>, <FRG> or <EOS> (P1 plans carry
      S = 1; with D1' every plan is covered);
    - B5: on a Q = 1 plan, the first of <QRY>, <FRG>, <EOS> after <LBL> is <QRY>;
    - a row cut at max_len before the decision is counted apart; so is a row that never reached
      <LBL> (cut, or ended, before the host): both rules act only from <LBL>.
    The URL strings cannot show this exactly: a delimiter written as a character inside a host
    field parses as a host end (e16_b5's own files: 5 of 180,000 URLs)."""
    lbl, pth, qry, frg = (tok.marker_id(m) for m in ("LBL", "PTH", "QRY", "FRG"))
    rs, eos = tok.rs_id, tok.eos_id
    slots = dict(zip(("P", "Q", "S"), tok.plan_slots()))
    ends = (pth, rs, qry, frg, eos)
    out = dict.fromkeys(("rows", "cut_before_host", "ended_before_host", "no_plan", "plan_mismatch",
                         "d1_checked", "d1_violations", "d1_cut", "b5_checked", "b5_violations", "b5_cut",
                         "b5_closed_before_host"), 0)
    if plans is not None and len(plans) != len(rows):
        raise ValueError(f"{len(rows)} rows, {len(plans)} plans")
    for n, r in enumerate(rows):
        out["rows"] += 1
        if lbl not in r:
            out["ended_before_host" if r and r[-1] == eos else "cut_before_host"] += 1
            continue
        i = r.index(lbl)
        # the plan a row carries: the slot tokens right before <LBL> (the supplied plan is
        # placed there, after the suffix); anything else is not a plan
        tail = r[max(0, i - len(slots)):i]
        own = ({f: (1 if t == b else 0) for (f, (a, b)), t in zip(slots.items(), tail)}
               if len(tail) == len(slots) and all(t in pair for pair, t in zip(slots.values(), tail))
               else dict.fromkeys(slots))
        if plans is not None:
            k = plans[n]
            plan = {"P": int(k[1]), "Q": int(k[3]), "S": int(k[5])}
            out["plan_mismatch"] += any(v is not None and v != plan[f] for f, v in own.items())
        else:
            plan = own
        if None in plan.values():
            out["no_plan"] += 1
            continue
        end = next((t for t in r[i + 1:] if t in ends), None)
        if end is None:
            out["d1_cut"] += 1
        else:
            out["d1_checked"] += 1
            out["d1_violations"] += end not in ((pth, rs) if plan["S"] else (qry, frg, eos))
        if plan["Q"]:
            if qry in r[:i]:                # generate's B5 closes at any <QRY>, even before <LBL>
                out["b5_closed_before_host"] += 1
                continue
            first = next((t for t in r[i + 1:] if t in (qry, frg, eos)), None)
            if first is None:
                out["b5_cut"] += 1
            else:
                out["b5_checked"] += 1
                out["b5_violations"] += first != qry
    return out


def _draw(rule, tok, label: str, n: int, seed: int, tables, device, plan_trace, query_trace,
          record: Optional[dict]) -> List[str]:
    """`e17_common.free_urls`'s and `control_urls`'s body with B5 on: the same tables, seeds and
    decode. With `record`, `uie.model.generate`'s `ids_out` and `slash_trace` (which only
    collect) are asked for, and `record` receives the URLs cut at max_len before their end
    token, the token-level rules (`token_rules`) and the mass D1 moved (section 6)."""
    import e13_common as E13
    kw = decode_kwargs()                          # E17.decode_kwargs(True): raw + D1 + D1' + B5
    if query_trace is not None:
        kw["query_trace"] = query_trace
    rows: Optional[list] = [] if record is not None else None
    strace: Optional[list] = [] if record is not None else None
    if rows is not None:
        kw["ids_out"] = rows
        kw["slash_trace"] = strace
    urls = E13.draw_and_generate(rule, tok, label, n, seed, tables, lambda k: kw, device=device,
                                 plan_trace=plan_trace)
    if record is not None:
        record["cut_at_max_len"] = [tok.decode(r) for r in rows if not r or r[-1] != tok.eos_id]
        record["token_rules"] = token_rules(tok, rows, row_plans(tables, n, seed))
        record["slash_moved_mean"] = sum(strace) / len(strace) if strace else None
        record["slash_rows_moved"] = sum(x > 0 for x in strace)
    return urls


def free_urls(model, tok, label: str, g: int, shipped, presence_slash, *, device=None,
              n: int = N_FREE, plan_trace: Optional[list] = None, query_trace: Optional[list] = None,
              rule: Optional[PieceRule] = None, record: Optional[dict] = None) -> List[str]:
    """FULL FREE at E12's seeds with e16_b5's decode and the piece rule (E17's call, B5 on)."""
    import e12_common as E12
    from e8_common import seed_free
    rule = rule or PieceRule(model, tok)
    tables = E12.supplied_tables(presence_slash, shipped.counts, label)
    return _draw(rule, tok, label, n, seed_free(g), tables, device, plan_trace, query_trace, record)


def control_urls(model, tok, label: str, state: str, g: int, state_tables, presence_slash, *,
                 device=None, n: Optional[int] = None, plan_trace: Optional[list] = None,
                 query_trace: Optional[list] = None, rule: Optional[PieceRule] = None,
                 record: Optional[dict] = None) -> List[str]:
    """FULL CONTROL at E8's seeds with e16_b5's decode and the piece rule (E17's call, B5 on)."""
    import e12_common as E12
    from e8_common import n_ctrl, seed_ctrl
    rule = rule or PieceRule(model, tok)
    n = n if n is not None else n_ctrl(label, state)
    tables = E12.control_tables(presence_slash, state_tables, label, state)
    return _draw(rule, tok, label, n, seed_ctrl(label, state, g), tables, device, plan_trace,
                 query_trace, record)


def jobs(arm: str) -> List[dict]:
    """E12's 50 jobs for the arm, at E12's seeds."""
    return E17.jobs(arm)


class _ArmView:
    """This module seen as one arm, for the V2 scripts that read `P1.ARM` and `P1.jobs()`
    (v2_arm.py): every name is this module's, with `ARM` the arm's."""

    def __init__(self, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(arm)
        self.ARM = arm

    def jobs(self, arm: Optional[str] = None) -> List[dict]:
        return jobs(arm or self.ARM)

    def __getattr__(self, name: str):
        return globals()[name] if name in globals() else getattr(sys.modules[__name__], name)


def arm_view(arm: str) -> _ArmView:
    return _ArmView(arm)
