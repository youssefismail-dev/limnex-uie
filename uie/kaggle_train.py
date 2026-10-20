"""
uie/kaggle_train.py

Training entry point for a Kaggle GPU session.

Runs unchanged locally (CPU, tiny `--steps`) and on Kaggle (T4/P100). It travels
inside the upload bundle, so the notebook needs no network access and no pip
install from an unpublished repository.

    python -m uie.kaggle_train --data /kaggle/input/<dataset> --out /kaggle/working

Everything it writes — checkpoint, metrics, samples — lands in `--out`, which on
Kaggle is the only directory that survives the session.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from uie.evaluation import boundary_nll, field_nll, shape_nll
from uie.model import ModelConfig, URLTransformer, generate
from uie.tokenizer import URLTokenizer
from uie.train import (TrainConfig, URLDataset, encode_corpus, evaluate,
                       save_checkpoint, train)

#: Languages given their own control token. The tail is enormous (159 observed)
#: and mostly unusable as a conditioning signal, so rare ones fold into "und" —
#: a token seen a hundred times teaches nothing and costs an embedding row.
TOP_LANGUAGES = 24


def resolve(directory: Path, stem: str) -> Path:
    """Find `<stem>.jsonl` or `<stem>.jsonl.gz`.

    Kaggle DECOMPRESSES gzip members while processing an uploaded dataset, so a
    bundle shipped as `train.jsonl.gz` arrives as `train.jsonl`. Assuming either
    name breaks on the other platform, so accept both.
    """
    for name in (f"{stem}.jsonl", f"{stem}.jsonl.gz"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"neither {stem}.jsonl nor {stem}.jsonl.gz in {directory}; "
        f"present: {sorted(p.name for p in directory.iterdir())}")


def load_rows(path: Path) -> Tuple[List[str], List[str], List[str]]:
    urls: List[str] = []
    labels: List[str] = []
    langs: List[str] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            urls.append(row["url"])
            labels.append(row["label"])
            langs.append(row.get("lang", "und"))
    return urls, labels, langs


#: The four training streams. `corpus` decides which FIELDS a row teaches:
#: name rows supply gradient for [SUF]/[LBL]/[SUB], structure rows for
#: [PTH]/[QRY]. The two populations are sampled differently on purpose and each
#: is wrong about the other's fields, so neither may teach both.
STREAMS = [
    ("name_train", "benign", "name"),
    ("structure_train", "benign", "structure"),
    ("phishing_name_train", "phishing", "name"),
    ("phishing_structure_train", "phishing", "structure"),
]
VAL_STREAMS = [
    ("name_val", "benign", "name"),
    ("structure_val", "benign", "structure"),
    ("phishing_name_val", "phishing", "name"),
    ("phishing_structure_val", "phishing", "structure"),
]


def load_streams(data: Path, streams) -> Tuple[List[str], List[str], List[str],
                                               List[str], Dict[str, int]]:
    """Load every stream, tagging each row with its class and its corpus.

    Also returns the LABEL COUNT per class — the number of distinct names, which
    is what `class_alpha` weights by. The name corpora hold exactly one URL per
    name by construction, so their row count IS the label count; structure rows
    are additional pages of names already counted and must not inflate it.
    """
    urls: List[str] = []
    labels: List[str] = []
    langs: List[str] = []
    corpora: List[str] = []
    label_counts: Dict[str, int] = {}
    for stem, label, corpus in streams:
        try:
            path = resolve(data, stem)
        except FileNotFoundError:
            print(f"  {stem:<28} MISSING — skipped", flush=True)
            continue
        u, _, g = load_rows(path)
        urls.extend(u)
        labels.extend([label] * len(u))
        langs.extend(g)
        corpora.extend([corpus] * len(u))
        if corpus == "name":
            label_counts[label] = label_counts.get(label, 0) + len(u)
        print(f"  {stem:<28}{len(u):>10,} rows  ({label}/{corpus})", flush=True)
    return urls, labels, langs, corpora, label_counts


def _safe_stdout() -> None:
    """Make stdout tolerate any character the model can produce.

    Raw Unicode is vanishingly rare in URLs (25 distinct non-ASCII characters,
    195 occurrences in ~120M) but it IS in the vocabulary, so generation can emit
    it. On a Windows console (cp1252) printing one raises UnicodeEncodeError and
    kills the run *after* training has finished — losing the samples and the
    result file to a display problem.
    """
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run(data: Path, out: Path = Path("/kaggle/working"), *,
        steps: int = 20_000, batch_size: int = 256, lr: float = 3e-4,
        d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
        eval_every: int = 1_000, class_alpha: float = 1.0,
        frame: bool = False, factorized: bool = False,
        boundary_weight: float = 1.0,
        fields: bool = False, field_dropout: float = 0.0,
        presence_plan: bool = False, root_slash: bool = False,
        loss_norm: str = "batch", loss_constant: Optional[float] = None,
        seed_init: bool = False, grad_clip_late: Optional[float] = None,
        field_dropout_joint: bool = False, shared_start: bool = False,
        slash_plan: bool = False, max_len: int = 256, frame_every: bool = False,
        e12_start: bool = False, context_prefixes: Optional[str] = None,
        context_p: float = 0.0, query_weight: float = 1.0,
        qry_extra: Optional[str] = None, accum_steps: int = 1) -> dict:
    """Train and write artifacts. Callable directly — no subprocess.

    A notebook must be able to call this IN-PROCESS. Shelling out to
    `!python -m uie.kaggle_train` spawns a fresh interpreter that does not
    inherit the notebook's `sys.path`, so the package it just added to the path
    is invisible to the child and the run dies with ModuleNotFoundError. Keeping
    the entry point importable removes that whole class of failure.
    """
    _safe_stdout()
    args = argparse.Namespace(
        data=Path(data), out=Path(out), steps=steps, batch_size=batch_size,
        lr=lr, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        eval_every=eval_every, class_alpha=class_alpha, frame=frame,
        factorized=factorized, boundary_weight=boundary_weight,
        fields=fields, field_dropout=field_dropout,
        presence_plan=presence_plan, root_slash=root_slash,
        loss_norm=loss_norm, loss_constant=loss_constant, seed_init=seed_init,
        grad_clip_late=grad_clip_late, field_dropout_joint=field_dropout_joint,
        shared_start=shared_start, slash_plan=slash_plan, max_len=max_len,
        frame_every=frame_every, e12_start=e12_start,
        context_prefixes=context_prefixes, context_p=context_p,
        query_weight=query_weight, qry_extra=qry_extra, accum_steps=accum_steps)
    return _run(args)


def main() -> None:
    _safe_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="directory holding train.jsonl[.gz] and val.jsonl[.gz]")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1_000)
    ap.add_argument("--class-alpha", type=float, default=1.0)
    ap.add_argument("--boundary-weight", type=float, default=1.0,
                    help="S4: loss weight on path segment starts. 1.0 is "
                         "exactly E5, so the default run is unchanged.")
    # E5. Emits <F:domain>/<F:page> and, because the model can then tell the two
    # sampling frames apart, DISABLES field-scoped content masking -- path
    # content trains on every row instead of structure rows only. Off by default
    # so an E4 run stays bit-identical; turning it on changes the vocabulary and
    # the loss, so it requires a fresh run rather than a resume.
    # E6. Factorises the output head so P(shape) and P(character) are separate
    # distributions. Targets the measured entropy-leakage mechanism; costs
    # d_model+1 parameters. Off by default so E5 stays reproducible.
    ap.add_argument("--factorized", action="store_true",
                    help="E6: factorised output head (shape vs character)")
    ap.add_argument("--frame", action="store_true",
                    help="E5: condition on the sampling frame and drop "
                         "field-scoped content masking")
    # E7 (research/e7/PREREG_E7.md). <P:*>/<Q:*> presence controls with
    # control dropout. Off by default, so every earlier run is unchanged.
    ap.add_argument("--fields", action="store_true",
                    help="E7: field-presence controls <P:0/1/?> <Q:0/1/?>")
    ap.add_argument("--field-dropout", type=float, default=0.0,
                    help="E7: P(control shown as ?) per row, per batch; "
                         "registered at 0.5")
    # E8 (research/native/PREREG_E8.md). Off by default: every earlier run is
    # unchanged.
    ap.add_argument("--presence-plan", action="store_true",
                    help="E8: presence plan after the suffix, trained, no dropout")
    ap.add_argument("--root-slash", action="store_true",
                    help="E8: the <RS> token for a bare root slash")
    # E9 (research/native/PREREG_E9.md). Defaults reproduce every earlier run.
    ap.add_argument("--loss-norm", choices=("batch", "fixed"), default="batch",
                    help="E9: 'fixed' = every scored token weighs the same")
    ap.add_argument("--loss-constant", type=float, default=None,
                    help="E9: C for 'fixed' (default: computed from the data)")
    ap.add_argument("--seed-init", action="store_true",
                    help="E9: seed the model's initial weights with the training seed")
    # E11 (research/native/PREREG_E11.md). Defaults reproduce every earlier run.
    ap.add_argument("--grad-clip-late", type=float, default=None,
                    help="E11: the clip cap after step 1,000 (registered: 5.0)")
    ap.add_argument("--field-dropout-joint", action="store_true",
                    help="E11: hide both presence controls together, or neither")
    ap.add_argument("--shared-start", action="store_true",
                    help="E11: build the <RS> model's start from the start without it")
    # E12 (research/native/PREREG_E12.md). Defaults reproduce every earlier run.
    ap.add_argument("--slash-plan", action="store_true",
                    help="E12: the slash plan <S:v> after <P:v> <Q:v>")
    ap.add_argument("--max-len", type=int, default=256,
                    help="E12 R1: the sequence budget (registered: 384)")
    ap.add_argument("--frame-every", action="store_true",
                    help="E12: the frame's embedding added at every position")
    ap.add_argument("--e12-start", action="store_true",
                    help="E12: E11's shared start, extended by seeded new rows")
    # E14 (research/e14/PREREG_E14.md). The default reproduces every earlier run.
    ap.add_argument("--query-weight", type=float, default=1.0,
                    help="E14 Q: loss weight on every step inside the query, training "
                         "rows only (registered: 4.0, with a pinned --loss-constant)")
    # E15-B (research/e15/PREREG_E15.md). The default reproduces every earlier run.
    ap.add_argument("--qry-extra", type=str, default=None,
                    help="E15-B: new query-bearing rows (JSON lines), injected alongside the "
                         "old ones with query-only targets")
    # E16 (research/e16/PREREG_E16.md). The default reproduces every earlier run.
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="E16: each batch in this many parts (gradient accumulation); "
                         "the batches and the summed gradient are unchanged")
    _run(ap.parse_args())


def new_model(cfg: ModelConfig, seed: Optional[int] = None) -> URLTransformer:
    """Build the model. E9: with a seed, the initial weights are fixed by it, so
    two arms built with one seed start identical. None is every earlier run:
    the initial weights come from whatever state the process's generator is in
    (E8_RESULTS.md section 3)."""
    if seed is not None:
        torch.manual_seed(seed)
    return URLTransformer(cfg)


#: E11 (PREREG_E11 D2): the seed of the generator that draws `<RS>`'s row.
RS_ROW_SEED = 20260929


def new_model_shared(cfg: ModelConfig, tokenizer: URLTokenizer,
                     seed: int) -> Tuple[URLTransformer, str]:
    """E11's shared start (research/native/PREREG_E11.md D2).

    `<RS>` adds one token, and the seeded initialisation draws every weight in
    one stream, so a 108-token model built from the seed would differ from E10's
    107-token start in EVERY weight. Instead: build the start the model would
    have without `<RS>` (the same seed, one row fewer -- E10's own start), copy
    every weight, and insert `<RS>`'s row at its place, drawn N(0, 0.02) -- the
    model's own initialisation -- from a separate generator.

    Returns the model and the fingerprint of the start without `<RS>`, which a
    run records so it can be checked against E10's.
    """
    from dataclasses import replace
    from uie.free import weights_sha256
    if not getattr(tokenizer, "root_slash", False):
        raise ValueError("the shared start extends a vocabulary by <RS>; this "
                         "tokenizer has none")
    base = new_model(replace(cfg, vocab_size=cfg.vocab_size - 1), seed)
    base_sha256 = weights_sha256(base)
    model = URLTransformer(cfg)
    rs = tokenizer.rs_id
    gen = torch.Generator().manual_seed(RS_ROW_SEED)
    made: Dict[int, torch.Tensor] = {}       # tied weights share storage: one row each
    state = {}
    for name, t in model.state_dict().items():
        b = base.state_dict()[name]
        if t.shape == b.shape:
            state[name] = b.clone()
            continue
        if t.shape[0] != b.shape[0] + 1 or t.shape[1:] != b.shape[1:]:
            raise ValueError(f"{name}: {tuple(b.shape)} -> {tuple(t.shape)} is not "
                             "one extra vocabulary row")
        key = b.data_ptr()
        if key not in made:
            row = (torch.randn(t.shape[1:], generator=gen) * 0.02).to(b.dtype)
            made[key] = torch.cat([b[:rs], row.unsqueeze(0), b[rs:]], dim=0)
        state[name] = made[key]
    model.load_state_dict(state)
    return model, base_sha256


#: E12 (PREREG_E12 D5): the seeds of the generators that draw the new rows.
S_ROWS_SEED = 20261001        # <S:0>, <S:1>
POS_ROWS_SEED = 20261002      # positions 256..383 (R1)


def new_model_e12(cfg: ModelConfig, tokenizer: URLTokenizer,
                  seed: int) -> Tuple[URLTransformer, Dict[str, str]]:
    """E12's start (research/native/PREREG_E12.md D5).

    Every arm starts from E11's shared start, rebuilt exactly as E11 built it (E10's
    seeded start plus `<RS>`'s row): the same tokenizer without the slash plan, 256
    positions. Every row is then copied by TOKEN and by POSITION; the rows E11 did
    not have are drawn N(0, 0.02), the model's own initialisation, from separate
    seeded generators -- `<S:0>` and `<S:1>` in vocabulary order, then positions
    256 onward. An arm with neither addition starts exactly at E11's start.

    Returns the model and the fingerprints a run records: E10's start (the part
    without `<RS>`) and E11's start, which V-id checks against their recorded values.
    """
    from dataclasses import replace
    from uie.free import weights_sha256
    e11_tok = URLTokenizer.from_state_dict({**tokenizer.state_dict(), "slash_plan": False})
    e11_cfg = replace(cfg, vocab_size=e11_tok.vocab_size, max_len=256, frame_every=False)
    e11_model, e10_sha256 = new_model_shared(e11_cfg, e11_tok, seed)
    prints = {"e10_start_sha256": e10_sha256, "e11_start_sha256": weights_sha256(e11_model)}
    model = URLTransformer(cfg)
    s_gen = torch.Generator().manual_seed(S_ROWS_SEED)
    p_gen = torch.Generator().manual_seed(POS_ROWS_SEED)
    old = e11_model.state_dict()
    rows = None                                # tied weights: one table, built once
    state = {}
    for name, t in model.state_dict().items():
        b = old[name]
        if t.shape == b.shape:
            state[name] = b.clone()
        elif name in ("tok_emb.weight", "head.weight"):
            if rows is None:
                rows = torch.stack([
                    b[e11_tok.stoi[tok]] if tok in e11_tok.stoi else
                    (torch.randn(b.shape[1:], generator=s_gen) * 0.02).to(b.dtype)
                    for tok in tokenizer.itos])
            state[name] = rows
        elif name == "pos_emb.weight" and t.shape[0] > b.shape[0] and t.shape[1:] == b.shape[1:]:
            extra = (torch.randn((t.shape[0] - b.shape[0],) + tuple(b.shape[1:]),
                                 generator=p_gen) * 0.02).to(b.dtype)
            state[name] = torch.cat([b, extra], dim=0)
        else:
            raise ValueError(f"{name}: {tuple(b.shape)} -> {tuple(t.shape)} is not an "
                             "E12 extension (new tokens or positions)")
    model.load_state_dict(state)
    # V-id: the drawn rows' fingerprints, so arms that share them can show it
    import hashlib
    new_tok = [i for i, tok in enumerate(tokenizer.itos) if tok not in e11_tok.stoi]
    emb = model.tok_emb.weight.detach()
    prints["new_token_rows_sha256"] = (hashlib.sha256(emb[new_tok].cpu().numpy().tobytes()).hexdigest()
                                       if new_tok else None)
    pos = model.pos_emb.weight.detach()
    prints["new_position_rows_sha256"] = (hashlib.sha256(pos[256:].cpu().numpy().tobytes()).hexdigest()
                                          if pos.shape[0] > 256 else None)
    return model, prints


def _run(args: argparse.Namespace) -> dict:
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    print("train streams:")
    tr_urls, tr_labels, tr_langs, tr_corp, label_counts = load_streams(
        args.data, STREAMS)
    print("val streams:")
    va_urls, va_labels, va_langs, va_corp, _ = load_streams(args.data, VAL_STREAMS)
    if not tr_urls:
        raise FileNotFoundError(
            f"no training streams found in {args.data}; expected "
            f"{[s for s, _, _ in STREAMS]}")
    print(f"loaded {len(tr_urls):,} train / {len(va_urls):,} val")
    print(f"label counts (class_alpha weights by these): {label_counts}")

    counts: Dict[str, int] = {}
    for lang in tr_langs:
        counts[lang] = counts.get(lang, 0) + 1
    keep = {lang for lang, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP_LANGUAGES]}
    keep.add("und")
    tr_langs = [lang if lang in keep else "und" for lang in tr_langs]
    va_langs = [lang if lang in keep else "und" for lang in va_langs]
    print(f"languages: {len(counts)} observed -> {len(keep)} control tokens")

    fields = bool(getattr(args, "fields", False))
    field_dropout = float(getattr(args, "field_dropout", 0.0))
    presence_plan = bool(getattr(args, "presence_plan", False))
    root_slash = bool(getattr(args, "root_slash", False))
    if presence_plan:
        # E8: the plan is always true and trained as a target, so the free
        # mode -- the model filling its own plan -- IS the validation task:
        # -log P(url) with the plan scored. There is nothing to hide.
        if not fields:
            raise ValueError("presence_plan needs fields=True")
        if field_dropout:
            raise ValueError("presence_plan is never hidden: field_dropout must "
                             "be 0 (PREREG_E8 section 3.2)")
    elif fields and not field_dropout:
        # Validation is scored in the unconditional (`?`) mode so it stays
        # comparable with E5; with no dropout that mode is never trained and
        # checkpoint selection would read an untrained input.
        raise ValueError("E7 needs field_dropout > 0 (registered: 0.5)")
    if field_dropout and not fields:
        raise ValueError("field_dropout needs fields=True")
    # E11 (research/native/PREREG_E11.md D2-D4)
    grad_clip_late = getattr(args, "grad_clip_late", None)
    field_dropout_joint = bool(getattr(args, "field_dropout_joint", False))
    shared_start = bool(getattr(args, "shared_start", False))
    if field_dropout_joint and (presence_plan or not field_dropout):
        raise ValueError("field_dropout_joint hides prefix controls: it needs "
                         "field_dropout > 0 and no presence plan")
    if shared_start and not (root_slash and getattr(args, "seed_init", False)):
        raise ValueError("shared_start extends the seeded start by <RS>: it needs "
                         "root_slash and seed_init")
    # E12 (research/native/PREREG_E12.md D2-D5)
    slash_plan = bool(getattr(args, "slash_plan", False))
    max_len = int(getattr(args, "max_len", 256))
    frame_every = bool(getattr(args, "frame_every", False))
    e12_start = bool(getattr(args, "e12_start", False))
    if frame_every and not args.frame:
        raise ValueError("frame_every reads the frame token: it needs frame=True")
    if e12_start and (shared_start or not (root_slash and getattr(args, "seed_init", False))):
        raise ValueError("e12_start extends E11's shared start: it needs root_slash and "
                         "seed_init, and replaces shared_start")
    tokenizer = URLTokenizer(languages=sorted(keep), include_frame=args.frame,
                             include_fields=fields, root_slash=root_slash,
                             presence_plan=presence_plan,
                             slash_plan=slash_plan).fit(tr_urls, min_count=1)
    if presence_plan:
        print("E8: presence PLAN ON -> <SUF> suffix <P:v> <Q:v> <LBL> ..., true "
              "values, trained as targets, no dropout; validation and selection "
              "score the plan (the free-mode likelihood)")
    elif fields:
        print(f"E7: presence controls ON -> <P:0/1/?> <Q:0/1/?>, control "
              f"dropout {field_dropout}; validation and selection are "
              f"UNCONDITIONAL (controls shown as ?), comparable with E5")
    if root_slash:
        print("E8: root slash ON -> <RS> for a bare root slash (host/, host/?q)")
    if slash_plan:
        print("E12: slash PLAN ON -> <P:v> <Q:v> <S:v>, S = 1 when a '/' follows the host")
    if max_len != 256:
        print(f"E12: max_len {max_len} (R1)")
    if frame_every:
        print("E12: the frame at every position ON (no new parameters)")
    if args.frame:
        print("E5: frame conditioning ON -> <F:domain>/<F:page> emitted, "
              "field-scoped content masking DISABLED")
    print(f"vocab: {tokenizer.vocab_size}")

    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=args.d_model,
                      n_layers=args.n_layers, n_heads=args.n_heads,
                      d_ff=4 * args.d_model,
                      struct_ids=(tokenizer.struct_ids()
                                  if getattr(args, "factorized", False) else ()),
                      max_len=max_len, frame_every=frame_every)
    if getattr(args, "factorized", False):
        print(f"E6: factorised head ON -> {len(cfg.struct_ids)} structural "
              f"tokens split from {tokenizer.vocab_size - len(cfg.struct_ids)} "
              f"content tokens; character entropy can no longer reach the "
              f"shape decision")
    seed = TrainConfig().seed if getattr(args, "seed_init", False) else None
    shared_base_sha256 = None
    e12_prints = None
    if e12_start:
        model, e12_prints = new_model_e12(cfg, tokenizer, seed)
        shared_base_sha256 = e12_prints["e10_start_sha256"]
        print(f"E12: start = E11's shared start ({e12_prints['e11_start_sha256'][:12]}...), "
              f"new rows from seeds {S_ROWS_SEED} (tokens) and {POS_ROWS_SEED} (positions)")
    elif shared_start:
        model, shared_base_sha256 = new_model_shared(cfg, tokenizer, seed)
        print(f"E11: shared start ON -> the seeded start without <RS> "
              f"({shared_base_sha256[:12]}...), <RS>'s row drawn from seed {RS_ROW_SEED}")
    else:
        model = new_model(cfg, seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M parameters")

    bw = getattr(args, "boundary_weight", 1.0)
    if bw != 1.0:
        print(f"S4: boundary weight {bw} on path segment starts "
              f"(first char after <PTH> or '/'); 1.0 would be E5")
    # VALIDATION ROWS KEEP WEIGHT 1.0. The eval loss must stay comparable
    # across arms; re-weighting it too would make a run that merely shifted
    # its own scoring look like a run that improved.
    # E14 Q: like S4's weight, the query weight reaches the TRAINING rows only.
    qw = float(getattr(args, "query_weight", 1.0) or 1.0)
    if qw != 1.0:
        print(f"E14: query weight {qw} on every step inside the query (training rows "
              f"only; validation unweighted); 1.0 would be E12")
    tr_enc, tr_dropped = encode_corpus(tr_urls, tr_labels, tokenizer,
                                       cfg.max_len, tr_langs, tr_corp,
                                       boundary_weight=bw, query_weight=qw)
    hidden = ("PTH", "QRY") if fields and not presence_plan else ()
    va_enc, va_dropped = encode_corpus(va_urls, va_labels, tokenizer,
                                       cfg.max_len, va_langs, va_corp,
                                       hide_fields=hidden)
    train_set = URLDataset(tr_enc, label_counts=label_counts)
    val_set = URLDataset(va_enc, label_counts=label_counts)
    val_cond = None
    if fields and not presence_plan:
        va_cond_enc, _ = encode_corpus(va_urls, va_labels, tokenizer,
                                       cfg.max_len, va_langs, va_corp)
        val_cond = URLDataset(va_cond_enc, label_counts=label_counts)
    print(f"encoded: {train_set.counts()}  dropped {tr_dropped:,} over max_len "
          f"({100*tr_dropped/max(len(tr_urls),1):.2f}%)")

    train_cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size,
                            lr=args.lr, eval_every=args.eval_every,
                            class_alpha=getattr(args, "class_alpha", 1.0),
                            boundary_weight=bw, field_dropout=field_dropout,
                            loss_norm=getattr(args, "loss_norm", "batch"),
                            loss_constant=getattr(args, "loss_constant", None),
                            grad_clip_late=grad_clip_late,
                            field_dropout_joint=field_dropout_joint,
                            # E13 C: off (None, 0.0) unless the arm sets it.
                            context_prefixes=(str(getattr(args, "context_prefixes"))
                                              if getattr(args, "context_prefixes", None)
                                              else None),
                            context_p=float(getattr(args, "context_p", 0.0) or 0.0),
                            query_weight=qw,
                            # E15-B: off (None) unless the arm sets it.
                            qry_extra=(str(getattr(args, "qry_extra"))
                                       if getattr(args, "qry_extra", None) else None),
                            # E16: 1 (one pass per batch) unless the arm sets it.
                            accum_steps=int(getattr(args, "accum_steps", 1) or 1))
    if train_cfg.accum_steps != 1:
        print(f"E16: gradient accumulation ON -> each batch in {train_cfg.accum_steps} "
              f"parts, weighted by scored mass; the batches are unchanged")
    if grad_clip_late is not None:
        print(f"E11: clip cap {train_cfg.grad_clip} for the first "
              f"{train_cfg.grad_clip_switch:,} steps, {grad_clip_late} after")
    if field_dropout_joint:
        print(f"E11: joint dropout ON -> both controls hidden together with "
              f"p = {field_dropout}, otherwise both shown")
    if train_cfg.loss_norm == "fixed":
        print(f"E9: length-neutral objective ON -> clipped gradients x T_b / C "
              f"(C = {train_cfg.loss_constant or 'computed from the data'})")
    started = time.time()
    result = train(model, train_set, val_set, tokenizer, train_cfg,
                   device=device, out_dir=args.out, log_every=200)
    print(f"\ntrained in {result['seconds']:.0f}s")
    print("final:", {k: round(v, 4) for k, v in result["final"].items()})

    save_checkpoint(model, tokenizer, train_cfg, args.out / "final.pt")

    final_conditioned = None
    if val_cond is not None:
        # The same final model, controls SHOWN. Carries more information than
        # the unconditional score, so it should sit at or below it.
        final_conditioned = evaluate(model, val_cond, train_cfg,
                                     tokenizer.pad_id, device)
        model.train()
        print("final (controls shown):",
              {k: round(v, 4) for k, v in final_conditioned.items()})

    # Per-field train/val gap. This is what decides whether the NEXT run scales
    # capacity: E2's label field sat at train 1.201 / val 2.346, an overfitting
    # signature that more parameters would have widened. Measuring it here means
    # the decision arrives with the run rather than needing a separate pass.
    print("\nper-field NLL (nats/char)")
    print(f"  {'field':<6}{'class':<10}{'train':>9}{'val':>9}{'gap':>9}")
    gaps: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label in train_set.classes:
        tr_sample = [u for u, l in zip(tr_urls, tr_labels) if l == label][:3_000]
        va_sample = [u for u, l in zip(va_urls, va_labels) if l == label][:3_000]
        on_train = field_nll(model, tokenizer, tr_sample, label, device,
                             hide_fields=hidden)
        on_val = field_nll(model, tokenizer, va_sample, label, device,
                           hide_fields=hidden)
        gaps[label] = {"train": on_train, "val": on_val,
                       "gap": {f: on_val[f] - on_train[f]
                               for f in on_val if f in on_train}}
        for f in sorted(gaps[label]["gap"], key=lambda k: -gaps[label]["gap"][k]):
            print(f"  {f:<6}{label:<10}{on_train[f]:>9.3f}{on_val[f]:>9.3f}"
                  f"{gaps[label]['gap'][f]:>+9.3f}", flush=True)
    # S4's PRE-REGISTERED OUTCOME, computed here so the run reports its own
    # verdict and no separate local pass is needed to read it.
    #
    #   E5 baseline, benign held-out:  boundary ce 3.168 / overconf +0.329
    #                                  mid      ce 1.410 / overconf +0.032
    #
    # SUCCESS  boundary ce falls >= 0.10 nats and boundary overconf <= +0.15
    # FAIL     boundary ce falls < 0.05 (the mechanism did not reach the defect;
    #          a temperature could already buy 0.034 without retraining)
    # REJECT   mid-segment ce rises > 0.05 (bought at the expense of the other
    #          94% of positions) or any field's val NLL regresses
    print("\nboundary vs mid-segment (nats/char)  -- S4 primary outcome")
    print(f"  {'class':<10}{'split':<7}{'kind':<10}{'n':>9}{'ce':>9}"
          f"{'h':>9}{'overconf':>10}")
    boundary: Dict[str, Dict[str, Dict[str, float]]] = {}
    shape: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label in train_set.classes:
        tr_sample = [u for u, l in zip(tr_urls, tr_labels) if l == label][:3_000]
        va_sample = [u for u, l in zip(va_urls, va_labels) if l == label][:3_000]
        boundary[label] = {"train": boundary_nll(model, tokenizer, tr_sample,
                                                 label, device,
                                                 hide_fields=hidden),
                           "val": boundary_nll(model, tokenizer, va_sample,
                                               label, device,
                                               hide_fields=hidden)}
        # STOP/STRUCTURE NLL. S4b exists because S4 damaged exactly this while
        # every content field stayed flat, so it is a first-class guardrail
        # rather than a diagnostic afterthought.
        shape[label] = {"train": shape_nll(model, tokenizer, tr_sample, label,
                                           device, hide_fields=hidden),
                        "val": shape_nll(model, tokenizer, va_sample, label,
                                         device, hide_fields=hidden)}
        sv = shape[label]["val"]
        print(f"  {label:<10}shape  EOS NLL {sv.get('eos_nll', 0):.3f}   "
              f"marker NLL {sv.get('marker_nll', 0):.3f}", flush=True)
        for split in ("train", "val"):
            r = boundary[label][split]
            for kind in ("boundary", "mid"):
                if f"{kind}_n" not in r:
                    continue
                print(f"  {label:<10}{split:<7}{kind:<10}{r[f'{kind}_n']:>9,.0f}"
                      f"{r[f'{kind}_ce']:>9.3f}{r[f'{kind}_h']:>9.3f}"
                      f"{r[f'{kind}_overconf']:>+10.3f}", flush=True)
        v = boundary[label]["val"]
        if "boundary_ce" in v:
            # A boundary train/val gap much WIDER than mid-segment's would mean
            # the openings are memorised, in which case up-weighting them is the
            # wrong intervention no matter what the training loss did.
            t = boundary[label]["train"]
            print(f"  {label:<10}gap    boundary "
                  f"{v['boundary_ce'] - t['boundary_ce']:+.3f}   mid "
                  f"{v['mid_ce'] - t['mid_ce']:+.3f}   "
                  f"overconf ratio {v.get('overconf_ratio', 0):.1f}x")
    model.train()

    samples: Dict[str, Dict[str, List[str]]] = {}
    for label in train_set.classes:
        samples[label] = {}
        for decoder in ("typical", "top_p", "temperature"):
            samples[label][decoder] = generate(
                model, tokenizer, label=label, n=20, decoder=decoder,
                seed=20260803, device=device)
        print(f"\n--- {label} (typical) ---")
        for url in samples[label]["typical"][:10]:
            print("   ", url[:100])

    # E7 QUICK HONOURING CHECK -- an early signal in the log, NOT the registered
    # C1 test (that runs locally at n = 4,000 under the adopted decoder).
    honoring: Dict[str, Dict[str, float]] = {}
    if presence_plan or field_dropout_joint:
        # E8: joint plans only (there are no partial ones), on a fixed suffix.
        # E11: a joint-dropout model never trains a partial state either.
        # A log signal, NOT the registered K1a reading.
        from uie.evaluation import FIELDS as _FIELDS
        from uie.factorization import parse as _parse
        print("\nE8 quick honouring check (n=200 per plan, temperature, suffix com)")
        for label in train_set.classes:
            honoring[label] = {}
            for want_p in (True, False):
                for want_q in (True, False):
                    # E12: with the slash plan, every state a URL can be in (a path
                    # implies the slash); the slash is honoured when the URL has a
                    # path component exactly when S = 1.
                    slashes = (((True,) if want_p else (True, False)) if slash_plan
                               else (None,))
                    for want_s in slashes:
                        plan = {"PTH": want_p, "QRY": want_q}
                        if want_s is not None:
                            plan["SLASH"] = want_s
                        urls = generate(model, tokenizer, label=label, n=200,
                                        decoder="temperature", seed=20260923,
                                        device=device, suffix="com", fields=plan)
                        ok = sum(1 for u in urls
                                 if bool(_FIELDS["PTH"](u)) == want_p
                                 and bool(_FIELDS["QRY"](u)) == want_q
                                 and (want_s is None or bool(_parse(u).path) == want_s)
                                 ) / len(urls)
                        key = f"P{int(want_p)}Q{int(want_q)}" + (
                            "" if want_s is None else f"S{int(want_s)}")
                        honoring[label][key] = round(ok, 4)
                        print(f"  {label:<10}{key}  honoured {100 * ok:5.1f}%", flush=True)
    elif fields:
        from uie.evaluation import FIELDS as _FIELDS
        print("\nE7 quick honouring check (n=200, temperature, other control ?)")
        for label in train_set.classes:
            honoring[label] = {}
            for field in ("PTH", "QRY"):
                for want in (True, False):
                    urls = generate(model, tokenizer, label=label, n=200,
                                    decoder="temperature", seed=20260923,
                                    device=device, fields={field: want})
                    has = sum(1 for u in urls if _FIELDS[field](u)) / len(urls)
                    rate = has if want else 1.0 - has
                    honoring[label][f"{field}:{int(want)}"] = round(rate, 4)
                    print(f"  {label:<10}{field}:{int(want)}  honoured "
                          f"{100 * rate:5.1f}%", flush=True)

    payload = {
        "params": n_params,
        "model_config": {k: getattr(cfg, k) for k in
                         ("vocab_size", "max_len", "d_model", "n_layers",
                          "n_heads", "d_ff")},
        "train_config": {"steps": args.steps, "batch_size": args.batch_size,
                         "lr": args.lr},
        "dropped_over_max_len": tr_dropped,
        "counts": train_set.counts(),
        "history": result["history"],
        "final": result["final"],
        "field_nll": gaps,
        "boundary_nll": boundary,
        "shape_nll": shape,
        "boundary_weight": bw,
        # E7 readback: a run that silently lost its controls would look healthy
        # everywhere else.
        "fields": fields,
        "field_dropout": field_dropout,
        "presence_plan": presence_plan,     # E8 readback
        "root_slash": root_slash,
        # E9 readback: the objective, its constant, the shared start, and what
        # clipping did by batch length
        "loss_norm": result.get("loss_norm"),
        "loss_constant": result.get("loss_constant"),
        "seed_init": bool(getattr(args, "seed_init", False)),
        "init_weights_sha256": result.get("init_weights_sha256"),
        "grad_diagnostics": result.get("grad_diagnostics"),
        # E11 readback (PREREG_E11 D2-D4)
        "grad_clip": train_cfg.grad_clip,
        "grad_clip_late": grad_clip_late,
        "grad_clip_switch": train_cfg.grad_clip_switch,
        "field_dropout_joint": field_dropout_joint,
        "shared_start": {"on": shared_start, "base_sha256": shared_base_sha256,
                         "rs_row_seed": RS_ROW_SEED if shared_start else None},
        # E12 readback (PREREG_E12 D2-D5)
        "slash_plan": slash_plan,
        "max_len": max_len,
        "frame_every": frame_every,
        "e12_start": {"on": e12_start, **(e12_prints or {}),
                      "s_rows_seed": S_ROWS_SEED if e12_start else None,
                      "pos_rows_seed": POS_ROWS_SEED if e12_start else None},
        # E13 readback (PREREG_E13 section 2.1, V5): None when the arm is off
        "context_swap": result.get("context_swap"),
        # E14 readback (PREREG_E14 section 2.1, V5): None when the arm is off
        "query_weight": result.get("query_weight"),
        # E15-B readback (PREREG_E15 section 10.4, V-batch, V-mass, V-stream): None when off
        "qry_extra": result.get("qry_extra"),
        # E16 readback (PREREG_E16 section 3): 1 when off
        "accum_steps": train_cfg.accum_steps,
        "vocab_size": tokenizer.vocab_size,
        "dropped_over_max_len_val": va_dropped,
        "final_conditioned": final_conditioned,
        "honoring_quick": honoring,
        "seconds": result["seconds"],
        "wall_seconds": round(time.time() - started, 1),
        "samples": samples,
    }
    (args.out / "train_result.json").write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out/'final.pt'} and {args.out/'train_result.json'}")
    return payload


if __name__ == "__main__":
    main()
