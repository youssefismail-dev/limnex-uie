"""
uie/tokenizer.py

Turn the URL factorization into a token sequence a transformer can model.

The scheme
----------
A URL becomes field markers interleaved with characters:

    [BOS] [C:phishing] [L:und] [SCH]https [SUF]com [LBL]bankx-alert
          [SUB]login [SUB]secure [PTH]verify [EOS]

Three properties fall out of this, and each was a deliberate choice:

**Presence is encoded by the marker itself.** A URL with no query simply has no
`[QRY]` token. So the model learns which components exist as part of learning the
sequence — no separate "shape" distribution is needed, unlike the n-gram
baseline, which had to model it explicitly.

**Structure the grammar owns is never emitted.** The `://`, the dots between host
labels, the `/` that begins a path, the `?` and `#` separators — none appear as
tokens. The model cannot misplace a delimiter it never produces. This is the same
guarantee `path_body` gives the baseline, extended to every separator.

**Inside a field, characters are unconstrained.** Backslashes, percent-encoding,
doubled slashes and `@` userinfo all survive, because deliberate grammar abuse is
a real attack class and a model that cannot represent it cannot model phishing.

The alphabet this yields is small: ~110 distinct characters over 1.5M URLs, of
which 25 are non-ASCII appearing 195 times in ~120M characters (0.0002%) — URLs
are overwhelmingly percent-encoded and punycode. A character model over an
alphabet that small is cheap, so `min_count=1` is used and nothing is folded into
`<unk>`: trimming buys a handful of embedding rows and costs exact round-trip.

Why the suffix comes before the label
-------------------------------------
In the URL string the registrable label precedes its suffix (`shopwave` `.io`),
but the sequence reverses them. Generation is causal, so ordering the suffix
first lets the label be invented *conditioned on the TLD* — a `.jp` domain and a
`.gov` domain are named differently, and the model can only use that if it knows
the suffix before choosing the name.

The suffix is spelled out as characters rather than given an ID from a lookup
table. That is what lets a TLD the model never saw still condition generation
sensibly: `.finance` resembles `.credit` in spelling and behaviour. At generation
time the suffix is *supplied* (sampled from an external, updateable frequency
table) rather than generated, so a new TLD needs no retraining.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from uie.factorization import URLParts, parse, render
from uie.ngram import path_body, path_from_body

__all__ = ["URLTokenizer", "Fields"]


@dataclass(frozen=True)
class Fields:
    """A decoded token sequence, before reassembly into a URL."""

    scheme: Optional[str] = None
    userinfo: Optional[str] = None
    suffix: str = ""
    label: str = ""
    subdomains: Tuple[str, ...] = ()
    port: Optional[str] = None
    path: Optional[str] = None          # body only, without the leading "/"
    query: Optional[str] = None
    fragment: Optional[str] = None


#: Field markers, in the order the model emits them. Order is causal: each field
#: is generated knowing everything before it.
_MARKERS = ["SCH", "USR", "SUF", "LBL", "SUB", "PRT", "PTH", "QRY", "FRG"]

_BOS, _EOS, _PAD, _UNK = "<bos>", "<eos>", "<pad>", "<unk>"

#: Sampling frames. NOT two halves of one distribution -- two different ways of
#: drawing from the same web:
#:
#:     domain   one URL per registrable name, over EVERY name
#:     page     up to 40 URLs from a single multi-page site
#:
#: They disagree systematically about paths: measured on the benign corpus, the
#: domain frame has 72.3% of URLs carrying a path at a conditional length of 28.3
#: characters, the page frame 97.2% at 39.7. `field_mask` currently handles that
#: disagreement by MASKING structure-field content outside the structure corpus,
#: which leaves shape and content under different teachers -- the asymmetry that
#: produced both the empty-path bug and the short-path bias.
#:
#: Naming the frame instead lets the model learn P(path | frame) directly, so no
#: masking is needed and every row can teach content. `<F:*>` is supplied at
#: generation exactly like `<C:*>`.
_FRAMES = ("domain", "page")

#: E7 -- field-presence controls (research/e7/PREREG_E7.md). PTH and QRY only.
#:
#: Presence is this tokenizer's OWN definition -- a structure field opens only
#: when its body is non-empty (see `encode`) -- which is also the gate's:
#: `evaluation.FIELDS` drops empty values. So `<Q:1>` means exactly "the gate
#: will see a query". "?" is the hidden value: control dropout trains it, and
#: generation uses it for unconditional sampling.
_FIELD_CONTROLS = {"PTH": "P", "QRY": "Q"}
_FIELD_VALUES = {True: "1", False: "0", None: "?"}

#: E8 (research/native/PREREG_E8.md) -- the root slash of a URL with no path
#: body (`host/`, `host/?q`). Real bare hosts carry it 97.9-100% of the time;
#: without a token for it the model can never write it (NATIVE_REPAIR_PLAN K3).
#: It is NOT a path: `<PTH>` still means a non-empty body, so every presence
#: definition, control and gate reading is unchanged.
_ROOT_SLASH = "<RS>"

#: E12 (research/native/PREREG_E12.md D2) -- the SLASH PLAN: a third plan token,
#: `<S:1>` when a "/" follows the host (a path body or the bare root slash) and
#: `<S:0>` when none does (`host`, `host?q`, `host#f`). It follows `<P:v> <Q:v>`,
#: is always true in training, and is supplied from the data at generation, as P
#: and Q are. P = 1 implies S = 1.
_SLASH_PLAN = ("<S:0>", "<S:1>")
_SLASH_FIELD = "SLASH"


class URLTokenizer:
    """Character-level tokenizer over the factorization, with control tokens."""

    def __init__(self, *, classes: Sequence[str] = ("benign", "phishing"),
                 languages: Sequence[str] = ("und",),
                 include_scheme: bool = False,
                 include_language: bool = False,
                 include_frame: bool = False,
                 include_fields: bool = False,
                 root_slash: bool = False,
                 presence_plan: bool = False,
                 slash_plan: bool = False) -> None:
        """E8 flags (research/native/PREREG_E8.md), both off by default so every
        earlier tokenizer is byte-identical:

        `root_slash=True` adds `<RS>`, written where `<PTH>` would be when the
        path is the bare root slash. `presence_plan=True` (needs
        `include_fields`) moves presence from the prefix to a PLAN right after
        the suffix, `<SUF> suffix <P:v> <Q:v> <LBL> ...`, always carrying the true
        values -- trained as targets, never hidden.

        `include_scheme=False` DROPS the scheme before tokenization.

        The scheme is not a property of how a URL is named; it is a property of
        when and how the URL was observed. Measured across the corpus:

            benign historical (2019-2022)   76.6% https
            benign live       (2024-2026)   92.5% https
            phishing          (2024-2026)   64.4% https

        Most of the benign/phishing gap is the web's TLS migration showing
        through a feed that captures sites at different moments, not a fact
        about phishing. A model given the scheme learns `http => phishing`, and
        a discriminator rewards it for doing so.

        SNOW Guard hit this exact artifact and fixed it the same way (V3.5,
        scheme stripping, which flipped zero predictions on its control set and
        improved recall). Removing the field is strictly better than balancing
        it in the corpus: a shortcut the model never receives cannot be learned,
        and no future sampling change can reintroduce it.

        `True` is retained for loading pre-E2.3 checkpoints, whose vocabulary
        and field order assume the scheme is present.

        `include_language=False` DROPS the `<L:*>` control token.

        It was a class marker. The phishing feed carries no language label, so
        `und` was 100% of phishing rows against 1.6% of benign — any non-`und`
        token guaranteed the class, and `<C:phishing>` + `<L:eng>` was entirely
        out of distribution. Two control surfaces DESIGN.md 4.5 advertises as
        independent were a function of one another.

        Neither repair survived measurement. Dropout discards real signal.
        Inferring the language from URL text -- the symmetric fix, same
        procedure for both classes -- turns out to be almost entirely the TLD
        restated:

            majority-class baseline        0.4380
            full URL                       0.8081
            TLD ONLY                       0.7750
            URL without the TLD            0.6734

        The whole URL beyond its TLD adds 3.3 points. The model already
        generates the suffix, conditions every later field on it, and 4.5 makes
        it externally supplyable, so an inferred token would launder the suffix
        through a second channel at 19% label noise.

        Language conditioning therefore moves to the SUFFIX: supply `.jp` and
        the model names in Japanese style. That covers 77.5% of the signal and
        costs one advertised control -- a Turkish site on `.com` can no longer
        be requested directly.
        """
        self.classes = list(classes)
        self.languages = list(languages)
        self.include_scheme = bool(include_scheme)
        self.include_language = bool(include_language)
        self.include_frame = bool(include_frame)
        self.include_fields = bool(include_fields)
        self.root_slash = bool(root_slash)
        self.presence_plan = bool(presence_plan)
        self.slash_plan = bool(slash_plan)
        if self.presence_plan and not self.include_fields:
            raise ValueError("presence_plan uses the presence tokens; it needs "
                             "include_fields=True")
        if self.slash_plan and not (self.presence_plan and self.root_slash):
            raise ValueError("slash_plan is a third plan token over the root slash: "
                             "it needs presence_plan=True and root_slash=True")
        self._chars: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        self._build([])

    # --- vocabulary -------------------------------------------------------

    def _build(self, chars: Sequence[str]) -> None:
        self._chars = list(chars)
        specials = [_PAD, _BOS, _EOS, _UNK]
        markers = [f"<{m}>" for m in _MARKERS]
        control = [f"<C:{c}>" for c in self.classes]
        if self.include_frame:
            control += [f"<F:{f}>" for f in _FRAMES]
        if self.include_language:
            control += [f"<L:{lang}>" for lang in self.languages]
        if self.include_fields:
            control += [f"<{c}:{v}>" for c in _FIELD_CONTROLS.values()
                        for v in ("0", "1", "?")]
        if self.root_slash:
            control.append(_ROOT_SLASH)
        if self.slash_plan:
            control.extend(_SLASH_PLAN)
        self.itos = specials + markers + control + self._chars
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def fit(self, urls: Iterable[str], *, min_count: int = 1,
            languages: Optional[Iterable[str]] = None) -> "URLTokenizer":
        """Learn the character vocabulary from a corpus.

        `min_count` drops characters seen fewer times than the threshold. They
        become `<unk>` — improbable rather than impossible, and encoding stays
        total so no URL is unrepresentable.
        """
        counts: Counter = Counter()
        for url in urls:
            parts = parse(url)
            for text in (parts.scheme, parts.userinfo, parts.public_suffix,
                         parts.registrable_label, parts.port,
                         path_body(parts.path), parts.query, parts.fragment):
                counts.update(text or "")
            for sub in parts.subdomains:
                counts.update(sub)
        if languages is not None:
            self.languages = sorted({lang for lang in languages} | {"und"})
        self._build([c for c, n in sorted(counts.items()) if n >= min_count])
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[_PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[_BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[_EOS]

    def marker_id(self, name: str) -> int:
        return self.stoi[f"<{name}>"]

    # --- encoding ---------------------------------------------------------

    def _chars_to_ids(self, text: str) -> List[int]:
        unk = self.stoi[_UNK]
        return [self.stoi.get(ch, unk) for ch in text]

    def struct_ids(self) -> Tuple[int, ...]:
        """Field markers plus EOS: the STRUCTURAL alphabet.

        These are the tokens that decide URL SHAPE. Handing them to
        `ModelConfig.struct_ids` factorises the output head so that character
        entropy cannot leak into the shape decision -- the mechanism measured on
        E5, where mid-path EOS mass rose 0.111% -> 3.022% (27x) purely as a
        function of the model's uncertainty about the next character.
        """
        return tuple(sorted([self.marker_id(m) for m in _MARKERS]
                            + [self.eos_id]
                            + ([self.stoi[_ROOT_SLASH]] if self.root_slash else [])))

    # --- E8: root slash and presence plan ----------------------------------------

    @property
    def rs_id(self) -> int:
        if not self.root_slash:
            raise ValueError("this tokenizer has no root-slash token (before E8)")
        return self.stoi[_ROOT_SLASH]

    def plan_slots(self) -> List[Tuple[int, int]]:
        """The legal tokens of each plan slot, in order: [(P:0, P:1), (Q:0, Q:1)],
        then (S:0, S:1) with the slash plan (E12). Index 0 is absent, 1 present."""
        if not self.presence_plan:
            raise ValueError("this tokenizer has no presence plan")
        slots = [(self.field_control_id(f, False), self.field_control_id(f, True))
                 for f in _FIELD_CONTROLS]
        if self.slash_plan:
            slots.append((self.stoi[_SLASH_PLAN[0]], self.stoi[_SLASH_PLAN[1]]))
        return slots

    def plan_fields(self) -> Tuple[str, ...]:
        """The fields a supplied plan must set: PTH, QRY, and SLASH with E12's
        slash plan."""
        return tuple(_FIELD_CONTROLS) + ((_SLASH_FIELD,) if self.slash_plan else ())

    def plan_ids(self, fields: Dict[str, bool]) -> List[int]:
        """A SUPPLIED plan: every plan field must be set, True or False. A partial
        plan is refused -- there are no partial states (NATIVE_REPAIR_PLAN). With
        the slash plan (E12), SLASH is a plan field and a path implies it."""
        if not self.presence_plan:
            raise ValueError("this tokenizer has no presence plan")
        fields = dict(fields or {})
        known = self.plan_fields()
        unknown = set(fields) - set(known)
        if unknown:
            raise KeyError(f"no presence plan for {sorted(unknown)}; "
                           f"known: {sorted(known)}")
        if any(not isinstance(fields.get(f), bool) for f in known):
            raise ValueError(f"a plan sets every field to True or False; got {fields}")
        ids = [self.field_control_id(f, fields[f]) for f in _FIELD_CONTROLS]
        if self.slash_plan:
            if fields["PTH"] and not fields[_SLASH_FIELD]:
                raise ValueError("a path implies the slash: PTH=True needs SLASH=True")
            ids.append(self.stoi[_SLASH_PLAN[int(fields[_SLASH_FIELD])]])
        return ids

    # --- E7 field-presence controls ------------------------------------------

    def _require_fields(self) -> None:
        if not self.include_fields:
            raise ValueError(
                "this tokenizer has no field-presence controls (trained before "
                "E7). A presence request cannot be honoured, so it is refused "
                "rather than silently ignored.")

    def field_control_id(self, field: str, present: Optional[bool]) -> int:
        """Id of `<P:*>` / `<Q:*>`: True -> 1, False -> 0, None -> ? (hidden)."""
        self._require_fields()
        if field not in _FIELD_CONTROLS:
            raise KeyError(f"no presence control for {field!r}; "
                           f"known: {sorted(_FIELD_CONTROLS)}")
        if present is not None and not isinstance(present, bool):
            raise ValueError(f"presence must be True, False or None; "
                             f"got {present!r}")
        return self.stoi[f"<{_FIELD_CONTROLS[field]}:{_FIELD_VALUES[present]}>"]

    def field_control_ids(self, fields: Optional[Dict[str, Optional[bool]]] = None
                          ) -> List[int]:
        """Prefix ids for GENERATION: the requested presence, `?` elsewhere."""
        self._require_fields()
        fields = dict(fields or {})
        unknown = set(fields) - set(_FIELD_CONTROLS)
        if unknown:
            raise KeyError(f"no presence control for {sorted(unknown)}; "
                           f"known: {sorted(_FIELD_CONTROLS)}")
        return [self.field_control_id(f, fields.get(f)) for f in _FIELD_CONTROLS]

    def field_control_positions(self) -> Dict[str, int]:
        """Where each presence control sits in an encoded sequence.

        Fixed, because the prefix is: BOS, class, then frame and language when
        present, then the controls in `_FIELD_CONTROLS` order. Control dropout
        swaps the token at these positions for its `?` form.
        """
        self._require_fields()
        if self.presence_plan:
            raise ValueError("a presence plan sits after the suffix, at no fixed "
                             "position, and is never hidden: control dropout does "
                             "not apply (PREREG_E8 section 3.2)")
        start = 2 + int(self.include_frame) + int(self.include_language)
        return {f: start + i for i, f in enumerate(_FIELD_CONTROLS)}

    def encode(self, url: str, label: str, language: str = "und",
               frame: str = "domain",
               hide_fields: Sequence[str] = ()) -> List[int]:
        """URL -> token ids, prefixed by its control tokens.

        `hide_fields` (E7) encodes those presence controls as `?` -- the
        unconditional form, used for control dropout and for validation that
        must stay comparable with pre-E7 checkpoints.
        """
        parts = parse(url)
        cls_tok = f"<C:{label}>"
        if cls_tok not in self.stoi:
            raise KeyError(f"unknown class {label!r}; known: {self.classes}")
        ids = [self.bos_id, self.stoi[cls_tok]]
        if self.include_frame:
            if frame not in _FRAMES:
                raise KeyError(f"unknown frame {frame!r}; known: {_FRAMES}")
            ids.append(self.stoi[f"<F:{frame}>"])
        if self.include_language:
            lang_tok = f"<L:{language}>"
            if lang_tok not in self.stoi:
                lang_tok = "<L:und>"
            ids.append(self.stoi[lang_tok])
        # PRESENCE IS DERIVED EXACTLY AS THE FIELDS ARE OPENED BELOW, so a
        # control can never disagree with the sequence it prefixes: a training
        # row that said <Q:1> above a URL with no <QRY> would teach the control
        # to be ignored.
        body_present = bool(path_body(parts.path) if parts.path else "")
        present = {"PTH": body_present,
                   "QRY": bool(parts.has_query and parts.query)}
        if self.slash_plan:
            # E12: a "/" follows the host -- a path body, or the bare root slash
            # that <RS> writes below. Derived from the same `parts.path`, so the
            # plan can never disagree with the sequence it prefixes.
            present[_SLASH_FIELD] = bool(parts.path)
        if self.presence_plan:
            # E8: no controls in the prefix; the plan follows the suffix below.
            if hide_fields:
                raise ValueError("a presence plan is never hidden (no dropout)")
        elif self.include_fields:
            hidden = set(hide_fields)
            unknown = hidden - set(_FIELD_CONTROLS)
            if unknown:
                raise KeyError(f"cannot hide {sorted(unknown)}; "
                               f"known: {sorted(_FIELD_CONTROLS)}")
            ids.extend(self.field_control_id(f, None if f in hidden else present[f])
                       for f in _FIELD_CONTROLS)
        elif hide_fields:
            raise ValueError("hide_fields needs a tokenizer with field controls")

        def field(name: str, text: str) -> None:
            ids.append(self.marker_id(name))
            ids.extend(self._chars_to_ids(text))

        if self.include_scheme and parts.has_scheme:
            field("SCH", parts.scheme)
        if parts.has_userinfo:
            field("USR", parts.userinfo)
        field("SUF", parts.public_suffix)
        if self.presence_plan:
            # E8: the TRUE plan, right after the suffix it may depend on.
            ids.extend(self.plan_ids(present))
        field("LBL", parts.registrable_label)
        for sub in parts.subdomains:
            field("SUB", sub)
        if parts.port:
            field("PRT", parts.port)

        # A STRUCTURE FIELD OPENS ONLY IF ITS BODY IS NON-EMPTY.
        #
        # These three used to open on `parts.path` / `has_query` / `has_fragment`,
        # and "/" is truthy, so `example.com/` emitted <PTH> followed immediately
        # by EOS. That put the two branches of ONE decision under two different
        # loss weights: `field_mask` treats markers and EOS as SHAPE (weight 1.0
        # on every row) but the characters of a structure field as CONTENT
        # (weight 0 outside the structure corpus). So on a name-corpus row:
        #
        #     example.com/          <PTH> w=1.0   EOS w=1.0    <- taught
        #     example.com/login     <PTH> w=1.0   'l' w=0.0    <- masked
        #
        # The name corpus supplied the evidence FOR an empty body while its
        # counter-evidence was masked out. Measured on the E3 checkpoint: the
        # marker matched training exactly (100.0% vs 100.0%) while non-empty
        # bodies ran 33.3 points under for phishing and 13.9 under for benign —
        # and phishing is worse precisely because its name corpus is 80.6% of
        # rows against benign's 35.5%.
        #
        # Gating on the BODY makes "does this URL have a path" a pure marker
        # decision, trained on all rows as shape, and leaves content as pure
        # characters, scoped to the structure corpus. The two branches become
        # the same kind of token, so the asymmetry cannot arise. It is also
        # semantically right: `example.com/` and `example.com` are one resource,
        # as are `a.com/x?` and `a.com/x`.
        body = path_body(parts.path) if parts.path else ""
        if body:
            field("PTH", body)
        elif self.root_slash and parts.path:
            # E8: the bare root slash (the only path with an empty body).
            ids.append(self.stoi[_ROOT_SLASH])
        if parts.has_query and parts.query:
            field("QRY", parts.query)
        if parts.has_fragment and parts.fragment:
            field("FRG", parts.fragment)

        ids.append(self.eos_id)
        return ids

    # --- decoding ---------------------------------------------------------

    def decode_fields(self, ids: Sequence[int]) -> Fields:
        """Token ids -> field values. Tolerant of malformed model output."""
        marker_ids = {self.marker_id(m): m for m in _MARKERS}
        rs_id = self.stoi[_ROOT_SLASH] if self.root_slash else None
        root_slash = False
        current: Optional[str] = None
        buffer: List[str] = []
        collected: Dict[str, List[str]] = {}

        def flush() -> None:
            if current is not None:
                collected.setdefault(current, []).append("".join(buffer))
            buffer.clear()

        for token in ids:
            if token in (self.bos_id, self.pad_id):
                continue
            if token == self.eos_id:
                break
            if token == rs_id:
                flush()                    # E8: a path with an empty body
                current = None
                root_slash = True
                continue
            if token in marker_ids:
                flush()
                current = marker_ids[token]
                continue
            if current is None:
                continue                   # control tokens and stray characters
            if 0 <= token < len(self.itos):
                tok = self.itos[token]
                if len(tok) == 1:          # a character, not a special token
                    buffer.append(tok)
        flush()

        def one(name: str) -> Optional[str]:
            values = collected.get(name)
            return values[0] if values else None

        path = one("PTH")
        if path is None and root_slash:
            path = ""                      # renders as the root slash
        return Fields(
            scheme=one("SCH"), userinfo=one("USR"),
            suffix=one("SUF") or "", label=one("LBL") or "",
            subdomains=tuple(collected.get("SUB", ())),
            port=one("PRT"), path=path,
            query=one("QRY"), fragment=one("FRG"),
        )

    @staticmethod
    def fields_to_url(fields: Fields) -> str:
        """Reassemble a URL. The grammar supplies every separator."""
        host = ".".join([s for s in fields.subdomains if s] +
                        ([fields.label] if fields.label else []) +
                        ([fields.suffix] if fields.suffix else []))
        parts = URLParts(
            scheme=fields.scheme or "", has_scheme=fields.scheme is not None,
            userinfo=fields.userinfo or "",
            has_userinfo=fields.userinfo is not None,
            subdomains=fields.subdomains, registrable_label=fields.label,
            public_suffix=fields.suffix, host_raw=host, port=fields.port or "",
            path=path_from_body(fields.path) if fields.path is not None else "",
            query=fields.query or "", has_query=fields.query is not None,
            fragment=fields.fragment or "",
            has_fragment=fields.fragment is not None,
        )
        return render(parts)

    def decode(self, ids: Sequence[int]) -> str:
        return self.fields_to_url(self.decode_fields(ids))

    # --- persistence ------------------------------------------------------

    def state_dict(self) -> dict:
        state = {"classes": self.classes, "languages": self.languages,
                 "chars": self._chars, "include_scheme": self.include_scheme,
                 "include_language": self.include_language,
                 "include_frame": self.include_frame}
        # Written only when on, so a pre-E7 tokenizer's state -- and any hash of
        # it -- stays byte-for-byte what it was.
        if self.include_fields:
            state["include_fields"] = True
        if self.root_slash:
            state["root_slash"] = True
        if self.presence_plan:
            state["presence_plan"] = True
        if self.slash_plan:
            state["slash_plan"] = True
        return state

    @classmethod
    def from_state_dict(cls, state: dict) -> "URLTokenizer":
        # Checkpoints written before E2.3 have no `include_scheme` key and were
        # trained WITH the scheme, so the default here must be True. Defaulting
        # to the new behaviour would silently re-encode old checkpoints against
        # a field layout their weights were never trained on.
        tok = cls(classes=state["classes"], languages=state["languages"],
                  include_scheme=state.get("include_scheme", True),
                  include_language=state.get("include_language", True),
                  # Pre-E5 checkpoints have no frame token and were trained
                  # without one, so this must default to False -- the mirror of
                  # the include_scheme reasoning above.
                  include_frame=state.get("include_frame", False),
                  # Pre-E7 checkpoints have no presence controls and were
                  # trained without them. Same reasoning as the frame above.
                  include_fields=state.get("include_fields", False),
                  # Pre-E8: neither flag existed; both default off.
                  root_slash=state.get("root_slash", False),
                  presence_plan=state.get("presence_plan", False),
                  # Pre-E12: no slash plan.
                  slash_plan=state.get("slash_plan", False))
        tok._build(state["chars"])
        return tok
