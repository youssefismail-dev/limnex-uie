"""
research/v2/v2_p2_tokenizer.py

V2-P2's tokenizer (research/v2/PREREG_V2_P2.md section 4.6; Amendment 1, section 18.2):
`e16_cap`'s tokenizer, with learned pieces inside some fields.

- **Ids:** every `e16_cap` token keeps its id. The rest table's pieces follow, then (arm B
  only) the host table's.
- **Fields:**
  - PTH, QRY and FRG are written in rest pieces;
  - LBL and SUB in host pieces when the host table is present;
  - every other field (SUF, USR, PRT, SCH) stays in characters, as do markers, controls
    and plan tokens.
- **Encoding** is the character encoding, then each piece field's run of characters
  rewritten as pieces:
  - canonical (merges in the order they were learned) by default;
  - under BPE-dropout (Provilkov et al., 2020) when a generator is passed: at each step,
    every applicable merge occurrence is dropped with probability p, and the lowest-ranked
    kept occurrence (leftmost on a tie) is merged; segmentation stops when none is kept.
    At p = 0 this is the canonical encoding exactly.
- **Lossless:** decoding concatenates piece texts, so `decode(encode(u))` equals
  `e16_cap`'s tokenizer's decode of u, for every split.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from uie.tokenizer import _MARKERS, Fields, URLTokenizer  # noqa: E402
from v2_bpe import BPETable  # noqa: E402

#: which table writes which field (section 4.6)
FIELD_TABLE = {"PTH": "rest", "QRY": "rest", "FRG": "rest", "LBL": "host", "SUB": "host"}
REST_FIELDS = ("PTH", "QRY", "FRG")
HOST_FIELDS = ("LBL", "SUB")


def table_sha256(state: dict) -> str:
    """A table's fingerprint, exactly `v2_bpe.BPETable.sha256`: its alphabet and merges."""
    import hashlib
    import json
    core = json.dumps({"alphabet": list(state["alphabet"]), "merges": [list(m) for m in state["merges"]]},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(core.encode("utf-8")).hexdigest()


def rules_sha256() -> str:
    """V-T3's per-field allowed sets: which table each field may hold pieces of."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(FIELD_TABLE, sort_keys=True).encode("utf-8")).hexdigest()


def dropout_encode(table: BPETable, text: str, rng: random.Random, p: float) -> List[str]:
    """BPE-dropout (Provilkov et al., 2020), as Amendment 1 specifies it."""
    pieces = list(text)
    rank = table.rank
    while len(pieces) > 1:
        best_r = best_i = None
        for i in range(len(pieces) - 1):
            r = rank.get((pieces[i], pieces[i + 1]))
            if r is None or (p and rng.random() < p):
                continue
            if best_r is None or r < best_r:
                best_r, best_i = r, i
        if best_i is None:
            break
        pieces[best_i:best_i + 2] = [pieces[best_i] + pieces[best_i + 1]]
    return pieces


class HybridTokenizer(URLTokenizer):
    """`e16_cap`'s tokenizer plus pieces (see the module docstring)."""

    def __init__(self, *, base_state: dict, rest: Optional[dict] = None, host: Optional[dict] = None,
                 dropout: float = 0.0, cache_size: int = 500_000) -> None:
        base = URLTokenizer.from_state_dict(base_state)
        super().__init__(classes=base.classes, languages=base.languages,
                         include_scheme=base.include_scheme, include_language=base.include_language,
                         include_frame=base.include_frame, include_fields=base.include_fields,
                         root_slash=base.root_slash, presence_plan=base.presence_plan,
                         slash_plan=base.slash_plan)
        self._build(base._chars)
        if self.itos != base.itos:
            raise ValueError("the base vocabulary was not rebuilt exactly")
        self.base = base
        self.n_base = len(self.itos)
        self.dropout = float(dropout)
        self.tables: Dict[str, BPETable] = {}
        self._table_state: Dict[str, dict] = {}
        self.piece_ids: Dict[Tuple[str, str], int] = {}
        self.piece_text: Dict[int, str] = {}
        self.piece_table: Dict[int, str] = {}
        for name, state in (("rest", rest), ("host", host)):
            if state is None:
                continue
            table = BPETable(state["alphabet"], [tuple(m) for m in state["merges"]])
            self.tables[name] = table
            self._table_state[name] = {"alphabet": list(table.alphabet),
                                       "merges": [list(m) for m in table.merges]}
            for a, b in table.merges:
                s = a + b
                if (name, s) in self.piece_ids:
                    continue
                tid = len(self.itos)
                tok = f"\x00{name}:{s}"           # never a character, marker or control token
                self.itos.append(tok)
                self.stoi[tok] = tid
                self.piece_ids[(name, s)] = tid
                self.piece_text[tid] = s
                self.piece_table[tid] = name
        self._marker_of = {self.marker_id(m): m for m in _MARKERS}
        self._char_of = {i: t for i, t in enumerate(self.itos[:self.n_base]) if len(t) == 1}
        self._unk = self.stoi["<unk>"]
        self._cache: Dict[Tuple[str, str], List[int]] = {}
        self._cache_size = int(cache_size)

    # --- the piece ids ----------------------------------------------------------

    def ids_of_table(self, name: str) -> List[int]:
        return sorted(i for i, t in self.piece_table.items() if t == name)

    def field_tables(self) -> Dict[str, str]:
        """field -> the table whose pieces it may hold (the piece rule, section 4.6)."""
        return {f: t for f, t in FIELD_TABLE.items() if t in self.tables}

    def _piece_ids(self, table: str, pieces: Sequence[str]) -> List[int]:
        out = []
        for s in pieces:
            if len(s) == 1:
                out.append(self.stoi.get(s, self._unk))
            else:
                out.append(self.piece_ids[(table, s)])
        return out

    def _run(self, table: str, text: str, rng: Optional[random.Random], p: float) -> List[int]:
        tb = self.tables[table]
        if rng is None or not p:
            hit = self._cache.get((table, text))
            if hit is None:
                hit = self._piece_ids(table, tb.encode(text))
                if len(self._cache) >= self._cache_size:
                    self._cache.clear()
                self._cache[(table, text)] = hit
            return list(hit)
        return self._piece_ids(table, dropout_encode(tb, text, rng, p))

    # --- encoding ---------------------------------------------------------------

    def encode_chars(self, url: str, label: str, language: str = "und", frame: str = "domain",
                     hide_fields: Sequence[str] = ()) -> List[int]:
        """`e16_cap`'s encoding of the URL (characters only)."""
        return super().encode(url, label, language, frame=frame, hide_fields=hide_fields)

    def pieces_from_chars(self, ids: Sequence[int], rng: Optional[random.Random] = None,
                          p: Optional[float] = None) -> List[int]:
        """Rewrite each piece field's run of characters as pieces."""
        p = self.dropout if (rng is not None and p is None) else (p or 0.0)
        ftab = self.field_tables()
        out: List[int] = []
        current: Optional[str] = None
        run: List[str] = []

        def flush() -> None:
            if run:
                out.extend(self._run(ftab[current], "".join(run), rng, p))
                run.clear()

        for t in ids:
            if t in self._marker_of:
                flush()
                current = self._marker_of[t]
                out.append(t)
                continue
            ch = self._char_of.get(t)
            if ch is not None and current in ftab:
                run.append(ch)
                continue
            flush()
            out.append(t)
        flush()
        return out

    def encode(self, url: str, label: str, language: str = "und", frame: str = "domain",
               hide_fields: Sequence[str] = (), rng: Optional[random.Random] = None,
               p: Optional[float] = None) -> List[int]:
        return self.pieces_from_chars(self.encode_chars(url, label, language, frame, hide_fields), rng, p)

    # --- decoding ---------------------------------------------------------------

    def token_text(self, t: int) -> Optional[str]:
        """The text a content token writes: a character or a piece; None otherwise."""
        if t >= self.n_base:
            return self.piece_text.get(t)
        return self._char_of.get(t)

    def decode_fields(self, ids: Sequence[int]) -> Fields:
        """`URLTokenizer.decode_fields`, with pieces written as their text."""
        marker_ids = self._marker_of
        rs_id = self.stoi["<RS>"] if self.root_slash else None
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
                flush()
                current = None
                root_slash = True
                continue
            if token in marker_ids:
                flush()
                current = marker_ids[token]
                continue
            if current is None:
                continue
            text = self.token_text(token) if 0 <= token < len(self.itos) else None
            if text is not None:
                buffer.append(text)
        flush()

        def one(name: str) -> Optional[str]:
            values = collected.get(name)
            return values[0] if values else None

        path = one("PTH")
        if path is None and root_slash:
            path = ""
        return Fields(scheme=one("SCH"), userinfo=one("USR"), suffix=one("SUF") or "",
                      label=one("LBL") or "", subdomains=tuple(collected.get("SUB", ())),
                      port=one("PRT"), path=path, query=one("QRY"), fragment=one("FRG"))

    def chars_of(self, ids: Sequence[int]) -> List[int]:
        """The character encoding a piece encoding stands for (pieces spelled out)."""
        out: List[int] = []
        for t in ids:
            if t >= self.n_base:
                out.extend(self.stoi.get(c, self._unk) for c in self.piece_text[t])
            else:
                out.append(t)
        return out

    # --- persistence ------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"hybrid": "V2-P2", "base": self.base.state_dict(), "tables": self._table_state,
                "dropout": self.dropout}

    @classmethod
    def from_state_dict(cls, state: dict) -> "HybridTokenizer":
        if state.get("hybrid") != "V2-P2":
            raise ValueError("not a V2-P2 tokenizer state")
        t = state.get("tables", {})
        return cls(base_state=state["base"], rest=t.get("rest"), host=t.get("host"),
                   dropout=state.get("dropout", 0.0))
