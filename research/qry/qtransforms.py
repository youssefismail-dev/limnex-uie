"""Deterministic transforms over QUERY strings, each destroying one property.

Same discipline as research/tell/transforms.py, for the same reason: the one
withdrawn headline in this project came from a transform that claimed to remove
repetition and silently destroyed depth, delimiters, length and edges as well.
Every function here is paired with a conservation test in
tests/test_qry_transforms.py that pins what it must LEAVE ALONE, not what it
outputs.

Input is a query body as `FIELDS["QRY"]` yields it -- no leading '?', e.g.
`id=131&page=2`. Output is the same shape.

The registered decomposition (PREREGISTRATION.md, Q3) is:

    parameter count            key/value structure and separators
    parameter-name identity    percent-encoding presence and correctness
    parameter-value length     character-class composition
    ordering and repetition    value entropy

SUBSTITUTION IS AVOIDED THROUGHOUT. Replacing a token with some other real token
manufactures signal -- measured on the path work, where four substitutive
transforms produced floors of 0.907, 0.294 and 0.13 and had to be discarded.
Everything here either masks a character class in place (length preserved) or
reorders; nothing swaps in content drawn from elsewhere.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Tuple

Transform = Callable[[str], str]

_PCT = re.compile(r"%[0-9A-Fa-f]{2}")


def split_params(q: str) -> List[Tuple[str, str, str]]:
    """Query -> [(name, sep, value)]. `sep` is '=' or '' when there is no value.

    Separators are preserved per-parameter rather than assumed, because whether
    a bare flag (`?debug`) appears at all is part of "key/value structure and
    separators" and must survive transforms that are not aimed at it.
    """
    out = []
    for part in q.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            out.append((k, "=", v))
        else:
            out.append((part, "", ""))
    return out


def join_params(parts: List[Tuple[str, str, str]]) -> str:
    return "&".join(k + s + v for k, s, v in parts)


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #

def t_identity(q: str) -> str:
    """E_0. Must reproduce the published excess or the harness is misplumbed."""
    return q


def t_null(q: str) -> str:
    """A no-op that is NOT identity: round-trips through the same parse/emit
    path every real transform uses. Any difference from `identity` is pure
    measurement noise, and that spread is what the thresholds are read against."""
    return join_params(split_params(q))


# --------------------------------------------------------------------------- #
# Destructive -- necessity
# --------------------------------------------------------------------------- #

def t_mask_names(q: str) -> str:
    """Parameter NAME IDENTITY -> 'a' * len. Conserves count, separators,
    name lengths, values, order, encoding."""
    return join_params([("a" * len(k), s, v) for k, s, v in split_params(q)])


def t_mask_name_chars(q: str) -> str:
    """Parameter-name LEXICAL IDENTITY only: letters -> 'a', digits -> '0'
    INSIDE names, punctuation kept.

    `mask_names` was supposed to do this and does not. It rewrites a name as
    'a' * len, which also deletes intra-name punctuation -- `utm_source` becomes
    `aaaaaaaaaa`, not `aaa_aaaaaa`. Three of the ten most common real parameter
    names carry an underscore, and `skeleton` (which PRESERVES that punctuation)
    loses no signal at all, so a collapse under `mask_names` cannot be
    attributed to name identity: character-class layout inside the name is
    confounded with it.

    This transform removes identity and nothing else. Read against `mask_names`,
    the difference between the two isolates intra-name punctuation.
    """
    out = []
    for k, sep, v in split_params(q):
        masked = "".join("0" if c.isdigit() else ("a" if c.isalpha() else c)
                         for c in k)
        out.append((masked, sep, v))
    return join_params(out)


def t_mask_values(q: str) -> str:
    """Parameter VALUE IDENTITY -> 'a' * len. Conserves count, names,
    separators, value LENGTHS, order. Kills value character composition."""
    return join_params([(k, s, "a" * len(v)) for k, s, v in split_params(q)])


def t_fixlen_values(q: str, width: int = 8) -> str:
    """Every value to a CONSTANT width. Kills value length AND identity.

    Read against `mask_values`, which kills identity but keeps length: the
    difference between the two isolates LENGTH specifically. Pads with 'a',
    a character the corpus already uses everywhere -- a novel pad character
    would itself become a perfect tell.
    """
    return join_params([(k, s, (v[:width] if len(v) >= width
                                else v + "a" * (width - len(v))) if s else v)
                        for k, s, v in split_params(q)])


def t_mask_digits(q: str) -> str:
    """All digits -> '0'. Kills digit IDENTITY, conserves digit POSITION and
    count, every length, and every other character."""
    return re.sub(r"\d", "0", q)


def t_mask_alpha(q: str) -> str:
    """All letters -> 'a'. Kills lexical content everywhere, conserves digit
    layout, separators, lengths and order."""
    return re.sub(r"[^\W\d_]", "a", q, flags=re.UNICODE)


def t_mask_encoding(q: str) -> str:
    """`%XX` -> `%00`. Kills percent-encoding IDENTITY while conserving its
    PRESENCE, its positions and every length."""
    return _PCT.sub("%00", q)


def t_sort_params(q: str) -> str:
    """Sort parameters. Kills ORDER only -- count, names, values, separators
    and repetition all survive byte-identically."""
    return join_params(sorted(split_params(q)))


# --------------------------------------------------------------------------- #
# Q3.1 -- finer shape decomposition
#
# Added after the first Q3 ladder established that SHAPE is sufficient
# (`skeleton` retains the whole tell) but left the dominant shape component
# unresolved: every individual component came back null at 3 split seeds.
#
# These target the specific dimensions the DESCRIPTIVE half flagged, and they
# are all DESTRUCTIVE or NORMALISING, applied identically to both sides. None
# substitutes content drawn from elsewhere -- the counterfactual "give the model
# the real distribution" is done by destroying the real side's excess down to
# the generated profile, not by transplanting tokens.
# --------------------------------------------------------------------------- #

def _clip(q: str, k: int) -> str:
    return join_params([(n, s, v[:k]) for n, s, v in split_params(q)])


def t_clip_values_20(q: str) -> str:
    """Truncate every value to <= 20 characters."""
    return _clip(q, 20)


def t_clip_values_15(q: str) -> str:
    """Truncate every value to <= 15 characters.

    THE TARGETED TRANSFORM. The descriptive half found real and generated values
    share a median of 4 while their p90s are 25 and 15 -- the model is not
    writing short values wrongly, it is missing the long tail. 15 is the
    GENERATED p90, so applying this to both sides removes the real side's excess
    tail and leaves the generated side largely untouched. If the tail is what
    the discriminator uses, excess should fall here and nowhere else.

    Destructive, not substitutive: nothing is transplanted, the same function is
    applied to both sides, and a value already shorter than the cap is returned
    byte-identical.
    """
    return _clip(q, 15)


def t_clip_values_10(q: str) -> str:
    """Truncate every value to <= 10 characters. Third point of a dose-response
    series with 20 and 15: a real effect should grow monotonically as the cap
    tightens, which a coincidence has no reason to do."""
    return _clip(q, 10)


def t_first_param(q: str) -> str:
    """Keep only the FIRST parameter. Normalises parameter count to 1.

    ASYMMETRIC EVIDENCE, and it is read only in one direction. Count cannot be
    removed without also removing the content of later parameters, so a COLLAPSE
    here is ambiguous between the two. SURVIVAL is clean: if excess persists
    when every query has exactly one parameter, parameter count is not the
    carrier.
    """
    parts = split_params(q)
    return join_params(parts[:1]) if parts else q


def t_fixlen_names(q: str, width: int = 8) -> str:
    """Pad or truncate every NAME to a constant width. Kills name LENGTH.

    Read against `mask_name_chars`, which kills name identity while conserving
    length: the difference isolates name length specifically.
    """
    out = []
    for n, s, v in split_params(q):
        out.append((n[:width] if len(n) >= width else n + "a" * (width - len(n)),
                    s, v))
    return join_params(out)


def t_mask_pct(q: str) -> str:
    """`%XX` -> `aaa`. Kills percent-encoding PRESENCE while conserving length.

    Distinct from `mask_encoding`, which kills the identity of the escape but
    leaves the '%' marker in place. Real queries are percent-encoded three times
    as often as generated ones (9.89% vs 3.15%), so PRESENCE is the property the
    descriptive half flagged, not which byte was escaped.
    """
    return _PCT.sub("aaa", q)


def t_normalize_seps(q: str) -> str:
    """Force every parameter into `name=value` form.

    A bare flag (`debug`) gains an '='. Kills the bare-flag / empty-value
    distinction, which is the "key/value structure and separators" component.
    Name and value content survive; only the separator profile is normalised.
    """
    return join_params([(n, "=", v) for n, s, v in split_params(q)])


def t_mask_punct(q: str) -> str:
    """Punctuation INSIDE names and values -> 'x'; '&' and '=' untouched.

    Kills intra-field punctuation identity while conserving every length, every
    position, and the structural separators that define the parameter layout.
    """
    out = []
    for n, s, v in split_params(q):
        f = lambda t: "".join(c if c.isalnum() else "x" for c in t)
        out.append((f(n), s, f(v)))
    return join_params(out)


def t_alnum_flat(q: str) -> str:
    """All alphanumerics -> 'a'; punctuation and separators verbatim.

    Read against `skeleton`, which is identical except that it preserves the
    DIGIT-vs-LETTER distinction (digits -> '0'). The difference between the two
    isolates that distinction, which no single transform can do alone. The
    descriptive half found generated values 55.6% purely numeric against real's
    42.8%, so this is a measured candidate rather than a speculative one.
    """
    return "".join("a" if c.isalnum() else c for c in q)


def t_clip15_first(q: str) -> str:
    """PRE-SPECIFIED INTERACTION: the long-value tail and parameter count
    together, and nothing else.

    Registered in advance and run ONLY if both components are individually weak,
    so that a combinatorial grid is never needed. If the pair explains what
    neither does alone, the tell is distributed across them rather than carried
    by either.
    """
    return t_first_param(_clip(q, 15))


# --------------------------------------------------------------------------- #
# Isolating -- sufficiency
# --------------------------------------------------------------------------- #

def t_skeleton(q: str) -> str:
    """ISOLATING: letters -> 'a', digits -> '0', punctuation verbatim.

    Retains structure, separators, lengths and character-CLASS layout, and
    nothing else. High excess here means shape alone is sufficient.
    """
    out = []
    for ch in q:
        out.append("0" if ch.isdigit() else ("a" if ch.isalpha() else ch))
    return "".join(out)


def t_structure_only(q: str) -> str:
    """ISOLATING: names and values both masked to their own lengths.

    Retains parameter count, separators and both length profiles. This is
    `mask_names` and `mask_values` applied together.
    """
    return join_params([("a" * len(k), s, "a" * len(v))
                        for k, s, v in split_params(q)])


def t_count_only(q: str) -> str:
    """ISOLATING, coarsest: retains the parameter COUNT and nothing else.

    'id=131&page=2' -> 'a=a&a=a'. The negative control of the ladder: a large
    excess here would mean the tell is carried by how many parameters the model
    writes, independent of what any of them contain.
    """
    return "&".join(["a=a"] * len(split_params(q)))


#: Order is the order they are reported in.
TRANSFORMS: Dict[str, Transform] = {
    "identity": t_identity,
    "null": t_null,
    "mask_names": t_mask_names,
    "mask_name_chars": t_mask_name_chars,
    "mask_values": t_mask_values,
    "fixlen_values": t_fixlen_values,
    "mask_digits": t_mask_digits,
    "mask_alpha": t_mask_alpha,
    "mask_encoding": t_mask_encoding,
    "sort_params": t_sort_params,
    "clip_values_20": t_clip_values_20,
    "clip_values_15": t_clip_values_15,
    "clip_values_10": t_clip_values_10,
    "first_param": t_first_param,
    "fixlen_names": t_fixlen_names,
    "mask_pct": t_mask_pct,
    "normalize_seps": t_normalize_seps,
    "mask_punct": t_mask_punct,
    "alnum_flat": t_alnum_flat,
    "clip15_first": t_clip15_first,
    "skeleton": t_skeleton,
    "structure_only": t_structure_only,
    "count_only": t_count_only,
}

#: Which registered component each speaks to, and in which direction.
COMPONENT = {
    "identity": ("--", "baseline"),
    "null": ("--", "null control"),
    "mask_names": ("param-name identity + intra-name punct  CONFOUNDED", "destructive"),
    "mask_name_chars": ("param-name identity ONLY", "destructive"),
    "mask_values": ("value composition", "destructive"),
    "fixlen_values": ("value LENGTH (vs mask_values)", "destructive"),
    "mask_digits": ("char class: digits", "destructive"),
    "mask_alpha": ("char class: letters", "destructive"),
    "mask_encoding": ("percent-encoding identity", "destructive"),
    "sort_params": ("ordering", "destructive"),
    "clip_values_20": ("value tail > 20", "destructive"),
    "clip_values_15": ("value tail > 15 (generated p90)", "destructive"),
    "clip_values_10": ("value tail > 10", "destructive"),
    "first_param": ("parameter count -> 1", "destructive  ONE-WAY"),
    "fixlen_names": ("name LENGTH (vs mask_name_chars)", "destructive"),
    "mask_pct": ("percent-encoding PRESENCE", "destructive"),
    "normalize_seps": ("separator structure", "destructive"),
    "mask_punct": ("intra-field punctuation", "destructive"),
    "alnum_flat": ("digit-vs-alpha distinction (vs skeleton)", "destructive"),
    "clip15_first": ("tail + count  PRE-SPECIFIED INTERACTION", "destructive"),
    "skeleton": ("structure + char class", "isolating"),
    "structure_only": ("count + separators + lengths", "isolating"),
    "count_only": ("parameter count", "isolating  NEGATIVE CONTROL"),
}
