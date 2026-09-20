"""Kinds, keys, modes and the per-leaf simplify policy.

`mode` (which kinds `compute()` composes) and `simplify` (how hard each leaf
is looked at) share one machinery here. Membership itself is decided by
`check.is_kind` (re-exported below); this module lowers the many key
spellings and resolves the per-leaf policy.

Keys are named, and names are the documented path:

* a transformation-set NAME, case-insensitively: ``"affine"``, ``"linear"``,
  ``"rigid"``, ``"rotation"``, ``"bijective"``, ``"injective"``,
  ``"surjective"``, ``"scaling"`` (the diagonal set); a mathematical symbol:
  ``"Aff"``, ``"SO"``, ``"SO(3)"`` (the ``(n)`` fixes the dimension). A NAME
  means the *set*: ``"affine"`` covers ``Linear``, ``Scaling``,
  ``Subspace(Affine)``, ``InverseAffine``, ...;
* a wrapper or field key: ``"subspace"``, ``"inverse"``, ``"projection"``,
  ``"meta"``, ``"field"``, ``"displacements"``, ``"coordinates"``. These have
  no hierarchy node and are matched by ``isinstance``;
* a hierarchy type (e.g. ``hierarchy.AffineTransformation``) means that
  *set*; a concrete class (e.g. ``Affine``) means ``isinstance(t, Affine)``
  (only ``Affine`` and its Python subclasses), NOT the affine set;
* an ``int`` fixes only the dimension (every kind of that ``ndim``);
* a callable ``(t, compute) -> bool`` is an extension hook.

Membership (`check.is_kind`) has two levels: ``compute=False`` (analytic,
structure only) and ``compute=True`` (numeric, values + rank). **Analytic
invertibility is assumed from shape**: a square matrix transformation is
presumed invertible (hence bijective), a wide one surjective, a tall one
injective, without reading any value; numeric verifies with rank and may
retract the assumption. Resolution (deciding which mode admits a leaf, or
which simplify entry applies) always runs at analytic and never reads a
value. The `simplify` ladder is ``none < analytic < numeric``.
"""

# stdlib
from collections.abc import Mapping

# dependencies
import typing_extensions as tx

# bagof
from bagof.converters import get_converter

# datamodel
from brainhops.datamodel import hierarchy
from brainhops.datamodel.enums import SimplifyPolicy

# The single membership predicate lives in `check`, next to the checker
# registry and the name resolution it dispatches on. It is re-exported here
# because `mode`/`simplify` resolution (below) is its main caller.
# `KIND_ALIASES` (the field/wrapper name -> class map) is consulted when
# lowering a key that `parseType` does not resolve.
from .check import KIND_ALIASES, is_kind  # noqa: F401

# typing
if tx.TYPE_CHECKING:
    # internals
    from .base import Transformation

# `mode` and `simplify` share one machinery. A *kind* is what a key denotes:
# a hierarchy node (set membership, decided by `check.is_kind`, which sees
# through the wrappers via the checker registry), a class or tuple of classes
# (plain `isinstance`, for the meta/field wrappers), or a callable
# `(t, compute) -> bool` extension hook. `mode` selects which kinds
# `compute()` composes; `simplify` selects, per kind, how hard each leaf is
# looked at (the `SimplifyPolicy` ladder).
#
# This module keeps a tight import discipline (`hierarchy`, `enums`, `bagof`
# only) so `base`, `inverse`, `meta` can import it at the top level without a
# cycle. It only *lowers* keys and resolves the per-leaf simplify policy;
# membership itself is `check.is_kind`, and the checkers that back it
# register from the concrete/wrapper modules at their own import time.

Kind = tx.Union[
    type,
    tx.Tuple[type, ...],
    tx.Callable[["Transformation", SimplifyPolicy], bool],
]
ModePair = tx.Tuple[Kind, tx.Optional[int]]
_ModePair = ModePair  # back-compat alias for the private name others import

_ModeScalar = tx.Union[str, int, type, tx.Tuple[type, ...]]
ModeLike = tx.Union[
    None, _ModeScalar, ModePair, tx.Iterable[tx.Union[_ModeScalar, ModePair]]
]

SimplifyTable = tx.Dict[tx.Optional[ModePair], SimplifyPolicy]
SimplifyLike = tx.Union[
    None,
    bool,
    str,
    SimplifyPolicy,
    type,
    tx.Iterable[tx.Union[str, type]],
    tx.Mapping[
        tx.Optional[tx.Union[str, type, ModePair]],
        tx.Union[bool, str, SimplifyPolicy],
    ],
]


# ======================================================================
#                    S I M P L I F Y   P O L I C Y
# ======================================================================

_POLICY_RANK = {
    SimplifyPolicy.none: 0,
    SimplifyPolicy.analytic: 1,
    SimplifyPolicy.numeric: 2,
}

# `bagof` converter: resolves a `SimplifyPolicy` by value and (after
# lower-casing) by name.
_to_policy = get_converter(SimplifyPolicy)


def _safest(policies: tx.Iterable[SimplifyPolicy]) -> SimplifyPolicy:
    # The least aggressive (safest) policy. Never compare policies with `<`:
    # `SimplifyPolicy` is a `StrEnum`, so string order would rank
    # ``"analytic" < "none"``. Always go through `_POLICY_RANK`.
    return min(policies, key=_POLICY_RANK.__getitem__)


def _lower_policy(value: tx.Any) -> SimplifyPolicy:
    """Normalize a single policy value to a [`SimplifyPolicy`][]."""
    if value is True:
        return SimplifyPolicy.numeric
    if value is False or value is None:
        return SimplifyPolicy.none
    if isinstance(value, SimplifyPolicy):
        return value
    if isinstance(value, str):
        value = value.lower()
    try:
        return _to_policy(value)
    except ValueError as e:
        raise ValueError(
            f"invalid simplify policy {value!r}; expected one of "
            "False/'none', 'analytic', True/'numeric'"
        ) from e


# ======================================================================
#                    K I N D S   A N D   K E Y S
# ======================================================================
#
# A *kind* is what a key denotes: a hierarchy node (set membership, decided
# by `is_kind`, which sees through the wrappers via the checker registry), a
# concrete/field/wrapper class or tuple of classes (plain `isinstance`), or a
# callable `(t, compute) -> bool` extension hook. Name resolution and the
# checker registry live in `check`/`checkers`; this module only *lowers* the
# many accepted key spellings to a `(kind, ndim)` pair and resolves the
# per-leaf `simplify` policy. It never decides membership -- that is `is_kind`.


def _is_kind(k: tx.Any) -> bool:
    # A kind is a type, a tuple of types, or a callable.
    if isinstance(k, type):
        return True
    if isinstance(k, tuple):
        return all(isinstance(c, type) for c in k)
    return callable(k)


def _lower_key(key: tx.Any) -> ModePair:
    """Normalize any mode/simplify key into a `(kind, ndim)` pair.

    A string is a hierarchy NAME/SYMBOL (via `hierarchy.parseType`) or a
    registered field/wrapper alias (`check.KIND_ALIASES`). A type is kept
    as is -- `is_kind` later reads a hierarchy node as a *set* and any other
    class (a concrete transform, a field, a wrapper) by `isinstance`.
    """
    if isinstance(key, str):
        try:
            # (node, ndim); NAME case-insensitive
            return hierarchy.parseType(key)
        except ValueError:
            alias = KIND_ALIASES.get(key.lower())
            if alias is not None:
                return alias, None
            raise ValueError(
                f"unknown transformation kind {key!r}: not a hierarchy "
                f"name/symbol and not one of {sorted(KIND_ALIASES)}"
            ) from None
    if isinstance(key, type):
        # A hierarchy node (a set) or a concrete/field/wrapper class (matched
        # by `isinstance`); `is_kind` tells them apart. Both lower to the
        # class with no dimension.
        return key, None
    if (
        isinstance(key, tuple)
        and len(key) == 2
        and _is_kind(key[0])
        and isinstance(key[1], (int, type(None)))
        and not isinstance(key[1], bool)
    ):
        return key  # already a proper pair: lowering is idempotent
    if (
        isinstance(key, tuple)
        and key
        and all(isinstance(k, type) for k in key)
    ):
        return key, None  # tuple of classes -> isinstance
    if isinstance(key, int) and not isinstance(key, bool):
        return hierarchy.Transformation, key
    if callable(key):
        return key, None
    raise ValueError(
        f"invalid kind {key!r}; expected a type name, a hierarchy type, "
        "or a wrapper/field key"
    )


# ======================================================================
#                            M O D E S
# ======================================================================


def _lower_modes(mode: ModeLike) -> tx.List[ModePair]:
    """Convert any (list of) mode-like input into a list of `(kind, ndim)`
    pairs.

    `None` admits every kind (compose everything). `False` -- and an empty
    list -- admit no kind (compose nothing): the caller then simplifies leaves
    without composing them. Anything else lowers key by key.
    """
    if mode is None:
        return [(hierarchy.Transformation, None)]
    if mode is False:
        return []  # compose nothing (simplify-only)
    if isinstance(mode, (str, int, type)) or _is_proper_mode(mode):
        mode = [mode]
    return [m if _is_proper_mode(m) else _lower_key(m) for m in mode]


def _is_proper_mode(mode: tx.Any) -> bool:
    """A proper mode is a `(kind, ndim)` pair: `kind` is a hierarchy node, a
    class, a tuple of classes or a callable, and `ndim` is an int or None."""
    if not isinstance(mode, tuple):
        return False
    if len(mode) != 2:
        return False
    if not _is_kind(mode[0]):
        return False
    if isinstance(mode[1], bool) or not isinstance(mode[1], (int, type(None))):
        return False
    return True


def _mode_children(mode: ModePair) -> list:
    children = []
    seen = set()
    kind, ndim = mode
    subclasses = getattr(kind, "__subclasses__", lambda: [])
    for child in subclasses():
        if (child, ndim) not in seen:
            seen.add((child, ndim))
            children.append((child, ndim))
    return children


def _matches_mode(t: "Transformation", mode: ModePair) -> bool:
    kind, ndim = mode
    # Resolution always runs at analytic (`compute=False`): it never reads a
    # value to decide which mode admits a leaf or which simplify entry applies.
    if not is_kind(t, kind, compute=False):
        return False
    if ndim is None:
        return True
    # FIXME
    #   In many transforms, the ndim can be guessed from the content of the
    #   xform, even if the input/output spaces are not set (e.g. the shape of
    #   the matrix or the field). The current implementation is a stricter
    #   bound.
    if t.input is None or t.output is None:
        return False
    if t.input.axes is None or t.output.axes is None:
        return False
    return len(t.input.axes) == ndim and len(t.output.axes) == ndim


def _mode_admits(t: "Transformation", modes: tx.List[ModePair]) -> bool:
    return any(_matches_mode(t, m) for m in modes)


# ======================================================================
#                        S I M P L I F Y   T A B L E S
# ======================================================================


def _lower_simplify(value: SimplifyLike) -> SimplifyTable:
    """Normalize any accepted `simplify=` input into a simplify table: a
    ``dict`` mapping each lowered kind pair (and the `None` fallback) to a
    [`SimplifyPolicy`][].

    * `None` / `False` / `"none"` -> `{None: none}`.
    * `"analytic"` / a `SimplifyPolicy` -> `{None: <policy>}`.
    * `True` / `"numeric"` -> `{None: numeric}`.
    * a key (str or type) or an iterable of keys -> analytic on those kinds,
      `none` fallback (restrict-to-these).
    * a mapping `{key: policy}` -> the fallback is the `None` entry, or
      `analytic` when absent; other keys are lowered with `_lower_key`,
      values with `_lower_policy`; colliding lowered keys keep the safest.
    """
    # An already-lowered table (a Mapping) passes through (idempotent); any
    # Mapping is accepted (e.g. a MappingProxyType), not only `dict`.
    if isinstance(value, Mapping):
        return _lower_mapping(value)
    # bool / None first (bool is an int).
    if value is None or value is True or value is False:
        return {None: _lower_policy(value)}
    if isinstance(value, SimplifyPolicy):
        return {None: value}
    if isinstance(value, str):
        # A policy word (none/analytic/numeric) is a fallback; anything else
        # is a restrict-to-this-kind key.
        if value.lower() in ("none", "analytic", "numeric"):
            return {None: _lower_policy(value)}
        return {
            None: SimplifyPolicy.none,
            _lower_key(value): SimplifyPolicy.analytic,
        }
    if isinstance(value, type):
        return {
            None: SimplifyPolicy.none,
            _lower_key(value): SimplifyPolicy.analytic,
        }
    if _is_proper_mode(value):
        return {None: SimplifyPolicy.none, value: SimplifyPolicy.analytic}
    if _iterable(value):
        table: SimplifyTable = {None: SimplifyPolicy.none}
        for item in value:
            _add_entry(table, _lower_key(item), SimplifyPolicy.analytic)
        return table
    raise ValueError(f"invalid simplify policy: {value!r}")


def _lower_mapping(value: Mapping) -> SimplifyTable:
    fallback = SimplifyPolicy.analytic  # the `None`-absent default
    table: SimplifyTable = {}
    for key, policy in value.items():
        if key is None:
            fallback = _lower_policy(policy)
        else:
            _add_entry(table, _lower_key(key), _lower_policy(policy))
    table[None] = fallback
    return table


def _add_entry(
    table: SimplifyTable, pair: ModePair, policy: SimplifyPolicy
) -> None:
    # Colliding lowered keys keep the SAFEST policy (the same tie-break the
    # resolution rule uses for several matching entries).
    if pair in table:
        table[pair] = _safest((table[pair], policy))
    else:
        table[pair] = policy


def _iterable(value: tx.Any) -> bool:
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _resolve_simplify(
    t: "Transformation", table: SimplifyTable
) -> SimplifyPolicy:
    matched = [
        policy
        for pair, policy in table.items()
        if pair is not None and _matches_mode(t, pair)
    ]
    return _safest(matched) if matched else table[None]


def _table_is_noop(table: SimplifyTable) -> bool:
    return all(p is SimplifyPolicy.none for p in table.values())


def _table_needs_numeric(table: SimplifyTable) -> bool:
    return any(p is SimplifyPolicy.numeric for p in table.values())
