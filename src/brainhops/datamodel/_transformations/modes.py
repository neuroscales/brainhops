"""Kinds, keys, modes and the level-aware membership predicate.

`mode` (which kinds `compute()` composes) and `simplify` (how hard each leaf
is looked at) share one machinery here.

Keys are named, and names are the documented path:

* a transformation-set NAME, case-insensitively: ``"affine"``, ``"linear"``,
  ``"rigid"``, ``"rotation"``, ``"bijective"``, ``"injective"``,
  ``"surjective"``; a mathematical symbol: ``"Aff"``, ``"SO"``, ``"SO(3)"``
  (the ``(n)`` fixes the dimension);
* a wrapper or field key: ``"subspace"``, ``"inverse"``, ``"projection"``,
  ``"meta"``, ``"field"``, ``"displacementfield"``, ``"coordinatesfield"``,
  and ``"scaling"`` (the diagonal set);
* a hierarchy type (e.g. ``hierarchy.AffineTransformation``) is also
  accepted; a concrete transformation class (e.g. ``Affine``) is mapped to
  the hierarchy set it registered to, so ``Affine`` means the affine *set*
  (covering ``Linear``, ``Scaling``, ``Subspace(Affine)``, ``InverseAffine``,
  …), not ``isinstance(t, Affine)``;
* an ``int`` fixes only the dimension (every kind of that ``ndim``);
* a callable ``(t, policy) -> bool`` is an extension hook.

Membership is level-aware, on the ladder ``none < analytic < numeric``.
**Analytic invertibility is assumed from shape**: a square matrix
transformation is presumed invertible (hence bijective), a wide one
surjective, a tall one injective, without reading any value; ``numeric``
verifies with rank and may retract the assumption. Resolution (deciding
which mode admits a leaf, or which simplify entry applies) always runs at
``analytic`` and never reads a value.
"""

# stdlib
from collections.abc import Mapping
from functools import lru_cache

# dependencies
import typing_extensions as tx

# bagof
from bagof.converters import get_converter

# datamodel
from brainhops.datamodel import hierarchy
from brainhops.datamodel.enums import SimplifyPolicy

# internals
from . import registries

# typing
if tx.TYPE_CHECKING:
    # internals
    from .base import Transformation

# `mode` and `simplify` share one machinery. A *kind* is what a key denotes:
# a hierarchy node (set membership, decided by the level-aware predicate
# `is_member`, which sees through wrappers), a class or tuple of classes
# registered as a "class kind" (plain `isinstance`, for the meta/field
# wrappers), or a callable `(t, policy) -> bool` extension hook. `mode`
# selects which kinds `compute()` composes; `simplify` selects, per kind,
# how hard each leaf is looked at (the `SimplifyPolicy` ladder).
#
# This module keeps a tight import discipline (`hierarchy`, `enums`,
# `registries`, `bagof` only) so `base`, `inverse`, `meta` can import it at
# the top level without a cycle. Concrete classes reach it through
# `register_kind` at their own import time and through the `_is_member` hooks
# on their instances.

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

KINDS: tx.Dict[str, Kind] = {}  # name (lowercase) -> kind
_CLASS_KINDS: tx.Set[type] = (
    set()
)  # classes whose subclasses match by isinstance


def register_kind(name: str, kind: Kind) -> None:
    """Name a kind for policy / mode keys.

    A class (or tuple of classes) registered here also becomes
    `isinstance`-addressable as a key, together with its subclasses -- this
    is how the meta/field wrappers (`"inverse"`, `"subspace"`, `"field"`, …)
    are matched, since they have no hierarchy node.
    """
    KINDS[name.lower()] = kind
    for cls in kind if isinstance(kind, tuple) else (kind,):
        if (
            isinstance(cls, type)
            and hierarchy.TransformationBaseClass not in cls.__mro__
        ):
            _CLASS_KINDS.add(cls)


def _all_nodes() -> tx.FrozenSet[type]:
    return frozenset(hierarchy.NAMETOCLASS.values())


@lru_cache(maxsize=None)  # noqa: UP033
def _registered_node(cls: type) -> type:
    """The hierarchy set a concrete class registered to, found by scanning
    the nodes (no class->node registry). ABC virtual subclassing answers
    `issubclass`."""
    nodes = {n for n in _all_nodes() if issubclass(cls, n)}
    # The minimal (most specific) node: drop any node that has a strict
    # subset also in the set.
    minimal = [
        n
        for n in nodes
        if not any(m is not n and issubclass(m, n) for m in nodes)
    ]
    if minimal == [hierarchy.Transformation]:
        raise ValueError(
            f"{cls.__name__} has no hierarchy set; use a name such as "
            "'affine', or 'field'/'inverse'/'subspace'/'projection' for a "
            "field or wrapper"
        )
    if len(minimal) != 1:
        raise ValueError(
            f"{cls.__name__} is registered to several unrelated sets "
            f"{[m.__name__ for m in minimal]}; name the set you mean"
        )
    return minimal[0]


def _is_kind(k: tx.Any) -> bool:
    # A kind is a type, a tuple of types, or a callable.
    if isinstance(k, type):
        return True
    if isinstance(k, tuple):
        return all(isinstance(c, type) for c in k)
    return callable(k)


def _lower_key(key: tx.Any) -> ModePair:
    """Normalize any mode/simplify key into a `(kind, ndim)` pair."""
    if isinstance(key, str):
        try:
            # (node, ndim); NAME case-insensitive
            return hierarchy.parseType(key)
        except ValueError:
            if key.lower() in KINDS:
                return _lower_key(KINDS[key.lower()])
            raise ValueError(
                f"unknown transformation kind {key!r}: not a hierarchy "
                f"name/symbol and not one of {sorted(KINDS)}"
            ) from None
    if isinstance(key, type):
        if (
            hierarchy.TransformationBaseClass in key.__mro__
        ):  # a hierarchy node
            return key, None
        if any(issubclass(key, c) for c in _CLASS_KINDS):  # meta / field class
            return key, None
        base = registries.TRANSFORMATION
        if base is not None and issubclass(key, base):  # a concrete transform
            return _registered_node(key), None
        raise ValueError(f"invalid kind {key!r}")
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
    pairs. `None` admits every kind; an empty list is a no-op."""
    if mode is None:
        return [(hierarchy.Transformation, None)]
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
    if not is_member(t, kind, SimplifyPolicy.analytic):
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


# ======================================================================
#                        M E M B E R S H I P
# ======================================================================


def is_member(
    t: "Transformation",
    kind: Kind,
    policy: tx.Union[
        SimplifyPolicy, bool, str, None
    ] = SimplifyPolicy.analytic,
) -> bool:
    """Whether `t`'s membership in `kind` is *established* at inspection
    level `policy` (`none < analytic < numeric`). Not-established is not
    refuted. Resolution (mode/simplify) only ever asks at `analytic`.

    `kind` may be a hierarchy node, a class kind (a wrapper/field class or a
    tuple of classes, matched by `isinstance`), a callable `(t, policy) ->
    bool`, or a **key**: a name (e.g. `"affine"`, `"Aff"`, `"SO(3)"`,
    `"inverse"`), an `int` dimension, or a concrete transformation class
    (mapped to its hierarchy set). A key's `ndim` is ignored here -- this is
    a set-membership question, not a dimension gate.
    """
    policy = _lower_policy(policy)
    # Accept name / int keys and concrete-class keys by normalizing them to a
    # resolved kind (ndim dropped). A hierarchy node, a registered class kind
    # (or a subclass of one), a tuple, or a callable is already resolved.
    if isinstance(kind, str) or (
        isinstance(kind, int) and not isinstance(kind, bool)
    ):
        kind = _lower_key(kind)[0]
    elif (
        isinstance(kind, type)
        and hierarchy.TransformationBaseClass not in kind.__mro__
        and not any(issubclass(kind, c) for c in _CLASS_KINDS)
    ):
        kind = _lower_key(kind)[0]
    if (
        isinstance(kind, type)
        and hierarchy.TransformationBaseClass in kind.__mro__
    ):
        # A hierarchy set.
        if isinstance(t, kind):
            return True  # declared (the `.register` promise)
        if policy is SimplifyPolicy.none:
            return False
        return t._is_member(kind, policy)  # established from structure/values
    if isinstance(kind, (type, tuple)):
        return isinstance(t, kind)  # a class kind
    if callable(kind):
        return bool(kind(t, policy))
    raise TypeError(f"invalid kind {kind!r}")


# --- shared node tables (mathematical facts about the lattice) ---------

_INVERTIBLE_OF = {
    hierarchy.AffineTransformation: hierarchy.InvertibleAffineTransformation,
    hierarchy.LinearTransformation: hierarchy.InvertibleLinearTransformation,
    hierarchy.DiagonalTransformation: (
        hierarchy.InvertibleDiagonalTransformation
    ),
    hierarchy.MultiplicativeTransformation: (
        hierarchy.InvertibleMultiplicativeTransformation
    ),
    hierarchy.MatrixTransformation: hierarchy.InvertibleMatrixTransformation,
    hierarchy.Transformation: hierarchy.BijectiveTransformation,
}
_NONINVERTIBLE_OF = {v: k for k, v in _INVERTIBLE_OF.items()}

# Sets NOT closed under blockdiag(., I): the uniform-scale / conformal
# families (lifting a uniform scaling into a larger identity block breaks
# uniformity). Everything else is liftable, including Injective/Surjective.
_NOT_LIFTABLE = {
    hierarchy.MultiplicativeTransformation,
    hierarchy.InvertibleMultiplicativeTransformation,
    hierarchy.PositiveMultiplicativeTransformation,
    hierarchy.Dilation,
    hierarchy.PositiveDilation,
    hierarchy.ConformalEuclideanTransformation,
    hierarchy.SpecialConformalEuclideanTransformation,
    hierarchy.ConformalOrthogonalTransformation,
    hierarchy.SpecialConformalOrthogonalTransformation,
}


def _liftable(node: type) -> bool:
    return node not in _NOT_LIFTABLE


# Sets closed under composition with a coordinate permutation P (any parity).
_PERMUTATION_CLOSED = {
    hierarchy.Transformation,
    hierarchy.InjectiveTransformation,
    hierarchy.SurjectiveTransformation,
    hierarchy.BijectiveTransformation,
    hierarchy.Isomorphism,
    hierarchy.Diffeomorphism,
    hierarchy.MatrixTransformation,
    hierarchy.InvertibleMatrixTransformation,
    hierarchy.AffineTransformation,
    hierarchy.InvertibleAffineTransformation,
    hierarchy.ConformalEuclideanTransformation,
    hierarchy.EuclideanTransformation,
    hierarchy.LinearTransformation,
    hierarchy.InvertibleLinearTransformation,
    hierarchy.ConformalOrthogonalTransformation,
    hierarchy.OrthogonalTransformation,
    hierarchy.GeneralizedPermutation,
    hierarchy.SignedPermutation,
    hierarchy.Permutation,
}
# Additionally closed when P is EVEN (det P = +1).
_EVEN_PERMUTATION_CLOSED = _PERMUTATION_CLOSED | {
    hierarchy.VolumePreservingDiffeomorphism,
    hierarchy.PositiveDefiniteMatrixTransformation,
    hierarchy.PositiveAffineTransformation,
    hierarchy.SpecialAffineTransformation,
    hierarchy.SpecialConformalEuclideanTransformation,
    hierarchy.SpecialEuclideanTransformation,
    hierarchy.PositiveLinearTransformation,
    hierarchy.SpecialLinearTransformation,
    hierarchy.SpecialConformalOrthogonalTransformation,
    hierarchy.SpecialOrthogonalTransformation,
}


def _maximal(nodes: tx.Iterable[type]) -> list:
    # The maximal (largest) sets under inclusion: drop any node that is a
    # strict subset of another node in the set.
    nodes = list(nodes)
    return [
        n
        for n in nodes
        if not any(m is not n and issubclass(n, m) for m in nodes)
    ]


@lru_cache(maxsize=None)  # noqa: UP033
def _lift_targets(node: type) -> tuple:
    # `blockdiag(inner, I) ∈ node` iff `inner ∈ M` for some M in these
    # targets. If `node` is liftable it is its own target; else the maximal
    # liftable strict subnodes of `node`.
    if _liftable(node):
        return (node,)
    return tuple(
        _maximal(
            n
            for n in _all_nodes()
            if issubclass(n, node) and n is not node and _liftable(n)
        )
    )


@lru_cache(maxsize=None)  # noqa: UP033
def _permute_targets(node: type, even: bool) -> tuple:
    # `P ∘ blockdiag(...) ∈ node` (P a coordinate permutation of the given
    # parity) iff the lifted map ∈ M for some M in these targets.
    closed = _EVEN_PERMUTATION_CLOSED if even else _PERMUTATION_CLOSED
    if node in closed:
        return (node,)
    return tuple(
        _maximal(
            n
            for n in _all_nodes()
            if issubclass(n, node) and n is not node and n in closed
        )
    )


@lru_cache(maxsize=None)  # noqa: UP033
def _bijective_targets(node: type) -> tuple:
    # The inverse of `T` is in `node` iff `T ∈ M` for some M in these
    # targets: `node` itself when it is bijective (closed under inversion),
    # else the maximal bijective subnodes of `node`.
    if issubclass(node, hierarchy.BijectiveTransformation):
        return (node,)
    return tuple(
        _maximal(
            n
            for n in _all_nodes()
            if issubclass(n, node)
            and issubclass(n, hierarchy.BijectiveTransformation)
        )
    )


def _reindex_is_even(input_axes: tx.Any, output_axes: tx.Any) -> bool:
    """Parity of the coordinate permutation a reindexing subspace applies:
    position `input_axes[j]` maps to `output_axes[j]`. Returns True for an
    even permutation (det +1). A length or set mismatch is treated as odd
    (det-sign nodes are then dropped by the caller's `even=False` path).
    """
    if input_axes is None or output_axes is None:
        return True
    a, b = list(input_axes), list(output_axes)
    if len(a) != len(b) or sorted(a) != sorted(b):
        return False
    pos = {axis: j for j, axis in enumerate(a)}
    perm = [pos[b[j]] for j in range(len(a))]
    seen = [False] * len(perm)
    swaps = 0
    for i in range(len(perm)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        swaps += length - 1
    return swaps % 2 == 0
