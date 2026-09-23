"""Dispatch a kind-membership question to a checker.

`is_kind(t, kind, compute=False)` is the single membership predicate used by
both `mode` (which kinds `compute()` composes) and `simplify` (how hard each
leaf is looked at). It answers whether `t`'s membership in `kind` is
*established*: declared (the `.register` promise, answered by `isinstance`),
or established from structure (`compute=False`, analytic) or values
(`compute=True`, numeric). Not-established is never refuted.

A concrete or wrapper transform establishes membership beyond its declared
type through a *checker*, registered against its own class in the shared
registry (`registries.CHECKERS`) -- the same pattern as the composer and
converter registries. `@checker(SourceType, KindNode)` declares "a
`SourceType` may be established as a member of `KindNode`"; the function
`checker(t, compute) -> bool` decides it for a given instance. The checker
implementations live in `checkers`, next to nothing but each other, and
register at import time; this module only holds the machinery.

Dispatch
--------
For a hierarchy-set query, declared membership (`isinstance`) settles it
first. Otherwise every registered checker whose source is an ancestor of
`type(t)` and whose kind node is a subset of the query is applicable, and
their results are OR-ed: establishing `t in C` for a `C subset of kind`
proves `t in kind`, and a DAG needs the OR (a wide affine is `Surjective`
via the surjective checker, not via the invertible one). When several
sources register the same kind node, the one nearest `type(t)` in the class
hierarchy wins, mirroring ordinary method resolution -- so a wrapper (e.g.
`Inverse`) shadows the concrete base it also inherits from, and a typed
`InverseAffine` uses the recursing `Inverse` checker rather than the matrix
checker that would read (and materialize) it.
"""

# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel import hierarchy

# internals
from .registries import CHECKERS, CHECKERS_FASTMAP, distance

# name (lower-case) -> a concrete/field/wrapper class (or tuple of classes),
# or a hierarchy node, for keys that `parseType` does not resolve
# (`"inverse"`, `"subspace"`, `"displacements"`, `"scaling"`, ...). Populated
# by `checkers` at import time.
KIND_ALIASES: tx.Dict[str, tx.Any] = {}


def register_kind_alias(name: str, cls: tx.Any) -> None:
    """Name a class (or tuple / hierarchy node) as a kind key.

    Wrapper and field kinds have no hierarchy node -- their membership
    depends on their contents -- so a name that resolves to one is matched by
    `isinstance` (see [`is_kind`][]). A name that resolves to a hierarchy
    node (e.g. `"scaling"` -> the diagonal set) keeps set semantics.
    """
    KIND_ALIASES[name.lower()] = cls


def checker(source: type, kind: type) -> tx.Callable:
    """Register a checker establishing `source` instances in the set `kind`.

    Used two-arg, like [`converter`][]: ``@checker(SourceType, KindNode)`` on
    a function ``f(t, compute) -> bool``. ``compute=False`` inspects structure
    only (analytic); ``compute=True`` reads values (numeric).
    """

    def _register(func: tx.Callable) -> tx.Callable:
        CHECKERS[(source, kind)] = func
        CHECKERS_FASTMAP.clear()
        return func

    return _register


def _source_checkers(src: type) -> tx.Tuple:
    # The (KindNode, checker) pairs that apply to `src`, cached per concrete
    # source type. A source applies when it is an ancestor of `src` (finite
    # `distance`); for a given kind node, the source *nearest in the MRO*
    # wins -- ordinary method resolution -- so a wrapper (e.g. `Inverse`)
    # shadows the concrete base it also inherits from, and a typed
    # `InverseAffine` uses the recursing `Inverse` checker rather than the
    # matrix checker that would read (and materialize) it. (MRO position,
    # not `distance`: a `Generic` subscription such as `Inverse[Affine]`
    # inserts an intermediate that would make `distance` rank the concrete
    # base nearer, yet the wrapper still precedes it in the MRO.)
    cached = CHECKERS_FASTMAP.get(src)
    if cached is not None:
        return cached
    mro = src.__mro__
    best: tx.Dict[type, tx.Tuple[int, tx.Callable]] = {}
    for (source, kind), func in CHECKERS.items():
        if distance(src, source) == float("inf"):
            continue  # not an ancestor of `src`
        try:
            mro_rank = mro.index(source)
        except ValueError:
            mro_rank = len(mro)  # ancestor not on the MRO (virtual base)
        current = best.get(kind)
        if current is None or mro_rank < current[0]:
            best[kind] = (mro_rank, func)
    pairs = tuple((kind, func) for kind, (_r, func) in best.items())
    CHECKERS_FASTMAP[src] = pairs
    return pairs


def _resolve_kind(kind: tx.Any) -> tx.Any:
    # Resolve a key to a hierarchy node, a concrete/field/wrapper class (or
    # tuple), or a callable hook. A string is a NAME/SYMBOL (via `parseType`)
    # or a registered alias; a bare int is a dimension (the whole set).
    if isinstance(kind, str):
        try:
            node, _ = hierarchy.parseType(kind)
            return node
        except ValueError:
            alias = KIND_ALIASES.get(kind.lower())
            if alias is not None:
                return alias
            raise ValueError(
                f"unknown transformation kind {kind!r}: not a hierarchy "
                f"name/symbol and not one of {sorted(KIND_ALIASES)}"
            ) from None
    if isinstance(kind, bool):
        raise TypeError(f"invalid kind {kind!r}")
    if isinstance(kind, int):
        return hierarchy.Transformation
    return kind


def is_kind(t: tx.Any, kind: tx.Any, compute: bool = False) -> bool:
    """Whether `t`'s membership in `kind` is established at level `compute`.

    ``compute=False`` (analytic) inspects structure only -- shapes, `None`
    parameters, axis lists -- and assumes invertibility from shape (a square
    matrix is presumed invertible, a wide one surjective, a tall one
    injective). ``compute=True`` (numeric) reads values and refines with
    rank. Not-established is not refuted.

    `kind` may be a hierarchy set (a node, or a NAME/SYMBOL string such as
    ``"affine"``, ``"SO(3)"``), a concrete/field/wrapper class or key
    (matched by `isinstance`: ``"inverse"``, ``"displacements"``, `Affine`),
    a tuple of classes, or a callable ``(t, compute) -> bool`` hook.
    """
    kind = _resolve_kind(kind)
    if isinstance(kind, tuple):
        return isinstance(t, kind)
    if isinstance(kind, type):
        if hierarchy.TransformationBaseClass in kind.__mro__:  # a set
            return _is_kind_node(t, kind, bool(compute))
        return isinstance(t, kind)  # a concrete/field/wrapper class kind
    if callable(kind):
        return bool(kind(t, compute))
    raise TypeError(f"invalid kind {kind!r}")


def _is_kind_node(t: tx.Any, node: type, compute: bool) -> bool:
    # Declared membership (the `.register` promise) settles it first; then a
    # checker may *establish* membership in `node` through any sub-set it
    # decides. See the module docstring for why the applicable checkers are
    # OR-ed rather than reduced to a single one.
    if isinstance(t, node):
        return True
    for kind, func in _source_checkers(type(t)):
        if issubclass(kind, node) and func(t, compute):
            return True
    return False
