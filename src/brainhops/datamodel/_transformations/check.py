"""Dispatch a kind-membership question to a checker.

`is_kind(t, kind, compute=False)` is the single membership predicate
used by both `compute(mode)` (which kinds do we compose) and
`simplify(policy)` (how hard each leaf is looked at). It answers whether
`t`'s membership in `kind` is *established*: either via subclassing
(the `.register` promise, answered by `isinstance`), or established
from structure (`compute=False`, analytic) or values
(`compute=True`, numeric). Not-established is never refuted.

Two kinds of kind
-----------------
A `kind` is either

* a **set node** -- a class defined in [`hierarchy`][], such as
  `hierarchy.AffineTransformation`. It denotes a *set* of maps, so
  membership can be established beyond the declared type: a `Linear` whose
  matrix happens to be diagonal is a member of the diagonal set. Set nodes
  are the only kinds a checker may be registered against.

* a **class kind** -- any other class (or tuple of classes): a concrete
  transform (`Affine`), a field (`CartesianField`), a wrapper
  (`InverseAffine`), a container (`Sequence`). It denotes a *representation*,
  not a set, so membership is exactly `isinstance` and nothing establishes
  it beyond that. `"affine"` is the set; `Affine` is the class.

A set node is recognized structurally: it is a real (non-virtual) subclass
of [`hierarchy.TransformationBaseClass`][]. A concrete transform is only ever
a *virtual* subclass of the node it registered to, so it never passes.

Checkers
--------
A concrete or wrapper transform establishes membership in a set node beyond
its declared type through a *checker*, registered against its own class in
the shared registry (`registries.CHECKERS`) -- the same pattern as the
composer and converter registries. `@checker(SourceType, KindNode)` declares
"a `SourceType` may be established as a member of `KindNode`"; the function
`checker(t, compute) -> bool` decides it for a given instance. Used with a
single argument, `@checker(KindNode)`, the source is read from the first
parameter's type hint. The checker implementations live in `checkers`, next
to nothing but each other, and register at import time; this module only
holds the machinery.

Dispatch
--------
For a set-node query, declared membership (`isinstance`) settles it
affirmatively first. Otherwise every registered checker whose source is an
ancestor of `type(t)` and whose kind node is a subset of the query is
applicable, and their results are OR-ed: establishing `t in C` for a
`C subset of kind` proves `t in kind`, and a DAG needs the OR (a wide affine
is `Surjective` via the surjective checker, not via the invertible one).
When several sources register the same kind node, the one nearest `type(t)`
in the class hierarchy wins, mirroring ordinary method resolution -- so a
wrapper (e.g. `Inverse`) shadows the concrete base it also inherits from,
and a typed `InverseAffine` uses the recursing `Inverse` checker rather than
the matrix checker that would read (and materialize) it.
"""

# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel import hierarchy

# internals
from .registries import CHECKERS, CHECKERS_FASTMAP, KIND_ALIASES

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation  # noqa: F401

SetNode = tx.Type[hierarchy.TransformationBaseClass]
ClassKind = tx.Union[type, tx.Tuple[type, ...]]
Kind = tx.Union[SetNode, ClassKind]
Hook = tx.Callable[["Transformation", bool], bool]
"""An ad-hoc kind: a predicate `(t, compute) -> bool` used as a kind."""

KindLike = tx.Union[
    Kind,
    Hook,
    str,  # a NAME/SYMBOL or a registered alias
    int,  # a dimension (the whole set, at that dimension)
]
FamilyLike = tx.Union[
    hierarchy.TransformationFamily,
    tx.Tuple[KindLike, tx.Optional[int]],
    KindLike,
]


# --- Public API -------------------------------------------------------


def register_kind_alias(name: str, cls: Kind) -> None:
    """Name a class (or tuple of classes, or set node) as a kind key.

    Wrapper and field kinds have no set node -- their membership depends on
    their contents -- so a name that resolves to one is matched by
    `isinstance` (see [`is_kind`][]). A name that resolves to a set node
    (e.g. `"scaling"` -> the diagonal set) keeps set semantics.
    """
    KIND_ALIASES[name.lower()] = cls


@tx.overload
def checker(kind: tx.Union[SetNode, tx.Tuple[SetNode, ...]]) -> tx.Callable:
    """Register a checker whose source is read from its type hints."""
    ...


@tx.overload
def checker(source: tx.Type["Transformation"], kind: SetNode) -> tx.Callable:
    """Register a checker for an explicit source type."""
    ...


def checker(*args) -> tx.Callable:
    """Register a checker establishing `source` instances in the set `kind`.

    Used two-arg, like [`converter`][]: `@checker(SourceType, KindNode)` on
    a function `f(t, compute) -> bool`. Used one-arg, `@checker(KindNode)`,
    the source is the type of the function's first parameter.
    `compute=False` inspects structure only (analytic); `compute=True` reads
    values (numeric).

    `kind` may be a tuple of set nodes, to register the same function
    against several of them at once.
    """
    if len(args) == 2:
        source, kinds = args
    elif len(args) == 1:
        source, kinds = None, args[0]
    else:
        raise TypeError("checker() takes 1 or 2 positional arguments")

    if not isinstance(kinds, tuple):
        kinds = (kinds,)
    for kind in kinds:
        if not is_set_node(kind):
            raise TypeError(
                f"a checker may only be registered against a hierarchy set "
                f"node, not {kind!r}. A class kind is matched by isinstance "
                f"and needs no checker."
            )

    def _register(func: tx.Callable) -> tx.Callable:
        source_ = source or next(iter(tx.get_type_hints(func).values()))
        for kind in kinds:
            CHECKERS[(source_, kind)] = func
        CHECKERS_FASTMAP.clear()
        return func

    return _register


def is_set_node(kind: tx.Any) -> bool:
    """Whether `kind` denotes a *set* of maps rather than a representation.

    True for a class defined in [`hierarchy`][] (a real subclass of
    [`hierarchy.TransformationBaseClass`][]), False for a concrete
    transform, a field, a wrapper, a container, or a tuple of those. A
    concrete transform is only ever a *virtual* subclass of the node it
    registered to, and virtual registration does not touch the MRO, so the
    test separates the two cleanly.
    """
    if not isinstance(kind, type):
        return False
    return hierarchy.TransformationBaseClass in kind.__mro__


def is_kind(t: tx.Any, kind: KindLike, compute: bool = False) -> bool:
    """
    Whether `t`'s membership in `kind` is established at level `compute`.

    `compute=False` (analytic) inspects structure only and assumes
    invertibility from shape (a square matrix is presumed invertible,
    a wide one surjective, a tall one injective).

    `compute=True` (numeric) reads values and refines with rank.
    Not-established is not refuted.

    Parameters
    ----------
    t : Transformation
        The transformation whose membership is questioned.
    kind : kind-like
        A set node, a class kind (or tuple of class kinds), a string that
        names one -- a hierarchy NAME or SYMBOL (`"affine"`, `"SO(3)"`) or
        a registered alias (`"inverse"`, `"displacements"`) -- or an ad-hoc
        predicate `(t, compute) -> bool`.
    compute : bool, default=False
        Whether values may be read to establish membership.

    Returns
    -------
    bool
        Whether membership is established.
    """
    kind = lower_kind(kind)
    if is_set_node(kind):
        return _is_in_set(t, kind, bool(compute))
    if isinstance(kind, (type, tuple)):
        # A class kind denotes a representation, not a set: `isinstance`
        # is the whole answer.
        return isinstance(t, kind)
    if callable(kind):
        # An ad-hoc kind: a predicate written at the call site, given the
        # same `(t, compute)` signature a registered checker has.
        return bool(kind(t, bool(compute)))
    raise TypeError(f"invalid kind {kind!r}")


# --- Public helpers ---------------------------------------------------


def lower_kind(kind_like: KindLike) -> Kind:
    """
    Resolve a kind-like object to a kind.

    Parameters
    ----------
    kind_like : kind-like
        A set node, a class kind (or tuple of class kinds), or a string
        that names one.

    Returns
    -------
    kind : Kind
        A set node, or a class kind.

    Raises
    ------
    ValueError
        If `kind_like` is a string that does not name a known kind.
    """
    if isinstance(kind_like, str):
        alias = KIND_ALIASES.get(kind_like.lower())
        if alias is not None:
            return alias
        try:
            return hierarchy.TransformationFamily.parse(kind_like).kind
        except ValueError:
            raise ValueError(
                f"unknown transformation kind {kind_like!r}: not a hierarchy "
                f"name/symbol and not one of {sorted(KIND_ALIASES)}"
            ) from None
    return kind_like


def lower_family(family_like: FamilyLike) -> hierarchy.TransformationFamily:
    """
    Resolve a family-like object to a [`hierarchy.TransformationFamily`][].

    This is the single entry point that turns a user-written mode or
    simplify key into the `(kind, ndim)` pair the tables are keyed by.

    Parameters
    ----------
    family_like : family-like
        A [`hierarchy.TransformationFamily`][], a `(kind, ndim)` pair, a
        bare dimension, a kind, or a string that names a kind.

    Returns
    -------
    family : hierarchy.TransformationFamily
        The lowered family.

    Raises
    ------
    ValueError
        If the kind is a string that names nothing known, or if the input
        is not family-like at all.
    """
    # A `(kind, ndim)` pair names its dimension explicitly.
    if isinstance(family_like, tuple) and len(family_like) == 2:
        kind, ndim = family_like
        if ndim is None or (
            isinstance(ndim, int) and not isinstance(ndim, bool)
        ):
            return hierarchy.TransformationFamily.parse(
                lower_kind(kind), ndim
            )
    # A string is resolved through the alias table first, so that a name
    # the hierarchy does not know (a wrapper, a field, a friendlier
    # spelling) lowers like any other key. Otherwise it goes to
    # `TransformationFamily.parse` *whole*, so that a dimensioned symbol
    # such as `"SO(3)"` keeps its dimension -- `lower_kind` would drop it.
    if isinstance(family_like, str):
        alias = KIND_ALIASES.get(family_like.lower())
        if alias is not None:
            return hierarchy.TransformationFamily(alias, None)
        return hierarchy.TransformationFamily.parse(family_like)
    return hierarchy.TransformationFamily.parse(lower_kind(family_like))


# --- Private helpers --------------------------------------------------


def _is_in_set(t: "Transformation", node: SetNode, compute: bool) -> bool:
    """
    Declared membership (the `.register` promise) settles it affirmatively;
    otherwise a checker may *establish* membership in `node` through any
    sub-set it decides.

    See the module docstring for why the applicable checkers are OR-ed
    rather than reduced to a single one.
    """
    if isinstance(t, node):
        return True
    for kind, func in _source_checkers(type(t)):
        # Belonging to any subset of the target means belonging to the
        # target.
        if issubclass(kind, node) and func(t, compute):
            return True
    return False


def _source_checkers(src: type) -> tx.Tuple[tx.Tuple[SetNode, tx.Callable]]:
    # The (KindNode, checker) pairs that apply to `src`, cached per
    # concrete source type. A source applies when it is an ancestor of
    # `src`; for a given kind node, the source *nearest in the MRO* wins --
    # ordinary method resolution -- so a wrapper (e.g. `Inverse`) shadows
    # the concrete base it also inherits from, and a typed `InverseAffine`
    # uses the recursing `Inverse` checker rather than the matrix checker
    # that would read (and materialize) it. (MRO position, not a hierarchy
    # `distance`: a `Generic` subscription such as `Inverse[Affine]` inserts
    # an intermediate that would make `distance` rank the concrete base
    # nearer, yet the wrapper still precedes it in the MRO.)
    cached = CHECKERS_FASTMAP.get(src)
    if cached is not None:
        return cached

    mro = src.__mro__
    best: tx.Dict[SetNode, tx.Tuple[int, tx.Callable]] = {}
    for (source, kind), func in CHECKERS.items():
        if not issubclass(src, source):
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
