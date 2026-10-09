"""Dispatcher for questions of kind membership.

[`is_kind`][] answers `is_kind(t, kind, compute=False)`, the predicate
that `compute` and `simplify` share to decide which kinds a mode
composes and how closely a policy inspects each leaf. Membership is
established by declaration (`isinstance`), by structure (the analytic
level, `compute=False`) or by values (the numeric level,
`compute=True`). A membership that is not established is never a
refutation.

## Two kinds of kind

A kind node is a genuine subclass of
[`TransformationKind`][brainhops.datamodel.kinds.TransformationKind],
such as `kinds.Affine`. A node denotes a set of maps, and a
transformation can be shown to belong to it beyond its declared type,
as a diagonal `Linear` belongs to the diagonal set. Only nodes have
checkers. A class kind is a concrete transformation, field, wrapper or
container class, such as `Affine` or `Sequence`. It denotes a
representation, and membership in it is exactly `isinstance`: `"affine"`
names a set, whereas `Affine` names a class.

## Checkers

A transformation class establishes membership in a node through a
checker registered against it:

```python
def _(query: Affine, kind: type[kinds.Translation], compute: bool) -> bool
```

This checker decides, for each instance, whether an `Affine` is a
translation. The `(source, node)` pair is read from the first two type
hints, or given explicitly with `@checker(Source, Node)`. The checkers
themselves live in `checkers`, which registers them on import.

## Dispatch

An `isinstance` check settles the question affirmatively, and is the
whole answer for a class kind. Otherwise, every applicable checker is
asked and the answers are combined with a logical OR. A checker applies
when its source is the nearest supertype of `type(t)` registered for
its node, so that a wrapper such as `Inverse` shadows its concrete
base, and when its node is a subset of the queried kind. A checker
receives the queried kind rather than its own node, so that it can
reason about the question actually asked.

The OR is necessary because the lattice of kinds is not a tree. An
`Affine` with a permutation matrix reaches the orthogonal set through
the permutation node, whereas one with a rotation matrix reaches it
through the special orthogonal node.

Candidates are enumerated by a [`Function`][bagof.dispatchers.Function],
in which the kind is matched contravariantly through
[`Super`][bagof.dispatchers.Super]. Since the single-winner dispatch of
that library would drop checkers, this module performs the reduction
itself and memoizes the selection, which depends only on
`(type(t), kind)`.
"""

__all__ = [
    "Checker",
    "Family",
    "FamilyLike",
    "Kind",
    "KindLike",
    "checker",
    "get_checker",
    "is_family",
    "is_kind",
    "normalize_family",
    "normalize_kind",
    "register_kind_alias",
]

import abc

import typing_extensions as tx
from bagof.dispatchers import Function, Super

from brainhops.datamodel import kinds
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.systems import AxisList

if tx.TYPE_CHECKING:
    from ..base import Transformation as _Transformation

Transformation: tx.TypeAlias = "_Transformation"
Kind: tx.TypeAlias = tx.Type[kinds.TransformationKind]
"""A node of the kind hierarchy, or a class kind.

Every concrete transformation class is registered into the hierarchy,
so a class kind is also a valid value.
"""

Family: tx.TypeAlias = kinds.TransformationFamily
Key: tx.TypeAlias = tx.Tuple[tx.Type[Transformation], Kind]
Checker: tx.TypeAlias = tx.Callable[[Transformation, Kind, bool], bool]
"""A checker, called as `(t, kind, compute)`.

A checker decides whether `t` is established as a member of a kind
node, analytically or numerically according to `compute`. It is keyed
by a source type and a node, and receives the queried kind, which may
be a superset of that node.
"""

Selection: tx.TypeAlias = tx.Tuple[Checker, ...]
"""Checker implementations combined by one query, in order."""

KindLike: tx.TypeAlias = tx.Union[
    Kind,
    str,  # a kind name, symbol or registered alias
]
FamilyLike: tx.TypeAlias = tx.Union[
    Family,
    tx.Tuple[KindLike, tx.Optional[int]],
    KindLike,
    int,  # a dimension, for the whole set at that dimension
]

KIND_ALIASES: tx.Dict[str, Kind] = {}
"""Additional kind names, not covered by the kind hierarchy.

The keys are lower-case names, and each value is a kind node or a class
kind matched by `isinstance`. The table is populated by `checkers` on
import.
"""


class IsKind:
    """Membership predicate and registry of checkers.

    The predicate enumerates the applicable checkers with a [`Function`][],
    keeps the most specific source for each node, and combines the answers
    with a logical OR. The registry can also be queried by `(source, node)`
    key, with `key in is_kind` and `is_kind.get(key)`, which `checkers`
    uses to detect clashes.
    """

    def __init__(self) -> None:
        self._function = Function("is_kind")
        # Values are shims (see `_add`); `get` recovers the implementation.
        self._registry: tx.Dict[Key, tx.Any] = {}
        # Selections per `(type(x), kind)`, tagged with the ABC cache token
        # under which they were computed. Every registration resets the memo.
        self._memo: tx.Optional[tx.Tuple[object, tx.Dict[Key, Selection]]]
        self._memo = None

    def register(self, *args: tx.Any) -> tx.Any:
        """Register a checker, or return a decorator that registers one.

        In the bare form, `@is_kind.register`, the `(source, node)` pair is
        read from the first two type hints, and the second hint is unwrapped
        from `type[...]`. The explicit form, `is_kind.register(Source, Node)`,
        serves generated checkers that have no hints. The original function is
        returned, so decorators can be stacked.

        Raises
        ------
        TypeError
            If the arguments match neither form, or if the target is a class
            kind rather than a node.
        """
        if len(args) == 2 and all(isinstance(a, type) for a in args):
            source, node = args

            def decorator(func: Checker) -> Checker:
                self._add(source, node, func)
                return func

            return decorator
        if (
            len(args) == 1
            and callable(args[0])
            and not isinstance(args[0], type)
        ):
            func = args[0]
            source, node = self._key_from_func(func)
            self._add(source, node, func)
            return func
        raise TypeError(
            "a checker is registered from a function, or against a "
            f"(source, node) pair of types, not {args!r}"
        )

    @staticmethod
    def _key_from_func(func: Checker) -> Key:
        hints = tx.get_type_hints(func)
        inp, out, *_ = hints.values()
        if tx.get_origin(out) is type:
            out = tx.get_args(out)[0]
        return inp, out

    def _add(self, source: type, node: Kind, func: Checker) -> None:
        if not kinds.is_transformation_set(node):
            raise TypeError(
                f"a checker may only be registered against a kind node, not "
                f"{node!r}. A class kind is matched by isinstance and needs "
                f"no checker."
            )

        # A fresh shim per `(source, node)` keeps the methods distinct when one
        # implementation serves many nodes, and carries the pair used for
        # grouping.
        def shim(x: tx.Any, kind: type, compute: bool) -> bool:
            # The library overlays the source and kind hints at registration,
            # so these annotations are never read. They use resolvable types
            # because the library cannot resolve this module's
            # forward-reference aliases.
            return func(x, kind, compute)

        shim.__module__ = getattr(func, "__module__", shim.__module__)
        shim.__qualname__ = f"is_kind[{source.__name__} in {node.__name__}]"
        shim._impl = func  # type: ignore[attr-defined]
        shim._source = source  # type: ignore[attr-defined]
        shim._node = node  # type: ignore[attr-defined]

        # `type[Super[node]]` accepts a queried kind `K` exactly when the node
        # is a subset of `K`. Two checkers on one source with incomparable
        # nodes tie for a query above both, which matters only for
        # single-winner dispatch: here every candidate is asked.
        self._function.register((source, tx.Type[Super[node]]))(shim)
        self._registry[(source, node)] = shim
        self._memo = None  # a new checker can change any selection

    def __contains__(self, key: Key) -> bool:
        return key in self._registry

    def get(self, key: Key, default: tx.Any = None) -> tx.Optional[Checker]:
        """Return the checker registered for `(source, node)`, or `default`."""
        shim = self._registry.get(key)
        return default if shim is None else shim._impl

    def __call__(
        self, x: Transformation, kind: KindLike, compute: bool = False
    ) -> bool:
        """Return whether the membership of `x` in `kind` is established.

        At the analytic level, only the structure is inspected. A matrix
        transformation, for example, is presumed invertible when it is square,
        surjective when it is wide and injective when it is tall. The numeric
        level reads the values and refines these presumptions with the rank.

        Parameters
        ----------
        x
            Transformation in question.
        kind
            Kind node, class kind, or a string naming one: a kind name or
            symbol such as `"affine"` or `"SO(3)"`, or a registered alias such
            as `"inverse"`.
        compute
            Whether values may be read.

        Returns
        -------
        bool
            Whether membership is established.
        """
        kind = normalize_kind(kind)
        if isinstance(x, kind):
            return True
        if not kinds.is_transformation_set(kind):
            # A class kind is a representation: isinstance is the answer.
            return False
        compute = bool(compute)

        # The selection depends only on types and is memoized, but checkers
        # read values, so the disjunction runs on every call. No applicable
        # checker means not established.
        for impl in self._selection(x, kind, compute):
            if impl(x, kind, compute):
                return True
        return False

    def _selection(
        self, x: Transformation, kind: Kind, compute: bool
    ) -> Selection:
        """Return the memoized checker implementations for `(type(x), kind)`.

        Enumeration is expensive, because the library tests every method
        against the argument values without caching. The selection depends on
        `(type(x), kind)` alone, since applicability reduces to
        `isinstance(x, source)` and `issubclass(node, kind)`. The memo is reset
        by a new checker, and by a new virtual subclass in any ABC hierarchy
        through [`abc.get_cache_token`][abc.get_cache_token].

        No lock is needed: the memo is a `(token, dict)` pair replaced as a
        whole, and concurrent misses compute the same selection.
        """
        token = abc.get_cache_token()
        memo = self._memo
        if memo is None or memo[0] != token:
            memo = self._memo = (token, {})
        key = (type(x), kind)
        selection = memo[1].get(key)
        if selection is None:
            selection = memo[1][key] = self._select(x, kind, compute)
        return selection

    def _select(
        self, x: Transformation, kind: Kind, compute: bool
    ) -> Selection:
        # Keep, for each node, the nearest source on the MRO of `type(x)`, so
        # that a wrapper override shadows its concrete base. A subscripted
        # generic such as `Inverse[Affine]` inserts a real intermediate class,
        # which the MRO already orders first.
        mro = type(x).__mro__

        def source_rank(source: type) -> int:
            try:
                return mro.index(source)
            except ValueError:
                return len(mro)  # ancestors off the MRO rank last

        best: tx.Dict[Kind, tx.Tuple[int, tx.Any]] = {}
        for method in self._function.candidates(x, kind, compute):
            shim = method.function
            rank = source_rank(shim._source)
            current = best.get(shim._node)
            if current is None or rank < current[0]:
                best[shim._node] = (rank, shim)

        # One implementation may win several nodes. It reads the queried kind,
        # not its node, so it is kept once.
        selection: tx.List[Checker] = []
        seen: tx.Set[int] = set()
        for _rank, shim in best.values():
            impl = shim._impl
            if id(impl) in seen:
                continue
            seen.add(id(impl))
            selection.append(impl)
        return tuple(selection)


is_kind: IsKind = IsKind()
"""Membership predicate shared by the transformation operations."""

checker = is_kind.register
"""Decorator that registers a checker with [`is_kind`][]."""

get_checker = is_kind.get
"""Return the checker registered for a transformation type and kind node."""


def is_family(x: Transformation, family: FamilyLike) -> bool:
    """Return whether a transformation belongs to a family.

    A [`TransformationFamily`][brainhops.datamodel.kinds.TransformationFamily]
    pairs a kind with an optional number of dimensions. A transformation
    belongs to the family when its membership in the kind is established
    analytically, without reading values, and when its endpoints do not
    contradict the number of dimensions.

    Parameters
    ----------
    x
        Transformation in question.
    family
        Family, `(kind, ndim)` pair, bare number of dimensions, kind, or
        string naming a kind.

    Returns
    -------
    bool
        Whether the family admits `x`.
    """
    family = normalize_family(family)
    if family.kind is not None and not is_kind(x, family.kind, compute=False):
        return False
    if family.ndim is None:
        return True
    # `space` holds only unknown axes, so only axis counts can clash. An
    # endpoint that does not state its axis count cannot contradict the
    # dimension. TODO: when the spaces are unset, ndim could often be inferred
    # from the content, such as the matrix or field shape.
    space = AxisList([...]).expand(family.ndim)
    return all(get_axes(e).compatible_with(space) for e in (x.input, x.output))


def register_kind_alias(
    name: str, cls: tx.Union[Kind, tx.Tuple[Kind, ...]]
) -> None:
    """Register a name for a class kind or a kind node.

    The name is stored in lower case. Wrapper and field kinds have no node,
    because their membership depends on their contents, so a name for such
    a class is matched by `isinstance` (see [`is_kind`][]). A name that
    resolves to a node, such as `"scaling"` for the diagonal set, keeps the
    semantics of a set. When a tuple is given, each class is assigned to
    the same name in turn, so only the last one is kept.
    """
    if not isinstance(cls, tuple):
        cls = (cls,)
    for a_cls in cls:
        KIND_ALIASES[name.lower()] = a_cls


def normalize_kind(kind_like: KindLike) -> Kind:
    """Resolve a kind-like value to a kind.

    Parameters
    ----------
    kind_like
        Kind node, class kind, or a string naming one.

    Returns
    -------
    Kind
        The kind node or class kind.

    Raises
    ------
    ValueError
        If a string names no known kind.
    TypeError
        If the value is neither a string nor a class, for example a tuple
        of kinds.
    """
    if isinstance(kind_like, str):
        alias = KIND_ALIASES.get(kind_like.lower())
        if alias is not None:
            return alias
        try:
            return kinds.TransformationKind.parse(kind_like)
        except ValueError:
            raise ValueError(
                f"unknown transformation kind {kind_like!r}: not a kind "
                f"name/symbol and not one of {sorted(KIND_ALIASES)}"
            ) from None
    if not isinstance(kind_like, type):
        # isinstance would accept a tuple of classes, so the refusal must be
        # explicit: one kind is one class. Query the shared base, or ask twice.
        raise TypeError(
            f"not a transformation kind: {kind_like!r}. A kind is one class "
            f"-- a node of `kinds`, or a concrete transformation class -- or "
            f"a string naming one."
        )
    return kind_like


def normalize_family(family_like: FamilyLike) -> Family:
    """Resolve a family-like value to a family.

    This is the single entry point that turns a mode or simplification key
    written by a user into the
    [`TransformationFamily`][brainhops.datamodel.kinds.TransformationFamily]
    that tables are keyed by.

    Parameters
    ----------
    family_like
        Family, `(kind, ndim)` pair, bare number of dimensions, kind, or
        string naming a kind.

    Returns
    -------
    Family
        The normalized family.

    Raises
    ------
    ValueError
        If a string names no known kind, or if the value is not family-like.
    """
    # A `(kind, ndim)` pair is the only tuple read as a family; a tuple of
    # kinds is not a kind.
    kind_like, ndims = family_like, ()
    if isinstance(family_like, tuple):
        if not kinds.is_family_tuple(family_like):
            raise ValueError(
                f"Invalid (kind, ndim) pair: {family_like!r}. A family pairs "
                f"one kind with one dimension (or None)."
            )
        kind_like, *ndims = family_like
    # Strings go through the alias table first, so that names missing from the
    # hierarchy normalize like the others.
    if isinstance(kind_like, str):
        alias = KIND_ALIASES.get(kind_like.lower())
        if alias is not None:
            return kinds.TransformationFamily(alias, *ndims)
    return kinds.TransformationFamily.parse(kind_like, *ndims)
