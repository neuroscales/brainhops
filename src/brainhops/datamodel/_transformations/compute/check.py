"""Dispatcher for questions of kind membership.

A kind is a named family of transformations, such as translations,
rotations or affine maps. [`is_kind`][] answers questions such as "is
this transformation a translation?", and it is called as
`is_kind(t, kind, compute=False)`. The operations `compute` and
`simplify` both rely on it: `compute` uses it to decide which
transformations a mode allows it to compose, and `simplify` uses it to
decide how closely a policy inspects each transformation of a chain.

A transformation can be shown to belong to a kind in three ways: by its
declared type, which is tested with `isinstance`; by its structure,
without reading any values (the analytic level, `compute=False`); or by
its values, such as the entries of a matrix (the numeric level,
`compute=True`). A negative answer only means that membership could not
be shown. It never proves that the transformation lies outside the
kind.

## Two kinds of kind

A kind is one of two things. A kind node is a class of the kind
hierarchy, that is, a genuine subclass of
[`TransformationKind`][brainhops.datamodel.kinds.TransformationKind]
such as `kinds.Affine`. A kind node stands for a set of maps, so a
transformation can belong to it even when its declared type does not
say so: a `Linear` transformation whose matrix happens to be diagonal
belongs to the set of diagonal maps. Only kind nodes have checkers.

A class kind is a concrete transformation, field, wrapper or container
class, such as `Affine` or `Sequence`. A class kind stands for a
representation rather than a set, and membership in it is exactly
`isinstance`. For example, the string `"affine"` names the set of
affine maps, whereas `Affine` names the class that stores an affine
matrix.

## Checkers

A checker is a function that decides whether instances of one
transformation class belong to one kind node. It is registered with
the [`checker`][] decorator:

```python
def _(query: Affine, kind: type[kinds.Translation], compute: bool) -> bool
```

This checker decides, for a given `Affine`, whether it is a
translation. The transformation class (the source) and the kind node
are read from the first two type hints, or they are given explicitly
with `@checker(Source, Node)`. The checkers themselves are defined in
the `checkers` module, which registers them when it is imported.

## Dispatch

When `t` is an instance of the queried kind, the answer is true, and
for a class kind this `isinstance` test is the whole answer. Otherwise,
every applicable checker is asked, and the answer is true as soon as
one of them succeeds. A checker applies when two conditions hold.
First, its source must be the nearest base class of `type(t)` among the
sources registered for the same node, so that a checker for a wrapper
such as `Inverse` takes precedence over a checker for a concrete base
class of that wrapper. Second, its node must be a subset of the queried
kind, because showing that `t` belongs to a smaller set proves that it
belongs to the larger one. A checker receives the queried kind rather
than its own node, so that it can reason about the question that was
actually asked.

Several checkers must be asked because the kinds do not form a tree,
and one set can be reached through several smaller sets that do not
contain each other. For example, an `Affine` with a permutation matrix
reaches the orthogonal set through the permutation node, whereas one
with a rotation matrix reaches it through the special orthogonal node.

The applicable checkers are found by a
[`Function`][bagof.dispatchers.Function], in which each checker is
registered for the kinds above its node through
[`Super`][bagof.dispatchers.Super]. The dispatch of that library
normally picks a single winner, which would drop checkers, so this
module collects every candidate and combines the answers itself. The
selection of checkers depends only on `type(t)` and on the queried
kind, so it is memoized.
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
so a class kind is also a valid value of this type.
"""

Family: tx.TypeAlias = kinds.TransformationFamily
Key: tx.TypeAlias = tx.Tuple[tx.Type[Transformation], Kind]
Checker: tx.TypeAlias = tx.Callable[[Transformation, Kind, bool], bool]
"""A function that decides whether a transformation belongs to a kind.

Checkers let [`is_kind`][] recognize a membership that the declared
type does not show, for example that an `Affine` is in fact a
translation. Each checker is registered for one transformation class
and one kind node, and it is called as `checker(t, kind, compute)`.
The argument `kind` is the kind that was queried, which may be a larger
set than the node that the checker was registered for. The flag
`compute` states whether the checker may read values, such as matrix
entries, or must decide from the structure alone. A checker returns
true only when it can show that `t` belongs to `kind`.
"""

Selection: tx.TypeAlias = tx.Tuple[Checker, ...]
"""Checkers that answer one query, in the order in which they are asked."""

KindLike: tx.TypeAlias = tx.Union[
    Kind,
    str,  # a kind name, symbol or registered alias
]
FamilyLike: tx.TypeAlias = tx.Union[
    Family,
    tx.Tuple[KindLike, tx.Optional[int]],
    KindLike,
    int,  # a number of dimensions, admitting any kind at that dimension
]

KIND_ALIASES: tx.Dict[str, Kind] = {}
"""Kind names that the kind hierarchy does not define itself.

The keys are lower-case names, and each value is a kind node or a class
kind, which is matched by `isinstance`. The `checkers` module fills the
table when it is imported.
"""


# --- Machinery --------------------------------------------------------


class IsKind:
    """Membership predicate and registry of checkers.

    Calling the predicate answers a membership question. The call collects
    the applicable checkers with a [`Function`][], keeps for each kind node
    the checker whose source class is most specific, and returns true as
    soon as one of the kept checkers succeeds. The registry can also be
    looked up by `(source, node)` pair, with `key in is_kind` and
    `is_kind.get(key)`. The `checkers` module uses these lookups to detect
    clashing registrations.
    """

    def __init__(self) -> None:
        self._function = Function("is_kind")
        # The values are shims created by `_add`, and `get` returns the
        # implementation behind each shim.
        self._registry: tx.Dict[Key, tx.Any] = {}
        # The memo maps each `(type(x), kind)` pair to its selection, and it is
        # tagged with the ABC cache token under which the selections were
        # computed. Every registration resets the memo.
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

        # A new shim is created for every `(source, node)` pair. Separate shims
        # keep the registered methods distinct when one implementation serves
        # several nodes, and each shim carries the pair that `_select` uses
        # for grouping.
        def shim(x: tx.Any, kind: type, compute: bool) -> bool:
            # The library replaces these hints with the source and kind given
            # at registration, so the annotations below are never read. They
            # still use resolvable types, because the library cannot resolve
            # the forward-reference aliases of this module.
            return func(x, kind, compute)

        shim.__module__ = getattr(func, "__module__", shim.__module__)
        shim.__qualname__ = f"is_kind[{source.__name__} in {node.__name__}]"
        shim._impl = func  # type: ignore[attr-defined]
        shim._source = source  # type: ignore[attr-defined]
        shim._node = node  # type: ignore[attr-defined]

        # `type[Super[node]]` accepts a queried kind `K` exactly when the node
        # is a subset of `K`. Two checkers for the same source whose nodes do
        # not contain each other tie for a query about a kind that contains
        # both nodes. A tie would matter only for a dispatch that picks a
        # single winner, and here every candidate is asked.
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
        """Return whether `x` can be shown to belong to `kind`.

        At the analytic level, only the structure of `x` is inspected. A matrix
        transformation, for example, is presumed invertible when it is square,
        surjective when it is wide and injective when it is tall. The numeric
        level reads the values and refines these presumptions with the rank.

        Parameters
        ----------
        x : Transformation
            Transformation in question.
        kind : KindLike
            Kind node, class kind, or a string naming one: a kind name or
            symbol such as `"affine"` or `"SO(3)"`, or a registered alias such
            as `"inverse"`.
        compute : bool, default=False
            Whether values, such as matrix entries, may be read.

        Returns
        -------
        bool
            Whether membership could be shown. False does not prove that `x`
            lies outside `kind`.
        """
        kind = normalize_kind(kind)
        if isinstance(x, kind):
            return True
        if not kinds.is_transformation_set(kind):
            # A class kind is a representation, so isinstance is the answer.
            return False
        compute = bool(compute)

        # The selection depends only on types and is memoized. The checkers
        # themselves may read values, so they are asked again on every call.
        # When no checker applies, membership cannot be shown.
        for impl in self._selection(x, kind, compute):
            if impl(x, kind, compute):
                return True
        return False

    def _selection(
        self, x: Transformation, kind: Kind, compute: bool
    ) -> Selection:
        """Return the memoized checker implementations for `(type(x), kind)`.

        Enumerating the candidates is expensive, because the library tests
        every method against the argument values without caching. The
        selection depends on `(type(x), kind)` alone, since whether a checker
        applies depends only on `isinstance(x, source)` and
        `issubclass(node, kind)`. The memo is reset by a new checker, and by a
        new virtual subclass in any ABC hierarchy through
        [`abc.get_cache_token`][abc.get_cache_token].

        No lock is needed, because the memo is a `(token, dict)` pair that is
        replaced as a whole, and two threads that miss the memo at the same
        time compute the same selection.
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
        # For each node, only the checker whose source comes first on the MRO
        # of `type(x)` is kept, so that a checker for a wrapper takes
        # precedence over a checker for its concrete base class. A subscripted
        # generic such as `Inverse[Affine]` inserts a real intermediate class,
        # which the MRO already places first.
        mro = type(x).__mro__

        def source_rank(source: type) -> int:
            try:
                return mro.index(source)
            except ValueError:
                return len(mro)  # sources absent from the MRO rank last

        best: tx.Dict[Kind, tx.Tuple[int, tx.Any]] = {}
        for method in self._function.candidates(x, kind, compute):
            shim = method.function
            rank = source_rank(shim._source)
            current = best.get(shim._node)
            if current is None or rank < current[0]:
                best[shim._node] = (rank, shim)

        # One implementation may be selected for several nodes. It receives the
        # queried kind rather than its node, so asking it once is enough.
        selection: tx.List[Checker] = []
        seen: tx.Set[int] = set()
        for _rank, shim in best.values():
            impl = shim._impl
            if id(impl) in seen:
                continue
            seen.add(id(impl))
            selection.append(impl)
        return tuple(selection)


# --- Public API -------------------------------------------------------

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
    belongs to the family when it can be shown to belong to the kind
    analytically, without reading values, and when its input and output
    systems do not contradict the number of dimensions.

    Parameters
    ----------
    x : Transformation
        Transformation in question.
    family : FamilyLike
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
    # `space` holds axes that state nothing except their number, so an
    # endpoint can clash with it only through its number of axes. An endpoint
    # that does not state its number of axes cannot contradict the dimension.
    # TODO: when the spaces are unset, the number of dimensions could often
    # be inferred from the content, such as the shape of a matrix or field.
    space = AxisList([...]).expand(family.ndim)
    return all(get_axes(e).compatible_with(space) for e in (x.input, x.output))


# --- Public helpers ---------------------------------------------------


def register_kind_alias(
    name: str, cls: tx.Union[Kind, tx.Tuple[Kind, ...]]
) -> None:
    """Register a name for a class kind or a kind node.

    The name is stored in lower case. Wrapper and field kinds have no node,
    because their membership depends on their contents, so a name for such
    a class is matched by `isinstance` (see [`is_kind`][]). A name that
    resolves to a node, such as `"scaling"` for the diagonal set, is still
    answered as a question about a set, with checkers. When a tuple is
    given, each class is assigned to the same name in turn, so only the
    last one is kept.
    """
    if not isinstance(cls, tuple):
        cls = (cls,)
    for a_cls in cls:
        KIND_ALIASES[name.lower()] = a_cls


def normalize_kind(kind_like: KindLike) -> Kind:
    """Resolve a kind-like value to a kind.

    Parameters
    ----------
    kind_like : KindLike
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
        # isinstance would accept a tuple of classes, so a tuple must be
        # refused explicitly, because a kind is a single class. A caller who
        # needs several kinds can query their shared base, or ask once for each
        # kind.
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
    family_like : FamilyLike
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
    # A `(kind, ndim)` pair is the only tuple that is read as a family, and a
    # tuple of kinds is refused.
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
