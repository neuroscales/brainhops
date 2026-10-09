"""Dispatch a kind-membership question to a checker.

`is_kind(t, kind, compute=False)` is the single membership predicate used by
both `compute(mode)` (which kinds do we compose) and `simplify(policy)` (how
hard each leaf is looked at). It answers whether `t`'s membership in `kind`
is *established*: either by declaration (the `.register` promise, answered
by `isinstance`), or from structure (`compute=False`, analytic) or from
values (`compute=True`, numeric). Not-established is never refuted.

Two kinds of kind
-----------------
A `kind` is either

* a **kind node** -- a class defined in [`kinds`][], such as `kinds.Affine`.
  It denotes a *set* of maps, so membership can be established beyond the
  declared type: a `Linear` whose matrix happens to be diagonal is a member
  of the diagonal set. Kind nodes are the only kinds a checker may be
  registered against.

* a **class kind** -- a concrete transform (`Affine`), a field
  (`CartesianField`), a wrapper (`InverseAffine`), a container (`Sequence`).
  It denotes a *representation*, not a set, so membership is exactly
  `isinstance` and nothing establishes it beyond that. `"affine"` is the
  set; `Affine` is the class.

Either way a kind is *one* class, never several: one kind is one node of the
hierarchy, and a concrete transform is a node of it too -- it is registered
into it (`@kinds.Affine.register`), so `issubclass(Affine,
kinds.TransformationKind)` holds. What separates the two is
[`kinds.is_kind_node`][]: a kind node is a *real* (non-virtual) subclass of
[`kinds.TransformationKind`][], while a concrete transform is only ever a
virtual subclass of the node it registered to. What the separation decides is
whether membership can be established beyond `isinstance`.

Checkers
--------
A concrete or wrapper transform establishes membership in a kind node beyond
its declared type through a *checker*, registered against its own class in
[`is_kind`][]. `@checker` on a function

    def _(query: Affine, kind: type[kinds.Translation], compute: bool) -> bool

declares "an `Affine` may be established as a member of the translation
set", and decides it for a given instance. The pair of types is read from
the first two hints, so a checker is written the way it is called; the
explicit form `@checker(Source, Node)` keys a function that carries no such
hints (a generated one). The implementations live in `checkers`, next to
nothing but each other, and register at import time; this module only holds
the machinery.

Dispatch
--------
Declared membership (`isinstance`) settles it affirmatively first, and a
class kind stops there -- it *is* `isinstance`. Otherwise every checker that
*applies* is asked, and their answers are OR-ed. A checker applies when

* its **source** is a supertype of `type(t)`, and it is the nearest such
  source registered against its node -- ordinary method resolution, so a
  wrapper (`Inverse`) shadows the concrete base it also inherits from, and a
  typed `InverseAffine` uses the recursing `Inverse` checker rather than the
  matrix checker that would read (and materialize) it;

* its **node** is a **subset** of the queried kind. That direction is what
  makes the answer sound: establishing `t in C` for a `C subset of kind`
  proves `t in kind`, while the converse proves nothing. It is also why a
  checker is passed the node that was *queried* rather than the one it was
  registered against -- a wrapper reasons about the question it was asked.

The disjunction is not belt and braces: the lattice is a DAG, and two
routes into a kind are routinely *incomparable*, so neither can stand for
the other. An `Affine` holding a permutation matrix is established in the
orthogonal set through the permutation node, and one holding a rotation
through the special-orthogonal node; a `Scaling` is established in the
bijections through the invertible-diagonal node, and in the translations
through the identity node. Ask only the nearest of those and the answer is
still sound, but it turns on which sibling happens to sit nearer -- so all
of them are asked, and the answers are OR-ed.

Machinery
---------
The enumeration of the applicable checkers is a
[`bagof.dispatchers`][] `Function`. Each checker is registered as a method
`(query: Source, kind: type[Super[Node]], compute)`: the source is matched
*covariantly* (`Source` a supertype of `type(t)`, the ordinary argument
direction), the kind *contravariantly* through a lower bound
([`Super`][bagof.dispatchers.Super]) so that node `N` answers a query about
kind `K` when `N` is a subset of `K` (the kinds lattice is plain
subclassing), and `compute` is left unannotated so it is carried through to
the checker without taking part in specificity. `Function.candidates`
enumerates *every* applicable method rather than picking one winner; the
single-winner model the library offers `convert`, `compose` and `simplify`
would ask only one checker and drop the rest, changing the answer.

What the library does not do, and this module keeps, is the *reduction*: the
candidates are grouped by node, the source nearest on `type(t)`'s MRO is kept
per node (the override that shadows an inherited base), and the survivors'
booleans are OR-ed. The library does not cache that enumeration either (it
tests every method against the argument values), so the selection -- which
depends on `(type(t), kind)` alone -- is memoized per that pair; only the
OR, which reads values, runs on every call. The full rationale, and the
enhancements this rests on, are in the migration memo under `docs/design/`.
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

# stdlib
import abc

# dependencies
import typing_extensions as tx
from bagof.dispatchers import Function, Super

# datamodel
from brainhops.datamodel import kinds
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.systems import AxisList

# typing
if tx.TYPE_CHECKING:
    from ..base import Transformation as _Transformation

Transformation: tx.TypeAlias = "_Transformation"
Kind: tx.TypeAlias = tx.Type[kinds.TransformationKind]
"""
One node of the [`kinds`][] hierarchy -- a set node or, since every
concrete transform is registered into the hierarchy, a class kind.
"""

Family: tx.TypeAlias = kinds.TransformationFamily
Key: tx.TypeAlias = tx.Tuple[tx.Type[Transformation], Kind]
Checker: tx.TypeAlias = tx.Callable[[Transformation, Kind, bool], bool]
"""
A checker decides whether a transform is *established* in a kind node:
`checker(t, kind, compute) -> bool` (`compute=False` analytic, `True`
numeric). It is keyed by `(source transform type, kind node)` and is given
the node that was queried, which may be a superset of the one it was
registered against.
"""

Selection: tx.TypeAlias = tx.Tuple[Checker, ...]
"""The checker implementations one `(type, kind)` query ORs, in order."""

KindLike: tx.TypeAlias = tx.Union[
    Kind,
    str,  # a NAME/SYMBOL or a registered alias
]
FamilyLike: tx.TypeAlias = tx.Union[
    Family,
    tx.Tuple[KindLike, tx.Optional[int]],
    KindLike,
    int,  # a dimension (the whole set, at that dimension)
]

KIND_ALIASES: tx.Dict[str, Kind] = {}
"""
The kinds a [`kinds`][] NAME or SYMBOL does not already name, keyed by name
(lower-case). A kind is either a kind node or a class matched by
`isinstance`; see the module docstring for what separates the two. The table
is populated by `checkers` at import time.
"""


# --- Machinery --------------------------------------------------------


class IsKind:
    """
    The membership predicate and the registry of its checkers.

    A [`bagof.dispatchers`][] `Function` enumerates the checkers that apply
    to a `(transformation, kind)` question -- source covariant, kind
    contravariant via a [`Super`][bagof.dispatchers.Super] lower bound, and
    `compute` carried (see the module docstring). This class wraps it with
    the reduction the library does not do: a per-node most-specific grouping
    feeding an OR.

    It also stays *dict-like* over its `(source, node)` keys -- `key in
    is_kind`, `is_kind.get(key)` -- because `checkers` reads the registry to
    decide whether a generated checker would clash with a hand-written one.
    """

    def __init__(self) -> None:
        self._function = Function("is_kind")
        # The registered checkers, keyed by the `(source, node)` pair a call
        # would look up. The values are the *shims* (see `_add`), from which
        # the underlying implementation is read back through `get`.
        self._registry: tx.Dict[Key, tx.Any] = {}
        # The per-`(type(x), kind)` selection memo (see `_selection`), with
        # the ABC cache token it was filled under. `None` until first use,
        # and reset to `None` by every registration.
        self._memo: tx.Optional[tx.Tuple[object, tx.Dict[Key, Selection]]]
        self._memo = None

    # -- registration ---------------------------------------------------

    def register(self, *args: tx.Any) -> tx.Any:
        """Register a checker, or return a decorator that does.

        Used bare (`@is_kind.register`), the `(source, node)` pair is read
        from the function's first two hints, the second unwrapped from its
        `type[...]`. Used with explicit types
        (`is_kind.register(Source, Node)`), those key it instead -- the form
        a generated checker that carries no hints needs. Either way the
        original function is returned, so the decorators stack.
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

        # A fresh shim per `(source, node)`. Distinct callables keep the
        # library's methods distinct even where one implementation is
        # registered against many nodes (the wrappers), and carry the
        # `(source, node)` the per-node grouping needs. The shim forwards the
        # *queried* kind, not the registered node -- a wrapper reasons about
        # the question it was asked.
        def shim(x: tx.Any, kind: type, compute: bool) -> bool:
            # The library overlays the source and kind hints onto the first
            # two parameters at registration, so the `Any`/`type` here are
            # placeholders it never reads. `compute` keeps a plain `bool` on
            # every method -- identical, so it is carried through without
            # taking part in specificity. (The parameters are annotated with
            # resolvable types rather than the module's forward-reference
            # aliases, which the library cannot resolve at registration.)
            return func(x, kind, compute)

        shim.__module__ = getattr(func, "__module__", shim.__module__)
        shim.__qualname__ = f"is_kind[{source.__name__} in {node.__name__}]"
        shim._impl = func  # type: ignore[attr-defined]
        shim._source = source  # type: ignore[attr-defined]
        shim._node = node  # type: ignore[attr-defined]

        # `compute` (the shim's third parameter) carries the same `bool` hint
        # on every method, so it is passed through to the checker without
        # taking part in specificity -- the library has no first-class context
        # parameter. The kind is matched by a lower bound: `type[Super[node]]`
        # accepts a queried kind `K` exactly when `node` is a subset of `K`,
        # the contravariant direction.
        #
        # Two checkers on one source against *incomparable* nodes tie for a
        # query about a kind above both, with neither more specific. That is
        # an ambiguity only for single-winner dispatch, which is never used
        # here: is_kind asks every candidate and OR-s them, and `candidates`
        # returns tied methods silently (bagof-dispatchers reports ambiguity
        # only when a single-winner call hits it, not at registration).
        self._function.register((source, tx.Type[Super[node]]))(shim)
        self._registry[(source, node)] = shim
        self._memo = None  # a new checker can change any selection

    # -- dict-like lookup (for `checkers`) ------------------------------

    def __contains__(self, key: Key) -> bool:
        return key in self._registry

    def get(self, key: Key, default: tx.Any = None) -> tx.Optional[Checker]:
        """Get the checker registered for a `(source, node)` pair, or None."""
        shim = self._registry.get(key)
        return default if shim is None else shim._impl

    # -- calling --------------------------------------------------------

    def __call__(
        self, x: Transformation, kind: KindLike, compute: bool = False
    ) -> bool:
        """
        Whether `x`'s membership in `kind` is established at level `compute`.

        For example, for a `Linear` transformation:

        * `compute=False` (analytic) inspects structure only and assumes
          invertibility from shape (a square matrix is presumed invertible,
          a wide one surjective, a tall one injective).

        * `compute=True` (numeric) reads values and refines with rank.

        Parameters
        ----------
        x : Transformation
            The transformation whose membership is questioned.
        kind : kind-like
            A kind node, a class kind, or a string that names one -- a
            [`kinds`][] NAME or SYMBOL (`"affine"`, `"SO(3)"`) or a
            registered alias (`"inverse"`, `"displacements"`).
        compute : bool, default=False
            Whether values may be read to establish membership.

        Returns
        -------
        bool
            Whether membership is established.
        """
        kind = normalize_kind(kind)
        if isinstance(x, kind):
            return True
        if not kinds.is_transformation_set(kind):
            # A class kind denotes a representation, not a set: `isinstance`
            # is the whole answer, and nothing establishes it beyond that.
            return False
        compute = bool(compute)

        # OR the selected implementations. The *selection* depends on types
        # only and is memoized (see `_selection`); the checkers read values,
        # so the disjunction itself runs on every call. Nothing applying is
        # the answer "not established" -- an empty disjunction, never an
        # error.
        for impl in self._selection(x, kind, compute):
            if impl(x, kind, compute):
                return True
        return False

    # -- selection ------------------------------------------------------

    def _selection(
        self, x: Transformation, kind: Kind, compute: bool
    ) -> Selection:
        """
        The checker implementations to OR for `(type(x), kind)`, memoized.

        Enumerating the candidates is the expensive part of a call: the
        library tests every registered method against the argument values
        and does not cache the enumeration. It is also a function of
        `(type(x), kind)` alone, so it is memoized on that key:

        * every method is registered as `(source, type[Super[node]], bool)`
          by `_add`, with `source` and `node` plain classes. Whether it
          applies is `isinstance(x, source)` -- a function of `type(x)`,
          since no class in the hierarchy defines a value-reading
          `__instancecheck__` -- and `issubclass(node, kind)` -- a function
          of the kind value;
        * `compute` is coerced to a `bool` before it gets here and every
          method hints it `bool`, so it always applies: it is carried to
          the checkers, never dispatched on;
        * the reduction below reads only the candidates and `type(x)`'s
          MRO.

        So any two calls with the same `(type(x), kind)` select the same
        implementations, in the same order. What can change the answer is
        the registry, not the arguments: a new checker (`_add` drops the
        memo) or a new virtual subclass anywhere in an ABC hierarchy (the
        memo is keyed on `abc.get_cache_token()`, as the library's own
        dispatch cache is).

        Concurrent calls are safe without a lock. The memo is an
        `(token, dict)` pair swapped in whole, so a stale dict is never
        re-labelled with a fresh token; two threads missing on one key
        both compute the same selection, and the last `dict` write wins,
        harmlessly (a benign race); a selection computed against a
        registry that changed underneath it lands in a dict that is
        already discarded.
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
        # The library enumerates every applicable checker (source a supertype
        # of `type(x)`, node a subset of `kind`). Group them by node and keep,
        # per node, the checker whose source is nearest on `type(x)`'s MRO --
        # the override that lets a wrapper shadow the concrete base it
        # inherits from. Nearest *on the MRO* rather than by a distance
        # metric: a `Generic` subscription such as `Inverse[Affine]` inserts a
        # real intermediate class, which the MRO already orders ahead of the
        # concrete base, exactly the precedence the subclass declared.
        mro = type(x).__mro__

        def source_rank(source: type) -> int:
            try:
                return mro.index(source)
            except ValueError:
                return len(mro)  # an ancestor that is not on the MRO

        best: tx.Dict[Kind, tx.Tuple[int, tx.Any]] = {}
        for method in self._function.candidates(x, kind, compute):
            shim = method.function
            rank = source_rank(shim._source)
            current = best.get(shim._node)
            if current is None or rank < current[0]:
                best[shim._node] = (rank, shim)

        # One implementation may win several nodes (a wrapper), and is kept
        # once: it reads the queried kind, not the node, so the calls would
        # be identical. The order is the survivors' order, as before.
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
"""Public membership predicate, that dispatches on its input types."""

checker = is_kind.register
"""Decorator to register a checker function."""

get_checker = is_kind.get
"""Get the checker function for a `(transformation type, kind node)` pair."""


def is_family(x: Transformation, family: FamilyLike) -> bool:
    """Whether a transform belongs to a [`kinds.TransformationFamily`][].

    A family is a kind and, optionally, a dimensionality; a transform
    belongs to it when its kind membership is established (at analytic:
    resolution never reads a value to decide which family admits a leaf)
    and its endpoints do not contradict the dimension.

    Parameters
    ----------
    x : Transformation
        The transformation whose membership is questioned.
    family : family-like
        A [`kinds.TransformationFamily`][], a `(kind, ndim)` pair, a bare
        dimension, a kind, or a string that names a kind.

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
    # An endpoint that does not say how many axes it has cannot contradict
    # the dimension, so it does not reject: the family asks for a dimension
    # it can read, not for one it must be told. An open endpoint contradicts
    # it only when it states more axes than the dimension allows.
    #
    # FIXME
    #   In many transforms, the ndim can be guessed from the content of the
    #   xform, even if the input/output spaces are not set (e.g. the shape
    #   of the matrix or the field).
    # `space` holds only unknown axes, so only the numbers of axes can
    # clash with it.
    space = AxisList([...]).expand(family.ndim)
    return all(get_axes(e).compatible_with(space) for e in (x.input, x.output))


# --- Public helpers ---------------------------------------------------


def register_kind_alias(
    name: str, cls: tx.Union[Kind, tx.Tuple[Kind, ...]]
) -> None:
    """Name a class, or a kind node, as a kind key.

    Wrapper and field kinds have no kind node -- their membership depends on
    their contents -- so a name that resolves to one is matched by
    `isinstance` (see [`is_kind`][]). A name that resolves to a kind node
    (e.g. `"scaling"` -> the diagonal set) keeps set semantics.
    """
    if not isinstance(cls, tuple):
        cls = (cls,)
    for a_cls in cls:
        KIND_ALIASES[name.lower()] = a_cls


def normalize_kind(kind_like: KindLike) -> Kind:
    """
    Resolve a kind-like object to a kind.

    Parameters
    ----------
    kind_like : kind-like
        A kind node, a class kind, or a string that names one.

    Returns
    -------
    kind : Kind
        A kind node, or a class kind.

    Raises
    ------
    ValueError
        If `kind_like` is a string that does not name a known kind.
    TypeError
        If `kind_like` is not a kind at all -- a tuple of kinds, say, which
        is several kinds and therefore none.
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
        # `isinstance` would accept a tuple of classes, so the refusal has
        # to be explicit: one kind is one node of the hierarchy. Ask about
        # the base the classes share, or ask twice.
        raise TypeError(
            f"not a transformation kind: {kind_like!r}. A kind is one class "
            f"-- a node of `kinds`, or a concrete transformation class -- or "
            f"a string naming one."
        )
    return kind_like


def normalize_family(family_like: FamilyLike) -> Family:
    """
    Resolve a family-like object to a [`kinds.TransformationFamily`][].

    This is the single entry point that turns a user-written mode or
    simplify key into the `(kind, ndim)` pair the tables are keyed by.

    Parameters
    ----------
    family_like : family-like
        A [`kinds.TransformationFamily`][], a `(kind, ndim)` pair, a bare
        dimension, a kind, or a string that names a kind.

    Returns
    -------
    family : kinds.TransformationFamily
        The normalized family.

    Raises
    ------
    ValueError
        If the kind is a string that names nothing known, or if the input
        is not family-like at all.
    """
    # A `(kind, ndim)` pair names its dimension explicitly. It is the only
    # tuple a family reads: a tuple of *kinds* is not a kind.
    kind_like, ndims = family_like, ()
    if isinstance(family_like, tuple):
        if not kinds.is_family_tuple(family_like):
            raise ValueError(
                f"Invalid (kind, ndim) pair: {family_like!r}. A family pairs "
                f"one kind with one dimension (or None)."
            )
        kind_like, *ndims = family_like
    # A string goes through the alias table first, so that a name the kind
    # hierarchy does not know (a wrapper, a field, a friendlier spelling)
    # normalizes like any other key.
    if isinstance(kind_like, str):
        alias = KIND_ALIASES.get(kind_like.lower())
        if alias is not None:
            return kinds.TransformationFamily(alias, *ndims)
    return kinds.TransformationFamily.parse(kind_like, *ndims)
