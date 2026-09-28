"""The registries every dispatcher and every cyclic import goes through.

Two unrelated things live here, for the same reason: both must be
importable by every module of the package without forming a cycle.

* [`Dispatcher`][], the registry base class. A dispatcher is a `dict` of
  functions keyed by types, called with values, that picks the function
  whose declared types sit *nearest* those of its arguments -- the way a
  method override does, but on several arguments at once. `convert` and
  `is_kind` are dispatchers; see their modules for what their keys mean.

* the late-bound singletons (`INVERSE`, `ADAPT`, `SEQUENCE`,
  `TRANSFORMATION`) and the tables (`COMPOSERS`, `SIMPLIFIERS`) that a
  module registers into at import time, so that a lower layer can reach a
  higher one without importing it.
"""

# stdlib
from functools import lru_cache

# dependencies
import typing_extensions as tx

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation
    from .inverse import Inverse
    from .sequence import Sequence


# --- dispatch ---------------------------------------------------------


def type_distance(
    t1: tx.Union[type, tx.Tuple[type, ...]],
    t2: tx.Union[type, tx.Tuple[type, ...]],
    oriented: bool = True
) -> float:
    """
    Compute the distance between two types or tuple of types in the
    class hierarchy.
    """
    if not isinstance(t1, tuple):
        t1 = (t1,)
    if not isinstance(t2, tuple):
        t2 = (t2,)
    if len(t1) != len(t2):
        raise ValueError("type_distance() requires tuples of the same length")
    return sum(_type_distance(a, b, oriented=oriented) for a, b in zip(t1, t2))


@lru_cache(maxsize=None)  # noqa: UP033
def _type_distance(
    t1: type, t2: type, oriented: bool = True
) -> float:
    """Compute the distance between two types in the class hierarchy."""
    # TODO: handle type hints (Union, Any)
    if t1 is t2:
        return 0
    if issubclass(t1, t2):
        n = len(t1.__bases__)
        return 1 + min(
            # favor earlier bases: a fractional penalty per base position,
            # too small to ever outweigh a whole step, so that two bases at
            # the same distance are separated the way the MRO separates
            # them, rather than by an accident of registration order.
            _type_distance(base, t2, oriented=False) + (i/n)
            for i, base in enumerate(t1.__bases__)
        )
    if issubclass(t2, t1) and not oriented:
        return _type_distance(t2, t1, oriented=True)
    return float("inf")


Key = tx.Union[type, tx.Tuple[type, ...]]
KEY = tx.TypeVar("KEY", bound=Key, default=Key)
FN = tx.TypeVar("FN", bound=tx.Callable, default=tx.Callable, covariant=True)


class Dispatcher(dict, tx.Generic[KEY, FN]):
    """
    A registry of functions keyed by types, with nearest-match dispatch.

    A subclass says what its keys mean and how it is called, by overriding

    | method            | what it decides                              |
    |-------------------|----------------------------------------------|
    | `key_from_func`   | the key a bare `@register` reads from hints  |
    | `key_from_args`   | the key a call looks up                      |
    | `key_distance`    | how a call's key is scored against a registered one |
    | `candidate_group` | which registrations compete for one answer   |
    | `__apply__`       | how the function is called                   |
    | `__error__`       | what a call with no registered function does |
    | `__pre_check__`   | what a call refuses outright                 |
    | `__post_check__`  | what a result is checked for                 |

    Every hook but `key_distance`/`key_from_func` receives the call's own
    arguments, so a subclass can raise with the values in hand.

    A dispatcher answers with *one* function -- [`get`][], the nearest --
    which is what a total operation wants (exactly one converter produces an
    `Affine`). A partial one wants them all, nearest first, to try in turn:
    that is [`candidates`][], and `get` is its first element.
    """

    def __new__(cls, *args, **kwargs) -> tx.Self:
        obj = super().__new__(cls)
        obj._CACHE = {}
        obj._CANDIDATES = {}
        return obj

    @tx.overload
    def register(self, *types: type) -> tx.Callable[[FN], FN]:
        """Return a decorator to register a function for the given types."""
        ...

    @tx.overload
    def register(self, func: FN) -> FN:
        """Register a function for the types inferred from its signature."""
        ...

    def register(
        self, *args, _func: tx.Optional[FN] = None, **kwargs
    ) -> tx.Callable[[FN], FN]:
        if _func is None and args and not isinstance(args[0], type):
            return self.register(*args[1:], _func=args[0], **kwargs)

        if _func is None:
            def decorator(func: FN) -> FN:
                return self.register(*args, _func=func, **kwargs)
            return decorator

        if not args:
            args = self.key_from_func(_func)
        elif len(args) == 1:
            args = args[0]
        self[args] = _func
        self._CACHE.clear()
        self._CANDIDATES.clear()
        return _func

    @classmethod
    def key_from_func(cls, func: tx.Callable) -> KEY:
        """Guess the key for a function from its type hints."""
        hints = tx.get_type_hints(func)
        types = tuple(
            tx.get_args(hint)[0] if tx.get_origin(hint) is type else hint
            for hint in hints.values()
        )
        if not types:
            raise ValueError(f"no type hints found for {func}")
        return types[0] if len(types) == 1 else types

    @classmethod
    def key_from_args(cls, *args, **kwargs) -> KEY:
        """Guess the key for a function from its arguments."""
        args = tuple(
            type(arg) if not isinstance(arg, type) else arg
            for arg in args
        )
        return args[0] if len(args) == 1 else args

    @classmethod
    def key_distance(cls, key: KEY, registered: KEY) -> tx.Any:
        """
        Score a registered key against the key a call looks up.

        `None` means "does not apply"; otherwise the lowest score wins, and
        any orderable score will do -- a subclass that ranks two axes
        against each other returns a tuple, read lexicographically.

        By default every element is *covariant*: the called type must be a
        subtype of the registered one, and the score is how far apart the
        two sit.
        """
        distance = type_distance(key, registered)
        return None if distance == float("inf") else distance

    def get(self, key: KEY, default: tx.Any = None) -> tx.Optional[FN]:
        """Get the function registered for the given key, or None."""
        MISSING = object()
        if (func := super().get(key, MISSING)) is not MISSING:
            return func
        if (func := self._CACHE.get(key, MISSING)) is not MISSING:
            return func
        # fallback to nearest match in the hierarchy
        candidates = self.candidates(key)
        func = candidates[0] if candidates else default
        self._CACHE[key] = func
        return func

    @classmethod
    def candidate_group(cls, registered: KEY) -> tx.Any:
        """
        The group a registered key competes in.

        Registrations in one group answer the same question, so only the
        nearest of them is a candidate -- the way a method override hides
        the method it overrides. By default each registration is its own
        group, so every applicable function is a candidate.
        """
        return registered

    def candidates(self, key: KEY) -> tx.Tuple[FN, ...]:
        """
        The functions that apply to `key`, nearest first.

        One per group (see [`candidate_group`][]), and one per function: a
        function registered several times is a candidate once, since it is
        called with the arguments rather than with the key it matched.

        The tuple is cached per key, and invalidated by a registration.
        """
        cached = self._CANDIDATES.get(key)
        if cached is not None:
            return cached
        # `order` keeps the sort total (and stable) when two registrations
        # tie, since a function is not orderable.
        nearest: tx.Dict[tx.Any, tx.Tuple[tx.Any, int, FN]] = {}
        for order, (registered, func) in enumerate(self.items()):
            distance = self.key_distance(key, registered)
            if distance is None:
                continue  # does not apply
            group = self.candidate_group(registered)
            current = nearest.get(group)
            if current is None or distance < current[0]:
                nearest[group] = (distance, order, func)
        found, seen = [], set()
        for _distance, _order, func in sorted(
            nearest.values(), key=lambda candidate: candidate[:2]
        ):
            if id(func) not in seen:
                seen.add(id(func))
                found.append(func)
        self._CANDIDATES[key] = found = tuple(found)
        return found

    def __apply__(self, func: FN, *args, **kwargs) -> tx.Any:
        return func(*args, **kwargs)

    def __error__(self, *args, **kwargs) -> tx.Any:
        raise ValueError(
            f"no function registered for {self.key_from_args(*args, **kwargs)}"
        )

    def __pre_check__(self, *args, **kwargs) -> None:
        pass

    def __post_check__(self, result: tx.Any, *args, **kwargs) -> tx.Any:
        return result

    def __call__(self, *args, **kwargs) -> tx.Any:
        """Apply the function registered for the given types."""
        self.__pre_check__(*args, **kwargs)
        func = self.get(self.key_from_args(*args, **kwargs))
        if func is None:
            # No function applies: the subclass decides whether that is an
            # error or an answer.
            return self.__error__(*args, **kwargs)
        result = self.__apply__(func, *args, **kwargs)
        result = self.__post_check__(result, *args, **kwargs)
        return result


# --- inverse ----------------------------------------------------------

INVERSE_CACHE = "_inverse_param_cache"
"""
The materialized inverse parameter is cached on the *forward* transform,
under this attribute name, rather than on the wrapper. A wrapper rebuilt
by `replace` or `.to(...)` keeps the same forward transform, so the cache
survives the rebuild and the inversion is not run again. The cache
assumes the forward transform is not mutated in place after it is
wrapped: a forward transform whose parameter is replaced by editing the
same object would keep serving the stale inverse.
"""

INVERSE: tx.Optional[tx.Type["Inverse"]] = None
"""Registered `Inverse` class, to avoid cyclic imports."""


def register_inverse(cls: tx.Type["Inverse"]) -> None:
    global INVERSE
    INVERSE = cls
    return cls


# --- adapt ------------------------------------------------------------

ADAPT: tx.Optional[tx.Callable[..., "Sequence"]] = None
"""
The adaptor lives in `adaptors`, which imports this module. It
registers itself here at import time, so the sequence machinery can call
it without importing that module at load time and forming a cycle. The
adaptor reconciles two consecutive transforms: it bridges a boundary
where the two systems merely reorder, rescale or flip their shared axes,
and it embeds a transform that acts on a subset of a boundary's axes in
the fuller axis space, leaving the extra axes as the identity, so a
lower-dimensional transform meets a higher-dimensional neighbour without
a dimensionality change.
"""


def register_adapt(func: tx.Callable[..., "Sequence"]) -> None:
    """Register the routine that reconciles two consecutive transforms."""
    global ADAPT
    ADAPT = func
    return func


# --- sequence ---------------------------------------------------------

SEQUENCE: tx.Optional[tx.Type["Sequence"]] = None
"""Registered `Sequence` class, to avoid cyclic imports."""


def register_sequence(cls: tx.Type["Sequence"]) -> None:
    global SEQUENCE
    SEQUENCE = cls
    return cls


# --- base -------------------------------------------------------------

# The concrete `base.Transformation` root, registered at its import time so
# that `modes._lower_key` can recognize a concrete transformation class as a
# key without importing `base` at the top level (which would cycle).
TRANSFORMATION: tx.Optional[tx.Type["Transformation"]] = None


def register_transformation(cls: tx.Type["Transformation"]) -> None:
    """Register the concrete `Transformation` root class."""
    global TRANSFORMATION
    TRANSFORMATION = cls
    return cls

# --- compose ----------------------------------------------------------

PairOfTypes = tx.Tuple[tx.Type["Transformation"], tx.Type["Transformation"]]

Composer = tx.Callable[["Transformation", "Transformation"], "Transformation"]
ComposerRegistry = tx.Dict[PairOfTypes, Composer]
"""
A composer *fuses* two transforms into one by reading their parameters
(multiplying matrices, resampling fields). Cost-free rewrites that decide
from types and object identity alone are not composers: they are two-argument
simplifiers (see `SIMPLIFIERS`), which `compose` consults first.
"""

COMPOSERS: ComposerRegistry = {}
COMPOSERS_FASTMAP: tx.Dict[PairOfTypes, tx.Tuple[Composer, ...]] = {}
"""
The fastmap caches, per concrete pair, the ordered tuple of candidate
composers `compose` should try, already sorted by dispatch order.
"""

# --- simplify ---------------------------------------------------------

LeafSimplifier = tx.Callable[..., "Transformation"]
"""
A one-argument simplifier: `f(t, policy) -> Transformation`. It is *total*
(it always returns a transform, possibly `t` itself) and it rewrites one
transform into an equivalent, cheaper one. `policy` is a `SimplifyTable`,
which the simplifier resolves against `t`.
"""

PairSimplifier = tx.Callable[..., tx.Optional["Transformation"]]
"""
A two-argument simplifier:
`f(first, second, policy) -> Transformation | None`.
It is *partial*: `None` means "these two do not collapse", which is the
common answer and so must not be an exception. The arguments are in
**application order** -- `first` is applied before `second`, exactly as the
two read in a [`Sequence`][]. Note that this is the reverse of `compose`,
which takes its operands in matrix order; `compose` flips them at the single
point where it delegates here.
"""

SimplifierKey = tx.Union[
    tx.Type["Transformation"],                # a leaf simplifier
    PairOfTypes,                              # a pair simplifier
]
SimplifierRegistry = tx.Dict[
    SimplifierKey, tx.Union[LeafSimplifier, PairSimplifier]
]

SIMPLIFIERS: SimplifierRegistry = {}
SIMPLIFIERS_FASTMAP: SimplifierRegistry = {}
"""
One registry holds both arities, keyed by a single type (leaf) or by a pair
of types (pair). `simplify()` dispatches on the number of transforms it is
given, so the two never collide.
"""
