# stdlib
import inspect
import itertools
import types as _types

# dependencies
import typing_extensions as tx

# core
from brainhops._core.typing import safe_get_origin

# internals
from . import registries
from .errors import CompositionError
from .registries import COMPOSERS, COMPOSERS_FASTMAP, distance

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation

# `types.UnionType` (the runtime type of a PEP 604 `X | Y` union) only
# exists on Python 3.10+. It is `None` on older interpreters.
_UnionTypes = (tx.Union,)
if hasattr(_types, "UnionType"):
    _UnionTypes += (_types.UnionType,)

# Which composer functions accept a `mode` keyword. A composer opts into
# mode-awareness simply by declaring a `mode` parameter; every other
# composer stays a plain two-argument function and is called without one.
# The map is filled lazily and cleared whenever a composer is
# (re)registered, alongside the dispatch fastmap.
_ACCEPTS_MODE: tx.Dict[tx.Callable, bool] = {}


def composer(func: tx.Callable) -> tx.Callable:
    """
    Decorator to register a function as a composer of two transformations.
    """
    types = tuple(tx.get_type_hints(func).values())[:2]
    COMPOSERS[types] = func
    COMPOSERS_FASTMAP.clear()
    _ACCEPTS_MODE.clear()
    return func


def _accepts_mode(func: tx.Callable) -> bool:
    # Whether a composer declares a `mode` parameter, cached per function.
    cached = _ACCEPTS_MODE.get(func)
    if cached is None:
        try:
            cached = "mode" in inspect.signature(func).parameters
        except (TypeError, ValueError):
            cached = False
        _ACCEPTS_MODE[func] = cached
    return cached


def _call(func: tx.Callable, x1, x2, mode):  # noqa: ANN001, ANN202
    # Call a composer, passing `mode` only when it accepts one.
    if _accepts_mode(func):
        return func(x1, x2, mode=mode)
    return func(x1, x2)


def _cancels(first: "Transformation", second: "Transformation") -> bool:
    # `first` is applied before `second`. The two cancel when `second` is
    # the inverse of `first`, or `first` is the inverse of `second`. An
    # inverse names the transform it undoes as its `forward`, so the test
    # is a plain identity check that materializes neither field. This
    # covers both a typed inverse and a generic `Inverse(forward=X)`.
    inverse_cls = registries.INVERSE
    if inverse_cls is None:
        return False
    if isinstance(second, inverse_cls) and second.forward is first:
        return True
    if isinstance(first, inverse_cls) and first.forward is second:
        return True
    return False


def compose(
    x1: "Transformation",
    x2: "Transformation",
    *,
    mode: tx.Optional[tx.Any] = None,
) -> "Transformation":
    """
    Dispatch the composition of two transformations to the appropriate
    composer function.

    `x2` is applied first, then `x1`, so the result maps ``x -> x1(x2(x))``.
    A `mode` (a list of ``(type, ndim)`` pairs, or any mode-like value) is
    threaded to any composer that declares one, so a composer can decline a
    composition the mode was told to leave alone.
    """
    # Identity-cancel rule, applied *before* type dispatch. When `x2` is
    # applied first and `x1` undoes it (or vice versa), the composition is
    # the identity and neither field is materialized. This must run before
    # dispatch, because the distance-based dispatch would otherwise prefer a
    # numeric composer such as ``(Affine, Affine)`` and invert numerically.
    if _cancels(first=x2, second=x1):
        # local import to avoid a load-time cycle: `concrete` is imported by
        # the composer modules, which import this one.
        from .concrete import Identity

        return Identity(input=x2.input, output=x1.output)

    t1, t2 = type(x1), type(x2)
    if (t1, t2) in COMPOSERS_FASTMAP:
        func = COMPOSERS_FASTMAP[(t1, t2)]
        return _call(func, x1, x2, mode)
    best_distance, best_func = float("inf"), None
    for (T1, T2), FUNC in COMPOSERS.items():
        origin1 = safe_get_origin(T1)
        if origin1 in _UnionTypes:
            T1s = tx.get_args(T1)
        else:
            T1s = (T1,)
        origin2 = safe_get_origin(T2)
        if origin2 in _UnionTypes:
            T2s = tx.get_args(T2)
        else:
            T2s = (T2,)
        for T1, T2 in itertools.product(T1s, T2s):
            dist = distance(t1, T1) + distance(t2, T2)
            if dist < best_distance:
                best_distance, best_func = dist, FUNC
    if best_distance < float("inf"):
        COMPOSERS_FASTMAP[(t1, t2)] = best_func
        return _call(best_func, x1, x2, mode)
    raise CompositionError(f"No composer found for types: {t1}, {t2}")
