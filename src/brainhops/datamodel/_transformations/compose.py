# stdlib
import itertools
import types as _types

# dependencies
import typing_extensions as tx

# core
from brainhops._core.typing import safe_get_origin

# internals
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


def composer(func: tx.Callable) -> tx.Callable:
    """
    Decorator to register a function as a composer of two transformations.
    """
    types = tuple(tx.get_type_hints(func).values())[:2]
    COMPOSERS[types] = func
    COMPOSERS_FASTMAP.clear()
    return func


def compose(x1: "Transformation", x2: "Transformation") -> "Transformation":
    """
    Dispatch the composition of two transformations to the appropriate
    composer function.
    """
    t1, t2 = type(x1), type(x2)
    if (t1, t2) in COMPOSERS_FASTMAP:
        func = COMPOSERS_FASTMAP[(t1, t2)]
        return func(x1, x2)
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
        return best_func(x1, x2)
    raise CompositionError(f"No composer found for types: {t1}, {t2}")
