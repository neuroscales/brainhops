# stdlib
from functools import partial

# dependencies
import typing_extensions as tx

# internals
from .errors import ConversionError
from .registries import CONVERTERS, CONVERTERS_FASTMAP, distance

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation


@tx.overload
def converter(
    Ti: tx.Type["Transformation"], To: tx.Type["Transformation"]
) -> tx.Callable:
    """Return a decorator to register a converter of between types."""
    ...


@tx.overload
def converter(func: tx.Callable) -> tx.Callable:
    """Register a converter of between types (based on hints)."""
    ...


def converter(*args, **kwargs) -> tx.Callable:
    """Decorator to register a converter of between transformation types."""
    if len(args) == 2:
        Ti, To = args
        return partial(converter, Ti=Ti, To=To)
    func = args[0]
    if kwargs:
        types = kwargs["Ti"], kwargs["To"]
    else:
        types = tuple(tx.get_type_hints(func).values())
    CONVERTERS[types] = func
    CONVERTERS_FASTMAP.clear()
    return func


def convert(
    x: "Transformation", cls: tx.Type["Transformation"], **kwargs
) -> "Transformation":
    """Convert a transform to a different type."""
    # TODO: implement using the CONVERTERS map,
    #       similarly to _compose() and _COMPOSERS.
    t1, t2 = type(x), cls
    if (t1, t2) in CONVERTERS_FASTMAP:
        func = CONVERTERS_FASTMAP[(t1, t2)]
        return func(x, **kwargs)
    best_distance, best_func = float("inf"), None
    for (T1, T2), FUNC in CONVERTERS.items():
        if not isinstance(T1, type):
            T1 = type(T1)
        if not isinstance(T2, type):
            T2 = type(T2)
        dist = distance(t1, T1) + distance(t2, T2)
        if dist < best_distance:
            best_distance, best_func = dist, FUNC
    if best_distance < float("inf"):
        CONVERTERS_FASTMAP[(t1, t2)] = best_func
        return best_func(x, **kwargs)
    raise ConversionError(f"No converter found for: {t1} -> {t2}")
