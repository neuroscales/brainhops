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
    """Convert a transform to a different type.

    Raises [`ConversionError`][] when no converter applies -- and also when
    the one that did apply produced something that is not a `cls`. The
    second check is not belt and braces: the dispatch below scores the
    *target* by `distance(cls, T2)`, which is finite whenever the converter
    produces a **supertype** of what was asked for. The catch-all same-type
    converter, declared `Transformation -> Transformation`, therefore
    matches every request, and without this check a conversion to a type
    nothing can produce would quietly hand back the original type instead
    of saying so.
    """
    t1, t2 = type(x), cls
    func = CONVERTERS_FASTMAP.get((t1, t2))
    if func is None:
        best_distance, best_func = float("inf"), None
        for (T1, T2), FUNC in CONVERTERS.items():
            if not isinstance(T1, type):
                T1 = type(T1)
            if not isinstance(T2, type):
                T2 = type(T2)
            dist = distance(t1, T1) + distance(t2, T2)
            if dist < best_distance:
                best_distance, best_func = dist, FUNC
        if best_distance == float("inf"):
            raise ConversionError(f"No converter found for: {t1} -> {t2}")
        func = best_func
        CONVERTERS_FASTMAP[(t1, t2)] = func
    result = func(x, **kwargs)
    if not isinstance(result, cls):
        raise ConversionError(
            f"No converter found for: {t1} -> {t2}. The nearest one "
            f"produced a {type(result).__name__}."
        )
    return result
