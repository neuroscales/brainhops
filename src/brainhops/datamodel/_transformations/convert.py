"""Dispatch a conversion to the converter that produces the wanted type.

`convert(x, cls)` turns a transform `x` into one of type `cls`. It is a
*total, single-winner* dispatch: exactly one converter is chosen -- the one
whose declared input is the nearest supertype of `type(x)` and whose declared
output is the nearest supertype of `cls` -- and it is run once. There is no
decline-and-fall-through here (a converter that cannot preserve the map raises
[`LossyConversionError`][]; it does not hand off), so the operation maps
cleanly onto [`bagof.dispatchers`][]' most-specific-single-winner model.

What makes `convert` unusual is that it dispatches on `cls`, a *type passed
as a value*, as well as on `type(x)`. That is expressed with a
[`type`][]`[Out]` parameter: `bagof.dispatchers` reads a `type[...]` hint as
*value-dependent* and selects the overload whose `Out` is the nearest
ancestor of the class value handed in, exactly the covariant scoring the
bespoke registry did by hand. The catch-all `Transformation -> Transformation`
converter therefore matches every request, which is why the winner's result is
still checked against `cls`: a request for a type nothing can produce would
otherwise be answered by that catch-all with the original type.

The converter implementations themselves take only `(x, **kwargs)` -- `cls`
is a dispatch key, not an argument they read -- so each registration wraps the
implementation in a thin shim that accepts (and drops) the `cls` value.
"""

# dependencies
import typing_extensions as tx
from bagof.dispatchers import AmbiguousMethodError, Function, NoMethodError

# internals
from .errors import ConversionError

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation as _Transformation

Transformation: tx.TypeAlias = "_Transformation"
TransformationType: tx.TypeAlias = tx.Type[Transformation]
Converter: tx.TypeAlias = tx.Callable[..., Transformation]
FROM = tx.TypeVar("FROM", bound=Transformation, default=Transformation)
TO = tx.TypeVar("TO", bound=Transformation, default=Transformation)


_converters: Function = Function("convert")
"""The dispatched function every registered converter joins."""


def _key_from_func(func: Converter) -> tx.Tuple[type, type]:
    """Read `(input, output)` from a converter's first hint and its return."""
    hints = tx.get_type_hints(func)
    inp, *_, out = hints.values()
    return inp, out


def _register(inp: type, out: type, func: Converter) -> Converter:
    """Register `func` as the converter from `inp` to `out`.

    The converter is keyed by its input type and, via a `type[out]`
    parameter, by the target class *value* a call passes. The wrapper is
    what `bagof.dispatchers` dispatches and calls; it forwards to the real
    converter, which never sees the target class.
    """

    def _method(x: FROM, cls: tx.Type[TO], **kwargs: tx.Any) -> TO:
        return func(x, **kwargs)

    _method.__module__ = getattr(func, "__module__", _method.__module__)
    _method.__qualname__ = (
        f"convert[{inp.__name__} -> {getattr(out, '__name__', out)}]"
    )
    _converters.register((inp, tx.Type[out]))(_method)
    return func


@tx.overload
def converter(func: Converter) -> Converter:
    """Register a converter, reading `(input, output)` from its hints."""
    ...


@tx.overload
def converter(
    inp: TransformationType, out: TransformationType
) -> tx.Callable[[Converter], Converter]:
    """Return a decorator registering a converter from `inp` to `out`."""
    ...


def converter(*args: tx.Any) -> tx.Any:
    """Register a converter function.

    Used bare (`@converter`), the input type is read from the first
    parameter's hint and the output type from the return hint. Used with
    explicit types (`@converter(Linear, Affine)`), those key it instead --
    the form a generated converter that carries no hints needs. Either way
    the original function is returned, so the decorators stack.
    """
    if len(args) == 1 and callable(args[0]) and not isinstance(args[0], type):
        func = args[0]
        inp, out = _key_from_func(func)
        return _register(inp, out, func)
    if len(args) == 2 and all(isinstance(a, type) for a in args):
        inp, out = args

        def decorator(func: Converter) -> Converter:
            return _register(inp, out, func)

        return decorator
    raise TypeError(
        "converter() takes a function, or an (input, output) pair of "
        f"transformation types, not {args!r}"
    )


def convert(x: FROM, cls: tx.Type[TO], **kwargs: tx.Any) -> TO:
    """Convert a transform to a different type.

    The converter whose input is the nearest supertype of `type(x)` and
    whose output is the nearest supertype of `cls` is chosen and run.

    !!! note

        Raises [`ConversionError`][] when no converter applies, but also
        when the one that did apply produced something that is not of the
        requested type. The second check is not belt and braces: the
        dispatch scores the *target* by type subclassing, which is finite
        whenever the converter produces a **supertype** of what was asked
        for. The catch-all same-type converter, declared
        `Transformation -> Transformation`, therefore matches every
        request, and without this check a conversion to a type nothing can
        produce would quietly hand back the original type instead of saying
        so.

    Parameters
    ----------
    x : Transformation
        The transform to convert.
    cls : type[Transformation]
        The type to convert to.

    Returns
    -------
    Transformation
        A transform of the requested type.

    Raises
    ------
    ConversionError
        When no converter applies, or when the one that did apply produced
        something that is not a `cls`.
    """
    try:
        result = _converters(x, cls, **kwargs)
    except (NoMethodError, AmbiguousMethodError):
        raise ConversionError(
            f"No converter found for: {type(x)} -> {cls}"
        ) from None
    if not isinstance(result, cls):
        raise ConversionError(
            f"No converter found for: {type(x)} -> {cls}. The nearest one "
            f"produced a {type(result).__name__}."
        )
    return result
