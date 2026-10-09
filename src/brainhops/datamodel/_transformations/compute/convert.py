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

A converter takes `(x, cls, **kwargs)`: `cls` is a dispatch key, but it is
also handed over, because the class a converter is asked for is finer than the
one it is keyed by. The `Affine -> Affine` converter answers a request for any
`Affine` subclass -- an `AffineExponential`, or an io subclass -- and rebuilds
the transform as that class, not as the `Affine` it is registered under. That
is exactly what the dispatcher calls a method with, so a converter is
registered as it is written, with nothing in between.
"""

# stdlib
from functools import partial

# dependencies
import typing_extensions as tx
from bagof.dispatchers import AmbiguousMethodError, Function, NoMethodError

# api
from brainhops.errors import ConversionError

# typing
if tx.TYPE_CHECKING:
    from ..base import Transformation as _Transformation

Transformation: tx.TypeAlias = "_Transformation"
TransformationType: tx.TypeAlias = tx.Type[Transformation]
Converter: tx.TypeAlias = tx.Callable[..., Transformation]
FROM = tx.TypeVar("FROM", bound=Transformation, default=Transformation)
TO = tx.TypeVar("TO", bound=Transformation, default=Transformation)


_convert: Function = Function("convert")
"""The dispatched function every registered converter joins."""

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
    the form a generated converter that carries no hints needs. A key may
    be a `TypeVar`, which stands for its bound, as the hints a bare
    `@converter` reads do. Either way the original function is returned,
    so the decorators stack.
    """
    if len(args) == 1 and callable(args[0]) and not _is_key(args[0]):
        func = args[0]
        return _register(*_key_from_func(func), func)
    if len(args) == 2 and all(map(_is_key, args)):
        return partial(_register, *args)
    raise TypeError(
        "converter() takes a function, or an (input, output) pair of "
        f"transformation types, not {args!r}"
    )


def _is_key(arg: tx.Any) -> bool:
    """Whether `arg` keys a converter: a class, or a `TypeVar` for one."""
    return isinstance(arg, (type, tx.TypeVar))


def _name(key: tx.Any) -> str:
    """The name a converter key goes by: a class's, or a `TypeVar`'s."""
    return getattr(key, "__name__", str(key))


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
        result = _convert(x, cls, **kwargs)
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


def _key_from_func(func: Converter) -> tx.Tuple[type, type]:
    """Read `(input, output)` off a converter's own hints.

    The input is the first parameter's hint, and the output the return's
    -- which is the type its `cls` parameter is a `type[...]` of, since a
    converter returns what it was asked for.
    """
    hints = tx.get_type_hints(func)
    out = hints.pop("return")
    inp = next(iter(hints.values()))
    return inp, out


def _register(inp: type, out: type, func: Converter) -> Converter:
    """Register `func` as the converter from `inp` to `out`.

    The converter is keyed by its input type and, via a `type[out]`
    parameter, by the target class *value* a call passes. The function is
    registered as it is: it already takes `(x, cls, **kwargs)`, which is
    what the dispatcher calls a method with.

    Its `__qualname__` is set to the pair it was keyed by, since every
    converter is written as `_` and the name is what a dispatch error
    reports. A converter registered for more than one pair -- the
    same-type catch-all is -- keeps the name of the last.
    """
    func.__qualname__ = f"convert[{_name(inp)} -> {_name(out)}]"
    _convert.register((inp, tx.Type[out]))(func)
    return func
