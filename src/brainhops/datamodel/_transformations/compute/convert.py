"""Dispatcher for converting a transformation to a requested type.

`convert(x, cls)` runs exactly one converter: the one whose input type
is the nearest supertype of `type(x)` and whose output type is the
nearest supertype of `cls`. There is no fall-through, so a lossy
conversion raises
[`LossyConversionError`][brainhops.errors.LossyConversionError]
instead of handing over to another converter.

The dispatch selects on `cls`, a type passed as a value, through a
parameter annotated `type[Out]`, which the dispatching
[`Function`][bagof.dispatchers.Function] matches against the nearest
ancestor of the class. Converters are called as `(x, cls, **kwargs)`,
because the requested class may be finer than the key: an `Affine` to
`Affine` converter rebuilds its result as the requested subclass.
"""

from functools import partial

import typing_extensions as tx
from bagof.dispatchers import AmbiguousMethodError, Function, NoMethodError

from brainhops.errors import ConversionError

if tx.TYPE_CHECKING:
    from ..base import Transformation as _Transformation

Transformation: tx.TypeAlias = "_Transformation"
TransformationType: tx.TypeAlias = tx.Type[Transformation]
Converter: tx.TypeAlias = tx.Callable[..., Transformation]
FROM = tx.TypeVar("FROM", bound=Transformation, default=Transformation)
TO = tx.TypeVar("TO", bound=Transformation, default=Transformation)


_convert: Function = Function("convert")
"""Dispatched function on which every converter is registered."""


@tx.overload
def converter(func: Converter) -> Converter:
    """Register a converter keyed by its type hints."""
    ...


@tx.overload
def converter(
    inp: TransformationType, out: TransformationType
) -> tx.Callable[[Converter], Converter]:
    """Return a decorator that registers a converter for `inp -> out`."""
    ...


def converter(*args: tx.Any) -> tx.Any:
    """Register a converter.

    In the bare form, `@converter`, the input type is read from the hint of
    the first parameter and the output type from the return hint. The
    explicit form, `@converter(Linear, Affine)`, serves generated
    converters without hints. A key may also be a type variable, which
    stands for its bound. The original function is returned, so decorators
    can be stacked.

    Raises
    ------
    TypeError
        If the arguments match neither form.
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
    """Return whether a value can key a converter."""
    return isinstance(arg, (type, tx.TypeVar))


def _name(key: tx.Any) -> str:
    """Return the display name of a key."""
    return getattr(key, "__name__", str(key))


def convert(x: FROM, cls: tx.Type[TO], **kwargs: tx.Any) -> TO:
    """Convert a transformation to another type.

    !!! note
        A converter whose output is a supertype of `cls` also qualifies,
        and the catch-all converter between transformations matches every
        request. The type of the result is therefore checked, so that an
        unproducible type is refused rather than silently answered with
        the original type.

    Parameters
    ----------
    x
        Transformation to convert.
    cls
        Requested transformation type.

    Returns
    -------
    TO
        A transformation of type `cls`.

    Raises
    ------
    ConversionError
        If no converter applies, if several apply equally, or if the result
        of the chosen converter is not an instance of `cls`.
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
    """Read the input and output types from the hints of a converter.

    The output type is the return hint, which is also the class that `cls`
    is a `type[...]` of.
    """
    hints = tx.get_type_hints(func)
    out = hints.pop("return")
    inp = next(iter(hints.values()))
    return inp, out


def _register(inp: type, out: type, func: Converter) -> Converter:
    """Register `func` for the conversion `inp -> out`.

    The method is keyed by the input type and, through its `type[out]`
    parameter, by the requested class. Every converter is named `_`, so its
    qualified name is set to `convert[In -> Out]` for error messages; a
    converter registered for several pairs keeps the last name.
    """
    func.__qualname__ = f"convert[{_name(inp)} -> {_name(out)}]"
    _convert.register((inp, tx.Type[out]))(func)
    return func
