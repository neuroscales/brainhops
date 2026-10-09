"""Dispatcher for converting a transformation to a requested type.

`convert(x, cls)` turns a transformation `x` into a transformation of
type `cls` by running exactly one converter. The chosen converter is the
one whose input type is the nearest supertype of `type(x)` and whose
output type is the nearest supertype of `cls`. If that converter cannot
convert without loss, it raises
[`LossyConversionError`][brainhops.errors.LossyConversionError]; the
request is never handed over to another converter.

The converter is chosen from the type of `x` and also from `cls`, which
is a class passed as a value. To make this possible, each converter
annotates its second parameter as `type[Out]`, and the dispatching
[`Function`][bagof.dispatchers.Function] selects the converter whose
`Out` is the nearest ancestor of `cls`. Converters receive `cls` as an
argument, `(x, cls, **kwargs)`, because the requested class may be a
subclass of the output type that the converter is registered for. For
example, the converter from `Affine` to `Affine` also answers a request
for a subclass of `Affine`, and it builds its result as that subclass.
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
    """Register a converter for the types named by its hints."""
    ...


@tx.overload
def converter(
    inp: TransformationType, out: TransformationType
) -> tx.Callable[[Converter], Converter]:
    """Return a decorator that registers a converter from `inp` to `out`."""
    ...


def converter(*args: tx.Any) -> tx.Any:
    """Register a converter.

    In the bare form, `@converter`, the input type is read from the hint of
    the first parameter and the output type from the return hint. The
    explicit form, `@converter(Linear, Affine)`, is meant for generated
    converters, which have no hints. Either type may also be given as a
    type variable, which then stands for its bound. The original function
    is returned, so decorators can be stacked.

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
    """Return whether a value can serve as a converter's input or output."""
    return isinstance(arg, (type, tx.TypeVar))


def _name(key: tx.Any) -> str:
    """Return the display name of a key."""
    return getattr(key, "__name__", str(key))


def convert(x: FROM, cls: tx.Type[TO], **kwargs: tx.Any) -> TO:
    """Convert a transformation to another type.

    !!! note
        A converter whose output type is a supertype of `cls` also matches
        the request, and the catch-all converter from `Transformation` to
        `Transformation` matches every request. The type of the result is
        therefore checked, so that a request for a type that no converter
        can produce is refused instead of being answered silently with a
        transformation of the original type.

    Parameters
    ----------
    x : Transformation
        Transformation to convert.
    cls : type of Transformation
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

    The input type is the hint of the first parameter. The output type is
    the return hint, which names the same class as the `type[...]` hint of
    the `cls` parameter.
    """
    hints = tx.get_type_hints(func)
    out = hints.pop("return")
    inp = next(iter(hints.values()))
    return inp, out


def _register(inp: type, out: type, func: Converter) -> Converter:
    """Register `func` for the conversion `inp -> out`.

    The converter is registered for the argument types `(inp, type[out])`,
    so the dispatcher chooses it from the type of the transformation and
    from the requested class. Every converter is defined under the name
    `_`, so its qualified name is replaced with `convert[In -> Out]` to
    make error messages readable. A converter registered for several pairs
    keeps the name of the last pair.
    """
    func.__qualname__ = f"convert[{_name(inp)} -> {_name(out)}]"
    _convert.register((inp, tx.Type[out]))(func)
    return func
