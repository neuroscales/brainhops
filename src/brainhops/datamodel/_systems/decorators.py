import typing_extensions as tx
from bagof.converters import Converter
from bagof.magic import ConvertTo, fields


class _NoneReadsAsDefault:
    """Field converter that reads `None` as the default of the field.

    Other values are passed to the converter that the type hint would have
    had without this one. For example, `axes=None` means the default of the
    `axes` field, which is `[...]` for a variadic system and the axes of the
    system for a fixed system. Only the field knows its default, so each
    field gets its own converter, and `bind_axes_default` links that
    converter to its field. The converter is not registered for
    `AxisSequence`, because the registry would then return this converter
    when it is asked for the converter of the hint.
    """

    def __init__(self, hint: tx.Any) -> None:
        self.hint = hint
        self.field: tx.Any = None
        self._convert: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None

    def __call__(self, value: tx.Any) -> tx.Any:
        if value is None and self.field is not None:
            factory = self.field.factory
            value = factory() if callable(factory) else self.field.default
        if self._convert is None:
            self._convert = Converter.get(self.hint)
        return self._convert(value)


class Axes:
    """Annotation for an `axes` field in which `None` stands for the default.

    `Axes[hint]` is the type `hint` with a converter that replaces `None`
    by the default of the field.
    """

    def __class_getitem__(cls, hint: tx.Any) -> tx.Any:
        return tx.Annotated[hint, ConvertTo(_NoneReadsAsDefault(hint))]


def bind_axes_default(cls: type) -> type:
    """Link the converter of the `axes` field of `cls` to that field."""
    for field in fields(cls):
        if field.name == "axes" and isinstance(
            field.converter, _NoneReadsAsDefault
        ):
            field.converter.field = field
    return cls
