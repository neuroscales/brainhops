__all__ = ["StrEnum", "IntEnum", "Enum", "to_enum"]

from enum import Enum, IntEnum

import typing_extensions as tx

from .compat import StrEnum


def to_enum(enum: tx.Type[Enum]) -> tx.Callable[[tx.Any], tx.Any]:
    """
    The converter of a free-text value with a list of known terms (an
    enum): a known term becomes the enum member, any other string stays
    a string, and `None` stays `None`. Any other value is a `TypeError`.

    `bagof` does not do this for a `Union[Enum, str]` by itself: its
    union converter returns a value that already fits a branch as it is,
    so a string stays a string and the enum converter is never tried.
    Registering a converter for the union does not replace this
    function either: `bagof` matches a parametrised union by equality,
    so each enum, and each wrapping of its union (`Optional[...]`, the
    metadata `Maybe[...]`, which flatten into other unions), would need
    its own registration, and a shape left out would silently keep the
    string. The explicit `ConvertTo(to_enum(E))` on the field cannot be
    missed.
    """

    def convert(value: tx.Any) -> tx.Any:
        if value is None or isinstance(value, enum):
            return value
        if isinstance(value, str):
            try:
                return enum(value)
            except ValueError:
                return str(value)
        raise TypeError(
            f"Expected a {enum.__name__} or a str, not {type(value).__name__}."
        )

    return convert
