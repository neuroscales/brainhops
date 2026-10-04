__all__ = ["StrEnum", "IntEnum", "Enum", "term"]

from enum import Enum, IntEnum

import typing_extensions as tx

from .compat import StrEnum


def term(enum: tx.Type[Enum]) -> tx.Callable[[tx.Any], tx.Any]:
    """
    The converter of a free-text value with a list of known terms (an
    enum): a known term becomes the enum member, any other string stays
    a string, and `None` stays `None`. `bagof` does not do this for a
    `Union[Enum, str]` by itself (a string already satisfies the union).
    Any other value is a `TypeError`.
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
