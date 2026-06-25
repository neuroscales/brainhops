__all__ = ["EnumConverter"]

import enum
import typing_extensions as _tx

from .abc import HintConverter, register
from .utils import ConversionError


# ----------------------------------------------------------------------
#
#   Enums
#
# ----------------------------------------------------------------------


ENUM = _tx.TypeVar("ENUM", bound=enum.Enum)


@register(enum.Enum, enum.IntEnum, enum.StrEnum, enum.Flag, enum.IntFlag)
class EnumConverter(HintConverter[ENUM]):

    def _init_check(self):
        return issubclass(self.type, enum.Enum)

    def _convert(self, value: _tx.Any) -> ENUM:
        if isinstance(value, self.type):
            return value
        if value in self.type:
            return self.type(value)
        try:
            return self.type[value]
        except KeyError as e:
            raise ConversionError(
                f"Value {value} cannot be converted to enum {self.type}"
            ) from e


@register(_tx.Literal)
class LiteralConverter(HintConverter[_tx.Literal]):

    def _init_check(self):
        return hasattr(self.type, "__args__")

    def _convert(self, value: _tx.Any) -> _tx.Any:
        if value in self.type.__args__:
            return value
        raise ConversionError(
            f"Value {value} is not a valid literal for {self.type}"
        )
