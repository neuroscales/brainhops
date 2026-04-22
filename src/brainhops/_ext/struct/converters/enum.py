__all__ = ["EnumConverter"]

import enum
import typing_extensions as _tx

from .abc import HintConverter, _register
from .utils import ConversionError


# ----------------------------------------------------------------------
#
#   Enums
#
# ----------------------------------------------------------------------


ENUM = _tx.TypeVar("ENUM", bound=enum.Enum)


@_register(enum.Enum)
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
