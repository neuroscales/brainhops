__all__ = ["ObjectConverter", "TypeConverter"]
import typing_extensions as _tx

from .abc import HintConverter, register
from .utils import ConversionError, _get_origin

# ----------------------------------------------------------------------
#
#   Builtin bases
#
# ----------------------------------------------------------------------


OBJ = _tx.TypeVar("OBJ", bound=object)
TYP = _tx.TypeVar("TYP", bound=type)


@register(object)
class ObjectConverter(HintConverter[OBJ]):

    _DEFAULT = object

    def _init_check(self):
        return issubclass(self.origin, self._DEFAULT)

    def _convert(self, value: _tx.Any) -> object:
        if not isinstance(value, self.type):
            value = self.type(value)
        return value


@register(type, _tx.Type)
class TypeConverter(HintConverter[TYP]):

    def _init_check(self):
        return _get_origin(self.type) in (type, _tx.Type)

    def _convert(self, value: _tx.Any) -> type:
        if not isinstance(value, type):
            raise ConversionError(f"Value {value} is not a type")
        if _tx.get_args(self.type):
            if not issubclass(value, _tx.get_args(self.type)[0]):
                raise ConversionError(
                    f"Value {value} is not a subclass of "
                    f"{_tx.get_args(self.type)[0]}"
                )
        return value
