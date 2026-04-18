__all__ = [
    "AnyConverter", 
    "UnionConverter", 
    "OptionalConverter", 
    "NoneConverter"
]

import typing_extensions as _tx
import types as _t

from .abc import HintConverter, _register
from .utils import _get_origin, T, ConversionError


# ----------------------------------------------------------------------
#
#   Meta
#
# ----------------------------------------------------------------------


@_register(_tx.Any)
class AnyConverter(HintConverter[_tx.Any]):

    _DEFAULT = _tx.Any

    def _init_check(self) -> bool:
        return self.type is _tx.Any

    def _convert(self, value: _tx.Any) -> _tx.Any:
        return value
    

@_register(_tx.Union, _t.UnionType)
class UnionConverter(HintConverter[T]):

    def _init_check(self) -> bool:
        return _get_origin(self.type) in (_tx.Union, _t.UnionType)

    def _convert_error_message(self, value: _tx.Any) -> str:
        return (
            f"Value {value} cannot be converted to any of the types "
            f"in {self.type}"
        )

    def _convert(self, value: _tx.Any) -> T:
        args = _tx.get_args(self.type)
        if None in args or _t.NoneType in args and value is None:
            return None
        for arg in args:
            converter = HintConverter(arg)
            try:
                return converter(value)
            except Exception as e:
                continue
        raise e
    

@_register(_tx.Optional)
class OptionalConverter(HintConverter[T]):
        
    def _init_check(self):
        return _get_origin(self.type) is _tx.Optional

    def _init_error_message(self) -> str:
        return (
            f"OptionalConverter can only be used with Optional types,  "
            f"got {self.type}"
        )
        
    def _convert(self, value: _tx.Any) -> T:
        if value is None:
            return None
        converter = HintConverter(_tx.get_args(self.type)[0])
        return converter(value)


@_register(None, _t.NoneType)
class NoneConverter(HintConverter[None]):

    _DEFAULT = None
    
    def _init_check(self):
        return self.type is None or self.type is _t.NoneType

    def _init_error_message(self):
        return (
            f"NoneConverter can only be used with None or NoneType, "
            f"got {self.type}"
        )

    def _convert(self, value: _tx.Any) -> None:
        if value is not None:
            raise ConversionError(f"Value {value} is not None")
        return None
