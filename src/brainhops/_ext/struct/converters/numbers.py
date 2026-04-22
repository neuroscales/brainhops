__all__ = [
    "NumberConverter", 
    "ComplexConverter", 
    "RealConverter", 
    "RationalConverter", 
    "IntegralConverter",
]
import typing_extensions as _tx
import numbers
import fractions

from .abc import _register
from .base import ObjectConverter


# ----------------------------------------------------------------------
#
#   Numbers
#
# ----------------------------------------------------------------------


NUMBER = _tx.TypeVar("NUMBER", bound=numbers.Number)
COMPLEX = _tx.TypeVar("COMPLEX", bound=numbers.Complex)
REAL = _tx.TypeVar("REAL", bound=numbers.Real)
RATIONAL = _tx.TypeVar("RATIONAL", bound=numbers.Rational)
INTEGRAL = _tx.TypeVar("INTEGRAL", bound=numbers.Integral)


@_register(numbers.Number)
class NumberConverter(ObjectConverter[NUMBER]):

    _DEFAULT = numbers.Number
    _FALLBACKS = (int, fractions.Fraction, float, complex)
    
    def _convert(self, value: _tx.Any) -> NUMBER:
        if isinstance(value, self.type):
            return value
        for fallback in self._FALLBACKS:
            if not issubclass(fallback, self.type):
                continue
            try:
                return fallback(value)
            except Exception as e:
                ...
        raise e


@_register(numbers.Complex)
class ComplexConverter(NumberConverter[COMPLEX]):
    
    _DEFAULT = numbers.Complex
    

@_register(numbers.Real)
class RealConverter(ComplexConverter[REAL]):
    
    _DEFAULT = numbers.Real
    

@_register(numbers.Rational)
class RationalConverter(RealConverter[RATIONAL]):
    
    _DEFAULT = numbers.Rational
    

@_register(numbers.Integral)
class IntegralConverter(RationalConverter[INTEGRAL]):
    
    _DEFAULT = numbers.Integral


@_register(bool)
class BoolConverter(IntegralConverter[bool]):

    _DEFAULT = bool
    _FALLBACKS = (bool,)
    _FALSE_VALUES = ("false", "0", "no", "n", "f")

    def _convert(self, value):
        if isinstance(value, str):
            if value.lower() in self._FALSE_VALUES:
                return False
            else:
                return True
        return super()._convert(value)