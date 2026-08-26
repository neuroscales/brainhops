__all__ = [
    "NumberConverter",
    "ComplexConverter",
    "RealConverter",
    "RationalConverter",
    "IntegralConverter",
]
import fractions
import numbers

import typing_extensions as _tx

from .abc import register
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


@register(numbers.Number)
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
            except Exception:
                ...
        raise e


@register(numbers.Complex)
class ComplexConverter(NumberConverter[COMPLEX]):

    _DEFAULT = numbers.Complex


@register(numbers.Real)
class RealConverter(ComplexConverter[REAL]):

    _DEFAULT = numbers.Real


@register(numbers.Rational)
class RationalConverter(RealConverter[RATIONAL]):

    _DEFAULT = numbers.Rational


@register(numbers.Integral)
class IntegralConverter(RationalConverter[INTEGRAL]):

    _DEFAULT = numbers.Integral


@register(bool)
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
