__all__ = []
# stdlib
import numbers

# externals
import typing_extensions as _tx

from . import cupy_typing as cpt

# internals
from .abc import register
from .base import ObjectConverter

# optionals
try:
    import cupy as cp
    ndarray = cp.ndarray
except ImportError:
    cp = None


if cp:

    _PY_TO_NPDTYPE = {
        object: cp.object_,
        str: cp.str_,
        bool: cp.bool_,
        int: cp.integer,
        float: cp.floating,
        complex: cp.complexfloating,
        numbers.Number: cp.number,
        numbers.Complex: cp.complexfloating,
        numbers.Real: cp.floating,
        numbers.Rational: cp.floating,
        numbers.Integral: cp.integer,
    }
    _NPGENERIC = {
        cp.generic: None,
        cp.number: None,
        cp.complexfloating: complex,
        cp.floating: float,
        cp.integer: int,
    }

    NDARRAY = _tx.TypeVar("NDARRAY", bound=cp.ndarray)


    @register(cp.ndarray, cpt.ndarray)
    class CupyArrayConverter(ObjectConverter[NDARRAY]):
        """An adaptor for cupy arrays."""

        _DEFAULT = cp.ndarray
        _FALLBACK = cp.asanyarray

        @property
        def origin(self):
            origin = super().origin
            if origin is cpt.ndarray:
                origin = cp.ndarray
            return origin

        @property
        def dtype(self):
            dtype = self.args[1] if self.args else None
            dtype = _tx.get_args(dtype)
            dtype = dtype[0] if dtype else None
            dtype = _PY_TO_NPDTYPE.get(dtype, dtype)
            return dtype

        def _convert(self, value):
            origin, dtype = self.origin, self.dtype
            # First, ensure ndarray to have access to dtype
            if not isinstance(value, cp.ndarray):
                # If we have a good dtype hint, use it already
                value = cp.asanyarray(value, _NPGENERIC.get(dtype, dtype))
            # Then, convert dtype if necessary
            if dtype and not issubclass(value.dtype.type, dtype):
                dtype = _NPGENERIC.get(dtype, dtype)
                value = value.astype(dtype)
            # Finally, convert to proper subclass if necessary
            if not isinstance(value, origin):
                value = value.view(origin)
            return value


    __all__.append("CupyArrayConverter")
