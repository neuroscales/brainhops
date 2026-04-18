__all__ = []
# stdlib
import numbers

# externals
import typing_extensions as _tx

# internals
from .abc import _register
from .base import ObjectConverter
from . import dask_typing as dkt

# optionals
try:
    import dask.array as da
    import numpy as np
except ImportError:
    da = None


if da:

    _PY_TO_NPDTYPE = {
        object: np.object_,
        str: np.str_,
        bool: np.bool_,
        int: np.integer,
        float: np.floating,
        complex: np.complexfloating,
        numbers.Number: np.number,
        numbers.Complex: np.complexfloating,
        numbers.Real: np.floating,
        numbers.Rational: np.floating,
        numbers.Integral: np.integer,
    }
    _NPGENERIC = {
        np.generic: None,
        np.number: None,
        np.complexfloating: complex,
        np.floating: float,
        np.integer: int,
    }

    NDARRAY = _tx.TypeVar("NDARRAY", bound=da.Array)


    @_register(da.Array, dkt.Array)
    class DaskArrayConverter(ObjectConverter[NDARRAY]):
        """An adaptor for dask arrays."""

        _DEFAULT = da.Array
        _FALLBACK = da.asanyarray

        @property
        def origin(self):
            origin = super().origin
            if origin is dkt.Array:
                origin = da.Array
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
            if not isinstance(value, da.Array):
                # If we have a good dtype hint, use it already
                value = da.asanyarray(value, _NPGENERIC.get(dtype, dtype))
            # Then, convert dtype if necessary
            if dtype and not issubclass(value.dtype.type, dtype):
                dtype = _NPGENERIC.get(dtype, dtype)
                value = value.astype(dtype)
            # Finally, convert to proper subclass if necessary
            if not isinstance(value, origin):
                # NOTE: dask does not have `ndarray.view(cls)` to 
                # reinterpret as a subclass. da.Array(value) does not 
                # convert, but this condition should only be hit if 
                # origin is a da.Array subclass, which we hope 
                # implements a conversion constructor.
                value = origin(value)
            return value


    __all__.append("DaskArrayConverter")
