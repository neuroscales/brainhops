__all__ = [
    "NDArrayConverter",
]
# stdlib
import typing_extensions as _tx
import numbers
from inspect import isabstract

# internals
from .abc import _register
from .base import ObjectConverter
from . import numpy_typing as npt

# optionals
try:
    import numpy as np
    ndarray = np.ndarray
except ImportError:
    np = None

try:
    import pandas as pd
    DataFrame, Series = pd.DataFrame, pd.Series
except ImportError:
    pd = None


@_register(npt.ArrayProtocol)
class NDArrayConverter(ObjectConverter[npt.ArrayProtocol]):
    """A converter for array-like objects."""

    _DEFAULT = npt.ArrayProtocol
    _FALLBACK = np.asanyarray

    def _convert(self, value):
        origin = self.origin
        if isinstance(value, origin):
            return value
        if isabstract(origin):
            return self._FALLBACK(value)
        return origin(value)


if np:

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

    NDARRAY = _tx.TypeVar("NDARRAY", bound=np.ndarray)


    @_register(np.ndarray, npt.ndarray)
    class NumpyArrayConverter(ObjectConverter[NDARRAY]):
        """An adaptor for numpy arrays."""

        _DEFAULT = np.ndarray
        _FALLBACK = np.asanyarray

        @property
        def dtype(self):
            dtype = self.args[1] if self.args else None
            dtype = _tx.get_args(dtype)
            dtype = dtype[0] if dtype else None
            dtype = _PY_TO_NPDTYPE.get(dtype, dtype)
            return dtype

        def _convert(self, value):
            dtype = self.dtype
            # First, ensure ndarray to have access to dtype
            if not isinstance(value, np.ndarray):
                # If we have a good dtype hint, use it already
                value = np.asanyarray(value, _NPGENERIC.get(dtype, dtype))
            # Then, convert dtype if necessary
            if dtype and not issubclass(value.dtype.type, dtype):
                dtype = _NPGENERIC.get(dtype, dtype)
                value = value.astype(dtype)
            # Finally, convert to proper subclass if necessary
            if not isinstance(value, self.origin):
                value = self.origin(value)
            return value


    __all__.append("NumpyArrayConverter")

if pd:

    DATAFRAME = _tx.TypeVar("DATAFRAME", bound=pd.DataFrame)
    SERIES = _tx.TypeVar("SERIES", bound=pd.Series)


    @_register(pd.DataFrame)
    class DataFrameConverter(ObjectConverter[DATAFRAME]):
        """An adaptor for pandas DataFrames."""

        _DEFAULT = pd.DataFrame


    @_register(pd.Series)
    class SeriesConverter(ObjectConverter[SERIES]):
        """An adaptor for pandas Series."""

        _DEFAULT = pd.Series


    __all__.extend(["DataFrameConverter", "SeriesConverter"])