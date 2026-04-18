__all__ = [
    "NDArrayConverter",
]
# stdlib
import typing_extensions as _tx
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
    _FALLBACKS = (np.asanyarray,)

    def _convert(self, value):
        origin = self.origin
        if isinstance(value, origin):
            return value
        if isabstract(origin):
            for fallback in self._FALLBACK:
                try:
                    return fallback(value)
                except Exception as e:
                    ...
            raise e
        return origin(value)


if np:

    NDARRAY = _tx.TypeVar("NDARRAY", bound=np.ndarray)

    @_register(np.ndarray, npt.ndarray)
    class NumpyArrayConverter(ObjectConverter[NDARRAY]):
        """An adaptor for numpy arrays."""

        _DEFAULT = np.ndarray

        @property
        def dtype(self):
            return self.args[1] if self.args else None

        def _convert(self, value):
            if not isinstance(value, self.type):
                value = np.asanyarray(value)
            # TODO: convert dtype if necessary
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