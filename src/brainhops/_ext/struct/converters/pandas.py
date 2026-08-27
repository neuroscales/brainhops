__all__ = []
# stdlib
import typing_extensions as _tx

# internals
from .abc import register
from .base import ObjectConverter

# optionals
try:
    import pandas as pd
    DataFrame, Series = pd.DataFrame, pd.Series
except ImportError:
    pd = None


if pd:

    DATAFRAME = _tx.TypeVar("DATAFRAME", bound=pd.DataFrame)
    SERIES = _tx.TypeVar("SERIES", bound=pd.Series)


    @register(pd.DataFrame)
    class DataFrameConverter(ObjectConverter[DATAFRAME]):
        """An adaptor for pandas DataFrames."""

        _DEFAULT = pd.DataFrame


    @register(pd.Series)
    class SeriesConverter(ObjectConverter[SERIES]):
        """An adaptor for pandas Series."""

        _DEFAULT = pd.Series


    __all__.extend(["DataFrameConverter", "SeriesConverter"])