__all__ = [
    "Const", "HiddenConst", "ArrayProtocol",
    "npscalar", "npvector", "npmatrix",
    "npt", "cpt", "dkt",
]
# externals
import typing_extensions as _tx

# internals
from brainhops.struct import Frozen, NoInit, NoRepr
from brainhops.struct.converters import numpy_typing as npt
from brainhops.struct.converters import cupy_typing as cpt
from brainhops.struct.converters import dask_typing as dkt


T = _tx.TypeVar("T")
Const = _tx.Annotated[T, Frozen(), NoInit()]
HiddenConst = _tx.Annotated[Const[T], NoRepr()]
ArrayProtocol = npt.ArrayProtocol

npscalar = _tx.Union[T, npt.ndarray[_tx.Tuple[()], npt.dtype[T]]]
npvector = npt.ndarray[_tx.Tuple[int], npt.dtype[T]]
npmatrix = npt.ndarray[_tx.Tuple[int, int], npt.dtype[T]]