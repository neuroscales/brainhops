__all__ = [
    "Const", "HiddenConst", "ArrayProtocol",
    "npscalar", "npvector", "npmatrix",
    "npt", "cpt", "dkt",
]
# externals
import typing_extensions as _tx
from bagof.converters import cupy_typing as cpt
from bagof.converters import dask_typing as dkt
from bagof.converters import numpy_typing as npt

# externals
from bagof.magic import Frozen, NoInit, NoRepr

T = _tx.TypeVar("T")
Const = _tx.Annotated[T, Frozen(), NoInit()]
HiddenConst = _tx.Annotated[Const[T], NoRepr()]
ArrayProtocol = npt.ArrayProtocol
ArrayLike = npt.ArrayLike

npscalar = _tx.Union[T, npt.ndarray[_tx.Tuple[()], npt.dtype[T]]]
npvector = npt.ndarray[_tx.Tuple[int], npt.dtype[T]]
npmatrix = npt.ndarray[_tx.Tuple[int, int], npt.dtype[T]]


def get_origin(type: _tx.Any, unfold: _tx.Any = None) -> _tx.Any:
    origin = _tx.get_origin(type)
    if origin is None:
        return type
    if unfold == "all":
        if _tx.get_args(type):
            return get_origin(_tx.get_args(type)[0], unfold=unfold)
    if unfold:
        if not isinstance(unfold, (list, tuple, set)):
            unfold = (unfold,)
        if origin in unfold:
            return get_origin(_tx.get_args(type)[0], unfold=unfold)
    return origin
