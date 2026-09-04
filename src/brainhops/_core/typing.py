__all__ = [
    "Const",
    "HiddenConst",
    "ArrayProtocol",
    "npscalar",
    "npvector",
    "npmatrix",
    "npt",
    "cpt",
    "dkt",
]
# externals
import typing_extensions as tx
from bagof.hints import array as art
from bagof.hints import cupy as cpt
from bagof.hints import dask as dkt
from bagof.hints import numpy as npt
from bagof.magic import Frozen, NoInit, NoRepr

T = tx.TypeVar("T")
Const = tx.Annotated[T, Frozen(), NoInit()]
HiddenConst = tx.Annotated[Const[T], NoRepr()]
ArrayProtocol = art.ArrayProtocol
ArrayLike = art.ArrayLike

npscalar = tx.Union[T, npt.ndarray[tx.Tuple[()], npt.dtype[T]]]
npvector = npt.ndarray[tx.Tuple[int], npt.dtype[T]]
npmatrix = npt.ndarray[tx.Tuple[int, int], npt.dtype[T]]


def get_origin(type: tx.Any, unfold: tx.Any = None) -> tx.Any:
    origin = tx.get_origin(type)
    if origin is None:
        return type
    if unfold == "all":
        if tx.get_args(type):
            return get_origin(tx.get_args(type)[0], unfold=unfold)
    if unfold:
        if not isinstance(unfold, (list, tuple, set)):
            unfold = (unfold,)
        if origin in unfold:
            return get_origin(tx.get_args(type)[0], unfold=unfold)
    return origin
