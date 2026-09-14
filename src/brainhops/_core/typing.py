__all__ = [
    "Const",
    "HiddenConst",
    "ArrayProtocol",
    "ArrayLike",
    "safe_get_origin",
    "npscalar",
    "npvector",
    "npmatrix",
    "art",
    "npt",
    "cpt",
    "dkt",
]
# externals
import typing_extensions as tx
from bagof.core.magic import safe_get_origin
from bagof.hints import array as art
from bagof.hints import cupy as cpt
from bagof.hints import dask as dkt
from bagof.hints import numpy as npt
from bagof.hints.array import ArrayLike, ArrayProtocol
from bagof.hints.numpy import dtype, ndarray
from bagof.magic import Frozen, NoInit, NoRepr

T = tx.TypeVar("T")
Const = tx.Annotated[T, Frozen(), NoInit()]
HiddenConst = tx.Annotated[Const[T], NoRepr()]

npscalar = tx.Union[T, ndarray[tx.Tuple[()], dtype[T]]]
npvector = ndarray[tx.Tuple[int], dtype[T]]
npmatrix = ndarray[tx.Tuple[int, int], dtype[T]]
