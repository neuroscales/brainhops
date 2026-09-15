"""
!!! warning "Internal module"
    `brainhops._core.typing` is internal. It is documented here only to
    explain the type hints that appear in the public API. Do not import
    it, and do not rely on anything in it: its contents can change or
    disappear without notice.
"""

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
"""A field that is frozen and cannot be set through the constructor."""

HiddenConst = tx.Annotated[Const[T], NoRepr()]
"""A `Const` field that is also hidden from the generated `__repr__`."""

npscalar = tx.Union[T, ndarray[tx.Tuple[()], dtype[T]]]
"""A plain scalar, or a zero-dimensional array holding one."""

npvector = ndarray[tx.Tuple[int], dtype[T]]
"""A one-dimensional array."""

npmatrix = ndarray[tx.Tuple[int, int], dtype[T]]
"""A two-dimensional array."""
