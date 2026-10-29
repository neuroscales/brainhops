"""Type hints shared across the package.

!!! warning "Internal module"
    `brainhops._core.typing` is an internal module. It is documented only
    to explain the type hints that appear in the public interface, and it
    should not be imported or relied upon. It may change or disappear
    without notice.
"""

__all__ = [
    "Const",
    "Clip",
    "HiddenConst",
    "ArrayProtocol",
    "ArrayLike",
    "DTypeLike",
    "Shape",
    "ShapeLike",
    "NativeNumber",
    "safe_get_origin",
    "npscalar",
    "npvector",
    "npmatrix",
    "art",
    "npt",
    "cpt",
    "dkt",
    "is_instance_or_subclass",
]
import sys

import typing_extensions as tx
from bagof.core.magic import safe_get_origin
from bagof.hints import array as art
from bagof.hints import cupy as cpt
from bagof.hints import dask as dkt
from bagof.hints import numpy as npt
from bagof.hints.array import ArrayLike, ArrayProtocol, DTypeLike
from bagof.hints.numpy import dtype, ndarray
from bagof.magic import Default, Frozen, NoInit, NoRepr

if sys.version_info >= (3, 10):
    from types import EllipsisType
else:
    EllipsisType = type(Ellipsis)

T = tx.TypeVar("T")


def is_instance_or_subclass(
    obj: tx.Any, cls: tx.Union[type, tx.Tuple[type, ...]]
) -> bool:
    """Return whether an object is an instance or a subclass of a class.

    Classes are tested with `issubclass` and other objects with
    `isinstance`, so the function returns `False` for unrelated classes and
    for objects such as `None`.
    """
    if isinstance(obj, type):
        return issubclass(obj, cls)
    return isinstance(obj, cls)


Const: tx.TypeAlias = tx.Annotated[T, Frozen(), NoInit()]
"""Field that is frozen and cannot be set through the constructor."""

HiddenConst: tx.TypeAlias = tx.Annotated[Const[T], NoRepr()]
"""Constant field that is also hidden from the generated `__repr__`."""

Internal: tx.TypeAlias = tx.Annotated[
    tx.Optional[T], NoRepr(), NoInit(), Default(None)
]
"""Internal attribute that is not exposed to the user."""


# The hints of the small parameters of transformations, such as the
# matrix of an affine, are converted by the NumPy converter of bagof
# (bagof-converters 0.3). It computes a Dask array and copies a CuPy
# array to memory, so that a parameter is a NumPy array whatever the
# backend of the value it is built from, and it fits the array to the
# shape of the hint with the rules of NumPy broadcasting.

npscalar: tx.TypeAlias = tx.Union[T, ndarray[tx.Tuple[()], dtype[T]]]
"""Scalar, given either as a plain value or as a 0-d array."""

npvector: tx.TypeAlias = ndarray[tx.Tuple[int], dtype[T]]
"""One-dimensional NumPy array, converted from an array of any backend.

A scalar becomes a vector of one element. An array with more
dimensions is refused, unless its extra leading dimensions have length
one, in which case they are removed.
"""

npmatrix: tx.TypeAlias = ndarray[tx.Tuple[int, int], dtype[T]]
"""Two-dimensional NumPy array, converted from an array of any backend.

A scalar becomes a matrix of one element and a vector becomes a matrix
of one row. An array with more dimensions, such as a stack of matrices,
is refused, unless its extra leading dimensions have length one, in
which case they are removed.
"""

nparray: tx.TypeAlias = ndarray[tx.Tuple[int, ...], dtype[T]]
"""NumPy array of any shape, converted from an array of any backend."""

Clip: tx.TypeAlias = tx.Optional[
    tx.Union[
        float,
        tx.Tuple[float, float],
        tx.Sequence[tx.Tuple[float, float]],
    ]
]
"""Voxel margin: symmetric scalar, shared pair, per-axis pairs, or None."""

CartesianIndexElement: tx.TypeAlias = tx.Union[
    tx.SupportsIndex, slice, None, EllipsisType
]
"""
Element of a Cartesian index, which can be an integer, a slice, or `...`.
"""

CartesianIndex: tx.TypeAlias = tx.Union[
    CartesianIndexElement, tx.Tuple[CartesianIndexElement, ...]
]
"""
Cartesian index, which can be a single index or a tuple of indices.
"""

Shape: tx.TypeAlias = tx.Tuple[int, ...]
"""Any shape."""

ShapeLike: tx.TypeAlias = tx.Iterable[int]
"""Any shape-like object, such as a list or a tuple."""

NativeNumberType: tx.TypeAlias = tx.Union[
    tx.Type[bool], tx.Type[int], tx.Type[float], tx.Type[complex]
]
"""Native number types."""

NativeNumber: tx.TypeAlias = tx.Union[bool, int, float, complex]
"""Native number values."""
