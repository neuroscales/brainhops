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
    "safe_get_origin",
    "npscalar",
    "npvector",
    "npmatrix",
    "art",
    "npt",
    "cpt",
    "dkt",
    "is_instance_or_subclass",
    "ToNumpyParameter",
]
import sys

import typing_extensions as tx
from bagof.converters import Converter
from bagof.core.magic import safe_get_origin
from bagof.hints import array as art
from bagof.hints import cupy as cpt
from bagof.hints import dask as dkt
from bagof.hints import numpy as npt
from bagof.hints.array import ArrayLike, ArrayProtocol, DTypeLike
from bagof.hints.numpy import dtype, ndarray
from bagof.magic import Default, Frozen, NoInit, NoRepr

from brainhops import backends

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


Const = tx.Annotated[T, Frozen(), NoInit()]
"""Field that is frozen and cannot be set through the constructor."""

HiddenConst = tx.Annotated[Const[T], NoRepr()]
"""Constant field that is also hidden from the generated `__repr__`."""

Internal = tx.Annotated[tx.Optional[T], NoRepr(), NoInit(), Default(None)]
"""Internal attribute that is not exposed to the user."""


class ToNumpyParameter(Converter):
    """Converter of a small parameter, such as a matrix, to a NumPy array.

    Small parameters are NumPy arrays whatever the backend setting (see
    [`brainhops.backends`][]), but a user may build one from an array of
    another backend, such as the CuPy matrix of an affine
    transformation. The NumPy converter of bagof calls
    `numpy.asanyarray`, which CuPy refuses. This converter first moves
    the value to memory with
    [`to_backend`][brainhops.backends.to_backend], which copies an array
    from the GPU and computes a Dask array, and then lets the NumPy
    converter of bagof check the shape and the type of the values.

    It is the metadata of the annotated hints [`npscalar`][],
    [`npvector`][] and [`npmatrix`][], and it receives the NumPy hint
    that they annotate.
    """

    def __call__(self, value: tx.Any) -> tx.Any:
        """Return the value as a NumPy array of the annotated hint.

        Parameters
        ----------
        value : Any
            An array of any backend, or a value that NumPy converts, such
            as a list.

        Returns
        -------
        np.ndarray
            The NumPy array.
        """
        if backends._is_loaded(value):
            value = backends.to_backend(value, "numpy")
        return Converter.get(self.hint)(value)


npscalar = tx.Union[
    T, tx.Annotated[ndarray[tx.Tuple[()], dtype[T]], ToNumpyParameter]
]
"""Scalar, given either as a plain value or as a 0-d array."""

npvector = tx.Annotated[ndarray[tx.Tuple[int], dtype[T]], ToNumpyParameter]
"""One-dimensional NumPy array, converted from an array of any backend."""

npmatrix = tx.Annotated[
    ndarray[tx.Tuple[int, int], dtype[T]], ToNumpyParameter
]
"""Two-dimensional NumPy array, converted from an array of any backend."""

Clip = tx.Optional[
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
