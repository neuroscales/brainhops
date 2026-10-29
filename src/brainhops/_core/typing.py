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
    converter of bagof convert the values to the type that the hint
    names.

    The NumPy converter of bagof does not check the shape that the hint
    names (bagof-converters 0.2.1), so this converter fits the array to
    that shape with the rules of NumPy broadcasting. An array with fewer
    dimensions than the hint names receives dimensions of length one on
    the left, so that a scalar becomes a vector of one element. A
    dimension that the hint gives as a `Literal` size is broadcast to
    that size. An array with more dimensions than the hint names is
    refused, because broadcasting never removes a dimension. A stack of
    matrices, for example, is not a matrix. A hint whose shape is
    variadic, such as `Tuple[int, ...]`, accepts any shape.

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
        if isinstance(value, dkt.Array):
            value = value.compute()
        if isinstance(value, cpt.ndarray):
            value = value.get()
        return self._fit(Converter.get(self.hint)(value))

    def _fit(self, array: tx.Any) -> tx.Any:
        # Fit the array to the shape that the hint names, with the rules
        # of NumPy broadcasting.
        sizes = _shape_of(self.hint)
        if sizes is None:
            return array
        ndim = len(sizes)
        if array.ndim > ndim:
            raise self.value_error(
                array,
                f"The array must have {ndim} dimensions, but it has "
                f"{array.ndim}. Dimensions are never removed, so an array "
                f"with more dimensions, such as a stack, is refused.",
            )
        if array.ndim < ndim:
            array = array.reshape((1,) * (ndim - array.ndim) + array.shape)
        target = tuple(
            n if size is None else size for size, n in zip(sizes, array.shape)
        )
        if target != array.shape:
            import numpy as np

            try:
                array = np.array(np.broadcast_to(array, target))
            except ValueError as error:
                raise self.value_error(
                    array,
                    f"The array of shape {array.shape} cannot be broadcast "
                    f"to the shape {target}.",
                ) from error
        return array


def _shape_of(hint: tx.Any) -> tx.Optional[tx.Tuple[tx.Optional[int], ...]]:
    """Return the sizes that the shape of an array hint names.

    The hint is `ndarray[Tuple[...], dtype[...]]`. Each dimension of the
    result is the size that a `Literal` names, or None for any size. None
    is returned when the hint names no shape or a variadic one.
    """
    args = tx.get_args(hint)
    if not args or safe_get_origin(args[0]) is not tuple:
        return None
    dims = tx.get_args(args[0])
    if dims == ((),):
        # Python 3.10 and earlier spell `Tuple[()]` with an empty tuple as
        # its only argument.
        dims = ()
    if Ellipsis in dims:
        return None
    sizes = []
    for dim in dims:
        if safe_get_origin(dim) is tx.Literal:
            sizes.append(int(tx.get_args(dim)[0]))
        else:
            sizes.append(None)
    return tuple(sizes)


npscalar: tx.TypeAlias = tx.Union[
    T, tx.Annotated[ndarray[tx.Tuple[()], dtype[T]], ToNumpyParameter]
]
"""Scalar, given either as a plain value or as a 0-d array."""

npvector: tx.TypeAlias = tx.Annotated[
    ndarray[tx.Tuple[int], dtype[T]], ToNumpyParameter
]
"""One-dimensional NumPy array, converted from an array of any backend."""

npmatrix: tx.TypeAlias = tx.Annotated[
    ndarray[tx.Tuple[int, int], dtype[T]], ToNumpyParameter
]
"""Two-dimensional NumPy array, converted from an array of any backend."""

nparray: tx.TypeAlias = tx.Annotated[
    ndarray[tx.Tuple[int, ...], dtype[T]], ToNumpyParameter
]
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
