# This file is part of a port of torch-interpol 0.3.0 to NumPy and
# CuPy. Copyright (c) 2021 Yael Balbastre. Released under the MIT
# license, whose text is in the LICENSE file of this package.
"""Choose the array package, NumPy or CuPy, that computations use.

Every function of the package creates its arrays with the package of its
input, so that an input on a GPU is processed on the GPU and nothing is
copied back to the host. CuPy is imported only when it is installed, and
the package works without it.
"""

# stdlib
import math
from types import ModuleType

# dependencies
import numpy as np
import typing_extensions as tx

try:
    import cupy as _cupy
except ImportError:  # pragma: no cover - CuPy is optional
    _cupy = None
"""The CuPy module, or None when CuPy is not installed.

The tests replace this attribute with a fake CuPy that runs on the CPU.
"""


def get_namespace(*arrays: tx.Any) -> ModuleType:
    """Return the array package of a group of arrays.

    The package is CuPy when CuPy is installed and at least one of the
    arrays is a CuPy array, and NumPy otherwise. Arrays of the other
    package must then be converted with [`to_namespace`][], which uploads
    them to the GPU explicitly.

    Parameters
    ----------
    *arrays : array
        The arrays, of which the ones that are not arrays are ignored.

    Returns
    -------
    module
        The module `numpy` or `cupy`.
    """
    if _cupy is not None:
        for array in arrays:
            if isinstance(array, _cupy.ndarray):
                return _cupy
    return np


def is_dask(array: tx.Any) -> bool:
    """Return whether an array is a dask array, without importing dask.

    The functions of the package hand dask arrays to the module
    `_dask`, which is the only one that imports dask, so that dask
    remains an optional dependency.

    Parameters
    ----------
    array : object
        The object to test.

    Returns
    -------
    bool
        True when the object is a dask array.
    """
    return hasattr(array, "__dask_graph__") and hasattr(array, "chunks")


def to_namespace(
    xp: ModuleType, array: tx.Any, dtype: tx.Any = None
) -> tx.Any:
    """Convert an array to the package `xp`, without leaving the device.

    A NumPy array is uploaded to the GPU when `xp` is CuPy. An array
    that already belongs to `xp` is returned unchanged when its data
    type is the requested one.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy`.
    array : array
        The array to convert.
    dtype : data type, optional
        The data type of the result. By default, the data type of the
        array is kept.

    Returns
    -------
    array
        The converted array.
    """
    if dtype is not None and getattr(array, "dtype", None) == np.dtype(dtype):
        dtype = None
    if xp is not np and isinstance(array, np.ndarray):
        return xp.asarray(array if dtype is None else array.astype(dtype))
    return xp.asarray(array, dtype=dtype)


class BufferPool:
    """Arrays that one computation allocates once and reuses.

    The interpolation repeats the same steps for every tap and every
    block of points, and each step writes into an array of the same
    shape as in the previous block. Allocating these arrays again each
    time is slow, in particular for large arrays, whose memory the
    system has to provide afresh. A pool keeps one flat array per name,
    and gives a view of its first values with the requested shape. A
    request for more values than the array holds, or for another data
    type, replaces the array.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` that allocates the arrays.
    """

    def __init__(self, xp: ModuleType) -> None:
        self._xp = xp
        self._arrays: tx.Dict[tx.Hashable, tx.Any] = {}

    def get(
        self, name: tx.Hashable, shape: tx.Sequence[int], dtype: tx.Any
    ) -> tx.Any:
        """Return a contiguous array of the pool, with undefined values.

        Parameters
        ----------
        name : hashable
            The name of the array in the pool. Two arrays that are used
            at the same time must have different names.
        shape : sequence of int
            The shape of the array.
        dtype : data type
            The data type of the array.

        Returns
        -------
        array
            A view of the first values of the flat array of the pool.
        """
        shape = tuple(shape)
        size = int(math.prod(shape))
        dtype = np.dtype(dtype)
        array = self._arrays.get(name)
        if array is None or array.size < size or array.dtype != dtype:
            array = self._xp.empty(size, dtype=dtype)
            self._arrays[name] = array
        return array[:size].reshape(shape)

    def release(self, *names: tx.Hashable) -> None:
        """Drop arrays from the pool, so that their memory can be freed.

        Parameters
        ----------
        *names : hashable
            The names of the arrays. Names that the pool does not hold
            are ignored.
        """
        for name in names:
            self._arrays.pop(name, None)
