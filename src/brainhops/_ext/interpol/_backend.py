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


def _get_namespace(*arrays: tx.Any) -> ModuleType:
    """Return the array package of a group of arrays.

    The package is CuPy when CuPy is installed and at least one of the
    arrays is a CuPy array, and NumPy otherwise. Arrays of the other
    package must then be converted with [`_asarray`][], which uploads
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


def _asarray(xp: ModuleType, array: tx.Any, dtype: tx.Any = None) -> tx.Any:
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
