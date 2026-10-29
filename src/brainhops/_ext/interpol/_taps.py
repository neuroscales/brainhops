# This file is part of a port of torch-interpol 0.3.0 to NumPy and
# CuPy. Copyright (c) 2021 Yael Balbastre. Released under the MIT
# license, whose text is in the LICENSE file of this package.
"""Samples and weights that interpolation reads along one axis."""

# stdlib
from types import ModuleType

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from ._backend import BufferPool
from ._bounds import bound_index
from ._splines import first_tap, tap_weight


def axis_taps(
    xp: ModuleType,
    coords: tx.Any,
    n: int,
    order: int,
    bound: str,
    dtype: tx.Any,
    stride: int = 1,
    pool: tx.Optional[BufferPool] = None,
    axis: int = 0,
    origin: int = 0,
) -> tx.List[tx.Tuple[tx.Any, tx.Any]]:
    """Return the samples and the weights that coordinates read.

    Interpolation at the coordinate `x` along an axis of length `n` adds
    `order + 1` samples, each multiplied by a weight. For each of these
    taps, this function returns the indices of the samples, which the
    boundary condition has brought inside the axis, and the weights, in
    which the signs of the boundary condition are included. Every array
    comes from the pool, so that a call for the next block of points
    reuses the arrays of the previous one.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the coordinates.
    coords : array
        The coordinates along the axis, in voxels, of a floating data
        type.
    n : int
        The length of the axis.
    order : int
        The spline order, from 0 to 7.
    bound : str
        The boundary condition, one of the names in `_BOUNDS`.
    dtype : data type
        The data type of the interpolated values. The weights have the
        real floating type of the same precision.
    stride : int, default=1
        The factor by which the indices are multiplied, which turns
        them into offsets in the flattened spatial axes.
    pool : BufferPool, optional
        The pool that provides the arrays. By default, a new pool is
        created, and the arrays are new.
    axis : int, default=0
        The spatial axis of the coordinates, which names the arrays of
        the taps in the pool, so that the taps of every axis can be held
        at the same time.
    origin : int, default=0
        The index along the axis of the first sample of the array that
        is read, when that array holds only a window of the axis. The
        indices are shifted by it once the boundary condition has
        brought them inside the axis of length `n`, so that they index
        the window. The window must hold every sample that the
        coordinates read.

    Returns
    -------
    list of tuple
        One pair `(index, weight)` of arrays for each tap, with the
        shape of `coords`. The arrays belong to the pool, and are
        overwritten by the next call with the same axis.
    """
    if pool is None:
        pool = BufferPool(xp)
    shape = coords.shape
    real = np.finfo(dtype).dtype
    offset = first_tap(order)
    # The first sample is floor(x - offset), and t is the distance from
    # it to x, computed in the precision of the coordinates and then
    # rounded to the precision of the weights.
    first = pool.get("first", shape, coords.dtype)
    if offset:
        xp.subtract(coords, offset, out=first)
        xp.floor(first, out=first)
    else:
        xp.floor(coords, out=first)
    t = pool.get("fraction", shape, real)
    xp.subtract(coords, first, out=t)
    base = pool.get("base", shape, np.intp)
    xp.copyto(base, first, casting="unsafe")
    scratch = pool.get("scratch", shape, real)
    integer_scratch = pool.get("integer scratch", shape, np.intp)
    taps = []
    for tap in range(order + 1):
        index = pool.get(("index", axis, tap), shape, np.intp)
        weight = pool.get(("weight", axis, tap), shape, real)
        tap_weight(xp, order, tap, t, out=weight, scratch=scratch)
        xp.add(base, tap, out=index)
        bound_index(xp, index, n, bound, weight, integer_scratch, scratch)
        if origin:
            xp.subtract(index, origin, out=index)
        if stride != 1:
            xp.multiply(index, stride, out=index)
        taps.append((index, weight))
    return taps
