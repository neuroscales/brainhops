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
from ._bounds import _bound_index
from ._splines import _first_tap, _tap_weight


def _axis_taps(
    xp: ModuleType,
    coords: tx.Any,
    n: int,
    order: int,
    bound: str,
    dtype: tx.Any,
    stride: int = 1,
) -> tx.List[tx.Tuple[tx.Any, tx.Any]]:
    """Return the samples and the weights that coordinates read.

    Interpolation at the coordinate `x` along an axis of length `n` adds
    `order + 1` samples, each multiplied by a weight. For each of these
    taps, this function returns the indices of the samples, which the
    boundary condition has brought inside the axis, and the weights, in
    which the signs of the boundary condition are included.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the coordinates.
    coords : array
        The coordinates along the axis, in voxels.
    n : int
        The length of the axis.
    order : int
        The spline order, from 0 to 7.
    bound : str
        The boundary condition, one of the names in `_BOUNDS`.
    dtype : data type
        The floating data type of the weights.
    stride : int, default=1
        The factor by which the indices are multiplied, which turns
        them into offsets in the flattened spatial axes.

    Returns
    -------
    list of tuple
        One pair `(index, weight)` of arrays for each tap, with the
        shape of `coords`.
    """
    offset = _first_tap(order)
    first = xp.floor(coords - offset if offset else coords)
    t = (coords - first).astype(dtype, copy=False)
    first = first.astype(np.intp)
    taps = []
    for tap in range(order + 1):
        index, sign = _bound_index(xp, first + tap, n, bound)
        if stride != 1:
            xp.multiply(index, stride, out=index)
        weight = _tap_weight(order, tap, t)
        if sign is not None:
            xp.multiply(weight, sign, out=weight, casting="unsafe")
        taps.append((index, weight))
    return taps


def _interpolation_rows(
    n: int, order: int, bound: str, rows: tx.Sequence[int]
) -> np.ndarray:
    """Return rows of the matrix that samples coefficients at integers.

    Sampling the spline of coefficients `c` at the integer coordinates
    `0, ..., n - 1` gives `A @ c`. The prefilter inverts `A`, which this
    function builds on the host, one row per requested coordinate.

    Parameters
    ----------
    n : int
        The length of the axis.
    order : int
        The spline order, from 0 to 7.
    bound : str
        The boundary condition, one of the names in `_BOUNDS`.
    rows : sequence of int
        The coordinates, which are the indices of the rows.

    Returns
    -------
    np.ndarray
        The rows, with shape `(len(rows), n)` and data type float64.
    """
    coords = np.asarray(rows, dtype=np.float64)
    out = np.zeros((len(coords), n))
    line = np.arange(len(coords))
    for index, weight in _axis_taps(np, coords, n, order, bound, np.float64):
        np.add.at(out, (line, index), weight)
    return out
