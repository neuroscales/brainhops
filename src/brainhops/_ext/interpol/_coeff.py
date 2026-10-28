# This file is part of a port of torch-interpol 0.3.0 to NumPy and
# CuPy. Copyright (c) 2021 Yael Balbastre. Released under the MIT
# license, whose text is in the LICENSE file of this package.
"""Spline prefilter, which turns values into spline coefficients.

A spline of order 2 or more does not pass through its coefficients, so
interpolating the samples of an image directly would blur it. The
prefilter computes the coefficients whose spline takes the value of the
image at every sample. Along one axis, sampling the coefficients `c` at
the integer coordinates gives `A @ c`, where the matrix `A` depends on
the order and on the boundary condition, and the prefilter computes
`c = inv(A) @ f` from the values `f`.

For the symmetric and periodic boundaries `dct1`, `dct2` and `dft`, the
inverse is a recursive filter, with one causal and one anticausal pass
for each pole of the spline. These routines come from torch-interpol,
which ported them from the C code of John Ashburner in SPM, itself
derived from the code of Philippe Thevenaz, and from SciPy for `dct2`.
The theory is in the papers of Unser, Aldroubi and Eden (IEEE
Transactions on Signal Processing 41(2):821-848, 1993) and of Unser
(IEEE Signal Processing Magazine 16(6):22-38, 1999).

For the other boundaries, the matrix `A` differs from the matrix of the
`dct2` boundary only in its first and last `order // 2` rows. The
coefficients are then computed with the `dct2` filter, and corrected
exactly with the Woodbury identity, at the cost of `order` extra
multiply-adds per sample. torch-interpol instead applied the `dct1`
filter to the zero boundary and the `dct2` filter to the replicate
boundary, so that its coefficients did not interpolate the values near
the edges, at every order for the zero boundary and from order 4 for
the replicate boundary. It did not support the antisymmetric
boundaries either.
"""

# stdlib
import functools
import math
from types import ModuleType

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from ._backend import _asarray, _get_namespace
from ._bounds import _normalize_bound
from ._splines import _MAX_ORDER
from ._taps import _interpolation_rows

_FILTERED = ("dct1", "dct2", "dft")
"""The boundaries whose prefilter is a recursive filter."""

_LOG_EPS = math.log(1e-17)
"""Logarithm of the size below which a power of a pole is neglected."""


@functools.lru_cache(maxsize=None)
def _poles(order: int) -> tx.Tuple[float, ...]:
    """Return the poles of the prefilter of a spline.

    Parameters
    ----------
    order : int
        The spline order, from 0 to 7.

    Returns
    -------
    tuple of float
        The poles, all in the interval (-1, 0). Orders 0 and 1 have
        none.

    Raises
    ------
    NotImplementedError
    """
    if order in (0, 1):
        return ()
    if order == 2:
        return (math.sqrt(8.0) - 3.0,)
    if order == 3:
        return (math.sqrt(3.0) - 2.0,)
    if order == 4:
        return (
            math.sqrt(664.0 - math.sqrt(438976.0)) + math.sqrt(304.0) - 19.0,
            math.sqrt(664.0 + math.sqrt(438976.0)) - math.sqrt(304.0) - 19.0,
        )
    if order == 5:
        return (
            math.sqrt(67.5 - math.sqrt(4436.25)) + math.sqrt(26.25) - 6.5,
            math.sqrt(67.5 + math.sqrt(4436.25)) - math.sqrt(26.25) - 6.5,
        )
    if order == 6:
        return (
            -0.488294589303044755130118038883789062112279161239377608394,
            -0.081679271076237512597937765737059080653379610398148178525368,
            -0.00141415180832581775108724397655859252786416905534669851652709,
        )
    if order == 7:
        return (
            -0.5352804307964381655424037816816460718339231523426924148812,
            -0.122554615192326690515272264359357343605486549427295558490763,
            -0.0091486948096082769285930216516478534156925639545994482648003,
        )
    raise NotImplementedError(f"Spline order {order} is not implemented.")


def _gain(order: int) -> float:
    # The gain makes the filter return a constant signal unchanged.
    gain = 1.0
    for z in _poles(order):
        gain *= (1.0 - z) * (1.0 - 1.0 / z)
    return gain


def _horizon(z: float, n: int) -> int:
    # Number of samples after which the powers of the pole are
    # negligible.
    return min(n, int(math.ceil(_LOG_EPS / math.log(abs(z)))))


# --- initial conditions -----------------------------------------------------
#
# Each function returns host vectors `v` and the slice of the samples
# that they weight, so that the initial value is `v @ x[slice]`. The
# vectors are cached per axis length and pole.


@functools.lru_cache(maxsize=None)
def _causal_init(n: int, z: float, bound: str) -> tx.Tuple[np.ndarray, slice]:
    if bound == "dct1":
        horizon = _horizon(z, n)
        if horizon < n:
            # The tail of the sum is negligible, and the mirrored copy
            # of the array is never reached.
            return z ** np.arange(horizon, dtype=np.float64), slice(horizon)
        # The extended array has period 2 * (n - 1), and the sum over
        # one period is exact.
        k = np.arange(n, dtype=np.float64)
        v = z**k + z ** (2 * n - 2 - k)
        v[0] = 1.0
        v[-1] = z ** (n - 1)
        return v / (1.0 - z ** (2 * n - 2)), slice(None)
    if bound == "dct2":
        # Ported from SciPy (ni_splines.c), with the period 2 * n summed
        # in closed form.
        k = np.arange(n, dtype=np.float64)
        zn = z**n
        v = (z**k + zn * z ** (n - 1 - k)) * (z / (1.0 - zn * zn))
        v[0] += 1.0
        return v, slice(None)
    if bound == "dft":
        horizon = _horizon(z, n)
        # x[0] + z * x[n - 1] + z**2 * x[n - 2] + ..., over one period.
        v = z ** np.arange(horizon - 1, 0, -1, dtype=np.float64)
        v = np.concatenate([[1.0], v]) / (1.0 - z**horizon)
        if horizon == 1:
            return v, slice(0, 1)
        index = np.concatenate([[0], np.arange(n - horizon + 1, n)])
        order = np.argsort(index)
        return v[order], slice(None) if horizon == n else index[order]
    raise ValueError(f"No recursive filter for the boundary {bound!r}.")


@functools.lru_cache(maxsize=None)
def _anticausal_init(
    n: int, z: float, bound: str
) -> tx.Tuple[np.ndarray, tx.Any]:
    if bound == "dct1":
        v = np.asarray([z * z, z]) / (z * z - 1.0)
        return v, slice(n - 2, n)
    if bound == "dct2":
        return np.asarray([z / (z - 1.0)]), slice(n - 1, n)
    if bound == "dft":
        horizon = _horizon(z, n)
        # z * x[n - 1] + z**2 * x[0] + z**3 * x[1] + ..., over one
        # period.
        v = z ** np.arange(2, horizon + 1, dtype=np.float64)
        v = np.concatenate([v, [z]]) / (z**horizon - 1.0)
        if horizon == n:
            return v, slice(None)
        index = np.concatenate([np.arange(horizon - 1), [n - 1]])
        return v, index
    raise ValueError(f"No recursive filter for the boundary {bound!r}.")


def _apply_init(
    xp: ModuleType, x: tx.Any, init: tx.Tuple[np.ndarray, tx.Any]
) -> tx.Any:
    v, where = init
    v = _asarray(xp, v, x.dtype)
    if isinstance(where, np.ndarray):
        where = _asarray(xp, where)
    return xp.tensordot(v, x[where], axes=1)


# --- recursive filter -------------------------------------------------------


def _recursive_filter(
    xp: ModuleType, x: tx.Any, order: int, bound: str
) -> None:
    """Apply the recursive prefilter in place along the first axis.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the array.
    x : array
        A contiguous array of a floating data type, with shape `(n, m)`,
        which is overwritten with its coefficients along the first axis.
    order : int
        The spline order, from 2 to 7.
    bound : str
        One of the boundaries `dct1`, `dct2` and `dft`.
    """
    n = x.shape[0]
    if n == 1:
        return
    x *= _gain(order)
    tmp = xp.empty_like(x[0])
    for z in _poles(order):
        x[0] = _apply_init(xp, x, _causal_init(n, z, bound))
        for i in range(1, n):
            xp.multiply(x[i - 1], z, out=tmp)
            x[i] += tmp
        x[n - 1] = _apply_init(xp, x, _anticausal_init(n, z, bound))
        for i in range(n - 2, -1, -1):
            x[i] -= x[i + 1]
            x[i] *= -z


# --- exact correction for the other boundaries ------------------------------


def _base(bound: str) -> str:
    return bound if bound in _FILTERED else "dct2"


@functools.lru_cache(maxsize=None)
def _correction(
    n: int, order: int, bound: str
) -> tx.Optional[tx.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return the terms of the exact correction of the base filter.

    The matrix `A` of the boundary is `B + S @ E`, where `B` is the
    matrix of the base boundary, `S` selects the rows in which the two
    matrices differ and `E` holds the differences. By the Woodbury
    identity, `inv(A) @ f = c - H @ (E @ c)` with `c = inv(B) @ f` and
    `H = G @ inv(I + E @ G)`, where `G = inv(B) @ S` is computed with
    the recursive filter. `E` is only nonzero in a few columns near the
    edges, so only those columns are kept.

    Parameters
    ----------
    n : int
        The length of the axis.
    order : int
        The spline order, from 2 to 7.
    bound : str
        The boundary condition, one of the names in `_BOUNDS`.

    Returns
    -------
    tuple of np.ndarray or None
        The kept columns, the matching columns of `E` and the matrix
        `H`, or None when the base filter is exact.
    """
    base = _base(bound)
    if base == bound:
        return None
    half = order // 2
    rows = sorted(set(range(min(half, n))) | set(range(max(0, n - half), n)))
    diff = _interpolation_rows(n, order, bound, rows)
    diff -= _interpolation_rows(n, order, base, rows)
    keep = np.abs(diff).max(axis=1) > 0
    if not keep.any():
        return None
    rows = np.asarray(rows)[keep]
    diff = diff[keep]
    columns = np.flatnonzero(np.abs(diff).max(axis=0) > 0)
    g = np.zeros((n, len(rows)))
    g[rows, np.arange(len(rows))] = 1.0
    _recursive_filter(np, g, order, base)
    small = diff[:, columns]
    h = g @ np.linalg.inv(np.eye(len(rows)) + small @ g[columns])
    return columns, small, h


def _filter_axis0(xp: ModuleType, x: tx.Any, order: int, bound: str) -> None:
    # Prefilter a contiguous (n, m) array in place along its first axis.
    _recursive_filter(xp, x, order, _base(bound))
    correction = _correction(x.shape[0], order, bound)
    if correction is not None:
        columns, small, h = (_asarray(xp, a) for a in correction)
        small = small.astype(x.dtype)
        h = h.astype(x.dtype)
        x -= xp.matmul(h, xp.matmul(small, x[columns]))


# --- public functions -------------------------------------------------------


def spline_coeff1d(
    input: tx.Any,
    order: int = 3,
    bound: str = "reflect",
    axis: int = -1,
) -> tx.Any:
    """Compute the spline coefficients of an array along one axis.

    The coefficients are those whose spline, sampled at the integer
    coordinates with [`pull`][brainhops._ext.interpol.pull] and the same
    order and boundary, gives back the input exactly. The computation
    runs with the package of the input, NumPy or CuPy.

    For the boundaries `mirror`, `reflect` and `grid-wrap`, the
    coefficients are computed with the recursive filter of
    torch-interpol, which makes one causal and one anticausal pass over
    the axis for each pole of the spline. For the other boundaries, the
    matrix that samples the coefficients differs from the matrix of
    `reflect` only in its first and last `order // 2` rows. The
    coefficients are then computed with the filter of `reflect`, and
    corrected exactly with the Woodbury identity, which costs about
    `order` more multiply-adds per value. torch-interpol applied the
    filter of `mirror` to the zero boundary and the filter of `reflect`
    to the replicate boundary without a correction. Its coefficients
    were therefore wrong near the edges, at every order for the zero
    boundary and from order 4 for the replicate boundary.

    Parameters
    ----------
    input : array
        The values, of any shape. An integer array is converted to
        float64.
    order : int, default=3
        The spline order, from 0 to 7. Orders 0 and 1 return a copy of
        the input, since their splines pass through their coefficients.
    bound : str, default="reflect"
        The boundary condition, by any name of the table in the
        documentation of the package.
    axis : int, default=-1
        The axis to filter.

    Returns
    -------
    array
        The coefficients, with the shape of the input.

    Raises
    ------
    ValueError
    """
    xp = _get_namespace(input)
    order = _normalize_order(order)
    bound = _normalize_bound(bound)
    input = _asarray(xp, input)
    dtype = _normalize_dtype(input.dtype)
    if order < 2:
        return input.astype(dtype, copy=True)
    axis = axis % input.ndim
    moved = xp.moveaxis(input, axis, 0)
    shape = moved.shape
    # The copy is laid out with the filtered axis first, so that each
    # step of the recursion reads and writes a contiguous block.
    x = moved.astype(dtype, order="C", copy=True).reshape((shape[0], -1))
    _filter_axis0(xp, x, order, bound)
    return xp.moveaxis(x.reshape(shape), 0, axis)


def spline_coeff(
    input: tx.Any,
    order: tx.Any = 3,
    bound: tx.Any = "reflect",
    ndim: tx.Optional[int] = None,
) -> tx.Any:
    """Compute the spline coefficients of an array along its last axes.

    The prefilter runs along each of the last `ndim` axes in turn, with
    the order and the boundary of that axis. The coefficients are those
    whose spline, sampled at the integer coordinates with
    [`pull`][brainhops._ext.interpol.pull], gives back the input
    exactly. The computation runs with the package of the input, NumPy
    or CuPy.

    Parameters
    ----------
    input : array
        The values, with shape `(*batch, *spatial)`.
    order : int or sequence of int, default=3
        The spline order, from 0 to 7, for all axes or for each of the
        spatial axes.
    bound : str or sequence of str, default="reflect"
        The boundary condition, for all axes or for each of the spatial
        axes, by any name of the table in the documentation of the
        package.
    ndim : int, optional
        The number of spatial axes. By default, it is the length of
        `order` or `bound` when one of them is a sequence, and all the
        axes of the input otherwise.

    Returns
    -------
    array
        The coefficients, with the shape of the input and a floating
        data type. Its axes may be laid out in memory in another order
        than theirs, which [`pull`][brainhops._ext.interpol.pull] reads
        without a copy.

    Raises
    ------
    ValueError
    """
    xp = _get_namespace(input)
    input = _asarray(xp, input)
    if ndim is None:
        ndim = _sequence_length(order, bound) or input.ndim
    orders = _normalize_per_axis(order, ndim, "order")
    bounds = _normalize_per_axis(bound, ndim, "bound")
    out = None
    # The first spatial axis is filtered last, so that the result is
    # contiguous when the input has no batch axes and that axis is
    # filtered.
    for d in reversed(range(ndim)):
        axis = input.ndim - ndim + d
        if _normalize_order(orders[d]) < 2:
            continue
        out = spline_coeff1d(
            input if out is None else out, orders[d], bounds[d], axis
        )
    if out is None:
        return input.astype(_normalize_dtype(input.dtype), copy=True)
    return out


# --- helpers ----------------------------------------------------------------

_ORDER_NAMES = {
    "nearest": 0,
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
}


def _normalize_order(order: tx.Any) -> int:
    """Return a spline order given as an integer or by its name.

    Parameters
    ----------
    order : int or str
        The order, from 0 to 7, or a name such as "linear" or "cubic".

    Returns
    -------
    int
        The order.

    Raises
    ------
    ValueError
    """
    if isinstance(order, str):
        if order.lower() not in _ORDER_NAMES:
            raise ValueError(f"Unknown spline order: {order!r}.")
        return _ORDER_NAMES[order.lower()]
    value = int(order)
    if value != order or not 0 <= value <= _MAX_ORDER:
        raise ValueError(
            f"The spline order must be an integer from 0 to {_MAX_ORDER}, "
            f"not {order!r}."
        )
    return value


def _normalize_dtype(dtype: tx.Any) -> np.dtype:
    """Return the data type in which an array is interpolated.

    Parameters
    ----------
    dtype : data type
        The data type of the input.

    Returns
    -------
    np.dtype
        The data type itself when it is floating or complex, and float64
        otherwise.
    """
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return dtype
    if dtype.kind == "c":
        return dtype
    return np.dtype(np.float64)


def _sequence_length(*values: tx.Any) -> int:
    for value in values:
        if isinstance(value, (list, tuple)):
            return len(value)
    return 0


def _normalize_per_axis(
    value: tx.Any, ndim: int, name: str
) -> tx.List[tx.Any]:
    """Return one value per axis from a scalar or a sequence.

    Parameters
    ----------
    value : object or sequence
        A value for all axes, or a sequence of one value per axis.
    ndim : int
        The number of axes.
    name : str
        The name of the argument, for the error message.

    Returns
    -------
    list
        The `ndim` values.

    Raises
    ------
    ValueError
    """
    if isinstance(value, (list, tuple)):
        if len(value) != ndim:
            raise ValueError(
                f"Expected {ndim} values for {name!r}, got {len(value)}."
            )
        return list(value)
    return [value] * ndim
