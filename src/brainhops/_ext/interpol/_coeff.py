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

For the boundaries `dct1`, `dct2`, `dst1`, `dst2` and `dft`, the
inverse is a recursive filter, with one causal and one anticausal pass
for each pole of the spline. Each pass needs its value at the first
sample that it computes. These boundaries extend the input of every
pass in the same way as they extend the coefficients, so that this
value is a sum over one period of the extended input, which the
functions below compute in closed form. The recursive filter comes from
torch-interpol, which ported it from the C code of John Ashburner in
SPM, itself derived from the code of Philippe Thevenaz, and from SciPy
for `dct2`. The theory is in the papers of Unser, Aldroubi and Eden
(IEEE Transactions on Signal Processing 41(2):821-848, 1993) and of
Unser (IEEE Signal Processing Magazine 16(6):22-38, 1999).

The boundaries `zero` and `replicate` have no such rule from order 4,
because the inputs of the passes after the first one do not follow the
extension of the coefficients. At orders 2 and 3, their matrices `A`
are those of `dst1` and `dct2`, whose filters therefore give their
coefficients. From order 4, the coefficients are first computed with
the filter of `dst1` or `dct2`, and then corrected near the edges. Away
from the edges, every row of `A` is the same convolution, so that the
correction is a sum of powers of the poles that decay away from each
edge. Its weights solve a small linear system, written on the first and
last `order // 2` rows of `A`. torch-interpol instead applied the
`dct1` filter to the zero boundary and the `dct2` filter to the
replicate boundary, so that its coefficients did not interpolate the
values near the edges, at every order for the zero boundary and from
order 4 for the replicate boundary. It did not support the
antisymmetric boundaries either.
"""

# stdlib
import fractions
import functools
import math
from types import ModuleType

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from ._backend import _asarray, _get_namespace, _is_dask
from ._bounds import _bound_index, _normalize_bound
from ._splines import _MAX_ORDER

_POLES: tx.Dict[int, tx.Tuple[float, ...]] = {
    0: (),
    1: (),
    2: (-0.171572875253809902396622551581,),
    3: (-0.267949192431122706472553658494,),
    4: (
        -0.361341225900220177092212841326,
        -0.0137254292973391213603312269391,
    ),
    5: (
        -0.430575347099973791851434783494,
        -0.0430962882032646538227123768226,
    ),
    6: (
        -0.488294589303044755130118038884,
        -0.0816792710762375125979377657371,
        -0.00141415180832581775108724397656,
    ),
    7: (
        -0.535280430796438165542403781682,
        -0.122554615192326690515272264359,
        -0.00914869480960827692859302165165,
    ),
}
"""The poles of the prefilter of each spline order.

The spline of order `p` takes the values `b[m]` at the integers `m`
from `-h` to `h`, with `h = p // 2`. The poles are the `h` roots in the
interval (-1, 0) of the polynomial `sum(b[m] * z**(h - m))`, from the
largest in magnitude to the smallest. They were computed with mpmath at
60 digits and rounded to 30 significant digits. Formulas with square
roots exist up to order 5, but they subtract nearly equal numbers, and
their values in float64 have a relative error of up to 2e-13.
"""

_BASE = {"zero": "dst1", "replicate": "dct2"}
"""The boundary whose filter starts the prefilter of `zero` and
`replicate`, and which gives it entirely at orders 2 and 3."""

_LOG_EPS = math.log(1e-17)
"""Logarithm of the size below which a power of a pole is neglected."""


def _poles(order: int) -> tx.Tuple[float, ...]:
    """Return the poles of the prefilter of a spline.

    Parameters
    ----------
    order : int
        The spline order, from 0 to 7.

    Returns
    -------
    tuple of float
        The poles, all in the interval (-1, 0), from the largest in
        magnitude to the smallest. Orders 0 and 1 have none.

    Raises
    ------
    NotImplementedError
    """
    if order not in _POLES:
        raise NotImplementedError(f"Spline order {order} is not implemented.")
    return _POLES[order]


@functools.lru_cache(maxsize=None)
def _stencil(order: int) -> tx.Tuple[fractions.Fraction, ...]:
    """Return the values of a spline at the integers, as exact fractions.

    The centred B-spline of order `p` is nonzero at the integers `m`
    from `-h` to `h`, with `h = p // 2`. Its value at `m` is
    `sum((-1)**k * comb(p + 1, k) * max(t, 0)**p for k) / p!`, with
    `t = m + (p + 1) / 2 - k` and `k` from 0 to `p + 1`. Sampling the
    coefficients of a spline at the integers is the convolution by
    these values.

    Parameters
    ----------
    order : int
        The spline order, from 0 to 7.

    Returns
    -------
    tuple of fractions.Fraction
        The `2 * h + 1` values, from `m = -h` to `m = h`.
    """
    h = order // 2
    centre = fractions.Fraction(order + 1, 2)
    values = []
    for m in range(-h, h + 1):
        total = fractions.Fraction(0)
        for k in range(order + 2):
            t = m + centre - k
            if t > 0:
                total += (-1) ** k * math.comb(order + 1, k) * t**order
        values.append(total / math.factorial(order))
    return tuple(values)


def _gain(order: int) -> float:
    # The gain makes the filter return a constant signal unchanged. It
    # is the product of (1 - z) * (1 - 1 / z) over the poles z, which
    # equals 1 / b[h] exactly, with the notation of `_POLES`, and which
    # is an integer.
    return float(1 / _stencil(order)[-1])


def _horizon(z: float) -> int:
    # The smallest exponent K for which the power z**K of the pole is
    # below 1e-17 in magnitude, so that it is negligible next to 1 in
    # float64.
    return int(math.ceil(_LOG_EPS / math.log(abs(z))))


# --- initial conditions -----------------------------------------------------
#
# A pole z makes a causal pass y[k] = x[k] + z * y[k - 1], then an
# anticausal pass w[k] = z * (w[k + 1] - y[k]). When the input x is
# extended past the axis by the boundary, the exact first values of the
# two passes are
#
#     y[0] = sum(z**j * x[-j] for j >= 0),
#     t = sum(z**j * x[n - 1 + j] for j >= 1),
#     w[n - 1] = -z / (1 - z**2) * (y[n - 1] + t).
#
# The functions below write these sums over one period of the
# extension. They write t with the values of y, since the causal pass
# has overwritten x when t is needed. When n - 1 is at least the
# horizon K of the pole, the powers z**K and above are dropped, which
# changes the result by less than its rounding error.
#
# Each function returns terms `(start, stop, weights)`, and the initial
# value is the sum of `weights @ x[start:stop]` over the terms. The
# weights are cached per axis length and pole, and have at most K
# values.

_Terms = tx.Tuple[tx.Tuple[int, int, np.ndarray], ...]


@functools.lru_cache(maxsize=256)
def _causal_init(n: int, z: float, bound: str) -> _Terms:
    horizon = _horizon(z)
    if n - 1 >= horizon:
        powers = z ** np.arange(horizon, dtype=np.float64)
        if bound == "dct1":
            return ((0, horizon, powers),)
        if bound == "dft":
            # x[0] + z * x[n - 1] + z**2 * x[n - 2] + ...
            return (
                (0, 1, np.ones(1)),
                (n - horizon, n, z * powers[::-1]),
            )
        # The value is x[0] plus a weighted sum of the first samples,
        # which the boundary reflects to the left of the axis, about
        # -1/2 for dct2 and dst2 and about -1 for dst1, with a change of
        # sign for dst1 and dst2.
        factor = {"dct2": z, "dst2": -z, "dst1": -z * z}[bound]
        weights = factor * powers
        weights[0] += 1.0
        return ((0, horizon, weights),)
    k = np.arange(n, dtype=np.float64)
    if bound == "dct1":
        # One period has 2 * n - 2 samples, in which the first and the
        # last sample appear once and the others twice.
        weights = z**k + z ** (2 * n - 2 - k)
        weights[0] = 1.0
        weights[-1] = z ** (n - 1)
        weights /= 1.0 - z ** (2 * n - 2)
    elif bound == "dct2":
        weights = z * (z**k + z ** (2 * n - 1 - k)) / (1.0 - z ** (2 * n))
        weights[0] += 1.0
    elif bound == "dst2":
        weights = -z * (z**k - z ** (2 * n - 1 - k)) / (1.0 - z ** (2 * n))
        weights[0] += 1.0
    elif bound == "dst1":
        # One period has 2 * n + 2 samples, two of which are zero.
        weights = z * z * (z ** (2 * n - k) - z**k)
        weights /= 1.0 - z ** (2 * n + 2)
        weights[0] += 1.0
    elif bound == "dft":
        weights = z ** (n - k)
        weights[0] = 1.0
        weights /= 1.0 - z**n
    else:
        raise ValueError(f"No recursive filter for the boundary {bound!r}.")
    return ((0, n, weights),)


@functools.lru_cache(maxsize=256)
def _anticausal_init(n: int, z: float, bound: str) -> _Terms:
    if bound == "dct1":
        # The samples after the last one are y[n - 2], y[n - 3], ...
        weights = np.asarray([z * z, z]) / (z * z - 1.0)
        return ((n - 2, n, weights),)
    if bound == "dct2":
        return ((n - 1, n, np.asarray([z / (z - 1.0)])),)
    if bound == "dst2":
        return ((n - 1, n, np.asarray([-z / (1.0 + z)])),)
    if bound == "dst1":
        return ((n - 1, n, np.asarray([-z])),)
    if bound != "dft":
        raise ValueError(f"No recursive filter for the boundary {bound!r}.")
    # The causal pass keeps the input periodic, and the samples after
    # the last one are y[0], y[1], ...
    horizon = _horizon(z)
    if n - 1 >= horizon:
        weights = -(z ** np.arange(2, horizon + 1, dtype=np.float64))
        return ((0, horizon - 1, weights), (n - 1, n, np.asarray([-z])))
    weights = -(z ** np.arange(2, n + 2, dtype=np.float64))
    weights[-1] = -z
    return ((0, n, weights / (1.0 - z**n)),)


def _apply_init(xp: ModuleType, x: tx.Any, terms: _Terms) -> tx.Any:
    value = None
    for start, stop, weights in terms:
        if stop - start == 1:
            term = x[start] * float(weights[0])
        else:
            weights = _asarray(xp, weights, x.dtype)
            term = xp.tensordot(weights, x[start:stop], axes=1)
        value = term if value is None else value + term
    return value


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
        One of the boundaries `dct1`, `dct2`, `dst1`, `dst2` and `dft`.
        The boundary `dct1` needs an axis of length 2 or more.
    """
    n = x.shape[0]
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


# --- edge correction for the zero and replicate boundaries ------------------


def _sampling_rows(
    n: int, order: int, bound: str, rows: tx.Sequence[int]
) -> np.ndarray:
    """Return rows of the matrix that samples coefficients at integers.

    Sampling the spline of coefficients `c` at the integer coordinates
    `0, ..., n - 1` gives `A @ c`, whose row `r` is the sum of
    `b[m] * c[r - m]` over the values `b` of [`_stencil`][], with `c`
    extended past the axis by the boundary. The values `b` are rounded
    from exact fractions, so that the rows are exact to rounding.

    Parameters
    ----------
    n : int
        The length of the axis.
    order : int
        The spline order, from 0 to 7.
    bound : str
        The boundary condition, one of the names in `_BOUNDS`.
    rows : sequence of int
        The indices of the rows.

    Returns
    -------
    np.ndarray
        The rows, with shape `(len(rows), n)` and data type float64.
    """
    h = order // 2
    rows = np.asarray(rows, dtype=np.intp)
    offsets = np.arange(-h, h + 1, dtype=np.intp)
    index = np.ascontiguousarray(rows[:, None] - offsets)
    weight = np.empty(index.shape)
    weight[...] = [float(b) for b in _stencil(order)]
    _bound_index(np, index, n, bound, weight)
    out = np.zeros((len(rows), n))
    np.add.at(out, (np.arange(len(rows))[:, None], index), weight)
    return out


@functools.lru_cache(maxsize=64)
def _edge_correction(
    n: int, order: int, bound: str
) -> tx.Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Return the terms that correct the filter of another boundary.

    The filter of the base boundary in `_BASE` gives coefficients `cb`
    with `B @ cb = f`, where `B` is the matrix of the base boundary. The
    rows of `B` and of the matrix `A` of the boundary are the same
    convolution away from the edges, in the rows `h` to `n - 1 - h`
    with `h = order // 2`. The difference `c - cb` therefore solves the
    homogeneous recurrence of the convolution in those rows, and is a
    sum of the sequences `z**k` and `z**(n - 1 - k)` over the poles `z`.
    The weights of this sum make the first and last `h` rows of
    `A @ c` equal to `f`, which is `B @ cb` in those rows. They solve a
    system of size `2 * h`, whose right-hand side `(B - A) @ cb` only
    reads a few coefficients at each edge.

    Parameters
    ----------
    n : int
        The length of the axis, at least `2 * (order // 2)`.
    order : int
        The spline order, from 4 to 7.
    bound : str
        The boundary `zero` or `replicate`.

    Returns
    -------
    width : int
        The number of coefficients at each edge that the right-hand side
        reads.
    left : np.ndarray
        The matrix of shape `(2 * h, width)` that gives the weights from
        the first `width` coefficients.
    right : np.ndarray
        The matrix of shape `(2 * h, width)` that gives the weights from
        the last `width` coefficients.
    powers : np.ndarray
        The matrix of shape `(length, h)` of the powers `z**k` of the
        poles, for the first `length` samples, beyond which they are
        negligible.
    """
    h = order // 2
    poles = np.asarray(_poles(order))
    rows = list(range(h)) + list(range(n - h, n))
    exact = _sampling_rows(n, order, bound, rows)
    diff = _sampling_rows(n, order, _BASE[bound], rows) - exact
    # The rows only read the samples near the edges, so that the system
    # is built from the sequences at those samples.
    band = np.flatnonzero(np.abs(exact).max(axis=0) > 0)
    k = band[:, None].astype(np.float64)
    sequences = np.concatenate([poles**k, poles ** (n - 1 - k)], axis=1)
    system = exact[:, band] @ sequences
    read = np.flatnonzero(np.abs(diff).max(axis=0) > 0)
    width = 1 + int(np.minimum(read, n - 1 - read).max())
    left = np.linalg.solve(system, diff[:, :width])
    right = np.linalg.solve(system, diff[:, n - width :])
    length = min(n, max(_horizon(z) for z in poles))
    powers = poles ** np.arange(length, dtype=np.float64)[:, None]
    return width, left, right, powers


@functools.lru_cache(maxsize=16)
def _small_inverse(n: int, order: int, bound: str) -> np.ndarray:
    # The inverse of the matrix A of an axis too short for the edge
    # correction, which has no rows away from the edges.
    return np.linalg.inv(_sampling_rows(n, order, bound, range(n)))


def _filter_axis0(xp: ModuleType, x: tx.Any, order: int, bound: str) -> None:
    # Prefilter a contiguous (n, m) array in place along its first axis.
    n = x.shape[0]
    h = order // 2
    if n == 1 and bound in ("dct1", "dct2", "dft", "replicate"):
        # The extension of a single sample is constant, and A = [1].
        return
    if bound not in _BASE or h == 1:
        _recursive_filter(xp, x, order, _BASE.get(bound, bound))
        return
    if n < 2 * h:
        inverse = _asarray(xp, _small_inverse(n, order, bound), x.dtype)
        x[...] = xp.matmul(inverse, x)
        return
    width, left, right, powers = _edge_correction(n, order, bound)
    left, right, powers = (
        _asarray(xp, a, x.dtype) for a in (left, right, powers)
    )
    _recursive_filter(xp, x, order, _BASE[bound])
    weights = xp.matmul(left, x[:width])
    weights += xp.matmul(right, x[n - width :])
    length = powers.shape[0]
    # The sequences z**k start at the first sample, and the sequences
    # z**(n - 1 - k) at the last one.
    x[:length] += xp.matmul(powers, weights[:h])
    x[n - length :] += xp.matmul(powers[::-1], weights[h:])


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

    For the boundaries `mirror`, `reflect`, `grid-wrap`, `antimirror`
    and `antireflect`, the coefficients are computed with the recursive
    filter of torch-interpol, which makes one causal and one anticausal
    pass over the axis for each pole of the spline. Each pass starts
    from a value that is exact for the boundary, so that no correction
    is needed. The zero and the replicate boundaries use the filters of
    `antimirror` and `reflect`, which are exact for them at orders 2
    and 3. From order 4, the coefficients near each edge are then
    corrected exactly, which costs a few multiply-adds per value within
    the first and last 39 to 63 samples, depending on the order.
    torch-interpol applied the filter of
    `mirror` to the zero boundary and the filter of `reflect` to the
    replicate boundary without a correction. Its coefficients were
    therefore wrong near the edges, at every order for the zero
    boundary and from order 4 for the replicate boundary.

    A dask array gives a dask array with the same chunks. The
    coefficients of a chunk depend on the values within the horizon of
    the filter, from 23 samples at order 2 to 63 samples at order 7,
    beyond which the weight of a value is below 1e-17. Each chunk is
    therefore computed by one task, which runs the filter above on the
    chunk extended by the horizon on each side, and keeps the chunk.
    The task reads all the chunks that overlap this window, however
    short they are, and the whole axis when the axis is short. At an
    edge of the array, the window keeps the boundary condition of the
    array. Where the window cuts the array, the boundary condition that
    the filter applies changes the coefficients of the chunk by less
    than their rounding errors, since the cut is at least the horizon
    away from the chunk.

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
    if _is_dask(input):
        from ._dask import _spline_coeff1d_dask

        return _spline_coeff1d_dask(input, order, bound, axis)
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
    or CuPy. A dask array gives a dask array with the same chunks,
    computed as explained in [`spline_coeff1d`][].

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
    if not _is_dask(input):
        input = _asarray(_get_namespace(input), input)
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
