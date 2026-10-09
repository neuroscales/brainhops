"""Functions of `scipy.ndimage` for dask arrays.

[`map_coordinates`][], [`spline_filter`][] and [`spline_filter1d`][] take
the arguments of SciPy, including `order` for the degree, and return its
results to about 1e-12, but they run lazily. A block of coordinates reads
a window of the input around its stencils, and a chunk of coefficients
reads a halo from its neighbours. Windows are extended past the edges as
SciPy extends the whole array, so the boundary conditions are those of
SciPy.

The module replaces `dask_image.ndinterp`, whose `map_coordinates` reads
outside an empty crop when all the coordinates of a chunk lie before an
axis, and whose `spline_filter` is accurate to 1e-6 only, refuses the
wrap mode, and cannot overlap an axis shorter than the halo.
"""

import math

import dask
import dask.array as da
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# brainhops.backends imports this module while it is itself being
# imported, so its functions are looked up at call time.
import brainhops.backends as backends

__all__ = ["map_coordinates", "spline_filter", "spline_filter1d"]


_SPLINE_POLES = {
    2: 0.17157287525380990,
    3: 0.26794919243112270,
    4: 0.36134122590022018,
    5: 0.43057534709997379,
}
"""Magnitude of the dominant pole of the spline prefilter, per degree.

The prefilter is recursive: a coefficient depends on the sample `k`
steps away with a weight of about `pole**k`.
"""

_PREFILTER_TOLERANCE = 1e-12
"""Weight below which a sample is left out of a coefficient."""

_SCIPY_PREPAD = {"nearest": 12, "grid-constant": 12}
"""Number of samples that SciPy pads before prefiltering, per mode.

In `map_coordinates`, SciPy pads 12 edge samples for `nearest` and 12
samples of `cval` for `grid-constant` (`_prepad_for_spline_filter`).
"""

_LOCAL_MODES = ("nearest", "reflect", "mirror", "grid-wrap", "grid-constant")
"""Modes with which a block of coordinates can be sampled from a window."""


def _halo(degree: int) -> int:
    """Return the number of samples that a prefilter reads on each side.

    Beyond this halo, the weight of a sample is below
    `_PREFILTER_TOLERANCE`. The halo of an unsupported degree is zero.
    """
    pole = _SPLINE_POLES.get(int(degree))
    if pole is None:
        return 0
    return int(math.ceil(math.log(_PREFILTER_TOLERANCE) / math.log(pole)))


def _fold(k: np.ndarray, n: int, mode: str) -> np.ndarray:
    """Return the sample that each index of an extended axis holds.

    The axis, of length `n`, is extended according to `mode`, as SciPy does.
    The index -1 is returned where the extension holds `cval`, which happens
    in the `grid-constant` mode.

    Raises
    ------
    ValueError
        If the mode is not supported.
    """
    if mode == "mirror":
        if n == 1:
            return np.zeros_like(k)
        period = 2 * n - 2
        k = np.mod(k, period)
        return np.where(k >= n, period - k, k)
    if mode == "reflect":
        period = 2 * n
        k = np.mod(k, period)
        return np.where(k >= n, period - 1 - k, k)
    if mode == "grid-wrap":
        return np.mod(k, n)
    if mode == "nearest":
        return np.clip(k, 0, n - 1)
    if mode == "grid-constant":
        return np.where((k >= 0) & (k < n), k, -1)
    raise ValueError(f"No sample mapping for mode {mode!r}.")


def _axis_plan(
    lo: int, hi: int, n: int, mode: str, degree: int, prefilter: bool
) -> tx.Tuple[np.ndarray, int, np.ndarray]:
    """Plan the sampling of one axis whose stencils read `[lo, hi)`.

    The plan holds the samples to read, the halo to cut from each end of
    the window after prefiltering, and the position of each index of
    `[lo, hi)` in the cut window, where -1 stands for `cval`. Without a
    prefilter, the samples are read directly, extended by the mode. With
    a prefilter, the window covers the array extended by the mode, after
    the padding that SciPy applies, plus the halo of the prefilter.
    """
    index = np.arange(lo, hi)
    if not prefilter:
        return _fold(index, n, mode), 0, np.arange(hi - lo)
    pad = _SCIPY_PREPAD.get(mode, 0)
    padded = n + 2 * pad
    halo = _halo(degree)
    index = index + pad
    if pad:
        coeff = _fold(index, padded, mode)
        used = coeff[coeff >= 0]
        if used.size == 0:
            return np.full(1, -1), 0, np.full(hi - lo, -1)
        start, stop = int(used.min()), int(used.max()) + 1
        local = np.where(coeff >= 0, coeff - start, -1)
        # The prefilter of SciPy extends the padded array past its own edge,
        # with reflect for nearest and mirror for grid-constant.
        window = np.arange(start - halo, stop + halo)
        outer = "reflect" if mode == "nearest" else "mirror"
        signal = _fold(_fold(window, padded, outer) - pad, n, mode)
    else:
        local = np.arange(hi - lo)
        signal = _fold(np.arange(lo - halo, hi + halo), n, mode)
    return signal, halo, local


def _take(
    array: ArrayProtocol,
    positions: tx.Sequence[np.ndarray],
    cval: float,
) -> ArrayProtocol:
    """Return `array[np.ix_(*positions)]`, where -1 reads `cval`."""
    nx = backends.get_array_backend(array)
    mask = None
    for axis, pos in enumerate(positions):
        array = nx.take(array, nx.asarray(np.maximum(pos, 0)), axis=axis)
        if (pos < 0).any():
            shape = [1] * len(positions)
            shape[axis] = -1
            outside = (pos < 0).reshape(shape)
            mask = outside if mask is None else mask | outside
    if mask is not None:
        array = nx.where(
            nx.asarray(mask), nx.asarray(cval, dtype=array.dtype), array
        )
    return array


def _gather(
    input: ArrayProtocol, indices: tx.Sequence[np.ndarray], cval: float
) -> tx.Optional[ArrayProtocol]:
    """Return `input[np.ix_(*indices)]` as a concrete array.

    The index -1 reads `cval`, and only the samples that are read are
    computed. `None` is returned when an axis reads nothing.
    """
    patch = input
    positions = []
    for axis, index in enumerate(indices):
        used = np.unique(index[index >= 0])
        if used.size == 0:
            return None
        if int(used[-1]) - int(used[0]) + 1 == used.size:
            key = slice(int(used[0]), int(used[-1]) + 1)
        else:
            key = used
        patch = patch[(slice(None),) * axis + (key,)]
        positions.append(
            np.where(index >= 0, np.searchsorted(used, index), -1)
        )
    if hasattr(patch, "compute"):
        patch = patch.compute()
    return _take(patch, positions, cval)


def _map_coordinates_1block(
    coords: ArrayProtocol,
    input: ArrayProtocol,
    degree: int,
    mode: str,
    cval: float,
    prefilter: bool,
) -> ArrayProtocol:
    """Sample the input at one concrete block of coordinates.

    The coordinates have shape `(ndim, N)`. The block reads only a window
    around its stencils (see `_axis_plan`), prefilters it when requested,
    and samples it with the concrete ndimage backend. The result matches
    SciPy to `_PREFILTER_TOLERANCE`, with two exceptions: SciPy
    approximates its `reflect` prefilter near the edges of a short axis
    (about 2e-6 at degree 5 on seven samples), and a non-finite coordinate
    gives NaN here, or `cval` for integer arrays. Modes outside
    `_LOCAL_MODES` cannot be reproduced by a window, so they read the
    whole array.
    """
    degree = int(degree)
    prefilter = bool(prefilter) and degree > 1
    opts = dict(degree=degree, mode=mode, cval=cval, prefilter=prefilter)
    dtype = input.dtype
    if mode not in _LOCAL_MODES:
        whole = input.compute() if hasattr(input, "compute") else input
        return backends.get_ndimage_backend(whole).map_coordinates(
            whole,
            coords,
            order=degree,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        )

    nx = backends.get_array_backend(coords)
    count = int(coords.shape[1])
    finite = nx.all(nx.isfinite(coords), axis=0)
    if not bool(nx.all(finite)):
        fill = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else cval
        output = nx.full((count,), fill, dtype=dtype)
        if bool(nx.any(finite)):
            output[finite] = _map_coordinates_1block(
                coords[:, finite], input, **opts
            )
        return output
    if count == 0:
        return nx.empty((0,), dtype=dtype)

    margin = degree // 2 + 1
    starts, signals, halos, locals_ = [], [], [], []
    for c, n in zip(coords, input.shape):
        lo = int(math.floor(float(c.min()))) - margin
        hi = int(math.ceil(float(c.max()))) + margin + 1
        signal, halo, local = _axis_plan(
            lo, hi, int(n), mode, degree, prefilter
        )
        starts.append(lo)
        signals.append(signal)
        halos.append(halo)
        locals_.append(local)

    window = _gather(input, signals, cval)
    if window is None:
        return nx.full((count,), cval, dtype=dtype)
    nd = backends.get_ndimage_backend(window)
    if prefilter:
        window = nd.spline_filter(
            window, degree, output=np.float64, mode="mirror"
        )
        window = window[
            tuple(slice(h, window.shape[i] - h) for i, h in enumerate(halos))
        ]
        window = _take(window, locals_, cval)
    offset = backends.get_array_backend(coords).asarray(
        starts, dtype=coords.dtype
    )
    output = nd.map_coordinates(
        window,
        coords - offset[:, None],
        order=degree,
        mode="mirror",
        prefilter=False,
    )
    return output.astype(dtype, copy=False)


def _map_coordinates_block(
    coords: ArrayProtocol, input: ArrayProtocol, **opts
) -> ArrayProtocol:
    """Sample the input at one block of coordinates of any shape."""
    ndim, *shape = coords.shape
    flat = coords.reshape((ndim, -1))
    return _map_coordinates_1block(flat, input, **opts).reshape(shape)


def map_coordinates(
    input: ArrayProtocol,
    coordinates: ArrayProtocol,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> ArrayProtocol:
    """Lazy and edge-exact `scipy.ndimage.map_coordinates`.

    The coordinates have shape `(ndim, *shape)`, and the result is a dask
    array of shape `shape`. Each block of coordinates is one task, which
    reads the window that its stencils need, extended by the boundary
    condition and padded by the halo of the prefilter. Dask coordinates
    keep their chunks, and other coordinates are cut at the default chunk
    size of dask. The `constant` and `wrap` modes cannot be reproduced by
    a window, so in these modes each task reads the whole input.

    Unlike `dask_image.ndinterp`, which crops the input around the
    coordinates of each chunk and clips the crop to the grid, the window
    keeps the samples onto which `reflect`, `mirror` and `grid-wrap` fold,
    and the prefilter runs over the array rather than over the crop.
    """
    opts = dict(degree=order, mode=mode, cval=cval, prefilter=prefilter)
    coords = coordinates
    if isinstance(coords, da.Array):
        coords = coords.rechunk({0: -1})
    else:
        coords = da.from_array(
            coords, chunks=(-1,) + ("auto",) * (coords.ndim - 1)
        )
    if isinstance(input, da.Array):
        source = dask.delayed(da.Array)(
            input.dask, input.name, input.chunks, input.dtype
        )
    else:
        source = dask.delayed(input)
    return da.map_blocks(
        _map_coordinates_block,
        coords,
        source,
        dtype=input.dtype,
        chunks=coords.chunks[1:],
        drop_axis=0,
        **opts,
    )


_FILTER_EXTENSION = {
    "mirror": "mirror",
    "constant": "mirror",
    "grid-constant": "mirror",
    "wrap": "mirror",
    "reflect": "reflect",
    "nearest": "reflect",
    "grid-wrap": "grid-wrap",
}
"""How `scipy.ndimage.spline_filter` extends an array, per mode.

The array is extended past its edges, and the coefficients equal those of the
array extended in this way, to about 1e-15, except that SciPy approximates
`reflect` near the edges of a short axis, with errors of about 1e-4 at degree 5
on five samples.
"""


def _spline_filter1d_block(
    block: ArrayProtocol, degree: int, dim: int
) -> ArrayProtocol:
    """Prefilter one concrete block along one axis, in float64."""
    nd = backends.get_ndimage_backend(block)
    return nd.spline_filter1d(
        block, degree, axis=dim, output=np.float64, mode="mirror"
    )


def _merge_chunks(
    chunks: tx.Sequence[int], minimum: int
) -> tx.Tuple[int, ...]:
    """Merge neighbouring chunks until each one is at least `minimum` long.

    A short last chunk is merged into the previous one, and a single chunk
    that is shorter than `minimum` is left as it is.
    """
    merged: tx.List[int] = []
    for chunk in chunks:
        if merged and merged[-1] < minimum:
            merged[-1] += int(chunk)
        else:
            merged.append(int(chunk))
    if len(merged) > 1 and merged[-1] < minimum:
        merged[-2] += merged.pop()
    return tuple(merged)


def _dask_spline_filter1d(
    input: ArrayProtocol, degree: int, dim: int, extension: str
) -> ArrayProtocol:
    """Prefilter a dask array along one axis, chunk by chunk.

    The axis is extended at each end by the halo of the prefilter, as
    SciPy extends it, and each chunk is prefiltered with the halo of its
    neighbours through `map_overlap` before the extension is cut.
    """
    halo = _halo(degree)
    size = int(input.shape[dim])
    index = (slice(None),) * dim
    left = _fold(np.arange(-halo, 0), size, extension)
    right = _fold(np.arange(size, size + halo), size, extension)
    padded = da.concatenate(
        [input[index + (left,)], input, input[index + (right,)]], axis=dim
    )
    # map_overlap needs chunks at least as long as the halo and would
    # merge short ones into whole-axis chunks, so they are merged here.
    chunks = _merge_chunks(input.chunks[dim], halo)
    if min(chunks) < halo:
        chunks = (size,)
    padded = padded.rechunk({dim: (halo, *chunks, halo)})
    filtered = da.map_overlap(
        _spline_filter1d_block,
        padded,
        depth={dim: halo},
        boundary="none",
        trim=True,
        dtype=np.float64,
        degree=degree,
        dim=dim,
    )
    return filtered[index + (slice(halo, halo + size),)]


def spline_filter(
    input: ArrayProtocol,
    order: int = 3,
    output: tx.Any = np.float64,
    mode: str = "mirror",
) -> ArrayProtocol:
    """Lazy `scipy.ndimage.spline_filter`, computed chunk by chunk.

    The filter is separable, so it is applied one axis at a time. The
    result keeps the chunks of the input and is always float64.
    """
    input = da.asarray(input)
    _check_filter(order, output, mode)
    result = input.astype(np.float64)
    for axis in range(input.ndim):
        result = _dask_spline_filter1d(
            result, int(order), axis, _FILTER_EXTENSION[mode]
        )
    return result.rechunk(input.chunks)


def spline_filter1d(
    input: ArrayProtocol,
    order: int = 3,
    axis: int = -1,
    output: tx.Any = np.float64,
    mode: str = "mirror",
) -> ArrayProtocol:
    """Lazy `scipy.ndimage.spline_filter1d`, computed chunk by chunk.

    The result keeps the chunks of the input, and the output is always
    float64.
    """
    input = da.asarray(input)
    _check_filter(order, output, mode)
    axis = int(axis) % input.ndim
    result = _dask_spline_filter1d(
        input.astype(np.float64), int(order), axis, _FILTER_EXTENSION[mode]
    )
    return result.rechunk(input.chunks)


def _check_filter(degree: int, output: tx.Any, mode: str) -> None:
    """Refuse the arguments that the prefilter of SciPy refuses.

    Raises
    ------
    RuntimeError
        If the order is not supported.
    ValueError
        If the mode is unknown.
    TypeError
        If the output type is not float64.
    """
    if int(degree) not in _SPLINE_POLES:
        raise RuntimeError("spline order not supported")
    if mode not in _FILTER_EXTENSION:
        raise ValueError(f"Unknown spline boundary mode {mode!r}.")
    if np.dtype(output) != np.float64:
        raise TypeError("A dask array is prefiltered into float64 only.")
