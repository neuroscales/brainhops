"""
The ndimage functions of the dask backend: interpolation and prefilter.

They follow `scipy.ndimage` -- same arguments, same results, to about
1e-12 -- on dask arrays, lazily, and never read the whole array: a block
of coordinates reads a window of the input around its stencils, and a
chunk of spline coefficients reads the halo of its neighbours. Each
window is extended past the edges of the array the way scipy extends the
whole array, so the boundary conditions are scipy's.

They replace `dask_image.ndinterp`, whose `map_coordinates` reads outside
of an empty crop for coordinates beyond the start of an axis, and whose
`spline_filter` is accurate to 1e-6 only, refuses the `wrap` mode, and
cannot overlap an axis shorter than its halo.
"""

# stdlib
import math

# dependencies
import dask
import dask.array as da
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# core
# `brainhops.backends` imports this module while it is itself being
# imported, so its functions are looked up when they are called.
import brainhops.backends as backends

__all__ = ["map_coordinates", "spline_filter", "spline_filter1d"]


_SPLINE_POLES = {
    2: 0.17157287525380990,
    3: 0.26794919243112270,
    4: 0.36134122590022018,
    5: 0.43057534709997379,
}
"""
The magnitude of the dominant pole of the b-spline prefilter, per order.

The prefilter is a recursive filter: the coefficient at one sample depends
on a sample `k` steps away through a weight that decays as `pole ** k`.
"""

_PREFILTER_TOLERANCE = 1e-12
"""The weight below which a sample is left out of a coefficient."""

_SCIPY_PREPAD = {"nearest": 12, "grid-constant": 12}
"""
The samples scipy pads an array with before it prefilters it, per mode.

`scipy.ndimage.map_coordinates` pads by twelve edge samples for `nearest`,
and by twelve `cval` samples for `grid-constant`, and prefilters the padded
array (`_prepad_for_spline_filter`). Its other modes prefilter the array
as it is.
"""

_LOCAL_MODES = ("nearest", "reflect", "mirror", "grid-wrap", "grid-constant")
"""The modes a block of coordinates can be sampled with from a window."""


def _halo(order: int) -> int:
    """
    The samples a prefilter of this order reads on each side of a window.

    Beyond it, a sample weighs less than `_PREFILTER_TOLERANCE` in any
    coefficient of the window, so the coefficients of a window padded by
    this halo are those of the whole extended array.
    """
    pole = _SPLINE_POLES.get(int(order))
    if pole is None:
        return 0
    return int(math.ceil(math.log(_PREFILTER_TOLERANCE) / math.log(pole)))


def _fold(k: np.ndarray, n: int, mode: str) -> np.ndarray:
    """
    The sample that each index `k` of an extended axis of length `n` holds.

    The axis is extended by `mode`, as `scipy.ndimage` extends it, and an
    index that the extension fills with `cval` is `-1`.
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
    lo: int, hi: int, n: int, mode: str, order: int, prefilter: bool
) -> tx.Tuple[np.ndarray, int, np.ndarray]:
    """
    How to sample one axis where the stencils read indices `[lo, hi)`.

    Returns the samples of the input to read (`-1` for `cval`), the halo
    to cut from each end of the window once it is prefiltered, and, for
    each index in `[lo, hi)`, the position of its coefficient in the cut
    window (`-1` for `cval`).

    Without a prefilter the stencils read the samples directly, extended
    by `mode`. With one, they read spline coefficients, which scipy takes
    from the array extended by `mode` -- after padding it, for `nearest`
    and `grid-constant` -- so the window is that extension around the
    indices read, plus a halo the prefilter runs over and that is then
    cut. The coefficients beyond scipy's padded array are its edge
    coefficient for `nearest`, and `cval` for `grid-constant`.
    """
    index = np.arange(lo, hi)
    if not prefilter:
        return _fold(index, n, mode), 0, np.arange(hi - lo)
    pad = _SCIPY_PREPAD.get(mode, 0)
    padded = n + 2 * pad
    halo = _halo(order)
    index = index + pad
    if pad:
        coeff = _fold(index, padded, mode)
        used = coeff[coeff >= 0]
        if used.size == 0:
            return np.full(1, -1), 0, np.full(hi - lo, -1)
        start, stop = int(used.min()), int(used.max()) + 1
        local = np.where(coeff >= 0, coeff - start, -1)
        # The padded array is extended past its own edge by scipy's
        # prefilter: `reflect` for `nearest`, `mirror` for `grid-constant`.
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
    """
    `array[np.ix_(*positions)]`, where a position of `-1` reads `cval`.
    """
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
    """
    `input[np.ix_(*indices)]`, materialized, where `-1` reads `cval`.

    Only the samples that are read are computed: each axis is sliced to
    the indices it uses, by a slice when they are contiguous and by a list
    otherwise -- such as the two ends of a `grid-wrap` axis. `None` when an
    axis reads no sample at all, and the whole window is `cval`.
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
    order: int,
    mode: str,
    cval: float,
    prefilter: bool,
) -> ArrayProtocol:
    """Sample `input` at one materialized block of `(ndim, N)` coordinates.

    The block reads a window of `input` around the stencils of its
    coordinates, extended by `mode` the way scipy extends the whole array
    (see [`_axis_plan`][]), so it never reads more than its stencils and,
    with a prefilter, a halo around them. The window is prefiltered, when
    asked, and sampled with its own concrete (scipy or cupy) ndimage.

    The result is scipy's on the whole array, to `_PREFILTER_TOLERANCE`,
    with two exceptions. scipy approximates its `reflect` prefilter near
    the edges of a short axis -- by about 2e-6 at order five on seven
    samples -- and this does not. A point
    with a coordinate that is not finite is no position at all, and scipy
    returns `cval`, NaN or an edge sample for it depending on the mode and
    the order: here it is NaN, or `cval` for an array of integers.

    The `wrap` mode does not extend the array in any way a window can
    reproduce, so it reads the whole array, and so does any mode this
    module does not know.
    """
    order = int(order)
    prefilter = bool(prefilter) and order > 1
    opts = dict(order=order, mode=mode, cval=cval, prefilter=prefilter)
    dtype = input.dtype
    if mode not in _LOCAL_MODES:
        whole = input.compute() if hasattr(input, "compute") else input
        return backends.get_ndimage_backend(whole).map_coordinates(
            whole, coords, **opts
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

    margin = order // 2 + 1
    starts, signals, halos, locals_ = [], [], [], []
    for c, n in zip(coords, input.shape):
        lo = int(math.floor(float(c.min()))) - margin
        hi = int(math.ceil(float(c.max()))) + margin + 1
        signal, halo, local = _axis_plan(
            lo, hi, int(n), mode, order, prefilter
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
            window, order, output=np.float64, mode="mirror"
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
        order=order,
        mode="mirror",
        prefilter=False,
    )
    return output.astype(dtype, copy=False)


def _map_coordinates_block(
    coords: ArrayProtocol, input: ArrayProtocol, **opts
) -> ArrayProtocol:
    """Sample `input` at one `(ndim, *shape)` block of coordinates."""
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
    """`scipy.ndimage.map_coordinates`, lazy and exact at the edges.

    `coordinates` has shape `(ndim, *shape)` and the result has shape
    `shape`. Either may be a dask array; the result is one.

    `dask_image.ndinterp.map_coordinates` is not used. It crops the input
    around the coordinates of each chunk and clips that crop to the grid,
    so a chunk whose coordinates all lie beyond the start of an axis gets
    an empty crop, and `scipy` then reads outside of a zero-length array:
    the result is whatever memory happens to hold. A crop clipped at the
    end of an axis is not empty, but it drops the samples that `reflect`,
    `mirror` or `grid-wrap` fold back onto, and its prefilter runs over
    the crop rather than over the array.

    Here each block of coordinates is sampled by one task that reads a
    window of the input around its stencils, extended by the boundary
    condition as scipy extends the whole array, and padded by the halo a
    prefilter needs (see [`_map_coordinates_1block`][]). A block never
    reads more of the input than that, wherever its points are -- a
    `grid-wrap` window at an edge reads a few samples from each end.

    A window covers the points of its block, so the blocks keep the
    spatial layout of the coordinates: dask coordinates keep their own
    chunks, and any other array is cut into dask's default chunk size,
    rather than sampled as one block that would read the whole input. The
    input is passed to the tasks as a delayed rebuild of itself, so each
    task only computes its window.
    """
    opts = dict(order=order, mode=mode, cval=cval, prefilter=prefilter)
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
"""
How `scipy.ndimage.spline_filter` extends an array past its edges, per mode.

Its coefficients are those of the array extended this way, to about 1e-15,
except that it approximates the `reflect` extension near the edges of a
short axis: by about 1e-4 at order five on five samples.
"""


def _spline_filter1d_block(
    block: ArrayProtocol, order: int, dim: int
) -> ArrayProtocol:
    """Prefilter one materialized block along one axis."""
    nd = backends.get_ndimage_backend(block)
    return nd.spline_filter1d(
        block, order, axis=dim, output=np.float64, mode="mirror"
    )


def _merge_chunks(
    chunks: tx.Sequence[int], minimum: int
) -> tx.Tuple[int, ...]:
    """Merge neighbouring chunks until each is at least `minimum` long.

    The last chunk is merged into the one before it when it is short. A
    single chunk shorter than `minimum` is left as it is.
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
    input: ArrayProtocol, order: int, dim: int, extension: str
) -> ArrayProtocol:
    """Prefilter a dask array along one axis, chunk by chunk.

    The axis is extended by the prefilter's halo (see [`_halo`][]) at
    each end, the way scipy extends it: a few samples, read from the same
    end of the axis or, for `grid-wrap`, from the other end. Each chunk is
    then prefiltered with the halo of its neighbours, which dask shares
    through the graph (`map_overlap`), and the extension is cut. Chunks
    shorter than the halo are merged with their neighbours first, until
    they are as long as it.
    """
    halo = _halo(order)
    size = int(input.shape[dim])
    index = (slice(None),) * dim
    left = _fold(np.arange(-halo, 0), size, extension)
    right = _fold(np.arange(size, size + halo), size, extension)
    padded = da.concatenate(
        [input[index + (left,)], input, input[index + (right,)]], axis=dim
    )
    # `map_overlap` needs chunks at least as long as the halo, and merges
    # shorter ones itself -- into chunks as long as the whole axis. So
    # each run of short chunks is merged with its neighbours here, and the
    # two halos are one chunk each.
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
        order=order,
        dim=dim,
    )
    return filtered[index + (slice(halo, halo + size),)]


def spline_filter(
    input: ArrayProtocol,
    order: int = 3,
    output: tx.Any = np.float64,
    mode: str = "mirror",
) -> ArrayProtocol:
    """`scipy.ndimage.spline_filter`, chunk by chunk on a dask array.

    The prefilter is separable, so the array is prefiltered one axis at a
    time (see [`spline_filter1d`][]), each chunk reading only the halo of
    its neighbours along that axis. The result keeps the input's chunks,
    never needs the whole array at once, and is scipy's to
    `_PREFILTER_TOLERANCE`. It is always of type `float64`.
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
    """`scipy.ndimage.spline_filter1d`, chunk by chunk on a dask array.

    See [`_dask_spline_filter1d`][]. The result keeps the input's chunks
    and is always of type `float64`.
    """
    input = da.asarray(input)
    _check_filter(order, output, mode)
    axis = int(axis) % input.ndim
    result = _dask_spline_filter1d(
        input.astype(np.float64), int(order), axis, _FILTER_EXTENSION[mode]
    )
    return result.rechunk(input.chunks)


def _check_filter(order: int, output: tx.Any, mode: str) -> None:
    """Refuse what scipy's prefilter refuses, and an output it cannot be."""
    if int(order) not in _SPLINE_POLES:
        raise RuntimeError("spline order not supported")
    if mode not in _FILTER_EXTENSION:
        raise ValueError(f"Unknown spline boundary mode {mode!r}.")
    if np.dtype(output) != np.float64:
        raise TypeError("A dask array is prefiltered into float64 only.")
