# stdlib
import itertools
import math
from types import ModuleType

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayLike, ArrayProtocol

# core
from brainhops._core.dependencies import da, dk, np
from brainhops.backends import (
    best_backend,
    copy_array,
    get_array_backend,
    get_ndimage_backend,
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
    nd = get_ndimage_backend(block)
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


def _spline_filter(
    input: ArrayProtocol, nd: ModuleType, order: int, mode: str
) -> ArrayProtocol:
    """Apply an ndimage spline prefilter, chunk by chunk for a dask array.

    `dask_image.spline_filter` prefilters through `map_overlap` with an
    overlap sized for a tolerance of 1e-8 -- its coefficients are off by
    up to 1e-6 at order five -- refuses the `wrap` mode, and cannot build
    the overlap when an axis is shorter than it. It is not used.

    The prefilter is separable, so a dask array is prefiltered one axis at
    a time (see [`_dask_spline_filter1d`][]), each chunk reading only the
    halo of its neighbours along that axis. The result keeps the input's
    chunks, never needs the whole array at once, and is scipy's to
    `_PREFILTER_TOLERANCE`.
    """
    if da is None or not isinstance(input, da.Array):
        return nd.spline_filter(input, order=order, mode=mode)
    if int(order) not in _SPLINE_POLES:
        raise RuntimeError("spline order not supported")
    if mode not in _FILTER_EXTENSION:
        raise ValueError(f"Unknown spline boundary mode {mode!r}.")
    output = input.astype(np.float64)
    for dim in range(input.ndim):
        output = _dask_spline_filter1d(
            output, int(order), dim, _FILTER_EXTENSION[mode]
        )
    return output.rechunk(input.chunks)


def _scipy_boundary(bound: tx.Union[str, float]) -> tx.Tuple[str, float]:
    """Translate a boundary condition into a scipy ``(mode, cval)`` pair.

    A constant boundary maps to scipy's ``"grid-constant"`` mode, whether
    it is named (the ``"constant"`` condition) or given as a numeric fill
    value. That mode treats every coordinate beyond the grid as the fill
    value for a spline of any order, which is the zero-padding an FNIRT
    coefficient field is evaluated with. Scipy's ``"constant"`` mode is not
    equivalent for an order above one, so it is not used. Every other
    named condition passes through unchanged.
    """
    if isinstance(bound, str):
        return ("grid-constant" if bound == "constant" else str(bound)), 0.0
    return "grid-constant", bound


def _autoreshape(map_coordinates: tx.Callable) -> tx.Callable:

    def _map_coordinates(
        input: ArrayProtocol, coords: ArrayProtocol, **kwargs
    ) -> ArrayProtocol:
        ndim, *oshape = coords.shape
        coords = coords.reshape((ndim, -1))
        output = map_coordinates(input, coords, **kwargs)
        return output.reshape(oshape)

    return _map_coordinates


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
    nx = get_array_backend(array)
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
        return get_ndimage_backend(whole).map_coordinates(
            whole, coords, **opts
        )

    nx = get_array_backend(coords)
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
    nd = get_ndimage_backend(window)
    if prefilter:
        window = nd.spline_filter(
            window, order, output=np.float64, mode="mirror"
        )
        window = window[
            tuple(slice(h, window.shape[i] - h) for i, h in enumerate(halos))
        ]
        window = _take(window, locals_, cval)
    offset = get_array_backend(coords).asarray(starts, dtype=coords.dtype)
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


def _dask_map_coordinates(
    input: ArrayProtocol, coords: ArrayProtocol, **opts
) -> ArrayProtocol:
    """``map_coordinates`` for a dask array, lazy and exact at the edges.

    `coords` has shape `(ndim, *shape)` and the result has shape `shape`.

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
    if isinstance(coords, da.Array):
        coords = coords.rechunk({0: -1})
    else:
        coords = da.from_array(
            coords, chunks=(-1,) + ("auto",) * (coords.ndim - 1)
        )
    if isinstance(input, da.Array):
        source = dk.delayed(da.Array)(
            input.dask, input.name, input.chunks, input.dtype
        )
    else:
        source = dk.delayed(input)
    return da.map_blocks(
        _map_coordinates_block,
        coords,
        source,
        dtype=input.dtype,
        chunks=coords.chunks[1:],
        drop_axis=0,
        **opts,
    )


def _map_coordinates_for(nx: ModuleType, nd: ModuleType) -> tx.Callable:
    """The (autoreshaping) ``map_coordinates`` to use for a backend.

    A dask array is sampled by [`_dask_map_coordinates`][], which is exact
    for coordinates outside of the grid, rather than by `dask_image`'s,
    which is not -- and which older `dask-image` releases do not have.
    Every other backend uses its own ndimage package.
    """
    if da is not None and nx is da:
        return _dask_map_coordinates
    return _autoreshape(nd.map_coordinates)


def _over_batch(
    nx: ModuleType,
    input: ArrayProtocol,
    batch: tx.Tuple[int, ...],
    func: tx.Callable[[ArrayProtocol], ArrayProtocol],
    shape: tx.Tuple[int, ...],
    output: tx.Optional[ArrayProtocol] = None,
) -> ArrayProtocol:
    """Apply `func` to each `input[index]` of a batch, into `(*batch, *shape)`.

    A concrete array is filled item by item, into `output` when given. A
    dask array is assembled by stacking the results instead: assigning one
    dask array into another rebuilds the graph of every chunk it covers,
    which takes minutes on a finely chunked array.
    """
    indices = itertools.product(*[range(s) for s in batch])
    if da is not None and nx is da:
        dtype = input.dtype
        parts = [
            nx.asarray(func(input[index])).astype(dtype) for index in indices
        ]
        if not batch:
            return parts[0]
        return nx.stack(parts).reshape(tuple(batch) + tuple(shape))
    if output is None:
        output = nx.empty_like(input, shape=tuple(batch) + tuple(shape))
    for index in indices:
        output[index] = func(input[index])
    return output


def pull(
    input: ArrayProtocol,
    coords: ArrayProtocol,
    order: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> ArrayProtocol:
    """
    Interpolate an array using a coordinates field.

    Parameters
    ----------
    input : array-like
        The array to be interpolated. Shape (*batch, *spatial_in)
    coords : array-like
        The coordinates field. Shape (*spatial_out, ndim)
    order : {0..5}
        The interpolation order. 0=nearest, 1=linear, 2=quadratic, etc.
    bound : {'nearest', 'reflect', 'mirror', 'grid-wrap', 'wrap'} or float
        The boundary condition. If a string, one of:
        - 'nearest': nearest edge value   (a a a a | a b c d | d d d d)
        - 'reflect': reflect at edge      (d c b a | a b c d | d c b a)
        - 'mirror': mirror at edge        (d c b | a b c d | c b a)
        - 'grid-wrap': wrap around        (a b c d | a b c d | a b c d)
        - 'wrap': wrap around with shift  (d b c d | a b c d | b c a b)
        If a float, the constant value to use beyond the edge.
    coeff : bool
        If True, the input image is assumed to already contain spline
        coefficients. If False, the input image will be prefiltered
        before interpolation.

    Returns
    -------
    array-like
        The interpolated array. Shape (*batch, *spatial_out)

    """
    # Get backends
    nx = best_backend(input, coords)
    nd = get_ndimage_backend(nx)
    # Get dimensions
    ndim = coords.shape[-1]
    batch = input.shape[:-ndim]
    # Prepare for map_coordinates
    coords = nx.moveaxis(coords, -1, 0)
    mode, cval = _scipy_boundary(bound)
    order = int(order)
    opts = {"order": order, "mode": mode, "cval": cval, "prefilter": not coeff}
    # Interpolate each batch
    map_coordinates = _map_coordinates_for(nx, nd)
    return _over_batch(
        nx,
        input,
        batch,
        lambda x: map_coordinates(x, coords, **opts),
        tuple(coords.shape[1:]),
    )


def pull_axes(
    input: ArrayProtocol,
    coords: ArrayProtocol,
    axes: tx.Sequence[int],
    order: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> ArrayProtocol:
    """
    Interpolate an array along a subset of its axes.

    The named axes are the spatial axes that the coordinates field
    addresses. They are moved to the end of the array, the interpolation
    is applied over them, and every other axis is carried along as a
    batch dimension. This samples a low-dimensional field over a few axes
    of a larger array without building a full-dimensional coordinates
    field over the whole array.

    Parameters
    ----------
    input : array-like
        The array to interpolate. Shape `(*d0, *d1, ...)`.
    coords : array-like
        The coordinates field. Shape `(*spatial_out, len(axes))`.
    axes : sequence of int
        The axes of `input` that the coordinates field addresses, in the
        order the components of the field address them.
    order : {0..5}
        The interpolation order.
    bound : str or float
        The boundary condition, as accepted by [`pull`][].
    coeff : bool
        Whether the input already contains spline coefficients.

    Returns
    -------
    array-like
        The interpolated array. Its leading axes are the axes of `input`
        that were not named, kept in their original order, followed by
        the output spatial axes of the field. Shape
        `(*batch, *spatial_out)`.
    """
    # Get backends
    nx = best_backend(input, coords)
    axes = [int(a) % input.ndim for a in axes]
    ndim = len(axes)
    dest = list(range(input.ndim - ndim, input.ndim))
    moved = nx.moveaxis(input, axes, dest)
    return pull(moved, coords, order, bound, coeff)


def spline_matrix(
    n_in: int,
    coords_1d: ArrayProtocol,
    order: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> ArrayProtocol:
    """
    Build the weight matrix of a one-dimensional spline resampling.

    The result is a matrix `W` of shape `(n_out, n_in)`, where `n_out`
    is the number of output samples in `coords_1d`. Applying `W` to a
    one-dimensional array of length `n_in` reproduces the interpolation
    that [`pull`][] performs at the same coordinates, because spline
    interpolation is linear in the sample values. The prefilter is baked
    into `W` when `coeff` is false, so the matrix maps values, not
    coefficients, unless `coeff` is true.

    The matrix is applied along one axis of a larger array with a single
    contraction, which replaces a batched call to [`pull`][] over that
    axis. This is faster than a batched pull for a one-dimensional step.

    A constant boundary cannot be expressed as a linear weight, so `W`
    carries only the in-bounds contribution when `bound` is a float. An
    output sample that reads beyond the grid then has a row that sums to
    less than one, and the constant fill is added separately as
    `cval * (1 - W.sum(axis=1))`. This is the partition-of-unity
    correction, and omitting it leaves the constant-boundary result wrong
    by an amount proportional to the fill value.

    Parameters
    ----------
    n_in : int
        The length of the input axis.
    coords_1d : array-like
        The output coordinates along the axis. Shape `(n_out,)`.
    order : {0..5}
        The interpolation order.
    bound : str or float
        The boundary condition, as accepted by [`pull`][]. A float
        selects a constant boundary, and the returned matrix then holds
        only the in-bounds weights.
    coeff : bool
        Whether the input already contains spline coefficients.

    Returns
    -------
    array-like
        The weight matrix. Shape `(n_out, n_in)`.
    """
    nx = get_array_backend(coords_1d)
    bound0 = bound if isinstance(bound, str) else 0.0
    basis = nx.eye(n_in)
    coords = nx.reshape(nx.asarray(coords_1d), (-1, 1))
    return pull(basis, coords, order, bound0, coeff).T


def pull_field(
    field: ArrayLike,
    coords: ArrayLike,
    order: int,
    bound: tx.Literal["nearest", "reflect", "mirror", "grid-wrap", "wrap"],
    coeff: bool,
) -> ArrayLike:
    """
    Interpolate an displacement or coordinates field using a coordinates field.

    Parameters
    ----------
    field : array-like
        The field to be interpolated. Shape (*batch, *spatial_in, ndim)
    coords : array-like
        The coordinates field. Shape (*spatial_out, ndim)
    order : {0..5}
        The interpolation order. 0=nearest, 1=linear, 2=quadratic, etc.
    bound : {'nearest', 'reflect', 'mirror', 'grid-wrap', 'wrap'} or float
        The boundary condition. If a string, one of:
        - 'nearest': nearest edge value   (a a a a | a b c d | d d d d)
        - 'reflect': reflect at edge      (d c b a | a b c d | d c b a)
        - 'mirror': mirror at edge        (d c b | a b c d | c b a)
        - 'grid-wrap': wrap around        (a b c d | a b c d | a b c d)
        - 'wrap': wrap around with shift  (d b c d | a b c d | b c a b)
        If a float, the constant value to use beyond the edge.
    coeff : bool
        If True, the input image is assumed to already contain spline
        coefficients. I.e., it has already been prefiltered with the
        appropriate spline filter. If False, the input image will be
        prefiltered before interpolation.

    Returns
    -------
    array-like
        The interpolated field. Shape (*batch, *spatial_out, ndim)

    """
    nx = best_backend(field, coords)
    field = nx.moveaxis(field, -1, 0)
    field = pull(field, coords, order, bound, coeff)
    field = nx.moveaxis(field, 0, -1)
    return field


def coeff2value(
    input: ArrayLike,
    order: int,
    bound: tx.Literal["nearest", "reflect", "mirror", "grid-wrap", "wrap"],
    inplace: bool = False,
    ndim: tx.Optional[int] = None,
) -> ArrayLike:
    """
    Convert an array of spline coefficients to values by applying the
    appropriate inverse spline filter.

    This is equivalent to (and implemented as) interpolating the field
    at the center of each voxel.

    Parameters
    ----------
    input : array-like
        The array of spline coefficients. Shape (*batch, *spatial)
    order : {0..5}
        The interpolation order. 0=nearest, 1=linear, 2=quadratic, etc.
    bound : {'nearest', 'reflect', 'mirror', 'grid-wrap', 'wrap'} or float
        The boundary condition. If a string, one of:
        - 'nearest': nearest edge value   (a a a a | a b c d | d d d d)
        - 'reflect': reflect at edge      (d c b a | a b c d | d c b a)
        - 'mirror': mirror at edge        (d c b | a b c d | c b a)
        - 'grid-wrap': wrap around        (a b c d | a b c d | a b c d)
        - 'wrap': wrap around with shift  (d b c d | a b c d | b c a b)
        If a float, the constant value to use beyond the edge.
    inplace : bool
        If True, the conversion will be done in-place (i.e. the input
        array will be modified). If False, a new array will be created
        for the output.
    ndim : int, optional
        The number of spatial dimensions. If None, all input dimensions
        are assumed to be spatial. If an integer, the last `ndim`
        dimensions are assumed to be spatial.

    Returns
    -------
    array-like
        The array of values. Shape (*batch, *spatial)
    """
    # Splines of order 0 and 1 interpolate their coefficients: the
    # coefficients are the values.
    if int(order) < 2:
        return input if inplace else copy_array(input)
    # Get packages
    nx = get_array_backend(input)
    nd = get_ndimage_backend(nx)
    # Get dimensions
    # NOTE: `ndim or input.ndim` treats ndim=0 (or None) as "all dimensions
    # are spatial"; a genuine ndim=0 is meaningless for interpolation, so
    # collapsing it into the "all" case is intentional.
    ndim = ndim or input.ndim
    batch = input.shape[:-ndim]
    # Create coordinates field for interpolation, sampling at the center of
    # each voxel of the *spatial* dimensions (the last `ndim` axes).
    grid = nx.meshgrid(
        *(nx.arange(s) for s in input.shape[-ndim:]), indexing="ij"
    )
    grid = nx.stack(grid, axis=0)
    # Prepare for map_coordinates
    mode, cval = _scipy_boundary(bound)
    order = int(order)
    opts = {"order": order, "mode": mode, "cval": cval, "prefilter": False}
    map_coordinates = _map_coordinates_for(nx, nd)
    return _over_batch(
        nx,
        input,
        batch,
        lambda x: map_coordinates(x, grid, **opts),
        tuple(input.shape[-ndim:]),
        output=input if inplace else None,
    )


def coeff2value_field(
    field: ArrayLike,
    order: int,
    bound: tx.Literal["nearest", "reflect", "mirror", "grid-wrap", "wrap"],
    inplace: bool = False,
) -> ArrayLike:
    """
    Convert a field of spline coefficients to values by applying the
    appropriate inverse spline filter.

    This is equivalent to (and implemented as) interpolating the field
    at the center of each voxel.

    Parameters
    ----------
    field : array-like
        The field of spline coefficients. Shape (*batch, *spatial, ndim)
    order : {0..5}
        The interpolation order. 0=nearest, 1=linear, 2=quadratic, etc.
    bound : {'nearest', 'reflect', 'mirror', 'grid-wrap', 'wrap'} or float
        The boundary condition. If a string, one of:
        - 'nearest': nearest edge value   (a a a a | a b c d | d d d d)
        - 'reflect': reflect at edge      (d c b a | a b c d | d c b a)
        - 'mirror': mirror at edge        (d c b | a b c d | c b a)
        - 'grid-wrap': wrap around        (a b c d | a b c d | a b c d)
        - 'wrap': wrap around with shift  (d b c d | a b c d | b c a b)
        If a float, the constant value to use beyond the edge.
    inplace : bool
        If True, the conversion will be done in-place (i.e. the input
        array will be modified). If False, a new array will be created
        for the output.

    Returns
    -------
    array-like
        The field of values. Shape (*batch, *spatial, ndim)
    """
    nx = get_array_backend(field)
    ndim = field.shape[-1]
    field = nx.moveaxis(field, -1, 0)
    field = coeff2value(field, order, bound, inplace=inplace, ndim=ndim)
    field = nx.moveaxis(field, 0, -1)
    return field


def value2coeff(
    input: ArrayLike,
    order: int,
    bound: tx.Literal["nearest", "reflect", "mirror", "grid-wrap", "wrap"],
    inplace: bool = False,
    ndim: tx.Optional[int] = None,
) -> ArrayLike:
    """
    Convert an array of values to spline coefficients by applying the
    appropriate spline filter.

    Parameters
    ----------
    input : array-like
        The array of values. Shape (*batch, *spatial)
    order : {0..5}
        The interpolation order. 0=nearest, 1=linear, 2=quadratic, etc.
    bound : {'nearest', 'reflect', 'mirror', 'grid-wrap', 'wrap'} or float
        The boundary condition. If a string, one of:
        - 'nearest': nearest edge value   (a a a a | a b c d | d d d d)
        - 'reflect': reflect at edge      (d c b a | a b c d | d c b a)
        - 'mirror': mirror at edge        (d c b | a b c d | c b a)
        - 'grid-wrap': wrap around        (a b c d | a b c d | a b c d)
        - 'wrap': wrap around with shift  (d b c d | a b c d | b c a b)
        If a float, the constant value to use beyond the edge.
    inplace : bool
        If True, the conversion will be done in-place (i.e. the input
        array will be modified). If False, a new array will be created
        for the output.
    ndim : int, optional
        The number of spatial dimensions. If None, all input dimensions
        are assumed to be spatial. If an integer, the last `ndim`
        dimensions are assumed to be spatial.

    Returns
    -------
    array-like
        The array of spline coefficients. Shape (*batch, *spatial)
    """
    # Splines of order 0 and 1 interpolate their coefficients: the
    # coefficients are the values.
    if int(order) < 2:
        return input if inplace else copy_array(input)
    # Get packages
    nx = get_array_backend(input)
    nd = get_ndimage_backend(input)
    # Get dimensions
    # NOTE: `ndim or input.ndim` treats ndim=0 (or None) as "all dimensions
    # are spatial"; a genuine ndim=0 is meaningless for a spline filter, so
    # collapsing it into the "all" case is intentional.
    ndim = ndim or input.ndim
    batch = input.shape[:-ndim]
    # Prepare for spline_filter.
    # `spline_filter` only accepts `order`, `output` and `mode` -- there is no
    # `cval` argument, so a float `bound` (a constant fill value) cannot be
    # forwarded. We prefilter a float bound with mode='constant', matching how
    # `coeff2value`/`map_coordinates` treat it, which keeps the round trip
    # exact for string bounds and consistent (if not a strict inverse) for
    # float bounds.
    mode = bound if isinstance(bound, str) else "constant"
    opts = dict(order=order, mode=mode)
    return _over_batch(
        nx,
        input,
        batch,
        lambda x: _spline_filter(x, nd, **opts),
        tuple(input.shape[-ndim:]),
        output=input if inplace else None,
    )


def value2coeff_field(
    field: ArrayLike,
    order: int,
    bound: tx.Literal["nearest", "reflect", "mirror", "grid-wrap", "wrap"],
    inplace: bool = False,
) -> ArrayLike:
    """
    Convert a field of values to spline coefficients by applying the
    appropriate spline filter.

    Parameters
    ----------
    field : array-like
        The field of values. Shape (*batch, *spatial, ndim)
    order : {0..5}
        The interpolation order. 0=nearest, 1=linear, 2=quadratic, etc.
    bound : {'nearest', 'reflect', 'mirror', 'grid-wrap', 'wrap'} or float
        The boundary condition. If a string, one of:
        - 'nearest': nearest edge value   (a a a a | a b c d | d d d d)
        - 'reflect': reflect at edge      (d c b a | a b c d | d c b a)
        - 'mirror': mirror at edge        (d c b | a b c d | c b a)
        - 'grid-wrap': wrap around        (a b c d | a b c d | a b c d)
        - 'wrap': wrap around with shift  (d b c d | a b c d | b c a b)
        If a float, the constant value to use beyond the edge.
    inplace : bool
        If True, the conversion will be done in-place (i.e. the input
        array will be modified). If False, a new array will be created
        for the output.

    Returns
    -------
    array-like
        The field of spline coefficients. Shape (*batch, *spatial, ndim)
    """
    nx = get_array_backend(field)
    ndim = field.shape[-1]
    field = nx.moveaxis(field, -1, 0)
    field = value2coeff(field, order, bound, inplace=inplace, ndim=ndim)
    field = nx.moveaxis(field, 0, -1)
    return field
