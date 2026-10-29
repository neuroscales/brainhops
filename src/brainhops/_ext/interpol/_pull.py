# This file is part of a port of torch-interpol 0.3.0 to NumPy and
# CuPy. Copyright (c) 2021 Yael Balbastre. Released under the MIT
# license, whose text is in the LICENSE file of this package.
"""Interpolation of an array at arbitrary coordinates."""

# stdlib
import itertools
import math
from types import ModuleType

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from ._backend import BufferPool, get_namespace, is_dask, to_namespace
from ._bounds import normalize_bound
from ._coeff import (
    normalize_dtype,
    normalize_order,
    normalize_per_axis,
    spline_coeff,
)
from ._taps import axis_taps

_TOLERANCE = 5e-2
"""Margin, in voxels, that the mask of `extrapolate=False` allows.

It keeps the samples that rounding errors put just outside the edge
samples. The value comes from torch-interpol.
"""


_BLOCK_VALUES = 65536
"""Default number of values in each temporary array of a block.

The blocked strategy of [`pull`][] processes the output in blocks small
enough that the arrays of a block stay in the cache of the processor
while all the taps are added. A block holds at most `_BLOCK_POINTS`
points, and as many batch items as fit in this number of values. With
float32 values, the arrays of one tap then take 256 KiB, a quarter of
the 1 MiB second-level cache of each core of the machine on which the
value was measured.
"""

_BLOCK_POINTS = 16384
"""Most points that a block of the blocked strategy holds.

The indices and the weights of the taps of a block are computed for
its points, and their size grows with the number of points and with
the order, but not with the number of batch items.
"""

_TAP_SCRATCH = (
    "first",
    "fraction",
    "base",
    "scratch",
    "integer scratch",
    "inside",
    "mask",
    "distance",
)
"""Names of the arrays of the pool that only the taps of a block use."""

_STRATEGIES = ("auto", "blocked", "batched")
"""The values of the argument `strategy` of [`pull`][]."""


def pull(
    input: tx.Any,
    grid: tx.Any,
    order: tx.Any = 1,
    bound: tx.Any = "reflect",
    extrapolate: tx.Any = True,
    prefilter: bool = False,
    strategy: str = "auto",
    block_size: tx.Optional[int] = None,
) -> tx.Any:
    """Interpolate an array at the coordinates of a grid.

    Use this function to resample an image with a transformation: the
    grid holds, for each output voxel, the coordinates of the point of
    the input that it takes its value from. The interpolation is a
    B-spline of the given order, separately along each axis, and the
    boundary condition says how the input continues beyond its edges.
    The computation runs with the package of the arrays, NumPy or CuPy,
    and every intermediate array is created with that package.

    Coordinates are in voxels, and the sample `i` of an axis lies at the
    coordinate `i`. The length `ndim` of the last axis of the grid is the
    number of spatial axes. The last `ndim` axes of the input are its
    spatial axes, and the `ndim` axes of the grid before its last one are
    the spatial axes of the output. A list of scattered points therefore
    needs a grid such as `(npoints, 1, 1, 3)` in three dimensions. The
    other, leading axes of the input and of the grid are batch axes,
    which broadcast against each other with the rules of NumPy. This
    convention is the one of torch-interpol.

    Each output value is a sum over the `order + 1` samples, or taps,
    that each axis reads. The code loops in Python over these taps and
    over blocks of the output, and each tap gathers the samples of all
    the points and batch items of a block at once. The two strategies
    differ in the size of the blocks:

    - The blocked strategy splits the output points, and the batch
      items when there are many, into blocks of at most `block_size`
      values, and adds all the taps of a block before it moves on to
      the next one. The indices and the weights of the taps are
      computed once per block of points and reused for all its batch
      items. With the default size, the arrays of a block stay in the
      cache of the processor, so that each tap reads and writes the
      cache instead of the main memory. On a processor, this made the
      resampling of volumes of 128 to 192 voxels per side 1.6 to 2.4
      times faster than the batched strategy, and the temporary arrays
      take a few megabytes instead of hundreds.
    - The batched strategy makes a single block of all the points and
      all the batch items. Each step is then one operation on whole
      arrays, which suits a GPU, where every operation launches a
      kernel and the memory is fast, so that more blocks would only add
      launches. Its temporary arrays take about `2 * ndim * (order + 1)`
      arrays of the size of the grid, for the indices and the weights,
      and two arrays of the size of the output.

    The default strategy, "auto", is the blocked one for NumPy arrays
    and the batched one for CuPy arrays. Both give the same values.

    When the input or the grid is a dask array, the function returns a
    dask array of the same values, which dask computes block by block.
    The output has the chunks of the grid along its spatial axes. Each
    block of the output is one task, which reads only the window of the
    input that its coordinates need, and runs the computation above on
    that window. With the prefilter, the coefficients of the window are
    computed from the values within a margin of 23 to 63 samples around
    it, depending on the order, which gives the coefficients of the
    whole input to rounding errors. The strategy and the block size
    apply to each block. For dask arrays, the default strategy is the
    batched one, because the many small operations of the blocked
    strategy are serialized by the lock of the Python interpreter under
    the threaded scheduler of dask. Dask is only imported when an
    argument is a dask array.

    Order 0 takes the nearest sample, and a coordinate exactly halfway
    between two samples takes the upper one, as in SciPy. Outside the
    array, the tie is broken before the boundary condition is applied.
    SciPy applies `mirror` and `reflect` to the coordinate first, so that
    at exact ties beyond the edges, the two can take different samples.

    Parameters
    ----------
    input : array
        The values, or the spline coefficients when `prefilter` is false
        and the order is 2 or more, with shape
        `(*batch_input, *spatial_in)`.
    grid : array
        The coordinates, with shape `(*batch_grid, *spatial_out, ndim)`,
        where `spatial_out` has `ndim` axes.
    order : int or sequence of int, default=1
        The spline order, from 0 to 7, for all axes or for each spatial
        axis.
    bound : str or sequence of str, default="reflect"
        The boundary condition, for all axes or for each spatial axis,
        by any name of the table in the documentation of the package.
    extrapolate : bool or str, default=True
        What to return at coordinates outside the input. True uses the
        boundary condition. False returns zero beyond the centres of the
        edge samples, with a tolerance of 0.05 voxel. The string "edges"
        returns zero beyond the outer edges of the edge samples, which
        lie half a voxel farther.
    prefilter : bool, default=False
        Whether to compute the spline coefficients of the input first,
        with [`spline_coeff`][brainhops._ext.interpol.spline_coeff], so
        that the spline passes through the values of the input.
    strategy : {"auto", "blocked", "batched"}, default="auto"
        How the computation is split, as explained above. The value
        "auto" chooses the blocked strategy for NumPy arrays and the
        batched one for CuPy arrays.
    block_size : int, optional
        The largest number of values, that is, batch items times points,
        in each temporary array of the blocked strategy. A block also
        holds at most 16384 points. By default, it is 65536. The batched
        strategy ignores it.

    Returns
    -------
    array
        The interpolated values, with shape `(*batch, *spatial_out)`,
        where `batch` is the broadcast of `batch_input` and
        `batch_grid`. The data type is that of the input, or float64 for
        an integer input.

    Raises
    ------
    ValueError
    """
    if is_dask(input) or is_dask(grid):
        from ._dask import pull_dask

        return pull_dask(
            input,
            grid,
            order,
            bound,
            extrapolate,
            prefilter,
            strategy,
            block_size,
        )
    xp = get_namespace(input, grid)
    input = to_namespace(xp, input)
    grid = to_namespace(xp, grid)
    orders, bounds, margin, blocked = check_arguments(
        xp, input, grid, order, bound, extrapolate, strategy, block_size
    )
    dtype = normalize_dtype(input.dtype)
    if prefilter and max(orders) > 1:
        input = spline_coeff(input, orders, bounds, len(orders))
    input = input.astype(dtype, copy=False)
    if grid.dtype.kind != "f":
        grid = grid.astype(np.float64)
    return pull_arrays(
        xp, input, grid, orders, bounds, margin, blocked, block_size
    )


def check_arguments(
    xp: ModuleType,
    input: tx.Any,
    grid: tx.Any,
    order: tx.Any,
    bound: tx.Any,
    extrapolate: tx.Any,
    strategy: str,
    block_size: tx.Optional[int],
) -> tx.Tuple[tx.List[int], tx.List[str], tx.Optional[float], bool]:
    """Check the arguments of [`pull`][] and return them in normal form.

    The eager and the dask versions of [`pull`][] share this function,
    which only reads the shapes of the arrays.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` with which the values are
        computed.
    input : array
        The argument `input` of [`pull`][].
    grid : array
        The argument `grid` of [`pull`][].
    order : int or sequence of int
        The argument `order` of [`pull`][].
    bound : str or sequence of str
        The argument `bound` of [`pull`][].
    extrapolate : bool or str
        The argument `extrapolate` of [`pull`][].
    strategy : str
        The argument `strategy` of [`pull`][].
    block_size : int or None
        The argument `block_size` of [`pull`][].

    Returns
    -------
    orders : list of int
        The spline order of each spatial axis.
    bounds : list of str
        The boundary condition of each spatial axis, one of the names
        in `_BOUNDS`.
    margin : float or None
        The margin of [`_inside`][], or None to keep every point.
    blocked : bool
        Whether the blocked strategy is used.

    Raises
    ------
    ValueError
    """
    ndim = grid.shape[-1]
    if input.ndim < ndim or grid.ndim < ndim + 1:
        raise ValueError(
            f"The grid has {ndim} coordinates, so the input needs at "
            f"least {ndim} axes and the grid at least {ndim + 1}, but "
            f"they have {input.ndim} and {grid.ndim}."
        )
    orders = [
        normalize_order(o) for o in normalize_per_axis(order, ndim, "order")
    ]
    bounds = [
        normalize_bound(b) for b in normalize_per_axis(bound, ndim, "bound")
    ]
    margin = _normalize_extrapolate(extrapolate)
    blocked = _select_strategy(xp, strategy) == "blocked"
    if block_size is not None and int(block_size) < 1:
        raise ValueError(f"The block size must be positive: {block_size}.")
    return orders, bounds, margin, blocked


def pull_arrays(
    xp: ModuleType,
    input: tx.Any,
    grid: tx.Any,
    orders: tx.List[int],
    bounds: tx.List[str],
    margin: tx.Optional[float],
    blocked: bool,
    block_size: tx.Optional[int],
    shape: tx.Optional[tx.Sequence[int]] = None,
    origin: tx.Optional[tx.Sequence[int]] = None,
) -> tx.Any:
    """Interpolate an array, or a window of it, at checked coordinates.

    This function does the work of [`pull`][] once the arguments are
    checked and the input is prefiltered. The input can also be a
    window of a larger array, which is how the dask version of
    [`pull`][] samples one block of coordinates: each spatial axis of
    the window holds the samples `origin[d]` to
    `origin[d] + input.shape[d] - 1` of an axis of length `shape[d]`.
    The boundary condition, the zero weights of the `zero` boundary and
    the mask of `extrapolate` then apply to the whole array, and the
    samples that the coordinates read must all lie in the window.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the arrays.
    input : array
        The values or the coefficients, with shape
        `(*batch_input, *window)` and the data type of the output.
    grid : array
        The coordinates, of a floating data type, with shape
        `(*batch_grid, *spatial_out, ndim)`.
    orders : list of int
        The spline order of each spatial axis.
    bounds : list of str
        The boundary condition of each spatial axis, one of the names in
        `_BOUNDS`.
    margin : float or None
        The margin of [`_inside`][], or None to keep every point.
    blocked : bool
        Whether to use the blocked strategy rather than the batched one.
    block_size : int or None
        The argument `block_size` of [`pull`][].
    shape : sequence of int, optional
        The spatial shape of the whole array. By default, it is the
        spatial shape of the input.
    origin : sequence of int, optional
        The index of the first sample of the window along each spatial
        axis. By default, it is zero.

    Returns
    -------
    array
        The interpolated values, with shape `(*batch, *spatial_out)`.
    """
    ndim = grid.shape[-1]
    dtype = input.dtype
    window = tuple(input.shape[-ndim:])
    spatial_in = window if shape is None else tuple(shape)
    origin = (0,) * ndim if origin is None else tuple(origin)
    spatial_out = tuple(grid.shape[-ndim - 1 : -1])
    batch_input = tuple(input.shape[:-ndim])
    batch_grid = tuple(grid.shape[: -ndim - 1])
    batch = tuple(np.broadcast_shapes(batch_input, batch_grid))
    npoints = int(math.prod(spatial_out))
    if npoints == 0 or int(math.prod(batch)) == 0:
        # An empty output has nothing to interpolate, and the views below
        # cannot be built when an axis has a length of zero.
        return xp.zeros(batch + spatial_out, dtype=dtype)
    # The flat index of a voxel is the sum of its indices in the window
    # times these.
    strides = [int(math.prod(window[d + 1 :])) for d in range(ndim)]
    pool = BufferPool(xp)

    # When NumPy's `take` writes into an output array and checks the
    # indices, it goes through a buffer, which `mode="clip"` avoids. The
    # indices are always valid, so clipping never changes them.
    take_options = {"mode": "clip"} if xp is np else {}

    shared = int(math.prod(batch_grid)) == 1
    if shared:
        # The same points for every batch item: the input is read in
        # place, as an array `(outer, voxels, inner)`, and each tap
        # gathers whole rows of the batch. The batch axes of a block are
        # the outer and the inner ones.
        grid = grid.reshape((1, 1, npoints, ndim))
        view, outer, inner = _as_view3d(xp, input, len(batch_input))
        block_batch = (view.shape[0], view.shape[2])
        out = xp.empty((view.shape[0], npoints, view.shape[2]), dtype=dtype)
        grid_batch = (1, 1)
    else:
        # Points that depend on the batch: each tap gathers from the flat
        # input with indices that broadcast over the batch.
        grid_batch = (1,) * (len(batch) - len(batch_grid)) + batch_grid
        grid = grid.reshape(grid_batch + (npoints, ndim))
        flat, offset = _flat_with_offsets(xp, input, batch, ndim)
        block_batch = batch
        out = xp.empty(batch + (npoints,), dtype=dtype)

    nitems = int(math.prod(block_batch))
    if blocked:
        size = _BLOCK_VALUES if block_size is None else int(block_size)
        # The inner batch axis is kept whole when it fits in a block,
        # since a block of it can only be gathered by slower indexing.
        whole = size // max(1, block_batch[-1]) if shared else size
        points = max(1, min(npoints, _BLOCK_POINTS, whole))
        items = max(1, size // points)
    else:
        points = max(1, npoints)
        items = max(1, nitems)

    def gather_view(sub: tx.Any, columns: slice) -> tx.Any:
        # Gather from rows of the view. A take into the buffer needs a
        # contiguous source, or it copies the source first, so that a
        # block of the inner axis, which is not contiguous, is gathered
        # by indexing instead.
        if columns == slice(None):

            def gather(index: tx.Any, values: tx.Any) -> tx.Any:
                return xp.take(sub, index, axis=1, out=values, **take_options)

        else:

            def gather(index: tx.Any, values: tx.Any) -> tx.Any:
                return sub[:, index, columns]

        return gather

    def gather_flat(offset: tx.Any, shape: tx.Tuple[int, ...]) -> tx.Any:
        flat_index = pool.get("flat index", shape, np.intp)

        def gather(index: tx.Any, values: tx.Any) -> tx.Any:
            xp.add(offset, index, out=flat_index)
            return xp.take(flat, flat_index, out=values, **take_options)

        return gather

    for start in range(0, npoints, points):
        span = slice(start, min(start + points, npoints))
        previous: tx.Optional[tx.Tuple[slice, ...]] = None
        taps: tx.List[tx.List[tx.Tuple[tx.Any, tx.Any]]] = []
        for select in _batch_blocks(block_batch, items):
            # The indices and the weights depend on the batch items only
            # along the batch axes of the grid, and are reused otherwise.
            grid_select = _broadcast_select(select, grid_batch)
            if grid_select != previous:
                previous = grid_select
                coords = grid[grid_select + (span,)]
                taps = _block_taps(
                    xp,
                    coords,
                    spatial_in,
                    orders,
                    bounds,
                    dtype,
                    strides,
                    margin,
                    pool,
                    origin,
                )
                if not blocked:
                    # The single block of the batched strategy does not
                    # need the scratch arrays of the taps again, and they
                    # are as large as the grid.
                    pool.release(*_TAP_SCRATCH)
                if shared:
                    # The weights multiply arrays `(outer, points, inner)`.
                    taps = [
                        [(i.reshape(-1), w.reshape((1, -1, 1))) for i, w in t]
                        for t in taps
                    ]
            if shared:
                target = out[select[0], span, select[1]]
                gather = gather_view(view[select[0]], select[1])
            else:
                target = out[select + (span,)]
                offset_select = _broadcast_select(select, offset.shape)
                gather = gather_flat(offset[offset_select], target.shape)
            _accumulate(xp, taps, target, gather, pool)

    if shared:
        # The batch axes of the view are those of the input, in the
        # order of the memory, and are put back in their order.
        axes = outer + inner
        out = xp.moveaxis(out, 1, -1)
        out = out.reshape(tuple(input.shape[a] for a in axes) + spatial_out)
        order_back = [axes.index(a) for a in range(len(batch_input))]
        order_back += list(range(len(batch_input), out.ndim))
        out = xp.transpose(out, order_back)
    return out.reshape(batch + spatial_out)


def _select_strategy(xp: ModuleType, strategy: str) -> str:
    """Return the strategy that [`pull`][] uses for arrays of a package.

    The choice is made here only. NumPy arrays use the blocked strategy,
    whose blocks stay in the cache of the processor. CuPy arrays use the
    batched strategy, because on a GPU each operation is a kernel that
    processes a whole array, and the memory is fast enough that smaller
    blocks would only add kernel launches.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the arrays.
    strategy : str
        The argument `strategy` of [`pull`][], one of `_STRATEGIES`.

    Returns
    -------
    str
        The strategy, "blocked" or "batched".

    Raises
    ------
    ValueError
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Use one of {_STRATEGIES}."
        )
    if strategy == "auto":
        return "blocked" if xp is np else "batched"
    return strategy


def _batch_blocks(
    shape: tx.Tuple[int, ...], size: int
) -> tx.Iterator[tx.Tuple[slice, ...]]:
    """Split batch axes into blocks of at most a given number of items.

    The blocks are made of whole trailing axes as far as they fit, so
    that a block is as contiguous as possible in a C-ordered array. The
    first axis that does not fit is split into ranges, and the axes
    before it are taken one index at a time.

    Parameters
    ----------
    shape : tuple of int
        The shape of the batch axes.
    size : int
        The largest number of items in a block, at least 1.

    Yields
    ------
    tuple of slice
        One slice per axis, which selects the items of a block.
    """
    split = len(shape)
    inner = 1
    while split > 0 and inner * shape[split - 1] <= size:
        split -= 1
        inner *= shape[split]
    if split == 0:
        yield (slice(None),) * len(shape)
        return
    split -= 1
    chunk = size // inner
    rest = (slice(None),) * (len(shape) - split - 1)
    for lead in itertools.product(*[range(n) for n in shape[:split]]):
        head = tuple(slice(k, k + 1) for k in lead)
        for start in range(0, shape[split], chunk):
            stop = min(start + chunk, shape[split])
            yield head + (slice(start, stop),) + rest


def _broadcast_select(
    select: tx.Tuple[slice, ...], shape: tx.Sequence[int]
) -> tx.Tuple[slice, ...]:
    """Return the selection of a block in an array that broadcasts.

    An array that broadcasts against the batch has length 1 along some
    batch axes, and its single item belongs to every block along them.

    Parameters
    ----------
    select : tuple of slice
        The slices that select a block of the batch axes.
    shape : sequence of int
        The shape of the array, whose first axes broadcast against the
        batch axes.

    Returns
    -------
    tuple of slice
        The slices that select the block in the array.
    """
    return tuple(s if n > 1 else slice(None) for s, n in zip(select, shape))


def _block_taps(
    xp: ModuleType,
    coords: tx.Any,
    spatial: tx.Tuple[int, ...],
    orders: tx.List[int],
    bounds: tx.List[str],
    dtype: tx.Any,
    strides: tx.List[int],
    margin: tx.Optional[float],
    pool: BufferPool,
    origin: tx.Sequence[int],
) -> tx.List[tx.List[tx.Tuple[tx.Any, tx.Any]]]:
    """Return the indices and the weights of the taps of a block.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the coordinates.
    coords : array
        The coordinates of the points of the block, with the
        coordinates of each point along the last axis.
    spatial : tuple of int
        The spatial shape of the whole input, whose boundary conditions
        apply.
    orders : list of int
        The spline order of each axis.
    bounds : list of str
        The boundary condition of each axis.
    dtype : data type
        The data type of the interpolated values.
    strides : list of int
        The strides of the spatial axes in the flattened spatial axes of
        the window that is read.
    margin : float or None
        The margin of [`_inside`][], or None to keep every point.
    pool : BufferPool
        The pool that provides the arrays.
    origin : sequence of int
        The index of the first sample of the window along each spatial
        axis.

    Returns
    -------
    list of list of tuple
        For each axis, one pair `(flat index, weight)` per tap, with
        the shape of `coords` without its last axis. The flat index is
        the index in the window along the axis times its stride.
    """
    taps = []
    for d in range(len(spatial)):
        taps.append(
            axis_taps(
                xp,
                coords[..., d],
                spatial[d],
                orders[d],
                bounds[d],
                dtype,
                strides[d],
                pool,
                d,
                origin[d],
            )
        )
    if margin is not None:
        inside = _inside(xp, coords, spatial, margin, pool)
        for _, weight in taps[0]:
            xp.multiply(weight, inside, out=weight)
    return taps


def _normalize_extrapolate(extrapolate: tx.Any) -> tx.Optional[float]:
    """Return the margin beyond the edge samples where values are kept.

    Parameters
    ----------
    extrapolate : bool or str
        The argument `extrapolate` of [`pull`][].

    Returns
    -------
    float or None
        The margin in voxels, or None when values are kept everywhere.

    Raises
    ------
    ValueError
    """
    if isinstance(extrapolate, str):
        if extrapolate.lower() in ("edges", "hist"):
            return 0.5 + _TOLERANCE
        raise ValueError(f"Unknown value of extrapolate: {extrapolate!r}.")
    return None if extrapolate else _TOLERANCE


def _inside(
    xp: ModuleType,
    coords: tx.Any,
    shape: tx.Sequence[int],
    margin: float,
    pool: BufferPool,
) -> tx.Any:
    """Return where coordinates lie inside an array, up to a margin.

    A coordinate along an axis of length `n` is inside when it lies
    between `-margin` and `n - 1 + margin`, that is, when its distance
    to the centre `(n - 1) / 2` of the axis is less than
    `(n - 1) / 2 + margin`.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the coordinates.
    coords : array
        The coordinates, with the coordinates of each point along the
        last axis.
    shape : sequence of int
        The spatial shape of the array.
    margin : float
        The margin, in voxels.
    pool : BufferPool
        The pool that provides the arrays.

    Returns
    -------
    array
        A boolean array with the shape of `coords` without its last
        axis, which belongs to the pool.
    """
    points = coords.shape[:-1]
    inside = pool.get("inside", points, bool)
    mask = pool.get("mask", points, bool)
    distance = pool.get("distance", points, coords.dtype)
    for d, n in enumerate(shape):
        half = (n - 1) / 2
        xp.subtract(coords[..., d], half, out=distance)
        xp.abs(distance, out=distance)
        xp.less(distance, half + margin, out=mask if d else inside)
        if d:
            xp.logical_and(inside, mask, out=inside)
    return inside


def _as_view3d(
    xp: ModuleType, input: tx.Any, nbatch: int
) -> tx.Tuple[tx.Any, tx.List[int], tx.List[int]]:
    """Return the input as a three-dimensional array, without a copy.

    The spatial axes are flattened into the middle axis, whose length
    is the number of voxels. The batch axes that come before the
    spatial axes in memory are flattened into the first axis, and those
    that come after them into the last axis. In this way, an array
    whose axes were moved, for example to resample one axis of a
    volume, is read in place, and each gathered voxel reads a
    contiguous run of the inner batch axes. An input that does not have
    this layout is copied to a contiguous array.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the input.
    input : array
        The input, with shape `(*batch, *spatial)`.
    nbatch : int
        The number of batch axes.

    Returns
    -------
    view : array
        The array, with shape `(outer, voxels, inner)`.
    outer : list of int
        The batch axes flattened into the first axis.
    inner : list of int
        The batch axes flattened into the last axis.
    """
    spatial = list(range(nbatch, input.ndim))
    nvoxels = int(math.prod(input.shape[nbatch:]))
    # Axes of length 1 can go anywhere, and are kept in their place.
    strides = [
        s if n > 1 else None for s, n in zip(input.strides, input.shape)
    ]
    batch = list(range(nbatch))
    long_spatial = [a for a in spatial if strides[a] is not None]
    if long_spatial:
        smallest = min(strides[a] for a in long_spatial)
        largest = max(strides[a] for a in long_spatial)
        outer = [
            a for a in batch if strides[a] is None or strides[a] > largest
        ]
        inner = [a for a in batch if a not in outer]
        if all(strides[a] < smallest for a in inner if strides[a] is not None):
            # The batch axes are sorted by decreasing stride within each
            # group, so that each group can be flattened.
            def key(a: int) -> int:
                return -(strides[a] or 0)

            outer.sort(key=key)
            inner.sort(key=key)
            moved = xp.transpose(input, outer + spatial + inner)
            if moved.flags.c_contiguous:
                shape = (int(math.prod(moved.shape[: len(outer)])), nvoxels)
                return moved.reshape(shape + (-1,)), outer, inner
    view = xp.ascontiguousarray(input).reshape((-1, nvoxels, 1))
    return view, batch, []


def _flat_with_offsets(
    xp: ModuleType, input: tx.Any, batch: tx.Tuple[int, ...], ndim: int
) -> tx.Tuple[tx.Any, tx.Any]:
    """Return the input as a vector, and the offsets of its batch items.

    The flat index of the voxel `v` of the batch item `b` is
    `offset[b] + v`. The offsets have the broadcast batch shape, with a
    trailing axis of length 1 for the points, and are zero along the
    batch axes that the input broadcasts. In this way, the input is
    never copied once per item of the broadcast batch.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the input.
    input : array
        The input, with shape `(*batch_input, *spatial)`.
    batch : tuple of int
        The broadcast batch shape.
    ndim : int
        The number of spatial axes.

    Returns
    -------
    flat : array
        The input as a contiguous vector.
    offset : array
        The offsets, with shape `(*batch, 1)` and an integer data type.
    """
    nvoxels = int(math.prod(input.shape[-ndim:]))
    shape = (1,) * (len(batch) + ndim - input.ndim) + tuple(input.shape)
    offset = xp.zeros((1,) * len(batch) + (1,), dtype=np.intp)
    stride = nvoxels
    for k in reversed(range(len(batch))):
        if shape[k] > 1:
            steps = xp.arange(shape[k], dtype=np.intp) * stride
            view = [1] * (len(batch) + 1)
            view[k] = shape[k]
            offset = offset + steps.reshape(tuple(view))
        stride *= shape[k]
    return xp.ascontiguousarray(input).reshape(-1), offset


def _accumulate(
    xp: ModuleType,
    taps: tx.List[tx.List[tx.Tuple[tx.Any, tx.Any]]],
    out: tx.Any,
    gather: tx.Callable[[tx.Any, tx.Any], tx.Any],
    pool: BufferPool,
) -> None:
    """Write the sum of the weighted samples of every tap into `out`.

    The flat indices and the weights of the combinations of taps are
    built axis by axis, so that only one partial index and one partial
    weight per axis are held in memory at a time. The samples of the
    taps of the last axis are summed before they are multiplied by the
    weight of the other axes, which saves one multiplication per
    combination. The first combination is summed in `out` itself, which
    therefore does not need to be filled with zeros first. Every step
    writes into arrays of the pool, because allocating an array for
    each tap is slow.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the arrays.
    taps : list of list of tuple
        For each axis, one pair `(flat index, weight)` per tap, computed
        for the points of the block. The weights have a shape that
        multiplies the gathered samples.
    out : array
        The block of the output, which is overwritten. It may be a view
        that is not contiguous.
    gather : callable
        The function that returns the samples at flat indices, for the
        points and the batch items of the block, with the shape of
        `out`. It writes them into the array that it receives when it
        can, and returns that array.
    pool : BufferPool
        The pool that provides the arrays.
    """
    # The recursion over the axes is a loop over the combinations of
    # the taps of all but the last axis, which reuses the partial
    # products of the axes whose tap did not change.
    ndim = len(taps)
    values = pool.get("values", out.shape, out.dtype)
    total = pool.get("total", out.shape, out.dtype) if ndim > 1 else out
    partial: tx.List[tx.Tuple[tx.Any, tx.Any]] = [(None, None)] * ndim
    previous: tx.Tuple[int, ...] = ()
    first = True
    for combination in itertools.product(*[range(len(t)) for t in taps[:-1]]):
        changed = 0
        while changed < len(previous) and (
            previous[changed] == combination[changed]
        ):
            changed += 1
        previous = combination
        for d in range(changed, len(combination)):
            tap_index, tap_weight = taps[d][combination[d]]
            if d:
                index, weight = partial[d - 1]
                tap_index = xp.add(
                    index,
                    tap_index,
                    out=pool.get(("index", d), index.shape, index.dtype),
                )
                tap_weight = xp.multiply(
                    weight,
                    tap_weight,
                    out=pool.get(("weight", d), weight.shape, weight.dtype),
                )
            partial[d] = (tap_index, tap_weight)
        index, weight = partial[ndim - 2] if ndim > 1 else (None, None)
        # The taps of the last axis are summed first, and the sum is
        # then multiplied by the weight of the taps of the other axes.
        acc = out if first else total
        for k, (tap_index, tap_weight) in enumerate(taps[-1]):
            if index is not None:
                tap_index = xp.add(
                    index,
                    tap_index,
                    out=pool.get("last index", index.shape, index.dtype),
                )
            samples = gather(tap_index, values)
            if k == 0:
                xp.multiply(samples, tap_weight, out=acc)
            else:
                xp.multiply(samples, tap_weight, out=samples)
                xp.add(acc, samples, out=acc)
        if weight is not None:
            xp.multiply(acc, weight, out=acc)
        if not first:
            xp.add(out, acc, out=out)
        first = False
