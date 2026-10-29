# This file is part of a port of torch-interpol 0.3.0 to NumPy and
# CuPy. Copyright (c) 2021 Yael Balbastre. Released under the MIT
# license, whose text is in the LICENSE file of this package.
"""Interpolation and prefilter of dask arrays, one block at a time.

The functions [`pull`][brainhops._ext.interpol.pull],
[`spline_coeff`][brainhops._ext.interpol.spline_coeff] and
[`spline_coeff1d`][brainhops._ext.interpol.spline_coeff1d] call this
module when one of their arrays is a dask array. The module is the only
one of the package that imports dask, which therefore remains an
optional dependency. The results equal those of the same functions on
the whole array, to rounding errors, and each task computes them with
the eager code of the package on the part of the array that it needs.

Interpolation is local: the value at a coordinate reads `order + 1`
samples along each axis, which the boundary condition brings inside the
axis. For each block of the grid, one task therefore reads the window
of the input that holds every sample its coordinates read, and runs the
eager interpolation on that window. The window is found by folding the
range of the samples read, with the boundary condition, over the whole
axis. For a smooth transformation, it is a little larger than the block,
while a coordinate that the boundary wraps around the axis makes it
grow to the whole axis. The zero weights of the `zero` boundary and the
mask of the argument `extrapolate` are computed for the whole axis.

The prefilter is not local, but it nearly is. The coefficient at a
sample is a sum over all the samples of the extended input, with weights
that decrease as `|z|**k` with the distance `k`, where `z` is the
largest pole of the prefilter. Beyond the horizon `K`, at which
`|z|**K` falls below 1e-17, the weights no longer change the result in
float64. The coefficients of a chunk, or of the window of a task, are
therefore computed by the eager prefilter on that range of samples
extended by `K` samples on each side. An edge of this extended window
that is an edge of the array keeps the boundary condition of the array,
which is exact. An edge that cuts the array is at least `K` samples away
from the samples that are kept, and the condition that the eager
prefilter applies there changes their coefficients by less than their
rounding error. This holds for every boundary condition and every order,
including the zero and replicate boundaries, whose correction near an
edge reaches up to `K` samples into the axis. The coefficients of the
whole array are the exact solution for its boundary condition, and the
interpolation extends them with the same condition, so that no padding
of the array is needed, unlike in `brainhops._core.dask_ndimage`, which
reproduces the padding that SciPy applies before its prefilter.
"""

# stdlib
import itertools
import math
from types import ModuleType

# dependencies
import dask
import dask.array as da
import numpy as np
import typing_extensions as tx
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

# internals
from ._backend import _asarray, _get_namespace, _is_dask
from ._bounds import _bound_index, _normalize_bound
from ._coeff import (
    _horizon,
    _normalize_dtype,
    _normalize_order,
    _poles,
    spline_coeff,
    spline_coeff1d,
)
from ._pull import _check_arguments, _pull
from ._splines import _first_tap

_Slices = tx.Tuple[slice, ...]


# --- arrays and packages ----------------------------------------------------


def _namespace(*arrays: tx.Any) -> ModuleType:
    """Return the package, NumPy or CuPy, of the blocks of some arrays.

    A dask array keeps an empty array of the package of its blocks in
    its attribute `_meta`, which says whether its blocks are on a GPU.

    Parameters
    ----------
    *arrays : array
        Dask arrays, or NumPy or CuPy arrays.

    Returns
    -------
    module
        The module `cupy` when the blocks of one of the arrays are CuPy
        arrays, and `numpy` otherwise.
    """
    return _get_namespace(*[getattr(a, "_meta", a) for a in arrays])


def _to_namespace(xp: ModuleType, array: tx.Any) -> tx.Any:
    """Move a dask array or a concrete array to the package `xp`.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy`.
    array : array
        A dask array, or a NumPy or CuPy array.

    Returns
    -------
    array
        The array, whose blocks belong to `xp` when it is a dask array.
    """
    if not _is_dask(array):
        return _asarray(xp, array)
    if _namespace(array) is xp:
        return array
    meta = xp.empty((0,) * array.ndim, dtype=array.dtype)
    return array.map_blocks(xp.asarray, dtype=array.dtype, meta=meta)


_Piece = tx.Tuple[int, slice, slice]
"""A piece of a chunk along one axis: the index of the chunk, the slice
of the chunk that the piece covers, and the slice of the window where it
goes."""


def _tiles(
    name: str, axes: tx.Sequence[tx.Sequence[_Piece]]
) -> tx.Tuple[tx.List[tx.Any], tx.List[tx.Tuple[_Slices, _Slices]], tuple]:
    """Return how a window of a dask array is assembled from its chunks.

    Parameters
    ----------
    name : str
        The name of the dask array.
    axes : sequence of sequence of tuple
        For each axis of the array, the pieces of the chunks that the
        window covers along that axis, from [`_runs`][].

    Returns
    -------
    keys : list of tuple
        The keys of the chunks that the window reads, one per
        combination of the pieces of the axes.
    pieces : list of tuple
        For each key, the slices of the chunk to read, and the slices of
        the window to write them into.
    shape : tuple of int
        The shape of the window.
    """
    keys = []
    pieces = []
    for combination in itertools.product(*axes):
        keys.append((name,) + tuple(chunk for chunk, _, _ in combination))
        source = tuple(s for _, s, _ in combination)
        target = tuple(t for _, _, t in combination)
        pieces.append((source, target))
    shape = tuple(axis[-1][2].stop for axis in axes)
    return keys, pieces, shape


def _assemble(
    parts: tx.List[tx.Any],
    pieces: tx.List[tx.Tuple[_Slices, _Slices]],
    shape: tx.Tuple[int, ...],
) -> tx.Any:
    """Copy the pieces of some chunks into a window, from [`_tiles`][].

    Parameters
    ----------
    parts : list of array
        The chunks, as NumPy or CuPy arrays.
    pieces : list of tuple
        For each chunk, the slices of the chunk to read and the slices
        of the window to write them into.
    shape : tuple of int
        The shape of the window.

    Returns
    -------
    array
        The window. It is a view of the chunk when it lies in a single
        chunk, and a new array otherwise, into which each sample is
        copied once.
    """
    if len(parts) == 1:
        return parts[0][pieces[0][0]]
    xp = _get_namespace(*parts)
    window = xp.empty(shape, dtype=parts[0].dtype)
    for part, (source, target) in zip(parts, pieces):
        window[target] = part[source]
    return window


def _whole(chunks: tx.Sequence[int], index: int) -> tx.List[_Piece]:
    """Return the piece of one whole chunk along an axis.

    Parameters
    ----------
    chunks : sequence of int
        The lengths of the chunks of the axis.
    index : int
        The index of the chunk.

    Returns
    -------
    list of tuple
        The single piece, in the form of [`_runs`][].
    """
    return [(index, slice(None), slice(0, int(chunks[index])))]


def _edges(chunks: tx.Sequence[int]) -> np.ndarray:
    """Return the index of the first sample of every chunk of an axis.

    Parameters
    ----------
    chunks : sequence of int
        The lengths of the chunks.

    Returns
    -------
    np.ndarray
        The `len(chunks) + 1` indices at which the chunks start, the
        last one being the length of the axis.
    """
    return np.concatenate([[0], np.cumsum(chunks, dtype=np.int64)])


# --- prefilter --------------------------------------------------------------


def _halo(order: int) -> int:
    """Return the number of samples beyond which the prefilter is local.

    Parameters
    ----------
    order : int
        The spline order, from 0 to 7.

    Returns
    -------
    int
        The horizon of the largest pole of the prefilter, which is 0 for
        the orders 0 and 1, that have no prefilter, and 63 at order 7.
    """
    poles = _poles(order)
    return _horizon(poles[0]) if poles else 0


def _runs(edges: np.ndarray, start: int, stop: int) -> tx.List[_Piece]:
    """Return the pieces of the chunks that a run of samples covers.

    The run goes from the sample `start` to the sample `stop - 1` of a
    periodic axis, so that it can wrap around the axis when `start` is
    negative or `stop` is larger than the length of the axis.

    Parameters
    ----------
    edges : np.ndarray
        The indices at which the chunks of the axis start, from
        [`_edges`][].
    start : int
        The first sample of the run.
    stop : int
        The sample after the last one of the run.

    Returns
    -------
    list of tuple
        For each piece, in the order of the run, the index of its chunk,
        the slice of that chunk that it covers, and its place in the
        run, as a slice that starts at 0 for the sample `start`.
    """
    n = int(edges[-1])
    pieces = []
    position = start
    while position < stop:
        index = position % n
        chunk = int(np.searchsorted(edges, index, side="right")) - 1
        end = min(int(edges[chunk + 1]), index + stop - position)
        source = slice(index - int(edges[chunk]), end - int(edges[chunk]))
        target = slice(position - start, position - start + end - index)
        pieces.append((chunk, source, target))
        position += end - index
    return pieces


def _extend(
    start: int, stop: int, n: int, halo: int, bound: str
) -> tx.Tuple[int, int, str]:
    """Return the window that the prefilter of some samples needs.

    The window adds `halo` samples on each side of the samples `start`
    to `stop - 1`, and is cut at the edges of the axis, where the eager
    prefilter applies the boundary condition of the array. A window that
    reaches both edges is the whole axis. With the periodic boundary
    `dft`, a window instead continues past one edge with the samples of
    the other edge, unless it covers the whole period. The eager
    prefilter then applies the boundary `dct2` at both edges of the
    window, which are cuts, where any boundary condition will do.

    Parameters
    ----------
    start : int
        The first sample whose coefficient is needed.
    stop : int
        The sample after the last one whose coefficient is needed.
    n : int
        The length of the axis.
    halo : int
        The number of samples to add on each side, from [`_halo`][].
    bound : str
        The boundary condition of the axis, one of the names in
        `_BOUNDS`.

    Returns
    -------
    first : int
        The first sample of the window, which is negative when a `dft`
        window continues past the start of the axis.
    last : int
        The sample after the last one of the window, which is larger
        than `n` when a `dft` window continues past the end of the axis.
    bound : str
        The boundary condition that the eager prefilter applies to the
        window.
    """
    first, last = start - halo, stop + halo
    if bound != "dft":
        return max(first, 0), min(last, n), bound
    if last - first >= n:
        return 0, n, bound
    return first, last, "dct2"


def _coeff_block(
    parts: tx.List[tx.Any],
    pieces: tx.List[tx.Tuple[_Slices, _Slices]],
    shape: tx.Tuple[int, ...],
    axis: int,
    order: int,
    bound: str,
    keep: slice,
) -> tx.Any:
    """Compute the coefficients of one chunk from the window around it.

    Parameters
    ----------
    parts : list of array
        The chunks of the input that the window covers.
    pieces : list of tuple
        How the window is made of the chunks, from [`_tiles`][].
    shape : tuple of int
        The shape of the window.
    axis : int
        The axis to filter.
    order : int
        The spline order, from 2 to 7.
    bound : str
        The boundary condition that the eager prefilter applies at both
        edges of the window.
    keep : slice
        The samples of the window, along the axis, that are the chunk.

    Returns
    -------
    array
        The coefficients of the chunk, as a new array.
    """
    window = _assemble(parts, pieces, shape)
    coeff = spline_coeff1d(window, order, bound, axis)
    xp = _get_namespace(coeff)
    return xp.ascontiguousarray(coeff[(slice(None),) * axis + (keep,)])


def _spline_coeff1d_dask(
    input: tx.Any, order: tx.Any, bound: tx.Any, axis: int
) -> tx.Any:
    """Compute the spline coefficients of a dask array along one axis.

    This function is the dask version of
    [`spline_coeff1d`][brainhops._ext.interpol.spline_coeff1d], which
    calls it, and takes the same arguments. The result has the chunks
    of the input. Each chunk is one task, which reads the chunks of the
    input that overlap the window from `K` samples before the chunk to
    `K` samples after it, where `K` is the horizon of [`_halo`][], cut
    or continued at the edges of the array as [`_extend`][] explains.
    On an axis of at most `2 * K + 1` samples, every window is the
    whole axis.

    Parameters
    ----------
    input : dask.array.Array
        The values, of any shape.
    order : int
        The spline order, from 0 to 7.
    bound : str
        The boundary condition, by any name of the table in the
        documentation of the package.
    axis : int
        The axis to filter.

    Returns
    -------
    dask.array.Array
        The coefficients, with the shape and the chunks of the input.
    """
    order = _normalize_order(order)
    bound = _normalize_bound(bound)
    dtype = _normalize_dtype(input.dtype)
    input = input.astype(dtype)
    if order < 2 or input.size == 0:
        return input
    axis = axis % input.ndim
    n = int(input.shape[axis])
    edges = _edges(input.chunks[axis])
    halo = _halo(order)
    name = "interpol-spline-coeff1d-" + tokenize(input, order, bound, axis)
    layer: tx.Dict[tx.Any, tx.Any] = {}
    plans = []
    for chunk in range(len(input.chunks[axis])):
        start, stop = int(edges[chunk]), int(edges[chunk + 1])
        first, last, window_bound = _extend(start, stop, n, halo, bound)
        keep = slice(start - first, stop - first)
        plans.append((_runs(edges, first, last), window_bound, keep))
    for index in itertools.product(*[range(b) for b in input.numblocks]):
        runs, window_bound, keep = plans[index[axis]]
        axes = [_whole(c, i) for c, i in zip(input.chunks, index)]
        axes[axis] = runs
        keys, pieces, shape = _tiles(input.name, axes)
        layer[(name,) + index] = (
            _coeff_block,
            keys,
            pieces,
            shape,
            axis,
            order,
            window_bound,
            keep,
        )
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[input])
    return da.Array(graph, name, input.chunks, meta=input._meta)


# --- interpolation ----------------------------------------------------------


def _axis_window(
    xp: ModuleType, coords: tx.Any, n: int, order: int, bound: str
) -> tx.Tuple[int, int]:
    """Return the range of samples that coordinates read along an axis.

    The coordinate `x` reads the samples `floor(x - offset)` to
    `floor(x - offset) + order`, with the offset of
    [`_first_tap`][brainhops._ext.interpol._splines._first_tap], before
    the boundary condition brings them inside the axis. The samples
    read by a group of coordinates are therefore those of a run of
    consecutive indices, from the smallest coordinate to the largest,
    brought inside the axis. The run is computed in the precision of the
    coordinates, as [`_axis_taps`][brainhops._ext.interpol._taps._axis_taps]
    computes it, and extended by one sample on each side, which guards
    against any difference in rounding. A run that covers a period of
    every boundary condition reads the whole axis, and so do coordinates
    that are not finite or that are too large to be converted to
    integers exactly.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the coordinates.
    coords : array
        The coordinates along the axis, of a floating data type.
    n : int
        The length of the axis.
    order : int
        The spline order, from 0 to 7.
    bound : str
        The boundary condition, one of the names in `_BOUNDS`.

    Returns
    -------
    start : int
        The first sample that is read.
    stop : int
        The sample after the last one that is read.
    """
    if coords.size == 0:
        return 0, min(n, 1)
    offset = coords.dtype.type(_first_tap(order))
    first = float(xp.floor(xp.min(coords) - offset))
    last = float(xp.floor(xp.max(coords) - offset))
    if not (math.isfinite(first) and math.isfinite(last)):
        return 0, n
    if max(abs(first), abs(last)) > 2**52:
        return 0, n
    start, stop = int(first) - 1, int(last) + order + 2
    # The periods of the boundaries are at most 2 * n + 2 samples long.
    if stop - start >= 2 * n + 2:
        return 0, n
    index = np.arange(start, stop, dtype=np.intp)
    _bound_index(np, index, n, bound, np.ones(index.shape))
    return int(index.min()), int(index.max()) + 1


def _windows(
    xp: ModuleType,
    grid: tx.Any,
    shape: tx.Sequence[int],
    orders: tx.Sequence[int],
    bounds: tx.Sequence[str],
) -> tx.List[tx.Tuple[int, int]]:
    """Return the window of the input that a block of a grid reads.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the grid.
    grid : array
        The coordinates of the block, with the coordinates of each point
        along the last axis.
    shape : sequence of int
        The spatial shape of the input.
    orders : sequence of int
        The spline order of each spatial axis.
    bounds : sequence of str
        The boundary condition of each spatial axis.

    Returns
    -------
    list of tuple
        For each spatial axis, the first sample of the window and the
        sample after its last one.
    """
    return [
        _axis_window(xp, grid[..., d], int(n), orders[d], bounds[d])
        for d, n in enumerate(shape)
    ]


class _Source:
    """The input of a lazy grid, which every task slices for itself.

    When the grid is a dask array, its coordinates, and therefore the
    windows of the input that its blocks read, are only known when the
    tasks run. The input is then a single value of the graph, held in
    this wrapper so that dask does not mistake it for an argument to
    compute, and each task computes the window that it reads.

    Parameters
    ----------
    array : array
        The input, a dask array or a NumPy or CuPy array.
    """

    def __init__(self, array: tx.Any) -> None:
        self.array = array

    def read(
        self, axes: tx.Sequence[tx.Sequence[tx.Tuple[slice, slice]]]
    ) -> tx.Any:
        """Return a window of the input as a NumPy or CuPy array.

        A dask input is computed with the synchronous scheduler, which
        runs inside the task that asks for the window, whatever
        scheduler runs that task.

        Parameters
        ----------
        axes : sequence of sequence of tuple
            For each axis of the input, the pieces of the axis that the
            window covers, each as a slice of the input and the slice of
            the window where it goes. An axis has several pieces when
            the window wraps around it.

        Returns
        -------
        array
            The window.
        """
        combinations = list(itertools.product(*axes))
        parts = [self.array[tuple(s for s, _ in c)] for c in combinations]
        if _is_dask(self.array):
            parts = list(dask.compute(*parts, scheduler="sync"))
        shape = tuple(axis[-1][1].stop for axis in axes)
        whole = tuple(slice(None) for _ in shape)
        pieces = [(whole, tuple(t for _, t in c)) for c in combinations]
        return _assemble(parts, pieces, shape)


def _pull_parts(
    grid: tx.Any,
    parts: tx.List[tx.Any],
    pieces: tx.List[tx.Tuple[_Slices, _Slices]],
    shape: tx.Tuple[int, ...],
    origin: tx.Sequence[int],
    options: tx.Dict[str, tx.Any],
) -> tx.Any:
    """Interpolate one block of a concrete grid from its window.

    Parameters
    ----------
    grid : array
        The block of the grid.
    parts : list of array
        The chunks of the input that the window covers.
    pieces : list of tuple
        How the window is made of the chunks, from [`_tiles`][].
    shape : tuple of int
        The shape of the window.
    origin : sequence of int
        The index of the first sample of the window along each spatial
        axis.
    options : dict
        The other keyword arguments of
        [`_pull`][brainhops._ext.interpol._pull._pull].

    Returns
    -------
    array
        The interpolated values of the block.
    """
    xp = _get_namespace(grid)
    window = _assemble(parts, pieces, shape)
    return _pull(xp, window, grid, origin=origin, **options)


def _pull_source(
    grid: tx.Any,
    source: _Source,
    batch: tx.Sequence[slice],
    prefilter: bool,
    options: tx.Dict[str, tx.Any],
) -> tx.Any:
    """Interpolate one block of a lazy grid, reading its window first.

    With the prefilter, the window that the task reads from the input
    is extended by the halo of the prefilter along each axis, as in
    [`_extend`][], and the coefficients of the samples that the block
    reads are computed from it. In this way, the task does not compute
    the coefficients of the chunks that its window overlaps, which other
    tasks would compute again.

    Parameters
    ----------
    grid : array
        The block of the grid.
    source : _Source
        The whole input.
    batch : sequence of slice
        The slices of the batch axes of the input that the block reads.
    prefilter : bool
        Whether the input holds values, whose coefficients are computed
        here, rather than coefficients.
    options : dict
        The other keyword arguments of
        [`_pull`][brainhops._ext.interpol._pull._pull].

    Returns
    -------
    array
        The interpolated values of the block.
    """
    xp = _get_namespace(grid)
    shape = options["shape"]
    orders = options["orders"]
    bounds = options["bounds"]
    windows = _windows(xp, grid, shape, orders, bounds)
    nbatch = len(batch)
    axes = [
        [(s, slice(0, len(range(n)[s])))]
        for s, n in zip(batch, source.array.shape)
    ]
    keep = [slice(None)] * nbatch
    filters = []
    for d, (start, stop) in enumerate(windows):
        halo = _halo(orders[d]) if prefilter else 0
        first, last, bound = _extend(start, stop, shape[d], halo, bounds[d])
        runs = _runs(np.asarray([0, shape[d]]), first, last)
        axes.append([(s, t) for _, s, t in runs])
        keep.append(slice(start - first, stop - first))
        filters.append(bound)
    window = _asarray(xp, source.read(axes))
    if prefilter:
        for d in reversed(range(len(windows))):
            if orders[d] > 1:
                window = spline_coeff1d(
                    window, orders[d], filters[d], nbatch + d
                )
        window = window[tuple(keep)]
    origin = [start for start, _ in windows]
    return _pull(xp, window, grid, origin=origin, **options)


def _batch_chunks(
    input: tx.Any, grid: tx.Any, batch: tx.Tuple[int, ...], ndim: int
) -> tx.List[tx.Tuple[int, ...]]:
    """Return the chunks of the batch axes of the output.

    Along a batch axis, the output takes the chunks of the input when
    the input has the axis and is a dask array, then those of the grid,
    and otherwise a single chunk.

    Parameters
    ----------
    input : array
        The input, with shape `(*batch_input, *spatial_in)`.
    grid : array
        The grid, with shape `(*batch_grid, *spatial_out, ndim)`.
    batch : tuple of int
        The broadcast batch shape.
    ndim : int
        The number of spatial axes.

    Returns
    -------
    list of tuple of int
        The chunks of each batch axis.
    """
    chunks = []
    for k, length in enumerate(batch):
        chunk = (length,)
        for array, nbatch in ((input, input.ndim - ndim), (grid, None)):
            nbatch = grid.ndim - ndim - 1 if nbatch is None else nbatch
            axis = k - (len(batch) - nbatch)
            if axis >= 0 and array.shape[axis] > 1 and _is_dask(array):
                chunk = tuple(array.chunks[axis])
                break
        chunks.append(chunk)
    return chunks


def _rechunk_batch(
    array: tx.Any, nbatch: int, chunks: tx.List[tx.Tuple[int, ...]]
) -> tx.Any:
    """Give the batch axes of a dask array the chunks of the output.

    Parameters
    ----------
    array : dask.array.Array
        The array, whose first `nbatch` axes are batch axes.
    nbatch : int
        The number of batch axes of the array.
    chunks : list of tuple of int
        The chunks of the batch axes of the output, which has at least
        `nbatch` of them.

    Returns
    -------
    dask.array.Array
        The array, rechunked along its batch axes whose length is not 1.
    """
    offset = len(chunks) - nbatch
    new = {
        k: chunks[k + offset]
        for k in range(nbatch)
        if array.shape[k] > 1 and array.chunks[k] != chunks[k + offset]
    }
    return array.rechunk(new) if new else array


def _pull_dask(
    input: tx.Any,
    grid: tx.Any,
    order: tx.Any,
    bound: tx.Any,
    extrapolate: tx.Any,
    prefilter: bool,
    strategy: str,
    block_size: tx.Optional[int],
) -> tx.Any:
    """Interpolate an array at the coordinates of a grid, lazily.

    This function is the dask version of
    [`pull`][brainhops._ext.interpol.pull], which calls it when the
    input or the grid is a dask array, and takes the same arguments.
    The output has the chunks of the grid along the spatial axes, and
    along the batch axes the chunks of the input, or those of the grid
    where the input has no batch axis. A grid that is not a dask array
    is cut into chunks that follow those of the input when the input
    and the output have the same spatial shape, and into chunks of the
    default size of dask otherwise.

    Each block of the output is one task, which runs the eager
    interpolation on the window of the input that the coordinates of the
    block read (see [`_axis_window`][]). The graph takes one of two
    forms:

    - When the grid is a NumPy or a CuPy array and the input a dask
      array, the windows are known when the graph is built. Each task
      depends on the chunks of the input that its window overlaps, which
      dask computes once for all the tasks. With the prefilter, these
      chunks are those of the coefficients from
      [`_spline_coeff1d_dask`][], computed once per chunk.
    - When the grid is a dask array, the windows are only known when the
      tasks run. Each task then computes the part of the input that its
      window covers, with the synchronous scheduler, so that a chunk of
      the input that several windows overlap is computed by each of
      them. An input that is expensive to compute should therefore be
      persisted first. With the prefilter, each task reads its window
      extended by the halo of the prefilter, and computes the
      coefficients of its window itself, which costs less than computing
      the coefficients of every chunk that the window overlaps.

    The default strategy, "auto", is the batched one for every block,
    because the threaded scheduler of dask runs several blocks at once,
    and the many small operations of the blocked strategy would wait for
    each other to hold the lock of the Python interpreter.

    Parameters
    ----------
    input : array
        The argument `input` of [`pull`][brainhops._ext.interpol.pull].
    grid : array
        The argument `grid` of [`pull`][brainhops._ext.interpol.pull].
    order : int or sequence of int
        The argument `order` of [`pull`][brainhops._ext.interpol.pull].
    bound : str or sequence of str
        The argument `bound` of [`pull`][brainhops._ext.interpol.pull].
    extrapolate : bool or str
        The argument `extrapolate` of
        [`pull`][brainhops._ext.interpol.pull].
    prefilter : bool
        The argument `prefilter` of
        [`pull`][brainhops._ext.interpol.pull].
    strategy : str
        The argument `strategy` of [`pull`][brainhops._ext.interpol.pull],
        which applies to each block.
    block_size : int or None
        The argument `block_size` of
        [`pull`][brainhops._ext.interpol.pull], which applies to each
        block.

    Returns
    -------
    dask.array.Array
        The interpolated values, with shape `(*batch, *spatial_out)`.
    """
    xp = _namespace(input, grid)
    input = _to_namespace(xp, input)
    grid = _to_namespace(xp, grid)
    orders, bounds, margin, blocked = _check_arguments(
        xp, input, grid, order, bound, extrapolate, strategy, block_size
    )
    blocked = blocked and strategy != "auto"
    ndim = len(orders)
    dtype = _normalize_dtype(input.dtype)
    # With a concrete grid and a dask input, each task depends on the
    # chunks of the input that its window overlaps. Otherwise, it reads
    # its window from the whole input when it runs.
    explicit = _is_dask(input) and not _is_dask(grid)
    prefilter = bool(prefilter) and max(orders) > 1
    if prefilter and explicit:
        # The coefficients of each chunk are computed once.
        input = spline_coeff(input, orders, bounds, ndim)
        prefilter = False
    if input.dtype != dtype:
        input = input.astype(dtype)
    if grid.dtype.kind != "f":
        grid = grid.astype(np.float64)

    spatial_in = tuple(int(n) for n in input.shape[-ndim:])
    spatial_out = tuple(int(n) for n in grid.shape[-ndim - 1 : -1])
    nbatch_input = input.ndim - ndim
    nbatch_grid = grid.ndim - ndim - 1
    batch = tuple(
        np.broadcast_shapes(input.shape[:-ndim], grid.shape[: -ndim - 1])
    )
    nbatch = len(batch)
    batch_chunks = _batch_chunks(input, grid, batch, ndim)
    meta = xp.empty((0,) * (nbatch + ndim), dtype=dtype)

    if not explicit and _is_dask(grid):
        grid = _rechunk_batch(grid, nbatch_grid, batch_chunks)
        grid = grid.rechunk({grid.ndim - 1: -1})
    else:
        concrete = grid
        if _is_dask(input) and spatial_in == spatial_out:
            spatial_chunks = tuple(input.chunks[-ndim:])
        else:
            spatial_chunks = ("auto",) * ndim
        chunks = da.core.normalize_chunks(
            tuple(
                batch_chunks[k + nbatch - nbatch_grid] if n > 1 else (n,)
                for k, n in enumerate(grid.shape[:nbatch_grid])
            )
            + spatial_chunks
            + ((ndim,),),
            grid.shape,
            dtype=grid.dtype,
        )
        meta_grid = xp.empty((0,) * grid.ndim, dtype=grid.dtype)
        grid = da.from_array(
            grid, chunks=chunks, asarray=False, meta=meta_grid
        )
    if _is_dask(input):
        input = _rechunk_batch(input, nbatch_input, batch_chunks)
    out_chunks = tuple(batch_chunks) + tuple(grid.chunks[nbatch_grid:-1])
    if math.prod(batch) * math.prod(spatial_out) == 0:
        empty = xp.zeros(batch + spatial_out, dtype=dtype)
        return da.from_array(
            empty, chunks=out_chunks, asarray=False, meta=meta
        )

    options = dict(
        shape=spatial_in,
        orders=orders,
        bounds=bounds,
        margin=margin,
        blocked=blocked,
        block_size=block_size,
    )
    name = "interpol-pull-" + tokenize(
        input, grid, orders, bounds, margin, blocked, block_size, prefilter
    )
    layer: tx.Dict[tx.Any, tx.Any] = {}
    dependencies = [grid]
    if explicit:
        dependencies.append(input)
        grid_edges = [_edges(c) for c in grid.chunks]
        input_edges = [_edges(c) for c in input.chunks[nbatch_input:]]
    else:
        source_key = (name + "-source",)
        layer[source_key] = _Source(input)
        batch_edges = [
            _edges(c) for c in batch_chunks[nbatch - nbatch_input :]
        ]
    for index in itertools.product(*[range(len(c)) for c in out_chunks]):
        batch_index = index[:nbatch]
        grid_index = _batch_block(batch_index, grid.shape[:nbatch_grid])
        grid_index += index[nbatch:] + (0,)
        input_index = _batch_block(batch_index, input.shape[:nbatch_input])
        if explicit:
            select = tuple(
                slice(int(e[i]), int(e[i + 1]))
                for e, i in zip(grid_edges, grid_index)
            )
            windows = _windows(
                xp, concrete[select], spatial_in, orders, bounds
            )
            axes = [_whole(c, i) for c, i in zip(input.chunks, input_index)]
            axes += [
                _runs(edges, start, stop)
                for edges, (start, stop) in zip(input_edges, windows)
            ]
            keys, pieces, shape = _tiles(input.name, axes)
            task = (
                _pull_parts,
                (grid.name,) + grid_index,
                keys,
                pieces,
                shape,
                [start for start, _ in windows],
                options,
            )
        else:
            select = tuple(
                slice(int(e[i]), int(e[i + 1])) if n > 1 else slice(None)
                for e, i, n in zip(batch_edges, input_index, input.shape)
            )
            task = (
                _pull_source,
                (grid.name,) + grid_index,
                source_key,
                select,
                prefilter,
                options,
            )
        layer[(name,) + index] = task
    graph = HighLevelGraph.from_collections(
        name, layer, dependencies=dependencies
    )
    return da.Array(graph, name, out_chunks, meta=meta)


def _batch_block(
    index: tx.Tuple[int, ...], shape: tx.Sequence[int]
) -> tx.Tuple[int, ...]:
    """Return the block of an array that broadcasts against the batch.

    Parameters
    ----------
    index : tuple of int
        The index of a block of the output along its batch axes.
    shape : sequence of int
        The batch shape of the array, which may have fewer axes than
        the output, and whose axes of length 1 broadcast.

    Returns
    -------
    tuple of int
        The index of the block of the array along its batch axes, which
        is 0 along the axes of length 1.
    """
    index = index[len(index) - len(shape) :]
    return tuple(i if n > 1 else 0 for i, n in zip(index, shape))
