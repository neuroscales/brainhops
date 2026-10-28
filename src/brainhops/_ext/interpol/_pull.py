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
from ._backend import _asarray, _get_namespace
from ._bounds import _normalize_bound
from ._coeff import (
    _normalize_dtype,
    _normalize_order,
    _normalize_per_axis,
    spline_coeff,
)
from ._taps import _axis_taps

_TOLERANCE = 5e-2
"""Margin, in voxels, that the mask of `extrapolate=False` allows.

It keeps the samples that rounding errors put just outside the edge
samples. The value comes from torch-interpol.
"""


def pull(
    input: tx.Any,
    grid: tx.Any,
    order: tx.Any = 1,
    bound: tx.Any = "reflect",
    extrapolate: tx.Any = True,
    prefilter: bool = False,
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

    The computation is fully batched: the indices and the weights along
    each axis are computed once for all the points, and the only loop in
    Python is over the `order + 1` samples, or taps, that each axis
    reads. Each tap gathers the samples of all the points and all the
    batch items at once. The memory that the computation uses is
    therefore about `2 * ndim * (order + 1)` arrays of the size of the
    grid, for the indices and the weights, and two arrays of the size of
    the output, for the gathered samples and their sum.

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
    xp = _get_namespace(input, grid)
    input = _asarray(xp, input)
    grid = _asarray(xp, grid)
    ndim = grid.shape[-1]
    if input.ndim < ndim or grid.ndim < ndim + 1:
        raise ValueError(
            f"The grid has {ndim} coordinates, so the input needs at "
            f"least {ndim} axes and the grid at least {ndim + 1}, but "
            f"they have {input.ndim} and {grid.ndim}."
        )
    orders = [
        _normalize_order(o) for o in _normalize_per_axis(order, ndim, "order")
    ]
    bounds = [
        _normalize_bound(b) for b in _normalize_per_axis(bound, ndim, "bound")
    ]
    margin = _normalize_extrapolate(extrapolate)
    dtype = _normalize_dtype(input.dtype)
    if prefilter and max(orders) > 1:
        input = spline_coeff(input, orders, bounds, ndim)
    input = input.astype(dtype, copy=False)
    if grid.dtype.kind != "f":
        grid = grid.astype(np.float64)

    spatial_in = tuple(input.shape[-ndim:])
    spatial_out = tuple(grid.shape[-ndim - 1 : -1])
    batch_input = tuple(input.shape[:-ndim])
    batch_grid = tuple(grid.shape[: -ndim - 1])
    batch = tuple(np.broadcast_shapes(batch_input, batch_grid))
    npoints = int(math.prod(spatial_out))
    # The flat index of a voxel is the sum of its indices times these.
    strides = [int(math.prod(spatial_in[d + 1 :])) for d in range(ndim)]

    # When NumPy's `take` writes into an output array and checks the
    # indices, it goes through a buffer, which `mode="clip"` avoids. The
    # indices are always valid, so clipping never changes them.
    take_options = {"mode": "clip"} if xp is np else {}

    if int(math.prod(batch_grid)) == 1:
        # The same points for every batch item: the input is read in
        # place, and each tap gathers whole rows of the batch.
        grid = grid.reshape((npoints, ndim))
        view, outer, inner = _as_view3d(xp, input, len(batch_input))

        def gather(index: tx.Any, values: tx.Any) -> None:
            xp.take(view, index, axis=1, out=values, **take_options)

        def spread(weight: tx.Any) -> tx.Any:
            return weight[:, None]

        shape = (view.shape[0], npoints, view.shape[2])
    else:
        # Points that depend on the batch: each tap gathers from the flat
        # input with indices that broadcast over the batch.
        grid = grid.reshape(batch_grid + (npoints, ndim))
        flat, offset = _flat_with_offsets(xp, input, batch, ndim)
        flat_index = xp.empty(batch + (npoints,), dtype=np.intp)

        def gather(index: tx.Any, values: tx.Any) -> None:
            xp.add(offset, index, out=flat_index)
            xp.take(flat, flat_index, out=values, **take_options)

        def spread(weight: tx.Any) -> tx.Any:
            return weight

        shape = batch + (npoints,)

    taps = []
    for d in range(ndim):
        pairs = _axis_taps(
            xp,
            grid[..., d],
            spatial_in[d],
            orders[d],
            bounds[d],
            dtype,
            strides[d],
        )
        if margin is not None and d == 0:
            inside = _inside(grid, spatial_in, margin)
            for _, weight in pairs:
                xp.multiply(weight, inside, out=weight, casting="unsafe")
        taps.append(pairs)
    out = xp.zeros(shape, dtype=dtype)
    _accumulate(xp, taps, out, gather, spread)

    if int(math.prod(batch_grid)) == 1:
        # The batch axes of the view are those of the input, in the
        # order of the memory, and are put back in their order.
        axes = outer + inner
        out = xp.moveaxis(out, 1, -1)
        out = out.reshape(tuple(input.shape[a] for a in axes) + spatial_out)
        order_back = [axes.index(a) for a in range(len(batch_input))]
        order_back += list(range(len(batch_input), out.ndim))
        out = xp.transpose(out, order_back)
    return out.reshape(batch + spatial_out)


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


def _inside(coords: tx.Any, shape: tx.Sequence[int], margin: float) -> tx.Any:
    inside = None
    for d, n in enumerate(shape):
        c = coords[..., d]
        mask = (c > -margin) & (c < n - 1 + margin)
        inside = mask if inside is None else inside & mask
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
    gather: tx.Callable[[tx.Any, tx.Any], None],
    spread: tx.Callable[[tx.Any], tx.Any],
) -> None:
    """Add the weighted samples of every combination of taps into `out`.

    The flat indices and the weights of the combinations are built axis
    by axis, so that only one partial index and one partial weight per
    axis are held in memory at a time. The samples of the taps of the
    last axis are summed before they are multiplied by the weight of the
    other axes, which saves one multiplication per combination. Every
    step writes into arrays that are allocated once, because allocating
    a large array for each tap is slow.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the arrays.
    taps : list of list of tuple
        For each axis, one pair `(flat index, weight)` per tap, computed
        for all the points.
    out : array
        The output, which the samples are added to.
    gather : callable
        The function that writes the samples at flat indices, for all
        the points and all the batch items, into an array of the shape
        of `out`.
    spread : callable
        The function that gives the weights of the points the shape that
        multiplies the gathered samples.
    """
    # The buffers of each axis are allocated once. The recursion is a
    # loop over the combinations of the taps of all but the last axis,
    # so that no closure refers to itself and the buffers are freed as
    # soon as the function returns.
    ndim = len(taps)
    indices = [xp.empty_like(taps[d][0][0]) for d in range(ndim)]
    weights = [xp.empty_like(taps[d][0][1]) for d in range(ndim)]
    values = xp.empty_like(out)
    total = xp.empty_like(out)
    partial: tx.List[tx.Tuple[tx.Any, tx.Any]] = [(None, None)] * ndim
    previous: tx.Tuple[int, ...] = ()
    for combination in itertools.product(*[range(len(t)) for t in taps[:-1]]):
        # Combine the taps of the leading axes, reusing the partial
        # products of the axes whose tap did not change.
        changed = 0
        while changed < len(previous) and (
            previous[changed] == combination[changed]
        ):
            changed += 1
        previous = combination
        for d in range(changed, len(combination)):
            k = combination[d]
            tap_index, tap_weight = taps[d][k]
            index, weight = partial[d - 1] if d else (None, None)
            if index is not None:
                tap_index = xp.add(index, tap_index, out=indices[d])
                tap_weight = xp.multiply(weight, tap_weight, out=weights[d])
            partial[d] = (tap_index, tap_weight)
        index, weight = partial[ndim - 2] if ndim > 1 else (None, None)
        # The taps of the last axis are summed first, and the sum is
        # then multiplied by the weight of the taps of the other axes.
        for k, (tap_index, tap_weight) in enumerate(taps[-1]):
            if index is not None:
                tap_index = xp.add(index, tap_index, out=indices[-1])
            target = total if k == 0 else values
            gather(tap_index, target)
            xp.multiply(target, spread(tap_weight), out=target)
            if k:
                xp.add(total, values, out=total)
        if weight is not None:
            xp.multiply(total, spread(weight), out=total)
        xp.add(out, total, out=out)
