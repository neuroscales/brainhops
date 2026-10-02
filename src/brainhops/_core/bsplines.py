# stdlib
import functools
import itertools
from types import ModuleType

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayLike, ArrayProtocol

# core
from brainhops._core.dependencies import da, np
from brainhops.backends import (
    best_backend,
    copy_array,
    get_array_backend,
    get_ndimage_backend,
)


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


def _map_coordinates_for(nx: ModuleType, nd: ModuleType) -> tx.Callable:
    """The (autoreshaping) ``map_coordinates`` to use for a backend.

    The dask backend's (see [`brainhops._core.dask_ndimage`][]) samples
    coordinates of shape `(ndim, *shape)` block by block, keeping their
    spatial layout, so it is not flattened; every other backend's is.
    """
    if da is not None and nx is da:
        return nd.map_coordinates
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
    # The coefficients are those that `coeff2value` interpolates exactly.
    # scipy's prefilter computes them for the bounds whose evaluation
    # extends the coefficients the way the prefilter extends the values.
    # `nearest` and constant bounds are not among them, and are solved.
    mode, cval = _scipy_boundary(bound)
    if mode in _SOLVED_MODES:
        bound = 0.0 if mode == "grid-constant" else mode

        def convert(x: ArrayProtocol) -> ArrayProtocol:
            return _interpolating_coefficients(x, int(order), bound, cval)

    else:

        def convert(x: ArrayProtocol) -> ArrayProtocol:
            return nd.spline_filter(x, order=order, mode=mode)

    output = _over_batch(
        nx,
        input,
        batch,
        convert,
        tuple(input.shape[-ndim:]),
        output=input if inplace else None,
    )
    if da is not None and isinstance(output, da.Array):
        output = output.rechunk(input.chunks)
    return output


_SOLVED_MODES = ("nearest", "grid-constant", "reflect")
"""
The modes whose coefficients `value2coeff` solves for rather than filters.

scipy's prefilter extends the values by reflection for `nearest`, and by
mirroring for `constant`, while its evaluation extends the coefficients
by clamping for `nearest` and by `cval` for `grid-constant` (which is what
constant bounds are evaluated with). The two do not match, and filtered
coefficients do not interpolate the values near the edges (see #250).
scipy's `reflect` prefilter does match its evaluation, but approximates
it near the edges of a short axis, by up to 1e-4 at order five.
"""

_EDGE = 48
"""
The samples at each end of an axis whose coefficients a boundary changes.

A boundary changes the coefficients by an amount that decays as the
spline's pole to the power of the distance to the edge; the largest pole,
at order five, is 0.43, and 0.43 ** 48 is about 3e-18.
"""


@functools.lru_cache(maxsize=None)
def _boundary_solvers(
    size: int, order: int, bound: tx.Union[str, float]
) -> tx.Tuple[
    tx.Optional[np.ndarray],
    tx.Optional[tx.Tuple[np.ndarray, np.ndarray]],
    tx.Optional[tx.Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
]:
    """
    How to solve for the coefficients of one axis under a boundary.

    The values of an axis are its coefficients times `A`, the matrix of
    [`spline_matrix`][] under `bound` -- clamped coefficients for
    `nearest`, reflected ones for `reflect`, and zeros past the edges for
    `0.0`. A short axis is solved with the inverse of `A`, returned first.

    A long one is solved by scipy's recursive `mirror` prefilter, whose
    matrix `M` is the same but with mirrored coefficients past the edges,
    then corrected near each end. `inv(A) - inv(M)` is negligible farther
    than `_EDGE` samples from either end, and its corner blocks have rank
    one per pole of the prefilter -- one at orders 2 and 3, two at orders
    4 and 5: they change the initial value of each pole's recursion. So
    the correction at each end is returned as a pair `(U, V)`, applied as
    `U @ (V @ values[edge])`, which costs a few operations per sample of
    the edge. Both are trimmed to the samples the correction reaches at
    this order: about 20 at order 2, 28 at order 3 and 44 at order 5. At
    the low end `V` reads the first `V.shape[1]` values and `U` corrects
    the first `U.shape[0]` coefficients; at the high end, the last ones.

    The last item is `inv(A) @ 1`, which a constant bound needs.
    """
    edge = _EDGE
    length = size if size <= 2 * edge else 4 * edge
    grid = np.arange(length, dtype=np.float64)
    matrix = np.asarray(spline_matrix(length, grid, order, bound, True))
    inverse = np.linalg.inv(matrix)
    if length == size:
        return inverse, None, None, inverse.sum(axis=1)
    mirror = np.asarray(spline_matrix(length, grid, order, "mirror", True))
    correction = inverse - np.linalg.inv(mirror)
    low = _low_rank(correction[:edge, :edge])
    # The high corner is trimmed, and returned, from its far end.
    high = _low_rank(correction[-edge:, -edge:][::-1, ::-1])
    high = (high[0][::-1], high[1][:, ::-1])
    # The coefficients of a constant under `mirror` are that constant.
    ones = np.ones(size)
    ones[: low[0].shape[0]] += low[0] @ low[1].sum(axis=1)
    ones[size - high[0].shape[0] :] += high[0] @ high[1].sum(axis=1)
    return None, low, high, ones


_CORRECTION_TOLERANCE = 1e-17
"""The relative size below which an entry of a correction is dropped."""


def _low_rank(block: np.ndarray) -> tx.Tuple[np.ndarray, np.ndarray]:
    """
    A corner block, starting at the edge, as `U @ V` of the rank it has to
    machine precision, with `U`'s rows and `V`'s columns cut where they
    fall below `_CORRECTION_TOLERANCE` of the largest entry.
    """
    u, s, vt = np.linalg.svd(block)
    rank = max(1, int((s > s[0] * 1e-14).sum()))
    u, vt = u[:, :rank] * s[:rank], vt[:rank]
    scale = np.abs(block).max() * _CORRECTION_TOLERANCE
    rows = np.flatnonzero(np.abs(u).max(axis=1) * np.abs(vt).max() > scale)
    cols = np.flatnonzero(np.abs(vt).max(axis=0) * np.abs(u).max() > scale)
    return u[: rows[-1] + 1], vt[:, : cols[-1] + 1]


def _along(
    nx: ModuleType, matrix: np.ndarray, x: ArrayProtocol, axis: int
) -> ArrayProtocol:
    """`matrix` applied to `x` along one of its axes."""
    matrix = nx.asarray(matrix, dtype=np.float64)
    return nx.moveaxis(nx.tensordot(matrix, x, axes=([1], [axis])), 0, axis)


def _edge_terms(
    nx: ModuleType,
    pair: tx.Tuple[np.ndarray, np.ndarray],
    slab: ArrayProtocol,
    axis: int,
    length: int,
    at_end: bool,
) -> tx.List[tx.Tuple[np.ndarray, ArrayProtocol]]:
    """
    The rank-one terms `(u, w)` of the edge correction `U @ (V @ slab)`.

    `w` holds `V @ slab` along `axis`, kept as an axis of length one, and
    `u` is a column of `U` laid along `axis` and padded with zeros to
    `length` (at its start when the edge is at the end of the axis), so
    that `u * w` broadcasts to the corrected part of the array without
    materializing anything larger than `w`.
    """
    u, v = pair
    weights = nx.tensordot(
        nx.asarray(v, dtype=np.float64), slab, axes=([1], [axis])
    )
    shape = [1] * slab.ndim
    shape[axis] = -1
    terms = []
    for j in range(u.shape[1]):
        column = np.zeros(length)
        if at_end:
            column[length - u.shape[0] :] = u[:, j]
        else:
            column[: u.shape[0]] = u[:, j]
        terms.append((column.reshape(shape), nx.expand_dims(weights[j], axis)))
    return terms


def _add_edges(
    like: ArrayProtocol,
    axis: int,
    low: tx.Tuple[tx.Tuple[np.ndarray, np.ndarray], ArrayProtocol],
    high: tx.Tuple[tx.Tuple[np.ndarray, np.ndarray], ArrayProtocol],
) -> ArrayProtocol:
    """
    A dask array `like`, with the edge corrections `low` and `high` --
    each a `(U, V)` pair and the slab of values `V` reads -- added at the
    two ends of `axis`.

    The axis is split at chunk boundaries: the chunks the corrections
    reach are corrected by broadcasting their rank-one terms, which dask
    fuses into one task per chunk, and the chunks in between are passed
    through untouched, so the result keeps the chunks of `like`.
    """
    size = int(like.shape[axis])
    depth_low = int(low[0][0].shape[0])
    depth_high = int(high[0][0].shape[0])
    bounds = np.cumsum((0,) + tuple(like.chunks[axis]))
    stop = int(bounds[np.searchsorted(bounds, depth_low)])
    start = int(
        bounds[np.searchsorted(bounds, size - depth_high, "right") - 1]
    )
    index = (slice(None),) * axis
    if stop > start:
        stop, start = size, 0
    pieces = []
    for (pair, slab), first, last, at_end in (
        (low, 0, stop, False),
        (high, start, size, True),
    ):
        part = like[index + (slice(first, last),)]
        for column, weights in _edge_terms(
            da, pair, slab, axis, last - first, at_end
        ):
            part = part + column * weights
        pieces.append(part)
    if (stop, start) == (size, 0):
        # The two corrections share chunks: they are both added to the
        # whole axis.
        whole = pieces[0]
        for column, weights in _edge_terms(
            da, high[0], high[1], axis, size, True
        ):
            whole = whole + column * weights
        return whole
    middle = like[index + (slice(stop, start),)]
    return da.concatenate([pieces[0], middle, pieces[1]], axis=axis)


def _constant_correction(
    ones: tx.Sequence[np.ndarray], cval: float
) -> tx.Iterator[tx.Tuple[tx.Tuple[slice, ...], np.ndarray]]:
    """
    Where, and by how much, coefficients filtered as if they were zero past
    the edges must change to be `cval` past the edges.

    With `L` the per-axis solve and `ones[d] = L_d @ 1`, the coefficients
    are `L @ (values - cval) + cval = L @ values + cval * (1 - prod_d
    ones[d])`. Each `ones[d]` is one but within its edge depths, so the
    correction is zero but on the slabs along the faces of the array.
    They are yielded, each sample once, as `(slices, correction)` pairs.
    """
    ndim = len(ones)
    edges = [np.flatnonzero(np.abs(o - 1) > 0) for o in ones]
    for axis in range(ndim):
        size = len(ones[axis])
        near = edges[axis]
        if near.size == 0:
            continue
        for part in (near[near < size // 2], near[near >= size // 2]):
            if part.size == 0:
                continue
            region = []
            factors = []
            for d in range(ndim):
                n = len(ones[d])
                if d == axis:
                    sl = slice(int(part[0]), int(part[-1]) + 1)
                elif d < axis and edges[d].size:
                    # Samples already in an earlier axis's slabs are skipped.
                    lo = int(edges[d][edges[d] < n // 2].max(initial=-1)) + 1
                    hi = int(edges[d][edges[d] >= n // 2].min(initial=n))
                    sl = slice(lo, hi)
                else:
                    sl = slice(0, n)
                region.append(sl)
                shape = [1] * ndim
                shape[d] = -1
                factors.append(ones[d][sl].reshape(shape))
            if any(sl.stop <= sl.start for sl in region):
                continue
            product = factors[0]
            for factor in factors[1:]:
                product = product * factor
            yield tuple(region), cval * (1 - product)


def _interpolating_coefficients(
    values: ArrayProtocol,
    order: int,
    bound: tx.Union[str, float],
    cval: float,
) -> ArrayProtocol:
    """
    The spline coefficients that interpolate `values` exactly, when the
    coefficients past the edges are clamped (`bound="nearest"`), reflected
    (`bound="reflect"`), or are `cval` (`bound=0.0`).

    Each axis is prefiltered in turn, by scipy's recursive filter with
    corrected initial values at each end (see [`_boundary_solvers`][]):
    the cost is the filter's. A numpy array is filtered in place after the
    first axis, so it takes no more memory than the filter does. A dask
    array keeps its chunks, and each chunk reads only its neighbours and
    the samples near each end of its axes.

    Coefficients that are `cval` past the edges interpolate `values` when
    `coefficients - cval`, zero past the edges, interpolate `values -
    cval`. The solve is linear and separable, so `cval` changes the
    coefficients only near the faces of the array, and is applied there
    (see [`_constant_correction`][]) rather than to a copy of the values.
    """
    nx = get_array_backend(values)
    nd = get_ndimage_backend(values)
    lazy = da is not None and isinstance(values, da.Array)
    coeff = values
    all_ones = []
    for axis in range(values.ndim):
        size = int(values.shape[axis])
        inverse, low, high, ones = _boundary_solvers(size, order, bound)
        if inverse is not None:
            coeff = _along(nx, inverse, coeff, axis)
        else:
            index = (slice(None),) * axis
            # The corrections read the values before they are filtered.
            low = (low, coeff[index + (slice(0, low[1].shape[1]),)])
            high = (
                high,
                coeff[index + (slice(size - high[1].shape[1], size),)],
            )
            if not lazy:
                low = _edge_terms(
                    np, low[0], low[1], axis, low[0][0].shape[0], False
                )
                high = _edge_terms(
                    np, high[0], high[1], axis, high[0][0].shape[0], True
                )
            inplace = not lazy and coeff is not values
            filtered = nd.spline_filter1d(
                coeff,
                order,
                axis=axis,
                output=coeff if inplace else np.float64,
                mode="mirror",
            )
            if inplace:
                filtered = coeff
            if lazy:
                coeff = _add_edges(filtered, axis, low, high)
            else:
                for terms, region in (
                    (low, slice(0, low[0][0].shape[axis])),
                    (high, slice(size - high[0][0].shape[axis], size)),
                ):
                    target = filtered[index + (region,)]
                    for column, weights in terms:
                        target += column * weights
                coeff = filtered
        all_ones.append(ones)
    if cval:
        if lazy:
            # `cval * (1 - prod_d ones[d])`, broadcast chunk by chunk.
            shape = [1] * coeff.ndim
            shape[0] = -1
            product = da.from_array(
                all_ones[0].reshape(shape),
                chunks=(coeff.chunks[0],) + (1,) * (coeff.ndim - 1),
            )
            for d in range(1, coeff.ndim):
                shape = [1] * coeff.ndim
                shape[d] = -1
                product = product * all_ones[d].reshape(shape)
            coeff = coeff + cval * (1 - product)
        else:
            for region, correction in _constant_correction(all_ones, cval):
                coeff[region] += correction
    return coeff


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
