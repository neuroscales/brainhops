import functools
import itertools
from types import ModuleType

import typing_extensions as tx
from bagof.hints.array import ArrayLike, ArrayProtocol

from brainhops._core.dependencies import da, np
from brainhops.backends import (
    best_backend,
    copy_array,
    get_array_backend,
    get_ndimage_backend,
)


def _scipy_boundary(bound: tx.Union[str, float]) -> tx.Tuple[str, float]:
    """Translate a boundary condition into the `(mode, cval)` of SciPy.

    A constant boundary, `"constant"` or a numeric fill value, becomes
    `"grid-constant"`, so that every coordinate beyond the grid takes the
    fill value at any degree, as in the zero padding of FNIRT coefficient
    fields. The `"constant"` mode of SciPy differs above degree 1.
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
    """Return the `map_coordinates` function of a backend.

    The dask function samples coordinates of shape `(ndim, *shape)` block
    by block, so only the functions of other backends are wrapped to
    flatten the coordinates.
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
    """Apply a function to each item of the batch dimensions of an array.

    Concrete arrays are filled item by item, into `output` when given.
    Dask results are stacked instead, because assigning a dask array into
    another rebuilds the graph of every chunk it covers.
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
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> ArrayProtocol:
    """Interpolate an array at the coordinates of a field.

    Parameters
    ----------
    input
        Array with shape `(*batch, *spatial_in)`.
    coords
        Coordinates with shape `(*spatial_out, ndim)`.
    degree
        Degree of the spline, from 0 to 5: 0 is nearest neighbour, 1 is
        linear, 2 is quadratic, and so on.
    bound
        Boundary condition, given by name or as a constant fill value. A
        number, or the name `"constant"` for zero, is the value beyond the
        edge. The names are:

          - 'nearest': nearest edge value   (a a a a | a b c d | d d d d)
          - 'reflect': reflect at edge      (d c b a | a b c d | d c b a)
          - 'mirror': mirror at edge        (d c b | a b c d | c b a)
          - 'grid-wrap': wrap around        (a b c d | a b c d | a b c d)
          - 'wrap': wrap around with shift  (d b c d | a b c d | b c a b)

    coeff
        Whether `input` already holds spline coefficients. When false, the
        input is prefiltered before it is interpolated.

    Returns
    -------
    ArrayProtocol
        The interpolated array, with shape `(*batch, *spatial_out)`.
    """
    nx = best_backend(input, coords)
    nd = get_ndimage_backend(nx)
    ndim = coords.shape[-1]
    batch = input.shape[:-ndim]
    coords = nx.moveaxis(coords, -1, 0)
    mode, cval = _scipy_boundary(bound)
    degree = int(degree)
    opts = {
        "order": degree,
        "mode": mode,
        "cval": cval,
        "prefilter": not coeff,
    }
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
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> ArrayProtocol:
    """Interpolate an array along a subset of its axes.

    The named axes, which the coordinates address in the order of their
    components, are moved to the end, and [`pull`][] is applied. The other
    axes become batch axes. A full-dimensional field of coordinates over a
    large array is thereby avoided.

    Parameters
    ----------
    input
        Array to interpolate.
    coords
        Coordinates with shape `(*spatial_out, len(axes))`.
    axes
        Axes of `input` that the coordinates address.
    degree
        Degree of the spline, from 0 to 5.
    bound
        Boundary condition, as in [`pull`][].
    coeff
        Whether `input` already holds spline coefficients.

    Returns
    -------
    ArrayProtocol
        The interpolated array. The unnamed axes come first, in their
        original order, followed by the spatial axes of the output.
    """
    nx = best_backend(input, coords)
    axes = [int(a) % input.ndim for a in axes]
    ndim = len(axes)
    dest = list(range(input.ndim - ndim, input.ndim))
    moved = nx.moveaxis(input, axes, dest)
    return pull(moved, coords, degree, bound, coeff)


def spline_matrix(
    n_in: int,
    coords_1d: ArrayProtocol,
    degree: int,
    bound: tx.Union[str, float],
    coeff: bool,
) -> ArrayProtocol:
    """Return the weight matrix of a one-dimensional spline resampling.

    Interpolation is linear, so the matrix `W`, applied along an axis of
    length `n_in`, reproduces [`pull`][] at `coords_1d` with a single
    contraction. When `coeff` is false, `W` includes the prefilter and maps
    values. With a numeric `bound`, `W` only holds the contributions from
    inside the array, and the fill value must be added as
    `cval * (1 - W.sum(axis=1))`.

    Parameters
    ----------
    n_in
        Length of the input axis.
    coords_1d
        Coordinates with shape `(n_out,)`.
    degree
        Degree of the spline, from 0 to 5.
    bound
        Boundary condition, as in [`pull`][].
    coeff
        Whether `W` maps coefficients rather than values.

    Returns
    -------
    ArrayProtocol
        The matrix `W`, with shape `(n_out, n_in)`.
    """
    nx = get_array_backend(coords_1d)
    bound0 = bound if isinstance(bound, str) else 0.0
    basis = nx.eye(n_in)
    coords = nx.reshape(nx.asarray(coords_1d), (-1, 1))
    return pull(basis, coords, degree, bound0, coeff).T


def pull_field(
    field: ArrayLike,
    coords: ArrayLike,
    degree: int,
    bound: tx.Literal["nearest", "reflect", "mirror", "grid-wrap", "wrap"],
    coeff: bool,
) -> ArrayLike:
    """Interpolate a field of displacements or coordinates.

    Parameters
    ----------
    field
        Field with shape `(*batch, *spatial_in, ndim)`.
    coords
        Coordinates with shape `(*spatial_out, ndim)`.
    degree
        Degree of the spline, from 0 to 5.
    bound
        Boundary condition, as in [`pull`][].
    coeff
        Whether `field` has already been prefiltered with the appropriate
        spline filter.

    Returns
    -------
    ArrayLike
        The interpolated field, with shape `(*batch, *spatial_out, ndim)`.
    """
    nx = best_backend(field, coords)
    field = nx.moveaxis(field, -1, 0)
    field = pull(field, coords, degree, bound, coeff)
    field = nx.moveaxis(field, 0, -1)
    return field


def coeff2value(
    input: ArrayLike,
    degree: int,
    bound: tx.Literal["nearest", "reflect", "mirror", "grid-wrap", "wrap"],
    inplace: bool = False,
    ndim: tx.Optional[int] = None,
) -> ArrayLike:
    """Convert spline coefficients to values.

    The conversion is the inverse of the spline filter, and is computed by
    interpolating the coefficients at the centres of the voxels.

    Parameters
    ----------
    input
        Coefficients with shape `(*batch, *spatial)`.
    degree
        Degree of the spline, from 0 to 5.
    bound
        Boundary condition, as in [`pull`][].
    inplace
        Whether to write the result into `input`.
    ndim
        Number of spatial axes. By default, all axes are spatial.

    Returns
    -------
    ArrayLike
        The values, with the same shape as `input`.
    """
    # Splines of degree 0 and 1 interpolate their coefficients, so the
    # coefficients are the values.
    if int(degree) < 2:
        return input if inplace else copy_array(input)
    nx = get_array_backend(input)
    nd = get_ndimage_backend(nx)
    # An ndim of 0 means that all axes are spatial; zero spatial axes would
    # be meaningless for interpolation.
    ndim = ndim or input.ndim
    batch = input.shape[:-ndim]
    grid = nx.meshgrid(
        *(nx.arange(s) for s in input.shape[-ndim:]), indexing="ij"
    )
    grid = nx.stack(grid, axis=0)
    mode, cval = _scipy_boundary(bound)
    degree = int(degree)
    opts = {"order": degree, "mode": mode, "cval": cval, "prefilter": False}
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
    degree: int,
    bound: tx.Literal["nearest", "reflect", "mirror", "grid-wrap", "wrap"],
    inplace: bool = False,
) -> ArrayLike:
    """Convert the spline coefficients of a field to values.

    The conversion is that of [`coeff2value`][], for a field with a trailing
    axis of components.

    Parameters
    ----------
    field
        Coefficients with shape `(*batch, *spatial, ndim)`.
    degree
        Degree of the spline, from 0 to 5.
    bound
        Boundary condition, as in [`pull`][].
    inplace
        Whether to write the result into `field`.

    Returns
    -------
    ArrayLike
        The values, with the same shape as `field`.
    """
    nx = get_array_backend(field)
    ndim = field.shape[-1]
    field = nx.moveaxis(field, -1, 0)
    field = coeff2value(field, degree, bound, inplace=inplace, ndim=ndim)
    field = nx.moveaxis(field, 0, -1)
    return field


def value2coeff(
    input: ArrayLike,
    degree: int,
    bound: tx.Literal["nearest", "reflect", "mirror", "grid-wrap", "wrap"],
    inplace: bool = False,
    ndim: tx.Optional[int] = None,
) -> ArrayLike:
    """Convert values to spline coefficients.

    Parameters
    ----------
    input
        Values with shape `(*batch, *spatial)`.
    degree
        Degree of the spline, from 0 to 5.
    bound
        Boundary condition, as in [`pull`][].
    inplace
        Whether to write the result into `input`.
    ndim
        Number of spatial axes. By default, all axes are spatial.

    Returns
    -------
    ArrayLike
        The coefficients, with the same shape as `input`.
    """
    # Splines of degree 0 and 1 interpolate their coefficients.
    if int(degree) < 2:
        return input if inplace else copy_array(input)
    nx = get_array_backend(input)
    nd = get_ndimage_backend(input)
    # An ndim of 0 means that all axes are spatial, since zero spatial axes
    # would be meaningless for a spline filter.
    ndim = ndim or input.ndim
    batch = input.shape[:-ndim]
    # The coefficients must be those that coeff2value interpolates exactly.
    # The prefilter of SciPy yields them only where its evaluation extends the
    # coefficients as the prefilter extends the values, so the coefficients
    # are solved for in the other modes.
    mode, cval = _scipy_boundary(bound)
    if mode in _SOLVED_MODES:
        bound = 0.0 if mode == "grid-constant" else mode

        def convert(x: ArrayProtocol) -> ArrayProtocol:
            return _interpolating_coefficients(x, int(degree), bound, cval)

    else:

        def convert(x: ArrayProtocol) -> ArrayProtocol:
            return nd.spline_filter(x, order=degree, mode=mode)

    input = ensure_floating(input)
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
"""Modes whose coefficients are solved for rather than filtered.

The prefilter of SciPy extends the values by reflection for `nearest` and
by mirroring for a constant, whereas its evaluation clamps the
coefficients or uses `cval`, so filtered coefficients do not interpolate
the values near the edges (#250). The `reflect` prefilter is approximate
near the edges of short axes, by up to 1e-4 at degree 5.
"""

_EDGE = 48
"""Number of samples at each end of an axis that a boundary affects.

A boundary changes the coefficients of these samples. The change decays as
`pole**distance`. The largest pole, at degree 5, is 0.43, and `0.43**48` is
about 3e-18.
"""


@functools.lru_cache(maxsize=None)
def _boundary_solvers(
    size: int, degree: int, bound: tx.Union[str, float]
) -> tx.Tuple[
    tx.Optional[np.ndarray],
    tx.Optional[tx.Tuple[np.ndarray, np.ndarray]],
    tx.Optional[tx.Tuple[np.ndarray, np.ndarray]],
    np.ndarray,
]:
    """Return how to solve for the coefficients of one axis under a boundary.

    The values are the coefficients times the [`spline_matrix`][] `A` of
    the boundary. A short axis is solved with `inv(A)`, the first item. A
    long axis is filtered with the mirror prefilter of SciPy, of matrix
    `M`, and corrected near each end: `inv(A) - inv(M)` is negligible
    beyond `_EDGE` samples, and its corner blocks have a rank of one per
    pole, since the boundary only changes the initial value of each
    recursion. Each correction is a pair `(U, V)`, applied as
    `U @ (V @ values)` to the first, or last, coefficients. The last item
    is `inv(A) @ 1`, which a constant boundary needs. Results are cached.
    """
    edge = _EDGE
    length = size if size <= 2 * edge else 4 * edge
    grid = np.arange(length, dtype=np.float64)
    matrix = np.asarray(spline_matrix(length, grid, degree, bound, True))
    inverse = np.linalg.inv(matrix)
    if length == size:
        return inverse, None, None, inverse.sum(axis=1)
    mirror = np.asarray(spline_matrix(length, grid, degree, "mirror", True))
    correction = inverse - np.linalg.inv(mirror)
    low = _low_rank(correction[:edge, :edge])
    # The high corner is trimmed from its far end, so it is flipped.
    high = _low_rank(correction[-edge:, -edge:][::-1, ::-1])
    high = (high[0][::-1], high[1][:, ::-1])
    # The mirror coefficients of a constant are that constant.
    ones = np.ones(size)
    ones[: low[0].shape[0]] += low[0] @ low[1].sum(axis=1)
    ones[size - high[0].shape[0] :] += high[0] @ high[1].sum(axis=1)
    return None, low, high, ones


_CORRECTION_TOLERANCE = 1e-17
"""Relative size below which an entry of a correction is dropped."""


def _low_rank(block: np.ndarray) -> tx.Tuple[np.ndarray, np.ndarray]:
    """Factor a corner block, starting at the edge, as `U @ V`.

    The factors have the numerical rank of the block. The rows of `U` and
    the columns of `V` are cut where they fall below the tolerance, relative
    to the largest entry.
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
    """Apply a matrix to an array along one axis."""
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
    """Return the rank-one terms of the edge correction `U @ (V @ slab)`.

    Each term is a pair `(u, w)`. The weights `w = V @ slab` are taken along
    the axis, which is kept with length one. Each `u` is a column of `U` laid
    along the axis and padded with zeros to `length`, at the start when the
    edge is at the end. The product `u * w` then broadcasts to the corrected
    part, without materialising anything larger than `w`.
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
    """Add edge corrections at both ends of an axis of a dask array.

    Each correction is a pair `(U, V)` with the slab that `V` reads. Only
    the chunks that a correction reaches are changed, by broadcasting its
    rank-one terms, so the result keeps the chunks of `like`.
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
        # The corrections overlap in a chunk, so both are added to the whole
        # axis.
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
    """Yield the corrections that turn a zero boundary into a constant one.

    With `L` the solve along each axis and `ones[d] = L_d @ 1`, the
    coefficients `L @ (values - cval) + cval` equal
    `L @ values + cval * (1 - prod_d ones[d])`. Each `ones[d]` is one away
    from the edges, so the corrections only cover slabs along the faces.
    Pairs of slices and corrections are yielded, covering each sample once.
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
                    # Skip samples that the slabs of an earlier axis cover.
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
    degree: int,
    bound: tx.Union[str, float],
    cval: float,
) -> ArrayProtocol:
    """Return the coefficients that interpolate values exactly.

    The coefficients beyond the edges are clamped for `nearest`, reflected
    for `reflect`, or equal to `cval` when `bound` is `0.0`. Each axis is
    prefiltered in turn by the recursive filter of SciPy and corrected at
    each end (see `_boundary_solvers`), so the cost is that of the filter.
    NumPy arrays are filtered in place after the first axis, and dask
    arrays keep their chunks. The solve is linear and separable, so
    `cval` only changes the coefficients near the faces, where it is
    applied (see `_constant_correction`).
    """
    nx = get_array_backend(values)
    nd = get_ndimage_backend(values)
    lazy = da is not None and isinstance(values, da.Array)
    coeff = values
    all_ones = []
    for axis in range(values.ndim):
        size = int(values.shape[axis])
        inverse, low, high, ones = _boundary_solvers(size, degree, bound)
        if inverse is not None:
            coeff = _along(nx, inverse, coeff, axis)
        else:
            index = (slice(None),) * axis
            # Corrections read the values before the in-place filter.
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
                degree,
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
    degree: int,
    bound: tx.Literal["nearest", "reflect", "mirror", "grid-wrap", "wrap"],
    inplace: bool = False,
) -> ArrayLike:
    """Convert the values of a field to spline coefficients.

    The conversion is that of [`value2coeff`][], for a field with a trailing
    axis of components.

    Parameters
    ----------
    field
        Values with shape `(*batch, *spatial, ndim)`.
    degree
        Degree of the spline, from 0 to 5.
    bound
        Boundary condition, as in [`pull`][].
    inplace
        Whether to write the result into `field`.

    Returns
    -------
    ArrayLike
        The coefficients, with the same shape as `field`.
    """
    nx = get_array_backend(field)
    ndim = field.shape[-1]
    field = nx.moveaxis(field, -1, 0)
    field = value2coeff(field, degree, bound, inplace=inplace, ndim=ndim)
    field = nx.moveaxis(field, 0, -1)
    return field


def ensure_floating(values: ArrayProtocol) -> ArrayProtocol:
    # Integer and boolean values have non-integer coefficients, which an
    # integer array would truncate.
    if values.dtype.kind in "biu":
        return values.astype("float32")
    return values
