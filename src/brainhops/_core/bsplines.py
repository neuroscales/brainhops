# stdlib
import itertools

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayLike, ArrayProtocol

# core
from brainhops.backends import (
    best_backend,
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
        input: ArrayProtocol,
        coords: ArrayProtocol, **kwargs
    ) -> ArrayProtocol:
        ndim, *oshape = coords.shape
        coords = coords.reshape((ndim, -1))
        output = map_coordinates(input, coords, **kwargs)
        return output.reshape(oshape)

    return _map_coordinates


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
    output = nx.empty_like(input, shape=batch + coords.shape[1:])
    mode, cval = _scipy_boundary(bound)
    order = int(order)
    opts = {"order": order, "mode": mode, "cval": cval, "prefilter": not coeff}
    # Interpolate each batch
    map_coordinates = _autoreshape(nd.map_coordinates)
    for index in itertools.product(*[range(s) for s in batch]):
        output[index] = map_coordinates(input[index], coords, **opts)
    return output


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
    output = nx.empty_like(input) if not inplace else input
    mode, cval = _scipy_boundary(bound)
    order = int(order)
    opts = {"order": order, "mode": mode, "cval": cval, "prefilter": False}
    map_coordinates = _autoreshape(nd.map_coordinates)
    for index in itertools.product(*[range(s) for s in batch]):
        output[index] = map_coordinates(input[index], grid, **opts)
    return output


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
    output = nx.empty_like(input) if not inplace else input
    mode = bound if isinstance(bound, str) else "constant"
    opts = dict(order=order, mode=mode)
    for index in itertools.product(*[range(s) for s in batch]):
        output[index] = nd.spline_filter(input[index], **opts)
    return output


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
