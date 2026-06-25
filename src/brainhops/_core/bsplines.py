# stdlib
import itertools

# dependencies
import typing_extensions as _tx

# locals
from .typing import ArrayProtocol
from .backends import get_array_backend, get_ndimage_backend


def pull(
    input: ArrayProtocol,
    coords: ArrayProtocol,
    order: int,
    bound: _tx.Union[str, float],
    coeff: bool
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
    # Get packages
    ab = get_array_backend(input)
    ib = get_ndimage_backend(input)
    # Get dimensions
    ndim = coords.shape[-1]
    batch = input.shape[:-ndim]
    # Prepare for map_coordinates
    coords = ab.moveaxis(coords, -1, 0)
    output = ab.empty_like(input, shape=batch + coords.shape[1:])
    mode = "constant" if not isinstance(bound, str) else bound
    cval = 0 if isinstance(bound, str) else bound
    opts = dict(order=order, mode=mode, cval=cval, prefilter=not coeff)
    # Interpolate each batch
    for index in itertools.product(*[range(s) for s in batch]):
        output[index] = ib.map_coordinates(input[index], coords, **opts)
    return output


def pull_field(field, coords, order, bound, coeff):
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
    ab = get_array_backend(field)
    field = ab.moveaxis(field, -1, 0)
    field = pull(field, coords, order, bound, coeff)
    field = ab.moveaxis(field, 0, -1)
    return field


def coeff2value(input, order, bound, inplace=False, ndim=None):
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
    ab = get_array_backend(input)
    ib = get_ndimage_backend(input)
    # Get dimensions
    ndim = ndim or input.ndim
    batch = input.shape[:-ndim]
    # Create coordinates field for interpolation
    grid = ab.meshgrid(*(ab.arange(s) for s in input.shape[:-ndim]), indexing='ij')
    grid = ab.stack(grid, axis=0)
    # Prepare for map_coordinates
    output = ab.empty_like(input) if not inplace else input
    mode = "constant" if not isinstance(bound, str) else bound
    cval = 0 if isinstance(bound, str) else bound
    opts = dict(order=order, mode=mode, cval=cval, prefilter=False)
    for index in itertools.product(*[range(s) for s in batch]):
        output[index] = ib.map_coordinates(input[index], grid, **opts)
    return output


def coeff2value_field(field, order, bound, inplace=False):
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
    ab = get_array_backend(field)
    ndim = field.shape[-1]
    field = ab.moveaxis(field, -1, 0)
    field = coeff2value(field, order, bound, inplace=inplace, ndim=ndim)
    field = ab.moveaxis(field, 0, -1)
    return field


def value2coeff(input, order, bound, inplace=False, ndim=None):
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
    ab = get_array_backend(input)
    ib = get_ndimage_backend(input)
    # Get dimensions
    ndim = ndim or input.ndim
    batch = input.shape[:-ndim]
    # Create coordinates field for interpolation
    grid = ab.meshgrid(*(ab.arange(s) for s in input.shape[:-ndim]), indexing='ij')
    grid = ab.stack(grid, axis=0)
    # Prepare for map_coordinates
    output = ab.empty_like(input) if not inplace else input
    mode = "constant" if not isinstance(bound, str) else bound
    cval = 0 if isinstance(bound, str) else bound
    opts = dict(order=order, mode=mode, cval=cval)
    for index in itertools.product(*[range(s) for s in batch]):
        output[index] = ib.spline_filter(input[index], **opts)
    return output


def value2coeff_field(field, order, bound, inplace=False):
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
    ab = get_array_backend(field)
    ndim = field.shape[-1]
    field = ab.moveaxis(field, -1, 0)
    field = value2coeff(field, order, bound, inplace=inplace, ndim=ndim)
    field = ab.moveaxis(field, 0, -1)
    return field
