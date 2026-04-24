# stdlib
import itertools
from types import ModuleType

# externals
import typing_extensions as _tx

# internals
from .typing import ArrayProtocol, npt, cpt, dkt

# optionals
try:
    import scipy.ndimage as npndi
except ImportError:
    npndi = None
try:
    import cupyx.scipy.ndimage as cpndi
except ImportError:
    cpndi = None
try:
    import dask_image.ndinterp as dkndi
except ImportError:
    dkndi = None


def _get_origin(type: _tx.Any, unfold: _tx.Any = None) -> _tx.Any:
    origin = _tx.get_origin(type)
    if origin is None:
        return type
    if unfold == "all":
        if _tx.get_args(type):
            return _get_origin(_tx.get_args(type)[0], unfold=unfold)
    if unfold:
        if not isinstance(unfold, (list, tuple, set)):
            unfold = (unfold,)
        if origin in unfold:
            return _get_origin(_tx.get_args(type)[0], unfold=unfold)
    return origin


def _get_array_package(x: ArrayProtocol) -> ModuleType:
    """Determine the array package for a given array
    
    One of: numpy, cupy, dask.array
    """
    if npt.np and isinstance(x, npt.np.ndarray):
        return npt.np
    if cpt.cp and isinstance(x, cpt.cp.ndarray):
        return cpt.cp
    if dkt.da and isinstance(x, dkt.da.ndarray):
        return dkt.da
    raise TypeError(f"Unsupported array type: {type(x)}")


def _get_ndimage_package(x: ArrayProtocol) -> ModuleType:
    """Determine the ndimage package for a given array

    One of: scipy.ndimage, cupyx.scipy.ndimage, dask_image.ndinterp
    """
    if cpt.cp and cpndi and isinstance(x, cpt.cp.ndarray):
        return cpndi
    if npt.np and npndi and isinstance(x, npt.np.ndarray):
        return npndi
    if dkt.da and isinstance(x, dkt.da.ndarray):
        if dkndi:
            return dkndi
        if npndi:
            return npndi
    raise TypeError(f"Unsupported array type: {type(x)}")


def _pull(
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
    nx = _get_array_package(input)
    ndx = _get_ndimage_package(input)
    # Get dimensions
    ndim = coords.shape[-1]
    batch = input.shape[:-ndim]
    # Prepare for map_coordinates
    coords = nx.moveaxis(coords, -1, 0)
    output = nx.empty_like(input, shape=batch + coords.shape[1:])
    mode = "constant" if not isinstance(bound, str) else bound
    cval = 0 if isinstance(bound, str) else bound
    opts = dict(order=order, mode=mode, cval=cval, prefilter=not coeff)
    # Interpolate each batch
    for index in itertools.product(*[range(s) for s in batch]):
        output[index] = ndx.map_coordinates(input[index], coords, **opts)
    return output


def _pull_field(field, coords, order, bound, coeff):
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
    nx = _get_array_package(field)
    field = nx.moveaxis(field, -1, 0)
    field = _pull(field, coords, order, bound, coeff)
    field = nx.moveaxis(field, 0, -1)
    return field


def _coeff2value(input, order, bound, inplace=False, ndim=None):
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
    nx = _get_array_package(input)
    ndx = _get_ndimage_package(input)
    # Get dimensions
    ndim = ndim or input.ndim
    batch = input.shape[:-ndim]
    # Create coordinates field for interpolation
    grid = nx.meshgrid(*(nx.arange(s) for s in input.shape[:-ndim]), indexing='ij')
    grid = nx.stack(grid, axis=0)
    # Prepare for map_coordinates
    output = nx.empty_like(input) if not inplace else input
    mode = "constant" if not isinstance(bound, str) else bound
    cval = 0 if isinstance(bound, str) else bound
    opts = dict(order=order, mode=mode, cval=cval, prefilter=False)
    for index in itertools.product(*[range(s) for s in batch]):
        output[index] = ndx.map_coordinates(input[index], grid, **opts)
    return output


def _coeff2value_field(field, order, bound, inplace=False):
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
    nx = _get_array_package(field)
    ndim = field.shape[-1]
    field = nx.moveaxis(field, -1, 0)
    field = _coeff2value(field, order, bound, inplace=inplace, ndim=ndim)
    field = nx.moveaxis(field, 0, -1)
    return field


def _value2coeff(input, order, bound, inplace=False, ndim=None):
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
    nx = _get_array_package(input)
    ndx = _get_ndimage_package(input)
    # Get dimensions
    ndim = ndim or input.ndim
    batch = input.shape[:-ndim]
    # Create coordinates field for interpolation
    grid = nx.meshgrid(*(nx.arange(s) for s in input.shape[:-ndim]), indexing='ij')
    grid = nx.stack(grid, axis=0)
    # Prepare for map_coordinates
    output = nx.empty_like(input) if not inplace else input
    mode = "constant" if not isinstance(bound, str) else bound
    cval = 0 if isinstance(bound, str) else bound
    opts = dict(order=order, mode=mode, cval=cval)
    for index in itertools.product(*[range(s) for s in batch]):
        output[index] = ndx.spline_filter(input[index], **opts)
    return output


def _value2coeff_field(field, order, bound, inplace=False):
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
    nx = _get_array_package(field)
    ndim = field.shape[-1]
    field = nx.moveaxis(field, -1, 0)
    field = _value2coeff(field, order, bound, inplace=inplace, ndim=ndim)
    field = nx.moveaxis(field, 0, -1)
    return field