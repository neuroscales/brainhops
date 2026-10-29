"""Spline interpolation of images and fields, and their coefficients.

The functions of this module resample arrays with B-splines, for
example when an image is resliced or when a field of displacements is
composed with another one, and convert between the values of a field
and its spline coefficients. They are thin wrappers over
[`brainhops._ext.interpol`][], which does the work with NumPy, CuPy or
dask. The wrappers add what brainhops needs on top of it: the names of
[`BoundaryCondition`][brainhops.datamodel.enums.BoundaryCondition], a
numeric fill value beyond the edges, the margins of `clip`, and the
rules that choose the data type of the output.

A numeric boundary is a fill value `cval` beyond the edges. The
interpolation is linear, and a spline reproduces constants, so the
array is interpolated as `pull(x - cval, "zero") + cval`. This holds
with and without the prefilter, which then sets the coefficients beyond
the edges to `cval`.
"""

# stdlib
import math
from numbers import Real

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayLike, ArrayProtocol
from bagof.hints.numpy import DTypeLike

# brainhops
from brainhops._ext import interpol
from brainhops.datamodel.enums import BoundaryCondition, BoundaryLike

# core
from .backends import (
    array_backend,
    copy_array,
    is_cupy,
    is_dask,
    is_numpy,
    promote_backends,
    to_backend,
    to_backend_of,
    to_promoted_backend,
)
from .dependencies import np
from .typing import Clip

CLIP_TOLERANCE = 1e-5
"""Distance, in voxels, by which the margins of `clip` are widened.

A coordinate that should fall exactly on the limit of a margin, such as
the centre of an edge sample with `clip=0`, can land just outside it
because of rounding errors. The tolerance keeps such a coordinate. It
is the tolerance of the argument `clip` of
[`pull`][brainhops._ext.interpol.pull], so that both layers keep the
same samples.
"""


def interpol_boundary(bound: BoundaryLike) -> tx.Tuple[str, float]:
    """Return the boundary name of `interpol` and the fill value.

    A name is read through
    [`BoundaryCondition`][brainhops.datamodel.enums.BoundaryCondition],
    so that every alias is accepted, including `wrap`, which `interpol`
    refuses and which brainhops reads as `grid-wrap`. The constant
    boundary, whether it is given by name or as a number, becomes the
    zero boundary of `interpol`, together with its fill value, which is
    zero for the name. The constant boundary of brainhops is therefore
    the `grid-constant` mode of SciPy, not its `constant` mode.

    Parameters
    ----------
    bound : str or float
        A name or an alias of a boundary condition, or a fill value.

    Returns
    -------
    name : str
        The name that [`pull`][brainhops._ext.interpol.pull] accepts.
    cval : float
        The fill value beyond the edges, which is zero for a name.

    Raises
    ------
    ValueError
    """
    if isinstance(bound, str):
        name = BoundaryCondition(bound).value
        return ("zero" if name == "constant" else name), 0.0
    return "zero", float(bound)


def normalize_clip(
    clip: Clip, ndim: int
) -> tx.Optional[tx.Tuple[tx.Tuple[float, float], ...]]:
    """Return the margins of `clip` as one pair per sampled axis.

    Parameters
    ----------
    clip : float or tuple or sequence of tuple, optional
        The argument `clip` of [`pull`][].
    ndim : int
        The number of sampled axes.

    Returns
    -------
    tuple of tuple of float, optional
        The pair `(lower, upper)` of each axis, or None when `clip` is
        None.

    Raises
    ------
    ValueError
    """
    if clip is None:
        return None
    if isinstance(clip, Real):
        pairs = [(clip, clip)] * ndim
    else:
        try:
            items = tuple(clip)
            if len(items) == 2 and all(isinstance(x, Real) for x in items):
                pairs = [items] * ndim
            else:
                pairs = [tuple(pair) for pair in items]
            if len(pairs) != ndim or any(len(pair) != 2 for pair in pairs):
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(
                "clip must be a scalar, a (lower, upper) pair, or one "
                "pair per sampled dimension"
            ) from error
    if any(
        not isinstance(x, Real) or not math.isfinite(x)
        for pair in pairs
        for x in pair
    ):
        raise ValueError("clip margins must be finite real numbers")
    return tuple((float(lower), float(upper)) for lower, upper in pairs)


def clip_mask(
    coords: ArrayProtocol,
    shape: tx.Sequence[int],
    clip: tx.Optional[tx.Sequence[tx.Tuple[float, float]]],
) -> tx.Optional[ArrayProtocol]:
    """Return where coordinates lie inside the margins of `clip`.

    The mask is true where every coordinate lies in the closed interval
    `[-lower, size - 1 + upper]` of its axis, widened by
    [`CLIP_TOLERANCE`][] on each side.

    Parameters
    ----------
    coords : ArrayProtocol
        Coordinates with shape `(*spatial, ndim)`.
    shape : sequence of int
        The length of each sampled axis.
    clip : sequence of tuple of float, optional
        The margins, as returned by [`normalize_clip`][].

    Returns
    -------
    ArrayProtocol, optional
        The mask, with shape `spatial`, or None when `clip` is None.
    """
    if clip is None:
        return None
    mask = None
    for axis, (size, (lower, upper)) in enumerate(zip(shape, clip)):
        component = coords[..., axis]
        inside = (component >= -lower - CLIP_TOLERANCE) & (
            component <= size - 1 + upper + CLIP_TOLERANCE
        )
        mask = inside if mask is None else mask & inside
    return mask


def _sample(
    input: ArrayProtocol,
    coords: ArrayProtocol,
    degree: int,
    bound: BoundaryLike,
    prefilter: bool,
    clip: tx.Optional[tx.Tuple[tx.Tuple[float, float], ...]] = None,
) -> ArrayProtocol:
    """Interpolate an array with `interpol`, with any number of points.

    The grid of [`pull`][brainhops._ext.interpol.pull] has as many
    output axes as coordinates. Coordinates with another number of
    output axes, such as a list of points, are therefore flattened into
    a grid of shape `(npoints, 1, ..., 1, ndim)`, and the output is
    given back its shape. A numeric boundary is handled as explained in
    the documentation of the module. The margins of `clip` are passed to
    `interpol`, which zeroes the samples beyond them as it adds the
    taps. With a nonzero fill value, the fill value is added only inside
    the margins, so that the clipped samples stay zero.
    """
    name, cval = interpol_boundary(bound)
    ndim = coords.shape[-1]
    spatial = tuple(coords.shape[:-1])
    batch = tuple(input.shape[: input.ndim - ndim])
    grid = coords
    if len(spatial) != ndim:
        grid = coords.reshape((-1,) + (1,) * (ndim - 1) + (ndim,))
    if cval:
        input = input - cval
    output = interpol.pull(
        input, grid, degree, name, clip=clip, prefilter=prefilter
    )
    if cval:
        mask = clip_mask(coords, input.shape[input.ndim - ndim :], clip)
        if mask is None:
            output = output + cval
        else:
            mask = to_backend_of(mask.reshape(grid.shape[:-1]), output)
            xp = array_backend(output)
            output = output + xp.where(mask, cval, 0.0).astype(output.dtype)
    if len(spatial) != ndim:
        output = output.reshape(batch + spatial)
    return output


def pull(
    input: ArrayProtocol,
    coords: ArrayProtocol,
    degree: int,
    bound: BoundaryLike,
    coeff: bool,
    *,
    clip: Clip = None,
    dtype: tx.Optional[DTypeLike] = None,
) -> ArrayProtocol:
    """Interpolate an array at the coordinates of a field.

    This function samples an image or a field wherever a transformation
    sends each output voxel, with a spline of the given degree. The
    leading axes of the input that the coordinates do not address are
    batch axes, such as channels or time points, and all their items
    are sampled together. The work is done by
    [`pull`][brainhops._ext.interpol.pull], whose prefilter is exact for
    every boundary condition.

    Parameters
    ----------
    input : ArrayProtocol
        Array with shape `(*batch, *spatial_in)`.
    coords : ArrayProtocol
        Coordinates with shape `(*spatial_out, ndim)`, in voxels.
    degree : int
        Degree of the spline, from 0 to 7: 0 is nearest neighbour, 1 is
        linear, 2 is quadratic, and so on.
    bound : str or float
        Boundary condition, which says how the array continues beyond
        its edges. A number is a fill value beyond the edges, and the
        name `"constant"` stands for a fill value of zero. The names
        are those of
        [`BoundaryCondition`][brainhops.datamodel.enums.BoundaryCondition]
        and their aliases.
    coeff : bool
        Whether `input` already holds spline coefficients. When false,
        the input is prefiltered before it is interpolated.
    clip : float or tuple or sequence of tuple, optional
        Margins, in voxels, beyond which the output is zero. A scalar
        sets symmetric margins. A `(lower, upper)` pair sets margins
        shared by all sampled axes, and a sequence of pairs sets them
        in the order of the coordinates. A sample is zero when one of
        its coordinates lies outside the closed interval
        `[-lower, size - 1 + upper]`, widened by 1e-5 voxel so that
        rounding errors do not drop a coordinate on a limit, as in
        [`pull`][brainhops._ext.interpol.pull]. None keeps every sample.
        A margin of 0.5 includes the outer edges of the edge voxels, 0
        their centres and -0.5 their inner edges. Inside the margins,
        `bound` still gives the samples that the splines read beyond
        the edges. Clipped samples are zero even when `bound` is a
        nonzero number, and the coordinates are never clamped.
    dtype : DTypeLike, optional
        Output dtype. None preserves the input dtype. An explicit dtype
        promotes the input and output dtypes to choose the floating
        precision of the computation, which is at least float32.
        Integer-only combinations use float64. A floating output keeps
        the interpolated fractions, and an integer output is rounded
        half away from zero and saturated.

    Returns
    -------
    ArrayProtocol
        The interpolated array, with shape `(*batch, *spatial_out)`.
    """
    backend = promote_backends(input, coords)
    input = to_backend(input, backend)
    # Concrete coordinates are kept concrete for a dask input, so that
    # interpol knows the window of each block when it builds the graph.
    # It moves them to the device of the chunks itself.
    if not (is_dask(input) and (is_numpy(coords) or is_cupy(coords))):
        coords = to_backend(coords, backend)
    output_dtype = np.dtype(input.dtype if dtype is None else dtype)
    input = input.astype(sampling_dtype(input.dtype, dtype), copy=False)
    clip = normalize_clip(clip, coords.shape[-1])
    output = _sample(input, coords, int(degree), bound, not coeff, clip)
    return cast_output(output, output_dtype)


def sampling_dtype(
    input_dtype: DTypeLike, output_dtype: tx.Optional[DTypeLike]
) -> np.dtype:
    """Return the data type in which an array is interpolated.

    Without an output dtype, a floating or complex input is interpolated
    in its own dtype and any other input in float64. An explicit output
    dtype is promoted with the input dtype, with float32 as the lowest
    floating precision, and integer-only combinations use float64 so
    that the fractions are kept.

    Parameters
    ----------
    input_dtype : DTypeLike
        The dtype of the input.
    output_dtype : DTypeLike, optional
        The requested dtype of the output.

    Returns
    -------
    np.dtype
        The dtype of the computation.
    """
    input_dtype = np.dtype(input_dtype)
    if output_dtype is None:
        return (
            input_dtype
            if np.issubdtype(input_dtype, np.inexact)
            else np.dtype(float)
        )
    promoted = np.result_type(input_dtype, output_dtype)
    if not np.issubdtype(promoted, np.inexact):
        return np.result_type(promoted, np.float64)
    return np.result_type(promoted, np.float32)


def cast_output(values: ArrayProtocol, dtype: DTypeLike) -> ArrayProtocol:
    """Convert interpolated values to the dtype of the output.

    An integer output is rounded half away from zero and saturated to
    the range of its dtype, and a boolean output keeps the nonzero
    values after truncation, as SciPy does.

    Parameters
    ----------
    values : ArrayProtocol
        The interpolated values.
    dtype : DTypeLike
        The dtype of the output.

    Returns
    -------
    ArrayProtocol
        The values in the dtype of the output.
    """
    dtype = np.dtype(dtype)
    if values.dtype == dtype:
        return values
    xp = array_backend(values)
    if dtype.kind == "b":
        return xp.trunc(values).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        # Adding 0.5 in float32 can move a value just below a rounding
        # threshold onto the threshold. Convert before that addition.
        values = values.astype(
            np.result_type(values.dtype, np.float64), copy=False
        )
        rounded = xp.trunc(xp.where(values > 0, values + 0.5, values - 0.5))
        info = np.iinfo(dtype)
        return xp.clip(rounded, info.min, info.max).astype(dtype)
    return values.astype(dtype)


def pull_axes(
    input: ArrayProtocol,
    coords: ArrayProtocol,
    axes: tx.Sequence[int],
    degree: int,
    bound: BoundaryLike,
    coeff: bool,
    *,
    clip: Clip = None,
    dtype: tx.Optional[DTypeLike] = None,
) -> ArrayProtocol:
    """Interpolate an array along a subset of its axes.

    The axes listed in `axes` are moved to the end of the array, in the
    order in which the components of the coordinates address them, and
    [`pull`][] is then applied. The other axes become batch axes. In this
    way, a large array can be resampled along a few of its axes without
    building a field of coordinates over all of its dimensions.

    Parameters
    ----------
    input : ArrayProtocol
        Array to interpolate.
    coords : ArrayProtocol
        Coordinates with shape `(*spatial_out, len(axes))`.
    axes : sequence of int
        Axes of `input` that the coordinates address.
    degree : int
        Degree of the spline, from 0 to 7.
    bound : str or float
        Boundary condition, as in [`pull`][].
    coeff : bool
        Whether `input` already holds spline coefficients.
    clip : float or tuple or sequence of tuple, optional
        Margins beyond which the output is zero, as in [`pull`][]. The
        pairs of a sequence follow `axes`, not the order of the axes of
        the array.
    dtype : DTypeLike, optional
        Output dtype, as in [`pull`][]. None preserves the input dtype.

    Returns
    -------
    ArrayProtocol
        The interpolated array. The axes that are not listed in `axes`
        come first, in their original order, and are followed by the
        spatial axes of the output.
    """
    input, coords = to_promoted_backend(input, coords)
    xp = array_backend(input)
    axes = [int(a) % input.ndim for a in axes]
    ndim = len(axes)
    dest = list(range(input.ndim - ndim, input.ndim))
    moved = xp.moveaxis(input, axes, dest)
    return pull(moved, coords, degree, bound, coeff, clip=clip, dtype=dtype)


def pull_field(
    field: ArrayLike,
    coords: ArrayLike,
    degree: int,
    bound: BoundaryLike,
    coeff: bool,
) -> ArrayLike:
    """Interpolate a field of displacements or coordinates.

    Parameters
    ----------
    field : ArrayLike
        Field with shape `(*batch, *spatial_in, ndim)`.
    coords : ArrayLike
        Coordinates with shape `(*spatial_out, ndim)`.
    degree : int
        Degree of the spline, from 0 to 7.
    bound : str or float
        Boundary condition, as in [`pull`][].
    coeff : bool
        Whether `field` already holds spline coefficients.

    Returns
    -------
    ArrayLike
        The interpolated field, with shape `(*batch, *spatial_out, ndim)`.
    """
    backend = promote_backends(field, coords)
    xp = array_backend(backend)
    field = to_backend(field, backend)
    coords = to_backend(coords, backend)
    field = xp.moveaxis(field, -1, 0)
    field = pull(field, coords, degree, bound, coeff)
    field = xp.moveaxis(field, 0, -1)
    return field


def _write(
    input: ArrayProtocol, output: ArrayProtocol, inplace: bool
) -> ArrayProtocol:
    """Return the output, written into the input when `inplace` is set.

    A dask array is immutable, so the output is returned as it is.
    """
    if not inplace or is_dask(input):
        return output
    input[...] = output
    return input


def coeff2value(
    input: ArrayLike,
    degree: int,
    bound: BoundaryLike,
    inplace: bool = False,
    ndim: tx.Optional[int] = None,
) -> ArrayLike:
    """Convert spline coefficients to values.

    The values are those of the spline at the centres of the voxels,
    which is the inverse of the conversion of [`value2coeff`][]. A
    spline at integer coordinates is a separable convolution of its
    coefficients, so each spatial axis is sampled in turn by a
    one-dimensional [`pull`][brainhops._ext.interpol.pull], which reads
    `degree + 1` coefficients per value and axis.

    Parameters
    ----------
    input : ArrayLike
        Coefficients with shape `(*batch, *spatial)`.
    degree : int
        Degree of the spline, from 0 to 7.
    bound : str or float
        Boundary condition, as in [`pull`][].
    inplace : bool, default=False
        Whether to write the result into `input`.
    ndim : int, optional
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
    xp = array_backend(input)
    # An ndim of 0 means that all axes are spatial, since zero spatial axes
    # would be meaningless for interpolation.
    ndim = ndim or input.ndim
    name, cval = interpol_boundary(bound)
    values = input - cval if cval else input
    for axis in range(input.ndim - ndim, input.ndim):
        size = int(values.shape[axis])
        # The grid is a NumPy array, which interpol moves to the device
        # of the input, and whose chunks follow those of a dask input.
        grid = np.arange(size, dtype=np.float64)[:, None]
        moved = xp.moveaxis(values, axis, -1)
        moved = interpol.pull(moved, grid, int(degree), name)
        values = xp.moveaxis(moved, -1, axis)
    if cval:
        values = values + cval
    return _write(input, values, inplace)


def coeff2value_field(
    field: ArrayLike,
    degree: int,
    bound: BoundaryLike,
    inplace: bool = False,
) -> ArrayLike:
    """Convert the spline coefficients of a field to values.

    The conversion is the one performed by [`coeff2value`][], applied to a
    field whose last axis holds the components of each vector.

    Parameters
    ----------
    field : ArrayLike
        Coefficients with shape `(*batch, *spatial, ndim)`.
    degree : int
        Degree of the spline, from 0 to 7.
    bound : str or float
        Boundary condition, as in [`pull`][].
    inplace : bool, default=False
        Whether to write the result into `field`.

    Returns
    -------
    ArrayLike
        The values, with the same shape as `field`.
    """
    field = to_backend(field)
    xp = array_backend(field)
    ndim = field.shape[-1]
    field = xp.moveaxis(field, -1, 0)
    field = coeff2value(field, degree, bound, inplace=inplace, ndim=ndim)
    field = xp.moveaxis(field, 0, -1)
    return field


def value2coeff(
    input: ArrayLike,
    degree: int,
    bound: BoundaryLike,
    inplace: bool = False,
    ndim: tx.Optional[int] = None,
) -> ArrayLike:
    """Convert values to spline coefficients.

    The coefficients are those of the spline that passes through the
    values at the centres of the voxels, so that [`coeff2value`][]
    gives the values back. They are computed by
    [`spline_coeff`][brainhops._ext.interpol.spline_coeff], which is
    exact for every boundary condition. With a numeric boundary, the
    coefficients beyond the edges equal the fill value.

    Parameters
    ----------
    input : ArrayLike
        Values with shape `(*batch, *spatial)`.
    degree : int
        Degree of the spline, from 0 to 7.
    bound : str or float
        Boundary condition, as in [`pull`][].
    inplace : bool, default=False
        Whether to write the result into `input`.
    ndim : int, optional
        Number of spatial axes. By default, all axes are spatial.

    Returns
    -------
    ArrayLike
        The coefficients, with the same shape as `input`.
    """
    # Splines of degree 0 and 1 interpolate their coefficients, so the
    # values are the coefficients.
    if int(degree) < 2:
        return input if inplace else copy_array(input)
    # An ndim of 0 means that all axes are spatial, since zero spatial axes
    # would be meaningless for a spline filter.
    ndim = ndim or input.ndim
    name, cval = interpol_boundary(bound)
    values = ensure_floating(input)
    if cval:
        values = values - cval
    coeff = interpol.spline_coeff(values, int(degree), name, ndim)
    if cval:
        coeff = coeff + cval
    return _write(input, coeff, inplace)


def value2coeff_field(
    field: ArrayLike,
    degree: int,
    bound: BoundaryLike,
    inplace: bool = False,
) -> ArrayLike:
    """Convert the values of a field to spline coefficients.

    The conversion is the one performed by [`value2coeff`][], applied to a
    field whose last axis holds the components of each vector.

    Parameters
    ----------
    field : ArrayLike
        Values with shape `(*batch, *spatial, ndim)`.
    degree : int
        Degree of the spline, from 0 to 7.
    bound : str or float
        Boundary condition, as in [`pull`][].
    inplace : bool, default=False
        Whether to write the result into `field`.

    Returns
    -------
    ArrayLike
        The coefficients, with the same shape as `field`.
    """
    field = to_backend(field)
    xp = array_backend(field)
    ndim = field.shape[-1]
    field = xp.moveaxis(field, -1, 0)
    field = value2coeff(field, degree, bound, inplace=inplace, ndim=ndim)
    field = xp.moveaxis(field, 0, -1)
    return field


def ensure_floating(values: ArrayProtocol) -> ArrayProtocol:
    """Return the values in a floating dtype.

    Integer and boolean values have coefficients that are not integers,
    which an integer array would truncate, so they are converted to
    float32. Other values are returned unchanged.

    Parameters
    ----------
    values : ArrayProtocol
        The values.

    Returns
    -------
    ArrayProtocol
        The values, in a floating or complex dtype.
    """
    if values.dtype.kind in "biu":
        return values.astype("float32")
    return values
