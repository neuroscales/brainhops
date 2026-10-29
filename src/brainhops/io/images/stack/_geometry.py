"""The geometry of an image stack: from the record to the transformations
of the image, and back."""

# stdlib
import math

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel._internal_api import get_axes
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.transformations import Affine, Transformation
from brainhops.io.images.base import _utils_raster as raster

# this format
from ._raw import PLANES, StackRaw

_Fields = tx.Tuple[
    tx.Optional[tx.Tuple[float, float]], tx.Optional[float], tx.Optional[str]
]
"""The geometry fields of a record: `pixel_size`, `spacing` and `unit`."""


def stack_transformations(record: StackRaw) -> tx.List[Transformation]:
    """Return the voxel-to-physical scaling that a record describes.

    The size of the pixels scales the axes along the columns and the rows
    of the files, and the spacing scales the stacking axis, with its sign.
    An axis whose size is unknown is scaled by one, without a unit, as for
    a single TIFF or Pillow image. A record without members gives no
    transformation.

    Parameters
    ----------
    record : StackRaw
        The record.

    Returns
    -------
    list[Transformation]
        One [`Scaling`][brainhops.datamodel.transformations.Scaling] from
        the voxels to `"physical"`, or nothing.
    """
    if not record.members:
        return []
    rows, columns = PLANES[record.axis].lower()
    scales: tx.Dict[str, raster.AxisScale] = {}
    if record.pixel_size is not None:
        scales[columns] = (record.pixel_size[0], record.unit)
        scales[rows] = (record.pixel_size[1], record.unit)
    if record.spacing is not None:
        scales[record.axis] = (record.spacing, record.unit)
    return raster.raster_transformations(model_axes(record.axes), scales)


def encoded_geometry(
    xform: tx.Optional[Transformation], axis: str, ndim: int
) -> _Fields:
    """Return the geometry fields of a record from a transformation.

    A record holds the size of the pixels, the spacing of the slices and
    one unit of length, so only a transformation that scales each axis
    onto an axis with a unit of length is recorded, as the raster writers
    record it. A translation is not recorded, and a rotation or a shear
    gives no size. The pixel size is the magnitude of the scale along the
    columns and the rows of the files, and the spacing keeps the sign of
    the scale along the stacking axis. Every size is converted to the unit
    along the columns, or along the stacking axis when the pixel size is
    unknown.

    Parameters
    ----------
    xform : Transformation, optional
        The preferred transformation of the image.
    axis : str
        The stacking axis.
    ndim : int
        The number of axes of the data, 3 or 4 with channels.

    Returns
    -------
    pixel_size : tuple[float, float] or None
        The size of the pixels along the columns and the rows of the files.
    spacing : float or None
        The distance from one slice to the next.
    unit : str or None
        The unit of length of the sizes.
    """
    if xform is None:
        return None, None, None
    try:
        matrix = np.asarray(xform.to(Affine).matrix, dtype=float)
        output = list(get_axes(xform.output))
    except Exception:  # noqa: BLE001
        return None, None, None
    if matrix.shape != (ndim, ndim + 1) or len(output) != ndim:
        return None, None, None
    linear = matrix[:, :-1]
    if np.any(linear - np.diag(np.diag(linear))):
        return None, None, None
    names = [axis.name for axis in model_axes(_codes(axis, ndim))]
    sizes: tx.Dict[str, tx.Tuple[float, str]] = {}
    for name in ("x", "y", "z"):
        i = names.index(name)
        scale = float(linear[i, i])
        unit = getattr(output[i], "unit", None)
        if not math.isfinite(scale) or scale == 0 or not _is_length(unit):
            continue
        sizes[name] = (scale, str(unit))
    rows, columns = PLANES[axis].lower()
    if columns in sizes and rows in sizes:
        unit = sizes[columns][1]
        pixel_size = (
            abs(sizes[columns][0]),
            abs(raster.convert_length(sizes[rows][0], sizes[rows][1], unit)),
        )
    elif axis in sizes:
        unit, pixel_size = sizes[axis][1], None
    else:
        return None, None, None
    spacing = None
    if axis in sizes:
        spacing = raster.convert_length(sizes[axis][0], sizes[axis][1], unit)
    return pixel_size, spacing, unit


def stored_geometry(
    fmt: tx.Optional[str], channels: bool, fields: _Fields
) -> _Fields:
    """Return the part of the geometry fields that the files can record.

    A Pillow file records no size that the reader of a stack uses, so it
    records nothing. A TIFF file records the size of its pixels when they
    have a unit of length, through the TIFF writer, which then writes the
    sizes in OME-XML. It records the spacing of the slices too when it
    also records the size of its pixels and has no channels, because it is
    then written as a volume of one slice, whose `PhysicalSizeZ` holds the
    magnitude of the spacing. A file with channels is written as a plane,
    so that an RGB image stays one, and it records no spacing.

    Parameters
    ----------
    fmt : str, optional
        The reader of the files, `"tiff"` or `"pillow"`.
    channels : bool
        Whether the files have channels.
    fields : tuple
        The fields `pixel_size`, `spacing` and `unit`.

    Returns
    -------
    tuple
        The fields that the files record.
    """
    pixel_size, spacing, unit = fields
    if fmt != "tiff" or pixel_size is None or not _is_length(unit):
        return None, None, None
    if channels or spacing is None:
        return pixel_size, None, unit
    return pixel_size, abs(spacing), unit


def model_axes(codes: str) -> tx.List[Axis]:
    """Return the axes of the data of a stack from the storage codes of
    its volume, in the brainhops order `x, y, z[, c]`."""
    axes = raster.storage_axes(codes)
    return [axes[p] for p in raster.canonical_permutation(axes)]


def _codes(axis: str, ndim: int) -> str:
    """Return the storage codes of a stacked volume of `ndim` axes."""
    return axis.upper() + PLANES[axis] + "S" * (ndim - 3)


def _is_length(unit: tx.Any) -> bool:
    """Tell whether a unit is a unit of length."""
    if unit is None:
        return False
    try:
        raster.convert_length(1.0, unit, unit)
    except ValueError:
        return False
    return True
