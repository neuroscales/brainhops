"""The geometry of TIFF images: decoded from a record, and encoded for the
writer.

A TIFF file records the size of a pixel of the finest level of a series,
and sometimes the position of its first pixel. The transformation of a
level is decoded from these and from the shapes of the levels, as the
documentation of [`brainhops.io.images.tiff`][] describes. The writer
stores the part of a transformation that TIFF can hold: a size and an
origin along each axis.
"""

# stdlib
import math
from numbers import Number

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic

# internals
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.transformations import (
    Affine,
    Identity,
    Scaling,
    Sequence,
    Transformation,
)
from brainhops.io.images.base import _utils_raster as raster

# this format
from . import _utils as backend
from ._raw import TiffRaw

# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def record_transformations(
    metadata: tx.Any, level: int
) -> tx.List[Transformation]:
    """Decode the transformations of a level from the record of metadata.

    The single transformation scales the pixels of the level onto the
    `"physical"` system, with a translation when the file records an
    origin or when the level is coarser than the finest one. Metadata
    without a record, or a level that the record does not describe,
    gives no transformation.

    Parameters
    ----------
    metadata : TiffMetadata or None
        The metadata, which holds the record of a series.
    level : int
        The index of the level in the record.

    Returns
    -------
    list of Transformation
        The transformations of the level.
    """
    record = getattr(metadata, "raw", None)
    if not isinstance(record, TiffRaw) or not record.levels:
        return []
    if not 0 <= level < len(record.levels):
        return []
    return [
        level_geometry(
            record.storage_axes,
            record.levels[0].shape,
            record.levels[level].shape,
            record,
        )
    ]


def level_geometry(
    codes: str,
    base_shape: tx.Sequence[int],
    level_shape: tx.Sequence[int],
    record: TiffRaw,
    pixel_size: tx.Any = None,
    unit: tx.Any = None,
    origin: tx.Any = None,
) -> Transformation:
    """Return the transformation of one level of a series.

    The transformation is computed from the record of the series, unless
    the reader options `pixel_size`, `unit` and `origin` override it.

    Parameters
    ----------
    codes : str
        The axes of the series as tifffile stores them.
    base_shape : sequence of int
        The stored shape of the finest level.
    level_shape : sequence of int
        The stored shape of the level.
    record : TiffRaw
        The record of the series.
    pixel_size : float or sequence of float or mapping, optional
        The pixel size, which overrides the pixel size of the record.
    unit : str or Unit, optional
        The unit of `pixel_size`, or the unit to which the sizes of the
        record are converted.
    origin : bool or float or sequence of float or mapping, optional
        The position of the first pixel, which overrides the positions of
        the record. `False` ignores them.

    Returns
    -------
    Transformation
        A [`Scaling`][], or an [`Affine`][] when there is a translation.
    """
    axes = raster.storage_axes(codes)
    perm = raster.canonical_permutation(axes)
    axes = [axes[p] for p in perm]
    scales = raster.resolve_pixel_size(
        axes, record.scales or {}, pixel_size=pixel_size, unit=unit
    )
    mine = _overridden(pixel_size, axes)
    for name, sign in (record.signs or {}).items():
        if name in scales and name not in mine:
            size, u = scales[name]
            scales[name] = (sign * size, u)
    factors = backend.level_factors(
        [base_shape[p] for p in perm], [level_shape[p] for p in perm]
    )
    return raster.level_transformation(
        axes, scales, factors, _origin(axes, scales, record, origin)
    )


def _overridden(pixel_size: tx.Any, axes: tx.Sequence[Axis]) -> tx.Set[str]:
    """Return the names of the spatial axes whose size `pixel_size` sets."""
    if pixel_size is None:
        return set()
    if isinstance(pixel_size, tx.Mapping):
        return set(pixel_size)
    return {axis.name for axis in raster.spatial_axes(axes)}


def _origin(
    axes: tx.Sequence[Axis],
    scales: tx.Mapping[str, tx.Tuple[float, tx.Any]],
    record: TiffRaw,
    origin: tx.Any,
) -> tx.Dict[str, float]:
    """Return the position of the first sample along each spatial axis, by
    name, in the unit of the axis.

    When `origin` is given, it sets the positions in the unit of the pixel
    size, and `False` ignores the positions recorded by the file. Otherwise,
    the OME plane positions of the file are used for the axes whose size and
    unit are known. A position without a
    unit, or in the OME "reference frame", is taken in the unit of the pixel
    size of the file.
    """
    names = [axis.name for axis in raster.spatial_axes(axes)]
    if origin is False:
        return {}
    if origin is not None:
        if isinstance(origin, tx.Mapping):
            out = {name: float(value) for name, value in origin.items()}
        elif isinstance(origin, (Number, np.number)):
            out = {name: float(origin) for name in names}
        else:
            values = [float(v) for v in np.ravel(origin)]
            if len(values) != len(names):
                raise ValueError(
                    f"origin gives {len(values)} positions, but the image "
                    f"has {len(names)} spatial axes {names}."
                )
            out = dict(zip(names, values))
        unknown = set(out) - set(names)
        if unknown:
            raise ValueError(
                f"origin names axes that are not spatial axes of this "
                f"image: {sorted(unknown)}."
            )
        return out
    out = {}
    recorded = record.scales or {}
    for name, (value, unit) in (record.origin or {}).items():
        if name not in names or name not in scales:
            continue
        target = scales[name][1]
        if target is None:
            continue
        source = backend.length_unit(unit)
        if source is None:
            if name not in recorded:
                continue
            source = recorded[name][1]
        try:
            out[name] = raster.convert_length(value, source, target)
        except ValueError:
            continue
    return out


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


class Geometry(Magic, frozen=True):
    """The part of the geometry of an image that a TIFF writer can store."""

    sizes: tx.Dict[str, float]  # positive, in `units`
    units: tx.Dict[str, str]  # normalised by `length_unit`
    signs: tx.Dict[str, float]  # -1 for a flipped axis
    origin: tx.Dict[str, float]  # in `units`
    time: tx.Optional[tx.Tuple[float, str]]  # interval and unit

    def size(self, name: str, unit: str) -> float:
        """Return the size along an axis, converted to another unit."""
        return raster.convert_length(self.sizes[name], self.units[name], unit)


def write_geometry(
    xform: tx.Optional[Transformation], axes: tx.Sequence[Axis]
) -> Geometry:
    """Return the pixel size, origin and time interval of the preferred
    transformation, as far as TIFF can store them.

    A spatial axis has a size when the transformation scales it onto an axis
    with a unit of length, and sizes are given only if every spatial axis has
    one. A rotation or a shear gives no size.
    """
    empty = Geometry(sizes={}, units={}, signs={}, origin={}, time=None)
    parts = _affine_parts(xform, len(axes))
    if parts is None:
        return empty
    diag, shift = parts
    output = getattr(getattr(xform, "output", None), "axes", None)
    try:
        output = list(output) if output is not None else None
    except TypeError:
        output = None
    if output is None or len(output) != len(axes):
        return empty
    sizes, units, signs, origin = {}, {}, {}, {}
    time = None
    for i, axis in enumerate(axes):
        unit = getattr(output[i], "unit", None)
        group = raster.axis_group(axis)
        scale = float(diag[i])
        if not math.isfinite(scale) or scale == 0:
            continue
        if group == "space":
            name = backend.length_unit(None if unit is None else str(unit))
            if name is None:
                # A spatial unit that is not a length gives no axis a size.
                return Geometry(
                    sizes={}, units={}, signs={}, origin={}, time=time
                )
            sizes[axis.name] = abs(scale)
            units[axis.name] = name
            if scale < 0:
                signs[axis.name] = -1.0
            if float(shift[i]):
                origin[axis.name] = float(shift[i])
        elif group == "time" and time is None:
            name = backend.time_unit(None if unit is None else str(unit))
            if name is not None and scale > 0:
                time = (scale, name)
    return Geometry(
        sizes=sizes, units=units, signs=signs, origin=origin, time=time
    )


def _affine_parts(
    xform: tx.Optional[Transformation], ndim: int
) -> tx.Optional[tx.Tuple[np.ndarray, np.ndarray]]:
    """Return the diagonal and translation of a transformation that scales and
    translates each axis separately, or `None` for any other transformation.
    """
    if xform is None or isinstance(xform, Identity):
        return np.ones(ndim), np.zeros(ndim)
    if isinstance(xform, Sequence):
        try:
            xform = xform.compute()
        except Exception:
            return None
    if isinstance(xform, Scaling):
        scale = getattr(xform, "scale", None)
        if scale is None:
            return np.ones(ndim), np.zeros(ndim)
        scale = np.atleast_1d(np.asarray(scale, dtype=float))
        if scale.size != ndim:
            return None
        return scale, np.zeros(ndim)
    affine = xform
    if not isinstance(affine, Affine):
        try:
            affine = xform.to(Affine)
        except Exception:
            return None
    matrix = getattr(affine, "matrix", None)
    if matrix is None:
        return np.ones(ndim), np.zeros(ndim)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (ndim, ndim + 1):
        return None
    linear = matrix[:, :-1]
    if np.any(linear - np.diag(np.diag(linear))):
        return None
    return np.diag(linear).copy(), matrix[:, -1].copy()
