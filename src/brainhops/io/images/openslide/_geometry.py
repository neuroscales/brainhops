"""The geometry of the levels of a slide, decoded from its record."""

# stdlib
import math

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel.transformations import Transformation
from brainhops.io.images.base import _utils_raster as raster

# this format
from ._raw import OpenSlideRaw

# The data of a level is an F-ordered (x, y, c) array: OpenSlide returns
# C-ordered (y, x, samples) pixels, which the lazy array transposes.
_STORAGE = "YXS"


def record_transformations(
    metadata: tx.Any, level: int
) -> tx.List[Transformation]:
    """Return the transformations of a level that a record decodes to.

    The list holds the transformation of [`level_geometry`][] for the
    record of the metadata, and it is empty without metadata, or for a
    level that the record does not describe.

    Parameters
    ----------
    metadata : OpenSlideMetadata or None
        The metadata of the slide.
    level : int
        The index of the level in the record.

    Returns
    -------
    list of Transformation
        The transformations of the level.
    """
    record = None if metadata is None else metadata.raw
    if record is None or not 0 <= level < len(record.levels):
        return []
    return [level_geometry(record, level)]


def level_geometry(
    record: OpenSlideRaw,
    level: int,
    pixel_size: tx.Any = None,
    unit: tx.Any = None,
) -> Transformation:
    """Return the pixel-to-physical transformation of a level.

    The full-resolution pixel size, which the slide records in micrometres
    as `openslide.mpp-x` and `openslide.mpp-y`, is multiplied by the
    downsampling factor `f` of the level, and pixel `i` is centred on the
    full-resolution coordinate `f * i + (f - 1) / 2`, so that every level
    covers the whole slide. Without a recorded pixel size, the axes have
    no unit.

    Parameters
    ----------
    record : OpenSlideRaw
        The record of the slide.
    level : int
        The index of the level in the record.
    pixel_size : float or Sequence[float] or Mapping[str, float], optional
        The full-resolution pixel size, which overrides the one of the
        slide.
    unit : str or Unit, optional
        The unit of `pixel_size`, or the unit to which micrometres are
        converted.

    Returns
    -------
    Transformation
        A [`Scaling`][brainhops.datamodel.transformations.Scaling], or an
        [`Affine`][brainhops.datamodel.transformations.Affine] with a
        translation for a coarser level.
    """
    axes = raster.storage_axes(_STORAGE)
    axes = [axes[p] for p in raster.canonical_permutation(axes)]
    scales = raster.resolve_pixel_size(
        axes, _mpp(record.properties), pixel_size=pixel_size, unit=unit
    )
    factor = float(record.levels[level].downsample)
    return raster.level_transformation(axes, scales, [factor, factor, 1], {})


def _mpp(properties: tx.Mapping[str, str]) -> tx.Dict[str, tx.Any]:
    """Return the full-resolution pixel size in micrometres, for the axes whose
    size the slide records.
    """
    out = {}
    for name in ("x", "y"):
        try:
            size = float(properties.get(f"openslide.mpp-{name}", ""))
        except ValueError:
            continue
        if math.isfinite(size) and size > 0:
            out[name] = (size, "um")
    return out
