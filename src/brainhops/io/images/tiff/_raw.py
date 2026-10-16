"""The record of a series of a TIFF file."""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import ParserContentError
from brainhops.io.images.base import _utils_raster as raster

# this format
from . import _utils as backend

_Scales = tx.Dict[str, tx.Tuple[float, tx.Any]]


class TiffLevelRaw(Raw):
    """Storage description of one level of a series of a TIFF file.

    A series of a TIFF file, such as an image, a stack or a hyperstack,
    may hold a pyramid of levels, finest first. Each level is stored in
    its own pages, or in the SubIFDs of the pages of the finest level, and
    this record describes how one level is stored, without its pixels.
    """

    shape: tx.Tuple[int, ...] = ()
    """The shape of the level in the order of `TiffRaw.storage_axes`,
    slowest first, as tifffile reads it."""

    dtype: str = "uint8"
    """The name of the NumPy data type of the pixels, such as `"uint16"`."""


class TiffRaw(Raw):
    """Record of one series of a TIFF file, without its pixels.

    A TIFF file holds one or more series, which tifffile assembles from
    its pages, and a series may hold a pyramid of levels. The record
    describes the series that was read: its levels, its axes, and what
    the file records about it, which is its metadata dialect, its
    OME-XML or ImageJ metadata, its tags of interest and the geometry
    that these give. [`TiffImage`][brainhops.io.images.tiff.TiffImage]
    and [`TiffMultiScaleImage`][brainhops.io.images.tiff.TiffMultiScaleImage]
    hold it through [`TiffMetadata`][brainhops.io.images.tiff.TiffMetadata],
    and every level of a pyramid holds the same record, in which its
    index `level` finds it.

    The record is read from the IFDs of the file and never from its
    pixels. It is never changed in place. A TIFF file has no header that
    can be written apart from its pixels, and tifffile encodes again
    everything that it writes, so an image that was read and not changed
    is written by copying the bytes of its file, and any other image is
    encoded again from its data, its transformations and this record.
    """

    dialect: tx.Optional[str] = None
    """The metadata dialect: `"ome"`, `"imagej"`, or `None` for a plain
    TIFF file."""

    storage_axes: str = ""
    """The axes of the series as tifffile stores them, slowest first, such
    as `"TZCYX"` or `"YXS"`."""

    levels: tx.Tuple[TiffLevelRaw, ...] = ()
    """The storage description of each level of the series, finest first."""

    series: int = 0
    """The index of the series in the file."""

    n_series: int = 1
    """The number of series in the file."""

    bigtiff: bool = False
    """Whether the file is a BigTIFF file."""

    tags: tx.Optional[tx.Dict[str, tx.Any]] = None
    """The tags of interest of the first page, by name, such as the
    resolution, the description, the software, the date, the artist and
    the copyright."""

    ome_xml: tx.Optional[str] = None
    """The OME-XML of an OME-TIFF file."""

    ome_index: tx.Optional[int] = None
    """The position of the OME Image element that describes the series,
    among all the Image elements of the OME-XML."""

    imagej: tx.Optional[tx.Dict[str, tx.Any]] = None
    """The ImageJ metadata of an ImageJ hyperstack."""

    scales: tx.Optional[_Scales] = None
    """The positive size and unit of a pixel of the finest level along each
    axis (x, y, z and t) whose size the file records."""

    signs: tx.Optional[tx.Dict[str, float]] = None
    """The sign -1.0 of each axis whose step is negative, by name, as an
    ImageJ `spacing` below 0 gives along z."""

    origin: tx.Optional[_Scales] = None
    """The position and unit of the first pixel along each axis whose
    position the OME planes record, by name. A unit of `None` means the
    unit of the pixel size."""

    sources: tx.Optional[tx.Dict[str, str]] = None
    """The dialect that each size of `scales` comes from: `"ome"`,
    `"imagej"` or `"resolution"`."""


def read_record(tif: tx.Any, series: int = 0) -> TiffRaw:
    """Read the record of a series of an open TIFF file.

    The record is read from the IFDs, the OME-XML and the ImageJ metadata
    of the file, as tifffile parses them, and no pixel is read.

    Parameters
    ----------
    tif : tifffile.TiffFile
        The open file.
    series : int, default=0
        The index of the series. A negative index counts from the end.

    Returns
    -------
    TiffRaw
        The record of the series.

    Raises
    ------
    IndexError
        If the file has no such series.
    ParserContentError
        If a level of the series has another number of axes than the
        finest level, or is downsampled along an axis that is not
        spatial, since such a level cannot be placed in space.
    """
    backend._select(tif, int(series), 0)
    index = int(series) % len(tif.series)
    one = tif.series[index]
    found = backend.series_metadata(tif, index)
    codes = str(one.axes)
    levels = tuple(
        TiffLevelRaw(
            shape=tuple(int(n) for n in level.shape),
            dtype=np.dtype(level.dtype).name,
        )
        for level in one.levels
    )
    _check_levels(codes, levels)
    return TiffRaw(
        dialect=found.dialect,
        storage_axes=codes,
        levels=levels,
        series=index,
        n_series=len(tif.series),
        bigtiff=bool(tif.is_bigtiff),
        tags=found.tags,
        ome_xml=found.ome_xml,
        ome_index=found.ome_index,
        imagej=found.imagej,
        scales=found.scales,
        signs=found.signs,
        origin=found.origin,
        sources=found.sources,
    )


def _check_levels(codes: str, levels: tx.Sequence[TiffLevelRaw]) -> None:
    """Check that the coarser levels of a series can be placed in space.

    Raises
    ------
    ParserContentError
        If a level is downsampled along an axis that is not spatial, or
        has another number of axes than the finest level.
    """
    axes = raster.storage_axes(codes)
    base = levels[0].shape
    for level in levels[1:]:
        factors = backend.level_factors(base, level.shape)
        for axis, factor in zip(axes, factors):
            if factor != 1 and raster.axis_group(axis) != "space":
                raise ParserContentError(
                    f"A pyramid level is downsampled along the non-spatial "
                    f"axis {axis.name!r}, which cannot be placed in space."
                )
