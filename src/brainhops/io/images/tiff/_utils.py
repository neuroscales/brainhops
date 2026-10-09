"""The tifffile backend of the TIFF reader.

This module opens a file, selects a series and a pyramid level, reads the
pixels and gathers what the file records about the pixel size. Coordinate
systems and transformations belong to the shared raster conventions of
`brainhops.io.images.base._utils_raster`, as for the Pillow backend.

## Dialects

The pixel size can be stored in three ways. Along each axis, the first of them
that gives a size wins:

1. OME-TIFF: the OME-XML of the first `ImageDescription` records
   `PhysicalSizeX`, `Y` and `Z` with their units (micrometres if absent) and
   `TimeIncrement` (seconds if no unit is given). The origin is the position of
   the plane whose Z, C and T indices are 0.
2. ImageJ: the hyperstack records a length `unit` (with `yunit` and `zunit`
   when they differ), in which pixels measure `1 / XResolution` by
   `1 / YResolution`. The slice step is `spacing` (negative if the slices run
   backwards), and the frame interval is `finterval` in `tunit` (seconds by
   default).
3. Resolution tags count pixels per `ResolutionUnit`: 1 for none (an aspect
   ratio only), 2 for the inch (the default), 3 for the centimetre, and the
   tifffile extensions 4 and 5 for the millimetre and the micrometre.

Absent and placeholder sizes are left out. tifffile reports missing resolution
tags as 1 pixel per inch, so their absence is detected from the tags
themselves, and a resolution of 72 or 96 dpi, or of one pixel per inch or
centimetre, is a placeholder.
"""

__all__ = [
    "EXTENSIONS",
    "MAGICS",
    "RESOLUTION_UNITS",
    "TAGS_OF_INTEREST",
    "TiffMetadata",
    "TiffSource",
    "is_tiff",
    "is_whole_slide",
    "length_unit",
    "time_unit",
    "resolution_scales",
    "imagej_scales",
    "ome_image_index",
    "ome_image",
    "ome_channel_names",
    "ome_scales",
    "series_metadata",
    "tags_of_interest",
    "level_factors",
    "read_series",
]

import math
import weakref
import xml.etree.ElementTree as ElementTree
from io import BytesIO

import numpy as np
import typing_extensions as tx
from bagof.magic import Magic

from brainhops._core import path
from brainhops._core.dependencies import tifffile
from brainhops.io.base.parsers import ParserContentError
from brainhops.io.images.base import _utils_raster as raster

EXTENSIONS: tx.Tuple[str, ...] = (
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
    ".btf",
    ".tf8",
    ".tf2",
)
"""The extensions of TIFF files, including BigTIFF and OME-TIFF."""

MAGICS: tx.Tuple[bytes, ...] = (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+")
"""The first four bytes of TIFF files (both byte orders) and BigTIFF."""

RESOLUTION_UNITS: tx.Dict[int, tx.Optional[str]] = {
    1: None,
    2: "inch",
    3: "cm",
    4: "mm",
    5: "um",
}
"""The units of the `ResolutionUnit` tag: TIFF codes 1 to 3, plus the tifffile
extensions 4 and 5. `None` means no unit (an aspect ratio).
"""

TAGS_OF_INTEREST: tx.Dict[int, str] = {
    269: "DocumentName",
    270: "ImageDescription",
    271: "Make",
    272: "Model",
    282: "XResolution",
    283: "YResolution",
    296: "ResolutionUnit",
    305: "Software",
    306: "DateTime",
    315: "Artist",
    316: "HostComputer",
    33432: "Copyright",
}
"""The tags of the first page that the reader keeps, by code."""

_TIFF_INCH = 2  # default of ResolutionUnit

# Normalised spellings of length units: OME symbols, ImageJ words (including
# the escaped micro sign) and the usual abbreviations.
_LENGTH_UNITS = {
    "m": "m",
    "meter": "m",
    "metre": "m",
    "dm": "dm",
    "cm": "cm",
    "centimeter": "cm",
    "centimetre": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "millimetre": "mm",
    "um": "um",
    "µm": "um",
    "μm": "um",
    "\\u00b5m": "um",
    "\\u03bcm": "um",
    "micron": "um",
    "microns": "um",
    "micrometer": "um",
    "micrometre": "um",
    "nm": "nm",
    "nanometer": "nm",
    "nanometre": "nm",
    "pm": "pm",
    "picometer": "pm",
    "picometre": "pm",
    "å": "angstrom",
    "Å": "angstrom",
    "angstrom": "angstrom",
    "in": "inch",
    "inch": "inch",
    "inches": "inch",
    "km": "km",
}

_TIME_UNITS = {
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "ms": "ms",
    "msec": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "us": "us",
    "µs": "us",
    "μs": "us",
    "\\u00b5s": "us",
    "microsecond": "us",
    "microseconds": "us",
    "ns": "ns",
    "nanosecond": "ns",
    "nanoseconds": "ns",
    "min": "min",
    "minute": "min",
    "minutes": "min",
    "h": "h",
    "hr": "h",
    "hour": "h",
    "hours": "h",
}


def _require_tifffile() -> tx.Any:
    if tifffile is None:
        raise ImportError(
            "Reading this TIFF file needs tifffile: pip install "
            "brainhops[tiff]"
        )
    return tifffile


# ----------------------------------------------------------------------
#   SNIFFING
# ----------------------------------------------------------------------


def is_tiff(head: bytes) -> bool:
    """Return whether the first bytes of a file are a TIFF or BigTIFF magic."""
    return bytes(head[:4]) in MAGICS


# ----------------------------------------------------------------------
#   UNITS
# ----------------------------------------------------------------------


def _unit(value: tx.Any, table: tx.Mapping[str, str]) -> tx.Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text in table:
        return table[text]
    return table.get(text.lower())


def length_unit(value: tx.Any) -> tx.Optional[str]:
    r"""Normalise the spelling of a length unit.

    `"micron"`, `"µm"` and the escaped `"µm"` of ImageJ all become `"um"`.
    `None` is returned for anything that is not a physical length, such as
    `"pixel"`, the OME `"reference frame"` or an unknown unit.
    """
    return _unit(value, _LENGTH_UNITS)


def time_unit(value: tx.Any) -> tx.Optional[str]:
    """Normalise the spelling of a time unit, such as `"sec"` to `"s"`, or
    return `None` if the unit is unknown.
    """
    return _unit(value, _TIME_UNITS)


def _positive(value: tx.Any) -> tx.Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _rational(value: tx.Any) -> tx.Optional[float]:
    """Return the value of a TIFF rational, a `(num, den)` pair or a number."""
    if value is None:
        return None
    try:
        if isinstance(value, (tuple, list)):
            if len(value) != 2 or not value[1]:
                return None
            return float(value[0]) / float(value[1])
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# ----------------------------------------------------------------------
#   METADATA
# ----------------------------------------------------------------------

_Scales = tx.Dict[str, raster.AxisScale]


class TiffMetadata(Magic, frozen=True):
    """What a TIFF file says about the geometry of one series."""

    dialect: tx.Optional[str]
    """`"ome"`, `"imagej"`, or `None` for a plain TIFF file."""

    scales: _Scales
    """The positive size and unit of a sample along each axis (x, y, z, t)
    whose size is known.
    """

    signs: tx.Dict[str, float]
    """The sign, -1.0, of the axes with a negative step (an ImageJ `spacing`
    below 0), by name.
    """

    origin: _Scales
    """The position and unit of the first sample, by axis name, where the file
    records it (OME plane positions).
    """

    sources: tx.Dict[str, str]
    """The dialect that each size of `scales` comes from: `"ome"`, `"imagej"`
    or `"resolution"`.
    """

    ome_xml: tx.Optional[str]
    """The OME-XML of an OME-TIFF file."""

    ome_index: tx.Optional[int]
    """The position, among the OME Image elements, of the one of the series."""

    imagej: tx.Optional[tx.Dict[str, tx.Any]]
    """The ImageJ metadata of an ImageJ hyperstack."""

    tags: tx.Dict[str, tx.Any]
    """The tags of interest of the first page, by name."""


def tags_of_interest(page: tx.Any) -> tx.Dict[str, tx.Any]:
    """Return the tags of interest of a page, by name.

    When a tag is repeated, such as several `ImageDescription` tags, the first
    one is kept.
    """
    out: tx.Dict[str, tx.Any] = {}
    for tag in page.tags.values():
        name = TAGS_OF_INTEREST.get(tag.code)
        if name is None or name in out:
            continue
        value = tag.value
        if hasattr(value, "value"):  # unwrap enum values such as RESUNIT
            value = value.value
        out[name] = value
    return out


def resolution_scales(page: tx.Any) -> _Scales:
    """Return the pixel width (x) and height (y) given by the resolution tags
    of a page.

    Nothing is returned for absent tags, an aspect ratio only, or placeholders.
    The tags are read directly, because tifffile reports their absence as 1
    pixel per inch.
    """
    tags = page.tags
    xres = _positive(_rational(tags.valueof(282)))
    yres = _positive(_rational(tags.valueof(283)))
    if xres is None or yres is None:
        return {}
    code = tags.valueof(296)
    code = _TIFF_INCH if code is None else int(code)
    unit = RESOLUTION_UNITS.get(code)
    if unit is None:
        return {}
    if unit in ("inch", "cm"):
        per_inch = 1.0 if unit == "inch" else 2.54
        if (xres == 1 and yres == 1) or raster.is_default_dpi(
            (xres * per_inch, yres * per_inch)
        ):
            return {}
    if unit == "inch":
        # Unit('inch') does not convert reliably
        return {
            "x": (raster.dpi_to_size(xres, "mm"), "mm"),
            "y": (raster.dpi_to_size(yres, "mm"), "mm"),
        }
    return {"x": (1.0 / xres, unit), "y": (1.0 / yres, unit)}


def imagej_scales(
    metadata: tx.Optional[tx.Mapping[str, tx.Any]],
    page: tx.Any,
    names: tx.Collection[str],
) -> tx.Tuple[_Scales, tx.Dict[str, float]]:
    """Return the sizes recorded by an ImageJ hyperstack and the signs of its
    steps.

    Parameters
    ----------
    metadata : Mapping[str, Any]
        The `imagej_metadata` of tifffile.
    page : TiffPage
        The first page, whose resolution tags hold the pixel size.
    names : Collection[str]
        The axes of the series, the only ones given a size.

    Returns
    -------
    scales : dict[str, tuple[float, str]]
        The size and unit of each axis whose size is known.
    signs : dict[str, float]
        The signs of the steps, such as `{"z": -1.0}`.
    """
    scales: _Scales = {}
    signs: tx.Dict[str, float] = {}
    if not metadata:
        return scales, signs
    unit = length_unit(metadata.get("unit"))
    units = {
        "x": unit,
        "y": length_unit(metadata.get("yunit")) or unit,
        "z": length_unit(metadata.get("zunit")) or unit,
    }
    for name, code in (("x", 282), ("y", 283)):
        resolution = _positive(_rational(page.tags.valueof(code)))
        if name in names and units[name] and resolution is not None:
            scales[name] = (1.0 / resolution, units[name])
    if "z" in names and units["z"]:
        try:
            spacing = float(metadata.get("spacing"))
        except (TypeError, ValueError):
            spacing = float("nan")
        if math.isfinite(spacing) and spacing != 0:
            scales["z"] = (abs(spacing), units["z"])
            if spacing < 0:
                signs["z"] = -1.0
    if "t" in names:
        interval = _positive(metadata.get("finterval"))
        tunit = time_unit(metadata.get("tunit", "s"))
        if interval is not None and tunit is not None:
            scales["t"] = (interval, tunit)
    return scales, signs


def _local(tag: str) -> str:
    """Return an XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def _children(element: tx.Any, name: str) -> tx.List[tx.Any]:
    return [child for child in element if _local(child.tag) == name]


def _ome_images(ome_xml: str) -> tx.List[tx.Any]:
    try:
        root = ElementTree.fromstring(ome_xml)
    except ElementTree.ParseError:
        return []
    return _children(root, "Image")


def ome_image_index(
    ome_xml: str, series: tx.Sequence[tx.Any], index: int
) -> tx.Optional[int]:
    """Return the position, among the OME Image elements, of the image that
    describes series `index`, or `None`.

    tifffile makes one series per OME Image, in order, but skips the images
    whose planes it cannot find. Each series is therefore matched to the next
    image of the same name and size. `series` holds all series, or anything
    with a `name`, `axes` and `shape`.
    """
    images = _ome_images(ome_xml)
    position = 0
    for i, one in enumerate(series[: index + 1]):
        sizes = dict(zip(one.axes, one.shape))
        found = None
        for j in range(position, len(images)):
            pixels = _children(images[j], "Pixels")
            if not pixels:
                continue
            name = images[j].get("Name")
            if getattr(one, "name", None) and name and one.name != name:
                continue
            if any(
                key in sizes
                and pixels[0].get(f"Size{key}") is not None
                and int(pixels[0].get(f"Size{key}")) != int(sizes[key])
                for key in ("X", "Y")
            ):
                continue
            found = j
            break
        if found is None:
            return None
        position = found + 1
        if i == index:
            return found
    return None


def ome_image(
    ome_xml: str, index: int
) -> tx.Tuple[tx.Optional[tx.Any], tx.Optional[tx.Any]]:
    """Return the OME Image element at a position and its Pixels element, or
    `(None, None)`.
    """
    images = _ome_images(ome_xml)
    if index is None or not 0 <= index < len(images):
        return None, None
    pixels = _children(images[index], "Pixels")
    return images[index], (pixels[0] if pixels else None)


def ome_channel_names(pixels: tx.Any) -> tx.List[tx.Optional[str]]:
    """Return the channel names of an OME Pixels element."""
    return [channel.get("Name") for channel in _children(pixels, "Channel")]


def ome_scales(
    pixels: tx.Any, names: tx.Collection[str]
) -> tx.Tuple[_Scales, _Scales]:
    """Return the sizes and origin recorded by an OME Pixels element.

    Parameters
    ----------
    pixels : Element
        The Pixels element (see [`ome_image`][]).
    names : Collection[str]
        The axes of the series, the only ones given a size.

    Returns
    -------
    scales : dict[str, tuple[float, str]]
        The size and unit of each axis whose unit is a length.
    origin : dict[str, tuple[float, str or None]]
        The position of the first plane along x, y and z, where recorded, with
        a unit of `None` when the file names none.
    """
    scales: _Scales = {}
    for name in ("x", "y", "z"):
        key = f"PhysicalSize{name.upper()}"
        size = _positive(pixels.get(key))
        unit = length_unit(pixels.get(f"{key}Unit", "µm"))
        if name in names and size is not None and unit is not None:
            scales[name] = (size, unit)
    if "t" in names:
        size = _positive(pixels.get("TimeIncrement"))
        unit = time_unit(pixels.get("TimeIncrementUnit", "s"))
        if size is not None and unit is not None:
            scales["t"] = (size, unit)

    origin: _Scales = {}
    planes = _children(pixels, "Plane")
    first = None
    for plane in planes:
        if all(int(plane.get(f"The{key}", "0") or 0) == 0 for key in "ZCT"):
            first = plane
            break
    if first is None and planes:
        first = planes[0]
    if first is not None:
        for name in ("x", "y", "z"):
            key = f"Position{name.upper()}"
            if name not in names or first.get(key) is None:
                continue
            try:
                value = float(first.get(key))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            unit = first.get(f"{key}Unit")
            origin[name] = (value, None if unit is None else str(unit))
    return scales, origin


def series_metadata(tif: tx.Any, index: int = 0) -> TiffMetadata:
    """Return what a TIFF file says about the geometry of series `index`, with
    OME taking precedence over ImageJ and the resolution tags along each axis.
    """
    series = tif.series[index]
    page = series.keyframe if series.keyframe is not None else tif.pages[0]
    names = {axis.name for axis in raster.storage_axes(series.axes)}
    scales: _Scales = {}
    sources: tx.Dict[str, str] = {}
    signs: tx.Dict[str, float] = {}
    origin: _Scales = {}

    for name, scale in resolution_scales(page).items():
        if name in names:
            scales[name] = scale
            sources[name] = "resolution"

    dialect = None
    imagej = None
    if tif.is_imagej:
        dialect = "imagej"
        imagej = dict(tif.imagej_metadata or {})
        found, signs = imagej_scales(imagej, page, names)
        for name, scale in found.items():
            scales[name] = scale
            sources[name] = "imagej"

    ome_xml = None
    ome_index = None
    if tif.is_ome:
        dialect = "ome"
        ome_xml = tif.ome_metadata
        if ome_xml:
            ome_index = ome_image_index(ome_xml, tif.series, index)
        _, pixels = ome_image(ome_xml, ome_index) if ome_xml else (None, None)
        if pixels is not None:
            found, origin = ome_scales(pixels, names)
            for name, scale in found.items():
                scales[name] = scale
                sources[name] = "ome"
                signs.pop(name, None)

    return TiffMetadata(
        dialect=dialect,
        scales=scales,
        signs=signs,
        origin=origin,
        sources=sources,
        ome_xml=ome_xml,
        ome_index=ome_index,
        imagej=imagej,
        tags=tags_of_interest(tif.pages[0]),
    )


def level_factors(
    base: tx.Sequence[int], level: tx.Sequence[int]
) -> tx.List[float]:
    """Return the downsampling factor of a pyramid level along each axis.

    The factor is the base shape divided by the shape of the level, without
    rounding: a level of 51 pixels under a base of 101 is downsampled by
    101/51, so that both cover the same extent.
    """
    if len(base) != len(level):
        raise ParserContentError(
            f"A pyramid level of shape {tuple(level)} does not have as "
            f"many axes as its base level, of shape {tuple(base)}."
        )
    factors = []
    for b, n in zip(base, level):
        if n <= 0:
            raise ParserContentError(
                f"A pyramid level has an empty axis (shape {tuple(level)})."
            )
        factors.append(float(b) / float(n))
    return factors


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------

_LOCAL_PROTOCOLS = frozenset({"", "file", "local"})


def _is_local(file: tx.Any) -> bool:
    try:
        return path.Path(file).protocol.lower() in _LOCAL_PROTOCOLS
    except Exception:
        return False


class TiffSource:
    """Where a TIFF file can be opened again to read its pixels.

    A local file is opened again by name, which allows memory-mapping. A remote
    file, a stream or bytes are held in memory.
    """

    def __init__(
        self,
        filename: tx.Optional[str] = None,
        content: tx.Optional[bytes] = None,
    ) -> None:
        if (filename is None) == (content is None):
            raise ValueError("Give either a filename or the content.")
        self.filename = filename
        self.content = content

    @classmethod
    def from_file(cls, file: tx.Any) -> "TiffSource":
        """Return the source of a path, a file object or bytes."""
        if isinstance(file, (bytes, bytearray, memoryview)):
            return cls(content=bytes(file))
        if isinstance(file, (str, path.PathLike)):
            if _is_local(file):
                return cls(filename=str(path.Path(file)))
            with path.Path(file).open("rb") as f:
                return cls(content=f.read())
        if hasattr(file, "read"):
            return cls(content=file.read())
        raise TypeError(f"Cannot read a TIFF file from {type(file)}.")

    def open(self) -> tx.Any:
        """Open the file with tifffile, as a context manager."""
        tf = _require_tifffile()
        try:
            if self.filename is not None:
                return tf.TiffFile(self.filename)
            return tf.TiffFile(BytesIO(self.content))
        except (tf.TiffFileError, ValueError, OSError) as e:
            raise ParserContentError(
                f"tifffile cannot read this file: {e}"
            ) from e


# tifffile flags of the vendor whole-slide formats, which OpenSlide reads with
# a dedicated reader (brainhops.io.images.openslide).
_WHOLE_SLIDE_FLAGS = ("is_svs", "is_ndpi", "is_philips", "is_scn", "is_bif")


def is_whole_slide(tif: tx.Any) -> bool:
    """Return whether a TIFF file is a vendor whole-slide image (Aperio SVS,
    Hamamatsu NDPI, Philips, Leica SCN or Ventana BIF), judged by its first
    page.
    """
    page = tif.pages.first
    return any(getattr(page, flag, False) for flag in _WHOLE_SLIDE_FLAGS)


def _select(tif: tx.Any, series: int, level: int) -> tx.Any:
    n = len(tif.series)
    if not -n <= series < n:
        raise IndexError(
            f"This file has {n} series, so it has no series {series}."
        )
    one = tif.series[series]
    levels = len(one.levels)
    if not -levels <= level < levels:
        raise IndexError(
            f"Series {series} of this file has {levels} level(s), so it "
            f"has no level {level}."
        )
    return one.levels[level]


def read_series(
    source: TiffSource,
    series: int = 0,
    level: int = 0,
    mmap: bool = True,
    lazy: tx.Optional[bool] = None,
) -> tx.Any:
    """Read the pixels of one level of one series.

    The array is C-ordered, in the layout of tifffile (`series.axes`), and is
    read in the first applicable way:

    1. Memory-mapped copy-on-write, with `mmap=True`, when the file is local
       and the level is stored uncompressed and contiguously.
    2. As a dask array over the Zarr store of tifffile, when `lazy` is true, or
       `None` with dask as the array backend, and dask and zarr are installed.
    3. Eagerly, into a NumPy array.

    Raises
    ------
    IndexError
        If there is no such series or level.
    ParserContentError
        If tifffile cannot decode the level.
    """
    with source.open() as tif:
        one = _select(tif, series, level)
        if (
            mmap
            and source.filename is not None
            and one.dataoffset is not None
            and one.dtype is not None
        ):
            dtype = np.dtype(tif.byteorder + one.dtype.char)
            return np.memmap(
                source.filename,
                dtype=dtype,
                mode="c",
                offset=one.dataoffset,
                shape=tuple(one.shape),
                order="C",
            )
        if lazy is None:
            from brainhops.backends import get_array_backend

            try:
                import dask.array as da
            except ImportError:  # pragma: no cover
                da = None
            lazy = da is not None and get_array_backend() is da
        if lazy:
            array = _read_lazily(source, series, level)
            if array is not None:
                return array
        try:
            return one.asarray()
        except Exception as e:
            raise ParserContentError(
                f"tifffile cannot decode series {series}, level {level}: {e}"
            ) from e


def _read_lazily(source: TiffSource, series: int, level: int) -> tx.Any:
    """Return a dask array over the Zarr store of tifffile, or `None` if dask
    or zarr is missing.
    """
    try:
        import dask.array as da
        import zarr
    except ImportError:
        return None
    tf = _require_tifffile()
    if source.filename is not None:
        store = tf.imread(
            source.filename, aszarr=True, series=series, level=level
        )
    else:
        store = tf.imread(
            BytesIO(source.content), aszarr=True, series=series, level=level
        )
    try:
        array = da.from_zarr(zarr.open(store, mode="r"))
    except Exception:
        store.close()
        return None
    _close_with(store)
    return array


def _close_with(store: tx.Any) -> None:
    """Close the files of a tifffile Zarr store when the store is collected.

    The store keeps the file it opens for chunks until `store.close()`, which a
    dask graph never calls, so only garbage collection would close it, with a
    `ResourceWarning`. The finalizer holds the cache rather than the store, so
    that it keeps neither the store nor the dask array alive.
    """
    cache = getattr(store, "_filecache", None)
    if cache is not None:
        weakref.finalize(store, cache.clear)
