"""
Decoding TIFF files with tifffile, and reading the pixel size their
metadata dialects record.

This is the tifffile *backend* of the TIFF image reader. It opens a file,
selects a series and a pyramid level, reads its pixels -- memory-mapped,
lazily, or eagerly -- and gathers what the file says about the size of a
pixel. It knows nothing of coordinate systems or transformations, which
are the business of the shared raster conventions in
`brainhops.io.images.base._utils_raster`. It mirrors
`brainhops.io.images.pillow._utils`, the Pillow backend of the raster
reader.

## Dialects

A TIFF file stores its pixel size in one of three ways, which this module
reads in this order of precedence (a dialect wins over the next one for
every axis it gives a size for):

1. **OME-TIFF.** The OME-XML in the first `ImageDescription` gives
   `PhysicalSizeX`, `PhysicalSizeY` and `PhysicalSizeZ`, each with its
   unit (`PhysicalSizeXUnit`, ..., micrometres when absent, as the schema
   says), and `TimeIncrement` (seconds when its unit is absent). The
   position of the first plane (`PositionX`, `PositionY`, `PositionZ` of
   the plane whose Z, C and T indices are all 0) is the origin. The axis order
   (`DimensionOrder`) is applied by tifffile.
2. **ImageJ.** An ImageJ hyperstack records its unit of length (`unit`,
   with `yunit` and `zunit` when they differ) in its `ImageDescription`;
   the pixel width and height are `1 / XResolution` and `1 / YResolution`
   in that unit, and the slice step is `spacing`, which is negative when
   the slices run backwards. The frame interval is `finterval`, in
   `tunit` (seconds by default).
3. **Resolution tags.** `XResolution` and `YResolution`, in pixels per
   `ResolutionUnit`: 1 (none: an aspect ratio only, so the size is
   unknown), 2 (inch, the default when the tag is absent), 3
   (centimetre), and tifffile's extensions 4 (millimetre) and 5
   (micrometre).

A size that is absent, or is a placeholder, is *unknown* and is left out.
tifffile reports missing resolution tags as 1 pixel per inch, so their
absence is detected from the tags themselves; a resolution in inches or
centimetres equal to 72 or 96 dpi, or to one pixel per unit, is a
placeholder (see `is_default_dpi`).
"""

__all__ = [
    "EXTENSIONS",
    "MAGICS",
    "RESOLUTION_UNITS",
    "TAGS_OF_INTEREST",
    "TiffMetadata",
    "TiffSource",
    "is_tiff",
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

# stdlib
import math
import xml.etree.ElementTree as ElementTree
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic

# internals
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
"""The file extensions of TIFF files, BigTIFF and OME-TIFF included."""

MAGICS: tx.Tuple[bytes, ...] = (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+")
"""The first four bytes of a TIFF file (little- and big-endian) and of a
BigTIFF file."""

RESOLUTION_UNITS: tx.Dict[int, tx.Optional[str]] = {
    1: None,
    2: "inch",
    3: "cm",
    4: "mm",
    5: "um",
}
"""The units of the `ResolutionUnit` tag: the TIFF ones (1 to 3) and
tifffile's extensions (4 and 5). `None` is "no unit": the resolution is
an aspect ratio."""

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
"""The tags of the first page that the reader keeps (by code)."""

_TIFF_INCH = 2  # The default of `ResolutionUnit`.

# Spellings of units of length, normalized: OME's symbols, ImageJ's words
# (and its escaped micro sign), and the usual abbreviations.
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
    """Whether the first bytes of a file are those of a TIFF (or
    BigTIFF) file."""
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
    """
    A unit of length as the metadata spells it, normalized (`"micron"`,
    `"µm"` and ImageJ's escaped `"\\u00B5m"` are all `"um"`), or `None`
    when it is not a physical unit of length (`"pixel"`, OME's
    `"reference frame"`, or anything unknown).
    """
    return _unit(value, _LENGTH_UNITS)


def time_unit(value: tx.Any) -> tx.Optional[str]:
    """A unit of time as the metadata spells it, normalized (`"sec"` is
    `"s"`), or `None` when it is unknown."""
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
    """The value of a TIFF rational (a `(numerator, denominator)` pair),
    or of a number."""
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
    """What a TIFF file says about the geometry of one of its series."""

    dialect: tx.Optional[str]
    """`"ome"`, `"imagej"`, or `None` for a plain TIFF file."""

    scales: _Scales
    """The size of a sample and its unit, by axis name (`"x"`, `"y"`,
    `"z"`, `"t"`), for the axes whose size is known. Always positive."""

    signs: tx.Dict[str, float]
    """The sign of an axis whose step is negative (an ImageJ `spacing`
    below zero), by axis name: `-1.0`."""

    origin: _Scales
    """The position of the first sample and its unit, by axis name, for
    the axes whose position is known (OME plane positions)."""

    sources: tx.Dict[str, str]
    """Which dialect each size in `scales` was taken from: `"ome"`,
    `"imagej"` or `"resolution"`."""

    ome_xml: tx.Optional[str]
    """The OME-XML of the file, if it is an OME-TIFF."""

    ome_index: tx.Optional[int]
    """The position, among the OME `Image` elements, of the one that
    describes the series."""

    imagej: tx.Optional[tx.Dict[str, tx.Any]]
    """The ImageJ metadata of the file, if it is an ImageJ hyperstack."""

    tags: tx.Dict[str, tx.Any]
    """The tags of interest of the first page, by name
    (`TAGS_OF_INTEREST`)."""


def tags_of_interest(page: tx.Any) -> tx.Dict[str, tx.Any]:
    """The tags of interest of a page, by name. The first of repeated
    tags (several `ImageDescription`s, say) is kept."""
    out: tx.Dict[str, tx.Any] = {}
    for tag in page.tags.values():
        name = TAGS_OF_INTEREST.get(tag.code)
        if name is None or name in out:
            continue
        value = tag.value
        if hasattr(value, "value"):  # an enum, such as RESUNIT
            value = value.value
        out[name] = value
    return out


def resolution_scales(page: tx.Any) -> _Scales:
    """
    The pixel width (`"x"`) and height (`"y"`) the resolution tags of a
    page give, or nothing when they are absent, give an aspect ratio only
    (`ResolutionUnit` 1), or are placeholders.

    The tags are read directly: tifffile reports their absence as one
    pixel per inch, which would pass for a resolution.
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
        # In millimetres: `Unit("inch")` does not convert reliably.
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
    """
    The sizes an ImageJ hyperstack records, and the signs of its steps.

    Parameters
    ----------
    metadata : Mapping
        The ImageJ metadata (tifffile's `imagej_metadata`).
    page : TiffPage
        The first page, whose resolution tags hold the pixel size in
        ImageJ's unit.
    names : Collection[str]
        The names of the axes of the series (`"x"`, `"y"`, `"z"`,
        `"t"`, ...): sizes are given only for these.

    Returns
    -------
    scales : dict[str, (float, str)]
        The size and unit of each axis whose size is known.
    signs : dict[str, float]
        `{"z": -1.0}` when the slices run backwards.
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
    """An XML tag without its namespace."""
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
    """
    The position, among the OME `Image` elements, of the image that
    describes series `index` of a file.

    tifffile makes one series per OME `Image`, in order, but skips the
    images whose planes it cannot find. So each series is matched to the
    next image of the same name and size.

    Parameters
    ----------
    ome_xml : str
        The OME-XML.
    series : Sequence[TiffPageSeries]
        All the series of the file (anything with `name`, `axes` and
        `shape`).
    index : int
        The series to describe.

    Returns
    -------
    int | None
        The position of the image, or `None` if none matches.
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
    """The OME `Image` element at a position, and its `Pixels` element,
    or `(None, None)`."""
    images = _ome_images(ome_xml)
    if index is None or not 0 <= index < len(images):
        return None, None
    pixels = _children(images[index], "Pixels")
    return images[index], (pixels[0] if pixels else None)


def ome_channel_names(pixels: tx.Any) -> tx.List[tx.Optional[str]]:
    """The names of the channels of an OME `Pixels` element."""
    return [channel.get("Name") for channel in _children(pixels, "Channel")]


def ome_scales(
    pixels: tx.Any, names: tx.Collection[str]
) -> tx.Tuple[_Scales, _Scales]:
    """
    The sizes, and the origin, an OME `Pixels` element records.

    Parameters
    ----------
    pixels : Element
        The `Pixels` element (see
        `ome_image`).
    names : Collection[str]
        The names of the axes of the series: sizes are given only for
        these.

    Returns
    -------
    scales : dict[str, (float, str)]
        The size and unit of each axis whose size is known. A size whose
        unit is not a unit of length (`"pixel"`, `"reference frame"`) is
        unknown.
    origin : dict[str, (float, str | None)]
        The position of the first plane along `x`, `y` and `z`, for those
        the file records. Its unit is `None` when the file names none (the
        caller decides what that means).
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
    """
    What a TIFF file says about the geometry of one of its series, with
    the precedence OME > ImageJ > resolution tags applied axis by axis.

    Parameters
    ----------
    tif : tifffile.TiffFile
        The open file.
    index : int
        The series.

    Returns
    -------
    TiffMetadata
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
    """
    The downsampling factor of a pyramid level along each axis: the ratio
    of the shape of the base level to its own.

    The ratio is not rounded: a level of 51 samples under a base of 101
    is downsampled by 101 / 51, so that both cover the same extent.
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
    """
    Where a TIFF file can be opened again, to read its pixels after the
    metadata has been read.

    A local file is reopened by name, which lets its pixels be
    memory-mapped. Anything else -- a remote file, a stream, bytes -- is
    held in memory as the bytes of the file, and decoded from there.
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
        """The source of a path, a file object or bytes."""
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
        """Open the file with tifffile (a context manager)."""
        tf = _require_tifffile()
        try:
            if self.filename is not None:
                return tf.TiffFile(self.filename)
            return tf.TiffFile(BytesIO(self.content))
        except (tf.TiffFileError, ValueError, OSError) as e:
            raise ParserContentError(
                f"tifffile cannot read this file: {e}"
            ) from e


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
    """
    Read the pixels of one level of one series, C-ordered, as tifffile
    lays them out (`series.axes`).

    The pixels are read in the first of these ways that applies:

    1. **Memory-mapped** (`mmap=True`), when the file is local and the
       level is stored uncompressed and contiguously. Nothing is read until
       the array is indexed. The map is copy-on-write: writing into the
       array does not change the file.
    2. **Lazily**, as a dask array over tifffile's Zarr store, when `lazy`
       is true, or is `None` and dask is the array backend
       (`brainhops.backends`). This needs dask and zarr; without them the
       pixels are read eagerly.
    3. **Eagerly**, decoded into a numpy array.

    Raises
    ------
    IndexError
        If the file has no such series or level.
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
    """A dask array over tifffile's Zarr store, or `None` if dask or zarr
    is missing."""
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
        return da.from_zarr(zarr.open(store, mode="r"))
    except Exception:
        store.close()
        return None
