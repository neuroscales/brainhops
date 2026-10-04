"""
The machinery shared by every OpenSlide format: opening a slide, reading
its levels region by region, and its geometry.

The per-vendor classes (`_formats.py`) only say which vendor they read,
under which extensions and hints.
"""

# stdlib
import functools
import math
import os
import shutil
import tempfile
import weakref

# dependencies
import numpy as np
import typing_extensions as tx

from brainhops._core import dependencies as deps

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.io.base._dispatch import _to_filename
from brainhops.io.base.parsers import (
    BinaryFileParser,
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
    SnifferExistsError,
)
from brainhops.io.images.base import FileBasedImage
from brainhops.io.images.base import _utils_raster as raster

# ----------------------------------------------------------------------
#   CONSTANTS
# ----------------------------------------------------------------------

# OpenSlide decodes every level as RGBA rows: C-ordered (y, x, samples).
# The alpha channel only marks the pixels outside the scanned area, which
# are composited onto the background colour, so the image is RGB.
_STORAGE = "YXS"

# The dask chunks of a lazily read level are whole tiles, grouped so that
# a chunk is at least this many pixels wide and tall.
_MIN_CHUNK = 1024

_LOCAL_PROTOCOLS = frozenset({"", "file", "local"})

# The single-scale class of each vendor, by OpenSlide vendor name, which
# the multiscale classes build their levels with (filled by `_formats`).
_LEVEL_CLASSES: tx.Dict[str, type] = {}


def _require_openslide() -> tx.Any:
    openslide = deps.openslide
    if openslide is None:
        raise ImportError(
            "Reading whole-slide images needs openslide-python and the "
            "OpenSlide library: pip install brainhops[openslide]"
        )
    return openslide


# ----------------------------------------------------------------------
#   OPENING A SLIDE
# ----------------------------------------------------------------------


def _local_path(file: tx.Any) -> tx.Optional[str]:
    """The local path of a file OpenSlide can open by name, or `None`:
    a local path, or an open file with the name of a local file."""
    if isinstance(file, (str, path.PathLike)):
        try:
            if path.Path(file).protocol.lower() not in _LOCAL_PROTOCOLS:
                return None
        except Exception:
            return None
        local = str(path.Path(file))
    else:
        local = getattr(file, "name", None)
        if not isinstance(local, str):
            return None
    return local if os.path.isfile(local) else None


@functools.lru_cache(maxsize=64)
def _detect(filename: str, mtime: int, size: int) -> tx.Tuple[str, int]:
    """The OpenSlide vendor of a file and its number of levels, or
    `("", 0)` if OpenSlide cannot read it. Cached by modification time
    and size, since every OpenSlide format sniffs the same file."""
    del mtime, size  # part of the cache key only
    openslide = deps.openslide
    try:
        vendor = openslide.OpenSlide.detect_format(filename)
        if not vendor:
            return "", 0
        with openslide.OpenSlide(filename) as slide:
            return str(vendor), int(slide.level_count)
    except Exception:
        return "", 0


# What a file OpenSlide reads may start with, at an offset: a TIFF or
# BigTIFF header (Aperio, Hamamatsu NDPI, Leica, Philips, Trestle,
# Ventana, generic tiled TIFF), a Zeiss CZI segment, a Sakura SQLite
# database, or DICOM's preamble. The other formats (Hamamatsu VMS and
# VMU, MIRAX) are text files that OpenSlide recognizes by their
# extension.
_SLIDE_MAGIC = (
    (0, (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")),
    (0, (b"ZISRAWFILE",)),
    (0, (b"SQLite format 3\x00",)),
    (128, (b"DICM",)),
)
_SLIDE_EXTENSIONS = (".vms", ".vmu", ".mrxs")


def _may_be_slide(filename: str) -> bool:
    """Whether OpenSlide may read a file, judged from its first bytes or
    its extension, so that a file that is no slide is declined without
    importing OpenSlide."""
    if filename.lower().endswith(_SLIDE_EXTENSIONS):
        return True
    try:
        with open(filename, "rb") as f:
            head = f.read(132)
    except OSError:
        return False
    return any(
        head[offset:].startswith(magic) for offset, magic in _SLIDE_MAGIC
    )


def _detect_file(filename: str) -> tx.Tuple[str, int]:
    if not _may_be_slide(filename) or deps.openslide is None:
        return "", 0
    stat = os.stat(filename)
    return _detect(os.path.abspath(filename), stat.st_mtime_ns, stat.st_size)


def _release(state: tx.Dict[str, tx.Any]) -> None:
    slide = state.pop("slide", None)
    if slide is not None:
        slide.close()
    tempdir = state.pop("tempdir", None)
    if tempdir is not None:
        shutil.rmtree(tempdir, ignore_errors=True)


class _Slide:
    """
    An OpenSlide slide, opened on first use and closed (and its temporary
    copy, if any, deleted) when the last image or array reading from it
    is collected.
    """

    def __init__(
        self, filename: str, tempdir: tx.Optional[str] = None
    ) -> None:
        self.filename = filename
        self._state: tx.Dict[str, tx.Any] = {"tempdir": tempdir}
        weakref.finalize(self, _release, self._state)

    @classmethod
    def from_content(cls, content: bytes, suffix: str) -> "_Slide":
        """A slide held in memory, spilled into a temporary file: OpenSlide
        only opens files by name. A format made of several files (MIRAX,
        Hamamatsu VMS) cannot be read this way."""
        tempdir = tempfile.mkdtemp(prefix="brainhops-openslide-")
        filename = os.path.join(tempdir, "slide" + suffix)
        with open(filename, "wb") as f:
            f.write(content)
        return cls(filename, tempdir)

    @property
    def slide(self) -> tx.Any:
        slide = self._state.get("slide")
        if slide is None:
            os_ = _require_openslide()
            try:
                slide = os_.OpenSlide(self.filename)
            except Exception as e:
                raise ParserContentError(
                    f"OpenSlide cannot read this file: {e}"
                ) from e
            self._state["slide"] = slide
        return slide

    @property
    def vendor(self) -> tx.Optional[str]:
        return self.slide.properties.get("openslide.vendor")


# ----------------------------------------------------------------------
#   LAZY LEVEL
# ----------------------------------------------------------------------


def _hex_colour(text: tx.Optional[str]) -> np.ndarray:
    """An `RRGGBB` colour as three bytes, white if there is none."""
    try:
        value = int(str(text), 16)
    except (TypeError, ValueError):
        value = 0xFFFFFF
    return np.array(
        [(value >> 16) & 255, (value >> 8) & 255, value & 255], dtype=np.uint16
    )


def _bounding(key: tx.Any, n: int) -> tx.Tuple[int, int, tx.Any]:
    """The range `[start, stop)` an index along one axis needs, and the
    index into that range."""
    if isinstance(key, (int, np.integer)):
        i = int(key)
        if not -n <= i < n:
            raise IndexError(f"Index {i} is out of bounds for size {n}.")
        i %= n
        return i, i + 1, 0
    if isinstance(key, slice):
        start, stop, step = key.indices(n)
        indices = range(start, stop, step)
        if not len(indices):
            return 0, 0, slice(0, 0)
        lo, hi = min(indices), max(indices) + 1
        # The region ends at the last index either way.
        return lo, hi, slice(start - lo, None, step)
    return 0, n, key  # an array index: read the whole axis


class _SlideArray:
    """
    One level of a slide, F-ordered (`x, y, c`, RGB), read region by
    region: indexing it reads only the region it covers, with OpenSlide's
    `read_region`. Pixels outside the scanned area are the slide's
    background colour.
    """

    dtype = np.dtype(np.uint8)
    ndim = 3

    def __init__(self, slide: _Slide, level: int) -> None:
        handle = slide.slide
        self._slide = slide
        self.level = level
        width, height = handle.level_dimensions[level]
        self.shape = (int(width), int(height), 3)
        self.downsample = float(handle.level_downsamples[level])
        self._background = _hex_colour(
            handle.properties.get("openslide.background-color")
        )

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return self.size

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"<slide level {self.level}, shape {self.shape}, uint8>"

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Any = None
    ) -> np.ndarray:
        array = self._read(0, 0, self.shape[0], self.shape[1])
        return array if dtype is None else array.astype(dtype)

    def __getitem__(self, key: tx.Any) -> np.ndarray:
        """Read the region an index covers. Integers and slices index as
        in numpy; integer or boolean arrays index each axis on its own
        (orthogonally)."""
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            i = next(i for i, k in enumerate(key) if k is Ellipsis)
            fill = (slice(None),) * (self.ndim - len(key) + 1)
            key = key[:i] + fill + key[i + 1 :]
        if len(key) > self.ndim or any(k is None for k in key):
            raise IndexError(
                "A slide level takes at most three indices (x, y, c), "
                "and no new axis."
            )
        key = key + (slice(None),) * (self.ndim - len(key))
        x0, x1, kx = _bounding(key[0], self.shape[0])
        y0, y1, ky = _bounding(key[1], self.shape[1])
        out = self._read(x0, y0, x1 - x0, y1 - y0)
        drop = []
        for axis, k in enumerate((kx, ky, key[2])):
            if isinstance(k, (int, np.integer)):
                out = np.take(out, [int(k)], axis=axis)
                drop.append(axis)
            elif isinstance(k, slice):
                out = out[(slice(None),) * axis + (k,)]
            else:
                k = np.asarray(k)
                if k.dtype == bool:
                    k = np.nonzero(k)[0]
                out = np.take(out, k, axis=axis)
        return out.squeeze(axis=tuple(drop)) if drop else out

    def _read(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """The region of this level at `(x, y)` (in its own pixels) of
        size `(width, height)`, as an `(x, y, c)` RGB array."""
        if width <= 0 or height <= 0:
            return np.zeros((max(width, 0), max(height, 0), 3), np.uint8)
        location = (
            int(round(x * self.downsample)),
            int(round(y * self.downsample)),
        )
        image = self._slide.slide.read_region(
            location, self.level, (int(width), int(height))
        )
        rgba = np.asarray(image)
        alpha = rgba[..., 3:4]
        rgb = rgba[..., :3]
        if not (alpha == 255).all():
            # OpenSlide (python) returns un-premultiplied RGBA: composite
            # it onto the background colour.
            a = alpha.astype(np.uint16)
            rgb = (
                rgb.astype(np.uint16) * a + self._background * (255 - a) + 127
            ) // 255
            rgb = rgb.astype(np.uint8)
        return np.ascontiguousarray(rgb).transpose(1, 0, 2)


def _level_data(
    slide: _Slide, level: int, lazy: tx.Optional[bool]
) -> ArrayProtocol:
    """A level, as a dask array of whole tiles when `lazy` is true (or is
    `None` and dask is the array backend), else as a `_SlideArray`."""
    array = _SlideArray(slide, level)
    if lazy is None:
        from brainhops.backends import get_array_backend

        try:
            import dask.array as da
        except ImportError:  # pragma: no cover
            da = None
        lazy = da is not None and get_array_backend() is da
    if not lazy:
        return array
    try:
        import dask.array as da
    except ImportError:
        return array
    properties = slide.slide.properties
    chunks = []
    for name, n in (("width", array.shape[0]), ("height", array.shape[1])):
        tile = properties.get(f"openslide.level[{level}].tile-{name}")
        try:
            tile = max(int(tile), 1)
        except (TypeError, ValueError):
            tile = 256
        chunks.append(min(n, tile * max(1, math.ceil(_MIN_CHUNK / tile))))
    return da.from_array(
        array,
        chunks=(chunks[0], chunks[1], 3),
        asarray=False,
        fancy=False,
        lock=False,
        meta=np.empty((0, 0, 0), dtype=np.uint8),
    )


# ----------------------------------------------------------------------
#   GEOMETRY AND METADATA
# ----------------------------------------------------------------------


def _mpp(properties: tx.Mapping[str, str]) -> tx.Dict[str, tx.Any]:
    """The pixel size of the full-resolution level, in micrometres, along
    the axes for which the slide records one (`openslide.mpp-x/y`)."""
    out = {}
    for name in ("x", "y"):
        try:
            size = float(properties.get(f"openslide.mpp-{name}", ""))
        except ValueError:
            continue
        if math.isfinite(size) and size > 0:
            out[name] = (size, "um")
    return out


def _bounds(
    properties: tx.Mapping[str, str],
) -> tx.Optional[tx.Tuple[int, int, int, int]]:
    try:
        return tuple(
            int(properties[f"openslide.bounds-{key}"])
            for key in ("x", "y", "width", "height")
        )
    except (KeyError, ValueError):
        return None


def _level_geometry(
    slide: _Slide,
    level: int,
    pixel_size: tx.Any = None,
    unit: tx.Any = None,
) -> tx.Any:
    """
    The pixel-to-physical transformation of a level: the full-resolution
    pixel size (`openslide.mpp-x/y`, or the caller's) times the level's
    downsampling factor, with the level covering the extent of the full
    resolution (pixel `i` centred on full-resolution pixel
    `f * i + (f - 1) / 2`).
    """
    handle = slide.slide
    axes = raster.storage_axes(_STORAGE)
    axes = [axes[p] for p in raster.canonical_permutation(axes)]
    scales = raster.resolve_pixel_size(
        axes, _mpp(handle.properties), pixel_size=pixel_size, unit=unit
    )
    factor = float(handle.level_downsamples[level])
    return raster.level_transformation(axes, scales, [factor, factor, 1], {})


# ----------------------------------------------------------------------
#   SHARED READER MACHINERY
# ----------------------------------------------------------------------


class OpenSlideFormat:
    """
    An image read with [OpenSlide](https://openslide.org/). Each vendor
    format derives from it, so that the hint `"openslide"` selects them
    all, and `"openslide.<vendor>"` one of them.
    """

    HINTS = ("openslide",)

    VENDOR: tx.ClassVar[tx.Optional[str]] = None
    """The vendor name OpenSlide gives to the format
    (`OpenSlide.detect_format`)."""

    SCORE: tx.ClassVar[float] = Confidence.CERTAIN
    """How confident the format is in a file OpenSlide attributes to its
    vendor."""


class _OpenSlideMixin(OpenSlideFormat):
    """Sniffing, opening and the slide's metadata, shared by the
    single-scale and multiscale OpenSlide images."""

    @classmethod
    def _score(cls, levels: int, level: tx.Optional[int]) -> float:
        """The score of a file of this vendor, with `levels` levels, when
        `level` is (or is not) asked for."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def _sniff_local(
        cls,
        filename: str,
        error: tx.Union[bool, tx.Type[Exception]],
        level: tx.Optional[int],
    ) -> float:
        vendor, levels = _detect_file(filename)
        if cls.VENDOR is not None and vendor == cls.VENDOR:
            return cls._score(levels, level)
        if error:
            if error is True:
                error = SnifferContentError
            raise error(f"This file is not a {cls.VENDOR} slide.")
        return Confidence.NO

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Score how confident the class is that a file is a slide of its
        vendor, as OpenSlide detects it. Only a local file is sniffed.
        """
        local = _local_path(filename)
        if local is None:
            if error:
                if error is True:
                    error = SnifferExistsError
                raise error(f"No such local file: {filename}")
            return Confidence.NO
        return cls._sniff_local(local, error, level)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Score an open file, by the name of the local file it was opened
        from. A stream with no such name is not sniffed (`NO`): OpenSlide
        reads files by name, so it is read only when asked for by hint.
        """
        local = _local_path(file)
        if local is None:
            if error:
                if error is True:
                    error = SnifferContentError
                raise error("OpenSlide only sniffs local files.")
            return Confidence.NO
        return cls._sniff_local(local, error, level)

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Content in memory is not sniffed (`NO`); it is read only when
        asked for by hint."""
        if error:
            if error is True:
                error = SnifferContentError
            raise error("OpenSlide only sniffs local files.")
        return Confidence.NO

    @classmethod
    def _suffix(cls, name: tx.Optional[str]) -> str:
        """The extension of a temporary copy: OpenSlide detects some
        formats by it."""
        if name:
            for ext in sorted(cls.EXTENSIONS, key=len, reverse=True):
                if name.lower().endswith(ext):
                    return ext
            ext = os.path.splitext(name)[1]
            if ext:
                return ext
        return cls.EXTENSIONS[0] if cls.EXTENSIONS else ""

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """
        Read the slide from a file. A local file is opened by OpenSlide
        directly; a remote one is copied into a temporary file first (which
        only works for the formats held in a single file). See
        `from_source` for the options.
        """
        local = _local_path(filename)
        if local is not None:
            return cls.from_source(_Slide(local), **kwargs)
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        with filename.open("rb") as f:
            content = f.read()
        slide = _Slide.from_content(
            content, cls._suffix(_to_filename(filename))
        )
        return cls.from_source(slide, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Read the slide from an open file: by its name if it is a local
        file, or else from a temporary copy. See `from_source` for the
        options."""
        local = _local_path(file)
        if local is not None:
            return cls.from_source(_Slide(local), **kwargs)
        with preserve_position(file):
            content = file.read()
        slide = _Slide.from_content(content, cls._suffix(_to_filename(file)))
        return cls.from_source(slide, **kwargs)

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Read the slide from the bytes of a file, through a temporary
        copy. See `from_source` for the options."""
        slide = _Slide.from_content(bytes(content), cls._suffix(None))
        return cls.from_source(slide, **kwargs)

    @classmethod
    def _check_vendor(cls, slide: _Slide) -> None:
        vendor = slide.vendor
        if vendor != cls.VENDOR:
            raise ParserContentError(
                f"This file is a {vendor!r} slide, not a {cls.VENDOR!r} "
                f"one, so {cls.__name__} does not read it."
            )

    @staticmethod
    def _metadata(slide: _Slide) -> tx.Dict[str, tx.Any]:
        handle = slide.slide
        properties = dict(handle.properties)
        return dict(
            vendor=properties.get("openslide.vendor"),
            properties=properties,
            n_levels=int(handle.level_count),
            level_downsamples=tuple(
                float(f) for f in handle.level_downsamples
            ),
            background_color=properties.get("openslide.background-color"),
            bounds=_bounds(properties),
            associated_images=tuple(handle.associated_images.keys()),
        )

    def associated_image(self, name: str) -> np.ndarray:
        """
        An associated image of the slide (`"label"`, `"macro"`,
        `"thumbnail"`, ... as listed in `associated_images`), as an
        F-ordered `(x, y, c)` RGB array, composited onto white.
        """
        slide = getattr(self, "_slide", None)
        if slide is None:
            raise KeyError(name)
        image = slide.slide.associated_images[name]
        rgba = np.asarray(image.convert("RGBA"))
        a = rgba[..., 3:4].astype(np.uint16)
        rgb = rgba[..., :3].astype(np.uint16) * a + 255 * (255 - a) + 127
        rgb = (rgb // 255).astype(np.uint8)
        return np.ascontiguousarray(rgb).transpose(1, 0, 2)


# The annotations of the metadata kept on both kinds of image.
_Properties = tx.Annotated[
    tx.Optional[tx.Dict[str, str]],
    tx.Doc(
        "Every property OpenSlide reports for the slide, by name: the "
        "standard `openslide.*` ones and the vendor's own."
    ),
]
_Vendor = tx.Annotated[
    tx.Optional[str],
    tx.Doc("The vendor of the slide, as OpenSlide names it."),
]
_NLevels = tx.Annotated[
    tx.Optional[int], tx.Doc("The number of levels of the slide.")
]
_Downsamples = tx.Annotated[
    tx.Optional[tx.Tuple[float, ...]],
    tx.Doc("The downsampling factor of each level (`level_downsamples`)."),
]
_Background = tx.Annotated[
    tx.Optional[str],
    tx.Doc(
        "The background colour of the slide (`RRGGBB`), which fills the "
        "pixels outside the scanned area; white if absent."
    ),
]
_Bounds = tx.Annotated[
    tx.Optional[tx.Tuple[int, int, int, int]],
    tx.Doc(
        "The bounding box of the scanned area (`x, y, width, height`, in "
        "full-resolution pixels), if the slide records one."
    ),
]
_Associated = tx.Annotated[
    tx.Optional[tx.Tuple[str, ...]],
    tx.Doc(
        "The names of the slide's associated images (label, macro, "
        "thumbnail, ...), read with `associated_image`."
    ),
]


# ----------------------------------------------------------------------
#   SINGLE-SCALE IMAGE
# ----------------------------------------------------------------------


class OpenSlideImage(
    _OpenSlideMixin, BinaryFileParser, FileBasedImage, SingleScaleImage
):
    """
    One level of a whole-slide image, read with OpenSlide: the base of
    the single-scale image of every vendor.

    The data is F-ordered `(x, y, c)` RGB `uint8`, and is read region by
    region when it is indexed (or as a dask array of whole tiles), never
    in full unless asked for (`numpy.asarray(image.data)`). The only
    transformation is a scaling from the pixel system to a `"physical"`
    one, in micrometres when the slide records its pixel size
    (`openslide.mpp-x/y`) and the identity, in no unit, otherwise.
    """

    vendor: _Vendor = None
    properties: _Properties = None
    level: tx.Annotated[
        tx.Optional[int],
        tx.Doc("The level that was read (0: full resolution)."),
    ] = None
    n_levels: _NLevels = None
    level_downsamples: _Downsamples = None
    background_color: _Background = None
    bounds: _Bounds = None
    associated_images: _Associated = None

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        slide = getattr(self, "_slide", None)
        if slide is None:
            return None
        return _level_data(slide, self.level or 0, self._lazy)

    @classmethod
    def _score(cls, levels: int, level: tx.Optional[int]) -> float:
        """The vendor's score when a level is asked for, or the slide has
        a single level; less (yielding to the multiscale image) for a
        pyramid."""
        if level is not None or levels < 2:
            return cls.SCORE
        return 0.8 * cls.SCORE

    @classmethod
    def from_source(
        cls,
        slide: _Slide,
        level: tx.Optional[int] = None,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        lazy: tx.Optional[bool] = None,
        **kwargs,
    ) -> tx.Self:
        """
        Read one level of a slide.

        Parameters
        ----------
        slide : _Slide
            The slide.
        level : int, optional
            The level to read: 0 (the default) is the full resolution;
            negative values count from the end. A level covers the same
            extent as the full resolution (see `OpenSlideMultiScaleImage`).
        pixel_size : float | Sequence[float] | Mapping[str, float], optional
            The full-resolution pixel size, which overrides the slide's.
        unit : str | Unit, optional
            The unit of `pixel_size`, or the unit to convert the slide's
            (micrometres) to.
        lazy : bool, optional
            Read the pixels as a dask array of whole tiles. By default,
            only when dask is the array backend; otherwise `data` reads
            the region it is indexed with.

        Raises
        ------
        ParserContentError
            If OpenSlide cannot read the file, or it is a slide of another
            vendor.
        IndexError
            If the slide has no such level.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} does not take the option(s) "
                f"{', '.join(sorted(kwargs))}."
            )
        cls._check_vendor(slide)
        count = int(slide.slide.level_count)
        index = 0 if level is None else int(level)
        if not -count <= index < count:
            raise IndexError(
                f"This slide has {count} level(s), so it has no level {index}."
            )
        index %= count
        image = cls(
            transformations=[_level_geometry(slide, index, pixel_size, unit)],
            level=index,
            **cls._metadata(slide),
        )
        image._slide = slide
        image._lazy = lazy
        return image


# ----------------------------------------------------------------------
#   MULTISCALE IMAGE
# ----------------------------------------------------------------------


class OpenSlideMultiScaleImage(
    _OpenSlideMixin, BinaryFileParser, FileBasedImage, MultiScaleImage
):
    """
    A whole-slide image, read with OpenSlide, as a pyramid: the base of
    the multiscale image of every vendor.

    Each OpenSlide level is a single-scale image of the same vendor,
    finest first, read region by region. A level's pixel size is the
    full-resolution pixel size times its downsampling factor
    (`level_downsamples`), and it covers the same extent as the full
    resolution: pixel `i` of a level downsampled by `f` is centred on the
    full-resolution pixel coordinate `f * i + (f - 1) / 2`, as in a TIFF
    or OME-Zarr pyramid. Every level maps onto the same `"physical"`
    system, so the pyramid's own transformations are empty.
    """

    vendor: _Vendor = None
    properties: _Properties = None
    n_levels: _NLevels = None
    level_downsamples: _Downsamples = None
    background_color: _Background = None
    bounds: _Bounds = None
    associated_images: _Associated = None

    PRIORITY: tx.ClassVar[int] = FileBasedImage.PRIORITY + 1
    """
    Content held in memory is not sniffed, so when it is read by hint the
    pyramid and its single-scale class would tie: the pyramid is tried
    first (and declines a `level`).
    """

    @classmethod
    def _score(cls, levels: int, level: tx.Optional[int]) -> float:
        """The vendor's score for a pyramid when no level is asked for,
        and `NO` otherwise."""
        if level is not None or levels < 2:
            return Confidence.NO
        return cls.SCORE

    @classmethod
    def from_source(
        cls,
        slide: _Slide,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        lazy: tx.Optional[bool] = None,
        **kwargs,
    ) -> tx.Self:
        """
        Read every level of a slide. The options are those of the
        single-scale image's `from_source`, but for `level`.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} does not take the option(s) "
                f"{', '.join(sorted(kwargs))}."
            )
        cls._check_vendor(slide)
        single = _LEVEL_CLASSES[cls.VENDOR]
        levels = [
            single.from_source(
                slide,
                level=index,
                pixel_size=pixel_size,
                unit=unit,
                lazy=lazy,
            )
            for index in range(int(slide.slide.level_count))
        ]
        image = cls(images=levels, transformations=[], **cls._metadata(slide))
        image._slide = slide
        return image
