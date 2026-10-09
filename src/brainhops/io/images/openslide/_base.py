"""Machinery shared by every OpenSlide format.

This module opens slides, reads their levels region by region and computes
their geometry. The vendor classes in `_formats` only add a vendor name,
extensions and hints.
"""

import functools
import math
import os
import shutil
import tempfile
import weakref

import numpy as np
import typing_extensions as tx

from brainhops._core import path
from brainhops._core.dependencies import openslide
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.io.base._dispatch import _to_filename
from brainhops.io.base.parsers import (
    BinaryFileReader,
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

# OpenSlide returns C-ordered (y, x, samples) RGBA pixels. The alpha channel
# only marks the pixels outside the scanned area, and it is composited onto
# the background colour.
_STORAGE = "YXS"

# Each dask chunk groups whole tiles until it is at least this many pixels
# wide along each axis (or covers the whole axis).
_MIN_CHUNK = 1024

_LOCAL_PROTOCOLS = frozenset({"", "file", "local"})

# This mapping gives the single-scale class of each vendor name. The multiscale
# classes use it to build their levels, and `_formats` fills it.
_LEVEL_CLASSES: tx.Dict[str, type] = {}


def _require_openslide() -> tx.Any:
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
    """Return the local path of a path or open file, or `None` if OpenSlide
    cannot open it by name.
    """
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
    """Return the vendor and level count of a file, or `("", 0)`.

    Every format class sniffs the same file, so the result is cached. The
    modification time and size of the file are part of the cache key, so
    that a modified file is detected again.
    """
    del mtime, size  # these arguments only serve as the cache key
    try:
        vendor = openslide.OpenSlide.detect_format(filename)
        if not vendor:
            return "", 0
        with openslide.OpenSlide(filename) as slide:
            return str(vendor), int(slide.level_count)
    except Exception:
        return "", 0


def _detect_file(filename: str) -> tx.Tuple[str, int]:
    if openslide is None:
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
    """An OpenSlide slide, opened on first use.

    The slide is closed, and its temporary copy deleted, when the last object
    using it is garbage-collected.
    """

    def __init__(
        self, filename: str, tempdir: tx.Optional[str] = None
    ) -> None:
        self.filename = filename
        self._state: tx.Dict[str, tx.Any] = {"tempdir": tempdir}
        weakref.finalize(self, _release, self._state)

    @classmethod
    def from_content(cls, content: bytes, suffix: str) -> "_Slide":
        """Return a slide whose content is written to a temporary file.

        OpenSlide opens slides only by name, so the content must be written to
        a file first. Multi-file formats cannot be read this way, because the
        temporary directory holds only the one file.
        """
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
    """Convert an `RRGGBB` string to three bytes, white if it is invalid."""
    try:
        value = int(str(text), 16)
    except (TypeError, ValueError):
        value = 0xFFFFFF
    return np.array(
        [(value >> 16) & 255, (value >> 8) & 255, value & 255], dtype=np.uint16
    )


def _bounding(key: tx.Any, n: int) -> tx.Tuple[int, int, tx.Any]:
    """Return the range `[start, stop)` that an index reads along one axis, and
    the index into that range.
    """
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
        # The range ends after the last index, whatever the sign of the step.
        return lo, hi, slice(start - lo, None, step)
    return 0, n, key  # an array index reads the whole axis


class _SlideArray:
    """One level of a slide, as an F-ordered `(x, y, c)` RGB `uint8` array that
    reads only the region it is indexed with.
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
        """Read the region that an index covers.

        Integers and slices behave as in NumPy, while integer and boolean
        arrays index each axis independently.
        """
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
        """Read a region given in level pixels, as an `(x, y, c)` RGB array."""
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
            # read_region returns RGBA that is not premultiplied by alpha
            a = alpha.astype(np.uint16)
            rgb = (
                rgb.astype(np.uint16) * a + self._background * (255 - a) + 127
            ) // 255
            rgb = rgb.astype(np.uint8)
        return np.ascontiguousarray(rgb).transpose(1, 0, 2)


def _level_data(
    slide: _Slide, level: int, lazy: tx.Optional[bool]
) -> ArrayProtocol:
    """Return the data of a level, as a [`_SlideArray`][] or a dask array.

    The data is a dask array of whole tiles if `lazy` is true, or if `lazy` is
    `None` and dask is the array backend. Otherwise, and whenever dask is not
    installed, the data is a [`_SlideArray`][].
    """
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
    """Return the pixel-to-physical transformation of a level.

    The full-resolution pixel size is multiplied by the downsampling factor `f`
    of the level, and pixel `i` is centred on the full-resolution coordinate
    `f * i + (f - 1) / 2`.
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
    """Base class of every image read with OpenSlide.

    Each vendor format derives from this class, so that the hint `"openslide"`
    selects every vendor and `"openslide.<vendor>"` selects one. See
    [OpenSlide](https://openslide.org).
    """

    HINTS = ("openslide",)

    VENDOR: tx.ClassVar[tx.Optional[str]] = None
    """The vendor name, as `OpenSlide.detect_format` reports it."""

    SCORE: tx.ClassVar[float] = Confidence.CERTAIN
    """The confidence for a file that OpenSlide attributes to the vendor."""


class _OpenSlideMixin(OpenSlideFormat):
    """Methods that sniff, open and describe slides, shared by single-scale and
    multiscale images.
    """

    @classmethod
    def _score(cls, levels: int, level: tx.Optional[int]) -> float:
        """Return the score for a slide of this vendor, given its number of
        levels and the requested level.
        """
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
        """Return the confidence that a local file is a slide of this vendor,
        as OpenSlide detects it.
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
        """Return the confidence that an open file is a slide of this vendor.

        OpenSlide sniffs the file through the name of the local file that it
        was opened from. A stream without such a name scores `NO`, so it is
        read only by hint.
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
        """Return `NO`, because content in memory is read only by hint."""
        if error:
            if error is True:
                error = SnifferContentError
            raise error("OpenSlide only sniffs local files.")
        return Confidence.NO

    @classmethod
    def _suffix(cls, name: tx.Optional[str]) -> str:
        """Return the extension for a temporary copy, since OpenSlide detects
        some formats by their suffix.
        """
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
        """Read a slide from a file.

        A local file is opened in place, and a remote file is first copied to a
        temporary file. The options are those of `from_source`.
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
        """Read a slide from an open file.

        A local file is opened by name, and other files are copied to a
        temporary file. The options are those of `from_source`.
        """
        local = _local_path(file)
        if local is not None:
            return cls.from_source(_Slide(local), **kwargs)
        with preserve_position(file):
            content = file.read()
        slide = _Slide.from_content(content, cls._suffix(_to_filename(file)))
        return cls.from_source(slide, **kwargs)

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Read a slide from bytes, through a temporary copy.

        The options are those of `from_source`.
        """
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
        """Read an associated image of the slide.

        Associated images are the label, macro, thumbnail and other images
        listed in `associated_images`. They are returned as F-ordered
        `(x, y, c)` RGB `uint8` arrays, composited onto white.

        Raises
        ------
        KeyError
            If there is no associated image of that name.
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


# The metadata attributes below are shared by single-scale and multiscale
# images.
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
        "pixels outside the scanned area. It is white if the slide records "
        "none."
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
    _OpenSlideMixin, BinaryFileReader, FileBasedImage, SingleScaleImage
):
    """A single level of a whole-slide image, read with OpenSlide.

    This class is the base of the single-scale image of every vendor. Its data
    is an F-ordered `(x, y, c)` RGB `uint8` array that reads only the indexed
    region, or a dask array of whole tiles. Its transformation scales pixels to
    micrometres when the slide records `openslide.mpp-x` and `openslide.mpp-y`,
    and is otherwise the identity, with no unit.
    """

    vendor: _Vendor = None
    properties: _Properties = None
    level: tx.Annotated[
        tx.Optional[int],
        tx.Doc("The level that was read, where 0 is the full resolution."),
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
        """Return the vendor score for a requested level or a single-level
        slide, and 0.8 times that score for a pyramid, which yields to the
        multiscale class.
        """
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
        """Read one level of a slide.

        Parameters
        ----------
        slide : _Slide
            The slide.
        level : int, optional
            The level, 0 (the full resolution) by default. A negative level
            counts from the end.
        pixel_size : float or Sequence[float] or Mapping[str, float], optional
            The full-resolution pixel size, overriding the one of the slide.
        unit : str or Unit, optional
            The unit of `pixel_size`, or the unit to convert micrometres to.
        lazy : bool, optional
            Whether the data is a dask array of whole tiles. By default, it is
            one only when dask is the array backend.

        Raises
        ------
        ParserContentError
            If OpenSlide cannot read the file, or the slide is of another
            vendor.
        IndexError
            If the slide has no such level.
        TypeError
            If an unknown option is given.
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
    _OpenSlideMixin, BinaryFileReader, FileBasedImage, MultiScaleImage
):
    """A whole-slide image read with OpenSlide, as a pyramid.

    This class is the base of the multiscale image of every vendor. Its levels
    are single-scale images of the same vendor, finest first, which all cover
    the whole slide and map to the same `"physical"` system, as in TIFF and
    OME-Zarr pyramids. The pyramid therefore has no transformation of its own.
    """

    vendor: _Vendor = None
    properties: _Properties = None
    n_levels: _NLevels = None
    level_downsamples: _Downsamples = None
    background_color: _Background = None
    bounds: _Bounds = None
    associated_images: _Associated = None

    PRIORITY: tx.ClassVar[int] = FileBasedImage.PRIORITY + 1
    """Content in memory is not sniffed, so when it is read by hint, the
    multiscale and single-scale classes of a vendor would tie. The higher
    priority makes the multiscale class be tried first. That class declines a
    `level` option, which leaves such a request to the single-scale class.
    """

    @classmethod
    def _score(cls, levels: int, level: tx.Optional[int]) -> float:
        """Return the vendor score when no level is requested and the slide is
        a pyramid, and `NO` otherwise.
        """
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
        """Read every level of a slide.

        The options are those of [`OpenSlideImage.from_source`][], except
        `level`.
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
