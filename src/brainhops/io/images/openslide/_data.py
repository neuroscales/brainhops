"""Access to the slides that OpenSlide reads, and the lazy array of a level.

OpenSlide opens a slide by the name of a local file, and it reads the pixels
of a level region by region. This module finds the local file of a path or
of a stream, copies content into a temporary file when there is none, and
detects the vendor of a file. It also defines the lazy array that a level of
a slide holds in `rawdata`, which reads only the region that it is
indexed with.
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

# internals
from brainhops._core import path
from brainhops._core.dependencies import openslide
from brainhops.io.base._utils_files import is_local
from brainhops.io.base.parsers import ParserContentError

# this format
from ._raw import OpenSlideRaw

# Each dask chunk groups whole tiles until it is at least this many pixels
# wide along each axis, or until it covers the whole axis.
_MIN_CHUNK = 1024

# The size of a tile along an axis for which the slide records none.
_DEFAULT_TILE = 256


def require_openslide() -> tx.Any:
    """Return the module `openslide`.

    Raises
    ------
    ImportError
        If openslide-python or the OpenSlide library is not installed.
    """
    if openslide is None:
        raise ImportError(
            "Reading whole-slide images needs openslide-python and the "
            "OpenSlide library: pip install brainhops[openslide]"
        )
    return openslide


# ----------------------------------------------------------------------
#   FILES
# ----------------------------------------------------------------------


def local_path(file: tx.Any) -> tx.Optional[str]:
    """Return the local file name of a path or of an open file.

    OpenSlide opens slides by name only. A path names a local file when its
    protocol is local, and an open file does when its `name` attribute
    names an existing file.

    Parameters
    ----------
    file : str or PathLike or IO
        A path or an open file.

    Returns
    -------
    str or None
        The name of the local file, or `None` when OpenSlide cannot open
        the file by name.
    """
    if isinstance(file, (str, path.PathLike)):
        try:
            if not is_local(file):
                return None
        except Exception:
            return None
        local = str(path.Path(file))
    else:
        local = getattr(file, "name", None)
        if not isinstance(local, str):
            return None
    return local if os.path.isfile(local) else None


def detect(filename: str) -> tx.Tuple[str, int]:
    """Return the vendor and the number of levels of a local file.

    Every format class of the package sniffs the same file, so the result
    is cached. The modification time and the size of the file are part of
    the key of the cache, so that a file that changed is detected again.

    Parameters
    ----------
    filename : str
        The name of a local file.

    Returns
    -------
    tuple of (str, int)
        The vendor, as `OpenSlide.detect_format` names it, and the number
        of levels, or `("", 0)` when OpenSlide cannot read the file or is
        not installed.
    """
    if openslide is None:
        return "", 0
    stat = os.stat(filename)
    return _detect(os.path.abspath(filename), stat.st_mtime_ns, stat.st_size)


class _Slide:
    """A slide that OpenSlide reads, opened on first use.

    A slide is the local file that OpenSlide opens. The lazy arrays of the
    levels of the slide share one `_Slide`, which opens the file once and
    keeps it open. The file is closed, and a temporary copy is deleted,
    when the last object that uses the slide is garbage-collected. A record
    never holds a slide, so that metadata keeps no open file.
    """

    def __init__(
        self, filename: str, tempdir: tx.Optional[str] = None
    ) -> None:
        self.filename = filename
        """The name of the local file that OpenSlide opens."""
        self._state: tx.Dict[str, tx.Any] = {"tempdir": tempdir}
        weakref.finalize(self, _release, self._state)

    @classmethod
    def from_content(
        cls, content: bytes, name: tx.Optional[str] = None
    ) -> "_Slide":
        """Return a slide whose content is written to a temporary file.

        OpenSlide opens slides by name only, so content without a local file
        is first written to a file in a new temporary directory. The file
        keeps the extension of `name`, because OpenSlide recognises a MIRAX
        slide by its extension. A slide made of several files, such as a
        MIRAX, Hamamatsu VMS or DICOM slide, cannot be read this way, since
        the directory holds only the one file.

        Parameters
        ----------
        content : bytes
            The content of the file.
        name : str, optional
            The name of the original file, whose extension is kept.

        Returns
        -------
        _Slide
            The slide, which deletes the temporary directory when it is
            garbage-collected.
        """
        tempdir = tempfile.mkdtemp(prefix="brainhops-openslide-")
        filename = os.path.join(tempdir, "slide" + _extension(name))
        with open(filename, "wb") as f:
            f.write(content)
        return cls(filename, tempdir)

    @property
    def handle(self) -> tx.Any:
        """The open `openslide.OpenSlide` object of the slide.

        Raises
        ------
        ParserContentError
            If OpenSlide cannot read the file.
        """
        handle = self._state.get("handle")
        if handle is None:
            handle = open_handle(self.filename)
            self._state["handle"] = handle
        return handle


def open_handle(filename: str) -> tx.Any:
    """Open a local file with OpenSlide.

    Parameters
    ----------
    filename : str
        The name of a local file.

    Returns
    -------
    openslide.OpenSlide
        The open slide, which the caller closes.

    Raises
    ------
    ImportError
        If OpenSlide is not installed.
    ParserContentError
        If OpenSlide cannot read the file.
    """
    module = require_openslide()
    try:
        return module.OpenSlide(filename)
    except Exception as e:
        raise ParserContentError(
            f"OpenSlide cannot read this file: {e}"
        ) from e


def read_associated(slide: _Slide, name: str) -> np.ndarray:
    """Read an associated image of a slide, composited onto white.

    Parameters
    ----------
    slide : _Slide
        The slide.
    name : str
        The name of the associated image, such as `"label"`.

    Returns
    -------
    np.ndarray
        The image, as an F-ordered `(x, y, c)` RGB `uint8` array.

    Raises
    ------
    KeyError
        If the slide has no associated image of that name.
    """
    image = slide.handle.associated_images[name]
    rgba = np.asarray(image.convert("RGBA"))
    a = rgba[..., 3:4].astype(np.uint16)
    rgb = rgba[..., :3].astype(np.uint16) * a + 255 * (255 - a) + 127
    rgb = (rgb // 255).astype(np.uint8)
    return np.ascontiguousarray(rgb).transpose(1, 0, 2)


# ----------------------------------------------------------------------
#   LAZY LEVEL
# ----------------------------------------------------------------------


class _SlideArray:
    """A lazy array of one level of a slide, which reads only what is indexed.

    The array holds the slide, the record of the slide and the index of the
    level in the record, and it knows its shape, its type and its background
    colour from the record, without opening the file. It presents the level
    as an F-ordered `(x, y, c)` RGB `uint8` array whose row 0 is the top of
    the slide. Indexing it reads the region that the index covers with
    `read_region`, and `numpy.asarray` reads the whole level. OpenSlide
    decodes RGBA pixels whose alpha channel only marks the pixels outside
    the scanned area, and these pixels are composited onto the background
    colour of the slide.

    The reader option `lazy`, which the array keeps, decides whether the
    data of the level is this array itself or a dask array of whole tiles
    over it, as [`view`][] describes.
    """

    def __init__(
        self,
        slide: _Slide,
        record: OpenSlideRaw,
        level: int,
        lazy: tx.Optional[bool] = None,
    ) -> None:
        self.slide = slide
        """The slide whose level the array reads."""
        self.record = record
        """The record of the slide."""
        self.level = int(level)
        """The index of the level in the record."""
        self.lazy = lazy
        """Whether the data is a dask array, or `None` for the array backend
        to decide."""

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the level: width, height and three samples."""
        return tuple(int(n) for n in self.record.levels[self.level].shape)

    @property
    def dtype(self) -> np.dtype:
        """The type of the pixels."""
        return np.dtype(self.record.levels[self.level].dtype)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the level."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of values of the level."""
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """The number of bytes of the level once it is read."""
        return self.size * self.dtype.itemsize

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"_SlideArray(level={self.level}, shape={self.shape}, uint8)"

    def view(self) -> tx.Any:
        """Return the data of the level, without reading any pixel.

        The data is a dask array whose chunks group whole tiles when `lazy`
        is true, or when `lazy` is `None` and dask is the array backend.
        Otherwise, and whenever dask is not installed, the data is this
        array, which reads the region that it is indexed with.

        Returns
        -------
        _SlideArray or dask.array.Array
            The data of the level.
        """
        lazy = self.lazy
        try:
            import dask.array as da
        except ImportError:
            return self
        if lazy is None:
            from brainhops.backends import get_array_backend

            lazy = get_array_backend() is da
        if not lazy:
            return self
        return da.from_array(
            self,
            chunks=_chunks(self.record, self.level, self.shape),
            asarray=False,
            fancy=False,
            lock=False,
            meta=np.empty((0, 0, 0), dtype=self.dtype),
        )

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        array = self._read(0, 0, self.shape[0], self.shape[1])
        return array if dtype is None else array.astype(dtype)

    def __getitem__(self, key: tx.Any) -> np.ndarray:
        """Read the region that an index covers.

        Integers and slices behave as in NumPy, while integer and boolean
        arrays index each axis independently. An empty selection, such as
        the one that Dask reads to learn the type of an array, is answered
        without reading the slide.
        """
        selected = np.broadcast_to(np.empty((), self.dtype), self.shape)[key]
        if selected.size == 0:
            return np.empty(selected.shape, self.dtype)
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
        downsample = self.record.levels[self.level].downsample
        location = (int(round(x * downsample)), int(round(y * downsample)))
        image = self.slide.handle.read_region(
            location, self.level, (int(width), int(height))
        )
        rgba = np.asarray(image)
        alpha = rgba[..., 3:4]
        rgb = rgba[..., :3]
        if not (alpha == 255).all():
            # read_region returns RGBA that is not premultiplied by alpha
            background = _hex_colour(self.record.background_color)
            a = alpha.astype(np.uint16)
            rgb = (
                rgb.astype(np.uint16) * a + background * (255 - a) + 127
            ) // 255
            rgb = rgb.astype(np.uint8)
        return np.ascontiguousarray(rgb).transpose(1, 0, 2)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _detect(filename: str, mtime: int, size: int) -> tx.Tuple[str, int]:
    """Return the vendor and the number of levels of a file, or `("", 0)`.

    The modification time and the size only serve as the key of the cache.
    """
    del mtime, size
    try:
        vendor = openslide.OpenSlide.detect_format(filename)
        if not vendor:
            return "", 0
        with openslide.OpenSlide(filename) as slide:
            return str(vendor), int(slide.level_count)
    except Exception:
        return "", 0


def _release(state: tx.Dict[str, tx.Any]) -> None:
    """Close the slide of a `_Slide` and delete its temporary directory."""
    handle = state.pop("handle", None)
    if handle is not None:
        handle.close()
    tempdir = state.pop("tempdir", None)
    if tempdir is not None:
        shutil.rmtree(tempdir, ignore_errors=True)


def _extension(name: tx.Optional[str]) -> str:
    """Return the extension of a file name, or an empty string."""
    if not name:
        return ""
    return os.path.splitext(str(name))[1]


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


def _chunks(
    record: OpenSlideRaw, level: int, shape: tx.Tuple[int, ...]
) -> tx.Tuple[int, int, int]:
    """Return the dask chunks of a level, which group whole tiles.

    The size of a tile is read from the properties
    `openslide.level[<i>].tile-width` and `tile-height` of the record, and
    is 256 pixels along an axis for which the slide records none.
    """
    chunks = []
    for name, n in (("width", shape[0]), ("height", shape[1])):
        tile = record.properties.get(f"openslide.level[{level}].tile-{name}")
        try:
            tile = max(int(tile), 1)
        except (TypeError, ValueError):
            tile = _DEFAULT_TILE
        chunks.append(min(n, tile * max(1, math.ceil(_MIN_CHUNK / tile))))
    return chunks[0], chunks[1], shape[2]
