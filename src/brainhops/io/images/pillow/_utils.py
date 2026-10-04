"""
Decoding and encoding raster images with Pillow.

This is the Pillow *backend*: it turns a file into a C-ordered `numpy`
array plus a description of its axes and its resolution, and back. It
knows nothing of coordinate systems or transformations, which are the
business of the shared raster conventions in
`brainhops.io.images.base._utils_raster`. Keeping the two apart lets any
raster reader -- the Pillow image reader, or a TIFF reader that falls back on
Pillow when tifffile is not installed -- decode with Pillow and still
follow the same conventions.

!!! warning "Decompression bombs"
    Pillow refuses to decode an image of more than twice
    `PIL.Image.MAX_IMAGE_PIXELS` pixels (about 179 million pixels by
    default), raising `PIL.Image.DecompressionBombError`, and warns above
    `MAX_IMAGE_PIXELS`. This guards against small files that decode into
    huge arrays. To read a larger image you trust, raise the limit
    yourself: `PIL.Image.MAX_IMAGE_PIXELS = None`.

!!! note "No lazy access"
    Pillow decodes a whole frame at once, so the array is read in full when
    the image is opened.
"""

__all__ = [
    "PillowRaster",
    "EXTENSIONS",
    "SNIFF_FORMATS",
    "DPI_FORMATS",
    "sniff_pillow",
    "read_pillow",
    "pillow_to_array",
    "pillow_dpi",
    "array_to_pillow",
    "encode_pillow",
    "can_write",
    "format_for_name",
]

# stdlib
import warnings
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic

# internals
from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops.io.base.parsers import ParserContentError, WriterError

# Pillow is imported where it is used, rather than here: this module is
# imported to sniff any file, and `sniff_pillow` declines a file that
# starts with none of the magic numbers of `SNIFF_FORMATS` without it.
if tx.TYPE_CHECKING:
    from PIL import Image

EXTENSIONS: tx.Tuple[str, ...] = (
    ".png",
    ".apng",
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".mpo",
    ".bmp",
    ".dib",
    ".gif",
    ".webp",
    ".ppm",
    ".pgm",
    ".pbm",
    ".pnm",
    ".pfm",
    ".jp2",
    ".j2k",
    ".jpx",
    ".jpf",
    ".j2c",
    ".tga",
    ".pcx",
    ".qoi",
)
"""
The file extensions the Pillow reader claims.

The list is curated rather than taken from Pillow's registry, which also
lists formats Pillow only identifies (HDF5, GRIB, FITS) or that other
readers handle better. TIFF is left out on purpose: TIFF files are claimed
by the dedicated TIFF reader, and Pillow reads one only as a fallback.
"""

SNIFF_FORMATS: tx.Tuple[str, ...] = (
    "PNG",
    "JPEG",
    "BMP",
    "GIF",
    "WEBP",
    "PPM",
    "JPEG2000",
    "QOI",
    "TIFF",
)
"""
The Pillow formats recognized from the content of a file.

Each starts with a magic number. Formats that have none, or a weak one
(TGA, PCX, DIB), are recognized from the file extension alone, so that
arbitrary binary data is never mistaken for them.
"""

# How each of `SNIFF_FORMATS` starts, as Pillow's plugins recognize it
# (their `_accept`), so that a file that starts otherwise is declined
# without importing Pillow.
_TIFF_PREFIXES = (
    b"MM\x00\x2a",
    b"II\x2a\x00",
    b"MM\x2a\x00",
    b"II\x00\x2a",
    b"MM\x00\x2b",
    b"II\x2b\x00",
)
_MAGIC_NUMBERS: tx.Dict[str, tx.Callable[[bytes], bool]] = {
    "PNG": lambda h: h.startswith(b"\x89PNG\r\n\x1a\n"),
    "JPEG": lambda h: h.startswith(b"\xff\xd8\xff"),
    "BMP": lambda h: h.startswith(b"BM"),
    "GIF": lambda h: h.startswith((b"GIF87a", b"GIF89a")),
    "WEBP": lambda h: h.startswith(b"RIFF") and h[8:12] == b"WEBP",
    "PPM": lambda h: len(h) >= 2 and h[:1] == b"P" and h[1] in b"0123456fy",
    "JPEG2000": lambda h: h.startswith(
        (b"\xff\x4f\xff\x51", b"\x00\x00\x00\x0cjP  \r\n\x87\n")
    ),
    "QOI": lambda h: h.startswith(b"qoif"),
    "TIFF": lambda h: h.startswith(_TIFF_PREFIXES),
}

DPI_FORMATS: tx.FrozenSet[str] = frozenset({"PNG", "JPEG", "BMP", "TIFF"})
"""The Pillow formats that can store a resolution in dots per inch."""

# The modes that store an index per pixel into a colour table.
_PALETTE_MODES = ("P", "PA")


class PillowRaster(Magic, frozen=True, eq=False):
    """
    One decoded frame of a raster file.

    Equality is identity: comparing two frames would compare their
    arrays element-wise.
    """

    array: np.ndarray
    """The pixels, C-ordered: `(rows, columns)` or `(rows, columns,
    samples)`."""

    axes: str
    """The storage axes of `array`: `"YX"` or `"YXS"`."""

    format: tx.Optional[str]
    """Pillow's name for the file format, such as `"PNG"`."""

    mode: str
    """Pillow's mode of the frame as stored, before any conversion, such as
    `"P"`, `"RGB"` or `"I;16"`."""

    info: tx.Dict[str, tx.Any]
    """Pillow's `info` dictionary: the format-specific metadata."""

    frame: int
    """The index of the decoded frame."""

    n_frames: int
    """The number of frames in the file."""

    dpi: tx.Optional[tx.Tuple[float, float]]
    """The resolution the file records, `(x, y)` in dots per inch, or
    `None`. It is reported as it is, placeholders included."""


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _open(
    file: tx.Any, formats: tx.Optional[tx.Sequence[str]] = None
) -> "Image.Image":
    """Open an image with Pillow, reporting a file it cannot identify as
    a parser error."""
    from PIL import Image, UnidentifiedImageError

    if isinstance(file, (str, path.PathLike)):
        file = str(file)
    try:
        return Image.open(file, formats=formats)
    except Image.DecompressionBombError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        raise ParserContentError(
            f"Pillow cannot read this file as an image: {e}"
        ) from e


def sniff_pillow(
    file: tx.Any, formats: tx.Optional[tx.Sequence[str]] = SNIFF_FORMATS
) -> tx.Optional[str]:
    """
    Pillow's name for the format of a file, if it recognizes one.

    Only the header is read. A file object is left where it was.

    Parameters
    ----------
    file : path or file object
        The file to identify.
    formats : Sequence[str], optional
        The Pillow formats to consider. By default, those recognizable
        from their content
        (`SNIFF_FORMATS`).

    Returns
    -------
    str | None
        The format name, such as `"PNG"`, or `None`.
    """

    if formats is not None:
        # Narrow the formats down to those whose magic number the file
        # starts with, before importing Pillow: most files sniffed here
        # are not raster images at all.
        head = _head(file)
        if head is not None:
            formats = [
                fmt
                for fmt in formats
                if fmt not in _MAGIC_NUMBERS or _MAGIC_NUMBERS[fmt](head)
            ]
            if not formats:
                return None

    from PIL import Image

    if formats is not None:
        Image.init()
        formats = [fmt for fmt in formats if fmt in Image.OPEN]

    def identify(f: tx.Any) -> tx.Optional[str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with Image.open(f, formats=formats) as im:
                    return im.format
            except Image.DecompressionBombError:
                # It *is* an image, just a large one: say so, and let the
                # reader report the limit.
                return "unknown"
            except Exception:
                return None

    if hasattr(file, "read"):
        with preserve_position(file):
            return identify(file)
    return identify(str(file))


def _head(file: tx.Any) -> tx.Optional[bytes]:
    """The first bytes of a file, or `None` if they cannot be read."""
    try:
        if hasattr(file, "read"):
            with preserve_position(file):
                return bytes(file.read(16))
        with open(str(file), "rb") as f:
            return f.read(16)
    except Exception:
        return None


def pillow_to_array(
    im: "Image.Image", palette: bool = True
) -> tx.Tuple[np.ndarray, str]:
    """
    The pixels of a decoded Pillow image, as a C-ordered array.

    | Mode                     | Array                                    |
    | ------------------------ | ---------------------------------------- |
    | `1`                      | `bool`, `(rows, columns)`                |
    | `L`                      | `uint8`, `(rows, columns)`               |
    | `LA`, `La`               | `uint8`, `(rows, columns, 2)`            |
    | `RGB`, `YCbCr`, `LAB`,   | `uint8`, `(rows, columns, 3)`            |
    | `HSV`                    |                                          |
    | `RGBA`, `RGBa`, `RGBX`,  | `uint8`, `(rows, columns, 4)`            |
    | `CMYK`                   |                                          |
    | `I;16`, `I;16B`, ...     | `uint16`, `(rows, columns)`              |
    | `I`                      | `int32`, `(rows, columns)`               |
    | `F`                      | `float32`, `(rows, columns)`             |
    | `P`, `PA`                | see `palette`                            |

    The premultiplied modes `La` and `RGBa` are converted to `LA` and
    `RGBA`. The array is in native byte order, and is writable.

    Parameters
    ----------
    im : PIL.Image.Image
        The decoded image.
    palette : bool
        A palette image (`P` or `PA`) stores an index per pixel and a
        colour table. When `True` (the default), the colours are looked up:
        the image becomes `RGB`, or `RGBA` if it has transparency. When
        `False`, the indices are returned, as `uint8`.

    Returns
    -------
    array : np.ndarray
        The pixels.
    axes : str
        `"YX"`, or `"YXS"` when there are several samples per pixel.
    """
    mode = im.mode
    if mode in _PALETTE_MODES:
        if palette:
            alpha = mode == "PA" or "transparency" in im.info
            im = im.convert("RGBA" if alpha else "RGB")
        elif mode == "PA":
            im = im.getchannel(0)
    elif mode == "La":
        im = im.convert("LA")
    elif mode == "RGBa":
        im = im.convert("RGBA")
    elif mode.startswith("BGR;"):
        im = im.convert("RGB")
    try:
        array = np.array(im)
    except Exception as e:
        raise ParserContentError(
            f"Pillow cannot convert an image of mode {mode!r} to an array: {e}"
        ) from e
    if array.dtype.byteorder not in ("=", "|"):
        array = array.astype(array.dtype.newbyteorder("="))
    if array.ndim == 2:
        return array, "YX"
    if array.ndim == 3:
        return array, "YXS"
    raise ParserContentError(
        f"Pillow decoded an array of {array.ndim} dimensions, which is not "
        f"a raster image."
    )


def pillow_dpi(
    info: tx.Mapping[str, tx.Any],
) -> tx.Optional[tx.Tuple[float, float]]:
    """
    The resolution a file records, `(x, y)` in dots per inch, or `None`.

    Pillow reports it as `info["dpi"]` for PNG (`pHYs`), JPEG (JFIF
    density, or EXIF), BMP and TIFF, converted from dots per centimetre or
    per metre when that is what the file stores. A file that records only
    an aspect ratio (a PNG `pHYs` or JFIF density with no unit) has none.
    """
    dpi = info.get("dpi")
    if dpi is None:
        return None
    try:
        values = [float(v) for v in np.ravel(np.asarray(dpi, dtype=float))]
    except (TypeError, ValueError):
        return None
    if len(values) == 1:
        values = values * 2
    if len(values) != 2:
        return None
    return values[0], values[1]


def read_pillow(
    file: tx.Any,
    frame: int = 0,
    palette: bool = True,
    formats: tx.Optional[tx.Sequence[str]] = None,
) -> PillowRaster:
    """
    Decode one frame of a raster file with Pillow.

    Parameters
    ----------
    file : path or file object
        The file to read. A file object is not closed.
    frame : int
        The frame to read from a multi-frame file (an animated GIF, PNG or
        WebP, a multi-page TIFF, ...). Negative values count from the end.
        Pillow composites the frames of an animated GIF: frame `i` is what
        is displayed at step `i`, and the frames after the first are
        decoded as `RGB` or `RGBA`.
    palette : bool
        Look up the colours of a palette image (see
        `pillow_to_array`).
    formats : Sequence[str], optional
        The Pillow formats to try. By default, all of them.

    Returns
    -------
    PillowRaster
        The pixels and what the file says about them.

    Raises
    ------
    ParserContentError
        If Pillow cannot read the file.
    IndexError
        If the file has no frame `frame`.
    PIL.Image.DecompressionBombError
        If the image is larger than Pillow's safety limit.
    """
    start = None
    if hasattr(file, "tell"):
        try:
            start = file.tell()
        except Exception:
            start = None
    with _open(file, formats) as im:
        n_frames = int(getattr(im, "n_frames", 1) or 1)
        index = int(frame)
        if index < 0:
            index += n_frames
        if not 0 <= index < n_frames:
            raise IndexError(
                f"This file has {n_frames} frame(s), so it has no frame "
                f"{frame}."
            )
        from PIL import Image

        try:
            if index:
                im.seek(index)
            im.load()
        except Image.DecompressionBombError:
            raise
        except Exception as e:
            raise ParserContentError(
                f"Pillow cannot decode this image: {e}"
            ) from e
        mode = im.mode
        info = dict(im.info)
        array, axes = pillow_to_array(im, palette=palette)
        fmt = im.format
    if fmt == "PNG" and "sCAL" not in info:
        scal = _png_scal(file, start)
        if scal is not None:
            info["sCAL"] = scal
    return PillowRaster(
        array=array,
        axes=axes,
        format=fmt,
        mode=mode,
        info=info,
        frame=index,
        n_frames=n_frames,
        dpi=pillow_dpi(info),
    )


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_scal(
    file: tx.Any, start: tx.Optional[int] = None
) -> tx.Optional[tx.Tuple[int, float, float]]:
    """
    The physical scale a PNG file records in its `sCAL` chunk, or `None`.

    Pillow skips this chunk, so it is read here. It is returned as
    `(unit, width, height)`: the size of a pixel along x and y, in metres
    when `unit` is 1 and in radians when it is 2.
    """

    def scan(f: tx.Any) -> tx.Optional[tx.Tuple[int, float, float]]:
        if f.read(8) != _PNG_SIGNATURE:
            return None
        while True:
            head = f.read(8)
            if len(head) < 8:
                return None
            length = int.from_bytes(head[:4], "big")
            kind = head[4:]
            if kind in (b"IDAT", b"IEND"):
                return None  # sCAL must come before the image data
            if kind != b"sCAL":
                f.seek(length + 4, 1)  # data and CRC
                continue
            data = f.read(length)
            try:
                width, height = data[1:].split(b"\0")[:2]
                return int(data[0]), float(width), float(height)
            except (ValueError, IndexError):
                return None

    try:
        if hasattr(file, "read"):
            if start is None:
                return None
            with preserve_position(file):
                file.seek(start)
                return scan(file)
        with open(str(file), "rb") as f:
            return scan(f)
    except Exception:
        return None


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _numpy(array: tx.Any) -> np.ndarray:
    """Materialize any array (numpy, dask, cupy) as a numpy array."""
    if hasattr(array, "get") and not isinstance(array, np.ndarray):
        array = array.get()  # cupy
    if hasattr(array, "compute") and not isinstance(array, np.ndarray):
        array = array.compute()  # dask
    return np.asarray(array)


def array_to_pillow(array: tx.Any) -> "Image.Image":
    """
    Build a Pillow image from a C-ordered array of pixels.

    | Array                                      | Mode                   |
    | ------------------------------------------ | ---------------------- |
    | `bool`, `(rows, columns)`                  | `1`                    |
    | `uint8`, `(rows, columns)`                 | `L`                    |
    | `uint8`, `(rows, columns, 2 / 3 / 4)`      | `LA` / `RGB` / `RGBA`  |
    | `uint16`, `(rows, columns)`                | `I;16`                 |
    | `int32`, `(rows, columns)`                 | `I`                    |
    | `float32`, `(rows, columns)`               | `F`                    |

    A trailing axis of one sample is dropped. Whether a file format can
    store the mode is up to that format (JPEG cannot store `I;16`, say).

    Raises
    ------
    WriterError
        If no Pillow mode holds the array: another data type (convert the
        data first, e.g. with `numpy.clip` and `astype`), more than four
        samples per pixel, or several samples of anything but `uint8`.
    """
    array = _numpy(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim not in (2, 3):
        raise WriterError(
            f"A raster image is stored as rows and columns, with an "
            f"optional axis of samples, but this array has {array.ndim} "
            f"dimensions."
        )
    dtype = array.dtype
    if array.ndim == 3:
        samples = array.shape[-1]
        if dtype != np.uint8:
            raise WriterError(
                f"Pillow stores several samples per pixel only as 8-bit "
                f"unsigned integers (uint8), so an image of {samples} "
                f"samples of type {dtype} cannot be written. Convert it to "
                f"uint8, or write each channel separately."
            )
        if samples not in (2, 3, 4):
            raise WriterError(
                f"Pillow stores 1 (grey), 2 (grey + alpha), 3 (RGB) or 4 "
                f"(RGBA) samples per pixel, not {samples}."
            )
    elif dtype.kind == "b":
        pass
    elif dtype.kind == "u" and dtype.itemsize in (1, 2):
        pass
    elif dtype.kind == "i" and dtype.itemsize == 4:
        pass
    elif dtype.kind == "f" and dtype.itemsize == 4:
        pass
    else:
        raise WriterError(
            f"Pillow cannot store pixels of type {dtype}. It stores bool, "
            f"uint8, uint16, int32 and float32 (and the formats that can "
            f"hold each are fewer still: PNG holds bool, uint8 and uint16, "
            f"and JPEG only uint8). Convert the data first."
        )
    from PIL import Image

    array = np.ascontiguousarray(array)
    if array.dtype.byteorder not in ("=", "|"):
        array = array.astype(array.dtype.newbyteorder("="))
    try:
        return Image.fromarray(array)
    except Exception as e:
        raise WriterError(
            f"Pillow cannot build an image from an array of type "
            f"{array.dtype} and shape {array.shape}: {e}"
        ) from e


def format_for_name(name: tx.Any) -> tx.Optional[str]:
    """Pillow's name for the format a file name's extension calls for,
    or `None`."""
    if name is None:
        return None
    from PIL import Image

    name = str(name).lower()
    Image.init()
    extensions = Image.registered_extensions()
    best = None
    for extension, fmt in extensions.items():
        if name.endswith(extension) and (
            best is None or len(extension) > len(best[0])
        ):
            best = (extension, fmt)
    return best[1] if best else None


def can_write(format: str) -> bool:
    """Whether Pillow can write files in a format (by Pillow's name)."""
    from PIL import Image

    Image.init()
    return str(format).upper() in Image.SAVE


def encode_pillow(
    im: "Image.Image",
    format: str,
    dpi: tx.Optional[tx.Tuple[float, float]] = None,
    **options,
) -> bytes:
    """
    Encode a Pillow image in a file format.

    The image is encoded in memory, so a failure leaves no partial file.

    Parameters
    ----------
    im : PIL.Image.Image
        The image.
    format : str
        Pillow's name for the format, such as `"PNG"`.
    dpi : (float, float), optional
        The resolution to record, `(x, y)` in dots per inch. It is passed
        on only to the formats that can store one
        (`DPI_FORMATS`).
    **options
        Passed on to Pillow's `Image.save`, e.g. `quality=95` for JPEG.

    Raises
    ------
    WriterError
        If Pillow does not know the format, or the format cannot store the
        image (JPEG cannot store 16-bit pixels, say).
    """
    format = str(format).upper()
    if not can_write(format):
        raise WriterError(f"Pillow cannot write files in the {format} format.")
    if dpi is not None and format in DPI_FORMATS:
        options["dpi"] = (float(dpi[0]), float(dpi[1]))
    buffer = BytesIO()
    try:
        im.save(buffer, format=format, **options)
    except (OSError, KeyError, ValueError, TypeError) as e:
        raise WriterError(
            f"The {format} format cannot store this image (Pillow mode "
            f"{im.mode!r}): {e}"
        ) from e
    return buffer.getvalue()
