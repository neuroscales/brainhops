"""Decoding and encoding raster images with Pillow.

This module decodes a file into a C-ordered NumPy array, a description of its
axes and a resolution, and it encodes such an array back into a file. It builds
no coordinate systems or transformations. These are left to the shared raster
conventions of `brainhops.io.images.base._utils_raster`, so that every raster
reader that decodes with Pillow (the Pillow reader, and the TIFF reader
without tifffile) follows the same conventions.

!!! warning "Decompression bombs"
    Pillow refuses images larger than twice `PIL.Image.MAX_IMAGE_PIXELS` (about
    179 million pixels) with a `DecompressionBombError`, and warns above the
    limit itself, to guard against small files that decode to huge images.
    Setting `PIL.Image.MAX_IMAGE_PIXELS = None` allows a trusted larger image.

!!! note "No lazy access"
    A whole frame is decoded and read into memory when the file is opened.
"""

__all__ = [
    "PillowHeader",
    "PillowRaster",
    "EXTENSIONS",
    "SNIFF_FORMATS",
    "DPI_FORMATS",
    "sniff_pillow",
    "read_pillow",
    "read_pillow_header",
    "pillow_layout",
    "pillow_to_array",
    "pillow_dpi",
    "array_to_pillow",
    "encode_pillow",
    "can_write",
    "format_for_name",
]

import warnings
from io import BytesIO

import numpy as np
import typing_extensions as tx
from bagof.magic import Magic
from PIL import Image, ImageMode, UnidentifiedImageError

from brainhops._core import path
from brainhops._core._internal_api import preserve_position
from brainhops.io.base.parsers import ParserContentError, WriterError

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
    ".tga",
    ".pcx",
    ".qoi",
)
"""The file extensions that the Pillow reader claims.

The list is curated rather than taken from Pillow, whose registry includes
identify-only formats (HDF5, GRIB, FITS) and formats that other readers handle
better. TIFF and JPEG 2000 are left to their dedicated readers.
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
"""The formats recognised from their content, by a magic number.

Formats with a weak magic number or none (TGA, PCX, DIB) are recognised only by
extension, so that arbitrary binary data is not mistaken for them.
"""

DPI_FORMATS: tx.FrozenSet[str] = frozenset({"PNG", "JPEG", "BMP", "TIFF"})
"""The formats that can store a resolution in dots per inch."""

# These modes store, for each pixel, an index into a colour table.
_PALETTE_MODES = ("P", "PA")


class PillowRaster(Magic, frozen=True, eq=False):
    """One decoded frame of a raster file.

    Equality is identity, since comparing frames would compare their arrays.
    """

    array: np.ndarray
    """The pixels, C-ordered `(rows, columns)` or `(rows, columns, samples)`.
    """

    axes: str
    """The storage axes of `array`, `"YX"` or `"YXS"`."""

    format: tx.Optional[str]
    """The Pillow name of the file format, such as `"PNG"`."""

    mode: str
    """The Pillow mode of the frame as stored, such as `"P"` or `"I;16"`."""

    info: tx.Dict[str, tx.Any]
    """The `info` dictionary of Pillow, with the format-specific metadata."""

    frame: int
    """The index of the decoded frame."""

    n_frames: int
    """The number of frames in the file."""

    dpi: tx.Optional[tx.Tuple[float, float]]
    """The resolution recorded by the file, in dots per inch along x and y.

    Placeholder resolutions, such as 72 dpi, are kept.
    """


class PillowHeader(Magic, frozen=True):
    """What the header of a raster file says about its first frame.

    A reader that must know the shape and the type of many files before it
    decodes any of them, such as the reader of image sequences, reads this
    header with [`read_pillow_header`][]. The header gives the array that
    [`read_pillow`][] would return for the first frame, without decoding
    the pixels.
    """

    format: tx.Optional[str]
    """The Pillow name of the file format, such as `"PNG"`."""

    mode: str
    """The Pillow mode of the first frame as stored, such as `"P"`."""

    info: tx.Dict[str, tx.Any]
    """The `info` dictionary of Pillow, with the format-specific metadata."""

    n_frames: int
    """The number of frames in the file."""

    shape: tx.Tuple[int, ...]
    """The shape of the decoded first frame, `(rows, columns[, samples])`."""

    dtype: str
    """The NumPy type of the decoded pixels, in native byte order."""

    axes: str
    """The storage axes of the decoded first frame, `"YX"` or `"YXS"`."""

    dpi: tx.Optional[tx.Tuple[float, float]]
    """The resolution recorded by the file, in dots per inch along x and y.

    Placeholder resolutions, such as 72 dpi, are kept.
    """


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def open_with_pillow(
    file: tx.Any, formats: tx.Optional[tx.Sequence[str]] = None
) -> "Image.Image":
    """Open a file with Pillow, raising a `ParserContentError` if it cannot be
    identified.
    """
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
    """Return the Pillow name of the format of a file, or `None`.

    Only the header is read, and the position of a file object is preserved.
    The formats considered default to [`SNIFF_FORMATS`][].
    """

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
                # An image beyond the size limit is still an image, and the
                # reader reports the limit when it decodes the image.
                return "unknown"
            except Exception:
                return None

    if hasattr(file, "read"):
        with preserve_position(file):
            return identify(file)
    return identify(str(file))


def pillow_to_array(
    im: "Image.Image", palette: bool = True
) -> tx.Tuple[np.ndarray, str]:
    """Return the pixels of a decoded Pillow image as a C-ordered array.

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

    The premultiplied modes `La` and `RGBa` become `LA` and `RGBA`, and the
    `BGR;*` modes become `RGB`. The array is native-endian and writable.

    Parameters
    ----------
    im : PIL.Image.Image
        The decoded image.
    palette : bool, optional
        Whether palette colours are looked up, giving RGB, or RGBA with
        transparency (the default), rather than returned as `uint8` indices.

    Returns
    -------
    array : numpy.ndarray
        The pixels.
    axes : str
        `"YX"` or `"YXS"`.
    """
    mode = im.mode
    target = _decoded_mode(mode, im.info, palette)
    if mode == "PA" and target == "L":
        im = im.getchannel(0)
    elif target != mode:
        im = im.convert(target)
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
    """Return the resolution recorded in `info["dpi"]`, along x and y, or
    `None`.

    Pillow reports the resolution for PNG, JPEG, BMP and TIFF. A file that
    records only an aspect ratio has no resolution.
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
    """Decode one frame of a raster file with Pillow.

    Parameters
    ----------
    file : str or PathLike or IO
        A path or a file object, which is not closed.
    frame : int, optional
        The index of the frame. A negative index counts from the end. Animated
        GIF frames are composited, so frame `i` is the picture displayed at
        step `i`.
    palette : bool, optional
        See [`pillow_to_array`][].
    formats : Sequence[str], optional
        The formats to consider, all by default.

    Returns
    -------
    PillowRaster
        The frame. The `info` of a PNG file also holds its `sCAL` chunk.

    Raises
    ------
    ParserContentError
        If Pillow cannot read or decode the file.
    IndexError
        If the file has no such frame.
    PIL.Image.DecompressionBombError
        If the image exceeds the limit of `PIL.Image.MAX_IMAGE_PIXELS`.
    """
    start = None
    if hasattr(file, "tell"):
        try:
            start = file.tell()
        except Exception:
            start = None
    with open_with_pillow(file, formats) as im:
        n_frames = int(getattr(im, "n_frames", 1) or 1)
        index = int(frame)
        if index < 0:
            index += n_frames
        if not 0 <= index < n_frames:
            raise IndexError(
                f"This file has {n_frames} frame(s), so it has no frame "
                f"{frame}."
            )
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
        scal = png_scal(file, start)
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


def pillow_layout(
    mode: str, info: tx.Mapping[str, tx.Any], palette: bool = True
) -> tx.Tuple[np.dtype, int]:
    """Return the type and the number of samples of a decoded frame.

    A frame of a given mode is decoded by [`pillow_to_array`][] into an
    array whose type and number of samples depend only on the mode, on
    whether the palette is looked up, and, for a palette image, on whether
    `info` records a transparent colour. This function computes them from
    the descriptor of the mode that Pillow publishes in `PIL.ImageMode`,
    so that the pixels need not be decoded. The descriptor gives the
    NumPy type string of a band (`typestr`) and the names of the bands
    (`bands`), as `src/PIL/ImageMode.py` of Pillow 10.4.0 defines them.
    Pillow builds the array interface of an image from the same two
    fields (`_conv_type_shape` in `src/PIL/Image.py`), and a test compares
    the result with decoded frames of each mode.

    Parameters
    ----------
    mode : str
        The Pillow mode of the frame as stored, such as `"P"` or `"I;16B"`.
    info : Mapping[str, Any]
        The `info` dictionary of the frame.
    palette : bool, default=True
        Whether palette colours are looked up, as in [`pillow_to_array`][].

    Returns
    -------
    dtype : numpy.dtype
        The type of the decoded pixels, in native byte order.
    samples : int
        The number of samples per pixel, where 1 means that the array has
        no axis of samples.

    Raises
    ------
    ParserContentError
        If Pillow does not describe the mode.
    """
    target = _decoded_mode(mode, info, palette)
    try:
        descriptor = ImageMode.getmode(target)
    except KeyError as e:
        raise ParserContentError(
            f"Pillow does not describe the mode {target!r}."
        ) from e
    dtype = np.dtype(descriptor.typestr)
    if dtype.byteorder not in ("=", "|"):
        dtype = dtype.newbyteorder("=")
    return dtype, len(descriptor.bands)


def read_pillow_header(file: tx.Any, palette: bool = True) -> PillowHeader:
    """Read the header of a raster file without decoding its pixels.

    Pillow reads only the header of a file when it opens it, so this
    function is much cheaper than [`read_pillow`][]. The shape and the type
    of the decoded first frame are computed with [`pillow_layout`][].

    Parameters
    ----------
    file : str or PathLike or IO
        A path or a file object, which is not closed.
    palette : bool, default=True
        Whether the decoded frame would look its palette colours up, as in
        [`pillow_to_array`][].

    Returns
    -------
    PillowHeader
        The header. The `info` of a PNG file also holds its `sCAL` chunk.

    Raises
    ------
    ParserContentError
        If Pillow cannot read the file, or does not describe its mode.
    """
    start = None
    if hasattr(file, "tell"):
        try:
            start = file.tell()
        except Exception:
            start = None
    with open_with_pillow(file) as im:
        n_frames = int(getattr(im, "n_frames", 1) or 1)
        mode = im.mode
        info = dict(im.info)
        width, height = im.size
        fmt = im.format
    if fmt == "PNG" and "sCAL" not in info:
        scal = png_scal(file, start)
        if scal is not None:
            info["sCAL"] = scal
    dtype, samples = pillow_layout(mode, info, palette)
    shape = (int(height), int(width))
    if samples > 1:
        shape += (samples,)
    return PillowHeader(
        format=fmt,
        mode=mode,
        info=info,
        n_frames=n_frames,
        shape=shape,
        dtype=str(dtype),
        axes="YXS" if samples > 1 else "YX",
        dpi=pillow_dpi(info),
    )


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_scal(
    file: tx.Any, start: tx.Optional[int] = None
) -> tx.Optional[tx.Tuple[int, float, float]]:
    """Return the PNG `sCAL` chunk, which Pillow skips, as
    `(unit, width, height)`, or `None`. The unit is 1 for metres and 2 for
    radians.
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
                return None  # sCAL precedes the image data
            if kind != b"sCAL":
                f.seek(length + 4, 1)  # skip the chunk data and its CRC
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
        with path.Path(file).open("rb") as f:
            return scan(f)
    except Exception:
        return None


def _decoded_mode(
    mode: str, info: tx.Mapping[str, tx.Any], palette: bool
) -> str:
    """Return the mode in which [`pillow_to_array`][] decodes a frame.

    A palette image becomes RGB, or RGBA if it has transparency, when its
    palette is looked up. Without the lookup, the indices of a `PA` image
    are its first channel, of mode `L`. The premultiplied modes `La` and
    `RGBa` become `LA` and `RGBA`, and the `BGR;*` modes become `RGB`.
    Every other mode is kept.
    """
    if mode in _PALETTE_MODES:
        if palette:
            alpha = mode == "PA" or "transparency" in info
            return "RGBA" if alpha else "RGB"
        return "L" if mode == "PA" else mode
    if mode == "La":
        return "LA"
    if mode == "RGBa":
        return "RGBA"
    if mode.startswith("BGR;"):
        return "RGB"
    return mode


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _numpy(array: tx.Any) -> np.ndarray:
    """Materialise a NumPy, dask or CuPy array as a NumPy array."""
    if hasattr(array, "get") and not isinstance(array, np.ndarray):
        array = array.get()
    if hasattr(array, "compute") and not isinstance(array, np.ndarray):
        array = array.compute()
    return np.asarray(array)


def array_to_pillow(array: tx.Any) -> "Image.Image":
    """Build a Pillow image from a C-ordered array of pixels.

    | Array                                      | Mode                   |
    | ------------------------------------------ | ---------------------- |
    | `bool`, `(rows, columns)`                  | `1`                    |
    | `uint8`, `(rows, columns)`                 | `L`                    |
    | `uint8`, `(rows, columns, 2 / 3 / 4)`      | `LA` / `RGB` / `RGBA`  |
    | `uint16`, `(rows, columns)`                | `I;16`                 |
    | `int32`, `(rows, columns)`                 | `I`                    |
    | `float32`, `(rows, columns)`               | `F`                    |

    A trailing axis of one sample is dropped. Whether the resulting mode can be
    stored depends on the format; JPEG, for example, cannot store `I;16`.

    Raises
    ------
    WriterError
        If the array has another type, more than four samples, or several
        samples that are not `uint8`.
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
    """Return the Pillow format for the extension of a file name, using the
    longest matching registered extension, or `None`.
    """
    if name is None:
        return None
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
    """Return whether Pillow can write a format, given by its Pillow name."""
    Image.init()
    return str(format).upper() in Image.SAVE


def encode_pillow(
    im: "Image.Image",
    format: str,
    dpi: tx.Optional[tx.Tuple[float, float]] = None,
    **options,
) -> bytes:
    """Encode a Pillow image in memory, so that a failure leaves no partial
    file.

    Parameters
    ----------
    im : PIL.Image.Image
        The image.
    format : str
        The Pillow name of the format, such as `"PNG"`.
    dpi : tuple of float, optional
        The resolution along x and y, which is passed only to the formats of
        [`DPI_FORMATS`][].
    **options : Any
        Options passed to `Image.save`, such as `quality=95` for JPEG.

    Raises
    ------
    WriterError
        If Pillow cannot write the format, or the format cannot store the
        image.
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
