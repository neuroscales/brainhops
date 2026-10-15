"""The record of a raster image read with Pillow: its header."""

# stdlib
import pickle
import warnings
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr, replace
from PIL import Image

# internals
from brainhops._core.streams import preserve_position
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    SnifferContentError,
)

# this format
from ._utils import _open, _png_scal, pillow_dpi, pillow_layout, sniff_pillow


class PillowRaw(Raw, BinaryFileReader):
    """The header of a raster file, as Pillow reads it.

    A raster file, such as a PNG or a JPEG file, holds one picture or a
    sequence of frames. Pillow reads the header of a file when it opens
    it, and decodes the pixels of a frame only when they are needed. The
    record holds what Pillow knows at that point about one frame: the
    format of the file, the mode and the size of the frame, its `info`
    dictionary, the index of the frame and the number of frames in the
    file. The record never holds the pixels.

    [`layout`][] gives the shape and the type of the array that the frame
    decodes to, so that a reader knows them before it decodes anything,
    as the reader of image sequences does. The record is held by
    [`PillowMetadata`][brainhops.io.images.pillow.PillowMetadata]. A
    record is frozen, and two records are equal when their fields are.
    """

    format: tx.Optional[str] = None
    """The Pillow name of the format of the file, such as `"PNG"`."""

    mode: str = "L"
    """The Pillow mode of the frame as stored, such as `"P"` or `"I;16"`."""

    size: tx.Tuple[int, int] = (0, 0)
    """The width and the height of the frame, in pixels, as Pillow gives
    them."""

    info: NoRepr[tx.Dict[str, tx.Any]] = {}
    """The `info` dictionary of the frame, as Pillow reads it with the header.

    It holds the metadata of the format, such as `"dpi"`, `"icc_profile"`
    or `"exif"`, and, for a PNG file, the `sCAL` chunk that Pillow skips,
    as `(unit, width, height)`. Keys that Pillow adds only when it decodes
    the pixels, such as the text chunks of a PNG file that follow its
    pixels, or the `"duration"` of a WebP frame, are not in it. Values
    that cannot be pickled are left out, so that the metadata can be
    copied between processes. The dictionary is never changed in place.
    """

    frame: int = 0
    """The index of the frame that the record describes."""

    n_frames: int = 1
    """The number of frames in the file."""

    @property
    def dpi(self) -> tx.Optional[tx.Tuple[float, float]]:
        """The resolution that the file records, in dots per inch along x
        and y, or `None`.

        Placeholder resolutions, such as 72 dpi, are kept.
        """
        return pillow_dpi(self.info)

    def layout(
        self, palette: bool = True
    ) -> tx.Tuple[tx.Tuple[int, ...], np.dtype]:
        """Return the shape and the type of the array that the frame decodes
        to.

        The frame decodes to `(rows, columns)`, or to `(rows, columns,
        samples)` when a pixel has several samples, as the lazy array of
        [`PillowImage`][brainhops.io.images.pillow.PillowImage] decodes
        it. The shape and the type follow from the mode, the size and,
        for a palette image, the transparency that `info` records, so
        nothing is decoded.

        Parameters
        ----------
        palette : bool, default=True
            Whether the colours of a palette image are looked up, or its
            indices kept.

        Returns
        -------
        shape : tuple of int
            The shape of the decoded frame.
        dtype : numpy.dtype
            The type of its pixels, in native byte order.

        Raises
        ------
        ParserContentError
            If Pillow does not describe the mode of the frame.
        """
        dtype, samples = pillow_layout(self.mode, self.info, palette)
        width, height = self.size
        shape = (int(height), int(width))
        if samples > 1:
            shape += (samples,)
        return shape, dtype

    def copy(self) -> "PillowRaw":
        """Return a copy of the record.

        The `info` dictionary is copied, and its values are shared.

        Returns
        -------
        PillowRaw
            The copy.
        """
        return replace(self, info=dict(self.info))

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an image that Pillow
        reads.

        A format that is recognised by its magic number scores `LIKELY`.
        TIFF and JPEG 2000 score only `WEAK`, because the dedicated TIFF and
        JPEG 2000 readers read them better. The position of the stream is
        restored.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when Pillow does not recognise the
            content.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        fmt = sniff_pillow(file)
        if fmt in ("TIFF", "JPEG2000"):
            return Confidence.WEAK
        if fmt is not None:
            return Confidence.LIKELY
        if error:
            if error is True:
                error = SnifferContentError
            raise error("Pillow does not recognize this content as an image.")
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold an image that Pillow reads.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when Pillow does not recognise the
            content.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def from_fileobj(
        cls, file: tx.BinaryIO, frame: int = 0, **kwargs
    ) -> "PillowRaw":
        """Read the header of a raster file from a stream.

        Pillow reads the file from the start of the stream, whatever its
        position, which is restored. The pixels are not decoded. Pillow
        moves to a frame other than the first by reading the frames before
        it, so reading the header of such a frame reads more of the file.

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        frame : int, default=0
            The frame to describe. A negative index counts from the end.
        **kwargs : Any
            Refused, since the record has no other option.

        Returns
        -------
        PillowRaw
            The record.

        Raises
        ------
        TypeError
            If another option is given.
        ParserContentError
            If Pillow cannot read the file.
        IndexError
            If the file has no such frame.
        PIL.Image.DecompressionBombError
            If the image exceeds the limit of `PIL.Image.MAX_IMAGE_PIXELS`.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} does not take the option(s) "
                f"{', '.join(sorted(kwargs))}."
            )
        with preserve_position(file):
            return _read_record(cls, file, frame)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> "PillowRaw":
        """Read the header of a raster file from its bytes.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            The options of [`from_fileobj`][].

        Returns
        -------
        PillowRaw
            The record.
        """
        return cls.from_fileobj(BytesIO(bytes(content)), **kwargs)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _read_record(cls: type, file: tx.BinaryIO, frame: int) -> PillowRaw:
    """Read the record of a frame from a stream, whose position is not
    restored.

    The size, the mode and the `info` of the frame are read before its
    pixels are decoded, since Pillow adds some keys to `info` only when it
    decodes them.
    """
    with _open(file) as im:
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
        except Image.DecompressionBombError:
            raise
        except Exception as e:
            raise ParserContentError(
                f"Pillow cannot read frame {index} of this file: {e}"
            ) from e
        fmt, mode, size = im.format, im.mode, tuple(im.size)
        info = dict(im.info)
    if fmt == "PNG" and "sCAL" not in info:
        scal = _png_scal(file, 0)
        if scal is not None:
            info["sCAL"] = scal
    return cls(
        format=fmt,
        mode=mode,
        size=(int(size[0]), int(size[1])),
        info=_picklable(info),
        frame=index,
        n_frames=n_frames,
    )


def _picklable(info: tx.Mapping[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    """Return the entries of an `info` dictionary whose values can be
    pickled."""
    kept = {}
    for key, value in info.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pickle.dumps(value)
        except Exception:  # noqa: BLE001
            continue
        kept[key] = value
    return kept
