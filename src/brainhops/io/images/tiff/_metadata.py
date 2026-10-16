"""The metadata of a TIFF file."""

# stdlib
from io import BytesIO

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core.streams import preserve_position
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    SnifferContentError,
)
from brainhops.io.metadata import MetadataFormat

# this format
from . import _utils as backend
from ._raw import TiffRaw, read_record


@register_format
class TiffMetadata(BinaryFileReader, MetadataFormat):
    """Metadata of a series of a TIFF file, held as a [`TiffRaw`][] record.

    The metadata of a TIFF file is what its IFDs, its OME-XML and its
    ImageJ metadata record about one of its series, without the pixels.
    [`TiffImage`][brainhops.io.images.tiff.TiffImage] and
    [`TiffMultiScaleImage`][brainhops.io.images.tiff.TiffMultiScaleImage]
    hold it, and every level of a pyramid holds the same metadata.

    Reading the metadata reads the IFDs with tifffile and never a pixel.
    The metadata cannot be written by itself, because a TIFF file has no
    header that can be written apart from its pixels: the IFDs of
    tifffile point to the pixels that follow them. The metadata keeps no
    open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = backend.EXTENSIONS
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("tiff", "tifffile")

    raw: NoRepr[tx.Optional[TiffRaw]] = None
    """The record of the series, or `None` for metadata without a record."""

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a TIFF file.

        The first four bytes are compared with the magic numbers of TIFF
        and BigTIFF, and the position of the stream is restored.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no TIFF file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        with preserve_position(file):
            head = file.read(4)
        if backend.is_tiff(head):
            return Confidence.LIKELY
        if error:
            if error is True:
                error = SnifferContentError
            raise error("This content is not a TIFF file.")
        return Confidence.NO

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold a TIFF file.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes hold no TIFF file.
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
        cls, file: tx.BinaryIO, series: int = 0, **kwargs
    ) -> "TiffMetadata":
        """Read the metadata of a series from an open TIFF file.

        The IFDs are read with tifffile, and the position of the stream is
        restored.

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        series : int, default=0
            The series, the first by default. A negative index counts from
            the end.
        **kwargs : Any
            Refused, since the metadata has no other option.

        Returns
        -------
        TiffMetadata
            The metadata, which holds the record of the series.

        Raises
        ------
        TypeError
            If another option is given.
        ParserContentError
            If tifffile cannot read the file.
        IndexError
            If the file has no such series.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} does not take the option(s) "
                f"{', '.join(sorted(kwargs))}."
            )
        tf = backend._require_tifffile()
        try:
            with preserve_position(file):
                with tf.TiffFile(file) as tif:
                    record = read_record(tif, series)
        except (tf.TiffFileError, ValueError, OSError) as e:
            raise ParserContentError(
                f"tifffile cannot read this file: {e}"
            ) from e
        return cls.from_raw(record)
