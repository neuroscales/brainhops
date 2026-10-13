"""The metadata of a raster image read with Pillow."""

# stdlib
from io import BytesIO

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileReader
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import PillowRaw
from ._utils import EXTENSIONS


@register_format
class PillowMetadata(MetadataFormat, BinaryFileReader):
    """Metadata of a raster image, which is its header as Pillow reads it.

    The metadata of a raster file, such as a PNG or a JPEG file, is held
    as a [`PillowRaw`][] record: the format of the file, the mode and the
    size of a frame, its `info` dictionary, and the number of frames.
    [`PillowImage`][brainhops.io.images.pillow.PillowImage] holds such
    metadata and a lazy array of the pixels of the frame. Reading the
    metadata never decodes the pixels, and a file cut after its header can
    still be read. The metadata keeps no open file.

    Pillow writes a header only together with the pixels that follow it,
    so the metadata cannot be written alone. The image writes it.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = EXTENSIONS
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("pillow",)

    raw: NoRepr[tx.Optional[PillowRaw]] = None
    """The record of the image, or `None` for metadata without a record."""

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an image that Pillow
        reads.

        The stream is tested with [`PillowRaw.sniff_fileobj`][].

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
        return PillowRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold an image that Pillow reads.

        Parameters
        ----------
        data : bytes
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
        return cls.sniff_fileobj(BytesIO(bytes(data)), error=error, **kwargs)

    @classmethod
    def from_fileobj(
        cls, file: tx.BinaryIO, frame: int = 0, **kwargs
    ) -> "PillowMetadata":
        """Read the metadata of a frame of a raster file from a stream.

        The record is read with [`PillowRaw.from_fileobj`][], which
        receives the options and refuses those it does not know.

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        frame : int, default=0
            The frame to describe. A negative index counts from the end.
        **kwargs : Any
            Options of the reader, which [`PillowRaw.from_fileobj`][]
            refuses.

        Returns
        -------
        PillowMetadata
            The metadata, which holds the record of the header.
        """
        return cls.from_raw(
            PillowRaw.from_fileobj(file, frame=frame, **kwargs)
        )
