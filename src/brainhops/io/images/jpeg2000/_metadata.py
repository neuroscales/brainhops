"""The metadata of a JPEG 2000 file."""

# stdlib
from io import BytesIO

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    WriterError,
)
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import Jpeg2000Raw


@register_format
class Jpeg2000Metadata(MetadataFormat, BinaryFileReader, BinaryFileWriter):
    """Metadata of a JPEG 2000 file, which is its header.

    The metadata of a JP2 file or a raw codestream is everything that the
    file says beside its coded image, held as a [`Jpeg2000Raw`][] record.
    [`Jpeg2000Image`][brainhops.io.images.jpeg2000.Jpeg2000Image] uses
    that record as the base of the file that it writes. Reading the
    metadata never decodes the image, and a file cut after the main header
    of its codestream can still be read. The metadata keeps no open file.

    A file is read through the record: [`from_fileobj`][] reads a
    [`Jpeg2000Raw`][] and wraps it. Writing the metadata writes the file
    up to its first tile, which is the header that the record was read
    from.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (
        ".jp2",
        ".jpx",
        ".jpf",
        ".j2k",
        ".j2c",
        ".jpc",
    )
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("jpeg2000", "jp2", "j2k", "j2c")

    raw: NoRepr[tx.Optional[Jpeg2000Raw]] = None
    """The record of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a JPEG 2000 header.

        The stream is tested with [`Jpeg2000Raw.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no JPEG 2000
            header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return Jpeg2000Raw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes start with a JPEG 2000 header.

        Parameters
        ----------
        data : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes hold no JPEG 2000
            header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(bytes(data)), error=error, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "Jpeg2000Metadata":
        """Read the metadata from a JPEG 2000 stream.

        The record is read with [`Jpeg2000Raw.from_fileobj`][], which
        receives the keyword arguments and refuses them.

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        **kwargs : Any
            Options of [`Jpeg2000Raw.from_fileobj`][], which refuses them.

        Returns
        -------
        Jpeg2000Metadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(Jpeg2000Raw.from_fileobj(file, **kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        """Return the header of the file, up to its first tile.

        The header is written with [`Jpeg2000Raw.to_bytes`][].

        Parameters
        ----------
        **kwargs : Any
            Options of [`Jpeg2000Raw.to_bytes`][], which refuses them.

        Returns
        -------
        bytes
            The header.

        Raises
        ------
        WriterError
            If the metadata has no record, or its record cannot be
            written.
        """
        if self.raw is None:
            raise WriterError(
                "This JPEG 2000 metadata has no record, so it has no header "
                "to write."
            )
        return self.raw.to_bytes(**kwargs)
