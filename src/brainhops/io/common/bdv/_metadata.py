"""The metadata of a BigDataViewer dataset."""

__all__ = ["BdvMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base._utils_files import write_file
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    WriterError,
)
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import BdvRaw


@register_format
class BdvMetadata(MetadataFormat, BinaryFileReader, BinaryFileWriter):
    """Metadata of a BigDataViewer dataset, which holds its XML file as a
    [`BdvRaw`][].

    The XML file of a dataset describes its views and their registrations,
    so the metadata of the file is the whole file. Reading the metadata
    reads the XML file only, never the HDF5 file that holds the voxels.
    An image read from the dataset adds to its record what the HDF5 file
    says about the resolution levels (see [`BdvRaw.pyramids`][]).

    The file is read and written as bytes, so that its line separators,
    which are CRLF in the files that BigDataViewer writes, are kept. The
    metadata keeps no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xml",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("bdv", "bigdataviewer")

    raw: NoRepr[tx.Optional[BdvRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a stream from the root element of its XML.

        The stream is scored by [`BdvRaw.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The binary stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not a
            BigDataViewer XML file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return BdvRaw.sniff_fileobj(file, error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score bytes from the root element of their XML.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not a
            BigDataViewer XML file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return BdvRaw.sniff_bytes(content, error=error)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "BdvMetadata":
        """Read the metadata from a binary stream of an XML file.

        The record is read by [`BdvRaw.from_fileobj`][].

        Parameters
        ----------
        file : BinaryIO
            The stream.
        **kwargs : Any
            Options of the reader, which [`BdvRaw.from_fileobj`][]
            refuses.

        Returns
        -------
        BdvMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(BdvRaw.from_fileobj(file, **kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of the XML file that the record holds.

        Metadata without a record has no file to write, and is refused,
        since a dataset needs at least its setups and time points.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer, which [`BdvRaw.to_bytes`][] refuses.

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        if self.raw is None:
            raise WriterError(
                "This BigDataViewer metadata holds no record, so it has no "
                "XML file to write."
            )
        return self.raw.to_bytes(**kwargs)

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the XML file, replacing a local file at once.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of the writer, which are refused.
        """
        write_file(filename, self.to_bytes(**kwargs))
