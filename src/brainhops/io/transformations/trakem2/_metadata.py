"""The metadata of a TrakEM2 project."""

__all__ = ["TrakEM2Metadata"]

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
from ._raw import TrakEM2Raw


@register_format
class TrakEM2Metadata(MetadataFormat, BinaryFileReader, BinaryFileWriter):
    """Metadata of a TrakEM2 project, which holds its XML file as a
    [`TrakEM2Raw`][].

    The XML file of a project describes its layers and the patches placed
    in them, and the affine of each patch is a small value in that file,
    so the metadata of the file is the whole file. Reading it never reads
    the images of the patches, which are separate files.

    The file is read and written as bytes, so that its encoding, which is
    ISO-8859-1 in the files that TrakEM2 writes, and its line separators
    are kept. The metadata keeps no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xml",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("trakem2",)

    raw: NoRepr[tx.Optional[TrakEM2Raw]] = None
    """The content of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a stream from the document type of its XML.

        The stream is scored by [`TrakEM2Raw.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The binary stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not a TrakEM2
            project.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return TrakEM2Raw.sniff_fileobj(file, error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score bytes from the document type of their XML.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not a TrakEM2
            project.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return TrakEM2Raw.sniff_bytes(content, error=error)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "TrakEM2Metadata":
        """Read the metadata from a binary stream of an XML file.

        The record is read by [`TrakEM2Raw.from_fileobj`][].

        Parameters
        ----------
        file : BinaryIO
            The stream.
        **kwargs : Any
            Options of the reader, which [`TrakEM2Raw.from_fileobj`][]
            refuses.

        Returns
        -------
        TrakEM2Metadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(TrakEM2Raw.from_fileobj(file, **kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of the XML file that the record holds.

        Metadata without a record has no project to write, and is
        refused.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer, which [`TrakEM2Raw.to_bytes`][] refuses.

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
                "This TrakEM2 metadata holds no record, so it has no project "
                "to write."
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

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        write_file(filename, self.to_bytes(**kwargs))
