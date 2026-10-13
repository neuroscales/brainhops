"""The metadata of an elastix transform parameter file."""

__all__ = ["ElastixMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base._utils_files import write_file
from brainhops.io.base.parsers import (
    TextFileReader,
    TextFileWriter,
    WriterError,
)
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import ElastixRaw


@register_format
class ElastixMetadata(MetadataFormat, TextFileReader, TextFileWriter):
    """Metadata of an elastix transform parameter file.

    The metadata holds the file as an [`ElastixRaw`][] record. A transform
    parameter file is short and is read in one pass, so the record holds
    the whole file, including the transform parameters, and the records
    of the files of its chain of initial transforms. The chain of an
    [`ElastixTransform`][brainhops.io.transformations.elastix.ElastixTransform]
    is therefore decoded from the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The metadata reads both syntaxes of elastix: as in
    elastix, a file whose name ends in `.toml` is read as TOML, and any
    other file in the classic text syntax.

    The text is read and written by the record. The metadata keeps no open
    file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt", ".toml")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("elastix", "transformix")

    raw: NoRepr[tx.Optional[ElastixRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score text as the text of a transform parameter file.

        The text is scored by [`ElastixRaw.sniff_text`][], in either
        syntax.

        Parameters
        ----------
        text : str
            The text of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the text is not a transform
            parameter file.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return ElastixRaw.sniff_text(text, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "ElastixMetadata":
        """Read the metadata of a file, with the records of its chain.

        The file is read by [`ElastixRaw.from_filename`][].

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of [`ElastixRaw.from_filename`][].

        Returns
        -------
        ElastixMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(ElastixRaw.from_filename(filename, **kwargs))

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "ElastixMetadata":
        """Read the metadata from an open file.

        The file is read by [`ElastixRaw.from_fileobj`][].

        Parameters
        ----------
        file : IO
            The open file, whose position is restored.
        **kwargs : Any
            Options of [`ElastixRaw.from_fileobj`][].

        Returns
        -------
        ElastixMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(ElastixRaw.from_fileobj(file, **kwargs))

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "ElastixMetadata":
        """Read the metadata from the text of a file.

        The text is read by [`ElastixRaw.from_text`][].

        Parameters
        ----------
        text : str
            The text of the file.
        **kwargs : Any
            Options of [`ElastixRaw.from_text`][].

        Returns
        -------
        ElastixMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(ElastixRaw.from_text(text, **kwargs))

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], **kwargs
    ) -> "ElastixMetadata":
        """Read the metadata from the lines of a file.

        The lines are read by [`ElastixRaw.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            Options of [`ElastixRaw.from_lines`][].

        Returns
        -------
        ElastixMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(ElastixRaw.from_lines(lines, **kwargs))

    # --- to -----------------------------------------------------------

    def to_text(self) -> str:
        """Return the text of the file that the record holds.

        Only the file of the record is written, and not the files of its
        chain.

        Returns
        -------
        str
            The text of the file, as [`ElastixRaw.to_text`][] writes it.

        Raises
        ------
        WriterError
            If the metadata holds no record, since a file without a
            transform is not a transform parameter file.
        """
        return _record(self).to_text()

    def to_lines(self) -> tx.Iterator[str]:
        """Yield the lines of the file that the record holds.

        Yields
        ------
        str
            The lines of [`to_text`][], without line endings.

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        return _record(self).to_lines()

    def to_fileobj(self, file: tx.IO) -> None:
        """Write the text of the file to an open text file.

        Parameters
        ----------
        file : IO
            The file, open for writing text.

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        file.write(self.to_text())

    def to_filename(self, filename: path.FilenameLike) -> None:
        """Write the file that the record holds, replacing a local file.

        Only the file of the record is written, and not the files of its
        chain. A local file is replaced at once.

        Parameters
        ----------
        filename : path-like
            The path of the file.

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        write_file(filename, self.to_bytes())


def _record(metadata: ElastixMetadata) -> ElastixRaw:
    """Return the record that metadata writes.

    Raises
    ------
    WriterError
        If the metadata holds no record. An empty record would make up a
        file without a transform, which elastix cannot read.
    """
    if metadata.raw is None:
        raise WriterError("This metadata holds no elastix transform to write.")
    return metadata.raw
