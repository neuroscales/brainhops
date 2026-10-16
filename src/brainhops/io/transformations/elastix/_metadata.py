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
    RAW: tx.ClassVar[type] = ElastixRaw

    raw: NoRepr[tx.Optional[ElastixRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    # Each reader forwards to the reader of the record of the same name,
    # since the bases would route a path, an open file and text to readers
    # that lose the line endings, the name of the file, or the text.

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return ElastixRaw.sniff_text(text, error=error, **kwargs)

    # --- to -----------------------------------------------------------

    def to_text(self) -> str:
        """Return the text of the file of the record, without its chain.

        Raises
        ------
        WriterError
            If the metadata holds no record, since a file without a
            transform is not a transform parameter file.
        """
        return _record(self).to_text()

    def to_lines(self) -> tx.Iterator[str]:
        return _record(self).to_lines()

    def to_fileobj(self, file: tx.IO) -> None:
        # The record writes its text as it is, where the base would write
        # `to_lines` with "\n" endings.
        _record(self).to_fileobj(file)

    def to_filename(self, filename: path.FilenameLike) -> None:
        """Write the file of the record, without its chain.

        A local file is replaced at once.
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
