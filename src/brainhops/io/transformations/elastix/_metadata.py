"""The metadata of an elastix transform parameter file."""

__all__ = ["ElastixMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import WriterError
from brainhops.io.metadata import (
    MetadataFormat,
    MetadataReader,
    MetadataWriter,
)

# this format
from ._raw import ElastixRaw


@register_format
class ElastixMetadata(MetadataFormat, MetadataReader, MetadataWriter):
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

    The file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter], because the
    readers of the bases would lose the line endings, the name of the
    file, or the text. A file is written without its chain. The metadata
    keeps no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt", ".toml")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("elastix", "transformix")
    RAW: tx.ClassVar[type] = ElastixRaw

    raw: NoRepr[tx.Optional[ElastixRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    def _record_to_write(self) -> ElastixRaw:
        """Return the record that the metadata writes.

        Returns
        -------
        ElastixRaw
            The record.

        Raises
        ------
        WriterError
            If the metadata holds no record. An empty record would make up a
            file without a transform, which elastix cannot read.
        """
        if self.raw is None:
            raise WriterError(
                "This metadata holds no elastix transform to write."
            )
        return self.raw
