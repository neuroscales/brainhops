"""The metadata of a TrakEM2 project."""

__all__ = ["TrakEM2Metadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.metadata import (
    MetadataFormat,
    MetadataReader,
    MetadataWriter,
)

# this format
from ._raw import TrakEM2Raw


@register_format
class TrakEM2Metadata(MetadataFormat, MetadataReader, MetadataWriter):
    """Metadata of a TrakEM2 project, which holds its XML file as a
    [`TrakEM2Raw`][].

    The XML file of a project describes its layers and the patches placed
    in them, and the affine of each patch is a small value in that file,
    so the metadata of the file is the whole file. Reading it never reads
    the images of the patches, which are separate files.

    The file is read and written as bytes, so that its encoding, which is
    ISO-8859-1 in the files that TrakEM2 writes, and its line separators
    are kept. The file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter], and
    metadata without a record has no project to write. The metadata keeps
    no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xml",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("trakem2",)
    RAW: tx.ClassVar[type] = TrakEM2Raw

    raw: NoRepr[tx.Optional[TrakEM2Raw]] = None
    """The content of the file, or `None` for metadata without a record."""
