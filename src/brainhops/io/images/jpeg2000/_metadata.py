"""The metadata of a JPEG 2000 file."""

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
from ._raw import Jpeg2000Raw


@register_format
class Jpeg2000Metadata(MetadataReader, MetadataWriter, MetadataFormat):
    """Metadata of a JPEG 2000 file, which is its header.

    The metadata of a JP2 file or a raw codestream is everything that the
    file says beside its coded image, held as a [`Jpeg2000Raw`][] record.
    [`Jpeg2000Image`][brainhops.io.images.jpeg2000.Jpeg2000Image] uses
    that record as the base of the file that it writes. Reading the
    metadata never decodes the image, and a file cut after the main header
    of its codestream can still be read. The metadata keeps no open file.

    The file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter]. Writing the
    metadata writes the file up to its first tile, which is the header
    that the record was read from, and metadata without a record has no
    header to write.
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
    RAW: tx.ClassVar[type] = Jpeg2000Raw

    raw: NoRepr[tx.Optional[Jpeg2000Raw]] = None
    """The record of the file, or `None` for metadata without a record."""
