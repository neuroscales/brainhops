"""The metadata of an MNI transform file."""

__all__ = ["XfmMetadata"]

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
from ._raw import XfmRaw


@register_format
class XfmMetadata(MetadataFormat, MetadataReader, MetadataWriter):
    """Metadata of an MNI transform file, which holds it as an [`XfmRaw`][].

    An MNI transform file is a short text file that is read in one pass,
    so its record holds the whole file: its comments and its transforms,
    including the matrices of its linear transforms and the names of the
    displacement volumes of its grid transforms. Reading the metadata of a
    file never reads a displacement volume, which is a separate MINC file.

    The file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter]. The metadata
    keeps no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xfm",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("minc", "xfm")
    RAW: tx.ClassVar[type] = XfmRaw

    raw: NoRepr[tx.Optional[XfmRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    def _record_to_write(self) -> XfmRaw:
        """Return the record that the metadata writes.

        Returns
        -------
        XfmRaw
            The record.

        Raises
        ------
        WriterError
            If the metadata holds no record. An empty record would make up a
            file without a transform, which libminc does not read.
        """
        if self.raw is None:
            raise WriterError("This metadata holds no transform to write.")
        return self.raw
