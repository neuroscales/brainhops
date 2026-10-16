"""The metadata of an MRtrix linear transform file."""

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
from ._raw import MrtrixLinearRaw


@register_format
class MrtrixLinearMetadata(MetadataReader, MetadataWriter, MetadataFormat):
    """Metadata of an MRtrix linear transform file, which holds the file.

    A linear transform file is a short text file that is read in one pass,
    so its record, an [`MrtrixLinearRaw`][], holds the whole file,
    including the matrix. The data of an
    [`MrtrixLinearTransform`][brainhops.io.transformations.mrtrix.MrtrixLinearTransform]
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The metadata keeps no open file.

    The file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter]. Metadata
    without a record is written as a new record, which holds the
    identity.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("mrtrix",)
    RAW: tx.ClassVar[type] = MrtrixLinearRaw

    raw: NoRepr[tx.Optional[MrtrixLinearRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    def _record_to_write(self) -> MrtrixLinearRaw:
        """Return the record, or a new record if the metadata has none.

        Returns
        -------
        MrtrixLinearRaw
            The record that the metadata writes.
        """
        return MrtrixLinearRaw() if self.raw is None else self.raw
