"""The metadata of an AFNI matrix file."""

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
from ._raw import AfniAffineRaw


@register_format
class AfniAffineMetadata(MetadataReader, MetadataWriter, MetadataFormat):
    """Metadata of an AFNI matrix file, which holds the whole file.

    An AFNI matrix file is a short text file that is read in one pass, so
    its record, an [`AfniAffineRaw`][], holds the whole file, including
    its matrices. The data of an
    [`AfniAffine`][brainhops.io.transformations.afni.AfniAffine]
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The metadata keeps no open file.

    The file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter]. Metadata
    without a record is written as a new record, which holds the
    identity.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".aff12.1D", ".1D")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("afni", "aff12")
    RAW: tx.ClassVar[type] = AfniAffineRaw

    raw: NoRepr[tx.Optional[AfniAffineRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    def _record_to_write(self) -> AfniAffineRaw:
        """Return the record, or a new record if the metadata has none.

        Returns
        -------
        AfniAffineRaw
            The record that the metadata writes.
        """
        return AfniAffineRaw() if self.raw is None else self.raw
