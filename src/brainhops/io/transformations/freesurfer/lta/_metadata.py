"""The metadata of an LTA file."""

__all__ = ["LtaMetadata"]

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
from ._raw import LtaRaw


@register_format
class LtaMetadata(MetadataFormat, MetadataReader, MetadataWriter):
    """Metadata of an LTA file, which holds the file as an [`LtaRaw`][].

    An LTA file is a short text file that is read in one pass, so its
    record holds the whole file, including the matrix of the affine. The
    data of an
    [`LtaTransformation`][brainhops.io.transformations.freesurfer.lta.LtaTransformation]
    therefore lives in the record of its metadata, and reading the metadata
    of a file reads the same content as reading the transformation.

    The file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter]. The metadata
    keeps no open file. Metadata without a record is written as a new
    record.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".lta",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("lta",)
    RAW: tx.ClassVar[type] = LtaRaw

    raw: NoRepr[tx.Optional[LtaRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    def _record_to_write(self) -> LtaRaw:
        """Return the record, or a new record if the metadata has none.

        Returns
        -------
        LtaRaw
            The record that the metadata writes.
        """
        return LtaRaw() if self.raw is None else self.raw
