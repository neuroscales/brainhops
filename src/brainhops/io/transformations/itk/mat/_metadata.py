"""The metadata of an ITK binary MATLAB (`.mat`) file."""

__all__ = ["MatMetadata"]

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
from ._raw import MatRaw


@register_format
class MatMetadata(MetadataReader, MetadataWriter, MetadataFormat):
    """Metadata of an ITK `.mat` file, which holds it as a [`MatRaw`][].

    A `.mat` file is a short binary file that is read in one pass, so its
    record holds every block of the file, parameters included, and the
    bytes of the file. The chain of a
    [`MatTransform`][brainhops.io.transformations.itk.mat.MatTransform]
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The file stores nothing else, such as a name or a
    version of ITK.

    The file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter]. The metadata
    keeps no open file. Metadata without a record is written as a new
    record.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("itk", "ants", "mat")
    RAW: tx.ClassVar[type] = MatRaw

    raw: NoRepr[tx.Optional[MatRaw]] = None
    """The blocks of the file, or `None` for metadata without a record."""

    def _record_to_write(self) -> MatRaw:
        """Return the record, or a new record if the metadata has none.

        Returns
        -------
        MatRaw
            The record that the metadata writes.
        """
        return MatRaw() if self.raw is None else self.raw
