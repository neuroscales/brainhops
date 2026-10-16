"""The metadata of a `register.dat` file."""

__all__ = ["RegisterDatMetadata"]

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
from ._raw import RegisterDatRaw


@register_format
class RegisterDatMetadata(MetadataReader, MetadataWriter, MetadataFormat):
    """Metadata of a `register.dat` file, which holds the file as a record.

    A `register.dat` file is a short text file that is read in one pass,
    so its record, a [`RegisterDatRaw`][], holds the whole file, including
    the registration matrix. The data of a
    [`RegisterDatTransformation`][brainhops.io.transformations.freesurfer.regdat.RegisterDatTransformation]
    therefore lives in the record of its metadata, and reading the metadata
    of a file reads the same content as reading the transformation.

    The file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter]. The metadata
    keeps no open file. Metadata without a record is written as a new
    record, which holds the identity.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".dat", ".reg")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("regdat",)
    RAW: tx.ClassVar[type] = RegisterDatRaw

    raw: NoRepr[tx.Optional[RegisterDatRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    def _record_to_write(self) -> RegisterDatRaw:
        """Return the record, or a new record if the metadata has none.

        Returns
        -------
        RegisterDatRaw
            The record that the metadata writes.
        """
        return RegisterDatRaw() if self.raw is None else self.raw
