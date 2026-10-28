"""The metadata of a NRRD image."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops.datamodel.metadata._report import takes_error_policy

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileWriter
from brainhops.io.metadata import MetadataFormat, MetadataReader

# this format
from ._raw import NrrdRaw


@register_format
class NrrdMetadata(MetadataReader, BinaryFileWriter, MetadataFormat):
    """Metadata of a NRRD image, which is its header.

    The metadata of a NRRD image is its header, held as an [`NrrdRaw`][]
    record. [`NrrdImage`][brainhops.io.images.nrrd.NrrdImage] uses that
    record as the base of the header that it writes. Reading the metadata
    never reads the samples, nor looks for the data files that a detached
    header names, and a file cut after its header can still be read. The
    metadata keeps no open file.

    A file is read through the record: [`from_fileobj`][] reads an
    [`NrrdRaw`][] and wraps it. Writing the metadata writes the header
    alone, as the file stores it, up to the first sample of attached
    samples.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nrrd", ".nhdr")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nrrd",)
    RAW: tx.ClassVar[type] = NrrdRaw

    raw: NoRepr[tx.Optional[NrrdRaw]] = None
    """The record of the image, or `None` for metadata without a record."""

    @takes_error_policy
    def to_bytes(self, **kwargs) -> bytes:
        """Return the header of the record as the file stores it.

        Metadata without a record is written as an empty header.

        Parameters
        ----------
        **kwargs : Any
            Options of [`NrrdRaw.to_bytes`][].

        Returns
        -------
        bytes
            The header.
        """
        if self.raw is None:
            return b""
        return self.raw.to_bytes(**kwargs)
