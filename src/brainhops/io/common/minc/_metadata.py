"""The metadata of a MINC image."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.metadata import MetadataFormat, MetadataReader

# this format
from ._raw import MincRaw


@register_format
class MincMetadata(MetadataFormat, MetadataReader):
    """Metadata of a MINC image, which is its header.

    The metadata of a MINC image is its header, held as a [`MincRaw`][]
    record. Reading the metadata never reads the voxels, and a file cut
    after its header can still be read. The metadata keeps no open file.
    MINC files cannot be written, so the metadata cannot be written
    either.

    A file is read through the record: [`from_fileobj`][] reads a
    [`MincRaw`][] and wraps it.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mnc", ".mnc.gz")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("minc",)
    RAW: tx.ClassVar[type] = MincRaw

    raw: NoRepr[tx.Optional[MincRaw]] = None
    """The record of the image, or `None` for metadata without a record."""
