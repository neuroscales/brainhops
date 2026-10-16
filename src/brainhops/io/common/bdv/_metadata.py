"""The metadata of a BigDataViewer dataset."""

__all__ = ["BdvMetadata"]

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
from ._raw import BdvRaw


@register_format
class BdvMetadata(MetadataFormat, MetadataReader, MetadataWriter):
    """Metadata of a BigDataViewer dataset, which holds its XML file as a
    [`BdvRaw`][].

    The XML file of a dataset describes its views and their registrations,
    so the metadata of the file is the whole file. Reading the metadata
    reads the XML file only, never the HDF5 file that holds the voxels.
    An image read from the dataset adds to its record what the HDF5 file
    says about the resolution levels (see [`BdvRaw.pyramids`][]).

    The file is read and written as bytes, so that its line separators,
    which are CRLF in the files that BigDataViewer writes, are kept. The
    file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter], and
    metadata without a record has no file to write, since a dataset needs
    at least its setups and time points. The metadata keeps no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xml",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("bdv", "bigdataviewer")
    RAW: tx.ClassVar[type] = BdvRaw

    raw: NoRepr[tx.Optional[BdvRaw]] = None
    """The content of the file, or `None` for metadata without a record."""
