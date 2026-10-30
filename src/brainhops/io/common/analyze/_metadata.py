"""The metadata of an Analyze 7.5 image."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.metadata import MetadataFormat, MetadataReader

# this format
from ._raw import AnalyzeRaw


@register_format
class AnalyzeMetadata(MetadataReader, MetadataFormat):
    """Metadata of an Analyze 7.5 image, which is its header.

    The metadata of an Analyze image is read from its header file alone,
    and held as an [`AnalyzeRaw`][] record, so the image file is never
    read. The metadata keeps no open file. Analyze files cannot be
    written, so the metadata cannot be written either, and no field of
    the vocabulary is decoded from the header yet.

    A file is read through the record: [`from_fileobj`][] reads an
    [`AnalyzeRaw`][] and wraps it.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".hdr",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("analyze",)
    RAW: tx.ClassVar[type] = AnalyzeRaw

    raw: NoRepr[tx.Optional[AnalyzeRaw]] = None
    """The header of the image, or `None` for metadata without a record."""
