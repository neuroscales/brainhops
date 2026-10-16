"""The metadata of an FSL FLIRT `.mat` file."""

__all__ = ["FlirtMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.metadata import MetadataFormat, MetadataReader

# this format
from ._raw import FlirtRaw


@register_format
class FlirtMetadata(MetadataFormat, MetadataReader):
    """Metadata of an FSL FLIRT `.mat` file, which holds a [`FlirtRaw`][].

    A FLIRT file is a short text file that is read in one pass, so its
    record holds the whole file, which is the FLIRT matrix. The FLIRT
    matrix of a
    [`FlirtTransform`][brainhops.io.transformations.fsl.flirt.FlirtTransform]
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The moving and reference images that place the matrix
    in world coordinates are not stored in the file, so they are options
    of the transformation and not part of its metadata.

    The text is read by the record: [`from_lines`][] parses a
    [`FlirtRaw`][] and wraps it. brainhops does not write FLIRT files, so
    the metadata cannot be written either.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("fsl", "flirt")
    RAW: tx.ClassVar[type] = FlirtRaw

    raw: NoRepr[tx.Optional[FlirtRaw]] = None
    """The matrix of the file, or `None` for metadata without a record."""
