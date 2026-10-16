"""The metadata of a Neuroglancer coordinate transform."""

__all__ = ["NeuroglancerTransformMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import WriterError
from brainhops.io.metadata import (
    MetadataFormat,
    MetadataReader,
    MetadataWriter,
)

# this format
from ._raw import NeuroglancerTransformRaw


@register_format
class NeuroglancerTransformMetadata(
    MetadataReader, MetadataWriter, MetadataFormat
):
    """Metadata of a Neuroglancer coordinate transform.

    The metadata holds the JSON object of the transform as a
    [`NeuroglancerTransformRaw`][]. The JSON is short and is read in one
    pass, so its record holds the whole transform, including its matrix,
    and reading the metadata reads the same content as reading the
    transformation.

    The file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter]. The metadata
    keeps no open file.

    The metadata answers to the hint `"neuroglancer.affine"`, as the
    transformation does. The hint `"neuroglancer"` names the metadata of
    Neuroglancer precomputed volumes.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".ngtransform.json",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("neuroglancer.affine",)
    RAW: tx.ClassVar[type] = NeuroglancerTransformRaw

    raw: NoRepr[tx.Optional[NeuroglancerTransformRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    def _record_to_write(self) -> NeuroglancerTransformRaw:
        """Return the record that the metadata writes.

        Returns
        -------
        NeuroglancerTransformRaw
            The record.

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        if self.raw is None:
            raise WriterError(
                "This metadata holds no Neuroglancer coordinate transform to "
                "write. A transform needs output dimensions, which only a "
                "record or a transformation gives."
            )
        return self.raw
