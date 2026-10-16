"""The metadata of a Convert3D matrix file."""

__all__ = ["C3dAffineMetadata"]

import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import WriterError
from brainhops.io.metadata import (
    MetadataFormat,
    MetadataReader,
    MetadataWriter,
)

from ._raw import C3dAffineRaw


@register_format
class C3dAffineMetadata(MetadataReader, MetadataWriter, MetadataFormat):
    """Metadata of a Convert3D matrix file, which holds the file as a record.

    A matrix file of Greedy, `c3d_affine_tool` or ITK-SNAP is a short
    text file that is read in one pass, so its record, a
    [`C3dAffineRaw`][], holds the whole file, including the matrix. The
    data of a
    [`C3dAffine`][brainhops.io.transformations.c3d.C3dAffine] therefore
    lives in the record of its metadata, and reading the metadata of a
    file reads the same content as reading the transformation. The file
    is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter]. The
    metadata keeps no open file.

    The content of the file holds no marker, so the metadata is never
    recognized from it. It is read from a file with the `.mat` extension
    or from a file selected by a hint.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("c3d", "greedy", "itksnap")
    RAW: tx.ClassVar[type] = C3dAffineRaw

    raw: NoRepr[tx.Optional[C3dAffineRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    def _record_to_write(self) -> C3dAffineRaw:
        """Return the record that the metadata writes.

        Returns
        -------
        C3dAffineRaw
            The record.

        Raises
        ------
        WriterError
            If the metadata has no record.
        """
        if self.raw is None:
            raise WriterError(
                "Metadata of a Convert3D matrix without a record cannot be "
                "written, because the file holds the matrix and nothing else. "
                "Write the transformation instead."
            )
        return self.raw
