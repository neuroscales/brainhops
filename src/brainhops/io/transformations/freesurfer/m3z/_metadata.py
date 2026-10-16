"""The metadata of a FreeSurfer morph (`.m3z`)."""

__all__ = ["M3zMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import WriterError
from brainhops.io.metadata import (
    MetadataFormat,
    MetadataReader,
    MetadataWriter,
)

# this format
from ._raw import M3zRaw


@register_format
class M3zMetadata(MetadataFormat, MetadataReader, MetadataWriter):
    """Metadata of a morph file, which holds the file as an [`M3zRaw`][].

    A morph is a single gzip stream in which the node arrays follow the
    header directly, so the file is read in one pass, and its record
    holds the whole file, including the positions of the nodes. The
    chain of an
    [`M3zMorph`][brainhops.io.transformations.freesurfer.m3z.M3zMorph]
    is therefore decoded from the record of its metadata, and reading
    the metadata of a file reads the same content as reading the morph.

    The file is read and written by the record, through
    [`MetadataReader`][brainhops.io.metadata.MetadataReader] and
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter]. The
    metadata keeps no open file. A copy of the record, as
    [`M3zRaw.copy`][] returns it, shares the arrays of the record, so that
    copying metadata never duplicates the nodes.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".m3z", ".m3d")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("m3z",)
    RAW: tx.ClassVar[type] = M3zRaw

    raw: NoRepr[tx.Optional[M3zRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    # --- to -----------------------------------------------------------

    def to_filename(
        self,
        filename: path.FilenameLike,
        compress: tx.Optional[bool] = None,
    ) -> None:
        """Write the record to a file.

        By default, the file is gzipped unless its name ends in `.m3d`,
        as in FreeSurfer.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        compress : bool, optional
            Whether to gzip the file. By default, the file is gzipped unless
            its name ends in `.m3d`.

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        self._record_to_write().to_filename(filename, compress=compress)

    def _record_to_write(self) -> M3zRaw:
        """Return the record that the metadata writes.

        Returns
        -------
        M3zRaw
            The record.

        Raises
        ------
        WriterError
            If the metadata holds no record. Unlike the record of a header,
            an empty record would make up a morph without nodes, which
            FreeSurfer cannot read.
        """
        if self.raw is None:
            raise WriterError("This metadata holds no morph to write.")
        return self.raw
