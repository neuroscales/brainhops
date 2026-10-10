"""The metadata of a FreeSurfer morph (`.m3z`)."""

__all__ = ["M3zMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    WriterError,
)
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import M3zRaw


@register_format
class M3zMetadata(MetadataFormat, BinaryFileReader, BinaryFileWriter):
    """Metadata of a morph file, which holds the file as an [`M3zRaw`][].

    A morph is a single gzip stream in which the node arrays follow the
    header directly, so the file is read in one pass, and its record
    holds the whole file, including the positions of the nodes. The
    chain of an
    [`M3zMorph`][brainhops.io.transformations.freesurfer.m3z.M3zMorph]
    is therefore decoded from the record of its metadata, and reading
    the metadata of a file reads the same content as reading the morph.

    The file is read and written by the record: [`from_fileobj`][] reads
    an [`M3zRaw`][] and wraps it, and [`to_bytes`][] encodes the record.
    The metadata keeps no open file. A copy of the record, as
    [`to_raw`][] returns it, shares the arrays of the record, so that
    copying metadata never duplicates the nodes.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".m3z", ".m3d")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("m3z",)

    raw: NoRepr[tx.Optional[M3zRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    def to_raw(self) -> tx.Optional[M3zRaw]:
        """Return a copy of the record that the caller is free to replace.

        The record is frozen, so a caller changes it with
        [`replace`][bagof.magic.replace]. The copy shares the arrays of
        the record, which are never changed in place, so that copying a
        record does not duplicate the nodes.

        Returns
        -------
        M3zRaw or None
            A copy of `raw`, or `None` when no record is held.
        """
        return None if self.raw is None else self.raw.copy()

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score an open binary file from its first bytes only.

        The file is scored by [`M3zRaw.sniff_fileobj`][].
        """
        return M3zRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score bytes, gzipped or not.

        The bytes are scored by [`M3zRaw.sniff_bytes`][].
        """
        return M3zRaw.sniff_bytes(content, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "M3zMetadata":
        """Read the metadata from the rest of an open binary file.

        The record is read by [`M3zRaw.from_fileobj`][], and the keyword
        arguments are other fields of the metadata.
        """
        return cls.from_raw(M3zRaw.from_fileobj(file), **kwargs)

    # --- to -----------------------------------------------------------

    def to_bytes(self, compress: bool = True) -> bytes:
        """Return the content of the morph file that the record holds.

        The content is gzipped (`.m3z`) unless `compress=False`
        (`.m3d`).

        Raises
        ------
        WriterError
            If the metadata holds no record, since a morph cannot be
            written without its nodes.
        """
        return _record(self).to_bytes(compress=compress)

    def to_filename(
        self,
        filename: path.FilenameLike,
        compress: tx.Optional[bool] = None,
    ) -> None:
        """Write the record to a file.

        By default, the file is gzipped unless its name ends in `.m3d`,
        as in FreeSurfer.

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        _record(self).to_filename(filename, compress=compress)


def _record(metadata: M3zMetadata) -> M3zRaw:
    """Return the record that metadata writes.

    Raises
    ------
    WriterError
        If the metadata holds no record. Unlike the record of a header,
        an empty record would make up a morph without nodes, which
        FreeSurfer cannot read.
    """
    if metadata.raw is None:
        raise WriterError("This metadata holds no morph to write.")
    return metadata.raw
