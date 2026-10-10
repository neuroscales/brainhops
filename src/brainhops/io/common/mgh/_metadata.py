"""The metadata of an MGH file."""

# stdlib
from io import BytesIO

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileReader, BinaryFileWriter
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import MghRaw


@register_format
class MghMetadata(MetadataFormat, BinaryFileReader, BinaryFileWriter):
    """Metadata of an MGH or MGZ file, which holds everything but the voxels.

    The metadata of an MGH file is its header, the footer of acquisition
    parameters and the trailing tags, held as an [`MghRaw`][] record.
    [`MghImage`][brainhops.io.images.freesurfer.mgh.MghImage] uses that
    record as the base of the file that it writes. Reading the metadata
    never decodes the voxels, and a file cut after its header can still be
    read. The metadata keeps no open file.

    A file is read through the record: [`from_fileobj`][] reads an
    [`MghRaw`][] and wraps it. Writing the metadata writes the fixed header
    of the record alone, without the footer and the tags, which a file
    stores after the voxels.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mgh", ".mgz", ".mgh.gz")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("mgh", "mgz")

    raw: NoRepr[tx.Optional[MghRaw]] = None
    """The record of the file, or `None` for metadata without a record."""

    def to_raw(self) -> tx.Optional[MghRaw]:
        """Return a copy of the record that the caller is free to change.

        Returns
        -------
        MghRaw or None
            A copy of `raw`, or `None` when no record is held.
        """
        return None if self.raw is None else self.raw.copy()

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an MGH header.

        The stream is tested with [`MghRaw.sniff_fileobj`][].
        """
        return MghRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes start with an MGH header."""
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "MghMetadata":
        """Read the metadata from an MGH stream.

        The record is read with [`MghRaw.from_fileobj`][], which receives
        the keyword arguments.
        """
        return cls.from_raw(MghRaw.from_fileobj(file, **kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        """Return the fixed header of the record as an MGH file stores it.

        Metadata without a record is written as the header of a new
        record.
        """
        raw = MghRaw() if self.raw is None else self.raw
        return raw.to_bytes(**kwargs)

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the fixed header to a path, compressed for `.mgz` and `.gz`.

        The header is written with [`MghRaw.to_filename`][].
        """
        raw = MghRaw() if self.raw is None else self.raw
        raw.to_filename(filename, **kwargs)
