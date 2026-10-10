"""The metadata of an MRtrix image."""

# stdlib
from io import BytesIO

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileReader, BinaryFileWriter
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import MrtrixRaw


@register_format
class MrtrixMetadata(MetadataFormat, BinaryFileReader, BinaryFileWriter):
    """Metadata of an MRtrix image, which is its header.

    The metadata of an MRtrix image is its header, held as an
    [`MrtrixRaw`][] record.
    [`MrtrixImage`][brainhops.io.images.mrtrix.MrtrixImage] uses that
    record as the base of the header that it writes. Reading the metadata
    never reads the voxels, and a file cut after its header can still be
    read. The metadata keeps no open file.

    A file is read through the record: [`from_fileobj`][] reads an
    [`MrtrixRaw`][] and wraps it. Writing the metadata writes the header
    alone, as the file stores it, up to the first voxel of a single-file
    image.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mif", ".mif.gz", ".mih")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("mrtrix",)

    raw: NoRepr[tx.Optional[MrtrixRaw]] = None
    """The record of the image, or `None` for metadata without a record."""

    def to_raw(self) -> tx.Optional[MrtrixRaw]:
        """Return a copy of the record that the caller is free to change.

        Returns
        -------
        MrtrixRaw or None
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
        """Return the confidence that a stream holds an MRtrix header.

        The stream is tested with [`MrtrixRaw.sniff_fileobj`][].
        """
        return MrtrixRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes start with an MRtrix header."""
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "MrtrixMetadata":
        """Read the metadata from an MRtrix stream, gzipped or not.

        The record is read with [`MrtrixRaw.from_fileobj`][], which
        receives the keyword arguments.
        """
        return cls.from_raw(MrtrixRaw.from_fileobj(file, **kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        """Return the header of the record as the file stores it.

        Metadata without a record is written as an empty header.
        """
        if self.raw is None:
            return b""
        return self.raw.to_bytes(**kwargs)
