"""The metadata of a NIfTI file."""

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
from ._raw import NiftiRaw


@register_format
class NiftiMetadata(MetadataFormat, BinaryFileReader, BinaryFileWriter):
    """Metadata of a NIfTI file, which holds the header of the file.

    The metadata of a NIfTI file is read from the header alone, so the
    voxels that follow the header are never read, and a file cut after its
    header can still be read. The header is held as a [`NiftiRaw`][]
    record, which [`NiftiImage`][brainhops.io.images.nifti.NiftiImage] uses
    as the base of the header that it writes. The metadata keeps no open
    file.

    A file is read through the record: [`from_fileobj`][] reads a
    [`NiftiRaw`][] and wraps it, and writing the metadata writes the header
    of the record without voxels.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nifti",)

    raw: NoRepr[tx.Optional[NiftiRaw]] = None
    """The header of the file, or `None` for metadata without a record."""

    def to_raw(self) -> tx.Optional[NiftiRaw]:
        """Return a copy of the record that the caller is free to change.

        Returns
        -------
        NiftiRaw or None
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
        """Return the confidence that a stream holds a NIfTI header.

        The stream is tested with [`NiftiRaw.sniff_fileobj`][].
        """
        return NiftiRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes start with a NIfTI header."""
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "NiftiMetadata":
        """Read the metadata from the header at the start of a stream.

        The header is read with [`NiftiRaw.from_fileobj`][], which receives
        the keyword arguments.
        """
        return cls.from_raw(NiftiRaw.from_fileobj(file, **kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        """Return the header of the record as a NIfTI file stores it.

        Metadata without a record is written as an empty NIfTI-1 header.
        """
        raw = NiftiRaw() if self.raw is None else self.raw
        return raw.to_bytes(**kwargs)
