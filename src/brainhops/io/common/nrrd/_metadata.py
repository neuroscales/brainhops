"""The metadata of a NRRD image."""

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
from ._raw import NrrdRaw


@register_format
class NrrdMetadata(MetadataFormat, BinaryFileReader, BinaryFileWriter):
    """Metadata of a NRRD image, which is its header.

    The metadata of a NRRD image is its header, held as an [`NrrdRaw`][]
    record. [`NrrdImage`][brainhops.io.images.nrrd.NrrdImage] uses that
    record as the base of the header that it writes. Reading the metadata
    never reads the samples, nor looks for the data files that a detached
    header names, and a file cut after its header can still be read. The
    metadata keeps no open file.

    A file is read through the record: [`from_fileobj`][] reads an
    [`NrrdRaw`][] and wraps it. Writing the metadata writes the header
    alone, as the file stores it, up to the first sample of attached
    samples.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nrrd", ".nhdr")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("nrrd",)

    raw: NoRepr[tx.Optional[NrrdRaw]] = None
    """The record of the image, or `None` for metadata without a record."""

    def to_raw(self) -> tx.Optional[NrrdRaw]:
        """Return a copy of the record that the caller is free to change.

        Returns
        -------
        NrrdRaw or None
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
        """Return the confidence that a stream holds a NRRD header.

        The stream is tested with [`NrrdRaw.sniff_fileobj`][].
        """
        return NrrdRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes start with a NRRD header."""
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "NrrdMetadata":
        """Read the metadata from a NRRD stream.

        The record is read with [`NrrdRaw.from_fileobj`][], which receives
        the keyword arguments.
        """
        return cls.from_raw(NrrdRaw.from_fileobj(file, **kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        """Return the header of the record as the file stores it.

        Metadata without a record is written as an empty header.
        """
        if self.raw is None:
            return b""
        return self.raw.to_bytes(**kwargs)
