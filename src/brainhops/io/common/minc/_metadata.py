"""The metadata of a MINC image."""

# stdlib
from io import BytesIO

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileReader
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import MincRaw


@register_format
class MincMetadata(MetadataFormat, BinaryFileReader):
    """Metadata of a MINC image, which is its header.

    The metadata of a MINC image is its header, held as a [`MincRaw`][]
    record. Reading the metadata never reads the voxels, and a file cut
    after its header can still be read. The metadata keeps no open file.
    MINC files cannot be written, so the metadata cannot be written
    either.

    A file is read through the record: [`from_fileobj`][] reads a
    [`MincRaw`][] and wraps it.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mnc", ".mnc.gz")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("minc",)

    raw: NoRepr[tx.Optional[MincRaw]] = None
    """The record of the image, or `None` for metadata without a record."""

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a MINC file.

        The stream is tested with [`MincRaw.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The stream, which may be gzipped.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no MINC
            file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return MincRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold a MINC file.

        Parameters
        ----------
        data : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes hold no MINC file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "MincMetadata":
        """Read the metadata from a MINC stream, gzipped or not.

        The record is read with [`MincRaw.from_fileobj`][], which
        receives the keyword arguments and refuses them.

        Parameters
        ----------
        file : BinaryIO
            The stream, which may be gzipped.
        **kwargs : Any
            Options of the reader, which [`MincRaw.from_fileobj`][] refuses.

        Returns
        -------
        MincMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(MincRaw.from_fileobj(file, **kwargs))
