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
    RAW: tx.ClassVar[type] = MrtrixRaw

    raw: NoRepr[tx.Optional[MrtrixRaw]] = None
    """The record of the image, or `None` for metadata without a record."""

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an MRtrix header.

        The stream is tested with [`MrtrixRaw.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The stream, which may be gzipped.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no MRtrix
            header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return MrtrixRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes start with an MRtrix header.

        Parameters
        ----------
        data : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes hold no MRtrix
            header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the header of the record as the file stores it.

        Metadata without a record is written as an empty header.

        Parameters
        ----------
        **kwargs : Any
            Options of [`MrtrixRaw.to_bytes`][].

        Returns
        -------
        bytes
            The header.
        """
        if self.raw is None:
            return b""
        return self.raw.to_bytes(**kwargs)
