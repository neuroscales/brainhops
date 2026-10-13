"""The metadata of an SPM normalisation file (`*_sn.mat`)."""

__all__ = ["SpmSnMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileReader
from brainhops.io.metadata import MetadataFormat

# this format
from ._sn_raw import SpmSnRaw


@register_format
class SpmSnMetadata(MetadataFormat, BinaryFileReader):
    """Metadata of an SPM normalisation file, which holds an [`SpmSnRaw`][].

    A `*_sn.mat` file is a small MATLAB file that is read in one pass, so
    its record holds the whole file, including the coefficients of the
    warp. The chain of an
    [`SpmSnTransformation`][brainhops.io.transformations.spm.SpmSnTransformation]
    is therefore decoded from the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation.

    The file is read by the record: [`from_fileobj`][] reads an
    [`SpmSnRaw`][] and wraps it. The metadata keeps no open file. It
    cannot be written, because brainhops does not write MATLAB files that
    SPM would read back as it wrote them (see
    [`brainhops.io.transformations.spm`][]).
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = ("_sn.mat",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("spm", "sn")

    raw: NoRepr[tx.Optional[SpmSnRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score an open binary file.

        The file is scored by [`SpmSnRaw.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The open binary file, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an SPM
            normalisation file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return SpmSnRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the bytes of a file.

        The bytes are scored by [`SpmSnRaw.sniff_bytes`][].

        Parameters
        ----------
        content : bytes
            The content of a file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not an SPM
            normalisation file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return SpmSnRaw.sniff_bytes(content, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "SpmSnMetadata":
        """Read the metadata from the rest of an open binary file.

        The record is read by [`SpmSnRaw.from_fileobj`][], and the keyword
        arguments are other fields of the metadata.

        Parameters
        ----------
        file : IO
            The open binary file.
        **kwargs : Any
            Other fields of the metadata.

        Returns
        -------
        SpmSnMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(SpmSnRaw.from_fileobj(file), **kwargs)
