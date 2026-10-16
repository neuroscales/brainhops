"""The metadata of an ITK binary MATLAB (`.mat`) file."""

__all__ = ["MatMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileReader, BinaryFileWriter
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import MatRaw


@register_format
class MatMetadata(MetadataFormat, BinaryFileReader, BinaryFileWriter):
    """Metadata of an ITK `.mat` file, which holds it as a [`MatRaw`][].

    A `.mat` file is a short binary file that is read in one pass, so its
    record holds every block of the file, parameters included, and the
    bytes of the file. The chain of a
    [`MatTransform`][brainhops.io.transformations.itk.mat.MatTransform]
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The file stores nothing else, such as a name or a
    version of ITK.

    The file is read and written by the record: [`from_fileobj`][] reads a
    [`MatRaw`][] and wraps it, and [`to_bytes`][] returns the bytes of the
    record. The metadata keeps no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("itk", "ants", "mat")
    RAW: tx.ClassVar[type] = MatRaw

    raw: NoRepr[tx.Optional[MatRaw]] = None
    """The blocks of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a binary stream as an ITK `.mat` file.

        The stream is scored by [`MatRaw.sniff_fileobj`][].

        Parameters
        ----------
        file : file object
            The stream, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not an ITK
            `.mat` file.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return MatRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score bytes as those of an ITK `.mat` file.

        The bytes are scored by [`MatRaw.sniff_bytes`][].

        Parameters
        ----------
        content : bytes
            The content, or its beginning.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not an ITK
            `.mat` file.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return MatRaw.sniff_bytes(content, error=error, **kwargs)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Reject text, which is never an ITK `.mat` file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the content.
        error : bool or type[Exception], default=False
            Whether to raise an error instead of returning 0.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            Always `Confidence.NO`.
        """
        return MatRaw.sniff_lines(lines, error=error, **kwargs)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of the ITK `.mat` file that the record holds.

        Metadata without a record is written as a new, empty record, which
        is a file without blocks.

        Parameters
        ----------
        **kwargs : Any
            Options of [`MatRaw.to_bytes`][].

        Returns
        -------
        bytes
            The content of the file.
        """
        raw = MatRaw() if self.raw is None else self.raw
        return raw.to_bytes(**kwargs)
