"""The metadata of an MNI transform file."""

__all__ = ["XfmMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import TextFileReader, TextFileWriter, WriterError
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import XfmRaw


@register_format
class XfmMetadata(MetadataFormat, TextFileReader, TextFileWriter):
    """Metadata of an MNI transform file, which holds the file as an [`XfmRaw`][].

    An MNI transform file is a short text file that is read in one pass,
    so its record holds the whole file: its comments and its transforms,
    including the matrices of its linear transforms and the names of the
    displacement volumes of its grid transforms. Reading the metadata of a
    file never reads a displacement volume, which is a separate MINC file.

    The text is read and written by the record: [`from_text`][] parses an
    [`XfmRaw`][] and wraps it, and [`to_text`][] returns the text of the
    record. The metadata keeps no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".xfm",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("minc", "xfm")

    raw: NoRepr[tx.Optional[XfmRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the text of a file as an MNI transform file.

        The text is scored by [`XfmRaw.sniff_text`][].

        Parameters
        ----------
        text : str
            The text of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the text is not an MNI transform
            file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return XfmRaw.sniff_text(text, error=error, **kwargs)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the lines of a file as an MNI transform file.

        The lines are scored by [`XfmRaw.sniff_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without line endings.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines are not an MNI
            transform file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return XfmRaw.sniff_lines(lines, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the bytes of a file as an MNI transform file.

        The bytes are scored by [`XfmRaw.sniff_bytes`][].

        Parameters
        ----------
        content : bytes
            The content of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not an MNI
            transform file.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return XfmRaw.sniff_bytes(content, error=error, **kwargs)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a file as an MNI transform file.

        The file is scored by [`XfmRaw.sniff_filename`][].

        Parameters
        ----------
        filename : path-like
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an MNI transform
            file.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return XfmRaw.sniff_filename(filename, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "XfmMetadata":
        """Read the metadata from the text of an MNI transform file.

        The text is parsed by [`XfmRaw.from_text`][].

        Parameters
        ----------
        text : str
            The text of the file.
        **kwargs : Any
            Options of [`XfmRaw.from_text`][], which refuses them.

        Returns
        -------
        XfmMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(XfmRaw.from_text(text, **kwargs))

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> "XfmMetadata":
        """Read the metadata from the lines of an MNI transform file.

        The lines are parsed by [`XfmRaw.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without line endings.
        **kwargs : Any
            Options of [`XfmRaw.from_lines`][], which refuses them.

        Returns
        -------
        XfmMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(XfmRaw.from_lines(lines, **kwargs))

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> "XfmMetadata":
        """Read the metadata from the bytes of an MNI transform file.

        The bytes are parsed by [`XfmRaw.from_bytes`][].

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            Options of [`XfmRaw.from_bytes`][].

        Returns
        -------
        XfmMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(XfmRaw.from_bytes(content, **kwargs))

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "XfmMetadata":
        """Read the metadata of an MNI transform file.

        The file is read by [`XfmRaw.from_filename`][], and no displacement
        volume is read.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of [`XfmRaw.from_bytes`][].

        Returns
        -------
        XfmMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(XfmRaw.from_filename(filename, **kwargs))

    # --- to -----------------------------------------------------------

    def to_text(self, **kwargs) -> str:
        """Return the text of the MNI transform file that the record holds.

        The text is written by [`XfmRaw.to_text`][].

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record, which refuses them.

        Returns
        -------
        str
            The text of the file.

        Raises
        ------
        WriterError
            If the metadata holds no record, since a file without a
            transform is not an MNI transform file.
        """
        return _record(self).to_text(**kwargs)

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the MNI transform file that the record holds.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record, which refuses them.

        Yields
        ------
        str
            The lines of the file, without line endings.

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        return _record(self).to_lines(**kwargs)

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of the MNI transform file that the record holds.

        Parameters
        ----------
        **kwargs : Any
            Options of [`XfmRaw.to_bytes`][].

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        return _record(self).to_bytes(**kwargs)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the MNI transform file that the record holds to an open file.

        Parameters
        ----------
        file : IO
            The open file, in text or binary mode.
        **kwargs : Any
            Options of [`XfmRaw.to_bytes`][].

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        _record(self).to_fileobj(file, **kwargs)

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the MNI transform file that the record holds.

        No displacement volume is written.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of [`XfmRaw.to_bytes`][].

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        _record(self).to_filename(filename, **kwargs)


def _record(metadata: XfmMetadata) -> XfmRaw:
    """Return the record that metadata writes.

    Raises
    ------
    WriterError
        If the metadata holds no record. An empty record would make up a
        file without a transform, which libminc does not read.
    """
    if metadata.raw is None:
        raise WriterError("This metadata holds no transform to write.")
    return metadata.raw
