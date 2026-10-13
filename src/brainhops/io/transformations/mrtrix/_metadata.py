"""The metadata of an MRtrix linear transform file."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base._utils_files import write_file
from brainhops.io.base.parsers import TextFileReader, TextFileWriter
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import MrtrixLinearRaw


@register_format
class MrtrixLinearMetadata(MetadataFormat, TextFileReader, TextFileWriter):
    """Metadata of an MRtrix linear transform file, which holds the file.

    A linear transform file is a short text file that is read in one pass,
    so its record, an [`MrtrixLinearRaw`][], holds the whole file,
    including the matrix. The data of an
    [`MrtrixLinearTransform`][brainhops.io.transformations.mrtrix.MrtrixLinearTransform]
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The metadata keeps no open file.

    The text is read and written by the record: [`from_text`][] parses an
    [`MrtrixLinearRaw`][] and wraps it, and [`to_text`][] returns the
    text of the record.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("mrtrix",)

    raw: NoRepr[tx.Optional[MrtrixLinearRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a text stream holds a transform.

        The stream is scored by [`MrtrixLinearRaw.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The stream, open in text mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no transform
            written by MRtrix.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return MrtrixLinearRaw.sniff_fileobj(file, error=error)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that lines hold a transform of MRtrix.

        The lines are scored by [`MrtrixLinearRaw.sniff_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines hold no transform
            written by MRtrix.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return MrtrixLinearRaw.sniff_lines(lines, error=error)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "MrtrixLinearMetadata":
        """Read the metadata from the text of a linear transform file.

        The text is parsed by [`MrtrixLinearRaw.from_text`][].

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            Options of the reader, which [`MrtrixLinearRaw.from_text`][]
            refuses.

        Returns
        -------
        MrtrixLinearMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(MrtrixLinearRaw.from_text(text, **kwargs))

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], **kwargs
    ) -> "MrtrixLinearMetadata":
        """Read the metadata from the lines of a linear transform file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without their line breaks.
        **kwargs : Any
            Options of the reader, which [`MrtrixLinearRaw.from_lines`][]
            refuses.

        Returns
        -------
        MrtrixLinearMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(MrtrixLinearRaw.from_lines(lines, **kwargs))

    def to_text(self, **kwargs) -> str:
        """Return the content of the file that the record holds.

        Metadata without a record is written as a new record, which holds
        the identity.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record, which it refuses.

        Returns
        -------
        str
            The content of the file, ending with a newline.
        """
        raw = MrtrixLinearRaw() if self.raw is None else self.raw
        return raw.to_text(**kwargs)

    def to_lines(self, **kwargs) -> tx.List[str]:
        """Return the lines of the file that the record holds.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record, which it refuses.

        Returns
        -------
        list of str
            The lines of the file, without their line breaks.
        """
        return self.to_text(**kwargs).splitlines()

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the content of the file to a text stream.

        Parameters
        ----------
        file : IO
            The stream, open in text mode.
        **kwargs : Any
            Options of the writer of the record, which it refuses.
        """
        file.write(self.to_text(**kwargs))

    def to_filename(self, filename: tx.Any, **kwargs) -> None:
        """Write the content of the file to a path.

        A local file is replaced at once.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of the writer of the record, which it refuses.
        """
        write_file(filename, self.to_bytes(**kwargs))
