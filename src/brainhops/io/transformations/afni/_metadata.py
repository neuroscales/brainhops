"""The metadata of an AFNI matrix file."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base._utils_files import write_file
from brainhops.io.base.parsers import TextFileReader, TextFileWriter
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import AfniAffineRaw


@register_format
class AfniAffineMetadata(MetadataFormat, TextFileReader, TextFileWriter):
    """Metadata of an AFNI matrix file, which holds the whole file.

    An AFNI matrix file is a short text file that is read in one pass, so
    its record, an [`AfniAffineRaw`][], holds the whole file, including
    its matrices. The data of an
    [`AfniAffine`][brainhops.io.transformations.afni.AfniAffine]
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The metadata keeps no open file.

    The text is read and written by the record: [`from_text`][] parses an
    [`AfniAffineRaw`][] and wraps it, and [`to_text`][] returns the text
    of the record.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".aff12.1D", ".1D")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("afni", "aff12")

    raw: NoRepr[tx.Optional[AfniAffineRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a text stream holds AFNI matrices.

        The stream is scored by [`AfniAffineRaw.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The stream, open in text mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no AFNI
            matrices.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return AfniAffineRaw.sniff_fileobj(file, error=error)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that lines hold AFNI matrices.

        The lines are scored by [`AfniAffineRaw.sniff_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines hold no AFNI
            matrices.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return AfniAffineRaw.sniff_lines(lines, error=error)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "AfniAffineMetadata":
        """Read the metadata from the text of a matrix file.

        The text is parsed by [`AfniAffineRaw.from_text`][].

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            Options of the reader, which [`AfniAffineRaw.from_text`][]
            refuses.

        Returns
        -------
        AfniAffineMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(AfniAffineRaw.from_text(text, **kwargs))

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], **kwargs
    ) -> "AfniAffineMetadata":
        """Read the metadata from the lines of a matrix file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without their line breaks.
        **kwargs : Any
            Options of the reader, which [`AfniAffineRaw.from_lines`][]
            refuses.

        Returns
        -------
        AfniAffineMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(AfniAffineRaw.from_lines(lines, **kwargs))

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
        raw = AfniAffineRaw() if self.raw is None else self.raw
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
