"""The metadata of a `register.dat` file."""

__all__ = ["RegisterDatMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import TextFileReader, TextFileWriter
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import RegisterDatRaw


@register_format
class RegisterDatMetadata(MetadataFormat, TextFileReader, TextFileWriter):
    """Metadata of a `register.dat` file, which holds the file as a record.

    A `register.dat` file is a short text file that is read in one pass,
    so its record, a [`RegisterDatRaw`][], holds the whole file, including
    the registration matrix. The data of a
    [`RegisterDatTransformation`][brainhops.io.transformations.freesurfer.regdat.RegisterDatTransformation]
    therefore lives in the record of its metadata, and reading the metadata
    of a file reads the same content as reading the transformation.

    The text is read and written by the record: [`from_lines`][] parses a
    [`RegisterDatRaw`][] and wraps it, and [`to_lines`][] yields the lines
    of the record. The metadata keeps no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".dat", ".reg")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("regdat",)
    RAW: tx.ClassVar[type] = RegisterDatRaw

    raw: NoRepr[tx.Optional[RegisterDatRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the lines of a file as those of a `register.dat` file.

        The lines are scored by [`RegisterDatRaw.sniff_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines are not those of a
            `register.dat` file.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return RegisterDatRaw.sniff_lines(lines, error=error, **kwargs)

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the `register.dat` file that the record holds.

        Metadata without a record is written as a new record, which holds
        the identity.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record.

        Yields
        ------
        str
            The lines of the file, without line endings.
        """
        raw = RegisterDatRaw() if self.raw is None else self.raw
        return raw.to_lines(**kwargs)

    def to_text(self, **kwargs) -> str:
        """Return the content of the `register.dat` file.

        The text is the text of the record, as [`RegisterDatRaw.to_text`][]
        writes it, so that it matches the files that the transformation
        writes.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record.

        Returns
        -------
        str
            The content of the file, ending with a newline.
        """
        raw = RegisterDatRaw() if self.raw is None else self.raw
        return raw.to_text(**kwargs)
