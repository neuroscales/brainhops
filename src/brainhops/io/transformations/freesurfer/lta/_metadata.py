"""The metadata of an LTA file."""

__all__ = ["LtaMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import TextFileReader, TextFileWriter
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import LtaRaw


@register_format
class LtaMetadata(MetadataFormat, TextFileReader, TextFileWriter):
    """Metadata of an LTA file, which holds the file as an [`LtaRaw`][].

    An LTA file is a short text file that is read in one pass, so its
    record holds the whole file, including the matrix of the affine. The
    data of an
    [`LtaTransformation`][brainhops.io.transformations.freesurfer.lta.LtaTransformation]
    therefore lives in the record of its metadata, and reading the metadata
    of a file reads the same content as reading the transformation.

    The text is read and written by the record: [`from_lines`][] parses an
    [`LtaRaw`][] and wraps it, and [`to_lines`][] yields the lines of the
    record. The metadata keeps no open file.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".lta",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("lta",)

    raw: NoRepr[tx.Optional[LtaRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a line as the first line of an LTA file.

        The line is scored by [`LtaRaw.sniff_line`][].
        """
        return LtaRaw.sniff_line(line, error=error, **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> "LtaMetadata":
        """Read the metadata from the lines of an LTA file.

        The lines are parsed by [`LtaRaw.from_lines`][], which receives the
        keyword arguments.
        """
        return cls.from_raw(LtaRaw.from_lines(lines, **kwargs))

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the LTA file that the record holds.

        Metadata without a record is written as a new, empty record.
        """
        raw = LtaRaw() if self.raw is None else self.raw
        return raw.to_lines(**kwargs)

    def to_text(self, **kwargs) -> str:
        """Return the content of the LTA file, ending with a newline.

        The text is the text of the record, as [`LtaRaw.to_text`][] writes
        it, so that it matches the files that the transformation writes.
        """
        raw = LtaRaw() if self.raw is None else self.raw
        return raw.to_text(**kwargs)
