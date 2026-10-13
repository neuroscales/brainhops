"""The metadata of an FSL FLIRT `.mat` file."""

__all__ = ["FlirtMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import TextFileReader
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import FlirtRaw


@register_format
class FlirtMetadata(MetadataFormat, TextFileReader):
    """Metadata of an FSL FLIRT `.mat` file, which holds a [`FlirtRaw`][].

    A FLIRT file is a short text file that is read in one pass, so its
    record holds the whole file, which is the FLIRT matrix. The FLIRT
    matrix of a
    [`FlirtTransform`][brainhops.io.transformations.fsl.flirt.FlirtTransform]
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The moving and reference images that place the matrix
    in world coordinates are not stored in the file, so they are options
    of the transformation and not part of its metadata.

    The text is read by the record: [`from_lines`][] parses a
    [`FlirtRaw`][] and wraps it. brainhops does not write FLIRT files, so
    the metadata cannot be written either.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("fsl", "flirt")

    raw: NoRepr[tx.Optional[FlirtRaw]] = None
    """The matrix of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the lines of a file as those of a FLIRT file.

        The lines are scored by [`FlirtRaw.sniff_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines do not hold a FLIRT
            matrix.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return FlirtRaw.sniff_lines(lines, error=error, **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> "FlirtMetadata":
        """Read the metadata from the lines of a FLIRT file.

        The lines are parsed by [`FlirtRaw.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            Options of [`FlirtRaw.from_lines`][].

        Returns
        -------
        FlirtMetadata
            The metadata, which holds the record of the file.
        """
        return cls.from_raw(FlirtRaw.from_lines(lines, **kwargs))
