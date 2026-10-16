"""The metadata of an ITK text `.tfm` file."""

__all__ = ["TfmMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import TextFileReader
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import TfmRaw


@register_format
class TfmMetadata(MetadataFormat, TextFileReader):
    """Metadata of an ITK text `.tfm` file, which holds it as a [`TfmRaw`][].

    A `.tfm` file is a short text file that is read in one pass, so its
    record holds every block of the file, parameters included. The chain
    of a
    [`TfmTransform`][brainhops.io.transformations.itk.tfm.TfmTransform]
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The file stores nothing else, such as a name or a
    version of ITK.

    The text is read by the record: [`from_lines`][] parses a [`TfmRaw`][]
    and wraps it. brainhops does not write `.tfm` files, so the metadata
    cannot be written either.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tfm",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("itk", "ants", "tfm")
    RAW: tx.ClassVar[type] = TfmRaw

    raw: NoRepr[tx.Optional[TfmRaw]] = None
    """The blocks of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a line as the first block of a `.tfm` file.

        The line is scored by [`TfmRaw.sniff_line`][].

        Parameters
        ----------
        line : str
            The first line of the file that is not a comment.
        error : bool or type[Exception], default=False
            Whether to raise an error when the line does not start a
            block.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return TfmRaw.sniff_line(line, error=error, **kwargs)
