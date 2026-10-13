"""The metadata of a Neuroglancer coordinate transform."""

__all__ = ["NeuroglancerTransformMetadata"]

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    TextFileReader,
    TextFileWriter,
    WriterError,
)
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import NeuroglancerTransformRaw


@register_format
class NeuroglancerTransformMetadata(
    MetadataFormat, TextFileReader, TextFileWriter
):
    """Metadata of a Neuroglancer coordinate transform.

    The metadata holds the JSON object of the transform as a
    [`NeuroglancerTransformRaw`][]. The JSON is short and is read in one
    pass, so its record holds the whole transform, including its matrix,
    and reading the metadata reads the same content as reading the
    transformation.

    The text is read and written by the record: [`from_text`][] parses
    a [`NeuroglancerTransformRaw`][] and wraps it, and [`to_text`][]
    returns the text of the record. The metadata keeps no open file.

    The metadata answers to the hint `"ngtransform"` only, because the
    hint `"neuroglancer"` also names the metadata of Neuroglancer
    volumes.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".ngtransform.json",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("ngtransform",)

    raw: NoRepr[tx.Optional[NeuroglancerTransformRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score text as a Neuroglancer coordinate transform.

        The text is scored by [`NeuroglancerTransformRaw.sniff_text`][].

        Parameters
        ----------
        text : str
            The content of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the text is not a transform.
        **kwargs : Any
            Options of the reader, which do not change the score.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return NeuroglancerTransformRaw.sniff_text(text, error=error)

    @classmethod
    def from_text(cls, text: str) -> "NeuroglancerTransformMetadata":
        """Read the metadata from the JSON text of a transform.

        The text is parsed by [`NeuroglancerTransformRaw.from_text`][].

        Parameters
        ----------
        text : str
            The JSON text of the transform.

        Returns
        -------
        NeuroglancerTransformMetadata
            The metadata, which holds the record of the transform.
        """
        return cls.from_raw(NeuroglancerTransformRaw.from_text(text))

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str]
    ) -> "NeuroglancerTransformMetadata":
        """Read the metadata from the lines of the JSON text of a transform.

        Parameters
        ----------
        lines : iterable of str
            The lines, without line endings.

        Returns
        -------
        NeuroglancerTransformMetadata
            The metadata, which holds the record of the transform.
        """
        return cls.from_raw(NeuroglancerTransformRaw.from_lines(lines))

    def to_text(self) -> str:
        """Return the JSON text of the record.

        Returns
        -------
        str
            The text, as [`NeuroglancerTransformRaw.to_text`][] writes it.

        Raises
        ------
        WriterError
            If the metadata holds no record, since a transform without
            output dimensions is not valid.
        """
        return _record(self).to_text()

    def to_lines(self) -> tx.Iterator[str]:
        """Yield the lines of the JSON text of the record.

        Yields
        ------
        str
            The lines, without line endings.
        """
        return _record(self).to_lines()

    def to_fileobj(self, file: tx.IO) -> None:
        """Write the JSON text of the record to a file object.

        Parameters
        ----------
        file : file object
            The file, open in text mode.
        """
        file.write(self.to_text())


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _record(metadata: NeuroglancerTransformMetadata) -> NeuroglancerTransformRaw:
    # The record, which metadata built without one does not have.
    if metadata.raw is None:
        raise WriterError(
            "This metadata holds no Neuroglancer coordinate transform to "
            "write. A transform needs output dimensions, which only a "
            "record or a transformation gives."
        )
    return metadata.raw
