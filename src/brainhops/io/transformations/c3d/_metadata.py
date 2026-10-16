"""The metadata of a Convert3D matrix file."""

__all__ = ["C3dAffineMetadata"]

import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    TextFileReader,
    TextFileWriter,
    WriterError,
)
from brainhops.io.metadata import MetadataFormat

from ._raw import C3dAffineRaw


@register_format
class C3dAffineMetadata(MetadataFormat, TextFileReader, TextFileWriter):
    """Metadata of a Convert3D matrix file, which holds the file as a record.

    A matrix file of Greedy, `c3d_affine_tool` or ITK-SNAP is a short
    text file that is read in one pass, so its record, a
    [`C3dAffineRaw`][], holds the whole file, including the matrix. The
    data of a
    [`C3dAffine`][brainhops.io.transformations.c3d.C3dAffine] therefore
    lives in the record of its metadata, and reading the metadata of a
    file reads the same content as reading the transformation. The
    metadata keeps no open file.

    The content of the file holds no marker, so the metadata is never
    recognized from it. It is read from a file with the `.mat` extension
    or from a file selected by a hint.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("c3d", "greedy", "itksnap")
    RAW: tx.ClassVar[type] = C3dAffineRaw

    raw: NoRepr[tx.Optional[C3dAffineRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Decline to recognize a Convert3D matrix from its lines.

        The lines are scored by [`C3dAffineRaw.sniff_lines`][], which
        always declines.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error instead of returning a confidence.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            `Confidence.NO`.
        """
        return C3dAffineRaw.sniff_lines(lines, error=error, **kwargs)

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Decline to recognize a Convert3D matrix from its content.

        The text is scored by [`C3dAffineRaw.sniff_text`][], which always
        declines.

        Parameters
        ----------
        text : str
            The content of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error instead of returning a confidence.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            `Confidence.NO`.
        """
        return C3dAffineRaw.sniff_text(text, error=error, **kwargs)

    def to_text(self, **kwargs) -> str:
        """Return the content of the file that the record holds.

        Parameters
        ----------
        **kwargs : Any
            Options of [`C3dAffineRaw.to_text`][], which takes none.

        Returns
        -------
        str
            The content of the file.

        Raises
        ------
        WriterError
            If the metadata has no record, since the file is made of the
            matrix alone.
        """
        return _record(self).to_text(**kwargs)

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the file that the record holds.

        Parameters
        ----------
        **kwargs : Any
            Options of [`C3dAffineRaw.to_lines`][], which takes none.

        Yields
        ------
        str
            The lines of the file, without line breaks.

        Raises
        ------
        WriterError
            If the metadata has no record.
        """
        return _record(self).to_lines(**kwargs)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _record(metadata: C3dAffineMetadata) -> C3dAffineRaw:
    """Return the record of metadata that is written.

    Raises
    ------
    WriterError
        If the metadata has no record.
    """
    if metadata.raw is None:
        raise WriterError(
            "Metadata of a Convert3D matrix without a record cannot be "
            "written, because the file holds the matrix and nothing else. "
            "Write the transformation instead."
        )
    return metadata.raw
