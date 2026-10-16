"""The metadata of a NiftyReg affine file."""

__all__ = ["NiftyRegAffineMetadata"]

import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, TextFileWriter, WriterError
from brainhops.io.common._arrays import TxtArrayReader, is_numeric_array
from brainhops.io.metadata import MetadataFormat

from ._raw import NiftyRegAffineRaw, is_homogeneous


@register_format
class NiftyRegAffineMetadata(MetadataFormat, TxtArrayReader, TextFileWriter):
    """Metadata of a NiftyReg affine file, which holds the file as a record.

    An affine file of `reg_aladin` is a short text file that is read in
    one pass, so its record, a [`NiftyRegAffineRaw`][], holds the whole
    file, including the matrix. The data of a
    [`NiftyRegAffine`][brainhops.io.transformations.niftyreg.NiftyRegAffine]
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The metadata keeps no open file.

    Nothing marks the file as NiftyReg's, so the metadata scores
    `Confidence.WEAK`, as the transformation does, and loses a `.txt` file
    to the metadata of a plain matrix unless it is selected with
    `hint="niftyreg"` or `hint="aladin"`.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("niftyreg", "aladin")
    RAW: tx.ClassVar[type] = NiftyRegAffineRaw
    SNIFF_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    NAMED_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    # The affine file is tiny, so a large file is not read whole only to be
    # rejected.
    SNIFF_LIMIT: tx.ClassVar[tx.Optional[int]] = 1 << 16

    raw: NoRepr[tx.Optional[NiftyRegAffineRaw]] = None
    """The content of the file, or `None` for metadata without a record."""

    # --- ArrayReader hooks --------------------------------------------

    @classmethod
    def _accepts_array(cls, array: np.ndarray) -> bool:
        return is_numeric_array(array) and array.ndim == 2

    @classmethod
    def _array_confidence(cls, array: np.ndarray, **kwargs) -> float:
        """Score `Confidence.WEAK` for a homogeneous (4, 4) matrix."""
        return cls.SNIFF_CONFIDENCE if is_homogeneous(array) else 0.0

    # --- to -----------------------------------------------------------

    def to_text(self, **kwargs) -> str:
        """Return the content of the file that the record holds.

        Parameters
        ----------
        **kwargs : Any
            Options of [`NiftyRegAffineRaw.to_text`][], which takes none.

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
            Options of [`NiftyRegAffineRaw.to_lines`][], which takes none.

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


def _record(metadata: NiftyRegAffineMetadata) -> NiftyRegAffineRaw:
    """Return the record of metadata that is written.

    Raises
    ------
    WriterError
        If the metadata has no record.
    """
    if metadata.raw is None:
        raise WriterError(
            "Metadata of a NiftyReg affine without a record cannot be "
            "written, because the file holds the matrix and nothing else. "
            "Write the transformation instead."
        )
    return metadata.raw
