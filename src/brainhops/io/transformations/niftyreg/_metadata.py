"""The metadata of a NiftyReg affine file."""

__all__ = ["NiftyRegAffineMetadata"]

import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, WriterError
from brainhops.io.common._arrays import TxtArrayReader, is_numeric_array
from brainhops.io.metadata import MetadataFormat, MetadataWriter

from ._raw import NiftyRegAffineRaw, is_homogeneous


@register_format
class NiftyRegAffineMetadata(MetadataFormat, TxtArrayReader, MetadataWriter):
    """Metadata of a NiftyReg affine file, which holds the file as a record.

    An affine file of `reg_aladin` is a short text file that is read in
    one pass, so its record, a [`NiftyRegAffineRaw`][], holds the whole
    file, including the matrix. The data of a
    [`NiftyRegAffine`][brainhops.io.transformations.niftyreg.NiftyRegAffine]
    therefore lives in the record of its metadata, and reading the
    metadata of a file reads the same content as reading the
    transformation. The metadata keeps no open file.

    The file is scored as a text array by
    [`TxtArrayReader`][brainhops.io.common._arrays.TxtArrayReader],
    because [`NiftyRegAffineRaw`][] has no sniffer, but its text is read
    by the record. The file is written by the record, through
    [`MetadataWriter`][brainhops.io.metadata.MetadataWriter].

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

    # --- from ---------------------------------------------------------
    # The record class has no sniffer, so the metadata cannot list
    # `MetadataReader`, and passes the text to the record here. Every
    # other input reaches one of these two methods through the adapters
    # of `TxtArrayReader`.

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Read the metadata from the text of a file, through the record.

        Parameters
        ----------
        text : str
            The text of the file.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        Self
            The metadata.
        """
        return cls.from_raw(NiftyRegAffineRaw.from_text(text, **kwargs))

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Read the metadata from the lines of a file, through the record.

        Parameters
        ----------
        lines : Iterable[str]
            The lines of the file.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        Self
            The metadata.
        """
        return cls.from_raw(NiftyRegAffineRaw.from_lines(lines, **kwargs))

    # --- to -----------------------------------------------------------

    def _record_to_write(self) -> NiftyRegAffineRaw:
        """Return the record that the metadata writes.

        Returns
        -------
        NiftyRegAffineRaw
            The record.

        Raises
        ------
        WriterError
            If the metadata has no record.
        """
        if self.raw is None:
            raise WriterError(
                "Metadata of a NiftyReg affine without a record cannot be "
                "written, because the file holds the matrix and nothing else. "
                "Write the transformation instead."
            )
        return self.raw
