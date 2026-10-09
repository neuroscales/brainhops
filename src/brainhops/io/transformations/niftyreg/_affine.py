"""NiftyReg affine matrices, as written by `reg_aladin`."""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    TextFileReader,
    TextFileWriter,
    WriterError,
)
from brainhops.io.common._arrays import TxtArrayParser, is_numeric_array
from brainhops.io.transformations.base import TransformationFormat
from brainhops.io.transformations.base.affines import RASToRAS
from brainhops.io.transformations.base.conversions import (
    format_options,
    unrepresentable,
)

from ._formats import NiftyRegAffineFormat

_SHAPE = (4, 4)
"""Shape of the matrix that NiftyReg reads and writes."""


def _is_homogeneous(array: np.ndarray) -> bool:
    """Return whether an array is (4, 4) with a last row of `[0, 0, 0, 1]`."""
    if not is_numeric_array(array) or tuple(array.shape) != _SHAPE:
        return False
    last = np.asarray(array[-1], dtype=np.float64)
    return bool(np.allclose(last, [0.0, 0.0, 0.0, 1.0], atol=1e-6))


@register_format
class NiftyRegAffine(
    NiftyRegAffineFormat,
    TxtArrayParser,
    TextFileReader,
    TextFileWriter,
    RASToRAS,
    TransformationFormat,
):
    """
    Affine written by `reg_aladin -aff`.

    The file is plain text holding the four rows of a (4, 4) homogeneous
    matrix, with four whitespace-separated numbers per line
    (`reg_tool_WriteAffineFile`). NiftyReg documents the matrix as `Affine *
    Reference = Floating`: it maps the reference world to the floating world,
    which is the resampling direction of the data model, so it is read without
    inversion. The world is NIfTI RAS in millimetres, with no sign flip, and no
    image is needed to read the file.

    Nothing marks the file as NiftyReg's, so this reader scores
    `Confidence.WEAK`, below the generic matrix reader
    ([`TxtMatrixAffine`][brainhops.io.transformations.matrix.TxtMatrixAffine])
    and FLIRT. It is selected with `hint="niftyreg"`, `"niftyreg.aladin"` or
    `"aladin"`. The matrix is written in the same layout, with enough digits
    for each number to be read back exactly, whereas NiftyReg itself prints
    `%.7g`.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt",)
    HINTS = ("aladin",)
    SNIFF_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    NAMED_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    # The affine file is tiny, so a large file is not read whole only to be
    # rejected.
    SNIFF_LIMIT: tx.ClassVar[tx.Optional[int]] = 1 << 16

    # --- ArrayParser hooks --------------------------------------------

    @classmethod
    def _accepts_array(cls, array: np.ndarray) -> bool:
        return is_numeric_array(array) and array.ndim == 2

    @classmethod
    def _array_confidence(cls, array: np.ndarray, **kwargs) -> float:
        """Score `Confidence.WEAK` for a homogeneous (4, 4) matrix."""
        return cls.SNIFF_CONFIDENCE if _is_homogeneous(array) else 0.0

    @classmethod
    def from_array(
        cls, array: np.ndarray, key: tx.Optional[str] = None, **kwargs
    ) -> tx.Self:
        """
        Build an affine from the (4, 4) homogeneous matrix read from the file.
        """
        array = np.asarray(array, dtype=np.float64)
        if tuple(array.shape) != _SHAPE:
            raise ParserContentError(
                f"A NiftyReg affine is a (4, 4) matrix, not a matrix of "
                f"shape {tuple(array.shape)}."
            )
        if not _is_homogeneous(array):
            raise ParserContentError(
                f"The last row of a NiftyReg affine is [0, 0, 0, 1], not "
                f"{array[-1].tolist()}."
            )
        return cls(matrix=array[:-1], **kwargs)

    # --- writing ------------------------------------------------------

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the four lines of the file, one matrix row each."""
        matrix = self.homogeneous_matrix
        if matrix is None:
            matrix = np.eye(4)
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != _SHAPE:
            raise WriterError(
                f"A NiftyReg affine maps 3-D to 3-D, and this one has a "
                f"matrix of shape {matrix.shape}."
            )
        for row in matrix:
            yield " ".join(repr(float(value)) for value in row)


@converter
def _(t: RASToRAS, cls: tx.Type[NiftyRegAffine], **kwargs) -> NiftyRegAffine:
    """Copy an explicitly RAS-to-RAS affine into NiftyReg's format."""
    format_options(t, cls, kwargs)
    return cls(matrix=t.matrix)


@converter
def _(
    t: _xforms.Affine, cls: tx.Type[NiftyRegAffine], **kwargs
) -> NiftyRegAffine:
    """Refuse affines that do not declare the format's RAS-to-RAS map."""
    format_options(t, cls, kwargs)
    raise unrepresentable(
        t,
        cls,
        "NiftyReg stores an affine from RAS world coordinates to RAS world "
        "coordinates, and this affine does not declare those endpoints.",
    )
