import numpy as np
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Alias, Magic

from brainhops._core.peek import peekable_lines
from brainhops._core.typing import ArrayLike
from brainhops.datamodel.images import Image
from brainhops.io.base.parsers import (
    Confidence,
    SnifferContentError,
    TextFileParser,
)

# io
from brainhops.io.common._arrays import ArrayContainerError, read_text_rows
from brainhops.io.common.nifti._header import _NiftiObject

# A nibabel image or header, or a brainhops image.
_ImageLike = tx.Union[_NiftiObject, Image]


class FlirtMatrixParser(Magic, TextFileParser, repr=HIDE_IF_NONE):
    """Parser for FSL FLIRT `.mat` files.

    The file holds a plain-text (4, 4) matrix, one row per line. The matrix
    carries no image geometry, so the reference and moving images are needed to
    turn it into a world-space transform.
    """

    flirt_matrix: tx.Optional[ArrayLike] = None
    """The raw FLIRT matrix, from moving to reference scaled millimetres."""

    moving: tx.Annotated[
        tx.Optional[_ImageLike], Alias(("moving", "mov", "src"))
    ] = None
    """The moving (source) image."""

    reference: tx.Annotated[
        tx.Optional[_ImageLike], Alias(("reference", "ref"))
    ] = None
    """The reference image."""

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        rows = _read_matrix_rows(lines)
        if (
            len(rows) == 4
            and all(len(row) == 4 for row in rows)
            and _looks_like_affine(rows)
        ):
            return Confidence.LIKELY
        if error:
            if error is True:
                error = SnifferContentError
            raise error("Not a FLIRT (4, 4) matrix.")
        return Confidence.NO

    # --- from ---------------------------------------------------------

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        moving = kwargs.pop("moving", None)
        if moving is None:
            moving = kwargs.pop("src", None)
        reference = kwargs.pop("reference", None)
        if reference is None:
            reference = kwargs.pop("ref", None)
        rows = _read_matrix_rows(lines)
        if not rows:
            raise SnifferContentError("Empty FLIRT matrix file.")
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            raise SnifferContentError(
                "A FLIRT matrix must have rows of equal length, but the rows "
                f"have lengths {sorted(len(row) for row in rows)}."
            )
        return cls(
            flirt_matrix=np.array(rows, dtype=np.float64),
            moving=moving,
            reference=reference,
        )


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


def _read_matrix_rows(lines: tx.Iterable[str]) -> tx.List[tx.List[float]]:
    """Read whitespace-separated rows of floats, skipping blank lines.

    FLIRT writes no comments, so none are recognised. Content with a
    non-numeric value yields an empty list.
    """
    if isinstance(lines, peekable_lines):
        lines = list(lines)
    text_lines = []
    for line in lines:
        if not isinstance(line, str):
            break
        text_lines.append(line)
    try:
        return read_text_rows(text_lines, comments=None, separators=r"\s+")
    except ArrayContainerError:
        return []


def _looks_like_affine(rows: tx.List[tx.List[float]]) -> bool:
    """Whether the last row of the matrix is `[0, 0, 0, 1]`."""
    last = rows[-1]
    return (
        abs(last[0]) < 1e-6
        and abs(last[1]) < 1e-6
        and abs(last[2]) < 1e-6
        and abs(last[3] - 1.0) < 1e-6
    )
