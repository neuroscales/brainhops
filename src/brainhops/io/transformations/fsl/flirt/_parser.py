# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Alias, Factory, Magic

from brainhops._core.peek import peekable_lines
from brainhops._core.typing import ArrayLike

# datamodel
from brainhops.datamodel.images import Image

# core
from brainhops.datamodel.metadata import MetadataField

# io
from brainhops.io.base.arrays import ArrayContainerError, read_text_rows
from brainhops.io.base.nifti import _NiftiObject
from brainhops.io.base.parsers import (
    Confidence,
    SnifferContentError,
    TextFileParser,
)
from brainhops.io.transformations.base._metadata import (
    sync_metadata,
)

# locals
from ._metadata import FlirtMetadata

# The moving and reference images may be a nibabel header or image, or a
# brainhops image. This is the type FLIRT accepts for either of them.
_ImageLike = tx.Union[_NiftiObject, Image]


class FlirtMatrixParser(Magic, TextFileParser, repr=HIDE_IF_NONE):
    """Reader for a FLIRT `.mat` file.

    A FLIRT `.mat` file is a plain text `(4, 4)` affine matrix, one row
    per line, whitespace separated. The matrix alone carries no image
    geometry, so the reference and moving images must be supplied for the
    matrix to be turned into a world-space transformation.
    """

    flirt_matrix: tx.Optional[ArrayLike] = None
    """The raw `(4, 4)` FLIRT matrix, as read from the file.

    An array-like of shape `(4, 4)` mapping moving-image scaled-mm
    coordinates to reference-image scaled-mm coordinates. It is converted
    to a NumPy array and used to build the world-space affine.
    """

    moving: tx.Annotated[
        tx.Optional[_ImageLike], Alias(("moving", "mov", "src"))
    ] = None
    """The moving (source) image, a nibabel image or header, or a
    brainhops image."""

    reference: tx.Annotated[
        tx.Optional[_ImageLike], Alias(("reference", "ref"))
    ] = None
    """The reference image, a nibabel image or header, or a brainhops
    image."""

    metadata: MetadataField[
        FlirtMetadata,
        Factory(FlirtMetadata),
        tx.Doc(
            """
            The paths of the moving and reference images, when they were
            read from files (`moving`, `fixed`). A `.mat` file stores no
            metadata. See
            [`FlirtMetadata`][brainhops.io.transformations.fsl.flirt.FlirtMetadata].
            """
        ),
    ]

    def __post_init__(self) -> None:
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        sync_metadata(self, FlirtMetadata, None, image=self)

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
    """Read whitespace-separated float rows, skipping blank lines.

    The parsing is shared with the other plain-matrix readers
    (`brainhops.io.base.arrays`). FLIRT writes whitespace-separated
    values and no comments, so only those are accepted here. A
    non-numeric value makes the whole content unreadable (`[]`).
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
    """Whether the last row of a `(4, 4)` matrix is `[0, 0, 0, 1]`."""
    last = rows[-1]
    return (
        abs(last[0]) < 1e-6
        and abs(last[1]) < 1e-6
        and abs(last[2]) < 1e-6
        and abs(last[3] - 1.0) < 1e-6
    )
