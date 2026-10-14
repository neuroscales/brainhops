"""The record of an FSL FLIRT `.mat` file, which holds its matrix."""

__all__ = ["FlirtRaw"]

import typing_extensions as tx
from bagof.magic import Magic

from brainhops._core.peek import peekable_lines
from brainhops.io.base.parsers import (
    Confidence,
    SnifferContentError,
    TextFileReader,
)

# io
from brainhops.io.common._arrays import ArrayContainerError, read_text_rows

_Matrix = tx.Tuple[tx.Tuple[float, ...], ...]

_IDENTITY: _Matrix = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class FlirtRaw(Magic, TextFileReader):
    """Record of an FSL FLIRT `.mat` file, which holds its matrix as read.

    A FLIRT `.mat` file is a plain-text matrix, one row per line, which
    maps the scaled millimetres of the moving image to those of the
    reference image. The file is read in one pass, so the record holds the
    whole file: its matrix, as a tuple of rows of floats. The file holds no
    image geometry, so the record holds none either.

    The metadata of a FLIRT file,
    [`FlirtMetadata`][brainhops.io.transformations.fsl.flirt.FlirtMetadata],
    holds a record, and a
    [`FlirtTransform`][brainhops.io.transformations.fsl.flirt.FlirtTransform]
    reads its FLIRT matrix from the record of its metadata. A record is
    never changed in place, so that several objects can share it.
    brainhops does not write FLIRT files, so the record has no writer and
    does not keep the text of the file.
    """

    matrix: _Matrix = _IDENTITY
    """The FLIRT matrix, as a tuple of rows of floats."""

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the likelihood that lines hold a FLIRT matrix.

        The lines must hold four rows of four numbers, whose last row is
        `0 0 0 1`, and they then score `Confidence.LIKELY`.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines do not hold a FLIRT
            matrix. With `True`, the error is a `SnifferContentError`.
        **kwargs : Any
            Options of the reader, which are not used.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
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
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> "FlirtRaw":
        """Read the matrix of a FLIRT file from its lines.

        The rows are separated by white space, and blank lines are
        skipped. FLIRT writes no comments, so none are recognised.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            Options of the reader. There are none, so any option is
            refused.

        Returns
        -------
        FlirtRaw
            The record, which holds the matrix of the file.

        Raises
        ------
        TypeError
            If an option is given.
        SnifferContentError
            If the lines hold no matrix, or rows of different lengths.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__}.from_lines() takes no options, but was "
                f"given {', '.join(sorted(kwargs))}."
            )
        rows = _read_matrix_rows(lines)
        if not rows:
            raise SnifferContentError("Empty FLIRT matrix file.")
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            raise SnifferContentError(
                "A FLIRT matrix must have rows of equal length, but the rows "
                f"have lengths {sorted(len(row) for row in rows)}."
            )
        return cls(matrix=tuple(tuple(float(v) for v in row) for row in rows))


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
