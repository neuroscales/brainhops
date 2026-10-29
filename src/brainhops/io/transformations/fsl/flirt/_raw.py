"""The record of an FSL FLIRT `.mat` file, which holds its matrix."""

__all__ = ["FlirtRaw"]

import math

import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops._core._internal_api import peekable_lines
from brainhops.datamodel.metadata._internal_api import takes_error_policy
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import (
    Confidence,
    SnifferContentError,
    TextFileReader,
    TextFileWriter,
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


class FlirtRaw(Raw, TextFileReader, TextFileWriter):
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
    never changed in place, so that several objects can share it. The
    record also keeps the lines of the file, so that a file that is read
    and written again keeps its bytes (see [`to_lines`][]).
    """

    matrix: _Matrix = _IDENTITY
    """The FLIRT matrix, as a tuple of rows of floats."""

    lines: NoRepr[tx.Optional[tx.Tuple[str, ...]]] = None
    """The lines of the file that the matrix was read from.

    The lines are written back as they are while they still hold the
    matrix of the record. They are `None` for a record that was encoded
    from a matrix.
    """

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
        lines = tuple(_text_lines(lines))
        rows = _read_matrix_rows(lines)
        if not rows:
            raise SnifferContentError("Empty FLIRT matrix file.")
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            raise SnifferContentError(
                "A FLIRT matrix must have rows of equal length, but the rows "
                f"have lengths {sorted(len(row) for row in rows)}."
            )
        matrix = tuple(tuple(float(v) for v in row) for row in rows)
        return cls(matrix=matrix, lines=lines)

    # --- to -----------------------------------------------------------

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the file, without their line breaks.

        The lines that the record was read from are returned while they
        still hold its matrix. Otherwise each row of the matrix is written
        on its own line, as `write_ascii_matrix` of FSL writes it: every
        number is followed by two spaces. Each number is written with the
        fewest digits that read back as the same float, whereas FLIRT
        writes six significant digits.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer. There are none, so any option is
            refused.

        Returns
        -------
        iterator of str
            The lines of the file.

        Raises
        ------
        TypeError
            If an option is given.
        """
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.to_lines() takes no options, but "
                f"was given {', '.join(sorted(kwargs))}."
            )
        if self.lines is not None and _still_holds(self):
            return iter(self.lines)
        return iter(
            [
                "".join(_format_number(v) + "  " for v in row)
                for row in self.matrix
            ]
        )

    @takes_error_policy
    def to_text(self, **kwargs) -> str:
        """Return the text of the file, ending with a newline.

        The text is made of the lines of [`to_lines`][].

        Parameters
        ----------
        **kwargs : Any
            Options of the writer. There are none, so any option is
            refused.

        Returns
        -------
        str
            The content of the file.

        Raises
        ------
        TypeError
            If an option is given.
        """
        return "".join(line + "\n" for line in self.to_lines(**kwargs))


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


def _text_lines(lines: tx.Iterable[tx.Any]) -> tx.Iterator[str]:
    """Yield the lines of a file without their line breaks.

    A reader that peeks at lines ends them with a sentinel that is not a
    string, which ends the file.
    """
    if isinstance(lines, peekable_lines):
        lines = list(lines)
    for line in lines:
        if not isinstance(line, str):
            break
        yield line.rstrip("\r\n")


def _still_holds(record: FlirtRaw) -> bool:
    """Tell whether the lines of a record hold its matrix."""
    rows = _read_matrix_rows(list(record.lines))
    matrix = tuple(tuple(float(v) for v in row) for row in rows)
    return matrix == tuple(tuple(row) for row in record.matrix)


def _format_number(value: float) -> str:
    """Return the shortest text that reads back as the same float.

    An integral value is written without a decimal point, as a C++ stream
    writes it, so that `1.0` is written `1`.
    """
    value = float(value)
    if value == 0:
        return "-0" if math.copysign(1.0, value) < 0 else "0"
    if math.isfinite(value) and value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    return repr(value)


def _looks_like_affine(rows: tx.List[tx.List[float]]) -> bool:
    """Whether the last row of the matrix is `[0, 0, 0, 1]`."""
    last = rows[-1]
    return (
        abs(last[0]) < 1e-6
        and abs(last[1]) < 1e-6
        and abs(last[2]) < 1e-6
        and abs(last[3] - 1.0) < 1e-6
    )
