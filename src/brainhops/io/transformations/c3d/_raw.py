"""The record of a Convert3D matrix file, which holds the whole file."""

__all__ = ["C3dAffineRaw"]

import math

import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops.datamodel.metadata._report import takes_error_policy
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
    TextFileReader,
    TextFileWriter,
)

_Matrix = tx.Tuple[tx.Tuple[float, ...], ...]


class C3dAffineRaw(Raw, TextFileReader, TextFileWriter):
    """Record of a Convert3D matrix file, which holds the file as parsed.

    Greedy, `c3d_affine_tool` and ITK-SNAP save an affine as a text file
    of sixteen numbers, the rows of a homogeneous (4, 4) matrix in RAS
    world coordinates. The file is short and is read in one pass, so the
    record holds the whole file: the matrix, the characters that end each
    row, and the text that the matrix was read from. The metadata of the
    file,
    [`C3dAffineMetadata`][brainhops.io.transformations.c3d.C3dAffineMetadata],
    holds a record, and a
    [`C3dAffine`][brainhops.io.transformations.c3d.C3dAffine] reads its
    matrix from the record of its metadata. A record is never changed in
    place, so that several objects can share it.

    The three programs read the numbers in the same way, as sixteen
    whitespace-separated values, but they write them in two layouts. Each
    of them prints the numbers with the default precision of a C++
    stream, which is six significant digits. Greedy writes a `vnl_matrix`
    with its `operator<<`, which puts a space after every number,
    including the last number of a row. `c3d_affine_tool -o` and ITK-SNAP
    separate the numbers of a row with single spaces and end the row
    without one. The record keeps the layout in `row_end`, so that a
    matrix encoded again is written in the layout of the file that it
    replaces.

    !!! note "Sources"
        The reader and writer of `c3d_affine_tool` are `ras_read` and
        `ras_write` in `utilities/AffineTransformTool.cxx` of
        <https://github.com/pyushkevich/c3d> (commit `16ea136`). Greedy
        reads and writes the file in `ReadAffineMatrixViaCache` and
        `WriteAffineMatrixViaCache` of `src/GreedyAPI.cxx` in
        <https://github.com/pyushkevich/greedy> (commit `f5212c5`).
        ITK-SNAP does so in `ReadAsRASMatrix` and `WriteAsRASMatrix` of
        `Common/AffineTransformHelper.cxx` in
        <https://github.com/pyushkevich/itksnap> (commit `cc07850`). The
        two ways of printing a matrix are `operator<<` of `vnl_matrix`
        and `print` of `vnl_matrix_fixed`, in `core/vnl/vnl_matrix.hxx`
        and `core/vnl/vnl_matrix_fixed.hxx` of
        <https://github.com/vxl/vxl> (commit `265dbf6`).
    """

    matrix: _Matrix = ()
    """The homogeneous (4, 4) matrix, as a tuple of rows of floats."""

    row_end: str = ""
    """The characters that end each row before its line break.

    They are a space in the files of Greedy and nothing in the files of
    `c3d_affine_tool` and ITK-SNAP.
    """

    text: NoRepr[tx.Optional[str]] = None
    """The text of the file that the matrix was read from.

    The text is written back as it is while it still describes the
    matrix and the layout of the record, so that a file that is read and
    written again keeps its bytes, including the digits of its numbers.
    Otherwise the record is formatted again. The text is `None` for a
    record that was encoded from a matrix.
    """

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Decline to recognize a Convert3D matrix from its content.

        A Convert3D matrix file holds sixteen numbers and nothing else, so
        its content cannot tell it from a FLIRT matrix, a NiftyReg affine
        or any other plain matrix. The format is therefore never chosen
        from the content, and is selected by a hint such as
        `hint="greedy"` instead.

        Parameters
        ----------
        text : str
            The content of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error instead of returning a confidence.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `Confidence.NO`.
        """
        if error:
            if error is True:
                error = SnifferContentError
            raise error(
                "A Convert3D matrix holds sixteen numbers and no marker, so "
                "it is not recognized from its content. Select it with "
                'hint="c3d", hint="greedy" or hint="itksnap".'
            )
        return Confidence.NO

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Decline to recognize a Convert3D matrix from its lines.

        See [`sniff_text`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error instead of returning a confidence.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `Confidence.NO`.
        """
        return cls.sniff_text("", error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "C3dAffineRaw":
        """Parse the text of a Convert3D matrix file, which the record keeps.

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            No option is accepted.

        Returns
        -------
        C3dAffineRaw
            The record of the file.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the text does not hold the sixteen numbers of a homogeneous
            (4, 4) matrix.
        """
        _no_options(cls, "from_text", kwargs)
        return cls(
            matrix=_parse_matrix(text), row_end=_row_end(text), text=text
        )

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> "C3dAffineRaw":
        """Parse the lines of a Convert3D matrix file.

        A line without a line break is given one, so that the record keeps
        the text of a file whose lines were split.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            No option is accepted.

        Returns
        -------
        C3dAffineRaw
            The record of the file.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the lines do not hold the sixteen numbers of a homogeneous
            (4, 4) matrix.
        """
        _no_options(cls, "from_lines", kwargs)
        text = "".join(
            line if line.endswith("\n") else line + "\n" for line in lines
        )
        return cls.from_text(text)

    # --- to -----------------------------------------------------------

    @takes_error_policy
    def to_text(self, **kwargs) -> str:
        """Return the text of the file, ending with a newline.

        The text that the record was read from is returned while it still
        describes the matrix and the layout of the record. Otherwise each
        row of the matrix is written on its own line, with its numbers
        separated by single spaces and followed by `row_end`. Each number
        is written with the fewest digits that read back as the same
        float, so the file keeps the precision of the matrix, whereas
        Greedy, `c3d_affine_tool` and ITK-SNAP keep six significant
        digits.

        Parameters
        ----------
        **kwargs : Any
            No option is accepted.

        Returns
        -------
        str
            The content of the file.

        Raises
        ------
        TypeError
            If an option is given.
        """
        _no_options(type(self), "to_text", kwargs)
        if self.text is not None and _still_describes(self):
            return self.text
        return "".join(
            " ".join(_format_number(value) for value in row)
            + self.row_end
            + "\n"
            for row in self.matrix
        )

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the file, without their line breaks.

        The lines are those of [`to_text`][].

        Parameters
        ----------
        **kwargs : Any
            No option is accepted.

        Yields
        ------
        str
            The lines of the file.
        """
        yield from self.to_text(**kwargs).splitlines()


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _no_options(cls: type, method: str, options: tx.Dict[str, tx.Any]) -> None:
    """Refuse the options that a reader or a writer does not take."""
    if options:
        raise TypeError(
            f"{cls.__name__}.{method}() takes no options, but was given "
            f"{', '.join(sorted(options))}."
        )


def _parse_matrix(text: str) -> _Matrix:
    """Read the homogeneous (4, 4) matrix that a file holds.

    The numbers are read as `c3d_affine_tool` reads them, separated by
    any whitespace. Exactly sixteen numbers are accepted, so a 2-D matrix
    of nine numbers is refused, and the last row must be `0 0 0 1`, so
    that the matrix is the matrix of an affine.

    Raises
    ------
    ParserContentError
        If the text does not hold such a matrix.
    """
    tokens = text.split()
    try:
        values = [float(token) for token in tokens]
    except ValueError as e:
        raise ParserContentError(
            f"A Convert3D matrix holds numbers only: {e}."
        ) from e
    if len(values) != 16:
        raise ParserContentError(
            f"A Convert3D matrix holds the sixteen numbers of a (4, 4) "
            f"matrix, but the text holds {len(values)}. Matrices of 2-D "
            f"transformations are not supported."
        )
    matrix = tuple(tuple(values[4 * i : 4 * i + 4]) for i in range(4))
    last = matrix[3]
    if any(abs(a - b) > 1e-6 for a, b in zip(last, (0.0, 0.0, 0.0, 1.0))):
        raise ParserContentError(
            f"The last row of a Convert3D affine is 0 0 0 1, not "
            f"{' '.join(repr(v) for v in last)}."
        )
    return matrix


def _row_end(text: str) -> str:
    """Return the characters that end the rows of a file.

    They are a space when every line that holds numbers ends with one, as
    in the files of Greedy, and nothing otherwise.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and all(line.endswith(" ") for line in lines):
        return " "
    return ""


def _still_describes(record: C3dAffineRaw) -> bool:
    """Tell whether the text of a record holds its matrix and layout."""
    try:
        matrix = _parse_matrix(record.text)
    except ParserContentError:
        return False
    return (
        matrix == tuple(tuple(row) for row in record.matrix)
        and _row_end(record.text) == record.row_end
    )


def _format_number(value: float) -> str:
    """Return the shortest text that reads back as the same float.

    An integral value is written without a decimal point, as a C++ stream
    writes it, so that `1.0` is written `1` and `-0.0` is written `-0`.
    """
    value = float(value)
    if value == 0:
        return "-0" if math.copysign(1.0, value) < 0 else "0"
    if math.isfinite(value) and value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    return repr(value)
