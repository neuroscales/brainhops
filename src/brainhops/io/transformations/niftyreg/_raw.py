"""The record of a NiftyReg affine file, which holds the whole file."""

__all__ = ["NiftyRegAffineRaw"]

import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops.datamodel.metadata._report import takes_error_policy
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import (
    ParserContentError,
    TextFileReader,
    TextFileWriter,
)
from brainhops.io.common._arrays import (
    ArrayContainerError,
    TxtArrayReader,
    read_text_array,
)

_Matrix = tx.Tuple[tx.Tuple[float, ...], ...]

_SHAPE = (4, 4)
"""Shape of the matrix that NiftyReg reads and writes."""


class NiftyRegAffineRaw(Raw, TextFileReader, TextFileWriter):
    """Record of a NiftyReg affine file, which holds the file as parsed.

    `reg_aladin -aff` saves an affine as a text file that holds the four
    rows of a homogeneous (4, 4) matrix in RAS world coordinates. The file
    is short and is read in one pass, so the record holds the whole file:
    the matrix and the text that it was read from. The metadata of the
    file,
    [`NiftyRegAffineMetadata`][brainhops.io.transformations.niftyreg.NiftyRegAffineMetadata],
    holds a record, and a
    [`NiftyRegAffine`][brainhops.io.transformations.niftyreg.NiftyRegAffine]
    reads its matrix from the record of its metadata. A record is never
    changed in place, so that several objects can share it.

    NiftyReg writes each row as four numbers printed with `%.7g` and
    separated by single spaces, and reads the file as a stream of sixteen
    numbers (`reg_tool_WriteAffineFile` and `reg_tool_ReadAffineFile` in
    `reg-io/_reg_ReadWriteMatrix.cpp` of
    <https://github.com/KCL-BMEIS/niftyreg>, commit `79a1762`). The record
    is read as whitespace-separated text whose comments start with `#`, as
    the reader of plain text matrices reads it, so it accepts every file
    that NiftyReg writes.
    """

    matrix: _Matrix = ()
    """The homogeneous (4, 4) matrix, as a tuple of rows of floats."""

    text: NoRepr[tx.Optional[str]] = None
    """The text of the file that the matrix was read from.

    The text is written back as it is while it still describes the
    matrix of the record, so that a file that is read and written again
    keeps its bytes, including the digits of its numbers. Otherwise the
    record is formatted again. The text is `None` for a record that was
    encoded from a matrix.
    """

    # --- from ---------------------------------------------------------

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "NiftyRegAffineRaw":
        """Parse the text of a NiftyReg affine file, which the record keeps.

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            No option is accepted.

        Returns
        -------
        NiftyRegAffineRaw
            The record of the file.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the text does not hold a homogeneous (4, 4) matrix.
        """
        _no_options(cls, "from_text", kwargs)
        return cls(matrix=_parse_matrix(text), text=text)

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], **kwargs
    ) -> "NiftyRegAffineRaw":
        """Parse the lines of a NiftyReg affine file.

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
        NiftyRegAffineRaw
            The record of the file.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the lines do not hold a homogeneous (4, 4) matrix.
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
        describes the matrix of the record. Otherwise each row of the
        matrix is written on its own line, with its numbers separated by
        single spaces. Each number is written with the fewest digits that
        read back as the same float, so the file keeps the precision of the
        matrix, whereas NiftyReg prints `%.7g`.

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
            " ".join(repr(float(value)) for value in row) + "\n"
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

    Raises
    ------
    ParserContentError
        If the text is not a (4, 4) matrix whose last row is `0 0 0 1`.
    """
    try:
        array = read_text_array(
            text.splitlines(), separators=TxtArrayReader.SEPARATORS
        )
    except ArrayContainerError as e:
        raise ParserContentError(str(e)) from e
    if tuple(array.shape) != _SHAPE:
        raise ParserContentError(
            f"A NiftyReg affine is a (4, 4) matrix, not a matrix of "
            f"shape {tuple(array.shape)}."
        )
    if not is_homogeneous(array):
        raise ParserContentError(
            f"The last row of a NiftyReg affine is [0, 0, 0, 1], not "
            f"{array[-1].tolist()}."
        )
    return tuple(tuple(float(v) for v in row) for row in array)


def is_homogeneous(array: np.ndarray) -> bool:
    """Return whether a (4, 4) matrix has a last row of `[0, 0, 0, 1]`."""
    if tuple(np.shape(array)) != _SHAPE:
        return False
    last = np.asarray(array[-1], dtype=np.float64)
    return bool(np.allclose(last, [0.0, 0.0, 0.0, 1.0], atol=1e-6))


def _still_describes(record: NiftyRegAffineRaw) -> bool:
    """Tell whether the text of a record holds the matrix of the record."""
    try:
        matrix = _parse_matrix(record.text)
    except ParserContentError:
        return False
    return matrix == tuple(tuple(row) for row in record.matrix)
