"""The record of an AFNI matrix file, which holds the whole file as it is
parsed."""

# stdlib
import os

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core._internal_api import preserve_position
from brainhops.datamodel.metadata._report import takes_error_policy
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
    TextFileReader,
    TextFileWriter,
)
from brainhops.io.common._arrays import (
    ArrayContainerError,
    is_numeric_array,
    read_text_array,
)

_SIGNATURE = "DICOM-to-DICOM"
"""The words of the comment that `3dAllineate` and `3dvolreg` write
before their matrices: `# 3dAllineate matrices (DICOM-to-DICOM,
row-by-row):`."""

_COMMENT = "# AFNI matrices (DICOM-to-DICOM, row-by-row):"
"""The comment written before matrices that were not read from a file."""

_AFF12 = ".aff12.1d"
"""The ending of the name of a file that `-1Dmatrix_save` writes, in
lower case."""

_NAMED = (".aff12.1d", ".1d")
"""The endings of the names of AFNI matrix files, in lower case."""

_SEPARATORS = r"\s+"
"""The separators of the values of a row, which are runs of white space."""

_SNIFF_LIMIT = 16 << 20
"""The number of characters that sniffing reads at most.

The volume registration of a long time series writes one row per volume,
so a few megabytes are allowed before a file is turned down unread.
"""


class AfniAffineRaw(TextFileReader, TextFileWriter):
    """The content of an AFNI matrix file, as the file holds it.

    An AFNI matrix file holds the 12 numbers of one or more `(3, 4)`
    matrices, which map the DICOM (LPS) coordinates of a base dataset to
    those of a source dataset. `3dAllineate -1Dmatrix_save`, `3dvolreg
    -1Dmatrix_save`, `align_epi_anat.py` and `cat_matvec -ONELINE` write
    one matrix per row, after a comment, and `cat_matvec` writes a single
    matrix as 3 lines of 4 numbers. A matrix file is short and is read in
    one pass, so the record holds the whole file: the matrices as stored,
    the comment above them, the layout of the rows, and the text that the
    file was read from, so that a file that is read and written again
    keeps its bytes.

    The record is held by
    [`AfniAffineMetadata`][brainhops.io.transformations.afni.AfniAffineMetadata],
    which never changes it in place. Its matrices are read-only, and a
    writer builds a new record. Two records are equal when they hold the
    same matrices, the same comment and the same layout.

    Parameters
    ----------
    matrices : array_like, optional
        The `(n, 3, 4)` matrices, or a single `(3, 4)` matrix. They
        default to the identity.
    comment : str, optional
        The comment lines above the matrices, each with its `#`.
    oneline : bool, default=True
        Whether each matrix is a row of 12 numbers, rather than a single
        matrix written as 3 lines of 4 numbers.
    stored : str, optional
        The text that the record was read from.
    """

    def __init__(
        self,
        matrices: tx.Optional[tx.Any] = None,
        comment: tx.Optional[str] = None,
        oneline: bool = True,
        stored: tx.Optional[str] = None,
    ) -> None:
        if matrices is None:
            matrices = np.eye(3, 4)
        matrices = np.array(matrices, dtype=np.float64).reshape(-1, 3, 4)
        matrices.flags.writeable = False
        self.matrices = matrices
        """The `(n, 3, 4)` matrices, as stored, which are read-only."""
        self.comment = comment
        """The comment lines above the matrices, or `None`."""
        self.oneline = bool(oneline)
        """Whether each matrix is written as one row of 12 numbers."""
        self.stored = stored
        """The text that the record was read from, or `None`.

        The text is written back as it is while it still describes the
        record, whichever program wrote it, and the record is formatted as
        the writer spells it otherwise.
        """

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a text stream holds AFNI matrices.

        The name of the stream, when it has one, takes part in the score,
        as [`sniff_lines`][] describes. At most 16 megabytes of the stream
        are read, and its position is restored.

        Parameters
        ----------
        file : IO
            The stream, open in text mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no AFNI
            matrices.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        name = getattr(file, "name", None)
        if isinstance(name, (str, bytes, os.PathLike)):
            name = os.fsdecode(name)
        else:
            name = None
        try:
            with preserve_position(file):
                text = file.read(_SNIFF_LIMIT + 1)
            if isinstance(text, bytes):
                text = text.decode("utf-8")
        except UnicodeDecodeError as e:
            return _declined(error, "The content is not text.", e)
        if len(text) > _SNIFF_LIMIT:
            return _declined(error, "The content is too long.", None)
        return _score(text.splitlines(), name, error)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that lines hold AFNI matrices.

        The lines must hold matrices that AFNI reads: rows of 12 numbers,
        3 lines of 4 numbers, 4 lines whose last is `0 0 0 1`, or 9
        numbers. The comment that `3dAllineate` and `3dvolreg` write
        (`... (DICOM-to-DICOM, row-by-row):`), or a name that ends with
        `.aff12.1D`, scores `Confidence.CERTAIN`. A `.1D` name scores
        `Confidence.LIKELY` for rows of 12 numbers or 3 lines of 4, and
        `Confidence.WEAK` for the other layouts, which any software may
        save, so that the generic matrix reader keeps them. Without a name,
        rows of 12 numbers score `Confidence.WEAK`, and any other layout
        `Confidence.NO`. Lines have no name.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines hold no AFNI
            matrices.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return _score(list(lines), None, error)

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "AfniAffineRaw":
        """Parse the text of an AFNI matrix file, keeping the text.

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            Refused, since the record has no option.

        Returns
        -------
        AfniAffineRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the text does not hold matrices that AFNI reads.
        """
        _no_options(kwargs)
        lines = text.splitlines()
        array = _array(lines)
        matrices = _matrices(array)
        if matrices is None:
            raise ParserContentError(
                f"A {array.shape} table is not an AFNI matrix file: "
                f"expected rows of 12 numbers, or 3 lines of 4."
            )
        return cls(
            matrices=matrices,
            comment=_comment(lines),
            oneline=array.shape[1] == 12,
            stored=text,
        )

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> "AfniAffineRaw":
        """Parse the lines of an AFNI matrix file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without their line breaks.
        **kwargs : Any
            Refused, since the record has no option.

        Returns
        -------
        AfniAffineRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the lines do not hold matrices that AFNI reads.
        """
        return cls.from_text("".join(line + "\n" for line in lines), **kwargs)

    # --- writing ------------------------------------------------------

    @takes_error_policy
    def to_text(self, **kwargs) -> str:
        """Return the content of the file, ending with a newline.

        The text that the record was read from is returned while it still
        describes the record. Otherwise the comment is written, and then
        each matrix as one row of 12 numbers, or the single matrix as 3
        lines of 4 numbers, with every digit of a double.

        Parameters
        ----------
        **kwargs : Any
            Refused, since the writer has no option.

        Returns
        -------
        str
            The content of the file.

        Raises
        ------
        TypeError
            If an option is given.
        """
        _no_options(kwargs)
        if self.stored is not None and _parsed(self.stored) == self:
            return self.stored
        lines = self.comment.splitlines() if self.comment else []
        if self.oneline:
            lines += [
                " " + " ".join(_format(v) for v in matrix.ravel())
                for matrix in self.matrices
            ]
        else:
            lines += [
                " ".join(_format(v) for v in row) for row in self.matrices[0]
            ]
        return "".join(line + "\n" for line in lines)

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.List[str]:
        """Return the lines of the file, without their line breaks.

        Parameters
        ----------
        **kwargs : Any
            Refused, since the writer has no option.

        Returns
        -------
        list of str
            The lines of [`to_text`][].
        """
        return self.to_text(**kwargs).splitlines()

    def __repr__(self) -> str:
        return (
            f"AfniAffineRaw(matrices={self.matrices.tolist()!r}, "
            f"comment={self.comment!r}, oneline={self.oneline!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AfniAffineRaw):
            return NotImplemented
        return (
            np.array_equal(self.matrices, other.matrices)
            and self.comment == other.comment
            and self.oneline == other.oneline
        )

    __hash__ = None


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _no_options(kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options of a reader or a writer that takes none.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(
            f"Unknown AFNI matrix file option(s): {', '.join(kwargs)}"
        )


def _array(lines: tx.List[str]) -> np.ndarray:
    """Return the table of numbers that lines hold.

    Raises
    ------
    ParserContentError
        If the lines do not hold a table of numbers.
    """
    try:
        return read_text_array(lines, comments="#", separators=_SEPARATORS)
    except ArrayContainerError as e:
        raise ParserContentError(str(e)) from e


def _matrices(array: np.ndarray) -> tx.Optional[np.ndarray]:
    """Return the `(n, 3, 4)` matrices that a table holds, or `None`.

    AFNI reads the following layouts (`3dAllineate -1Dmatrix_apply`,
    `cat_matvec`).

    - Rows of 12 numbers, one matrix per row, with its three rows one
      after the other (`-1Dmatrix_save`, `cat_matvec -ONELINE`).
    - A single matrix as 3 lines of 4 numbers (`cat_matvec`), or as 4
      lines that end with `0 0 0 1` (`cat_matvec -4x4`).
    - 9 numbers, as 3 lines of 3 or one line, which are a matrix without
      a shift.
    """
    if not is_numeric_array(array) or array.ndim != 2 or not array.size:
        return None
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        return None
    rows, cols = array.shape
    if cols == 12:
        return array.reshape(rows, 3, 4)
    if (rows, cols) == (3, 4):
        return array[None]
    if (rows, cols) == (4, 4) and np.allclose(array[3], [0, 0, 0, 1]):
        return array[None, :3]
    if (rows, cols) in ((3, 3), (1, 9)):
        matrix = np.zeros((1, 3, 4))
        matrix[0, :, :3] = array.reshape(3, 3)
        return matrix
    return None


def _comment(lines: tx.Sequence[str]) -> tx.Optional[str]:
    """Return the comment lines before the first value, or `None`."""
    comments = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if not text.startswith("#"):
            break
        comments.append(text)
    return "\n".join(comments) or None


def _score(
    lines: tx.List[str],
    name: tx.Optional[str],
    error: tx.Union[bool, tx.Type[Exception]],
) -> float:
    """Score lines as AFNI matrices, as `AfniAffineRaw.sniff_lines` says."""
    try:
        array = _array(lines)
    except ParserContentError as e:
        return _declined(error, str(e), e)
    if _matrices(array) is None:
        return _declined(
            error, f"A {array.shape} table is not an AFNI matrix file.", None
        )
    lower = None if name is None else name.lower()
    if (lower is not None and lower.endswith(_AFF12)) or _SIGNATURE in (
        _comment(lines) or ""
    ):
        return Confidence.CERTAIN
    # Rows of 12 numbers, and 3 lines of 4, are the layouts of AFNI. A
    # square matrix is what any software saves, so a `.1D` file that holds
    # one is left to the generic matrix reader.
    own = array.shape[1] == 12 or array.shape == (3, 4)
    if lower is not None and lower.endswith(_NAMED):
        return Confidence.LIKELY if own else Confidence.WEAK
    if array.shape[1] == 12:
        return Confidence.WEAK
    return _declined(
        error, "A matrix in a file not named .1D is not claimed by AFNI.", None
    )


def _parsed(text: str) -> AfniAffineRaw:
    """Parse stored text into a record, to compare it with another."""
    lines = text.splitlines()
    array = _array(lines)
    return AfniAffineRaw(
        matrices=_matrices(array),
        comment=_comment(lines),
        oneline=array.shape[1] == 12,
    )


def _format(value: float) -> str:
    """Return a value with every digit of a double.

    The value is the `repr` of the float, made more compact for whole
    numbers, which are written as integers.
    """
    value = float(value)
    if value == 0:
        return "0"
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _declined(
    error: tx.Union[bool, tx.Type[Exception]],
    message: str,
    failure: tx.Optional[Exception],
) -> float:
    """Decline sniffed content, raising the requested error if any."""
    if error:
        if error is True:
            error = SnifferContentError
        raise error(message) from failure
    return Confidence.NO
