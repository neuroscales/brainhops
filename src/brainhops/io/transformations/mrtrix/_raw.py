"""The record of an MRtrix linear transform file, which holds the whole
file as it is parsed."""

# stdlib
from collections import OrderedDict

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core.streams import preserve_position
from brainhops.datamodel.metadata._internal_api import takes_error_policy
from brainhops.io.base._internal_api import declined, no_options
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    TextFileReader,
    TextFileWriter,
)
from brainhops.io.common._internal_api import (
    ArrayContainerError,
    read_text_array,
)

_NDIM = 3
"""The number of spatial dimensions of an MRtrix linear transform."""

_SEPARATORS = r"[\s,;]+"
"""The separators of the values of a row, which are spaces, commas,
semicolons and tabs (`load_matrix_2D_vector`, `core/math/math.h`)."""

_HISTORY = "command_history"
"""The key in which MRtrix records the commands that wrote a file."""

_CENTRE = "centre"
"""The key in which `mrregister` records the centre of rotation."""

_OPTION_OWNER = "MRtrix linear transform"
"""What refuses unknown options, as the error message names it."""

_VERSION = "(version="
"""What MRtrix appends to every `command_history` entry (`core/app.cpp`)."""

_SNIFF_LIMIT = 1 << 20
"""The number of characters that sniffing reads at most.

A linear transform is a few lines long, so a larger file is not claimed,
and a large file is never read whole only to be rejected.
"""


class MrtrixLinearRaw(TextFileReader, TextFileWriter):
    """The content of an MRtrix linear transform file, as the file holds it.

    `mrtransform -linear` and `mrregister -rigid` or `-affine` read and
    write a text matrix (`load_transform` and `save_transform`, in
    `core/math/math.h`): three or four rows of four values, after
    `# key: value` comments such as `command_history` and `centre`. A
    linear transform file is short and is read in one pass, so the record
    holds the whole file: the first three rows of the matrix, the comments
    as keys and values, and the text that the file was read from, so that
    a file that is read and written again keeps its bytes.

    The record is held by
    [`MrtrixLinearMetadata`][brainhops.io.transformations.mrtrix.MrtrixLinearMetadata],
    which never changes it in place. Its matrix is read-only, and a writer
    works on the copy that [`copy`][] returns. Two records are equal when
    they hold the same matrix and the same keys.

    Parameters
    ----------
    matrix : array_like, optional
        The `(3, 4)` matrix, or a `(4, 4)` matrix whose last row is
        dropped. It defaults to the identity.
    keyval : mapping, optional
        The keys and values of the comments, where the lines of a
        repeated key are joined by newlines.
    stored : str, optional
        The text that the record was read from.
    """

    def __init__(
        self,
        matrix: tx.Optional[tx.Any] = None,
        keyval: tx.Optional[tx.Mapping[str, str]] = None,
        stored: tx.Optional[str] = None,
    ) -> None:
        if matrix is None:
            matrix = np.eye(_NDIM, _NDIM + 1)
        matrix = np.array(matrix, dtype=np.float64)
        if matrix.shape == (_NDIM + 1, _NDIM + 1):
            matrix = matrix[:_NDIM]
        if matrix.shape != (_NDIM, _NDIM + 1):
            raise ValueError(
                f"An MRtrix linear transform is a (3, 4) or (4, 4) matrix, "
                f"not a matrix of shape {matrix.shape}."
            )
        matrix.flags.writeable = False
        self.matrix = matrix
        """The `(3, 4)` matrix, which is read-only."""
        self.keyval = OrderedDict(keyval or {})
        """The keys and values of the comments of the file."""
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
        """Return the confidence that a text stream holds a transform.

        At most a megabyte of the stream is read, and its position is
        restored.

        Parameters
        ----------
        file : IO
            The stream, open in text mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no transform
            written by MRtrix.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        try:
            with preserve_position(file):
                text = file.read(_SNIFF_LIMIT + 1)
            if isinstance(text, bytes):
                text = text.decode("utf-8")
        except UnicodeDecodeError as e:
            return declined(error, "The content is not text.", e)
        if len(text) > _SNIFF_LIMIT:
            return declined(error, "The content is too long.", None)
        return cls.sniff_lines(text.splitlines(), error=error)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that lines hold a transform of MRtrix.

        The lines must hold a `(3, 4)` or `(4, 4)` matrix, whose comments
        carry the signature of MRtrix: a `command_history` entry with the
        version that MRtrix appends scores `Confidence.CERTAIN`, and a
        `command_history` or `centre` entry without it scores
        `Confidence.LIKELY`. A plain text matrix scores `Confidence.NO`,
        because plain matrices, FLIRT files and MRtrix transforms look
        alike, and it is read as a transform only when it is asked for.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines hold no transform
            written by MRtrix.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        lines = list(lines)
        try:
            _matrix(lines)
        except ParserContentError as e:
            return declined(error, str(e), e)
        score = _signature(_comment_keyval(lines))
        if not score:
            return declined(
                error,
                "A text matrix without the comments that MRtrix writes. "
                "Read it with hint='mrtrix.linear' if it is an MRtrix "
                "transform.",
                None,
            )
        return score

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "MrtrixLinearRaw":
        """Parse the text of a linear transform file.

        Only the first three rows are read, as MRtrix reads them, and the
        text is kept.

        Parameters
        ----------
        text : str
            The content of the file.
        **kwargs : Any
            Refused, since the record has no option.

        Returns
        -------
        MrtrixLinearRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the text does not hold a `(3, 4)` or `(4, 4)` matrix.
        """
        no_options(kwargs, _OPTION_OWNER)
        lines = text.splitlines()
        return cls(
            matrix=_matrix(lines),
            keyval=_comment_keyval(lines),
            stored=text,
        )

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], **kwargs
    ) -> "MrtrixLinearRaw":
        """Parse the lines of a linear transform file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without their line breaks.
        **kwargs : Any
            Refused, since the record has no option.

        Returns
        -------
        MrtrixLinearRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the lines do not hold a `(3, 4)` or `(4, 4)` matrix.
        """
        return cls.from_text("".join(line + "\n" for line in lines), **kwargs)

    # --- writing ------------------------------------------------------

    @takes_error_policy
    def to_text(self, **kwargs) -> str:
        """Return the content of the file, ending with a newline.

        The text that the record was read from is returned while it still
        describes the record. Otherwise the record is formatted as MRtrix
        writes it (`save_transform`): the comments, three rows of four
        values, and `0 0 0 1`.

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
        no_options(kwargs, _OPTION_OWNER)
        if self.stored is not None and _parsed(self.stored) == self:
            return self.stored
        return "".join(line + "\n" for line in self.to_lines())

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.List[str]:
        """Return the lines of the file as MRtrix writes them.

        Parameters
        ----------
        **kwargs : Any
            Refused, since the writer has no option.

        Returns
        -------
        list of str
            The comments, three rows of four values, and `0 0 0 1`,
            without their line breaks.

        Raises
        ------
        TypeError
            If an option is given.
        """
        no_options(kwargs, _OPTION_OWNER)
        lines = []
        for key, value in self.keyval.items():
            for line in str(value).split("\n"):
                lines.append(f"# {key}: {line}")
        for row in self.matrix:
            lines.append(" ".join(repr(float(v)) for v in row))
        lines.append("0 0 0 1")
        return lines

    def copy(self) -> "MrtrixLinearRaw":
        """Return a copy of the record that can be changed freely.

        The matrix is shared, since it is read-only, and the text that the
        record was read from is kept, since [`to_text`][] writes it only
        while it still describes the record.

        Returns
        -------
        MrtrixLinearRaw
            The copy.
        """
        return MrtrixLinearRaw(
            matrix=self.matrix, keyval=self.keyval, stored=self.stored
        )

    def __repr__(self) -> str:
        return (
            f"MrtrixLinearRaw(matrix={self.matrix.tolist()!r}, "
            f"keyval={dict(self.keyval)!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MrtrixLinearRaw):
            return NotImplemented
        return (
            np.array_equal(self.matrix, other.matrix)
            and self.keyval == other.keyval
        )

    __hash__ = None


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _matrix(lines: tx.List[str]) -> np.ndarray:
    """Return the first three rows of the matrix that lines hold.

    Raises
    ------
    ParserContentError
        If the lines do not hold a `(3, 4)` or `(4, 4)` matrix.
    """
    try:
        array = read_text_array(lines, comments="#", separators=_SEPARATORS)
    except ArrayContainerError as e:
        raise ParserContentError(str(e)) from e
    if array.shape not in ((_NDIM, _NDIM + 1), (_NDIM + 1, _NDIM + 1)):
        raise ParserContentError(
            f"An MRtrix linear transform has 3 or 4 rows of 4 values, not "
            f"an array of shape {array.shape}."
        )
    return array[:_NDIM]


def _comment_keyval(lines: tx.Iterable[str]) -> "OrderedDict[str, str]":
    """Return the `# key: value` comments of an MRtrix text file.

    MRtrix writes its keys as comments, one line per entry
    (`File::KeyValue::write` with the prefix `"# "`). The lines of a
    repeated key are joined by newlines, as in an image header, and other
    comments are dropped.
    """
    keyval: OrderedDict[str, str] = OrderedDict()
    for line in lines:
        if "#" not in line:
            continue
        comment = line.split("#", 1)[1].strip()
        key, sep, value = comment.partition(":")
        key = key.strip()
        if not sep or not key or " " in key:
            continue
        value = value.strip()
        if key in keyval:
            keyval[key] += "\n" + value
        else:
            keyval[key] = value
    return keyval


def _signature(keyval: tx.Mapping[str, str]) -> float:
    """Return how surely the comments of a text matrix come from MRtrix."""
    history = keyval.get(_HISTORY, "")
    if _VERSION in history:
        return Confidence.CERTAIN
    if _HISTORY in keyval or _CENTRE in keyval:
        return Confidence.LIKELY
    return Confidence.NO


def _parsed(text: str) -> MrtrixLinearRaw:
    """Parse stored text into a record, to compare it with another."""
    lines = text.splitlines()
    return MrtrixLinearRaw(
        matrix=_matrix(lines), keyval=_comment_keyval(lines)
    )
