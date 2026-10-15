"""The record of a `register.dat` file, which holds the whole file."""

__all__ = ["RegisterDatRaw"]

import re

import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
    TextFileReader,
    TextFileWriter,
    WriterError,
)

_Matrix = tx.Tuple[tx.Tuple[float, ...], ...]

_IDENTITY: _Matrix = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

_UNKNOWN_SUBJECT = "subject-unknown"
"""The subject that FreeSurfer writes when it knows none."""


# The example of a file has lines longer than the limit.
# ruff: disable[E501]
class RegisterDatRaw(Raw, TextFileReader, TextFileWriter):
    """Record of a FreeSurfer `register.dat` file, as it was read.

    A `register.dat` file is the registration that `tkregister2` and
    `bbregister` write with `--reg`. It is a short text file, read in one
    pass, so the record holds the whole file: the name of the subject,
    the in-plane resolution and the slice thickness of the movable
    volume, an intensity scale that only `tkregister2` uses for display,
    the 4x4 registration matrix, and the rounding method that the matrix
    was made for. The matrix maps the tkregister RAS coordinates of the
    reference (target) volume to those of the movable volume, as
    [`RegisterDatTransformation`][brainhops.io.transformations.freesurfer.regdat.RegisterDatTransformation]
    describes. The file holds no volume geometry.

    The file is written as follows by `regio_print_register` in
    `utils/registerio.cpp` of FreeSurfer (commit 4f8728e, lines 161-202):
    the subject, three numbers in `%f` format, three rows of the matrix
    with each value in `%18.15e` format followed by a space, the fixed
    last row `0 0 0 1`, and the rounding method. FreeSurfer reads the file
    with `fscanf` (`regio_read_register`, lines 43-158), so any white
    space separates the values, the numbers are read in single precision,
    and the rounding line may be missing.

    The record also keeps the lines of the file, which are written back
    as they are while they still encode the other fields, so that a file
    that is read and written again is kept byte for byte. A record is
    never changed in place, so that several objects can share it.

    !!! example "A file in the layout that `tkregister2` writes"
        ```
        bert
        2.000000
        3.000000
        0.150000
        9.998000000000000e-01 -1.400000000000000e-02 1.000000000000000e-02 1.250000000000000e+00
        1.400000000000000e-02 9.999000000000000e-01 0.000000000000000e+00 -2.500000000000000e+00
        -1.000000000000000e-02 0.000000000000000e+00 9.999000000000000e-01 3.000000000000000e-01
        0 0 0 1
        round
        ```
    """

    # ruff: enable[E501]

    subject: str = _UNKNOWN_SUBJECT
    """The name of the subject, which cannot contain white space."""

    inplane_resolution: float = 1.0
    """The in-plane resolution of the movable volume, in millimetres.

    `tkregister2` writes the first voxel size of the movable volume here,
    and FreeSurfer reads it only as the voxel size of a movable volume
    whose geometry is not given.
    """

    slice_thickness: float = 1.0
    """The slice thickness of the movable volume, in millimetres.

    `tkregister2` writes the third voxel size of the movable volume here.
    """

    intensity: float = 1.0
    """The intensity scale that `tkregister2` uses to display the volume."""

    matrix: _Matrix = _IDENTITY
    """The 4x4 registration matrix, as a tuple of rows of floats."""

    float2int: tx.Optional[str] = "round"
    """The method that rounds voxel coordinates, as the file names it.

    It is `"round"` (or `"rint"`), `"floor"`, or a name that starts with
    `"tkreg"`, as `float2int_code` in `utils/resample.cpp` of FreeSurfer
    accepts them. It is `None` when the file has no such line, which
    FreeSurfer reads as the method of `tkregister`. That method truncates
    the coordinates, and `tkregister2` corrects the matrix of such a file
    (`MRIfixTkReg` in `utils/mri.cpp`), whereas `lta_convert` does not.
    The matrix is read here as it is stored, as `lta_convert` reads it.
    """

    lines: NoRepr[tx.Optional[tx.Tuple[str, ...]]] = None
    """The lines of the file as it was read, or `None` for a new record.

    They are written back as they are while they still encode the other
    fields of the record.
    """

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the likelihood that lines are those of a `register.dat`.

        The content must hold a subject, three positive numbers, a 4x4
        matrix whose last row is `0 0 0 1`, and a known rounding method
        or nothing after the matrix. Content that ends with a rounding
        method scores `Confidence.LIKELY`, and content that ends with the
        matrix scores `Confidence.MAYBE`, since it has a weaker signature. Content with more words is left to other
        formats, although FreeSurfer would ignore them.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines are not those of a
            `register.dat` file.
        **kwargs : Any
            Options of the reader, which are not used.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        words = _words(lines, _MAX_WORDS + 1)
        reason = _signature_error(words)
        if reason is None:
            if len(words) == _MAX_WORDS:
                return Confidence.LIKELY
            return Confidence.MAYBE
        if error:
            if error is True:
                error = SnifferContentError
            raise error(f"Not a register.dat file: {reason}")
        return Confidence.NO

    # --- from ---------------------------------------------------------

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Read a record from the lines of a `register.dat` file.

        The values are read as `regio_read_register` reads them, separated
        by any white space. Words after the rounding method are ignored, as
        FreeSurfer ignores them.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without line endings.
        **kwargs : Any
            Options of the reader. There are none, so any option is
            refused.

        Returns
        -------
        RegisterDatRaw
            The record, which keeps the lines.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the lines are not those of a `register.dat` file.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__}.from_lines() takes no options, but was "
                f"given {', '.join(sorted(kwargs))}."
            )
        lines = tuple(lines)
        return cls(**_decode(lines), lines=lines)

    # --- to -----------------------------------------------------------

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the `register.dat` file that the record holds.

        The lines that were read are yielded while they still encode the
        fields of the record. Otherwise, the fields are written as
        `regio_print_register` writes them.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer. There are none, so any option is
            refused.

        Yields
        ------
        str
            The lines of the file, without line endings.

        Raises
        ------
        TypeError
            If an option is given.
        WriterError
            If the subject or the rounding method cannot be written.
        """
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.to_lines() takes no options, but "
                f"was given {', '.join(sorted(kwargs))}."
            )
        if self.lines is not None and _still_encodes(self):
            return iter(self.lines)
        return iter(_encode(self))

    def to_text(self, **kwargs) -> str:
        """Return the content of the `register.dat` file.

        The content ends with a newline, as the file that FreeSurfer
        writes does.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer, as [`to_lines`][] takes them.

        Returns
        -------
        str
            The content of the file.
        """
        return "\n".join(self.to_lines(**kwargs)) + "\n"


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------

# A file holds a subject, three numbers, 16 numbers of the matrix and an
# optional rounding method.
_MIN_WORDS = 20
_MAX_WORDS = 21

# The numbers that `fscanf("%f")` reads in full, in decimal notation. A
# number that `fscanf` reads only in part breaks the next read, so it is
# refused here.
_NUMBER = re.compile(
    r"[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)",
    re.IGNORECASE,
)


def _words(lines: tx.Iterable[str], limit: tx.Optional[int] = None) -> list:
    """Return the words of the lines, at most `limit` of them."""
    words = []
    for line in lines:
        words.extend(line.split())
        if limit is not None and len(words) >= limit:
            return words[:limit]
    return words


def _is_number(word: str) -> bool:
    return _NUMBER.fullmatch(word) is not None


def _float2int_known(word: str) -> bool:
    # The names that `float2int_code` in `utils/resample.cpp` accepts,
    # ignoring case: "round", "rint", "floor", and any name whose first
    # five letters are "tkreg".
    word = word.lower()
    return word in ("round", "rint", "floor") or word[:5] == "tkreg"


def _signature_error(words: tx.List[str]) -> tx.Optional[str]:
    """Return why words are not those of a file, or `None` if they are."""
    if not _MIN_WORDS <= len(words) <= _MAX_WORDS:
        return f"expected 20 or 21 values, found {len(words)} or more"
    numbers = words[1:_MIN_WORDS]
    if not all(_is_number(word) for word in numbers):
        return "expected three numbers and a 4x4 matrix after the subject"
    if not all(float(word) > 0 for word in numbers[:2]):
        return "expected positive voxel sizes"
    if tuple(float(word) for word in numbers[-4:]) != (0.0, 0.0, 0.0, 1.0):
        return "expected the last row of the matrix to be 0 0 0 1"
    if len(words) == _MAX_WORDS and not _float2int_known(words[-1]):
        return f"unknown rounding method {words[-1]!r}"
    return None


def _decode(lines: tx.Sequence[str]) -> tx.Dict[str, tx.Any]:
    """Return the fields that lines encode, as `regio_read_register` reads.

    Raises
    ------
    ParserContentError
        If the lines are not those of a `register.dat` file.
    """
    words = _words(lines)
    if len(words) < _MIN_WORDS:
        raise ParserContentError(
            f"A register.dat file holds a subject, three numbers and a 4x4 "
            f"matrix, which are 20 values, but this content has "
            f"{len(words)}."
        )
    numbers = words[1:_MIN_WORDS]
    for word in numbers:
        if not _is_number(word):
            raise ParserContentError(
                f"A register.dat file holds numbers after its subject, but "
                f"{word!r} is not a number."
            )
    numbers = [float(word) for word in numbers]
    float2int = words[_MIN_WORDS] if len(words) > _MIN_WORDS else None
    if float2int is not None and not _float2int_known(float2int):
        raise ParserContentError(
            f"The rounding method of a register.dat file is round, rint, "
            f"floor or tkregister, not {float2int!r}."
        )
    flat = numbers[3:]
    return dict(
        subject=words[0],
        inplane_resolution=numbers[0],
        slice_thickness=numbers[1],
        intensity=numbers[2],
        matrix=tuple(tuple(flat[4 * i : 4 * i + 4]) for i in range(4)),
        float2int=float2int,
    )


def _fields(record: RegisterDatRaw) -> tx.Dict[str, tx.Any]:
    """Return the fields of a record that its lines encode."""
    return dict(
        subject=record.subject,
        inplane_resolution=float(record.inplane_resolution),
        slice_thickness=float(record.slice_thickness),
        intensity=float(record.intensity),
        matrix=tuple(tuple(float(v) for v in row) for row in record.matrix),
        float2int=record.float2int,
    )


def _still_encodes(record: RegisterDatRaw) -> bool:
    """Whether the lines of a record still encode its other fields."""
    try:
        return _decode(record.lines) == _fields(record)
    except (ParserContentError, TypeError, ValueError):
        return False


def _encode(record: RegisterDatRaw) -> tx.List[str]:
    """Return the lines that `regio_print_register` writes for a record.

    Raises
    ------
    WriterError
        If the subject or the rounding method cannot be written.
    """
    subject = record.subject or _UNKNOWN_SUBJECT
    if len(subject.split()) != 1 or subject.strip() != subject:
        raise WriterError(
            f"The subject of a register.dat file is read as one word, so "
            f"it cannot be {subject!r}."
        )
    if record.float2int is not None and not (
        len(record.float2int.split()) == 1
        and _float2int_known(record.float2int)
    ):
        raise WriterError(
            f"The rounding method of a register.dat file is round, rint, "
            f"floor or tkregister, not {record.float2int!r}."
        )
    matrix = tuple(tuple(float(v) for v in row) for row in record.matrix)
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise WriterError("A register.dat file holds a 4x4 matrix.")
    lines = [
        subject,
        format(record.inplane_resolution, "f"),
        format(record.slice_thickness, "f"),
        format(record.intensity, "f"),
    ]
    for row in matrix[:3]:
        lines.append("".join(format(value, "18.15e") + " " for value in row))
    if matrix[3] == (0.0, 0.0, 0.0, 1.0):
        # FreeSurfer writes the last row as it is fixed.
        lines.append("0 0 0 1")
    else:
        lines.append(
            "".join(format(value, "18.15e") + " " for value in matrix[3])
        )
    if record.float2int is not None:
        lines.append(record.float2int)
    return lines
