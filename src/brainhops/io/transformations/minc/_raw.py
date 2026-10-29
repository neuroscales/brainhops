"""The record of an MNI transform file, which holds the whole file."""

__all__ = ["XfmRaw"]

# stdlib

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic, field

# internals
from brainhops._core import path
from brainhops.datamodel.metadata._internal_api import takes_error_policy
from brainhops.io.base._internal_api import Raw
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
    TextFileReader,
    TextFileWriter,
    WriterError,
)

# this format
from . import _syntax as syntax

_Row = tx.Tuple[float, ...]
_Rows = tx.Tuple[_Row, ...]


class XfmRaw(Raw, TextFileReader, TextFileWriter):
    """Record of an MNI transform file, which holds the file as it is parsed.

    An MNI transform file (`.xfm`) is a short text file that is read in
    one pass, so the record holds the whole file: the comments at its top
    and the transforms that it concatenates, in the order of the file.
    Each transform is one of the blocks [`XfmRaw.Linear`][],
    [`XfmRaw.Grid`][] and [`XfmRaw.ThinPlateSpline`][]. A grid transform
    names the MINC volume of its displacements, which the record does not
    read.

    The record also keeps the text that it was read from, so that a file
    that is read and written again without changes is written byte for
    byte, with all its comments and its layout. The text is written only
    while it still parses to the comments and the transforms of the
    record. A record that was changed, with
    [`replace`][bagof.magic.replace], is written in the layout of libminc
    instead, with the comments of its top.

    A record is frozen and never changed in place, so that several objects
    can share it. The metadata of a file,
    [`XfmMetadata`][brainhops.io.transformations.minc.XfmMetadata], holds
    a record.
    """

    class Linear(Magic, frozen=True):
        """A linear transform, from `Transform_Type = Linear;`.

        The matrix maps world coordinates of the source, in RAS
        millimetres, to world coordinates of the target. libminc applies
        its inverse instead when `inverted` is set.
        """

        matrix: _Rows = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        )
        """The three rows of the `(3, 4)` matrix, as stored."""

        inverted: bool = False
        """Whether the block sets `Invert_Flag = True;`."""

        @property
        def homogeneous(self) -> np.ndarray:
            """The `(4, 4)` homogeneous matrix, as stored."""
            matrix = np.eye(4)
            matrix[:3] = np.asarray(self.matrix, dtype=np.float64)
            return matrix

    class Grid(Magic, frozen=True):
        """A grid transform, from `Transform_Type = Grid_Transform;`.

        The displacements are stored in a MINC volume, which `volume`
        names as the file writes it. A relative name is relative to the
        directory of the transform file.
        """

        volume: str = ""
        """The name of the displacement volume, as stored."""

        inverted: bool = False
        """Whether the block sets `Invert_Flag = True;`."""

    class ThinPlateSpline(Magic, frozen=True):
        """A thin-plate spline transform.

        The block starts with `Transform_Type =
        Thin_Plate_Spline_Transform;`. It is read and written back, but
        brainhops has no transformation that it can be decoded into.
        """

        ndim: int = 3
        """The number of dimensions, `Number_Dimensions`."""

        points: _Rows = ()
        """The control points, one row of `ndim` values per point."""

        displacements: _Rows = ()
        """The rows of `Displacements`, `ndim` values each."""

        inverted: bool = False
        """Whether the block sets `Invert_Flag = True;`."""

    Block = tx.Union[Linear, Grid, ThinPlateSpline]

    comments: tx.Tuple[str, ...] = ()
    """The comments before the first transform, each with its comment
    character."""

    transforms: tx.Tuple[Block, ...] = ()
    """The transforms that the file concatenates, in the order of the file,
    which is the order in which they are applied."""

    text: tx.Optional[str] = field(default=None, repr=False)
    """The text that the record was read from, or `None`."""

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the text of a file as an MNI transform file.

        A text that libminc reads is certainly an MNI transform file. A
        text that starts with the header line `MNI Transform File` but that
        libminc would not read scores `Confidence.MAYBE`, so that reading
        it reports what is wrong.

        Parameters
        ----------
        text : str
            The text of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the text is not an MNI transform
            file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return score(sniff_transforms(text) is not None, text, error)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the lines of a file as an MNI transform file.

        The lines are joined and scored by [`sniff_text`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without line endings.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines are not an MNI
            transform file.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_text("\n".join(lines), error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the bytes of a file as an MNI transform file.

        The bytes are decoded as [`from_bytes`][] decodes them, and scored
        by [`sniff_text`][].

        Parameters
        ----------
        content : bytes
            The content of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not an MNI
            transform file.
        **kwargs : Any
            Options of the reader. `encoding` is used, and the others are
            ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        text = decode(content, kwargs.get("encoding", "utf-8"))
        return cls.sniff_text(text, error=error)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a file as an MNI transform file.

        The file is read as bytes and scored by [`sniff_bytes`][], so that
        a comment that is not valid UTF-8 does not make it unreadable.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an MNI transform
            file.
        **kwargs : Any
            Options of the reader, passed to [`sniff_bytes`][].

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        content = read_bytes(filename)
        if content is None:
            return super().sniff_filename(filename, error=error, **kwargs)
        return cls.sniff_bytes(content, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "XfmRaw":
        """Read a record from the text of a file.

        The text is parsed as libminc's `input_transform` parses it, and
        it is kept in the record.

        Parameters
        ----------
        text : str
            The text of the file.
        **kwargs : Any
            Refused, since the reader has no option.

        Returns
        -------
        XfmRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If libminc would not read the text.
        """
        _no_options(cls, kwargs)
        comments, transforms = parse(text)
        return cls(comments=comments, transforms=transforms, text=text)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> "XfmRaw":
        """Read a record from the lines of a file.

        The lines are joined by newlines and read by [`from_text`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file, without line endings.
        **kwargs : Any
            Refused, since the reader has no option.

        Returns
        -------
        XfmRaw
            The record.
        """
        return cls.from_text("\n".join(lines) + "\n", **kwargs)

    @classmethod
    def from_bytes(
        cls, content: bytes, encoding: str = "utf-8", **kwargs
    ) -> "XfmRaw":
        """Read a record from the bytes of a file.

        Bytes that do not decode are kept as they are, so that they are
        written back unchanged, and so are the line endings.

        Parameters
        ----------
        content : bytes
            The content of the file.
        encoding : str, default="utf-8"
            The encoding of the text.
        **kwargs : Any
            Refused, since the reader has no other option.

        Returns
        -------
        XfmRaw
            The record.
        """
        return cls.from_text(decode(content, encoding), **kwargs)

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> "XfmRaw":
        """Read a record from a file, as bytes.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of [`from_bytes`][].

        Returns
        -------
        XfmRaw
            The record.
        """
        return cls.from_bytes(read_file(filename), **kwargs)

    # --- to -----------------------------------------------------------

    @takes_error_policy
    def to_text(self, **kwargs) -> str:
        """Return the text of the file that the record holds.

        The text that the record was read from is returned while it still
        parses to the comments and the transforms of the record. Otherwise
        the record is written as libminc's `output_transform` writes a
        file: the header line, the comments, a blank line, then each
        transform.

        Parameters
        ----------
        **kwargs : Any
            Refused, since the writer has no option.

        Returns
        -------
        str
            The text of the file, which ends with a newline.

        Raises
        ------
        TypeError
            If an option is given.
        WriterError
            If the record holds no transform, which libminc would not
            read, or a volume name that cannot be written.
        """
        _no_options(type(self), kwargs)
        if not self.transforms:
            raise WriterError(
                "An MNI transform file holds at least one transform, and "
                "libminc does not read a file without one."
            )
        # The text is kept as AFNI and MRtrix keep theirs (item 143 of the
        # design memo).
        encoded = _format(self)
        if self.text is not None:
            try:
                stored = XfmRaw.from_text(self.text)
            except ParserContentError:
                return encoded
            if _format(stored) == encoded:
                return self.text
        return encoded

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the file that the record holds.

        Parameters
        ----------
        **kwargs : Any
            Refused, since the writer has no option.

        Yields
        ------
        str
            The lines of [`to_text`][], without line endings.
        """
        yield from self.to_text(**kwargs).splitlines()

    @takes_error_policy
    def to_bytes(self, encoding: str = "utf-8", **kwargs) -> bytes:
        """Return the bytes of the file that the record holds.

        Parameters
        ----------
        encoding : str, default="utf-8"
            The encoding of the text.
        **kwargs : Any
            Refused, since the writer has no other option.

        Returns
        -------
        bytes
            The text of [`to_text`][], encoded. Bytes that were kept
            because they did not decode are written back unchanged.
        """
        return self.to_text(**kwargs).encode(encoding, "surrogateescape")


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def parse(text: str) -> tx.Tuple[tx.Tuple[str, ...], tx.Tuple[tx.Any, ...]]:
    """Parse the text of a file into its comments and its transforms.

    The text is read as libminc's `input_transform` reads it: the header
    line, then transforms until the end of the text. The comments are
    those that come before the first transform.

    Raises
    ------
    ParserContentError
        If libminc would not read the text.
    """
    cursor = syntax.Cursor(text)
    header = cursor.string("\0")
    if header != syntax.HEADER:
        raise cursor.error(
            f"The file must start with {syntax.HEADER!r}, not {header!r}."
        )
    comments = None
    transforms = []
    while True:
        keyword = cursor.string("=")
        if comments is None:
            comments = tuple(cursor.comments)
        if keyword is None:
            break
        if keyword != syntax.TYPE:
            raise cursor.error(
                f'Expected "{syntax.TYPE} =", found {keyword!r}.'
            )
        cursor.expect("=")
        transforms.append(_parse_transform(cursor))
    if not transforms:
        raise cursor.error("The file holds no transform.")
    return comments, tuple(transforms)


def sniff_transforms(text: str) -> tx.Optional[tx.Tuple[tx.Any, ...]]:
    """Return the transforms of a text, or `None` if libminc refuses it."""
    try:
        return parse(text)[1]
    except ParserContentError:
        return None


def read_file(filename: path.FilenameLike) -> bytes:
    """Return the bytes of a file.

    Raises
    ------
    ParserExistsError
        If the file does not exist.
    """
    content = read_bytes(filename)
    if content is None:
        raise ParserExistsError(f"No such file: {filename}")
    return content


def _parse_transform(cursor: syntax.Cursor) -> tx.Any:
    """Parse one transform, after its `Transform_Type =`.

    This is libminc's `input_one_transform`.
    """
    kind = cursor.string(";")
    if kind is None:
        raise cursor.error("The transform type is missing.")
    cursor.expect(";")
    if kind not in (syntax.LINEAR, syntax.GRID, syntax.THIN_PLATE_SPLINE):
        raise cursor.error(
            f"Unknown transform type {kind!r}: libminc reads "
            f"{syntax.LINEAR!r}, {syntax.GRID!r} and "
            f"{syntax.THIN_PLATE_SPLINE!r}."
        )
    keyword = cursor.string("=")
    inverted = False
    if keyword == syntax.INVERT_FLAG:
        cursor.expect("=")
        flag = cursor.string(";")
        cursor.expect(";")
        if flag not in (syntax.TRUE, syntax.FALSE):
            raise cursor.error(
                f"Expected {syntax.TRUE} or {syntax.FALSE} after "
                f"{syntax.INVERT_FLAG} =, found {flag!r}."
            )
        inverted = flag == syntax.TRUE
        keyword = cursor.string("=")
    if kind == syntax.LINEAR:
        _expect_keyword(cursor, keyword, syntax.LINEAR_TRANSFORM)
        values = cursor.reals(12)
        cursor.expect(";")
        matrix = tuple(tuple(values[4 * i : 4 * i + 4]) for i in range(3))
        return XfmRaw.Linear(matrix=matrix, inverted=inverted)
    if kind == syntax.GRID:
        _expect_keyword(cursor, keyword, syntax.DISPLACEMENT_VOLUME)
        volume = cursor.string(";")
        if volume is None:
            raise cursor.error("The displacement volume is missing.")
        cursor.expect(";")
        return XfmRaw.Grid(volume=volume, inverted=inverted)
    _expect_keyword(cursor, keyword, syntax.NUMBER_DIMENSIONS)
    ndim = cursor.integer()
    cursor.expect(";")
    if ndim <= 0:
        raise cursor.error(f"Number_Dimensions must be positive, not {ndim}.")
    cursor.keyword(syntax.POINTS)
    points = cursor.reals()
    cursor.expect(";")
    if len(points) % ndim:
        raise cursor.error(
            f"The number of point coordinates ({len(points)}) must be a "
            f"multiple of the number of dimensions ({ndim})."
        )
    npoints = len(points) // ndim
    cursor.keyword(syntax.DISPLACEMENTS)
    displacements = cursor.reals((npoints + ndim + 1) * ndim)
    cursor.expect(";")
    return XfmRaw.ThinPlateSpline(
        ndim=ndim,
        points=_rows(points, ndim),
        displacements=_rows(displacements, ndim),
        inverted=inverted,
    )


def _expect_keyword(
    cursor: syntax.Cursor, keyword: tx.Optional[str], expected: str
) -> None:
    """Check the keyword that was read, and read its equal sign."""
    if keyword != expected:
        raise cursor.error(f'Expected "{expected} =", found {keyword!r}.')
    cursor.expect("=")


def _rows(values: tx.Sequence[float], width: int) -> _Rows:
    """Split a flat list of values into rows of `width` values."""
    return tuple(
        tuple(values[i : i + width]) for i in range(0, len(values), width)
    )


def decode(content: bytes, encoding: str) -> str:
    """Decode the bytes of a file, keeping the bytes that do not decode."""
    return bytes(content).decode(encoding, "surrogateescape")


def read_bytes(filename: path.FilenameLike) -> tx.Optional[bytes]:
    """Return the bytes of a file, or `None` if it does not exist."""
    if isinstance(filename, str):
        filename = path.Path(filename)
    if not path.exists(filename):
        return None
    with path.Path(filename).open("rb") as file:
        return file.read()


def score(
    valid: bool, text: str, error: tx.Union[bool, tx.Type[Exception]]
) -> float:
    """Return the score of a text that libminc reads, or of its header."""
    if valid:
        return Confidence.CERTAIN
    try:
        header = syntax.Cursor(text).string("\n")
    except ParserContentError:
        header = None
    if header == syntax.HEADER:
        return Confidence.MAYBE
    if error:
        if error is True:
            error = SnifferContentError
        raise error(
            f"Not an MNI transform file: it does not start with "
            f"{syntax.HEADER!r}."
        )
    return Confidence.NO


def _no_options(cls: type, kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options of a reader or a writer that takes none.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(
            f"{cls.__name__} takes no option, but was given "
            f"{', '.join(sorted(kwargs))}."
        )


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _format(raw: XfmRaw) -> str:
    """Return the text of a record, in the layout of libminc.

    The layout is that of `output_transform`: the header line, the
    comments, a blank line, and each transform. A comment without a
    comment character is given one, as libminc's `output_comments` does.

    Raises
    ------
    WriterError
        If a volume name cannot be written.
    """
    lines = [syntax.HEADER]
    for comment in raw.comments:
        for line in str(comment).split("\n"):
            if not line.startswith(tuple(syntax.COMMENT_CHARACTERS)):
                line = "%" + line
            lines.append(line)
    lines.append("")
    for block in raw.transforms:
        lines.extend(_format_block(block))
    return "\n".join(lines) + "\n"


def _format_block(block: tx.Any) -> tx.List[str]:
    """Return the lines of one transform, as `output_one_transform` does.

    libminc writes the inverse matrix of an inverted linear transform and
    no flag, while brainhops writes the flag and the matrix as stored, so
    that both are kept.
    """
    if isinstance(block, XfmRaw.Linear):
        kind = syntax.LINEAR
    elif isinstance(block, XfmRaw.Grid):
        kind = syntax.GRID
    elif isinstance(block, XfmRaw.ThinPlateSpline):
        kind = syntax.THIN_PLATE_SPLINE
    else:
        raise WriterError(
            f"An MNI transform file cannot hold a {type(block).__name__}."
        )
    lines = [f"{syntax.TYPE} = {kind};"]
    if block.inverted:
        lines.append(f"{syntax.INVERT_FLAG} = {syntax.TRUE};")
    if isinstance(block, XfmRaw.Linear):
        lines.append(f"{syntax.LINEAR_TRANSFORM} =")
        lines.extend(_format_rows(block.matrix))
    elif isinstance(block, XfmRaw.Grid):
        lines.append(
            f"{syntax.DISPLACEMENT_VOLUME} = {_format_volume(block.volume)};"
        )
    else:
        lines.append(f"{syntax.NUMBER_DIMENSIONS} = {int(block.ndim)};")
        lines.append(f"{syntax.POINTS} =")
        lines.extend(_format_rows(block.points))
        lines.append(f"{syntax.DISPLACEMENTS} =")
        lines.extend(_format_rows(block.displacements))
    return lines


def _format_rows(rows: _Rows) -> tx.List[str]:
    """Return the lines of a block of numbers, the last ending with `;`."""
    lines = [
        "".join(" " + syntax.format_number(value) for value in row)
        for row in rows
    ]
    if not lines:
        return [";"]
    lines[-1] += ";"
    return lines


def _format_volume(volume: str) -> str:
    """Return the text of a volume name, quoted if it holds a semicolon.

    Raises
    ------
    WriterError
        If the name cannot be read back as it is.
    """
    volume = str(volume)
    if (
        not volume
        or "\n" in volume
        or '"' in volume
        or volume != volume.strip()
    ):
        raise WriterError(
            f"The displacement volume {volume!r} cannot be written in an "
            f"MNI transform file, which reads a name without line breaks, "
            f"double quotes or surrounding blanks."
        )
    return f'"{volume}"' if ";" in volume else volume
