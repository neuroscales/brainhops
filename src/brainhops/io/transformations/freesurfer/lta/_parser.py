import re
from enum import Enum
from warnings import warn

import typing_extensions as tx
from bagof.magic import Magic, fields

from brainhops._core.path import FileOrContentLike, Path, PathLike, exists
from brainhops._core.peek import peekable_lines
from brainhops.io.base.parsers import (
    Confidence,
    SnifferContentError,
    TextFileParserWriter,
)

# Once comments are stripped, the first line of an LTA file gives the
# integer transformation type.
_FIRST_LINE = re.compile(r"^type\s*=\s*\d+$")


# ----------------------------------------------------------------------
#   Parser classes with public methods inherited by LtaStruct
# ----------------------------------------------------------------------


class LtaParser(Magic, TextFileParserWriter):
    """Mixin that lets a class be sniffed, read and written as LTA.

    `LtaStruct` and its blocks inherit their `sniff*`, `from_*` and `to_*`
    methods from this class, which follows the contract of
    [`TextFileParserWriter`][]. That base class provides the public entry
    points, such as `load` and `save`, and this class implements the steps
    that are specific to the format: `sniff_line`, `from_lines` and
    `to_lines`.

    A struct is read field by field, in declaration order. A field whose type
    is itself an `LtaParser` reads its own block of lines. Any other field
    reads one line of the form `key = values`, or a line of bare values in a
    block without keys. Comments and blank lines are skipped.
    """

    _HAS_KEYS = True

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the likelihood that a line is the first line of an LTA file.

        Once comments are stripped, the first line of an LTA file reads
        `type = <int>`.

        Parameters
        ----------
        line : str
            The first line that is neither blank nor a comment.
        error : bool or type of Exception, optional
            If not `False`, raise an error when the line is not the first line
            of an LTA file.

        Returns
        -------
        float
            A confidence between 0 and 1.
        """
        if line and _FIRST_LINE.match(line.split("#", 1)[0].strip()):
            return Confidence.LIKELY
        if error:
            if error is True:
                error = SnifferContentError
            raise error(
                f'Not an LTA file: expected "type = <int>", got {line!r}'
            )
        return Confidence.NO

    # --- from ---------------------------------------------------------

    @classmethod
    def from_(cls, other: FileOrContentLike) -> tx.Self:
        """Build an object from a file or from its content.

        !!! warning "Deprecated"
            Use `load` for a file, a file object or bytes, and `from_text` or
            `from_lines` for content in memory. Unlike `load`, `from_` reads a
            string that names no existing file as LTA content.

        Parameters
        ----------
        other : str, PathLike, IO, bytes or iterable of str
            The file, or its content.
        """
        warn(
            f"{cls.__name__}.from_() is deprecated: use load() for a file, "
            f"a file object or bytes, and from_text() or from_lines() for "
            f"content held in memory.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(other, str):
            # This method is the only one in which a string may hold
            # content rather than a path. Multi-line content is usually too
            # long to be a file name, so `exists` reports it as missing
            # instead of raising an error.
            if not exists(other):
                return cls.from_text(other)
            other = Path(other)
        if isinstance(other, (PathLike, bytes, bytearray)) or hasattr(
            other, "read"
        ):
            return cls.load(other)
        return cls.from_lines(other)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build an object from an iterable of LTA lines."""
        if kwargs:
            raise TypeError(
                f"{cls.__name__}.from_lines() takes no options, but was "
                f"given {', '.join(sorted(kwargs))}."
            )
        obj = cls()
        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)
        for field in fields(cls):
            key = field.name if cls._HAS_KEYS else None
            parse = LtaFieldParser(key, field.type)
            setattr(obj, field.name, parse(lines))
        return obj

    # --- to -----------------------------------------------------------

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the LTA lines of the object.

        Extra keyword arguments are passed to the writer of each field.
        """
        for field in fields(type(self)):
            value = getattr(self, field.name)
            key = field.name if self._HAS_KEYS else None
            write = LtaFieldWriter(key, **kwargs)
            yield from write(value)

    def to_text(self, **kwargs) -> str:
        """Return the LTA text of the object, ending with a newline."""
        return super().to_text(**kwargs) + "\n"


class VolumeInfoParser(LtaParser):
    """Parser of a volume-geometry block.

    The block opens with a `"<NAME> volume info"` header line, followed by the
    fields that describe the source or destination volume.
    """

    @classmethod
    def from_lines(
        cls, lines: tx.Iterable[str], **kwargs
    ) -> tx.Optional[tx.Self]:
        """Build the block from LTA lines positioned at its start.

        If the next line is not the header of this block, `None` is returned
        and no line is consumed.
        """
        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)
        line = lines.peek()
        if not line:
            return None
        if line != (f"{cls.NAME} volume info"):
            return None
        next(lines)
        return super().from_lines(lines, **kwargs)

    def to_lines(self, **kwargs) -> tx.Generator[str]:
        """Yield the lines of the block, including its header."""
        yield f"{self.NAME} volume info"
        yield from super().to_lines(fmt={float: "{:.15e}"}, **kwargs)


class MatrixParser(LtaParser):
    """Parser of an affine matrix block.

    The first line gives the element type and the numbers of rows and
    columns, and that many rows of entries follow.
    """

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build the matrix block from LTA lines positioned at its start."""
        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)

        line = next(lines, None)
        if not line:
            warn("expected affine block, but got nothing", stacklevel=1)
            return cls()

        shape = _read_values(line, (int,) * 3)
        if not shape:
            warn(
                f'expected affine block with shape, but got: "{line}"',
                stacklevel=1,
            )
            return cls()
        nelem, nrow, ncol = shape
        dtype = {1: float, 2: complex}.get(nelem)

        matrix = []
        for _ in range(nrow):
            row = _read_values(next(lines), (dtype,) * (ncol))
            matrix.append(row)

        return cls(matrix=tuple(matrix))

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the matrix block.

        Entries are written at full double precision so that the matrix
        survives a round trip. FreeSurfer itself writes six decimals, but it
        reads both precisions.
        """
        dtype = self.dtype
        fmt = "{:+.15e} {:+.15e}   " if dtype is complex else "{:+.15e}  "
        yield _write_values((int(self.matrix_type), *self.shape))
        for row in self.matrix:
            yield _write_values(row, sep=0, fmt=fmt).rstrip()


# ----------------------------------------------------------------------
#   High-level utilities to (recursively) read and write fields
# ----------------------------------------------------------------------


class LtaFieldParser:
    """Reader of a single field of an LTA struct.

    Calling the reader consumes as many lines as the field needs, which is
    one line or a block of lines, and returns the parsed value.
    """

    def __init__(self, key: tx.Optional[str], type: tx.Any) -> None:
        """Parameters
        ----------
        key : str or None
            If given, the line must read `key = *values` with a matching key.
            Otherwise the line reads `*values`.
        type : type hint
            The type of the value, which is a plain type, a `Tuple` or an
            `Optional`. For an `Optional` type, the reader returns `None` when
            the line is missing or empty.
        """
        self.key = key
        self.optional, self.type = _is_optional(type)

    def __call__(self, lines: tx.Iterator[str]) -> tx.Any:
        """Consume the lines of the field and return its parsed value."""
        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)
        types = self.type

        # A field that is a struct defers to the parser of that struct.
        if isinstance(types, type) and issubclass(types, LtaParser):
            value = types.from_lines(lines)
            if value is None and not self.optional:
                raise ValueError(
                    f'expected block for key "{self.key}", but got nothing'
                )
            return value

        if tx.get_origin(types) in (tx.Tuple, tuple):
            types = tx.get_args(types)
        if isinstance(types, tuple) and len(types) == 1:
            types = types[0]

        line = lines.peek()
        if not line:
            if self.optional:
                return None
            raise ValueError(
                f'expected line for key "{self.key}", but got EOF'
            )

        if self.key:
            key, value = _read_key(line, {self.key: types})
            if key != self.key:
                if self.optional:
                    return None
                raise ValueError(f'expected key "{self.key}", but got "{key}"')
            next(lines)
            return value

        else:
            value = _read_values(line, types)
            if value is None:
                if self.optional:
                    return None
                raise ValueError(
                    f'expected value(s) of type "{types}", but got: "{line}"'
                )
            next(lines)
            return value


class LtaFieldWriter:
    """Writer of a single field of an LTA struct.

    Calling the writer yields the line, or the block of lines, of a value.
    """

    def __init__(self, key: tx.Optional[str], **kwargs) -> None:
        """Parameters
        ----------
        key : str or None
            If given, the line reads `key = value`. Otherwise the line reads
            `*values`.
        """
        self.key = key
        self.kwargs = kwargs

    def __call__(self, value: tx.Any, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of `value`."""
        if value is None:
            return
        if isinstance(value, LtaParser):
            yield from value.to_lines(**kwargs)
        elif self.key:
            kwargs = {**self.kwargs, **kwargs}
            yield _write_key(self.key, value, **kwargs)
        else:
            kwargs = {**self.kwargs, **kwargs}
            yield _write_values(value, **kwargs)


# ----------------------------------------------------------------------
#   Low level parsers and writers
# ----------------------------------------------------------------------

_INT = r"(\d+)"
_FLOAT = r"([\+\-]?\d+\.?\d*(?:[eE][\+\-]?\d+)?)"
_COMPLEX = f"(?P<real>{_FLOAT})" + r"\s+" + f"(?P<imag>{_FLOAT})"
_STR = r"(.*)\s*$"
_KEY_VALUE = r"^(\S+)\s*=\s*(.*)$"
_WHITESPACE = r"\s*"
_PATTERNS = {
    int: _INT,
    float: _FLOAT,
    complex: _COMPLEX,
    str: _STR,
    "key_value": _KEY_VALUE,
    "whitespace": _WHITESPACE,
}

_COMPILED_PATTERNS = {
    key: re.compile(pattern) for key, pattern in _PATTERNS.items()
}


def _get_pattern(type_: tx.Any, compiled: bool = False) -> re.Pattern:
    # An enum uses the pattern of its underlying type.
    if isinstance(type_, type):
        if issubclass(type_, int):
            type_ = int
        elif issubclass(type_, str):
            type_ = str
    if compiled:
        return _COMPILED_PATTERNS[type_]
    return _PATTERNS[type_]


def _to_type(value: str, type_: tx.Any) -> tx.Any:
    if isinstance(type_, type):
        if issubclass(type_, Enum) and issubclass(type_, int):
            return type_(int(value))
    return type_(value)


def _read_key(
    line: str, key_dict: tx.Optional[dict] = None
) -> tx.Tuple[tx.Optional[str], tx.Optional[str]]:
    """Read a `key = value` line.

    `key_dict` maps known keys to the type, or sequence of types, of their
    values. The value of a known key is converted, and the value of any other
    key is returned as a string. `(None, None)` is returned when the line is
    not a `key = value` line.
    """
    key_dict = key_dict or dict()

    match = _get_pattern("key_value", compiled=True).match(line)
    if not match:
        return None, None

    key, value = match.group(1), match.group(2)
    if key in key_dict:
        format = key_dict[key]
        if isinstance(format, type):
            pattern = _get_pattern(format, compiled=True)
        else:
            pattern = re.compile(
                r"\s*".join([_get_pattern(fmt) for fmt in format])
            )
        match = pattern.match(value)
        if match:
            if match.groupdict():
                value = complex(*map(float, match.groupdict().values()))
            elif isinstance(format, type):
                value = _to_type(match.group(1), format)
            else:
                value = tuple(
                    _to_type(v, t) for v, t in zip(match.groups(), format)
                )
    return key, value


def _read_values(
    line: str, format: tx.Union[type, tx.Sequence[type]]
) -> tx.Optional[tx.Union[str, int, float, tx.Tuple]]:
    """Read a `*values` line, given the type or types of the values.

    The result is a single value, a tuple of values, or `None` when the line
    does not match.
    """
    pattern = _get_pattern("whitespace")
    if isinstance(format, type):
        reformat = [_get_pattern(format)]
    else:
        reformat = [_get_pattern(fmt) for fmt in format]
    pattern = re.compile(pattern + r"\s*".join(reformat))
    value = pattern.match(line)
    if value:
        if isinstance(format, type):
            value = _to_type(value.group(1), format)
        else:
            value = tuple(
                _to_type(v, t) for v, t in zip(value.groups(), format)
            )
    return value


def _write_key(
    key: str,
    value: tx.Union[
        tx.Sequence[tx.Union[int, float, str, Enum]], int, float, str, Enum
    ],
    sep: tx.Union[int, str] = 1,
    fmt: tx.Optional[tx.Union[str, tx.Dict[tx.Type, str]]] = None,
) -> str:
    """Write a `key = value` line.

    Parameters
    ----------
    key : str
        The key.
    value : int, float, str, Enum or sequence of these
        The value or values.
    sep : int or str, default=1
        The separator between values. An integer gives a number of spaces.
    fmt : str or dict, optional
        The format of the values. A string applies to all values, a dictionary
        maps value types to formats, and `None` uses a default for each type.
    """
    return f"{key:9s} = {_write_values(value, sep, fmt)}".rstrip()


def _write_values(
    value: tx.Union[
        tx.Sequence[tx.Union[int, float, str, Enum]], int, float, str, Enum
    ],
    sep: tx.Union[int, str] = 1,
    fmt: tx.Optional[tx.Union[str, tx.Dict[tx.Type, str]]] = None,
) -> str:
    """Write a `*values` line.

    The parameters are those of [`_write_key`][].
    """
    fmt = fmt or {}

    if isinstance(value, Enum):
        # An enum is written as its value, with its name in a trailing
        # comment.
        return str(value.value) + f"  # {value.name}"

    if isinstance(value, str):
        if isinstance(fmt, dict):
            fmt = fmt.get(str, "{:s}")
        return fmt.format(value)

    if isinstance(value, int):
        if isinstance(fmt, dict):
            fmt = fmt.get(int, "{:d}")
        return fmt.format(value)

    if isinstance(value, float):
        if isinstance(fmt, dict):
            fmt = fmt.get(float, "{:6.4f}")
        return fmt.format(value)

    if isinstance(value, complex):
        # A complex number is written as two values separated by a space.
        if isinstance(fmt, dict):
            fmt = fmt.get(complex, "{:+.6f} {:+.6f}   ")
        return fmt.format(value.real, value.imag)

    else:
        if isinstance(sep, int):
            sep = " " * sep
        return sep.join([_write_values(v, fmt=fmt) for v in value])


def _is_optional(type_: tx.Any) -> tx.Tuple[bool, tx.Any]:
    """Return whether a type hint is optional, and the underlying type."""
    if tx.get_origin(type_) is tx.Optional:
        return True, tx.get_args(type_)[0]
    if tx.get_origin(type_) is tx.Union and type(None) in tx.get_args(type_):
        args = [arg for arg in tx.get_args(type_) if arg is not type(None)]
        if len(args) == 1:
            return True, args[0]
        return True, type_
    return False, type_
