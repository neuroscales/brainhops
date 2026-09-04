# stdlib
import re
from enum import Enum
from warnings import warn

# dependencies
import typing_extensions as tx
from bagof.magic import Magic

# externals
from brainhops._core.path import Path, PathLike

# core
from brainhops._core.peek import peekable_lines

# typing
_FileLike = tx.Union[tx.IO, PathLike, str]
_FileOrContentLike = tx.Union[_FileLike, bytes, tx.Iterable[str]]


# ----------------------------------------------------------------------
#   Parser classes with public methods inherited by LTAStruct
# ----------------------------------------------------------------------


class LTAParser(Magic):
    _HAS_KEYS = True

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff(cls, other: _FileOrContentLike) -> bool:
        if isinstance(other, str):
            if Path(other).exists():
                return cls.sniff_file(other)
            return cls.sniff_text(other)
        if isinstance(other, PathLike):
            return cls.sniff_file(other)
        if hasattr(other, "read"):
            return cls.sniff_file(other)
        if isinstance(other, bytes):
            return cls.sniff_bytes(other)
        # otherwise: assume it is an iterable of strings
        return cls.sniff_lines(other)

    @classmethod
    def sniff_file(cls, fileobj: _FileLike) -> bool:
        if isinstance(fileobj, str):
            return cls.sniff_file(Path(fileobj))
        if isinstance(fileobj, PathLike):
            return cls.sniff_text(fileobj.read_text())
        # otherwise: assume it is a file-like object open in text mode
        # TODO: ensure cursor is not moved by this operation
        return cls.sniff_text(fileobj.read())

    @classmethod
    def sniff_bytes(cls, bytes: bytes, encoding: str = "utf-8") -> bool:
        return cls.sniff_text(bytes.decode(encoding))

    @classmethod
    def sniff_text(cls, text: str) -> bool:
        first_line = next(peekable_lines(text.splitlines()))
        if first_line:
            return cls.sniff_line(first_line)
        return False

    @classmethod
    def sniff_line(cls, line: str) -> bool:
        if re.match(r"^type\s*=\s*\d+$", line.strip()):
            return True
        return False

    # --- from ---------------------------------------------------------

    @classmethod
    def from_(cls, other: _FileOrContentLike) -> tx.Self:
        """
        Build an object from a file (path, file-like object or iterable
        of lines).

        Parameters
        ----------
        other : str | PathLike | IO | Iterable[str]
            Input file, or its content.

        Returns
        -------
        obj
            The parsed object.
        """
        if isinstance(other, str):
            if Path(other).exists():
                return cls.from_file(Path(other))
            else:
                return cls.from_text(other)
        if isinstance(other, PathLike) or hasattr(other, "read"):
            return cls.from_file(other)
        if isinstance(other, bytes):
            return cls.from_bytes(other)
        return cls.from_lines(other)

    @classmethod
    def from_file(cls, fileobj: _FileLike) -> tx.Self:
        """
        Build an object from a file (path or file-like object).

        Parameters
        ----------
        fileobj : str | PathLike | IO
            Input file.

        Returns
        -------
        obj
            The parsed object.
        """
        if isinstance(fileobj, str):
            return cls.from_file(Path(fileobj))
        if isinstance(fileobj, PathLike):
            return cls.from_text(fileobj.read_text())
        # otherwise: assume it is a file-like object open in text mode
        # TODO: how do I know if it is a text or bytes IO?
        return cls.from_lines(fileobj)

    @classmethod
    def from_text(cls, text: str) -> tx.Self:
        """
        Build an object from a string in LTA format.

        Parameters
        ----------
        text : str
            The content of an LTA file.

        Returns
        -------
        obj
            The parsed object.
        """
        return cls.from_lines(text.splitlines())

    @classmethod
    def from_bytes(cls, bytes: bytes, encoding: str = "utf-8") -> tx.Self:
        """
        Build an object from bytes in LTA format.

        Parameters
        ----------
        bytes : bytes
            The byte content of an LTA file.
        encoding : str
            The encoding to use for decoding the input bytes.

        Returns
        -------
        obj
            The parsed object.
        """
        return cls.from_text(bytes.decode(encoding))

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str]) -> tx.Self:
        """
        Build an object from an iterable over lines on an LTA files.

        Parameters
        ----------
        lines : Iterable[str]
            Iterable content of an LTA file.

        Returns
        -------
        obj
            The parsed object.
        """
        obj = cls()
        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)
        for field in cls.__struct_fields__.values():
            key = field.name if cls._HAS_KEYS else None
            parse = LTAFieldParser(key, field.type)
            setattr(obj, field.name, parse(lines))
        return obj

    # --- to -----------------------------------------------------------

    def to_file(self, fileobj: tx.Union[tx.IO, PathLike, str]) -> None:
        """
        Write the object to a file (path or file-like object).

        Parameters
        ----------
        fileobj : str | PathLike | IO
            Output file.

        Returns
        -------
        None
        """
        if isinstance(fileobj, str):
            return self.to_file(Path(fileobj))
        if isinstance(fileobj, PathLike):
            return fileobj.write_text(self.to_text())
        # TODO: how do I know if it is a text or bytes IO?
        fileobj.write(self.to_text())

    def to_bytes(self, encoding: str = "utf-8") -> bytes:
        """
        Convert the object to bytes in LTA format.

        Parameters
        ----------
        encoding : str
            The encoding to use for the output bytes.

        Returns
        -------
        bytes
            The byte representation of the object in LTA format.
        """
        return self.to_text().encode(encoding)

    def to_text(self) -> str:
        """
        Convert the object to a string in LTA format.

        Returns
        -------
        str
            The string representation of the object in LTA format.
        """
        return "\n".join(self.to_lines())

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """
        Convert the object to an iterable over lines of an LTA file.

        Additional keyword arguments can be passed to the underlying
        field formatter.

        Returns
        -------
        Iterator[str]
            An iterable over lines of an LTA file representing the object.
        """
        for field in self.__struct_fields__.values():
            value = getattr(self, field.name)
            key = field.name if self._HAS_KEYS else None
            write = LTAFieldWriter(key, **kwargs)
            yield from write(value)


class VolumeInfoParser(LTAParser):
    @classmethod
    def from_lines(cls, lines: tx.Iterable[str]) -> tx.Optional[tx.Self]:
        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)
        line = lines.peek()
        if not line:
            return None
        if line != (f"{cls.NAME} volume info"):
            return None
        next(lines)  # consume line
        return super().from_lines(lines)

    def to_lines(self, **kwargs) -> tx.Generator[str]:
        yield f"{self.NAME} volume info"
        yield from super().to_lines(fmt={float: "{:.15e}"}, **kwargs)


class MatrixParser(LTAParser):
    @classmethod
    def from_lines(cls, lines: tx.Iterable[str]) -> tx.Self:
        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)

        # Read first line to check affine shape
        line = next(lines, None)
        if not line:
            warn("expected affine block, but got nothing", stacklevel=1)
            return cls()

        # Parse shape
        shape = _read_values(line, (int,) * 3)
        if not shape:
            warn(
                f'expected affine block with shape, but got: "{line}"',
                stacklevel=1,
            )
            return cls()
        nelem, nrow, ncol = shape
        dtype = {1: float, 2: complex}.get(nelem)

        # Parse affines
        matrix = []
        for _ in range(nrow):
            row = _read_values(next(lines), (dtype,) * (ncol))
            matrix.append(row)

        # Return object
        return cls(matrix=tuple(matrix))

    def to_lines(self) -> tx.Iterator[str]:
        dtype = self.dtype
        fmt = "{:+.6f} {:+.6f}   " if dtype is complex else "{:+.6f}  "
        yield _write_values((int(self.matrix_type), *self.shape))
        for row in self.matrix:
            yield _write_values(row, sep=0, fmt=fmt)


# ----------------------------------------------------------------------
#   High-level utilities to (recursively) read and write fields
# ----------------------------------------------------------------------


class LTAFieldParser:
    def __init__(self, key: tx.Optional[str], type: tx.Any) -> None:
        """
        Parameters
        ----------
        key : str or None
            If not None, the line should be in `key = *values` format,
            and the key should match this value.
            If None, the line should be in `*values` format.
        type : type hint
            The type of the value to be parsed.
            Can be a plain type, a `Tuple` or an `Optional` type hint.
            If the type is optional, the parser will return None if the
            line is missing or empty.
        """
        self.key = key
        self.optional, self.type = _is_optional(type)

    def __call__(self, lines: tx.Iterator[str]) -> tx.Any:
        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)
        types = self.type

        # If field is a struct, defer
        if isinstance(types, type) and issubclass(types, LTAParser):
            value = types.from_lines(lines)
            if value is None and not self.optional:
                raise ValueError(
                    f'expected block for key "{self.key}", but got nothing'
                )
            return value

        # Convert type hint to actual type(s) for parsing
        if tx.get_origin(types) in (tx.Tuple, tuple):
            types = tx.get_args(types)
        if isinstance(types, tuple) and len(types) == 1:
            types = types[0]

        # Read line
        line = lines.peek()
        if not line:
            if self.optional:
                return None
            raise ValueError(
                f'expected line for key "{self.key}", but got EOF'
            )

        # Parse key = value(s)
        if self.key:
            key, value = _read_key(line, {self.key: types})
            if key != self.key:
                if self.optional:
                    return None
                raise ValueError(f'expected key "{self.key}", but got "{key}"')
            next(lines)  # consume line
            return value

        # Parse value(s)
        else:
            value = _read_values(line, types)
            if value is None:
                if self.optional:
                    return None
                raise ValueError(
                    f'expected value(s) of type "{types}", but got: "{line}"'
                )
            next(lines)  # consume line
            return value


class LTAFieldWriter:
    def __init__(self, key: tx.Optional[str], **kwargs) -> None:
        """
        Parameters
        ----------
        key : str or None
            If not None, the line should be in `key = value` format,
            and the key should match this value.
            If None, the line should be in `*values` format.
        """
        self.key = key
        self.kwargs = kwargs

    def __call__(self, value: tx.Any, **kwargs) -> tx.Iterator[str]:
        if value is None:
            return
        if isinstance(value, LTAParser):
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

# Regex patterns for different value types
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
    # Handle enums by using the pattern of their underlying type
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
    """Read one `key = value` line from an LTA file

    Parameters
    ----------
    line : str
    format : type in {int, float, str}

    Returns
    -------
    object or tuple or None

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
            if match.groupdict():  # complex
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
    """Read one `*values` line from an LTA file

    Parameters
    ----------
    line : str
    format : [sequence of] type
        One of {int, float, str}

    Returns
    -------
    object or tuple or None

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
    """Write a `key = value` line in an LTA file.

    Parameters
    ----------
    key : str
        Key to write.
    value : [sequence of] int or float or str or enum
        Value(s) to write.
    sep : int | str
        The separator to use between values.
        If an integer, use that many spaces.
    fmt : str or dict | None
        The format string(s) to use for values.
        If a string, use that for all values.
        If a dict, use the format corresponding to the type of each value.
        If None, use a default format based on the type of each value.

    Returns
    -------
    str

    """
    return f"{key:9s} = {_write_values(value, sep, fmt)}"


def _write_values(
    value: tx.Union[
        tx.Sequence[tx.Union[int, float, str, Enum]], int, float, str, Enum
    ],
    sep: tx.Union[int, str] = 1,
    fmt: tx.Optional[tx.Union[str, tx.Dict[tx.Type, str]]] = None,
) -> str:
    """Write a `*values` line in an LTA file.

    Parameters
    ----------
    value : [sequence of] int or float or str or enum
        Value(s) to write.
    sep : int | str
        The separator to use between values.
        If an integer, use that many spaces.
    fmt : str or dict | None
        The format string(s) to use for values.
        If a string, use that for all values.
        If a dict, use the format corresponding to the type of each value.
        If None, use a default format based on the type of each value.

    Returns
    -------
    str

    """
    fmt = fmt or {}

    if isinstance(value, Enum):
        # Enum -> defer to its value + comment with its name
        return str(value.value) + f"  # {value.name}"

    if isinstance(value, str):
        # Str -> use appropriate format
        if isinstance(fmt, dict):
            fmt = fmt.get(str, "{:s}")
        return fmt.format(value)

    if isinstance(value, int):
        # Int -> use appropriate format
        if isinstance(fmt, dict):
            fmt = fmt.get(int, "{:d}")
        return fmt.format(value)

    if isinstance(value, float):
        # Float -> use appropriate format
        if isinstance(fmt, dict):
            fmt = fmt.get(float, "{:6.4f}")
        return fmt.format(value)

    if isinstance(value, complex):
        # Complex -> write as two values, separated by a single space.
        if isinstance(fmt, dict):
            fmt = fmt.get(complex, "{:+.6f} {:+.6f}   ")
        return fmt.format(value.real, value.imag)

    else:
        # Sequence of values -> separate by sep.
        if isinstance(sep, int):
            sep = " " * sep
        return sep.join([_write_values(v, fmt=fmt) for v in value])


def _is_optional(type_: tx.Any) -> tx.Tuple[bool, tx.Any]:
    """
    Check if a type hint is optional, and return the underlying type.
    """
    if tx.get_origin(type_) is tx.Optional:
        return True, tx.get_args(type_)[0]
    if tx.get_origin(type_) is tx.Union and type(None) in tx.get_args(
        type_
    ):
        args = [arg for arg in tx.get_args(type_) if arg is not type(None)]
        if len(args) == 1:
            return True, args[0]
        return True, type_
    return False, type_
