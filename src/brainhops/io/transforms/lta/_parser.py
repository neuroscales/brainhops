# stdlib
import re
from enum import Enum
from os import PathLike
from itertools import tee
from pathlib import Path as LocalPath
from warnings import warn

# externals
import typing_extensions as _tx

# internals
from brainhops._ext.struct import Struct

# optionals
try:
    from upath import UPath
except ImportError:
    UPath = None
try:
    from cloudpathlib import AnyPath
except ImportError:
    AnyPath = None

Path = UPath or AnyPath or LocalPath


# Regex patterns for different value types
_INT = r'(\d+)'
_FLOAT = r'([\+\-]?\d+\.?\d*(?:[eE][\+\-]?\d+)?)'
_COMPLEX = f'(?P<real>{_FLOAT})' + r'\s+' + f'(?P<imag>{_FLOAT})'
_STR = r'(.*)\s*$'
_KEY_VALUE = r'^(\S+)\s*=\s*(.*)$'
_WHITESPACE = r'\s*'
_PATTERNS = {
    int: _INT, float: _FLOAT, complex: _COMPLEX, str: _STR,
    'key_value': _KEY_VALUE, 'whitespace': _WHITESPACE,
}

_COMPILED_PATTERNS = {
    key: re.compile(pattern)
    for key, pattern in _PATTERNS.items()
}

def _get_pattern(type_: _tx.Any, compiled: bool = False) -> re.Pattern:
    # Handle enums by using the pattern of their underlying type
    if isinstance(type_, type):
        if issubclass(type_, int):
            type_ = int
        elif issubclass(type_, str):
            type_ = str
    if compiled:
        return _COMPILED_PATTERNS[type_]
    return _PATTERNS[type_]


def _to_type(value: str, type_: _tx.Any) -> _tx.Any:
    if isinstance(type_, type):
        if issubclass(type_, Enum) and issubclass(type_, int):
            return type_(int(value))
    return type_(value)


def _read_key(
    line: str, key_dict: _tx.Optional[dict] = None
) -> _tx.Tuple[_tx.Optional[str], _tx.Optional[str]]:
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

    match = _get_pattern('key_value', compiled=True).match(line)
    if not match:
        return None, None

    key, value = match.group(1), match.group(2)
    if key in key_dict:
        format = key_dict[key]
        if isinstance(format, type):
            pattern = _get_pattern(format, compiled=True)
        else:
            pattern = re.compile(r'\s*'.join([_get_pattern(fmt) for fmt in format]))
        match = pattern.match(value)
        if match:
            if match.groupdict():  # complex
                value = complex(*map(float, match.groupdict().values()))
            elif isinstance(format, type):
                value = _to_type(match.group(1), format)
            else:
                value = tuple(_to_type(v, t)
                              for v, t in zip(match.groups(), format))
    return key, value


def _read_values(
    line: str, format: _tx.Union[type, _tx.Sequence[type]]
) -> _tx.Optional[_tx.Union[str, int, float, _tx.Tuple]]:
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
    pattern = _get_pattern('whitespace')
    if isinstance(format, type):
        reformat = [_get_pattern(format)]
    else:
        reformat = [_get_pattern(fmt) for fmt in format]
    pattern = re.compile(pattern + r'\s*'.join(reformat))
    value = pattern.match(line)
    if value:
        if isinstance(format, type):
            value = _to_type(value.group(1), format)
        else:
            value = tuple(_to_type(v, t)
                          for v, t in zip(value.groups(), format))
    return value


def _write_key(
    key: str,
    value: _tx.Any,
    sep: _tx.Union[int, str] = 1,
    fmt: _tx.Optional[_tx.Union[str, _tx.Dict[_tx.Type, str]]] = None
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
    return f'{key:9s} = {_write_values(value, sep, fmt)}'


def _write_values(
    value: _tx.Any,
    sep: _tx.Union[int, str] = 1,
    fmt: _tx.Optional[_tx.Union[str, _tx.Dict[_tx.Type, str]]] = None
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
            fmt = fmt.get(str, '{:s}')
        return fmt.format(value)

    if isinstance(value, int):
        # Int -> use appropriate format
        if isinstance(fmt, dict):
            fmt = fmt.get(int, '{:d}')
        return fmt.format(value)

    if isinstance(value, float):
        # Float -> use appropriate format
        if isinstance(fmt, dict):
            fmt = fmt.get(float, '{:6.4f}')
        return fmt.format(value)

    if isinstance(value, complex):
        # Complex -> write as two values, separated by a single space.
        if isinstance(fmt, dict):
            fmt = fmt.get(complex, '{:+.6f} {:+.6f}   ')
        return fmt.format(value.real, value.imag)

    else:
        # Sequence of values -> separate by sep.
        if isinstance(sep, int):
            sep = ' ' * sep
        return sep.join([_write_values(v, fmt=fmt) for v in value])


def _is_optional(type_: _tx.Any) -> _tx.Tuple[bool, _tx.Any]:
    if _tx.get_origin(type_) is _tx.Optional:
        return True, _tx.get_args(type_)[0]
    if _tx.get_origin(type_) is _tx.Union and type(None) in _tx.get_args(type_):
        args = [arg for arg in _tx.get_args(type_) if arg is not type(None)]
        if len(args) == 1:
            return True, args[0]
        return True, type_
    return False, type_


class _iterlines:
    """
    A peekable iterator over lines of an LTA file.

    Automatically trims whitespace and comments, and skips empty lines.
    """

    EMPTY = object()

    def __init__(self, lines: _tx.Iterable[str]):
        if not hasattr(lines, '__next__'):
            lines = iter(lines)
        self.lines = lines
        self.peeked = self.EMPTY

    def __next__(self) -> str:
        if self.peeked is not self.EMPTY:
            line, self.peeked = self.peeked, self.EMPTY
        else:
            line = self._next_valid()
        if line is None:
            raise StopIteration
        return line

    def __iter__(self):
        while True:
            try:
                yield next(self)
            except StopIteration:
                return

    def peek(self) -> _tx.Optional[str]:
        if self.peeked is not self.EMPTY:
            return self.peeked
        try:
            self.peeked = self._next_valid()
            return self.peeked
        except StopIteration:
            return None

    def _next_valid(self) -> _tx.Optional[str]:
        while True:
            line = next(self.lines, None)
            if line is None:
                return None
            line = line.split('\r\n')[0]  # remove eol (windows)
            line = line.split('\n')[0]    # remove eol (unix)
            line = line.split('#')[0]     # remove hanging comments
            line = line.strip()           # remove leading/trailing whitespaces
            if line:
                return line


class LTAFieldParser:

    def __init__(self, key: _tx.Optional[str], type: _tx.Any):
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

    def __call__(self, lines: _tx.Iterator[str]) -> _tx.Any:
        if not isinstance(lines, _iterlines):
            lines = _iterlines(lines)
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
        if _tx.get_origin(types) in (_tx.Tuple, tuple):
            types = _tx.get_args(types)
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
                raise ValueError(
                    f'expected key "{self.key}", but got "{key}"'
                )
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

    def __init__(self, key: _tx.Optional[str], **kwargs):
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

    def __call__(self, value: _tx.Any, **kwargs) -> _tx.Iterator[str]:
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


class LTAParser(Struct):

    _HAS_KEYS = True

    @classmethod
    def from_(
        cls, thing: _tx.Union[_tx.IO, PathLike, str, _tx.Iterable[str]]
    ):
        if isinstance(thing, (str, PathLike)) or hasattr(thing, 'read'):
            return cls.from_file(Path(thing))
        return cls.from_lines(thing)

    @classmethod
    def from_file(
        cls, fileobj: _tx.Union[_tx.IO, PathLike, str]
    ) -> _tx.Self:
        if isinstance(fileobj, str):
            return cls.from_file(Path(fileobj))
        if isinstance(fileobj, PathLike):
            with open(fileobj, 'r') as f:
                return cls.from_file(f)
        return cls.from_lines(fileobj)

    @classmethod
    def from_lines(cls, lines: _tx.Iterable[str]) -> _tx.Self:
        obj = cls()
        if not isinstance(lines, _iterlines):
            lines = _iterlines(lines)
        for field in cls.__struct_fields__.values():
            key = field.name if cls._HAS_KEYS else None
            parse = LTAFieldParser(key, field.type)
            setattr(obj, field.name, parse(lines))
        return obj

    def to_lines(self, **kwargs) -> _tx.Iterator[str]:
        for field in self.__struct_fields__.values():
            value = getattr(self, field.name)
            key = field.name if self._HAS_KEYS else None
            write = LTAFieldWriter(key, **kwargs)
            yield from write(value)


class VolumeInfoParser(LTAParser):

    @classmethod
    def from_lines(cls, lines: _tx.Iterable[str]) -> _tx.Optional[_tx.Self]:
        if not isinstance(lines, _iterlines):
            lines = _iterlines(lines)
        line = lines.peek()
        if not line:
            return None
        if line != (f'{cls.NAME} volume info'):
            return None
        next(lines)  # consume line
        return super().from_lines(lines)

    def to_lines(self, **kwargs):
        yield f'{self.NAME} volume info'
        yield from super().to_lines(fmt={float: '{:.15e}'}, **kwargs)


class MatrixParser(LTAParser):

    @classmethod
    def from_lines(cls, lines: _tx.Iterable[str]) -> _tx.Self:
        if not isinstance(lines, _iterlines):
            lines = _iterlines(lines)

        # Read first line to check affine shape
        line = next(lines, None)
        if not line:
            warn(f'expected affine block, but got nothing')
            return cls()

        # Parse shape
        shape = _read_values(line, (int,) * 3)
        if not shape:
            warn(f'expected affine block with shape, but got: "{line}"')
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

    def to_lines(self) -> _tx.Iterator[str]:
        dtype = self.dtype
        fmt = '{:+.6f} {:+.6f}   ' if dtype is complex else '{:+.6f}  '
        yield _write_values((int(self.matrix_type), *self.shape))
        for row in self.matrix:
            yield _write_values(row, sep=0, fmt=fmt)
