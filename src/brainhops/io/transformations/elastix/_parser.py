"""Reading and writing of the parameter maps of elastix parameter files.

A parameter map is a flat map from names to values, spelled in one of two
syntaxes (`itkParameterFileParser.cxx`): classic text (`.txt`), one
`(Name value ...)` per line with `//` comments, or TOML (`.toml`), one
`Name = value` or `Name = [value, ...]` per line with `#` comments.

Both are parsed into a dictionary from names to tuples, in which a quoted
value is a `str` and an unquoted one is an `int` or `float` when it reads
as a number. Transform parameters, millions of values for B-splines, are
parsed directly into a float64 array.
"""

import math
import re

import numpy as np
import typing_extensions as tx

from brainhops.io.base.parsers import ParserContentError

#: Parameters parsed into a float64 array instead of a tuple.
_ARRAYS = ("TransformParameters", "ITKTransformParameters")

#: Parameters that mark a transform. Registration parameter files also
#: name a `Transform`, but have none of these.
_TRANSFORM_KEYS = (
    "TransformParameters",
    "ITKTransformParameters",
    "NumberOfParameters",
)

_INT_RE = re.compile(r"^[+-]?\d+$")
# elastix rejects a name containing any of these characters
# (`GetParameterFromLine`).
_INVALID_NAME_RE = re.compile(r"[.,:;!@#$%^&\-+|<>?]")
_TOML_LINE_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+)\s*=\s*(?P<value>.*)$")

ParameterMap = tx.Dict[str, tx.Any]


# ----------------------------------------------------------------------
#   VALUES
# ----------------------------------------------------------------------


def _number(token: str) -> tx.Union[int, float, str]:
    """Read an unquoted token as a number, if it is one."""
    if _INT_RE.match(token):
        return int(token)
    try:
        value = float(token)
    except ValueError:
        return token
    # elastix `IsNumber` refuses "nan" and "inf", so they stay words.
    if math.isfinite(value):
        return value
    return token


def _values(name: str, values: tx.List[tx.Any]) -> tx.Any:
    """Store a parameter's values, as an array for transform parameters."""
    if name in _ARRAYS:
        try:
            return np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ParserContentError(
                f"The elastix parameter {name} holds a value that is not "
                f"a number."
            ) from e
    return tuple(values)


def _insert(pmap: ParameterMap, name: str, values: tx.List, line: str) -> None:
    if not name or _INVALID_NAME_RE.search(name):
        raise ParserContentError(
            f"Invalid elastix parameter name {name!r} in line: {line!r}"
        )
    if name in pmap:
        raise ParserContentError(
            f"The elastix parameter {name} is specified more than once."
        )
    pmap[name] = _values(name, values)


# ----------------------------------------------------------------------
#   TEXT SYNTAX
# ----------------------------------------------------------------------


def _split_text(line: str) -> tx.List[tx.Tuple[str, bool]]:
    """Split the inside of a `(...)` line into `(token, quoted)` pairs."""
    tokens = []
    current: tx.List[str] = []
    quoted = False
    for char in line:
        if char == '"':
            if quoted:
                tokens.append(("".join(current), True))
                current = []
            elif current:
                tokens.append(("".join(current), False))
                current = []
            quoted = not quoted
        elif char == " " and not quoted:
            if current:
                tokens.append(("".join(current), False))
                current = []
        else:
            current.append(char)
    if quoted:
        raise ParserContentError(
            f"This elastix parameter line has an odd number of quotes: "
            f"{line!r}"
        )
    if current:
        tokens.append(("".join(current), False))
    return tokens


def _strip_text(line: str) -> str:
    """Remove the comment, tabs and surrounding blanks from a text line.

    Unlike elastix, which cuts at the first `//`, a `//` inside quotes is
    kept, so that URLs and UNC paths survive.
    """
    line = line.replace("\t", " ")
    quoted = False
    for i, char in enumerate(line):
        if char == '"':
            quoted = not quoted
        elif not quoted and line.startswith("//", i):
            line = line[:i]
            break
    return line.strip()


def read_text_map(lines: tx.Iterable[str]) -> ParameterMap:
    """Parse the lines of a classic `.txt` parameter file."""
    pmap: ParameterMap = {}
    for raw in lines:
        line = _strip_text(raw)
        if not line:
            continue
        if not (line.startswith("(") and line.endswith(")")):
            raise ParserContentError(
                f"An elastix parameter line must be between brackets: {raw!r}"
            )
        tokens = _split_text(line[1:-1].strip())
        if not tokens or tokens[0][1]:
            raise ParserContentError(
                f"An elastix parameter line must start with a name: {raw!r}"
            )
        name = tokens[0][0]
        values = [
            tok if quoted else _number(tok) for tok, quoted in tokens[1:]
        ]
        _insert(pmap, name, values, raw)
    return pmap


def _format_value(value: tx.Any, toml: bool = False) -> str:
    if isinstance(value, str):
        if toml:
            value = value.replace("\\", "\\\\").replace('"', '\\"')
        elif '"' in value:
            # The text syntax has no escapes: a quote always ends a string.
            raise ValueError(
                f"An elastix text parameter cannot hold a quote: {value!r}"
            )
        return '"' + value + '"'
    if isinstance(value, (bool, np.bool_)):
        # elastix writes booleans as words, which the text syntax quotes.
        text = "true" if value else "false"
        return text if toml else f'"{text}"'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if value.is_integer() and abs(value) < 1e16:
        # elastix writes whole numbers without a decimal point.
        return str(int(value))
    return repr(value)


def format_text_map(pmap: ParameterMap) -> tx.Iterator[str]:
    """Yield the lines of a classic `.txt` parameter file."""
    for name, values in pmap.items():
        if isinstance(values, (str, int, float)):
            values = (values,)
        yield "(" + " ".join([name, *map(_format_value, values)]) + ")"


# ----------------------------------------------------------------------
#   TOML SYNTAX
# ----------------------------------------------------------------------
# elastix writes one parameter per line and reads TOML with toml++.
# Python parses TOML only from 3.11, so the subset that elastix writes
# (bare keys, basic strings, numbers, booleans and one-line arrays) is
# parsed here, and anything else is refused rather than misread.


def _split_toml_array(text: str, line: str) -> tx.List[str]:
    items: tx.List[str] = []
    current: tx.List[str] = []
    quoted = False
    escaped = False
    for char in text:
        if quoted:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            current.append(char)
        elif char == ",":
            items.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if quoted:
        raise ParserContentError(f"Unterminated string in line: {line!r}")
    last = "".join(current).strip()
    if last or items:
        items.append(last)
    if items and items[-1] == "":
        items.pop()
    return items


def _toml_scalar(text: str, line: str) -> tx.Any:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        body = text[1:-1]
        return re.sub(
            r"\\(.)",
            lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)),
            body,
        )
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1]
    if text in ("true", "false"):
        # elastix stores a boolean as the text it writes for it.
        return text
    value = _number(text.replace("_", ""))
    if isinstance(value, str):
        raise ParserContentError(
            f"Unsupported TOML value {text!r} in line: {line!r}"
        )
    return value


def _strip_toml(line: str) -> str:
    quoted = None
    for i, char in enumerate(line):
        if quoted:
            if char == quoted and (quoted == "'" or line[i - 1] != "\\"):
                quoted = None
        elif char in "\"'":
            quoted = char
        elif char == "#":
            return line[:i].strip()
    return line.strip()


def read_toml_map(lines: tx.Iterable[str]) -> ParameterMap:
    """Parse the lines of a TOML `.toml` parameter file."""
    pmap: ParameterMap = {}
    for raw in lines:
        line = _strip_toml(raw)
        if not line:
            continue
        match = _TOML_LINE_RE.match(line)
        if not match:
            raise ParserContentError(
                f"Not an elastix TOML parameter line: {raw!r}"
            )
        name, value = match.group("name"), match.group("value").strip()
        if value.startswith("["):
            if not value.endswith("]"):
                raise ParserContentError(
                    f"Multi-line TOML arrays are not supported: {raw!r}"
                )
            items = _split_toml_array(value[1:-1], raw)
            values = [_toml_scalar(item, raw) for item in items]
        else:
            values = [_toml_scalar(value, raw)]
        _insert(pmap, name, values, raw)
    return pmap


def format_toml_map(pmap: ParameterMap) -> tx.Iterator[str]:
    """Yield the lines of a TOML `.toml` parameter file."""
    for name, values in pmap.items():
        if isinstance(values, (str, int, float)):
            values = (values,)
        items = [_format_value(v, toml=True) for v in values]
        if len(items) == 1:
            yield f"{name} = {items[0]}"
        else:
            yield f"{name} = [" + ", ".join(items) + "]"


# ----------------------------------------------------------------------
#   ACCESSORS
# ----------------------------------------------------------------------


def is_transform_map(pmap: ParameterMap) -> bool:
    """Return whether a parameter map describes a transform."""
    return "Transform" in pmap and any(k in pmap for k in _TRANSFORM_KEYS)


def get_string(
    pmap: ParameterMap, name: str, default: tx.Optional[str] = None
) -> tx.Optional[str]:
    """Return the first value of a parameter as text."""
    values = pmap.get(name, ())
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not len(values):
        return default
    value = values[0]
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def get_floats(pmap: ParameterMap, name: str) -> tx.Optional[np.ndarray]:
    """Return the values of a parameter as a float64 vector, if it is set."""
    if name not in pmap:
        return None
    try:
        return np.asarray(pmap[name], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as e:
        raise ParserContentError(
            f"The elastix parameter {name} holds a value that is not a "
            f"number: {pmap[name]!r}"
        ) from e


def get_bool(pmap: ParameterMap, name: str, default: bool = False) -> bool:
    """Read a boolean parameter, which elastix writes as "true" or "false"."""
    value = get_string(pmap, name)
    if value is None:
        return default
    return value.strip().lower() == "true"
