"""NRRD types, and the syntax of header field values."""

# stdlib
import math
import re

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.io.base.parsers import (
    ParserContentError,
    WriterError,
)

# this format
from ._constants import (
    NONE,
    TYPE_NAMES,
    TYPES,
)

# ----------------------------------------------------------------------
#   TYPES
# ----------------------------------------------------------------------


def nrrd_dtype(name: str, endian: tx.Optional[str] = None) -> np.dtype:
    """Return the NumPy data type of a NRRD type name and byte order.

    Raises
    ------
    ParserContentError
        If the type is unknown.
    """
    key = " ".join(str(name).split()).lower()
    if key not in TYPES:
        raise ParserContentError(f"Unsupported NRRD type: {name!r}")
    dtype = np.dtype(TYPES[key])
    if endian is not None and dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    return dtype


def dtype_to_nrrd(dtype: tx.Any) -> str:
    """Return the NRRD type name of a NumPy data type.

    Raises
    ------
    WriterError
        If NRRD has no matching type.
    """
    if isinstance(dtype, str) and " ".join(dtype.split()).lower() in TYPES:
        return TYPE_NAMES[TYPES[" ".join(dtype.split()).lower()]]
    try:
        dtype = np.dtype(dtype)
    except TypeError as error:
        raise WriterError(f"NRRD has no type for {dtype!r}.") from error
    if dtype.kind == "b":
        return "uint8"
    if dtype.kind == "f" and dtype.itemsize < 4:
        return "float"
    code = f"{dtype.kind}{dtype.itemsize}"
    if code not in TYPE_NAMES:
        raise WriterError(f"NRRD has no type for {dtype}.")
    return TYPE_NAMES[code]


# ----------------------------------------------------------------------
#   VALUES
# ----------------------------------------------------------------------


def unescape(text: str) -> str:
    """Decode the escaped characters of a NRRD string.

    A NRRD header escapes a newline as `\\n` and any other character as
    a backslash followed by that character.

    Parameters
    ----------
    text : str
        The escaped text.

    Returns
    -------
    str
        The decoded text.
    """
    return re.sub(
        r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1), text
    )


def escape(text: str) -> str:
    """Escape the backslashes and the newlines of a NRRD string.

    This is the inverse of [`unescape`][].

    Parameters
    ----------
    text : str
        The text to escape.

    Returns
    -------
    str
        The escaped text.
    """
    return str(text).replace("\\", "\\\\").replace("\n", "\\n")


def parse_float(token: str) -> float:
    """Read a number of a NRRD header.

    Parameters
    ----------
    token : str
        The text of the number.

    Returns
    -------
    float
        The number.

    Raises
    ------
    ParserContentError
        If the text is not a number.
    """
    try:
        return float(token)
    except ValueError:
        raise ParserContentError(f"Not a NRRD number: {token!r}") from None


def format_float(value: float) -> str:
    """Write a number of a NRRD header.

    A whole number below `1e15` is written without a decimal point, and
    any other number is written with the shortest text that reads back
    as the same float.

    Parameters
    ----------
    value : float
        The number.

    Returns
    -------
    str
        The text of the number.
    """
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def parse_vectors(value: str) -> tx.List[tx.Optional[tx.List[float]]]:
    """Parse a list of vectors.

    `"none (1,0,0) (0,1,0)"` gives `[None, [1, 0, 0], [0, 1, 0]]`.
    """
    out: tx.List[tx.Optional[tx.List[float]]] = []
    for token in re.findall(r"\([^)]*\)|[^\s()]+", value):
        if token.lower() in NONE:
            out.append(None)
        elif token.startswith("("):
            out.append([parse_float(t) for t in token[1:-1].split(",")])
        else:
            raise ParserContentError(f"Not a NRRD vector: {token!r}")
    return out


def format_vectors(vectors: tx.Iterable[tx.Any]) -> str:
    """Write a list of vectors of a NRRD header.

    Each vector is written in parentheses, with its components separated
    by commas, and a missing vector is written as `none`.

    Parameters
    ----------
    vectors : Iterable[Any]
        The vectors, each a sequence of numbers or `None`.

    Returns
    -------
    str
        The text of the vectors, separated by spaces.
    """
    return " ".join(
        "none"
        if v is None
        else "(" + ",".join(format_float(x) for x in v) + ")"
        for v in vectors
    )


def parse_strings(value: str) -> tx.List[str]:
    """Parse quoted strings: `'"mm" "" "s"'` gives `["mm", "", "s"]`."""
    return [unescape(s) for s in re.findall(r'"((?:[^"\\]|\\.)*)"', value)]


def format_strings(values: tx.Iterable[str]) -> str:
    """Write a list of quoted strings of a NRRD header.

    Each string is written between double quotes, with its backslashes
    and double quotes escaped.

    Parameters
    ----------
    values : Iterable[str]
        The strings.

    Returns
    -------
    str
        The quoted strings, separated by spaces.
    """
    return " ".join(
        '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'
        for v in values
    )
