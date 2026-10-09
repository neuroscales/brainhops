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
    _NONE,
    _TYPE_NAMES,
    _TYPES,
)

# ----------------------------------------------------------------------
#   TYPES
# ----------------------------------------------------------------------


def nrrd_dtype(name: str, endian: tx.Optional[str] = None) -> np.dtype:
    """
    The numpy type of a NRRD `type`, in the byte order of `endian`
    (`"little"` or `"big"`; the native order when it is not given).

    Raises
    ------
    ParserContentError
        If the type is unknown, or is `block`.
    """
    key = " ".join(str(name).split()).lower()
    if key not in _TYPES:
        raise ParserContentError(f"Unsupported NRRD type: {name!r}")
    dtype = np.dtype(_TYPES[key])
    if endian is not None and dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    return dtype


def dtype_to_nrrd(dtype: tx.Any) -> str:
    """
    The NRRD `type` of a numpy type (or of a NRRD type name).

    Booleans are stored as `uint8` and half floats as `float`.

    Raises
    ------
    WriterError
        If NRRD has no type for it (complex numbers, strings, ...).
    """
    if isinstance(dtype, str) and " ".join(dtype.split()).lower() in _TYPES:
        return _TYPE_NAMES[_TYPES[" ".join(dtype.split()).lower()]]
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return "uint8"
    if dtype.kind == "f" and dtype.itemsize < 4:
        return "float"
    code = f"{dtype.kind}{dtype.itemsize}"
    if code not in _TYPE_NAMES:
        raise WriterError(f"NRRD has no type for {dtype}.")
    return _TYPE_NAMES[code]


# ----------------------------------------------------------------------
#   VALUES
# ----------------------------------------------------------------------


def _unescape(text: str) -> str:
    return re.sub(
        r"\\(.)", lambda m: "\n" if m.group(1) == "n" else m.group(1), text
    )


def _escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("\n", "\\n")


def _float(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        raise ParserContentError(f"Not a NRRD number: {token!r}") from None


def _format_float(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _parse_vectors(value: str) -> tx.List[tx.Optional[tx.List[float]]]:
    """`none (1,0,0) (0,1,0)` -> `[None, [1, 0, 0], [0, 1, 0]]`."""
    out: tx.List[tx.Optional[tx.List[float]]] = []
    for token in re.findall(r"\([^)]*\)|[^\s()]+", value):
        if token.lower() in _NONE:
            out.append(None)
        elif token.startswith("("):
            out.append([_float(t) for t in token[1:-1].split(",")])
        else:
            raise ParserContentError(f"Not a NRRD vector: {token!r}")
    return out


def _format_vectors(vectors: tx.Iterable[tx.Any]) -> str:
    return " ".join(
        "none"
        if v is None
        else "(" + ",".join(_format_float(x) for x in v) + ")"
        for v in vectors
    )


def _parse_strings(value: str) -> tx.List[str]:
    """`"mm" "" "s"` -> `["mm", "", "s"]`."""
    return [_unescape(s) for s in re.findall(r'"((?:[^"\\]|\\.)*)"', value)]


def _format_strings(values: tx.Iterable[str]) -> str:
    return " ".join(
        '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'
        for v in values
    )
