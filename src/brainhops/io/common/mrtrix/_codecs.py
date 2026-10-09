"""MRtrix data types and layouts."""

# stdlib
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
    _BIT,
    _DTYPES,
    _NAMES,
)


def mrtrix_dtype(spec: str) -> tx.Optional[np.dtype]:
    """
    The numpy data type of an MRtrix data type, or `None` for `Bit`.

    The byte order is the suffix's (`LE`, `BE`), or the machine's when
    there is none, as MRtrix does. The specifier is matched
    case-insensitively.

    Raises
    ------
    ParserContentError
        If the specifier is not an MRtrix data type.
    """
    key = spec.strip().lower()
    if key == _BIT:
        return None
    order = "="
    if key.endswith(("le", "be")) and key[:-2] in _DTYPES:
        order = "<" if key.endswith("le") else ">"
        key = key[:-2]
    if key not in _DTYPES:
        raise ParserContentError(f"Unknown MRtrix data type: {spec!r}")
    if key in ("int8", "uint8"):
        order = "|"
    return np.dtype(order + _DTYPES[key])


def dtype_to_mrtrix(dtype: tx.Any) -> str:
    """
    The MRtrix data type that stores a numpy data type exactly.

    A multi-byte type is always given its byte order (`LE` or `BE`), as
    MRtrix itself writes it. A boolean is `Bit`. A type MRtrix cannot
    store (`float16`, `float128`, strings, ...) raises `WriterError`.
    """
    if isinstance(dtype, str) and dtype.strip().lower() == _BIT:
        return "Bit"
    if isinstance(dtype, str) and _is_mrtrix_spec(dtype):
        # Already an MRtrix specifier: normalise its spelling.
        np_dtype = mrtrix_dtype(dtype)
        return dtype_to_mrtrix(np_dtype)
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return "Bit"
    for key, code in _DTYPES.items():
        if np.dtype(code) == dtype.newbyteorder("="):
            name = _NAMES[key]
            if dtype.itemsize == 1:
                return name
            order = dtype.byteorder
            if order in ("=", "|"):
                order = "<" if np.little_endian else ">"
            return name + ("LE" if order == "<" else "BE")
    raise WriterError(f"MRtrix cannot store an array of type {dtype}.")


def _is_mrtrix_spec(spec: str) -> bool:
    try:
        mrtrix_dtype(spec)
    except ParserContentError:
        return False
    return True


# ----------------------------------------------------------------------
#   LAYOUT
# ----------------------------------------------------------------------


def parse_layout(spec: str, ndim: int) -> tx.Tuple[int, ...]:
    """
    Parse an MRtrix `layout` into signed, one-based symbolic strides.

    `-0,-1,+2` becomes `(-1, -2, 3)`: the absolute value minus one is
    the rank of the axis in the file (0 changes fastest), and the sign
    is the direction the axis is stored in. The one-based spelling is
    the one MRtrix uses internally, and keeps the sign of rank 0.

    Raises
    ------
    ParserContentError
        If the layout is malformed, has the wrong number of axes, or
        does not give each axis a distinct rank.
    """
    entries = [entry.strip() for entry in spec.split(",")]
    strides = []
    for entry in entries:
        match = re.fullmatch(r"([+-]?)(\d+)", entry)
        if not match:
            raise ParserContentError(f"Malformed MRtrix layout: {spec!r}")
        stride = int(match.group(2)) + 1
        strides.append(-stride if match.group(1) == "-" else stride)
    if len(strides) != ndim:
        raise ParserContentError(
            f"The MRtrix layout {spec!r} has {len(strides)} axes, but the "
            f"image has {ndim}."
        )
    if sorted(abs(s) for s in strides) != list(range(1, ndim + 1)):
        raise ParserContentError(
            f"The MRtrix layout {spec!r} does not give each of the {ndim} "
            f"axes a distinct rank between 0 and {ndim - 1}."
        )
    return tuple(strides)


def format_layout(strides: tx.Sequence[int]) -> str:
    """Spell signed one-based symbolic strides as an MRtrix `layout`."""
    return ",".join(
        ("+" if s > 0 else "-") + str(abs(int(s)) - 1) for s in strides
    )


def default_layout(ndim: int) -> tx.Tuple[int, ...]:
    """The layout of a Fortran-ordered array: `+0,+1,+2,...`."""
    return tuple(range(1, ndim + 1))


def _storage_shape(
    dim: tx.Sequence[int], strides: tx.Sequence[int]
) -> tx.Tuple[tx.Tuple[int, ...], tx.List[int]]:
    """
    The C-ordered shape the values have in the file, and where each
    header axis sits in it.

    The slowest axis (highest rank) comes first in the C-ordered shape.
    `position[i]` is the index, in that shape, of header axis `i`.
    """
    ndim = len(dim)
    ranks = [abs(s) - 1 for s in strides]
    position = [ndim - 1 - rank for rank in ranks]
    shape = [0] * ndim
    for axis, pos in enumerate(position):
        shape[pos] = int(dim[axis])
    return tuple(shape), position


def storage_to_image(
    stored: np.ndarray, dim: tx.Sequence[int], strides: tx.Sequence[int]
) -> np.ndarray:
    """
    View the values as they are stored as an array of the header's axes.

    `stored` holds the values in file order, either flat or already
    shaped. The result is indexed `[i0, i1, ...]` in the order of the
    header's axes, and is a view of `stored` (a transposition, and a
    reversal of the negative axes), so a memory-mapped file stays
    memory-mapped.
    """
    shape, position = _storage_shape(dim, strides)
    stored = stored.reshape(shape)
    image = stored.transpose(position)
    flips = tuple(
        slice(None, None, -1) if s < 0 else slice(None) for s in strides
    )
    return image[flips]


def image_to_storage(image: tx.Any, strides: tx.Sequence[int]) -> np.ndarray:
    """
    Reorder an array of the header's axes into the order of the file.

    The inverse of `storage_to_image`. The result is a C-ordered view
    when possible; `np.ascontiguousarray` of it is what the file holds.
    """
    image = np.asarray(image)
    ndim = image.ndim
    flips = tuple(
        slice(None, None, -1) if s < 0 else slice(None) for s in strides
    )
    image = image[flips]
    _, position = _storage_shape(image.shape, strides)
    order = [0] * ndim
    for axis, pos in enumerate(position):
        order[pos] = axis
    return image.transpose(order)
