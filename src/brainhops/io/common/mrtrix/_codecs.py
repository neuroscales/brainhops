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
    Return the NumPy dtype of an MRtrix data type, or `None` for `Bit`.

    The name is case-insensitive. As in MRtrix, the byte order is given by
    an `LE` or `BE` suffix, and is the native one otherwise.

    Raises
    ------
    ParserContentError
        If the name is not an MRtrix data type.
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
    Return the MRtrix data type that stores a dtype exactly.

    Booleans are stored as `Bit`, and multi-byte types always carry an `LE`
    or `BE` suffix, as MRtrix writes them.

    Raises
    ------
    WriterError
        If MRtrix cannot store the dtype, such as float16 or strings.
    """
    if isinstance(dtype, str) and dtype.strip().lower() == _BIT:
        return "Bit"
    if isinstance(dtype, str) and _is_mrtrix_spec(dtype):
        # normalise the spelling of an MRtrix data type
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
    Parse a layout into signed one-based strides.

    For example, `-0,-1,+2` becomes `(-1, -2, 3)`. The absolute value minus
    one is the rank, from 0 for the fastest axis, and the sign is the
    storage direction. Counting from one keeps the sign of rank 0.

    Raises
    ------
    ParserContentError
        If the layout is malformed, has the wrong number of axes, or does
        not give the axes distinct ranks.
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
    """Spell signed one-based strides as a layout."""
    return ",".join(
        ("+" if s > 0 else "-") + str(abs(int(s)) - 1) for s in strides
    )


def default_layout(ndim: int) -> tx.Tuple[int, ...]:
    """Return the Fortran-order layout `+0,+1,+2,...`."""
    return tuple(range(1, ndim + 1))


def _storage_shape(
    dim: tx.Sequence[int], strides: tx.Sequence[int]
) -> tx.Tuple[tx.Tuple[int, ...], tx.List[int]]:
    """
    Return the C-ordered file shape and the position of each header axis
    in it.
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
    View stored values as an array indexed in header axis order.

    The result is a view, so that a memory map stays one.
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
    Reorder an array of header axes into file order.

    This is the inverse of [`storage_to_image`][].
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
