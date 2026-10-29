"""Low-level MGH/MGZ helpers: the leading header fields and streams."""

# dependencies
import typing_extensions as tx

# this format
from ._constants import MGH_TYPES, PREFIX


def read_prefix(fileobj: tx.BinaryIO) -> tx.Optional[tuple]:
    """
    Read the leading header fields of a decompressed stream, or `None` if
    the stream is too short.
    """
    raw = fileobj.read(PREFIX.size)
    if len(raw) < PREFIX.size:
        return None
    return PREFIX.unpack(raw)


def valid_prefix(prefix: tx.Optional[tuple]) -> bool:
    """Tell whether the leading fields describe an MGH volume."""
    if prefix is None:
        return False
    version, *dims, dtype, _dof, _flag = prefix
    return version == 1 and all(d > 0 for d in dims) and dtype in MGH_TYPES


def seekable(fileobj: tx.IO) -> bool:
    """Tell whether a stream can seek (no answer means no)."""
    try:
        return bool(fileobj.seekable())
    except Exception:
        return False
