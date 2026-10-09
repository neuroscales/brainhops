"""MRtrix voxel data: decoding, encoding and locating it."""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.io.base._utils_files import local_path as _local_path
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base.parsers import (
    ParserContentError,
    WriterError,
)

# this format
from ._codecs import (
    image_to_storage,
    storage_to_image,
)
from ._header import MrtrixHeader

# ----------------------------------------------------------------------
#   DATA
# ----------------------------------------------------------------------


def decode_data(header: MrtrixHeader, buffer: tx.Any) -> np.ndarray:
    """
    Decode the stored values into an array of the header's axes.

    `buffer` holds the voxel data, from its first byte: `bytes`, a
    `memoryview`, or a one-dimensional `uint8` array (a memory map). The
    result is a view of it whenever the data type allows one; packed bits
    are unpacked into a new boolean array. Intensity scaling is not
    applied.
    """
    count = header.count
    if header.is_bit:
        raw = np.frombuffer(buffer, dtype=np.uint8, count=header.nbytes)
        stored = np.unpackbits(raw, bitorder="big", count=count)
        stored = stored.astype(bool)
    else:
        if isinstance(buffer, np.ndarray):
            stored = buffer[: header.nbytes].view(header.dtype)
        else:
            stored = np.frombuffer(buffer, dtype=header.dtype, count=count)
    if stored.size < count:
        raise ParserContentError(
            f"The MRtrix data hold {stored.size} values, but the header "
            f"asks for {count}."
        )
    return storage_to_image(stored[:count], header.dim, header.layout)


def encode_data(header: MrtrixHeader, data: tx.Any) -> bytes:
    """
    Encode an array of the header's axes into the bytes of the file.

    The values are converted to the header's data type, rounding when a
    floating-point array is stored as integers, and laid out as the
    header's `layout` says.
    """
    array = np.asarray(data)
    if tuple(array.shape) != tuple(header.dim):
        raise WriterError(
            f"The data have shape {array.shape}, but the header says "
            f"{header.dim}."
        )
    stored = image_to_storage(array, header.layout)
    if header.is_bit:
        flat = np.ascontiguousarray(stored).reshape(-1).astype(bool)
        return np.packbits(flat, bitorder="big").tobytes()
    dtype = header.dtype
    if dtype.kind in "iu" and stored.dtype.kind in "fc":
        stored = np.round(np.real(stored))
    return np.ascontiguousarray(stored, dtype=dtype).tobytes()


def _read_buffer(file: tx.Any, offset: int, nbytes: int, mmap: bool) -> tx.Any:
    """
    The `nbytes` bytes at `offset` in an uncompressed data file.

    A local file is memory-mapped (read-only) unless `mmap` is false, so
    nothing is read until it is indexed. Any other path is read through
    its `open`.
    """
    local = _local_path(file)
    if local is not None and mmap and nbytes > 0:
        return np.memmap(
            local, dtype=np.uint8, mode="r", offset=offset, shape=(nbytes,)
        )
    with _open_path(file) as f:
        f.seek(offset)
        return f.read(nbytes)


def _is_gzip(file: tx.Any) -> bool:
    """Whether a path names a gzip-compressed file, from its content."""
    try:
        with _open_path(file) as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


def _data_location(header: MrtrixHeader) -> tx.Tuple[str, int]:
    """The `(name, offset)` of the data, checked as MRtrix does."""
    if header.file is None:
        raise ParserContentError(
            "The MRtrix header has no 'file' entry, so its data cannot be "
            "found."
        )
    name, offset = header.file
    if name == "." and offset <= 0:
        raise ParserContentError(
            "Invalid offset for an MRtrix image embedded in its header "
            f"file: {offset}"
        )
    return name, offset
