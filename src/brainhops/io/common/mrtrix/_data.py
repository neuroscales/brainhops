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
    Decode stored values into an array of header axes.

    The buffer holds the voxel data from their first byte. The result is a
    view of the buffer, except for `Bit` data, which are unpacked into a
    new boolean array. Intensity scaling is not applied.

    Raises
    ------
    ParserContentError
        If the buffer holds fewer values than the header requires.
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
    Encode an array of header axes into the bytes of the file.

    Floats stored as integers are rounded.

    Raises
    ------
    WriterError
        If the shape of the array disagrees with the header.
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
    Read `nbytes` bytes at `offset` in an uncompressed data file,
    memory-mapping a local file unless `mmap` is false.
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
    """Tell whether a path names a gzip file, judged by its content."""
    try:
        with _open_path(file) as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


def _data_location(header: MrtrixHeader) -> tx.Tuple[str, int]:
    """
    Return the `(name, offset)` of the data, checked as MRtrix does.

    Raises
    ------
    ParserContentError
        If the header has no `file` entry, or an embedded image has no
        positive offset.
    """
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
