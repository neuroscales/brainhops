"""AFNI `.BRIK` data: sub-brick types, decoding, scaling and
encoding, and the files of a dataset."""

# stdlib
import bz2
import gzip
import os
import re

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.streams import open_compressed
from brainhops.io.base._utils_files import local_path as _local_path
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base.parsers import (
    ParserContentError,
    ParserExistsError,
    WriterError,
)

# this format
from ._constants import (
    _BRICK_CODES,
    _BRICK_DTYPES,
    _BRICK_NAMES,
    _BRIK_SUFFIXES,
    _READABLE_SUFFIXES,
)
from ._header import AfniHeader

# ----------------------------------------------------------------------
#   DATA
# ----------------------------------------------------------------------


def brick_dtype(dtype: tx.Any) -> np.dtype:
    """
    The AFNI data type that stores an array of type `dtype` (without byte
    order), or the one named (`"byte"`, `"short"`, `"int"`, `"float"`,
    `"double"`, `"complex"`).

    A type AFNI has is kept. Booleans become bytes, `int8` shorts,
    `uint16` and `uint32` ints, wider integers doubles, `float16` floats
    and `complex128` complex (single precision).
    """
    if isinstance(dtype, str) and dtype.lower() in _BRICK_NAMES:
        return _BRICK_DTYPES[_BRICK_NAMES[dtype.lower()]]
    dtype = np.dtype(dtype).newbyteorder("=")
    if dtype in _BRICK_CODES:
        return dtype
    if dtype.kind == "b":
        return np.dtype(np.uint8)
    if dtype == np.int8:
        return np.dtype(np.int16)
    if dtype.kind in "iu" and dtype.itemsize <= 4:
        return np.dtype(np.int32) if dtype != np.uint32 else np.dtype(float)
    if dtype.kind in "iu":
        return np.dtype(np.float64)
    if dtype.kind == "f":
        return np.dtype(np.float32 if dtype.itemsize < 8 else np.float64)
    if dtype.kind == "c":
        return np.dtype(np.complex64)
    raise WriterError(f"AFNI cannot store data of type {dtype}.")


def brick_code(dtype: tx.Any) -> int:
    """The `BRICK_TYPES` code of an AFNI data type."""
    return _BRICK_CODES[np.dtype(dtype).newbyteorder("=")]


def decode_bricks(header: AfniHeader, buffer: tx.Any) -> np.ndarray:
    """
    Decode the BRIK into a `(nx, ny, nz, nvals)` array.

    `buffer` holds the (uncompressed) BRIK: `bytes`, a `memoryview` or a
    one-dimensional `uint8` array (a memory map). The result is a
    Fortran-ordered view of it when every sub-brick has the same type;
    sub-bricks of different types are converted to a common type and
    copied. Scaling (`BRICK_FLOAT_FACS`) is not applied.
    """
    shape = header.shape
    nvox = int(np.prod(shape))
    dtypes = header.dtypes
    size = len(buffer) if not isinstance(buffer, np.ndarray) else buffer.size
    if size < header.nbytes:
        raise ParserContentError(
            f"The AFNI BRIK holds {size} bytes, but the header asks for "
            f"{header.nbytes}."
        )

    def _brick(offset: int, dtype: np.dtype) -> np.ndarray:
        if isinstance(buffer, np.ndarray):
            chunk = buffer[offset : offset + nvox * dtype.itemsize]
            return chunk.view(dtype)
        return np.frombuffer(buffer, dtype=dtype, count=nvox, offset=offset)

    if len(set(dtypes)) == 1:
        if isinstance(buffer, np.ndarray):
            flat = buffer[: header.nbytes].view(dtypes[0])
        else:
            count = nvox * header.nvals
            flat = np.frombuffer(buffer, dtype=dtypes[0], count=count)
        return flat.reshape(shape + (header.nvals,), order="F")

    common = np.result_type(*dtypes)
    out = np.empty(shape + (header.nvals,), dtype=common, order="F")
    offset = 0
    for i, dtype in enumerate(dtypes):
        out[..., i] = _brick(offset, dtype).reshape(shape, order="F")
        offset += nvox * dtype.itemsize
    return out


def scale_bricks(header: AfniHeader, stored: tx.Any) -> tx.Any:
    """
    Apply the `BRICK_FLOAT_FACS` of the header to the stored values
    (whose last axis holds the sub-bricks, unless there is only one).

    Without a factor, the stored values are returned as they are, so a
    memory map stays one. With one, they are read and scaled, to at least
    single precision.
    """
    facs = np.asarray(header.float_facs, dtype=np.float64)
    if not np.any(facs) or np.all((facs == 0) | (facs == 1)):
        return stored
    facs = np.where(facs == 0, 1.0, facs)
    stored = np.asarray(stored)
    dtype = np.result_type(stored.dtype, np.float32)
    if stored.ndim == len(header.shape):
        return stored.astype(dtype) * dtype.type(facs[0])
    return stored.astype(dtype) * facs.astype(dtype)


def encode_bricks(header: AfniHeader, data: tx.Any) -> tx.Iterator[bytes]:
    """
    Encode a `(nx, ny, nz, nvals)` array (or `(nx, ny, nz)` for a single
    sub-brick) into the bytes of the BRIK, one sub-brick at a time.

    The values are divided by the header's `BRICK_FLOAT_FACS` (where not
    zero) and converted to its types and byte order, rounding when a
    floating-point array is stored as integers.

    Raises
    ------
    WriterError
        If the shape disagrees with the header, or integers do not fit
        the stored type.
    """
    array = np.asarray(data)
    shape = header.shape
    if array.ndim == 3:
        array = array[..., None]
    if tuple(array.shape) != tuple(shape) + (header.nvals,):
        raise WriterError(
            f"The data have shape {np.shape(data)}, but the AFNI header "
            f"says {shape} with {header.nvals} sub-brick(s)."
        )
    for i, (dtype, fac) in enumerate(zip(header.dtypes, header.float_facs)):
        brick = array[..., i]
        if fac:
            brick = brick / fac
        if dtype.kind in "iu":
            if brick.dtype.kind in "fc":
                brick = np.round(np.real(brick))
            if brick.size:
                info = np.iinfo(dtype)
                low, high = np.nanmin(brick), np.nanmax(brick)
                if low < info.min or high > info.max:
                    raise WriterError(
                        f"Sub-brick {i} holds values in [{low}, {high}], "
                        f"which AFNI's {dtype.name} cannot store."
                    )
        elif dtype.kind == "f" and brick.dtype.kind == "c":
            brick = np.real(brick)
        yield np.asarray(brick, dtype=dtype).tobytes(order="F")


# ----------------------------------------------------------------------
#   FILES
# ----------------------------------------------------------------------


def afni_dataset_files(
    filename: path.FilenameLike,
) -> tx.Tuple[tx.Any, tx.Any, str]:
    """
    The `.HEAD` and `.BRIK` files of the dataset a path names.

    The path may be that of the `.HEAD`, of the `.BRIK` (compressed or
    not), or the dataset's name without extension (`anat+orig`). The
    BRIK is the first one found among `.BRIK`, `.BRIK.gz`, `.BRIK.bz2`,
    `.BRIK.Z` (AFNI's own order), or the uncompressed name if none
    exists.

    Returns
    -------
    head : Path
        The `.HEAD` file.
    brik : Path
        The `.BRIK` file (which may not exist).
    stem : str
        The dataset's name, without directory nor extension.
    """
    if isinstance(filename, str):
        filename = path.Path(filename)
    name = filename.name
    stem, head_ext, brik_ext = name, ".HEAD", ".BRIK"
    match = re.search(r"\.(HEAD|head|BRIK|brik)(\.[A-Za-z0-9]+)?$", name)
    if match and match.group(2) in (None,) + _BRIK_SUFFIXES[1:]:
        stem = name[: match.start()]
        if match.group(1).islower():
            head_ext, brik_ext = ".head", ".brik"
    parent = filename.parent
    head = parent / (stem + head_ext)
    brik = None
    for suffix in _BRIK_SUFFIXES:
        candidate = parent / (stem + brik_ext + suffix)
        if path.exists(candidate):
            brik = candidate
            break
    if brik is None:
        brik = parent / (stem + brik_ext)
    return head, brik, stem


def _brik_suffix(brik: tx.Any) -> str:
    name = str(brik.name if hasattr(brik, "name") else brik)
    for suffix in _BRIK_SUFFIXES[1:]:
        if name.endswith(suffix):
            return suffix
    return ""


def _read_brik(header: AfniHeader, brik: tx.Any, mmap: bool) -> np.ndarray:
    """
    Read the BRIK of a dataset into a `(nx, ny, nz, nvals)` array.

    An uncompressed local BRIK whose sub-bricks share one type is
    memory-mapped (read-only) unless `mmap` is false, so nothing is read
    until the data are indexed. Anything else is read into memory.
    """
    if not path.exists(brik):
        raise ParserExistsError(
            f"No such file: {brik} (the BRIK of the AFNI dataset). A "
            f"dataset without a BRIK (a 'warp-on-demand' dataset) cannot "
            f"be read."
        )
    suffix = _brik_suffix(brik)
    if suffix not in _READABLE_SUFFIXES:
        raise ParserContentError(
            f"The AFNI BRIK {brik} is compressed with {suffix!r}, which "
            f"cannot be read here. Decompress it, or recompress it with "
            f"gzip."
        )
    nbytes = header.nbytes
    local = _local_path(brik)
    with _open_path(brik) as f:
        stream = open_compressed(f)
        compressed = stream is not f
        if (
            local is not None
            and mmap
            and not compressed
            and nbytes > 0
            and len(set(header.dtypes)) == 1
        ):
            size = os.path.getsize(local)
            if size < nbytes:
                raise ParserContentError(
                    f"The AFNI BRIK {brik} holds {size} bytes, but the "
                    f"header asks for {nbytes}."
                )
            buffer = np.memmap(
                local, dtype=np.uint8, mode="r", offset=0, shape=(nbytes,)
            )
        else:
            buffer = stream.read(nbytes)
    return decode_bricks(header, buffer)


def _write_brik(header: AfniHeader, data: tx.Any, brik: tx.Any) -> None:
    """Write the BRIK, compressed as its suffix says."""
    suffix = _brik_suffix(brik)
    with brik.open("wb") as f:
        if suffix == ".gz":
            out = gzip.GzipFile(fileobj=f, mode="wb")
        elif suffix == ".bz2":
            out = bz2.BZ2File(f, mode="wb")
        elif suffix:
            raise WriterError(
                f"Cannot write an AFNI BRIK compressed with {suffix!r}: "
                f"use '.BRIK', '.BRIK.gz' or '.BRIK.bz2'."
            )
        else:
            out = f
        try:
            for chunk in encode_bricks(header, data):
                out.write(chunk)
        finally:
            if out is not f:
                out.close()
