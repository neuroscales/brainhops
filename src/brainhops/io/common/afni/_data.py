"""AFNI `.BRIK` data: sub-brick types, decoding, scaling and
encoding, and the lazy array that reads a BRIK."""

# stdlib
import bz2
import gzip
import os
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr

# internals
from brainhops._core import path
from brainhops._core.properties import alias
from brainhops._core.streams import open_compressed
from brainhops._core.strided import StridedProxy
from brainhops.io.base._utils_files import is_local, replace_file
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
    _READABLE_SUFFIXES,
)
from ._files import _brik_suffix, afni_dataset_files
from ._raw import AfniRaw

# ----------------------------------------------------------------------
#   DATA
# ----------------------------------------------------------------------


def brick_dtype(dtype: tx.Any) -> np.dtype:
    """
    Return the AFNI dtype that stores a dtype, or the one an AFNI type
    name (`"short"`, `"float"`, ...) names.

    Supported dtypes are kept. Otherwise booleans become bytes, int8
    shorts, uint16 ints, uint32 and wider integers doubles, float16 floats,
    and complex128 single-precision complex.

    Raises
    ------
    WriterError
        If AFNI cannot store the dtype.
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
    """Return the `BRICK_TYPES` code of an AFNI dtype."""
    return _BRICK_CODES[np.dtype(dtype).newbyteorder("=")]


def decode_bricks(header: AfniRaw, buffer: tx.Any) -> np.ndarray:
    """
    Decode an uncompressed BRIK into an `(nx, ny, nz, nvals)` array.

    The buffer may be bytes, a memoryview or a one-dimensional uint8 array
    such as a memory map. When all sub-bricks share a type, the result is a
    view of the buffer; otherwise the sub-bricks are copied into their
    common type. Scale factors are not applied.

    Raises
    ------
    ParserContentError
        If the buffer is shorter than the header requires.
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


def _is_scaled(header: AfniRaw) -> bool:
    """Tell whether a scale factor of `BRICK_FLOAT_FACS` changes values."""
    facs = np.asarray(header.float_facs, dtype=np.float64)
    return bool(np.any(facs) and not np.all((facs == 0) | (facs == 1)))


def _stored_dtype(header: AfniRaw) -> np.dtype:
    """Return the type of the stored values once the sub-bricks are joined.

    Sub-bricks of one type keep it, with its byte order, and sub-bricks
    of several types are joined in their common type, as
    [`decode_bricks`][] does.
    """
    dtypes = header.dtypes
    if len(set(dtypes)) == 1:
        return dtypes[0]
    return np.result_type(*dtypes)


def scale_bricks(header: AfniRaw, stored: tx.Any) -> tx.Any:
    """
    Apply the `BRICK_FLOAT_FACS` scale factors to stored values.

    When no sub-brick is scaled, the values are returned unchanged, so that
    a memory map stays one. Otherwise they are scaled in at least single
    precision.
    """
    if not _is_scaled(header):
        return stored
    facs = np.asarray(header.float_facs, dtype=np.float64)
    facs = np.where(facs == 0, 1.0, facs)
    stored = np.asarray(stored)
    dtype = np.result_type(stored.dtype, np.float32)
    if stored.ndim == len(header.shape):
        return stored.astype(dtype) * dtype.type(facs[0])
    return stored.astype(dtype) * facs.astype(dtype)


def encode_bricks(header: AfniRaw, data: tx.Any) -> tx.Iterator[bytes]:
    """
    Encode an `(nx, ny, nz, nvals)` array into BRIK content, sub-brick by
    sub-brick.

    A three-dimensional array is a single sub-brick. Each sub-brick is
    divided by its nonzero scale factor, rounded if it is stored as
    integers, and converted to the type and byte order of the header.

    Parameters
    ----------
    header : AfniRaw
        The record that gives the shape, the number of sub-bricks, their
        types, their byte order and their scale factors.
    data : array_like
        The voxels, with shape `(nx, ny, nz, nvals)` or `(nx, ny, nz)`.

    Yields
    ------
    bytes
        The content of each sub-brick, in order.

    Raises
    ------
    WriterError
        If the shape disagrees with the header, or if integer values
        overflow their stored type.
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
#   LAZY ARRAY
# ----------------------------------------------------------------------

_CHUNK = 1 << 20
"""The number of bytes copied at a time from one BRIK to another."""


class _BrikProxy(StridedProxy):
    """A lazy array of the voxels that a BRIK file stores.

    The proxy holds the path of the BRIK and the record of the dataset,
    which says how many sub-bricks the BRIK holds and how each one is
    stored. It opens the file each time its values are needed and keeps
    no open file, so it can be pickled. Its shape is that of the image:
    `(x, y, z)` for a single sub-brick and `(x, y, z, sub-brick)`
    otherwise. The scale factors of `BRICK_FLOAT_FACS` are applied when
    the values are read, as nibabel's proxy applies the scaling of a
    NIfTI header.

    When the sub-bricks share a type, the BRIK is one Fortran-ordered
    array, and a read gets only the bytes that its selection needs (see
    [`StridedProxy`][brainhops._core.strided.StridedProxy]): a local BRIK
    that is not compressed is memory-mapped when `mmap` is true, and a
    compressed one is decompressed up to the end of the selection. The
    blocks that the proxy gives Dask are groups of sub-bricks, or slabs
    of slices for a single sub-brick. Sub-bricks of several types are
    joined in their common type, which needs the whole BRIK, so any read
    of such a BRIK decodes and scales all of it.

    A reader builds the proxy from the record of the dataset with
    [`from_record`][brainhops.io.common.afni._data._BrikProxy.from_record].
    """

    record: KwOnly[NoRepr[AfniRaw]]
    """The record of the dataset, which describes the BRIK."""

    @classmethod
    def from_record(
        cls, brik: tx.Any, record: AfniRaw, mmap: bool
    ) -> "_BrikProxy":
        """
        Return the proxy of the voxels of a BRIK that a record describes.

        The layout is read from the record: a BRIK is a Fortran-ordered
        array of shape `(x, y, z, sub-brick)`, with the last axis dropped
        for a single sub-brick, and the factors of `BRICK_FLOAT_FACS`
        scale each sub-brick. A factor of zero means no scaling.

        Parameters
        ----------
        brik : path-like
            The path of the BRIK file.
        record : AfniRaw
            The record of the dataset.
        mmap : bool
            Whether a local BRIK that is not compressed is memory-mapped.

        Returns
        -------
        _BrikProxy
            The proxy of the voxels.
        """
        nvals = record.nvals
        shape = record.shape + ((nvals,) if nvals > 1 else ())
        slope = None
        if _is_scaled(record):
            facs = np.asarray(record.float_facs, dtype=np.float64)
            facs = np.where(facs == 0, 1.0, facs)
            slope = facs if nvals > 1 else facs[0]
        return cls.contiguous(
            brik,
            record.dtypes[0],
            shape,
            order="F",
            slope=slope,
            mmap=mmap,
            record=record,
        )

    brik = alias("source")
    """The path of the BRIK file, which is the source of the proxy."""

    @property
    def _single_type(self) -> bool:
        # Whether the sub-bricks share a type, so that the BRIK is one array.
        return len(set(self.record.dtypes)) == 1

    @property
    def dtype(self) -> np.dtype:
        """The type of the values once they are scaled."""
        stored = _stored_dtype(self.record)
        if _is_scaled(self.record):
            return np.result_type(stored, np.float32)
        return stored

    nbytes_stored = alias("record.nbytes", readonly=True)
    """The number of bytes of the BRIK."""

    @property
    def base_chunks(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The block shape, or `None` for sub-bricks of several types."""
        if not self._single_type:
            return None
        return super().base_chunks

    def get_unscaled(self) -> np.ndarray:
        """Return all the stored values as a `(x, y, z, sub-brick)` array.

        The values are those of the whole BRIK, whatever part of it a view
        of the proxy selects, and the scale factors are not applied.
        """
        return _read_bricks(self.record, self.brik, self.mmap)

    def copy_to(self, out: tx.BinaryIO) -> None:
        """Write the stored bytes of the BRIK, decompressed, to a stream.

        The values are copied without being decoded, so that a dataset
        that is read and written again keeps the bytes of its BRIK.

        Raises
        ------
        ParserContentError
            If the BRIK holds fewer bytes than the record says.
        """
        remaining = self.record.nbytes
        with _open_path(self.brik) as f:
            stream = open_compressed(f)
            while remaining > 0:
                chunk = stream.read(min(remaining, _CHUNK))
                if not chunk:
                    raise ParserContentError(
                        f"The AFNI BRIK {self.brik} ends before the "
                        f"{self.record.nbytes} bytes that its header asks "
                        f"for."
                    )
                out.write(chunk)
                remaining -= len(chunk)

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        if self._single_type:
            try:
                return super()._read(index)
            except ValueError as error:
                raise ParserContentError(
                    f"The AFNI BRIK {self.brik} is shorter than the "
                    f"{self.record.nbytes} bytes that its header asks for. "
                    f"{error}"
                ) from error
        values = scale_bricks(self.record, self.get_unscaled())
        if self.record.nvals == 1:
            values = values[..., 0]
        return values[index]


def read_afni(
    filename: path.FilenameLike, mmap: bool = True
) -> tx.Tuple[AfniRaw, _BrikProxy]:
    """Read the record and a lazy array of the voxels of a dataset.

    The path may name the `.HEAD` file, the `.BRIK` file or the bare
    dataset. The record is read from the `.HEAD` file and validated, and
    the BRIK is checked, but its values are not read.

    Parameters
    ----------
    filename : path-like
        The path of the `.HEAD` file, of the `.BRIK` file, or of the
        dataset without an extension.
    mmap : bool, default=True
        Whether to memory-map a local BRIK that is not compressed.

    Returns
    -------
    raw : AfniRaw
        The validated record.
    proxy : array
        A lazy array of the voxels, with shape `(nx, ny, nz, nvals)`.

    Raises
    ------
    ParserExistsError
        If the `.HEAD` or the `.BRIK` file does not exist.
    ParserContentError
        If the header is invalid, or the BRIK is compressed in a way
        that cannot be read or is too short.
    """
    head, brik, _ = afni_dataset_files(filename)
    if not path.exists(head):
        raise ParserExistsError(f"No such file: {head}")
    record = AfniRaw.from_filename(head).validate()
    return record, brik_proxy(record, brik, mmap)


def brik_proxy(record: AfniRaw, brik: tx.Any, mmap: bool) -> _BrikProxy:
    """Return the lazy array of a BRIK, after checking that it can be read.

    The checks need no voxel: the BRIK must exist, be compressed with
    gzip or bzip2 or not at all, and, when it is a local file that is not
    compressed, hold the bytes that the record asks for.

    Raises
    ------
    ParserExistsError
        If the BRIK does not exist.
    ParserContentError
        If the BRIK is compressed in a way that cannot be read, or is too
        short.
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
    local = _local_path(brik)
    if local is not None and not suffix:
        size = os.path.getsize(local)
        if size < record.nbytes:
            raise ParserContentError(
                f"The AFNI BRIK {brik} holds {size} bytes, but the header "
                f"asks for {record.nbytes}."
            )
    return _BrikProxy.from_record(brik, record, mmap)


def _read_bricks(header: AfniRaw, brik: tx.Any, mmap: bool) -> np.ndarray:
    """
    Read the sub-bricks of a BRIK, memory-mapping the file when it is
    local, uncompressed, of a single type, and `mmap` is true.
    """
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
            buffer = np.memmap(
                local, dtype=np.uint8, mode="r", offset=0, shape=(nbytes,)
            )
        else:
            buffer = stream.read(nbytes)
    return decode_bricks(header, buffer)


def write_brik(header: AfniRaw, data: tx.Any, brik: tx.Any) -> None:
    """Write a BRIK, compressed according to its suffix.

    A [`_BrikProxy`][] that is not a view is copied as the file stores
    it. A local BRIK is written beside its target and moved over it, so
    that a proxy that reads the target copies the old file. When the
    proxy reads a remote file that is written, its bytes are read into
    memory first, since opening the file for writing empties it. Any
    other array is encoded with the types and the scale factors of the
    header.

    Raises
    ------
    WriterError
        If the suffix asks for a compression that cannot be written.
    """
    suffix = _brik_suffix(brik)
    if suffix not in ("", ".gz", ".bz2"):
        raise WriterError(
            f"Cannot write an AFNI BRIK compressed with {suffix!r}: "
            f"use '.BRIK', '.BRIK.gz' or '.BRIK.bz2'."
        )

    def write(f: tx.BinaryIO) -> None:
        if suffix == ".gz":
            out = gzip.GzipFile(fileobj=f, mode="wb")
        elif suffix == ".bz2":
            out = bz2.BZ2File(f, mode="wb")
        else:
            out = f
        try:
            if stores_brik(data):
                data.copy_to(out)
            elif isinstance(data, bytes):
                out.write(data)
            else:
                for chunk in encode_bricks(header, data):
                    out.write(chunk)
        finally:
            if out is not f:
                out.close()

    def write_named(name: str) -> None:
        with open(name, "xb") as f:
            write(f)

    if is_local(brik):
        replace_file(brik, write_named)
        return
    if stores_brik(data) and _same_file(data.brik, brik):
        copied = BytesIO()
        data.copy_to(copied)
        data = copied.getvalue()
    with brik.open("wb") as f:
        write(f)


def _same_file(first: tx.Any, second: tx.Any) -> bool:
    """Tell whether two paths name the same existing local file."""
    first, second = _local_path(first), _local_path(second)
    if first is None or second is None:
        return False
    return os.path.samefile(first, second)


def stores_brik(raw: tx.Any) -> bool:
    """Tell whether an array is a lazy array of a whole BRIK file.

    The bytes of such a lazy array can be copied as the BRIK stores
    them. A view of a lazy array, which selects a part of the BRIK or
    reorders its axes, is encoded like an array in memory.

    Parameters
    ----------
    raw : array
        The `rawdata` array of an object.

    Returns
    -------
    bool
        Whether `raw` is a lazy array of a BRIK that is not a view.
    """
    return isinstance(raw, _BrikProxy) and not raw.is_view


def reads_brik(raw: tx.Any, brik: tx.Any) -> bool:
    """Tell whether an array is a lazy array of a BRIK file.

    Parameters
    ----------
    raw : array
        The `rawdata` array of an object.
    brik : path-like
        The BRIK file.

    Returns
    -------
    bool
        Whether `raw` is a lazy array that reads `brik`.
    """
    return isinstance(raw, _BrikProxy) and _same_file(raw.brik, brik)
