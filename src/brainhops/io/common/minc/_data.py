"""MINC voxels: the lazy array that reads them with nibabel."""

# stdlib
import os
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops._core._internal_api import (
    StoredProxy,
    alias,
    is_local_file,
    open_compressed,
    preserve_position,
)
from brainhops.io.base.parsers import ParserContentError, ParserExistsError

# this format
from ._raw import MincRaw
from ._utils import apply_to_minc, read_minc_voxels

_CHUNK_BYTES = 128 * 2**20
"""The size of a block when Dask is not installed to give its own."""


class MincProxy(StoredProxy):
    """A lazy array of the voxels of a MINC image.

    The proxy holds where the file lies, which is a path or the bytes of
    a file that was read from a stream, and the record of the image. It
    reads the voxels with nibabel each time its values are needed, scaled
    to real values through `image-min` and `image-max`, and keeps no file
    open. A local file that is not compressed is opened by name, and any
    other file is decompressed into memory first. The values are
    presented in the order of the file, slowest dimension first.

    A read asks nibabel for the box that it selects, which reads only
    that part of a local file that is not compressed: MINC1 files are
    memory-mapped, and MINC2 files are read with h5py. A compressed
    file, or one read from a stream, is decompressed whole first. The
    blocks that the proxy gives Dask are slabs of the slowest dimension,
    as many as fit in the block size of Dask.
    """

    # --- magic attributes ----

    source: tx.Any
    """The path of the file, or the bytes of the decompressed file."""

    record: NoRepr[MincRaw]
    """The record of the image, which gives its shape and type."""

    # --- properties ----

    shape = alias("record.shape", readonly=True)
    """The shape of the voxels in file order, read from the record."""

    dtype = alias("record.scaled_dtype", readonly=True)
    """The type of the voxels once they are scaled."""

    @property
    def chunks(self) -> tx.Tuple[int, ...]:
        """The block shape: slabs of the slowest dimension.

        A block holds every value of the other dimensions, and as many
        slabs as fit in the block size of Dask, with at least one slab.
        """
        shape = self.shape
        if not shape:
            return shape
        slab = int(np.prod(shape[1:])) * np.dtype(self.dtype).itemsize
        count = max(1, _chunk_bytes() // max(1, slab))
        return (min(count, shape[0]),) + shape[1:]

    # --- methods

    def read_voxels(self) -> np.ndarray:
        """Return all the scaled voxels in file order.

        Raises
        ------
        ParserContentError
            If the file cannot be read, or its voxels no longer have the
            shape of the record.
        """
        values = apply_to_minc(
            _opened(self.source), self.record.version, read_minc_voxels
        )
        if tuple(values.shape) != self.shape:
            raise ParserContentError(
                f"The MINC voxels have shape {values.shape}, but the header "
                f"that the image was read with says {self.shape}."
            )
        return values

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        def read(container: tx.Any, mfile: tx.Any, version: int) -> np.ndarray:
            if tuple(mfile.get_data_shape()) != self.shape:
                raise ParserContentError(
                    f"The MINC voxels have shape {mfile.get_data_shape()}, "
                    f"but the header that the image was read with says "
                    f"{self.shape}."
                )
            return read_minc_voxels(container, mfile, version, index)

        return apply_to_minc(_opened(self.source), self.record.version, read)


def _opened(source: tx.Any) -> tx.Union[path.Path, bytes]:
    """Return what nibabel reads the voxels from: a name or the bytes.

    A local file that is not compressed is read by name, and any other
    file is read and decompressed into bytes.
    """
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    source = path.Path(source)
    if is_local_file(source):
        with source.open("rb") as f:
            if f.read(2) != b"\x1f\x8b":
                return source
    with source.open("rb") as f:
        return bytes(open_compressed(f).read())


def read_minc(file: tx.Any) -> tx.Tuple[MincRaw, MincProxy]:
    """Read the record and a lazy array of the voxels of a MINC image.

    The file is a path or a stream, gzipped or not. A path is read for
    its header alone, and the lazy array reads the voxels from it when
    they are needed. A stream has no name to open again, so its content
    is read into memory, decompressed, and its voxels are decoded only
    when they are needed.

    Parameters
    ----------
    file : path-like or file object
        The path of the image, or a stream open in binary mode.

    Returns
    -------
    raw : MincRaw
        The record.
    proxy : array
        A lazy array of the voxels.

    Raises
    ------
    ParserExistsError
        If the path does not exist.
    ParserContentError
        If the content is not a MINC file that nibabel can read.
    """
    if isinstance(file, str):
        file = path.Path(file)
    if isinstance(file, os.PathLike):
        if not path.exists(file):
            raise ParserExistsError(f"No such file: {file}")
        with path.Path(file).open("rb") as f:
            record = MincRaw.from_fileobj(f)
        return record, MincProxy(file, record)
    with preserve_position(file):
        content = bytes(open_compressed(file).read())
    record = MincRaw.from_fileobj(BytesIO(content))
    return record, MincProxy(content, record)


def _chunk_bytes() -> int:
    """Return the number of bytes of a block that Dask is configured for."""
    try:
        import dask
        from dask.utils import parse_bytes
    except ImportError:
        return _CHUNK_BYTES
    size = dask.config.get("array.chunk-size", _CHUNK_BYTES)
    return int(parse_bytes(size)) if isinstance(size, str) else int(size)
