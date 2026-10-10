"""MINC voxels: the lazy array that reads them with nibabel."""

# stdlib
import os
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.streams import open_compressed, preserve_position
from brainhops.io.base._utils_files import local_path as _local_path
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base.parsers import ParserContentError, ParserExistsError

# this format
from ._raw import MincRaw
from ._utils import _read_minc, _read_voxels


class _MincProxy:
    """A lazy array of the voxels of a MINC image.

    The proxy holds where the file lies, which is a path or the bytes of
    a file that was read from a stream, and the record of the image. It
    reads the voxels with nibabel each time its values are needed, scaled
    to real values through `image-min` and `image-max`, and keeps no file
    open. A local file that is not compressed is opened by name, and any
    other file is decompressed into memory first. The values are
    presented in the order of the file, slowest dimension first.
    """

    def __init__(self, source: tx.Any, record: MincRaw) -> None:
        self.source = source
        """The path of the file, or the bytes of the decompressed file."""
        self.record = record
        """The record of the image, which gives its shape and type."""

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the voxels in file order, read from the record."""
        return self.record.shape

    @property
    def ndim(self) -> int:
        """The number of dimensions of the voxels."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """The type of the voxels once they are scaled."""
        return self.record.scaled_dtype

    def read_voxels(self) -> np.ndarray:
        """Return the scaled voxels in file order.

        Raises
        ------
        ParserContentError
            If the file cannot be read, or its voxels no longer have the
            shape of the record.
        """
        values = _read_minc(
            _opened(self.source), self.record.version, _read_voxels
        )
        if tuple(values.shape) != self.shape:
            raise ParserContentError(
                f"The MINC voxels have shape {values.shape}, but the header "
                f"that the image was read with says {self.shape}."
            )
        return values

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        values = self.read_voxels()
        return values if dtype is None else values.astype(dtype)

    def __getitem__(self, index: tx.Any) -> np.ndarray:
        # An empty selection, such as the one that Dask reads to learn the
        # type of an array, is answered without reading the file.
        selected = np.broadcast_to(np.empty((), self.dtype), self.shape)[index]
        if selected.size == 0:
            return np.empty(selected.shape, self.dtype)
        return np.asarray(self)[index]

    def __repr__(self) -> str:
        return f"_MincProxy(shape={self.shape}, dtype={self.dtype})"


def _opened(source: tx.Any) -> tx.Union[str, bytes]:
    """Return what nibabel reads the voxels from: a name or the bytes.

    A local file that is not compressed is read by name, and any other
    file is read and decompressed into bytes.
    """
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    local = _local_path(source)
    if local is not None:
        with open(local, "rb") as f:
            if f.read(2) != b"\x1f\x8b":
                return local
    with _open_path(source) as f:
        return bytes(open_compressed(f).read())


def read_minc(file: tx.Any) -> tx.Tuple[MincRaw, _MincProxy]:
    """Read the record and a lazy array of the voxels of a MINC image.

    The file is a path or a stream, gzipped or not. A path is read for
    its header alone, and the lazy array reads the voxels from it when
    they are needed. A stream has no name to open again, so its content
    is read into memory, decompressed, and its voxels are decoded only
    when they are needed.

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
        with _open_path(file) as f:
            record = MincRaw.from_fileobj(f)
        return record, _MincProxy(file, record)
    with preserve_position(file):
        content = bytes(open_compressed(file).read())
    record = MincRaw.from_fileobj(BytesIO(content))
    return record, _MincProxy(content, record)
