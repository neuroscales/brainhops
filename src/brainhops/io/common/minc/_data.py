"""MINC voxels: the lazy array that reads them with nibabel."""

# stdlib
import os
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.properties import alias
from brainhops._core.proxy import ArrayProxy
from brainhops._core.streams import open_compressed, preserve_position
from brainhops.io.base._utils_files import local_path as _local_path
from brainhops.io.base._utils_files import open_path as _open_path
from brainhops.io.base.parsers import ParserContentError, ParserExistsError

# this format
from ._raw import MincRaw
from ._utils import _read_minc, _read_voxels


class _MincProxy(ArrayProxy):
    """A lazy array of the voxels of a MINC image.

    The proxy holds where the file lies, which is a path or the bytes of
    a file that was read from a stream, and the record of the image. It
    reads the voxels with nibabel each time its values are needed, scaled
    to real values through `image-min` and `image-max`, and keeps no file
    open. A local file that is not compressed is opened by name, and any
    other file is decompressed into memory first. The values are
    presented in the order of the file, slowest dimension first.
    Indexing the proxy returns another lazy array (see
    [`ArrayProxy`][]), but any read decodes the whole file for now.
    """

    def __init__(self, source: tx.Any, record: MincRaw) -> None:
        self.source = source
        """The path of the file, or the bytes of the decompressed file."""
        self.record = record
        """The record of the image, which gives its shape and type."""

    @property
    def base_shape(self) -> tx.Tuple[int, ...]:
        """The shape of the voxels in file order, read from the record."""
        return tuple(self.record.shape)

    dtype = alias("record.scaled_dtype")
    """The type of the voxels once they are scaled."""

    def read_voxels(self) -> np.ndarray:
        """Return all the scaled voxels in file order.

        The voxels are those of the whole file, whatever part of it a view
        of the proxy selects.

        Raises
        ------
        ParserContentError
            If the file cannot be read, or its voxels no longer have the
            shape of the record.
        """
        values = _read_minc(
            _opened(self.source), self.record.version, _read_voxels
        )
        if tuple(values.shape) != self.base_shape:
            raise ParserContentError(
                f"The MINC voxels have shape {values.shape}, but the header "
                f"that the image was read with says {self.base_shape}."
            )
        return values

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        return self.read_voxels()[index]


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
        with _open_path(file) as f:
            record = MincRaw.from_fileobj(f)
        return record, _MincProxy(file, record)
    with preserve_position(file):
        content = bytes(open_compressed(file).read())
    record = MincRaw.from_fileobj(BytesIO(content))
    return record, _MincProxy(content, record)
